#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deploy versioned Sphinx documentation to GitHub Pages.

Takes the HTML output from a Sphinx build (in ``docs/_build/html/``) and
commits it to the local ``gh-pages`` branch under a versioned folder
(e.g. ``/latest/``, ``/v1.10/``).  The caller is responsible for pushing
the branch to the remote.

The script also maintains:
  - /stable/        full copy of the highest MAJOR.MINOR version
  - /versions.json  version switcher data for the PyData Sphinx theme
  - /index.html     redirect to /stable/ (or /latest/ if no releases exist)
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
from pathlib import Path
from typing import List, Optional

BASE_URL = "https://nvidia.github.io/warp"
HTML_DIR = Path("docs/_build/html")

# Sphinx writes pickled doctrees to ``<outdir>/.doctrees/`` by default — large
# binary state (env.pickle alone is ~10 MB, with absolute paths embedded so it
# regenerates with different bytes every build). No user agent ever loads it,
# so excluding it from the deploy keeps gh-pages from accumulating tens of
# megabytes of churn per push (the failure mode that bloated newton-physics).
# ``__pycache__`` shows up in builds that import generators with stale .pyc.
_DEPLOY_EXCLUDE = shutil.ignore_patterns(".doctrees", "__pycache__", "*.pyc")


# Git Helpers
# ---------------------------------------------------------------------------

def git_run_cmd(*args: str, cwd: Optional[Path] = None) -> str:
    """Run a git command, printing it for CI visibility.

    Returns stdout. Raises ``subprocess.CalledProcessError`` on non-zero exit;
    stderr is printed before the exception is raised so the underlying error
    is visible (the default ``capture_output=True`` would otherwise swallow it).
    """
    cmd = ("git", *args)
    print(f"  $ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=cwd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        if result.stdout:
            print(result.stdout, flush=True)
        if result.stderr:
            print(result.stderr, file=sys.stderr, flush=True)
        result.check_returncode()
    return result.stdout.strip()


# Version Helpers
# ---------------------------------------------------------------------------

_VERSION_RE = re.compile(r"^\d+\.\d+$")


def resolve_version_folder(version: str) -> str:
    """Map a --version value to the target folder name on gh-pages.

    "latest" -> "latest", "1.10" -> "v1.10". Any other shape is rejected so
    passing a full patch version like "1.10.0" does not silently create a
    /v1.10.0/ folder. The pattern mirrors the regex enforced by the deploy
    workflow.
    """
    if version == "latest":
        return "latest"

    if not _VERSION_RE.match(version):
        raise ValueError(
            f"Invalid --version '{version}': expected 'latest' or 'MAJOR.MINOR' (e.g. '1.10')"
        )

    return f"v{version}"


def discover_versions(gh_pages_dir: Path) -> List[str]:
    """Return deployed MAJOR.MINOR versions sorted highest-first."""
    versions = []
    for entry in gh_pages_dir.iterdir():
        if entry.is_dir():
            m = re.match(r"^v(\d+\.\d+)$", entry.name)
            if m:
                versions.append(m.group(1))

    versions.sort(key=lambda v: tuple(int(x) for x in v.split(".")), reverse=True)
    return versions


def is_released(version: str) -> bool:
    """Whether a final ``v{version}.N`` tag exists in the local repo.

    Used to gate ``/stable/`` promotion on an actual release rather than the
    mere existence of a ``release-X.Y`` branch (which is created during the
    RC phase, before the tag is published). Requires the workflow's checkout
    step to have fetched tags (``fetch-depth: 0``).

    Pre-release / post-release tags such as ``v1.9.0rc1``, ``v1.0.0-beta.3``,
    or ``v1.7.2.post1`` are deliberately excluded: they don't represent a
    published stable release. The git glob is broad to keep the candidate
    list small, then a strict regex enforces ``vX.Y.N`` only.
    """
    result = subprocess.run(
        ("git", "tag", "-l", f"v{version}.*"),
        capture_output=True, text=True, check=False,
    )
    pattern = re.compile(rf"^v{re.escape(version)}\.\d+$")
    return any(pattern.fullmatch(tag) for tag in result.stdout.splitlines())


# Content Generation
# ---------------------------------------------------------------------------

def generate_versions_json(
    gh_pages_dir: Path,
    versions: List[str],
    released: List[str],
    has_latest: bool,
) -> None:
    """Write versions.json for the PyData Sphinx theme version switcher.

    The highest *released* version is flagged ``preferred`` (and labeled
    "stable"). Versions deployed without a corresponding tag are labeled
    "(prerelease)" so RC reviewers can find them in the dropdown without
    them being treated as stable.
    """
    entries = []

    if has_latest:
        entries.append({
            "name": "latest (main)",
            "version": "latest",
            "url": f"{BASE_URL}/latest/",
        })

    released_set = set(released)
    preferred = released[0] if released else None

    for ver in versions:
        entry: dict = {
            "version": ver,
            "url": f"{BASE_URL}/v{ver}/",
        }
        if ver == preferred:
            entry["name"] = f"{ver} (stable)"
            entry["preferred"] = True
        elif ver not in released_set:
            entry["name"] = f"{ver} (prerelease)"
        else:
            entry["name"] = ver
        entries.append(entry)

    path = gh_pages_dir / "versions.json"
    path.write_text(json.dumps(entries, indent=2) + "\n")
    print(f"Wrote {path} with {len(entries)} entries.")


def generate_root_redirect(gh_pages_dir: Path, target: str) -> None:
    """Write a root index.html that redirects to *target* (e.g. "stable/")."""
    html = f"""\
<!DOCTYPE html>
<html>
<head>
  <meta http-equiv="refresh" content="0; url={target}" />
  <script>window.location.href = "{target}";</script>
</head>
<body>
  <p>Redirecting to <a href="{target}">{target}</a>...</p>
</body>
</html>
"""
    (gh_pages_dir / "index.html").write_text(html)
    print(f"Root redirect -> {target}")


def generate_404_redirect(gh_pages_dir: Path, target: str) -> None:
    """Write 404.html: redirect pre-versioning paths; show a GH-style not-found for the rest.

    GitHub Pages has no server-side redirect mechanism: it serves /404.html
    for any path that doesn't resolve. This catches old bookmarks like
    ``/user_guide/installation.html`` (pre-versioning) and rewrites them to
    ``/<target>user_guide/installation.html``. Paths already under a known
    version folder (``stable/``, ``latest/``, ``vX.Y/``) are genuine 404s
    within a version — render a body that mimics GitHub Pages' own default
    404 page (which is what users would see if /404.html didn't exist).
    """
    # Project Pages live under a path prefix (e.g. "/warp/"); strip it before
    # matching so the version-folder regex sees a relative path.
    prefix = urllib.parse.urlparse(BASE_URL).path.rstrip("/") + "/"

    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta http-equiv="X-UA-Compatible" content="IE=edge,chrome=1" />
  <meta name="viewport" content="width=device-width" />
  <title>Page not found &middot; Warp</title>
  <script>
(function () {{
  var prefix = {json.dumps(prefix)};
  var path = location.pathname;
  if (path.indexOf(prefix) === 0) {{
    path = path.slice(prefix.length);
  }}
  // Already under a known version: this is a genuine 404 within that
  // version. Fall through to the not-found body below.
  if (/^(stable|latest|v\\d+\\.\\d+)(\\/|$)/.test(path)) {{
    return;
  }}
  // Pre-versioning URL: rewrite under the current default version.
  location.replace(prefix + {json.dumps(target)} + path + location.search + location.hash);
}})();
  </script>
  <style>
    body {{
      background: #f1f1f1;
      color: #222;
      font: 14px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
      margin: 0;
    }}
    .container {{
      margin: 0 auto;
      max-width: 600px;
      padding: 20px 0 40px;
      text-align: center;
    }}
    h1 {{
      color: #222;
      font-size: 144px;
      font-weight: 800;
      letter-spacing: -2px;
      line-height: 1;
      margin: 0;
    }}
    h2 {{
      color: #5a5a5a;
      font-size: 24px;
      font-weight: 400;
      margin: 12px 0 24px;
    }}
    p {{
      color: #5a5a5a;
      font-size: 14px;
      margin: 0.5em 0;
    }}
    a {{ color: #76b900; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>404</h1>
    <h2>File not found.</h2>
    <p>The page you requested does not exist in this version of the Warp documentation.</p>
    <p><a href="{prefix}stable/">Latest stable docs</a> &nbsp;&middot;&nbsp; <a href="{prefix}latest/">Development docs</a></p>
  </div>
</body>
</html>
"""
    (gh_pages_dir / "404.html").write_text(html)
    print(f"404 fallback -> {target}")


def update_stable(gh_pages_dir: Path, stable_version: str) -> None:
    """Replace /stable/ with a full copy of the highest version folder.

    We deliberately do a real recursive copy (not redirect stubs): downstream
    Sphinx projects use ``intersphinx_mapping`` against
    ``https://nvidia.github.io/warp/stable/objects.inv``, and pytorch/pytorch#182007
    showed what happens when an alias drops ``objects.inv`` /
    ``searchindex.js`` / ``.buildinfo`` — every dependent project's docs build
    breaks. The ``.doctrees`` exclusion in ``_DEPLOY_EXCLUDE`` already
    happened upstream of here (the v{X.Y} folder doesn't contain them).
    """
    stable_dir = gh_pages_dir / "stable"
    source_dir = gh_pages_dir / f"v{stable_version}"

    if stable_dir.exists():
        shutil.rmtree(stable_dir)
    shutil.copytree(source_dir, stable_dir)
    print(f"/stable/ -> v{stable_version}")


# Main
# ---------------------------------------------------------------------------

def run(version: str, metadata_only: bool = False) -> None:
    """Deploy the built docs (or, with ``metadata_only``, just refresh
    ``/stable/`` and ``versions.json`` against the existing ``/vX.Y/``).

    The metadata-only path is for the tag-trigger workflow: a tag is gating
    ``/stable/`` promotion but the version folder content is whatever the
    ``release-X.Y`` branch line currently has. Rebuilding from the tag
    commit could regress ``/vX.Y/`` if the branch is ahead of the tag.
    """
    folder = resolve_version_folder(version)
    print(f"Deploying to /{folder}/  (metadata_only={metadata_only})")

    if not metadata_only and not HTML_DIR.exists():
        raise FileNotFoundError(f"Built docs not found at {HTML_DIR}")

    branch_exists = bool(git_run_cmd("branch", "--list", "gh-pages"))

    # Refuse to silently orphan when origin already has a gh-pages branch.
    # Without this, a local run on a fresh clone would create an empty branch,
    # and the workflow's force-push would then wipe history.
    if not branch_exists:
        remote = subprocess.run(
            ("git", "ls-remote", "--heads", "origin", "gh-pages"),
            capture_output=True, text=True, check=False,
        )
        if remote.returncode == 0 and remote.stdout.strip():
            raise RuntimeError(
                "origin has a gh-pages branch but it is not present locally. "
                "Refusing to orphan and overwrite remote history. Fetch first:\n"
                "  git fetch origin gh-pages:gh-pages"
            )

    # Use a worktree at the existing gh-pages branch (or, on first-ever
    # bootstrap with neither local nor remote gh-pages, an orphan branch
    # created in place). Committing in the worktree updates the local
    # gh-pages ref directly, and git enforces that the deploy commit
    # is a descendant of the prior tip rather than us doing it via FETCH_HEAD.
    with tempfile.TemporaryDirectory() as tmp:
        wt = Path(tmp) / "gh-pages"
        if branch_exists:
            git_run_cmd("worktree", "add", str(wt), "gh-pages")
        else:
            # First-ever deploy: gh-pages doesn't exist anywhere. Stand up
            # a detached worktree at HEAD, switch to a fresh orphan branch,
            # and wipe the working tree so the deploy starts clean. We avoid
            # `git worktree add --orphan` (git >= 2.42 only) and use the
            # older orphan-checkout flow instead.
            git_run_cmd("worktree", "add", "--detach", str(wt), "HEAD")
            git_run_cmd("checkout", "--orphan", "gh-pages", cwd=wt)
            git_run_cmd("rm", "-rf", ".", cwd=wt)

        try:
            # 1. Deploy the built HTML into the target version folder, or
            # (metadata-only) assert the folder is already there. Tag-trigger
            # runs use metadata-only when /vX.Y/ exists; if it doesn't, the
            # workflow falls back to a full deploy from the tag commit.
            target = wt / folder
            if metadata_only:
                if not target.exists():
                    raise RuntimeError(
                        f"--metadata-only: /{folder}/ doesn't exist on "
                        f"gh-pages (deploy the branch first to populate it)"
                    )
            else:
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(HTML_DIR, target, ignore=_DEPLOY_EXCLUDE)
                print(f"Copied {HTML_DIR} -> {target}")

            # 2. Discover all deployed release versions, and the released
            # subset (those with a v{X.Y}.* tag in the repo). Only released
            # versions are eligible to back /stable/; RC pushes to a
            # release-X.Y branch deploy to /vX.Y/ but don't touch /stable/.
            versions = discover_versions(wt)
            released = [v for v in versions if is_released(v)]
            has_latest = (wt / "latest").is_dir()
            print(
                f"Deployed versions: {versions}  released: {released}  "
                f"(has latest: {has_latest})"
            )

            # 3. Update /stable/ and root + 404 redirects.
            if released:
                update_stable(wt, released[0])
                generate_root_redirect(wt, "stable/")
                generate_404_redirect(wt, "stable/")
            else:
                stable_dir = wt / "stable"
                if stable_dir.exists():
                    shutil.rmtree(stable_dir)
                    print("Removed stale /stable/ (no released versions)")
                if has_latest:
                    generate_root_redirect(wt, "latest/")
                    generate_404_redirect(wt, "latest/")

            # 4. Generate versions.json for the version switcher.
            generate_versions_json(wt, versions, released, has_latest)

            # 5. Ensure .nojekyll exists at the root.
            (wt / ".nojekyll").touch()

            # 6. Commit on top of gh-pages directly.
            git_run_cmd("add", "-A", cwd=wt)
            if not git_run_cmd("status", "--porcelain", cwd=wt):
                print("No changes to commit.")
                return

            # Worktrees share .git/config with the main repo, so writing
            # commit identity / `commit.gpgsign` / `core.hooksPath` via
            # `git config` would mutate the operator's working repo.
            # Override per-invocation instead. Auto-generated build
            # artifact: signing would require a key in the CI runner /
            # local agent, and any operator-installed hooks (pre-commit,
            # commit-msg, …) would block a non-interactive deploy.
            git_run_cmd(
                "-c", "user.email=actions@github.com",
                "-c", "user.name=GitHub Actions",
                "-c", "commit.gpgsign=false",
                "-c", "core.hooksPath=/dev/null",
                "commit",
                "-m", f"Deploy docs: {folder}",
                cwd=wt,
            )
        finally:
            git_run_cmd("worktree", "remove", "--force", str(wt))

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deploy versioned Sphinx documentation to GitHub Pages.",
    )
    parser.add_argument(
        "--version",
        required=True,
        help=(
            "Version identifier for this deployment. "
            "'latest' deploys to /latest/; a MAJOR.MINOR string (e.g. '1.10') "
            "deploys to /vMAJOR.MINOR/."
        ),
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help=(
            "Skip the HTML deploy and only refresh /stable/, /index.html, "
            "/404.html, and /versions.json against the existing /vX.Y/ "
            "content. Fails if the version folder doesn't already exist. "
            "Used by the tag-push trigger to advance /stable/ without "
            "regressing /vX.Y/ to the tagged commit's content."
        ),
    )
    args = parser.parse_args()
    try:
        run(args.version, metadata_only=args.metadata_only)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
