#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Emit a JSON contributor list for Warp release-notes drafting.

Deterministic, stdlib-only. Enumerates commits in <base>..<head>, extracts GH
refs and author identity, resolves GitHub usernames best-effort, and classifies
each unique contributor as ``nvidia`` / ``external`` per the rules in
``../references/contributor-attribution.md``.

The ``gh`` CLI is used opportunistically for two lookups: resolving the GitHub
login for a commit, and checking NVIDIA org membership. The org-membership
check uses ``/orgs/<org>/members/<login>``, which only returns 204 to fellow
org members; outside callers see 404 even for actual members and any
private-membership staff fall through to ``external``. The skill is for the
release manager (an NVIDIA org member by definition), so the assumption holds.
When ``gh`` is unavailable both lookups are skipped and only ``@nvidia.com``
emails register as internal.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


def _git(repo: Path, *args: str) -> str:
    """Run ``git -C repo args...`` and return stdout, exiting clearly on failure."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        sys.exit("list_contributors: 'git' executable not found on PATH")
    if result.returncode != 0:
        sys.exit(
            f"list_contributors: git {' '.join(args)} failed in {repo} "
            f"(exit {result.returncode}): {result.stderr.strip()}"
        )
    return result.stdout


def get_commits_in_range(repo: Path, base: str, head: str, *, drop_cherry_picked: bool = True) -> list[dict]:
    """Enumerate commits in base..head with metadata, body, and file list.

    Returns oldest-first; merges skipped. Each record carries ``sha``,
    ``subject``, ``author_name``, ``author_email``, ``body``, ``files``.

    Uses a single ``git log`` call (custom format + ``--name-only``) to avoid
    per-commit subprocess overhead. The ASCII Record Separator (``\\x1e``)
    brackets the format so commit bodies (which often contain newlines) parse
    safely; the Unit Separator (``\\x1f``) splits fields within a record.

    With ``drop_cherry_picked=True`` (default), uses git's symmetric-difference
    range with ``--cherry-pick --right-only`` so commits whose patch-id has an
    equivalent in ``base`` are filtered out. This catches the common case of a
    fix being cherry-picked onto a prior bugfix release branch and tagged there
    (with a different SHA than the original on the release branch). Without
    this filter, contributors of those re-applied commits would be surfaced
    again as if they were new in this release.
    """
    if drop_cherry_picked:
        range_args = ["--cherry-pick", "--right-only", f"{base}...{head}"]
    else:
        range_args = [f"{base}..{head}"]
    fmt = "\x1e" + "\x1f".join(["%H", "%s", "%an", "%ae", "%b"]) + "\x1e"
    stdout = _git(
        repo,
        "log",
        "--no-merges",
        "--reverse",
        "--name-only",
        f"--format={fmt}",
        *range_args,
    )
    commits: list[dict] = []
    # Output: \x1e<header>\x1e<file-block>\x1e<header>\x1e<file-block>...
    # parts[0] is leading text before the first separator (empty in practice).
    # Records then alternate header / file-block.
    parts = stdout.split("\x1e")
    parts_iter = iter(parts[1:])
    for header, file_block in zip(parts_iter, parts_iter, strict=True):
        sha, subject, author_name, author_email, body = header.split("\x1f", 4)
        files = [f for f in file_block.split("\n") if f]
        commits.append(
            {
                "sha": sha,
                "subject": subject,
                "author_name": author_name,
                "author_email": author_email,
                "body": body,
                "files": files,
            }
        )
    return commits


_GH_REF_RE = re.compile(r"\bGH-(\d+)")
_PR_TRAILER_RE = re.compile(r"\(#(\d+)\)\s*$")
_NOREPLY_RE = re.compile(r"^\d+\+([^@]+)@users\.noreply\.github\.com$", re.IGNORECASE)
_LEGACY_NOREPLY_RE = re.compile(r"^([^@]+)@users\.noreply\.github\.com$", re.IGNORECASE)
_CHANGELOG_SECTION_RE = re.compile(r"^## \[([^\]]+)\]")

# GitHub usernames: 1-39 chars, alphanumeric or hyphens, no leading/trailing
# hyphen, no consecutive hyphens. Used to validate noreply-extracted logins
# before they flow into the JSON output (and from there into release-note
# @mentions).
_GH_LOGIN_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9])){0,38}$")


def parse_changelog_gh_refs(changelog_text: str) -> dict[int, list[str]]:
    """Map each ``GH-NNNN`` reference to the list of CHANGELOG section labels
    it appears in (e.g. ``"Unreleased"``, ``"1.12.1"``).

    Same ref can appear under multiple sections (deprecation in one release,
    removal in another); the returned list preserves the order of appearance
    in the file (newest section first, since CHANGELOG.md is in reverse-
    chronological order).
    """
    refs: dict[int, list[str]] = {}
    current: str | None = None
    for line in changelog_text.splitlines():
        m = _CHANGELOG_SECTION_RE.match(line)
        if m:
            current = m.group(1)
            continue
        if current is None:
            continue
        for gh in _GH_REF_RE.finditer(line):
            num = int(gh.group(1))
            sections = refs.setdefault(num, [])
            if current not in sections:
                sections.append(current)
    return refs


def extract_gh_refs(subject: str, body: str) -> list[int]:
    """Return sorted, deduplicated GH issue numbers referenced in subject+body."""
    nums: set[int] = set()
    for text in (subject, body):
        for match in _GH_REF_RE.finditer(text):
            nums.add(int(match.group(1)))
    return sorted(nums)


def extract_pr_number(subject: str) -> int | None:
    """Return the PR number from a squash-commit subject's `(#NNNN)` trailer."""
    m = _PR_TRAILER_RE.search(subject)
    return int(m.group(1)) if m else None


def get_changelog_added_gh_refs_for_range(
    repo: Path, base: str, head: str, *, drop_cherry_picked: bool = True
) -> dict[str, list[int]]:
    """Return ``{sha: [gh_ref, ...]}`` for commits in range that added GH refs to CHANGELOG.md.

    Single ``git log -p`` pass over the range, filtered to ``CHANGELOG.md``.
    Catches commits whose subject/body omit the GH ref but whose CHANGELOG
    entry carries it; otherwise the changelog-section attribution misses
    those contributions. The format string brackets each commit's SHA with
    ``\\x1e`` so we can recover record boundaries before parsing the patch.
    """
    if drop_cherry_picked:
        range_args = ["--cherry-pick", "--right-only", f"{base}...{head}"]
    else:
        range_args = [f"{base}..{head}"]
    fmt = "\x1e%H\x1e"
    stdout = _git(
        repo,
        "log",
        "--no-merges",
        "--reverse",
        "--no-color",
        "-p",
        "--unified=0",
        f"--format={fmt}",
        *range_args,
        "--",
        "CHANGELOG.md",
    )
    refs_per_sha: dict[str, list[int]] = {}
    parts = stdout.split("\x1e")
    parts_iter = iter(parts[1:])
    for sha, diff in zip(parts_iter, parts_iter, strict=True):
        nums: set[int] = set()
        for line in diff.splitlines():
            if not line.startswith("+") or line.startswith("+++"):
                continue
            for match in _GH_REF_RE.finditer(line):
                nums.add(int(match.group(1)))
        if nums:
            refs_per_sha[sha] = sorted(nums)
    return refs_per_sha


def resolve_gh_login_from_email(email: str) -> str | None:
    """Return the GitHub login extracted from a noreply email, else None.

    The candidate is validated against GitHub's handle grammar (1-39 chars,
    alphanumeric or hyphens, no leading/trailing hyphen, no consecutive
    hyphens) before being returned. Without validation, malformed local parts
    (spaces, dots, commas) would flow into the JSON output and ultimately into
    rendered ``@mentions`` in the release notes, potentially pinging the
    wrong user or no user at all.
    """
    candidate: str | None = None
    m = _NOREPLY_RE.match(email.strip())
    if m:
        candidate = m.group(1)
    else:
        m = _LEGACY_NOREPLY_RE.match(email.strip())
        if m:
            candidate = m.group(1)
    if candidate is None:
        return None
    if not _GH_LOGIN_RE.match(candidate):
        return None
    return candidate


def gh_available() -> bool:
    """Return True if ``gh`` is installed and authenticated."""
    if shutil.which("gh") is None:
        return False
    try:
        result = subprocess.run(
            ["gh", "auth", "status"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return result.returncode == 0


def gh_lookup_login_for_commit(sha: str, repo_slug: str = "NVIDIA/warp") -> str | None:
    """Look up the GitHub login for a commit via gh api. Returns None on failure."""
    try:
        result = subprocess.run(
            [
                "gh",
                "api",
                f"/repos/{repo_slug}/commits/{sha}",
                "--jq",
                ".author.login // empty",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if result.returncode != 0:
        return None
    login = result.stdout.strip()
    return login or None


def gh_check_org_membership(login: str, org: str) -> bool:
    """Return True if ``login`` is a member of ``org`` (public OR private).

    Uses ``/orgs/<org>/members/<login>``, which returns 204 No Content for
    any member when the authenticated caller is also an org member, and 404
    for everyone else. The skill assumes the release manager running it is
    an NVIDIA org member, so this catches staff who keep their org membership
    private (the GitHub default) and commit from personal or
    ``users.noreply.github.com`` addresses.

    Outside callers see 404 even for actual members; in that case
    private-membership staff fall through to ``external`` in :func:`classify`.
    """
    try:
        result = subprocess.run(
            ["gh", "api", f"/orgs/{org}/members/{login}"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return result.returncode == 0


def classify(email: str, *, is_org_member: bool) -> str:
    """Apply the classification rules from ``contributor-attribution.md``.

    Returns ``"nvidia"`` if any positive-internal signal matches (NVIDIA
    email OR confirmed NVIDIA org membership), else ``"external"``. Order
    matters: first match wins.
    """
    email_lower = email.strip().lower()

    # 1. NVIDIA email. Free, deterministic, no API call.
    if email_lower.endswith("@nvidia.com") or email_lower.endswith("@exchange.nvidia.com"):
        return "nvidia"

    # 2. Confirmed NVIDIA org member. Catches staff who commit from personal
    #    or users.noreply.github.com addresses; required because GitHub
    #    defaults org membership to private and most staff don't toggle it
    #    public, so the public-profile / public-members APIs miss them.
    if is_org_member:
        return "nvidia"

    # 3. Everything else.
    return "external"


def aggregate_contributors(
    commits: list[dict],
    use_gh: bool,
    repo_slug: str,
    *,
    changelog_refs: dict[int, list[str]] | None = None,
    target_section_labels: tuple[str, ...] = (),
) -> list[dict]:
    """Aggregate per-contributor records from the commit list.

    Groups commits by ``(author_name, author_email)``. For each group:
    - resolves a GitHub login (email pattern -> gh api -> None);
    - classifies as nvidia/external (using NVIDIA org membership when ``gh``
      is available and a login was resolved);
    - records a ``pr_summary`` of (pr_number, subject) pairs.

    When ``changelog_refs`` is provided, each contributor is annotated with
    ``changelog_sections`` (the union of section labels their GH refs land in)
    and ``already_shipped`` (True iff at least one ref is in the changelog and
    none fall in ``target_section_labels``). Contributors without any GH refs
    in the changelog map have ``already_shipped=False`` (we cannot prove the
    work shipped previously, so we keep them as candidates).
    """
    org = repo_slug.split("/", 1)[0] if "/" in repo_slug else repo_slug

    by_key: dict[tuple[str, str], dict] = {}
    for commit in commits:
        key = (commit["author_name"], commit["author_email"])
        by_key.setdefault(
            key,
            {
                "name": commit["author_name"],
                "email": commit["author_email"],
                "commits": [],
            },
        )["commits"].append(commit)

    contributors = []
    for (name, email), info in by_key.items():
        # Resolve login from email pattern first (free, deterministic).
        login = resolve_gh_login_from_email(email)

        # Fallback to gh api on the FIRST commit only; cache by author key.
        if login is None and use_gh and not email.lower().endswith("@nvidia.com"):
            sample_sha = info["commits"][0]["sha"]
            login = gh_lookup_login_for_commit(sample_sha, repo_slug)

        # Org-membership lookup. Catches NVIDIA staff who commit from personal
        # or noreply addresses; relies on the caller being an NVIDIA org
        # member so the endpoint can see private memberships.
        is_org_member = False
        if login is not None and use_gh and not email.lower().endswith("@nvidia.com"):
            is_org_member = gh_check_org_membership(login, org)

        classification = classify(email, is_org_member=is_org_member)

        pr_summary: list[dict] = []
        for commit in info["commits"]:
            pr = extract_pr_number(commit["subject"])
            pr_summary.append(
                {
                    "pr": pr,
                    "subject": commit["subject"],
                    "sha": commit["sha"],
                }
            )

        record = {
            "name": name,
            "email": email,
            "gh_login": login,
            "classification": classification,
            "commit_count": len(info["commits"]),
            "pr_summary": pr_summary,
        }

        if changelog_refs is not None:
            sections: list[str] = []
            # already_shipped requires positive proof every commit was thanked
            # in a prior release: every commit must have at least one ref that
            # maps into the changelog, and none of those mappings may target
            # the current release. A commit with no changelog mapping at all
            # is unproven work and keeps the contributor as a current candidate.
            every_commit_in_prior = bool(target_section_labels)
            for commit in info["commits"]:
                commit_sections: set[str] = set()
                for ref in commit.get("gh_refs", ()) or ():
                    for section in changelog_refs.get(ref, ()):
                        commit_sections.add(section)
                        if section not in sections:
                            sections.append(section)
                if every_commit_in_prior:
                    in_prior = bool(commit_sections) and not any(s in target_section_labels for s in commit_sections)
                    if not in_prior:
                        every_commit_in_prior = False
            record["changelog_sections"] = sections
            record["already_shipped"] = every_commit_in_prior

        contributors.append(record)

    # Sort: external first (alphabetical by login), then nvidia.
    order = {"external": 0, "nvidia": 1}
    contributors.sort(key=lambda c: (order[c["classification"]], (c["gh_login"] or c["name"]).lower()))
    return contributors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Base ref (previous release tag)")
    parser.add_argument("--head", required=True, help="Head ref (release branch / tag)")
    parser.add_argument(
        "--repo-slug",
        default="NVIDIA/warp",
        help=(
            "GitHub repo slug for gh api lookups (default: NVIDIA/warp). The "
            "owning org (everything before the first '/') is also used for "
            "the org-membership check in classify()."
        ),
    )
    parser.add_argument(
        "--repo",
        default=None,
        help="Path to the git repo. Defaults to git rev-parse --show-toplevel from cwd.",
    )
    parser.add_argument(
        "--no-gh",
        action="store_true",
        help="Skip gh api lookups even when gh is available (useful for offline runs).",
    )
    parser.add_argument(
        "--include-cherry-picked",
        action="store_true",
        help=(
            "Include commits whose patch-id has an equivalent in <base> "
            "(default: filter via 'git log --cherry-pick --right-only')."
        ),
    )
    parser.add_argument(
        "--changelog",
        default=None,
        help=(
            "Path to CHANGELOG.md at <head>. When provided, each contributor "
            "is annotated with the CHANGELOG sections their GH refs land in, "
            "and 'already_shipped' is set when none of those sections are the "
            "target."
        ),
    )
    parser.add_argument(
        "--target-version",
        default=None,
        help=(
            "Target release version (e.g. '1.13.0'). Combined with the "
            "implicit 'Unreleased' label, defines which CHANGELOG sections "
            "count as the current target. Required when --changelog is set."
        ),
    )
    args = parser.parse_args()

    if args.changelog and not args.target_version:
        sys.exit("list_contributors: --changelog requires --target-version")
    if args.target_version and not args.changelog:
        sys.exit("list_contributors: --target-version requires --changelog")

    if args.repo is None:
        try:
            repo_str = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True, timeout=10).strip()
        except subprocess.CalledProcessError as exc:
            sys.exit(f"list_contributors: not in a git repo: {exc}")
        except subprocess.TimeoutExpired as exc:
            sys.exit(f"list_contributors: 'git rev-parse --show-toplevel' timed out: {exc}")
        repo = Path(repo_str)
    else:
        repo = Path(args.repo)

    use_gh = (not args.no_gh) and gh_available()

    drop_cherry_picked = not args.include_cherry_picked
    commits = get_commits_in_range(repo, args.base, args.head, drop_cherry_picked=drop_cherry_picked)
    changelog_added = get_changelog_added_gh_refs_for_range(
        repo,
        args.base,
        args.head,
        drop_cherry_picked=drop_cherry_picked,
    )

    # Enrich each commit with PR number, CHANGELOG-added refs, and merged refs.
    # The ``body`` from get_commits_in_range feeds GH-ref extraction and is
    # popped to keep the JSON output schema lean.
    for commit in commits:
        body = commit.pop("body")
        msg_refs = extract_gh_refs(commit["subject"], body)
        commit["pr"] = extract_pr_number(commit["subject"])
        changelog_refs_added = changelog_added.get(commit["sha"], [])
        commit["changelog_added_gh_refs"] = changelog_refs_added
        commit["gh_refs"] = sorted(set(msg_refs).union(changelog_refs_added))

    changelog_refs: dict[int, list[str]] | None = None
    target_section_labels: tuple[str, ...] = ()
    if args.changelog:
        changelog_path = Path(args.changelog)
        try:
            changelog_text = changelog_path.read_text()
        except OSError as exc:
            sys.exit(f"list_contributors: cannot read --changelog: {exc}")
        changelog_refs = parse_changelog_gh_refs(changelog_text)
        target_section_labels = (args.target_version, "Unreleased")

    contributors = aggregate_contributors(
        commits,
        use_gh=use_gh,
        repo_slug=args.repo_slug,
        changelog_refs=changelog_refs,
        target_section_labels=target_section_labels,
    )

    output = {
        "base": args.base,
        "head": args.head,
        "gh_used": use_gh,
        "drop_cherry_picked": drop_cherry_picked,
        "target_section_labels": list(target_section_labels) if target_section_labels else None,
        "commit_count": len(commits),
        "commits": commits,
        "contributors": contributors,
    }
    json.dump(output, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
