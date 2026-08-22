#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Emit a JSON commit list for Warp release-candidate reporting.

Deterministic, stdlib-only. Enumerates commits in <base>..<head>, extracts
GH refs, identifies main equivalents via subject matching, and computes
soak-time metrics. No analysis, no network.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path
from types import MappingProxyType

# ASCII Unit Separator (U+001F). Not expected inside a git commit subject or
# author name, so safe to use as a field delimiter in log output.
_LOG_DELIM = "\x1f"
_LOG_FMT = _LOG_DELIM.join(["%H", "%s", "%an", "%cs"])


def _git(repo: Path, *args: str) -> str:
    """Run ``git -C repo args...`` and return stdout, exiting clearly on failure.

    On non-zero exit, surfaces git's stderr (which the raw
    ``subprocess.CalledProcessError`` hides). On missing git binary, reports
    that explicitly rather than the cryptic ``FileNotFoundError``.
    """
    stdout, _stderr = _git_capture(repo, *args)
    return stdout


def _git_capture(repo: Path, *args: str) -> tuple[str, str]:
    """Like ``_git`` but also returns stderr.

    Used by ``_resolve_sha`` to detect ambiguous-ref warnings, which ``git
    rev-parse --verify`` prints to stderr without affecting the exit code.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        sys.exit("list_commits: 'git' executable not found on PATH")
    if result.returncode != 0:
        sys.exit(
            f"list_commits: git {' '.join(args)} failed in {repo} (exit {result.returncode}): {result.stderr.strip()}"
        )
    return result.stdout, result.stderr


def get_commits_in_range(repo: Path, base: str, head: str) -> list[dict]:
    """Enumerate commits in base..head, oldest first, skipping merges.

    Returns one dict per commit with keys: sha, subject, author, committer_date.
    """
    stdout = _git(
        repo,
        "log",
        "--no-merges",
        "--reverse",
        f"--format={_LOG_FMT}",
        f"{base}..{head}",
    )
    commits = []
    for line in stdout.splitlines():
        if not line:
            continue
        sha, subject, author, committer_date = line.split(_LOG_DELIM, 3)
        commits.append(
            {
                "sha": sha,
                "subject": subject,
                "author": author,
                "committer_date": committer_date,
            }
        )
    return commits


# Word boundary before GH prevents matching "Regraph-42" etc.
_GH_REF_RE = re.compile(r"\bGH-(\d+)")


def extract_gh_refs(subject: str, body: str) -> list[int]:
    """Return sorted, deduplicated GH issue numbers referenced in subject+body."""
    nums = set()
    for text in (subject, body):
        for match in _GH_REF_RE.finditer(text):
            nums.add(int(match.group(1)))
    return sorted(nums)


def get_commit_body(repo: Path, sha: str) -> str:
    """Fetch the full commit message body (everything after the subject line)."""
    return _git(repo, "log", "-1", "--format=%b", sha).rstrip("\n")


def get_commit_files(repo: Path, sha: str) -> list[str]:
    """Return the list of file paths changed by a single commit.

    Uses --root so the initial commit (with no parent) also enumerates files.
    """
    stdout = _git(repo, "diff-tree", "--no-commit-id", "--name-only", "-r", "--root", sha)
    return [line for line in stdout.splitlines() if line]


def days_between(earlier_iso: str, later_iso: str) -> int:
    """Days from earlier_iso to later_iso (YYYY-MM-DD). May be negative."""
    d1 = date.fromisoformat(earlier_iso)
    d2 = date.fromisoformat(later_iso)
    return (d2 - d1).days


# Sentinel marking a subject whose main-side commit could not be unambiguously
# resolved (it appears more than once on main_ref). Wrapped in MappingProxyType
# so an accidental in-place mutation in find_main_equivalent doesn't corrupt
# every ambiguous slot at once.
_AMBIGUOUS = MappingProxyType({"_ambiguous": True})


def build_main_subject_index(repo: Path, base: str, main_ref: str) -> dict:
    """Build subject → {sha, committer_date} map from <base>..<main_ref> commits.

    Walks the range oldest-first (``git log --reverse``) so the first insertion
    per subject is the original occurrence on main. A subject that appears more
    than once is marked ambiguous via the ``_AMBIGUOUS`` sentinel; downstream,
    ``find_main_equivalent`` distinguishes ambiguous from missing so the
    consuming report can flag the two cases differently.
    """
    stdout = _git(
        repo,
        "log",
        "--no-merges",
        "--reverse",
        f"--format={_LOG_FMT}",
        f"{base}..{main_ref}",
    )
    index: dict[str, dict] = {}
    for line in stdout.splitlines():
        if not line:
            continue
        sha, subject, _author, committer_date = line.split(_LOG_DELIM, 3)
        if subject in index:
            # Second occurrence: mark ambiguous so downstream doesn't pick
            # whichever landed first as the canonical main-side commit.
            index[subject] = _AMBIGUOUS
        else:
            index[subject] = {"sha": sha, "committer_date": committer_date}
    return index


def find_main_equivalent(index: dict, subject: str) -> tuple[str, dict | None]:
    """Look up a head commit's main equivalent by exact subject.

    Returns one of:
    - ``("unique", {sha, committer_date})`` — exactly one match on main_ref.
    - ``("ambiguous", None)`` — subject appears more than once on main_ref;
      caller should flag rather than pick a candidate.
    - ``("missing", None)`` — subject not present on main_ref.
    """
    entry = index.get(subject)
    if entry is None:
        return "missing", None
    if entry.get("_ambiguous"):
        return "ambiguous", None
    return "unique", entry


def _resolve_sha(repo: Path, ref: str) -> str:
    """Resolve a git ref to its commit SHA, peeling annotated tags.

    Fails on ambiguous refs: ``git rev-parse --verify`` exits 0 but prints a
    warning to stderr when a name resolves to both a tag and a branch. Treat
    any non-empty stderr from ``rev-parse`` as a hard error so the report
    doesn't silently use one of two candidate SHAs.
    """
    # Plain `rev-parse <annotated-tag>` returns the tag-object SHA, not the
    # commit SHA. `^{commit}` peels through tag objects so v1.12.1 etc. resolve
    # to the underlying commit (which is what gist/report URLs need).
    stdout, stderr = _git_capture(repo, "rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}")
    if stderr.strip():
        sys.exit(f"list_commits: ambiguous ref {ref!r}: {stderr.strip()}")
    return stdout.strip()


def build_commit_entry(
    repo: Path,
    commit: dict,
    subject_index: dict,
    report_date: str,
) -> dict:
    """Enrich a commit dict with files, gh_refs, days, main_equivalent."""
    body = get_commit_body(repo, commit["sha"])
    gh_refs = extract_gh_refs(commit["subject"], body)
    files = get_commit_files(repo, commit["sha"])
    days_since_merge = days_between(commit["committer_date"], report_date)
    if days_since_merge < 0:
        sys.exit(
            f"list_commits: --report-date {report_date} predates committer_date "
            f"{commit['committer_date']} for commit {commit['sha'][:12]}; "
            "refusing to emit negative days_since_merge."
        )
    state, main_match = find_main_equivalent(subject_index, commit["subject"])
    if main_match is not None:
        main_sha = main_match["sha"]
        days_in_main = days_between(main_match["committer_date"], report_date)
    else:
        main_sha = None
        days_in_main = None
    return {
        "sha": commit["sha"],
        "subject": commit["subject"],
        "author": commit["author"],
        "committer_date": commit["committer_date"],
        "days_since_merge": days_since_merge,
        "main_equivalent_sha": main_sha,
        "main_match_state": state,
        "days_in_main": days_in_main,
        "files": files,
        "gh_refs": gh_refs,
    }


def parse_args(argv):
    """Parse CLI args; see ``--help``."""
    p = argparse.ArgumentParser(
        description="Emit commit metadata as JSON for RC report generation.",
    )
    p.add_argument("--base", required=True, help="Base git ref (e.g. v1.12.1)")
    p.add_argument("--head", required=True, help="Head git ref (e.g. upstream/release-1.13)")
    p.add_argument("--report-date", required=True, help="Report date YYYY-MM-DD")
    p.add_argument(
        "--main-ref",
        default="upstream/main",
        help="Main-branch ref for cherry-pick detection (default: upstream/main)",
    )
    return p.parse_args(argv)


def main(argv=None):
    """Build the commit-list JSON for <base>..<head> and print it to stdout."""
    args = parse_args(argv if argv is not None else sys.argv[1:])

    # Validate --report-date upfront rather than discovering the problem
    # deep inside the per-commit loop.
    try:
        date.fromisoformat(args.report_date)
    except ValueError:
        sys.exit(f"list_commits: --report-date must be YYYY-MM-DD, got {args.report_date!r}")

    repo = Path.cwd()
    base_sha = _resolve_sha(repo, args.base)
    head_sha = _resolve_sha(repo, args.head)
    main_sha = _resolve_sha(repo, args.main_ref)

    commits = get_commits_in_range(repo, args.base, args.head)
    empty_range = not commits
    if empty_range:
        sys.stderr.write(
            f"list_commits: warning: {args.base}..{args.head} contains no commits. "
            f"Resolved: base={base_sha[:12]}, head={head_sha[:12]}. "
            "Check that --base is an ancestor of --head.\n"
        )

    subject_index = build_main_subject_index(repo, args.base, args.main_ref)
    empty_main_index = not subject_index
    if empty_main_index and not empty_range:
        # No commits on main between base and main_ref means every head commit
        # will resolve as main_match_state="missing". Surface this rather than
        # let the consumer interpret it as "nothing has soaked on main yet."
        sys.stderr.write(
            f"list_commits: warning: {args.base}..{args.main_ref} contains no commits; "
            f"main_match_state will be 'missing' for every commit in range. "
            f"Resolved: base={base_sha[:12]}, main_ref={main_sha[:12]}. "
            "Check that --main-ref is correct and not equal to --base.\n"
        )

    entries = [build_commit_entry(repo, c, subject_index, args.report_date) for c in commits]

    out = {
        "resolved": {
            "report_date": args.report_date,
            "base": {"ref": args.base, "sha": base_sha},
            "head": {"ref": args.head, "sha": head_sha},
            "main_ref": {"ref": args.main_ref, "sha": main_sha},
            "empty_range": empty_range,
            "empty_main_index": empty_main_index,
        },
        "commits": entries,
    }
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
