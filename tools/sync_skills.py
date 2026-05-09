#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synchronize project skills for Claude Code and Codex CLI.

Warp keeps project-level skills under both ``.claude/skills`` and ``.codex/skills`` so each tool can discover the
same release-management workflows. This script plans and applies changes between those trees without guessing when
both sides have diverged.

The command has three operating modes:

- Auto mode compares each side against ``HEAD`` and copies the side that changed.
- Forced mode (``--from``) treats one side as the source of truth and mirrors it to the other side.
- Check mode (``--check``) prints the plan and never mutates the filesystem.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Side:
    """Metadata for one skill tree."""

    name: str
    """Short display name used in diagnostics."""

    root: Path
    """Filesystem root of the skill tree."""

    git_prefix: str
    """Repository-relative prefix used when reading the tree from ``HEAD``."""


@dataclass(frozen=True)
class Action:
    """A single planned sync operation."""

    kind: str
    """Operation type: ``copy`` or ``delete``."""

    src: str
    """Side that won the conflict decision."""

    dst: str
    """Side that will be updated."""

    rel: str
    """Skill-tree-relative file path."""

    reason: str
    """Human-readable explanation for why the action was selected."""


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Sync .claude/skills and .codex/skills with safe bidirectional drift detection.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report drift and exit 1 if a sync would change files.",
    )
    parser.add_argument(
        "--from",
        dest="force_source",
        choices=("claude", "codex"),
        help="Force one side to be the source of truth.",
    )
    parser.add_argument(
        "--prefer-mtime",
        action="store_true",
        help="Resolve both-sides-changed conflicts by choosing the newer file mtime.",
    )
    return parser.parse_args()


def git_root() -> Path:
    """Return the repository root for this script, independent of the caller's current directory."""

    proc = subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--show-toplevel"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        print(proc.stderr.strip() or "failed to resolve git root", file=sys.stderr)
        raise SystemExit(1)
    return Path(proc.stdout.strip())


def list_files(base: Path) -> dict[str, Path]:
    """Return non-cache files below ``base``, keyed by slash-separated paths relative to ``base``.

    A missing tree is treated as empty. That lets forced mode seed a destination tree while validation still ensures
    the chosen source tree exists.
    """

    if not base.exists():
        return {}
    out: dict[str, Path] = {}
    for path in base.rglob("*"):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        if path.is_file() or path.is_symlink():
            out[path.relative_to(base).as_posix()] = path
    return out


def read_current(path: Path | None) -> bytes | None:
    """Read a working-tree file, returning ``None`` when the file is absent."""

    if path is None or not path.exists():
        return None
    return path.read_bytes()


def read_head(root: Path, git_path: str) -> bytes | None:
    """Read a repository path from ``HEAD``, returning ``None`` for files not tracked there."""

    proc = subprocess.run(
        ["git", "-C", str(root), "show", f"HEAD:{git_path}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout


def changed(current: bytes | None, base: bytes | None) -> bool:
    """Return whether a working-tree value differs from its ``HEAD`` value."""

    return current != base


def mtime(path: Path | None) -> float:
    """Return a path's mtime, or ``0.0`` when it does not exist."""

    if path is not None and path.exists():
        return path.stat().st_mtime
    return 0.0


def newest_side(rel: str, claude_files: dict[str, Path], codex_files: dict[str, Path]) -> str | None:
    """Return the side with the newer file for ``rel`` when mtime can break a conflict."""

    claude_path = claude_files.get(rel)
    codex_path = codex_files.get(rel)
    if claude_path is None or codex_path is None:
        return None

    claude_mtime = mtime(claude_path)
    codex_mtime = mtime(codex_path)
    if claude_mtime == codex_mtime:
        return None
    return "claude" if claude_mtime > codex_mtime else "codex"


def build_action(
    winner: str,
    rel: str,
    reason: str,
    claude_files: dict[str, Path],
    codex_files: dict[str, Path],
) -> Action:
    """Build the copy/delete action needed to make the losing side match the winning side."""

    loser = "codex" if winner == "claude" else "claude"
    winner_files = claude_files if winner == "claude" else codex_files
    if rel in winner_files:
        return Action("copy", winner, loser, rel, reason)
    return Action("delete", winner, loser, rel, reason)


def plan_forced(source: str, claude_files: dict[str, Path], codex_files: dict[str, Path]) -> list[Action]:
    """Plan a one-way mirror from ``source`` to the other skill tree."""

    dest = "codex" if source == "claude" else "claude"
    source_files = claude_files if source == "claude" else codex_files
    dest_files = codex_files if source == "claude" else claude_files
    actions: list[Action] = []

    for rel in sorted(set(source_files) | set(dest_files)):
        source_bytes = read_current(source_files.get(rel))
        dest_bytes = read_current(dest_files.get(rel))
        if source_bytes == dest_bytes:
            continue
        kind = "copy" if rel in source_files else "delete"
        actions.append(Action(kind, source, dest, rel, f"forced {source} source"))

    return actions


def plan_auto(
    root: Path,
    sides: dict[str, Side],
    claude_files: dict[str, Path],
    codex_files: dict[str, Path],
    prefer_mtime: bool,
) -> tuple[list[Action], list[str]]:
    """Plan a bidirectional sync from working-tree drift relative to ``HEAD``.

    Auto mode is conservative: when exactly one side has changed relative to ``HEAD``, that side wins. If both sides
    changed differently, the script reports a conflict unless ``--prefer-mtime`` can choose a winner. This keeps the
    mirror operation from silently overwriting independent edits.
    """

    actions: list[Action] = []
    conflicts: list[str] = []

    for rel in sorted(set(claude_files) | set(codex_files)):
        claude_current = read_current(claude_files.get(rel))
        codex_current = read_current(codex_files.get(rel))
        if claude_current == codex_current:
            continue

        # ``None`` represents "missing" on either side, both in the working tree and in HEAD. A create/delete is just
        # another changed value, so the same logic handles additions, edits, and removals.
        claude_base = read_head(root, f"{sides['claude'].git_prefix}/{rel}")
        codex_base = read_head(root, f"{sides['codex'].git_prefix}/{rel}")
        claude_changed = changed(claude_current, claude_base)
        codex_changed = changed(codex_current, codex_base)

        if claude_changed and not codex_changed:
            actions.append(build_action("claude", rel, "Claude side changed", claude_files, codex_files))
            continue
        if codex_changed and not claude_changed:
            actions.append(build_action("codex", rel, "Codex side changed", claude_files, codex_files))
            continue

        if prefer_mtime:
            winner = newest_side(rel, claude_files, codex_files)
            if winner is not None:
                actions.append(build_action(winner, rel, f"{winner} side has newer mtime", claude_files, codex_files))
                continue

        if claude_changed and codex_changed:
            conflicts.append(f"{rel}: both sides changed differently")
        else:
            conflicts.append(f"{rel}: trees differ but neither side differs from HEAD; use --prefer-mtime or --from")

    return actions, conflicts


def apply_action(action: Action, sides: dict[str, Side]) -> None:
    """Apply one planned sync action to the destination tree."""

    src_path = sides[action.src].root / action.rel
    dst_path = sides[action.dst].root / action.rel

    if action.kind == "delete":
        if dst_path.exists() or dst_path.is_symlink():
            dst_path.unlink()
        return

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst_path)


def prune_empty_dirs(base: Path) -> None:
    """Remove empty directories left behind by delete actions."""

    if not base.exists():
        return
    for dirpath, _, _ in os.walk(base, topdown=False):
        path = Path(dirpath)
        if path == base:
            continue
        # ``os.walk()`` reports each parent's original ``dirnames`` snapshot. Let ``rmdir()`` decide emptiness after
        # earlier child removals instead.
        try:
            path.rmdir()
        except OSError:
            pass


def describe(action: Action) -> str:
    """Return the human-readable form printed for a planned or applied action."""

    arrow = f"{action.src} -> {action.dst}"
    if action.kind == "copy":
        return f"copy {arrow}: {action.rel} ({action.reason})"
    return f"delete from {action.dst}: {action.rel} ({action.reason})"


def validate_trees(args: argparse.Namespace, claude: Path, codex: Path) -> int:
    """Validate that the requested mode has the skill trees it needs."""

    if args.force_source == "claude" and not claude.is_dir():
        print(f"no source tree at {claude}", file=sys.stderr)
        return 1
    if args.force_source == "codex" and not codex.is_dir():
        print(f"no source tree at {codex}", file=sys.stderr)
        return 1
    if args.force_source is None and (not claude.is_dir() or not codex.is_dir()):
        print(
            f"auto mode requires both {claude} and {codex} to exist; use --from claude or --from codex to seed one side",
            file=sys.stderr,
        )
        return 1
    return 0


def run(args: argparse.Namespace) -> int:
    """Plan and optionally apply a skill-tree sync."""

    root = git_root()
    claude = root / ".claude" / "skills"
    codex = root / ".codex" / "skills"
    if status := validate_trees(args, claude, codex):
        return status

    sides = {
        "claude": Side("claude", claude, ".claude/skills"),
        "codex": Side("codex", codex, ".codex/skills"),
    }

    # ``--check`` must be read-only: even seeding a missing destination tree would surprise CI and review tools.
    if not args.check:
        codex.mkdir(parents=True, exist_ok=True)
    claude_files = list_files(claude)
    codex_files = list_files(codex)

    if args.force_source is not None:
        actions = plan_forced(args.force_source, claude_files, codex_files)
        conflicts: list[str] = []
    else:
        actions, conflicts = plan_auto(root, sides, claude_files, codex_files, args.prefer_mtime)

    if conflicts:
        print("conflicts detected:", file=sys.stderr)
        for conflict in conflicts:
            print(f"  {conflict}", file=sys.stderr)
        return 2

    if not actions:
        print("in sync")
        return 0

    if args.check:
        print("drift detected:")
        for action in actions:
            print(f"  would {describe(action)}")
        return 1

    for action in actions:
        apply_action(action, sides)
        print(describe(action))
    prune_empty_dirs(claude)
    prune_empty_dirs(codex)
    print("skills synced")
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
