#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the release-audit public API diff helper."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("diff_public_api.py")


def run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")


def commit_all(repo: Path, message: str) -> None:
    run(["git", "add", "."], repo)
    run(["git", "commit", "-m", message], repo)


class DiffPublicApiTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name)

        run(["git", "init"], self.repo)
        run(["git", "config", "user.email", "test@example.com"], self.repo)
        run(["git", "config", "user.name", "Test User"], self.repo)

    def write_base_api(self) -> None:
        write(
            self.repo / "pkg" / "__init__.py",
            """
            from pkg._impl import keep as keep
            from pkg._impl import inserted as inserted
            from pkg._impl import make_kw as make_kw
            """,
        )
        write(
            self.repo / "pkg" / "_impl.py",
            """
            def keep(a, b=1):
                return a + b

            def inserted(a, b=1):
                return a + b

            def make_kw(a, b=1):
                return a + b
            """,
        )
        write(
            self.repo / "pkg" / "__init__.pyi",
            """
            from typing import overload

            def keep(a: int, b: int = ...) -> int: ...
            def removed_stub(a: int) -> int: ...

            @overload
            def overloaded(x: int) -> int: ...
            @overload
            def overloaded(x: str) -> str: ...

            class StubClass:
                old_attr: int
                def kept(self, x: int) -> int: ...
                def removed_method(self) -> None: ...
            """,
        )

    def write_head_api(self) -> None:
        write(
            self.repo / "pkg" / "__init__.py",
            """
            from pkg._impl import keep as keep
            from pkg._impl import inserted as inserted
            from pkg._impl import make_kw as make_kw
            """,
        )
        write(
            self.repo / "pkg" / "_impl.py",
            """
            def keep(a, b=1):
                return a + b

            def inserted(a, new=False, b=1):
                return a + b

            def make_kw(a, *, b=1):
                return a + b
            """,
        )
        write(
            self.repo / "pkg" / "__init__.pyi",
            """
            from typing import overload

            def keep(a: int, b: int = ...) -> int: ...

            @overload
            def overloaded(x: int) -> int: ...

            class StubClass:
                def kept(self, x: int) -> int: ...
            """,
        )

    def run_helper(self) -> dict:
        proc = run(
            [
                sys.executable,
                str(SCRIPT),
                "--repo",
                str(self.repo),
                "--base",
                "base",
                "--head",
                "HEAD",
                "--module",
                "pkg",
            ],
            self.repo,
        )
        return json.loads(proc.stdout)

    def test_reports_signature_breaks_and_stub_removals(self) -> None:
        self.write_base_api()
        commit_all(self.repo, "base")
        run(["git", "tag", "base"], self.repo)

        self.write_head_api()
        commit_all(self.repo, "head")

        report = self.run_helper()
        changes = {(change["kind"], change["symbol"], tuple(change["reasons"])) for change in report["changes"]}

        self.assertIn(
            (
                "signature_change",
                "pkg.inserted",
                ("inserted_positional_parameter",),
            ),
            changes,
        )
        self.assertIn(
            (
                "signature_change",
                "pkg.make_kw",
                ("positional_to_keyword_only",),
            ),
            changes,
        )
        self.assertIn(
            ("public_stub_removal", "pkg.removed_stub", ("removed_stub_symbol",)),
            changes,
        )
        self.assertIn(
            ("public_stub_removal", "pkg.overloaded", ("removed_stub_overload",)),
            changes,
        )
        self.assertIn(
            ("public_stub_removal", "pkg.StubClass.old_attr", ("removed_stub_attribute",)),
            changes,
        )
        self.assertIn(
            ("public_stub_removal", "pkg.StubClass.removed_method", ("removed_stub_symbol",)),
            changes,
        )


if __name__ == "__main__":
    unittest.main()
