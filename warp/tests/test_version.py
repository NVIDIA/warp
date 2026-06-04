# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import types
import unittest

import warp as wp
from warp._src.context import _verify_library_version, get_warp_clang_version, get_warp_version


class TestVersion(unittest.TestCase):
    """Tests for native library version verification using string comparison."""

    def test_get_warp_version_returns_string(self):
        """Test that get_warp_version() returns a string."""
        version = get_warp_version()
        self.assertIsInstance(version, str)
        self.assertRegex(version, r"^\d+\.\d+\.\d+")

    def test_get_warp_clang_version_returns_string(self):
        """Test that get_warp_clang_version() returns a string."""
        version = get_warp_clang_version()
        self.assertIsInstance(version, str)
        self.assertRegex(version, r"^\d+\.\d+\.\d+")

    def test_dll_versions_match_python_version(self):
        """Test that native library versions match Python package version exactly."""
        python_version = wp.config.version
        warp_version = get_warp_version()
        warp_clang_version = get_warp_clang_version()

        self.assertEqual(
            warp_version, python_version, f"Warp library version {warp_version} != Python version {python_version}"
        )
        self.assertEqual(
            warp_clang_version,
            python_version,
            f"warp-clang library version {warp_clang_version} != Python version {python_version}",
        )

    def test_version_format_validation(self):
        """Test that versions follow PEP 440 versioning format."""
        # PEP 440 pattern: [EPOCH!]MAJOR.MINOR.PATCH[{a|b|rc}N][.postN][.devN][+LOCAL]
        # Examples: 1.10.0, 1.10.0.dev0, 1.10.0.dev20251017, 1.10.0a1, 1.10.0rc1, 1!1.10.0
        pep440_pattern = (
            r"^(\d+!)?"  # Optional epoch
            r"\d+\.\d+\.\d+"  # Major.minor.patch
            r"(\.?(a|alpha|b|beta|rc)\.?\d+)?"  # Optional alpha/beta/rc (with or without dots)
            r"(\.post\d+)?"  # Optional post release
            r"(\.dev\d+)?"  # Optional dev release
            r"(\+[a-zA-Z0-9.]+)?$"  # Optional local version identifier
        )

        warp_version = get_warp_version()
        warp_clang_version = get_warp_clang_version()

        self.assertIsNotNone(re.match(pep440_pattern, warp_version))
        self.assertIsNotNone(re.match(pep440_pattern, warp_clang_version))


class TestVerifyLibraryVersion(unittest.TestCase):
    """Tests for the ``_verify_library_version`` helper that enforces native library versions."""

    @staticmethod
    def _lib(**symbols):
        """Stand in for a loaded native library exposing the given version functions."""
        return types.SimpleNamespace(**symbols)

    def test_matching_version_does_not_raise(self):
        """A version equal to the expected one passes without raising."""
        _verify_library_version(self._lib(wp_version=lambda: b"1.2.3"), "warp", "wp_version", "1.2.3")

    def test_mismatched_version_raises(self):
        """A decoded-but-different version raises RuntimeError naming both versions."""
        lib = self._lib(wp_warp_clang_version=lambda: b"9.9.9")
        with self.assertRaisesRegex(RuntimeError, "Version mismatch") as cm:
            _verify_library_version(lib, "warp-clang", "wp_warp_clang_version", "1.2.3")
        message = str(cm.exception)
        self.assertIn("1.2.3", message)
        self.assertIn("9.9.9", message)
        self.assertIn("multiple Warp installations", message)

    def test_missing_symbol_raises(self):
        """A missing version symbol raises, naming the symbol and omitting the expected version."""
        with self.assertRaisesRegex(RuntimeError, "does not export") as cm:
            _verify_library_version(self._lib(), "warp", "wp_version", "1.2.3")
        message = str(cm.exception)
        self.assertIn("wp_version", message)
        self.assertNotIn("1.2.3", message)

    def test_null_return_raises(self):
        """A NULL (None) return raises an empty-version error, omitting the expected version."""
        with self.assertRaisesRegex(RuntimeError, "empty version") as cm:
            _verify_library_version(self._lib(wp_version=lambda: None), "warp", "wp_version", "1.2.3")
        self.assertNotIn("1.2.3", str(cm.exception))

    def test_empty_return_raises(self):
        """An empty version string raises an empty-version error, omitting the expected version."""
        with self.assertRaisesRegex(RuntimeError, "empty version") as cm:
            _verify_library_version(self._lib(wp_version=lambda: b""), "warp", "wp_version", "1.2.3")
        self.assertNotIn("1.2.3", str(cm.exception))

    def test_call_exception_is_wrapped_and_chained(self):
        """Any exception from the call/decode is wrapped (no expected version) and chained via __cause__."""

        def boom():
            raise ValueError("native call failed")

        with self.assertRaisesRegex(RuntimeError, "Failed to read") as cm:
            _verify_library_version(self._lib(wp_version=boom), "warp", "wp_version", "1.2.3")
        self.assertIsInstance(cm.exception.__cause__, ValueError)
        self.assertNotIn("1.2.3", str(cm.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
