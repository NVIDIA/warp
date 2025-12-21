# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import unittest

import warp as wp
from warp._src.context import get_warp_clang_version, get_warp_version


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


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
