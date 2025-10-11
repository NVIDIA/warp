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
        python_version = wp._src.config.version
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
        """Test that versions follow semantic versioning format."""
        # Pattern: MAJOR.MINOR.PATCH[.SUFFIX][+LOCALVERSION]
        semver_pattern = r"^\d+\.\d+\.\d+(?:\.\w+\d+)?(?:\+\S+)?$"

        warp_version = get_warp_version()
        warp_clang_version = get_warp_clang_version()

        self.assertIsNotNone(re.match(semver_pattern, warp_version))
        self.assertIsNotNone(re.match(semver_pattern, warp_clang_version))


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
