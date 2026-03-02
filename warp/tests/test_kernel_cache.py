# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import tempfile
import unittest
from unittest.mock import patch

import warp._src.build
import warp.config


class TestKernelCache(unittest.TestCase):
    def setUp(self):
        self._original_cache_dir = warp.config.kernel_cache_dir
        self._original_env = os.environ.pop("WARP_CACHE_PATH", None)

    def tearDown(self):
        warp.config.kernel_cache_dir = self._original_cache_dir
        if self._original_env is None:
            os.environ.pop("WARP_CACHE_PATH", None)
        else:
            os.environ["WARP_CACHE_PATH"] = self._original_env

    def test_cache_path_includes_version(self):
        """init_kernel_cache appends the Warp version to user-supplied paths."""
        with tempfile.TemporaryDirectory() as tmp:
            warp._src.build.init_kernel_cache(path=tmp)
            expected = os.path.join(os.path.realpath(tmp), warp.config.version)
            self.assertEqual(warp.config.kernel_cache_dir, expected)
            self.assertTrue(os.path.isdir(expected))

    def test_cache_env_var_includes_version(self):
        """WARP_CACHE_PATH also gets a version subdirectory."""
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["WARP_CACHE_PATH"] = tmp
            warp._src.build.init_kernel_cache()
            expected = os.path.join(os.path.realpath(tmp), warp.config.version)
            self.assertEqual(warp.config.kernel_cache_dir, expected)
            self.assertTrue(os.path.isdir(expected))

    def test_stale_artifacts_warning(self):
        """Warn when the unversioned base directory contains stale wp_ artifacts."""
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "wp_stale_module"))
            with patch("warp._src.utils.warn") as mock_warn:
                warp._src.build.init_kernel_cache(path=tmp)
                mock_warn.assert_called_once()
                self.assertIn("previous Warp version", mock_warn.call_args[0][0])

    def test_no_stale_artifacts_warning(self):
        """No warning when the base directory is clean."""
        with tempfile.TemporaryDirectory() as tmp:
            with patch("warp._src.utils.warn") as mock_warn:
                warp._src.build.init_kernel_cache(path=tmp)
                mock_warn.assert_not_called()

    def test_default_cache_path_includes_version(self):
        """The default cache path (no env var, no explicit path) includes the version."""
        warp._src.build.init_kernel_cache()
        self.assertIn(warp.config.version, warp.config.kernel_cache_dir)


if __name__ == "__main__":
    unittest.main(verbosity=2)
