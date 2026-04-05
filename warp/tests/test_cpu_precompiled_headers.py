# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for CPU precompiled header support."""

import glob
import os
import tempfile
import unittest

import numpy as np

import warp as wp
import warp._src.build

_original_use_precompiled_headers = None


def setUpModule():
    global _original_use_precompiled_headers
    _original_use_precompiled_headers = wp.config.use_precompiled_headers
    wp.config.use_precompiled_headers = True


def tearDownModule():
    wp.config.use_precompiled_headers = _original_use_precompiled_headers


def _use_temp_cache():
    """Context manager that switches to a temporary kernel cache for cold compilation."""
    return _TempCacheContext()


class _TempCacheContext:
    def __enter__(self):
        self._original_cache_dir = wp.config.kernel_cache_dir
        self._tmp = tempfile.TemporaryDirectory(prefix="wp_pch_test_")
        warp._src.build.init_kernel_cache(path=self._tmp.name)
        return self

    def __exit__(self, *args):
        wp.config.kernel_cache_dir = self._original_cache_dir
        self._tmp.cleanup()


class TestCpuPrecompiledHeaders(unittest.TestCase):
    def test_basic(self):
        """Verify that a kernel compiles correctly with PCH and a .pch file is generated."""

        with _use_temp_cache():

            @wp.kernel(module="unique")
            def scale(a: wp.array[float]):
                tid = wp.tid()
                a[tid] = float(tid) * 2.0

            a = wp.zeros(10, dtype=float, device="cpu")
            wp.launch(scale, dim=10, inputs=[a], device="cpu")

            expected = np.arange(10, dtype=np.float32) * 2.0
            np.testing.assert_allclose(a.numpy(), expected)

            # Verify PCH was actually generated
            pch_dir = wp._src.context.runtime.get_clang_pch_dir()
            self.assertIsNotNone(pch_dir)
            pch_files = glob.glob(os.path.join(pch_dir, "*.pch"))
            self.assertGreater(len(pch_files), 0, "No .pch file generated despite use_precompiled_headers=True")

    def test_disabled(self):
        """Verify that compilation works when PCH is disabled."""
        old_val = wp.config.use_precompiled_headers
        try:
            wp.config.use_precompiled_headers = False

            with _use_temp_cache():

                @wp.kernel(module="unique")
                def offset(a: wp.array[float]):
                    tid = wp.tid()
                    a[tid] = float(tid) + 1.0

                a = wp.zeros(5, dtype=float, device="cpu")
                wp.launch(offset, dim=5, inputs=[a], device="cpu")

                np.testing.assert_allclose(a.numpy(), [1, 2, 3, 4, 5])
        finally:
            wp.config.use_precompiled_headers = old_val

    def test_fallback(self):
        """Verify graceful fallback when PCH file is corrupted."""
        pch_dir = wp._src.context.runtime.get_clang_pch_dir()
        if pch_dir is None:
            self.skipTest("CPU PCH dir not available")

        with _use_temp_cache():
            # Snapshot PCH files before compiling so we can identify which
            # files this test created (the PCH dir may already contain files
            # from prior test suites reusing the same worker process).
            pre_existing = set(glob.glob(os.path.join(pch_dir, "*.pch")))

            # Ensure PCH exists by compiling something first
            @wp.kernel(module="unique")
            def iota(a: wp.array[float]):
                tid = wp.tid()
                a[tid] = float(tid)

            a = wp.zeros(3, dtype=float, device="cpu")
            wp.launch(iota, dim=3, inputs=[a], device="cpu")

            post_compile = set(glob.glob(os.path.join(pch_dir, "*.pch")))
            new_pch_files = sorted(post_compile - pre_existing)

            # iota may reuse a pre-existing PCH (same flags).  Only target
            # block_dim=1 files (CPU launches always use block_dim=1) to
            # avoid corrupting unrelated PCH files left by prior test suites.
            target_files = new_pch_files or sorted(
                f for f in post_compile if os.path.basename(f).startswith("builtin_bd1")
            )
            self.assertGreater(len(target_files), 0, "No PCH files found to corrupt")

            for pch_file in target_files:
                with open(pch_file, "wb") as f:
                    f.write(b"not a valid AST file")

            # Next compilation should detect the invalid PCH and fall back
            @wp.kernel(module="unique")
            def triple(b: wp.array[float]):
                tid = wp.tid()
                b[tid] = float(tid) * 3.0

            b = wp.zeros(3, dtype=float, device="cpu")
            wp.launch(triple, dim=3, inputs=[b], device="cpu")

            np.testing.assert_allclose(b.numpy(), [0, 3, 6])

            # Verify the corrupt PCH files were cleaned up
            for pch_file in target_files:
                self.assertFalse(os.path.exists(pch_file), f"Corrupt PCH not cleaned up: {pch_file}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
