# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for indexing a thread tile by its logical lane."""

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import run_python_subprocess


def _run_block_dim_mismatch(mode):
    wp.config.mode = mode
    wp.config.enable_cpu_blocks = True
    wp.config.quiet = True
    block_dim = 8

    @wp.kernel(module="tile_block_dim_mismatch")
    def kernel(out: wp.array[wp.int32]):
        block_id, lane = wp.tid()
        values = wp.tile(wp.int32(42))
        out[block_id * block_dim + lane] = values[lane]

    out = wp.zeros(2 * block_dim, dtype=wp.int32, device="cpu")
    wp.launch(kernel, dim=(2, block_dim), outputs=[out], device="cpu", block_dim=block_dim)
    np.testing.assert_array_equal(out.numpy(), np.full(2 * block_dim, 42, dtype=np.int32))
    print("BLOCK_DIM_MISMATCH_OK", flush=True)


class TestTileBlockDimMismatch(unittest.TestCase):
    def test_thread_tile_uses_logical_block_dimension(self):
        result = run_python_subprocess(
            "import sys; import warp.tests.tile.test_tile_block_dim_mismatch as m; "
            "m._run_block_dim_mismatch(sys.argv[1])",
            wp.config.mode,
            timeout=60,
            hide_gpu=True,
        )
        self.assertEqual(result.returncode, 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
        self.assertIn("BLOCK_DIM_MISMATCH_OK", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
