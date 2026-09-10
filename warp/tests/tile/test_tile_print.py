# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for printing tiles."""

import subprocess
import sys
import unittest

import warp as wp


@wp.kernel(enable_backward=False)
def print_register_tile(x: wp.array[wp.int32]):
    tile = wp.tile_load(x, shape=(8,), storage="register")
    wp.print(tile)


def _run_register_tile_cpu_blocks():
    wp.config.enable_cpu_blocks = True
    wp.config.quiet = True

    x = wp.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=wp.int32, device="cpu")
    wp.launch_tiled(print_register_tile, dim=1, inputs=[x], block_dim=4, device="cpu")
    wp.synchronize_device("cpu")


class TestTilePrint(unittest.TestCase):
    def test_register_tile_cpu_blocks(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import warp.tests.tile.test_tile_print as m; m._run_register_tile_cpu_blocks()",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
        self.assertIn("[1 2 3 4 5 6 7 8] = tile(shape=(8), storage=register)", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
