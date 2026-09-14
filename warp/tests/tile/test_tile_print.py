# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for printing tiles."""

import unittest

import warp as wp
from warp.tests.unittest_utils import add_function_test, get_test_devices, run_python_subprocess


@wp.kernel(enable_backward=False)
def print_register_tile(x: wp.array[wp.int32]):
    tile = wp.tile_load(x, shape=(8,), storage="register")
    wp.print(tile)


@wp.kernel(enable_backward=False, module="test_shared_and_register_tile_print")
def print_shared_and_register_tiles():
    shared_tile = wp.tile_ones(shape=(4, 3), dtype=float, storage="shared")
    register_tile = wp.tile_ones(shape=(4, 3), dtype=float)

    wp.print(shared_tile)
    wp.print(register_tile)


def test_shared_and_register_tile_print_executes(test, device):
    """Compile and execute shared and register tile printing."""
    wp.launch_tiled(print_shared_and_register_tiles, dim=1, block_dim=64, device=device)
    wp.synchronize_device(device)


def _run_register_tile_cpu_blocks(mode):
    wp.config.mode = mode
    wp.config.enable_cpu_blocks = True
    wp.config.quiet = True

    x = wp.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=wp.int32, device="cpu")
    wp.launch_tiled(print_register_tile, dim=1, inputs=[x], block_dim=4, device="cpu")
    wp.synchronize_device("cpu")


class TestTilePrint(unittest.TestCase):
    def test_register_tile_cpu_blocks(self):
        result = run_python_subprocess(
            "import sys; import warp.tests.tile.test_tile_print as m; m._run_register_tile_cpu_blocks(sys.argv[1])",
            wp.config.mode,
            timeout=60,
            hide_gpu=True,
        )
        self.assertEqual(result.returncode, 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
        self.assertIn("[1 2 3 4 5 6 7 8] = tile(shape=(8), storage=register)", result.stdout)


add_function_test(
    TestTilePrint,
    "test_shared_and_register_tile_print_executes",
    test_shared_and_register_tile_print_executes,
    devices=get_test_devices(),
    check_output=False,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
