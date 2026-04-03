# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

wp.init()  # For wp._src.context.runtime.core.wp_is_mathdx_enabled()

TILE_M = wp.constant(8)
TILE_N = wp.constant(4)
TILE_K = wp.constant(8)

# num threads per-tile
TILE_DIM = 32


@wp.kernel()
def tile_math_matmul_kernel(
    ga: wp.array2d(dtype=wp.float16), gb: wp.array2d(dtype=wp.float32), gc: wp.array2d(dtype=wp.float64)
):
    i, j = wp.tid()
    a = wp.tile_load(ga, shape=(TILE_M, TILE_K), offset=(i * TILE_M, j * TILE_K))
    b = wp.tile_load(gb, shape=(TILE_K, TILE_N), offset=(i * TILE_K, j * TILE_N))
    c = wp.tile_load(gc, shape=(TILE_M, TILE_N), offset=(i * TILE_M, j * TILE_N))
    wp.tile_matmul(a, b, c, alpha=0.5, beta=-1.3)
    wp.tile_store(gc, c, offset=(i * TILE_M, j * TILE_N))


def test_tile_math_matmul(test, device):
    rng = np.random.default_rng(42)

    A = rng.random((TILE_M, TILE_K), dtype=np.float64).astype(np.float16)
    B = rng.random((TILE_K, TILE_N), dtype=np.float32)
    C = rng.random((TILE_M, TILE_N), dtype=np.float64)

    A_wp = wp.array(A, requires_grad=True, device=device)
    B_wp = wp.array(B, requires_grad=True, device=device)
    C_wp = wp.array(C, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_math_matmul_kernel,
            dim=[1, 1],
            inputs=[A_wp, B_wp, C_wp],
            block_dim=TILE_DIM,
            device=device,
        )

    # verify forward pass
    assert_np_equal(C_wp.numpy(), 0.5 * A @ B - 1.3 * C, tol=1e-2)

    adj_C = np.ones_like(C)

    tape.backward(grads={C_wp: wp.array(adj_C, device=device)})

    assert_np_equal(A_wp.grad.numpy(), 0.5 * adj_C @ B.T, tol=1e-2)
    assert_np_equal(B_wp.grad.numpy(), 0.5 * A.T @ adj_C, tol=1e-2)
    assert_np_equal(C_wp.grad.numpy(), -1.3 * adj_C, tol=1e-2)


all_devices = get_test_devices()


class TestTileMathDx(unittest.TestCase):
    pass


# check_output=False so we can enable libmathdx's logging without failing the tests
add_function_test(
    TestTileMathDx, "test_tile_math_matmul", test_tile_math_matmul, devices=all_devices, check_output=False
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
