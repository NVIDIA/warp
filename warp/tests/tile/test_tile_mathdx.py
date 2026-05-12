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
def tile_math_matmul_kernel(ga: wp.array2d[wp.float16], gb: wp.array2d[wp.float32], gc: wp.array2d[wp.float64]):
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


@wp.kernel
def tile_pipelined_gemm_kernel(A: wp.array2d[float], B: wp.array2d[float], C: wp.array2d[float]):
    i, j = wp.tid()

    sum = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=wp.float32)
    a = wp.tile_load(A, shape=(TILE_M, TILE_K), offset=(i * TILE_M, 0), storage="register")
    b = wp.tile_load(B, shape=(TILE_K, TILE_N), offset=(0, j * TILE_N), storage="register")

    count = int(A.shape[1] / TILE_K)
    for k in range(1, count):
        a_next = wp.tile_load(A, shape=(TILE_M, TILE_K), offset=(i * TILE_M, k * TILE_K), storage="register")
        b_next = wp.tile_load(B, shape=(TILE_K, TILE_N), offset=(k * TILE_K, j * TILE_N), storage="register")

        wp.tile_matmul(a, b, sum)
        a = a_next
        b = b_next

    wp.tile_matmul(a, b, sum)
    wp.tile_store(C, sum, offset=(i * TILE_M, j * TILE_N))


@wp.kernel
def tile_reassign_after_matmul_kernel(
    A: wp.array2d[float],
    B: wp.array2d[float],
    C_sum: wp.array2d[float],
    C_reassigned: wp.array2d[float],
    C_direct: wp.array2d[float],
):
    a = wp.tile_load(A, shape=(TILE_M, TILE_K), offset=(0, 0), storage="register")
    b = wp.tile_load(B, shape=(TILE_K, TILE_N), offset=(0, 0), storage="register")

    sum = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=wp.float32)
    wp.tile_matmul(a, b, sum)
    wp.tile_store(C_sum, sum)

    a_next = wp.tile_load(A, shape=(TILE_M, TILE_K), offset=(TILE_M, 0), storage="register")
    a = a_next

    wp.tile_store(C_reassigned, a)
    wp.tile_store(C_direct, a_next)


def test_tile_matmul_pipelined_reassign(test, device):
    M = TILE_M * 3
    K = TILE_K * 3
    N = TILE_N * 5

    rng = np.random.default_rng(42)
    A = rng.random((M, K), dtype=np.float32)
    B = rng.random((K, N), dtype=np.float32)
    C = np.zeros((M, N), dtype=np.float32)

    A_wp = wp.array(A, requires_grad=True, device=device)
    B_wp = wp.array(B, requires_grad=True, device=device)
    C_wp = wp.array(C, requires_grad=True, device=device)

    wp.launch_tiled(
        tile_pipelined_gemm_kernel,
        dim=(int(M / TILE_M), int(N / TILE_N)),
        inputs=[A_wp, B_wp, C_wp],
        block_dim=TILE_DIM,
        device=device,
    )

    assert_np_equal(C_wp.numpy(), A @ B, tol=1.0e-4)


def test_tile_matmul_reassign_backward(test, device):
    M = TILE_M * 2
    K = TILE_K
    N = TILE_N

    rng = np.random.default_rng(42)
    A = rng.random((M, K), dtype=np.float32)
    B = rng.random((K, N), dtype=np.float32)
    C_sum = np.zeros((TILE_M, TILE_N), dtype=np.float32)
    C_reassigned = np.zeros((TILE_M, TILE_K), dtype=np.float32)
    C_direct = np.zeros((TILE_M, TILE_K), dtype=np.float32)

    A_wp = wp.array(A, requires_grad=True, device=device)
    B_wp = wp.array(B, requires_grad=True, device=device)
    C_sum_wp = wp.array(C_sum, requires_grad=True, device=device)
    C_reassigned_wp = wp.array(C_reassigned, requires_grad=True, device=device)
    C_direct_wp = wp.array(C_direct, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_reassign_after_matmul_kernel,
            dim=1,
            inputs=[A_wp, B_wp, C_sum_wp, C_reassigned_wp, C_direct_wp],
            block_dim=TILE_DIM,
            device=device,
        )

    assert_np_equal(C_sum_wp.numpy(), A[:TILE_M, :] @ B, tol=1.0e-4)
    assert_np_equal(C_reassigned_wp.numpy(), A[TILE_M:, :], tol=1.0e-4)
    assert_np_equal(C_direct_wp.numpy(), A[TILE_M:, :], tol=1.0e-4)

    adj_sum = np.ones_like(C_sum)
    adj_reassigned = np.ones_like(C_reassigned)
    adj_direct = np.ones_like(C_direct)
    tape.backward(
        grads={
            C_sum_wp: wp.array(adj_sum, device=device),
            C_reassigned_wp: wp.array(adj_reassigned, device=device),
            C_direct_wp: wp.array(adj_direct, device=device),
        }
    )

    expected_A_grad = np.zeros_like(A)
    expected_A_grad[:TILE_M, :] = adj_sum @ B.T
    expected_A_grad[TILE_M:, :] = adj_reassigned + adj_direct
    assert_np_equal(A_wp.grad.numpy(), expected_A_grad, tol=1.0e-4)
    assert_np_equal(B_wp.grad.numpy(), A[:TILE_M, :].T @ adj_sum, tol=1.0e-4)


all_devices = get_test_devices()


class TestTileMathDx(unittest.TestCase):
    pass


# check_output=False so we can enable libmathdx's logging without failing the tests
add_function_test(
    TestTileMathDx, "test_tile_math_matmul", test_tile_math_matmul, devices=all_devices, check_output=False
)
add_function_test(
    TestTileMathDx,
    "test_tile_matmul_pipelined_reassign",
    test_tile_matmul_pipelined_reassign,
    devices=all_devices,
    check_output=False,
)
add_function_test(
    TestTileMathDx,
    "test_tile_matmul_reassign_backward",
    test_tile_matmul_reassign_backward,
    devices=all_devices,
    check_output=False,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
