# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

TILE_M = wp.constant(8)
TILE_N = wp.constant(8)
TILE_K = wp.constant(8)
TILE_K2 = wp.constant(2 * TILE_K)
TILE_N4 = wp.constant(4)

TILE_DIM = 32


@wp.kernel
def tile_matmul_strided_view_kernel(A: wp.array2d[float], B: wp.array2d[float], C: wp.array2d[float]):
    # the view of the left half keeps the parent's row stride, so it is non-dense
    a_full = wp.tile_load(A, shape=(TILE_M, TILE_K2))
    a_sub = wp.tile_view(a_full, offset=(0, 0), shape=(TILE_M, TILE_K))
    b = wp.tile_load(B, shape=(TILE_K, TILE_N))
    c = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=float)
    wp.tile_matmul(a_sub, b, c)
    wp.tile_store(C, c)


def test_tile_matmul_strided_view(test, device):
    rng = np.random.default_rng(42)
    A = rng.standard_normal((TILE_M, 2 * TILE_K)).astype(np.float32)
    B = rng.standard_normal((TILE_K, TILE_N)).astype(np.float32)

    A_wp = wp.array(A, device=device)
    B_wp = wp.array(B, device=device)
    C_wp = wp.zeros((TILE_M, TILE_N), dtype=float, device=device)

    wp.launch_tiled(
        tile_matmul_strided_view_kernel, dim=[1], inputs=[A_wp, B_wp], outputs=[C_wp], block_dim=TILE_DIM, device=device
    )

    assert_np_equal(C_wp.numpy(), A[:, :TILE_K] @ B, tol=1e-5)


def test_tile_matmul_strided_view_backward(test, device):
    # the backward GEMMs also read and write the strided view
    rng = np.random.default_rng(42)
    A = rng.standard_normal((TILE_M, 2 * TILE_K)).astype(np.float32)
    B = rng.standard_normal((TILE_K, TILE_N)).astype(np.float32)

    A_wp = wp.array(A, requires_grad=True, device=device)
    B_wp = wp.array(B, requires_grad=True, device=device)
    C_wp = wp.zeros((TILE_M, TILE_N), dtype=float, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_matmul_strided_view_kernel,
            dim=[1],
            inputs=[A_wp, B_wp],
            outputs=[C_wp],
            block_dim=TILE_DIM,
            device=device,
        )

    adj_C = np.ones((TILE_M, TILE_N), dtype=np.float32)
    tape.backward(grads={C_wp: wp.array(adj_C, device=device)})

    A_grad = np.zeros_like(A)
    A_grad[:, :TILE_K] = adj_C @ B.T
    assert_np_equal(A_wp.grad.numpy(), A_grad, tol=1e-5)
    assert_np_equal(B_wp.grad.numpy(), A[:, :TILE_K].T @ adj_C, tol=1e-5)


@wp.kernel
def tile_matmul_strided_view_fp16_kernel(
    A: wp.array2d[wp.float16], B: wp.array2d[wp.float16], C: wp.array2d[wp.float16]
):
    a_full = wp.tile_load(A, shape=(TILE_M, TILE_K2))
    a_sub = wp.tile_view(a_full, offset=(0, 0), shape=(TILE_M, TILE_K))
    b = wp.tile_load(B, shape=(TILE_K, TILE_N4))
    c = wp.tile_zeros(shape=(TILE_M, TILE_N4), dtype=wp.float16)
    wp.tile_matmul(a_sub, b, c)
    wp.tile_store(C, c)


def test_tile_matmul_strided_view_fp16(test, device):
    """Verify FP16 tile matrix multiplication with a strided operand.

    cuBLASDx 0.4.0 hard-fails this case under CUDA 12.8.0, 12.8.1,
    and 12.9.0 (NVBUG 5218000), so Warp exercises the scalar fallback
    on those versions. CUDA 12.9.1 is known to work. See:
    https://docs.nvidia.com/cuda/cublasdx/0.4.0/release_notes.html#known-issues
    """
    rng = np.random.default_rng(42)
    A = rng.standard_normal((TILE_M, 2 * TILE_K)).astype(np.float16)
    B = rng.standard_normal((TILE_K, TILE_N4)).astype(np.float16)

    A_wp = wp.array(A, device=device)
    B_wp = wp.array(B, device=device)
    C_wp = wp.zeros((TILE_M, TILE_N4), dtype=wp.float16, device=device)

    wp.launch_tiled(
        tile_matmul_strided_view_fp16_kernel,
        dim=[1],
        inputs=[A_wp, B_wp],
        outputs=[C_wp],
        block_dim=64,
        device=device,
    )

    ref = A[:, :TILE_K].astype(np.float32) @ B.astype(np.float32)
    assert_np_equal(C_wp.numpy().astype(np.float32), ref, tol=5e-2)


@wp.kernel
def tile_matmul_broadcast_kernel(A: wp.array2d[float], B: wp.array2d[float], C: wp.array2d[float]):
    # a zero-stride broadcast operand makes tile_matmul fall back to the scalar path
    a = wp.tile_load(A, shape=(TILE_M, TILE_K))
    b_row = wp.tile_load(B, shape=(1, TILE_N))
    b = wp.tile_broadcast(b_row, shape=(TILE_K, TILE_N))
    c = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=float)
    wp.tile_matmul(a, b, c)
    wp.tile_store(C, c)


def test_tile_matmul_broadcast_fallback(test, device):
    rng = np.random.default_rng(42)
    A = rng.standard_normal((TILE_M, TILE_K)).astype(np.float32)
    B = rng.standard_normal((1, TILE_N)).astype(np.float32)

    A_wp = wp.array(A, device=device)
    B_wp = wp.array(B, device=device)
    C_wp = wp.zeros((TILE_M, TILE_N), dtype=float, device=device)

    wp.launch_tiled(
        tile_matmul_broadcast_kernel, dim=[1], inputs=[A_wp, B_wp], outputs=[C_wp], block_dim=TILE_DIM, device=device
    )

    assert_np_equal(C_wp.numpy(), A @ np.broadcast_to(B, (TILE_K, TILE_N)), tol=1e-5)


class TestTileMatmulStrides(unittest.TestCase):
    pass


devices = get_test_devices()

add_function_test(
    TestTileMatmulStrides, "test_tile_matmul_strided_view", test_tile_matmul_strided_view, devices=devices
)
add_function_test(
    TestTileMatmulStrides,
    "test_tile_matmul_strided_view_backward",
    test_tile_matmul_strided_view_backward,
    devices=devices,
)
add_function_test(
    TestTileMatmulStrides, "test_tile_matmul_strided_view_fp16", test_tile_matmul_strided_view_fp16, devices=devices
)
add_function_test(
    TestTileMatmulStrides, "test_tile_matmul_broadcast_fallback", test_tile_matmul_broadcast_fallback, devices=devices
)

if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
