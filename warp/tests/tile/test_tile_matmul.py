# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import sys
import unittest
from typing import Any

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

TILE_M = wp.constant(8)
TILE_N = wp.constant(4)
TILE_K = wp.constant(8)

# num threads per-tile
TILE_DIM = 64


@wp.kernel
def tile_grouped_gemm(A: wp.array3d(dtype=float), B: wp.array3d(dtype=float), C: wp.array3d(dtype=float)):
    # output tile index
    i = wp.tid()

    a = wp.tile_load(A[i], shape=(TILE_M, TILE_K))
    b = wp.tile_load(B[i], shape=(TILE_K, TILE_N))

    sum = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=wp.float32)

    wp.tile_matmul(a, b, sum)

    wp.tile_store(C[i], sum)


def test_tile_grouped_gemm(test, device):
    batch_count = 56

    M = TILE_M
    N = TILE_N
    K = TILE_K

    rng = np.random.default_rng(42)
    A = rng.random((batch_count, M, K), dtype=np.float32)
    B = rng.random((batch_count, K, N), dtype=np.float32)
    C = A @ B

    A_wp = wp.array(A, requires_grad=True, device=device)
    B_wp = wp.array(B, requires_grad=True, device=device)
    C_wp = wp.zeros((batch_count, TILE_M, TILE_N), requires_grad=True, device=device)

    with wp.Tape():
        wp.launch_tiled(
            tile_grouped_gemm, dim=[batch_count], inputs=[A_wp, B_wp, C_wp], block_dim=TILE_DIM, device=device
        )

    # TODO: 32 mismatched elements
    assert_np_equal(C_wp.numpy(), C, 1e-6)


@wp.kernel
def tile_gemm(A: wp.array2d(dtype=Any), B: wp.array2d(dtype=Any), C: wp.array2d(dtype=Any)):
    # output tile index
    i, j = wp.tid()

    sum = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=A.dtype)

    M = A.shape[0]
    N = B.shape[1]
    K = A.shape[1]

    count = int(K / TILE_K)

    for k in range(0, count):
        a = wp.tile_load(A, shape=(TILE_M, TILE_K), offset=(i * TILE_M, k * TILE_K))
        b = wp.tile_load(B, shape=(TILE_K, TILE_N), offset=(k * TILE_K, j * TILE_N))

        # sum += a*b
        wp.tile_matmul(a, b, sum)

    wp.tile_store(C, sum, offset=(i * TILE_M, j * TILE_N))


wp.overload(
    tile_gemm, {"A": wp.array2d(dtype=wp.float16), "B": wp.array2d(dtype=wp.float16), "C": wp.array2d(dtype=wp.float16)}
)
wp.overload(
    tile_gemm, {"A": wp.array2d(dtype=wp.float32), "B": wp.array2d(dtype=wp.float32), "C": wp.array2d(dtype=wp.float32)}
)
wp.overload(
    tile_gemm, {"A": wp.array2d(dtype=wp.float64), "B": wp.array2d(dtype=wp.float64), "C": wp.array2d(dtype=wp.float64)}
)


def test_tile_gemm(dtype):
    def test(test, device):
        M = TILE_M * 7
        K = TILE_K * 6
        N = TILE_N * 5

        rng = np.random.default_rng(42)
        A = rng.random((M, K), dtype=float).astype(wp.dtype_to_numpy(dtype))
        B = rng.random((K, N), dtype=float).astype(wp.dtype_to_numpy(dtype))
        C = np.zeros((M, N), dtype=float).astype(wp.dtype_to_numpy(dtype))

        A_wp = wp.array(A, requires_grad=True, device=device)
        B_wp = wp.array(B, requires_grad=True, device=device)
        C_wp = wp.array(C, requires_grad=True, device=device)

        with wp.Tape() as tape:
            wp.launch_tiled(
                tile_gemm,
                dim=(int(M / TILE_M), int(N / TILE_N)),
                inputs=[A_wp, B_wp, C_wp],
                block_dim=TILE_DIM,
                device=device,
            )

        assert_np_equal(C_wp.numpy(), A @ B, tol=1.0e-1)

        adj_C = np.ones_like(C)

        tape.backward(grads={C_wp: wp.array(adj_C, device=device)})

        assert_np_equal(A_wp.grad.numpy(), adj_C @ B.T, tol=1.0e-1)
        assert_np_equal(B_wp.grad.numpy(), A.T @ adj_C, 1.0e-1)

    return test


@wp.kernel
def test_tile_transpose_matmul_kernel(input: wp.array2d(dtype=float), output: wp.array2d(dtype=float)):
    x = wp.tile_load(input, shape=(TILE_M, TILE_N))
    y = wp.tile_transpose(x)

    z = wp.tile_zeros(dtype=float, shape=(TILE_N, TILE_N))
    wp.tile_matmul(y, x, z)

    wp.tile_store(output, z)


def test_tile_transpose_matmul(test, device):
    rng = np.random.default_rng(42)
    input = wp.array(rng.random((TILE_M, TILE_N), dtype=np.float32), device=device)
    output = wp.zeros((TILE_N, TILE_N), dtype=float, device=device)

    wp.launch_tiled(
        test_tile_transpose_matmul_kernel, dim=[1], inputs=[input, output], block_dim=TILE_DIM, device=device
    )

    assert_np_equal(output.numpy(), input.numpy().T @ input.numpy(), 1e-6)


@wp.kernel
def test_tile_matmul_return_form_kernel(
    A: wp.array2d(dtype=float), B: wp.array2d(dtype=float), C: wp.array2d(dtype=float)
):
    """Test the c = wp.tile_matmul(a, b) form which returns a fresh tile."""
    a = wp.tile_load(A, shape=(TILE_M, TILE_K))
    b = wp.tile_load(B, shape=(TILE_K, TILE_N))

    # Use the return form (not the accumulate form)
    # This tests that we don't read from the uninitialized output tile
    c = wp.tile_matmul(a, b)

    wp.tile_store(C, c)


def test_tile_matmul_return_form(test, device):
    """Test that c = wp.tile_matmul(a, b) works correctly with verify_fp.

    This specifically tests a fix where the return form was incorrectly
    reading from the uninitialized output tile (which could contain NaN
    when verify_fp is enabled, causing the result to be NaN).
    """
    # Enable verify_fp to trigger NaN initialization of tiles
    old_verify_fp = wp.config.verify_fp
    wp.config.verify_fp = True

    try:
        M = TILE_M
        K = TILE_K
        N = TILE_N

        rng = np.random.default_rng(42)
        A = rng.random((M, K), dtype=np.float32)
        B = rng.random((K, N), dtype=np.float32)
        expected = A @ B

        A_wp = wp.array(A, device=device)
        B_wp = wp.array(B, device=device)
        C_wp = wp.zeros((M, N), dtype=float, device=device)

        wp.launch_tiled(
            test_tile_matmul_return_form_kernel,
            dim=[1],
            inputs=[A_wp, B_wp, C_wp],
            block_dim=TILE_DIM,
            device=device,
        )

        result = C_wp.numpy()

        # Check that result doesn't contain NaN (which would happen with the bug)
        test.assertFalse(np.any(np.isnan(result)), "Result contains NaN values")

        # Check correctness
        assert_np_equal(result, expected, tol=1e-5)
    finally:
        wp.config.verify_fp = old_verify_fp


def _set_module_mathdx_gemm(enabled):
    """Helper to toggle enable_mathdx_gemm on this test module."""
    this_module = sys.modules[__name__]
    wp.set_module_options({"enable_mathdx_gemm": enabled}, module=this_module)


def test_tile_matmul_no_mathdx_gemm_config(test, device):
    """Test that wp.config.enable_mathdx_gemm=False propagates to module options."""
    old_config = wp.config.enable_mathdx_gemm

    try:
        wp.config.enable_mathdx_gemm = False
        test.assertFalse(wp.config.enable_mathdx_gemm)

        this_module = sys.modules[__name__]
        wp.set_module_options({"enable_mathdx_gemm": False}, module=this_module)

        batch_count = 4

        rng = np.random.default_rng(42)
        A = rng.random((batch_count, TILE_M, TILE_K), dtype=np.float32)
        B = rng.random((batch_count, TILE_K, TILE_N), dtype=np.float32)
        expected = A @ B

        A_wp = wp.array(A, device=device)
        B_wp = wp.array(B, device=device)
        C_wp = wp.zeros((batch_count, TILE_M, TILE_N), dtype=float, device=device)

        wp.launch_tiled(
            tile_grouped_gemm, dim=[batch_count], inputs=[A_wp, B_wp, C_wp], block_dim=TILE_DIM, device=device
        )

        assert_np_equal(C_wp.numpy(), expected, tol=1e-5)
    finally:
        wp.config.enable_mathdx_gemm = old_config
        _set_module_mathdx_gemm(True)


ODD_M = wp.constant(7)
ODD_N = wp.constant(5)
ODD_K = wp.constant(3)


@wp.kernel
def tile_odd_gemm_kernel(A: wp.array2d(dtype=float), B: wp.array2d(dtype=float), C: wp.array2d(dtype=float)):
    """Kernel with non-power-of-2, non-multiple-of-4 tile dimensions."""
    a = wp.tile_load(A, shape=(ODD_M, ODD_K))
    b = wp.tile_load(B, shape=(ODD_K, ODD_N))

    sum = wp.tile_zeros(shape=(ODD_M, ODD_N), dtype=wp.float32)
    wp.tile_matmul(a, b, sum)

    wp.tile_store(C, sum)


def test_tile_matmul_odd_shapes(test, device):
    """Test tile_matmul with odd (non-power-of-2) tile dimensions."""
    M = ODD_M
    K = ODD_K
    N = ODD_N

    rng = np.random.default_rng(42)
    A = rng.random((M, K), dtype=np.float32)
    B = rng.random((K, N), dtype=np.float32)
    expected = A @ B

    A_wp = wp.array(A, requires_grad=True, device=device)
    B_wp = wp.array(B, requires_grad=True, device=device)
    C_wp = wp.zeros((M, N), requires_grad=True, dtype=float, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_odd_gemm_kernel,
            dim=[1],
            inputs=[A_wp, B_wp, C_wp],
            block_dim=TILE_DIM,
            device=device,
        )

    assert_np_equal(C_wp.numpy(), expected, tol=1e-5)

    adj_C = np.ones_like(expected)
    tape.backward(grads={C_wp: wp.array(adj_C, device=device)})

    assert_np_equal(A_wp.grad.numpy(), adj_C @ B.T, tol=1e-5)
    assert_np_equal(B_wp.grad.numpy(), A.T @ adj_C, tol=1e-5)


# Large tile tests to exercise register blocking (BM=8x4, 4x4, 2x2 paths).
# With block_dim=64 and min_blocks=16:
#   64x32: blocks_8x4 = 8*8 = 64 >= 16 -> BM=8, BN=4
#   65x33: blocks_8x4 = 9*9 = 81 >= 16 -> BM=8, BN=4 (with boundary checks)
LARGE_M = wp.constant(64)
LARGE_N = wp.constant(32)
LARGE_K = wp.constant(8)

LARGE_ODD_M = wp.constant(65)
LARGE_ODD_N = wp.constant(33)


@wp.kernel
def tile_large_gemm_kernel(A: wp.array2d(dtype=float), B: wp.array2d(dtype=float), C: wp.array2d(dtype=float)):
    """Large aligned tile to exercise BM=8, BN=4 register blocking."""
    a = wp.tile_load(A, shape=(LARGE_M, LARGE_K))
    b = wp.tile_load(B, shape=(LARGE_K, LARGE_N))

    sum = wp.tile_zeros(shape=(LARGE_M, LARGE_N), dtype=wp.float32)
    wp.tile_matmul(a, b, sum)

    wp.tile_store(C, sum)


@wp.kernel
def tile_large_odd_gemm_kernel(A: wp.array2d(dtype=float), B: wp.array2d(dtype=float), C: wp.array2d(dtype=float)):
    """Large unaligned tile to exercise register blocking with boundary checks."""
    a = wp.tile_load(A, shape=(LARGE_ODD_M, LARGE_K))
    b = wp.tile_load(B, shape=(LARGE_K, LARGE_ODD_N))

    sum = wp.tile_zeros(shape=(LARGE_ODD_M, LARGE_ODD_N), dtype=wp.float32)
    wp.tile_matmul(a, b, sum)

    wp.tile_store(C, sum)


@wp.kernel
def tile_large_gemm_return_kernel(A: wp.array2d(dtype=float), B: wp.array2d(dtype=float), C: wp.array2d(dtype=float)):
    """Large tile using the return form c = wp.tile_matmul(a, b)."""
    a = wp.tile_load(A, shape=(LARGE_M, LARGE_K))
    b = wp.tile_load(B, shape=(LARGE_K, LARGE_N))

    c = wp.tile_matmul(a, b)

    wp.tile_store(C, c)


def test_tile_matmul_large_aligned(test, device):
    """Test large aligned tiles (64x32x8) to exercise BM=8, BN=4 register blocking with backward."""
    M, K, N = LARGE_M, LARGE_K, LARGE_N

    rng = np.random.default_rng(42)
    A = rng.random((M, K), dtype=np.float32)
    B = rng.random((K, N), dtype=np.float32)
    expected = A @ B

    A_wp = wp.array(A, requires_grad=True, device=device)
    B_wp = wp.array(B, requires_grad=True, device=device)
    C_wp = wp.zeros((M, N), requires_grad=True, dtype=float, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(tile_large_gemm_kernel, dim=[1], inputs=[A_wp, B_wp, C_wp], block_dim=TILE_DIM, device=device)

    assert_np_equal(C_wp.numpy(), expected, tol=1e-4)

    adj_C = np.ones_like(expected)
    tape.backward(grads={C_wp: wp.array(adj_C, device=device)})

    assert_np_equal(A_wp.grad.numpy(), adj_C @ B.T, tol=1e-4)
    assert_np_equal(B_wp.grad.numpy(), A.T @ adj_C, tol=1e-4)


def test_tile_matmul_large_unaligned(test, device):
    """Test large unaligned tiles (65x33x8) to exercise register blocking boundary checks with backward."""
    M, K, N = LARGE_ODD_M, LARGE_K, LARGE_ODD_N

    rng = np.random.default_rng(42)
    A = rng.random((M, K), dtype=np.float32)
    B = rng.random((K, N), dtype=np.float32)
    expected = A @ B

    A_wp = wp.array(A, requires_grad=True, device=device)
    B_wp = wp.array(B, requires_grad=True, device=device)
    C_wp = wp.zeros((M, N), requires_grad=True, dtype=float, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_large_odd_gemm_kernel, dim=[1], inputs=[A_wp, B_wp, C_wp], block_dim=TILE_DIM, device=device
        )

    assert_np_equal(C_wp.numpy(), expected, tol=1e-4)

    adj_C = np.ones_like(expected)
    tape.backward(grads={C_wp: wp.array(adj_C, device=device)})

    assert_np_equal(A_wp.grad.numpy(), adj_C @ B.T, tol=1e-4)
    assert_np_equal(B_wp.grad.numpy(), A.T @ adj_C, tol=1e-4)


def test_tile_matmul_return_form_backward(test, device):
    """Test the return form c = wp.tile_matmul(a, b) with backward pass."""
    M, K, N = LARGE_M, LARGE_K, LARGE_N

    rng = np.random.default_rng(42)
    A = rng.random((M, K), dtype=np.float32)
    B = rng.random((K, N), dtype=np.float32)
    expected = A @ B

    A_wp = wp.array(A, requires_grad=True, device=device)
    B_wp = wp.array(B, requires_grad=True, device=device)
    C_wp = wp.zeros((M, N), requires_grad=True, dtype=float, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_large_gemm_return_kernel, dim=[1], inputs=[A_wp, B_wp, C_wp], block_dim=TILE_DIM, device=device
        )

    assert_np_equal(C_wp.numpy(), expected, tol=1e-4)

    adj_C = np.ones_like(expected)
    tape.backward(grads={C_wp: wp.array(adj_C, device=device)})

    assert_np_equal(A_wp.grad.numpy(), adj_C @ B.T, tol=1e-4)
    assert_np_equal(B_wp.grad.numpy(), A.T @ adj_C, tol=1e-4)


# Float64 large tile test to exercise fma() specialization of muladd<T>
LARGE_M_F64 = wp.constant(32)
LARGE_N_F64 = wp.constant(16)
LARGE_K_F64 = wp.constant(8)


@wp.kernel
def tile_large_gemm_f64_kernel(
    A: wp.array2d(dtype=wp.float64), B: wp.array2d(dtype=wp.float64), C: wp.array2d(dtype=wp.float64)
):
    """Large tile in float64 to exercise fma() specialization."""
    a = wp.tile_load(A, shape=(LARGE_M_F64, LARGE_K_F64))
    b = wp.tile_load(B, shape=(LARGE_K_F64, LARGE_N_F64))

    sum = wp.tile_zeros(shape=(LARGE_M_F64, LARGE_N_F64), dtype=wp.float64)
    wp.tile_matmul(a, b, sum)

    wp.tile_store(C, sum)


def test_tile_matmul_large_fp64(test, device):
    """Test large tile matmul with float64 to exercise fma() and register blocking."""
    M, K, N = LARGE_M_F64, LARGE_K_F64, LARGE_N_F64

    rng = np.random.default_rng(42)
    A = rng.random((M, K), dtype=np.float64)
    B = rng.random((K, N), dtype=np.float64)
    expected = A @ B

    A_wp = wp.array(A, requires_grad=True, device=device)
    B_wp = wp.array(B, requires_grad=True, device=device)
    C_wp = wp.zeros((M, N), requires_grad=True, dtype=wp.float64, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_large_gemm_f64_kernel, dim=[1], inputs=[A_wp, B_wp, C_wp], block_dim=TILE_DIM, device=device
        )

    assert_np_equal(C_wp.numpy(), expected, tol=1e-10)

    adj_C = np.ones_like(expected)
    tape.backward(grads={C_wp: wp.array(adj_C, device=device)})

    assert_np_equal(A_wp.grad.numpy(), adj_C @ B.T, tol=1e-10)
    assert_np_equal(B_wp.grad.numpy(), A.T @ adj_C, tol=1e-10)


def make_no_mathdx_test(func):
    """Wrap a test function to run with enable_mathdx_gemm=False (scalar GEMM fallback)."""

    def wrapper(test, device):
        _set_module_mathdx_gemm(False)
        try:
            func(test, device)
        finally:
            _set_module_mathdx_gemm(True)

    return wrapper


class TestTileMatmul(unittest.TestCase):
    pass


devices = get_test_devices()

# All tile_matmul tests to run in both mathdx and scalar modes
tile_matmul_tests = [
    ("test_tile_gemm_fp16", test_tile_gemm(wp.float16)),
    ("test_tile_gemm_fp32", test_tile_gemm(wp.float32)),
    ("test_tile_gemm_fp64", test_tile_gemm(wp.float64)),
    ("test_tile_grouped_gemm", test_tile_grouped_gemm),
    ("test_tile_transpose_matmul", test_tile_transpose_matmul),
    ("test_tile_matmul_return_form", test_tile_matmul_return_form),
    ("test_tile_matmul_odd_shapes", test_tile_matmul_odd_shapes),
    ("test_tile_matmul_large_aligned", test_tile_matmul_large_aligned),
    ("test_tile_matmul_large_unaligned", test_tile_matmul_large_unaligned),
    ("test_tile_matmul_return_form_backward", test_tile_matmul_return_form_backward),
    ("test_tile_matmul_large_fp64", test_tile_matmul_large_fp64),
]

# Register each test in both modes: default (mathdx) and scalar (no mathdx)
for name, func in tile_matmul_tests:
    add_function_test(TestTileMatmul, name, func, devices=devices)
    add_function_test(TestTileMatmul, f"{name}_no_mathdx", make_no_mathdx_test(func), devices=devices)

# Config propagation test (unique, only needs one mode)
add_function_test(
    TestTileMatmul, "test_tile_matmul_no_mathdx_gemm_config", test_tile_matmul_no_mathdx_gemm_config, devices=devices
)

if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
