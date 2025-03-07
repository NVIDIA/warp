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

import functools
import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

wp.init()  # For wp.context.runtime.core.is_mathdx_enabled()

TILE_M = wp.constant(8)
TILE_N = wp.constant(4)
TILE_K = wp.constant(8)

# num threads per-tile
TILE_DIM = 32
FFT_SIZE_FP32 = 64
FFT_SIZE_FP64 = 64


@wp.kernel()
def tile_math_matmul_kernel(
    ga: wp.array2d(dtype=wp.float16), gb: wp.array2d(dtype=wp.float32), gc: wp.array2d(dtype=wp.float64)
):
    i, j = wp.tid()
    a = wp.tile_load(ga, shape=(TILE_M, TILE_K), offset=(i * TILE_M, j * TILE_K))
    b = wp.tile_load(gb, shape=(TILE_K, TILE_N), offset=(i * TILE_K, j * TILE_N))
    c = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=wp.float64)
    wp.tile_matmul(a, b, c)
    wp.tile_store(gc, c, offset=(i * TILE_M, j * TILE_N))


def test_tile_math_matmul(test, device):
    rng = np.random.default_rng(42)

    A = rng.random((TILE_M, TILE_K), dtype=np.float64).astype(np.float16)
    B = rng.random((TILE_K, TILE_N), dtype=np.float32)
    C = np.zeros((TILE_M, TILE_N), dtype=np.float64)

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
    assert_np_equal(C_wp.numpy(), A @ B, tol=1e-2)

    adj_C = np.ones_like(C)

    tape.backward(grads={C_wp: wp.array(adj_C, device=device)})

    assert_np_equal(A_wp.grad.numpy(), adj_C @ B.T, tol=1e-2)
    assert_np_equal(B_wp.grad.numpy(), A.T @ adj_C, tol=1e-2)


@wp.kernel()
def tile_math_fft_kernel_vec2f(gx: wp.array2d(dtype=wp.vec2f), gy: wp.array2d(dtype=wp.vec2f)):
    i, j = wp.tid()
    xy = wp.tile_load(gx, shape=(FFT_SIZE_FP32, FFT_SIZE_FP32))
    wp.tile_fft(xy)
    wp.tile_store(gy, xy)


@wp.kernel()
def tile_math_fft_kernel_vec2d(gx: wp.array2d(dtype=wp.vec2d), gy: wp.array2d(dtype=wp.vec2d)):
    i, j = wp.tid()
    xy = wp.tile_load(gx, shape=(FFT_SIZE_FP64, FFT_SIZE_FP64))
    wp.tile_fft(xy)
    wp.tile_store(gy, xy)


@unittest.skipUnless(wp.context.runtime.core.is_mathdx_enabled(), "Warp was not built with MathDx support")
def test_tile_math_fft(test, device, wp_dtype):
    np_real_dtype = {wp.vec2f: np.float32, wp.vec2d: np.float64}[wp_dtype]
    np_cplx_dtype = {wp.vec2f: np.complex64, wp.vec2d: np.complex128}[wp_dtype]
    kernel = {wp.vec2d: tile_math_fft_kernel_vec2d, wp.vec2f: tile_math_fft_kernel_vec2f}[wp_dtype]
    fft_size = {wp.vec2d: FFT_SIZE_FP64, wp.vec2f: FFT_SIZE_FP32}[wp_dtype]

    rng = np.random.default_rng(42)

    # Warp doesn't really have a complex64 type,
    # so we use 2 float32 to represent a single complex64 number and then convert it to vec2f

    X = rng.random((fft_size, 2 * fft_size), dtype=np_real_dtype)
    Y = np.zeros_like(X)

    X_wp = wp.array2d(X, requires_grad=True, dtype=wp_dtype, device=device)
    Y_wp = wp.array2d(Y, requires_grad=True, dtype=wp_dtype, device=device)

    X_c64 = X.view(np_cplx_dtype).reshape(fft_size, fft_size)
    Y_c64 = np.fft.fft(X_c64, axis=-1)

    with wp.Tape() as tape:
        wp.launch_tiled(kernel, dim=[1, 1], inputs=[X_wp, Y_wp], block_dim=TILE_DIM, device=device)

    Y_wp_c64 = Y_wp.numpy().view(np_cplx_dtype).reshape(fft_size, fft_size)

    assert_np_equal(Y_wp_c64, Y_c64, tol=1.0e-4)

    # TODO: implement and test backward pass


@wp.kernel()
def tile_math_cholesky(
    gA: wp.array2d(dtype=wp.float64),
    gD: wp.array1d(dtype=wp.float64),
    gL: wp.array2d(dtype=wp.float64),
    gx: wp.array1d(dtype=wp.float64),
    gy: wp.array1d(dtype=wp.float64),
):
    i, j = wp.tid()
    # Load A, D & x
    a = wp.tile_load(gA, shape=(TILE_M, TILE_M), storage="shared")
    d = wp.tile_load(gD, shape=TILE_M, storage="shared")
    x = wp.tile_load(gx, shape=TILE_M, storage="shared")
    # Compute L st LL^T = A + diag(D)
    b = wp.tile_diag_add(a, d)
    l = wp.tile_cholesky(b)
    # Solve for y in LL^T y = x
    y = wp.tile_cholesky_solve(l, x)
    # Store L & y
    wp.tile_store(gL, l)
    wp.tile_store(gy, y)


def test_tile_math_cholesky(test, device):
    A_h = np.ones((TILE_M, TILE_M), dtype=np.float64)
    D_h = 8.0 * np.ones(TILE_M, dtype=np.float64)
    L_h = np.zeros_like(A_h)
    X_h = np.arange(TILE_M, dtype=np.float64)
    Y_h = np.zeros_like(X_h)

    A_np = A_h + np.diag(D_h)
    L_np = np.linalg.cholesky(A_np)
    Y_np = np.linalg.solve(A_np, X_h)

    A_wp = wp.array2d(A_h, requires_grad=True, dtype=wp.float64, device=device)
    D_wp = wp.array2d(D_h, requires_grad=True, dtype=wp.float64, device=device)
    L_wp = wp.array2d(L_h, requires_grad=True, dtype=wp.float64, device=device)
    X_wp = wp.array2d(X_h, requires_grad=True, dtype=wp.float64, device=device)
    Y_wp = wp.array2d(Y_h, requires_grad=True, dtype=wp.float64, device=device)

    wp.launch_tiled(
        tile_math_cholesky, dim=[1, 1], inputs=[A_wp, D_wp, L_wp, X_wp, Y_wp], block_dim=TILE_DIM, device=device
    )
    wp.synchronize_device()

    assert np.allclose(Y_wp.numpy(), Y_np) and np.allclose(L_wp.numpy(), L_np)

    # TODO: implement and test backward pass


all_devices = get_test_devices()
cuda_devices = get_cuda_test_devices()


class TestTileMathDx(unittest.TestCase):
    pass


# check_output=False so we can enable libmathdx's logging without failing the tests
add_function_test(
    TestTileMathDx, "test_tile_math_matmul", test_tile_math_matmul, devices=all_devices, check_output=False
)
add_function_test(
    TestTileMathDx, "test_tile_math_cholesky", test_tile_math_cholesky, devices=all_devices, check_output=False
)
add_function_test(
    TestTileMathDx,
    "test_tile_math_fft_vec2f",
    functools.partial(test_tile_math_fft, wp_dtype=wp.vec2f),
    devices=cuda_devices,
    check_output=False,
)
add_function_test(
    TestTileMathDx,
    "test_tile_math_fft_vec2d",
    functools.partial(test_tile_math_fft, wp_dtype=wp.vec2d),
    devices=cuda_devices,
    check_output=False,
)

if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=True)
