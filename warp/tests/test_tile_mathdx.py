# Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

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
    a = wp.tile_load(ga, i, j, m=TILE_M, n=TILE_K)
    b = wp.tile_load(gb, i, j, m=TILE_K, n=TILE_N)
    c = wp.tile_zeros(m=TILE_M, n=TILE_N, dtype=wp.float64)
    wp.tile_matmul(a, b, c)
    wp.tile_store(gc, i, j, c)


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
    xy = wp.tile_load(gx, i, j, m=FFT_SIZE_FP32, n=FFT_SIZE_FP32)
    wp.tile_fft(xy)
    wp.tile_store(gy, i, j, xy)


@wp.kernel()
def tile_math_fft_kernel_vec2d(gx: wp.array2d(dtype=wp.vec2d), gy: wp.array2d(dtype=wp.vec2d)):
    i, j = wp.tid()
    xy = wp.tile_load(gx, i, j, m=FFT_SIZE_FP64, n=FFT_SIZE_FP64)
    wp.tile_fft(xy)
    wp.tile_store(gy, i, j, xy)


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


devices = get_cuda_test_devices()


@unittest.skipUnless(wp.context.runtime.core.is_mathdx_enabled(), "Warp was not built with MathDx support")
class TestTileMathDx(unittest.TestCase):
    pass


# check_output=False so we can enable libmathdx's logging without failing the tests
add_function_test(TestTileMathDx, "test_tile_math_matmul", test_tile_math_matmul, devices=devices, check_output=False)
add_function_test(
    TestTileMathDx,
    "test_tile_math_fft_vec2f",
    functools.partial(test_tile_math_fft, wp_dtype=wp.vec2f),
    devices=devices,
    check_output=False,
)
add_function_test(
    TestTileMathDx,
    "test_tile_math_fft_vec2d",
    functools.partial(test_tile_math_fft, wp_dtype=wp.vec2d),
    devices=devices,
    check_output=False,
)

if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=True)
