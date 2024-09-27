# Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

wp.init()  # For wp.context.runtime.core.is_mathdx_enabled()

TILE_M = wp.constant(8)
TILE_N = wp.constant(4)
TILE_K = wp.constant(8)

N_FFT = wp.constant(128)

# num threads per-tile
TILE_DIM = 64


@wp.kernel()
def tile_math_matmul_kernel(
    ga: wp.array2d(dtype=wp.float64), gb: wp.array2d(dtype=wp.float64), gc: wp.array2d(dtype=wp.float64)
):
    i, j = wp.tid()
    a = wp.tile_load(ga, i, j, m=TILE_M, n=TILE_K)
    b = wp.tile_load(gb, i, j, m=TILE_K, n=TILE_N)
    c = wp.tile_zeros(m=TILE_M, n=TILE_N, dtype=wp.float64)
    wp.tile_matmul(a, b, c)
    wp.tile_store(gc, i, j, c)


def test_tile_math_matmul(test, device):
    rng = np.random.default_rng(42)

    A = rng.random((TILE_M, TILE_K), dtype=np.float64)
    B = rng.random((TILE_K, TILE_N), dtype=np.float64)
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
    assert_np_equal(C_wp.numpy(), A @ B)

    adj_C = np.ones_like(C)

    tape.backward(grads={C_wp: wp.array(adj_C, device=device)})

    assert_np_equal(A_wp.grad.numpy(), adj_C @ B.T)
    assert_np_equal(B_wp.grad.numpy(), A.T @ adj_C)


@wp.kernel()
def tile_math_fft_kernel(gx: wp.array2d(dtype=wp.vec2f), gy: wp.array2d(dtype=wp.vec2f)):
    i, j = wp.tid()
    xy = wp.tile_load(gx, i, j, m=N_FFT, n=N_FFT)
    wp.tile_fft(xy)
    wp.tile_store(gy, i, j, xy)


def test_tile_math_fft(test, device):
    rng = np.random.default_rng(42)

    # Warp doesn't really have a complex64 type,
    # so we use 2 float32 to represent a single complex64 number and then convert it to vec2f

    X = rng.random((N_FFT, 2 * N_FFT), dtype=np.float32)
    Y = np.zeros_like(X)

    X_wp = wp.array2d(X, requires_grad=True, dtype=wp.vec2f, device=device)
    Y_wp = wp.array2d(Y, requires_grad=True, dtype=wp.vec2f, device=device)

    X_c64 = X.view(np.complex64).reshape(N_FFT, N_FFT)
    Y_c64 = np.fft.fft(X_c64, axis=-1)

    with wp.Tape() as tape:
        wp.launch_tiled(
            tile_math_fft_kernel, 
            dim=[1, 1], 
            inputs=[X_wp, Y_wp], 
            block_dim=TILE_DIM, 
            device=device)

    Y_wp_c64 = Y_wp.numpy().view(np.complex64).reshape(N_FFT, N_FFT)

    assert_np_equal(Y_wp_c64, Y_c64, tol=1.0e-4)

    # TODO: implement and test backward pass


devices = get_cuda_test_devices()


@unittest.skipUnless(wp.context.runtime.core.is_mathdx_enabled(), "Warp was not built with MathDx support")
class TestTileMathDx(unittest.TestCase):
    pass


add_function_test(TestTileMathDx, "test_tile_math_matmul", test_tile_math_matmul, devices=devices)
add_function_test(TestTileMathDx, "test_tile_math_fft", test_tile_math_fft, devices=devices)

if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
