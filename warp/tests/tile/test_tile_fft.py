# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

wp.init()  # For wp._src.context.runtime.core.wp_is_mathdx_enabled()

# num threads per-tile
TILE_DIM = 32
FFT_SIZE_FP32 = 64
FFT_SIZE_FP64 = 64
FFT_3D_DIM0 = 2
FFT_3D_DIM1 = 4


@wp.kernel()
def tile_fft_kernel_vec2f(gx: wp.array2d(dtype=wp.vec2f), gy: wp.array2d(dtype=wp.vec2f)):
    xy = wp.tile_load(gx, shape=(FFT_SIZE_FP32, FFT_SIZE_FP32))
    wp.tile_fft(xy)
    wp.tile_store(gy, xy)


@wp.kernel()
def tile_fft_kernel_vec2d(gx: wp.array2d(dtype=wp.vec2d), gy: wp.array2d(dtype=wp.vec2d)):
    xy = wp.tile_load(gx, shape=(FFT_SIZE_FP64, FFT_SIZE_FP64))
    wp.tile_fft(xy)
    wp.tile_store(gy, xy)


@wp.kernel()
def tile_ifft_kernel_vec2f(gx: wp.array2d(dtype=wp.vec2f), gy: wp.array2d(dtype=wp.vec2f)):
    xy = wp.tile_load(gx, shape=(FFT_SIZE_FP32, FFT_SIZE_FP32))
    wp.tile_ifft(xy)
    wp.tile_store(gy, xy)


@wp.kernel()
def tile_ifft_kernel_vec2d(gx: wp.array2d(dtype=wp.vec2d), gy: wp.array2d(dtype=wp.vec2d)):
    xy = wp.tile_load(gx, shape=(FFT_SIZE_FP64, FFT_SIZE_FP64))
    wp.tile_ifft(xy)
    wp.tile_store(gy, xy)


@wp.kernel()
def tile_fft_3d_kernel_vec2f(gx: wp.array3d(dtype=wp.vec2f), gy: wp.array3d(dtype=wp.vec2f)):
    xy = wp.tile_load(gx, shape=(FFT_3D_DIM0, FFT_3D_DIM1, FFT_SIZE_FP32))
    wp.tile_fft(xy)
    wp.tile_store(gy, xy)


@wp.kernel()
def tile_fft_3d_kernel_vec2d(gx: wp.array3d(dtype=wp.vec2d), gy: wp.array3d(dtype=wp.vec2d)):
    xy = wp.tile_load(gx, shape=(FFT_3D_DIM0, FFT_3D_DIM1, FFT_SIZE_FP64))
    wp.tile_fft(xy)
    wp.tile_store(gy, xy)


@wp.kernel()
def tile_ifft_3d_kernel_vec2f(gx: wp.array3d(dtype=wp.vec2f), gy: wp.array3d(dtype=wp.vec2f)):
    xy = wp.tile_load(gx, shape=(FFT_3D_DIM0, FFT_3D_DIM1, FFT_SIZE_FP32))
    wp.tile_ifft(xy)
    wp.tile_store(gy, xy)


@wp.kernel()
def tile_ifft_3d_kernel_vec2d(gx: wp.array3d(dtype=wp.vec2d), gy: wp.array3d(dtype=wp.vec2d)):
    xy = wp.tile_load(gx, shape=(FFT_3D_DIM0, FFT_3D_DIM1, FFT_SIZE_FP64))
    wp.tile_ifft(xy)
    wp.tile_store(gy, xy)


@unittest.skipUnless(wp._src.context.runtime.core.wp_is_mathdx_enabled(), "Warp was not built with MathDx support")
def test_tile_fft(test, device, wp_dtype, kernel, data_shape):
    np_real_dtype = {wp.vec2f: np.float32, wp.vec2d: np.float64}[wp_dtype]
    np_cplx_dtype = {wp.vec2f: np.complex64, wp.vec2d: np.complex128}[wp_dtype]
    fft_size = data_shape[-1] // 2
    complex_shape = (*data_shape[:-1], fft_size)

    rng = np.random.default_rng(42)

    X = rng.random(data_shape, dtype=np_real_dtype)
    Y = np.zeros_like(X)

    X_wp = wp.array(X, requires_grad=True, dtype=wp_dtype, device=device)
    Y_wp = wp.array(Y, requires_grad=True, dtype=wp_dtype, device=device)

    X_c = X.view(np_cplx_dtype).reshape(complex_shape)
    Y_c = np.fft.fft(X_c, axis=-1)

    with wp.Tape() as tape:
        wp.launch_tiled(kernel, dim=[1], inputs=[X_wp, Y_wp], block_dim=TILE_DIM, device=device)

    # verify forward pass
    Y_wp_c = Y_wp.numpy().view(np_cplx_dtype).reshape(complex_shape)
    assert_np_equal(Y_wp_c, Y_c, tol=1.0e-4)

    # verify backward pass
    # The adjoint of FFT is IFFT (unnormalized)
    adj_Y = rng.random(data_shape, dtype=np_real_dtype)
    adj_Y_c = adj_Y.view(np_cplx_dtype).reshape(complex_shape)

    tape.backward(grads={Y_wp: wp.array(adj_Y, dtype=wp_dtype, device=device)})

    # Expected gradient: IFFT of adj_Y (unnormalized, so multiply by fft_size)
    expected_grad_c = np.fft.ifft(adj_Y_c, axis=-1) * fft_size
    actual_grad_c = X_wp.grad.numpy().view(np_cplx_dtype).reshape(complex_shape)

    assert_np_equal(actual_grad_c, expected_grad_c, tol=1.0e-4)


@unittest.skipUnless(wp._src.context.runtime.core.wp_is_mathdx_enabled(), "Warp was not built with MathDx support")
def test_tile_ifft(test, device, wp_dtype, kernel, data_shape):
    np_real_dtype = {wp.vec2f: np.float32, wp.vec2d: np.float64}[wp_dtype]
    np_cplx_dtype = {wp.vec2f: np.complex64, wp.vec2d: np.complex128}[wp_dtype]
    fft_size = data_shape[-1] // 2
    complex_shape = (*data_shape[:-1], fft_size)

    rng = np.random.default_rng(42)

    X = rng.random(data_shape, dtype=np_real_dtype)
    Y = np.zeros_like(X)

    X_wp = wp.array(X, requires_grad=True, dtype=wp_dtype, device=device)
    Y_wp = wp.array(Y, requires_grad=True, dtype=wp_dtype, device=device)

    X_c = X.view(np_cplx_dtype).reshape(complex_shape)
    # Warp's IFFT is unnormalized, equivalent to NumPy's ifft * N
    Y_c = np.fft.ifft(X_c, axis=-1) * fft_size

    with wp.Tape() as tape:
        wp.launch_tiled(kernel, dim=[1], inputs=[X_wp, Y_wp], block_dim=TILE_DIM, device=device)

    # verify forward pass
    Y_wp_c = Y_wp.numpy().view(np_cplx_dtype).reshape(complex_shape)
    assert_np_equal(Y_wp_c, Y_c, tol=1.0e-4)

    # verify backward pass
    # The adjoint of IFFT is FFT (unnormalized)
    adj_Y = rng.random(data_shape, dtype=np_real_dtype)
    adj_Y_c = adj_Y.view(np_cplx_dtype).reshape(complex_shape)

    tape.backward(grads={Y_wp: wp.array(adj_Y, dtype=wp_dtype, device=device)})

    # Expected gradient: FFT of adj_Y
    expected_grad_c = np.fft.fft(adj_Y_c, axis=-1)
    actual_grad_c = X_wp.grad.numpy().view(np_cplx_dtype).reshape(complex_shape)

    assert_np_equal(actual_grad_c, expected_grad_c, tol=1.0e-4)


cuda_devices = get_cuda_test_devices()


class TestTileFFT(unittest.TestCase):
    pass


# check_output=False so we can enable libmathdx's logging without failing the tests
add_function_test(
    TestTileFFT,
    "test_tile_fft_2d_vec2f",
    functools.partial(
        test_tile_fft,
        wp_dtype=wp.vec2f,
        kernel=tile_fft_kernel_vec2f,
        data_shape=(FFT_SIZE_FP32, 2 * FFT_SIZE_FP32),
    ),
    devices=cuda_devices,
    check_output=False,
)
add_function_test(
    TestTileFFT,
    "test_tile_fft_2d_vec2d",
    functools.partial(
        test_tile_fft,
        wp_dtype=wp.vec2d,
        kernel=tile_fft_kernel_vec2d,
        data_shape=(FFT_SIZE_FP64, 2 * FFT_SIZE_FP64),
    ),
    devices=cuda_devices,
    check_output=False,
)
add_function_test(
    TestTileFFT,
    "test_tile_ifft_2d_vec2f",
    functools.partial(
        test_tile_ifft,
        wp_dtype=wp.vec2f,
        kernel=tile_ifft_kernel_vec2f,
        data_shape=(FFT_SIZE_FP32, 2 * FFT_SIZE_FP32),
    ),
    devices=cuda_devices,
    check_output=False,
)
add_function_test(
    TestTileFFT,
    "test_tile_ifft_2d_vec2d",
    functools.partial(
        test_tile_ifft,
        wp_dtype=wp.vec2d,
        kernel=tile_ifft_kernel_vec2d,
        data_shape=(FFT_SIZE_FP64, 2 * FFT_SIZE_FP64),
    ),
    devices=cuda_devices,
    check_output=False,
)
add_function_test(
    TestTileFFT,
    "test_tile_fft_3d_vec2f",
    functools.partial(
        test_tile_fft,
        wp_dtype=wp.vec2f,
        kernel=tile_fft_3d_kernel_vec2f,
        data_shape=(FFT_3D_DIM0, FFT_3D_DIM1, 2 * FFT_SIZE_FP32),
    ),
    devices=cuda_devices,
    check_output=False,
)
add_function_test(
    TestTileFFT,
    "test_tile_fft_3d_vec2d",
    functools.partial(
        test_tile_fft,
        wp_dtype=wp.vec2d,
        kernel=tile_fft_3d_kernel_vec2d,
        data_shape=(FFT_3D_DIM0, FFT_3D_DIM1, 2 * FFT_SIZE_FP64),
    ),
    devices=cuda_devices,
    check_output=False,
)
add_function_test(
    TestTileFFT,
    "test_tile_ifft_3d_vec2f",
    functools.partial(
        test_tile_ifft,
        wp_dtype=wp.vec2f,
        kernel=tile_ifft_3d_kernel_vec2f,
        data_shape=(FFT_3D_DIM0, FFT_3D_DIM1, 2 * FFT_SIZE_FP32),
    ),
    devices=cuda_devices,
    check_output=False,
)
add_function_test(
    TestTileFFT,
    "test_tile_ifft_3d_vec2d",
    functools.partial(
        test_tile_ifft,
        wp_dtype=wp.vec2d,
        kernel=tile_ifft_3d_kernel_vec2d,
        data_shape=(FFT_3D_DIM0, FFT_3D_DIM1, 2 * FFT_SIZE_FP64),
    ),
    devices=cuda_devices,
    check_output=False,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
