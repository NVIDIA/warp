# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tile FFT tests exercising the cooperative scalar fallback path.

This file mirrors the kernel definitions and test logic from test_tile_fft.py
so that the no-MathDx variants compile as a separate Warp module. Setting
``enable_mathdx_fft=False`` at the module level routes ``tile_fft`` /
``tile_ifft`` and their adjoints through ``wp::tile_fft_entry``'s cooperative
shared-memory branch on GPU (or the CPU sequential branch on CPU). This is
the fallback path used when ``enable_mathdx_fft=False``, which is also the
path selected when Warp is built without libmathdx.
"""

import functools
import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

# Disable MathDx FFT for all kernels defined in this module.
wp.set_module_options({"enable_mathdx_fft": False})

# num threads per-tile
TILE_DIM = 32
FFT_SIZE_FP32 = 64
FFT_SIZE_FP64 = 64
FFT_3D_DIM0 = 2
FFT_3D_DIM1 = 4


@wp.kernel
def tile_fft_kernel_vec2f(gx: wp.array2d[wp.vec2f], gy: wp.array2d[wp.vec2f]):
    xy = wp.tile_load(gx, shape=(FFT_SIZE_FP32, FFT_SIZE_FP32))
    wp.tile_fft(xy)
    wp.tile_store(gy, xy)


@wp.kernel
def tile_fft_kernel_vec2d(gx: wp.array2d[wp.vec2d], gy: wp.array2d[wp.vec2d]):
    xy = wp.tile_load(gx, shape=(FFT_SIZE_FP64, FFT_SIZE_FP64))
    wp.tile_fft(xy)
    wp.tile_store(gy, xy)


@wp.kernel
def tile_ifft_kernel_vec2f(gx: wp.array2d[wp.vec2f], gy: wp.array2d[wp.vec2f]):
    xy = wp.tile_load(gx, shape=(FFT_SIZE_FP32, FFT_SIZE_FP32))
    wp.tile_ifft(xy)
    wp.tile_store(gy, xy)


@wp.kernel
def tile_ifft_kernel_vec2d(gx: wp.array2d[wp.vec2d], gy: wp.array2d[wp.vec2d]):
    xy = wp.tile_load(gx, shape=(FFT_SIZE_FP64, FFT_SIZE_FP64))
    wp.tile_ifft(xy)
    wp.tile_store(gy, xy)


@wp.kernel
def tile_fft_3d_kernel_vec2f(gx: wp.array3d[wp.vec2f], gy: wp.array3d[wp.vec2f]):
    xy = wp.tile_load(gx, shape=(FFT_3D_DIM0, FFT_3D_DIM1, FFT_SIZE_FP32))
    wp.tile_fft(xy)
    wp.tile_store(gy, xy)


@wp.kernel
def tile_fft_3d_kernel_vec2d(gx: wp.array3d[wp.vec2d], gy: wp.array3d[wp.vec2d]):
    xy = wp.tile_load(gx, shape=(FFT_3D_DIM0, FFT_3D_DIM1, FFT_SIZE_FP64))
    wp.tile_fft(xy)
    wp.tile_store(gy, xy)


@wp.kernel
def tile_ifft_3d_kernel_vec2f(gx: wp.array3d[wp.vec2f], gy: wp.array3d[wp.vec2f]):
    xy = wp.tile_load(gx, shape=(FFT_3D_DIM0, FFT_3D_DIM1, FFT_SIZE_FP32))
    wp.tile_ifft(xy)
    wp.tile_store(gy, xy)


@wp.kernel
def tile_ifft_3d_kernel_vec2d(gx: wp.array3d[wp.vec2d], gy: wp.array3d[wp.vec2d]):
    xy = wp.tile_load(gx, shape=(FFT_3D_DIM0, FFT_3D_DIM1, FFT_SIZE_FP64))
    wp.tile_ifft(xy)
    wp.tile_store(gy, xy)


# Tolerances chosen with ~5-15x headroom over max abs errors measured on
# CPU and CUDA (RTX 5090, sm_120) across all kernels here: ~7e-6 worst case
# for fp32 and ~7e-15 for fp64 (forward and adjoint).
_FFT_TOL = {wp.vec2f: 3.0e-5, wp.vec2d: 1.0e-13}


def test_tile_fft(test, device, wp_dtype, kernel, data_shape):
    np_real_dtype = {wp.vec2f: np.float32, wp.vec2d: np.float64}[wp_dtype]
    np_cplx_dtype = {wp.vec2f: np.complex64, wp.vec2d: np.complex128}[wp_dtype]
    tol = _FFT_TOL[wp_dtype]
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

    Y_wp_c = Y_wp.numpy().view(np_cplx_dtype).reshape(complex_shape)
    assert_np_equal(Y_wp_c, Y_c, tol=tol)

    adj_Y = rng.random(data_shape, dtype=np_real_dtype)
    adj_Y_c = adj_Y.view(np_cplx_dtype).reshape(complex_shape)

    tape.backward(grads={Y_wp: wp.array(adj_Y, dtype=wp_dtype, device=device)})

    expected_grad_c = np.fft.ifft(adj_Y_c, axis=-1) * fft_size
    actual_grad_c = X_wp.grad.numpy().view(np_cplx_dtype).reshape(complex_shape)

    assert_np_equal(actual_grad_c, expected_grad_c, tol=tol)


def test_tile_ifft(test, device, wp_dtype, kernel, data_shape):
    np_real_dtype = {wp.vec2f: np.float32, wp.vec2d: np.float64}[wp_dtype]
    np_cplx_dtype = {wp.vec2f: np.complex64, wp.vec2d: np.complex128}[wp_dtype]
    tol = _FFT_TOL[wp_dtype]
    fft_size = data_shape[-1] // 2
    complex_shape = (*data_shape[:-1], fft_size)

    rng = np.random.default_rng(42)

    X = rng.random(data_shape, dtype=np_real_dtype)
    Y = np.zeros_like(X)

    X_wp = wp.array(X, requires_grad=True, dtype=wp_dtype, device=device)
    Y_wp = wp.array(Y, requires_grad=True, dtype=wp_dtype, device=device)

    X_c = X.view(np_cplx_dtype).reshape(complex_shape)
    Y_c = np.fft.ifft(X_c, axis=-1) * fft_size

    with wp.Tape() as tape:
        wp.launch_tiled(kernel, dim=[1], inputs=[X_wp, Y_wp], block_dim=TILE_DIM, device=device)

    Y_wp_c = Y_wp.numpy().view(np_cplx_dtype).reshape(complex_shape)
    assert_np_equal(Y_wp_c, Y_c, tol=tol)

    adj_Y = rng.random(data_shape, dtype=np_real_dtype)
    adj_Y_c = adj_Y.view(np_cplx_dtype).reshape(complex_shape)

    tape.backward(grads={Y_wp: wp.array(adj_Y, dtype=wp_dtype, device=device)})

    expected_grad_c = np.fft.fft(adj_Y_c, axis=-1)
    actual_grad_c = X_wp.grad.numpy().view(np_cplx_dtype).reshape(complex_shape)

    assert_np_equal(actual_grad_c, expected_grad_c, tol=tol)


test_devices = get_test_devices()


class TestTileFFTNoMathDx(unittest.TestCase):
    pass


add_function_test(
    TestTileFFTNoMathDx,
    "test_tile_fft_2d_vec2f",
    functools.partial(
        test_tile_fft,
        wp_dtype=wp.vec2f,
        kernel=tile_fft_kernel_vec2f,
        data_shape=(FFT_SIZE_FP32, 2 * FFT_SIZE_FP32),
    ),
    devices=test_devices,
    check_output=False,
)
add_function_test(
    TestTileFFTNoMathDx,
    "test_tile_fft_2d_vec2d",
    functools.partial(
        test_tile_fft,
        wp_dtype=wp.vec2d,
        kernel=tile_fft_kernel_vec2d,
        data_shape=(FFT_SIZE_FP64, 2 * FFT_SIZE_FP64),
    ),
    devices=test_devices,
    check_output=False,
)
add_function_test(
    TestTileFFTNoMathDx,
    "test_tile_ifft_2d_vec2f",
    functools.partial(
        test_tile_ifft,
        wp_dtype=wp.vec2f,
        kernel=tile_ifft_kernel_vec2f,
        data_shape=(FFT_SIZE_FP32, 2 * FFT_SIZE_FP32),
    ),
    devices=test_devices,
    check_output=False,
)
add_function_test(
    TestTileFFTNoMathDx,
    "test_tile_ifft_2d_vec2d",
    functools.partial(
        test_tile_ifft,
        wp_dtype=wp.vec2d,
        kernel=tile_ifft_kernel_vec2d,
        data_shape=(FFT_SIZE_FP64, 2 * FFT_SIZE_FP64),
    ),
    devices=test_devices,
    check_output=False,
)
add_function_test(
    TestTileFFTNoMathDx,
    "test_tile_fft_3d_vec2f",
    functools.partial(
        test_tile_fft,
        wp_dtype=wp.vec2f,
        kernel=tile_fft_3d_kernel_vec2f,
        data_shape=(FFT_3D_DIM0, FFT_3D_DIM1, 2 * FFT_SIZE_FP32),
    ),
    devices=test_devices,
    check_output=False,
)
add_function_test(
    TestTileFFTNoMathDx,
    "test_tile_fft_3d_vec2d",
    functools.partial(
        test_tile_fft,
        wp_dtype=wp.vec2d,
        kernel=tile_fft_3d_kernel_vec2d,
        data_shape=(FFT_3D_DIM0, FFT_3D_DIM1, 2 * FFT_SIZE_FP64),
    ),
    devices=test_devices,
    check_output=False,
)
add_function_test(
    TestTileFFTNoMathDx,
    "test_tile_ifft_3d_vec2f",
    functools.partial(
        test_tile_ifft,
        wp_dtype=wp.vec2f,
        kernel=tile_ifft_3d_kernel_vec2f,
        data_shape=(FFT_3D_DIM0, FFT_3D_DIM1, 2 * FFT_SIZE_FP32),
    ),
    devices=test_devices,
    check_output=False,
)
add_function_test(
    TestTileFFTNoMathDx,
    "test_tile_ifft_3d_vec2d",
    functools.partial(
        test_tile_ifft,
        wp_dtype=wp.vec2d,
        kernel=tile_ifft_3d_kernel_vec2d,
        data_shape=(FFT_3D_DIM0, FFT_3D_DIM1, 2 * FFT_SIZE_FP64),
    ),
    devices=test_devices,
    check_output=False,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
