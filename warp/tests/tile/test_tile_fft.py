# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import unittest
from typing import Any

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
# Non-power-of-two size exercises the CPU direct DFT fallback.
FFT_SIZE_NPO2 = 48


@wp.kernel
def tile_fft_kernel_vec2f(gx: wp.array2d[wp.vec2f], gy: wp.array2d[wp.vec2f]):
    xy = wp.tile_load(gx, shape=(FFT_SIZE_FP32, FFT_SIZE_FP32))
    wp.tile_fft(xy)
    wp.tile_store(gy, xy)


# The module disables the backward pass, and only the kernel enables it.
@wp.kernel(module="unique", module_options={"enable_backward": False}, enable_backward=True)
def tile_fft_kernel_vec2f_kernel_backward_enabled(gx: wp.array2d[wp.vec2f], gy: wp.array2d[wp.vec2f]):
    xy = wp.tile_load(gx, shape=(FFT_SIZE_FP32, FFT_SIZE_FP32))
    wp.tile_fft(xy)
    wp.tile_store(gy, xy)


# The module disables the backward pass, and only the kernel enables it.
@wp.kernel(module="unique", module_options={"enable_backward": False}, enable_backward=True)
def tile_ifft_kernel_vec2f_kernel_backward_enabled(gx: wp.array2d[wp.vec2f], gy: wp.array2d[wp.vec2f]):
    xy = wp.tile_load(gx, shape=(FFT_SIZE_FP32, FFT_SIZE_FP32))
    wp.tile_ifft(xy)
    wp.tile_store(gy, xy)


def make_tile_fft_shared_helper_kernels(index):
    """Return a forward-only kernel and a backward kernel that share a new generic ``tile_fft()`` helper.

    Both kernels go in one module, so the same build of the helper serves both.
    """

    @wp.func
    def helper(gx: Any, gy: Any):
        xy = wp.tile_load(gx, shape=(FFT_SIZE_FP32, FFT_SIZE_FP32))
        wp.tile_fft(xy)
        wp.tile_store(gy, xy)

    @wp.kernel(
        module="test_tile_fft_shared_helper", enable_backward=False, name=f"tile_fft_shared_helper_forward_{index}"
    )
    def forward_kernel(gx: wp.array2d[wp.vec2f], gy: wp.array2d[wp.vec2f]):
        helper(gx, gy)

    @wp.kernel(module="test_tile_fft_shared_helper", name=f"tile_fft_shared_helper_backward_{index}")
    def backward_kernel(gx: wp.array2d[wp.vec2f], gy: wp.array2d[wp.vec2f]):
        helper(gx, gy)

    return forward_kernel, backward_kernel


# Kernels are built in the order of their content hashes, so several independent pairs catch a result
# that depends on which kernel of a pair is built first.
tile_fft_shared_helper_kernels = [make_tile_fft_shared_helper_kernels(i) for i in range(4)]


def test_tile_fft_shared_helper_backward(test, device):
    """Differentiate generic ``tile_fft()`` helpers that forward-only kernels in their module also call."""
    if not wp._src.context.runtime.core.wp_is_mathdx_enabled():
        test.skipTest("Warp was not built with MathDx support")

    data_shape = (FFT_SIZE_FP32, 2 * FFT_SIZE_FP32)
    X = np.random.default_rng(42).random(data_shape, dtype=np.float32)
    expected = np.fft.fft(X.view(np.complex64), axis=-1)

    for forward_kernel, backward_kernel in tile_fft_shared_helper_kernels:
        with test.subTest(kernel=backward_kernel.key):
            X_wp = wp.array(X, dtype=wp.vec2f, device=device)
            Y_wp = wp.zeros_like(X_wp)
            wp.launch_tiled(forward_kernel, dim=[1], inputs=[X_wp], outputs=[Y_wp], block_dim=TILE_DIM, device=device)
            assert_np_equal(Y_wp.numpy().view(np.complex64).reshape(expected.shape), expected, tol=1.0e-4)

            test_tile_fft(test, device, wp.vec2f, backward_kernel, data_shape)


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


# Non-power-of-two CPU FFT kernels. These exercise the direct-DFT fallback and
# are CPU-only — a non-power-of-two fft_size violates the GPU codegen
# constraints. ``module="test_fft_npo2"`` places them in a shared Warp module
# separate from the surrounding test module, so the module is only compiled
# when launched on CPU.
@wp.kernel(module="test_fft_npo2")
def tile_fft_kernel_vec2f_npo2(gx: wp.array2d[wp.vec2f], gy: wp.array2d[wp.vec2f]):
    xy = wp.tile_load(gx, shape=(FFT_SIZE_NPO2, FFT_SIZE_NPO2))
    wp.tile_fft(xy)
    wp.tile_store(gy, xy)


@wp.kernel(module="test_fft_npo2")
def tile_fft_kernel_vec2d_npo2(gx: wp.array2d[wp.vec2d], gy: wp.array2d[wp.vec2d]):
    xy = wp.tile_load(gx, shape=(FFT_SIZE_NPO2, FFT_SIZE_NPO2))
    wp.tile_fft(xy)
    wp.tile_store(gy, xy)


@wp.kernel(module="test_fft_npo2")
def tile_ifft_kernel_vec2f_npo2(gx: wp.array2d[wp.vec2f], gy: wp.array2d[wp.vec2f]):
    xy = wp.tile_load(gx, shape=(FFT_SIZE_NPO2, FFT_SIZE_NPO2))
    wp.tile_ifft(xy)
    wp.tile_store(gy, xy)


@wp.kernel(module="test_fft_npo2")
def tile_ifft_kernel_vec2d_npo2(gx: wp.array2d[wp.vec2d], gy: wp.array2d[wp.vec2d]):
    xy = wp.tile_load(gx, shape=(FFT_SIZE_NPO2, FFT_SIZE_NPO2))
    wp.tile_ifft(xy)
    wp.tile_store(gy, xy)


def test_tile_fft(test, device, wp_dtype, kernel, data_shape, block_dim=TILE_DIM):
    # CUDA path requires MathDx for cuFFTDx; CPU path has a built-in implementation.
    wp_device = wp.get_device(device)
    if wp_device.is_cuda and not wp._src.context.runtime.core.wp_is_mathdx_enabled():
        test.skipTest("Warp was not built with MathDx support")
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
        wp.launch_tiled(kernel, dim=[1], inputs=[X_wp, Y_wp], block_dim=block_dim, device=device)

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


def test_tile_ifft(test, device, wp_dtype, kernel, data_shape, block_dim=TILE_DIM):
    # CUDA path requires MathDx for cuFFTDx; CPU path has a built-in implementation.
    wp_device = wp.get_device(device)
    if wp_device.is_cuda and not wp._src.context.runtime.core.wp_is_mathdx_enabled():
        test.skipTest("Warp was not built with MathDx support")
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
        wp.launch_tiled(kernel, dim=[1], inputs=[X_wp, Y_wp], block_dim=block_dim, device=device)

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


def test_tile_fft_func_composition(test, device, inverse, mode, mathdx, wp_dtype=wp.vec2f, backward=True):
    """Preserve FFT values and gradients when a helper reads the register tile."""

    @wp.func
    def read_element(t: wp.tile[wp_dtype, 2, 64]):
        return t[0, 1]

    @wp.kernel(module="unique", enable_backward=backward, module_options={"enable_mathdx_fft": mathdx})
    def compute(x: wp.array2d[wp_dtype], y: wp.array2d[wp_dtype], sample: wp.array[wp_dtype]):
        _, lane = wp.tid()
        t = wp.tile_load(x, shape=(2, 64), storage="register")

        if wp.static(mode == "before"):
            value = read_element(t)
            if lane == 0:
                sample[0] = value

        if wp.static(inverse):
            wp.tile_ifft(t)
        else:
            wp.tile_fft(t)

        if wp.static(mode != "before"):
            value = read_element(t)
            if lane == 0:
                sample[0] = value

        wp.tile_store(y, t)

    np_dtype = np.float32 if wp_dtype == wp.vec2f else np.float64
    complex_dtype = np.complex64 if wp_dtype == wp.vec2f else np.complex128
    shape = (2, 64)
    rng = np.random.default_rng(42)
    data = rng.standard_normal((*shape, 2)).astype(np_dtype)
    x = wp.array(data, dtype=wp_dtype, device=device, requires_grad=backward)
    y = wp.zeros_like(x)
    sample = wp.zeros(1, dtype=wp_dtype, device=device, requires_grad=backward)

    values = data.view(complex_dtype).reshape(shape)
    transform_input = values
    expected = np.fft.ifft(transform_input, axis=-1) * 64 if inverse else np.fft.fft(transform_input, axis=-1)
    expected_sample = transform_input[0, 1] if mode == "before" else expected[0, 1]
    expected_output = expected
    tol = 3.0e-5 if wp_dtype == wp.vec2f else 1.0e-12

    with wp.Tape() as tape:
        wp.launch_tiled(compute, dim=[1], inputs=[x], outputs=[y, sample], block_dim=32, device=device)
    np.testing.assert_allclose(y.numpy().view(complex_dtype).reshape(shape), expected_output, rtol=tol, atol=tol)
    np.testing.assert_allclose(sample.numpy().view(complex_dtype).ravel(), [expected_sample], rtol=tol, atol=tol)

    if backward:
        seed = rng.standard_normal((*shape, 2)).astype(np_dtype)
        sample_seed = np.array([[0.3, -0.7]], dtype=np_dtype)
        tape.backward(
            grads={
                y: wp.array(seed, dtype=wp_dtype, device=device),
                sample: wp.array(sample_seed, dtype=wp_dtype, device=device),
            }
        )
        adj_output = seed.view(complex_dtype).reshape(shape).copy()
        adj_transform = adj_output
        if mode != "before":
            adj_transform[0, 1] += sample_seed.view(complex_dtype).item()
        expected_grad = np.fft.fft(adj_transform, axis=-1) if inverse else np.fft.ifft(adj_transform, axis=-1) * 64
        if mode == "before":
            expected_grad[0, 1] += sample_seed.view(complex_dtype).item()
        np.testing.assert_allclose(x.grad.numpy().view(complex_dtype).reshape(shape), expected_grad, rtol=tol, atol=tol)


test_devices = get_test_devices()


class TestTileFFT(unittest.TestCase):
    pass


add_function_test(
    TestTileFFT,
    "test_tile_fft_after_func_forward_only",
    functools.partial(test_tile_fft_func_composition, inverse=False, mode="after", mathdx=True, backward=False),
    devices=get_cuda_test_devices(),
    check_output=False,
)


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
    devices=test_devices,
    check_output=False,
)
add_function_test(
    TestTileFFT,
    "test_tile_fft_2d_vec2f_kernel_backward_enabled",
    functools.partial(
        test_tile_fft,
        wp_dtype=wp.vec2f,
        kernel=tile_fft_kernel_vec2f_kernel_backward_enabled,
        data_shape=(FFT_SIZE_FP32, 2 * FFT_SIZE_FP32),
    ),
    devices=test_devices,
    check_output=False,
)
add_function_test(
    TestTileFFT,
    "test_tile_ifft_2d_vec2f_kernel_backward_enabled",
    functools.partial(
        test_tile_ifft,
        wp_dtype=wp.vec2f,
        kernel=tile_ifft_kernel_vec2f_kernel_backward_enabled,
        data_shape=(FFT_SIZE_FP32, 2 * FFT_SIZE_FP32),
    ),
    devices=test_devices,
    check_output=False,
)
# CUDA only: the CPU path has no adjoint LTOs, and a CPU build first would leave the helper marked as
# reached by a backward kernel, hiding a failure in any rule that reads the mark
add_function_test(
    TestTileFFT,
    "test_tile_fft_shared_helper_backward",
    test_tile_fft_shared_helper_backward,
    devices=get_selected_cuda_test_devices(),
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
    devices=test_devices,
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
    devices=test_devices,
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
    devices=test_devices,
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
    devices=test_devices,
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
    devices=test_devices,
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
    devices=test_devices,
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
    devices=test_devices,
    check_output=False,
)
add_function_test(
    TestTileFFT,
    "test_tile_fft_cpu_npo2_vec2f",
    functools.partial(
        test_tile_fft,
        wp_dtype=wp.vec2f,
        kernel=tile_fft_kernel_vec2f_npo2,
        data_shape=(FFT_SIZE_NPO2, 2 * FFT_SIZE_NPO2),
        block_dim=1,
    ),
    devices=get_cpu_test_devices(),
    check_output=False,
)
add_function_test(
    TestTileFFT,
    "test_tile_fft_cpu_npo2_vec2d",
    functools.partial(
        test_tile_fft,
        wp_dtype=wp.vec2d,
        kernel=tile_fft_kernel_vec2d_npo2,
        data_shape=(FFT_SIZE_NPO2, 2 * FFT_SIZE_NPO2),
        block_dim=1,
    ),
    devices=get_cpu_test_devices(),
    check_output=False,
)
add_function_test(
    TestTileFFT,
    "test_tile_ifft_cpu_npo2_vec2f",
    functools.partial(
        test_tile_ifft,
        wp_dtype=wp.vec2f,
        kernel=tile_ifft_kernel_vec2f_npo2,
        data_shape=(FFT_SIZE_NPO2, 2 * FFT_SIZE_NPO2),
        block_dim=1,
    ),
    devices=get_cpu_test_devices(),
    check_output=False,
)
add_function_test(
    TestTileFFT,
    "test_tile_ifft_cpu_npo2_vec2d",
    functools.partial(
        test_tile_ifft,
        wp_dtype=wp.vec2d,
        kernel=tile_ifft_kernel_vec2d_npo2,
        data_shape=(FFT_SIZE_NPO2, 2 * FFT_SIZE_NPO2),
        block_dim=1,
    ),
    devices=get_cpu_test_devices(),
    check_output=False,
)

cpu_block_fft_tests = (
    (
        "test_tile_fft_2d_vec2f_cpu_blocks",
        functools.partial(
            test_tile_fft,
            wp_dtype=wp.vec2f,
            kernel=tile_fft_kernel_vec2f,
            data_shape=(FFT_SIZE_FP32, 2 * FFT_SIZE_FP32),
        ),
    ),
    (
        "test_tile_fft_2d_vec2d_cpu_blocks",
        functools.partial(
            test_tile_fft,
            wp_dtype=wp.vec2d,
            kernel=tile_fft_kernel_vec2d,
            data_shape=(FFT_SIZE_FP64, 2 * FFT_SIZE_FP64),
        ),
    ),
    (
        "test_tile_ifft_2d_vec2f_cpu_blocks",
        functools.partial(
            test_tile_ifft,
            wp_dtype=wp.vec2f,
            kernel=tile_ifft_kernel_vec2f,
            data_shape=(FFT_SIZE_FP32, 2 * FFT_SIZE_FP32),
        ),
    ),
    (
        "test_tile_ifft_2d_vec2d_cpu_blocks",
        functools.partial(
            test_tile_ifft,
            wp_dtype=wp.vec2d,
            kernel=tile_ifft_kernel_vec2d,
            data_shape=(FFT_SIZE_FP64, 2 * FFT_SIZE_FP64),
        ),
    ),
    (
        "test_tile_fft_3d_vec2f_cpu_blocks",
        functools.partial(
            test_tile_fft,
            wp_dtype=wp.vec2f,
            kernel=tile_fft_3d_kernel_vec2f,
            data_shape=(FFT_3D_DIM0, FFT_3D_DIM1, 2 * FFT_SIZE_FP32),
        ),
    ),
    (
        "test_tile_fft_3d_vec2d_cpu_blocks",
        functools.partial(
            test_tile_fft,
            wp_dtype=wp.vec2d,
            kernel=tile_fft_3d_kernel_vec2d,
            data_shape=(FFT_3D_DIM0, FFT_3D_DIM1, 2 * FFT_SIZE_FP64),
        ),
    ),
    (
        "test_tile_ifft_3d_vec2f_cpu_blocks",
        functools.partial(
            test_tile_ifft,
            wp_dtype=wp.vec2f,
            kernel=tile_ifft_3d_kernel_vec2f,
            data_shape=(FFT_3D_DIM0, FFT_3D_DIM1, 2 * FFT_SIZE_FP32),
        ),
    ),
    (
        "test_tile_ifft_3d_vec2d_cpu_blocks",
        functools.partial(
            test_tile_ifft,
            wp_dtype=wp.vec2d,
            kernel=tile_ifft_3d_kernel_vec2d,
            data_shape=(FFT_3D_DIM0, FFT_3D_DIM1, 2 * FFT_SIZE_FP64),
        ),
    ),
)
for name, func in cpu_block_fft_tests:
    add_function_test(
        TestTileFFT,
        name,
        func,
        devices=get_cpu_test_devices(),
        check_output=False,
        enable_cpu_blocks=True,
    )


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
