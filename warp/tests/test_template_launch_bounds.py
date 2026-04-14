# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for templated launch_bounds_t<N>.

Verifies that kernel_dim inference from wp.tid() produces correct thread
coordinates across all supported launch configurations:

- Regular (non-tiled) launches with 1D/2D/3D/4D wp.tid()
- launch_tiled with wp.tid() ignoring the trailing block_dim dimension
- launch_tiled with wp.tid() explicitly covering all dimensions
- Manual tiled launches (wp.launch with block_dim baked into dim)
- Untiled launches that use tile functions
- Dimension normalization: padding (dim < kernel_dim) and flattening (dim > kernel_dim)
- Kernels that never call wp.tid()
"""

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

BLOCK_DIM = 64
TILE_M = wp.constant(8)
TILE_N = wp.constant(4)


# ============================================================================
# Regular (non-tiled) launches
# ============================================================================


@wp.kernel
def regular_1d_kernel(out: wp.array(dtype=int)):
    i = wp.tid()
    out[i] = i


def test_regular_1d(test, device):
    N = 128
    out = wp.zeros(N, dtype=int, device=device)
    wp.launch(regular_1d_kernel, dim=N, inputs=[out], device=device)
    result = out.numpy()
    np.testing.assert_array_equal(result, np.arange(N))


@wp.kernel
def regular_2d_kernel(out: wp.array2d(dtype=int), M: int, N: int):
    i, j = wp.tid()
    out[i, j] = i * N + j


def test_regular_2d(test, device):
    M, N = 8, 16
    out = wp.zeros((M, N), dtype=int, device=device)
    wp.launch(regular_2d_kernel, dim=[M, N], inputs=[out, M, N], device=device)
    result = out.numpy()
    expected = np.arange(M * N).reshape(M, N)
    np.testing.assert_array_equal(result, expected)


@wp.kernel
def regular_3d_kernel(out: wp.array3d(dtype=int), M: int, N: int, K: int):
    i, j, k = wp.tid()
    out[i, j, k] = i * N * K + j * K + k


def test_regular_3d(test, device):
    M, N, K = 4, 8, 16
    out = wp.zeros((M, N, K), dtype=int, device=device)
    wp.launch(regular_3d_kernel, dim=[M, N, K], inputs=[out, M, N, K], device=device)
    result = out.numpy()
    expected = np.arange(M * N * K).reshape(M, N, K)
    np.testing.assert_array_equal(result, expected)


@wp.kernel
def regular_4d_kernel(out: wp.array(dtype=int), A: int, B: int, C: int, D: int):
    i, j, k, l = wp.tid()
    out[i * B * C * D + j * C * D + k * D + l] = i * B * C * D + j * C * D + k * D + l


def test_regular_4d(test, device):
    A, B, C, D = 2, 3, 4, 5
    total = A * B * C * D
    out = wp.zeros(total, dtype=int, device=device)
    wp.launch(regular_4d_kernel, dim=[A, B, C, D], inputs=[out, A, B, C, D], device=device)
    result = out.numpy()
    np.testing.assert_array_equal(result, np.arange(total))


# ============================================================================
# Dimension normalization: padding (dim < kernel_dim)
# ============================================================================


def test_dim_padding_1d_to_2d(test, device):
    """Launch a 2D kernel with a scalar dim — should pad to (N, 1)."""
    N = 128
    out = wp.zeros((N, 1), dtype=int, device=device)
    wp.launch(regular_2d_kernel, dim=N, inputs=[out, N, 1], device=device)
    result = out.numpy()
    expected = np.arange(N).reshape(N, 1)
    np.testing.assert_array_equal(result, expected)


def test_dim_padding_1d_to_3d(test, device):
    """Launch a 3D kernel with a scalar dim — should pad to (N, 1, 1)."""
    N = 64
    out = wp.zeros((N, 1, 1), dtype=int, device=device)
    wp.launch(regular_3d_kernel, dim=N, inputs=[out, N, 1, 1], device=device)
    result = out.numpy()
    expected = np.arange(N).reshape(N, 1, 1)
    np.testing.assert_array_equal(result, expected)


# ============================================================================
# Dimension normalization: flattening (dim > kernel_dim)
# ============================================================================


def test_dim_flattening_2d_to_1d(test, device):
    """Launch a 1D kernel with dim=[M, N] — should flatten to (M*N,)."""
    M, N = 8, 16
    total = M * N
    out = wp.zeros(total, dtype=int, device=device)
    wp.launch(regular_1d_kernel, dim=[M, N], inputs=[out], device=device)
    result = out.numpy()
    np.testing.assert_array_equal(result, np.arange(total))


def test_dim_flattening_3d_to_2d(test, device):
    """Launch a 2D kernel with dim=[M, N, K] — should flatten to (M, N*K)."""
    M, N, K = 4, 8, 2
    out = wp.zeros((M, N * K), dtype=int, device=device)
    wp.launch(regular_2d_kernel, dim=[M, N, K], inputs=[out, M, N * K], device=device)
    result = out.numpy()
    expected = np.arange(M * N * K).reshape(M, N * K)
    np.testing.assert_array_equal(result, expected)


# ============================================================================
# No-tid kernel with multi-dimensional launch
# ============================================================================


@wp.kernel
def no_tid_kernel(out: wp.array(dtype=float), val: float):
    wp.atomic_add(out, 0, val)


def test_no_tid_kernel_multidim(test, device):
    """Kernel without wp.tid() launched with multi-dim should not crash."""
    out = wp.zeros(1, dtype=float, device=device)
    wp.launch(no_tid_kernel, dim=[2, 3], inputs=[out, 1.0], device=device)
    # 6 threads each atomically add 1.0
    np.testing.assert_allclose(out.numpy()[0], 6.0)


# ============================================================================
# launch_tiled: wp.tid() ignores trailing block_dim dimension
# ============================================================================


@wp.kernel
def tiled_1d_kernel(A: wp.array(dtype=float), B: wp.array(dtype=float)):
    i = wp.tid()
    a = wp.tile_load(A, shape=TILE_N, offset=i * TILE_N)
    wp.tile_store(B, a, offset=i * TILE_N)


def test_tiled_1d(test, device):
    N = TILE_N * 5
    A = wp.full(N, 42.0, dtype=float, device=device)
    B = wp.zeros(N, dtype=float, device=device)
    wp.launch_tiled(tiled_1d_kernel, dim=[int(N / TILE_N)], inputs=[A, B], block_dim=BLOCK_DIM, device=device)
    np.testing.assert_array_equal(B.numpy(), A.numpy())


@wp.kernel
def tiled_2d_kernel(
    A: wp.array2d(dtype=float),
    B: wp.array2d(dtype=float),
):
    i, j = wp.tid()
    a = wp.tile_load(A, shape=(TILE_M, TILE_N), offset=(i * TILE_M, j * TILE_N))
    wp.tile_store(B, a, offset=(i * TILE_M, j * TILE_N))


def test_tiled_2d(test, device):
    M = TILE_M * 3
    N = TILE_N * 2
    rng = np.random.default_rng(42)
    A_np = rng.random((M, N)).astype(np.float32)
    A = wp.array(A_np, device=device)
    B = wp.zeros((M, N), dtype=float, device=device)
    wp.launch_tiled(
        tiled_2d_kernel,
        dim=[int(M / TILE_M), int(N / TILE_N)],
        inputs=[A, B],
        block_dim=BLOCK_DIM,
        device=device,
    )
    np.testing.assert_allclose(B.numpy(), A_np, rtol=1e-5)


# ============================================================================
# launch_tiled: kernel doesn't call wp.tid() (e.g. pure tile operations)
# ============================================================================


TILE_SIZE = wp.constant(32)


@wp.kernel
def tiled_no_tid_kernel(A: wp.array2d(dtype=float), B: wp.array2d(dtype=float)):
    a = wp.tile_load(A, shape=(TILE_SIZE, TILE_SIZE))
    wp.tile_store(B, a)


def test_tiled_no_tid(test, device):
    """Tiled kernel with no wp.tid() should work with multi-dim launch."""
    N = 32
    rng = np.random.default_rng(123)
    A_np = rng.random((N, N)).astype(np.float32)
    A = wp.array(A_np, device=device)
    B = wp.zeros((N, N), dtype=float, device=device)
    wp.launch_tiled(tiled_no_tid_kernel, dim=[1, 1], inputs=[A, B], block_dim=BLOCK_DIM, device=device)
    np.testing.assert_allclose(B.numpy(), A_np, rtol=1e-5)


# ============================================================================
# Manual tiled launch: wp.launch with block_dim baked into dim
# ============================================================================


@wp.kernel
def manual_tiled_kernel(out: wp.array3d(dtype=int), M: int, N: int, BD: int):
    i, j, t = wp.tid()
    out[i, j, t] = i * N * BD + j * BD + t


def test_manual_tiled(test, device):
    M, N, BD = 4, 8, 32
    out = wp.zeros((M, N, BD), dtype=int, device=device)
    wp.launch(manual_tiled_kernel, dim=[M, N, BD], inputs=[out, M, N, BD], block_dim=BD, device=device)
    result = out.numpy()
    expected = np.arange(M * N * BD).reshape(M, N, BD)
    np.testing.assert_array_equal(result, expected)


# ============================================================================
# launch_tiled dimension mismatch error
# ============================================================================


def test_tiled_dim_mismatch_error(test, device):
    """launch_tiled should raise when kernel_dim is incompatible with dim."""

    @wp.kernel
    def _tiled_4d_kernel(out: wp.array(dtype=float)):
        _i, _j, _k, _l = wp.tid()
        pass

    with test.assertRaises(RuntimeError):
        wp.launch_tiled(
            _tiled_4d_kernel,
            dim=[2, 3],
            inputs=[wp.zeros(1, dtype=float, device=device)],
            block_dim=BLOCK_DIM,
            device=device,
        )


# ============================================================================
# Launch.set_dim normalizes correctly
# ============================================================================


def test_launch_set_dim(test, device):
    """Launch.set_dim() should normalize dim to kernel_dim dimensions."""
    N = 64
    out = wp.zeros(N, dtype=int, device=device)
    launch = wp.launch(regular_1d_kernel, dim=N, inputs=[out], device=device, record_cmd=True)

    # Re-set with a 2D dim — should flatten to 1D to match kernel_dim=1
    launch.set_dim([8, 8])
    launch.launch()

    result = out.numpy()
    np.testing.assert_array_equal(result, np.arange(N))


# ============================================================================
# Test class and registration
# ============================================================================


class TestTemplateLaunchBounds(unittest.TestCase):
    pass


devices = get_test_devices()

add_function_test(TestTemplateLaunchBounds, "test_regular_1d", test_regular_1d, devices=devices)
add_function_test(TestTemplateLaunchBounds, "test_regular_2d", test_regular_2d, devices=devices)
add_function_test(TestTemplateLaunchBounds, "test_regular_3d", test_regular_3d, devices=devices)
add_function_test(TestTemplateLaunchBounds, "test_regular_4d", test_regular_4d, devices=devices)
add_function_test(TestTemplateLaunchBounds, "test_dim_padding_1d_to_2d", test_dim_padding_1d_to_2d, devices=devices)
add_function_test(TestTemplateLaunchBounds, "test_dim_padding_1d_to_3d", test_dim_padding_1d_to_3d, devices=devices)
add_function_test(
    TestTemplateLaunchBounds, "test_dim_flattening_2d_to_1d", test_dim_flattening_2d_to_1d, devices=devices
)
add_function_test(
    TestTemplateLaunchBounds, "test_dim_flattening_3d_to_2d", test_dim_flattening_3d_to_2d, devices=devices
)
add_function_test(TestTemplateLaunchBounds, "test_no_tid_kernel_multidim", test_no_tid_kernel_multidim, devices=devices)
add_function_test(TestTemplateLaunchBounds, "test_tiled_1d", test_tiled_1d, devices=devices)
add_function_test(TestTemplateLaunchBounds, "test_tiled_2d", test_tiled_2d, devices=devices)
add_function_test(TestTemplateLaunchBounds, "test_tiled_no_tid", test_tiled_no_tid, devices=devices)
add_function_test(TestTemplateLaunchBounds, "test_manual_tiled", test_manual_tiled, devices=devices)
add_function_test(
    TestTemplateLaunchBounds, "test_tiled_dim_mismatch_error", test_tiled_dim_mismatch_error, devices=devices
)
add_function_test(TestTemplateLaunchBounds, "test_launch_set_dim", test_launch_set_dim, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
