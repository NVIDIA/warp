# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility tests for launch bounds and ``wp.tid()`` mapping."""

import unittest
from unittest import mock

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

BLOCK_DIM = 64
TILE_M = wp.constant(8)
TILE_N = wp.constant(4)


@wp.kernel
def regular_1d_kernel(out: wp.array[int]):
    i = wp.tid()
    out[i] = i


def test_regular_1d(test, device):
    n = 128
    out = wp.zeros(n, dtype=int, device=device)
    wp.launch(regular_1d_kernel, dim=n, inputs=[out], device=device)
    np.testing.assert_array_equal(out.numpy(), np.arange(n))


@wp.kernel
def regular_2d_kernel(out: wp.array2d[int], m: int, n: int):
    i, j = wp.tid()
    out[i, j] = i * n + j


def test_regular_2d(test, device):
    m, n = 8, 16
    out = wp.zeros((m, n), dtype=int, device=device)
    wp.launch(regular_2d_kernel, dim=[m, n], inputs=[out, m, n], device=device)
    np.testing.assert_array_equal(out.numpy(), np.arange(m * n).reshape(m, n))


@wp.kernel
def regular_3d_kernel(out: wp.array3d[int], m: int, n: int, k: int):
    i, j, z = wp.tid()
    out[i, j, z] = i * n * k + j * k + z


def test_regular_3d(test, device):
    m, n, k = 4, 8, 16
    out = wp.zeros((m, n, k), dtype=int, device=device)
    wp.launch(regular_3d_kernel, dim=[m, n, k], inputs=[out, m, n, k], device=device)
    np.testing.assert_array_equal(out.numpy(), np.arange(m * n * k).reshape(m, n, k))


@wp.kernel
def regular_4d_kernel(out: wp.array4d[int], b: int, c: int, d: int):
    i, j, k, l = wp.tid()
    out[i, j, k, l] = i * b * c * d + j * c * d + k * d + l


def test_regular_4d(test, device):
    a, b, c, d = 2, 3, 4, 5
    out = wp.zeros((a, b, c, d), dtype=int, device=device)
    wp.launch(regular_4d_kernel, dim=[a, b, c, d], inputs=[out, b, c, d], device=device)
    np.testing.assert_array_equal(out.numpy(), np.arange(a * b * c * d).reshape(a, b, c, d))


@wp.kernel
def count_visits_1d(counts: wp.array[wp.int32]):
    i = wp.tid()
    wp.atomic_add(counts, i, 1)


@wp.kernel
def count_visits_2d(counts: wp.array2d[wp.int32]):
    i, j = wp.tid()
    wp.atomic_add(counts, i, j, 1)


@wp.kernel
def count_visits_3d(counts: wp.array3d[wp.int32]):
    i, j, k = wp.tid()
    wp.atomic_add(counts, i, j, k, 1)


@wp.kernel
def count_visits_4d(counts: wp.array4d[wp.int32]):
    i, j, k, l = wp.tid()
    wp.atomic_add(counts, i, j, k, l, 1)


_COUNT_KERNELS = {1: count_visits_1d, 2: count_visits_2d, 3: count_visits_3d, 4: count_visits_4d}


def _dim_tuple(dim):
    return (dim,) if isinstance(dim, int) else tuple(dim)


def _prod(values):
    total = 1
    for value in values:
        total *= value
    return total


def _dim_matrix_case(device, launch_dim, kernel_dim):
    launch_dim = _dim_tuple(launch_dim)
    padded = launch_dim + (1,) * max(0, kernel_dim - len(launch_dim))
    kept = padded[:kernel_dim]
    counts = wp.zeros(kept, dtype=wp.int32, device=device)
    wp.launch(_COUNT_KERNELS[kernel_dim], dim=launch_dim, inputs=[counts], device=device)
    return counts.numpy(), kept


_DIM_CASES = [
    (128, 2, "scalar_into_2d_pads_to_(N,1)"),
    (64, 3, "scalar_into_3d_pads_to_(N,1,1)"),
    ((8, 16), 1, "2d_launch_over_1d_kernel_aliases_16x"),
    ((4, 8, 3), 2, "3d_launch_over_2d_kernel_aliases_3x"),
    ((4, 8, 3, 2), 2, "4d_launch_over_2d_kernel_aliases_6x"),
]


def test_dim_matrix(test, device):
    """Preserve padding and aliasing semantics for mismatched launch/tid arities."""
    for launch_dim, kernel_dim, label in _DIM_CASES:
        launch_tuple = _dim_tuple(launch_dim)
        counts, kept = _dim_matrix_case(device, launch_tuple, kernel_dim)
        expected = np.full(kept, _prod(launch_tuple) // _prod(kept), dtype=np.int32)
        np.testing.assert_array_equal(counts, expected, err_msg=label)


@wp.kernel
def no_tid_kernel(out: wp.array[float], val: float):
    wp.atomic_add(out, 0, val)


def test_no_tid_kernel_multidim(test, device):
    out = wp.zeros(1, dtype=float, device=device)
    wp.launch(no_tid_kernel, dim=[2, 3], inputs=[out, 1.0], device=device)
    np.testing.assert_allclose(out.numpy()[0], 6.0)


@wp.kernel
def tiled_1d_kernel(src: wp.array[float], dst: wp.array[float]):
    i = wp.tid()
    tile = wp.tile_load(src, shape=TILE_N, offset=i * TILE_N)
    wp.tile_store(dst, tile, offset=i * TILE_N)


def test_tiled_1d(test, device):
    n = TILE_N * 5
    src = wp.full(n, 42.0, dtype=float, device=device)
    dst = wp.zeros(n, dtype=float, device=device)
    wp.launch_tiled(tiled_1d_kernel, dim=[int(n / TILE_N)], inputs=[src, dst], block_dim=BLOCK_DIM, device=device)
    np.testing.assert_array_equal(dst.numpy(), src.numpy())


@wp.kernel
def tiled_2d_kernel(src: wp.array2d[float], dst: wp.array2d[float]):
    i, j = wp.tid()
    tile = wp.tile_load(src, shape=(TILE_M, TILE_N), offset=(i * TILE_M, j * TILE_N))
    wp.tile_store(dst, tile, offset=(i * TILE_M, j * TILE_N))


def test_tiled_2d(test, device):
    m = TILE_M * 3
    n = TILE_N * 2
    rng = np.random.default_rng(42)
    src_np = rng.random((m, n)).astype(np.float32)
    src = wp.array(src_np, device=device)
    dst = wp.zeros((m, n), dtype=float, device=device)
    wp.launch_tiled(
        tiled_2d_kernel,
        dim=[int(m / TILE_M), int(n / TILE_N)],
        inputs=[src, dst],
        block_dim=BLOCK_DIM,
        device=device,
    )
    np.testing.assert_allclose(dst.numpy(), src_np, rtol=1e-5)


@wp.kernel
def no_tid_counter(out: wp.array[wp.int32]):
    wp.atomic_add(out, 0, 1)


def test_tiled_no_tid_multidim_thread_count(test, device):
    m, n, block_dim = 2, 3, 32
    effective_block_dim = 1 if device.is_cpu else block_dim
    out = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch_tiled(no_tid_counter, dim=[m, n], inputs=[out], block_dim=block_dim, device=device)
    test.assertEqual(out.numpy()[0], m * n * effective_block_dim)


def test_tiled_accepts_numpy_integer_scalar_dim(test, device):
    n, block_dim = np.int32(5), 32
    out = wp.zeros(1, dtype=wp.int32, device=device)
    wp.launch_tiled(no_tid_counter, dim=n, inputs=[out], block_dim=block_dim, device=device)
    test.assertEqual(out.numpy()[0], int(n))


class IndexLike:
    def __init__(self, value):
        self.value = value

    def __index__(self):
        return self.value


def test_canonicalize_dim_contract(test, device):
    from warp._src import context  # noqa: PLC0415

    dim = (1, 2, 3)

    with mock.patch.object(context.operator, "index", wraps=context.operator.index) as index:
        test.assertIs(context._canonicalize_dim(dim), dim)
        test.assertEqual(context._canonicalize_dim([1, 2, 3]), dim)
        index.assert_not_called()

    test.assertEqual(context._canonicalize_dim(np.int32(5)), (5,))
    test.assertEqual(context._canonicalize_dim([np.int32(2), np.int64(3), IndexLike(4)]), (2, 3, 4))

    with test.assertRaises(TypeError):
        context._canonicalize_dim(1.5)

    with test.assertRaises(TypeError):
        context._canonicalize_dim([1, 2.5])


def test_build_launch_bounds_from_tuple_preserves_large_coord_mult(test, device):
    from warp._src import context  # noqa: PLC0415

    large_mult = 2**33 + 3
    bounds = context._build_launch_bounds_from_tuple((1, large_mult), 1)

    test.assertEqual(bounds.shape[0], 1)
    test.assertEqual(bounds.coord_mult, large_mult)
    test.assertEqual(bounds.size, large_mult)


def test_launch_normalizes_dim_once(test, device):
    from warp._src import context  # noqa: PLC0415

    out = wp.zeros(1, dtype=int, device=device)

    # Protect the launch-dim fast path: launch() should reuse the normalized
    # tuple instead of canonicalizing again while building launch bounds.
    with mock.patch.object(context, "_canonicalize_dim", wraps=context._canonicalize_dim) as canonicalize_dim:
        wp.launch(regular_1d_kernel, dim=(1,), inputs=[out], device=device)

    canonicalize_dim.assert_called_once()


def test_launch_rejects_negative_dim(test, device):
    out = wp.zeros(1, dtype=int, device=device)

    with test.assertRaisesRegex(ValueError, "non-negative"):
        wp.launch(regular_1d_kernel, dim=-1, inputs=[out], device=device)

    launch = wp.launch(regular_1d_kernel, dim=1, inputs=[out], device=device, record_cmd=True)
    with test.assertRaisesRegex(ValueError, "non-negative"):
        launch.set_dim([1, -1])

    counter = wp.zeros(1, dtype=wp.int32, device=device)
    with test.assertRaisesRegex(ValueError, "non-negative"):
        wp.launch_tiled(no_tid_counter, dim=np.int32(-1), inputs=[counter], block_dim=32, device=device)


@wp.kernel
def manual_tiled_kernel(out: wp.array3d[int], m: int, n: int, block_dim: int):
    i, j, t = wp.tid()
    out[i, j, t] = i * n * block_dim + j * block_dim + t


def test_manual_tiled(test, device):
    m, n, block_dim = 4, 8, 32
    out = wp.zeros((m, n, block_dim), dtype=int, device=device)
    wp.launch(
        manual_tiled_kernel, dim=[m, n, block_dim], inputs=[out, m, n, block_dim], block_dim=block_dim, device=device
    )
    np.testing.assert_array_equal(out.numpy(), np.arange(m * n * block_dim).reshape(m, n, block_dim))


def _run_tiled_case(device, user_dim, kernel_dim, block_dim):
    effective_block_dim = 1 if device.is_cpu else block_dim
    full_dim = (*user_dim, effective_block_dim)
    if len(full_dim) >= kernel_dim:
        kept = full_dim[:kernel_dim]
    else:
        kept = full_dim + (1,) * (kernel_dim - len(full_dim))

    counts = wp.zeros(kept, dtype=wp.int32, device=device)
    wp.launch_tiled(_COUNT_KERNELS[kernel_dim], dim=user_dim, inputs=[counts], block_dim=block_dim, device=device)
    return counts.numpy()


def test_tiled_matrix_1d_user(test, device):
    m, block_dim = 4, 8
    effective_block_dim = 1 if device.is_cpu else block_dim

    np.testing.assert_array_equal(
        _run_tiled_case(device, (m,), 1, block_dim), np.full(m, effective_block_dim, dtype=np.int32)
    )
    np.testing.assert_array_equal(
        _run_tiled_case(device, (m,), 2, block_dim), np.ones((m, effective_block_dim), dtype=np.int32)
    )
    np.testing.assert_array_equal(
        _run_tiled_case(device, (m,), 3, block_dim), np.ones((m, effective_block_dim, 1), dtype=np.int32)
    )
    np.testing.assert_array_equal(
        _run_tiled_case(device, (m,), 4, block_dim), np.ones((m, effective_block_dim, 1, 1), dtype=np.int32)
    )


def test_tiled_matrix_2d_user(test, device):
    m, n, block_dim = 3, 4, 8
    effective_block_dim = 1 if device.is_cpu else block_dim

    np.testing.assert_array_equal(
        _run_tiled_case(device, (m, n), 1, block_dim), np.full(m, n * effective_block_dim, dtype=np.int32)
    )
    np.testing.assert_array_equal(
        _run_tiled_case(device, (m, n), 2, block_dim), np.full((m, n), effective_block_dim, dtype=np.int32)
    )
    np.testing.assert_array_equal(
        _run_tiled_case(device, (m, n), 3, block_dim), np.ones((m, n, effective_block_dim), dtype=np.int32)
    )
    np.testing.assert_array_equal(
        _run_tiled_case(device, (m, n), 4, block_dim), np.ones((m, n, effective_block_dim, 1), dtype=np.int32)
    )


def test_tiled_matrix_3d_user(test, device):
    m, n, k, block_dim = 2, 3, 2, 8
    effective_block_dim = 1 if device.is_cpu else block_dim

    np.testing.assert_array_equal(
        _run_tiled_case(device, (m, n, k), 1, block_dim),
        np.full(m, n * k * effective_block_dim, dtype=np.int32),
    )
    np.testing.assert_array_equal(
        _run_tiled_case(device, (m, n, k), 2, block_dim),
        np.full((m, n), k * effective_block_dim, dtype=np.int32),
    )
    np.testing.assert_array_equal(
        _run_tiled_case(device, (m, n, k), 3, block_dim),
        np.full((m, n, k), effective_block_dim, dtype=np.int32),
    )
    np.testing.assert_array_equal(
        _run_tiled_case(device, (m, n, k), 4, block_dim),
        np.ones((m, n, k, effective_block_dim), dtype=np.int32),
    )


def test_launch_set_dim(test, device):
    m, n = 8, 8
    out = wp.zeros(m * n, dtype=int, device=device)
    launch = wp.launch(regular_1d_kernel, dim=m, inputs=[out], device=device, record_cmd=True)

    launch.set_dim([m, n])
    launch.launch()

    expected = np.zeros(m * n, dtype=int)
    expected[:m] = np.arange(m)
    np.testing.assert_array_equal(out.numpy(), expected)


def test_launch_rejects_unsupported_rank(test, device):
    out = wp.zeros(1, dtype=int, device=device)

    with test.assertRaisesRegex(ValueError, "at most 4 dimensions"):
        wp.launch(regular_1d_kernel, dim=(1, 1, 1, 1, 2), inputs=[out], device=device)

    with test.assertRaisesRegex(ValueError, "at most 4 dimensions"):
        wp.launch(regular_1d_kernel, dim=(0, 1, 1, 1, 2), inputs=[out], device=device)


def test_launch_bounds_factory_specializes_dimensionality(test, device):
    from warp._src.types import _launch_bounds_classes, launch_bounds_t  # noqa: PLC0415

    bounds_1d = launch_bounds_t((4,))
    bounds_2d = launch_bounds_t((4, 5))

    test.assertIs(type(bounds_1d), _launch_bounds_classes[1])
    test.assertIs(type(bounds_2d), _launch_bounds_classes[2])
    test.assertNotEqual(type(bounds_1d), type(bounds_2d))
    test.assertEqual(len(bounds_1d.shape), 1)
    test.assertEqual(len(bounds_2d.shape), 2)


class TestTemplateLaunchBounds(unittest.TestCase):
    pass


devices = get_test_devices()

add_function_test(TestTemplateLaunchBounds, "test_regular_1d", test_regular_1d, devices=devices)
add_function_test(TestTemplateLaunchBounds, "test_regular_2d", test_regular_2d, devices=devices)
add_function_test(TestTemplateLaunchBounds, "test_regular_3d", test_regular_3d, devices=devices)
add_function_test(TestTemplateLaunchBounds, "test_regular_4d", test_regular_4d, devices=devices)
add_function_test(TestTemplateLaunchBounds, "test_dim_matrix", test_dim_matrix, devices=devices)
add_function_test(TestTemplateLaunchBounds, "test_no_tid_kernel_multidim", test_no_tid_kernel_multidim, devices=devices)
add_function_test(TestTemplateLaunchBounds, "test_tiled_1d", test_tiled_1d, devices=devices)
add_function_test(TestTemplateLaunchBounds, "test_tiled_2d", test_tiled_2d, devices=devices)
add_function_test(
    TestTemplateLaunchBounds,
    "test_tiled_no_tid_multidim_thread_count",
    test_tiled_no_tid_multidim_thread_count,
    devices=devices,
)
add_function_test(
    TestTemplateLaunchBounds,
    "test_tiled_accepts_numpy_integer_scalar_dim",
    test_tiled_accepts_numpy_integer_scalar_dim,
    devices=["cpu"],
)
add_function_test(
    TestTemplateLaunchBounds,
    "test_canonicalize_dim_contract",
    test_canonicalize_dim_contract,
    devices=["cpu"],
)
add_function_test(
    TestTemplateLaunchBounds,
    "test_build_launch_bounds_from_tuple_preserves_large_coord_mult",
    test_build_launch_bounds_from_tuple_preserves_large_coord_mult,
    devices=["cpu"],
)
add_function_test(
    TestTemplateLaunchBounds,
    "test_launch_normalizes_dim_once",
    test_launch_normalizes_dim_once,
    devices=["cpu"],
)
add_function_test(
    TestTemplateLaunchBounds,
    "test_launch_rejects_negative_dim",
    test_launch_rejects_negative_dim,
    devices=["cpu"],
)
add_function_test(TestTemplateLaunchBounds, "test_manual_tiled", test_manual_tiled, devices=devices)
add_function_test(TestTemplateLaunchBounds, "test_tiled_matrix_1d_user", test_tiled_matrix_1d_user, devices=devices)
add_function_test(TestTemplateLaunchBounds, "test_tiled_matrix_2d_user", test_tiled_matrix_2d_user, devices=devices)
add_function_test(TestTemplateLaunchBounds, "test_tiled_matrix_3d_user", test_tiled_matrix_3d_user, devices=devices)
add_function_test(TestTemplateLaunchBounds, "test_launch_set_dim", test_launch_set_dim, devices=devices)
add_function_test(
    TestTemplateLaunchBounds,
    "test_launch_rejects_unsupported_rank",
    test_launch_rejects_unsupported_rank,
    devices=devices,
)
add_function_test(
    TestTemplateLaunchBounds,
    "test_launch_bounds_factory_specializes_dimensionality",
    test_launch_bounds_factory_specializes_dimensionality,
    devices=["cpu"],
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
