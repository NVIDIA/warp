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
- Launch-dim rank validation: under-rank and over-rank both raise ValueError
- Kernels that never call wp.tid()
"""

import unittest
from types import SimpleNamespace

import numpy as np

import warp as wp

# Private helpers live under warp._src; tests import them directly.
from warp._src.context import _build_rank_error, _prepare_launch_dim, _tid_unpack
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
# Launch-dim rank validation: mismatch raises ValueError with migration options
# ============================================================================


def test_launch_dim_over_rank_kernel_dim_1_error(test, device):
    """dim rank > kernel_dim=1: error lists all 4 intent options."""
    N = 3
    out = wp.zeros(N * N, dtype=int, device=device)
    with test.assertRaises(ValueError) as cm:
        wp.launch(regular_1d_kernel, dim=(N, N), inputs=[out], device=device)
    msg = str(cm.exception)
    test.assertIn("kernel_dim=1", msg)
    test.assertIn("flat linear index over 9 threads: launch with dim=9", msg)
    test.assertIn("first-dim index (0..2, no repetition): launch with dim=3", msg)
    test.assertIn("first-dim index with repetition: unpack as `i, _ = wp.tid()`", msg)
    test.assertIn("per-dim indexing: unpack as `i, j = wp.tid()`", msg)


def test_launch_dim_over_rank_kernel_dim_2_error(test, device):
    """dim rank > kernel_dim=2: launch-only options omitted (they'd require kernel edit too)."""
    M, N, K = 2, 3, 4
    out = wp.zeros((M, N * K), dtype=int, device=device)
    with test.assertRaises(ValueError) as cm:
        wp.launch(regular_2d_kernel, dim=(M, N, K), inputs=[out, M, N * K], device=device)
    msg = str(cm.exception)
    test.assertIn("kernel_dim=2", msg)
    # Guard: misleading launch-only fixes must NOT appear when kernel_dim >= 2.
    test.assertNotIn("flat linear", msg)
    test.assertNotIn(f"launch with dim={M * N * K}", msg)
    test.assertNotIn("no repetition", msg)
    # Kernel-side fixes still appear.
    test.assertIn("first-dim index with repetition: unpack as `i, _, _ = wp.tid()`", msg)
    test.assertIn("per-dim indexing: unpack as `i, j, k = wp.tid()`", msg)


def test_launch_dim_under_rank_error(test, device):
    """dim rank < kernel_dim: error lists the two fix options."""
    N = 10
    out = wp.zeros((N, 1), dtype=int, device=device)
    with test.assertRaises(ValueError) as cm:
        wp.launch(regular_2d_kernel, dim=N, inputs=[out, N, 1], device=device)
    msg = str(cm.exception)
    test.assertIn("kernel_dim=2", msg)
    test.assertIn("keep 2-D kernel, launch with matching rank: dim=(10, 1)", msg)
    test.assertIn("change kernel to unpack 1 variable: `i = wp.tid()`", msg)


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
    """launch_tiled should raise ValueError with migration hint when kernel_dim is incompatible with dim."""

    @wp.kernel
    def _tiled_4d_kernel(out: wp.array(dtype=float)):
        _i, _j, _k, _l = wp.tid()
        pass

    with test.assertRaises(ValueError) as cm:
        wp.launch_tiled(
            _tiled_4d_kernel,
            dim=[2, 3],
            inputs=[wp.zeros(1, dtype=float, device=device)],
            block_dim=BLOCK_DIM,
            device=device,
        )
    msg = str(cm.exception)
    test.assertIn("kernel_dim=4", msg)
    # tiled hint should appear because this is a tiled launch
    test.assertIn("For launch_tiled, dim may also match kernel_dim - 1", msg)


# ============================================================================
# Launch.set_dim rank validation
# ============================================================================


def test_set_dim_matching_rank(test, device):
    """Launch.set_dim() with rank matching kernel_dim should succeed."""
    N = 64
    out = wp.zeros(N, dtype=int, device=device)
    launch = wp.launch(regular_1d_kernel, dim=N, inputs=[out], device=device, record_cmd=True)

    # Re-set with a new 1D dim — matches kernel_dim=1, should succeed
    launch.set_dim(32)
    launch.launch()

    result = out.numpy()
    # only the first 32 entries were written; rest remain 0
    np.testing.assert_array_equal(result[:32], np.arange(32))
    np.testing.assert_array_equal(result[32:], np.zeros(N - 32))


def test_set_dim_rank_mismatch_error(test, device):
    """Launch.set_dim() with rank != kernel_dim should raise ValueError."""
    N = 64
    out = wp.zeros(N, dtype=int, device=device)
    launch = wp.launch(regular_1d_kernel, dim=N, inputs=[out], device=device, record_cmd=True)

    with test.assertRaises(ValueError) as cm:
        launch.set_dim([8, 8])
    msg = str(cm.exception)
    test.assertIn("kernel_dim=1", msg)
    test.assertIn("flat linear index over 64 threads", msg)


def test_set_dim_preserves_tiled_flag(test, device):
    """Launch.set_dim() on a recorded tiled launch should preserve tiled semantics."""
    N = TILE_N * 5
    A = wp.full(N, 42.0, dtype=float, device=device)
    B = wp.zeros(N, dtype=float, device=device)
    launch = wp.launch_tiled(
        tiled_1d_kernel,
        dim=[int(N / TILE_N)],
        inputs=[A, B],
        block_dim=BLOCK_DIM,
        device=device,
        record_cmd=True,
    )
    # set_dim should accept user-rank dim (len == kernel_dim for this kernel)
    launch.set_dim([int(N / TILE_N)])
    test.assertTrue(launch.bounds.tiled, "tiled flag should be preserved after set_dim")
    launch.launch()
    np.testing.assert_array_equal(B.numpy(), A.numpy())


def test_set_dim_tiled_explicit_block_axis(test, device):
    """Replaying a recorded launch_tiled whose kernel unpacks the block axis.

    When ``wp.tid()`` covers the block_dim axis (kernel_dim == len(dim) + 1),
    ``_construct_tiled_bounds`` appends block_dim to the shape and leaves the
    C-struct ``bounds.tiled`` flag False. ``set_dim`` must still recognize the
    launch as tiled (via ``self.tiled``) and route through
    ``_construct_tiled_bounds`` — otherwise the non-tiled path would raise
    ``ValueError`` because ``len(dim) == kernel_dim - 1``.
    """

    @wp.kernel
    def _tiled_block_axis_kernel(out: wp.array(dtype=int)):
        _i, _t = wp.tid()

    N = 4
    out = wp.zeros(N * BLOCK_DIM, dtype=int, device=device)
    launch = wp.launch_tiled(
        _tiled_block_axis_kernel,
        dim=[N],
        inputs=[out],
        block_dim=BLOCK_DIM,
        device=device,
        record_cmd=True,
    )
    # This path leaves bounds.tiled False — block axis is baked into the shape.
    test.assertFalse(launch.bounds.tiled)
    test.assertTrue(launch.tiled)
    # Regression: previously set_dim consulted bounds.tiled and raised on this
    # shape. It must now accept the same user-rank dim the record call used.
    launch.set_dim([N])
    launch.launch()


# ============================================================================
# Unit tests for launch-dim preparation helpers
# ============================================================================


def _stub_kernel(key: str = "foo", *, kernel_dim: int = 1, max_tid_dim: int | None = None):
    """Produce a minimal object matching the attributes _prepare_launch_dim reads.

    ``max_tid_dim`` defaults to ``kernel_dim`` (the common case where the kernel
    has at least one wp.tid() call). Pass ``max_tid_dim=0`` to simulate a kernel
    with no wp.tid() calls at all.
    """
    if max_tid_dim is None:
        max_tid_dim = kernel_dim
    adj = SimpleNamespace(kernel_dim=kernel_dim, max_tid_dimensionality=max_tid_dim)
    return SimpleNamespace(key=key, adj=adj)


class TestTidUnpack(unittest.TestCase):
    def test_scalar(self):
        self.assertEqual(_tid_unpack(1), "i = wp.tid()")

    def test_two(self):
        self.assertEqual(_tid_unpack(2), "i, j = wp.tid()")

    def test_three(self):
        self.assertEqual(_tid_unpack(3), "i, j, k = wp.tid()")

    def test_four(self):
        self.assertEqual(_tid_unpack(4), "i, j, k, l = wp.tid()")

    def test_beyond_supported(self):
        # degrades gracefully when n > len(_TID_NAMES)
        self.assertEqual(_tid_unpack(5), "... = wp.tid()  # 5 variables")

    def test_zero(self):
        # guards against pathological input; real callers always pass n >= 1
        self.assertEqual(_tid_unpack(0), "... = wp.tid()  # 0 variables")


class TestBuildRankError(unittest.TestCase):
    def test_over_rank_kernel_dim_1_lists_all_four_options(self):
        msg = _build_rank_error((3, 3), kernel_dim=1, kernel=_stub_kernel("foo"), tiled=False)
        self.assertIn("Launch dim (3, 3) has rank 2", msg)
        self.assertIn("kernel 'foo'", msg)
        self.assertIn("kernel_dim=1", msg)
        self.assertIn("flat linear index over 9 threads: launch with dim=9", msg)
        self.assertIn("first-dim index (0..2, no repetition): launch with dim=3", msg)
        self.assertIn("first-dim index with repetition: unpack as `i, _ = wp.tid()`", msg)
        self.assertIn("per-dim indexing: unpack as `i, j = wp.tid()`", msg)

    def test_over_rank_kernel_dim_2_omits_launch_only_options(self):
        # With kernel_dim >= 2 the launch-only options would require a second
        # kernel edit to land — don't offer them.
        msg = _build_rank_error((2, 3, 4), kernel_dim=2, kernel=_stub_kernel("baz"), tiled=False)
        self.assertIn("kernel_dim=2", msg)
        self.assertNotIn("flat linear", msg)
        self.assertNotIn("launch with dim=24", msg)
        self.assertNotIn("no repetition", msg)
        self.assertIn("first-dim index with repetition: unpack as `i, _, _ = wp.tid()`", msg)
        self.assertIn("per-dim indexing: unpack as `i, j, k = wp.tid()`", msg)

    def test_under_rank_lists_two_options(self):
        msg = _build_rank_error((10,), kernel_dim=2, kernel=_stub_kernel("bar"), tiled=False)
        self.assertIn("Launch dim (10,) has rank 1", msg)
        self.assertIn("kernel_dim=2", msg)
        self.assertIn("keep 2-D kernel, launch with matching rank: dim=(10, 1)", msg)
        self.assertIn("change kernel to unpack 1 variable: `i = wp.tid()`", msg)

    def test_under_rank_tiled_lists_kernel_dim_minus_one_option(self):
        # For tiled launches, dim rank may also match kernel_dim - 1; offer a
        # concrete option alongside the rank-matching one.
        msg = _build_rank_error((2, 3), kernel_dim=4, kernel=_stub_kernel("baz"), tiled=True)
        self.assertIn("kernel_dim=4", msg)
        # rank-kernel_dim option
        self.assertIn("keep 4-D kernel, launch with matching rank: dim=(2, 3, 1, 1)", msg)
        # rank-(kernel_dim - 1) option — the new one
        self.assertIn("keep 4-D kernel with implicit block_dim axis, launch with dim=(2, 3, 1)", msg)

    def test_under_rank_non_tiled_omits_kernel_dim_minus_one_option(self):
        # Non-tiled launches have no kernel_dim - 1 escape hatch.
        msg = _build_rank_error((2, 3), kernel_dim=4, kernel=_stub_kernel("baz"), tiled=False)
        self.assertNotIn("implicit block_dim axis", msg)

    def test_tiled_flag_appends_hint(self):
        msg = _build_rank_error((3, 3), kernel_dim=1, kernel=_stub_kernel("foo"), tiled=True)
        self.assertIn("For launch_tiled, dim may also match kernel_dim - 1", msg)

    def test_non_tiled_omits_hint(self):
        msg = _build_rank_error((3, 3), kernel_dim=1, kernel=_stub_kernel("foo"), tiled=False)
        self.assertNotIn("launch_tiled", msg)


class TestPrepareLaunchDim(unittest.TestCase):
    def test_matching_rank_returns_canonicalized(self):
        kernel = _stub_kernel(kernel_dim=2)
        self.assertEqual(_prepare_launch_dim((3, 4), kernel), (3, 4))

    def test_scalar_int_canonicalized_for_1d(self):
        kernel = _stub_kernel(kernel_dim=1)
        self.assertEqual(_prepare_launch_dim(10, kernel), (10,))

    def test_list_canonicalized(self):
        kernel = _stub_kernel(kernel_dim=2)
        self.assertEqual(_prepare_launch_dim([3, 4], kernel), (3, 4))

    def test_zero_tid_kernel_flattens_multidim(self):
        # max_tid_dimensionality=0 → any dim is accepted, flattened to total count
        kernel = _stub_kernel(kernel_dim=1, max_tid_dim=0)
        self.assertEqual(_prepare_launch_dim((3, 4, 5), kernel), (60,))

    def test_zero_tid_kernel_preserves_1d(self):
        kernel = _stub_kernel(kernel_dim=1, max_tid_dim=0)
        self.assertEqual(_prepare_launch_dim(100, kernel), (100,))

    def test_over_rank_raises_value_error(self):
        kernel = _stub_kernel(kernel_dim=1)
        with self.assertRaises(ValueError) as cm:
            _prepare_launch_dim((3, 3), kernel)
        self.assertIn("kernel_dim=1", str(cm.exception))

    def test_under_rank_raises_value_error(self):
        kernel = _stub_kernel(kernel_dim=2)
        with self.assertRaises(ValueError) as cm:
            _prepare_launch_dim(10, kernel)
        self.assertIn("kernel_dim=2", str(cm.exception))

    def test_tiled_accepts_rank_minus_one(self):
        # tiled=True allows len(dim) == kernel_dim - 1 (block_dim axis is implicit)
        kernel = _stub_kernel(kernel_dim=2)
        self.assertEqual(_prepare_launch_dim((5,), kernel, tiled=True), (5,))

    def test_tiled_still_rejects_greater_mismatch(self):
        kernel = _stub_kernel(kernel_dim=2)
        with self.assertRaises(ValueError):
            _prepare_launch_dim((3, 3, 3), kernel, tiled=True)

    def test_non_tiled_rejects_rank_minus_one(self):
        kernel = _stub_kernel(kernel_dim=2)
        with self.assertRaises(ValueError):
            _prepare_launch_dim((5,), kernel, tiled=False)


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
add_function_test(
    TestTemplateLaunchBounds,
    "test_launch_dim_over_rank_kernel_dim_1_error",
    test_launch_dim_over_rank_kernel_dim_1_error,
    devices=devices,
)
add_function_test(
    TestTemplateLaunchBounds,
    "test_launch_dim_over_rank_kernel_dim_2_error",
    test_launch_dim_over_rank_kernel_dim_2_error,
    devices=devices,
)
add_function_test(
    TestTemplateLaunchBounds,
    "test_launch_dim_under_rank_error",
    test_launch_dim_under_rank_error,
    devices=devices,
)
add_function_test(TestTemplateLaunchBounds, "test_no_tid_kernel_multidim", test_no_tid_kernel_multidim, devices=devices)
add_function_test(TestTemplateLaunchBounds, "test_tiled_1d", test_tiled_1d, devices=devices)
add_function_test(TestTemplateLaunchBounds, "test_tiled_2d", test_tiled_2d, devices=devices)
add_function_test(TestTemplateLaunchBounds, "test_tiled_no_tid", test_tiled_no_tid, devices=devices)
add_function_test(TestTemplateLaunchBounds, "test_manual_tiled", test_manual_tiled, devices=devices)
add_function_test(
    TestTemplateLaunchBounds, "test_tiled_dim_mismatch_error", test_tiled_dim_mismatch_error, devices=devices
)
add_function_test(TestTemplateLaunchBounds, "test_set_dim_matching_rank", test_set_dim_matching_rank, devices=devices)
add_function_test(
    TestTemplateLaunchBounds,
    "test_set_dim_rank_mismatch_error",
    test_set_dim_rank_mismatch_error,
    devices=devices,
)
add_function_test(
    TestTemplateLaunchBounds,
    "test_set_dim_preserves_tiled_flag",
    test_set_dim_preserves_tiled_flag,
    devices=devices,
)
add_function_test(
    TestTemplateLaunchBounds,
    "test_set_dim_tiled_explicit_block_axis",
    test_set_dim_tiled_explicit_block_axis,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
