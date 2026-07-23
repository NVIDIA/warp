# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

TILE_DIM = 64
TILE_M = 16
TILE_N = 32
TILE_O = 8


@wp.kernel
def test_tile_view_kernel(src: wp.array2d[float], dst: wp.array2d[float]):
    # load whole source into local memory
    a = wp.tile_load(src, shape=(TILE_M, TILE_N))

    # copy the source array row by row
    for i in range(TILE_M):
        # create a view on original array and store
        row = a[i]
        wp.tile_store(dst[i], row)


def test_tile_view(test, device):
    rng = np.random.default_rng(42)

    a = wp.array(rng.random((TILE_M, TILE_N), dtype=np.float32), requires_grad=True, device=device)
    b = wp.array(np.zeros((TILE_M, TILE_N), dtype=np.float32), requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(test_tile_view_kernel, dim=[1], inputs=[a, b], block_dim=32, device=device)

    assert_np_equal(b.numpy(), a.numpy())
    b.grad = wp.ones_like(b, device=device)
    tape.backward()

    assert_np_equal(a.grad.numpy(), np.ones_like(a.numpy()))


@wp.kernel
def test_tile_assign_1d_kernel(src: wp.array2d[float], dst: wp.array2d[float]):
    # load whole source into local memory
    a = wp.tile_load(src, shape=(TILE_M, TILE_N))
    b = wp.tile_zeros(dtype=float, shape=(TILE_M, TILE_N))

    # copy the source array row by row
    for i in range(int(TILE_M)):
        # create views onto source and dest rows
        row_src = a[i]
        row_dst = b[i]

        # copy onto dest row
        wp.tile_assign(row_dst, row_src)

    wp.tile_store(dst, b)


def test_tile_assign_1d(test, device):
    rng = np.random.default_rng(42)

    a = wp.array(rng.random((TILE_M, TILE_N), dtype=np.float32), requires_grad=True, device=device)
    b = wp.array(np.zeros((TILE_M, TILE_N), dtype=np.float32), requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(test_tile_assign_1d_kernel, dim=[1], inputs=[a, b], block_dim=32, device=device)

    assert_np_equal(b.numpy(), a.numpy())
    b.grad = wp.ones_like(b, device=device)
    tape.backward()

    assert_np_equal(a.grad.numpy(), np.ones_like(a.numpy()))


@wp.kernel
def test_tile_assign_2d_kernel(src: wp.array3d[float], dst: wp.array3d[float]):
    # load whole source into local memory
    a = wp.tile_load(src, shape=(TILE_M, TILE_N, TILE_O))
    b = wp.tile_zeros(dtype=float, shape=(TILE_M, TILE_N, TILE_O))

    # copy the source array slice by slice
    for i in range(TILE_M):
        # create views onto source and dest slice
        row_src = a[i]
        row_dst = b[i]

        # copy onto dest slice
        wp.tile_assign(row_dst, row_src)

    wp.tile_store(dst, b)


def test_tile_assign_2d(test, device):
    rng = np.random.default_rng(42)

    a = wp.array(rng.random((TILE_M, TILE_N, TILE_O), dtype=np.float32), requires_grad=True, device=device)
    b = wp.array(np.zeros((TILE_M, TILE_N, TILE_O), dtype=np.float32), requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(test_tile_assign_2d_kernel, dim=[1], inputs=[a, b], block_dim=32, device=device)

    assert_np_equal(b.numpy(), a.numpy())
    b.grad = wp.ones_like(b, device=device)
    tape.backward()

    assert_np_equal(a.grad.numpy(), np.ones_like(a.numpy()))


@wp.kernel
def test_tile_view_offset_kernel(src: wp.array2d[float], dst: wp.array2d[float]):
    # load whole source into local memory
    a = wp.tile_load(src, shape=(TILE_M, TILE_N))
    b = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=float)

    # copy the source array slice by slice
    for i in range(TILE_M // 4):
        # create views onto source and dest slice 4 rows at a time
        v = wp.tile_view(a, offset=(i * 4, 0), shape=(4, TILE_N))

        # copy onto dest slice
        wp.tile_assign(b, v, offset=(i * 4, 0))

    wp.tile_store(dst, b)


@wp.func
def affine_op(x: float):
    return x * 2.0 + 1.0


@wp.kernel
def test_tile_view_map_assign_column_kernel(src: wp.array2d[float], dst: wp.array2d[float], col_idx: int):
    # Explicitly use shared storage for the source tile.
    a = wp.tile_load(src, shape=(TILE_M, TILE_N), storage="shared")

    # Take one column as a 2D view (M x 1), map (register tile), then write back in place.
    col_view = wp.tile_view(a, offset=(0, col_idx), shape=(TILE_M, 1))
    tmp = wp.tile_map(affine_op, col_view)
    wp.tile_assign(a, tmp, offset=(0, col_idx))

    wp.tile_store(dst, a)


def test_tile_view_map_assign_column(test, device):
    rng = np.random.default_rng(42)

    col_idx = 7
    src_np = rng.random((TILE_M, TILE_N), dtype=np.float32)
    expected = src_np.copy()
    expected[:, col_idx] = expected[:, col_idx] * 2.0 + 1.0

    src = wp.array(src_np, dtype=float, device=device)
    dst = wp.zeros((TILE_M, TILE_N), dtype=float, device=device)

    wp.launch_tiled(
        test_tile_view_map_assign_column_kernel, dim=[1], inputs=[src, dst, col_idx], block_dim=32, device=device
    )

    assert_np_equal(dst.numpy(), expected, tol=1e-6)


def test_tile_view_map_assign_column_backward(test, device):
    rng = np.random.default_rng(42)

    col_idx = 7
    src_np = rng.random((TILE_M, TILE_N), dtype=np.float32)
    src = wp.array(src_np, dtype=float, requires_grad=True, device=device)
    dst = wp.zeros((TILE_M, TILE_N), dtype=float, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            test_tile_view_map_assign_column_kernel, dim=[1], inputs=[src, dst, col_idx], block_dim=32, device=device
        )

    dst.grad = wp.ones_like(dst, device=device)
    tape.backward()

    expected_grad = np.ones_like(src_np)
    expected_grad[:, col_idx] = 2.0
    assert_np_equal(src.grad.numpy(), expected_grad, tol=1e-6)


def test_tile_view_offset(test, device):
    rng = np.random.default_rng(42)

    a = wp.array(rng.random((TILE_M, TILE_N), dtype=np.float32), requires_grad=True, device=device)
    b = wp.array(np.zeros((TILE_M, TILE_N), dtype=np.float32), requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(test_tile_view_offset_kernel, dim=[1], inputs=[a, b], block_dim=32, device=device)

    assert_np_equal(b.numpy(), a.numpy())
    b.grad = wp.ones_like(b, device=device)
    tape.backward()

    assert_np_equal(a.grad.numpy(), np.ones_like(a.numpy()))


@wp.kernel
def test_tile_view_shared_assign_column_kernel(src: wp.array2d[float], dst: wp.array2d[float], col_idx: int):
    a = wp.tile_load(src, shape=(TILE_M, TILE_N), storage="shared")

    # Extract a column view from src, scale it, and assign back via shared-to-shared tile_assign.
    col_src = wp.tile_view(a, offset=(0, col_idx), shape=(TILE_M, 1))
    col_scaled = wp.tile_zeros(shape=(TILE_M, 1), dtype=float, storage="shared")
    wp.tile_assign(col_scaled, col_src)
    col_scaled_view = wp.tile_view(col_scaled, offset=(0, 0), shape=(TILE_M, 1))
    wp.tile_assign(a, col_scaled_view, offset=(0, col_idx))

    wp.tile_store(dst, a)


def test_tile_view_shared_assign_column_backward(test, device):
    """Verify adj_tile_assign zeroing for the shared-to-shared path.

    The kernel copies column col_idx through an intermediate shared tile
    (identity transform), so the gradient for that column should be 1.0,
    not 2.0 (which would indicate double-counting without zeroing).
    """
    rng = np.random.default_rng(42)

    col_idx = 5
    src_np = rng.random((TILE_M, TILE_N), dtype=np.float32)
    src = wp.array(src_np, dtype=float, requires_grad=True, device=device)
    dst = wp.zeros((TILE_M, TILE_N), dtype=float, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(
            test_tile_view_shared_assign_column_kernel,
            dim=[1],
            inputs=[src, dst, col_idx],
            block_dim=32,
            device=device,
        )

    # Forward: dst should equal src (identity copy through intermediate tile).
    assert_np_equal(dst.numpy(), src_np, tol=1e-6)

    dst.grad = wp.ones_like(dst, device=device)
    tape.backward()

    # All gradients should be 1.0. Without the zeroing fix the overwritten
    # column would accumulate an extra gradient contribution, yielding 2.0.
    expected_grad = np.ones_like(src_np)
    assert_np_equal(src.grad.numpy(), expected_grad, tol=1e-6)


@wp.kernel
def tile_view_non_dense_store_kernel(src: wp.array2d[float], dst: wp.array2d[float]):
    """Store a non-dense view to global memory.

    Parent tile is (TILE_M, TILE_N) with shared strides (TILE_N, 1).
    View is (TILE_M, TILE_O) with the parent's strides (TILE_N, 1),
    so stride[0]=TILE_N != shape[1]=TILE_O — non-dense in shared memory.

    Before the Dense guard fix, the vectorized path would read shared
    memory linearly (ignoring the stride gap), producing wrong output.
    """
    a = wp.tile_load(src, shape=(TILE_M, TILE_N), storage="shared")
    v = wp.tile_view(a, offset=(0, 0), shape=(TILE_M, TILE_O))
    wp.tile_store(dst, v)


def test_tile_view_non_dense_store(test, device):
    """Regression test: tile_store of a non-dense shared view must use
    the scalar path and produce correct results."""
    rng = np.random.default_rng(42)
    src_np = rng.random((TILE_M, TILE_N), dtype=np.float32)
    src = wp.array(src_np, dtype=float, device=device)
    dst = wp.zeros((TILE_M, TILE_O), dtype=float, device=device)

    wp.launch_tiled(
        tile_view_non_dense_store_kernel,
        dim=[1],
        inputs=[src, dst],
        block_dim=32,
        device=device,
    )

    # The view selects the first TILE_O columns of each row.
    expected = src_np[:, :TILE_O]
    assert_np_equal(dst.numpy(), expected)


# ---------------------------------------------------------------------------
# NumPy-style slicing
# ---------------------------------------------------------------------------


@wp.kernel
def tile_slice_rows_kernel(src: wp.array2d[float], dst: wp.array2d[float]):
    t = wp.tile_load(src, shape=(TILE_M, TILE_N))
    wp.tile_store(dst, t[4:12, :])


@wp.kernel
def tile_slice_cols_kernel(src: wp.array2d[float], dst: wp.array2d[float]):
    t = wp.tile_load(src, shape=(TILE_M, TILE_N))
    wp.tile_store(dst, t[:, 8:24])


@wp.kernel
def tile_slice_strided_kernel(src: wp.array2d[float], dst: wp.array2d[float]):
    t = wp.tile_load(src, shape=(TILE_M, TILE_N))
    wp.tile_store(dst, t[::2, ::2])


@wp.kernel
def tile_slice_reverse_kernel(src: wp.array2d[float], dst: wp.array2d[float]):
    t = wp.tile_load(src, shape=(TILE_M, TILE_N))
    wp.tile_store(dst, t[::-1, :])


@wp.kernel
def tile_slice_row_collapse_kernel(src: wp.array2d[float], dst: wp.array1d[float]):
    t = wp.tile_load(src, shape=(TILE_M, TILE_N))
    wp.tile_store(dst, t[5, :])


@wp.kernel
def tile_slice_assign_kernel(src: wp.array2d[float], dst: wp.array2d[float]):
    t = wp.tile_load(src, shape=(TILE_M, TILE_N))
    ones = wp.tile_ones(shape=(4, TILE_N), dtype=float)
    t[0:4, :] = ones
    wp.tile_store(dst, t)


@wp.kernel
def tile_advanced_index_kernel(src: wp.array2d[float], dst: wp.array2d[float]):
    t = wp.tile_load(src, shape=(TILE_M, TILE_N))
    idx = wp.tile_arange(0, 8, dtype=int) * 2  # [0, 2, 4, ..., 14]
    wp.tile_store(dst, t[idx, :])


@wp.kernel
def tile_advanced_index_dup_kernel(src: wp.array2d[float], idx: wp.array1d[int], dst: wp.array2d[float]):
    t = wp.tile_load(src, shape=(TILE_M, TILE_N))
    i = wp.tile_load(idx, shape=8)
    wp.tile_store(dst, t[i, :])


@wp.kernel
def tile_slice_assign_grad_kernel(src: wp.array2d[float], val: wp.array2d[float], dst: wp.array2d[float]):
    t = wp.tile_load(src, shape=(TILE_M, TILE_N))
    v = wp.tile_load(val, shape=(4, TILE_N))
    t[0:4, :] = v
    wp.tile_store(dst, t)


@wp.kernel
def tile_slice_neg_index_kernel(src: wp.array2d[float], dst: wp.array1d[float]):
    t = wp.tile_load(src, shape=(TILE_M, TILE_N))
    wp.tile_store(dst, t[-1, :])  # NumPy negative index: last row


@wp.kernel
def tile_slice_neg_step_oob_kernel(src: wp.array2d[float], dst: wp.array2d[float]):
    t = wp.tile_load(src, shape=(TILE_M, TILE_N))
    wp.tile_store(dst, t[100::-1, :])  # out-of-range start clamps to the full reverse


@wp.kernel
def tile_slice_neg_stop_kernel(src: wp.array2d[float], dst: wp.array2d[float]):
    t = wp.tile_load(src, shape=(TILE_M, TILE_N))
    wp.tile_store(dst, t[7:-100:-1, :])  # out-of-range negative stop must still include row 0


def _check_slice(test, device, kernel, np_index, out_shape, check_grad=True):
    """Check a tile-slicing kernel against the equivalent NumPy indexing expression.

    Args:
        test: Test case instance.
        device: Device to launch on.
        kernel: Kernel that slices a ``(TILE_M, TILE_N)`` tile with ``np_index``.
        np_index: NumPy index expression matching the kernel's subscript.
        out_shape: Shape of the sliced result.
        check_grad: Whether to also verify the backward pass scatters gradients
            back to the sliced source elements.
    """
    rng = np.random.default_rng(42)
    src_np = rng.random((TILE_M, TILE_N), dtype=np.float32)

    src = wp.array(src_np, dtype=float, requires_grad=True, device=device)
    dst = wp.zeros(out_shape, dtype=float, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(kernel, dim=[1], inputs=[src, dst], block_dim=32, device=device)

    assert_np_equal(dst.numpy(), src_np[np_index], tol=1e-6)

    if not check_grad:
        return

    adj_np = rng.random(out_shape, dtype=np.float32)
    dst.grad = wp.array(adj_np, dtype=float, device=device)
    tape.backward()

    expected_grad = np.zeros_like(src_np)
    expected_grad[np_index] = adj_np
    assert_np_equal(src.grad.numpy(), expected_grad, tol=1e-6)


def test_tile_slice_rows(test, device):
    _check_slice(test, device, tile_slice_rows_kernel, np.s_[4:12, :], (8, TILE_N))


def test_tile_slice_cols(test, device):
    _check_slice(test, device, tile_slice_cols_kernel, np.s_[:, 8:24], (TILE_M, 16))


def test_tile_slice_strided(test, device):
    _check_slice(test, device, tile_slice_strided_kernel, np.s_[::2, ::2], (TILE_M // 2, TILE_N // 2))


def test_tile_slice_reverse(test, device):
    _check_slice(test, device, tile_slice_reverse_kernel, np.s_[::-1, :], (TILE_M, TILE_N))


def test_tile_slice_row_collapse(test, device):
    _check_slice(test, device, tile_slice_row_collapse_kernel, np.s_[5, :], (TILE_N,))


def test_tile_slice_assign(test, device):
    rng = np.random.default_rng(42)
    src_np = rng.random((TILE_M, TILE_N), dtype=np.float32)

    src = wp.array(src_np, dtype=float, device=device)
    dst = wp.zeros((TILE_M, TILE_N), dtype=float, device=device)

    wp.launch_tiled(tile_slice_assign_kernel, dim=[1], inputs=[src, dst], block_dim=32, device=device)

    expected = src_np.copy()
    expected[0:4, :] = 1.0
    assert_np_equal(dst.numpy(), expected, tol=1e-6)


def test_tile_advanced_index(test, device):
    idx = np.arange(0, 8) * 2
    _check_slice(test, device, tile_advanced_index_kernel, np.s_[idx, :], (8, TILE_N))


def test_tile_advanced_index_duplicate_grad(test, device):
    # Duplicate indices must accumulate their gradients (atomic scatter). The
    # generic _check_slice reference assigns rather than accumulates, so this
    # case needs an np.add.at reference of its own.
    rng = np.random.default_rng(42)
    src_np = rng.random((TILE_M, TILE_N), dtype=np.float32)
    idx_np = np.array([0, 0, 3, 3, 5, 5, 7, 7], dtype=np.int32)

    src = wp.array(src_np, dtype=float, requires_grad=True, device=device)
    idx = wp.array(idx_np, dtype=int, device=device)
    dst = wp.zeros((8, TILE_N), dtype=float, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(tile_advanced_index_dup_kernel, dim=[1], inputs=[src, idx, dst], block_dim=32, device=device)

    assert_np_equal(dst.numpy(), src_np[idx_np], tol=1e-6)

    adj_np = rng.random((8, TILE_N), dtype=np.float32)
    dst.grad = wp.array(adj_np, dtype=float, device=device)
    tape.backward()

    expected_grad = np.zeros_like(src_np)
    np.add.at(expected_grad, idx_np, adj_np)  # accumulate contributions from repeated rows
    assert_np_equal(src.grad.numpy(), expected_grad, tol=1e-6)


def test_tile_slice_assign_grad(test, device):
    # Slice assignment must route gradients correctly: the overwritten region flows
    # to the assigned source, and the untouched region passes through to the base.
    rng = np.random.default_rng(42)
    src_np = rng.random((TILE_M, TILE_N), dtype=np.float32)
    val_np = rng.random((4, TILE_N), dtype=np.float32)

    src = wp.array(src_np, dtype=float, requires_grad=True, device=device)
    val = wp.array(val_np, dtype=float, requires_grad=True, device=device)
    dst = wp.zeros((TILE_M, TILE_N), dtype=float, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(tile_slice_assign_grad_kernel, dim=[1], inputs=[src, val, dst], block_dim=32, device=device)

    expected = src_np.copy()
    expected[0:4, :] = val_np
    assert_np_equal(dst.numpy(), expected, tol=1e-6)

    adj_np = rng.random((TILE_M, TILE_N), dtype=np.float32)
    dst.grad = wp.array(adj_np, dtype=float, device=device)
    tape.backward()

    # rows 0:4 are overwritten by val, so they contribute no gradient to src
    expected_src_grad = adj_np.copy()
    expected_src_grad[0:4, :] = 0.0
    assert_np_equal(src.grad.numpy(), expected_src_grad, tol=1e-6)
    assert_np_equal(val.grad.numpy(), adj_np[0:4, :], tol=1e-6)


def test_tile_slice_neg_index(test, device):
    # A negative integer index must wrap like NumPy (last row), not read out of bounds.
    _check_slice(test, device, tile_slice_neg_index_kernel, np.s_[-1, :], (TILE_N,))


def test_tile_slice_neg_step_oob_start(test, device):
    # An out-of-range start with a negative step clamps to length-1 (full reverse).
    _check_slice(test, device, tile_slice_neg_step_oob_kernel, np.s_[100::-1, :], (TILE_M, TILE_N))


def test_tile_slice_neg_stop(test, device):
    # An out-of-range negative stop with a negative step must still reach row 0.
    _check_slice(test, device, tile_slice_neg_stop_kernel, np.s_[7:-100:-1, :], (8, TILE_N))


def test_tile_advanced_index_float_rejected(test, device):
    @wp.kernel(module="unique", enable_backward=False)
    def kernel_fn():
        t = wp.tile_ones(shape=(TILE_M, TILE_N), dtype=float)
        idx = wp.tile_arange(0.0, 8.0, 1.0, dtype=float)  # non-integer index tile
        rows = t[idx, :]

    with test.assertRaisesRegex((RuntimeError, ValueError), r"integer index tile"):
        wp.launch_tiled(kernel_fn, dim=[1], inputs=[], block_dim=32, device=device)


def test_tile_slice_scalar_assign_rejected(test, device):
    @wp.kernel(module="unique", enable_backward=False)
    def kernel_fn():
        t = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=float)
        t[0:4, :] = 1.0  # scalar broadcast into a slice is not supported

    with test.assertRaisesRegex((RuntimeError, ValueError), r"requires a tile of matching shape"):
        wp.launch_tiled(kernel_fn, dim=[1], inputs=[], block_dim=32, device=device)


def test_tile_slice_augassign_rejected(test, device):
    @wp.kernel(module="unique", enable_backward=False)
    def kernel_fn():
        t = wp.tile_ones(shape=(TILE_M, TILE_N), dtype=float)
        v = wp.tile_ones(shape=(4, TILE_N), dtype=float)
        t[0:4, :] += v  # compound assignment to a slice is not supported

    with test.assertRaisesRegex((RuntimeError, ValueError), r"Compound assignment to a tile view"):
        wp.launch_tiled(kernel_fn, dim=[1], inputs=[], block_dim=32, device=device)


def test_tile_row_augassign_rejected(test, device):
    # A partial integer index (a row view) has no in-place path either.
    @wp.kernel(module="unique", enable_backward=False)
    def kernel_fn():
        t = wp.tile_ones(shape=(TILE_M, TILE_N), dtype=float)
        v = wp.tile_ones(shape=(TILE_N,), dtype=float)
        t[0] += v  # compound assignment to a row view is not supported

    with test.assertRaisesRegex((RuntimeError, ValueError), r"Compound assignment to a tile view"):
        wp.launch_tiled(kernel_fn, dim=[1], inputs=[], block_dim=32, device=device)


def test_tile_slice_assign_shape_mismatch(test, device):
    @wp.kernel(module="unique", enable_backward=False)
    def kernel_fn():
        t = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=float)
        src = wp.tile_ones(shape=(5, TILE_N), dtype=float)  # 5 rows into a 4-row slice
        t[0:4, :] = src

    with test.assertRaisesRegex((RuntimeError, ValueError), r"shape mismatch|does not fit"):
        wp.launch_tiled(kernel_fn, dim=[1], inputs=[], block_dim=32, device=device)


def test_tile_index_out_of_bounds(test, device):
    @wp.kernel(module="unique", enable_backward=False)
    def kernel_fn():
        t = wp.tile_ones(shape=(TILE_M, TILE_N), dtype=float)
        row = t[100, :]  # constant out-of-range integer index

    with test.assertRaisesRegex((RuntimeError, ValueError), r"out of bounds"):
        wp.launch_tiled(kernel_fn, dim=[1], inputs=[], block_dim=32, device=device)


def test_tile_slice_empty_rejected(test, device):
    @wp.kernel(module="unique", enable_backward=False)
    def kernel_fn():
        t = wp.tile_ones(shape=(TILE_M, TILE_N), dtype=float)
        empty = t[4:4, :]  # zero-length axis

    with test.assertRaisesRegex((RuntimeError, ValueError), r"empty tile|zero-length"):
        wp.launch_tiled(kernel_fn, dim=[1], inputs=[], block_dim=32, device=device)


@wp.kernel(enable_backward=False)
def tile_slice_assign_overlap_kernel(src: wp.array1d[wp.int32], dst: wp.array1d[wp.int32]):
    t = wp.tile_load(src, shape=(TILE_DIM,))
    t[1:] = t[:-1]  # source and destination views alias the same tile
    wp.tile_store(dst, t)


def test_tile_slice_assign_overlap(test, device):
    # Overlapping slice assignment must read the pre-assignment source values
    # (NumPy semantics): the copy is staged through registers, not interleaved.
    src_np = np.arange(TILE_DIM, dtype=np.int32)
    src = wp.array(src_np, device=device)
    dst = wp.zeros(TILE_DIM, dtype=wp.int32, device=device)

    wp.launch_tiled(tile_slice_assign_overlap_kernel, dim=[1], inputs=[src, dst], block_dim=32, device=device)

    expected = src_np.copy()
    expected[1:] = src_np[:-1]
    assert_np_equal(dst.numpy(), expected)


def test_tile_slice_float_index_rejected(test, device):
    @wp.kernel(module="unique", enable_backward=False)
    def kernel_fn():
        t = wp.tile_ones(shape=(TILE_M, TILE_N), dtype=float)
        row = t[1.5, :]  # non-integer scalar index

    with test.assertRaisesRegex((RuntimeError, TypeError), r"must be integers or slices"):
        wp.launch_tiled(kernel_fn, dim=[1], inputs=[], block_dim=32, device=device)


def test_tile_view_float_offset_rejected(test, device):
    # The explicit tile_view() API must reject non-integer offsets as well.
    @wp.kernel(module="unique", enable_backward=False)
    def kernel_fn():
        t = wp.tile_ones(shape=(TILE_M, TILE_N), dtype=float)
        row = wp.tile_view(t, offset=(1.5,))

    with test.assertRaisesRegex((RuntimeError, ValueError), r"must be integers or slices"):
        wp.launch_tiled(kernel_fn, dim=[1], inputs=[], block_dim=32, device=device)


def test_tile_reshape_strided_view_rejected(test, device):
    # Reshape aliases the source pointer with dense strides, so a strided or
    # reversed view would silently read the wrong elements (or out of bounds).
    @wp.kernel(module="unique", enable_backward=False)
    def kernel_fn():
        t = wp.tile_ones(shape=(TILE_M, TILE_N), dtype=float)
        flat = wp.tile_reshape(t[::2, ::2], shape=((TILE_M // 2) * (TILE_N // 2),))

    with test.assertRaisesRegex((RuntimeError, ValueError), r"requires a contiguous tile"):
        wp.launch_tiled(kernel_fn, dim=[1], inputs=[], block_dim=32, device=device)


def test_tile_view_shape_with_slice_rejected(test, device):
    # Passing an explicit shape alongside a slice-containing offset would
    # silently drop the shape, so it must be rejected.
    @wp.kernel(module="unique", enable_backward=False)
    def kernel_fn():
        t = wp.tile_ones(shape=(TILE_M, TILE_N), dtype=float)
        v = wp.tile_view(t, offset=(slice(0, 2, 1), slice(0, 3, 1)), shape=(2, 3))

    with test.assertRaisesRegex((RuntimeError, ValueError), r"'shape' together with a slice"):
        wp.launch_tiled(kernel_fn, dim=[1], inputs=[], block_dim=32, device=device)


# ---------------------------------------------------------------------------
# Runtime and negative index handling, chained subscripts, explicit slices
# ---------------------------------------------------------------------------


@wp.kernel
def tile_runtime_ref_partial_kernel(src: wp.array2d[float], indices: wp.array1d[int], dst: wp.array1d[float]):
    t = wp.tile_load(src, shape=(TILE_M, TILE_N))
    wp.tile_store(dst, t[indices[0], :])


@wp.kernel
def tile_runtime_ref_bare_kernel(src: wp.array2d[float], indices: wp.array1d[int], dst: wp.array1d[float]):
    t = wp.tile_load(src, shape=(TILE_M, TILE_N))
    wp.tile_store(dst, t[indices[0]])


@wp.kernel
def tile_runtime_ref_explicit_kernel(src: wp.array2d[float], indices: wp.array1d[int], dst: wp.array1d[float]):
    t = wp.tile_load(src, shape=(TILE_M, TILE_N))
    wp.tile_store(dst, wp.tile_view(t, offset=(indices[0],)))


def test_tile_runtime_ref_index(test, device):
    # A runtime integer index loaded from an array (a wp.ref[int32] during code
    # generation) must be accepted by the subscript, bare-index, and explicit
    # tile_view() routes; the referenced value is loaded at the call site.
    rng = np.random.default_rng(42)
    src_np = rng.random((TILE_M, TILE_N), dtype=np.float32)
    src = wp.array(src_np, dtype=float, device=device)
    indices = wp.array(np.array([5], dtype=np.int32), dtype=int, device=device)

    for kernel in (
        tile_runtime_ref_partial_kernel,
        tile_runtime_ref_bare_kernel,
        tile_runtime_ref_explicit_kernel,
    ):
        dst = wp.zeros(TILE_N, dtype=float, device=device)
        wp.launch_tiled(kernel, dim=[1], inputs=[src, indices, dst], block_dim=32, device=device)
        assert_np_equal(dst.numpy(), src_np[5], tol=1e-6)


@wp.kernel
def tile_runtime_neg_index_kernel(src: wp.array2d[float], offset: int, dst: wp.array1d[float]):
    t = wp.tile_load(src, shape=(TILE_M, TILE_N))
    i = offset  # runtime (non-constant) index
    wp.tile_store(dst, t[i, :])


def test_tile_runtime_neg_index(test, device):
    # A runtime (non-constant) negative index must wrap like NumPy, matching the
    # compile-time-constant t[-1]. -TILE_M wraps to row 0.
    rng = np.random.default_rng(42)
    src_np = rng.random((TILE_M, TILE_N), dtype=np.float32)
    src = wp.array(src_np, dtype=float, device=device)
    for offset in (-1, -TILE_M):
        dst = wp.zeros(TILE_N, dtype=float, device=device)
        wp.launch_tiled(tile_runtime_neg_index_kernel, dim=[1], inputs=[src, offset, dst], block_dim=32, device=device)
        assert_np_equal(dst.numpy(), src_np[offset], tol=1e-6)


@wp.kernel
def tile_gather_neg_index_kernel(src: wp.array2d[float], idx: wp.array1d[int], dst: wp.array2d[float]):
    t = wp.tile_load(src, shape=(TILE_M, TILE_N))
    i = wp.tile_load(idx, shape=4)
    wp.tile_store(dst, t[i, :])


def test_tile_gather_neg_index(test, device):
    # Negative values in an integer index tile must wrap against the gathered axis
    # (NumPy semantics) on both the forward gather and the backward scatter.
    rng = np.random.default_rng(42)
    src_np = rng.random((TILE_M, TILE_N), dtype=np.float32)
    idx_np = np.array([-1, 0, -TILE_M, 3], dtype=np.int32)  # wraps to [TILE_M-1, 0, 0, 3]

    src = wp.array(src_np, dtype=float, requires_grad=True, device=device)
    idx = wp.array(idx_np, dtype=int, device=device)
    dst = wp.zeros((4, TILE_N), dtype=float, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(tile_gather_neg_index_kernel, dim=[1], inputs=[src, idx, dst], block_dim=32, device=device)

    assert_np_equal(dst.numpy(), src_np[idx_np], tol=1e-6)

    adj_np = rng.random((4, TILE_N), dtype=np.float32)
    dst.grad = wp.array(adj_np, dtype=float, device=device)
    tape.backward()

    expected_grad = np.zeros_like(src_np)
    np.add.at(expected_grad, idx_np % TILE_M, adj_np)  # wrapped rows; duplicate wraps accumulate
    assert_np_equal(src.grad.numpy(), expected_grad, tol=1e-6)


@wp.kernel
def tile_chain_slice_kernel(src: wp.array1d[float], dst: wp.array1d[float]):
    t = wp.tile_load(src, shape=(TILE_DIM,))
    wp.tile_store(dst, t[::-1][::2])


@wp.kernel
def tile_chain_three_kernel(src: wp.array1d[float], dst: wp.array1d[float]):
    t = wp.tile_load(src, shape=(TILE_DIM,))
    wp.tile_store(dst, t[::-1][1:][::2])


@wp.kernel
def tile_chain_slice_gather_kernel(src: wp.array1d[float], idx: wp.array1d[int], dst: wp.array1d[float]):
    t = wp.tile_load(src, shape=(TILE_DIM,))
    i = wp.tile_load(idx, shape=4)
    wp.tile_store(dst, t[::2][i])


@wp.kernel
def tile_chain_gather_slice_kernel(src: wp.array1d[float], idx: wp.array1d[int], dst: wp.array1d[float]):
    t = wp.tile_load(src, shape=(TILE_DIM,))
    i = wp.tile_load(idx, shape=4)
    wp.tile_store(dst, t[i][::-1])


def test_tile_chained_subscript(test, device):
    # Chained tile subscripts (a[X][Y]...) apply each bracket to the previous
    # result: slice/slice, three-deep, slice/gather, and gather/slice.
    a = np.arange(TILE_DIM, dtype=np.float32)
    src = wp.array(a, dtype=float, requires_grad=True, device=device)

    def check(kernel, expected, extra_inputs=()):
        dst = wp.zeros(len(expected), dtype=float, device=device)
        wp.launch_tiled(kernel, dim=[1], inputs=[src, *extra_inputs, dst], block_dim=32, device=device)
        assert_np_equal(dst.numpy(), np.asarray(expected, dtype=np.float32), tol=1e-6)

    check(tile_chain_slice_kernel, a[::-1][::2])
    check(tile_chain_three_kernel, a[::-1][1:][::2])

    idx_np = np.array([0, 5, 5, 9], dtype=np.int32)
    idx = wp.array(idx_np, dtype=int, device=device)
    check(tile_chain_slice_gather_kernel, a[::2][idx_np], extra_inputs=(idx,))
    check(tile_chain_gather_slice_kernel, a[idx_np][::-1], extra_inputs=(idx,))

    # gradient through the slice/slice chain scatters back to the selected elements
    dst = wp.zeros(TILE_DIM // 2, dtype=float, requires_grad=True, device=device)
    with wp.Tape() as tape:
        wp.launch_tiled(tile_chain_slice_kernel, dim=[1], inputs=[src, dst], block_dim=32, device=device)
    adj_np = np.arange(1, TILE_DIM // 2 + 1, dtype=np.float32)
    dst.grad = wp.array(adj_np, dtype=float, device=device)
    tape.backward()
    expected_grad = np.zeros_like(a)
    expected_grad[::-1][::2] = adj_np
    assert_np_equal(src.grad.numpy(), expected_grad, tol=1e-6)


@wp.kernel
def tile_view_explicit_slice_kernel(src: wp.array1d[float], dst: wp.array1d[float]):
    t = wp.tile_load(src, shape=(TILE_DIM,))
    wp.tile_store(dst, wp.tile_view(t, offset=(slice(0, -1, 1),)))


@wp.kernel
def tile_view_explicit_slice_neg_start_kernel(src: wp.array1d[float], dst: wp.array1d[float]):
    t = wp.tile_load(src, shape=(TILE_DIM,))
    wp.tile_store(dst, wp.tile_view(t, offset=(slice(-3, TILE_DIM, 1),)))


def test_tile_view_explicit_slice(test, device):
    # An explicit slice offset must normalize its bounds against the parent shape,
    # matching the equivalent subscript syntax (t[0:-1], t[-3:]).
    a = np.arange(TILE_DIM, dtype=np.float32)
    src = wp.array(a, dtype=float, device=device)

    dst = wp.zeros(TILE_DIM - 1, dtype=float, device=device)
    wp.launch_tiled(tile_view_explicit_slice_kernel, dim=[1], inputs=[src, dst], block_dim=32, device=device)
    assert_np_equal(dst.numpy(), a[0:-1], tol=1e-6)

    dst2 = wp.zeros(3, dtype=float, device=device)
    wp.launch_tiled(tile_view_explicit_slice_neg_start_kernel, dim=[1], inputs=[src, dst2], block_dim=32, device=device)
    assert_np_equal(dst2.numpy(), a[-3:], tol=1e-6)


@wp.kernel
def tile_row_assign_kernel(src: wp.array2d[float], row_src: wp.array1d[float], dst: wp.array2d[float]):
    t = wp.tile_load(src, shape=(TILE_M, TILE_N))
    r = wp.tile_load(row_src, shape=(TILE_N,))
    t[0] = r  # partial integer index assignment (fewer indices than the tile rank)
    wp.tile_store(dst, t)


def test_tile_row_assign(test, device):
    # Partial integer index assignment (t[0] = row) must build a row view and assign
    # in place, matching t[0, :] = row, including gradient routing.
    rng = np.random.default_rng(42)
    src_np = rng.random((TILE_M, TILE_N), dtype=np.float32)
    row_np = rng.random(TILE_N, dtype=np.float32)

    src = wp.array(src_np, dtype=float, requires_grad=True, device=device)
    row = wp.array(row_np, dtype=float, requires_grad=True, device=device)
    dst = wp.zeros((TILE_M, TILE_N), dtype=float, requires_grad=True, device=device)

    with wp.Tape() as tape:
        wp.launch_tiled(tile_row_assign_kernel, dim=[1], inputs=[src, row, dst], block_dim=32, device=device)

    expected = src_np.copy()
    expected[0] = row_np
    assert_np_equal(dst.numpy(), expected, tol=1e-6)

    adj_np = rng.random((TILE_M, TILE_N), dtype=np.float32)
    dst.grad = wp.array(adj_np, dtype=float, device=device)
    tape.backward()

    expected_src_grad = adj_np.copy()
    expected_src_grad[0] = 0.0  # row 0 is overwritten, so it contributes no gradient to src
    assert_np_equal(src.grad.numpy(), expected_src_grad, tol=1e-6)
    assert_np_equal(row.grad.numpy(), adj_np[0], tol=1e-6)


@wp.kernel
def tile_chain_int_element_kernel(src: wp.array2d[float], dst: wp.array1d[float]):
    t = wp.tile_load(src, shape=(5, 7))  # register tile
    dst[0] = t[3][4]  # chained integer subscripts collapse to a single element


def test_tile_chain_int_element(test, device):
    # t[i][j] with all-integer brackets must resolve to the same element as t[i, j]
    # without going through the slice/view path (which would force shared storage).
    rng = np.random.default_rng(42)
    src_np = rng.random((5, 7), dtype=np.float32)
    src = wp.array(src_np, dtype=float, device=device)
    dst = wp.zeros(1, dtype=float, device=device)
    wp.launch_tiled(tile_chain_int_element_kernel, dim=[1], inputs=[src, dst], block_dim=32, device=device)
    assert_np_equal(dst.numpy(), src_np[3, 4:5], tol=1e-6)


def test_tile_view_offset_oob_rejected(test, device):
    # An explicit tile_view() offset is a raw coordinate, so an out-of-range constant
    # (negative or too large) must be rejected rather than silently read out of
    # bounds, matching the subscript path.
    @wp.kernel(module="unique", enable_backward=False)
    def neg_kernel():
        t = wp.tile_ones(shape=(TILE_M, TILE_N), dtype=float)
        row = wp.tile_view(t, offset=(-1,))  # negative constant offset

    with test.assertRaisesRegex((RuntimeError, ValueError), r"out of bounds"):
        wp.launch_tiled(neg_kernel, dim=[1], inputs=[], block_dim=32, device=device)

    @wp.kernel(module="unique", enable_backward=False)
    def pos_kernel():
        t = wp.tile_ones(shape=(TILE_M, TILE_N), dtype=float)
        row = wp.tile_view(t, offset=(TILE_M,))  # one past the last valid row

    with test.assertRaisesRegex((RuntimeError, ValueError), r"out of bounds"):
        wp.launch_tiled(pos_kernel, dim=[1], inputs=[], block_dim=32, device=device)


def test_tile_view_runtime_slice_bound_rejected(test, device):
    # An explicit slice offset with a runtime bound cannot infer the view shape at
    # code-gen time, so it must be rejected with a clear message (not crash).
    @wp.kernel(module="unique", enable_backward=False)
    def kernel_fn(n: int):
        t = wp.tile_ones(shape=(TILE_N,), dtype=float)
        v = wp.tile_view(t, offset=(slice(0, n, 1),))  # runtime stop bound

    with test.assertRaisesRegex((RuntimeError, ValueError), r"compile-time constant"):
        wp.launch_tiled(kernel_fn, dim=[1], inputs=[4], block_dim=32, device=device)


devices = get_test_devices()


class TestTileView(unittest.TestCase):
    pass


add_function_test(TestTileView, "test_tile_view", test_tile_view, devices=devices)
add_function_test(TestTileView, "test_tile_view_offset", test_tile_view_offset, devices=devices)
add_function_test(TestTileView, "test_tile_assign_1d", test_tile_assign_1d, devices=devices)
add_function_test(TestTileView, "test_tile_assign_2d", test_tile_assign_2d, devices=devices)
add_function_test(TestTileView, "test_tile_view_map_assign_column", test_tile_view_map_assign_column, devices=devices)
add_function_test(
    TestTileView,
    "test_tile_view_map_assign_column_backward",
    test_tile_view_map_assign_column_backward,
    devices=devices,
)
add_function_test(
    TestTileView,
    "test_tile_view_shared_assign_column_backward",
    test_tile_view_shared_assign_column_backward,
    devices=devices,
)
add_function_test(
    TestTileView,
    "test_tile_view_non_dense_store",
    test_tile_view_non_dense_store,
    devices=devices,
)
add_function_test(TestTileView, "test_tile_slice_rows", test_tile_slice_rows, devices=devices)
add_function_test(TestTileView, "test_tile_slice_cols", test_tile_slice_cols, devices=devices)
add_function_test(TestTileView, "test_tile_slice_strided", test_tile_slice_strided, devices=devices)
add_function_test(TestTileView, "test_tile_slice_reverse", test_tile_slice_reverse, devices=devices)
add_function_test(TestTileView, "test_tile_slice_row_collapse", test_tile_slice_row_collapse, devices=devices)
add_function_test(TestTileView, "test_tile_slice_assign", test_tile_slice_assign, devices=devices)
add_function_test(TestTileView, "test_tile_advanced_index", test_tile_advanced_index, devices=devices)
add_function_test(
    TestTileView, "test_tile_advanced_index_duplicate_grad", test_tile_advanced_index_duplicate_grad, devices=devices
)
add_function_test(TestTileView, "test_tile_slice_assign_grad", test_tile_slice_assign_grad, devices=devices)
add_function_test(TestTileView, "test_tile_slice_neg_index", test_tile_slice_neg_index, devices=devices)
add_function_test(
    TestTileView, "test_tile_slice_neg_step_oob_start", test_tile_slice_neg_step_oob_start, devices=devices
)
add_function_test(
    TestTileView, "test_tile_slice_assign_shape_mismatch", test_tile_slice_assign_shape_mismatch, devices=devices
)
add_function_test(TestTileView, "test_tile_index_out_of_bounds", test_tile_index_out_of_bounds, devices=devices)
add_function_test(TestTileView, "test_tile_slice_empty_rejected", test_tile_slice_empty_rejected, devices=devices)
add_function_test(TestTileView, "test_tile_slice_neg_stop", test_tile_slice_neg_stop, devices=devices)
add_function_test(
    TestTileView, "test_tile_advanced_index_float_rejected", test_tile_advanced_index_float_rejected, devices=devices
)
add_function_test(
    TestTileView, "test_tile_slice_scalar_assign_rejected", test_tile_slice_scalar_assign_rejected, devices=devices
)
add_function_test(
    TestTileView, "test_tile_slice_augassign_rejected", test_tile_slice_augassign_rejected, devices=devices
)
add_function_test(TestTileView, "test_tile_row_augassign_rejected", test_tile_row_augassign_rejected, devices=devices)
add_function_test(TestTileView, "test_tile_slice_assign_overlap", test_tile_slice_assign_overlap, devices=devices)
add_function_test(
    TestTileView, "test_tile_slice_float_index_rejected", test_tile_slice_float_index_rejected, devices=devices
)
add_function_test(
    TestTileView, "test_tile_view_float_offset_rejected", test_tile_view_float_offset_rejected, devices=devices
)
add_function_test(
    TestTileView, "test_tile_reshape_strided_view_rejected", test_tile_reshape_strided_view_rejected, devices=devices
)
add_function_test(
    TestTileView, "test_tile_view_shape_with_slice_rejected", test_tile_view_shape_with_slice_rejected, devices=devices
)
add_function_test(TestTileView, "test_tile_runtime_ref_index", test_tile_runtime_ref_index, devices=devices)
add_function_test(TestTileView, "test_tile_runtime_neg_index", test_tile_runtime_neg_index, devices=devices)
add_function_test(TestTileView, "test_tile_gather_neg_index", test_tile_gather_neg_index, devices=devices)
add_function_test(TestTileView, "test_tile_chained_subscript", test_tile_chained_subscript, devices=devices)
add_function_test(TestTileView, "test_tile_view_explicit_slice", test_tile_view_explicit_slice, devices=devices)
add_function_test(TestTileView, "test_tile_row_assign", test_tile_row_assign, devices=devices)
add_function_test(TestTileView, "test_tile_chain_int_element", test_tile_chain_int_element, devices=devices)
add_function_test(
    TestTileView, "test_tile_view_offset_oob_rejected", test_tile_view_offset_oob_rejected, devices=devices
)
add_function_test(
    TestTileView,
    "test_tile_view_runtime_slice_bound_rejected",
    test_tile_view_runtime_slice_bound_rejected,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
