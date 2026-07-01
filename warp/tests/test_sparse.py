# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import unittest

import numpy as np

import warp as wp
from warp.sparse import (
    BSR_STATUS_ROW_CAPACITY_EXCEEDED,
    BSR_STATUS_SUCCESS,
    BsrMatrix,
    bsr_assign,
    bsr_axpy,
    bsr_axpy_work_arrays,
    bsr_compress,
    bsr_copy,
    bsr_diag,
    bsr_from_triplets,
    bsr_get_diag,
    bsr_identity,
    bsr_mm,
    bsr_mm_work_arrays,
    bsr_mv,
    bsr_scale,
    bsr_set_from_triplets,
    bsr_set_transpose,
    bsr_set_zero,
    bsr_transposed,
    bsr_zeros,
)
from warp.tests.unittest_utils import *


def _get_block(mat, row, col, block_shape):
    return mat[row * block_shape[0] : (row + 1) * block_shape[0], col * block_shape[1] : (col + 1) * block_shape[1]]


def _triplets_to_dense(shape, rows, cols, values):
    mat = np.zeros(shape)

    rows = rows.numpy()
    cols = cols.numpy()
    values = values.numpy()

    block_shape = values.shape[1:] if values.ndim == 3 else (1, 1)

    for row, col, val in zip(rows, cols, values, strict=True):
        mat_block = _get_block(mat, row, col, block_shape)
        mat_block += val

    return mat


def _bsr_pruned(bsr):
    return bsr_from_triplets(
        rows_of_blocks=bsr.nrow,
        cols_of_blocks=bsr.ncol,
        rows=bsr.uncompress_rows(),
        columns=bsr.columns[: bsr.nnz],
        values=bsr.values[: bsr.nnz],
        prune_numerical_zeros=True,
    )


def _bsr_to_dense(bsr):
    mat = np.zeros(bsr.shape)

    offsets = bsr.offsets.numpy()
    row_counts = None if bsr.row_counts is None else bsr.row_counts.numpy()
    columns = bsr.columns.numpy()
    values = bsr.values.numpy()

    for row in range(bsr.nrow):
        beg = offsets[row]
        end = offsets[row + 1] if row_counts is None else beg + row_counts[row]

        for block in range(beg, end):
            mat_block = _get_block(mat, row, columns[block], bsr.block_shape)
            mat_block += values[block]

    return mat


def _assert_bsr_active_columns(test, bsr, expected):
    offsets = bsr.offsets.numpy()
    row_counts = None if bsr.row_counts is None else bsr.row_counts.numpy()
    columns = bsr.columns.numpy()

    test.assertEqual(len(expected), bsr.nrow)
    for row, expected_row in enumerate(expected):
        row_end = offsets[row + 1] if row_counts is None else offsets[row] + row_counts[row]
        np.testing.assert_array_equal(columns[offsets[row] : row_end], np.array(expected_row))


def _make_gapped_csr(device):
    bsr = bsr_zeros(2, 3, float, device=device, row_capacity=3)
    bsr_set_from_triplets(
        bsr,
        rows=wp.array([0, 0, 1, 1], dtype=int, device=device),
        columns=wp.array([0, 2, 1, 2], dtype=int, device=device),
        values=wp.array([1.0, 2.0, 3.0, 4.0], dtype=float, device=device),
        topology="padded",
    )
    return bsr


def test_csr_from_triplets(test, device):
    rng = np.random.default_rng(123)

    shape = (8, 6)
    n = 100

    rows = wp.array(rng.integers(0, high=shape[0], size=n, dtype=int), dtype=int, device=device)
    cols = wp.array(rng.integers(0, high=shape[1], size=n, dtype=int), dtype=int, device=device)
    vals = wp.array(rng.random(size=n), dtype=float, device=device)

    ref = _triplets_to_dense(shape, rows, cols, vals)

    csr = bsr_zeros(shape[0], shape[1], float, device=device)
    bsr_set_from_triplets(csr, rows, cols, vals)
    test.assertEqual(csr.block_size, 1)

    res = _bsr_to_dense(csr)

    assert_np_equal(res, ref, 0.0001)


def test_bsr_from_triplets(test, device):
    rng = np.random.default_rng(123)

    block_shape = (3, 2)
    nrow = 4
    ncol = 9
    shape = (block_shape[0] * nrow, block_shape[1] * ncol)
    n = 50

    rows = wp.array(rng.integers(0, high=nrow, size=n, dtype=int), dtype=int, device=device)
    cols = wp.array(rng.integers(0, high=ncol, size=n, dtype=int), dtype=int, device=device)
    vals = wp.array(rng.random(size=(n, block_shape[0], block_shape[1])), dtype=float, device=device)

    ref = _triplets_to_dense(shape, rows, cols, vals)

    bsr = bsr_zeros(nrow, ncol, wp.types.matrix(shape=block_shape, dtype=float), device=device)
    bsr_set_from_triplets(bsr, rows, cols, vals)
    test.assertEqual(bsr.block_size, block_shape[0] * block_shape[1])

    res = _bsr_to_dense(bsr)

    assert_np_equal(res, ref, 0.0001)

    topology_only_rows = wp.array([1, 0, 1], dtype=int, device=device)
    topology_only_cols = wp.array([2, 1, 2], dtype=int, device=device)

    topology_only = bsr_zeros(2, 3, float, device=device)
    bsr_set_from_triplets(topology_only, topology_only_rows, topology_only_cols, values=None)
    test.assertEqual(topology_only.nnz_sync(), 2)
    np.testing.assert_array_equal(topology_only.offsets.numpy(), np.array([0, 1, 2]))
    test.assertIsNone(topology_only.row_counts)
    np.testing.assert_array_equal(topology_only.columns.numpy()[: topology_only.nnz], np.array([1, 2]))

    padded_topology_only = bsr_zeros(2, 3, float, device=device)
    padded_topology_only.nnz = 5
    padded_topology_only.offsets = wp.array([0, 2, 5], dtype=int, device=device)
    padded_topology_only.row_counts = wp.array([0, 0], dtype=int, device=device)
    padded_topology_only.columns = wp.full(5, value=-9, dtype=int, device=device)
    padded_topology_only.values = wp.empty(5, dtype=float, device=device)
    bsr_set_from_triplets(
        padded_topology_only,
        topology_only_rows,
        topology_only_cols,
        values=None,
        topology="padded",
    )
    np.testing.assert_array_equal(padded_topology_only.offsets.numpy(), np.array([0, 2, 5]))
    np.testing.assert_array_equal(padded_topology_only.row_counts.numpy(), np.array([1, 1]))
    _assert_bsr_active_columns(test, padded_topology_only, [[1], [2]])

    # test zero-length inputs
    bsr_set_from_triplets(
        bsr,
        wp.array([], dtype=int, device=device),
        wp.array([], dtype=int, device=device),
        wp.array([], shape=(0, block_shape[0], block_shape[1]), dtype=float, device=device),
    )
    test.assertEqual(bsr.nnz, 0)

    # test passing indices with wrong data ty[e]
    rows = wp.array(rows.numpy().astype(float), dtype=float, device=device)
    cols = wp.array(cols.numpy().astype(float), dtype=float, device=device)
    with test.assertRaisesRegex(
        TypeError,
        r"Rows and columns arrays must be of type int32$",
    ):
        bsr_set_from_triplets(bsr, rows, cols, vals)


def test_bsr_from_triplets_prune_numerical_zeros(test, device):
    rows = wp.array([1, 0, 2, 3], dtype=int)
    cols = wp.array([0, 1, 2, 3], dtype=int)
    vals = wp.zeros(len(rows), dtype=float)

    A = bsr_from_triplets(
        rows_of_blocks=12,  # Number of rows of blocks
        cols_of_blocks=12,  # Number of columns of blocks
        rows=rows,  # Row indices
        columns=cols,  # Column indices
        values=vals,  # Block values
        prune_numerical_zeros=False,
    )
    assert A.nnz_sync() == 4

    A = bsr_from_triplets(
        rows_of_blocks=12,  # Number of rows of blocks
        cols_of_blocks=12,  # Number of columns of blocks
        rows=rows,  # Row indices
        columns=cols,  # Column indices
        values=vals,  # Block values
        prune_numerical_zeros=True,
    )
    assert A.nnz_sync() == 0


def test_bsr_gapped_layout(test, device):
    A = _make_gapped_csr(device)
    expected_A = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 4.0]])

    np.testing.assert_array_equal(A.offsets.numpy(), np.array([0, 3, 6]))
    np.testing.assert_array_equal(A.row_counts.numpy(), np.array([2, 2]))
    _assert_bsr_active_columns(test, A, [[0, 2], [1, 2]])
    np.testing.assert_array_equal(A.uncompress_rows().numpy(), np.array([0, 0, -1, 1, 1, -1]))
    assert_np_equal(_bsr_to_dense(A), expected_A)

    x = wp.array([5.0, 7.0, 11.0], dtype=float, device=device)
    assert_np_equal(bsr_mv(A, x).numpy(), np.array([27.0, 65.0]))
    assert_np_equal(bsr_get_diag(A).numpy(), np.array([1.0, 3.0]))

    compact_copy = bsr_copy(A)
    test.assertIsNone(compact_copy.row_counts)
    assert_np_equal(_bsr_to_dense(compact_copy), expected_A)

    padded_copy = bsr_copy(2.0 * A, topology="padded")
    np.testing.assert_array_equal(padded_copy.offsets.numpy(), A.offsets.numpy())
    np.testing.assert_array_equal(padded_copy.row_counts.numpy(), A.row_counts.numpy())
    assert_np_equal(_bsr_to_dense(padded_copy), 2.0 * expected_A)

    padded_dest = bsr_zeros(2, 3, float, device=device, row_capacity=3)
    bsr_assign(padded_dest, A, topology="padded")
    _assert_bsr_active_columns(test, padded_dest, [[0, 2], [1, 2]])
    assert_np_equal(_bsr_to_dense(padded_dest), expected_A)

    too_small_assign = bsr_zeros(2, 3, float, device=device, row_capacity=1)
    bsr_assign(too_small_assign, A, topology="padded")
    test.assertEqual(too_small_assign.status_sync(), BSR_STATUS_ROW_CAPACITY_EXCEEDED)

    axpy_x = bsr_from_triplets(
        2,
        3,
        rows=wp.array([0, 1, 1], dtype=int, device=device),
        columns=wp.array([1, 0, 2], dtype=int, device=device),
        values=wp.array([5.0, 7.0, 11.0], dtype=float, device=device),
    )
    axpy_y = _make_gapped_csr(device)
    bsr_axpy(axpy_x, axpy_y, alpha=2.0, beta=3.0, topology="padded")
    assert_np_equal(_bsr_to_dense(axpy_y), np.array([[3.0, 10.0, 6.0], [14.0, 9.0, 34.0]]))

    B = bsr_zeros(3, 2, float, device=device, row_capacity=2)
    bsr_set_from_triplets(
        B,
        rows=wp.array([0, 1, 2], dtype=int, device=device),
        columns=wp.array([0, 1, 0], dtype=int, device=device),
        values=wp.array([5.0, 7.0, 11.0], dtype=float, device=device),
        topology="padded",
    )
    mm_dest = bsr_zeros(2, 2, float, device=device, row_capacity=2)
    bsr_mm(A, B, mm_dest, topology="padded")
    _assert_bsr_active_columns(test, mm_dest, [[0], [0, 1]])
    assert_np_equal(_bsr_to_dense(mm_dest), np.array([[27.0, 0.0], [44.0, 21.0]]))

    too_small_mm = bsr_zeros(2, 2, float, device=device, row_capacity=1)
    bsr_mm(A, B, too_small_mm, topology="padded")
    test.assertEqual(too_small_mm.status_sync(), BSR_STATUS_ROW_CAPACITY_EXCEEDED)

    transpose_dest = bsr_zeros(3, 2, float, device=device, row_capacity=2)
    bsr_set_transpose(transpose_dest, A, topology="padded")
    _assert_bsr_active_columns(test, transpose_dest, [[0], [1], [0, 1]])
    assert_np_equal(_bsr_to_dense(transpose_dest), np.array([[1.0, 0.0], [0.0, 3.0], [2.0, 4.0]]))

    row_capacity = wp.array([2, 3], dtype=int, device=device)
    triplet_dest = bsr_zeros(2, 3, float, device=device, row_capacity=row_capacity)
    bsr_set_from_triplets(
        triplet_dest,
        rows=wp.array([1, 0, 1, 0, 1], dtype=int, device=device),
        columns=wp.array([2, 1, 2, 0, 1], dtype=int, device=device),
        values=wp.array([4.0, 1.0, 5.0, 2.0, 3.0], dtype=float, device=device),
        topology="padded",
    )
    _assert_bsr_active_columns(test, triplet_dest, [[0, 1], [1, 2]])
    assert_np_equal(_bsr_to_dense(triplet_dest), np.array([[2.0, 1.0, 0.0], [0.0, 3.0, 9.0]]))

    too_small_triplets = bsr_zeros(2, 3, float, device=device, row_capacity=1)
    bsr_set_from_triplets(
        too_small_triplets,
        rows=wp.array([0, 0, 1], dtype=int, device=device),
        columns=wp.array([0, 2, 1], dtype=int, device=device),
        values=wp.array([1.0, 2.0, 3.0], dtype=float, device=device),
        topology="padded",
    )
    test.assertEqual(too_small_triplets.status_sync(), BSR_STATUS_ROW_CAPACITY_EXCEEDED)

    compress_capacity = wp.array([4, 3], dtype=int, device=device)
    compress_candidate = bsr_zeros(2, 3, float, device=device, row_capacity=compress_capacity)
    compress_candidate.row_counts = wp.array([4, 3], dtype=int, device=device)
    compress_candidate.columns = wp.array([2, 0, 2, 1, 1, 2, 1], dtype=int, device=device)
    compress_candidate.values = wp.array([5.0, 2.0, -1.0, 3.0, 4.0, 8.0, -4.0], dtype=float, device=device)
    bsr_compress(compress_candidate, inplace=True, topology="padded")
    np.testing.assert_array_equal(compress_candidate.row_counts.numpy(), np.array([3, 2]))
    _assert_bsr_active_columns(test, compress_candidate, [[0, 1, 2], [1, 2]])
    assert_np_equal(_bsr_to_dense(compress_candidate), np.array([[2.0, 3.0, 4.0], [0.0, 0.0, 8.0]]))

    long_row_columns = (np.arange(160, dtype=np.int32) * 37 + 11) % 97
    long_row_values = (np.arange(160, dtype=np.float32) % 11.0) - 5.0
    expected_long_row_values = np.zeros(97, dtype=np.float32)
    for col, value in zip(long_row_columns, long_row_values, strict=True):
        expected_long_row_values[col] += value
    expected_long_row_columns = np.unique(long_row_columns[long_row_values != 0.0]).astype(np.int32)

    long_row_capacity = wp.array([long_row_columns.size, 0], dtype=int, device=device)
    long_row = bsr_zeros(2, 97, float, device=device, row_capacity=long_row_capacity)
    long_row.row_counts = wp.array([long_row_columns.size, 0], dtype=int, device=device)
    long_row.columns = wp.array(long_row_columns, dtype=int, device=device)
    long_row.values = wp.array(long_row_values, dtype=float, device=device)
    bsr_compress(long_row, inplace=True, topology="compact")
    test.assertIsNone(long_row.row_counts)
    test.assertEqual(long_row.nnz_sync(), expected_long_row_columns.size)
    np.testing.assert_array_equal(
        long_row.offsets.numpy(), np.array([0, expected_long_row_columns.size, expected_long_row_columns.size])
    )
    np.testing.assert_array_equal(long_row.columns.numpy()[: long_row.nnz], expected_long_row_columns)
    np.testing.assert_allclose(
        long_row.values.numpy()[: long_row.nnz], expected_long_row_values[expected_long_row_columns]
    )


def test_bsr_from_triplets_gradient(test, device):
    rng = np.random.default_rng(123)

    block_shape = (3, 3)
    nrow = 2
    ncol = 2

    n = 4
    rows = wp.array([1, 0, 0, 1], dtype=int, device=device)
    cols = wp.array([0, 1, 0, 0], dtype=int, device=device)

    vals = wp.array(
        rng.random(size=(n, block_shape[0], block_shape[1])), dtype=wp.mat33, device=device, requires_grad=True
    )

    with wp.Tape() as tape:
        mat = bsr_from_triplets(nrow, ncol, rows, cols, vals)

    assert mat.nnz_sync() == 3

    zero_block = np.zeros((3, 3))
    ones_block = np.ones((3, 3))

    mat.values.grad[0:1].fill_(1.0)
    tape.backward()
    assert_np_equal(vals.grad.numpy(), np.stack((zero_block, zero_block, ones_block, zero_block)))
    tape.zero()

    mat.values.grad[1:2].fill_(1.0)
    tape.backward()
    assert_np_equal(vals.grad.numpy(), np.stack((zero_block, ones_block, zero_block, zero_block)))
    tape.zero()

    mat.values.grad[2:3].fill_(1.0)
    tape.backward()
    assert_np_equal(vals.grad.numpy(), np.stack((ones_block, zero_block, zero_block, ones_block)))
    tape.zero()

    padded_vals = wp.array(
        rng.random(size=(n, block_shape[0], block_shape[1])), dtype=wp.mat33, device=device, requires_grad=True
    )
    padded = bsr_zeros(nrow, ncol, wp.mat33, device=device, row_capacity=ncol)
    padded.values = wp.empty(shape=(padded.nnz,), dtype=wp.mat33, device=device, requires_grad=True)

    with wp.Tape() as tape:
        bsr_set_from_triplets(padded, rows, cols, padded_vals, topology="padded")

    np.testing.assert_array_equal(padded.row_counts.numpy(), np.array([2, 1]))

    padded.values.grad[0:1].fill_(1.0)
    tape.backward()
    assert_np_equal(padded_vals.grad.numpy(), np.stack((zero_block, zero_block, ones_block, zero_block)))
    tape.zero()

    padded.values.grad[1:2].fill_(1.0)
    tape.backward()
    assert_np_equal(padded_vals.grad.numpy(), np.stack((zero_block, ones_block, zero_block, zero_block)))
    tape.zero()

    padded.values.grad[2:3].fill_(1.0)
    tape.backward()
    assert_np_equal(padded_vals.grad.numpy(), np.stack((ones_block, zero_block, zero_block, ones_block)))
    tape.zero()


def test_bsr_compress_gradient(test, device):
    def make_matrix():
        vals = wp.array([1.0, 2.0, 3.0, 0.0], dtype=float, device=device, requires_grad=True)
        mat = bsr_zeros(1, 3, float, device=device)
        mat.nnz = 4
        mat.offsets = wp.array([0, 4], dtype=int, device=device)
        mat.row_counts = wp.array([4], dtype=int, device=device)
        mat.columns = wp.array([1, 0, 1, 2], dtype=int, device=device)
        mat.values = vals
        return mat, vals

    A, vals = make_matrix()
    with wp.Tape() as tape:
        bsr_compress(A)

    test.assertEqual(A.nnz_sync(), 2)
    test.assertIsNone(A.row_counts)
    np.testing.assert_array_equal(A.offsets.numpy(), np.array([0, 2]))
    np.testing.assert_array_equal(A.columns.numpy()[: A.nnz], np.array([0, 1]))
    assert_np_equal(_bsr_to_dense(A), np.array([[2.0, 4.0, 0.0]]))

    A.values.grad[0:1].fill_(1.0)
    tape.backward()
    assert_np_equal(vals.grad.numpy(), np.array([0.0, 1.0, 0.0, 0.0]))
    tape.zero()

    A.values.grad[1:2].fill_(1.0)
    tape.backward()
    assert_np_equal(vals.grad.numpy(), np.array([1.0, 0.0, 1.0, 0.0]))
    tape.zero()

    padded, padded_vals = make_matrix()
    with wp.Tape() as tape:
        bsr_compress(padded, topology="padded")

    np.testing.assert_array_equal(padded.offsets.numpy(), np.array([0, 4]))
    np.testing.assert_array_equal(padded.row_counts.numpy(), np.array([2]))
    _assert_bsr_active_columns(test, padded, [[0, 1]])
    assert_np_equal(_bsr_to_dense(padded), np.array([[2.0, 4.0, 0.0]]))

    padded.values.grad[1:2].fill_(1.0)
    tape.backward()
    assert_np_equal(padded_vals.grad.numpy(), np.array([1.0, 0.0, 1.0, 0.0]))


def test_bsr_get_set_diag(test, device):
    rng = np.random.default_rng(123)

    block_shape = (3, 3)
    nrow = 4
    ncol = 4
    nnz = 6

    rows = wp.array([0, 1, 2, 3, 2, 1], dtype=int, device=device)
    cols = wp.array([1, 1, 1, 3, 2, 2], dtype=int, device=device)
    vals_np = rng.random(size=(nnz, block_shape[0], block_shape[1]))
    vals = wp.array(vals_np, dtype=float, device=device)

    bsr = bsr_zeros(nrow, ncol, wp.types.matrix(shape=block_shape, dtype=float), device=device)
    bsr_set_from_triplets(bsr, rows, cols, vals)

    diag = bsr_get_diag(bsr)
    diag_np = diag.numpy()

    assert_np_equal(diag_np[0], np.zeros(block_shape))
    assert_np_equal(diag_np[1], vals_np[1], tol=0.00001)
    assert_np_equal(diag_np[2], vals_np[4], tol=0.00001)
    assert_np_equal(diag_np[3], vals_np[3], tol=0.00001)

    # Passing out should produce the same result as allocating internally.
    sentinel = wp.full(shape=(nrow,), value=bsr.values.dtype(7.0), dtype=bsr.values.dtype, device=device)
    bsr_get_diag(bsr, out=sentinel)
    assert_np_equal(diag_np, sentinel.numpy(), tol=0.00001)

    # Test set_diag/get_diag round-trips with various block types

    # Array of blocks
    diag_bsr = bsr_diag(diag)
    bsr_get_diag(diag_bsr, out=diag)
    assert_np_equal(diag_np, diag.numpy())

    diag_scalar_np = rng.random(size=nrow)
    diag_scalar = wp.array(diag_scalar_np, device=device)
    diag_bsr = bsr_diag(diag_scalar)
    diag = bsr_get_diag(diag_bsr)
    assert_np_equal(diag_scalar_np, diag.numpy(), tol=0.000001)

    diag = bsr_get_diag(2.0 * diag_bsr)
    assert_np_equal(2.0 * diag_scalar_np, diag.numpy(), tol=0.000001)

    # Uniform block diagonal

    with test.assertRaisesRegex(ValueError, "BsrMatrix block type must be either warp matrix or scalar"):
        # 1d block type -- invalid
        diag_bsr = bsr_diag(diag=wp.vec3(vals_np[0, 0]), rows_of_blocks=nrow, cols_of_blocks=nrow + 1)

    diag_bsr = bsr_diag(diag=wp.mat33(vals_np[0]), rows_of_blocks=nrow, cols_of_blocks=nrow + 1)
    assert diag_bsr.values.shape[0] == nrow
    assert_np_equal(diag_bsr.values.numpy(), np.broadcast_to(vals_np[0], shape=(nrow, *block_shape)), tol=0.000001)

    diag_bsr = bsr_diag(diag=float(diag_scalar_np[0]), rows_of_blocks=nrow, cols_of_blocks=nrow + 1)
    assert diag_bsr.values.shape[0] == nrow
    assert_np_equal(diag_bsr.values.numpy(), np.full(nrow, diag_scalar_np[0]), tol=0.000001)

    # Identity matrix
    diag_bsr = bsr_identity(nrow, block_type=wp.mat44, device=device)
    assert diag_bsr.values.shape[0] == nrow
    assert_np_equal(diag_bsr.values.numpy(), np.broadcast_to(np.eye(4), shape=(nrow, 4, 4)), tol=0.000001)

    diag_csr = bsr_identity(nrow, block_type=wp.float64, device=device)
    np.testing.assert_array_equal(diag_csr.values.numpy(), np.ones(nrow, dtype=float))


def test_bsr_split_merge(test, device):
    rng = np.random.default_rng(123)

    block_shape = (4, 2)
    nrow = 4
    ncol = 8
    n = 20

    rows = wp.array(rng.integers(0, high=nrow, size=n, dtype=int), dtype=int, device=device)
    cols = wp.array(rng.integers(0, high=ncol, size=n, dtype=int), dtype=int, device=device)
    vals = wp.array(rng.random(size=(n, block_shape[0], block_shape[1])), dtype=float, device=device)

    bsr = bsr_zeros(nrow, ncol, wp.types.matrix(shape=block_shape, dtype=float), device=device)
    bsr_set_from_triplets(bsr, rows, cols, vals)
    ref = _bsr_to_dense(bsr)

    bsr_split = bsr_copy(bsr, block_shape=(2, 2))
    test.assertEqual(bsr_split.block_size, 4)
    res = _bsr_to_dense(bsr_split)
    assert_np_equal(res, ref, 0.0001)

    bsr_split = bsr_copy(bsr, block_shape=(1, 1))
    test.assertEqual(bsr_split.block_size, 1)
    res = _bsr_to_dense(bsr_split)
    assert_np_equal(res, ref, 0.0001)

    bsr_merge = bsr_copy(bsr, block_shape=(4, 4))
    test.assertEqual(bsr_merge.block_size, 16)
    res = _bsr_to_dense(bsr_merge)
    assert_np_equal(res, ref, 0.0001)

    bsr_merge = bsr_copy(bsr, block_shape=(8, 8))
    test.assertEqual(bsr_merge.block_size, 64)
    res = _bsr_to_dense(bsr_merge)
    assert_np_equal(res, ref, 0.0001)

    with test.assertRaisesRegex(ValueError, "Incompatible dest and src block shapes"):
        bsr_copy(bsr, block_shape=(3, 3))

    with test.assertRaisesRegex(ValueError, "Incompatible dest and src block shapes"):
        bsr_copy(bsr, block_shape=(5, 5))

    with test.assertRaisesRegex(
        ValueError,
        r"The requested block shape \(32, 32\) does not evenly divide the source matrix of total size \(16, 16\)",
    ):
        bsr_copy(bsr, block_shape=(32, 32))


def test_bsr_assign_masked(test, device):
    rng = np.random.default_rng(123)

    block_shape = (1, 2)
    nrow = 16
    ncol = 8
    shape = (block_shape[0] * nrow, block_shape[1] * ncol)
    n = 20

    rows = wp.array(rng.integers(0, high=nrow, size=n, dtype=int), dtype=int, device=device)
    cols = wp.array(rng.integers(0, high=ncol, size=n, dtype=int), dtype=int, device=device)
    vals = wp.array(rng.random(size=(n, block_shape[0], block_shape[1])), dtype=float, device=device)

    A = bsr_from_triplets(nrow, ncol, rows, cols, vals)

    # Extract coarse diagonal with copy + diag funcs, for reference
    A_coarse = bsr_copy(A, block_shape=(4, 4))
    ref = _bsr_to_dense(bsr_diag(bsr_get_diag(A_coarse)))

    # Extract coarse diagonal with masked assign (more memory efficient)
    diag_masked = bsr_diag(rows_of_blocks=shape[0] // 4, block_type=A_coarse.dtype, device=device)
    bsr_assign(src=A, dest=diag_masked, topology="masked")
    res = _bsr_to_dense(diag_masked)

    assert_np_equal(res, ref, 0.0001)


def make_test_bsr_transpose(block_shape, scalar_type):
    def test_bsr_transpose(test, device):
        rng = np.random.default_rng(123)

        nrow = 4
        ncol = 5
        nnz = 6

        rows = wp.array([0, 1, 2, 3, 2, 1], dtype=int, device=device)
        cols = wp.array([1, 4, 1, 3, 0, 2], dtype=int, device=device)

        vals_np = rng.random(size=(nnz, block_shape[0], block_shape[1]))
        vals = wp.array(vals_np, dtype=scalar_type, device=device).reshape((nnz, block_shape[0], block_shape[1]))

        bsr = bsr_zeros(nrow, ncol, wp.types.matrix(shape=block_shape, dtype=scalar_type), device=device)
        bsr_set_from_triplets(bsr, rows, cols, vals)
        ref = 2.0 * np.transpose(_bsr_to_dense(bsr))

        bsr_transposed = (2.0 * bsr).transpose().eval()

        res = _bsr_to_dense(bsr_transposed)
        assert_np_equal(res, ref, 0.0001)

        if block_shape[0] != block_shape[-1]:
            # test incompatible block shape
            with test.assertRaisesRegex(ValueError, "Destination block shape must be"):
                bsr_set_transpose(dest=bsr, src=bsr)

        # test masked transpose
        # remove some non zeros from src and dest matrices
        bsr_set_from_triplets(bsr, rows[:3], cols[:3], vals[:3])
        bsr_transposed = bsr_from_triplets(
            bsr_transposed.nrow,
            bsr_transposed.ncol,
            bsr_transposed.uncompress_rows()[:3],
            bsr_transposed.columns[:3],
            bsr_transposed.values[:3],
        )

        assert_np_equal(bsr_transposed.uncompress_rows().numpy()[:3], [0, 1, 1])
        assert_np_equal(bsr_transposed.columns.numpy()[:3], [2, 0, 2])
        bsr_set_transpose(bsr_transposed, bsr, topology="masked")
        assert _bsr_pruned(bsr_transposed).nnz_sync() == 2

    return test_bsr_transpose


def make_test_bsr_axpy(block_shape, scalar_type):
    def test_bsr_axpy(test, device):
        rng = np.random.default_rng(123)

        nrow = 2
        ncol = 3
        nnz = 6

        alphas = [-1.0, 0.0, 1.0]
        betas = [2.0, -1.0, 0.0]

        x_rows = wp.array(rng.integers(0, high=nrow, size=nnz, dtype=int), dtype=int, device=device)
        x_cols = wp.array(rng.integers(0, high=ncol, size=nnz, dtype=int), dtype=int, device=device)
        x_vals = wp.array(rng.random(size=(nnz, block_shape[0], block_shape[1])), dtype=scalar_type, device=device)
        x_vals = x_vals.reshape((nnz, block_shape[0], block_shape[1]))

        x = bsr_zeros(nrow, ncol, wp.types.matrix(shape=block_shape, dtype=scalar_type), device=device)
        bsr_set_from_triplets(x, x_rows, x_cols, x_vals)

        y_rows = wp.array(rng.integers(0, high=nrow, size=nnz, dtype=int), dtype=int, device=device)
        y_cols = wp.array(rng.integers(0, high=ncol, size=nnz, dtype=int), dtype=int, device=device)
        y_vals = wp.array(rng.random(size=(nnz, block_shape[0], block_shape[1])), dtype=scalar_type, device=device)
        y_vals = y_vals.reshape((nnz, block_shape[0], block_shape[1]))

        y = bsr_zeros(nrow, ncol, wp.types.matrix(shape=block_shape, dtype=scalar_type), device=device)
        bsr_set_from_triplets(y, y_rows, y_cols, y_vals)

        work_arrays = bsr_axpy_work_arrays()
        for alpha, beta in zip(alphas, betas, strict=True):
            ref = alpha * _bsr_to_dense(x) + beta * _bsr_to_dense(y)
            bsr_axpy(x, y, alpha, beta, work_arrays=work_arrays)

            res = _bsr_to_dense(y)
            assert_np_equal(res, ref, 0.0001)

        # test aliasing
        ref = 3.0 * _bsr_to_dense(y)
        y += y * 2.0
        res = _bsr_to_dense(y)
        assert_np_equal(res, ref, 0.0001)

        # test masked
        y_mask = bsr_from_triplets(nrow, ncol, y.uncompress_rows()[:1], y.columns[:1], y.values[:1])
        bsr_axpy(y, y_mask, topology="masked")
        assert y_mask.nnz_sync() == 1
        assert_np_equal(y_mask.values.numpy(), 2.0 * y.values[:1].numpy(), 0.0001)

        # test incompatible shapes
        y.ncol = y.ncol + 1
        with test.assertRaisesRegex(ValueError, "Matrices must have the same number of rows and columns"):
            bsr_axpy(x, y)

    return test_bsr_axpy


def make_test_bsr_mm(block_shape, scalar_type):
    def test_bsr_mm(test, device):
        rng = np.random.default_rng(123)

        x_nrow = 3
        x_ncol = 2
        x_block_shape = block_shape

        y_nrow = 2
        y_ncol = 3
        y_block_shape = block_shape[::-1]

        z_nrow = x_nrow
        z_ncol = y_ncol
        z_block_shape = (x_block_shape[0], y_block_shape[1])

        nnz = 6

        alphas = [-1.0, 0.0, 2.0]
        betas = [2.0, -1.0, 0.0]

        x_rows = wp.array(rng.integers(0, high=x_nrow, size=nnz, dtype=int), dtype=int, device=device)
        x_cols = wp.array(rng.integers(0, high=x_ncol, size=nnz, dtype=int), dtype=int, device=device)
        x_vals = wp.array(rng.random(size=(nnz, x_block_shape[0], x_block_shape[1])), dtype=scalar_type, device=device)
        x_vals = x_vals.reshape((nnz, x_block_shape[0], x_block_shape[1]))

        x = bsr_zeros(x_nrow, x_ncol, wp.types.matrix(shape=x_block_shape, dtype=scalar_type), device=device)
        bsr_set_from_triplets(x, x_rows, x_cols, x_vals)

        y_rows = wp.array(rng.integers(0, high=y_nrow, size=nnz, dtype=int), dtype=int, device=device)
        y_cols = wp.array(rng.integers(0, high=y_ncol, size=nnz, dtype=int), dtype=int, device=device)
        y_vals = wp.array(rng.random(size=(nnz, y_block_shape[0], y_block_shape[1])), dtype=scalar_type, device=device)
        y_vals = y_vals.reshape((nnz, y_block_shape[0], y_block_shape[1]))

        y = bsr_zeros(y_nrow, y_ncol, wp.types.matrix(shape=y_block_shape, dtype=scalar_type), device=device)
        bsr_set_from_triplets(y, y_rows, y_cols, y_vals)

        z_rows = wp.array(rng.integers(0, high=z_nrow, size=nnz, dtype=int), dtype=int, device=device)
        z_cols = wp.array(rng.integers(0, high=z_ncol, size=nnz, dtype=int), dtype=int, device=device)
        z_vals = wp.array(rng.random(size=(nnz, z_block_shape[0], z_block_shape[1])), dtype=scalar_type, device=device)
        z_vals = z_vals.reshape((nnz, z_block_shape[0], z_block_shape[1]))

        z = bsr_zeros(z_nrow, z_ncol, wp.types.matrix(shape=z_block_shape, dtype=scalar_type), device=device)
        bsr_set_from_triplets(z, z_rows, z_cols, z_vals)

        work_arrays = bsr_mm_work_arrays()
        for alpha, beta in zip(alphas, betas, strict=True):
            ref = alpha * (_bsr_to_dense(x) @ _bsr_to_dense(y)) + beta * _bsr_to_dense(z)

            bsr_mm(x, y, z, alpha, beta, work_arrays=work_arrays)

            res = _bsr_to_dense(z)
            assert_np_equal(res, ref, 0.0001)

        # test reusing topology from work arrays
        # (assumes betas[-1] = 0)
        bsr_mm(x, y, z, alpha, beta, work_arrays=work_arrays, reuse_topology=True)
        assert_np_equal(res, ref, 0.0001)

        # test masked mm
        z = bsr_diag(rows_of_blocks=z.nrow, block_type=z.dtype, device=z.device)
        bsr_mm(x, y, z, topology="masked")
        res = _bsr_to_dense(z)
        ref = _bsr_to_dense(bsr_diag(bsr_get_diag(x @ y)))
        assert_np_equal(res, ref, 0.0001)

        # using overloaded operators
        x = (alpha * x) @ y
        assert_np_equal(res, ref, 0.0001)

        # test aliasing of matrix arguments
        # x = alpha * z * x + beta * x
        alpha, beta = alphas[0], betas[0]
        ref = alpha * (_bsr_to_dense(z) @ _bsr_to_dense(x)) + beta * _bsr_to_dense(x)
        bsr_mm(z, x, x, alpha, beta)

        res = _bsr_to_dense(x)
        assert_np_equal(res, ref, 0.0001)

        # z = alpha * z * z + beta * z
        ref = alpha * (_bsr_to_dense(z) @ _bsr_to_dense(z)) + beta * _bsr_to_dense(z)
        bsr_mm(z, z, z, alpha, beta)

        res = _bsr_to_dense(z)
        assert_np_equal(res, ref, 0.0001)

        # test incompatible shapes
        if block_shape[0] != block_shape[-1]:
            with test.assertRaisesRegex(ValueError, "Incompatible block sizes"):
                bsr_mm(z, y)

        y.ncol = y.ncol * 2
        with test.assertRaisesRegex(ValueError, "Incompatible number of rows/columns"):
            bsr_mm(y, z)

    return test_bsr_mm


def make_test_bsr_mv(block_shape, scalar_type):
    def test_bsr_mv(test, device):
        rng = np.random.default_rng(123)

        nrow = 2
        ncol = 3
        nnz = 6

        alphas = [-1.0, 0.0, 1.0]
        betas = [2.0, -1.0, 0.0]
        A_rows = wp.array(rng.integers(0, high=nrow, size=nnz, dtype=int), dtype=int, device=device)
        A_cols = wp.array(rng.integers(0, high=ncol, size=nnz, dtype=int), dtype=int, device=device)
        A_vals = wp.array(rng.random(size=(nnz, block_shape[0], block_shape[1])), dtype=scalar_type, device=device)
        A_vals = A_vals.reshape((nnz, block_shape[0], block_shape[1]))

        A = bsr_zeros(nrow, ncol, wp.types.matrix(shape=block_shape, dtype=scalar_type), device=device)
        bsr_set_from_triplets(A, A_rows, A_cols, A_vals)

        if block_shape[1] == 1:
            x = wp.array(rng.random(size=ncol), dtype=scalar_type, device=device)
        else:
            x = wp.array(
                rng.random(size=(ncol, block_shape[1])),
                dtype=wp.types.vector(length=block_shape[1], dtype=scalar_type),
                device=device,
            )

        if block_shape[0] == 1:
            y = wp.array(rng.random(size=nrow), dtype=scalar_type, device=device)
        else:
            y = wp.array(
                rng.random(size=(nrow, block_shape[0])),
                dtype=wp.types.vector(length=block_shape[0], dtype=scalar_type),
                device=device,
            )

        work_buffer = wp.empty_like(y)
        for alpha, beta in zip(alphas, betas, strict=True):
            ref = alpha * _bsr_to_dense(A) @ x.numpy().flatten() + beta * y.numpy().flatten()

            if beta == 0.0:
                y = A @ x
            else:
                bsr_mv(A, x, y, alpha, beta, work_buffer=work_buffer)

            res = y.numpy().flatten()
            assert_np_equal(res, ref, 0.0001)

        # test transposed product
        ref = alpha * y.numpy().flatten() @ _bsr_to_dense(A)
        x = y @ (A * alpha)
        res = x.numpy().flatten()
        assert_np_equal(res, ref, 0.0001)

        # test aliasing
        AAt = bsr_mm(A, bsr_transposed(A))
        assert_np_equal(_bsr_to_dense(AAt), _bsr_to_dense(A) @ _bsr_to_dense(A).T, 0.0001)

        alpha, beta = alphas[0], betas[0]
        ref = alpha * _bsr_to_dense(AAt) @ y.numpy().flatten() + beta * y.numpy().flatten()
        bsr_mv(AAt, y, y, alpha, beta)
        res = y.numpy().flatten()
        assert_np_equal(res, ref, 0.0001)

        A.ncol = A.ncol + 1
        with test.assertRaisesRegex(ValueError, "Incompatible 'x'"):
            bsr_mv(A, x, y)

        A.ncol = A.ncol - 1
        A.nrow = A.nrow - 1
        with test.assertRaisesRegex(ValueError, "Incompatible 'y'"):
            bsr_mv(A, x, y)

    return test_bsr_mv


def make_test_bsr_multiply_deep(block_shape, scalar_type):
    def test_bsr_multiply_deep(test, device):
        """Test BSR matrix multiplication with deep matrices (many columns > 256)"""
        rng = np.random.default_rng(123)

        # Generate a dense matrix with few rows and many columns (> 256)
        nrow = (4 + block_shape[0] - 1) // block_shape[0]
        ncol = (600 + block_shape[1] - 1) // block_shape[1]

        # Create a dense "sparse" matrix
        values = rng.random(size=(nrow * ncol, block_shape[0], block_shape[1]))
        rows, cols = np.meshgrid(np.arange(nrow), np.arange(ncol))

        # Convert to warp arrays
        rows = wp.array(rows.flatten(), dtype=int, device=device)
        cols = wp.array(cols.flatten(), dtype=int, device=device)
        vals = wp.array(values, dtype=scalar_type, device=device)

        # Convert to BSR using bsr_from_triplets
        A = bsr_from_triplets(nrow, ncol, rows, cols, vals)

        # Get dense representation for numpy reference
        A_dense = _bsr_to_dense(A)

        # Multiply with itself transpose using bsr_mm
        # A @ A.T should result in a nrow x nrow matrix
        At = bsr_transposed(A)

        result = bsr_mm(A, At)

        # Check that the result is correct against numpy reference
        result_dense = _bsr_to_dense(result)
        ref_dense = A_dense @ A_dense.T

        assert_np_equal(result_dense, ref_dense, 0.0001)

        # Additional test: multiply A.T @ A (should be ncol x ncol)
        result2 = bsr_mm(At, A)
        result2_dense = _bsr_to_dense(result2)
        ref2_dense = A_dense.T @ A_dense

        assert_np_equal(result2_dense, ref2_dense, 0.0001)

        # Test matrix vector products
        x = wp.array(rng.random(size=A.shape[1]), dtype=A.scalar_type, device=device)
        y = wp.array(rng.random(size=A.shape[0]), dtype=A.scalar_type, device=device)
        bsr_mv(A, x, y)
        res = y.numpy().flatten()
        ref = A_dense @ x.numpy().flatten()
        assert_np_equal(res, ref, 0.0001 * block_shape[1])

        bsr_mv(A, y, x, transpose=True)
        res = x.numpy().flatten()
        ref = A_dense.T @ y.numpy().flatten()
        assert_np_equal(res, ref, 0.0001 * block_shape[1])

    return test_bsr_multiply_deep


def test_bsr_mm_max_new_nnz(test, device):
    """Test that BSR matrix multiplication with max_new_nnz works"""
    A = bsr_from_triplets(
        2,
        2,
        wp.array([0, 0, 1, 1], dtype=int, device=device),
        wp.array([0, 1, 0, 1], dtype=int, device=device),
        wp.array([1.0, 2.0, 3.0, 4.0], dtype=wp.float32, device=device),
    )
    B = bsr_from_triplets(
        2,
        2,
        wp.array([0, 0, 1, 1], dtype=int, device=device),
        wp.array([0, 1, 0, 1], dtype=int, device=device),
        wp.array([1.0, 2.0, 3.0, 4.0], dtype=wp.float32, device=device),
    )
    C = bsr_zeros(2, 2, wp.float32, device=device)

    # max_new_nnz big enough
    bsr_mm(A, B, C, max_new_nnz=4)
    test.assertEqual(C.nnz_sync(), 4)

    bsr_set_zero(C)
    test.assertEqual(C.nnz_sync(), 0)

    # We skip the rest of the test on Windows due to unreliable stdout capture
    if sys.platform == "win32":
        return

    # max_new_nnz too small, check warning
    capture = StdOutCapture()
    capture.begin()
    bsr_mm(A, B, C, max_new_nnz=2)
    test.assertEqual(C.nnz_sync(), 2)
    output = capture.end()

    # Check that the output contains warnings about "max_new_nnz" being exceeded.
    test.assertRegex(output, r"exceeded")


def test_capturability(test, device):
    """Test that BSR operations are graph-capturable"""

    N = 5
    M = 3

    C = bsr_diag(wp.zeros(N, dtype=wp.mat33, device=device))

    rows = wp.array([3, 4, 2, 0, 1], dtype=int, device=device)
    columns = wp.array([2, 0, 1, 2, 1], dtype=int, device=device)
    values = wp.ones(5, dtype=wp.mat33, device=device)

    def test_body():
        A = bsr_from_triplets(
            N,
            M,
            rows=rows,
            columns=columns,
            values=values,
        )
        B = A + bsr_copy(A * 2.0)
        bsr_mm(A, bsr_transposed(B), C, max_new_nnz=N * N)

    # ensure necessary modules are loaded and reset result
    test_body()
    bsr_set_zero(C)
    test.assertEqual(C.nnz_sync(), 0)

    with wp.ScopedDevice(device):
        with wp.ScopedCapture(force_module_load=False) as capture:
            test_body()

    assert_array_equal(bsr_get_diag(C), wp.zeros(N, dtype=wp.mat33, device=device))

    wp.capture_launch(capture.graph)
    test.assertEqual(C.nnz_sync(), 9)
    assert_array_equal(bsr_get_diag(C), wp.full(N, value=wp.mat33(9.0), dtype=wp.mat33, device=device))


def test_bsr_compress_compact_capturability(test, device):
    """Test that native compact BSR compression is graph-capturable."""

    A = bsr_zeros(2, 3, float, device=device)
    A.nnz = 7
    A.offsets = wp.array([0, 4, 7], dtype=int, device=device)
    capture_row_counts = wp.array([4, 3], dtype=int, device=device)
    A.row_counts = capture_row_counts
    A.columns = wp.array([2, 0, 2, 1, 1, 2, 1], dtype=int, device=device)
    A.values = wp.array([5.0, 2.0, -1.0, 3.0, 4.0, 8.0, -4.0], dtype=float, device=device)

    with wp.ScopedDevice(device):
        with wp.ScopedCapture(force_module_load=False) as capture:
            bsr_compress(A, inplace=True, topology="compact")

    wp.capture_launch(capture.graph)

    test.assertEqual(A.nnz_sync(), 5)
    test.assertIsNone(A.row_counts)
    np.testing.assert_array_equal(A.offsets.numpy(), np.array([0, 3, 5]))
    np.testing.assert_array_equal(A.columns.numpy()[: A.nnz], np.array([0, 1, 2, 1, 2]))
    assert_np_equal(_bsr_to_dense(A), np.array([[2.0, 3.0, 4.0], [0.0, 0.0, 8.0]]))


def test_padded_bsr_capture_constructs_matrix(test, device):
    """Test that padded BSR matrices can be constructed during CUDA graph capture."""

    uniform_offsets = wp.empty(3, dtype=int, device=device)
    uniform_status = wp.empty(1, dtype=int, device=device)
    row_capacity = wp.array([1, 0, 3], dtype=int, device=device)
    per_row_offsets = wp.empty(4, dtype=int, device=device)
    per_row_status = wp.empty(1, dtype=int, device=device)
    captured_matrices = []

    def reset_outputs():
        uniform_offsets.fill_(-1)
        uniform_status.fill_(-1)
        per_row_offsets.fill_(-1)
        per_row_status.fill_(-1)

    def test_body(retain=False):
        uniform = bsr_zeros(2, 3, float, device=device, row_capacity=2)
        per_row = bsr_zeros(3, 4, float, device=device, row_capacity=row_capacity, nnz_capacity=4)

        if retain:
            captured_matrices.extend((uniform, per_row))

        wp.copy(dest=uniform_offsets, src=uniform.offsets, count=uniform.nrow + 1)
        wp.copy(dest=uniform_status, src=uniform._ensure_status(), count=1)
        wp.copy(dest=per_row_offsets, src=per_row.offsets, count=per_row.nrow + 1)
        wp.copy(dest=per_row_status, src=per_row._ensure_status(), count=1)

    test_body()
    reset_outputs()

    with wp.ScopedCapture(device=device, force_module_load=False) as capture:
        test_body(retain=True)

    wp.capture_launch(capture.graph)

    np.testing.assert_array_equal(uniform_offsets.numpy(), np.array([0, 2, 4], dtype=np.int32))
    np.testing.assert_array_equal(uniform_status.numpy(), np.array([BSR_STATUS_SUCCESS], dtype=np.int32))
    np.testing.assert_array_equal(per_row_offsets.numpy(), np.array([0, 1, 1, 4], dtype=np.int32))
    np.testing.assert_array_equal(per_row_status.numpy(), np.array([BSR_STATUS_SUCCESS], dtype=np.int32))


def test_padded_bsr_capture_triplets_and_overflow(test, device):
    """Test padded BSR triplet capture, including row-capacity overflow status."""

    rows = wp.array([0, 0, 1], dtype=int, device=device)
    columns = wp.array([0, 2, 1], dtype=int, device=device)
    values = wp.array([1.0, 2.0, 3.0], dtype=float, device=device)

    success_offsets = wp.empty(3, dtype=int, device=device)
    success_row_counts = wp.empty(2, dtype=int, device=device)
    success_columns = wp.empty(4, dtype=int, device=device)
    success_values = wp.empty(4, dtype=float, device=device)
    success_status = wp.empty(1, dtype=int, device=device)
    overflow_status = wp.empty(1, dtype=int, device=device)
    captured_matrices = []

    def reset_outputs():
        for array in (
            success_offsets,
            success_row_counts,
            success_columns,
            success_values,
            success_status,
            overflow_status,
        ):
            array.fill_(-1)

    def test_body(retain=False):
        success = bsr_zeros(2, 3, float, device=device, row_capacity=2)
        overflow = bsr_zeros(2, 3, float, device=device, row_capacity=1)

        if retain:
            captured_matrices.extend((success, overflow))

        bsr_set_from_triplets(success, rows, columns, values, topology="padded")
        bsr_set_from_triplets(overflow, rows, columns, values, topology="padded")

        wp.copy(dest=success_offsets, src=success.offsets, count=success.nrow + 1)
        wp.copy(dest=success_row_counts, src=success.row_counts, count=success.nrow)
        wp.copy(dest=success_columns, src=success.columns, count=success.nnz)
        wp.copy(dest=success_values, src=success.values, count=success.nnz)
        wp.copy(dest=success_status, src=success._ensure_status(), count=1)
        wp.copy(dest=overflow_status, src=overflow._ensure_status(), count=1)

    test_body()
    reset_outputs()

    with wp.ScopedCapture(device=device, force_module_load=False) as capture:
        test_body(retain=True)

    wp.capture_launch(capture.graph)

    np.testing.assert_array_equal(success_offsets.numpy(), np.array([0, 2, 4], dtype=np.int32))
    np.testing.assert_array_equal(success_row_counts.numpy(), np.array([2, 1], dtype=np.int32))
    np.testing.assert_array_equal(success_columns.numpy()[:3], np.array([0, 2, 1], dtype=np.int32))
    assert_np_equal(success_values.numpy()[:3], np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(success_status.numpy(), np.array([BSR_STATUS_SUCCESS], dtype=np.int32))
    np.testing.assert_array_equal(overflow_status.numpy(), np.array([BSR_STATUS_ROW_CAPACITY_EXCEEDED], dtype=np.int32))


def test_padded_bsr_capture_reblock_copy(test, device):
    """Test lazy padded status allocation during captured reblocked BSR copy."""

    src = bsr_zeros(1, 1, wp.mat22, device=device, row_capacity=1)
    rows = wp.array([0], dtype=int, device=device)
    columns = wp.array([0], dtype=int, device=device)
    values = wp.array(np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32), dtype=float, device=device)
    bsr_set_from_triplets(src, rows, columns, values, topology="padded")

    copy_offsets = wp.empty(3, dtype=int, device=device)
    copy_row_counts = wp.empty(2, dtype=int, device=device)
    copy_columns = wp.empty(4, dtype=int, device=device)
    copy_values = wp.empty(4, dtype=float, device=device)
    copy_status = wp.empty(1, dtype=int, device=device)
    captured_matrices = []

    def reset_outputs():
        for array in (copy_offsets, copy_row_counts, copy_columns, copy_values, copy_status):
            array.fill_(-1)

    def test_body(retain=False):
        dest = bsr_copy(src, block_shape=(1, 1), topology="padded")

        if retain:
            captured_matrices.append(dest)

        wp.copy(dest=copy_offsets, src=dest.offsets, count=dest.nrow + 1)
        wp.copy(dest=copy_row_counts, src=dest.row_counts, count=dest.nrow)
        wp.copy(dest=copy_columns, src=dest.columns, count=dest.nnz)
        wp.copy(dest=copy_values, src=dest.values, count=dest.nnz)
        wp.copy(dest=copy_status, src=dest._ensure_status(), count=1)

    test_body()
    reset_outputs()

    with wp.ScopedCapture(device=device, force_module_load=False) as capture:
        test_body(retain=True)

    wp.capture_launch(capture.graph)

    np.testing.assert_array_equal(copy_offsets.numpy(), np.array([0, 2, 4], dtype=np.int32))
    np.testing.assert_array_equal(copy_row_counts.numpy(), np.array([2, 2], dtype=np.int32))
    np.testing.assert_array_equal(copy_columns.numpy(), np.array([0, 1, 0, 1], dtype=np.int32))
    assert_np_equal(copy_values.numpy(), np.array([1.0, 2.0, 3.0, 4.0]))
    np.testing.assert_array_equal(copy_status.numpy(), np.array([BSR_STATUS_SUCCESS], dtype=np.int32))


def test_padded_bsr_status_sync_cuda_capture_rejected(test, device):
    """Test that live CUDA capture rejects padded BSR status readback."""

    bsr = bsr_zeros(1, 1, float, device=device, row_capacity=1)
    test.assertEqual(bsr.status_sync(), BSR_STATUS_SUCCESS)

    with test.assertRaisesRegex(RuntimeError, "cannot read sparse status during a live CUDA graph capture"):
        with wp.ScopedCapture(device=device, force_module_load=False):
            bsr.status_sync()

    # Raising inside the capture must leave the device cleanly recoverable.
    test.assertFalse(device.is_capturing)

    with wp.ScopedCapture(device=device, force_module_load=False) as capture:
        bsr.clear_status()

    wp.capture_launch(capture.graph)
    test.assertEqual(bsr.status_sync(), BSR_STATUS_SUCCESS)


def test_padded_bsr_capture_per_row_without_nnz_capacity_rejected(test, device):
    """A per-row ``row_capacity`` array without an explicit ``nnz_capacity`` needs
    a host nnz readback, which is rejected with a clear error during a live CUDA
    graph capture rather than failing obscurely."""

    row_capacity = wp.array([1, 0, 3], dtype=int, device=device)

    # Warm up the construction path (with nnz_capacity) outside capture so the
    # force_module_load=False capture does not trigger an in-capture module load.
    bsr_zeros(3, 4, float, device=device, row_capacity=row_capacity, nnz_capacity=4)

    with test.assertRaisesRegex(RuntimeError, "device-to-host readback"):
        with wp.ScopedCapture(device=device, force_module_load=False):
            bsr_zeros(3, 4, float, device=device, row_capacity=row_capacity)


def test_bsr_alloc(test, device):
    rows_of_blocks, cols_of_blocks = 3, 4

    bsr: BsrMatrix[float] = bsr_zeros(
        rows_of_blocks,
        cols_of_blocks,
        block_type=float,
        device=device,
    )

    row_capacity = wp.array([1, 0, 3], dtype=int, device=device)
    reserved = bsr_zeros(rows_of_blocks, cols_of_blocks, block_type=float, device=device, row_capacity=row_capacity)
    bsr_set_from_triplets(
        reserved,
        rows=wp.array([0, 2, 2], dtype=int, device=device),
        columns=wp.array([0, 1, 3], dtype=int, device=device),
        values=wp.array([1.0, 2.0, 3.0], dtype=float, device=device),
        topology="padded",
    )
    test.assertEqual(reserved.nnz, 4)
    np.testing.assert_array_equal(reserved.offsets.numpy(), np.array([0, 1, 1, 4], dtype=np.int32))
    np.testing.assert_array_equal(reserved.row_counts.numpy(), np.array([1, 0, 2], dtype=np.int32))
    assert_np_equal(
        _bsr_to_dense(reserved), np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 3.0]])
    )

    bsr_set_zero(reserved, topology="padded")
    test.assertEqual(reserved.nnz, 4)
    np.testing.assert_array_equal(reserved.offsets.numpy(), np.array([0, 1, 1, 4], dtype=np.int32))
    np.testing.assert_array_equal(reserved.row_counts.numpy(), np.zeros(rows_of_blocks, dtype=np.int32))
    assert_np_equal(_bsr_to_dense(reserved), np.zeros((rows_of_blocks, cols_of_blocks)))

    reset_capacity = wp.array([0, 2, 1], dtype=int, device=device)
    bsr_set_zero(reserved, topology="padded", row_capacity=reset_capacity)
    test.assertEqual(reserved.nnz_sync(), 3)
    np.testing.assert_array_equal(reserved.offsets.numpy(), np.array([0, 0, 2, 3], dtype=np.int32))
    np.testing.assert_array_equal(reserved.row_counts.numpy(), np.zeros(rows_of_blocks, dtype=np.int32))

    overallocated_capacity = wp.array([1, 0, 1], dtype=int, device=device)
    overallocated = bsr_zeros(3, 4, float, device=device, row_capacity=overallocated_capacity, nnz_capacity=8)
    bsr_set_from_triplets(
        overallocated,
        rows=wp.array([0, 2], dtype=int, device=device),
        columns=wp.array([1, 3], dtype=int, device=device),
        values=wp.array([2.0, 5.0], dtype=float, device=device),
        topology="padded",
    )
    test.assertEqual(overallocated.nnz, 8)
    np.testing.assert_array_equal(overallocated.offsets.numpy(), np.array([0, 1, 1, 2], dtype=np.int32))
    np.testing.assert_array_equal(overallocated.row_counts.numpy(), np.array([1, 0, 1], dtype=np.int32))
    assert_np_equal(
        _bsr_to_dense(overallocated), np.array([[0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 5.0]])
    )

    # Notify of new nnz upper bound. Allocs buffers, but nnz_sync still 0
    bsr.notify_nnz_changed(10)
    assert bsr.columns.shape[0] >= 6
    assert bsr.values.shape[0] >= 6

    assert bsr.nnz == 10
    assert bsr.nnz_sync() == 0

    # Set offsets and sync. Should update actual nnz
    offsets = wp.array([0, 2, 3, 6], dtype=int, device=device)
    bsr.offsets.assign(offsets)
    bsr.notify_nnz_changed()

    assert bsr.nnz == 6
    assert bsr.nnz_sync() == 6
    assert bsr.columns.shape[0] >= 6
    assert bsr.values.shape[0] >= 6


devices = get_test_devices()
cuda_test_devices = get_selected_cuda_test_devices()
cuda_test_devices_with_mempool = get_selected_cuda_test_devices_with_mempool()


class TestSparse(unittest.TestCase):
    def test_bsr_copy_scale(self):
        nrow = 6
        bsize = 2

        diag_bsr = bsr_diag(diag=wp.mat22(np.eye(bsize, dtype=float) * 2.0), rows_of_blocks=nrow)
        diag_copy = bsr_copy(diag_bsr, scalar_type=wp.float64)

        self.assertTrue(
            wp.types.types_equal(diag_copy.values.dtype, wp.types.matrix(shape=(bsize, bsize), dtype=wp.float64))
        )
        bsr_scale(x=diag_copy, alpha=0.5)

        res = _bsr_to_dense(diag_copy)
        ref = np.eye(nrow * bsize)
        assert_np_equal(res, ref, 0.0001)

        bsr_scale(x=diag_copy, alpha=0.0)
        self.assertEqual(diag_copy.nrow, nrow)
        self.assertEqual(diag_copy.ncol, nrow)
        self.assertEqual(diag_copy.nnz, diag_bsr.nnz)

        diag_pruned = _bsr_pruned(diag_copy)
        self.assertEqual(diag_pruned.nnz_sync(), 0)


add_function_test(TestSparse, "test_csr_from_triplets", test_csr_from_triplets, devices=devices)
add_function_test(TestSparse, "test_bsr_from_triplets", test_bsr_from_triplets, devices=devices)
add_function_test(
    TestSparse,
    "test_bsr_from_triplets_prune_numerical_zeros",
    test_bsr_from_triplets_prune_numerical_zeros,
    devices=devices,
)
add_function_test(TestSparse, "test_bsr_gapped_layout", test_bsr_gapped_layout, devices=devices)
add_function_test(TestSparse, "test_bsr_get_diag", test_bsr_get_set_diag, devices=devices)
add_function_test(TestSparse, "test_bsr_split_merge", test_bsr_split_merge, devices=devices)
add_function_test(TestSparse, "test_bsr_assign_masked", test_bsr_assign_masked, devices=devices)
add_function_test(TestSparse, "test_bsr_from_triplets_gradient", test_bsr_from_triplets_gradient, devices=devices)
add_function_test(TestSparse, "test_bsr_compress_gradient", test_bsr_compress_gradient, devices=devices)

add_function_test(TestSparse, "test_csr_transpose", make_test_bsr_transpose((1, 1), wp.float32), devices=devices)
add_function_test(TestSparse, "test_bsr_transpose_1_3", make_test_bsr_transpose((1, 3), wp.float32), devices=devices)
add_function_test(TestSparse, "test_bsr_transpose_3_3", make_test_bsr_transpose((3, 3), wp.float64), devices=devices)

add_function_test(TestSparse, "test_csr_axpy", make_test_bsr_axpy((1, 1), wp.float32), devices=devices)
add_function_test(TestSparse, "test_bsr_axpy_1_3", make_test_bsr_axpy((1, 3), wp.float32), devices=devices)
add_function_test(TestSparse, "test_bsr_axpy_3_3", make_test_bsr_axpy((3, 3), wp.float64), devices=devices)

add_function_test(TestSparse, "test_csr_mm", make_test_bsr_mm((1, 1), wp.float32), devices=devices)
add_function_test(TestSparse, "test_bsr_mm_1_3", make_test_bsr_mm((1, 3), wp.float32), devices=devices)
add_function_test(TestSparse, "test_bsr_mm_3_3", make_test_bsr_mm((3, 3), wp.float64), devices=devices)

add_function_test(
    TestSparse, "test_bsr_multiply_deep_2_2", make_test_bsr_multiply_deep((2, 2), wp.float64), devices=devices
)
add_function_test(
    TestSparse,
    "test_bsr_multiply_deep_30_30",
    make_test_bsr_multiply_deep((30, 30), wp.float32),
    devices=cuda_test_devices,
)

add_function_test(TestSparse, "test_csr_mv", make_test_bsr_mv((1, 1), wp.float32), devices=devices)
add_function_test(TestSparse, "test_bsr_mv_1_3", make_test_bsr_mv((1, 3), wp.float32), devices=devices)
add_function_test(TestSparse, "test_bsr_mv_3_3", make_test_bsr_mv((3, 3), wp.float64), devices=devices)

add_function_test(TestSparse, "test_capturability", test_capturability, devices=cuda_test_devices_with_mempool)
add_function_test(
    TestSparse,
    "test_bsr_compress_compact_capturability",
    test_bsr_compress_compact_capturability,
    devices=cuda_test_devices_with_mempool,
)
add_function_test(
    TestSparse,
    "test_padded_bsr_capture_constructs_matrix",
    test_padded_bsr_capture_constructs_matrix,
    devices=cuda_test_devices_with_mempool,
)
add_function_test(
    TestSparse,
    "test_padded_bsr_capture_triplets_and_overflow",
    test_padded_bsr_capture_triplets_and_overflow,
    devices=cuda_test_devices_with_mempool,
)
add_function_test(
    TestSparse,
    "test_padded_bsr_capture_reblock_copy",
    test_padded_bsr_capture_reblock_copy,
    devices=cuda_test_devices_with_mempool,
)
add_function_test(
    TestSparse,
    "test_padded_bsr_status_sync_cuda_capture_rejected",
    test_padded_bsr_status_sync_cuda_capture_rejected,
    devices=cuda_test_devices_with_mempool,
)
add_function_test(
    TestSparse,
    "test_padded_bsr_capture_per_row_without_nnz_capacity_rejected",
    test_padded_bsr_capture_per_row_without_nnz_capacity_rejected,
    devices=cuda_test_devices_with_mempool,
)
add_function_test(TestSparse, "test_bsr_mm_max_new_nnz", test_bsr_mm_max_new_nnz, devices=devices, check_output=False)

add_function_test(TestSparse, "test_bsr_alloc", test_bsr_alloc, devices=devices)

if __name__ == "__main__":
    unittest.main(verbosity=2)
