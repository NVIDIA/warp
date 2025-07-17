# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np

import warp as wp
from warp.sparse import (
    bsr_assign,
    bsr_axpy,
    bsr_axpy_work_arrays,
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

    for row, col, val in zip(rows, cols, values):
        mat_block = _get_block(mat, row, col, block_shape)
        mat_block += val

    return mat


def _bsr_to_dense(bsr):
    mat = np.zeros(bsr.shape)

    offsets = bsr.offsets.numpy()
    columns = bsr.columns.numpy()
    values = bsr.values.numpy()

    for row in range(bsr.nrow):
        beg = offsets[row]
        end = offsets[row + 1]

        for block in range(beg, end):
            mat_block = _get_block(mat, row, columns[block], bsr.block_shape)
            mat_block += values[block]

    return mat


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
        diag_bsr = bsr_diag(diag=vals_np[0, 0], rows_of_blocks=nrow, cols_of_blocks=nrow + 1)

    diag_bsr = bsr_diag(diag=vals_np[0], rows_of_blocks=nrow, cols_of_blocks=nrow + 1)
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
    shape = (block_shape[0] * nrow, block_shape[1] * ncol)
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
    bsr_assign(src=A, dest=diag_masked, masked=True)
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

        bsr_transposed = (2.0 * bsr).transpose()

        res = _bsr_to_dense(bsr_transposed.eval())
        assert_np_equal(res, ref, 0.0001)

        if block_shape[0] != block_shape[-1]:
            # test incompatible block shape
            with test.assertRaisesRegex(ValueError, "Destination block shape must be"):
                bsr_set_transpose(dest=bsr, src=bsr)

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
        for alpha, beta in zip(alphas, betas):
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
        bsr_axpy(y, y_mask, masked=True)
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
        for alpha, beta in zip(alphas, betas):
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
        bsr_mm(x, y, z, masked=True)
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
                dtype=wp.vec(length=block_shape[1], dtype=scalar_type),
                device=device,
            )

        if block_shape[0] == 1:
            y = wp.array(rng.random(size=nrow), dtype=scalar_type, device=device)
        else:
            y = wp.array(
                rng.random(size=(nrow, block_shape[0])),
                dtype=wp.vec(length=block_shape[0], dtype=scalar_type),
                device=device,
            )

        work_buffer = wp.empty_like(y)
        for alpha, beta in zip(alphas, betas):
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


devices = get_test_devices()
cuda_test_devices = get_selected_cuda_test_devices()


class TestSparse(unittest.TestCase):
    def test_bsr_copy_scale(self):
        nrow = 6
        bsize = 2

        diag_bsr = bsr_diag(diag=np.eye(bsize, dtype=float) * 2.0, rows_of_blocks=nrow)
        diag_copy = bsr_copy(diag_bsr, scalar_type=wp.float64)

        self.assertTrue(wp.types.types_equal(diag_copy.values.dtype, wp.mat(shape=(bsize, bsize), dtype=wp.float64)))
        bsr_scale(x=diag_copy, alpha=0.5)

        res = _bsr_to_dense(diag_copy)
        ref = np.eye(nrow * bsize)
        assert_np_equal(res, ref, 0.0001)

        bsr_scale(x=diag_copy, alpha=0.0)
        self.assertEqual(diag_copy.nrow, nrow)
        self.assertEqual(diag_copy.ncol, nrow)
        self.assertEqual(diag_copy.nnz, 0)


add_function_test(TestSparse, "test_csr_from_triplets", test_csr_from_triplets, devices=devices)
add_function_test(TestSparse, "test_bsr_from_triplets", test_bsr_from_triplets, devices=devices)
add_function_test(
    TestSparse,
    "test_bsr_from_triplets_prune_numerical_zeros",
    test_bsr_from_triplets_prune_numerical_zeros,
    devices=devices,
)
add_function_test(TestSparse, "test_bsr_get_diag", test_bsr_get_set_diag, devices=devices)
add_function_test(TestSparse, "test_bsr_split_merge", test_bsr_split_merge, devices=devices)
add_function_test(TestSparse, "test_bsr_assign_masked", test_bsr_assign_masked, devices=devices)
add_function_test(TestSparse, "test_bsr_from_triplets_gradient", test_bsr_from_triplets_gradient, devices=devices)

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


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
