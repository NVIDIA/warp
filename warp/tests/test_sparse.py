# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import unittest

import numpy as np

import warp as wp
from warp.sparse import (
    bsr_axpy,
    bsr_axpy_work_arrays,
    bsr_copy,
    bsr_diag,
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
    assert np.all(diag_csr.values.numpy() == np.ones(nrow, dtype=float))


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

    with test.assertRaisesRegex(
        ValueError, r"Dest block shape \(5, 5\) is not an exact multiple of src block shape \(4, 2\)"
    ):
        bsr_copy(bsr, block_shape=(5, 5))

    with test.assertRaisesRegex(
        ValueError,
        "The total rows and columns of the src matrix cannot be evenly divided using the requested block shape",
    ):
        bsr_copy(bsr, block_shape=(32, 32))


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
        with test.assertRaisesRegex(ValueError, "Number of columns"):
            bsr_mv(A, x, y)

        A.ncol = A.ncol - 1
        A.nrow = A.nrow - 1
        with test.assertRaisesRegex(ValueError, "Number of rows"):
            bsr_mv(A, x, y)

    return test_bsr_mv


devices = get_test_devices()


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
add_function_test(TestSparse, "test_bsr_get_diag", test_bsr_get_set_diag, devices=devices)
add_function_test(TestSparse, "test_bsr_split_merge", test_bsr_split_merge, devices=devices)

add_function_test(TestSparse, "test_csr_transpose", make_test_bsr_transpose((1, 1), wp.float32), devices=devices)
add_function_test(TestSparse, "test_bsr_transpose_1_3", make_test_bsr_transpose((1, 3), wp.float32), devices=devices)
add_function_test(TestSparse, "test_bsr_transpose_3_3", make_test_bsr_transpose((3, 3), wp.float64), devices=devices)

add_function_test(TestSparse, "test_csr_axpy", make_test_bsr_axpy((1, 1), wp.float32), devices=devices)
add_function_test(TestSparse, "test_bsr_axpy_1_3", make_test_bsr_axpy((1, 3), wp.float32), devices=devices)
add_function_test(TestSparse, "test_bsr_axpy_3_3", make_test_bsr_axpy((3, 3), wp.float64), devices=devices)

add_function_test(TestSparse, "test_csr_mm", make_test_bsr_mm((1, 1), wp.float32), devices=devices)
add_function_test(TestSparse, "test_bsr_mm_1_3", make_test_bsr_mm((1, 3), wp.float32), devices=devices)
add_function_test(TestSparse, "test_bsr_mm_3_3", make_test_bsr_mm((3, 3), wp.float64), devices=devices)

add_function_test(TestSparse, "test_csr_mv", make_test_bsr_mv((1, 1), wp.float32), devices=devices)
add_function_test(TestSparse, "test_bsr_mv_1_3", make_test_bsr_mv((1, 3), wp.float32), devices=devices)
add_function_test(TestSparse, "test_bsr_mv_3_3", make_test_bsr_mv((3, 3), wp.float64), devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
