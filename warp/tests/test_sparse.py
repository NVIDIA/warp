import numpy as np
import warp as wp

from warp.sparse import bsr_zeros, bsr_set_from_triplets, bsr_get_diag, bsr_diag, bsr_set_transpose, bsr_axpy, bsr_mm
from warp.tests.test_base import *

wp.init()


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
    shape = (8, 6)
    n = 100

    rows = wp.array(np.random.randint(0, shape[0], n, dtype=int), dtype=int, device=device)
    cols = wp.array(np.random.randint(0, shape[1], n, dtype=int), dtype=int, device=device)
    vals = wp.array(np.random.rand(n), dtype=float, device=device)

    ref = _triplets_to_dense(shape, rows, cols, vals)

    csr = bsr_zeros(shape[0], shape[1], float, device=device)
    bsr_set_from_triplets(csr, rows, cols, vals)

    res = _bsr_to_dense(csr)

    assert_np_equal(ref, res, 0.0001)


def test_bsr_from_triplets(test, device):
    block_shape = (3, 2)
    nrow = 4
    ncol = 9
    shape = (block_shape[0] * nrow, block_shape[1] * ncol)
    n = 50

    rows = wp.array(np.random.randint(0, nrow, n, dtype=int), dtype=int, device=device)
    cols = wp.array(np.random.randint(0, ncol, n, dtype=int), dtype=int, device=device)
    vals = wp.array(np.random.rand(n, block_shape[0], block_shape[1]), dtype=float, device=device)

    ref = _triplets_to_dense(shape, rows, cols, vals)

    bsr = bsr_zeros(nrow, ncol, wp.types.matrix(shape=block_shape, dtype=float), device=device)
    bsr_set_from_triplets(bsr, rows, cols, vals)

    res = _bsr_to_dense(bsr)

    assert_np_equal(ref, res, 0.0001)


def test_bsr_get_diag(test, device):
    block_shape = (3, 3)
    nrow = 4
    ncol = 4
    nnz = 6

    rows = wp.array([0, 1, 2, 3, 2, 1], dtype=int, device=device)
    cols = wp.array([1, 1, 1, 3, 2, 2], dtype=int, device=device)
    vals_np = np.random.rand(nnz, block_shape[0], block_shape[1])
    vals = wp.array(vals_np, dtype=float, device=device)

    bsr = bsr_zeros(nrow, ncol, wp.types.matrix(shape=block_shape, dtype=float), device=device)
    bsr_set_from_triplets(bsr, rows, cols, vals)

    diag = bsr_get_diag(bsr)
    diag_np = diag.numpy()

    assert_np_equal(diag_np[0], np.zeros(block_shape))
    assert_np_equal(diag_np[1], vals_np[1], tol=0.00001)
    assert_np_equal(diag_np[2], vals_np[4], tol=0.00001)
    assert_np_equal(diag_np[3], vals_np[3], tol=0.00001)

    # Test round-trip
    diag_bsr = bsr_diag(diag)
    diag = bsr_get_diag(diag_bsr)
    assert_np_equal(diag_np, diag.numpy())


def make_test_bsr_transpose(block_shape, scalar_type):
    def test_bsr_transpose(test, device):
        nrow = 4
        ncol = 5
        nnz = 6

        rows = wp.array([0, 1, 2, 3, 2, 1], dtype=int, device=device)
        cols = wp.array([1, 4, 1, 3, 0, 2], dtype=int, device=device)

        vals_np = np.random.rand(nnz, block_shape[0], block_shape[1])
        vals = wp.array(vals_np, dtype=scalar_type, device=device).reshape((nnz, block_shape[0], block_shape[1]))

        bsr = bsr_zeros(nrow, ncol, wp.types.matrix(shape=block_shape, dtype=scalar_type), device=device)
        bsr_set_from_triplets(bsr, rows, cols, vals)
        ref = np.transpose(_bsr_to_dense(bsr))

        bsr_transposed = bsr_zeros(
            ncol, nrow, wp.types.matrix(shape=block_shape[::-1], dtype=scalar_type), device=device
        )
        bsr_set_transpose(dest=bsr_transposed, src=bsr)

        res = _bsr_to_dense(bsr_transposed)

        assert_np_equal(ref, res, 0.0001)

    return test_bsr_transpose


def make_test_bsr_axpy(block_shape, scalar_type):
    def test_bsr_axpy(test, device):
        nrow = 2
        ncol = 3
        nnz = 6

        alpha = -1.0
        beta = 2.0

        x_rows = wp.array(np.random.randint(0, nrow, nnz, dtype=int), dtype=int, device=device)
        x_cols = wp.array(np.random.randint(0, ncol, nnz, dtype=int), dtype=int, device=device)
        x_vals = wp.array(np.random.rand(nnz, block_shape[0], block_shape[1]), dtype=scalar_type, device=device)
        x_vals = x_vals.reshape((nnz, block_shape[0], block_shape[1]))

        x = bsr_zeros(nrow, ncol, wp.types.matrix(shape=block_shape, dtype=scalar_type), device=device)
        bsr_set_from_triplets(x, x_rows, x_cols, x_vals)

        y_rows = wp.array(np.random.randint(0, nrow, nnz, dtype=int), dtype=int, device=device)
        y_cols = wp.array(np.random.randint(0, ncol, nnz, dtype=int), dtype=int, device=device)
        y_vals = wp.array(np.random.rand(nnz, block_shape[0], block_shape[1]), dtype=scalar_type, device=device)
        y_vals = y_vals.reshape((nnz, block_shape[0], block_shape[1]))

        y = bsr_zeros(nrow, ncol, wp.types.matrix(shape=block_shape, dtype=scalar_type), device=device)
        bsr_set_from_triplets(y, y_rows, y_cols, y_vals)

        ref = alpha * _bsr_to_dense(x) + beta * _bsr_to_dense(y)

        bsr_axpy(x, y, alpha, beta)

        res = _bsr_to_dense(y)
        assert_np_equal(ref, res, 0.0001)

    return test_bsr_axpy


def make_test_bsr_mm(block_shape, scalar_type):
    def test_bsr_mm(test, device):
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

        alpha = -1.0
        beta = 2.0

        x_rows = wp.array(np.random.randint(0, x_nrow, nnz, dtype=int), dtype=int, device=device)
        x_cols = wp.array(np.random.randint(0, x_ncol, nnz, dtype=int), dtype=int, device=device)
        x_vals = wp.array(np.random.rand(nnz, x_block_shape[0], x_block_shape[1]), dtype=scalar_type, device=device)
        x_vals = x_vals.reshape((nnz, x_block_shape[0], x_block_shape[1]))

        x = bsr_zeros(x_nrow, x_ncol, wp.types.matrix(shape=x_block_shape, dtype=scalar_type), device=device)
        bsr_set_from_triplets(x, x_rows, x_cols, x_vals)

        y_rows = wp.array(np.random.randint(0, y_nrow, nnz, dtype=int), dtype=int, device=device)
        y_cols = wp.array(np.random.randint(0, y_ncol, nnz, dtype=int), dtype=int, device=device)
        y_vals = wp.array(np.random.rand(nnz, y_block_shape[0], y_block_shape[1]), dtype=scalar_type, device=device)
        y_vals = y_vals.reshape((nnz, y_block_shape[0], y_block_shape[1]))

        y = bsr_zeros(y_nrow, y_ncol, wp.types.matrix(shape=y_block_shape, dtype=scalar_type), device=device)
        bsr_set_from_triplets(y, y_rows, y_cols, y_vals)

        z_rows = wp.array(np.random.randint(0, z_nrow, nnz, dtype=int), dtype=int, device=device)
        z_cols = wp.array(np.random.randint(0, z_ncol, nnz, dtype=int), dtype=int, device=device)
        z_vals = wp.array(np.random.rand(nnz, z_block_shape[0], z_block_shape[1]), dtype=scalar_type, device=device)
        z_vals = z_vals.reshape((nnz, z_block_shape[0], z_block_shape[1]))

        z = bsr_zeros(z_nrow, z_ncol, wp.types.matrix(shape=z_block_shape, dtype=scalar_type), device=device)
        bsr_set_from_triplets(z, z_rows, z_cols, z_vals)

        ref = alpha * (_bsr_to_dense(x) @ _bsr_to_dense(y)) + beta * _bsr_to_dense(z)

        bsr_mm(x, y, z, alpha, beta)

        res = _bsr_to_dense(z)
        assert_np_equal(ref, res, 0.0001)

    return test_bsr_mm


def register(parent):
    devices = get_test_devices()

    class TestSparse(parent):
        pass

    add_function_test(TestSparse, "test_csr_from_triplets", test_csr_from_triplets, devices=devices)
    add_function_test(TestSparse, "test_bsr_from_triplets", test_bsr_from_triplets, devices=devices)
    add_function_test(TestSparse, "test_bsr_get_diag", test_bsr_get_diag, devices=devices)

    add_function_test(TestSparse, "test_csr_transpose", make_test_bsr_transpose((1, 1), wp.float32), devices=devices)
    add_function_test(
        TestSparse, "test_bsr_transpose_1_3", make_test_bsr_transpose((1, 3), wp.float32), devices=devices
    )
    add_function_test(
        TestSparse, "test_bsr_transpose_3_3", make_test_bsr_transpose((3, 3), wp.float64), devices=devices
    )

    add_function_test(TestSparse, "test_csr_axpy", make_test_bsr_axpy((1, 1), wp.float32), devices=devices)
    add_function_test(TestSparse, "test_bsr_axpy_1_3", make_test_bsr_axpy((1, 3), wp.float32), devices=devices)
    add_function_test(TestSparse, "test_bsr_axpy_3_3", make_test_bsr_axpy((3, 3), wp.float64), devices=devices)

    add_function_test(TestSparse, "test_csr_mm", make_test_bsr_mm((1, 1), wp.float32), devices=devices)
    add_function_test(TestSparse, "test_bsr_mm_1_3", make_test_bsr_mm((1, 3), wp.float32), devices=devices)
    add_function_test(TestSparse, "test_bsr_mm_3_3", make_test_bsr_mm((3, 3), wp.float64), devices=devices)

    return TestSparse


if __name__ == "__main__":
    c = register(unittest.TestCase)
    unittest.main(verbosity=2)
