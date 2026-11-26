# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from typing import Any

import numpy as np

import warp as wp
import warp.fem as fem
from warp.fem.linalg import householder_qr_decomposition, inverse_qr, symmetric_eigenvalues_qr
from warp.tests.fem.utils import vec6f
from warp.tests.unittest_utils import *

mat55f = wp.types.matrix(shape=(5, 5), dtype=wp.float32)
mat88f = wp.types.matrix(shape=(8, 8), dtype=wp.float32)
mat99d = wp.types.matrix(shape=(9, 9), dtype=wp.float64)
mat33bf = wp.types.matrix(shape=(3, 3), dtype=wp.bfloat16)


@wp.kernel(enable_backward=False)
def test_qr_eigenvalues():
    tol = 5.0e-7

    # zero
    Zero = wp.mat33(0.0)
    Id = wp.identity(n=3, dtype=float)
    D3, P3 = symmetric_eigenvalues_qr(Zero, tol * tol)
    wp.expect_eq(D3, wp.vec3(0.0))
    wp.expect_eq(P3, Id)

    # Identity
    D3, P3 = symmetric_eigenvalues_qr(Id, tol * tol)
    wp.expect_eq(D3, wp.vec3(1.0))
    wp.expect_eq(wp.transpose(P3) * P3, Id)

    # rank 1
    v = wp.vec4(0.0, 1.0, 1.0, 0.0)
    Rank1 = wp.outer(v, v)
    D4, P4 = symmetric_eigenvalues_qr(Rank1, tol * tol)
    wp.expect_near(wp.max(D4), wp.length_sq(v), tol)
    Err4 = wp.transpose(P4) * wp.diag(D4) * P4 - Rank1
    wp.expect_near(wp.ddot(Err4, Err4), 0.0, tol)

    # rank 2
    v2 = wp.vec4(0.0, 0.5, -0.5, 0.0)
    Rank2 = Rank1 + wp.outer(v2, v2)
    D4, P4 = symmetric_eigenvalues_qr(Rank2, tol * tol)
    wp.expect_near(wp.max(D4), wp.length_sq(v), tol)
    wp.expect_near(D4[0] + D4[1] + D4[2] + D4[3], wp.length_sq(v) + wp.length_sq(v2), tol)
    Err4 = wp.transpose(P4) * wp.diag(D4) * P4 - Rank2
    wp.expect_near(wp.ddot(Err4, Err4), 0.0, tol)

    # rank 4
    v3 = wp.vec4(1.0, 2.0, 3.0, 4.0)
    v4 = wp.vec4(2.0, 1.0, 0.0, -1.0)
    Rank4 = Rank2 + wp.outer(v3, v3) + wp.outer(v4, v4)
    D4, P4 = symmetric_eigenvalues_qr(Rank4, tol * tol)
    Err4 = wp.transpose(P4) * wp.diag(D4) * P4 - Rank4
    wp.expect_near(wp.ddot(Err4, Err4), 0.0, tol)

    # test robustness to low requested tolerance
    Rank6 = wp.matrix_from_cols(
        vec6f(0.00171076, 0.0, 0.0, 0.0, 0.0, 0.0),
        vec6f(0.0, 0.00169935, 6.14367e-06, -3.52589e-05, 3.02397e-05, -1.53458e-11),
        vec6f(0.0, 6.14368e-06, 0.00172217, 2.03568e-05, 1.74589e-05, -2.92627e-05),
        vec6f(0.0, -3.52589e-05, 2.03568e-05, 0.00172178, 2.53422e-05, 3.02397e-05),
        vec6f(0.0, 3.02397e-05, 1.74589e-05, 2.53422e-05, 0.00171114, 3.52589e-05),
        vec6f(0.0, 6.42993e-12, -2.92627e-05, 3.02397e-05, 3.52589e-05, 0.00169935),
    )
    D6, P6 = symmetric_eigenvalues_qr(Rank6, 0.0)
    Err6 = wp.transpose(P6) * wp.diag(D6) * P6 - Rank6
    wp.expect_near(wp.ddot(Err6, Err6), 0.0, 1.0e-13)


@wp.kernel(enable_backward=False)
def test_qr_inverse():
    rng = wp.rand_init(4356, wp.tid())
    M = wp.mat33(
        wp.randf(rng, 0.0, 10.0),
        wp.randf(rng, 0.0, 10.0),
        wp.randf(rng, 0.0, 10.0),
        wp.randf(rng, 0.0, 10.0),
        wp.randf(rng, 0.0, 10.0),
        wp.randf(rng, 0.0, 10.0),
        wp.randf(rng, 0.0, 10.0),
        wp.randf(rng, 0.0, 10.0),
        wp.randf(rng, 0.0, 10.0),
    )

    if wp.determinant(M) != 0.0:
        tol = 1.0e-8
        Mi = inverse_qr(M)
        Id = wp.identity(n=3, dtype=float)
        Err = M * Mi - Id
        wp.expect_near(wp.ddot(Err, Err), 0.0, tol)
        Err = Mi * M - Id
        wp.expect_near(wp.ddot(Err, Err), 0.0, tol)


@wp.kernel(enable_backward=False)
def test_qr_small_scales():
    base_h = wp.mat33h(3.0, 0.4, -0.2, 0.4, 2.0, 0.3, -0.2, 0.3, 1.5)
    identity_h = wp.identity(n=3, dtype=wp.float16)
    small_scale_h = wp.float16(1.0e-4)
    small_h = small_scale_h * base_h
    small_inverse_h = inverse_qr(small_h)
    inverse_error_h = small_h * small_inverse_h - identity_h
    wp.expect_near(wp.ddot(inverse_error_h, inverse_error_h), wp.float16(0.0), wp.float16(1.0e-2))

    eigenvalues_h, eigenvectors_h = symmetric_eigenvalues_qr(small_h, wp.float16(0.0))
    reconstruction_h = wp.transpose(eigenvectors_h) * wp.diag(eigenvalues_h) * eigenvectors_h / small_scale_h
    reconstruction_error_h = reconstruction_h - base_h
    wp.expect_near(wp.ddot(reconstruction_error_h, reconstruction_error_h), wp.float16(0.0), wp.float16(1.0e-2))

    base = wp.mat33(3.0, 0.4, -0.2, 0.4, 2.0, 0.3, -0.2, 0.3, 1.5)
    identity = wp.identity(n=3, dtype=float)
    tol = 1.0e-10

    small_scale = 1.0e-22
    small = small_scale * base
    small_inverse = inverse_qr(small)
    inverse_error = small * small_inverse - identity
    wp.expect_near(wp.ddot(inverse_error, inverse_error), 0.0, tol)

    eigenvalues, eigenvectors = symmetric_eigenvalues_qr(small, 0.0)
    reconstruction = wp.transpose(eigenvectors) * wp.diag(eigenvalues) * eigenvectors / small_scale
    reconstruction_error = reconstruction - base
    wp.expect_near(wp.ddot(reconstruction_error, reconstruction_error), 0.0, tol)

    base_d = wp.mat33d(3.0, 0.4, -0.2, 0.4, 2.0, 0.3, -0.2, 0.3, 1.5)
    identity_d = wp.identity(n=3, dtype=wp.float64)
    small_scale_d = wp.float64(1.0e-180)
    small_d = small_scale_d * base_d
    small_inverse_d = inverse_qr(small_d)
    inverse_error_d = small_d * small_inverse_d - identity_d
    wp.expect_near(wp.ddot(inverse_error_d, inverse_error_d), 0.0, 1.0e-24)

    eigenvalues_d, eigenvectors_d = symmetric_eigenvalues_qr(small_d, wp.float64(0.0))
    reconstruction_d = wp.transpose(eigenvectors_d) * wp.diag(eigenvalues_d) * eigenvectors_d / small_scale_d
    reconstruction_error_d = reconstruction_d - base_d
    wp.expect_near(wp.ddot(reconstruction_error_d, reconstruction_error_d), 0.0, 1.0e-24)

    # Routine-wide scaling cannot protect a tiny subblock in an otherwise order-one matrix.
    tiny = 1.0e-25
    graded = wp.mat33(1.0, 0.0, 0.0, 0.0, 0.0, tiny, 0.0, tiny, 0.0)
    graded_inverse = inverse_qr(graded)
    inverse_error = graded * graded_inverse - identity
    wp.expect_near(wp.ddot(inverse_error, inverse_error), 0.0, tol)

    eigenvalues, eigenvectors = symmetric_eigenvalues_qr(graded, 0.0)
    reconstruction = wp.transpose(eigenvectors) * wp.diag(eigenvalues) * eigenvectors
    wp.expect_near(reconstruction[1, 2] / tiny, 1.0, 1.0e-5)
    wp.expect_near(reconstruction[2, 1] / tiny, 1.0, 1.0e-5)


@wp.kernel(enable_backward=False)
def test_qr_eigenvalues_small_coupled_to_large():
    # Couplings below eps times the larger diagonal term still determine the small eigenvalue -c^2/a
    A = wp.mat22(1.0e10, 1.0e3, 1.0e3, 0.0)
    eigenvalues, eigenvectors = symmetric_eigenvalues_qr(A, 0.0)
    wp.expect_near(wp.min(eigenvalues) / -1.0e-4, 1.0, 1.0e-4)
    reconstruction = wp.transpose(eigenvectors) * wp.diag(eigenvalues) * eigenvectors
    wp.expect_near(reconstruction[0, 1] / 1.0e3, 1.0, 1.0e-4)

    A_d = wp.mat22d(1.0e20, 1.0e3, 1.0e3, 0.0)
    eigenvalues_d, eigenvectors_d = symmetric_eigenvalues_qr(A_d, wp.float64(0.0))
    wp.expect_near(wp.min(eigenvalues_d) / wp.float64(-1.0e-14), wp.float64(1.0), wp.float64(1.0e-10))
    reconstruction_d = wp.transpose(eigenvectors_d) * wp.diag(eigenvalues_d) * eigenvectors_d
    wp.expect_near(reconstruction_d[0, 1] / wp.float64(1.0e3), wp.float64(1.0), wp.float64(1.0e-10))


def test_qr_bfloat16_small_scales(test, device):
    @wp.kernel(module="unique", enable_backward=False)
    def check():
        base = mat33bf(
            wp.bfloat16(3.0),
            wp.bfloat16(0.4),
            wp.bfloat16(-0.2),
            wp.bfloat16(0.4),
            wp.bfloat16(2.0),
            wp.bfloat16(0.3),
            wp.bfloat16(-0.2),
            wp.bfloat16(0.3),
            wp.bfloat16(1.5),
        )
        identity = wp.identity(n=3, dtype=wp.bfloat16)
        small_scale = wp.bfloat16(1.0e-22)
        small = small_scale * base
        small_inverse = inverse_qr(small)
        inverse_error = small * small_inverse - identity
        wp.expect_near(wp.float32(wp.ddot(inverse_error, inverse_error)), 0.0, 1.0e-1)

        eigenvalues, eigenvectors = symmetric_eigenvalues_qr(small, wp.bfloat16(0.0))
        reconstruction = wp.transpose(eigenvectors) * wp.diag(eigenvalues) * eigenvectors / small_scale
        reconstruction_error = reconstruction - base
        wp.expect_near(wp.float32(wp.ddot(reconstruction_error, reconstruction_error)), 0.0, 1.0e-1)

    wp.launch(check, dim=1, device=device)


@wp.kernel(module="unique", enable_backward=False)
def compute_qr_large_scales(
    matrices: wp.array[Any],
    q: wp.array[Any],
    r: wp.array[Any],
    inverses: wp.array[Any],
    eigenvalues: wp.array[Any],
    eigenvectors: wp.array[Any],
):
    i = wp.tid()
    matrix = matrices[i]
    matrix_q, matrix_r = householder_qr_decomposition(matrix)
    q[i] = matrix_q
    r[i] = matrix_r
    inverses[i] = inverse_qr(matrix)
    d, p = symmetric_eigenvalues_qr(matrix, matrix.dtype(0.0))
    eigenvalues[i] = d
    eigenvectors[i] = p


def test_qr_large_scales(test, device):
    cases = [
        (wp.float16, (32.0, 64.0, 128.0, 1.0e3), 2.0e-2),
        (wp.float32, (1.0e18, 4.0e18, 8.0e18, 1.0e19, 1.0e30), 2.0e-6),
        (wp.float64, (1.0e153, 4.0e153, 8.0e153, 1.0e154, 1.0e200), 2.0e-14),
    ]
    if device.is_cpu or device.arch >= 80:
        cases.append((wp.bfloat16, (1.0e18, 4.0e18, 8.0e18, 1.0e19, 1.0e30), 8.0e-2))

    for dtype, scales, tolerance in cases:
        for n in (2, 3):
            matrix_type = wp.types.matrix(shape=(n, n), dtype=dtype)
            vector_type = wp.types.vector(length=n, dtype=dtype)
            # The 2x2 matrix matches GH-2008; the 3x3 matrix also exercises Hessenberg reduction.
            base = np.ones((n, n)) + np.eye(n)
            for scale in scales:
                with test.subTest(dtype=dtype, n=n, scale=scale):
                    matrices = wp.array([scale * base], dtype=matrix_type, device=device)
                    q = wp.empty_like(matrices)
                    r = wp.empty_like(matrices)
                    inverses = wp.empty_like(matrices)
                    eigenvalues = wp.empty(1, dtype=vector_type, device=device)
                    eigenvectors = wp.empty_like(matrices)
                    wp.launch(
                        compute_qr_large_scales,
                        dim=1,
                        inputs=[matrices, q, r, inverses, eigenvalues, eigenvectors],
                        device=device,
                    )
                    q, r, inverse, d, p = [
                        output.numpy().astype(np.float64)[0] for output in (q, r, inverses, eigenvalues, eigenvectors)
                    ]
                    for output in (q, r, inverse, d, p):
                        test.assertTrue(np.isfinite(output).all())
                    # Normalize before multiplying so the reference checks cannot overflow either.
                    a = matrices.numpy().astype(np.float64)[0] / scale
                    np.testing.assert_allclose(q @ (r / scale), a, rtol=tolerance, atol=tolerance)
                    np.testing.assert_allclose(q.T @ q, np.eye(n), rtol=tolerance, atol=tolerance)
                    np.testing.assert_allclose(np.tril(r / scale, -1), 0.0, atol=tolerance)
                    np.testing.assert_allclose(a @ (inverse * scale), np.eye(n), atol=tolerance)
                    np.testing.assert_allclose(np.sort(d / scale), np.linalg.eigvalsh(a), rtol=tolerance)
                    np.testing.assert_allclose(p.T @ np.diag(d / scale) @ p, a, rtol=tolerance, atol=tolerance)
                    np.testing.assert_allclose(p.T @ p, np.eye(n), rtol=tolerance, atol=tolerance)


@wp.kernel(enable_backward=False)
def reconstruct_qr_eigenvalues_zero_tolerance(matrices: wp.array[Any], reconstructions: wp.array[Any]):
    i = wp.tid()
    matrix = matrices[i]
    eigenvalues, eigenvectors = symmetric_eigenvalues_qr(matrix, matrix.dtype(0.0))
    reconstructions[i] = wp.transpose(eigenvectors) * wp.diag(eigenvalues) * eigenvectors


def _zero_tolerance_reconstruction_errors(matrices, matrix_type, device):
    matrices_device = wp.array(matrices, dtype=matrix_type, device=device)
    reconstructions = wp.empty_like(matrices_device)
    wp.launch(
        reconstruct_qr_eigenvalues_zero_tolerance,
        dim=matrices.shape[0],
        inputs=[matrices_device, reconstructions],
        device=device,
    )
    matrices = matrices.astype(np.float64)
    errors = reconstructions.numpy().astype(np.float64) - matrices
    return np.linalg.norm(errors, axis=(1, 2)) / np.linalg.norm(matrices, axis=(1, 2))


def _tridiagonal(diagonals, off_diagonals):
    n = diagonals.shape[-1]
    matrices = np.zeros((*diagonals.shape, n), dtype=diagonals.dtype)
    idx = np.arange(n)
    matrices[..., idx, idx] = diagonals
    matrices[..., idx[1:], idx[:-1]] = off_diagonals
    matrices[..., idx[:-1], idx[1:]] = off_diagonals
    return matrices


def test_qr_eigenvalues_zero_tolerance(test, device):
    rng = np.random.default_rng(20260922)
    matrices = rng.standard_normal((40, 9, 9))
    matrices += np.swapaxes(matrices, 1, 2)

    errors = _zero_tolerance_reconstruction_errors(matrices, mat99d, device)
    test.assertLess(np.max(errors), 1.0e-12)


def test_qr_eigenvalues_zero_tolerance_graded(test, device):
    # Graded couplings between exactly-zero diagonal terms, which stall deflation relative to the diagonal alone
    diagonals = np.array([[0.0, 2.5e-15, 0.0, 0.8, 0.0], [0.0, 2.0e-15, 0.0, 0.8, 0.0]], dtype=np.float32)
    off_diagonals = np.array(
        [[-2.0e-19, -0.75, -4.0e-9, 1.0e-16], [-2.0e-19, -0.75, -3.0e-9, 1.1e-16]], dtype=np.float32
    )
    errors = _zero_tolerance_reconstruction_errors(_tridiagonal(diagonals, off_diagonals), mat55f, device)
    test.assertLess(np.max(errors), 1.0e-5)

    if device.is_cuda:
        # Broader random coverage spanning 20 decades, kept off CPU to limit debug build test time
        rng = np.random.default_rng(1)
        count, n = 20000, 8
        diagonals = rng.choice([-1.0, 1.0], (count, n)) * 10.0 ** rng.uniform(-20.0, 0.0, (count, n))
        off_diagonals = rng.choice([-1.0, 1.0], (count, n - 1)) * 10.0 ** rng.uniform(-20.0, 0.0, (count, n - 1))
        diagonals[rng.random((count, n)) < 0.2] = 0.0
        off_diagonals[rng.random((count, n - 1)) < 0.1] = 0.0

        matrices = _tridiagonal(diagonals, off_diagonals).astype(np.float32)
        errors = _zero_tolerance_reconstruction_errors(matrices, mat88f, device)
        test.assertLess(np.max(errors), 1.0e-5)


def test_qr_eigenvalues_zero_tolerance_small_scale(test, device):
    # Rotations within blocks of small magnitude underflow unless the matrix is normalized first
    diagonals = np.array([[0.0, -1.5048e-26, -1.9483e-23, -2.3335e-22, -4.6156e-24, 0.0, 0.0, 0.0]], dtype=np.float32)
    off_diagonals = np.array(
        [[1.6187e-29, -8.7357e-24, 3.2934e-26, -2.7489e-21, 1.4257e-30, -1.9659e-27, 4.1052e-22]], dtype=np.float32
    )
    errors = _zero_tolerance_reconstruction_errors(_tridiagonal(diagonals, off_diagonals), mat88f, device)
    test.assertLess(np.max(errors), 1.0e-5)

    matrices = np.array(
        [[[2.1613e-4, 1.1754e-4, 5.8472e-5], [1.1754e-4, -1.5986e-4, -9.9778e-5], [5.8472e-5, -9.9778e-5, -6.5744e-5]]],
        dtype=np.float16,
    )
    errors = _zero_tolerance_reconstruction_errors(matrices, wp.mat33h, device)
    test.assertLess(np.max(errors), 1.0e-2)


def test_array_axpy(test, device):
    N = 10
    alpha = 0.5
    beta = 4.0

    x = wp.full(N, 2.0, device=device, dtype=float, requires_grad=True)
    y = wp.array(np.arange(N), device=device, dtype=wp.float64, requires_grad=True)

    tape = wp.Tape()
    with tape:
        fem.linalg.array_axpy(x=x, y=y, alpha=alpha, beta=beta)

    assert_np_equal(x.numpy(), np.full(N, 2.0))
    assert_np_equal(y.numpy(), alpha * x.numpy() + beta * np.arange(N))

    y.grad.fill_(1.0)
    tape.backward()

    assert_np_equal(x.grad.numpy(), alpha * np.ones(N))
    assert_np_equal(y.grad.numpy(), beta * np.ones(N))


devices = get_test_devices()
bfloat16_devices = [device for device in devices if device.is_cpu or device.arch >= 80]


class TestFemLinalg(unittest.TestCase):
    pass


add_kernel_test(TestFemLinalg, test_qr_eigenvalues, dim=1, devices=devices)
add_kernel_test(TestFemLinalg, test_qr_inverse, dim=100, devices=devices)
add_kernel_test(TestFemLinalg, test_qr_small_scales, dim=1, devices=devices)
add_function_test(TestFemLinalg, "test_qr_large_scales", test_qr_large_scales, devices=devices)
add_kernel_test(TestFemLinalg, test_qr_eigenvalues_small_coupled_to_large, dim=1, devices=devices)
add_function_test(
    TestFemLinalg, "test_qr_bfloat16_small_scales", test_qr_bfloat16_small_scales, devices=bfloat16_devices
)
add_function_test(
    TestFemLinalg, "test_qr_eigenvalues_zero_tolerance", test_qr_eigenvalues_zero_tolerance, devices=devices
)
add_function_test(
    TestFemLinalg,
    "test_qr_eigenvalues_zero_tolerance_graded",
    test_qr_eigenvalues_zero_tolerance_graded,
    devices=devices,
)
add_function_test(
    TestFemLinalg,
    "test_qr_eigenvalues_zero_tolerance_small_scale",
    test_qr_eigenvalues_zero_tolerance_small_scale,
    devices=devices,
)
add_function_test(TestFemLinalg, "test_array_axpy", test_array_axpy)

if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
