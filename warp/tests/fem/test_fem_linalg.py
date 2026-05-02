# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import warp as wp
import warp.fem as fem
from warp.fem.linalg import inverse_qr, symmetric_eigenvalues_qr
from warp.tests.fem.utils import vec6f
from warp.tests.unittest_utils import *


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


class TestFemLinalg(unittest.TestCase):
    pass


add_kernel_test(TestFemLinalg, test_qr_eigenvalues, dim=1, devices=devices)
add_kernel_test(TestFemLinalg, test_qr_inverse, dim=100, devices=devices)
add_function_test(TestFemLinalg, "test_array_axpy", test_array_axpy)

if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
