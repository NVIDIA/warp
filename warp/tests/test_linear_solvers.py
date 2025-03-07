# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from warp.optim.linear import bicgstab, cg, cr, gmres, preconditioner
from warp.tests.unittest_utils import *


def _check_linear_solve(test, A, b, func, *args, **kwargs):
    # test from zero
    x = wp.zeros_like(b)
    with wp.ScopedDevice(A.device):
        niter, err, atol = func(A, b, x, *args, use_cuda_graph=True, **kwargs)

    test.assertLessEqual(err, atol)

    # test with warm start
    with wp.ScopedDevice(A.device):
        niter_warm, err, atol = func(A, b, x, *args, use_cuda_graph=False, **kwargs)

    test.assertLessEqual(err, atol)

    if func in [cr, gmres]:
        # monotonic convergence
        test.assertLess(niter_warm, niter)

    # In CG and BiCGSTAB residual norm is evaluating from running residual
    # rather then being computed from scratch as Ax - b
    # This can lead to accumulated inaccuracies over iterations, esp in float32
    residual = A.numpy() @ x.numpy() - b.numpy()
    err_np = np.linalg.norm(residual)
    if A.dtype == wp.float64:
        test.assertLessEqual(err_np, 2.0 * atol)
    else:
        test.assertLessEqual(err_np, 32.0 * atol)


def _least_square_system(rng, n: int):
    C = rng.uniform(low=-100, high=100, size=(n, n))
    f = rng.uniform(low=-100, high=100, size=(n,))

    A = C @ C.T
    b = C @ f

    return A, b


def _make_spd_system(n: int, seed: int, dtype, device):
    rng = np.random.default_rng(seed)

    A, b = _least_square_system(rng, n)

    return wp.array(A, dtype=dtype, device=device), wp.array(b, dtype=dtype, device=device)


def _make_nonsymmetric_system(n: int, seed: int, dtype, device):
    rng = np.random.default_rng(seed)
    s = rng.uniform(low=0.1, high=10, size=(n,))

    A, b = _least_square_system(rng, n)
    A = A @ np.diag(s)

    return wp.array(A, dtype=dtype, device=device), wp.array(b, dtype=dtype, device=device)


def _make_indefinite_system(n: int, seed: int, dtype, device):
    rng = np.random.default_rng(seed)
    s = rng.uniform(low=0.1, high=10, size=(n,))

    A, b = _least_square_system(rng, n)
    A = A @ np.diag(s)

    return wp.array(A, dtype=dtype, device=device), wp.array(b, dtype=dtype, device=device)


def _make_identity_system(n: int, seed: int, dtype, device):
    rng = np.random.default_rng(seed)

    A = np.eye(n)
    b = rng.uniform(low=-1.0, high=1.0, size=(n,))

    return wp.array(A, dtype=dtype, device=device), wp.array(b, dtype=dtype, device=device)


def test_cg(test, device):
    A, b = _make_spd_system(n=64, seed=123, device=device, dtype=wp.float64)
    M = preconditioner(A, "diag")

    _check_linear_solve(test, A, b, cg, maxiter=1000)
    _check_linear_solve(test, A, b, cg, M=M, maxiter=1000)

    A, b = _make_spd_system(n=16, seed=321, device=device, dtype=wp.float32)
    M = preconditioner(A, "diag")

    _check_linear_solve(test, A, b, cg, maxiter=1000)
    _check_linear_solve(test, A, b, cg, M=M, maxiter=1000)

    A, b = _make_identity_system(n=5, seed=321, device=device, dtype=wp.float32)
    _check_linear_solve(test, A, b, cg, maxiter=30)


def test_cr(test, device):
    A, b = _make_spd_system(n=64, seed=123, device=device, dtype=wp.float64)
    M = preconditioner(A, "diag")

    _check_linear_solve(test, A, b, cr, maxiter=1000)
    _check_linear_solve(test, A, b, cr, M=M, maxiter=1000)

    A, b = _make_spd_system(n=16, seed=321, device=device, dtype=wp.float32)
    M = preconditioner(A, "diag")

    _check_linear_solve(test, A, b, cr, maxiter=1000)
    _check_linear_solve(test, A, b, cr, M=M, maxiter=1000)

    A, b = _make_identity_system(n=5, seed=321, device=device, dtype=wp.float32)
    _check_linear_solve(test, A, b, cr, maxiter=30)


def test_bicgstab(test, device):
    A, b = _make_nonsymmetric_system(n=64, seed=123, device=device, dtype=wp.float64)
    M = preconditioner(A, "diag")

    _check_linear_solve(test, A, b, bicgstab, maxiter=1000)
    _check_linear_solve(test, A, b, bicgstab, M=M, maxiter=1000)
    _check_linear_solve(test, A, b, bicgstab, M=M, maxiter=1000, is_left_preconditioner=True)

    A, b = _make_nonsymmetric_system(n=16, seed=321, device=device, dtype=wp.float32)
    M = preconditioner(A, "diag")

    _check_linear_solve(test, A, b, bicgstab, maxiter=1000)
    _check_linear_solve(test, A, b, bicgstab, M=M, maxiter=1000)
    _check_linear_solve(test, A, b, bicgstab, M=M, maxiter=1000, is_left_preconditioner=True)

    A, b = _make_indefinite_system(n=64, seed=121, device=device, dtype=wp.float64)
    M = preconditioner(A, "diag")

    _check_linear_solve(test, A, b, bicgstab, maxiter=1000)
    _check_linear_solve(test, A, b, bicgstab, M=M, maxiter=1000)
    _check_linear_solve(test, A, b, bicgstab, M=M, maxiter=1000, is_left_preconditioner=True)

    A, b = _make_identity_system(n=5, seed=321, device=device, dtype=wp.float32)
    _check_linear_solve(test, A, b, bicgstab, maxiter=30)


def test_gmres(test, device):
    A, b = _make_nonsymmetric_system(n=64, seed=456, device=device, dtype=wp.float64)
    M = preconditioner(A, "diag")

    _check_linear_solve(test, A, b, gmres, maxiter=1000, tol=1.0e-3)
    _check_linear_solve(test, A, b, gmres, M=M, maxiter=1000, tol=1.0e-5)
    _check_linear_solve(test, A, b, gmres, M=M, maxiter=1000, tol=1.0e-5, is_left_preconditioner=True)

    A, b = _make_nonsymmetric_system(n=64, seed=654, device=device, dtype=wp.float64)
    M = preconditioner(A, "diag")

    _check_linear_solve(test, A, b, gmres, maxiter=1000, tol=1.0e-3)
    _check_linear_solve(test, A, b, gmres, M=M, maxiter=1000, tol=1.0e-5)
    _check_linear_solve(test, A, b, gmres, M=M, maxiter=1000, tol=1.0e-5, is_left_preconditioner=True)

    A, b = _make_identity_system(n=5, seed=123, device=device, dtype=wp.float32)
    _check_linear_solve(test, A, b, gmres, maxiter=120)


class TestLinearSolvers(unittest.TestCase):
    pass


devices = get_test_devices()

add_function_test(TestLinearSolvers, "test_cg", test_cg, devices=devices)
add_function_test(TestLinearSolvers, "test_cr", test_cr, devices=devices)
add_function_test(TestLinearSolvers, "test_bicgstab", test_bicgstab, devices=devices)
add_function_test(TestLinearSolvers, "test_gmres", test_gmres, devices=devices)

if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
