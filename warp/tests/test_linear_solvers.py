# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools
import unittest

import numpy as np

import warp as wp
from warp._src.optim.linear import _run_solver_loop
from warp.optim.linear import CG, CR, GMRES, BiCGSTAB, aslinearoperator, bicgstab, cg, cr, gmres, preconditioner
from warp.tests.unittest_utils import *


def _check_linear_solve(test, A, b, func, *args, **kwargs):
    # test from zero
    x = wp.zeros_like(b)
    with wp.ScopedDevice(A.device):
        niter, err, atol = func(A, b, x, *args, use_cuda_graph=True, **kwargs)

    test.assertLessEqual(err, atol)

    # Test with capturable graph
    if A.device.is_cuda and wp.is_conditional_graph_supported():
        x.zero_()
        with wp.ScopedDevice(A.device):
            with wp.ScopedCapture() as capture:
                niter, err, atol = func(A, b, x, *args, use_cuda_graph=True, check_every=0, **kwargs)

            wp.capture_launch(capture.graph)

        niter = niter.numpy()[0]
        err = np.sqrt(err.numpy()[0])
        atol = np.sqrt(atol.numpy()[0])

        test.assertLessEqual(err, atol)

    # test with warm start
    with wp.ScopedDevice(A.device):
        niter_warm, err, atol = func(A, b, x, *args, use_cuda_graph=False, **kwargs)

    if isinstance(niter_warm, wp.array):
        niter_warm = niter_warm.numpy()[0]
        err = np.sqrt(err.numpy()[0])
        atol = np.sqrt(atol.numpy()[0])

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


def _batch_offsets(batch_sizes, device):
    offsets = np.concatenate([[0], np.cumsum(batch_sizes)]).astype(np.int32)
    return wp.array(offsets, dtype=int, device=device)


def test_batched_host_loop_per_batch_tolerance(test, device):
    r_norm_sq = wp.array([25.0, 4.0], dtype=wp.float32, device=device)
    atol_sq = wp.array([100.0, 1.0], dtype=wp.float32, device=device)
    next_r_norm_sq = wp.array([25.0, 0.25], dtype=wp.float32, device=device)
    cycle_count = 0

    def do_cycle():
        nonlocal cycle_count
        cycle_count += 1
        r_norm_sq.assign(next_r_norm_sq)

    niter, err, atol = _run_solver_loop(
        do_cycle=do_cycle,
        cycle_size=1,
        r_norm_sq=r_norm_sq,
        maxiter=3,
        atol_sq=atol_sq,
        callback=None,
        check_every=1,
        use_cuda_graph=False,
        device=r_norm_sq.device,
    )

    test.assertEqual(niter, 1)
    test.assertEqual(cycle_count, 1)
    test.assertLessEqual(err, atol)


def _check_batch_residuals(test, A_np_full, b_np_full, batch_sizes, x_full, tol, dtype):
    """Verify per-batch residuals match what _check_linear_solve uses."""
    scale = 32.0 if dtype == wp.float32 else 2.0
    x_np = x_full.numpy()
    offsets = np.concatenate([[0], np.cumsum(batch_sizes)])
    for i, (start, end) in enumerate(itertools.pairwise(offsets)):
        sl = slice(start, end)
        A_i = A_np_full[sl, sl].astype(np.float64)
        b_i = b_np_full[sl].astype(np.float64)
        res = np.linalg.norm(A_i @ x_np[sl].astype(np.float64) - b_i)
        atol_i = tol * np.linalg.norm(b_i)
        test.assertLessEqual(float(res), scale * atol_i, msg=f"batch {i}: residual {res:.2e} > {scale * atol_i:.2e}")


def _run_batched_spd_solver(test, device, solver, seed_base, dtype=wp.float32, batch_sizes=None, tol=1e-5):
    if batch_sizes is None:
        batch_sizes = [20] * 4
    rows = sum(batch_sizes)
    A_np_full = np.zeros((rows, rows), dtype=np.float64 if dtype == wp.float64 else np.float32)
    b_np_full = np.zeros(rows, dtype=A_np_full.dtype)

    for i, n in enumerate(batch_sizes):
        A_i, b_i = _make_spd_system(n, seed=seed_base + i, dtype=dtype, device="cpu")
        offset = sum(batch_sizes[:i])
        sl = slice(offset, offset + n)
        A_np_full[sl, sl] = A_i.numpy()
        b_np_full[sl] = b_i.numpy()

    A_full = wp.array(A_np_full, dtype=dtype, device=device)
    b_full = wp.array(b_np_full, dtype=dtype, device=device)
    x_full = wp.zeros_like(b_full)

    offsets = _batch_offsets(batch_sizes, device)
    A_op = aslinearoperator(A_full, batch_offsets=offsets)
    test.assertEqual(A_op.batch_count, len(batch_sizes))

    solver(A_op, b_full, x_full, tol=tol, maxiter=1000)

    _check_batch_residuals(test, A_np_full, b_np_full, batch_sizes, x_full, tol, dtype)


def test_batched_cg(test, device, dtype=wp.float32):
    _run_batched_spd_solver(test, device, cg, seed_base=0, dtype=dtype)


def test_batched_cr(test, device, dtype=wp.float32):
    _run_batched_spd_solver(test, device, cr, seed_base=100, dtype=dtype)


def test_batched_bicgstab(test, device, dtype=wp.float32):
    _run_batched_spd_solver(test, device, bicgstab, seed_base=200, dtype=dtype)


def test_batched_nonuniform(test, device, dtype=wp.float32):
    _run_batched_spd_solver(test, device, cg, seed_base=300, dtype=dtype, batch_sizes=[8, 15, 10, 12])


def test_batched_vector_offsets(test, device):
    diag = wp.array(((2.0, 2.0), (5.0, 5.0)), dtype=wp.vec2, device=device)
    b = wp.array(((2.0, 4.0), (10.0, 15.0)), dtype=wp.vec2, device=device)
    expected = np.array(((1.0, 2.0), (2.0, 3.0)), dtype=np.float32)

    offsets = _batch_offsets([2, 2], device)
    A = aslinearoperator(diag, batch_offsets=offsets)

    for solver, kwargs in (
        (cg, {}),
        (cr, {}),
        (bicgstab, {}),
        (gmres, {"restart": 2}),
    ):
        x = wp.zeros_like(b)
        solver(A, b, x, tol=1.0e-7, maxiter=4, check_every=1, use_cuda_graph=False, **kwargs)
        np.testing.assert_allclose(x.numpy(), expected, rtol=1.0e-5, atol=1.0e-5)


def _run_batched_gmres(test, device, dtype, batch_sizes, seed_base, tol, restart):
    rows = sum(batch_sizes)
    A_np_full = np.zeros((rows, rows), dtype=np.float64 if dtype == wp.float64 else np.float32)
    b_np_full = np.zeros(rows, dtype=A_np_full.dtype)

    for i, n in enumerate(batch_sizes):
        A_i, b_i = _make_nonsymmetric_system(n, seed=seed_base + i, dtype=dtype, device="cpu")
        offset = sum(batch_sizes[:i])
        sl = slice(offset, offset + n)
        A_np_full[sl, sl] = A_i.numpy()
        b_np_full[sl] = b_i.numpy()

    A_full = wp.array(A_np_full, dtype=dtype, device=device)
    b_full = wp.array(b_np_full, dtype=dtype, device=device)

    offsets = _batch_offsets(batch_sizes, device)
    A_op = aslinearoperator(A_full, batch_offsets=offsets)
    test.assertEqual(A_op.batch_count, len(batch_sizes))

    # Diagonal preconditioner (block-diagonal across the full system, so implicitly per-batch).
    M = preconditioner(A_full, "diag")

    # (description, kwargs) pairs — no precond, right precond, left precond
    cases = [
        ("none", {}),
        ("right", {"M": M}),
        ("left", {"M": M, "is_left_preconditioner": True}),
    ]
    for _label, kwargs in cases:
        x_full = wp.zeros_like(b_full)
        gmres(A_op, b_full, x_full, tol=tol, restart=restart, maxiter=1000, **kwargs)
        _check_batch_residuals(test, A_np_full, b_np_full, batch_sizes, x_full, tol, dtype)


def test_batched_gmres(test, device, dtype=wp.float32, batch_count=4, n=20):
    _run_batched_gmres(
        test,
        device,
        dtype,
        batch_sizes=[n] * batch_count,
        seed_base=456,
        tol=1e-3 if dtype == wp.float32 else 1e-5,
        restart=16,
    )


def test_batched_gmres_nonuniform(test, device, dtype=wp.float32):
    _run_batched_gmres(
        test,
        device,
        dtype,
        batch_sizes=[8, 15, 10, 12],
        seed_base=654,
        tol=1e-3 if dtype == wp.float32 else 1e-5,
        restart=16,
    )


def test_functor_reuse(test, device):
    # For each solver, construct a pre-allocated functor, then re-run on a different
    # (but compatible) system without re-allocating temporary buffers.
    cases = [
        (cg, CG, _make_spd_system, 32, {"maxiter": 500}),
        (cr, CR, _make_spd_system, 32, {"maxiter": 500}),
        (bicgstab, BiCGSTAB, _make_nonsymmetric_system, 32, {"maxiter": 500}),
        (gmres, GMRES, _make_nonsymmetric_system, 16, {"tol": 1.0e-3, "restart": 16, "maxiter": 256}),
    ]
    with wp.ScopedDevice(device):
        for func, klass, make_system, n, kwargs in cases:
            A1, b1 = make_system(n=n, seed=11, dtype=wp.float64, device=device)
            x1 = wp.zeros_like(b1)
            state = func(A1, b1, x1, run=False, **kwargs)
            test.assertIsInstance(state, klass)

            # First run with the original system
            _niter, err, atol = state()
            test.assertLessEqual(err, atol)

            # Second run with a *different* but compatible system
            A2, b2 = make_system(n=n, seed=22, dtype=wp.float64, device=device)
            x2 = wp.zeros_like(b2)
            _niter2, err2, atol2 = state(A=A2, b=b2, x=x2)
            test.assertLessEqual(err2, atol2)

            # Residual check in numpy to confirm x2 really solves A2 x2 = b2
            residual = A2.numpy() @ x2.numpy() - b2.numpy()
            test.assertLessEqual(np.linalg.norm(residual), 2.0 * atol2)


def test_functor_preconditioner(test, device):
    # CG and CR allow toggling M between None and a valid preconditioner between calls.
    with wp.ScopedDevice(device):
        A, b = _make_spd_system(n=32, seed=33, dtype=wp.float64, device=device)
        M = preconditioner(A, "diag")

        for func in (cg, cr):
            x = wp.zeros_like(b)
            state = func(A, b, x, maxiter=500, run=False)

            # No preconditioner on first call
            _, err, atol = state()
            test.assertLessEqual(err, atol)

            # With preconditioner on second call
            x.zero_()
            _, err2, atol2 = state(M=M)
            test.assertLessEqual(err2, atol2)


def test_functor_compat_errors(test, device):
    with wp.ScopedDevice(device):
        A, b = _make_spd_system(n=32, seed=44, dtype=wp.float64, device=device)
        x = wp.zeros_like(b)
        state = cg(A, b, x, maxiter=100, run=False)

        # Wrong b shape
        b_bad = wp.zeros(64, dtype=wp.float64, device=device)
        with test.assertRaises(ValueError):
            state(b=b_bad)

        # Wrong dtype
        A_bad, b_bad = _make_spd_system(n=32, seed=44, dtype=wp.float32, device=device)
        x_bad = wp.zeros_like(b_bad)
        with test.assertRaises(ValueError):
            state(A=A_bad, b=b_bad, x=x_bad)

        # BiCGSTAB requires M presence to match
        A2, b2 = _make_nonsymmetric_system(n=16, seed=45, dtype=wp.float64, device=device)
        x2 = wp.zeros_like(b2)
        M2 = preconditioner(A2, "diag")
        bic_state = bicgstab(A2, b2, x2, maxiter=100, run=False)  # M=None at construction
        with test.assertRaises(ValueError):
            bic_state(M=M2)


class TestLinearSolvers(unittest.TestCase):
    pass


devices = get_test_devices()

add_function_test(TestLinearSolvers, "test_cg", test_cg, devices=devices)
add_function_test(TestLinearSolvers, "test_cr", test_cr, devices=devices)
add_function_test(TestLinearSolvers, "test_bicgstab", test_bicgstab, devices=devices)
add_function_test(TestLinearSolvers, "test_gmres", test_gmres, devices=devices)
add_function_test(
    TestLinearSolvers,
    "test_batched_host_loop_per_batch_tolerance",
    test_batched_host_loop_per_batch_tolerance,
    devices=devices,
)
add_function_test(TestLinearSolvers, "test_batched_cg_f32", test_batched_cg, devices=devices)
add_function_test(TestLinearSolvers, "test_batched_cg_f64", test_batched_cg, devices=devices, dtype=wp.float64)
add_function_test(TestLinearSolvers, "test_batched_cr_f32", test_batched_cr, devices=devices)
add_function_test(TestLinearSolvers, "test_batched_bicgstab_f32", test_batched_bicgstab, devices=devices)
add_function_test(TestLinearSolvers, "test_batched_gmres_f32", test_batched_gmres, devices=devices)
add_function_test(TestLinearSolvers, "test_batched_gmres_f64", test_batched_gmres, devices=devices, dtype=wp.float64)
add_function_test(TestLinearSolvers, "test_batched_gmres_nonuniform", test_batched_gmres_nonuniform, devices=devices)
add_function_test(TestLinearSolvers, "test_batched_nonuniform", test_batched_nonuniform, devices=devices)
add_function_test(TestLinearSolvers, "test_batched_vector_offsets", test_batched_vector_offsets, devices=devices)
add_function_test(TestLinearSolvers, "test_functor_reuse", test_functor_reuse, devices=devices)
add_function_test(TestLinearSolvers, "test_functor_preconditioner", test_functor_preconditioner, devices=devices)
add_function_test(TestLinearSolvers, "test_functor_compat_errors", test_functor_compat_errors, devices=devices)

if __name__ == "__main__":
    unittest.main(verbosity=2)
