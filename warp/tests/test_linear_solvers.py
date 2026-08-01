# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools
import unittest

import numpy as np

import warp as wp
from warp._src.optim.linear import TiledDot, _create_segmented_tiled_dot_kernels, _run_solver_loop
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


def test_conjugate_solver_restart(test, device):
    rng = np.random.default_rng(2027)
    C = rng.standard_normal((12, 12))
    A_np = C.T @ C + 4.0 * np.eye(12)
    b_np = rng.standard_normal(12)

    def make_callback(iterations):
        def callback(iteration, _residual, _tolerance):
            iterations.append(iteration)

        return callback

    with wp.ScopedDevice(device):
        A = wp.array(A_np, dtype=wp.float64, device=device)
        b = wp.array(b_np, dtype=wp.float64, device=device)
        M = preconditioner(A, "diag")

        for solver in (cg, cr):
            for preconditioner_op in (None, M):
                with test.subTest(solver=solver.__name__, preconditioned=preconditioner_op is not None):
                    callback_iterations = []

                    x = wp.zeros_like(b)
                    niter, err, atol = solver(
                        A,
                        b,
                        x,
                        tol=1.0e-10,
                        maxiter=120,
                        M=preconditioner_op,
                        callback=make_callback(callback_iterations),
                        check_every=1,
                        use_cuda_graph=True,
                        restart=3,
                    )

                    test.assertLessEqual(err, atol)
                    test.assertEqual(niter % 3, 0)
                    test.assertEqual(callback_iterations[0], 0)
                    test.assertEqual(callback_iterations[-1], niter)
                    test.assertTrue(all(iteration % 3 == 0 for iteration in callback_iterations))

                    true_residual = A_np @ x.numpy() - b_np
                    test.assertLessEqual(np.linalg.norm(true_residual), atol)


def _make_cg_restart_drift_system(batch_count):
    dof_count = 130
    chain_count = dof_count - 1
    edge_count = chain_count - 1
    mass = 3.0 / chain_count
    edge_weight = 0.18**2 / (3.0 / edge_count)

    A_single = mass * np.eye(dof_count)
    for edge in range(edge_count):
        A_single[edge, edge] += edge_weight
        A_single[edge + 1, edge + 1] += edge_weight
        A_single[edge, edge + 1] -= edge_weight
        A_single[edge + 1, edge] -= edge_weight

    fixed = np.arange(0, chain_count, 17)
    A_single[fixed, :] = 0.0
    A_single[:, fixed] = 0.0
    A_single[fixed, fixed] = 1.0

    A = np.kron(np.eye(batch_count), A_single).astype(np.float32)
    rng = np.random.default_rng(7301)
    initial = (0.1 * rng.standard_normal((batch_count, dof_count))).astype(np.float32)
    b = (mass * initial).astype(np.float32)
    initial[:, fixed] = 0.0
    b[:, fixed] = 0.0

    return A, b.reshape(-1), initial.reshape(-1)


def test_cg_restart_float32_drift(test, device):
    interval = 32
    tolerance = 1.0e-6

    for batch_count in (1, 3):
        A_np, b_np, initial_np = _make_cg_restart_drift_system(batch_count)
        dof_count = b_np.shape[0] // batch_count

        with wp.ScopedDevice(device):
            A = wp.array(A_np, dtype=wp.float32, device=device)
            b = wp.array(b_np, dtype=wp.float32, device=device)
            M = preconditioner(A, "diag")
            offsets = None if batch_count == 1 else _batch_offsets([dof_count] * batch_count, device)
            A_op = aslinearoperator(A, batch_offsets=offsets)

            for preconditioner_name, preconditioner_op in (("none", None), ("Jacobi", M)):
                x = wp.array(initial_np, dtype=wp.float32, device=device)
                niter, err, atol = cg(
                    A_op,
                    b,
                    x,
                    tol=tolerance,
                    atol=0.0,
                    maxiter=512,
                    M=preconditioner_op,
                    check_every=interval,
                    use_cuda_graph=True,
                    restart=interval,
                )

                test.assertLessEqual(err, atol)
                test.assertEqual(niter % interval, 0)
                test.assertLessEqual(niter, 512)

                x_np = x.numpy().astype(np.float64)
                for batch_index in range(batch_count):
                    start = batch_index * dof_count
                    end = start + dof_count
                    A_batch = A_np[start:end, start:end].astype(np.float64)
                    b_batch = b_np[start:end].astype(np.float64)
                    residual = A_batch @ x_np[start:end] - b_batch
                    target = tolerance * np.linalg.norm(b_batch)
                    test.assertLessEqual(
                        np.linalg.norm(residual),
                        target,
                        msg=(
                            f"{preconditioner_name} batch {batch_index} did not reach "
                            "the requested true-residual tolerance"
                        ),
                    )


def test_conjugate_solver_restart_validation(test, device):
    A, b = _make_identity_system(n=3, seed=123, dtype=wp.float64, device=device)
    x = wp.zeros_like(b)

    for solver, state_type in ((cg, CG), (cr, CR)):
        for value in (True, 1.5):
            with test.assertRaises(TypeError):
                solver(A, b, x, run=False, restart=value)

        for value in (0, -1):
            with test.assertRaises(ValueError):
                solver(A, b, x, run=False, restart=value)

        state = solver(A, b, x, run=False, restart=np.int64(2))
        test.assertIsInstance(state, state_type)

        x.zero_()
        niter, err, atol = solver(A, b, x, maxiter=0.5, check_every=1, restart=1)
        test.assertEqual(niter, 1)
        test.assertLessEqual(err, atol)


def test_conjugate_solver_restart_conditional_graph(test, device):
    if not wp.is_conditional_graph_supported():
        test.skipTest("conditional CUDA graphs are not supported")

    with wp.ScopedDevice(device):
        A = wp.array((2.0, 3.0, 4.0), dtype=wp.float32, device=device)
        b = wp.array((2.0, 6.0, 12.0), dtype=wp.float32, device=device)
        for solver in (cg, cr):
            with test.subTest(solver=solver.__name__):
                x = wp.zeros_like(b)
                state = solver(
                    A,
                    b,
                    x,
                    tol=1.0e-6,
                    maxiter=6,
                    check_every=0,
                    use_cuda_graph=True,
                    run=False,
                    restart=3,
                )

                state()
                x.zero_()

                with wp.ScopedCapture() as capture:
                    niter, residual_sq, tolerance_sq = state()

                x.zero_()
                wp.capture_launch(capture.graph)

                test.assertEqual(niter.numpy()[0], 3)
                test.assertLessEqual(residual_sq.numpy()[0], tolerance_sq.numpy()[0])
                np.testing.assert_allclose(x.numpy(), (1.0, 2.0, 3.0), rtol=1.0e-5, atol=1.0e-5)

                wp.capture_launch(capture.graph)
                test.assertEqual(niter.numpy()[0], 0)
                test.assertLessEqual(residual_sq.numpy()[0], tolerance_sq.numpy()[0])


def test_gmres_conditional_graph_cycle_iterations(test, device):
    if not wp.is_conditional_graph_supported():
        test.skipTest("conditional CUDA graphs are not supported")

    with wp.ScopedDevice(device):
        A = wp.array((2.0, 3.0, 4.0), dtype=wp.float32, device=device)
        b = wp.array((2.0, 6.0, 12.0), dtype=wp.float32, device=device)
        x = wp.zeros_like(b)
        state = gmres(
            A,
            b,
            x,
            tol=1.0e-6,
            restart=3,
            maxiter=6,
            check_every=0,
            use_cuda_graph=True,
            run=False,
        )

        state()
        x.zero_()

        with wp.ScopedCapture() as capture:
            niter, residual_sq, tolerance_sq = state()

        x.zero_()
        wp.capture_launch(capture.graph)

        test.assertEqual(niter.numpy()[0], 3)
        test.assertLessEqual(residual_sq.numpy()[0], tolerance_sq.numpy()[0])
        np.testing.assert_allclose(x.numpy(), (1.0, 2.0, 3.0), rtol=1.0e-5, atol=1.0e-5)

        wp.capture_launch(capture.graph)
        test.assertEqual(niter.numpy()[0], 0)
        test.assertLessEqual(residual_sq.numpy()[0], tolerance_sq.numpy()[0])


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
    A_op = aslinearoperator(A_full, batch_offsets=offsets, max_batch_length=max(batch_sizes))
    test.assertEqual(A_op.batch_count, len(batch_sizes))
    test.assertEqual(A_op.max_batch_length, max(batch_sizes))

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
    A = aslinearoperator(diag, batch_offsets=offsets, max_batch_length=2)

    for solver, kwargs in (
        (cg, {}),
        (cr, {}),
        (bicgstab, {}),
        (gmres, {"restart": 2}),
    ):
        x = wp.zeros_like(b)
        solver(A, b, x, tol=1.0e-7, maxiter=4, check_every=1, use_cuda_graph=False, **kwargs)
        np.testing.assert_allclose(x.numpy(), expected, rtol=1.0e-5, atol=1.0e-5)


def test_batched_inactive_tail(test, device):
    diag = wp.array([2.0, 3.0, 4.0, 5.0, 6.0], dtype=wp.float64, device=device)
    b = wp.array([2.0, 6.0, 12.0, 100.0, 200.0], dtype=wp.float64, device=device)
    initial = np.array([0.0, 0.0, 0.0, 7.0, 8.0], dtype=np.float64)
    expected = np.array([1.0, 2.0, 3.0, 7.0, 8.0], dtype=np.float64)

    # Keep an explicit one-batch layout: the batched reduction must not include
    # the inactive tail when only one subproblem is present.
    offsets = _batch_offsets([3], device)
    A = aslinearoperator(diag, batch_offsets=offsets, max_batch_length=3)

    for solver, kwargs in (
        (cg, {}),
        (cr, {}),
        (bicgstab, {}),
        (gmres, {"restart": 2}),
    ):
        x = wp.array(initial, device=device)
        niter, err, atol = solver(A, b, x, tol=1.0e-8, maxiter=20, check_every=1, **kwargs)

        test.assertGreater(niter, 1)
        test.assertLessEqual(err, atol)
        np.testing.assert_allclose(x.numpy(), expected, rtol=1.0e-7, atol=1.0e-7)


def test_batched_vector_inactive_tail(test, device):
    diag = wp.array(((2.0, 3.0), (4.0, 5.0), (6.0, 7.0)), dtype=wp.vec2d, device=device)
    b = wp.array(((2.0, 6.0), (12.0, 20.0), (100.0, 200.0)), dtype=wp.vec2d, device=device)
    initial = np.array(((0.0, 0.0), (0.0, 0.0), (7.0, 8.0)), dtype=np.float64)
    expected = np.array(((1.0, 2.0), (3.0, 4.0), (7.0, 8.0)), dtype=np.float64)

    offsets = _batch_offsets([2, 2], device)
    A = aslinearoperator(diag, batch_offsets=offsets, max_batch_length=2)

    for solver, kwargs in (
        (cg, {}),
        (cr, {}),
        (bicgstab, {}),
        (gmres, {"restart": 2}),
    ):
        x = wp.array(initial, dtype=wp.vec2d, device=device)
        _, err, atol = solver(A, b, x, tol=1.0e-8, maxiter=8, check_every=1, use_cuda_graph=False, **kwargs)
        test.assertLessEqual(err, atol)
        np.testing.assert_allclose(x.numpy(), expected, rtol=1.0e-7, atol=1.0e-7)


def test_batched_gmres_right_preconditioned_inactive_tail(test, device):
    A = aslinearoperator(
        wp.array(
            ((2.0, 0.0, 0.0), (0.0, 3.0, 0.0), (5.0, 0.0, 1.0)),
            dtype=wp.float64,
            device=device,
        ),
        batch_offsets=_batch_offsets([2], device),
        max_batch_length=2,
    )
    M = aslinearoperator(wp.array((1.0, 1.0, 1.0), dtype=wp.float64, device=device))
    b = wp.array((2.0, 6.0, 7.0), dtype=wp.float64, device=device)
    x = wp.array((0.0, 0.0, 7.0), dtype=wp.float64, device=device)

    _, err, atol = gmres(
        A,
        b,
        x,
        M=M,
        is_left_preconditioner=False,
        restart=2,
        maxiter=2,
        check_every=1,
        tol=1.0e-12,
        use_cuda_graph=True,
    )

    test.assertLessEqual(err, atol)
    np.testing.assert_allclose(x.numpy(), (1.0, 2.0, 7.0), rtol=1.0e-12, atol=1.0e-12)


def test_tiled_dot_large_single_batch(test, device):
    length = 100_000
    active_length = length - 2
    values = np.ones(length, dtype=np.float32)
    values[active_length:] = 100.0

    a = wp.array(values, device=device)
    offsets = _batch_offsets([active_length], device)
    tiled_dot = TiledDot(length, wp.float32, device=device, batch_offsets=offsets)
    tiled_dot.compute(a, a)

    with wp.ScopedDevice(device):
        with wp.ScopedCapture(force_module_load=False) as capture:
            tiled_dot.compute(a, a)

        a.fill_(2.0)
        wp.capture_launch(capture.graph)

    test.assertEqual(tiled_dot.col().numpy()[0], 4 * active_length)


def test_tiled_dot_batched_tree(test, device):
    batch_sizes = [0, 1, 511, 512, 513, 65_537, 262_145]
    active_length = sum(batch_sizes)
    length = active_length + 17
    rng = np.random.default_rng(123)
    a_np = rng.standard_normal(length).astype(np.float32)
    b_np = rng.standard_normal(length).astype(np.float32)

    a = wp.array(a_np, device=device)
    b = wp.array(b_np, device=device)
    offsets = _batch_offsets(batch_sizes, device)
    batched_dot = TiledDot(
        length,
        wp.float32,
        device=device,
        batch_offsets=offsets,
        max_batch_length=max(batch_sizes),
    )
    level_batch_sizes = batch_sizes
    for level_offsets in batched_dot.segmented_offsets[1:]:
        level_batch_sizes = [
            max(1, (size + batched_dot.tile_size - 1) // batched_dot.tile_size) for size in level_batch_sizes
        ]
        expected_offsets = np.concatenate([[0], np.cumsum(level_batch_sizes)]).astype(np.int32)
        np.testing.assert_array_equal(level_offsets.numpy(), expected_offsets)

    batched_dot.compute(a, b)

    expected = np.empty(len(batch_sizes), dtype=np.float32)
    batch_offsets = np.concatenate([[0], np.cumsum(batch_sizes)])
    for batch, (start, end) in enumerate(itertools.pairwise(batch_offsets)):
        if start == end:
            expected[batch] = 0.0
            continue

        single_dot = TiledDot(end - start, wp.float32, device=device)
        single_dot.compute(a[start:end], b[start:end])
        expected[batch] = single_dot.col().numpy()[0]

    np.testing.assert_array_equal(batched_dot.col().numpy(), expected)

    with wp.ScopedDevice(device):
        with wp.ScopedCapture(force_module_load=False) as capture:
            batched_dot.compute(a, b)

        a.fill_(2.0)
        b.fill_(3.0)
        wp.capture_launch(capture.graph)

    np.testing.assert_array_equal(
        batched_dot.col().numpy(),
        6.0 * np.array(batch_sizes, dtype=np.float32),
    )


def test_tiled_dot_max_batch_length(test, device):
    batch_sizes = [20] * 64
    length = sum(batch_sizes)
    offsets = _batch_offsets(batch_sizes, device)
    a = wp.ones(length, dtype=wp.float32, device=device)

    bounded_dot = TiledDot(
        length,
        wp.float32,
        device=device,
        batch_offsets=offsets,
        max_batch_length=max(batch_sizes),
    )
    fallback_dot = TiledDot(length, wp.float32, device=device, batch_offsets=offsets)

    test.assertEqual(bounded_dot.rounds, 0)
    test.assertEqual(fallback_dot.rounds, 1)
    test.assertIsNotNone(bounded_dot.batch_dot_launch)
    test.assertIsNone(bounded_dot.segmented_dot_launch)
    np.testing.assert_array_equal(
        fallback_dot.segmented_offsets[1].numpy(),
        np.arange(len(batch_sizes) + 1, dtype=np.int32),
    )

    bounded_dot.compute(a, a)
    fallback_dot.compute(a, a)
    expected = np.full(len(batch_sizes), 20.0, dtype=np.float32)
    np.testing.assert_array_equal(bounded_dot.col().numpy(), expected)
    np.testing.assert_array_equal(fallback_dot.col().numpy(), expected)


def test_tiled_dot_plan_offsets_int_max(test, device):
    tile_size = 512
    input_offsets = wp.array([0, np.iinfo(np.int32).max], dtype=int, device=device)
    output_offsets = wp.empty_like(input_offsets)
    plan_offsets_kernel, _, _ = _create_segmented_tiled_dot_kernels(tile_size)

    wp.launch(
        plan_offsets_kernel,
        dim=tile_size,
        inputs=[input_offsets],
        outputs=[output_offsets],
        block_dim=tile_size,
        device=device,
    )

    np.testing.assert_array_equal(output_offsets.numpy(), (0, 4_194_304))


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
    A_op = aslinearoperator(A_full, batch_offsets=offsets, max_batch_length=max(batch_sizes))
    test.assertEqual(A_op.batch_count, len(batch_sizes))
    test.assertEqual(A_op.max_batch_length, max(batch_sizes))

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
        (cg, CG, _make_spd_system, 32, {"maxiter": 500, "restart": 32}),
        (cr, CR, _make_spd_system, 32, {"maxiter": 500, "restart": 32}),
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


def test_functor_iteration_buffer_reuse(test, device):
    A, b = _make_identity_system(n=5, seed=23, dtype=wp.float32, device=device)
    solver_cases = (
        (cg, {}),
        (cr, {}),
        (bicgstab, {}),
        (gmres, {"restart": 2}),
    )

    with wp.ScopedDevice(device):
        for func, kwargs in solver_cases:
            x = wp.zeros_like(b)
            state = func(
                A,
                b,
                x,
                maxiter=2,
                check_every=0,
                use_cuda_graph=False,
                run=False,
                **kwargs,
            )
            iteration_buffer = state._cur_iter_and_condition

            final_iteration, err, atol = state()
            test.assertEqual(final_iteration.ptr, iteration_buffer.ptr)
            expected_final_iteration = final_iteration.numpy()[0]
            test.assertLessEqual(err.numpy()[0], atol.numpy()[0])

            iteration_buffer.fill_(123)
            x.zero_()
            final_iteration, err, atol = state()
            test.assertEqual(final_iteration.ptr, iteration_buffer.ptr)
            test.assertEqual(final_iteration.numpy()[0], expected_final_iteration)
            test.assertLessEqual(err.numpy()[0], atol.numpy()[0])

            if A.device.is_cuda and wp.is_conditional_graph_supported():
                x.zero_()
                capture_state = func(
                    A,
                    b,
                    x,
                    maxiter=2,
                    check_every=0,
                    use_cuda_graph=True,
                    run=False,
                    **kwargs,
                )

                # Warm up the solver modules before the outer capture.
                capture_state()
                x.zero_()

                with wp.ScopedCapture(device=device, force_module_load=False) as capture:
                    final_iteration, err, atol = capture_state()

                wp.capture_launch(capture.graph)
                test.assertEqual(final_iteration.ptr, capture_state._cur_iter_and_condition.ptr)
                test.assertLessEqual(err.numpy()[0], atol.numpy()[0])


def test_functor_preconditioner(test, device):
    # CG and CR allow toggling M between None and a valid preconditioner between calls.
    with wp.ScopedDevice(device):
        A, b = _make_spd_system(n=32, seed=33, dtype=wp.float64, device=device)
        M = preconditioner(A, "diag")

        for func in (cg, cr):
            with test.subTest(solver=func.__name__):
                x = wp.zeros_like(b)
                state = func(A, b, x, maxiter=640, restart=32, run=False)

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

        # An explicit one-batch layout is not compatible with unbatched scratch.
        A_one_batch = aslinearoperator(A, batch_offsets=_batch_offsets([32], device))
        with test.assertRaisesRegex(ValueError, "batch_offsets"):
            state(A=A_one_batch)

        # Stateful batched solvers require fixed reduction-planning metadata.
        offsets = _batch_offsets([16, 16], device)
        A_batched = aslinearoperator(A, batch_offsets=offsets, max_batch_length=16)
        A_batched_bad = aslinearoperator(A, batch_offsets=offsets, max_batch_length=32)
        batched_state = cg(A_batched, b, x, maxiter=100, run=False)
        with test.assertRaisesRegex(ValueError, "max_batch_length"):
            batched_state(A=A_batched_bad)

        with test.assertRaisesRegex(ValueError, "requires batch_offsets"):
            aslinearoperator(A, max_batch_length=32)

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
devices_with_graph_capture_allocation = get_test_devices_with_graph_capture_allocation()

add_function_test(TestLinearSolvers, "test_cg", test_cg, devices=devices_with_graph_capture_allocation)
add_function_test(TestLinearSolvers, "test_conjugate_solver_restart", test_conjugate_solver_restart, devices=devices)
add_function_test(
    TestLinearSolvers,
    "test_cg_restart_float32_drift",
    test_cg_restart_float32_drift,
    devices=devices,
)
add_function_test(
    TestLinearSolvers,
    "test_conjugate_solver_restart_validation",
    test_conjugate_solver_restart_validation,
    devices=[wp.get_device("cpu")],
)
add_function_test(
    TestLinearSolvers,
    "test_conjugate_solver_restart_conditional_graph",
    test_conjugate_solver_restart_conditional_graph,
    devices=get_cuda_test_devices_with_mempool(),
)
add_function_test(
    TestLinearSolvers,
    "test_gmres_conditional_graph_cycle_iterations",
    test_gmres_conditional_graph_cycle_iterations,
    devices=get_cuda_test_devices_with_mempool(),
)
add_function_test(TestLinearSolvers, "test_cr", test_cr, devices=devices_with_graph_capture_allocation)
add_function_test(TestLinearSolvers, "test_bicgstab", test_bicgstab, devices=devices_with_graph_capture_allocation)
add_function_test(TestLinearSolvers, "test_gmres", test_gmres, devices=devices_with_graph_capture_allocation)
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
add_function_test(TestLinearSolvers, "test_batched_inactive_tail", test_batched_inactive_tail, devices=devices)
add_function_test(
    TestLinearSolvers,
    "test_batched_vector_inactive_tail",
    test_batched_vector_inactive_tail,
    devices=devices,
)
add_function_test(
    TestLinearSolvers,
    "test_batched_gmres_right_preconditioned_inactive_tail",
    test_batched_gmres_right_preconditioned_inactive_tail,
    devices=devices,
)
add_function_test(
    TestLinearSolvers,
    "test_tiled_dot_large_single_batch",
    test_tiled_dot_large_single_batch,
    devices=get_cuda_test_devices(),
)
add_function_test(
    TestLinearSolvers,
    "test_tiled_dot_batched_tree",
    test_tiled_dot_batched_tree,
    devices=get_cuda_test_devices(),
)
add_function_test(
    TestLinearSolvers,
    "test_tiled_dot_max_batch_length",
    test_tiled_dot_max_batch_length,
    devices=get_cuda_test_devices(),
)
add_function_test(
    TestLinearSolvers,
    "test_tiled_dot_plan_offsets_int_max",
    test_tiled_dot_plan_offsets_int_max,
    devices=get_cuda_test_devices(),
)
add_function_test(TestLinearSolvers, "test_functor_reuse", test_functor_reuse, devices=devices)
add_function_test(
    TestLinearSolvers,
    "test_functor_iteration_buffer_reuse",
    test_functor_iteration_buffer_reuse,
    devices=devices,
)
add_function_test(TestLinearSolvers, "test_functor_preconditioner", test_functor_preconditioner, devices=devices)
add_function_test(TestLinearSolvers, "test_functor_compat_errors", test_functor_compat_errors, devices=devices)

if __name__ == "__main__":
    unittest.main(verbosity=2)
