# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest
from pathlib import Path

import numpy as np

import warp as wp
import warp.sparse as sp
from warp.optim.linear import (
    KrylovSolve,
    MultiRHSOperator,
    aslinearoperator,
    preconditioner,
)
from warp.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices


def _matrix(a, device, dtype, block):
    rows, cols = np.nonzero(a)
    result = sp.bsr_from_triplets(
        len(a),
        len(a),
        wp.array(rows, dtype=int, device=device),
        wp.array(cols, dtype=int, device=device),
        wp.array(a[rows, cols], dtype=dtype, device=device),
    )
    return sp.bsr_copy(result, block_shape=(block, block)) if block != 1 else result


def _gmres(A, nrhs=1, **kwargs):
    kwargs.setdefault("residual", "fresh")
    return KrylovSolve(A, nrhs, method="gmres", **kwargs)


def _cg(A, nrhs, **kwargs):
    return KrylovSolve(A, nrhs, method="cg", **kwargs)


def _algorithms(device):
    """Return every batched product strategy available on ``device``."""
    result = ["loop", "rows"]
    if wp.get_device(device).is_cuda:
        result.append("columns")
    return result


def test_krylov_methods(test, device):
    """Check every method on batched and single RHSs with both residual modes and a preconditioner."""
    n = 24
    spd = np.diag(np.linspace(2.0, 5.0, n)) + np.diag(np.full(n - 1, -0.5), 1) + np.diag(np.full(n - 1, -0.5), -1)
    nonsymmetric = spd + np.diag(np.full(n - 1, 0.3), 1)
    rhs = np.random.default_rng(31).normal(size=(3, n))
    for method in ("cg", "cr", "bicgstab", "gmres"):
        a = nonsymmetric if method in ("bicgstab", "gmres") else spd
        matrix = _matrix(a, device, wp.float64, 1)
        expected = np.linalg.solve(a, rhs.T).T
        for residual in ("recurrence", "fresh"):
            for nrhs in (1, 3):
                action = KrylovSolve(
                    matrix, nrhs, method=method, M=preconditioner(matrix, "diag"), tol=1e-10, residual=residual
                )
                b = wp.array(rhs[:nrhs], dtype=wp.float64, device=device)
                out = wp.empty_like(b)
                action.solve(b, out)
                np.testing.assert_array_equal(action.converged.numpy(), np.ones(nrhs), err_msg=method)
                np.testing.assert_allclose(out.numpy(), expected[:nrhs], rtol=1e-8, atol=1e-8, err_msg=method)
    for invalid in ("CG", "minres", None):
        with test.assertRaisesRegex(ValueError, "Unsupported Krylov method"):
            KrylovSolve(matrix, method=invalid)
    with test.assertRaises(ValueError):
        KrylovSolve(matrix, method="cg", restart=8)
    with test.assertRaises(ValueError):
        KrylovSolve(aslinearoperator(matrix), method="cg", working_dtype=wp.float32)


def test_gmres_action(test, device, dtype):
    """Check solutions, fresh diagnostics and graph replay for indefinite and nonsymmetric systems."""
    n = 24
    a = np.diag(np.linspace(-3.0, 3.0, n)) + np.diag(np.full(n - 1, 0.1), -1)
    rng = np.random.default_rng(730)
    for nonsymmetric in (False, True):
        a += np.diag(np.full(n - 1, 0.2 if nonsymmetric else 0.1), 1)
        for block in (1, 3):
            matrix = _matrix(a, device, dtype, block)
            original = matrix.values.numpy().copy()
            for preconditioned in (False, True):
                pre = (
                    aslinearoperator(wp.array(1 / np.maximum(abs(a.diagonal()), 0.2), dtype=dtype, device=device))
                    if preconditioned
                    else None
                )
                action = _gmres(matrix, 3, M=pre, restart=n, maxiter=3 * n, tol=1e-5 if dtype == wp.float32 else 1e-12)
                b = wp.zeros((3, n), dtype=dtype, device=device)
                for alias in (False, True):
                    out = b if alias else wp.empty_like(b)
                    action.solve(b, out)
                    graph = None
                    if device.is_cuda:
                        wp.synchronize_device(device)
                        with wp.ScopedCapture(device=device, force_module_load=False) as capture:
                            action.reset_stats()
                            action.solve(b, out)
                        graph = capture.graph
                    for scale in (1.0, 1.7):
                        rhs = rng.normal(size=(3, n))
                        rhs[0] = 0
                        b.assign(rhs)
                        matrix.values.assign(original * scale)
                        if graph:
                            wp.capture_launch(graph)
                        else:
                            action.reset_stats()
                            action.solve(b, out)
                        actual = out.numpy()
                        error = np.linalg.norm(actual @ (a * scale).T - rhs, axis=1)
                        eps = 3e-5 if dtype == wp.float32 else 1e-10
                        np.testing.assert_allclose(actual, np.linalg.solve(a * scale, rhs.T).T, rtol=eps, atol=eps)
                        np.testing.assert_allclose(action.absolute_residual.numpy(), error, rtol=eps, atol=eps)
                        np.testing.assert_array_equal(action.converged.numpy(), [1, 1, 1])
                        test.assertEqual(action.stats.numpy()[2:].tolist(), [0, 3])
                        test.assertAlmostEqual(action.residual_maxima.numpy()[0], error.max(), delta=eps)
                        test.assertEqual(action.relative_residual.numpy()[0], 0.0)
                    if graph:
                        with tempfile.TemporaryDirectory() as folder:
                            dot = Path(folder) / "gmres.dot"
                            wp.capture_debug_dot_print(graph, str(dot))
                            content = dot.read_text()
                            for forbidden in ("HOST", "DtoH", "HtoD"):
                                test.assertNotIn(forbidden, content)


def test_gmres_action_failures(test, device):
    """Check that exhausted budgets and nonfinite inputs fail acceptance and reset semantics hold."""
    a = np.diag(np.arange(-16.0, 16.0) + 0.2)
    action = _gmres(_matrix(a, device, wp.float64, 1), 2, restart=1, maxiter=1)
    b = wp.zeros((2, 32), dtype=wp.float64, device=device)
    out = wp.empty_like(b)
    action.solve(b, out)
    np.testing.assert_array_equal(action.converged.numpy(), [1, 1])
    b.fill_(1.0)
    action.solve(b, out)
    np.testing.assert_array_equal(action.converged.numpy(), [0, 0])
    test.assertEqual(action.stats.numpy()[2:].tolist(), [2, 4])
    test.assertGreater(action.relative_residual.numpy().min(), 0.5)
    action.reset_stats()
    np.testing.assert_array_equal(action.stats.numpy(), [0, 0, 0, 0])
    np.testing.assert_array_equal(action.converged.numpy(), [0, 0])
    b.fill_(float("nan"))
    action.solve(b, out)
    np.testing.assert_array_equal(action.converged.numpy(), [0, 0])
    test.assertTrue(np.isinf(action.relative_residual.numpy()).all())


def test_gmres_action_validation(test, device):
    """Check that invalid arguments and arrays are rejected before output mutation."""
    a = _matrix(np.eye(8), device, wp.float64, 1)
    for kwargs in (
        {"nrhs": 0},
        {"nrhs": True},
        {"restart": 0},
        {"maxiter": False},
        {"tol": -1.0},
        {"tol": float("nan")},
        {"atol": float("inf")},
        {"tol": 0.0, "atol": 0.0},
        {"tol": 0.0},
    ):
        with test.assertRaises((ValueError, TypeError)):
            _gmres(a, **kwargs)
    with test.assertRaisesRegex(TypeError, "atol"):
        _gmres(a, atol=None)
    action = _gmres(a)
    out = wp.full(8, 7.0, dtype=wp.float64, device=device)
    for bad in (
        wp.ones(7, dtype=wp.float64, device=device),
        wp.ones(8, dtype=wp.float32, device=device),
        wp.ones(16, dtype=wp.float64, device=device)[::2],
    ):
        with test.assertRaises(ValueError):
            action.solve(bad, out)
        np.testing.assert_array_equal(out.numpy(), np.full(8, 7.0))
    buffer = wp.ones(9, dtype=wp.float64, device=device)
    with test.assertRaises(ValueError):
        action.solve(buffer[:8], buffer[1:])
    action.solve(out, out)
    np.testing.assert_allclose(out.numpy(), np.full(8, 7.0))
    with test.assertRaisesRegex(ValueError, "Unsupported residual mode"):
        _gmres(a, residual="FRESH")
    for capacity in (0, 2, True):
        with test.assertRaises((ValueError, TypeError)):
            action.with_nrhs(capacity)


def test_gmres_action_capacities_and_modes(test, device):
    """Check smaller capacities, update(), and recurrence versus fresh residual reporting."""
    n = 24
    a = np.diag(np.linspace(-3.0, 3.0, n) + 0.05) + np.diag(np.full(n - 1, 0.1), 1)
    matrix = _matrix(a, device, wp.float64, 1)
    rhs = np.random.default_rng(11).normal(size=(3, n))
    for mode in ("fresh", "recurrence"):
        action = _gmres(matrix, 3, restart=n, maxiter=2 * n, residual=mode)
        test.assertIs(action.residual, mode)
        small = action.with_nrhs(2)
        b = wp.array(rhs[:2], dtype=wp.float64, device=device)
        out = wp.empty_like(b)
        small.update()
        small.solve(b, out)
        np.testing.assert_allclose(out.numpy(), np.linalg.solve(a, rhs[:2].T).T, rtol=1e-9, atol=1e-9)
        true_residual = np.linalg.norm(out.numpy() @ a.T - rhs[:2], axis=1)
        np.testing.assert_array_equal(small.converged.numpy(), [1, 1])
        np.testing.assert_allclose(
            small.relative_residual.numpy(),
            small.absolute_residual.numpy() / np.linalg.norm(rhs[:2], axis=1),
            rtol=1e-12,
        )
        if mode == "fresh":
            np.testing.assert_allclose(small.absolute_residual.numpy(), true_residual, rtol=1e-6, atol=1e-13)
        # The parent's diagnostics are separate from the smaller capacity's.
        test.assertEqual(small.stats.numpy()[3], 2)
        test.assertEqual(action.stats.numpy()[3], 0)


def test_gmres_action_streams(test, device):
    """Check that eager solves issued from different streams share scratch safely."""
    n = 64
    a = np.diag(np.linspace(-3.0, 3.0, n) + 0.05) + np.diag(np.full(n - 1, 0.1), 1)
    action = _gmres(_matrix(a, device, wp.float64, 1), 2, restart=n, maxiter=2 * n)
    rhs = [np.random.default_rng(seed).normal(size=(2, n)) for seed in (5, 6)]
    inputs = [wp.array(r, dtype=wp.float64, device=device) for r in rhs]
    outputs = [wp.empty_like(b) for b in inputs]
    streams = [wp.Stream(device), wp.Stream(device)]
    for _ in range(4):
        for stream, b, out in zip(streams, inputs, outputs, strict=True):
            with wp.ScopedStream(stream, sync_enter=False, sync_exit=False):
                action.solve(b, out)
    wp.synchronize_device(device)
    for r, out in zip(rhs, outputs, strict=True):
        np.testing.assert_allclose(out.numpy(), np.linalg.solve(a, r.T).T, rtol=1e-9, atol=1e-9)


def test_gmres_action_default_tolerance(test, device):
    """Check that default tolerances follow the scalar type so float32 solves converge."""
    n = 24
    a = np.diag(np.linspace(1.0, 4.0, n)) + np.diag(np.full(n - 1, 0.3), 1)
    rhs = np.random.default_rng(3).normal(size=(2, n))
    for dtype, rtol in ((wp.float32, 1e-4), (wp.float64, 1e-10)):
        action = _gmres(_matrix(a, device, dtype, 1), 2, restart=n)
        b = wp.array(rhs, dtype=dtype, device=device)
        out = wp.empty_like(b)
        action.solve(b, out)
        np.testing.assert_array_equal(action.converged.numpy(), [1, 1])
        np.testing.assert_allclose(out.numpy(), np.linalg.solve(a, rhs.T).T, rtol=rtol, atol=rtol)


def test_gmres_action_block_diagonals(test, device):
    """Check that vector-valued diagonals of block systems count every scalar entry."""
    blocks = 5
    n = 3 * blocks
    a = np.diag(np.linspace(1.0, 4.0, n)) + np.diag(np.full(n - 1, 0.3), 1)
    rhs = np.random.default_rng(5).normal(size=(2, n))
    expected = np.linalg.solve(a, rhs.T).T
    matrix = _matrix(a, device, wp.float64, 3)
    for kind in ("diag", "diag_abs"):
        M = preconditioner(matrix, kind)
        test.assertEqual(M.shape, (blocks,))
        action = _gmres(matrix, 2, M=M, restart=n, tol=1e-12)
        b = wp.array(rhs, dtype=wp.float64, device=device)
        out = wp.empty_like(b)
        action.solve(b, out)
        np.testing.assert_array_equal(action.converged.numpy(), [1, 1])
        np.testing.assert_allclose(out.numpy(), expected, rtol=1e-9, atol=1e-9)
    diagonal = np.linspace(1.0, 4.0, n)
    action = _gmres(wp.array(diagonal.reshape(blocks, 3), dtype=wp.vec3d, device=device), 2, tol=1e-12)
    b = wp.array(rhs, dtype=wp.float64, device=device)
    out = wp.empty_like(b)
    action.solve(b, out)
    np.testing.assert_array_equal(action.converged.numpy(), [1, 1])
    np.testing.assert_allclose(out.numpy(), rhs / diagonal, rtol=1e-12)
    with test.assertRaises(ValueError):
        _gmres(matrix, M=wp.ones(blocks, dtype=wp.vec2d, device=device))


def test_gmres_action_stream_ordering(test, device):
    """Check that resets and smaller capacities join the stream ordering used by solves."""
    n = 32
    a = np.diag(np.linspace(1.0, 4.0, n)) + np.diag(np.full(n - 1, 0.3), 1)
    action = _gmres(_matrix(a, device, wp.float64, 1), 2, restart=n)
    rhs = np.random.default_rng(4).normal(size=(2, n))
    b = wp.array(rhs, dtype=wp.float64, device=device)
    out = wp.empty_like(b)
    first, second, third = wp.Stream(device), wp.Stream(device), wp.Stream(device)
    with wp.ScopedStream(first, sync_enter=False, sync_exit=False):
        action.solve(b, out)
    with wp.ScopedStream(second, sync_enter=False, sync_exit=False):
        action.reset_stats()
    # A later solve on the first stream must wait for the reset.
    test.assertIs(action._stream_order.last, second)
    with wp.ScopedStream(third, sync_enter=False, sync_exit=False):
        small = action.with_nrhs(1)
    test.assertIs(small._stream_order.last, third)
    with wp.ScopedStream(first, sync_enter=False, sync_exit=False):
        small.solve(b[:1], out[:1])
    wp.synchronize_device(device)
    np.testing.assert_array_equal(action.stats.numpy(), [0, 0, 0, 0])
    test.assertEqual(small.stats.numpy()[2:].tolist(), [0, 1])
    np.testing.assert_allclose(out.numpy()[0], np.linalg.solve(a, rhs[0]), rtol=1e-9)


def test_multiple_rhs(test, device, working_dtype):
    """Check solutions across layouts, widths, coefficient changes, aliases and smaller capacities."""
    n = 15
    a = np.diag(np.linspace(3.0, 5.0, n)) + np.diag(np.full(n - 1, -0.5), 1) + np.diag(np.full(n - 1, -0.5), -1)
    rng = np.random.default_rng(423)
    for block in (1, 3):
        for algorithm in _algorithms(device):
            matrix = _matrix(a, device, wp.float64, block)
            original = matrix.values.numpy().copy()
            for width in (1, 4, 7, 24, 33):
                solver = _cg(
                    matrix,
                    width,
                    working_dtype=working_dtype,
                    algorithm=algorithm,
                    tol=1e-5 if working_dtype == wp.float32 else 1e-12,
                    maxiter=40,
                )
                rhs = rng.normal(size=(width, n)) * np.geomspace(1e-3, 1e3, width)[:, None]
                if width > 1:
                    rhs[0] = 0
                b = wp.array(rhs, dtype=wp.float64, device=device)
                out = wp.empty_like(b)
                solver.solve(b, out)
                # Original coefficients, changed coefficients, and in-place output.
                for scale in (1.0, 1.7):
                    matrix.values.assign(original * scale)
                    b.assign(rhs)
                    solver.solve(b, b)
                    expected = np.linalg.solve(scale * a, rhs.T).T
                    error = np.linalg.norm(b.numpy() - expected, axis=1)
                    test.assertTrue(np.all(error <= 3e-5 * np.maximum(np.linalg.norm(expected, axis=1), 1e-8)))
                    test.assertEqual(solver.stats.numpy()[2], 0)
                matrix.values.assign(original)
                smaller = solver.with_nrhs(1)
                single_b = wp.array(rhs[-1:], dtype=wp.float64, device=device)
                smaller.solve(single_b, single_b)
                expected_single = np.linalg.solve(a, rhs[-1]).reshape(1, n)
                test.assertLessEqual(
                    np.linalg.norm(single_b.numpy() - expected_single),
                    3e-5 * max(np.linalg.norm(expected_single), 1e-8),
                )


def test_multiple_rhs_capture(test, device):
    """Check captured replay with prepared preconditioners and changing values."""
    n = 24
    a = np.diag(np.linspace(3.0, 5.0, n)) + np.diag(np.full(n - 1, -0.7), 1) + np.diag(np.full(n - 1, -0.7), -1)
    for block in (1, 3):
        matrix = _matrix(a, device, wp.float64, block)
        original = matrix.values.numpy().copy()
        pre = MultiRHSOperator(aslinearoperator(wp.array(1 / a.diagonal(), dtype=wp.float32, device=device)), 7)
        solver = _cg(matrix, 7, M=pre, working_dtype=wp.float32, tol=1e-5, maxiter=40)
        small = solver.with_nrhs(4, M=pre.with_nrhs(4))
        for action in (solver, small):
            b = wp.ones((action.nrhs, n), dtype=wp.float64, device=device)
            out = wp.empty_like(b)
            action.solve(b, out)
            graph = None
            if device.is_cuda:
                wp.synchronize_device(device)
                with wp.ScopedCapture(device=device, force_module_load=False) as capture:
                    action.reset_stats()
                    action.solve(b, out)
                graph = capture.graph
            for scale in (1.0, 2.0, 0.5):
                matrix.values.assign(original * scale)
                b.fill_(scale + 0.1)
                if graph:
                    wp.capture_launch(graph)
                else:
                    action.reset_stats()
                    action.solve(b, out)
                expected = np.linalg.solve(a * scale, np.full(n, scale + 0.1))
                np.testing.assert_allclose(out.numpy(), np.tile(expected, (action.nrhs, 1)), rtol=5e-5, atol=1e-6)
                test.assertEqual(action.stats.numpy()[2:].tolist(), [0, action.nrhs])
            if graph:
                with tempfile.TemporaryDirectory() as folder:
                    dot = Path(folder) / "cg.dot"
                    wp.capture_debug_dot_print(graph, str(dot))
                    for forbidden in ("HOST", "DtoH", "HtoD"):
                        test.assertNotIn(forbidden, dot.read_text())


def test_multiple_rhs_auto_row_count(test, device):
    """Check that prepared AUTO reads back row lengths while per-call AUTO does not."""
    from warp._src.sparse_multi_rhs import resolve  # noqa: PLC0415

    rows = 3000
    rng = np.random.default_rng(5)
    columns = rng.integers(0, rows, size=(rows, 40))
    row_index = np.repeat(np.arange(rows), 40)
    a = sp.bsr_from_triplets(
        rows,
        rows,
        wp.array(row_index, dtype=int, device=device),
        wp.array(columns.ravel(), dtype=int, device=device),
        wp.array(rng.normal(size=row_index.size), dtype=wp.float64, device=device),
    )
    operator = MultiRHSOperator(a, 8)
    expected = "columns" if wp.get_device(device).is_cuda else "rows"
    test.assertEqual(operator.algorithm, expected)
    test.assertEqual(operator.with_nrhs(2).algorithm, expected)
    test.assertEqual(resolve("auto", a, 8, wp.get_device(device), False), "rows")
    action = KrylovSolve(a, 8, method="gmres", maxiter=1)
    test.assertEqual(action.operator.algorithm, expected)


def test_mixed_precision_unsorted_columns(test, device):
    """Check that a mixed-precision solve keeps the source's slot order for unsorted CSR columns."""
    # A = [[2, 1], [0, 3]] with the first row stored as columns [1, 0]. A is nonsymmetric, so use GMRES.
    a = sp.bsr_zeros(2, 2, wp.float64, device=device)
    a.offsets = wp.array([0, 2, 3], dtype=int, device=device)
    a.columns = wp.array([1, 0, 1], dtype=int, device=device)
    a.values = wp.array([1.0, 2.0, 3.0], dtype=wp.float64, device=device)
    a.nnz = 3
    for residual in ("recurrence", "fresh"):
        action = _gmres(a, 1, working_dtype=wp.float32, tol=1e-6, residual=residual)
        b = wp.array([1.0, 2.0], dtype=wp.float64, device=device)
        out = wp.empty_like(b)
        action.solve(b, out)
        np.testing.assert_array_equal(action.converged.numpy(), [1])
        np.testing.assert_allclose(out.numpy(), [1.0 / 6.0, 2.0 / 3.0], rtol=1e-5)


def test_multiple_rhs_padded_mixed_precision(test, device):
    """Check that a padded source keeps its coefficients when copied to another working dtype."""
    n = 16
    a = np.diag(np.linspace(3.0, 5.0, n)) + np.diag(np.full(n - 1, -1.0), 1) + np.diag(np.full(n - 1, -1.0), -1)
    rows, cols = np.nonzero(a)
    padded = sp.bsr_zeros(n, n, wp.float64, device=device, row_capacity=5)
    sp.bsr_set_from_triplets(
        padded,
        wp.array(rows, dtype=int, device=device),
        wp.array(cols, dtype=int, device=device),
        wp.array(a[rows, cols], dtype=wp.float64, device=device),
        topology="padded",
    )
    rhs = np.random.default_rng(91).normal(size=(2, n))
    for working_dtype in (wp.float32, float):
        action = _cg(padded, 2, working_dtype=working_dtype, tol=1e-6, maxiter=100)
        test.assertEqual(action.working_dtype, wp.float32)
        b = wp.array(rhs, dtype=wp.float64, device=device)
        out = wp.empty_like(b)
        action.solve(b, out)
        np.testing.assert_array_equal(action.converged.numpy(), [1, 1])
        np.testing.assert_allclose(out.numpy(), np.linalg.solve(a, rhs.T).T, rtol=1e-4, atol=1e-5)


def test_multiple_rhs_operator(test, device):
    """Check aliased matvec semantics and that zero coefficients ignore nonfinite inputs."""
    a = np.diag(np.arange(1.0, 16.0)) + np.diag(np.full(14, 0.2), 1)
    for block in (1, 3):
        for algorithm in _algorithms(device):
            op = MultiRHSOperator(_matrix(a, device, wp.float64, block), 7, algorithm=algorithm)
            # The requested strategy must actually be selected.
            test.assertEqual(op.algorithm, algorithm)
            host = np.random.default_rng(17).normal(size=(7, 15))
            x = wp.array(host.ravel(), dtype=wp.float64, device=device)
            y = wp.array((0.3 * host).ravel(), dtype=wp.float64, device=device)
            op.matvec(x, y, x, 0.75, -0.25)
            np.testing.assert_allclose(
                x.numpy().reshape(host.shape), 0.75 * (host @ a.T) - 0.25 * 0.3 * host, rtol=1e-12, atol=1e-12
            )
            x.fill_(float("nan"))
            y.fill_(float("nan"))
            op.matvec(x, y, x, 0, 0)
            np.testing.assert_array_equal(x.numpy(), np.zeros(x.size))


def test_multiple_rhs_failure_validation(test, device):
    """Check argument validation and failure counting for exhausted or nonfinite solves."""
    a = _matrix(np.diag(np.arange(1.0, 16.0)), device, wp.float64, 1)
    for kwargs in (
        {"nrhs": 0},
        {"nrhs": True},
        {"nrhs": 2, "tol": float("nan")},
        {"nrhs": 2, "working_dtype": wp.float16},
        {"nrhs": 2, "maxiter": 0},
    ):
        with test.assertRaises((ValueError, TypeError)):
            _cg(a, **kwargs)
    with test.assertRaisesRegex(ValueError, "working dtype"):
        _cg(a, 2, working_dtype=wp.float32, M=aslinearoperator(wp.ones(15, dtype=wp.float64, device=device)))
    solver = _cg(a, 2, tol=1e-12, maxiter=1)
    b = wp.ones((2, 15), dtype=wp.float64, device=device)
    solver.solve(b, b)
    test.assertEqual(solver.stats.numpy().tolist(), [2, 1, 2, 2])
    out = wp.full((2, 15), 123.0, dtype=wp.float64, device=device)
    with test.assertRaises(ValueError):
        solver.solve(wp.ones((2, 15), dtype=wp.float32, device=device), out)
    np.testing.assert_array_equal(out.numpy(), np.full((2, 15), 123.0))
    b.fill_(float("nan"))
    solver.reset_stats()
    solver.solve(b, out)
    test.assertEqual(solver.stats.numpy()[2], 2)
    with test.assertRaises(TypeError):
        solver.with_nrhs(True)
    with test.assertRaises(TypeError):
        MultiRHSOperator(a, True)


def test_multiple_rhs_diagnostics(test, device):
    """Check per-RHS diagnostics for recurrence and fresh residuals with mixed precision."""
    n = 15
    a = np.diag(np.linspace(3.0, 5.0, n)) + np.diag(np.full(n - 1, -0.5), 1) + np.diag(np.full(n - 1, -0.5), -1)
    rhs = np.random.default_rng(5).normal(size=(3, n))
    rhs[1] = 0
    matrix = _matrix(a, device, wp.float64, 1)
    for mode in ("recurrence", "fresh"):
        action = _cg(matrix, 3, working_dtype=wp.float32, tol=1e-5, maxiter=40, residual=mode)
        test.assertIs(action.residual, mode)
        b = wp.array(rhs, dtype=wp.float64, device=device)
        action.update()
        action.solve(b, b)
        actual = b.numpy()
        true_residual = np.linalg.norm(actual @ a.T - rhs, axis=1)
        np.testing.assert_array_equal(action.converged.numpy(), [1, 1, 1])
        iterations = action.iterations.numpy()
        test.assertTrue(np.all(iterations == iterations[0]) and 0 < iterations[0] <= 40)
        test.assertEqual(action.relative_residual.numpy()[1], 0.0)
        # A zero RHS with atol=0 reports the documented float32 safety floor.
        np.testing.assert_allclose(action.tolerance.numpy()[1], 1e-18, rtol=1e-5)
        test.assertEqual(action.stats.numpy()[3], 3)
        if mode == "fresh":
            # Fresh residuals use the float64 source matrix and the returned float64 solution.
            np.testing.assert_allclose(action.absolute_residual.numpy(), true_residual, rtol=1e-6, atol=1e-12)
        test.assertTrue(np.all(action.absolute_residual.numpy() <= action.tolerance.numpy()))
        small = action.with_nrhs(1)
        single = wp.array(rhs[0], dtype=wp.float64, device=device)
        small.solve(single, single)
        np.testing.assert_allclose(single.numpy(), np.linalg.solve(a, rhs[0]), rtol=3e-5, atol=1e-7)
        test.assertEqual(small.stats.numpy()[3], 1)
    with test.assertRaisesRegex(ValueError, "Unsupported residual mode"):
        _cg(matrix, 3, residual="exact")


def test_multiple_rhs_warm_start(test, device):
    """Check that an initial guess, including the previous output, continues truncated solves."""
    n = 24
    a = np.diag(np.linspace(2.0, 5.0, n)) + np.diag(np.full(n - 1, -0.9), 1) + np.diag(np.full(n - 1, -0.9), -1)
    rhs = np.random.default_rng(11).normal(size=(2, n))
    expected = np.linalg.solve(a, rhs.T).T
    matrix = _matrix(a, device, wp.float64, 1)
    for method in ("cg", "gmres"):
        for residual in ("recurrence", "fresh"):
            action = KrylovSolve(
                matrix,
                2,
                method=method,
                tol=1e-10,
                maxiter=3,
                restart=None if method == "cg" else 3,
                residual=residual,
            )
            b = wp.array(rhs, dtype=wp.float64, device=device)
            out = wp.zeros_like(b)
            action.solve(b, out)
            first = action.absolute_residual.numpy().copy()
            test.assertFalse(np.any(action.converged.numpy()))
            action.solve(b, out, x0=out)
            second = action.absolute_residual.numpy()
            test.assertTrue(np.all(second < 0.5 * first), msg=f"{method} {residual}: {first} -> {second}")
            exact = wp.array(expected, dtype=wp.float64, device=device)
            action.solve(b, out, exact)
            np.testing.assert_array_equal(action.converged.numpy(), [1, 1])
            np.testing.assert_allclose(out.numpy(), expected, rtol=1e-8, atol=1e-10)
            # Tolerances stay relative to ||b||, not to the small initial residual.
            np.testing.assert_allclose(action.tolerance.numpy(), 1e-10 * np.linalg.norm(rhs, axis=1), rtol=1e-6)
    with test.assertRaises(ValueError):
        action.solve(b, out, x0=wp.zeros(n, dtype=wp.float64, device=device))
    with test.assertRaises(ValueError):
        action.solve(b, out, x0=wp.zeros((2, n), dtype=wp.float32, device=device))


def test_multiple_rhs_without_diagnostics(test, device):
    """Check that diagnostics=False returns device results without touching the diagnostic attributes."""
    n = 15
    a = np.diag(np.linspace(3.0, 5.0, n)) + np.diag(np.full(n - 1, -0.5), 1) + np.diag(np.full(n - 1, -0.5), -1)
    rhs = np.random.default_rng(13).normal(size=(3, n))
    matrix = _matrix(a, device, wp.float64, 1)
    action = _cg(matrix, 3, working_dtype=wp.float32, tol=1e-5, maxiter=40)
    b = wp.array(rhs, dtype=wp.float64, device=device)
    out = wp.empty_like(b)
    iterations, residual_sq, tolerance_sq = action.solve(b, out, diagnostics=False)
    np.testing.assert_allclose(out.numpy(), np.linalg.solve(a, rhs.T).T, rtol=3e-5, atol=1e-6)
    test.assertTrue(0 < iterations.numpy()[0] <= 40)
    test.assertTrue(np.all(residual_sq.numpy() <= tolerance_sq.numpy()))
    np.testing.assert_allclose(tolerance_sq.numpy(), (1e-5 * np.linalg.norm(rhs, axis=1)) ** 2, rtol=1e-4)
    for name in ("iterations", "converged", "stats", "absolute_residual", "residual_maxima"):
        test.assertFalse(np.any(getattr(action, name).numpy()), msg=name)
    returned = action.solve(b, out)
    test.assertEqual(action.stats.numpy()[3], 3)
    np.testing.assert_allclose(action.absolute_residual.numpy() ** 2, returned[1].numpy(), rtol=1e-5)
    # The relative residual uses the RHS norm that the solver stored while setting tolerances.
    np.testing.assert_allclose(action.state.rhs_norm_sq.numpy(), np.sum(rhs * rhs, axis=1), rtol=1e-6)
    np.testing.assert_allclose(
        action.relative_residual.numpy(), action.absolute_residual.numpy() / np.linalg.norm(rhs, axis=1), rtol=1e-5
    )
    fresh = _cg(matrix, 3, tol=1e-5, residual="fresh")
    with test.assertRaises(ValueError):
        fresh.solve(b, out, diagnostics=False)


def test_solver_state_rhs_norm(test, device):
    """Check that every functional solver state stores the squared RHS norm when it runs."""
    from warp.optim.linear import bicgstab, cg, cr, gmres  # noqa: PLC0415

    n = 12
    a = np.diag(np.linspace(3.0, 5.0, n)) + np.diag(np.full(n - 1, -0.5), 1) + np.diag(np.full(n - 1, -0.5), -1)
    rhs = np.random.default_rng(17).normal(size=n)
    matrix = _matrix(a, device, wp.float64, 1)
    for solver in (cg, cr, bicgstab, gmres):
        b = wp.array(rhs, dtype=wp.float64, device=device)
        x = wp.zeros_like(b)
        state = solver(matrix, b, x, tol=1e-8, check_every=0, run=False)
        np.testing.assert_array_equal(state.rhs_norm_sq.numpy(), [0.0])
        state()
        np.testing.assert_allclose(state.rhs_norm_sq.numpy(), [rhs @ rhs], rtol=1e-12, err_msg=solver.__name__)


def test_multiple_rhs_storage_change(test, device):
    """Check that reallocating the source matrix raises for every working precision."""
    a = np.diag(np.arange(1.0, 16.0))
    for working_dtype in (wp.float32, wp.float64):
        matrix = _matrix(a, device, wp.float64, 1)
        solver = _cg(matrix, 2, working_dtype=working_dtype, tol=1e-5, maxiter=20)
        b = wp.ones((2, 15), dtype=wp.float64, device=device)
        out = wp.empty_like(b)
        solver.solve(b, out)
        matrix.values = wp.clone(matrix.values)
        with test.assertRaisesRegex(ValueError, "storage changed"):
            solver.update()
        with test.assertRaisesRegex(ValueError, "storage changed"):
            solver.solve(b, out)
        # Row counts are part of the storage too, including switching between compact and padded rows.
        matrix = _matrix(a, device, wp.float64, 1)
        solver = _cg(matrix, 2, working_dtype=working_dtype, tol=1e-5, maxiter=20)
        for row_counts in (wp.ones(15, dtype=int, device=device), wp.ones(15, dtype=int, device=device)):
            matrix.row_counts = row_counts
            with test.assertRaisesRegex(ValueError, "storage changed"):
                solver.solve(b, out)


def test_multiple_rhs_default_tolerance(test, device):
    """Check that default tolerances follow the working precision so float32 solves converge."""
    n = 15
    a = np.diag(np.linspace(3.0, 5.0, n)) + np.diag(np.full(n - 1, -0.5), 1) + np.diag(np.full(n - 1, -0.5), -1)
    rhs = np.random.default_rng(8).normal(size=(2, n))
    for dtype, rtol in ((wp.float32, 1e-4), (wp.float64, 1e-10)):
        # Fresh residuals expose the accuracy float32 actually attains.
        action = _cg(_matrix(a, device, dtype, 1), 2, residual="fresh")
        b = wp.array(rhs, dtype=dtype, device=device)
        out = wp.empty_like(b)
        action.solve(b, out)
        np.testing.assert_array_equal(action.converged.numpy(), [1, 1])
        np.testing.assert_allclose(out.numpy(), np.linalg.solve(a, rhs.T).T, rtol=rtol, atol=rtol)


def test_multiple_rhs_stream_ordering(test, device):
    """Check that explicit updates, resets and smaller capacities join the solve stream ordering."""
    n = 15
    a = np.diag(np.linspace(3.0, 5.0, n)) + np.diag(np.full(n - 1, -0.5), 1) + np.diag(np.full(n - 1, -0.5), -1)
    rhs = np.random.default_rng(9).normal(size=(2, n))
    action = _cg(_matrix(a, device, wp.float64, 1), 2, working_dtype=wp.float32, tol=1e-5, maxiter=40)
    b = wp.array(rhs, dtype=wp.float64, device=device)
    out = wp.empty_like(b)
    first, second, third, fourth = (wp.Stream(device) for _ in range(4))
    with wp.ScopedStream(first, sync_enter=False, sync_exit=False):
        action.solve(b, out)
    # Each later call must wait for earlier work on the stream it was issued from.
    with wp.ScopedStream(second, sync_enter=False, sync_exit=False):
        action.update()
    test.assertIs(action._stream_order.last, second)
    with wp.ScopedStream(third, sync_enter=False, sync_exit=False):
        action.reset_stats()
    test.assertIs(action._stream_order.last, third)
    with wp.ScopedStream(fourth, sync_enter=False, sync_exit=False):
        small = action.with_nrhs(1)
    test.assertIs(small._stream_order.last, fourth)
    with wp.ScopedStream(first, sync_enter=False, sync_exit=False):
        small.solve(b[:1], out[:1])
    wp.synchronize_device(device)
    np.testing.assert_array_equal(action.stats.numpy(), [0, 0, 0, 0])
    test.assertEqual(small.stats.numpy()[2:].tolist(), [0, 1])
    np.testing.assert_allclose(out.numpy()[0], np.linalg.solve(a, rhs[0]), rtol=3e-5, atol=1e-7)


def test_multiple_rhs_streams(test, device):
    """Check that eager solves from different streams share scratch and capacities safely."""
    n = 64
    a = np.diag(np.linspace(3.0, 5.0, n)) + np.diag(np.full(n - 1, -0.5), 1) + np.diag(np.full(n - 1, -0.5), -1)
    solver = _cg(_matrix(a, device, wp.float64, 1), 4, tol=1e-12, maxiter=200)
    smaller = solver.with_nrhs(2)
    rhs = [np.random.default_rng(seed).normal(size=(width, n)) for seed, width in ((3, 4), (4, 2))]
    inputs = [wp.array(r, dtype=wp.float64, device=device) for r in rhs]
    outputs = [wp.empty_like(b) for b in inputs]
    streams = [wp.Stream(device), wp.Stream(device)]
    for _ in range(4):
        for stream, action, b, out in zip(streams, (solver, smaller), inputs, outputs, strict=True):
            with wp.ScopedStream(stream, sync_enter=False, sync_exit=False):
                action.solve(b, out)
    wp.synchronize_device(device)
    for r, out in zip(rhs, outputs, strict=True):
        np.testing.assert_allclose(out.numpy(), np.linalg.solve(a, r.T).T, rtol=1e-9, atol=1e-9)


def test_multiple_rhs_float32_source(test, device):
    """Check float32 sources with float32 and float64 working precision and block Jacobi."""
    n = 15
    a = np.diag(np.linspace(3.0, 5.0, n)) + np.diag(np.full(n - 1, -0.5), 1) + np.diag(np.full(n - 1, -0.5), -1)
    rhs = np.random.default_rng(73).normal(size=(4, n)).astype(np.float32)
    for block in (1, 3):
        matrix = _matrix(a, device, wp.float32, block)
        for dtype in (wp.float32, wp.float64):
            working = sp.bsr_copy(matrix, scalar_type=dtype)
            pre = preconditioner(working, ptype="block_jacobi_direct")
            action = _cg(matrix, 4, M=pre, working_dtype=dtype, tol=1e-6, maxiter=40)
            b = wp.array(rhs, dtype=wp.float32, device=device)
            out = wp.empty_like(b)
            action.solve(b, out)
            actual = out.numpy()
            relative = np.linalg.norm(actual @ a.astype(np.float32).T - rhs, axis=1) / np.linalg.norm(rhs, axis=1)
            test.assertLess(relative.max(), 3e-6)
            test.assertEqual(action.stats.numpy()[2], 0)


def test_multiple_rhs_tiny_rhs(test, device):
    """Check that float64 working precision judges tiny float32 RHSs without underflow."""
    n = 16
    a = np.diag(np.linspace(3.0, 5.0, n)) + np.diag(np.full(n - 1, -1.0), 1) + np.diag(np.full(n - 1, -1.0), -1)
    rhs = (np.random.default_rng(81).normal(size=(2, n)) * 1e-24).astype(np.float32)
    matrix = _matrix(a, device, wp.float32, 1)
    for residual in ("recurrence", "fresh"):
        b = wp.array(rhs, dtype=wp.float32, device=device)
        out = wp.empty_like(b)
        unfinished = _cg(matrix, 2, working_dtype=wp.float64, tol=1e-6, maxiter=1, residual=residual)
        unfinished.solve(b, out)
        np.testing.assert_array_equal(unfinished.converged.numpy(), [0, 0])
        test.assertTrue(np.all(unfinished.relative_residual.numpy() > 1e-3))
        action = _cg(matrix, 2, working_dtype=wp.float64, tol=1e-6, maxiter=100, residual=residual)
        action.solve(b, out)
        np.testing.assert_array_equal(action.converged.numpy(), [1, 1])
        np.testing.assert_allclose(out.numpy(), np.linalg.solve(a, rhs.T.astype(np.float64)).T, rtol=1e-4)


def test_multiple_rhs_block_diagonal(test, device):
    """Check that vector-valued diagonal preconditioners count every scalar of a block system."""
    n = 15
    a = np.diag(np.linspace(3.0, 5.0, n)) + np.diag(np.full(n - 1, -0.5), 1) + np.diag(np.full(n - 1, -0.5), -1)
    rhs = np.random.default_rng(83).normal(size=(2, n))
    matrix = _matrix(a, device, wp.float64, 3)
    M = preconditioner(matrix, ptype="diag")
    test.assertEqual(M.shape, (n // 3,))
    operator = MultiRHSOperator(M, 2)
    test.assertEqual(operator.shape, (2 * n, 2 * n))
    x = wp.ones(2 * n, dtype=wp.float64, device=device)
    y = wp.empty_like(x)
    operator.matvec(x, y, y, 1.0, 0.0)
    np.testing.assert_allclose(y.numpy(), np.tile(1.0 / np.diag(a), 2), rtol=1e-12)
    action = _cg(matrix, 2, M=M, tol=1e-10)
    b = wp.array(rhs, dtype=wp.float64, device=device)
    out = wp.empty_like(b)
    action.solve(b, out)
    np.testing.assert_array_equal(action.converged.numpy(), [1, 1])
    np.testing.assert_allclose(out.numpy(), np.linalg.solve(a, rhs.T).T, rtol=1e-8, atol=1e-8)


class TestSolveAction(unittest.TestCase):
    pass


for dtype in (wp.float32, wp.float64):
    add_function_test(
        TestSolveAction,
        f"test_gmres_action_{dtype.__name__}",
        test_gmres_action,
        devices=get_test_devices(),
        dtype=dtype,
    )
    add_function_test(
        TestSolveAction,
        f"test_multiple_rhs_{dtype.__name__}",
        test_multiple_rhs,
        devices=get_test_devices(),
        working_dtype=dtype,
    )
for func in (
    test_krylov_methods,
    test_gmres_action_failures,
    test_gmres_action_validation,
    test_gmres_action_capacities_and_modes,
    test_gmres_action_default_tolerance,
    test_gmres_action_block_diagonals,
    test_multiple_rhs_capture,
    test_multiple_rhs_operator,
    test_multiple_rhs_auto_row_count,
    test_multiple_rhs_padded_mixed_precision,
    test_mixed_precision_unsorted_columns,
    test_multiple_rhs_failure_validation,
    test_multiple_rhs_tiny_rhs,
    test_multiple_rhs_block_diagonal,
    test_multiple_rhs_float32_source,
    test_multiple_rhs_diagnostics,
    test_multiple_rhs_warm_start,
    test_multiple_rhs_without_diagnostics,
    test_solver_state_rhs_norm,
    test_multiple_rhs_storage_change,
    test_multiple_rhs_default_tolerance,
):
    add_function_test(TestSolveAction, func.__name__, func, devices=get_test_devices())
for func in (
    test_gmres_action_streams,
    test_gmres_action_stream_ordering,
    test_multiple_rhs_streams,
    test_multiple_rhs_stream_ordering,
):
    add_function_test(TestSolveAction, func.__name__, func, devices=get_cuda_test_devices())

if __name__ == "__main__":
    unittest.main(verbosity=2)
