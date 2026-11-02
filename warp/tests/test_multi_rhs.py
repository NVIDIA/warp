# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest
from pathlib import Path

import numpy as np

import warp as wp
import warp.sparse as sp
from warp.optim.linear import CGSolve, MultiRHSOperator, ResidualMode, aslinearoperator, preconditioner
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


def test_multiple_rhs(test, device, working_dtype):
    """Check solutions across layouts, widths, coefficient changes, aliases and smaller capacities."""
    n = 15
    a = np.diag(np.linspace(3.0, 5.0, n)) + np.diag(np.full(n - 1, -0.5), 1) + np.diag(np.full(n - 1, -0.5), -1)
    rng = np.random.default_rng(423)
    for block in (1, 3):
        for layout in ("rows", "columns"):
            matrix = _matrix(a, device, wp.float64, block)
            original = matrix.values.numpy().copy()
            for width in (1, 4, 7, 24, 33):
                solver = CGSolve(
                    matrix,
                    width,
                    working_dtype=working_dtype,
                    layout=layout,
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
        solver = CGSolve(matrix, 7, M=pre, working_dtype=wp.float32, tol=1e-5, maxiter=40)
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


def test_multiple_rhs_operator(test, device):
    """Check aliased matvec semantics and that zero coefficients ignore nonfinite inputs."""
    a = np.diag(np.arange(1.0, 16.0)) + np.diag(np.full(14, 0.2), 1)
    for block in (1, 3):
        for layout in ("rows", "columns"):
            op = MultiRHSOperator(_matrix(a, device, wp.float64, block), 7, layout=layout)
            # The coalesced path must actually be selected on CUDA; CPU always uses rows.
            test.assertEqual(op.layout, layout if device.is_cuda else "rows")
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
            CGSolve(a, **kwargs)
    with test.assertRaisesRegex(ValueError, "working dtype"):
        CGSolve(a, 2, working_dtype=wp.float32, M=aslinearoperator(wp.ones(15, dtype=wp.float64, device=device)))
    solver = CGSolve(a, 2, tol=1e-12, maxiter=1)
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
    for mode in (ResidualMode.RECURRENCE, ResidualMode.FRESH):
        action = CGSolve(matrix, 3, working_dtype=wp.float32, tol=1e-5, maxiter=40, residual=mode)
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
        test.assertEqual(action.stats.numpy()[3], 3)
        if mode == ResidualMode.FRESH:
            # Fresh residuals use the float64 source matrix and the returned float64 solution.
            np.testing.assert_allclose(action.absolute_residual.numpy(), true_residual, rtol=1e-6, atol=1e-12)
        test.assertTrue(np.all(action.absolute_residual.numpy() <= action.tolerance.numpy()))
        small = action.with_nrhs(1)
        single = wp.array(rhs[0], dtype=wp.float64, device=device)
        small.solve(single, single)
        np.testing.assert_allclose(single.numpy(), np.linalg.solve(a, rhs[0]), rtol=3e-5, atol=1e-7)
        test.assertEqual(small.stats.numpy()[3], 1)
    with test.assertRaises(TypeError):
        CGSolve(matrix, 3, residual="fresh")


def test_multiple_rhs_storage_change(test, device):
    """Check that reallocating a mixed-precision source matrix raises instead of using stale data."""
    a = np.diag(np.arange(1.0, 16.0))
    matrix = _matrix(a, device, wp.float64, 1)
    solver = CGSolve(matrix, 2, working_dtype=wp.float32, tol=1e-5, maxiter=20)
    b = wp.ones((2, 15), dtype=wp.float64, device=device)
    out = wp.empty_like(b)
    solver.solve(b, out)
    matrix.values = wp.clone(matrix.values)
    with test.assertRaisesRegex(ValueError, "storage changed"):
        solver.solve(b, out)


def test_multiple_rhs_streams(test, device):
    """Check that eager solves from different streams share scratch and capacities safely."""
    n = 64
    a = np.diag(np.linspace(3.0, 5.0, n)) + np.diag(np.full(n - 1, -0.5), 1) + np.diag(np.full(n - 1, -0.5), -1)
    solver = CGSolve(_matrix(a, device, wp.float64, 1), 4, tol=1e-12, maxiter=200)
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
            action = CGSolve(matrix, 4, M=pre, working_dtype=dtype, tol=1e-6, maxiter=40)
            b = wp.array(rhs, dtype=wp.float32, device=device)
            out = wp.empty_like(b)
            action.solve(b, out)
            actual = out.numpy()
            relative = np.linalg.norm(actual @ a.astype(np.float32).T - rhs, axis=1) / np.linalg.norm(rhs, axis=1)
            test.assertLess(relative.max(), 3e-6)
            test.assertEqual(action.stats.numpy()[2], 0)


class TestMultiRHS(unittest.TestCase):
    pass


for dtype in (wp.float32, wp.float64):
    add_function_test(
        TestMultiRHS,
        f"test_multiple_rhs_{dtype.__name__}",
        test_multiple_rhs,
        devices=get_test_devices(),
        working_dtype=dtype,
    )
for func in (test_multiple_rhs_capture, test_multiple_rhs_operator, test_multiple_rhs_failure_validation):
    add_function_test(TestMultiRHS, func.__name__, func, devices=get_test_devices())

add_function_test(
    TestMultiRHS, "test_multiple_rhs_float32_source", test_multiple_rhs_float32_source, devices=get_test_devices()
)
add_function_test(
    TestMultiRHS, "test_multiple_rhs_diagnostics", test_multiple_rhs_diagnostics, devices=get_test_devices()
)
add_function_test(
    TestMultiRHS, "test_multiple_rhs_storage_change", test_multiple_rhs_storage_change, devices=get_test_devices()
)
add_function_test(TestMultiRHS, "test_multiple_rhs_streams", test_multiple_rhs_streams, devices=get_cuda_test_devices())

if __name__ == "__main__":
    unittest.main(verbosity=2)
