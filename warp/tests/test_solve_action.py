# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest
from pathlib import Path

import numpy as np

import warp as wp
import warp.sparse as sp
from warp.optim.linear import GMRESSolve, aslinearoperator
from warp.tests.unittest_utils import add_function_test, get_test_devices


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


def test_gmres_action(test, device, dtype):
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
                action = GMRESSolve(
                    matrix, 3, M=pre, restart=n, maxiter=3 * n, tol=1e-5 if dtype == wp.float32 else 1e-12
                )
                b = wp.zeros((3, n), dtype=dtype, device=device)
                for alias in (False, True):
                    out = b if alias else wp.empty_like(b)
                    action.solve(b, out)
                    graph = None
                    if device.is_cuda:
                        wp.synchronize_device(device)
                        with wp.ScopedCapture(device=device) as capture:
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
    a = np.diag(np.arange(-16.0, 16.0) + 0.2)
    action = GMRESSolve(_matrix(a, device, wp.float64, 1), 2, restart=1, maxiter=1)
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
    ):
        with test.assertRaises((ValueError, TypeError)):
            GMRESSolve(a, **kwargs)
    action = GMRESSolve(a)
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
for func in (test_gmres_action_failures, test_gmres_action_validation):
    add_function_test(TestSolveAction, func.__name__, func, devices=get_test_devices())

if __name__ == "__main__":
    unittest.main(verbosity=2)
