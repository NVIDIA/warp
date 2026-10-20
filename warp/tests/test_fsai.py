# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools
import unittest
from unittest import mock

import numpy as np

import warp as wp
import warp.sparse as sp
from warp._src.optim import fsai as fsai_module
from warp.optim.linear import FSAI, cg, cr, preconditioner
from warp.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices


def _matrix(a, device, dtype=wp.float64, block=1):
    rows, cols = np.nonzero(a)
    result = sp.bsr_from_triplets(
        a.shape[0],
        a.shape[1],
        wp.array(rows, dtype=int, device=device),
        wp.array(cols, dtype=int, device=device),
        wp.array(a[rows, cols], dtype=dtype, device=device),
    )
    return sp.bsr_copy(result, block_shape=(block, block)) if block != 1 else result


def _dense(a):
    offsets, columns, values = a.offsets.numpy(), a.columns.numpy(), a.values.numpy()
    counts = None if a.row_counts is None else a.row_counts.numpy()
    result = np.zeros(a.shape, dtype=values.dtype)
    for row in range(a.nrow):
        start = offsets[row]
        end = offsets[row + 1] if counts is None else start + counts[row]
        result[row, columns[start:end]] = values[start:end]
    return result


def _path(n):
    return np.diag(np.full(n, 3.0)) + np.diag(-np.ones(n - 1), 1) + np.diag(-np.ones(n - 1), -1)


def _check_local_solves(test, pre, a, rtol=1e-10, atol=1e-12):
    g = _dense(pre.G)
    offsets, columns = pre.G.offsets.numpy(), pre.G.columns.numpy()
    np.testing.assert_array_equal(_dense(pre.GT), g.T)
    np.testing.assert_array_equal(g, np.tril(g))
    test.assertTrue(np.all(np.isfinite(g)))
    test.assertTrue(np.all(np.diag(g) > 0))
    for row in range(len(a)):
        support = columns[offsets[row] : offsets[row + 1]]
        test.assertEqual(support[-1], row)
        test.assertTrue(np.all(np.diff(support) > 0))
        rhs = np.zeros(len(support))
        rhs[-1] = 1
        z = np.linalg.solve(a[np.ix_(support, support)], rhs)
        np.testing.assert_allclose(g[row, support], z / np.sqrt(z[-1]), rtol=rtol, atol=atol)
    return g


def test_full_pattern(test, device, dtype):
    """Check that an unrestricted pattern reproduces the exact inverse and local solves."""
    rng = np.random.default_rng(7)
    a = rng.normal(size=(8, 8))
    a = a @ a.T + np.eye(8)
    for step_size in (1, 3):
        pre = FSAI(_matrix(a, device, dtype), max_row_size=8, kap_tolerance=0, max_step_size=step_size)
        g = _check_local_solves(test, pre, a, rtol=2e-5, atol=1e-6)
        np.testing.assert_allclose(g.T @ g, np.linalg.inv(a), rtol=2e-5, atol=1e-6)


def test_adaptive_support(test, device, dtype):
    """Check adaptive support selection against an independent dense greedy search."""
    rng = np.random.default_rng(193)
    a = rng.normal(size=(18, 18))
    a[rng.random(a.shape) < 0.65] = 0
    a = a + a.T
    np.fill_diagonal(a, np.sum(np.abs(a), axis=1) + 2)
    scaling = np.geomspace(0.01, 100, len(a))
    a *= scaling[:, None] * scaling[None, :]
    raw = _matrix(a, device, dtype, block=3)
    equilibrated = a / np.sqrt(np.diag(a)[:, None] * np.diag(a)[None, :])
    for step_size in (1, 3):
        for tolerance in (0.0, 0.003, 0.1):
            pre = FSAI(raw, max_row_size=8, kap_tolerance=tolerance, max_step_size=step_size)
            g = _dense(pre.G)
            # Independent dense greedy search; neither factor values nor supports
            # are obtained from the implementation under test.
            for i in range(len(a)):
                support = [i]
                z = np.ones(1)
                while len(support) < 8:
                    frontier = sorted(set(np.flatnonzero(np.any(a[support] != 0, axis=0))) - set(support))
                    frontier = [c for c in frontier if c < i]
                    scores = [abs(equilibrated[c, support] @ z) for c in frontier]
                    ranked = sorted(zip(scores, frontier, strict=True), key=lambda item: (-item[0], item[1]))
                    selected = [c for score, c in ranked if score > 0][: min(step_size, 8 - len(support))]
                    if not selected:
                        break
                    support.extend(selected)
                    old = z[0]
                    rhs = np.zeros(len(support))
                    rhs[0] = 1
                    z = np.linalg.solve(equilibrated[np.ix_(support, support)], rhs)
                    if tolerance > 0 and 1 - old / z[0] <= tolerance:
                        break
                expected = np.zeros(len(a))
                expected[support] = z / np.sqrt(z[0] * np.diag(a)[support])
                np.testing.assert_allclose(g[i], expected, rtol=3e-5, atol=1e-6)


def test_narrow_frontier_and_zero_tolerance(test, device):
    """Check that short frontiers and a disabled tolerance do not stop growth early."""
    # The frontier has one candidate. A short batch must not use a full batch's
    # budget; improvement rounding to zero must not stop a disabled tolerance.
    a = _path(32)
    for step_size in (1, 3):
        pre = FSAI(_matrix(a, device), max_row_size=32, max_step_size=step_size, kap_tolerance=0)
        np.testing.assert_array_equal(np.diff(pre.G.offsets.numpy()), np.arange(1, 33))
        _check_local_solves(test, pre, a)


def test_pivot_guard(test, device):
    """Check that a rejected pivot truncates the row and is counted."""
    a = np.array([[1.0, 1 - 1e-14, 0], [1 - 1e-14, 1.0, 0], [0, 0, 2.0]])
    for width in (1, 2):
        pre = FSAI(_matrix(a, device), max_row_size=width, max_step_size=6)
        test.assertEqual(pre.truncated_rows, 0 if width == 1 else 1)
        np.testing.assert_allclose(_dense(pre.G), np.diag(1 / np.sqrt(np.diag(a))))


def test_application(test, device, dtype):
    """Check ``G.T @ G`` application across storage types and aliasing."""
    rng = np.random.default_rng(18)
    a = _path(35)
    raw = _matrix(a, device, dtype)
    tol = 5e-3 if dtype == wp.float16 else 3e-6 if dtype == wp.float32 else 2e-12
    # Include both ordinary and compressed factors.
    for storage in (dtype,) if dtype == wp.float16 else dict.fromkeys((dtype, wp.float32)):
        pre = FSAI(raw, max_row_size=5, factor_dtype=storage)
        g = _dense(pre.G).astype(np.float64)
        np.testing.assert_array_equal(_dense(pre.GT), g.T)
        for alias in ("none", "xz", "yz", "xy", "xyz"):
            x = wp.array(rng.normal(size=len(a)), dtype=dtype, device=device)
            y = x if alias in ("xy", "xyz") else wp.array(rng.normal(size=len(a)), dtype=dtype, device=device)
            z = x if alias in ("xz", "xyz") else y if alias == "yz" else wp.empty_like(x)
            expected = 0.7 * g.T @ g @ x.numpy() - 1.2 * y.numpy()
            pre.matvec(x, y, z, 0.7, -1.2)
            np.testing.assert_allclose(z.numpy(), expected, rtol=tol, atol=tol)
        x = wp.ones(len(a), dtype=dtype, device=device)
        y = wp.empty_like(x)
        z = wp.empty_like(x)
        y.fill_(float("nan"))
        pre.matvec(x, y, z, 1, 0)
        np.testing.assert_allclose(z.numpy(), g.T @ g @ np.ones(len(a)), rtol=tol, atol=tol)
        x.fill_(float("nan"))
        y.fill_(2)
        pre.matvec(x, y, z, 0, 0.5)
        np.testing.assert_array_equal(z.numpy(), 1)


def test_solver_integration(test, device, dtype):
    """Check CG and CR convergence with FSAI against the true residual."""
    a = _path(24)
    tolerance = 1e-5 if dtype == wp.float32 else 1e-11
    for block in (1, 3):
        raw = _matrix(a, device, dtype, block)
        jacobi = FSAI(raw, max_row_size=1)
        np.testing.assert_allclose(_dense(jacobi.G), np.eye(len(a)) / np.sqrt(3), rtol=1e-6)
        pre = preconditioner(raw, "fsai")
        test.assertIsInstance(pre, FSAI)
        vector = dtype if block == 1 else wp.types.vector(block, dtype)
        b = wp.ones(len(a) // block, dtype=vector, device=device)
        for solver in (cg, cr):
            x = wp.zeros_like(b)
            with wp.ScopedDevice(device):
                solver(raw, b, x, M=pre, tol=tolerance, maxiter=100)
            # Check the true residual, independently of the solver's report.
            np.testing.assert_allclose(a @ x.numpy().ravel(), 1.0, atol=4 * tolerance, rtol=0)


def test_refit_and_rollback(test, device, dtype):
    """Check that refits reuse storage and failed updates leave the factor unchanged."""
    rng = np.random.default_rng(48)
    r = rng.normal(size=(12, 12))
    a = r @ r.T + np.eye(12) * 3
    b = a + np.diag(np.arange(12))
    for block in (1, 3):
        for storage in (dtype,) if dtype == wp.float16 else dict.fromkeys((dtype, wp.float32)):
            raw, updated = _matrix(a, device, dtype, block), _matrix(b, device, dtype, block)
            pre = FSAI(raw, max_row_size=5, reuse_pattern=True, factor_dtype=storage, max_step_size=3)
            pointers = (pre.G.values.ptr, pre.GT.values.ptr)
            topology = (pre.G.offsets.numpy().copy(), pre.G.columns.numpy().copy())
            test.assertIs(pre.update(updated), pre)
            rtol, atol = (8e-3, 5e-4) if dtype == wp.float16 else (2e-4, 2e-7)
            g = _check_local_solves(test, pre, b, rtol=rtol, atol=atol)
            test.assertEqual((pre.G.values.ptr, pre.GT.values.ptr), pointers)
            np.testing.assert_array_equal(pre.G.offsets.numpy(), topology[0])
            np.testing.assert_array_equal(pre.G.columns.numpy(), topology[1])
            for failure in ("diagonal", "pivot", "topology"):
                invalid = sp.bsr_copy(updated)
                if failure == "diagonal":
                    invalid.values.zero_()
                elif failure == "pivot":
                    # Positive diagonals are not enough: every selected pair is indefinite.
                    indefinite = np.full(a.shape, 10.0)
                    np.fill_diagonal(indefinite, 1.0)
                    invalid = _matrix(indefinite, device, dtype, block)
                else:
                    columns = invalid.columns.numpy().copy()
                    columns[0] += 1
                    invalid.columns.assign(columns)
                with test.assertRaises(ValueError):
                    pre.update(invalid)
                test.assertIs(pre.source, updated)
                np.testing.assert_array_equal(_dense(pre.G), g)
                np.testing.assert_array_equal(_dense(pre.GT), g.T)
            # Same object, changing values, then no-argument update.
            updated.values.assign(raw.values.numpy())
            pre.update()
            _check_local_solves(test, pre, a, rtol=rtol, atol=atol)


def test_stored_zeros_and_capacity(test, device):
    """Check that stored zeros are retained and unused capacity is ignored."""
    a = _path(3)
    a[0, 2] = a[2, 0] = 0
    raw = _matrix(a, device, block=3)
    count = raw.nnz_sync()
    columns, values = raw.columns, raw.values
    raw.columns = wp.full(16, -123, dtype=int, device=device)
    raw.values = wp.empty(16, dtype=raw.dtype, device=device)
    raw.values.fill_(float("nan"))
    wp.copy(raw.columns, columns, count=count)
    wp.copy(raw.values, values, count=count)
    raw.notify_nnz_changed(nnz=16)
    # Unused capacity must not enter topology checks, scalarization or refits.
    pre = FSAI(raw, max_row_size=3, kap_tolerance=0, reuse_pattern=True)
    updated = _matrix(a + 0.2 * np.ones((3, 3)), device, block=3)
    pre.update(updated)
    g = _check_local_solves(test, pre, a + 0.2)
    np.testing.assert_allclose(g.T @ g, np.linalg.inv(a + 0.2), atol=1e-12)
    # 1x1 matrix-valued BSR is distinct from scalar-valued CSR.
    scalar = _matrix(a, device)
    single = sp.bsr_zeros(3, 3, wp.types.matrix((1, 1), wp.float64), device=device)
    single.offsets = wp.clone(scalar.offsets)
    single.columns = wp.clone(scalar.columns)
    single.values = scalar.values.view(single.dtype)
    single.notify_nnz_changed(nnz=scalar.nnz_sync())
    _check_local_solves(test, FSAI(single, max_row_size=3, kap_tolerance=0, reuse_pattern=True).update(), a)


def test_padded_input(test, device):
    """Check that padded input matches compact input and is rejected for reuse."""
    a = _path(12)
    for block in (1, 3):
        raw = _matrix(a, device, block=block)
        padded = sp.bsr_zeros(raw.nrow, raw.ncol, raw.dtype, device=device, row_capacity=raw.ncol + 2)
        sp.bsr_assign(padded, raw, topology="padded")
        pre = FSAI(padded, max_row_size=5)
        expected = FSAI(raw, max_row_size=5)
        np.testing.assert_allclose(_dense(pre.G), _dense(expected.G), atol=1e-12)
        with test.assertRaisesRegex(ValueError, "compact"):
            FSAI(padded, reuse_pattern=True)
        # Compacting with bsr_compress, as the error message advises, makes reuse possible.
        sp.bsr_compress(padded, prune_numerical_zeros=False)
        test.assertIsNone(padded.row_counts)
        np.testing.assert_allclose(
            _dense(FSAI(padded, max_row_size=5, reuse_pattern=True).G), _dense(expected.G), atol=1e-12
        )


def test_empty(test, device):
    """Check construction, application and update on an empty matrix."""
    for dtype in (wp.float16, wp.float32, wp.float64):
        raw = sp.bsr_zeros(0, 0, dtype, device=device)
        for reuse in (False, True):
            pre = FSAI(raw, reuse_pattern=reuse)
            test.assertEqual(pre.G.nnz_sync(), 0)
            test.assertEqual(pre.shape, (0, 0))
            x = wp.empty(0, dtype=dtype, device=device)
            pre.matvec(x, x, x, 1, 0)
            if reuse:
                test.assertIs(pre.update(), pre)
            wp.synchronize_device(device)


def test_apply_dispatch(test, device):
    """Check that application runs the factor kernel for every storage type, never bsr_mv."""
    a = _path(12)
    raw = _matrix(a, device, wp.float64)
    x = wp.ones(12, dtype=wp.float64, device=device)
    y = wp.empty_like(x)
    for storage in (wp.float64, wp.float32):
        pre = FSAI(raw, max_row_size=5, factor_dtype=storage)
        with (
            mock.patch.object(fsai_module, "_matvec_kernel", wraps=fsai_module._matvec_kernel) as kernel,
            mock.patch.object(sp, "bsr_mv") as bsr_mv,
        ):
            pre.matvec(x, y, y, 1, 0)
        test.assertEqual([call.args[0] for call in kernel.call_args_list], [storage, storage])
        test.assertEqual(bsr_mv.call_count, 0)
        g = _dense(pre.G)
        np.testing.assert_allclose(y.numpy(), g.T @ g @ np.ones(12), rtol=1e-6, atol=1e-6)


def test_captured_update(test, device, dtype):
    """Check that a captured update refits, rejects bad input and reports status on the device."""
    a = _path(12)
    tol = 3e-3 if dtype == wp.float16 else 1e-5 if dtype == wp.float32 else 1e-12
    for block in (1, 3):
        raw = _matrix(a, device, dtype, block)
        pre = FSAI(raw, max_row_size=5, reuse_pattern=True)
        vector = dtype if block == 1 else wp.types.vector(block, dtype)
        x = wp.ones(12 // block, dtype=vector, device=device)
        y = wp.empty_like(x)
        # Eager warm-up loads every kernel module before capture.
        pre.update()
        pre.matvec(x, y, y, 1, 0)
        with wp.ScopedCapture(device=device, force_module_load=False) as cap:
            pre.update()
            pre.matvec(x, y, y, 1, 0)

        def replay(values=None, columns=None, raw=raw, pre=pre, cap=cap, y=y):
            if values is not None:
                raw.values.assign(values)
            if columns is not None:
                raw.columns.assign(columns)
            wp.capture_launch(cap.graph)
            return pre.update_status.numpy().copy(), _dense(pre.G).astype(np.float64), y.numpy().ravel()

        original_values = raw.values.numpy().copy()
        original_columns = raw.columns.numpy().copy()
        for multiplier in (2.0, 0.5):
            status, g, result = replay(_matrix(multiplier * a, device, dtype, block).values.numpy())
            np.testing.assert_array_equal(status, 0)
            np.testing.assert_allclose(result, g.T @ g @ np.ones(12), rtol=tol, atol=tol)
        reference = _dense(pre.G).astype(np.float64)
        _check_local_solves(
            test, pre, 0.5 * a, rtol=8e-3 if dtype == wp.float16 else 2e-4, atol=5e-4 if dtype == wp.float16 else 2e-7
        )

        # A nonpositive diagonal is reported on the device and the factor is kept.
        status, g, _ = replay(np.zeros_like(original_values))
        test.assertGreater(status[1], 0)
        np.testing.assert_array_equal(g, reference)
        np.testing.assert_array_equal(_dense(pre.GT).astype(np.float64), reference.T)

        # A changed topology is reported and the factor is kept.
        changed = original_columns.copy()
        changed[0] += 1
        status, g, _ = replay(_matrix(a, device, dtype, block).values.numpy(), changed)
        test.assertGreater(status[0], 0)
        np.testing.assert_array_equal(g, reference)

        # Restoring valid input clears the status and applies the new factor.
        status, g, result = replay(original_values, original_columns)
        np.testing.assert_array_equal(status, 0)
        test.assertFalse(np.array_equal(g, reference))
        np.testing.assert_allclose(result, g.T @ g @ np.ones(12), rtol=tol, atol=tol)
        wp.synchronize_device(device)


def test_conditional_graph_solve(test, device, dtype):
    """Check that FSAI refits and preconditions CG and CR in one graph with a conditional solver loop."""
    if not wp.is_conditional_graph_supported():
        test.skipTest("conditional CUDA graphs are not supported")
    a = _path(24)
    tolerance = 1e-5 if dtype == wp.float32 else 1e-11
    for solver in (cg, cr):
        raw = _matrix(a, device, dtype)
        pre = FSAI(raw, reuse_pattern=True)
        b = wp.ones(len(a), dtype=dtype, device=device)
        x = wp.zeros_like(b)
        values = raw.values.numpy().copy()
        with wp.ScopedDevice(device):
            # Eager warm-up loads every kernel module before capture.
            pre.update()
            solver(raw, b, x, M=pre, tol=tolerance, maxiter=100)
            with wp.ScopedCapture(force_module_load=False) as capture:
                pre.update()
                solver(raw, b, x, M=pre, tol=tolerance, maxiter=100, check_every=0, use_cuda_graph=True)
            for multiplier in (2.0, 0.5):
                raw.values.assign(values * multiplier)
                x.zero_()
                wp.capture_launch(capture.graph)
                np.testing.assert_array_equal(pre.update_status.numpy(), 0)
                # Check the true residual, independently of the solver's report.
                np.testing.assert_allclose(multiplier * a @ x.numpy(), 1.0, atol=4 * tolerance, rtol=0)


def test_concurrent_streams(test, device):
    """Check that one instance applied on concurrent streams never corrupts an output."""
    num_blocks, block_size = 200_000, 12
    count = num_blocks * block_size
    A = sp.bsr_identity(num_blocks, block_type=wp.types.matrix((block_size, block_size), wp.float32), device=device)
    pre = FSAI(A)
    stream0 = wp.Stream(device, priority=0)
    stream1 = wp.Stream(device, priority=-1)
    x0 = wp.full(count, 1.0, dtype=wp.float32, device=device)
    x1 = wp.full(count, 2.0, dtype=wp.float32, device=device)
    z0, z1 = wp.zeros_like(x0), wp.zeros_like(x1)
    pre.matvec(x0, z0, z0, 1.0, 0.0)
    wp.synchronize_device(device)
    for _ in range(100):
        # Zero each output on the stream that writes it so the memset stays ordered.
        with wp.ScopedStream(stream0, sync_enter=False, sync_exit=False):
            z0.zero_()
            pre.matvec(x0, z0, z0, 1.0, 0.0)
        with wp.ScopedStream(stream1, sync_enter=False, sync_exit=False):
            z1.zero_()
            pre.matvec(x1, z1, z1, 1.0, 0.0)
        wp.synchronize_device(device)
        np.testing.assert_array_equal(z0.numpy(), x0.numpy())
        np.testing.assert_array_equal(z1.numpy(), x1.numpy())


def test_streams_during_unrelated_capture(test, device):
    """Check that streams stay ordered while another stream captures an unrelated graph."""
    num_blocks, block_size, iterations = 200_000, 12, 10
    count = num_blocks * block_size
    A = sp.bsr_identity(num_blocks, block_type=wp.types.matrix((block_size, block_size), wp.float32), device=device)
    pre = FSAI(A)
    stream0 = wp.Stream(device, priority=0)
    stream1 = wp.Stream(device, priority=-1)
    capture_stream = wp.Stream(device)
    x0 = wp.full(count, 1.0, dtype=wp.float32, device=device)
    x1 = wp.full(count, 2.0, dtype=wp.float32, device=device)
    outputs0 = [wp.zeros_like(x0) for _ in range(iterations)]
    outputs1 = [wp.zeros_like(x1) for _ in range(iterations)]
    pre.matvec(x0, outputs0[0], outputs0[0], 1.0, 0.0)
    wp.synchronize_device(device)
    wp.capture_begin(device, stream=capture_stream, force_module_load=False, capture_mode=wp.CaptureMode.RELAXED)
    try:
        # The device reports capture, but stream0 and stream1 are not the ones capturing.
        test.assertTrue(device.is_capturing)
        for z0, z1 in zip(outputs0, outputs1, strict=True):
            with wp.ScopedStream(stream0, sync_enter=False, sync_exit=False):
                z0.zero_()
                pre.matvec(x0, z0, z0, 1.0, 0.0)
            with wp.ScopedStream(stream1, sync_enter=False, sync_exit=False):
                z1.zero_()
                pre.matvec(x1, z1, z1, 1.0, 0.0)
    finally:
        wp.capture_end(device, stream=capture_stream)
    wp.synchronize_device(device)
    for z0, z1 in zip(outputs0, outputs1, strict=True):
        np.testing.assert_array_equal(z0.numpy(), x0.numpy())
        np.testing.assert_array_equal(z1.numpy(), x1.numpy())


def test_event_only_on_stream_change(test, device):
    """Check that stream ordering records events only when the application stream changes."""
    num_blocks = 1000
    A = sp.bsr_identity(num_blocks, block_type=wp.types.matrix((3, 3), wp.float32), device=device)
    pre = FSAI(A)
    other = wp.Stream(device)
    x = wp.ones(num_blocks * 3, dtype=wp.float32, device=device)
    y = wp.zeros_like(x)
    pre.matvec(x, y, y, 1.0, 0.0)
    with mock.patch.object(wp.Stream, "record_event", autospec=True, side_effect=wp.Stream.record_event) as record:
        for _ in range(5):
            pre.matvec(x, y, y, 1.0, 0.0)
        test.assertEqual(record.call_count, 0)
        with wp.ScopedStream(other, sync_enter=False, sync_exit=False):
            pre.matvec(x, y, y, 1.0, 0.0)
        test.assertEqual(record.call_count, 1)
    wp.synchronize_device(device)
    np.testing.assert_array_equal(y.numpy(), x.numpy())


def test_refit_growth_order(test, device):
    """Check that an unchanged matrix refits when sorted pivots differ from growth-order pivots."""
    b = np.sqrt(0.0179)
    a = np.array([[1.0, b, 0.0], [b, 1.0, 0.99], [0.0, 0.99, 1.0]])
    pre = FSAI(_matrix(a, device), max_row_size=3, pivot_floor=0.01, reuse_pattern=True)
    test.assertEqual(pre.truncated_rows, 0)
    before = _dense(pre.G)
    pre.update(_matrix(a, device))
    np.testing.assert_allclose(_dense(pre.G), before, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(_dense(pre.GT), before.T, rtol=1e-12, atol=1e-12)


def test_refit_low_precision_diagonal(test, device):
    """Check that an unchanged float16 matrix refits when its equilibrated diagonal rounds off one."""
    a = np.array([[1.0, 3.16015625], [3.16015625, 10.0]])
    raw = _matrix(a, device, wp.float16)
    pre = FSAI(raw, reuse_pattern=True)
    before = _dense(pre.G)
    pre.update(raw)
    np.testing.assert_array_equal(_dense(pre.G), before)


def _pair_matrix(first, second):
    """Return a block diagonal matrix of two 2x2 pairs."""
    a = np.zeros((4, 4))
    a[:2, :2] = first
    a[2:, 2:] = second
    return a


def test_regularized_refit(test, device, dtype):
    """Check that a regularizing refit keeps updating rows next to a failing one."""
    healthy = np.array([[1.0, 0.5], [0.5, 1.0]])
    changed = np.array([[2.0, 0.5], [0.5, 1.0]])
    singular = np.array([[1.0, 1.0], [1.0, 1.0]])
    start = _matrix(_pair_matrix(healthy, healthy), device, dtype)
    tolerance = 1e-5 if dtype == wp.float32 else 1e-12

    def refit(pre, a):
        pre.source.values.assign(_matrix(a, device, dtype).values.numpy())
        return pre.update()

    # By default a singular pair rejects the whole update and keeps the old factor.
    strict = FSAI(start, max_row_size=2, reuse_pattern=True)
    before = _dense(strict.G)
    with test.assertRaisesRegex(ValueError, "SPD"):
        refit(strict, _pair_matrix(singular, changed))
    np.testing.assert_array_equal(_dense(strict.G), before)
    test.assertEqual(strict.update_status.numpy()[3], 0)

    # With regularization the healthy pair is refit and the singular row is reported.
    pre = FSAI(
        _matrix(_pair_matrix(healthy, healthy), device, dtype),
        max_row_size=2,
        reuse_pattern=True,
        regularize_refit=True,
    )
    refit(pre, _pair_matrix(singular, changed))
    np.testing.assert_array_equal(pre.update_status.numpy(), [0, 0, 0, 1])
    reference = FSAI(_matrix(_pair_matrix(healthy, changed), device, dtype), max_row_size=2)
    np.testing.assert_allclose(_dense(pre.G)[2:, 2:], _dense(reference.G)[2:, 2:], rtol=tolerance, atol=tolerance)
    g = _dense(pre.G)
    test.assertTrue(np.all(np.isfinite(g)))
    test.assertTrue(np.all(np.diag(g) > 0))
    np.testing.assert_array_equal(_dense(pre.GT), g.T)

    # An invalid diagonal falls back to a unit diagonal row and leaves other rows refit.
    broken = _pair_matrix(healthy, changed)
    broken[0, 0] = -1.0
    refit(pre, broken)
    status = pre.update_status.numpy()
    test.assertEqual(status[1], 1)
    test.assertGreaterEqual(status[2], 1)
    g = _dense(pre.G)
    np.testing.assert_array_equal(g[0], [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(g[2:, 2:], _dense(reference.G)[2:, 2:], rtol=tolerance, atol=tolerance)

    # A topology mismatch still rejects the update and keeps the factor.
    kept = _dense(pre.G)
    other = _matrix(
        _pair_matrix(healthy, healthy) + np.diag([0.0, 0.0, 0.0, 0.0]) + np.eye(4)[::-1] * 0.1, device, dtype
    )
    with test.assertRaisesRegex(ValueError, "sparsity"):
        pre.update(other)
    np.testing.assert_array_equal(_dense(pre.G), kept)


def test_skipped_topology_validation(test, device):
    """Check that disabling topology validation skips the comparison and keeps refit results."""
    a = _path(12)
    for block in (1, 3):
        raw = _matrix(a, device, block=block)
        checked = FSAI(raw, max_row_size=5, reuse_pattern=True)
        unchecked = FSAI(raw, max_row_size=5, reuse_pattern=True, validate_topology=False)
        updated = _matrix(0.5 * a + np.eye(12), device, block=block)
        compared = []
        for pre in (checked, unchecked):
            with mock.patch.object(wp, "launch", wraps=wp.launch) as launch:
                pre.update(updated)
            compared.append(sum(call.args[0] is fsai_module._compare_topology for call in launch.call_args_list))
        test.assertEqual(compared, [2, 0])
        np.testing.assert_array_equal(unchecked.update_status.numpy(), 0)
        np.testing.assert_array_equal(_dense(unchecked.G), _dense(checked.G))
        np.testing.assert_array_equal(_dense(unchecked.GT), _dense(checked.GT))


def _topology_only(a, device, dtype, block):
    """Return a matrix with the topology of ``a`` and no assembled values."""
    result = _matrix(a, device, dtype, block)
    result.values.zero_()
    return result


def test_deferred_pattern(test, device, dtype):
    """Check that deferred selection matches construction-time selection and then only refits."""
    a = _path(12)
    for block, (width, step, storage) in itertools.product((1, 3), ((5, 1, dtype), (8, 3, dtype), (5, 1, wp.float32))):
        options = {"max_row_size": width, "max_step_size": step, "reuse_pattern": True, "factor_dtype": storage}
        raw = _topology_only(a, device, dtype, block)
        pre = FSAI(raw, defer_pattern=True, **options)
        np.testing.assert_array_equal(pre.pattern_ready.numpy(), 0)
        test.assertIsNone(pre.truncated_rows)
        vector = dtype if block == 1 else wp.types.vector(block, dtype)
        x = wp.array(np.arange(1.0, 13.0).reshape(12 // block, -1).squeeze(), dtype=vector, device=device)
        y = wp.empty_like(x)
        # Before the first update the preconditioner is the identity, which a constant vector cannot show.
        pre.matvec(x, y, y, 1, 0)
        np.testing.assert_array_equal(y.numpy(), x.numpy())

        raw.values.assign(_matrix(a, device, dtype, block).values.numpy())
        reference = FSAI(_matrix(a, device, dtype, block), **options)
        pre.update()
        np.testing.assert_array_equal(pre.pattern_ready.numpy(), 1)
        np.testing.assert_array_equal(pre.update_status.numpy(), 0)
        test.assertEqual(pre.pattern_status.numpy()[1], reference.truncated_rows)
        np.testing.assert_array_equal(_dense(pre.G), _dense(reference.G))
        np.testing.assert_array_equal(_dense(pre.GT), _dense(reference.GT))
        np.testing.assert_array_equal(np.diff(pre.GT.offsets.numpy()), np.diff(reference.GT.offsets.numpy()))
        expected = FSAI(_matrix(a, device, dtype, block), **options)
        expected.matvec(x, y, y, 1, 0)
        pre.matvec(x, y, y, 1, 0)
        z = wp.empty_like(x)
        expected.matvec(x, z, z, 1, 0)
        np.testing.assert_array_equal(y.numpy(), z.numpy())

        # Later updates keep the selected supports and only refit their values.
        columns = pre.G.columns.numpy().copy()
        changed = 0.5 * a + np.diag(np.arange(12.0))
        changed[0, 11] = changed[11, 0] = 0.0
        reference.update(_matrix(changed, device, dtype, block))
        raw.values.assign(_matrix(changed, device, dtype, block).values.numpy())
        with mock.patch.object(wp, "launch", wraps=wp.launch) as launch:
            pre.update()
        launched = [call.args[0] for call in launch.call_args_list]
        test.assertNotIn(pre._refit.select, launched)
        np.testing.assert_array_equal(pre.G.columns.numpy(), columns)
        np.testing.assert_array_equal(_dense(pre.G), _dense(reference.G))
        np.testing.assert_array_equal(_dense(pre.GT), _dense(reference.GT))


def test_deferred_pattern_failures(test, device, dtype):
    """Check that a deferred first update retries after a rejected matrix or regularizes it."""
    healthy = np.array([[1.0, 0.5], [0.5, 1.0]])
    good = _pair_matrix(healthy, healthy)
    bad = _pair_matrix(np.array([[-1.0, 0.5], [0.5, 1.0]]), healthy)
    options = {"max_row_size": 2, "reuse_pattern": True, "defer_pattern": True}

    raw = _topology_only(good, device, dtype, 1)
    strict = FSAI(raw, **options)
    raw.values.assign(_matrix(bad, device, dtype).values.numpy())
    with test.assertRaisesRegex(ValueError, "positive diagonals"):
        strict.update()
    np.testing.assert_array_equal(strict.pattern_ready.numpy(), 0)
    np.testing.assert_array_equal(_dense(strict.G), np.eye(4))
    raw.values.assign(_matrix(good, device, dtype).values.numpy())
    strict.update()
    np.testing.assert_array_equal(strict.pattern_ready.numpy(), 1)
    np.testing.assert_array_equal(_dense(strict.G), _dense(FSAI(_matrix(good, device, dtype), max_row_size=2).G))

    # With regularization the unusable row keeps a diagonal-only support and is reported.
    raw = _topology_only(good, device, dtype, 1)
    pre = FSAI(raw, regularize_refit=True, **options)
    raw.values.assign(_matrix(bad, device, dtype).values.numpy())
    pre.update()
    np.testing.assert_array_equal(pre.pattern_ready.numpy(), 1)
    test.assertEqual(pre.update_status.numpy()[1], 1)
    g = _dense(pre.G)
    np.testing.assert_array_equal(g[0], [1.0, 0.0, 0.0, 0.0])
    reference = FSAI(_matrix(good, device, dtype), max_row_size=2)
    np.testing.assert_array_equal(g[2:, 2:], _dense(reference.G)[2:, 2:])
    np.testing.assert_array_equal(_dense(pre.GT), g.T)


def test_deferred_pattern_graph(test, device, dtype):
    """Check that a captured first update selects the pattern on replay and later replays only refit."""
    a = _path(24)
    options = {"max_row_size": 5, "reuse_pattern": True, "defer_pattern": True}
    # Load every kernel module eagerly: capture does not load modules.
    warm = FSAI(_matrix(a, device, dtype), **options)
    warm.update()
    x = wp.ones(24, dtype=dtype, device=device)
    y = wp.empty_like(x)
    warm.matvec(x, y, y, 1, 0)

    raw = _topology_only(a, device, dtype, 1)
    pre = FSAI(raw, **options)
    with wp.ScopedCapture(device=device, force_module_load=False) as cap:
        pre.update()
        pre.matvec(x, y, y, 1, 0)
    values = _matrix(a, device, dtype).values.numpy()
    reference = FSAI(_matrix(a, device, dtype), max_row_size=5, reuse_pattern=True)
    for multiplier in (1.0, 2.0, 0.5):
        raw.values.assign(multiplier * values)
        reference.update(_matrix(multiplier * a, device, dtype))
        wp.capture_launch(cap.graph)
        np.testing.assert_array_equal(pre.pattern_ready.numpy(), 1)
        np.testing.assert_array_equal(pre.update_status.numpy(), 0)
        np.testing.assert_array_equal(_dense(pre.G), _dense(reference.G))
        np.testing.assert_array_equal(_dense(pre.GT), _dense(reference.GT))
        g = _dense(pre.G)
        np.testing.assert_allclose(
            y.numpy(), g.T @ g @ np.ones(24), rtol=1e-5 if dtype == wp.float32 else 1e-12, atol=1e-6
        )

    # A matrix that would select other supports leaves the published pattern alone.
    columns = pre.G.columns.numpy().copy()
    stiff = a + np.eye(24) * 200
    raw.values.assign(_matrix(stiff, device, dtype).values.numpy())
    reference.update(_matrix(stiff, device, dtype))
    wp.capture_launch(cap.graph)
    np.testing.assert_array_equal(pre.G.columns.numpy(), columns)
    np.testing.assert_array_equal(_dense(pre.G), _dense(reference.G))
    np.testing.assert_array_equal(_dense(pre.GT), _dense(reference.GT))


def test_deferred_conditional_graph_solve(test, device, dtype):
    """Check that a deferred preconditioner selects, refits and preconditions in one conditional graph."""
    if not wp.is_conditional_graph_supported():
        test.skipTest("conditional CUDA graphs are not supported")
    a = _path(24)
    tolerance = 1e-5 if dtype == wp.float32 else 1e-11
    for solver in (cg, cr):
        warm_matrix = _matrix(a, device, dtype)
        warm = FSAI(warm_matrix, reuse_pattern=True, defer_pattern=True)
        b = wp.ones(24, dtype=dtype, device=device)
        x = wp.zeros_like(b)
        with wp.ScopedDevice(device):
            warm.update()
            solver(warm_matrix, b, x, M=warm, tol=tolerance, maxiter=100)
            raw = _topology_only(a, device, dtype, 1)
            pre = FSAI(raw, reuse_pattern=True, defer_pattern=True)
            with wp.ScopedCapture(force_module_load=False) as capture:
                pre.update()
                solver(raw, b, x, M=pre, tol=tolerance, maxiter=100, check_every=0, use_cuda_graph=True)
            values = warm_matrix.values.numpy().copy()
            for multiplier in (2.0, 0.5):
                raw.values.assign(values * multiplier)
                x.zero_()
                wp.capture_launch(capture.graph)
                np.testing.assert_array_equal(pre.update_status.numpy(), 0)
                np.testing.assert_allclose(multiplier * a @ x.numpy(), 1.0, atol=4 * tolerance, rtol=0)


def test_block_selection_matches_scalar(test, device, dtype):
    """Check that selection on block storage matches selection on the scalar-expanded matrix."""
    rng = np.random.default_rng(11)
    for block in (2, 3):
        count = 9
        a = np.zeros((count * block, count * block))
        for i in range(count):
            for j in range(i):
                if rng.random() < 0.35:
                    blk = rng.normal(size=(block, block)) * 0.3
                    # Entries stored as exact zeros inside a nonzero block must stay harmless.
                    blk[0, -1] = 0.0
                    a[i * block : (i + 1) * block, j * block : (j + 1) * block] = blk
                    a[j * block : (j + 1) * block, i * block : (i + 1) * block] = blk.T
        a += np.diag(np.abs(a).sum(1) + 1.0)
        if dtype == wp.float16:
            a = a.astype(np.float16).astype(np.float64)
        raw = _matrix(a, device, dtype, block)
        expanded = sp.bsr_copy(raw, block_shape=(1, 1))
        for width, step, tolerance in ((3, 1, 1e-3), (8, 3, 1e-3), (12, 1, 0.0)):
            options = {"max_row_size": width, "max_step_size": step, "kap_tolerance": tolerance}
            blocked = FSAI(raw, **options)
            scalar = FSAI(expanded, **options)
            test.assertEqual(blocked.truncated_rows, scalar.truncated_rows)
            for name in ("offsets", "columns", "values"):
                np.testing.assert_array_equal(getattr(blocked.G, name).numpy(), getattr(scalar.G, name).numpy())
            np.testing.assert_array_equal(blocked.GT.values.numpy(), scalar.GT.values.numpy())
            deferred = FSAI(_topology_only(a, device, dtype, block), reuse_pattern=True, defer_pattern=True, **options)
            deferred.source.values.assign(raw.values.numpy())
            deferred.update()
            np.testing.assert_array_equal(_dense(deferred.G), _dense(scalar.G))


def test_cached_views_follow_arrays(test, device):
    """Check that cached vector views never serve stale data across many and recycled arrays."""
    a = _path(12)
    for block in (1, 3):
        pre = FSAI(_matrix(a, device, wp.float32, block), max_row_size=5)
        g = _dense(pre.G).astype(np.float64)
        expected = g.T @ g
        vector = wp.float32 if block == 1 else wp.vec3f
        rng = np.random.default_rng(5)
        # More arrays than the cache holds, then freed and recreated ones.
        for _ in range(3):
            for _ in range(12):
                values = rng.normal(size=12).astype(np.float32)
                x = wp.array(values.reshape(-1, block) if block > 1 else values, dtype=vector, device=device)
                y = wp.zeros_like(x)
                pre.matvec(x, y, y, 1.0, 0.0)
                np.testing.assert_allclose(y.numpy().ravel(), expected @ values, rtol=1e-4, atol=1e-5)
                del x, y


def test_eager_update_skips_synchronization(test, device):
    """Check that eager updates only read the status back when they could raise."""
    a = _path(12)
    for options, reads in (({}, 1), ({"validate_topology": False}, 1), ({"regularize_refit": True}, 1)):
        pre = FSAI(_matrix(a, device), reuse_pattern=True, **options)
        with mock.patch.object(type(pre._refit), "check", autospec=True) as check:
            pre.update()
        test.assertEqual(check.call_count, reads)
    both = FSAI(_matrix(a, device), reuse_pattern=True, regularize_refit=True, validate_topology=False)
    with mock.patch.object(type(both._refit), "check", autospec=True) as check:
        both.update()
    test.assertEqual(check.call_count, 0)
    # The deferred first update still has to read the ready flag, later ones need no readback.
    deferred = FSAI(
        _matrix(a, device), reuse_pattern=True, defer_pattern=True, regularize_refit=True, validate_topology=False
    )
    deferred.update()
    test.assertTrue(deferred._refit.published)
    with mock.patch.object(wp.array, "numpy", autospec=True) as numpy:
        deferred.update()
    test.assertEqual(numpy.call_count, 0)


def test_scaling_and_conversion(test, device):
    """Check badly scaled inputs and factor conversion overflow handling."""
    a = _path(12)
    scale = np.geomspace(1e-8, 1e8, len(a))
    a *= scale[:, None] * scale[None, :]
    _check_local_solves(test, FSAI(_matrix(a, device), max_row_size=5), a, atol=1e-9)
    for magnitude in (1e-100, 1e100):
        with test.assertRaisesRegex(ValueError, "positive diagonal"):
            FSAI(_matrix(np.eye(3) * magnitude, device), factor_dtype=wp.float32)
    raw = _matrix(np.eye(3), device)
    pre = FSAI(raw, reuse_pattern=True, factor_dtype=wp.float32)
    before = _dense(pre.G)
    for magnitude in (1e-100, 1e100):
        with test.assertRaises(ValueError):
            pre.update(_matrix(np.eye(3) * magnitude, device))
        np.testing.assert_array_equal(_dense(pre.G), before)
        np.testing.assert_array_equal(_dense(pre.GT), before.T)


def test_capture_update(test, device):
    """Check that CUDA graph application sees refit factor values."""
    a = _path(12)
    for block in (1, 3):
        for storage in (wp.float32, wp.float64):
            raw = _matrix(a, device, block=block)
            pre = FSAI(raw, max_row_size=5, reuse_pattern=True, factor_dtype=storage)
            vector = wp.float64 if block == 1 else wp.vec3d
            x = wp.ones(12 // block, dtype=vector, device=device)
            y = wp.empty_like(x)
            y.fill_(float("nan"))
            # Load the kernel modules before capture; capture uses force_module_load=False.
            pre.matvec(x, y, y, 1, 0)
            with wp.ScopedCapture(device=device, force_module_load=False) as cap:
                pre.matvec(x, y, y, 1, 0)
            for multiplier in (2.0, 0.5):
                pre.update(_matrix(multiplier * a, device, block=block))
                wp.capture_launch(cap.graph)
                g = _dense(pre.G).astype(np.float64)
                np.testing.assert_allclose(y.numpy().ravel(), g.T @ g @ np.ones(12), rtol=1e-12, atol=1e-12)
            wp.synchronize_device(device)


def test_float16_factors(test, device):
    """Check float16 factors against local solves and the exact inverse."""
    # Reference the matrix after input quantization, not the original float64 data.
    rng = np.random.default_rng(72)
    r = rng.normal(size=(12, 12))
    a = r @ r.T + 4 * np.eye(12)
    a = a.astype(np.float16).astype(np.float64)
    for block in (1, 3):
        raw = _matrix(a, device, wp.float16, block)
        for width, step in ((1, 1), (5, 3), (12, 1), (12, 3)):
            pre = FSAI(raw, max_row_size=width, max_step_size=step, kap_tolerance=0)
            test.assertEqual(pre.scalar_type, wp.float16)
            test.assertEqual(pre.G.scalar_type, wp.float16)
            test.assertEqual(pre.GT.scalar_type, wp.float16)
            g = _check_local_solves(test, pre, a, rtol=1e-2, atol=1e-3).astype(np.float64)
            if width == 12:
                reference = np.linalg.inv(a)
                test.assertLess(np.linalg.norm(g.T @ g - reference) / np.linalg.norm(reference), 1e-2)
            if width == 1:
                np.testing.assert_allclose(g.T @ g, np.diag(1 / np.diag(a)), rtol=2e-3, atol=1e-6)
        with test.assertRaisesRegex(ValueError, "factor_dtype"):
            FSAI(raw, factor_dtype=wp.float32)
        # Ordinary construction accepts padded blocks for half precision, too.
        padded = sp.bsr_zeros(raw.nrow, raw.ncol, raw.dtype, device=device, row_capacity=raw.ncol + 1)
        sp.bsr_assign(padded, raw, topology="padded")
        np.testing.assert_array_equal(_dense(FSAI(padded).G), _dense(FSAI(raw).G))


def test_float16_pivot_guard(test, device):
    """Check float16 pivot truncation and invalid diagonal rejection."""
    # Same rank-deficient inputs as block Jacobi. FSAI truncates a factor row
    # at a rejected pivot, retaining a positive Gram factor rather than
    # replacing an entire diagonal block with the identity.
    for a in (np.array([[1, 2], [2, 4]]), np.array([[9, 21], [21, 49]])):
        pre = FSAI(_matrix(a, device, wp.float16, 2), max_row_size=2)
        test.assertEqual(pre.truncated_rows, 1)
        g = _dense(pre.G).astype(np.float64)
        np.testing.assert_allclose(g, np.diag(1 / np.sqrt(np.diag(a))), rtol=2e-3)
        test.assertGreater(np.linalg.eigvalsh(g.T @ g).min(), 0)
    for diagonal in (0, -1, float("nan"), float("inf")):
        with test.assertRaisesRegex(ValueError, "positive diagonals"):
            FSAI(_matrix(np.diag([1.0, diagonal]), device, wp.float16))


def test_float16_capture_refit(test, device):
    """Check that float16 CUDA graph application sees refit factor values."""
    a = _path(12)
    raw = _matrix(a, device, wp.float16, 3)
    pre = FSAI(raw, reuse_pattern=True, max_row_size=8, kap_tolerance=0)
    pointers = (pre.G.values.ptr, pre.GT.values.ptr)
    x = wp.ones(4, dtype=wp.vec3h, device=device)
    y = wp.empty_like(x)
    # Load the kernel modules before capture; capture uses force_module_load=False.
    pre.matvec(x, y, y, 1, 0)
    with wp.ScopedCapture(device=device, force_module_load=False) as cap:
        pre.matvec(x, y, y, 1, 0)
    for multiplier in (2.0, 0.5):
        pre.update(_matrix(multiplier * a, device, wp.float16, 3))
        test.assertEqual((pre.G.values.ptr, pre.GT.values.ptr), pointers)
        wp.capture_launch(cap.graph)
        g = _dense(pre.G).astype(np.float64)
        np.testing.assert_allclose(y.numpy().ravel(), g.T @ g @ np.ones(12), rtol=3e-3, atol=1e-3)
    before = y.numpy().copy()
    invalid = sp.bsr_copy(raw)
    invalid.values.zero_()
    with test.assertRaises(ValueError):
        pre.update(invalid)
    wp.capture_launch(cap.graph)
    np.testing.assert_array_equal(y.numpy(), before)


class TestFSAI(unittest.TestCase):
    def test_invalid_arguments(self):
        """Check that invalid arguments and matrices raise specific errors."""
        raw = _matrix(np.eye(3), "cpu")
        for option in ("max_row_size", "max_step_size"):
            for value in (0, -1, 65, 1.5):
                with self.subTest(option=option, value=value), self.assertRaisesRegex(ValueError, option):
                    FSAI(raw, **{option: value})
        for option, values in (
            ("kap_tolerance", (-1, 1, float("nan"), float("inf"))),
            ("pivot_floor", (0, 1, float("nan"))),
            ("factor_dtype", (wp.float16, wp.int32)),
        ):
            for value in values:
                with self.subTest(option=option, value=value), self.assertRaisesRegex(ValueError, option):
                    FSAI(raw, **{option: value})
        for invalid in (
            None,
            sp.bsr_zeros(2, 3, wp.float64, device="cpu"),
            sp.bsr_zeros(3, 2, wp.types.matrix((2, 3), wp.float64), device="cpu"),
        ):
            with self.assertRaisesRegex(ValueError, "square"):
                FSAI(invalid)
        with self.assertRaises(TypeError):
            FSAI(sp.bsr_zeros(3, 3, wp.bfloat16, device="cpu"))
        with self.assertRaisesRegex(ValueError, "reuse_pattern"):
            FSAI(raw).update()
        with self.assertRaisesRegex(ValueError, "defer_pattern"):
            FSAI(raw, defer_pattern=True)
        for diagonal in (0.0, -1.0, float("inf"), float("nan")):
            with self.assertRaisesRegex(ValueError, "positive diagonals"):
                FSAI(_matrix(np.diag([1.0, diagonal]), "cpu"))
        pre = FSAI(raw, reuse_pattern=True)
        with self.assertRaisesRegex(ValueError, "BSR"):
            pre.update(np.eye(3))
        for changed in (_matrix(np.eye(2), "cpu"), _matrix(np.eye(3), "cpu", wp.float32)):
            with self.assertRaisesRegex(ValueError, "shape, dtype, device"):
                pre.update(changed)


devices = get_test_devices()
for dtype in (wp.float32, wp.float64):
    for func in (
        test_full_pattern,
        test_adaptive_support,
        test_application,
        test_solver_integration,
        test_refit_and_rollback,
        test_regularized_refit,
        test_deferred_pattern,
        test_deferred_pattern_failures,
        test_block_selection_matches_scalar,
    ):
        add_function_test(TestFSAI, f"{func.__name__}_{dtype.__name__}", func, devices=devices, dtype=dtype)
for func in (
    test_narrow_frontier_and_zero_tolerance,
    test_pivot_guard,
    test_refit_growth_order,
    test_skipped_topology_validation,
    test_cached_views_follow_arrays,
    test_eager_update_skips_synchronization,
    test_refit_low_precision_diagonal,
    test_stored_zeros_and_capacity,
    test_padded_input,
    test_empty,
    test_scaling_and_conversion,
    test_apply_dispatch,
):
    add_function_test(TestFSAI, func.__name__, func, devices=devices)
add_function_test(TestFSAI, "test_capture_update", test_capture_update, devices=get_cuda_test_devices())
for dtype in (wp.float16, wp.float32, wp.float64):
    add_function_test(
        TestFSAI,
        f"test_captured_update_{dtype.__name__}",
        test_captured_update,
        devices=get_cuda_test_devices(),
        dtype=dtype,
    )
for dtype in (wp.float32, wp.float64):
    add_function_test(
        TestFSAI,
        f"test_conditional_graph_solve_{dtype.__name__}",
        test_conditional_graph_solve,
        devices=get_cuda_test_devices(),
        dtype=dtype,
    )
add_function_test(
    TestFSAI, "test_event_only_on_stream_change", test_event_only_on_stream_change, devices=get_cuda_test_devices()
)
for dtype in (wp.float32, wp.float64):
    for func in (test_deferred_pattern_graph, test_deferred_conditional_graph_solve):
        add_function_test(
            TestFSAI, f"{func.__name__}_{dtype.__name__}", func, devices=get_cuda_test_devices(), dtype=dtype
        )
add_function_test(TestFSAI, "test_concurrent_streams", test_concurrent_streams, devices=get_cuda_test_devices())
add_function_test(
    TestFSAI,
    "test_streams_during_unrelated_capture",
    test_streams_during_unrelated_capture,
    devices=get_cuda_test_devices(),
)

for func in (test_application, test_refit_and_rollback, test_block_selection_matches_scalar):
    add_function_test(TestFSAI, f"{func.__name__}_float16", func, devices=devices, dtype=wp.float16)
for func in (test_float16_factors, test_float16_pivot_guard):
    add_function_test(TestFSAI, func.__name__, func, devices=devices)
add_function_test(TestFSAI, "test_float16_capture_refit", test_float16_capture_refit, devices=get_cuda_test_devices())

if __name__ == "__main__":
    unittest.main(verbosity=2)
