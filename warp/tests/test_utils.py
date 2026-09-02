# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import gc
import io
import os
import subprocess
import sys
import unittest
import warnings
from unittest.mock import patch

from warp._src import context as _context
from warp._src import logger as _logger
from warp._src.logger import log_warning
from warp.tests.unittest_utils import *


def test_array_scan(test, device):
    rng = np.random.default_rng(123)

    dtype_cases = (
        (wp.int32, np.int32, 0),
        (wp.int64, np.int64, 0),
        (wp.float32, np.float32, 1e-3),
        (wp.float64, np.float64, 1e-12),
    )

    for wp_dtype, np_dtype, tolerance in dtype_cases:
        if np.issubdtype(np_dtype, np.integer):
            values = rng.integers(-1000, high=1000, size=100000, dtype=np_dtype)
        else:
            values = rng.uniform(low=-1e6, high=1e6, size=100000).astype(np_dtype)

        expected = np.cumsum(values)

        values = wp.array(values, dtype=wp_dtype, device=device)
        result_inc = wp.zeros_like(values)
        result_exc = wp.zeros_like(values)

        wp.utils.array_scan(values, result_inc, True)
        wp.utils.array_scan(values, result_exc, False)

        result_inc = result_inc.numpy().squeeze()
        result_exc = result_exc.numpy().squeeze()
        error_inc = np.max(np.abs(result_inc - expected)) / abs(expected[-1])
        error_exc = max(np.max(np.abs(result_exc[1:] - expected[:-1])), abs(result_exc[0])) / abs(expected[-2])

        test.assertTrue(error_inc <= tolerance)
        test.assertTrue(error_exc <= tolerance)


def test_array_scan_vector(test, device):
    rng = np.random.default_rng(123)

    dtype_cases = (
        (wp.vec3i, np.int32, 0),
        (wp.vec3l, np.int64, 0),
        (wp.vec3f, np.float32, 1e-3),
        (wp.vec3d, np.float64, 1e-12),
    )

    for wp_dtype, np_dtype, tolerance in dtype_cases:
        if np.issubdtype(np_dtype, np.integer):
            values = rng.integers(-1000, high=1000, size=(100000, 3), dtype=np_dtype)
        else:
            values = rng.uniform(low=-1e6, high=1e6, size=(100000, 3)).astype(np_dtype)

        expected = np.cumsum(values, axis=0)

        values = wp.array(values, dtype=wp_dtype, device=device)
        result_inc = wp.zeros_like(values)
        result_exc = wp.zeros_like(values)

        wp.utils.array_scan(values, result_inc, True)
        wp.utils.array_scan(values, result_exc, False)

        result_inc = result_inc.numpy()
        result_exc = result_exc.numpy()
        error_inc = np.max(np.abs(result_inc - expected)) / np.max(np.abs(expected[-1]))
        error_exc = max(np.max(np.abs(result_exc[1:] - expected[:-1])), np.max(np.abs(result_exc[0]))) / np.max(
            np.abs(expected[-2])
        )

        test.assertTrue(error_inc <= tolerance)
        test.assertTrue(error_exc <= tolerance)


def test_array_scan_strided_views(test, device):
    # Interleave data with padding to exercise 1D strided views and
    # verify array_scan() writes only through the output view.
    scalar_values = np.array((6, -3, 5, 0, -2, 8), dtype=np.int64)
    scalar_base = np.zeros(scalar_values.size * 2, dtype=np.int64)
    scalar_base[::2] = scalar_values

    scalar_values_base = wp.array(scalar_base, dtype=wp.int64, device=device)
    values = scalar_values_base[::2]  # Create a non-contiguous Warp array view.

    sentinel = np.int64(-12345)
    result_inc_base = wp.array(np.full_like(scalar_base, sentinel), dtype=wp.int64, device=device)
    result_exc_base = wp.array(np.full_like(scalar_base, sentinel), dtype=wp.int64, device=device)

    result_inc = result_inc_base[::2]
    result_exc = result_exc_base[::2]

    wp.utils.array_scan(values, result_inc, True)
    wp.utils.array_scan(values, result_exc, False)

    result_inc_base = result_inc_base.numpy()
    result_exc_base = result_exc_base.numpy()
    expected_inc = np.cumsum(scalar_values)
    expected_exc = np.zeros_like(scalar_values)
    expected_exc[1:] = expected_inc[:-1]

    np.testing.assert_array_equal(result_inc_base[::2], expected_inc)
    np.testing.assert_array_equal(result_exc_base[::2], expected_exc)
    np.testing.assert_array_equal(result_inc_base[1::2], sentinel)
    np.testing.assert_array_equal(result_exc_base[1::2], sentinel)

    vector_values = np.array(
        (
            (2.0, -1.0, 4.0),
            (0.5, 3.0, -2.0),
            (-1.5, 0.0, 5.0),
            (4.0, -2.0, 1.0),
        ),
        dtype=np.float64,
    )
    vector_base = np.zeros((vector_values.shape[0] * 2, 3), dtype=np.float64)
    vector_base[1::2] = vector_values

    vector_values_base = wp.array(vector_base, dtype=wp.vec3d, device=device)
    values = vector_values_base[1::2]  # Create a non-contiguous Warp array view.

    sentinel = -999.0
    result_inc_base = wp.array(np.full(vector_base.shape, sentinel, dtype=np.float64), dtype=wp.vec3d, device=device)
    result_exc_base = wp.array(np.full(vector_base.shape, sentinel, dtype=np.float64), dtype=wp.vec3d, device=device)

    result_inc = result_inc_base[1::2]
    result_exc = result_exc_base[1::2]

    wp.utils.array_scan(values, result_inc, True)
    wp.utils.array_scan(values, result_exc, False)

    result_inc_base = result_inc_base.numpy()
    result_exc_base = result_exc_base.numpy()
    expected_inc = np.cumsum(vector_values, axis=0)
    expected_exc = np.zeros_like(vector_values)
    expected_exc[1:] = expected_inc[:-1]

    np.testing.assert_allclose(result_inc_base[1::2], expected_inc)
    np.testing.assert_allclose(result_exc_base[1::2], expected_exc)
    np.testing.assert_allclose(result_inc_base[::2], sentinel)
    np.testing.assert_allclose(result_exc_base[::2], sentinel)


def test_array_scan_empty(test, device):
    values = wp.array((), dtype=int, device=device)
    result = wp.array((), dtype=int, device=device)
    wp.utils.array_scan(values, result)


def test_array_scan_error_sizes_mismatch(test, device):
    values = wp.zeros(123, dtype=int, device=device)
    result = wp.zeros(234, dtype=int, device=device)
    with test.assertRaisesRegex(
        RuntimeError,
        r"In and out array storage sizes do not match \(123 vs 234\)$",
    ):
        wp.utils.array_scan(values, result, True)


def test_array_scan_error_dtypes_mismatch(test, device):
    values = wp.zeros(123, dtype=int, device=device)
    result = wp.zeros(123, dtype=float, device=device)
    with test.assertRaisesRegex(
        RuntimeError,
        r"In and out array data types do not match \(int32 vs float32\)$",
    ):
        wp.utils.array_scan(values, result, True)


def test_array_scan_error_unsupported_dtype(test, device):
    values = wp.zeros(123, dtype=wp.vec3h, device=device)
    result = wp.zeros(123, dtype=wp.vec3h, device=device)
    with test.assertRaisesRegex(
        RuntimeError,
        r"Unsupported data type: vec3h$",
    ):
        wp.utils.array_scan(values, result, True)


def test_array_scan_error_device_failure_is_reported(test, device):
    """A failed device scan raises instead of leaving the output silently unwritten."""
    values = wp.zeros(16, dtype=wp.int32, device=device)
    result = wp.zeros(16, dtype=wp.int32, device=device)

    with (
        patch.object(_context.runtime.core, "wp_array_scan_int_device", lambda *args: False),
        test.assertRaises(RuntimeError),
    ):
        wp.utils.array_scan(values, result, True)


def test_radix_sort_pairs(test, device):
    keyTypes = [int, wp.uint32, wp.float32, wp.int64, wp.uint64, wp.float64]

    for keyType in keyTypes:
        keys = wp.array((7, 2, 8, 4, 1, 6, 5, 3, 0, 0, 0, 0, 0, 0, 0, 0), dtype=keyType, device=device)
        values = wp.array((1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0), dtype=int, device=device)
        wp.utils.radix_sort_pairs(keys, values, 8)
        assert_np_equal(keys.numpy()[:8], np.array((1, 2, 3, 4, 5, 6, 7, 8)))
        assert_np_equal(values.numpy()[:8], np.array((5, 2, 8, 4, 7, 6, 1, 3)))


def test_radix_sort_pairs_signed_keys(test, device):
    keyTypes = [int, wp.int64]

    for keyType in keyTypes:
        keys = wp.array((-1, 7, -8, 0, 4, -3, 2, 10, 0, 0, 0, 0, 0, 0, 0, 0), dtype=keyType, device=device)
        values = wp.array((1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0), dtype=int, device=device)
        wp.utils.radix_sort_pairs(keys, values, 8)
        assert_np_equal(keys.numpy()[:8], np.array((-8, -3, -1, 0, 2, 4, 7, 10)))
        assert_np_equal(values.numpy()[:8], np.array((3, 6, 1, 4, 7, 5, 2, 8)))


def test_radix_sort_pairs_bit_range(test, device):
    keyTypes = [int, wp.uint32, wp.int64, wp.uint64]

    for keyType in keyTypes:
        keys = wp.array((8, 1, 10, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0), dtype=keyType, device=device)
        values = wp.array((1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0), dtype=int, device=device)
        wp.utils.radix_sort_pairs(keys, values, 8, end_bit=2)
        assert_np_equal(keys.numpy()[:8], np.array((8, 4, 1, 5, 10, 6, 3, 7)))
        assert_np_equal(values.numpy()[:8], np.array((1, 5, 2, 6, 3, 7, 4, 8)))


def test_radix_sort_pairs_value_types(test, device):
    valueTypes = [wp.uint32, wp.float32, wp.int64, wp.uint64, wp.float64]

    for valueType in valueTypes:
        keys = wp.array((7, 2, 8, 4, 1, 6, 5, 3, 0, 0, 0, 0, 0, 0, 0, 0), dtype=int, device=device)
        values = wp.array((1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0), dtype=valueType, device=device)
        wp.utils.radix_sort_pairs(keys, values, 8)
        assert_np_equal(keys.numpy()[:8], np.array((1, 2, 3, 4, 5, 6, 7, 8)))
        assert_np_equal(values.numpy()[:8], np.array((5, 2, 8, 4, 7, 6, 1, 3)))


def test_radix_sort_pairs_64_bit_keys_8_byte_values(test, device):
    # Use duplicate 64-bit keys plus 8-byte payloads large enough to catch
    # unstable key/value movement or payload truncation.
    cases = (
        (
            wp.int64,
            np.array((5, -(2**40), -7, 0, 2**40, -7, 5), dtype=np.int64),
            wp.uint64,
            np.array((2**48 + 1, 2**48 + 2, 2**48 + 3, 2**48 + 4, 2**48 + 5, 2**48 + 6, 2**48 + 7), dtype=np.uint64),
        ),
        (
            wp.uint64,
            np.array((2**63 + 3, 11, 2**63 + 3, 0, 2**40, 11, 2**63 + 2), dtype=np.uint64),
            wp.float64,
            np.array((10.25, -5.5, 20.5, 0.0, 7.75, 11.125, -13.5), dtype=np.float64),
        ),
    )

    for key_type, keys, value_type, values in cases:
        with test.subTest(key_type=key_type, value_type=value_type):
            count = keys.size
            # radix_sort_pairs() uses the second half of each array as scratch storage.
            key_storage = np.zeros(count * 2, dtype=keys.dtype)
            value_storage = np.zeros(count * 2, dtype=values.dtype)
            key_storage[:count] = keys
            value_storage[:count] = values

            wp_keys = wp.array(key_storage, dtype=key_type, device=device)
            wp_values = wp.array(value_storage, dtype=value_type, device=device)
            wp.utils.radix_sort_pairs(wp_keys, wp_values, count)

            order = np.argsort(keys, kind="stable")
            np.testing.assert_array_equal(wp_keys.numpy()[:count], keys[order])
            if np.issubdtype(values.dtype, np.floating):
                np.testing.assert_allclose(wp_values.numpy()[:count], values[order])
            else:
                np.testing.assert_array_equal(wp_values.numpy()[:count], values[order])


def test_radix_sort_pairs_error_non_contiguous(test, device):
    values = wp.array(tuple(range(16)), dtype=int, device=device)
    keys = wp.array(tuple(range(16)), dtype=int, device=device)[::2]

    with test.assertRaisesRegex(
        RuntimeError,
        r"radix_sort_pairs\(\) requires a contiguous keys array, got non-contiguous keys with data types: int32, int32$",
    ):
        wp.utils.radix_sort_pairs(keys, values, 4)

    keys = wp.array(tuple(range(16)), dtype=int, device=device)
    values = wp.array(tuple(range(16)), dtype=int, device=device)[::2]

    with test.assertRaisesRegex(
        RuntimeError,
        r"radix_sort_pairs\(\) requires a contiguous values array, got non-contiguous values with data types: int32, int32$",
    ):
        wp.utils.radix_sort_pairs(keys, values, 4)


def _make_radix_sort_arrays(rng, count, device):
    # radix_sort_pairs() uses the second half of each array as scratch storage
    keys_np = rng.integers(0, 2**40, size=count, dtype=np.int64)
    values_np = np.arange(count, dtype=np.int32)
    keys = wp.zeros(2 * count, dtype=wp.int64, device=device)
    values = wp.zeros(2 * count, dtype=wp.int32, device=device)
    return keys_np, values_np, keys, values


def _upload_radix_sort_arrays(keys_np, values_np, keys, values):
    count = len(keys_np)
    keys.assign(np.concatenate([keys_np, np.zeros(count, dtype=keys_np.dtype)]))
    values.assign(np.concatenate([values_np, np.zeros(count, dtype=values_np.dtype)]))


def _check_radix_sort_arrays(keys_np, values_np, keys, values):
    count = len(keys_np)
    order = np.argsort(keys_np, kind="stable")
    np.testing.assert_array_equal(keys.numpy()[:count], keys_np[order])
    np.testing.assert_array_equal(values.numpy()[:count], values_np[order])


def test_radix_sort_pairs_graph_capture(test, device):
    """Verify captured sorts that grow the scratch buffer mid-capture replay correctly."""
    rng = np.random.default_rng(123)
    n_small, n_large = 2500, 100000

    ks_np, vs_np, ks, vs = _make_radix_sort_arrays(rng, n_small, device)
    kl_np, vl_np, kl, vl = _make_radix_sort_arrays(rng, n_large, device)

    # seed the global scratch buffer with a small sort
    wp.utils.radix_sort_pairs(ks, vs, n_small)

    with wp.ScopedCapture(device, force_module_load=False) as capture:
        wp.utils.radix_sort_pairs(ks, vs, n_small)
        wp.utils.radix_sort_pairs(kl, vl, n_large)  # needs bigger scratch mid-capture

    for _ in range(3):
        _upload_radix_sort_arrays(ks_np, vs_np, ks, vs)
        _upload_radix_sort_arrays(kl_np, vl_np, kl, vl)
        wp.capture_launch(capture.graph)
        _check_radix_sort_arrays(ks_np, vs_np, ks, vs)
        _check_radix_sort_arrays(kl_np, vl_np, kl, vl)

    # grow the global scratch buffer outside the capture, then relaunch the graph
    kh_np, vh_np, kh, vh = _make_radix_sort_arrays(rng, 4 * n_large, device)
    _upload_radix_sort_arrays(kh_np, vh_np, kh, vh)
    wp.utils.radix_sort_pairs(kh, vh, 4 * n_large)

    _upload_radix_sort_arrays(ks_np, vs_np, ks, vs)
    _upload_radix_sort_arrays(kl_np, vl_np, kl, vl)
    wp.capture_launch(capture.graph)
    _check_radix_sort_arrays(ks_np, vs_np, ks, vs)
    _check_radix_sort_arrays(kl_np, vl_np, kl, vl)

    # delete the graph, then keep sorting
    del capture
    gc.collect()
    _upload_radix_sort_arrays(kl_np, vl_np, kl, vl)
    wp.utils.radix_sort_pairs(kl, vl, n_large)
    _check_radix_sort_arrays(kl_np, vl_np, kl, vl)


def test_radix_sort_pairs_graph_capture_independent_graphs(test, device):
    """Verify each captured graph keeps working after other graphs and sorts resize scratch."""
    rng = np.random.default_rng(123)
    n_a, n_b = 2500, 100000

    ka_np, va_np, ka, va = _make_radix_sort_arrays(rng, n_a, device)
    kb_np, vb_np, kb, vb = _make_radix_sort_arrays(rng, n_b, device)

    with wp.ScopedCapture(device, force_module_load=False) as capture_a:
        wp.utils.radix_sort_pairs(ka, va, n_a)

    with wp.ScopedCapture(device, force_module_load=False) as capture_b:
        wp.utils.radix_sort_pairs(kb, vb, n_b)

    for _ in range(2):
        _upload_radix_sort_arrays(ka_np, va_np, ka, va)
        _upload_radix_sort_arrays(kb_np, vb_np, kb, vb)
        wp.capture_launch(capture_a.graph)
        wp.capture_launch(capture_b.graph)
        _check_radix_sort_arrays(ka_np, va_np, ka, va)
        _check_radix_sort_arrays(kb_np, vb_np, kb, vb)

    # destroying one graph must not affect the other
    del capture_b
    gc.collect()
    _upload_radix_sort_arrays(ka_np, va_np, ka, va)
    wp.capture_launch(capture_a.graph)
    _check_radix_sort_arrays(ka_np, va_np, ka, va)


def test_radix_sort_pairs_capture_while(test, device):
    """Verify sorts inside a conditional body graph, where graph allocations are not permitted."""
    if not wp.is_conditional_graph_supported():
        test.skipTest("Conditional graph nodes not supported")

    rng = np.random.default_rng(123)
    n = 100000

    kl_np, vl_np, kl, vl = _make_radix_sort_arrays(rng, n, device)

    counter = wp.empty(1, dtype=wp.int32, device=device)

    @wp.kernel
    def decrement_counter(c: wp.array[wp.int32]):
        c[0] = c[0] - 1

    def body():
        wp.utils.radix_sort_pairs(kl, vl, n)
        wp.launch(decrement_counter, dim=1, inputs=[counter], device=device)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, force_module_load=False) as capture:
        counter.fill_(1)
        wp.capture_while(counter, while_body=body)

    for _ in range(2):
        _upload_radix_sort_arrays(kl_np, vl_np, kl, vl)
        wp.capture_launch(capture.graph)
        _check_radix_sort_arrays(kl_np, vl_np, kl, vl)


def test_radix_sort_pairs_graph_capture_stream_destruction(test, device):
    """Verify destroying the capture stream does not free scratch a graph still uses."""
    rng = np.random.default_rng(123)
    n = 100000

    k_np, v_np, k, v = _make_radix_sort_arrays(rng, n, device)

    stream = wp.Stream(device)
    with wp.ScopedStream(stream):
        # seed the stream's scratch buffer, then capture a sort on the same stream
        _upload_radix_sort_arrays(k_np, v_np, k, v)
        wp.utils.radix_sort_pairs(k, v, n)

        with wp.ScopedCapture(device, stream=stream, force_module_load=False) as capture:
            wp.utils.radix_sort_pairs(k, v, n)

    # destroying the stream releases its cached scratch buffer
    graph = capture.graph
    del capture, stream
    gc.collect()

    for _ in range(2):
        _upload_radix_sort_arrays(k_np, v_np, k, v)
        wp.capture_launch(graph)
        _check_radix_sort_arrays(k_np, v_np, k, v)


def test_radix_sort_pairs_graph_capture_pause(test, device):
    """Verify growing the scratch buffer while a capture is paused preserves the paused graph."""
    # capture pause/resume requires CUDA 12.4+, same as conditional graph nodes
    if not wp.is_conditional_graph_supported():
        test.skipTest("Graph capture pause/resume not supported")

    from warp._src.context import capture_pause, capture_resume  # noqa: PLC0415

    rng = np.random.default_rng(123)
    n_small, n_large = 2500, 100000

    ks_np, vs_np, ks, vs = _make_radix_sort_arrays(rng, n_small, device)
    kl_np, vl_np, kl, vl = _make_radix_sort_arrays(rng, n_large, device)
    kp_np, vp_np, kp, vp = _make_radix_sort_arrays(rng, n_large, device)  # sorted while paused

    # seed the scratch buffer with a small sort
    _upload_radix_sort_arrays(ks_np, vs_np, ks, vs)
    wp.utils.radix_sort_pairs(ks, vs, n_small)

    _upload_radix_sort_arrays(kp_np, vp_np, kp, vp)

    wp.capture_begin(device, force_module_load=False)
    try:
        wp.utils.radix_sort_pairs(ks, vs, n_small)
        paused_graph = capture_pause(device=device)
        # grow the scratch buffer while the capture is paused
        wp.utils.radix_sort_pairs(kp, vp, n_large)
        capture_resume(paused_graph, device=device)
        wp.utils.radix_sort_pairs(kl, vl, n_large)
    finally:
        graph = wp.capture_end(device)

    # the sort issued while paused executed immediately
    _check_radix_sort_arrays(kp_np, vp_np, kp, vp)

    for _ in range(2):
        _upload_radix_sort_arrays(ks_np, vs_np, ks, vs)
        _upload_radix_sort_arrays(kl_np, vl_np, kl, vl)
        wp.capture_launch(graph)
        _check_radix_sort_arrays(ks_np, vs_np, ks, vs)
        _check_radix_sort_arrays(kl_np, vl_np, kl, vl)


def test_radix_sort_pairs_capture_if_forked_stream(test, device):
    """Verify conditional body sorts on the main and a forked stream do not leak scratch buffers."""
    if not wp.is_conditional_graph_supported():
        test.skipTest("Conditional graph nodes not supported")

    from warp._src.context import runtime  # noqa: PLC0415

    rng = np.random.default_rng(123)
    n = 100000

    ka_np, va_np, ka, va = _make_radix_sort_arrays(rng, n, device)
    kb_np, vb_np, kb, vb = _make_radix_sort_arrays(rng, n, device)

    main_stream = device.stream
    forked_stream = wp.Stream(device)
    fork_event = wp.Event(device)
    join_event = wp.Event(device)
    condition = wp.ones(1, dtype=wp.int32, device=device)

    def sort_on_both_streams():
        # sort on a forked stream and on the main stream within the conditional body
        forked_stream.wait_stream(main_stream, event=fork_event)
        with wp.ScopedStream(forked_stream):
            wp.utils.radix_sort_pairs(ka, va, n)
        wp.utils.radix_sort_pairs(kb, vb, n)
        main_stream.wait_stream(forked_stream, event=join_event)

    # warm up non-sort allocations before enabling allocation tracking
    _upload_radix_sort_arrays(ka_np, va_np, ka, va)
    _upload_radix_sort_arrays(kb_np, vb_np, kb, vb)
    wp.synchronize_device(device)

    core = runtime.core
    core.wp_alloc_tracker_enable(1)
    try:
        core.wp_alloc_tracker_reset()

        with wp.ScopedCapture(device=device, stream=main_stream, force_module_load=False) as capture:
            wp.capture_if(condition, on_true=sort_on_both_streams, stream=main_stream)

        wp.capture_launch(capture.graph)
        _check_radix_sort_arrays(ka_np, va_np, ka, va)
        _check_radix_sort_arrays(kb_np, vb_np, kb, vb)

        # destroying the graph must release the scratch buffers allocated during capture
        del capture
        gc.collect()
        wp.synchronize_device(device)
        test.assertEqual(core.wp_alloc_tracker_get_live_count(), 0)
    finally:
        core.wp_alloc_tracker_enable(0)


def test_radix_sort_pairs_graph_capture_multi_stream(test, device):
    """Verify sorts on multiple forked streams within a single graph capture."""
    rng = np.random.default_rng(123)
    n = 100000

    ka_np, va_np, ka, va = _make_radix_sort_arrays(rng, n, device)
    kb_np, vb_np, kb, vb = _make_radix_sort_arrays(rng, n, device)
    kc_np, vc_np, kc, vc = _make_radix_sort_arrays(rng, n, device)

    main_stream = device.stream
    stream_b = wp.Stream(device)
    stream_c = wp.Stream(device)
    fork_event_b = wp.Event(device)
    fork_event_c = wp.Event(device)
    join_event_b = wp.Event(device)
    join_event_c = wp.Event(device)

    with wp.ScopedCapture(device=device, stream=main_stream, force_module_load=False) as capture:
        # fork two streams that sort in parallel graph branches
        stream_b.wait_stream(main_stream, event=fork_event_b)
        stream_c.wait_stream(main_stream, event=fork_event_c)
        with wp.ScopedStream(stream_b):
            wp.utils.radix_sort_pairs(kb, vb, n)
        with wp.ScopedStream(stream_c):
            wp.utils.radix_sort_pairs(kc, vc, n)
        main_stream.wait_stream(stream_b, event=join_event_b)
        main_stream.wait_stream(stream_c, event=join_event_c)
        # and one sort on the main stream after the join
        wp.utils.radix_sort_pairs(ka, va, n)

    for _ in range(2):
        _upload_radix_sort_arrays(ka_np, va_np, ka, va)
        _upload_radix_sort_arrays(kb_np, vb_np, kb, vb)
        _upload_radix_sort_arrays(kc_np, vc_np, kc, vc)
        wp.capture_launch(capture.graph)
        _check_radix_sort_arrays(ka_np, va_np, ka, va)
        _check_radix_sort_arrays(kb_np, vb_np, kb, vb)
        _check_radix_sort_arrays(kc_np, vc_np, kc, vc)


def test_segmented_sort_pairs_graph_capture(test, device):
    """Verify segmented sorts that grow the scratch buffer mid-capture replay correctly."""
    rng = np.random.default_rng(123)
    n_small, n_large = 1000, 50000

    def make_arrays(count):
        keys_np = rng.random(count, dtype=np.float32)
        values_np = np.arange(count, dtype=np.int32)
        starts_np = np.array([0, count // 2], dtype=np.int32)
        ends_np = np.array([count // 2, count], dtype=np.int32)
        keys = wp.zeros(2 * count, dtype=wp.float32, device=device)
        values = wp.zeros(2 * count, dtype=wp.int32, device=device)
        starts = wp.array(starts_np, dtype=wp.int32, device=device)
        ends = wp.array(ends_np, dtype=wp.int32, device=device)
        return keys_np, values_np, starts_np, ends_np, keys, values, starts, ends

    def reset_sort_inputs(keys_np, values_np, keys, values):
        count = len(keys_np)
        keys.assign(np.concatenate([keys_np, np.zeros(count, dtype=np.float32)]))
        values.assign(np.concatenate([values_np, np.zeros(count, dtype=np.int32)]))

    def assert_sort_results(keys_np, values_np, starts_np, ends_np, keys, values):
        count = len(keys_np)
        expected_keys = keys_np.copy()
        expected_values = values_np.copy()
        for s, e in zip(starts_np, ends_np, strict=True):
            order = np.argsort(keys_np[s:e], kind="stable")
            expected_keys[s:e] = keys_np[s:e][order]
            expected_values[s:e] = values_np[s:e][order]
        np.testing.assert_array_equal(keys.numpy()[:count], expected_keys)
        np.testing.assert_array_equal(values.numpy()[:count], expected_values)

    small = make_arrays(n_small)
    large = make_arrays(n_large)

    # seed the global scratch buffer with a small sort
    reset_sort_inputs(small[0], small[1], small[4], small[5])
    wp.utils.segmented_sort_pairs(small[4], small[5], n_small, small[6], small[7])

    with wp.ScopedCapture(device, force_module_load=False) as capture:
        wp.utils.segmented_sort_pairs(small[4], small[5], n_small, small[6], small[7])
        wp.utils.segmented_sort_pairs(large[4], large[5], n_large, large[6], large[7])

    for _ in range(2):
        reset_sort_inputs(small[0], small[1], small[4], small[5])
        reset_sort_inputs(large[0], large[1], large[4], large[5])
        wp.capture_launch(capture.graph)
        assert_sort_results(*small[:4], small[4], small[5])
        assert_sort_results(*large[:4], large[4], large[5])


def test_segmented_sort_pairs(test, device):
    keyTypes = [int, wp.float32]

    for keyType in keyTypes:
        keys = wp.array((7, 2, 8, 4, 1, 6, 5, 3, 0, 0, 0, 0, 0, 0, 0, 0), dtype=keyType, device=device)
        values = wp.array((1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0), dtype=int, device=device)
        wp.utils.segmented_sort_pairs(
            keys,
            values,
            8,
            wp.array((0, 4), dtype=int, device=device),
            wp.array((4, 8), dtype=int, device=device),
        )
        assert_np_equal(keys.numpy()[:8], np.array((2, 4, 7, 8, 1, 3, 5, 6)))
        assert_np_equal(values.numpy()[:8], np.array((2, 4, 1, 3, 5, 8, 7, 6)))


def test_radix_sort_pairs_empty(test, device):
    keyTypes = [int, wp.uint32, wp.float32, wp.int64, wp.uint64, wp.float64]

    for keyType in keyTypes:
        keys = wp.array((), dtype=keyType, device=device)
        values = wp.array((), dtype=int, device=device)
        wp.utils.radix_sort_pairs(keys, values, 0)


def test_segmented_sort_pairs_empty(test, device):
    keyTypes = [int, wp.float32]

    for keyType in keyTypes:
        keys = wp.array((), dtype=keyType, device=device)
        values = wp.array((), dtype=int, device=device)
        wp.utils.segmented_sort_pairs(
            keys, values, 0, wp.array((), dtype=int, device=device), wp.array((), dtype=int, device=device)
        )


def test_radix_sort_pairs_error_insufficient_storage(test, device):
    keyTypes = [int, wp.uint32, wp.float32, wp.int64, wp.uint64, wp.float64]

    for keyType in keyTypes:
        keys = wp.array((1, 2, 3), dtype=keyType, device=device)
        values = wp.array((1, 2, 3), dtype=int, device=device)
        with test.assertRaisesRegex(
            RuntimeError,
            r"Keys and values array storage must be large enough to contain 2\*count elements$",
        ):
            wp.utils.radix_sort_pairs(keys, values, 3)


def test_segmented_sort_pairs_error_insufficient_storage(test, device):
    keyTypes = [int, wp.float32]

    for keyType in keyTypes:
        keys = wp.array((1, 2, 3), dtype=keyType, device=device)
        values = wp.array((1, 2, 3), dtype=int, device=device)
        with test.assertRaisesRegex(
            RuntimeError,
            r"Array storage must be large enough to contain 2\*count elements$",
        ):
            wp.utils.segmented_sort_pairs(
                keys,
                values,
                3,
                wp.array((0,), dtype=int, device=device),
                wp.array((3,), dtype=int, device=device),
            )


def test_radix_sort_pairs_error_unsupported_dtype(test, device):
    keyTypes = [wp.int32, wp.uint32, wp.float32, wp.int64, wp.uint64, wp.float64]

    for keyType in keyTypes:
        keys = wp.array((1.0, 2.0, 3.0), dtype=keyType, device=device)
        values = wp.array((1.0, 2.0, 3.0), dtype=wp.float16, device=device)
        with test.assertRaisesRegex(
            RuntimeError,
            rf"Unsupported keys and values data types: {keyType.__name__}, float16$",
        ):
            wp.utils.radix_sort_pairs(keys, values, 1)


def test_radix_sort_pairs_error_invalid_bit_range(test, device):
    keys = wp.array((1, 2, 3, 0, 0, 0), dtype=int, device=device)
    values = wp.array((1, 2, 3, 0, 0, 0), dtype=int, device=device)

    with test.assertRaisesRegex(RuntimeError, r"Invalid radix sort bit range \[4, 2\) for 32-bit keys$"):
        wp.utils.radix_sort_pairs(keys, values, 3, begin_bit=4, end_bit=2)

    with test.assertRaisesRegex(RuntimeError, r"Invalid radix sort bit range \[0, 33\) for 32-bit keys$"):
        wp.utils.radix_sort_pairs(keys, values, 3, end_bit=33)


def test_segmented_sort_pairs_error_unsupported_dtype(test, device):
    keyTypes = [wp.int32, wp.float32]

    for keyType in keyTypes:
        keys = wp.array((1.0, 2.0, 3.0), dtype=keyType, device=device)
        values = wp.array((1.0, 2.0, 3.0), dtype=float, device=device)
        with test.assertRaisesRegex(
            RuntimeError,
            rf"Unsupported data type: {keyType.__name__}$",
        ):
            wp.utils.segmented_sort_pairs(
                keys,
                values,
                1,
                wp.array((0,), dtype=int, device=device),
                wp.array((3,), dtype=int, device=device),
            )


def test_array_sum(test, device):
    for dtype in (wp.float32, wp.float64):
        with test.subTest(dtype=dtype):
            values = wp.array((1.0, 2.0, 3.0), dtype=dtype, device=device)
            test.assertEqual(wp.utils.array_sum(values), 6.0)

            values = wp.array((1.0, 2.0, 3.0), dtype=dtype, device=device)
            result = wp.empty(shape=(1,), dtype=dtype, device=device)
            wp.utils.array_sum(values, out=result)
            test.assertEqual(result.numpy()[0], 6.0)


def test_array_sum_error_out_dtype_mismatch(test, device):
    values = wp.array((1.0, 2.0, 3.0), dtype=wp.float32, device=device)
    result = wp.empty(shape=(1,), dtype=wp.float64, device=device)
    with test.assertRaisesRegex(
        RuntimeError,
        r"out array should have type float32$",
    ):
        wp.utils.array_sum(values, out=result)


def test_array_sum_error_out_shape_mismatch(test, device):
    values = wp.array((1.0, 2.0, 3.0), dtype=wp.float32, device=device)
    result = wp.empty(shape=(2,), dtype=wp.float32, device=device)
    with test.assertRaisesRegex(
        RuntimeError,
        r"out array should have shape \(1,\)$",
    ):
        wp.utils.array_sum(values, out=result)


def test_array_sum_error_unsupported_dtype(test, device):
    values = wp.array((1, 2, 3), dtype=int, device=device)
    with test.assertRaisesRegex(
        RuntimeError,
        r"Unsupported data type: int32$",
    ):
        wp.utils.array_sum(values)


def test_array_inner(test, device):
    for dtype in (wp.float32, wp.float64, wp.vec3):
        a = wp.array((1.0, 2.0, 3.0), dtype=dtype, device=device)
        b = wp.array((1.0, 2.0, 3.0), dtype=dtype, device=device)
        test.assertEqual(wp.utils.array_inner(a, b), 14.0)

        a = wp.array((1.0, 2.0, 3.0), dtype=dtype, device=device)
        b = wp.array((1.0, 2.0, 3.0), dtype=dtype, device=device)
        result = wp.empty(shape=(1,), dtype=wp._src.types.type_scalar_type(dtype), device=device)
        wp.utils.array_inner(a, b, out=result)
        test.assertEqual(result.numpy()[0], 14.0)

    # test with different instances of same type
    a = wp.array((1.0, 2.0, 3.0), dtype=wp.vec3, device=device)
    b = wp.array((1.0, 2.0, 3.0), dtype=wp.types.vector(3, float), device=device)
    test.assertEqual(wp.utils.array_inner(a, b), 14.0)


def test_array_inner_error_sizes_mismatch(test, device):
    a = wp.array((1.0, 2.0), dtype=wp.float32, device=device)
    b = wp.array((1.0, 2.0, 3.0), dtype=wp.float32, device=device)
    with test.assertRaisesRegex(
        RuntimeError,
        r"A and b array storage sizes do not match \(2 vs 3\)$",
    ):
        wp.utils.array_inner(a, b)


def test_array_inner_error_dtypes_mismatch(test, device):
    a = wp.array((1.0, 2.0, 3.0), dtype=wp.float32, device=device)
    b = wp.array((1.0, 2.0, 3.0), dtype=wp.float64, device=device)
    with test.assertRaisesRegex(
        RuntimeError,
        r"A and b array data types do not match \(float32 vs float64\)$",
    ):
        wp.utils.array_inner(a, b)


def test_array_inner_error_out_dtype_mismatch(test, device):
    a = wp.array((1.0, 2.0, 3.0), dtype=wp.float32, device=device)
    b = wp.array((1.0, 2.0, 3.0), dtype=wp.float32, device=device)
    result = wp.empty(shape=(1,), dtype=wp.float64, device=device)
    with test.assertRaisesRegex(
        RuntimeError,
        r"out array should have type float32$",
    ):
        wp.utils.array_inner(a, b, result)


def test_array_inner_error_out_shape_mismatch(test, device):
    a = wp.array((1.0, 2.0, 3.0), dtype=wp.float32, device=device)
    b = wp.array((1.0, 2.0, 3.0), dtype=wp.float32, device=device)
    result = wp.empty(shape=(2,), dtype=wp.float32, device=device)
    with test.assertRaisesRegex(
        RuntimeError,
        r"out array should have shape \(1,\)$",
    ):
        wp.utils.array_inner(a, b, result)


def test_array_inner_error_unsupported_dtype(test, device):
    a = wp.array((1, 2, 3), dtype=int, device=device)
    b = wp.array((1, 2, 3), dtype=int, device=device)
    with test.assertRaisesRegex(
        RuntimeError,
        r"Unsupported data type: int32$",
    ):
        wp.utils.array_inner(a, b)


def test_array_cast(test, device):
    values = wp.array((1, 2, 3), dtype=int, device=device)
    result = wp.empty(3, dtype=float, device=device)
    wp.utils.array_cast(values, result)
    test.assertEqual(result.dtype, wp.float32)
    test.assertEqual(result.shape, (3,))
    assert_np_equal(result.numpy(), np.array((1.0, 2.0, 3.0), dtype=float))

    values = wp.array((1, 2, 3, 4), dtype=int, device=device)
    result = wp.empty((2, 2), dtype=float, device=device)
    wp.utils.array_cast(values, result)
    test.assertEqual(result.dtype, wp.float32)
    test.assertEqual(result.shape, (2, 2))
    assert_np_equal(result.numpy(), np.array(((1.0, 2.0), (3.0, 4.0)), dtype=float))

    values = wp.array(((1, 2), (3, 4)), dtype=wp.vec2, device=device)
    result = wp.zeros(2, dtype=float, device=device)
    wp.utils.array_cast(values, result, count=1)
    test.assertEqual(result.dtype, wp.float32)
    test.assertEqual(result.shape, (2,))
    assert_np_equal(result.numpy(), np.array((1.0, 2.0), dtype=float))

    values = wp.array(((1, 2), (3, 4)), dtype=int, device=device)
    result = wp.zeros((2, 2), dtype=int, device=device)
    wp.utils.array_cast(values, result)
    test.assertEqual(result.dtype, wp.int32)
    test.assertEqual(result.shape, (2, 2))
    assert_np_equal(result.numpy(), np.array(((1, 2), (3, 4)), dtype=int))


def test_array_cast_error_unsupported_partial_cast(test, device):
    values = wp.array(((1, 2), (3, 4)), dtype=int, device=device)
    result = wp.zeros((2, 2), dtype=float, device=device)
    with test.assertRaisesRegex(
        RuntimeError,
        r"Partial cast is not supported for arrays with more than one dimension$",
    ):
        wp.utils.array_cast(values, result, count=1)


def parenthesized_multiline_lambda():
    # qd is intentionally unused to match a two-argument callback signature.
    # fmt: off
    return lambda q, qd: (
        q[0] == 0.0
        and q[1] == 0.0
    )
    # fmt: on


devices = get_test_devices()
cuda_devices = get_cuda_test_devices()


class TestUtils(unittest.TestCase):
    def test_create_warp_function_parenthesized_multiline_lambda(self):
        original_fn = parenthesized_multiline_lambda()
        created_fn, _ = wp.utils.create_warp_function(original_fn)

        self.assertTrue(callable(created_fn))
        q = (0.0, 0.0)
        qd = (0.0, 0.0)
        self.assertEqual(original_fn(q, qd), created_fn(q, qd))

    def test_create_warp_function_name_is_deterministic(self):
        script = (
            "import warp as wp\n"
            "from warp.tests.test_utils import parenthesized_multiline_lambda\n"
            "created_fn, _ = wp.utils.create_warp_function(parenthesized_multiline_lambda())\n"
            'print(f"created-function-key:{created_fn.key}")\n'
        )
        marker = "created-function-key:"
        keys = []

        for seed in ("1", "2"):
            env = os.environ.copy()
            env["PYTHONHASHSEED"] = seed
            result = subprocess.run(
                [sys.executable, "-c", script],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
            self.assertEqual(
                result.returncode,
                0,
                msg=f"Subprocess failed for PYTHONHASHSEED={seed}:\n{result.stderr}",
            )
            matching_lines = [line for line in result.stdout.splitlines() if line.startswith(marker)]
            self.assertEqual(
                len(matching_lines),
                1,
                msg=f"Expected one function-key marker in subprocess output:\n{result.stdout}",
            )
            keys.append(matching_lines[0].removeprefix(marker))

        self.assertEqual(keys[0], keys[1])
        self.assertRegex(keys[0], r"^func_[0-9a-f]{16}$")

    def test_warn(self):
        # Clear any state from prior tests in the same process.
        _logger._warnings_seen.clear()

        # Multiple warnings get printed out each time.
        with contextlib.redirect_stderr(io.StringIO()) as f:
            log_warning("hello, world!")
            log_warning("hello, world!")

        expected = "Warp UserWarning: hello, world!\nWarp UserWarning: hello, world!\n"

        self.assertEqual(f.getvalue(), expected)

        # Multiple similar DeprecationWarnings with once=True get printed out only once.
        with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()) as f:
            warnings.simplefilter("always")
            log_warning("once_msg_utils_test", category=DeprecationWarning, once=True)
            log_warning("once_msg_utils_test", category=DeprecationWarning, once=True)

        expected = "Warp DeprecationWarning: once_msg_utils_test\n"

        self.assertEqual(f.getvalue(), expected)

        # Clear once-seen state for next sub-test.
        _logger._warnings_seen.clear()

        # Multiple different DeprecationWarnings with once=True get printed out each time.
        with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()) as f:
            warnings.simplefilter("always")
            log_warning("foo_once_utils_test", category=DeprecationWarning, once=True)
            log_warning("bar_once_utils_test", category=DeprecationWarning, once=True)

        expected = "Warp DeprecationWarning: foo_once_utils_test\nWarp DeprecationWarning: bar_once_utils_test\n"

        self.assertEqual(f.getvalue(), expected)

    def test_transform_expand(self):
        t = (1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0)
        self.assertEqual(
            wp.transform_expand(t),
            wp.transformf(p=(1.0, 2.0, 3.0), q=(4.0, 3.0, 2.0, 1.0)),
        )

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_array_scan_error_devices_mismatch(self):
        values = wp.zeros(123, dtype=int, device="cpu")
        result = wp.zeros_like(values, device="cuda:0")
        with self.assertRaisesRegex(
            RuntimeError,
            r"In and out array storage devices do not match \(cpu vs cuda:0\)$",
        ):
            wp.utils.array_scan(values, result, True)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_radix_sort_pairs_error_devices_mismatch(self):
        keys = wp.array((1, 2, 3), dtype=int, device="cpu")
        values = wp.array((1, 2, 3), dtype=int, device="cuda:0")
        with self.assertRaisesRegex(
            RuntimeError,
            r"Keys and values array storage devices do not match \(cpu vs cuda:0\)$",
        ):
            wp.utils.radix_sort_pairs(keys, values, 1)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_array_inner_error_out_device_mismatch(self):
        a = wp.array((1.0, 2.0, 3.0), dtype=wp.float32, device="cpu")
        b = wp.array((1.0, 2.0, 3.0), dtype=wp.float32, device="cpu")
        result = wp.empty(shape=(1,), dtype=wp.float32, device="cuda:0")
        with self.assertRaisesRegex(
            RuntimeError,
            r"out storage device should match values array$",
        ):
            wp.utils.array_inner(a, b, result)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_array_sum_error_out_device_mismatch(self):
        values = wp.array((1.0, 2.0, 3.0), dtype=wp.float32, device="cpu")
        result = wp.empty(shape=(1,), dtype=wp.float32, device="cuda:0")
        with self.assertRaisesRegex(
            RuntimeError,
            r"out storage device should match values array$",
        ):
            wp.utils.array_sum(values, out=result)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_array_inner_error_devices_mismatch(self):
        a = wp.array((1.0, 2.0, 3.0), dtype=wp.float32, device="cpu")
        b = wp.array((1.0, 2.0, 3.0), dtype=wp.float32, device="cuda:0")
        with self.assertRaisesRegex(
            RuntimeError,
            r"A and b array storage devices do not match \(cpu vs cuda:0\)$",
        ):
            wp.utils.array_inner(a, b)

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_array_cast_error_devices_mismatch(self):
        values = wp.array((1, 2, 3), dtype=int, device="cpu")
        result = wp.empty(3, dtype=float, device="cuda:0")
        with self.assertRaisesRegex(
            RuntimeError,
            r"Array storage devices do not match \(cpu vs cuda:0\)$",
        ):
            wp.utils.array_cast(values, result)

    def test_mesh_adjacency(self):
        triangles = (
            (0, 3, 1),
            (0, 2, 3),
        )
        adj = wp._src.utils.MeshAdjacency(triangles, len(triangles))
        expected_edges = {
            (0, 3): (0, 3, 1, 2, 0, 1),
            (1, 3): (3, 1, 0, -1, 0, -1),
            (0, 1): (1, 0, 3, -1, 0, -1),
            (0, 2): (0, 2, 3, -1, 1, -1),
            (2, 3): (2, 3, 0, -1, 1, -1),
        }
        edges = {k: (e.v0, e.v1, e.o0, e.o1, e.f0, e.f1) for k, e in adj.edges.items()}
        self.assertDictEqual(edges, expected_edges)

    def test_mesh_adjacency_error_manifold(self):
        triangles = (
            (0, 3, 1),
            (0, 2, 3),
            (3, 0, 1),
        )

        with contextlib.redirect_stdout(io.StringIO()) as f:
            wp._src.utils.MeshAdjacency(triangles, len(triangles))

        self.assertEqual(f.getvalue(), "Detected non-manifold edge\n")

    def test_scoped_timer(self):
        with contextlib.redirect_stdout(io.StringIO()) as f:
            with wp.ScopedTimer("hello"):
                pass

        self.assertRegex(f.getvalue(), r"^hello took \d+\.\d+ ms$")

        with contextlib.redirect_stdout(io.StringIO()) as f:
            with wp.ScopedTimer("hello", detailed=True):
                pass

        self.assertRegex(f.getvalue(), r"^         \d+ function calls in \d+\.\d+ seconds")
        self.assertRegex(f.getvalue(), r"hello took \d+\.\d+ ms$")

    @unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
    def test_scoped_timer_cuda_timing(self):
        """Verify ``ScopedTimer`` returns usable CUDA memory-copy timing results."""
        src = wp.ones(16, dtype=wp.float32, device="cuda:0")
        dst = wp.zeros_like(src)

        with wp.ScopedTimer("copy", print=False, synchronize=True, cuda_filter=wp.TIMING_MEMCPY) as timer:
            wp.copy(dst, src)

        self.assertEqual(len(timer.timing_results), 1)
        result = timer.timing_results[0]
        self.assertEqual(result.device, wp.get_device("cuda:0"))
        self.assertEqual(result.name, "memcpy DtoD")
        self.assertEqual(result.filter, wp.TIMING_MEMCPY)
        self.assertGreaterEqual(result.elapsed, 0.0)


add_function_test(TestUtils, "test_array_scan", test_array_scan, devices=devices)
add_function_test(TestUtils, "test_array_scan_vector", test_array_scan_vector, devices=devices)
add_function_test(TestUtils, "test_array_scan_strided_views", test_array_scan_strided_views, devices=devices)
add_function_test(TestUtils, "test_array_scan_empty", test_array_scan_empty, devices=devices)
add_function_test(
    TestUtils, "test_array_scan_error_sizes_mismatch", test_array_scan_error_sizes_mismatch, devices=devices
)
add_function_test(
    TestUtils, "test_array_scan_error_dtypes_mismatch", test_array_scan_error_dtypes_mismatch, devices=devices
)
add_function_test(
    TestUtils, "test_array_scan_error_unsupported_dtype", test_array_scan_error_unsupported_dtype, devices=devices
)
add_function_test(TestUtils, "test_radix_sort_pairs", test_radix_sort_pairs, devices=devices)
add_function_test(
    TestUtils,
    "test_radix_sort_pairs_graph_capture",
    test_radix_sort_pairs_graph_capture,
    devices=get_selected_cuda_test_devices(),
)
add_function_test(
    TestUtils,
    "test_radix_sort_pairs_graph_capture_independent_graphs",
    test_radix_sort_pairs_graph_capture_independent_graphs,
    devices=get_selected_cuda_test_devices(),
)
add_function_test(
    TestUtils,
    "test_radix_sort_pairs_capture_if_forked_stream",
    test_radix_sort_pairs_capture_if_forked_stream,
    devices=get_selected_cuda_test_devices(),
)
add_function_test(
    TestUtils,
    "test_radix_sort_pairs_graph_capture_multi_stream",
    test_radix_sort_pairs_graph_capture_multi_stream,
    devices=get_selected_cuda_test_devices(),
)
add_function_test(
    TestUtils,
    "test_radix_sort_pairs_capture_while",
    test_radix_sort_pairs_capture_while,
    devices=get_selected_cuda_test_devices(),
)
add_function_test(
    TestUtils,
    "test_radix_sort_pairs_graph_capture_stream_destruction",
    test_radix_sort_pairs_graph_capture_stream_destruction,
    devices=get_selected_cuda_test_devices(),
)
add_function_test(
    TestUtils,
    "test_radix_sort_pairs_graph_capture_pause",
    test_radix_sort_pairs_graph_capture_pause,
    devices=get_selected_cuda_test_devices(),
)
add_function_test(
    TestUtils,
    "test_segmented_sort_pairs_graph_capture",
    test_segmented_sort_pairs_graph_capture,
    devices=get_selected_cuda_test_devices(),
)
add_function_test(TestUtils, "test_radix_sort_pairs_signed_keys", test_radix_sort_pairs_signed_keys, devices=devices)
add_function_test(TestUtils, "test_radix_sort_pairs_bit_range", test_radix_sort_pairs_bit_range, devices=devices)
add_function_test(TestUtils, "test_radix_sort_pairs_value_types", test_radix_sort_pairs_value_types, devices=devices)
add_function_test(
    TestUtils,
    "test_radix_sort_pairs_64_bit_keys_8_byte_values",
    test_radix_sort_pairs_64_bit_keys_8_byte_values,
    devices=devices,
)
add_function_test(TestUtils, "test_radix_sort_pairs_empty", test_radix_sort_pairs_empty, devices=devices)
add_function_test(
    TestUtils,
    "test_radix_sort_pairs_error_insufficient_storage",
    test_radix_sort_pairs_error_insufficient_storage,
    devices=devices,
)
add_function_test(
    TestUtils,
    "test_radix_sort_pairs_error_unsupported_dtype",
    test_radix_sort_pairs_error_unsupported_dtype,
    devices=devices,
)
add_function_test(
    TestUtils,
    "test_radix_sort_pairs_error_invalid_bit_range",
    test_radix_sort_pairs_error_invalid_bit_range,
    devices=devices,
)
add_function_test(
    TestUtils,
    "test_radix_sort_pairs_error_non_contiguous",
    test_radix_sort_pairs_error_non_contiguous,
    devices=devices,
)
add_function_test(TestUtils, "test_segmented_sort_pairs", test_segmented_sort_pairs, devices=devices)
add_function_test(TestUtils, "test_segmented_sort_pairs_empty", test_segmented_sort_pairs, devices=devices)
add_function_test(
    TestUtils,
    "test_segmented_sort_pairs_error_insufficient_storage",
    test_segmented_sort_pairs_error_insufficient_storage,
    devices=devices,
)
add_function_test(
    TestUtils,
    "test_segmented_sort_pairs_error_unsupported_dtype",
    test_segmented_sort_pairs_error_unsupported_dtype,
    devices=devices,
)
add_function_test(TestUtils, "test_array_sum", test_array_sum, devices=devices)
add_function_test(
    TestUtils, "test_array_sum_error_out_dtype_mismatch", test_array_sum_error_out_dtype_mismatch, devices=devices
)
add_function_test(
    TestUtils, "test_array_sum_error_out_shape_mismatch", test_array_sum_error_out_shape_mismatch, devices=devices
)
add_function_test(
    TestUtils, "test_array_sum_error_unsupported_dtype", test_array_sum_error_unsupported_dtype, devices=devices
)
add_function_test(TestUtils, "test_array_inner", test_array_inner, devices=devices)
add_function_test(
    TestUtils, "test_array_inner_error_sizes_mismatch", test_array_inner_error_sizes_mismatch, devices=devices
)
add_function_test(
    TestUtils, "test_array_inner_error_dtypes_mismatch", test_array_inner_error_dtypes_mismatch, devices=devices
)
add_function_test(
    TestUtils, "test_array_inner_error_out_dtype_mismatch", test_array_inner_error_out_dtype_mismatch, devices=devices
)
add_function_test(
    TestUtils, "test_array_inner_error_out_shape_mismatch", test_array_inner_error_out_shape_mismatch, devices=devices
)
add_function_test(
    TestUtils, "test_array_inner_error_unsupported_dtype", test_array_inner_error_unsupported_dtype, devices=devices
)
add_function_test(TestUtils, "test_array_cast", test_array_cast, devices=devices)
add_function_test(
    TestUtils,
    "test_array_scan_error_device_failure_is_reported",
    test_array_scan_error_device_failure_is_reported,
    devices=cuda_devices,
)
add_function_test(
    TestUtils,
    "test_array_cast_error_unsupported_partial_cast",
    test_array_cast_error_unsupported_partial_cast,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
