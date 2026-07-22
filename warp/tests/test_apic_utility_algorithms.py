# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for APIC capture of utility algorithms.

These tests intentionally use two ordered ``TestCase`` classes. On Linux with
glibc 2.41, a long-lived worker aborts when it runs these CUDA tests in order:

1. ``test_capture_with_array_scan_extended_metadata_cuda_0``
2. ``test_capture_with_radix_sort_extended_metadata_cuda_0``
3. ``test_capture_with_runlength_encode_cuda_0``
4. ``test_capture_with_segmented_sort_cuda_0``

The abort occurs during CUDA graph cleanup in ``cudaGraphExecDestroy`` and
prints ``free(): invalid next size (fast)``. It reproduces with CPython 3.12
and 3.14. See https://github.com/NVIDIA/warp/issues/1678.
"""

import os
import tempfile
import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel
def fill_descending_kernel(keys: wp.array[float], values: wp.array[wp.int32], count: wp.int32):
    i = wp.tid()
    keys[i] = float(count - 1 - i)
    values[i] = i


@wp.kernel
def fill_runs_kernel(values: wp.array[wp.int32]):
    # Produces consecutive runs of length 3: [0,0,0,1,1,1,2,2,2,...].
    i = wp.tid()
    values[i] = i // 3


# Preserve these separate, ordered TestCase classes; see the module docstring.
class TestApicSegmentedSort(unittest.TestCase):
    pass


class TestApicUtilityAlgorithms(unittest.TestCase):
    pass


def test_capture_with_array_scan(test, device):
    """Capture inclusive and exclusive array scans under APIC and verify replay.

    Newton's broad/narrow-phase calls into wp_array_scan_*_host, whose internal
    scratch -> output memcpy used to fail apic_resolve_ptr and silently drop.
    After the fix, scan internals use plain memcpy.
    """
    n = 32
    src = wp.array(np.ones(n, dtype=np.int32), dtype=wp.int32, device=device)
    dst_in = wp.zeros(n, dtype=wp.int32, device=device)
    dst_ex = wp.zeros(n, dtype=wp.int32, device=device)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.utils.array_scan(src, dst_in, inclusive=True)
        wp.utils.array_scan(src, dst_ex, inclusive=False)

    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    # Inclusive scan of [1]*n is [1,2,...,n]; exclusive is [0,1,...,n-1].
    np.testing.assert_allclose(dst_in.numpy(), np.arange(1, n + 1, dtype=np.int32))
    np.testing.assert_allclose(dst_ex.numpy(), np.arange(0, n, dtype=np.int32))


def test_save_load_array_scan_replay_with_updated_input(test, device):
    """Recompute a saved and loaded array scan against updated input.

    ``wp.utils.array_scan`` must be recorded into the byte stream so a saved +
    loaded graph recomputes against the current input rather than returning
    capture-time output.
    """
    original = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    updated = np.array([5, 1, 4, 1, 3], dtype=np.int32)

    src = wp.array(original, dtype=wp.int32, device=device)
    dst = wp.zeros_like(src)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.utils.array_scan(src, dst, inclusive=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "exclusive_scan")
        wp.capture_save(capture.graph, path, inputs={"src": src}, outputs={"dst": dst})

        loaded = wp.capture_load(path, device=device)
        loaded.set_param("src", wp.array(updated, dtype=wp.int32, device=device))
        wp.capture_launch(loaded)

        actual = wp.empty_like(dst)
        loaded.get_param("dst", actual)

    expected = np.zeros_like(updated)
    expected[1:] = np.cumsum(updated[:-1], dtype=updated.dtype)
    np.testing.assert_allclose(actual.numpy(), expected)


def test_capture_with_array_scan_extended_metadata(test, device):
    """Preserve dtype, vector lanes, and 1D strides in APIC scan records."""
    n = 6

    base = wp.array(np.arange(0, 2 * n, dtype=np.int64), dtype=wp.int64, device=device)
    dst_base = wp.zeros(2 * n, dtype=wp.int64, device=device)
    src = base[0 : 2 * n : 2]
    dst = dst_base[1 : 2 * n : 2]

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.utils.array_scan(src, dst, inclusive=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "strided_int64_scan")
        wp.capture_save(capture.graph, path, inputs={"base": base}, outputs={"dst_base": dst_base})

        updated_base_np = np.array([10, -1, 2, -1, 7, -1, 1, -1, 3, -1, 4, -1], dtype=np.int64)
        loaded = wp.capture_load(path, device=device)
        loaded.set_param("base", wp.array(updated_base_np, dtype=wp.int64, device=device))
        wp.capture_launch(loaded)

        actual = wp.empty_like(dst_base)
        loaded.get_param("dst_base", actual)

    expected = np.zeros(2 * n, dtype=np.int64)
    expected[1::2] = np.cumsum(updated_base_np[::2], dtype=np.int64)
    np.testing.assert_array_equal(actual.numpy(), expected)

    vec_src_np = np.array([[1.0, 10.0, 100.0], [2.0, 20.0, 200.0], [3.0, 30.0, 300.0]], dtype=np.float32)
    vec_src = wp.array(vec_src_np, dtype=wp.vec3, device=device)
    vec_dst = wp.zeros_like(vec_src)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture_vec:
        wp.utils.array_scan(vec_src, vec_dst, inclusive=False)
    wp.capture_launch(capture_vec.graph)
    np.testing.assert_allclose(
        vec_dst.numpy(), np.vstack([np.zeros(3, dtype=np.float32), np.cumsum(vec_src_np[:-1], axis=0)])
    )

    f64_src_np = np.array([0.25, 0.5, 1.25, 2.0], dtype=np.float64)
    f64_src = wp.array(f64_src_np, dtype=wp.float64, device=device)
    f64_dst = wp.zeros_like(f64_src)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture_f64:
        wp.utils.array_scan(f64_src, f64_dst, inclusive=True)
    wp.capture_launch(capture_f64.graph)
    np.testing.assert_allclose(f64_dst.numpy(), np.cumsum(f64_src_np))


def test_capture_with_segmented_sort(test, device):
    """Replay a captured segmented sort so it actually re-sorts on launch.

    wp.utils.segmented_sort_pairs on CPU dispatches to a host function that
    wasn't recorded into the APIC byte stream, so under graph capture/replay the
    sort silently didn't run and data stayed unsorted (Newton SAP broadphase
    ~10x slowdown). The fill kernel runs inside the capture so the keys at replay
    time differ from capture time, forcing the sort to actually replay.
    """
    n = 64
    # segmented_sort_pairs requires 2*count capacity for sort scratch.
    keys = wp.zeros(2 * n, dtype=wp.float32, device=device)
    values = wp.zeros(2 * n, dtype=wp.int32, device=device)
    segments = wp.array(np.array([0, n], dtype=np.int32), dtype=wp.int32, device=device)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.launch(fill_descending_kernel, dim=n, inputs=[keys, values, n], device=device)
        wp.utils.segmented_sort_pairs(keys=keys, values=values, count=n, segment_start_indices=segments)

    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    # Correct sort turns descending keys [n-1..0] into ascending [0..n-1],
    # and carries the original indices into values in reverse.
    np.testing.assert_allclose(keys.numpy()[:n], np.arange(0, n, dtype=np.float32))
    np.testing.assert_allclose(values.numpy()[:n], np.arange(n - 1, -1, -1, dtype=np.int32))


def test_save_load_segmented_sort(test, device):
    """Re-sort a saved and loaded segmented sort on replay.

    A captured segmented sort must be recorded into the byte stream so a saved +
    loaded graph re-sorts on replay rather than returning capture-time (unsorted)
    data.
    """
    n = 16
    keys = wp.zeros(2 * n, dtype=wp.float32, device=device)
    values = wp.zeros(2 * n, dtype=wp.int32, device=device)
    segments = wp.array(np.array([0, n], dtype=np.int32), dtype=wp.int32, device=device)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.launch(fill_descending_kernel, dim=n, inputs=[keys, values, n], device=device)
        wp.utils.segmented_sort_pairs(keys=keys, values=values, count=n, segment_start_indices=segments)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "segmented_sort")
        wp.capture_save(capture.graph, path, outputs={"keys": keys})

        loaded = wp.capture_load(path, device=device)
        wp.capture_launch(loaded)
        wp.synchronize_device(device)

        result = wp.empty(2 * n, dtype=wp.float32, device=device)
        loaded.get_param("keys", result)
        np.testing.assert_allclose(result.numpy()[:n], np.arange(0, n, dtype=np.float32))


def test_save_load_segmented_sort_explicit_end(test, device):
    """Match the recorded start-region span for explicit segment-end arrays.

    With explicit ``segment_end_indices`` the start array holds only
    ``num_segments`` entries (not ``num_segments + 1``), so the recorded
    start-region span must match the array. The earlier code always claimed
    ``num_segments + 1`` entries, over-running the explicit-end start array so
    save/load replay failed pointer resolution.
    """
    n = 16
    half = n // 2
    keys = wp.zeros(2 * n, dtype=wp.float32, device=device)
    values = wp.zeros(2 * n, dtype=wp.int32, device=device)
    # Two segments with SEPARATE start/end arrays, each num_segments (=2) entries.
    seg_start = wp.array(np.array([0, half], dtype=np.int32), dtype=wp.int32, device=device)
    seg_end = wp.array(np.array([half, n], dtype=np.int32), dtype=wp.int32, device=device)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.launch(fill_descending_kernel, dim=n, inputs=[keys, values, n], device=device)
        wp.utils.segmented_sort_pairs(
            keys=keys,
            values=values,
            count=n,
            segment_start_indices=seg_start,
            segment_end_indices=seg_end,
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "segmented_sort_explicit")
        wp.capture_save(capture.graph, path, outputs={"keys": keys})

        loaded = wp.capture_load(path, device=device)
        wp.capture_launch(loaded)
        wp.synchronize_device(device)

        result = wp.empty(2 * n, dtype=wp.float32, device=device)
        loaded.get_param("keys", result)

    # Each segment is sorted ascending independently: descending fill
    # [n-1 .. 0] becomes [half .. n-1] for segment 0 and [0 .. half-1] for segment 1.
    expected = np.concatenate([np.arange(half, n, dtype=np.float32), np.arange(0, half, dtype=np.float32)])
    np.testing.assert_allclose(result.numpy()[:n], expected)


def test_capture_with_radix_sort(test, device):
    """Replay a captured radix sort so it actually re-sorts on launch.

    wp.utils.radix_sort_pairs on CPU dispatches to a host function
    (wp_radix_sort_pairs_*_host) that, like the segmented sort, was invisible to
    the APIC byte stream and so didn't replay. The fill kernel runs inside the
    capture so replay-time keys differ from capture time.
    """
    n = 64
    keys = wp.zeros(2 * n, dtype=wp.float32, device=device)
    values = wp.zeros(2 * n, dtype=wp.int32, device=device)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.launch(fill_descending_kernel, dim=n, inputs=[keys, values, n], device=device)
        wp.utils.radix_sort_pairs(keys, values, n)

    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    np.testing.assert_allclose(keys.numpy()[:n], np.arange(0, n, dtype=np.float32))
    np.testing.assert_allclose(values.numpy()[:n], np.arange(n - 1, -1, -1, dtype=np.int32))


def test_capture_with_radix_sort_extended_metadata(test, device):
    """Preserve key dtype, bit range, and value size in APIC radix-sort records."""
    n = 4
    keys_np = np.array([0x0201, 0x0102, 0x0200, 0x0101, 0, 0, 0, 0], dtype=np.uint32)
    values_np = np.array([10, 20, 30, 40, 0, 0, 0, 0], dtype=np.int64)
    keys = wp.array(keys_np, dtype=wp.uint32, device=device)
    values = wp.array(values_np, dtype=wp.int64, device=device)

    f64_keys_np = np.array([3.0, 1.0, 2.0, 0.0, 0.0, 0.0], dtype=np.float64)
    f64_values_np = np.array([0, 1, 2, 0, 0, 0], dtype=np.int32)
    f64_keys = wp.array(f64_keys_np, dtype=wp.float64, device=device)
    f64_values = wp.array(f64_values_np, dtype=wp.int32, device=device)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.utils.radix_sort_pairs(keys, values, n, begin_bit=8, end_bit=16)
        wp.utils.radix_sort_pairs(f64_keys, f64_values, 3)

    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)

    np.testing.assert_array_equal(keys.numpy()[:n], np.array([0x0102, 0x0101, 0x0201, 0x0200], dtype=np.uint32))
    np.testing.assert_array_equal(values.numpy()[:n], np.array([20, 40, 10, 30], dtype=np.int64))
    np.testing.assert_allclose(f64_keys.numpy()[:3], np.array([1.0, 2.0, 3.0], dtype=np.float64))
    np.testing.assert_array_equal(f64_values.numpy()[:3], np.array([1, 2, 0], dtype=np.int32))


def test_capture_with_runlength_encode(test, device):
    """Replay a captured run-length encode so it restores the correct runs.

    wp.utils.runlength_encode on CPU dispatches to a host function
    (wp_runlength_encode_int_host) that, like the sorts, was invisible to the
    APIC byte stream and so didn't replay. The outputs are overwritten with
    sentinels after capture, so only a replayed encode can restore the correct
    runs; without recording the op, replay leaves the sentinels in place.
    """
    n = 9
    values = wp.zeros(n, dtype=wp.int32, device=device)
    run_values = wp.zeros(n, dtype=wp.int32, device=device)
    run_lengths = wp.zeros(n, dtype=wp.int32, device=device)
    run_count = wp.zeros(1, dtype=wp.int32, device=device)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.launch(fill_runs_kernel, dim=n, inputs=[values], device=device)
        wp.utils.runlength_encode(values, run_values, run_lengths, run_count=run_count, value_count=n)

    # Clobber the outputs so a no-op replay cannot leave capture-time values behind.
    run_values.fill_(-1)
    run_lengths.fill_(-1)
    run_count.fill_(-1)

    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    # [0,0,0,1,1,1,2,2,2] -> 3 runs of value 0/1/2, each length 3.
    test.assertEqual(int(run_count.numpy()[0]), 3)
    np.testing.assert_array_equal(run_values.numpy()[:3], np.array([0, 1, 2], dtype=np.int32))
    np.testing.assert_array_equal(run_lengths.numpy()[:3], np.array([3, 3, 3], dtype=np.int32))


def test_runlength_encode_host_return_rejected_during_cpu_apic_capture(test, device):
    """Reject the host-return run-length encode form during CPU APIC capture.

    Without an explicit ``run_count`` array the run count is returned on the
    host, which needs a D2H readback that cannot be recorded into the APIC byte
    stream, so capture raises ``NotImplementedError``. A zero ``value_count``
    short-circuits to a plain 0 before any readback, so it still captures
    cleanly.
    """
    values = wp.array(np.array([1, 1, 2], dtype=np.int32), dtype=wp.int32, device=device)
    run_values = wp.zeros(3, dtype=wp.int32, device=device)
    run_lengths = wp.zeros(3, dtype=wp.int32, device=device)

    with test.assertRaises(NotImplementedError):
        with wp.ScopedCapture(device=device, apic=True, force_module_load=False):
            wp.utils.runlength_encode(values, run_values, run_lengths)

    with wp.ScopedCapture(device=device, apic=True, force_module_load=False):
        test.assertEqual(wp.utils.runlength_encode(values, run_values, run_lengths, value_count=0), 0)


devices = get_test_devices()

add_function_test(
    TestApicSegmentedSort,
    "test_capture_with_segmented_sort",
    test_capture_with_segmented_sort,
    devices=devices,
)
add_function_test(
    TestApicSegmentedSort,
    "test_save_load_segmented_sort",
    test_save_load_segmented_sort,
    devices=devices,
)
add_function_test(
    TestApicSegmentedSort,
    "test_save_load_segmented_sort_explicit_end",
    test_save_load_segmented_sort_explicit_end,
    devices=devices,
)
add_function_test(
    TestApicUtilityAlgorithms,
    "test_capture_with_array_scan",
    test_capture_with_array_scan,
    devices=devices,
)
add_function_test(
    TestApicUtilityAlgorithms,
    "test_save_load_array_scan_replay_with_updated_input",
    test_save_load_array_scan_replay_with_updated_input,
    devices=devices,
)
add_function_test(
    TestApicUtilityAlgorithms,
    "test_capture_with_array_scan_extended_metadata",
    test_capture_with_array_scan_extended_metadata,
    devices=devices,
)
add_function_test(
    TestApicUtilityAlgorithms,
    "test_capture_with_radix_sort",
    test_capture_with_radix_sort,
    devices=devices,
)
add_function_test(
    TestApicUtilityAlgorithms,
    "test_capture_with_radix_sort_extended_metadata",
    test_capture_with_radix_sort_extended_metadata,
    devices=devices,
)
add_function_test(
    TestApicUtilityAlgorithms,
    "test_capture_with_runlength_encode",
    test_capture_with_runlength_encode,
    devices=devices,
)
add_function_test(
    TestApicUtilityAlgorithms,
    "test_runlength_encode_host_return_rejected_during_cpu_apic_capture",
    test_runlength_encode_host_return_rejected_during_cpu_apic_capture,
    devices=[d for d in devices if d.is_cpu],
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
