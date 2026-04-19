# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import io
import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


def _core():
    """Return the native core library handle (lazy, after wp.init)."""
    from warp._src.context import runtime  # noqa: PLC0415

    return runtime.core


def test_basic_tracking(test, device):
    with wp.ScopedMemoryTracker("test_basic", print=False):
        initial_count = _core().wp_alloc_tracker_get_total_alloc_count()
        a = wp.zeros(100, dtype=wp.float32, device=device)
        final_count = _core().wp_alloc_tracker_get_total_alloc_count()

    test.assertGreater(final_count, initial_count)
    test.assertGreaterEqual(_core().wp_alloc_tracker_get_total_alloc_bytes(), 100 * 4)

    del a


def test_free_tracking(test, device):
    with wp.ScopedMemoryTracker("test_free", print=False):
        _core().wp_alloc_tracker_reset()
        a = wp.zeros(64, dtype=wp.float32, device=device)
        live_before = _core().wp_alloc_tracker_get_live_count()
        del a
        gc.collect()
        live_after = _core().wp_alloc_tracker_get_live_count()

    test.assertGreater(live_before, 0)
    test.assertLess(live_after, live_before)


def test_scope_nesting(test, device):
    with wp.ScopedMemoryTracker("outer", print=False):
        _core().wp_alloc_tracker_reset()
        a = wp.zeros(10, dtype=wp.float32, device=device)
        with wp.ScopedMemoryTracker("inner", print=False):
            b = wp.zeros(20, dtype=wp.float32, device=device)

        report = _core().wp_alloc_tracker_report(0, 10).decode("utf-8")

    test.assertIn("outer", report)
    test.assertIn("outer/inner", report)

    del a, b


def test_report_output(test, device):
    buf = io.StringIO()
    with wp.ScopedMemoryTracker("report_test", print=False) as tracker:
        _core().wp_alloc_tracker_reset()
        a = wp.zeros(256, dtype=wp.float32, device=device)

    tracker.report(file=buf)
    output = buf.getvalue()

    test.assertIn("Allocation Tracking Report", output)
    test.assertIn("Total allocations:", output)
    test.assertIn("Live allocations:", output)

    del a


def test_no_overhead_when_inactive(test, device):
    """Verify that the C++ tracker is not enabled when no tracker is active."""
    old = wp.config.track_memory
    wp.config.track_memory = False
    try:
        test.assertFalse(_core().wp_alloc_tracker_is_enabled())
    finally:
        wp.config.track_memory = old


def test_callsite_capture(test, device):
    with wp.ScopedMemoryTracker("callsite", print=False):
        _core().wp_alloc_tracker_reset()
        a = wp.zeros(32, dtype=wp.float32, device=device)

        report = _core().wp_alloc_tracker_report(0, 10).decode("utf-8")

    test.assertIn("array[float32, 32]", report)

    for line in report.splitlines():
        if "(native)" in line:
            test.assertRegex(line, r"\(native:\w+\)")

    del a


def test_clear(test, device):
    with wp.ScopedMemoryTracker("clear_test", print=False) as tracker:
        a = wp.zeros(100, dtype=wp.float32, device=device)
        test.assertGreater(_core().wp_alloc_tracker_get_live_count(), 0)

        tracker.clear()
        test.assertEqual(_core().wp_alloc_tracker_get_live_count(), 0)
        test.assertEqual(_core().wp_alloc_tracker_get_total_alloc_count(), 0)

    del a


def test_native_hashgrid(test, device):
    """Verify that C++ HashGrid allocations are captured with a descriptive label."""
    with wp.ScopedMemoryTracker("hashgrid", print=False):
        _core().wp_alloc_tracker_reset()

        grid = wp.HashGrid(dim_x=10, dim_y=10, dim_z=10, device=device)

        live = _core().wp_alloc_tracker_get_live_count()
        total_bytes = _core().wp_alloc_tracker_get_current_bytes()
        report = _core().wp_alloc_tracker_report(0, 10).decode("utf-8")

    test.assertGreater(live, 0, "HashGrid should produce tracked allocations")
    test.assertGreater(total_bytes, 0)
    test.assertIn("(native:hashgrid)", report)

    del grid


def test_native_mesh(test, device):
    """Verify that C++ Mesh (BVH) allocations are captured with descriptive labels."""
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    indices = np.array([0, 1, 2, 0, 2, 3, 0, 1, 3, 1, 2, 3], dtype=np.int32)

    with wp.ScopedMemoryTracker("mesh", print=False):
        _core().wp_alloc_tracker_reset()

        mesh = wp.Mesh(
            points=wp.array(vertices, dtype=wp.vec3, device=device),
            indices=wp.array(indices, dtype=int, device=device),
        )

        live = _core().wp_alloc_tracker_get_live_count()
        total_bytes = _core().wp_alloc_tracker_get_current_bytes()
        report = _core().wp_alloc_tracker_report(0, 10).decode("utf-8")

    test.assertGreater(live, 0, "Mesh should produce tracked allocations")
    test.assertGreater(total_bytes, 0)
    test.assertIn("(native:mesh)", report)

    del mesh


def test_peak_bytes(test, device):
    """Verify that peak bytes tracks the high-water mark correctly."""
    with wp.ScopedMemoryTracker("peak", print=False):
        _core().wp_alloc_tracker_reset()

        a = wp.zeros(1000, dtype=wp.float32, device=device)
        peak_after_alloc = _core().wp_alloc_tracker_get_peak_bytes()

        del a
        gc.collect()

        peak_after_free = _core().wp_alloc_tracker_get_peak_bytes()
        current_after_free = _core().wp_alloc_tracker_get_current_bytes()

    test.assertGreaterEqual(peak_after_alloc, 1000 * 4)
    test.assertEqual(peak_after_alloc, peak_after_free, "Peak should not decrease after free")
    test.assertLess(current_after_free, peak_after_free)


def test_pinned_memory(test, device):
    """Verify that pinned host allocations are tracked and reported."""
    if not wp.is_cuda_available():
        return

    with wp.ScopedMemoryTracker("pinned", print=False):
        _core().wp_alloc_tracker_reset()

        a = wp.zeros(512, dtype=wp.float32, device="cpu", pinned=True)

        report = _core().wp_alloc_tracker_report(0, 10).decode("utf-8")
        live = _core().wp_alloc_tracker_get_live_count()

    test.assertGreater(live, 0)
    test.assertIn("pinned", report.lower())

    del a


def test_print_memory_report_error_when_inactive(test, device):
    """Verify that wp.print_memory_report() raises when tracking is not active."""
    test.assertFalse(_core().wp_alloc_tracker_is_enabled())
    with test.assertRaises(RuntimeError):
        wp.print_memory_report()


def test_report_after_exit(test, device):
    """Verify that report() can be called after __exit__ (reads internal state)."""
    buf_inside = io.StringIO()
    buf_after = io.StringIO()

    with wp.ScopedMemoryTracker("after_exit", print=False) as tracker:
        _core().wp_alloc_tracker_reset()
        a = wp.zeros(128, dtype=wp.float32, device=device)
        live_inside = _core().wp_alloc_tracker_get_live_count()
        tracker.report(file=buf_inside)
        del a
        gc.collect()
        live_after_free = _core().wp_alloc_tracker_get_live_count()

    tracker.report(file=buf_after)

    test.assertGreater(live_inside, 0)
    test.assertLess(live_after_free, live_inside, "Live count should decrease after freeing the array")
    test.assertIn("Allocation Tracking Report", buf_inside.getvalue())
    test.assertIn("Allocation Tracking Report", buf_after.getvalue())


def test_multithreaded_scopes(test, device):
    """Verify that thread-local scope stacks are independent."""
    import threading  # noqa: PLC0415

    results = {}

    def worker(name, size):
        with wp.ScopedMemoryTracker(name, print=False):
            a = wp.zeros(size, dtype=wp.float32, device=device)
            with wp.ScopedMemoryTracker("inner", print=False):
                b = wp.zeros(size, dtype=wp.float32, device=device)
            report = _core().wp_alloc_tracker_report(0, 10).decode("utf-8")
            results[name] = report
            del a, b

    with wp.ScopedMemoryTracker("mt_root", print=False):
        _core().wp_alloc_tracker_reset()

        t1 = threading.Thread(target=worker, args=("thread_a", 100))
        t2 = threading.Thread(target=worker, args=("thread_b", 200))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

    test.assertIn("thread_a", results.get("thread_a", ""))
    test.assertIn("thread_b", results.get("thread_b", ""))


def test_native_label_hashgrid_on_cpu(test, device):
    """Verify that host-side HashGrid allocations are tracked with labels."""
    if wp.get_device(device).is_cuda:
        return

    with wp.ScopedMemoryTracker("hashgrid_cpu", print=False):
        _core().wp_alloc_tracker_reset()

        grid = wp.HashGrid(dim_x=5, dim_y=5, dim_z=5, device=device)

        report = _core().wp_alloc_tracker_report(0, 10).decode("utf-8")

    test.assertIn("(native:hashgrid)", report)

    del grid


def test_native_label_bvh_on_cpu(test, device):
    """Verify that host-side BVH allocations are tracked with labels."""
    if wp.get_device(device).is_cuda:
        return

    lowers = wp.array(np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32), dtype=wp.vec3, device=device)
    uppers = wp.array(np.array([[1, 1, 1], [2, 2, 2]], dtype=np.float32), dtype=wp.vec3, device=device)

    with wp.ScopedMemoryTracker("bvh_cpu", print=False):
        _core().wp_alloc_tracker_reset()

        bvh = wp.Bvh(lowers, uppers)

        report = _core().wp_alloc_tracker_report(0, 10).decode("utf-8")

    test.assertIn("(native:bvh)", report)

    del bvh


devices = get_test_devices()


class TestAllocTracker(unittest.TestCase):
    pass


add_function_test(TestAllocTracker, "test_basic_tracking", test_basic_tracking, devices=devices)
add_function_test(TestAllocTracker, "test_free_tracking", test_free_tracking, devices=devices)
add_function_test(TestAllocTracker, "test_scope_nesting", test_scope_nesting, devices=devices)
add_function_test(TestAllocTracker, "test_report_output", test_report_output, devices=devices)
add_function_test(TestAllocTracker, "test_no_overhead_when_inactive", test_no_overhead_when_inactive, devices=devices)
add_function_test(TestAllocTracker, "test_callsite_capture", test_callsite_capture, devices=devices)
add_function_test(TestAllocTracker, "test_clear", test_clear, devices=devices)
add_function_test(TestAllocTracker, "test_native_hashgrid", test_native_hashgrid, devices=devices)
add_function_test(TestAllocTracker, "test_native_mesh", test_native_mesh, devices=devices)
add_function_test(TestAllocTracker, "test_peak_bytes", test_peak_bytes, devices=devices)
add_function_test(TestAllocTracker, "test_pinned_memory", test_pinned_memory, devices=devices)
add_function_test(
    TestAllocTracker,
    "test_print_memory_report_error_when_inactive",
    test_print_memory_report_error_when_inactive,
    devices=devices,
)
add_function_test(TestAllocTracker, "test_report_after_exit", test_report_after_exit, devices=devices)
add_function_test(TestAllocTracker, "test_multithreaded_scopes", test_multithreaded_scopes, devices=devices)
add_function_test(
    TestAllocTracker, "test_native_label_hashgrid_on_cpu", test_native_label_hashgrid_on_cpu, devices=devices
)
add_function_test(TestAllocTracker, "test_native_label_bvh_on_cpu", test_native_label_bvh_on_cpu, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
