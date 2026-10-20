# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for resources released while a CUDA graph capture is in progress."""

import contextlib
import ctypes
import gc
import os
import sys
import tempfile
import threading
import time
import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


class _SecondaryContext:
    """A second CUDA context on the same GPU, used as a distinct Warp device."""

    def __init__(self, device):
        core = wp._src.context.runtime.core
        self.context = core.wp_cuda_context_create(device.ordinal)
        if not self.context:
            raise RuntimeError(f"Failed to create a CUDA context on {device}")
        # creating a context pushes it; setting the previous one instead would leave an extra stack entry
        core.wp_cuda_context_pop_current()
        self.alias = f"{device.alias}_capture_release_test"
        self.device = wp.map_cuda_device(self.alias, self.context)

    def close(self, synchronize=True):
        if synchronize:
            wp.synchronize()
        wp.unmap_cuda_device(self.alias)
        # release the device's default stream while its context is still alive
        self.device = None
        gc.collect()
        wp._src.context.runtime.core.wp_cuda_context_destroy(self.context)


def _capture_while_releasing(device, resources):
    """Capture a fill on ``device`` and clear ``resources`` while the capture is in progress."""
    dst = wp.ones(4, dtype=float, device=device)
    # unlike wp.ScopedCapture, leave the current device as the capture left it
    device.make_current()

    wp.capture_begin(device, force_module_load=False)
    try:
        dst.zero_()
        resources.clear()
        gc.collect()
    finally:
        graph = wp.capture_end(device)

    return dst, graph


def _graph_mem(device):
    core = wp._src.context.runtime.core
    core.wp_cuda_device_graph_mem_trim(device.ordinal)
    return core.wp_cuda_device_get_graph_mem_current(device.ordinal)


def _drain_graph_destroy_callbacks(test, device, graph_mem):
    """End captures until graph memory drops below ``graph_mem``, checking the current device after each."""
    # graph-destroy callbacks arrive asynchronously on a CUDA thread
    deadline = time.monotonic() + 10.0
    while True:
        _capture_while_releasing(device, [])
        test.assertEqual(wp.get_device("cuda"), device)
        if _graph_mem(device) < graph_mem:
            return
        if time.monotonic() > deadline:
            test.fail("graph-destroy callback was not processed")
        time.sleep(0.01)


def test_pinned_array_del_during_capture(test, device):
    """Releasing a pinned host array during capture must not invalidate the capture."""
    with wp.ScopedDevice(device):
        resources = [wp.zeros(1024, dtype=float, device="cpu", pinned=True)]
        dst, graph = _capture_while_releasing(device, resources)

        wp.capture_launch(graph)
        np.testing.assert_array_equal(dst.numpy(), np.zeros(4, dtype=np.float32))


def test_loaded_graph_del_during_capture(test, device):
    """Releasing a graph loaded by ``wp.capture_load()`` during capture must not invalidate the capture."""
    with wp.ScopedDevice(device), tempfile.TemporaryDirectory() as tmpdir:
        values = wp.zeros(4, dtype=wp.uint8, device=device)
        with wp.ScopedCapture(device, apic=True, force_module_load=False) as apic_capture:
            values.zero_()
        path = os.path.join(tmpdir, "graph")
        wp.capture_save(apic_capture.graph, path, outputs={"values": values})

        resources = [wp.capture_load(path, device=device)]
        # build the CUDA graph and its executable, so they are released too
        wp.capture_launch(resources[0])
        dst, graph = _capture_while_releasing(device, resources)

        wp.capture_launch(graph)
        np.testing.assert_array_equal(dst.numpy(), np.zeros(4, dtype=np.float32))


def test_invalid_free_during_capture_reports_error(test, device):
    """A deferred stream-ordered free of an invalid pointer must report an error when the capture ends."""
    core = wp._src.context.runtime.core
    # with unified addressing, a host address is never a device allocation
    host_buffer = ctypes.create_string_buffer(1024)

    def free_invalid(during_capture):
        stderr = StdErrCapture()
        stderr.begin()
        try:
            with wp.ScopedCapture(device, force_module_load=False) if during_capture else contextlib.nullcontext():
                core.wp_free_device_async(device.context, ctypes.addressof(host_buffer), None)
            # clear the error, so it does not surface in later tests
            core.wp_cuda_context_check(device.context)
        finally:
            output = stderr.end()
        return output

    # not every driver validates the pointer, e.g. on Windows with a Blackwell GPU
    if "invalid argument" not in free_invalid(during_capture=False):
        test.skipTest("The driver does not report invalid stream-ordered frees")
    test.assertIn("invalid argument", free_invalid(during_capture=True))


def test_release_other_device_during_capture_keeps_current_device(test, device):
    """Releasing another device's resources during capture must not change the current device."""
    other = _SecondaryContext(device)
    try:
        with wp.ScopedDevice(device):
            # all deferred releases share one drain, so a single resource covers it
            resources = [wp.zeros(1024, dtype=float, device=other.device)]
            _capture_while_releasing(device, resources)
            test.assertEqual(wp.get_device("cuda"), device)
    finally:
        other.close()


def test_multi_device_graph_release_keeps_current_device(test, device):
    """Destroying a graph that owns another device's allocation must not change the current device."""
    other = _SecondaryContext(device)
    try:
        with wp.ScopedDevice(device):
            main_stream = wp.Stream(device)
            other_stream = wp.Stream(other.device)
            with wp.ScopedCapture(stream=main_stream, force_module_load=False) as capture:
                other_stream.wait_stream(main_stream)
                with wp.ScopedStream(other_stream, sync_enter=False):
                    graph_alloc = wp.zeros(4, dtype=float, device=other.device)
                main_stream.wait_stream(other_stream)
            wp.capture_launch(capture.graph, stream=main_stream)
            wp.synchronize()

            # the graph now owns the allocation and frees it from a graph-destroy callback
            resources = [capture.graph]
            del capture, graph_alloc, other_stream
            graph_mem = _graph_mem(device)
            _capture_while_releasing(device, resources)
            test.assertEqual(wp.get_device("cuda"), device)

            _drain_graph_destroy_callbacks(test, device, graph_mem)
    finally:
        other.close()


_CU_HOST_FN = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
_graph_destroyed = threading.Event()


# module-level, so it outlives any graph it is attached to
@_CU_HOST_FN
def _on_graph_destroyed(_user_data):
    _graph_destroyed.set()


def _load_cuda_driver():
    try:
        driver = ctypes.CDLL("nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1")
    except OSError:
        return None
    driver.cuUserObjectCreate.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        _CU_HOST_FN,
        ctypes.c_uint,
        ctypes.c_uint,
    ]
    driver.cuUserObjectCreate.restype = ctypes.c_int
    driver.cuGraphRetainUserObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint]
    driver.cuGraphRetainUserObject.restype = ctypes.c_int
    return driver


def _notify_graph_destroyed(test, driver, graph):
    """Set ``_graph_destroyed`` when ``graph`` is destroyed."""
    user_object = ctypes.c_void_p()
    no_destructor_sync = 1
    test.assertEqual(
        driver.cuUserObjectCreate(ctypes.byref(user_object), None, _on_graph_destroyed, 1, no_destructor_sync), 0
    )
    move = 1
    test.assertEqual(driver.cuGraphRetainUserObject(graph, user_object, 1, move), 0)


def test_destroyed_context_discards_queued_releases(test, device):
    """Releases still queued when their context is destroyed must be dropped, except those the context does not own."""
    driver = _load_cuda_driver()
    if driver is None:
        test.skipTest("CUDA driver library not found")

    other = _SecondaryContext(device)
    closed = False
    try:
        with wp.ScopedDevice(device):
            graph_dst = wp.zeros(4, dtype=float, device=other.device)
            with wp.ScopedCapture(other.device, force_module_load=False) as other_capture:
                graph_dst.zero_()
            _graph_destroyed.clear()
            _notify_graph_destroyed(test, driver, other_capture.graph.graph)

            # stream-ordered memory belongs to the device's memory pool, which outlives the context
            pool_bytes = 64 * 1024 * 1024
            pool_array = wp.empty(pool_bytes // 4, dtype=float, device=other.device)
            resources = [pool_array, graph_dst, other_capture.graph]
            del pool_array, graph_dst, other_capture
            wp.synchronize_device(other.device)
            pool_mem = wp.get_mempool_used_mem_current(device) if wp.is_mempool_enabled(other.device) else None

            dst = wp.ones(4, dtype=float, device=device)
            stderr = StdErrCapture()
            stderr.begin()
            try:
                # relaxed, so destroying the context does not invalidate the capture
                with wp.ScopedCapture(device, force_module_load=False, capture_mode=wp.CaptureMode.RELAXED):
                    dst.zero_()
                    resources.clear()
                    gc.collect()
                    other.close(synchronize=False)
                    closed = True
            finally:
                output = stderr.end()
            test.assertEqual(output, "")
            if pool_mem is not None:
                wp.synchronize_device(device)
                test.assertLessEqual(wp.get_mempool_used_mem_current(device), pool_mem - pool_bytes)
            # the destroy callback runs asynchronously on a CUDA thread
            test.assertTrue(_graph_destroyed.wait(timeout=10.0))
    finally:
        if not closed:
            other.close()


class TestCaptureRelease(unittest.TestCase):
    pass


devices = get_selected_cuda_test_devices()
# these tests allocate during capture
mempool_devices = get_selected_cuda_test_devices_with_mempool()

add_function_test(
    TestCaptureRelease, "test_pinned_array_del_during_capture", test_pinned_array_del_during_capture, devices=devices
)
add_function_test(
    TestCaptureRelease, "test_loaded_graph_del_during_capture", test_loaded_graph_del_during_capture, devices=devices
)
add_function_test(
    TestCaptureRelease,
    "test_invalid_free_during_capture_reports_error",
    test_invalid_free_during_capture_reports_error,
    devices=mempool_devices,
)
add_function_test(
    TestCaptureRelease,
    "test_release_other_device_during_capture_keeps_current_device",
    test_release_other_device_during_capture_keeps_current_device,
    devices=devices,
)
add_function_test(
    TestCaptureRelease,
    "test_multi_device_graph_release_keeps_current_device",
    test_multi_device_graph_release_keeps_current_device,
    devices=mempool_devices,
)
add_function_test(
    TestCaptureRelease,
    "test_destroyed_context_discards_queued_releases",
    test_destroyed_context_discards_queued_releases,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
