# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for graph capture and replay on CPU and CUDA devices."""

import ctypes
import gc
import time
import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import (
    add_function_test,
    get_selected_cuda_test_devices_with_mempool,
    get_test_devices,
    get_test_devices_with_cuda_graph_module_load,
    get_test_devices_with_mempool_and_cuda_graph_module_load,
)


@wp.kernel
def scale_kernel(input: wp.array(dtype=float), output: wp.array(dtype=float), s: float):
    i = wp.tid()
    output[i] = input[i] * s


@wp.kernel
def add_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float), output: wp.array(dtype=float)):
    i = wp.tid()
    output[i] = a[i] + b[i]


class TestGraph(unittest.TestCase):
    def test_cuda_graph_memory_bindings(self):
        core = wp._src.context.runtime.core

        def get_ctypes_binding(name):
            bindings = getattr(core, "ctypes", None)
            if bindings is not None and hasattr(bindings, name):
                return getattr(bindings, name)
            return getattr(core, name)

        get_current = get_ctypes_binding("wp_cuda_device_get_graph_mem_current")
        trim = get_ctypes_binding("wp_cuda_device_graph_mem_trim")

        self.assertEqual(get_current.argtypes, [ctypes.c_int])
        self.assertIs(get_current.restype, ctypes.c_uint64)
        self.assertEqual(trim.argtypes, [ctypes.c_int])
        self.assertIsNone(trim.restype)


def test_graph_single_kernel(test, device):
    n = 1024
    input_arr = wp.array(np.arange(n, dtype=np.float32), device=device)
    output_arr = wp.zeros(n, dtype=float, device=device)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, force_module_load=False) as capture:
        wp.launch(scale_kernel, dim=n, inputs=[input_arr, output_arr, 2.0], device=device)

    expected = np.arange(n, dtype=np.float32) * 2.0

    # Launch and verify
    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    np.testing.assert_allclose(output_arr.numpy(), expected)

    # Reset and replay
    output_arr.zero_()
    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    np.testing.assert_allclose(output_arr.numpy(), expected)


def test_graph_multiple_kernels(test, device):
    n = 512
    a = wp.array(np.ones(n, dtype=np.float32), device=device)
    b = wp.array(np.ones(n, dtype=np.float32) * 3.0, device=device)
    c = wp.zeros(n, dtype=float, device=device)
    d = wp.zeros(n, dtype=float, device=device)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, force_module_load=False) as capture:
        wp.launch(add_kernel, dim=n, inputs=[a, b, c], device=device)
        wp.launch(scale_kernel, dim=n, inputs=[c, d, 10.0], device=device)

    # c = a + b = 4.0, d = c * 10 = 40.0
    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    np.testing.assert_allclose(d.numpy(), np.full(n, 40.0))

    # Reset and replay
    c.zero_()
    d.zero_()
    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    np.testing.assert_allclose(d.numpy(), np.full(n, 40.0))


def test_graph_replay_multiple(test, device):
    n = 256
    input_arr = wp.array(np.ones(n, dtype=np.float32), device=device)
    output_arr = wp.zeros(n, dtype=float, device=device)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, force_module_load=False) as capture:
        wp.launch(add_kernel, dim=n, inputs=[input_arr, output_arr, output_arr], device=device)

    # Each replay adds input_arr to output_arr: output_arr += 1.0
    for _i in range(100):
        wp.capture_launch(capture.graph)

    wp.synchronize_device(device)
    np.testing.assert_allclose(output_arr.numpy(), np.full(n, 100.0))


def test_graph_memcpy(test, device):
    n = 256
    src = wp.array(np.arange(n, dtype=np.float32), device=device)
    dst = wp.zeros(n, dtype=float, device=device)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, force_module_load=False) as capture:
        wp.copy(dst, src)

    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    np.testing.assert_allclose(dst.numpy(), src.numpy())

    # Replay copies from original src (pointers are baked into the graph)
    dst.zero_()
    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    np.testing.assert_allclose(dst.numpy(), np.arange(n, dtype=np.float32))


def test_graph_memset(test, device):
    n = 256
    arr = wp.array(np.arange(n, dtype=np.float32), device=device)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, force_module_load=False) as capture:
        arr.zero_()

    # Launch to execute the captured zero_()
    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    np.testing.assert_allclose(arr.numpy(), np.zeros(n))

    # Fill with non-zero, then replay to zero again
    arr.fill_(1.0)
    np.testing.assert_allclose(arr.numpy(), np.ones(n))
    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    np.testing.assert_allclose(arr.numpy(), np.zeros(n))


def test_graph_alloc(test, device):
    """Array allocated inside capture scope, used by subsequent kernel."""
    n = 128
    input_arr = wp.array(np.arange(n, dtype=np.float32) + 1.0, device=device)
    output_arr = wp.zeros(n, dtype=float, device=device)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, force_module_load=False) as capture:
        tmp = wp.zeros(n, dtype=float, device=device)
        wp.launch(scale_kernel, dim=n, inputs=[input_arr, tmp, 2.0], device=device)
        wp.launch(add_kernel, dim=n, inputs=[tmp, input_arr, output_arr], device=device)

    # tmp = input*2, output = tmp + input = input*3
    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    expected = (np.arange(n, dtype=np.float32) + 1.0) * 3.0
    np.testing.assert_allclose(output_arr.numpy(), expected)

    # Replay — should produce same result
    output_arr.zero_()
    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    np.testing.assert_allclose(output_arr.numpy(), expected)


############################################################
#
# CUDA-only tests
#
############################################################


def test_cuda_graph_alloc_retained_release(test, device):
    """Release retained CUDA graph allocations after graph and user refs drop.

    Allocations made during capture and stored on a Python object that
    outlives the capture must be reclaimed once both the graph and the user
    reference are gone.
    """

    # Note: the test may fail if CUDA/Python/multiprocess stars don't align perfectly.
    # Disabled to avoid CI noise while we find a more robust measurement strategy.
    test.skipTest("Skipped due to flakiness in memory usage measurement")

    device = wp.get_device(device)

    n = 64 * 1024 * 1024  # allocation size should be large enough to spot a clear leak
    size_in_bytes = n * 4
    steps = 10
    substeps = 5

    wp_cuda_device_get_graph_mem_current = wp._src.context.runtime.core.wp_cuda_device_get_graph_mem_current
    wp_cuda_device_graph_mem_trim = wp._src.context.runtime.core.wp_cuda_device_graph_mem_trim

    class Holder:
        """Allocate inside capture, retain on self."""

        def __init__(self):
            base = wp.zeros(n, dtype=float, device=device)
            wp.load_module(device=device)
            with wp.ScopedCapture(device=device, force_module_load=False) as capture:
                for _ in range(substeps):
                    # graph allocation retained in self
                    self.scratch = wp.clone(base)
                    wp.launch(scale_kernel, dim=n, inputs=[self.scratch, self.scratch, 1.0], device=device)
            self.graph = capture.graph
            self.base = base

        def step(self):
            wp.capture_launch(self.graph)

    def cycle():
        h = Holder()
        for _ in range(steps):
            h.step()
        # h goes out of scope -> graph + scratch reference both dropped

    def settle():
        # Allow gc and deferred destructors to settle.
        # on_graph_destroy() runs on an internal CUDA thread and might lag the Python thread.
        # Pressure from concurrent processes could cause delays with callbacks or mempool management.
        # Deferred destructors are processed in synchronize_device().
        # Call wp_cuda_device_graph_mem_trim() to release any unused graph memory.
        gc.collect()
        wp.synchronize_device(device)  # finish GPU work, including graphs
        time.sleep(0.1)  # wait for async callbacks to arrive (e.g., on_graph_destroy)
        wp.synchronize_device(device)  # process deferred deallocations
        wp_cuda_device_graph_mem_trim(device.ordinal)

    # Warm up: first cycle establishes the steady-state graph mempool footprint.
    settle()
    cycle()
    settle()
    baseline = wp_cuda_device_get_graph_mem_current(device.ordinal)

    # Run several more cycles.
    n_cycles = 10
    for _ in range(n_cycles):
        cycle()
        settle()

    final = wp_cuda_device_get_graph_mem_current(device.ordinal)

    # A real leak would scale with n_cycles.
    test.assertLess(
        final - baseline,
        n_cycles * size_in_bytes,
        f"graph memory leak: baseline={baseline}, final={final} after {n_cycles} cycles",
    )


devices = get_test_devices()
devices_with_cuda_graph_module_load = get_test_devices_with_cuda_graph_module_load()
devices_with_mempool_and_cuda_graph_module_load = get_test_devices_with_mempool_and_cuda_graph_module_load()

add_function_test(
    TestGraph,
    "test_graph_single_kernel",
    test_graph_single_kernel,
    devices=devices_with_cuda_graph_module_load,
)
add_function_test(
    TestGraph,
    "test_graph_multiple_kernels",
    test_graph_multiple_kernels,
    devices=devices_with_cuda_graph_module_load,
)
add_function_test(
    TestGraph,
    "test_graph_replay_multiple",
    test_graph_replay_multiple,
    devices=devices_with_cuda_graph_module_load,
)
add_function_test(TestGraph, "test_graph_memcpy", test_graph_memcpy, devices=devices)
add_function_test(TestGraph, "test_graph_memset", test_graph_memset, devices=devices)
add_function_test(
    TestGraph,
    "test_graph_alloc",
    test_graph_alloc,
    devices=devices_with_mempool_and_cuda_graph_module_load,
)

# CUDA-only tests
cuda_devices = get_selected_cuda_test_devices_with_mempool()

add_function_test(
    TestGraph,
    "test_cuda_graph_alloc_retained_release",
    test_cuda_graph_alloc_retained_release,
    devices=cuda_devices,
)

if __name__ == "__main__":
    unittest.main(verbosity=2)
