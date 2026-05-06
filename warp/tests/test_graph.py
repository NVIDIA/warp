# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for graph capture and replay on CPU and CUDA devices."""

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import add_function_test, get_test_devices, get_test_devices_with_mempool


@wp.kernel
def scale_kernel(input: wp.array(dtype=float), output: wp.array(dtype=float), s: float):
    i = wp.tid()
    output[i] = input[i] * s


@wp.kernel
def add_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float), output: wp.array(dtype=float)):
    i = wp.tid()
    output[i] = a[i] + b[i]


class TestGraph(unittest.TestCase):
    pass


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


devices = get_test_devices()
devices_with_mempool = get_test_devices_with_mempool()

add_function_test(TestGraph, "test_graph_single_kernel", test_graph_single_kernel, devices=devices)
add_function_test(TestGraph, "test_graph_multiple_kernels", test_graph_multiple_kernels, devices=devices)
add_function_test(TestGraph, "test_graph_replay_multiple", test_graph_replay_multiple, devices=devices)
add_function_test(TestGraph, "test_graph_memcpy", test_graph_memcpy, devices=devices)
add_function_test(TestGraph, "test_graph_memset", test_graph_memset, devices=devices)
add_function_test(TestGraph, "test_graph_alloc", test_graph_alloc, devices=devices_with_mempool)

if __name__ == "__main__":
    unittest.main(verbosity=2)
