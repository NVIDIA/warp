# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import gc
import unittest

import numpy as np

import warp as wp
import warp.tests.deterministic.test_deterministic_counter as counter_module
import warp.tests.deterministic.test_deterministic_scatter as scatter_module
from warp.tests.deterministic.common import DeterministicTestBase, cuda_devices
from warp.tests.deterministic.test_deterministic_counter import counter_kernel, indexed_counter_kernel
from warp.tests.deterministic.test_deterministic_scatter import (
    _det_closure_transform_a,
    _make_deterministic_closure_kernel,
    func_scatter_add_kernel,
    scatter_add_kernel,
    sliced_2d_atomic_add_kernel,
    vec3_atomic_minmax_kernel,
)
from warp.tests.unittest_utils import add_function_test


def test_record_cmd_deterministic_launch(test, device):
    """Verify ``record_cmd=True`` works for deterministic CUDA launches."""
    n = 128
    out_size = 8
    rng = np.random.default_rng(19)

    data_np = rng.random(n, dtype=np.float32)
    indices_np = rng.integers(0, out_size, size=n, dtype=np.int32)

    data = wp.array(data_np, dtype=wp.float32, device=device)
    indices = wp.array(indices_np, dtype=wp.int32, device=device)
    output = wp.zeros(out_size, dtype=wp.float32, device=device)

    cmd = wp.launch(
        scatter_add_kernel,
        dim=n,
        inputs=[data, indices],
        outputs=[output],
        device=device,
        record_cmd=True,
    )

    np.testing.assert_array_equal(output.numpy(), np.zeros(out_size, dtype=np.float32))

    cmd.launch()
    expected = output.numpy().copy()

    output_2 = wp.zeros(out_size, dtype=wp.float32, device=device)
    cmd.set_param_by_name("output", output_2)
    cmd.launch()

    np.testing.assert_array_equal(output_2.numpy(), expected)

    output_3 = wp.zeros(out_size, dtype=wp.float32, device=device)
    with test.assertRaisesRegex(RuntimeError, "raw ctypes"):
        cmd.set_param_by_name_from_ctype("output", output_3.__ctype__())

    cmd.set_param_by_name("output", output_3)
    cmd.launch()

    np.testing.assert_array_equal(output_3.numpy(), expected)


def test_graph_capture_deterministic_launch(test, device):
    """Verify deterministic scatter launches can be captured and replayed."""
    n = 256
    rng = np.random.default_rng(29)
    data_np = rng.random(n, dtype=np.float32)
    indices_np = rng.integers(0, 8, size=n, dtype=np.int32)

    data = wp.array(data_np, dtype=wp.float32, device=device)
    indices = wp.array(indices_np, dtype=wp.int32, device=device)
    output = wp.zeros(8, dtype=wp.float32, device=device)

    wp.launch(scatter_add_kernel, dim=n, inputs=[data, indices], outputs=[output], device=device)
    output.zero_()

    with wp.ScopedCapture(device, force_module_load=False) as capture:
        wp.launch(scatter_add_kernel, dim=n, inputs=[data, indices], outputs=[output], device=device)

    test.assertGreater(len(capture.graph._deterministic_buffer_refs), 0)
    test.assertTrue(any(getattr(ref, "dtype", None) is wp.uint8 for ref in capture.graph._deterministic_buffer_refs))
    gc.collect()

    wp.capture_launch(capture.graph)
    first = output.numpy().copy()

    output.zero_()
    wp.capture_launch(capture.graph)
    second = output.numpy().copy()

    np.testing.assert_array_equal(first, second)


def test_graph_capture_sliced_array(test, device):
    """Verify deterministic sliced-array atomics can be captured and replayed."""
    # The graph suite enables run-to-run deterministic mode for the module that
    # defines the sliced-array kernel.
    test.assertEqual(wp.get_module_options(module=scatter_module)["deterministic"], "run_to_run")

    n = 256
    rows, cols = 8, 8
    rng = np.random.default_rng(201)

    data_np = rng.random(n, dtype=np.float32)
    row_np = rng.integers(0, rows, size=n, dtype=np.int32)
    col_np = rng.integers(0, cols, size=n, dtype=np.int32)

    data = wp.array(data_np, dtype=wp.float32, device=device)
    row_idx = wp.array(row_np, dtype=wp.int32, device=device)
    col_idx = wp.array(col_np, dtype=wp.int32, device=device)
    output = wp.zeros(shape=(rows, cols), dtype=wp.float32, device=device)

    wp.launch(sliced_2d_atomic_add_kernel, dim=n, inputs=[data, row_idx, col_idx], outputs=[output], device=device)
    output.zero_()

    with wp.ScopedCapture(device, force_module_load=False) as capture:
        wp.launch(sliced_2d_atomic_add_kernel, dim=n, inputs=[data, row_idx, col_idx], outputs=[output], device=device)

    wp.capture_launch(capture.graph)
    first = output.numpy().copy()

    output.zero_()
    wp.capture_launch(capture.graph)
    second = output.numpy().copy()

    np.testing.assert_array_equal(first, second)


def test_graph_capture_deterministic_closure_kernel(test, device):
    """Verify deterministic closure kernels can be captured and replayed."""
    kernel = _make_deterministic_closure_kernel(_det_closure_transform_a)

    n = 256
    rng = np.random.default_rng(31)
    data_np = rng.random(n, dtype=np.float32)
    data = wp.array(data_np, dtype=wp.float32, device=device)
    output = wp.zeros(8, dtype=wp.float32, device=device)

    wp.launch(kernel, dim=n, inputs=[data], outputs=[output], device=device)
    output.zero_()

    with wp.ScopedCapture(device, force_module_load=False) as capture:
        wp.launch(kernel, dim=n, inputs=[data], outputs=[output], device=device)

    wp.capture_launch(capture.graph)
    first = output.numpy().copy()

    output.zero_()
    wp.capture_launch(capture.graph)
    second = output.numpy().copy()

    np.testing.assert_array_equal(first, second)


def test_graph_capture_deterministic_func_kernel(test, device):
    """Verify deterministic ``@wp.func`` atomics remain capture-safe."""
    n = 256
    rng = np.random.default_rng(76)
    data_np = rng.random(n, dtype=np.float32)
    indices_np = rng.integers(0, 8, size=n, dtype=np.int32)

    data = wp.array(data_np, dtype=wp.float32, device=device)
    indices = wp.array(indices_np, dtype=wp.int32, device=device)
    output = wp.zeros(8, dtype=wp.float32, device=device)

    wp.launch(func_scatter_add_kernel, dim=n, inputs=[data, indices], outputs=[output], device=device)
    output.zero_()

    with wp.ScopedCapture(device, force_module_load=False) as capture:
        wp.launch(func_scatter_add_kernel, dim=n, inputs=[data, indices], outputs=[output], device=device)

    wp.capture_launch(capture.graph)
    first = output.numpy().copy()

    output.zero_()
    wp.capture_launch(capture.graph)
    second = output.numpy().copy()

    np.testing.assert_array_equal(first, second)


def test_graph_capture_vec3_atomic_minmax(test, device):
    """Verify composite deterministic reductions remain capture-safe."""
    n = 512
    rng = np.random.default_rng(70)
    points_np = rng.standard_normal((n, 3), dtype=np.float32)
    points = wp.array(points_np, dtype=wp.vec3, device=device)

    out_min = wp.empty(1, dtype=wp.vec3, device=device)
    out_max = wp.empty(1, dtype=wp.vec3, device=device)
    min_init = wp.vec3(np.inf, np.inf, np.inf)
    max_init = wp.vec3(-np.inf, -np.inf, -np.inf)

    out_min.fill_(min_init)
    out_max.fill_(max_init)
    wp.launch(vec3_atomic_minmax_kernel, dim=n, inputs=[points], outputs=[out_min, out_max], device=device)
    out_min.fill_(min_init)
    out_max.fill_(max_init)

    with wp.ScopedCapture(device, force_module_load=False) as capture:
        wp.launch(vec3_atomic_minmax_kernel, dim=n, inputs=[points], outputs=[out_min, out_max], device=device)

    wp.capture_launch(capture.graph)
    first_min = out_min.numpy().copy()
    first_max = out_max.numpy().copy()

    out_min.fill_(min_init)
    out_max.fill_(max_init)
    wp.capture_launch(capture.graph)
    second_min = out_min.numpy().copy()
    second_max = out_max.numpy().copy()

    np.testing.assert_array_equal(first_min, second_min)
    np.testing.assert_array_equal(first_max, second_max)


def test_graph_capture_consumed_return_counter(test, device):
    """Verify consumed-return atomic counters can be captured and replayed."""
    n = 64
    rng = np.random.default_rng(202)
    data_np = rng.random(n, dtype=np.float32)

    data = wp.array(data_np, dtype=wp.float32, device=device)
    counter = wp.zeros(1, dtype=wp.int32, device=device)
    output = wp.zeros(n, dtype=wp.float32, device=device)

    wp.launch(counter_kernel, dim=n, inputs=[data, counter], outputs=[output], device=device)
    counter.zero_()
    output.zero_()

    with wp.ScopedCapture(device, force_module_load=False) as capture:
        wp.launch(counter_kernel, dim=n, inputs=[data, counter], outputs=[output], device=device)

    wp.capture_launch(capture.graph)
    first_count = counter.numpy().copy()
    first = output.numpy().copy()

    counter.zero_()
    output.zero_()
    wp.capture_launch(capture.graph)
    second_count = counter.numpy().copy()
    second = output.numpy().copy()

    np.testing.assert_array_equal(first_count, second_count)
    np.testing.assert_array_equal(first, second)


def test_graph_capture_indexed_counter(test, device):
    """Verify data-dependent consumed-return counters can be captured and replayed."""
    n = 64
    bin_count = 4
    bins_np = (np.arange(n, dtype=np.int32) * 7) % bin_count
    values_np = np.arange(n, dtype=np.int32)

    values = wp.array(values_np, dtype=wp.int32, device=device)
    bins = wp.array(bins_np, dtype=wp.int32, device=device)
    counter = wp.zeros(bin_count, dtype=wp.int32, device=device)
    output = wp.full((bin_count, n), value=-1, dtype=wp.int32, device=device)

    wp.launch(indexed_counter_kernel, dim=n, inputs=[values, bins, counter], outputs=[output], device=device)
    counter.zero_()
    output.fill_(-1)

    with wp.ScopedCapture(device, force_module_load=False) as capture:
        wp.launch(indexed_counter_kernel, dim=n, inputs=[values, bins, counter], outputs=[output], device=device)

    wp.capture_launch(capture.graph)
    first_count = counter.numpy().copy()
    first = output.numpy().copy()

    counter.zero_()
    output.fill_(-1)
    wp.capture_launch(capture.graph)
    second_count = counter.numpy().copy()
    second = output.numpy().copy()

    np.testing.assert_array_equal(first_count, second_count)
    np.testing.assert_array_equal(first, second)


def test_counter_large_launch_rejected(test, device):
    """Verify counter prefix buffers fail clearly before oversized launches."""
    data = wp.ones(1, dtype=wp.float32, device=device)
    counter = wp.zeros(1, dtype=wp.int32, device=device)
    output = wp.zeros(1, dtype=wp.float32, device=device)

    with test.assertRaisesRegex(RuntimeError, "up to 2\\^31 - 1 threads"):
        wp.launch(counter_kernel, dim=(46341, 46341), inputs=[data, counter], outputs=[output], device=device)


def test_apic_capture_rejects_deterministic_cuda_kernel(test, device):
    """Verify APIC serialization fails explicitly for deterministic CUDA kernels."""
    n = 16
    data_np = np.ones(n, dtype=np.float32)
    indices_np = np.arange(n, dtype=np.int32) % 4

    data = wp.array(data_np, dtype=wp.float32, device=device)
    indices = wp.array(indices_np, dtype=wp.int32, device=device)
    output = wp.zeros(4, dtype=wp.float32, device=device)

    wp.launch(scatter_add_kernel, dim=n, inputs=[data, indices], outputs=[output], device=device)
    output.zero_()

    with test.assertRaisesRegex(RuntimeError, "APIC serialization"):
        with wp.ScopedCapture(device=device, apic=True, force_module_load=False):
            wp.launch(scatter_add_kernel, dim=n, inputs=[data, indices], outputs=[output], device=device)


class TestDeterministicGraph(DeterministicTestBase):
    """Test deterministic launches in CUDA graph capture paths."""

    deterministic_modules = (scatter_module, counter_module)


def _add(name, devices=cuda_devices):
    add_function_test(TestDeterministicGraph, name, globals()[name], devices=devices)


for _name in (
    "test_record_cmd_deterministic_launch",
    "test_graph_capture_deterministic_launch",
    "test_graph_capture_sliced_array",
    "test_graph_capture_deterministic_closure_kernel",
    "test_graph_capture_deterministic_func_kernel",
    "test_graph_capture_vec3_atomic_minmax",
    "test_graph_capture_consumed_return_counter",
    "test_graph_capture_indexed_counter",
    "test_counter_large_launch_rejected",
    "test_apic_capture_rejects_deterministic_cuda_kernel",
):
    _add(_name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
