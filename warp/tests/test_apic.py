# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for APIC (API Capture) graph serialization and loading."""

import os
import tempfile
import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import (
    add_function_test,
    get_cuda_test_devices,
    get_test_devices,
    get_test_devices_with_mempool,
)


@wp.kernel
def scale_kernel(input: wp.array(dtype=float), output: wp.array(dtype=float), s: float):
    i = wp.tid()
    output[i] = input[i] * s


@wp.kernel
def add_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float), output: wp.array(dtype=float)):
    i = wp.tid()
    output[i] = a[i] + b[i]


class TestApic(unittest.TestCase):
    pass


def test_save_apic_false_error(test, device):
    """capture_save() should raise when apic=False."""
    n = 64
    a = wp.array(np.ones(n, dtype=np.float32), device=device)
    b = wp.zeros(n, dtype=float, device=device)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=False, force_module_load=False) as capture:
        wp.launch(scale_kernel, dim=n, inputs=[a, b, 2.0], device=device)

    with tempfile.TemporaryDirectory() as tmpdir:
        with test.assertRaises(RuntimeError):
            wp.capture_save(capture.graph, os.path.join(tmpdir, "should_not_exist"))


def test_save_single_kernel(test, device):
    """Capture, save to .wrp, verify file exists."""
    n = 256
    a = wp.array(np.arange(n, dtype=np.float32), device=device)
    b = wp.zeros(n, dtype=float, device=device)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.launch(scale_kernel, dim=n, inputs=[a, b, 3.0], device=device)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_single")
        wp.capture_save(capture.graph, path, inputs={"a": a}, outputs={"b": b})

        wrp_path = path + ".wrp"
        test.assertTrue(os.path.exists(wrp_path), f"WRP file not found: {wrp_path}")
        test.assertGreater(os.path.getsize(wrp_path), 0, "WRP file is empty")


def test_save_load_round_trip(test, device):
    """Capture, save, load, launch, verify output matches."""
    n = 256
    a = wp.array(np.arange(n, dtype=np.float32), device=device)
    b = wp.zeros(n, dtype=float, device=device)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.launch(scale_kernel, dim=n, inputs=[a, b, 2.0], device=device)

    # Launch the original graph first to verify it works
    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    expected = np.arange(n, dtype=np.float32) * 2.0
    np.testing.assert_allclose(b.numpy(), expected)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_roundtrip")
        wp.capture_save(capture.graph, path, inputs={"a": a}, outputs={"b": b})

        # Load and launch
        loaded = wp.capture_load(path, device=device)
        test.assertTrue(loaded.is_loaded)
        test.assertIn("a", loaded.params)
        test.assertIn("b", loaded.params)

        wp.capture_launch(loaded)
        wp.synchronize_device(device)

        # Read back results via get_param
        result = wp.zeros(n, dtype=float, device=device)
        loaded.get_param("b", result)
        np.testing.assert_allclose(result.numpy(), expected)


def test_bindings_param_update(test, device):
    """set_param changes input, verify output changes."""
    n = 128
    a = wp.array(np.ones(n, dtype=np.float32), device=device)
    b = wp.zeros(n, dtype=float, device=device)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.launch(scale_kernel, dim=n, inputs=[a, b, 5.0], device=device)

    # Launch original graph to populate memory before saving
    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_param_update")
        wp.capture_save(capture.graph, path, inputs={"a": a}, outputs={"b": b})

        loaded = wp.capture_load(path, device=device)

        # First launch with original data (a=1.0) -> b should be 5.0
        wp.capture_launch(loaded)
        wp.synchronize_device(device)
        result = wp.zeros(n, dtype=float, device=device)
        loaded.get_param("b", result)
        np.testing.assert_allclose(result.numpy(), np.full(n, 5.0))  # 1.0 * 5.0

        # Update input with new data
        new_a = wp.array(np.full(n, 10.0, dtype=np.float32), device=device)
        loaded.set_param("a", new_a)

        # Launch with updated input
        wp.capture_launch(loaded)
        wp.synchronize_device(device)
        loaded.get_param("b", result)
        np.testing.assert_allclose(result.numpy(), np.full(n, 50.0))  # 10.0 * 5.0


def test_save_load_multiple_kernels(test, device):
    """Capture multiple different kernels, round-trip via .wrp."""
    n = 128
    a = wp.array(np.ones(n, dtype=np.float32) * 2.0, device=device)
    b = wp.array(np.ones(n, dtype=np.float32) * 3.0, device=device)
    c = wp.zeros(n, dtype=float, device=device)
    d = wp.zeros(n, dtype=float, device=device)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.launch(add_kernel, dim=n, inputs=[a, b, c], device=device)
        wp.launch(scale_kernel, dim=n, inputs=[c, d, 10.0], device=device)

    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    # c = 2 + 3 = 5, d = 5 * 10 = 50
    np.testing.assert_allclose(d.numpy(), np.full(n, 50.0))

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_multi")
        wp.capture_save(capture.graph, path, outputs={"d": d})

        loaded = wp.capture_load(path, device=device)
        wp.capture_launch(loaded)
        wp.synchronize_device(device)

        result = wp.zeros(n, dtype=float, device=device)
        loaded.get_param("d", result)
        np.testing.assert_allclose(result.numpy(), np.full(n, 50.0))


def test_save_load_memcpy(test, device):
    """Capture wp.copy, round-trip via .wrp."""
    n = 64
    src = wp.array(np.arange(n, dtype=np.float32), device=device)
    dst = wp.zeros(n, dtype=float, device=device)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.copy(dst, src)

    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    np.testing.assert_allclose(dst.numpy(), src.numpy())

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_memcpy")
        wp.capture_save(capture.graph, path, outputs={"dst": dst})

        loaded = wp.capture_load(path, device=device)
        wp.capture_launch(loaded)
        wp.synchronize_device(device)

        result = wp.zeros(n, dtype=float, device=device)
        loaded.get_param("dst", result)
        np.testing.assert_allclose(result.numpy(), np.arange(n, dtype=np.float32))


def test_save_load_memset(test, device):
    """Capture array.zero_(), round-trip via .wrp."""
    n = 64
    arr = wp.array(np.arange(n, dtype=np.float32), device=device)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        arr.zero_()

    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    np.testing.assert_allclose(arr.numpy(), np.zeros(n))

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_memset")
        wp.capture_save(capture.graph, path, outputs={"arr": arr})

        loaded = wp.capture_load(path, device=device)
        wp.capture_launch(loaded)
        wp.synchronize_device(device)

        result = wp.array(np.ones(n, dtype=np.float32), device=device)
        loaded.get_param("arr", result)
        np.testing.assert_allclose(result.numpy(), np.zeros(n))


def test_array_slicing(test, device):
    """Array slices sharing a base allocation map to the same region."""
    from warp._src.apic.capture import APICapture  # noqa: PLC0415

    n = 1024
    base_arr = wp.array(np.arange(n, dtype=np.float32), device=device)
    slice1 = base_arr[0:512]
    slice2 = base_arr[512:1024]

    apic = APICapture(device, wp._src.context.runtime, apic_savable=False)

    region_id_base, offset_base = apic.track_array(base_arr)
    region_id_1, offset_1 = apic.track_array(slice1)
    region_id_2, offset_2 = apic.track_array(slice2)

    # All should map to the same region
    test.assertEqual(region_id_base, region_id_1)
    test.assertEqual(region_id_base, region_id_2)

    # Offsets
    test.assertEqual(offset_base, 0)
    test.assertEqual(offset_1, 0)
    test.assertEqual(offset_2, 512 * 4)  # 512 * sizeof(float)

    apic.destroy()


def test_complex_pipeline(test, device):
    """Multi-stage pipeline: kernels + memcpy round-trip."""
    n = 128
    a = wp.array(np.ones(n, dtype=np.float32) * 2.0, device=device)
    b = wp.array(np.ones(n, dtype=np.float32) * 3.0, device=device)
    c = wp.zeros(n, dtype=float, device=device)
    d = wp.zeros(n, dtype=float, device=device)
    e = wp.zeros(n, dtype=float, device=device)
    f = wp.zeros(n, dtype=float, device=device)
    g = wp.zeros(n, dtype=float, device=device)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.launch(add_kernel, dim=n, inputs=[a, b, c], device=device)  # c = 5
        wp.copy(d, c)  # d = 5
        wp.launch(scale_kernel, dim=n, inputs=[d, e, 2.0], device=device)  # e = 10
        wp.copy(f, e)  # f = 10
        wp.launch(add_kernel, dim=n, inputs=[f, c, g], device=device)  # g = 15

    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    np.testing.assert_allclose(g.numpy(), np.full(n, 15.0))

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "complex_pipeline")
        wp.capture_save(capture.graph, path, inputs={"a": a, "b": b}, outputs={"g": g})

        loaded = wp.capture_load(path, device=device)

        new_a = wp.array(np.full(n, 10.0, dtype=np.float32), device=device)
        new_b = wp.array(np.full(n, 5.0, dtype=np.float32), device=device)
        loaded.set_param("a", new_a)
        loaded.set_param("b", new_b)

        wp.capture_launch(loaded)
        wp.synchronize_device(device)

        result = wp.zeros(n, dtype=float, device=device)
        loaded.get_param("g", result)
        # c=15, e=30, g=30+15=45
        np.testing.assert_allclose(result.numpy(), np.full(n, 45.0))


def test_internal_allocation(test, device):
    """Array allocated inside capture scope, used by subsequent ops."""
    n = 128
    input_data = wp.array(np.arange(n, dtype=np.float32) + 1.0, device=device)
    output_data = wp.zeros(n, dtype=float, device=device)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        tmp = wp.zeros(n, dtype=float, device=device)
        wp.launch(scale_kernel, dim=n, inputs=[input_data, tmp, 2.0], device=device)
        wp.launch(add_kernel, dim=n, inputs=[tmp, input_data, output_data], device=device)

    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    # tmp = input*2, output = tmp + input = input*3
    expected = (np.arange(n, dtype=np.float32) + 1.0) * 3.0
    np.testing.assert_allclose(output_data.numpy(), expected)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "alloc_graph")
        wp.capture_save(capture.graph, path, inputs={"input": input_data}, outputs={"output": output_data})

        loaded = wp.capture_load(path, device=device)
        new_input = wp.array(np.full(n, 10.0, dtype=np.float32), device=device)
        loaded.set_param("input", new_input)

        wp.capture_launch(loaded)
        wp.synchronize_device(device)

        result = wp.zeros(n, dtype=float, device=device)
        loaded.get_param("output", result)
        # tmp = 10*2=20, output = 20+10=30
        np.testing.assert_allclose(result.numpy(), np.full(n, 30.0))


def test_multiple_internal_allocations(test, device):
    """Multiple temporary arrays allocated inside capture scope."""
    n = 64
    input_data = wp.array(np.full(n, 2.0, dtype=np.float32), device=device)
    output_data = wp.zeros(n, dtype=float, device=device)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        t1 = wp.zeros(n, dtype=float, device=device)
        t2 = wp.zeros(n, dtype=float, device=device)
        t3 = wp.zeros(n, dtype=float, device=device)

        wp.launch(scale_kernel, dim=n, inputs=[input_data, t1, 2.0], device=device)  # t1 = 4
        wp.launch(scale_kernel, dim=n, inputs=[t1, t2, 3.0], device=device)  # t2 = 12
        wp.launch(add_kernel, dim=n, inputs=[t1, t2, t3], device=device)  # t3 = 16
        wp.launch(add_kernel, dim=n, inputs=[t3, input_data, output_data], device=device)  # out = 18

    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    np.testing.assert_allclose(output_data.numpy(), np.full(n, 18.0))

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "multi_alloc")
        wp.capture_save(capture.graph, path, inputs={"input": input_data}, outputs={"output": output_data})

        loaded = wp.capture_load(path, device=device)
        new_input = wp.array(np.full(n, 5.0, dtype=np.float32), device=device)
        loaded.set_param("input", new_input)

        wp.capture_launch(loaded)
        wp.synchronize_device(device)

        result = wp.zeros(n, dtype=float, device=device)
        loaded.get_param("output", result)
        # t1=10, t2=30, t3=40, output=45
        np.testing.assert_allclose(result.numpy(), np.full(n, 45.0))


def test_graph_execution_unchanged(test, device):
    """Normal graph execution still works with apic=True."""
    n = 1024
    input_data = wp.array(np.arange(n, dtype=np.float32), device=device)
    output_data = wp.zeros(n, dtype=float, device=device)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.launch(scale_kernel, dim=n, inputs=[input_data, output_data, 0.5], device=device)

    for _ in range(3):
        wp.capture_launch(capture.graph)

    wp.synchronize_device(device)
    expected = np.arange(n, dtype=np.float32) * 0.5
    np.testing.assert_allclose(output_data.numpy(), expected)


def test_save_load_with_param_update(test, device):
    """Full round-trip with set_param on multiple kernels."""
    n = 128
    a = wp.array(np.full(n, 2.0, dtype=np.float32), device=device)
    b = wp.array(np.full(n, 3.0, dtype=np.float32), device=device)
    c = wp.zeros(n, dtype=float, device=device)
    d = wp.zeros(n, dtype=float, device=device)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.launch(add_kernel, dim=n, inputs=[a, b, c], device=device)
        wp.launch(scale_kernel, dim=n, inputs=[c, d, 2.0], device=device)

    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "param_update_multi")
        wp.capture_save(capture.graph, path, inputs={"a": a, "b": b}, outputs={"c": c, "d": d})

        loaded = wp.capture_load(path, device=device)
        new_a = wp.array(np.full(n, 5.0, dtype=np.float32), device=device)
        new_b = wp.array(np.full(n, 7.0, dtype=np.float32), device=device)
        loaded.set_param("a", new_a)
        loaded.set_param("b", new_b)

        wp.capture_launch(loaded)
        wp.synchronize_device(device)

        result_c = wp.zeros(n, dtype=float, device=device)
        result_d = wp.zeros(n, dtype=float, device=device)
        loaded.get_param("c", result_c)
        loaded.get_param("d", result_d)

        # c = 5+7 = 12, d = 12*2 = 24
        np.testing.assert_allclose(result_c.numpy(), np.full(n, 12.0))
        np.testing.assert_allclose(result_d.numpy(), np.full(n, 24.0))


def test_save_load_memcpy_and_kernel(test, device):
    """Round-trip with memcpy + kernel pipeline."""
    n = 256
    src = wp.array(np.arange(n, dtype=np.float32) + 1.0, device=device)
    tmp = wp.zeros(n, dtype=float, device=device)
    dst = wp.zeros(n, dtype=float, device=device)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.copy(tmp, src)
        wp.launch(scale_kernel, dim=n, inputs=[tmp, dst, 2.0], device=device)

    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    expected = (np.arange(n, dtype=np.float32) + 1.0) * 2.0
    np.testing.assert_allclose(dst.numpy(), expected)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "memcpy_kernel")
        wp.capture_save(capture.graph, path, inputs={"src": src}, outputs={"dst": dst})

        loaded = wp.capture_load(path, device=device)
        new_src = wp.array(np.full(n, 10.0, dtype=np.float32), device=device)
        loaded.set_param("src", new_src)

        wp.capture_launch(loaded)
        wp.synchronize_device(device)

        result = wp.zeros(n, dtype=float, device=device)
        loaded.get_param("dst", result)
        np.testing.assert_allclose(result.numpy(), np.full(n, 20.0))


def test_save_load_fill(test, device):
    """Capture array.fill_(), round-trip via .wrp."""
    n = 64
    arr = wp.zeros(n, dtype=float, device=device)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        arr.fill_(42.0)

    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    np.testing.assert_allclose(arr.numpy(), np.full(n, 42.0))

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_fill")
        wp.capture_save(capture.graph, path, outputs={"arr": arr})

        loaded = wp.capture_load(path, device=device)
        wp.capture_launch(loaded)
        wp.synchronize_device(device)

        result = wp.zeros(n, dtype=float, device=device)
        loaded.get_param("arr", result)
        np.testing.assert_allclose(result.numpy(), np.full(n, 42.0))


def test_save_load_alloc_only(test, device):
    """Allocation + memcpy inside capture, no kernel launches — tests apic_record_alloc round-trip."""
    n = 64
    src = wp.array(np.arange(n, dtype=np.float32) + 1.0, device=device)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        # Allocate inside capture and copy data into it
        dst = wp.zeros(n, dtype=float, device=device)
        wp.copy(dst, src)

    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_alloc_only")
        wp.capture_save(capture.graph, path, inputs={"src": src}, outputs={"dst": dst})

        loaded = wp.capture_load(path, device=device)
        wp.capture_launch(loaded)
        wp.synchronize_device(device)

        result = wp.zeros(n, dtype=float, device=device)
        loaded.get_param("dst", result)
        np.testing.assert_allclose(result.numpy(), np.arange(n, dtype=np.float32) + 1.0)


def test_get_param_ptr(test, device):
    """get_param_ptr returns a non-zero int for a known name, None for an unknown
    name, and raises RuntimeError on a non-loaded graph."""
    n = 64
    a = wp.array(np.ones(n, dtype=np.float32), device=device)
    b = wp.zeros(n, dtype=float, device=device)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.launch(scale_kernel, dim=n, inputs=[a, b, 2.0], device=device)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_ptr")
        wp.capture_save(capture.graph, path, inputs={"a": a}, outputs={"b": b})

        loaded = wp.capture_load(path, device=device)

        # Known name: should return a non-zero integer device pointer.
        ptr = loaded.get_param_ptr("a")
        test.assertIsInstance(ptr, int, "expected an integer device pointer")
        test.assertNotEqual(ptr, 0, "expected a non-zero device pointer")

        # Unknown name: should return None.
        test.assertIsNone(loaded.get_param_ptr("nonexistent"))

    # Non-loaded graph: should raise RuntimeError mentioning loaded APIC graphs.
    with wp.ScopedCapture(device=device, force_module_load=False) as plain_capture:
        wp.launch(scale_kernel, dim=n, inputs=[a, b, 1.0], device=device)

    with test.assertRaisesRegex(RuntimeError, "loaded APIC"):
        plain_capture.graph.get_param_ptr("a")


devices = get_test_devices()
devices_with_mempool = get_test_devices_with_mempool()

add_function_test(TestApic, "test_save_apic_false_error", test_save_apic_false_error, devices=devices)
add_function_test(TestApic, "test_save_single_kernel", test_save_single_kernel, devices=devices)
add_function_test(TestApic, "test_save_load_round_trip", test_save_load_round_trip, devices=devices)
add_function_test(TestApic, "test_save_load_multiple_kernels", test_save_load_multiple_kernels, devices=devices)
add_function_test(TestApic, "test_save_load_memcpy", test_save_load_memcpy, devices=devices)
add_function_test(TestApic, "test_save_load_memset", test_save_load_memset, devices=devices)
add_function_test(TestApic, "test_bindings_param_update", test_bindings_param_update, devices=devices)
add_function_test(TestApic, "test_array_slicing", test_array_slicing, devices=devices)
add_function_test(TestApic, "test_complex_pipeline", test_complex_pipeline, devices=devices)
add_function_test(TestApic, "test_internal_allocation", test_internal_allocation, devices=devices_with_mempool)
add_function_test(
    TestApic,
    "test_multiple_internal_allocations",
    test_multiple_internal_allocations,
    devices=devices_with_mempool,
)
add_function_test(TestApic, "test_graph_execution_unchanged", test_graph_execution_unchanged, devices=devices)
add_function_test(TestApic, "test_save_load_with_param_update", test_save_load_with_param_update, devices=devices)
add_function_test(TestApic, "test_save_load_memcpy_and_kernel", test_save_load_memcpy_and_kernel, devices=devices)
add_function_test(
    TestApic, "test_save_load_fill", test_save_load_fill, devices=get_cuda_test_devices()
)  # CPU: wp_memtile_host not recorded
add_function_test(TestApic, "test_save_load_alloc_only", test_save_load_alloc_only, devices=devices_with_mempool)
add_function_test(TestApic, "test_get_param_ptr", test_get_param_ptr, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
