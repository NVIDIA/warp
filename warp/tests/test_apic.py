# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for APIC (API Capture) graph serialization and loading."""

import os
import tempfile
import unittest

import numpy as np

import warp as wp
import warp._src.context as wp_context
from warp._src.apic.capture import APICapture
from warp.tests.unittest_utils import (
    add_function_test,
    get_cuda_test_devices,
    get_test_devices,
    get_test_devices_with_cuda_graph_module_load,
    get_test_devices_with_mempool,
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


@wp.kernel
def touch_first_if_nonempty_kernel(empty_buf: wp.array(dtype=float), out: wp.array(dtype=float)):
    i = wp.tid()
    # Only touch empty_buf if its shape is non-zero — but the array is still
    # passed as a kernel argument, so APIC must record it without crashing.
    if empty_buf.shape[0] > 0:
        out[i] = empty_buf[0]
    else:
        out[i] = 7.0


@wp.struct
class _MiniHeightfieldData:
    data_offset: wp.int32
    nrow: wp.int32
    ncol: wp.int32
    hx: wp.float32
    hy: wp.float32
    min_z: wp.float32
    max_z: wp.float32


@wp.kernel
def two_d_strided_kernel(
    a: wp.array(dtype=int),
    b: wp.array(dtype=wp.transform),
    c: wp.array(dtype=wp.uint64),
    d: wp.array(dtype=float),
    e: wp.array(dtype=wp.vec4),
    f: wp.array(dtype=float),
    g: wp.array(dtype=int),
    h: wp.array(dtype=_MiniHeightfieldData),  # struct-array, possibly empty (mirrors heightfield_data)
    i: wp.array(dtype=wp.vec2i),
    j: wp.array(dtype=int),
    n_threads: int,
    out_pairs: wp.array(dtype=wp.vec3i),
    out_count: wp.array(dtype=int),
):
    tid, jj = wp.tid()
    cnt = j[0]
    for ii in range(tid, cnt, n_threads):
        if ii < cnt and jj == 0:
            wp.atomic_add(out_count, 0, 1)


def test_capture_2d_launch_minimal(test, device):
    """Minimal repro of basic_conveyor's narrow_phase_find_mesh_triangle_overlaps_kernel
    crash: 2D launch with 13 params (10 arrays + 1 scalar + 2 arrays), one of
    them potentially empty."""
    n = 23
    a = wp.zeros(n, dtype=int, device=device)
    b = wp.zeros(n, dtype=wp.transform, device=device)
    c = wp.zeros(n, dtype=wp.uint64, device=device)
    d = wp.zeros(n, dtype=float, device=device)
    e = wp.zeros(n, dtype=wp.vec4, device=device)
    f = wp.zeros(n, dtype=float, device=device)
    g = wp.zeros(n, dtype=int, device=device)
    h = wp.zeros(0, dtype=_MiniHeightfieldData, device=device)  # empty struct-array
    i = wp.zeros(8, dtype=wp.vec2i, device=device)
    j = wp.zeros(1, dtype=int, device=device)
    out_pairs = wp.zeros(64, dtype=wp.vec3i, device=device)
    out_count = wp.zeros(1, dtype=int, device=device)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.launch(
            two_d_strided_kernel,
            dim=[256, 128],
            inputs=[a, b, c, d, e, f, g, h, i, j, 256],
            outputs=[out_pairs, out_count],
            device=device,
        )

    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    # No assertion on result — we just want to confirm replay doesn't crash.


@wp.kernel
def _tile_using_kernel(out: wp.array(dtype=float)):
    # Touch wp.tile_zeros to make the kernel use tile shared storage; on CPU
    # this allocates a 256 KB tile_shared_storage_t on the kernel's stack frame.
    # If the APIC replay path also stack-allocates anything sizeable before
    # calling here, the combined frame can blow Windows' 1 MB thread stack
    # (regression: basic_conveyor on CPU APIC capture).
    t = wp.tile_zeros(shape=64, dtype=float)
    out[wp.tid()] = t[0]


def test_capture_replay_with_tile_kernel_no_stack_overflow(test, device):
    """Regression: a kernel that uses tile primitives allocates ~256 KB on
    the stack via tile_shared_storage_t. The APIC replay path used to keep
    sibling stack arrays (fwd_stack[512], adj_stack[512]) alongside the
    deep replay-call stack frame, which combined with the 256 KB the kernel
    itself takes was enough to blow the 1 MB Windows main-thread stack —
    Newton's basic_conveyor / basic_plotting / IK examples all hit this."""
    n = 32
    out = wp.zeros(n, dtype=float, device=device)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.launch_tiled(_tile_using_kernel, dim=n, inputs=[out], block_dim=64, device=device)

    # Multiple launches verify there's no slow leak / cumulative stack use.
    for _ in range(3):
        wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    np.testing.assert_allclose(out.numpy(), np.zeros(n, dtype=np.float32))


def test_save_load_tiled_nondefault_block_dim(test, device):
    """APIC save must copy the binary for the captured block_dim variant."""
    n = 32
    block_dim = 64
    out = wp.zeros(n, dtype=float, device=device)

    # Compile the default variant first so capture_save would have a plausible
    # but wrong binary to copy if it ignored the captured module executable's
    # block_dim.
    wp.load_module(device=device)
    wp.load_module(module=_tile_using_kernel.module, device=device, block_dim=block_dim)

    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.launch_tiled(_tile_using_kernel, dim=n, inputs=[out], block_dim=block_dim, device=device)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "tiled_block_dim")
        wp.capture_save(capture.graph, path, outputs={"out": out})

        module_infos = list(capture.graph._apic_capture.collected_modules.values())
        test.assertEqual(len(module_infos), 1)
        module_info = module_infos[0]
        test.assertIn(module_info["module_hash"][:7], module_info["binary_filename"])
        test.assertTrue(os.path.exists(os.path.join(path + "_modules", module_info["binary_filename"])))

        loaded = wp.capture_load(path, device=device)
        wp.capture_launch(loaded)
        wp.synchronize_device(device)

        result = wp.zeros(n, dtype=float, device=device)
        loaded.get_param("out", result)
        np.testing.assert_allclose(result.numpy(), np.zeros(n, dtype=np.float32))


@wp.kernel
def _vec3_after_int_kernel(
    pre: int,
    v: wp.vec3,  # 12 bytes, alignof(float)=4 — kernel struct lays this at offset 4 after `pre`
    post_int: int,
    post_float: float,
    arr_in: wp.array(dtype=int),
    arr_out: wp.array(dtype=float),
):
    i = wp.tid()
    arr_out[i] = float(arr_in[i] + pre + post_int) + v[0] + v[1] + v[2] + post_float


def test_capture_replay_vec3_scalar_alignment(test, device):
    """Regression: a vec3 param (size=12) appearing in the middle of a kernel
    signature has alignof=4 in C++, but the replay packer used a size-based
    heuristic (>= 8 → align 8) that introduced 4 bytes of phantom padding.
    Every subsequent param shifted, so array data pointers were read from
    the wrong offset and the kernel either AV'd or wrote garbage. Newton's
    _pos_residuals (used by the IK examples) is the canonical case."""
    n = 4
    arr_in = wp.array(np.arange(n, dtype=np.int32), device=device)
    arr_out = wp.zeros(n, dtype=float, device=device)
    v = wp.vec3(1.0, 2.0, 3.0)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.launch(_vec3_after_int_kernel, dim=n, inputs=[10, v, 100, 0.5, arr_in, arr_out], device=device)

    # Live execution sets the expected pattern; arr_out was zeroed before
    # capture and capture itself does not execute on CPU, so values come
    # from replay only.
    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)

    expected = np.arange(n, dtype=np.float32) + 10.0 + 100.0 + 1.0 + 2.0 + 3.0 + 0.5
    np.testing.assert_allclose(arr_out.numpy(), expected)


def test_capture_with_empty_array_input(test, device):
    """An empty array (shape=(0,), arr.ptr is None) used as a kernel input
    must not crash APIC capture in _find_base / track_array. This is the
    pattern Newton's narrow_phase hits with unused contact buffers."""
    n_out = 64
    out = wp.zeros(n_out, dtype=float, device=device)
    empty = wp.zeros(0, dtype=float, device=device)

    # Sanity: zero-length arrays carry arr.ptr == None (not 0)
    test.assertTrue(empty.ptr is None, f"expected None, got {empty.ptr!r}")

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.launch(touch_first_if_nonempty_kernel, dim=n_out, inputs=[empty, out], device=device)

    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    np.testing.assert_allclose(out.numpy(), np.full(n_out, 7.0, dtype=np.float32))


@wp.struct
class BigStruct96:
    """A 96-byte struct (24 wp.float32 fields, exceeds the 64-byte inline cap)."""

    f00: wp.float32
    f01: wp.float32
    f02: wp.float32
    f03: wp.float32
    f04: wp.float32
    f05: wp.float32
    f06: wp.float32
    f07: wp.float32
    f08: wp.float32
    f09: wp.float32
    f10: wp.float32
    f11: wp.float32
    f12: wp.float32
    f13: wp.float32
    f14: wp.float32
    f15: wp.float32
    f16: wp.float32
    f17: wp.float32
    f18: wp.float32
    f19: wp.float32
    f20: wp.float32
    f21: wp.float32
    f22: wp.float32
    f23: wp.float32


@wp.kernel
def big_struct_sum_kernel(s: BigStruct96, out: wp.array(dtype=float)):
    i = wp.tid()
    total = (
        s.f00
        + s.f01
        + s.f02
        + s.f03
        + s.f04
        + s.f05
        + s.f06
        + s.f07
        + s.f08
        + s.f09
        + s.f10
        + s.f11
        + s.f12
        + s.f13
        + s.f14
        + s.f15
        + s.f16
        + s.f17
        + s.f18
        + s.f19
        + s.f20
        + s.f21
        + s.f22
        + s.f23
    )
    out[i] = total


def test_capture_with_large_scalar_param(test, device):
    """A by-value struct param > 64 B (e.g. Newton's contact-writer struct)
    must capture and replay correctly via the per-launch scalar pool, both
    for in-memory replay and after a .wrp round-trip."""
    n = 8
    out = wp.zeros(n, dtype=float, device=device)

    s = BigStruct96()
    for i in range(24):
        setattr(s, f"f{i:02d}", float(i))
    expected = np.full(n, float(sum(range(24))), dtype=np.float32)  # 0+1+...+23 = 276

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.launch(big_struct_sum_kernel, dim=n, inputs=[s, out], device=device)

    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    np.testing.assert_allclose(out.numpy(), expected)

    # Round-trip through .wrp so the serializer/loader path is exercised
    # against the value-blob layout produced for a >64-byte scalar.
    out.zero_()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "large_scalar")
        wp.capture_save(capture.graph, path, outputs={"out": out})
        loaded = wp.capture_load(path, device=device)
        wp.capture_launch(loaded)
        wp.synchronize_device(device)
        result = wp.zeros(n, dtype=float, device=device)
        loaded.get_param("out", result)
        np.testing.assert_allclose(result.numpy(), expected)


@wp.kernel
def square_loss_kernel(x: wp.array(dtype=float), loss: wp.array(dtype=float)):
    i = wp.tid()
    wp.atomic_add(loss, 0, x[i] * x[i])


@wp.kernel
def assign_kernel(x: wp.array(dtype=float), y: wp.array(dtype=float)):
    i = wp.tid()
    y[i] = x[i]


def test_capture_backward_consumes_y_grad_cpu(test, device):
    """The adjoint replay path must consume an output array's seeded grad the
    same way live ``Tape.backward()`` does. Regression for the case where
    ``apic_pack_args_buf`` hardcoded ``arr->grad = 0`` and silently dropped
    the array's grad pointer and flags."""
    n = 4
    x = wp.array(np.arange(n, dtype=np.float32), device=device, requires_grad=True)
    y = wp.zeros(n, dtype=float, device=device, requires_grad=True)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        tape = wp.Tape()
        with tape:
            wp.launch(assign_kernel, dim=n, inputs=[x, y], device=device)
        tape.backward(grads={y: wp.ones(n, dtype=float, device=device)})

    # Seed the consumed-output's grad and replay. Live semantics: assign_kernel
    # backward reads y.grad, accumulates into x.grad, then zeros y.grad. APIC
    # replay must produce the same effect.
    x.grad.zero_()
    y.grad.fill_(1.0)
    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)

    np.testing.assert_allclose(x.grad.numpy(), np.ones(n, dtype=np.float32))
    np.testing.assert_allclose(y.grad.numpy(), np.zeros(n, dtype=np.float32))


def test_capture_backward_retain_grad_cpu(test, device):
    """``retain_grad=True`` must survive APIC capture: the array's
    ARRAY_FLAG_RETAIN_GRAD bit needs to round-trip via array_flags so that
    adj_array_store's consume path is skipped at replay."""
    n = 4
    x = wp.array(np.arange(n, dtype=np.float32), device=device, requires_grad=True)
    y = wp.zeros(n, dtype=float, device=device, requires_grad=True)
    y.retain_grad = True

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        tape = wp.Tape()
        with tape:
            wp.launch(assign_kernel, dim=n, inputs=[x, y], device=device)
        tape.backward(grads={y: wp.ones(n, dtype=float, device=device)})

    x.grad.zero_()
    y.grad.fill_(1.0)
    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)

    # With retain_grad=True, y.grad must NOT be consumed.
    np.testing.assert_allclose(x.grad.numpy(), np.ones(n, dtype=np.float32))
    np.testing.assert_allclose(y.grad.numpy(), np.ones(n, dtype=np.float32))


def test_capture_backward_kernel_cpu(test, device):
    """Backward kernel launches must record cleanly during APIC capture and
    replay correctly. Drives the diffsim path Newton needs."""
    n = 8
    x = wp.array(np.arange(n, dtype=np.float32), device=device, requires_grad=True)
    loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        tape = wp.Tape()
        with tape:
            wp.launch(square_loss_kernel, dim=n, inputs=[x, loss], device=device)
        tape.backward(loss)

    # Reset gradients and replay
    x.grad.zero_()
    loss.zero_()
    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)

    # d/dx (sum x_i^2) = 2 x
    np.testing.assert_allclose(x.grad.numpy(), 2.0 * np.arange(n, dtype=np.float32), rtol=1e-5)
    np.testing.assert_allclose(loss.numpy(), [float(np.sum(np.arange(n) ** 2))], rtol=1e-5)


@wp.struct
class StructWithArrays:
    """By-value struct holding two wp.array fields. The launch-param walker
    must emit DATA_PTR / NULL relocations for each nested array_t.data and
    array_t.grad slot, at the correct offsets within the struct blob."""

    pos: wp.array(dtype=wp.vec3)
    vel: wp.array(dtype=wp.vec3)
    radius: float


@wp.kernel
def step_particles_struct_kernel(p: StructWithArrays, dt: float):
    tid = wp.tid()
    p.pos[tid] = p.pos[tid] + p.vel[tid] * dt


def test_capture_struct_with_array_cpu(test, device):
    """A @wp.struct containing wp.array fields must capture + replay correctly:
    each nested array_t.data field is patched via a per-blob relocation, so
    replay writes into the same memory the live launch would have written.
    """
    n = 4
    p = StructWithArrays()
    p.pos = wp.array(np.zeros((n, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    p.vel = wp.array(np.ones((n, 3), dtype=np.float32), dtype=wp.vec3, device=device)
    p.radius = 0.5

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.launch(step_particles_struct_kernel, dim=n, inputs=[p, 0.5], device=device)

    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)

    expected = np.full((n, 3), 0.5, dtype=np.float32)
    np.testing.assert_allclose(p.pos.numpy(), expected, rtol=1e-5)

    # Round-trip through .wrp: the value-blob for `p` carries the nested
    # array_t.data relocations for pos / vel; the loader must rebuild them
    # against the loaded-graph's region pointers.
    p.pos.zero_()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "struct_with_array")
        wp.capture_save(capture.graph, path, outputs={"pos": p.pos})
        loaded = wp.capture_load(path, device=device)
        wp.capture_launch(loaded)
        wp.synchronize_device(device)
        result = wp.zeros(n, dtype=wp.vec3, device=device)
        loaded.get_param("pos", result)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)


@wp.kernel
def write_indexed_kernel(arr: wp.indexedarray(dtype=float, ndim=1), v: float):
    arr[wp.tid()] = v


def test_capture_indexedarray_cpu(test, device):
    """wp.indexedarray as a launch parameter must capture + replay correctly:
    the value blob is the indexedarray_t descriptor; the walker emits relocs
    for the nested array_t.data, array_t.grad, and each indices[d] pointer.
    """
    base_values = np.arange(10, dtype=np.float32)
    base = wp.array(base_values, dtype=float, device=device)
    indices = wp.array([1, 3, 5, 7, 9], dtype=int, device=device)
    iarr = wp.indexedarray1d(base, [indices])

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.launch(write_indexed_kernel, dim=iarr.size, inputs=[iarr, 42.0], device=device)

    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)

    expected = base_values.copy()
    expected[[1, 3, 5, 7, 9]] = 42.0
    np.testing.assert_allclose(base.numpy(), expected, rtol=1e-5)

    # Round-trip through .wrp: confirms the indexedarray_t value blob and
    # its three relocation kinds (data, grad/null, indices[d]) survive
    # serialization and reload.
    base.assign(base_values)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "indexedarray")
        wp.capture_save(capture.graph, path, outputs={"base": base})
        loaded = wp.capture_load(path, device=device)
        wp.capture_launch(loaded)
        wp.synchronize_device(device)
        result = wp.zeros(len(base_values), dtype=float, device=device)
        loaded.get_param("base", result)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)


@wp.kernel
def weighted_sample_sum_kernel(
    samples: wp.indexedarray(dtype=float, ndim=1),
    weights: wp.array(dtype=float),
    total: wp.array(dtype=float),
):
    i = wp.tid()
    wp.atomic_add(total, 0, samples[i] * weights[i])


def test_capture_indexedarray_adjoint_pack_cpu(test, device):
    """Custom-adjoint launch with an ``wp.indexedarray`` argument must capture
    without raising. The forward arg type is ``indexedarray``, but Warp's
    backward ABI represents the corresponding adjoint as a plain ``wp.array``
    (the underlying gradient buffer). APIC must walk the adjoint value blob
    using the array layout, not the indexedarray layout, or it would try to
    read ``.data`` / ``.indices`` slots that aren't there.

    The test stops at capture-end. Replay of CPU backward for indexedarray is
    a separate, pre-existing Warp/codegen issue (reproduces against
    ``omniverse/main`` without any APIC involvement); that's tracked outside
    this gist.
    """
    base_values = np.linspace(1.0, 6.0, 6, dtype=np.float32)
    base = wp.array(base_values, dtype=float, device=device, requires_grad=True)
    weights = wp.array([0.25, 0.5, 1.0], dtype=float, device=device)
    sample_ids = wp.array([1, 3, 5], dtype=wp.int32, device=device)
    samples = wp.indexedarray1d(base, [sample_ids])
    total = wp.zeros(1, dtype=float, device=device, requires_grad=True)
    total.grad.fill_(1.0)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.launch(
            weighted_sample_sum_kernel,
            dim=samples.size,
            inputs=[samples, weights],
            outputs=[total],
            adj_inputs=[base.grad, None],
            adj_outputs=[total.grad],
            adjoint=True,
            device=device,
        )

    # The bug was an AttributeError on .indices during adjoint pack — getting
    # here at all proves it's fixed. capture.graph should be valid even though
    # we don't replay it.
    test.assertIsNotNone(capture.graph)


@wp.kernel
def write_value_kernel(x: wp.array(dtype=float), value: float):
    x[wp.tid()] = value


@wp.kernel
def decrement_counter_kernel(c: wp.array(dtype=wp.int32)):
    c[0] = c[0] - 1


def test_capture_if_cpu(test, device):
    """APIC_OP_IF on CPU: condition selects which branch runs at replay."""
    n = 4
    out = wp.zeros(n, dtype=float, device=device)
    cond = wp.array([1], dtype=wp.int32, device=device)

    def on_true():
        wp.launch(write_value_kernel, dim=n, inputs=[out, 11.0], device=device)

    def on_false():
        wp.launch(write_value_kernel, dim=n, inputs=[out, 22.0], device=device)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.capture_if(cond, on_true=on_true, on_false=on_false)

    # Replay with cond=1 -> on_true branch -> out filled with 11.0
    out.zero_()
    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    np.testing.assert_allclose(out.numpy(), np.full(n, 11.0, dtype=np.float32))

    # Flip cond and replay -> on_false branch -> out filled with 22.0
    cond.assign([0])
    out.zero_()
    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    np.testing.assert_allclose(out.numpy(), np.full(n, 22.0, dtype=np.float32))


def test_capture_while_cpu(test, device):
    """APIC_OP_WHILE on CPU: body re-runs while the condition int32 is nonzero."""
    counter = wp.array([5], dtype=wp.int32, device=device)

    def body():
        # decrement counter by 1 each iteration
        wp.launch(decrement_counter_kernel, dim=1, inputs=[counter], device=device)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.capture_while(counter, while_body=body)

    counter.assign([5])
    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    test.assertEqual(int(counter.numpy()[0]), 0)

    # Replay again with a different starting value to verify the loop is
    # actually re-evaluated, not unrolled at capture time.
    counter.assign([3])
    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    test.assertEqual(int(counter.numpy()[0]), 0)


def test_capture_with_array_scan(test, device):
    """``wp.utils.array_scan`` interaction with APIC capture:

    - CPU: scan ops are not yet recorded into the byte stream; the helper
      raises so users don't silently get a graph with stale scan output.
      Tracks the gist's Issue 2.
    - CUDA: the scan kernels are picked up by ``cudaGraph`` stream capture,
      so capture + launch reproduces the live scan output.
    """
    n = 32
    src = wp.array(np.ones(n, dtype=np.int32), dtype=wp.int32, device=device)
    dst_in = wp.zeros(n, dtype=wp.int32, device=device)
    dst_ex = wp.zeros(n, dtype=wp.int32, device=device)

    wp.load_module(device=device)

    if wp.get_device(device).is_cpu:
        with test.assertRaises(NotImplementedError):
            with wp.ScopedCapture(device=device, apic=True, force_module_load=False):
                wp.utils.array_scan(src, dst_in, inclusive=True)
        return

    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.utils.array_scan(src, dst_in, inclusive=True)
        wp.utils.array_scan(src, dst_ex, inclusive=False)

    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    # Inclusive scan of [1]*n is [1,2,...,n]; exclusive is [0,1,...,n-1].
    np.testing.assert_allclose(dst_in.numpy(), np.arange(1, n + 1, dtype=np.int32))
    np.testing.assert_allclose(dst_ex.numpy(), np.arange(0, n, dtype=np.int32))


def test_end_recording_null_state_preserves_active(test, device):
    """wp_apic_end_recording(nullptr) must not clobber g_apic_state when
    another (valid) recording is active. Regression for Greptile fb1661ef.
    """
    n = 64
    a = wp.array(np.ones(n, dtype=np.float32), device=device)
    b = wp.zeros(n, dtype=float, device=device)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.launch(scale_kernel, dim=n, inputs=[a, b, 2.0], device=device)
        # While capture is active, an external caller (e.g. error-cleanup path
        # in another component) calls wp_apic_end_recording(NULL).
        core = wp_context.runtime.core
        active_before = core.wp_apic_get_recording_state()
        test.assertIsNotNone(active_before, "expected active recording state")
        core.wp_apic_end_recording(None)
        active_after = core.wp_apic_get_recording_state()
        # The active capture must still be recording.
        test.assertEqual(
            active_before,
            active_after,
            "wp_apic_end_recording(NULL) clobbered the active g_apic_state",
        )

    # Capture should still complete cleanly and replay correctly.
    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    np.testing.assert_allclose(b.numpy(), np.full(n, 2.0, dtype=np.float32))


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
devices_with_cuda_graph_module_load = get_test_devices_with_cuda_graph_module_load()
devices_with_mempool_and_cuda_graph_module_load = get_test_devices_with_mempool_and_cuda_graph_module_load()

add_function_test(
    TestApic,
    "test_save_apic_false_error",
    test_save_apic_false_error,
    devices=devices_with_cuda_graph_module_load,
)
add_function_test(
    TestApic,
    "test_save_single_kernel",
    test_save_single_kernel,
    devices=devices_with_cuda_graph_module_load,
)
add_function_test(
    TestApic,
    "test_save_load_round_trip",
    test_save_load_round_trip,
    devices=devices_with_cuda_graph_module_load,
)
add_function_test(
    TestApic,
    "test_save_load_multiple_kernels",
    test_save_load_multiple_kernels,
    devices=devices_with_cuda_graph_module_load,
)
add_function_test(TestApic, "test_save_load_memcpy", test_save_load_memcpy, devices=devices)
add_function_test(TestApic, "test_save_load_memset", test_save_load_memset, devices=devices)
add_function_test(
    TestApic,
    "test_bindings_param_update",
    test_bindings_param_update,
    devices=devices_with_cuda_graph_module_load,
)
add_function_test(TestApic, "test_array_slicing", test_array_slicing, devices=devices)
add_function_test(
    TestApic,
    "test_complex_pipeline",
    test_complex_pipeline,
    devices=devices_with_cuda_graph_module_load,
)
add_function_test(
    TestApic,
    "test_internal_allocation",
    test_internal_allocation,
    devices=devices_with_mempool_and_cuda_graph_module_load,
)
add_function_test(
    TestApic,
    "test_multiple_internal_allocations",
    test_multiple_internal_allocations,
    devices=devices_with_mempool_and_cuda_graph_module_load,
)
add_function_test(
    TestApic,
    "test_graph_execution_unchanged",
    test_graph_execution_unchanged,
    devices=devices_with_cuda_graph_module_load,
)
add_function_test(
    TestApic,
    "test_save_load_with_param_update",
    test_save_load_with_param_update,
    devices=devices_with_cuda_graph_module_load,
)
add_function_test(
    TestApic,
    "test_save_load_memcpy_and_kernel",
    test_save_load_memcpy_and_kernel,
    devices=devices_with_cuda_graph_module_load,
)
add_function_test(
    TestApic, "test_save_load_fill", test_save_load_fill, devices=get_cuda_test_devices()
)  # CPU: wp_memtile_host not recorded
add_function_test(TestApic, "test_save_load_alloc_only", test_save_load_alloc_only, devices=devices_with_mempool)
add_function_test(TestApic, "test_capture_with_empty_array_input", test_capture_with_empty_array_input, devices=devices)
add_function_test(
    TestApic,
    "test_capture_replay_with_tile_kernel_no_stack_overflow",
    test_capture_replay_with_tile_kernel_no_stack_overflow,
    devices=[d for d in devices if d.is_cpu],
)
add_function_test(
    TestApic,
    "test_save_load_tiled_nondefault_block_dim",
    test_save_load_tiled_nondefault_block_dim,
    devices=get_cuda_test_devices(),
)
add_function_test(
    TestApic,
    "test_capture_replay_vec3_scalar_alignment",
    test_capture_replay_vec3_scalar_alignment,
    devices=[d for d in devices if d.is_cpu],
)
add_function_test(
    TestApic,
    "test_capture_2d_launch_minimal",
    test_capture_2d_launch_minimal,
    devices=[d for d in devices if d.is_cpu],
)
add_function_test(
    TestApic, "test_capture_with_large_scalar_param", test_capture_with_large_scalar_param, devices=devices
)
add_function_test(TestApic, "test_capture_with_array_scan", test_capture_with_array_scan, devices=devices)
# Backward kernel capture is currently CPU-only; CUDA backward path doesn't go through APIC yet.
add_function_test(
    TestApic,
    "test_capture_backward_kernel_cpu",
    test_capture_backward_kernel_cpu,
    devices=[d for d in devices if d.is_cpu],
)
add_function_test(
    TestApic,
    "test_capture_backward_consumes_y_grad_cpu",
    test_capture_backward_consumes_y_grad_cpu,
    devices=[d for d in devices if d.is_cpu],
)
add_function_test(
    TestApic,
    "test_capture_struct_with_array_cpu",
    test_capture_struct_with_array_cpu,
    devices=[d for d in devices if d.is_cpu],
)
add_function_test(
    TestApic,
    "test_capture_indexedarray_cpu",
    test_capture_indexedarray_cpu,
    devices=[d for d in devices if d.is_cpu],
)
add_function_test(
    TestApic,
    "test_capture_indexedarray_adjoint_pack_cpu",
    test_capture_indexedarray_adjoint_pack_cpu,
    devices=[d for d in devices if d.is_cpu],
)
add_function_test(
    TestApic,
    "test_capture_backward_retain_grad_cpu",
    test_capture_backward_retain_grad_cpu,
    devices=[d for d in devices if d.is_cpu],
)
# Conditional / loop capture is currently CPU-only; CUDA uses the existing
# cudaGraphConditional* path during live capture, but loaded-graph replay of
# IF / WHILE on CUDA isn't implemented yet.
add_function_test(TestApic, "test_capture_if_cpu", test_capture_if_cpu, devices=[d for d in devices if d.is_cpu])
add_function_test(TestApic, "test_capture_while_cpu", test_capture_while_cpu, devices=[d for d in devices if d.is_cpu])
add_function_test(
    TestApic,
    "test_end_recording_null_state_preserves_active",
    test_end_recording_null_state_preserves_active,
    devices=devices,
)
add_function_test(TestApic, "test_get_param_ptr", test_get_param_ptr, devices=devices_with_cuda_graph_module_load)


if __name__ == "__main__":
    unittest.main(verbosity=2)
