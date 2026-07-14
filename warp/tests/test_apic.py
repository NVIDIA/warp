# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for APIC (API Capture) graph serialization and loading."""

import ctypes
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

import warp as wp
import warp._src.context as wp_context
from warp._src.apic.capture import APICapture
from warp.sparse import (
    BSR_STATUS_ROW_CAPACITY_EXCEEDED,
    bsr_set_from_triplets,
    bsr_set_transpose,
    bsr_zeros,
)
from warp.tests.unittest_utils import (
    add_function_test,
    get_cuda_test_devices,
    get_test_devices,
    get_test_devices_with_cuda_graph_module_load,
    get_test_devices_with_graph_capture_allocation,
    get_test_devices_with_graph_capture_allocation_and_cuda_graph_module_load,
)


@wp.kernel
def scale_kernel(input: wp.array(dtype=float), output: wp.array(dtype=float), s: float):
    i = wp.tid()
    output[i] = input[i] * s


@wp.kernel
def add_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float), output: wp.array(dtype=float)):
    i = wp.tid()
    output[i] = a[i] + b[i]


@wp.kernel
def fill_descending_kernel(keys: wp.array(dtype=float), values: wp.array(dtype=wp.int32), count: wp.int32):
    i = wp.tid()
    keys[i] = float(count - 1 - i)
    values[i] = i


@wp.kernel
def fill_runs_kernel(values: wp.array(dtype=wp.int32)):
    # Produces consecutive runs of length 3: [0,0,0,1,1,1,2,2,2,...].
    i = wp.tid()
    values[i] = i // 3


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


def test_save_load_capture_time_scratch_cuda(test, device):
    """A buffer allocated via the bare wp.array(shape=...) constructor *during* a
    CUDA APIC capture is graph-scoped. track_array marks it transient (keyed on this
    capture's apic_state) so capture_save serializes its size only and the rebuild
    regenerates it from the recorded kernels. A persistent array allocated *before* the
    capture must stay non-transient and round-trip with its real data (scoping invariant)."""
    n = 64
    a = wp.array(np.arange(n, dtype=np.float32), device=device)  # persistent input
    out = wp.zeros(n, dtype=float, device=device)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        # Bare constructor bypasses wp.empty(); allocated during capture -> transient.
        scratch = wp.array(shape=(n,), dtype=float, device=device)
        wp.launch(scale_kernel, dim=n, inputs=[a, scratch, 2.0], device=device)  # scratch = a * 2
        wp.launch(scale_kernel, dim=n, inputs=[scratch, out, 3.0], device=device)  # out = scratch * 3

    apic_capture = capture.graph._apic_capture
    scratch_region = apic_capture.get_region_id(scratch)
    input_region = apic_capture.get_region_id(a)
    # Capture-time scratch is transient; the pre-capture input is not (scoping invariant).
    test.assertIn(scratch_region, apic_capture._transient_regions)
    test.assertNotIn(input_region, apic_capture._transient_regions)

    expected = np.arange(n, dtype=np.float32) * 6.0
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "scratch_roundtrip")
        wp.capture_save(capture.graph, path, inputs={"a": a}, outputs={"out": out})
        loaded = wp.capture_load(path, device=device)
        wp.capture_launch(loaded)
        wp.synchronize_device(device)
        result = wp.zeros(n, dtype=float, device=device)
        loaded.get_param("out", result)
        np.testing.assert_allclose(result.numpy(), expected)


def test_apic_h2d_rejected_during_capture(test, device):
    """Host-to-device transfers are not recorded into the APIC byte stream, so
    initializing a CUDA array from host data during an APIC capture cannot be reproduced on
    replay. wp.copy() from a host array and wp.array(data=..., device=cuda) must be rejected
    rather than silently producing an uninitialized region on load."""
    n = 16
    host = np.arange(n, dtype=np.float32)
    dst = wp.zeros(n, dtype=float, device=device)
    src_cpu = wp.array(host, device="cpu")

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False):
        dst.fill_(0.0)  # a recorded op so the capture is non-empty
        with test.assertRaises(NotImplementedError):
            wp.copy(dst, src_cpu)
        with test.assertRaises(NotImplementedError):
            wp.array(host, dtype=float, device=device)


def test_apic_cuda_copy_gaps_rejected_during_capture(test, device):
    """Copy variants with no APIC byte-stream representation must fail loudly under a CUDA
    APIC capture rather than silently dropping from the saved graph: a non-contiguous
    (indexed/strided) CUDA copy, and a device-to-host copy. Contiguous same-device D2D is
    recorded and unaffected."""
    base = wp.zeros(8, dtype=wp.float32, device=device)
    idx = wp.array([0, 2, 4, 6], dtype=wp.int32, device=device)
    indexed_dst = wp.indexedarray1d(base, [idx])
    src4 = wp.zeros(4, dtype=wp.float32, device=device)
    cuda_src = wp.zeros(4, dtype=wp.float32, device=device)
    cpu_dst = wp.zeros(4, dtype=wp.float32, device="cpu")

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False):
        base.fill_(0.0)  # a recorded op so the capture is non-empty
        # Non-contiguous (indexed) CUDA copy: no APIC record -> must reject, not silently drop.
        with test.assertRaises(NotImplementedError):
            wp.copy(indexed_dst, src4)
        # Device-to-host copy: no host-destination region model yet -> must reject.
        with test.assertRaises(NotImplementedError):
            wp.copy(cpu_dst, cuda_src)


def test_apic_cuda_indexed_fill_rejected_during_capture(test, device):
    """Indexed CUDA fill has no APIC byte-stream representation yet, so it must fail
    loudly instead of silently dropping from the saved graph."""
    base = wp.zeros(8, dtype=wp.float32, device=device)
    idx = wp.array([0, 2, 4, 6], dtype=wp.int32, device=device)
    indexed = wp.indexedarray1d(base, [idx])

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False):
        base.fill_(0.0)  # a recorded op so the capture is non-empty
        with test.assertRaises(NotImplementedError):
            indexed.fill_(1.0)


def test_apic_capture_while_body_raises_cleanup(test, device):
    """A raising capture_while body must propagate cleanly and leave the device's
    capture state consistent (parent graph restored, parent capture resumed, APIC recording
    torn down, no leaked branch) so a subsequent capture works. The broken capture is never
    launched -- its while node has an empty body and would loop forever."""
    if device.is_cuda and not wp.is_conditional_graph_supported():
        test.skipTest("CUDA conditional graph nodes require Toolkit and driver 12.4+")

    cond = wp.ones(1, dtype=wp.int32, device=device)

    class Sentinel(Exception):
        pass

    def bad_body():
        raise Sentinel("boom")

    wp.load_module(device=device)
    with test.assertRaises(Sentinel):
        with wp.ScopedCapture(device=device, apic=True, force_module_load=False):
            wp.capture_while(cond, bad_body)

    # The capture was torn down (no leaked global recording state).
    test.assertIsNone(wp_context.runtime._apic_capture)

    # The device is reusable: a fresh capture records and replays correctly.
    out = wp.zeros(8, dtype=float, device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.launch(write_value_kernel, dim=8, inputs=[out, 7.0], device=device)
    out.zero_()
    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    np.testing.assert_array_equal(out.numpy(), np.full(8, 7.0, dtype=np.float32))


def test_apic_capture_if_body_raises_cleanup(test, device):
    """Companion to the capture_while case -- a raising capture_if branch body
    propagates cleanly (branch rolled back, parent graph restored and capture resumed) and
    leaves the device reusable for a subsequent capture."""
    if device.is_cuda and not wp.is_conditional_graph_supported():
        test.skipTest("CUDA conditional graph nodes require Toolkit and driver 12.4+")

    cond = wp.ones(1, dtype=wp.int32, device=device)

    class Sentinel(Exception):
        pass

    def bad_branch():
        raise Sentinel("boom")

    wp.load_module(device=device)
    with test.assertRaises(Sentinel):
        with wp.ScopedCapture(device=device, apic=True, force_module_load=False):
            wp.capture_if(cond, bad_branch)

    test.assertIsNone(wp_context.runtime._apic_capture)

    out = wp.zeros(8, dtype=float, device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.launch(write_value_kernel, dim=8, inputs=[out, 9.0], device=device)
    out.zero_()
    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    np.testing.assert_array_equal(out.numpy(), np.full(8, 9.0, dtype=np.float32))


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


def test_apic_alloc_grow_during_capture(test, device):
    """A NEW array allocated inside capture stays valid across multiple replays.

    Mirrors the Newton SolverVBD pattern of growing a buffer during capture: the temporary is
    allocated while recording, and the captured graph must re-resolve its region against the
    retained allocation on every replay (not just the first). Gated on graph-capture allocation
    capability because in-capture allocation on CUDA requires ``cudaMallocAsync``; CPU allocates
    during capture regardless.
    """
    n = 256
    input_data = wp.array(np.arange(n, dtype=np.float32) + 1.0, device=device)
    output_data = wp.zeros(n, dtype=float, device=device)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        # Fresh allocation made *during* capture (the buffer-grow analog).
        tmp = wp.empty(n, dtype=float, device=device)
        wp.launch(scale_kernel, dim=n, inputs=[input_data, tmp, 2.0], device=device)  # tmp = input*2
        wp.launch(add_kernel, dim=n, inputs=[tmp, input_data, output_data], device=device)  # out = input*3

    expected = (np.arange(n, dtype=np.float32) + 1.0) * 3.0

    # Replay twice, clobbering the output in between, to prove the captured temporary's region is
    # re-resolved against the still-live allocation on each launch, not just the first.
    for _ in range(2):
        output_data.zero_()
        wp.capture_launch(capture.graph)
        wp.synchronize_device(device)
        np.testing.assert_allclose(output_data.numpy(), expected)


def test_cpu_graph_alloc_not_leaked_on_relaunch(test, device):
    """In-graph allocations on CPU must not accumulate across relaunches.

    CPU APIC allocates a graph's in-capture buffers once (at capture time) and reuses them
    on every replay -- APIC_OP_ALLOC is a no-op during replay and the backing array is
    retained for the graph's lifetime -- so relaunching must perform no further host
    allocations. (CUDA achieves the analogous no-leak guarantee via its stream-ordered graph
    mempool; see test_cuda_graph_alloc_retained_release.) This guards against a regression
    that re-allocates, and leaks, on every launch. The allocation tracker counts host
    allocations, so the count must not grow across the relaunch loop.
    """
    core = wp_context.runtime.core
    n = 1024
    input_data = wp.array(np.arange(n, dtype=np.float32) + 1.0, device=device)
    output_data = wp.zeros(n, dtype=float, device=device)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        # Allocation made *inside* the capture; replay must reuse it, not re-allocate.
        tmp = wp.empty(n, dtype=float, device=device)
        wp.launch(scale_kernel, dim=n, inputs=[input_data, tmp, 2.0], device=device)  # tmp = input*2
        wp.launch(add_kernel, dim=n, inputs=[tmp, input_data, output_data], device=device)  # out = input*3

    expected = (np.arange(n, dtype=np.float32) + 1.0) * 3.0

    # Warm-up launch establishes steady state and checks correctness.
    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    np.testing.assert_allclose(output_data.numpy(), expected)

    # Count host allocations performed across many relaunches; reuse means this stays at zero,
    # whereas a per-launch re-allocation would scale with the relaunch count.
    was_enabled = bool(core.wp_alloc_tracker_is_enabled())
    core.wp_alloc_tracker_enable(1)
    try:
        before = core.wp_alloc_tracker_get_total_alloc_count()
        n_relaunch = 50
        for _ in range(n_relaunch):
            wp.capture_launch(capture.graph)
        wp.synchronize_device(device)
        allocations = core.wp_alloc_tracker_get_total_alloc_count() - before
    finally:
        if not was_enabled:
            core.wp_alloc_tracker_enable(0)

    test.assertEqual(
        allocations,
        0,
        f"CPU graph relaunch performed {allocations} host allocation(s) over {n_relaunch} relaunches; "
        "in-capture allocations should be reused, not re-allocated per launch",
    )

    # Still correct after the relaunch loop (reused buffers were not corrupted).
    output_data.zero_()
    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    np.testing.assert_allclose(output_data.numpy(), expected)


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
    block_dim = 64
    out = wp.zeros(n, dtype=float, device=device)

    wp.load_module(device=device)
    if device.is_cuda:
        wp.load_module(module=_tile_using_kernel.module, device=device, block_dim=block_dim)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.launch_tiled(_tile_using_kernel, dim=n, inputs=[out], block_dim=block_dim, device=device)

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

    # arr_out was zeroed before capture. On CPU the APIC capture only records
    # (replay is the sole writer); on CUDA the capture also executes live. The
    # kernel is idempotent, so replaying the captured graph yields the expected
    # pattern on either device.
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


def test_capture_backward_consumes_y_grad(test, device):
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


def test_capture_backward_retain_grad(test, device):
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


def test_capture_backward_kernel(test, device):
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


def test_capture_struct_with_array(test, device):
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


def test_capture_indexedarray(test, device):
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


def test_capture_indexedarray_adjoint_pack(test, device):
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
    if device.is_cuda and not wp.is_conditional_graph_supported():
        test.skipTest("CUDA conditional graph nodes require Toolkit and driver 12.4+")

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


def test_save_load_capture_if_cuda(test, device):
    """CUDA loaded-graph replay DOES support APIC conditional ops --
    apic_replay_ops_into_cuda_capture rebuilds the conditional nodes from the
    byte stream. Save/load a capture_if graph and confirm the recorded branch
    runs on the rebuilt graph."""
    if not wp.is_conditional_graph_supported():
        test.skipTest("CUDA conditional graph nodes require Toolkit and driver 12.4+")

    n = 4
    out = wp.zeros(n, dtype=float, device=device)
    cond = wp.array([1], dtype=wp.int32, device=device)  # selects on_true at capture

    def on_true():
        wp.launch(write_value_kernel, dim=n, inputs=[out, 11.0], device=device)

    def on_false():
        wp.launch(write_value_kernel, dim=n, inputs=[out, 22.0], device=device)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.capture_if(cond, on_true=on_true, on_false=on_false)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "capture_if")
        wp.capture_save(capture.graph, path, outputs={"out": out})

        loaded = wp.capture_load(path, device=device)
        wp.capture_launch(loaded)
        wp.synchronize_device(device)

        result = wp.zeros(n, dtype=float, device=device)
        loaded.get_param("out", result)
        np.testing.assert_allclose(result.numpy(), np.full(n, 11.0, dtype=np.float32))


def test_capture_while_cpu(test, device):
    """APIC_OP_WHILE on CPU: body re-runs while the condition int32 is nonzero."""
    if device.is_cuda and not wp.is_conditional_graph_supported():
        test.skipTest("CUDA conditional graph nodes require Toolkit and driver 12.4+")

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


def _make_const_writer(value):
    """Build a kernel that writes a compile-time constant. Two kernels from this
    factory share ``kernel.key`` (``_make_const_writer__locals__kernel``) but
    compile to distinct modules via ``module="unique"``."""
    constant = value

    @wp.kernel(module="unique")
    def kernel(out: wp.array(dtype=wp.int32)):
        out[0] = wp.static(constant)

    return kernel


def test_capture_distinct_modules_same_key(test, device):
    """Regression: kernels built with ``module="unique"`` from one factory share
    a ``kernel.key`` but compile to distinct functions. APIC keyed CPU replay
    kernels by key alone, so the second registration clobbered the first and
    replay dispatched the wrong compiled kernel (wrong results or out-of-bounds
    access). Each recorded launch must replay its own kernel."""
    ka = _make_const_writer(111)
    kb = _make_const_writer(222)
    test.assertEqual(ka.key, kb.key)  # same key, distinct compiled modules

    out_a = wp.zeros(1, dtype=wp.int32, device=device)
    out_b = wp.zeros(1, dtype=wp.int32, device=device)

    wp.load_module(ka.module, device=device)
    wp.load_module(kb.module, device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.launch(ka, dim=1, inputs=[out_a], device=device)
        wp.launch(kb, dim=1, inputs=[out_b], device=device)

    out_a.zero_()
    out_b.zero_()
    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    test.assertEqual(int(out_a.numpy()[0]), 111)
    test.assertEqual(int(out_b.numpy()[0]), 222)


def test_save_load_distinct_modules_same_key(test, device):
    """Saved CPU APIC graphs must also dispatch same-key unique-module kernels."""
    ka = _make_const_writer(111)
    kb = _make_const_writer(222)
    test.assertEqual(ka.key, kb.key)

    out_a = wp.zeros(1, dtype=wp.int32, device=device)
    out_b = wp.zeros(1, dtype=wp.int32, device=device)

    wp.load_module(ka.module, device=device)
    wp.load_module(kb.module, device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.launch(ka, dim=1, inputs=[out_a], device=device)
        wp.launch(kb, dim=1, inputs=[out_b], device=device)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "same_key_modules")
        wp.capture_save(capture.graph, path, outputs={"a": out_a, "b": out_b})

        loaded = wp.capture_load(path, device=device)
        wp.capture_launch(loaded)

        loaded_a = wp.empty_like(out_a)
        loaded_b = wp.empty_like(out_b)
        loaded.get_param("a", loaded_a)
        loaded.get_param("b", loaded_b)

    test.assertEqual(int(loaded_a.numpy()[0]), 111)
    test.assertEqual(int(loaded_b.numpy()[0]), 222)


def test_capture_with_array_scan(test, device):
    """Regression: Newton's broad/narrow-phase calls into wp_array_scan_*_host,
    whose internal scratch -> output memcpy used to fail apic_resolve_ptr and
    silently drop. After the fix, scan internals use plain memcpy."""
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


def test_cpu_helper_not_recorded_during_cuda_capture(test, device):
    """A CPU host helper invoked during a CUDA APIC capture must execute live,
    not record into the CUDA byte stream. The native host hooks are gated on the
    capture's device class, so a CPU array_scan run inside a CUDA capture runs
    immediately (its output is correct once the capture block exits) instead of
    being deferred/recorded into the CUDA capture. ``device`` is a CUDA
    device; the scan operates on CPU arrays."""
    n = 16
    src = wp.array(np.ones(n, dtype=np.int32), dtype=wp.int32, device="cpu")
    dst = wp.zeros(n, dtype=wp.int32, device="cpu")

    wp.load_module(device=device)
    # No host synchronization inside the CUDA capture (illegal under stream capture).
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False):
        wp.utils.array_scan(src, dst, inclusive=True)

    # If the CPU scan had been recorded (and thus deferred) into the CUDA capture
    # instead of executing live, dst would still be all zeros here.
    np.testing.assert_allclose(dst.numpy(), np.arange(1, n + 1, dtype=np.int32))


def test_capture_pause_resume_allows_unrecorded_allocation(test, device):
    """capture_pause()/capture_resume() must let callers build data outside the
    recorded APIC stream while surrounding launches still replay."""
    if device.is_cuda and not wp_context.is_conditional_graph_supported():
        test.skipTest("CUDA graph pause/resume requires CUDA Toolkit and driver 12.4+")

    n = 8
    base = np.arange(n, dtype=np.float32) + 1.0
    a = wp.array(base, device=device)
    out = wp.zeros(n, dtype=float, device=device)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.launch(scale_kernel, dim=n, inputs=[a, out, 2.0], device=device)  # out = a*2
        graph = wp_context.capture_pause(device=device)
        test.assertFalse(wp.get_device(device).is_capturing, "capture should be paused")
        # Allocation/work here must not be recorded into the graph.
        _scratch = wp.zeros(n, dtype=float, device=device)
        wp_context.capture_resume(graph, device=device)
        test.assertTrue(wp.get_device(device).is_capturing, "capture should have resumed")
        wp.launch(scale_kernel, dim=n, inputs=[out, out, 3.0], device=device)  # out = out*3

    out.zero_()
    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    # Both recorded launches replay across the pause: out = a*2*3 = 6a.
    np.testing.assert_allclose(out.numpy(), 6.0 * base)


def test_capture_pause_resume_suspends_apic_recording(test, device):
    """Work performed while APIC capture is paused must not be recorded into the
    APIC byte stream. Conditional graph construction uses an internal
    non-suspending pause path; this test covers the public pause/resume API."""
    if device.is_cuda and not wp_context.is_conditional_graph_supported():
        test.skipTest("CUDA graph pause/resume requires CUDA Toolkit and driver 12.4+")

    n = 8
    base = np.arange(n, dtype=np.float32) + 1.0
    src = wp.array(base, device=device)
    scratch = wp.zeros(n, dtype=float, device=device)
    paused = wp.zeros(n, dtype=float, device=device)
    out = wp.zeros(n, dtype=float, device=device)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.launch(scale_kernel, dim=n, inputs=[src, scratch, 2.0], device=device)
        graph = wp_context.capture_pause(device=device)
        test.assertFalse(wp.get_device(device).is_capturing, "capture should be paused")
        wp.launch(scale_kernel, dim=n, inputs=[src, paused, 5.0], device=device)
        wp_context.capture_resume(graph, device=device)
        test.assertTrue(wp.get_device(device).is_capturing, "capture should have resumed")
        wp.launch(add_kernel, dim=n, inputs=[scratch, paused, out], device=device)

    expected = base * 2.0
    paused.zero_()
    out.zero_()
    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    np.testing.assert_allclose(out.numpy(), expected)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "pause_resume")
        wp.capture_save(capture.graph, path, inputs={"src": src}, outputs={"out": out})

        loaded = wp.capture_load(path, device=device)
        result = wp.zeros(n, dtype=float, device=device)
        wp.capture_launch(loaded)
        wp.synchronize_device(device)
        loaded.get_param("out", result)
        np.testing.assert_allclose(result.numpy(), expected)


def test_bsr_nnz_sync_during_cpu_apic_capture(test, device):
    """BsrMatrix.nnz_sync() during a CPU APIC capture must not crash and must
    return the real block count: the readback is performed with recording paused
    so it runs live rather than recording a deferred host copy into an
    uninitialized buffer."""
    import warp.sparse as sparse  # noqa: PLC0415

    bsr = sparse.bsr_zeros(4, 4, wp.float32, device=device)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False):
        nnz = bsr.nnz_sync()

    test.assertIsInstance(nnz, int)
    test.assertEqual(nnz, 0)


def test_bsr_nnz_sync_after_recorded_topology_rejected(test, device):
    """A deferred topology update cannot provide a replay-time nnz during capture."""
    from warp.sparse import bsr_set_from_triplets, bsr_zeros  # noqa: PLC0415

    rows = wp.array(np.array([1, 0, 1], dtype=np.int32), dtype=wp.int32, device=device)
    columns = wp.array(np.array([2, 1, 2], dtype=np.int32), dtype=wp.int32, device=device)
    values = wp.array(np.array([1.0, 2.0, 3.0], dtype=np.float32), dtype=wp.float32, device=device)
    A = bsr_zeros(2, 3, block_type=wp.float32, device=device)

    with test.assertRaisesRegex(NotImplementedError, "topology update"):
        with wp.ScopedCapture(device=device, apic=True, force_module_load=False):
            bsr_set_from_triplets(A, rows, columns, values)
            A.nnz_sync()


def test_bsr_status_sync_during_cpu_apic_capture_rejected(test, device):
    """BsrMatrix.status_sync() must reject a host readback during a CPU APIC
    capture: a padded op recorded in the capture writes the status only on
    replay, so a status read inside the capture would observe a pre-replay
    value. Outside the capture the readback works normally."""
    from warp.sparse import BSR_STATUS_SUCCESS, bsr_zeros  # noqa: PLC0415

    bsr = bsr_zeros(2, 3, wp.float32, device=device, row_capacity=2)
    test.assertEqual(bsr.status_sync(), BSR_STATUS_SUCCESS)

    wp.load_module(device=device)
    with test.assertRaisesRegex(NotImplementedError, "CPU APIC graph capture"):
        with wp.ScopedCapture(device=device, apic=True, force_module_load=False):
            bsr.status_sync()


def test_save_load_array_scan_replay_with_updated_input(test, device):
    """``wp.utils.array_scan`` must be recorded into the byte stream so a saved
    + loaded graph recomputes against the current input rather than returning
    capture-time output. Regression for shi-eric's gist (Issue 2)."""
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
    """APIC scan records must preserve dtype, vector lanes, and 1D strides."""
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
    """Regression: wp.utils.segmented_sort_pairs on CPU dispatches to a host
    function that wasn't recorded into the APIC byte stream, so under graph
    capture/replay the sort silently didn't run and data stayed unsorted
    (Newton SAP broadphase ~10x slowdown). The fill kernel runs inside the
    capture so the keys at replay time differ from capture time, forcing the
    sort to actually replay."""
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
    """A captured segmented sort must be recorded into the byte stream so a
    saved + loaded graph re-sorts on replay rather than returning capture-time
    (unsorted) data."""
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
    """With explicit ``segment_end_indices`` the start array holds only
    ``num_segments`` entries (not ``num_segments + 1``), so the recorded
    start-region span must match the array. The earlier code always claimed
    ``num_segments + 1`` entries, over-running the explicit-end start array so
    save/load replay failed pointer resolution."""
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
    """Regression: wp.utils.radix_sort_pairs on CPU dispatches to a host function
    (wp_radix_sort_pairs_*_host) that, like the segmented sort, was invisible to
    the APIC byte stream and so didn't replay. The fill kernel runs inside the
    capture so replay-time keys differ from capture time."""
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
    """APIC radix-sort records must preserve key dtype, bit range, and value size."""
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


def test_borrow_temporary_not_recycled_during_apic_capture(test, device):
    """Regression: ``warp.fem`` temporaries borrowed during APIC capture must not
    be recycled by the temporary pool. The captured byte stream references them by
    pointer, so a release + re-borrow that handed the same pool memory back for a
    distinct captured region left replay reading/writing stale memory.
    Under an active capture the pool is bypassed, so a released buffer's memory is
    not aliased onto the next borrow.
    """
    from warp._src.fem import cache as fem_cache  # noqa: PLC0415

    store = fem_cache.TemporaryStore()
    n = 64

    with wp.ScopedCapture(device=device, apic=True, force_module_load=False):
        t1 = fem_cache.borrow_temporary(store, shape=(n,), dtype=wp.float32, device=device)
        ptr1 = t1.ptr
        # Without the fix this returns t1's buffer to the pool; with the fix it is
        # a no-op so t1 stays alive for the graph's lifetime.
        t1.release()
        t2 = fem_cache.borrow_temporary(store, shape=(n,), dtype=wp.float32, device=device)

    # The pool must not have recycled t1's memory onto t2 while capturing.
    test.assertNotEqual(t2.ptr, ptr1, "temporary pool recycled a buffer during APIC capture")
    # t1 must remain alive at its original address (release was neutralized).
    test.assertEqual(t1.ptr, ptr1, "captured temporary was released mid-capture")


@wp.kernel
def saxpy_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float), s: float, out: wp.array(dtype=float)):
    i = wp.tid()
    out[i] = a[i] + s * b[i]


@wp.kernel
def copy_scaled_kernel(src: wp.array(dtype=float), dst: wp.array(dtype=float), idx: int, s: float):
    dst[idx] = src[0] * s


def test_capture_replay_many_regions(test, device):
    """Regression for the O(1) region-resolution index in the CPU live-capture
    replay path (apic_resolve_state_region_ptr): a graph that registers many
    distinct array regions must resolve every one correctly on replay. Each of
    the `n` launches reads a distinct `src` array (a distinct region) and writes
    one slot of `out`, so replay must resolve `n` distinct regions; the outputs
    are clobbered after capture so only correct per-region resolution restores
    them. Guards the index build/lookup against off-by-one / stale-index bugs."""
    n = 300
    srcs = [wp.array([float(i + 1)], dtype=float, device=device) for i in range(n)]
    out = wp.zeros(n, dtype=float, device=device)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        for i in range(n):
            wp.launch(copy_scaled_kernel, dim=1, inputs=[srcs[i], out, i, 2.0], device=device)

    out.fill_(-1.0)  # clobber; only correct region resolution on replay restores it
    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    np.testing.assert_allclose(out.numpy(), 2.0 * (np.arange(n) + 1))

    # Replay again to exercise the cached (already-built) index path.
    out.fill_(-1.0)
    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    np.testing.assert_allclose(out.numpy(), 2.0 * (np.arange(n) + 1))


def test_capture_with_record_cmd_launch(test, device):
    """Regression: a reusable launch from ``wp.launch(..., record_cmd=True)``
    invoked during APIC capture must record an APIC_OP_KERNEL_LAUNCH (it
    previously called the kernel hook directly, bypassing the byte stream, so
    the launch was dropped on replay). The output is clobbered after capture, so
    only a replayed launch can restore it. Also covers ``set_param_at_index``."""
    n = 64
    a = wp.array(np.arange(n, dtype=np.float32), device=device)
    b = wp.array(np.full(n, 10.0, dtype=np.float32), device=device)
    out = wp.zeros(n, dtype=float, device=device)

    cmd = wp.launch(saxpy_kernel, dim=n, inputs=[a, b, 2.0], outputs=[out], record_cmd=True, device=device)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        cmd.launch()

    out.fill_(-999.0)  # clobber so a dropped launch leaves the sentinel
    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    np.testing.assert_allclose(out.numpy(), np.arange(n, dtype=np.float32) + 2.0 * 10.0)

    # set_param_at_index on a scalar must propagate into the recorded value blob.
    cmd.set_param_at_index(2, 3.0)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture2:
        cmd.launch()
    out.fill_(-999.0)
    wp.capture_launch(capture2.graph)
    wp.synchronize_device(device)
    np.testing.assert_allclose(out.numpy(), np.arange(n, dtype=np.float32) + 3.0 * 10.0)

    # set_param_at_index on an array arg must also propagate: the recorded
    # data-pointer relocation is derived from the retained fwd_args, so it must
    # target the updated array rather than the one passed at launch-record time.
    new_a = wp.array(np.arange(n, dtype=np.float32) + 100.0, device=device)
    cmd.set_param_at_index(0, new_a)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture3:
        cmd.launch()
    out.fill_(-999.0)
    wp.capture_launch(capture3.graph)
    wp.synchronize_device(device)
    np.testing.assert_allclose(out.numpy(), np.arange(n, dtype=np.float32) + 100.0 + 3.0 * 10.0)


def test_record_cmd_raw_array_ctype_rejected_during_apic_capture(test, device):
    """Raw ctypes array descriptors do not provide APIC relocation ownership."""
    n = 4
    a = wp.array(np.arange(n, dtype=np.float32), device=device)
    b = wp.ones(n, dtype=wp.float32, device=device)
    out = wp.zeros(n, dtype=float, device=device)

    cmd = wp.launch(saxpy_kernel, dim=n, inputs=[a, b, 2.0], outputs=[out], record_cmd=True, device=device)
    cmd.set_param_at_index_from_ctype(0, a.__ctype__())

    wp.load_module(device=device)
    with test.assertRaisesRegex(NotImplementedError, "raw array ctype"):
        with wp.ScopedCapture(device=device, apic=True, force_module_load=False):
            cmd.launch()


@wp.kernel
def fill_diag_triplets_kernel(
    rows: wp.array(dtype=wp.int32), columns: wp.array(dtype=wp.int32), values: wp.array(dtype=wp.float32)
):
    # Emit one diagonal triplet per row: (i, i) = i + 1.
    i = wp.tid()
    rows[i] = i
    columns[i] = i
    values[i] = float(i + 1)


def test_capture_with_bsr_from_triplets(test, device):
    """Regression: wp.sparse.bsr_set_from_triplets on CPU computes the matrix
    topology via a host function (wp_bsr_matrix_from_triplets_host) that, like the
    sorts/runlength_encode, was invisible to the APIC byte stream. The triplet
    arrays are produced by a kernel inside the capture (deferred at capture time),
    so without recording the op replay scatters fresh values into a frozen,
    capture-time-empty topology, leaving the matrix empty."""
    from warp.sparse import bsr_set_from_triplets, bsr_zeros  # noqa: PLC0415

    n = 8
    rows = wp.zeros(n, dtype=wp.int32, device=device)
    columns = wp.zeros(n, dtype=wp.int32, device=device)
    values = wp.zeros(n, dtype=wp.float32, device=device)
    A = bsr_zeros(n, n, block_type=wp.float32, device=device)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.launch(fill_diag_triplets_kernel, dim=n, inputs=[rows, columns, values], device=device)
        bsr_set_from_triplets(A, rows, columns, values)

    # Clobber topology + values so a no-op replay cannot leave capture-time data.
    A.offsets.zero_()
    A.columns.fill_(-1)
    A.values.zero_()

    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)

    # Expect an 8x8 diagonal matrix: nnz=8, columns=[0..7], values=[1..8].
    test.assertEqual(int(A.offsets.numpy()[n]), n)
    np.testing.assert_array_equal(A.columns.numpy()[:n], np.arange(n, dtype=np.int32))
    np.testing.assert_allclose(A.values.numpy()[:n], np.arange(1, n + 1, dtype=np.float32))


def test_capture_with_bsr_from_triplets_topology_only(test, device):
    """``bsr_set_from_triplets(values=None)`` still needs the recorded topology op."""
    from warp.sparse import bsr_set_from_triplets, bsr_zeros  # noqa: PLC0415

    rows = wp.array(np.array([1, 0, 1], dtype=np.int32), dtype=wp.int32, device=device)
    columns = wp.array(np.array([2, 1, 2], dtype=np.int32), dtype=wp.int32, device=device)
    A = bsr_zeros(2, 3, block_type=wp.float32, device=device)

    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        bsr_set_from_triplets(A, rows, columns, values=None)

    A.offsets.zero_()
    A.columns.fill_(-1)

    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)

    np.testing.assert_array_equal(A.offsets.numpy(), np.array([0, 1, 2], dtype=np.int32))
    np.testing.assert_array_equal(A.columns.numpy()[:2], np.array([1, 2], dtype=np.int32))


def test_capture_with_bsr_transpose(test, device):
    """Regression: wp.sparse.bsr_transposed on CPU computes the transposed topology
    via a host function (wp_bsr_transpose_host) that was invisible to the APIC byte
    stream. The source matrix is (re)assembled from triplets produced inside the
    capture, so replay must recompute both the from-triplets and the transpose
    topology."""
    from warp.sparse import bsr_set_from_triplets, bsr_set_transpose, bsr_zeros  # noqa: PLC0415

    n = 6
    rows = wp.zeros(n, dtype=wp.int32, device=device)
    columns = wp.zeros(n, dtype=wp.int32, device=device)
    values = wp.zeros(n, dtype=wp.float32, device=device)
    A = bsr_zeros(n, n, block_type=wp.float32, device=device)
    At = bsr_zeros(n, n, block_type=wp.float32, device=device)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.launch(fill_diag_triplets_kernel, dim=n, inputs=[rows, columns, values], device=device)
        bsr_set_from_triplets(A, rows, columns, values)
        bsr_set_transpose(At, A)

    At.offsets.zero_()
    At.columns.fill_(-1)
    At.values.zero_()

    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)

    # Diagonal matrix transposes to itself: nnz=6, columns=[0..5], values=[1..6].
    test.assertEqual(int(At.offsets.numpy()[n]), n)
    np.testing.assert_array_equal(At.columns.numpy()[:n], np.arange(n, dtype=np.int32))
    np.testing.assert_allclose(At.values.numpy()[:n], np.arange(1, n + 1, dtype=np.float32))


def test_capture_with_padded_bsr_transpose(test, device):
    """Padded BSR transpose replay must preserve row_counts and status pointers."""
    from warp.sparse import (  # noqa: PLC0415
        BSR_STATUS_ROW_CAPACITY_EXCEEDED,
        bsr_set_from_triplets,
        bsr_set_transpose,
        bsr_zeros,
    )

    rows = wp.array(np.array([0, 0, 1, 1], dtype=np.int32), dtype=wp.int32, device=device)
    columns = wp.array(np.array([0, 2, 1, 2], dtype=np.int32), dtype=wp.int32, device=device)
    values = wp.array(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), dtype=wp.float32, device=device)

    A = bsr_zeros(2, 3, block_type=wp.float32, device=device, row_capacity=3)
    bsr_set_from_triplets(A, rows, columns, values, topology="padded")

    At = bsr_zeros(3, 2, block_type=wp.float32, device=device, row_capacity=2)
    At_too_small = bsr_zeros(3, 2, block_type=wp.float32, device=device, row_capacity=1)

    wp.load_module(device=device)
    if device.is_cuda:
        # Specialize and load the internal values kernel before CUDA graph capture.
        bsr_set_transpose(At, A, topology="padded")
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        bsr_set_transpose(At, A, topology="padded")
        bsr_set_transpose(At_too_small, A, topology="padded")

    At.row_counts.zero_()
    At.columns.fill_(-1)
    At.values.zero_()
    At._ensure_status().zero_()
    At_too_small.row_counts.zero_()
    At_too_small.columns.fill_(-1)
    At_too_small.values.zero_()
    At_too_small._ensure_status().zero_()

    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)

    np.testing.assert_array_equal(At.row_counts.numpy(), np.array([1, 1, 2], dtype=np.int32))
    columns_np = At.columns.numpy()
    np.testing.assert_array_equal(columns_np[[0, 2, 4, 5]], np.array([0, 1, 0, 1], dtype=np.int32))
    values_np = At.values.numpy()
    np.testing.assert_allclose(values_np[[0, 2, 4, 5]], np.array([1.0, 3.0, 2.0, 4.0], dtype=np.float32))
    test.assertEqual(At.status_sync(), 0)

    np.testing.assert_array_equal(At_too_small.row_counts.numpy(), np.array([1, 1, 0], dtype=np.int32))
    test.assertEqual(At_too_small.status_sync(), BSR_STATUS_ROW_CAPACITY_EXCEEDED)


def test_capture_padded_bsr_transpose_rebuilds_offsets(test, device):
    """Regression: the padded BSR transpose reads the destination's
    row-capacity offsets but never writes them, so an APIC replay that found
    those offsets zeroed before capture_launch reconstructed nothing. The capture
    now restores the capacity layout (CPU from the recorded op, CUDA from a
    snapshot copy captured into the graph), so the captured graph rebuilds the
    destination even after every destination buffer is reset."""
    from warp.sparse import bsr_set_from_triplets, bsr_set_transpose, bsr_zeros  # noqa: PLC0415

    A = bsr_zeros(2, 2, block_type=wp.float32, device=device, row_capacity=2)
    bsr_set_from_triplets(
        A,
        rows=wp.array([0, 1], dtype=wp.int32, device=device),
        columns=wp.array([1, 0], dtype=wp.int32, device=device),
        values=wp.array([10.0, 20.0], dtype=wp.float32, device=device),
        topology="padded",
    )
    # Make inactive slack entries deterministic.
    A.columns.assign([1, 0, 0, 1])
    A.values.assign([10.0, 111.0, 20.0, 222.0])

    At = bsr_zeros(2, 2, block_type=wp.float32, device=device, row_capacity=2)

    wp.load_module(device=device)
    if device.is_cuda:
        # Specialize and load the internal values kernel before CUDA graph capture.
        bsr_set_transpose(At, A, topology="padded")
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        bsr_set_transpose(At, A, topology="padded")

    # Reset every destination buffer, including the row-capacity offsets.
    At.offsets.zero_()
    At.row_counts.zero_()
    At.columns.fill_(-1)
    At.values.zero_()

    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)

    np.testing.assert_array_equal(At.offsets.numpy(), np.array([0, 2, 4], dtype=np.int32))
    np.testing.assert_array_equal(At.row_counts.numpy(), np.array([1, 1], dtype=np.int32))
    np.testing.assert_array_equal(At.columns.numpy()[[0, 2]], np.array([1, 0], dtype=np.int32))


def test_save_load_padded_bsr_transpose_cuda_rebuild(test, device):
    """Save/load of a CUDA APIC padded bsr_set_transpose graph. The transpose
    allocates block-index scratch during capture (graph-scoped on a memory-pool
    device); capture_save stores that region's size only (its content regenerates
    on replay), and the rebuild (apic_replay_ops_into_cuda_capture) sizes the
    transposed-columns / block-index regions without dereferencing the device
    offsets on the host.

    A padded destination's offsets are the fixed row-capacity layout, which the
    transpose reads but never writes; a recorded device-to-device copy restores
    that layout before the transpose so replay is correct even if the destination
    offsets are stale (GH-1587). This test asserts the full transposed result
    (offsets, row_counts, and active columns/values) round-trips through save/load,
    and -- by zeroing the destination offsets before save so their content snapshot
    is no longer the capacity layout -- that the recorded restore copy alone
    re-establishes the layout on replay (the copy is the load-bearing mechanism)."""
    from warp.sparse import bsr_set_from_triplets, bsr_set_transpose, bsr_zeros  # noqa: PLC0415

    rows = wp.array(np.array([0, 0, 1, 1], dtype=np.int32), dtype=wp.int32, device=device)
    columns = wp.array(np.array([0, 2, 1, 2], dtype=np.int32), dtype=wp.int32, device=device)
    values = wp.array(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), dtype=wp.float32, device=device)
    A = bsr_zeros(2, 3, block_type=wp.float32, device=device, row_capacity=3)
    bsr_set_from_triplets(A, rows, columns, values, topology="padded")
    At = bsr_zeros(3, 2, block_type=wp.float32, device=device, row_capacity=2)

    wp.load_module(device=device)
    # Specialize and load the internal values kernel before CUDA graph capture.
    bsr_set_transpose(At, A, topology="padded")
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        bsr_set_transpose(At, A, topology="padded")

    # The 2x3 -> 3x2 padded transpose yields 3 rows at row_capacity 2 (6 slots);
    # row_counts [1, 1, 2] makes the active slots [0, 2, 4, 5] (offsets are the
    # fixed capacity layout [0, 2, 4, 6]). Inactive padding slots are not compared.
    expected_offsets = np.array([0, 2, 4, 6], dtype=np.int32)
    expected_row_counts = np.array([1, 1, 2], dtype=np.int32)
    active_slots = [0, 2, 4, 5]
    expected_columns = np.array([0, 1, 0, 1], dtype=np.int32)
    expected_values = np.array([1.0, 3.0, 2.0, 4.0], dtype=np.float32)
    outputs = {"offsets": At.offsets, "row_counts": At.row_counts, "columns": At.columns, "values": At.values}

    def assert_loaded_matches(loaded, label):
        off = wp.zeros_like(At.offsets)
        rc = wp.zeros_like(At.row_counts)
        cols = wp.zeros_like(At.columns)
        vals = wp.zeros_like(At.values)
        loaded.get_param("offsets", off)
        loaded.get_param("row_counts", rc)
        loaded.get_param("columns", cols)
        loaded.get_param("values", vals)
        wp.synchronize_device(device)
        np.testing.assert_array_equal(off.numpy(), expected_offsets, err_msg=f"{label}: offsets")
        np.testing.assert_array_equal(rc.numpy(), expected_row_counts, err_msg=f"{label}: row_counts")
        np.testing.assert_array_equal(
            cols.numpy().reshape(-1)[active_slots], expected_columns, err_msg=f"{label}: columns"
        )
        np.testing.assert_allclose(vals.numpy().reshape(-1)[active_slots], expected_values, err_msg=f"{label}: values")

    # (1) Normal save/load: the full transposed result must round-trip, and the
    # loaded graph must stay correct across repeated launches.
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "padded_transpose")
        wp.capture_save(capture.graph, path, outputs=outputs)
        loaded = wp.capture_load(path, device=device)
        for i in range(2):
            wp.capture_launch(loaded)
            wp.synchronize_device(device)
            assert_loaded_matches(loaded, f"normal save/load (launch {i + 1})")

    # (2) Stale-offsets save/load: zero the destination offsets before save so their
    # content snapshot is no longer the capacity layout. Correct replay then depends
    # solely on the recorded device-to-device capacity-restore copy (GH-1587) -- if
    # it were dropped from the byte stream, the transpose would read zeroed offsets
    # and produce wrong results.
    At.offsets.zero_()
    wp.synchronize_device(device)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "padded_transpose_stale")
        wp.capture_save(capture.graph, path, outputs=outputs)
        loaded = wp.capture_load(path, device=device)
        wp.capture_launch(loaded)
        wp.synchronize_device(device)
        assert_loaded_matches(loaded, "stale-offsets save/load")


def test_save_load_padded_bsr_transpose_too_small(test, device):
    """Save/load a padded transpose into a destination that is too small for the source.

    The destination's total block capacity (3) is smaller than the source's nnz
    upper bound (6), which is the capacity-overflow case the padded API reports
    through ``status``. The capture hooks used to claim the source bound for
    the destination-columns span, so the tracked region grew past the
    destination's real allocation: the CPU host snapshot read past the buffer
    and pointer resolution was corrupted for the rest of the capture (seen in
    CI as ``free(): invalid next size`` aborts). Record and replay now size
    that span from the destination capacity. The test asserts the exact-span
    record and resolve survive a save/load round-trip and that the overflow
    status is still reported.
    """
    rows = wp.array(np.array([0, 0, 1, 1], dtype=np.int32), dtype=wp.int32, device=device)
    columns = wp.array(np.array([0, 2, 1, 2], dtype=np.int32), dtype=wp.int32, device=device)
    values = wp.array(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), dtype=wp.float32, device=device)
    A = bsr_zeros(2, 3, block_type=wp.float32, device=device, row_capacity=3)
    bsr_set_from_triplets(A, rows, columns, values, topology="padded")

    # 3 destination rows at row_capacity 1: total capacity 3 < source nnz upper
    # bound 6, and destination row 2 receives 2 blocks so it overflows.
    At = bsr_zeros(3, 2, block_type=wp.float32, device=device, row_capacity=1)
    status = At._ensure_status()

    wp.load_module(device=device)
    if device.is_cuda:
        # Specialize and load the internal values kernel before CUDA graph capture.
        bsr_set_transpose(At, A, topology="padded")
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        bsr_set_transpose(At, A, topology="padded")

    # Reset every destination buffer so the loaded graph must rebuild them.
    At.row_counts.zero_()
    At.columns.fill_(-1)
    At.values.zero_()
    status.zero_()

    outputs = {"row_counts": At.row_counts, "columns": At.columns, "values": At.values, "status": status}
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "padded_transpose_too_small")
        wp.capture_save(capture.graph, path, outputs=outputs)
        loaded = wp.capture_load(path, device=device)
        for i in range(2):
            wp.capture_launch(loaded)

            rc = wp.zeros_like(At.row_counts)
            cols = wp.zeros_like(At.columns)
            vals = wp.zeros_like(At.values)
            st = wp.zeros_like(status)
            loaded.get_param("row_counts", rc)
            loaded.get_param("columns", cols)
            loaded.get_param("values", vals)
            loaded.get_param("status", st)

            label = f"launch {i + 1}"
            np.testing.assert_array_equal(
                rc.numpy(), np.array([1, 1, 0], dtype=np.int32), err_msg=f"{label}: row_counts"
            )
            np.testing.assert_array_equal(
                cols.numpy().reshape(-1)[:2], np.array([0, 1], dtype=np.int32), err_msg=f"{label}: columns"
            )
            np.testing.assert_allclose(
                vals.numpy().reshape(-1)[:2], np.array([1.0, 3.0], dtype=np.float32), err_msg=f"{label}: values"
            )
            test.assertEqual(st.numpy()[0], BSR_STATUS_ROW_CAPACITY_EXCEEDED, msg=f"{label}: status")


def test_apic_cpu_op_not_rejected_under_cuda_capture(test, device):
    """Device-scoping: the CPU-only ``NotImplementedError`` guards for
    non-contiguous ``fill_()`` / ``wp.copy()`` are scoped to a CPU APIC capture
    via ``apic_capture.device == array.device``. ``runtime._apic_capture`` is
    global, so a CUDA APIC capture active on ``device`` must not falsely reject
    an unrelated CPU op."""
    base = wp.zeros(8, dtype=wp.float32, device="cpu")
    indices = wp.array([0, 2, 4, 6], dtype=wp.int32, device="cpu")
    cpu_indexed = wp.indexedarray1d(base, [indices])
    cuda_arr = wp.zeros(4, dtype=wp.float32, device=device)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False):
        cuda_arr.fill_(2.0)  # keeps the captured CUDA graph non-empty
        # Before the device-scoping fix this raised NotImplementedError because
        # the active CUDA capture also set runtime._apic_capture.
        cpu_indexed.fill_(1.0)

    wp.synchronize_device(device)
    np.testing.assert_array_equal(base.numpy()[[0, 2, 4, 6]], np.ones(4, dtype=np.float32))


def test_apic_cpu_ops_scoped_to_capture_device(test, device):
    """Device-scoping: a CPU ``wp.launch`` / ``capture_if`` / ``capture_while``
    issued during a CUDA APIC capture must execute immediately on the host, not be
    recorded into the global ``runtime._apic_capture`` (which targets the CUDA device).
    Each guard checks ``runtime._apic_capture.device == device``; otherwise the CPU op
    is deferred into a CUDA capture that never replays host work, so its effect (the
    array write, the loop iterations) would not land. ``device`` here is the CUDA device."""
    cuda_arr = wp.zeros(4, dtype=wp.float32, device=device)
    cpu_launch_out = wp.zeros(4, dtype=wp.float32, device="cpu")
    cpu_if_out = wp.zeros(1, dtype=wp.float32, device="cpu")
    if_cond = wp.array([1], dtype=wp.int32, device="cpu")
    while_cond = wp.array([1], dtype=wp.int32, device="cpu")

    state = {"iters": 0}

    def on_true():
        cpu_if_out.fill_(5.0)

    def while_body():
        # A host loop calls this until while_cond is cleared; APIC recording would
        # instead run it exactly once. Clear the condition immediately (the fill_
        # must execute now, not be deferred) after three iterations.
        state["iters"] += 1
        if state["iters"] >= 3:
            while_cond.fill_(0)

    # Pre-load both modules so the in-capture CPU launch does not also exercise lazy
    # CPU compilation/loading, which is not the behavior under test.
    wp.load_module(device=device)
    wp.load_module(device="cpu")
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False):
        cuda_arr.fill_(2.0)  # keep the captured CUDA graph non-empty
        # All three target the CPU while the active APIC capture targets the CUDA device.
        wp.launch(write_value_kernel, dim=4, inputs=[cpu_launch_out, 3.0], device="cpu")
        wp.capture_if(if_cond, on_true)
        wp.capture_while(while_cond, while_body)

    # Each CPU op executed immediately on the host (not recorded into the CUDA capture),
    # so its effect is visible without replaying the captured graph.
    np.testing.assert_array_equal(cpu_launch_out.numpy(), np.full(4, 3.0, dtype=np.float32))
    np.testing.assert_array_equal(cpu_if_out.numpy(), np.full(1, 5.0, dtype=np.float32))
    test.assertEqual(state["iters"], 3)


def test_apic_capture_resume_rejects_finished_graph(test, device):
    """A CPU APIC graph keeps its capture object after capture_end()
    so it can be replayed. capture_resume() must refuse to restart recording on
    a finished (never-paused) graph rather than silently reopening it."""
    from warp._src.context import capture_resume  # noqa: PLC0415

    a = wp.zeros(4, dtype=wp.float32, device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        a.fill_(1.0)

    # The capture object is intentionally retained for replay.
    test.assertIsNotNone(capture.graph._apic_capture)
    with test.assertRaisesRegex(RuntimeError, "not paused"):
        capture_resume(capture.graph)

    # The finished graph still replays correctly.
    a.zero_()
    wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(a.numpy(), np.ones(4, dtype=np.float32))


def test_capture_with_runlength_encode(test, device):
    """Regression: wp.utils.runlength_encode on CPU dispatches to a host function
    (wp_runlength_encode_int_host) that, like the sorts, was invisible to the APIC
    byte stream and so didn't replay. The outputs are overwritten with sentinels
    after capture, so only a replayed encode can restore the correct runs; without
    recording the op, replay leaves the sentinels in place."""
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
    values = wp.array(np.array([1, 1, 2], dtype=np.int32), dtype=wp.int32, device=device)
    run_values = wp.zeros(3, dtype=wp.int32, device=device)
    run_lengths = wp.zeros(3, dtype=wp.int32, device=device)

    with test.assertRaises(NotImplementedError):
        with wp.ScopedCapture(device=device, apic=True, force_module_load=False):
            wp.utils.runlength_encode(values, run_values, run_lengths)

    with wp.ScopedCapture(device=device, apic=True, force_module_load=False):
        test.assertEqual(wp.utils.runlength_encode(values, run_values, run_lengths, value_count=0), 0)


def test_capture_auto_register_unknown_pointer_cpu(test, device):
    """``wp_memset_host`` against a pointer APIC's tracker has never seen
    previously dropped the op and emitted an error; the resulting graph
    replayed as a no-op and left the buffer non-zero. The recording hooks
    now auto-register unknown pointers as fresh regions so capture + replay
    completes.

    Bypass the high-level ``array.zero_()`` Python wrapper (which calls
    ``track_array`` first) and invoke ``wp_memset_host`` directly via the
    runtime ctypes binding to simulate a solver-owned buffer (e.g.
    mujoco-warp's internal ``mjw_data`` arrays) that never flows through
    Warp's tracking sites.
    """
    n = 64
    arr = wp.empty(n, dtype=wp.int32, device=device)
    arr.fill_(7)

    runtime = wp_context.runtime

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        runtime.core.wp_memset_host(ctypes.c_void_p(arr.ptr), ctypes.c_int(0), ctypes.c_size_t(arr.size * 4))

    # Re-prime with non-zero data, then replay; the captured memset must
    # zero the buffer back out (proves the op landed in the byte stream).
    arr.fill_(7)
    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    np.testing.assert_array_equal(arr.numpy(), np.zeros(n, dtype=np.int32))


def test_save_load_auto_registered_native_pointer_cpu(test, device):
    """Native-only auto-registered host regions must survive save/load.

    The scratch pointer below never flows through Python's APIC tracker. It is
    registered only by the native host hooks, then used as the source for a raw
    host-to-host memcpy into a tracked output. Saving the graph must serialize
    the scratch region too; otherwise loaded replay cannot resolve the source
    pointer.
    """
    n = 64
    output = wp.full(n, value=9, dtype=wp.uint8, device=device)
    runtime = wp_context.runtime

    scratch = runtime.core.wp_alloc_host(ctypes.c_size_t(n), None)
    test.assertTrue(scratch)
    try:
        # Seed the tail with data that must be copied as initial data when the
        # native region grows from n/2 bytes to n bytes during capture.
        runtime.core.wp_memset_host(ctypes.c_void_p(scratch), ctypes.c_int(5), ctypes.c_size_t(n))

        wp.load_module(device=device)
        with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
            runtime.core.wp_memset_host(ctypes.c_void_p(scratch), ctypes.c_int(0), ctypes.c_size_t(n // 2))
            runtime.core.wp_memcpy_h2h(ctypes.c_void_p(output.ptr), ctypes.c_void_p(scratch), ctypes.c_size_t(n))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "native_only_region")
            wp.capture_save(capture.graph, path, outputs={"output": output})

            # Prove loaded replay is independent of the original native pointer.
            runtime.core.wp_free_host(ctypes.c_void_p(scratch))
            scratch = None

            loaded = wp.capture_load(path, device=device)
            wp.capture_launch(loaded)
            wp.synchronize_device(device)

            result = wp.empty(n, dtype=wp.uint8, device=device)
            loaded.get_param("output", result)
            expected = np.concatenate((np.zeros(n // 2, dtype=np.uint8), np.full(n - n // 2, 5, dtype=np.uint8)))
            np.testing.assert_array_equal(result.numpy(), expected)
    finally:
        if scratch:
            runtime.core.wp_free_host(ctypes.c_void_p(scratch))


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
        # The host recording-state accessor is gated to CPU captures and the CUDA
        # accessor to CUDA captures; pick the one matching this capture's
        # device so this device-agnostic null-clobber check reads the active state.
        get_state = (
            core.wp_apic_get_recording_state if wp.get_device(device).is_cpu else core.wp_apic_get_cuda_recording_state
        )
        active_before = get_state()
        test.assertIsNotNone(active_before, "expected active recording state")
        core.wp_apic_end_recording(None)
        active_after = get_state()
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


def test_capture_save_aborts_on_mesh_registration_failure(test, device):
    """A device mesh whose arrays cannot be device-to-host snapshotted during
    capture_save must abort the save loudly rather than silently emit a ``.wrp``
    missing the mesh's data. A real ``cudaMemcpy`` failure is not deterministically
    inducible from Python, so this stubs the native mesh registration to report
    failure and asserts capture_save raises -- verifying the wp_apic_register_mesh
    return value is now checked."""
    n = 8
    a = wp.array(np.arange(n, dtype=np.float32), device=device)
    b = wp.zeros(n, dtype=float, device=device)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.launch(scale_kernel, dim=n, inputs=[a, b, 2.0], device=device)

    # Drive the mesh-registration loop in capture_save without building a real
    # mesh; the stub ignores the id and reports failure, so it is never dereferenced.
    capture.graph._apic_capture.collected_mesh_ids.add(1)

    with mock.patch.object(wp_context.runtime.core, "wp_apic_register_mesh", lambda *_args: False):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "mesh_snapshot_fail")
            with test.assertRaisesRegex(RuntimeError, "mesh"):
                wp.capture_save(capture.graph, path, outputs={"b": b})


def test_capture_save_aborts_on_region_snapshot_failure(test, device):
    """A device-region device-to-host snapshot failure during capture_save must
    abort the save (wp_apic_state_save returns false) instead of writing a ``.wrp``
    missing a referenced region's initial data. A real ``cudaMemcpy`` failure is not
    deterministically inducible from Python, so this stubs wp_apic_state_save to
    report failure and asserts capture_save raises."""
    n = 8
    a = wp.array(np.arange(n, dtype=np.float32), device=device)
    b = wp.zeros(n, dtype=float, device=device)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.launch(scale_kernel, dim=n, inputs=[a, b, 2.0], device=device)

    with mock.patch.object(wp_context.runtime.core, "wp_apic_state_save", lambda *_args: False):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "region_snapshot_fail")
            with test.assertRaisesRegex(RuntimeError, "Failed to save APIC graph"):
                wp.capture_save(capture.graph, path, outputs={"b": b})


devices = get_test_devices()
devices_with_graph_capture_allocation = get_test_devices_with_graph_capture_allocation()
devices_with_cuda_graph_module_load = get_test_devices_with_cuda_graph_module_load()
devices_with_graph_capture_allocation_and_cuda_graph_module_load = (
    get_test_devices_with_graph_capture_allocation_and_cuda_graph_module_load()
)

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
    devices=devices_with_graph_capture_allocation_and_cuda_graph_module_load,
)
add_function_test(
    TestApic,
    "test_multiple_internal_allocations",
    test_multiple_internal_allocations,
    devices=devices_with_graph_capture_allocation_and_cuda_graph_module_load,
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
add_function_test(
    TestApic, "test_save_load_alloc_only", test_save_load_alloc_only, devices=devices_with_graph_capture_allocation
)
add_function_test(
    TestApic,
    "test_apic_alloc_grow_during_capture",
    test_apic_alloc_grow_during_capture,
    devices=devices_with_graph_capture_allocation,
)
add_function_test(
    TestApic,
    "test_cpu_graph_alloc_not_leaked_on_relaunch",
    test_cpu_graph_alloc_not_leaked_on_relaunch,
    devices=[d for d in devices if d.is_cpu],
)
add_function_test(
    TestApic,
    "test_capture_pause_resume_allows_unrecorded_allocation",
    test_capture_pause_resume_allows_unrecorded_allocation,
    devices=devices_with_cuda_graph_module_load,
)
add_function_test(
    TestApic,
    "test_capture_pause_resume_suspends_apic_recording",
    test_capture_pause_resume_suspends_apic_recording,
    devices=devices_with_cuda_graph_module_load,
)
add_function_test(
    TestApic,
    "test_bsr_nnz_sync_during_cpu_apic_capture",
    test_bsr_nnz_sync_during_cpu_apic_capture,
    devices=[d for d in devices if d.is_cpu],
)
add_function_test(
    TestApic,
    "test_bsr_nnz_sync_after_recorded_topology_rejected",
    test_bsr_nnz_sync_after_recorded_topology_rejected,
    devices=[d for d in devices if d.is_cpu],
)
add_function_test(
    TestApic,
    "test_bsr_status_sync_during_cpu_apic_capture_rejected",
    test_bsr_status_sync_during_cpu_apic_capture_rejected,
    devices=[d for d in devices if d.is_cpu],
)
add_function_test(TestApic, "test_capture_with_empty_array_input", test_capture_with_empty_array_input, devices=devices)
add_function_test(
    TestApic,
    "test_capture_replay_with_tile_kernel_no_stack_overflow",
    test_capture_replay_with_tile_kernel_no_stack_overflow,
    devices=devices,
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
    devices=devices,
)
add_function_test(
    TestApic,
    "test_capture_2d_launch_minimal",
    test_capture_2d_launch_minimal,
    devices=devices,
)
add_function_test(
    TestApic, "test_capture_with_large_scalar_param", test_capture_with_large_scalar_param, devices=devices
)
add_function_test(TestApic, "test_capture_with_array_scan", test_capture_with_array_scan, devices=devices)
add_function_test(
    TestApic,
    "test_cpu_helper_not_recorded_during_cuda_capture",
    test_cpu_helper_not_recorded_during_cuda_capture,
    devices=get_cuda_test_devices(),
)
add_function_test(
    TestApic,
    "test_capture_with_array_scan_extended_metadata",
    test_capture_with_array_scan_extended_metadata,
    devices=devices,
)
add_function_test(
    TestApic,
    "test_capture_with_segmented_sort",
    test_capture_with_segmented_sort,
    devices=devices,
)
add_function_test(
    TestApic,
    "test_save_load_segmented_sort",
    test_save_load_segmented_sort,
    devices=devices,
)
add_function_test(
    TestApic,
    "test_save_load_segmented_sort_explicit_end",
    test_save_load_segmented_sort_explicit_end,
    devices=devices,
)
add_function_test(
    TestApic,
    "test_capture_with_radix_sort",
    test_capture_with_radix_sort,
    devices=devices,
)
add_function_test(
    TestApic,
    "test_capture_with_radix_sort_extended_metadata",
    test_capture_with_radix_sort_extended_metadata,
    devices=devices,
)
add_function_test(
    TestApic,
    "test_borrow_temporary_not_recycled_during_apic_capture",
    test_borrow_temporary_not_recycled_during_apic_capture,
    devices=devices_with_graph_capture_allocation,
)
add_function_test(
    TestApic,
    "test_capture_with_runlength_encode",
    test_capture_with_runlength_encode,
    devices=devices,
)
add_function_test(
    TestApic,
    "test_runlength_encode_host_return_rejected_during_cpu_apic_capture",
    test_runlength_encode_host_return_rejected_during_cpu_apic_capture,
    devices=[d for d in devices if d.is_cpu],
)
add_function_test(
    TestApic,
    "test_capture_replay_many_regions",
    test_capture_replay_many_regions,
    devices=devices,
)
add_function_test(
    TestApic,
    "test_capture_with_record_cmd_launch",
    test_capture_with_record_cmd_launch,
    devices=devices_with_cuda_graph_module_load,
)
add_function_test(
    TestApic,
    "test_record_cmd_raw_array_ctype_rejected_during_apic_capture",
    test_record_cmd_raw_array_ctype_rejected_during_apic_capture,
    devices=devices,
)
add_function_test(
    TestApic,
    "test_capture_with_bsr_from_triplets",
    test_capture_with_bsr_from_triplets,
    devices=[d for d in devices if d.is_cpu],
)
add_function_test(
    TestApic,
    "test_capture_with_bsr_from_triplets_topology_only",
    test_capture_with_bsr_from_triplets_topology_only,
    devices=[d for d in devices if d.is_cpu],
)
add_function_test(
    TestApic,
    "test_capture_with_bsr_transpose",
    test_capture_with_bsr_transpose,
    devices=[d for d in devices if d.is_cpu],
)
add_function_test(
    TestApic,
    "test_capture_with_padded_bsr_transpose",
    test_capture_with_padded_bsr_transpose,
    devices=devices,
)
add_function_test(
    TestApic,
    "test_capture_padded_bsr_transpose_rebuilds_offsets",
    test_capture_padded_bsr_transpose_rebuilds_offsets,
    devices=devices,
)
add_function_test(
    TestApic,
    "test_save_load_padded_bsr_transpose_cuda_rebuild",
    test_save_load_padded_bsr_transpose_cuda_rebuild,
    devices=[d for d in devices if d.is_cuda],
)
add_function_test(
    TestApic,
    "test_save_load_padded_bsr_transpose_too_small",
    test_save_load_padded_bsr_transpose_too_small,
    devices=devices,
)
add_function_test(
    TestApic,
    "test_apic_cpu_op_not_rejected_under_cuda_capture",
    test_apic_cpu_op_not_rejected_under_cuda_capture,
    devices=[d for d in devices if d.is_cuda],
)
add_function_test(
    TestApic,
    "test_apic_cpu_ops_scoped_to_capture_device",
    test_apic_cpu_ops_scoped_to_capture_device,
    devices=[d for d in devices if d.is_cuda],
)
add_function_test(
    TestApic,
    "test_save_load_capture_time_scratch_cuda",
    test_save_load_capture_time_scratch_cuda,
    devices=[d for d in devices if d.is_cuda],
)
add_function_test(
    TestApic,
    "test_apic_h2d_rejected_during_capture",
    test_apic_h2d_rejected_during_capture,
    devices=[d for d in devices if d.is_cuda],
)
add_function_test(
    TestApic,
    "test_apic_cuda_copy_gaps_rejected_during_capture",
    test_apic_cuda_copy_gaps_rejected_during_capture,
    devices=[d for d in devices if d.is_cuda],
)
add_function_test(
    TestApic,
    "test_apic_cuda_indexed_fill_rejected_during_capture",
    test_apic_cuda_indexed_fill_rejected_during_capture,
    devices=[d for d in devices if d.is_cuda],
)
add_function_test(
    TestApic,
    "test_apic_capture_while_body_raises_cleanup",
    test_apic_capture_while_body_raises_cleanup,
    devices=devices,
)
add_function_test(
    TestApic,
    "test_apic_capture_if_body_raises_cleanup",
    test_apic_capture_if_body_raises_cleanup,
    devices=devices,
)
add_function_test(
    TestApic,
    "test_apic_capture_resume_rejects_finished_graph",
    test_apic_capture_resume_rejects_finished_graph,
    devices=[d for d in devices if d.is_cpu],
)
add_function_test(
    TestApic,
    "test_save_load_array_scan_replay_with_updated_input",
    test_save_load_array_scan_replay_with_updated_input,
    devices=devices,
)
add_function_test(
    TestApic,
    "test_capture_auto_register_unknown_pointer_cpu",
    test_capture_auto_register_unknown_pointer_cpu,
    devices=[d for d in devices if d.is_cpu],
)
add_function_test(
    TestApic,
    "test_save_load_auto_registered_native_pointer_cpu",
    test_save_load_auto_registered_native_pointer_cpu,
    devices=[d for d in devices if d.is_cpu],
)
add_function_test(
    TestApic,
    "test_capture_backward_kernel",
    test_capture_backward_kernel,
    devices=devices,
)
add_function_test(
    TestApic,
    "test_capture_backward_consumes_y_grad",
    test_capture_backward_consumes_y_grad,
    devices=devices,
)
add_function_test(
    TestApic,
    "test_capture_struct_with_array",
    test_capture_struct_with_array,
    devices=devices,
)
add_function_test(
    TestApic,
    "test_capture_indexedarray",
    test_capture_indexedarray,
    devices=devices,
)
add_function_test(
    TestApic,
    "test_capture_indexedarray_adjoint_pack",
    test_capture_indexedarray_adjoint_pack,
    devices=devices,
)
add_function_test(
    TestApic,
    "test_capture_backward_retain_grad",
    test_capture_backward_retain_grad,
    devices=devices,
)
# Conditional / loop capture records APIC_OP_IF / APIC_OP_WHILE on both CPU
# (record-only) and CUDA (record-and-execute), and the loaded-graph rebuild
# reconstructs the conditional body sub-graphs on CUDA.
add_function_test(TestApic, "test_capture_if_cpu", test_capture_if_cpu, devices=devices)
add_function_test(TestApic, "test_capture_while_cpu", test_capture_while_cpu, devices=devices)
add_function_test(
    TestApic,
    "test_save_load_capture_if_cuda",
    test_save_load_capture_if_cuda,
    devices=[d for d in devices if d.is_cuda],
)
add_function_test(
    TestApic,
    "test_capture_distinct_modules_same_key",
    test_capture_distinct_modules_same_key,
    devices=devices,
)
add_function_test(
    TestApic,
    "test_save_load_distinct_modules_same_key",
    test_save_load_distinct_modules_same_key,
    devices=devices_with_cuda_graph_module_load,
)
add_function_test(
    TestApic,
    "test_end_recording_null_state_preserves_active",
    test_end_recording_null_state_preserves_active,
    devices=devices,
)
add_function_test(TestApic, "test_get_param_ptr", test_get_param_ptr, devices=devices_with_cuda_graph_module_load)
add_function_test(
    TestApic,
    "test_capture_save_aborts_on_mesh_registration_failure",
    test_capture_save_aborts_on_mesh_registration_failure,
    devices=devices_with_cuda_graph_module_load,
)
add_function_test(
    TestApic,
    "test_capture_save_aborts_on_region_snapshot_failure",
    test_capture_save_aborts_on_region_snapshot_failure,
    devices=devices_with_cuda_graph_module_load,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
