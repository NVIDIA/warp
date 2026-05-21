# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for graph capture and replay on CPU and CUDA devices."""

import ctypes
import enum
import gc
import time
import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import (
    add_function_test,
    assert_np_equal,
    get_cuda_device_pair_with_mempool_access_support,
    get_cuda_device_pair_with_peer_access_support,
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


@wp.kernel
def warmup_kernel(buf: wp.array(dtype=float)):
    tid = wp.tid()
    buf[tid] = buf[tid] + 0.0


@wp.kernel
def write_tid_kernel(ptr: wp.array(dtype=float)):
    tid = wp.tid()
    ptr[tid] = float(tid)


@wp.kernel
def slow_write_two_kernel(out: wp.array(dtype=float), spin: int):
    tid = wp.tid()
    s = float(0.0)
    for _ in range(spin):
        s = s + 1.0
    if s >= 0.0:
        out[tid] = 2.0


@wp.kernel
def copy_kernel(src: wp.array(dtype=float), dst: wp.array(dtype=float)):
    tid = wp.tid()
    dst[tid] = src[tid]


@wp.kernel
def accum_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float)):
    tid = wp.tid()
    a[tid] = a[tid] + b[tid]


graph_module_load_devices = get_test_devices_with_cuda_graph_module_load("all")


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
    np.testing.assert_allclose(output_arr.numpy(), expected)

    # Reset and replay
    output_arr.zero_()
    wp.capture_launch(capture.graph)
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
    np.testing.assert_allclose(d.numpy(), np.full(n, 40.0))

    # Reset and replay
    c.zero_()
    d.zero_()
    wp.capture_launch(capture.graph)
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

    np.testing.assert_allclose(output_arr.numpy(), np.full(n, 100.0))


def test_graph_memcpy(test, device):
    n = 256
    src = wp.array(np.arange(n, dtype=np.float32), device=device)
    dst = wp.zeros(n, dtype=float, device=device)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, force_module_load=False) as capture:
        wp.copy(dst, src)

    wp.capture_launch(capture.graph)
    np.testing.assert_allclose(dst.numpy(), src.numpy())

    # Replay copies from original src (pointers are baked into the graph)
    dst.zero_()
    wp.capture_launch(capture.graph)
    np.testing.assert_allclose(dst.numpy(), np.arange(n, dtype=np.float32))


def test_graph_memset(test, device):
    n = 256
    arr = wp.array(np.arange(n, dtype=np.float32), device=device)

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, force_module_load=False) as capture:
        arr.zero_()

    # Launch to execute the captured zero_()
    wp.capture_launch(capture.graph)
    np.testing.assert_allclose(arr.numpy(), np.zeros(n))

    # Fill with non-zero, then replay to zero again
    arr.fill_(1.0)
    np.testing.assert_allclose(arr.numpy(), np.ones(n))
    wp.capture_launch(capture.graph)
    np.testing.assert_allclose(arr.numpy(), np.zeros(n))


def test_graph_launch_verification_mode_checked_cuda_capture(test, device):
    n = 64
    input_arr = wp.array(np.arange(n, dtype=np.float32), device=device)
    output_arr = wp.zeros(n, dtype=float, device=device)

    wp.load_module(device=device)

    launch_verification_mode_saved = wp.config.launch_verification_mode
    wp.config.launch_verification_mode = wp.LaunchVerificationMode.CHECKED
    try:
        with wp.ScopedCapture(device=device, force_module_load=False) as capture:
            wp.launch(scale_kernel, dim=n, inputs=[input_arr, output_arr, 2.0], device=device)

        wp.capture_launch(capture.graph)
        np.testing.assert_allclose(output_arr.numpy(), np.arange(n, dtype=np.float32) * 2.0)
    finally:
        wp.config.launch_verification_mode = launch_verification_mode_saved


@unittest.skipUnless(
    get_cuda_device_pair_with_peer_access_support(graph_module_load_devices),
    "Requires devices with peer access and CUDA graph module-load support",
)
def test_graph_launch_verification_mode_checked_peer_access_cuda_capture(test, _):
    target_device, peer_device = get_cuda_device_pair_with_peer_access_support(graph_module_load_devices)
    n = 64
    with wp.ScopedMempool(target_device, False), wp.ScopedMempool(peer_device, False):
        input_arr = wp.array(np.arange(n, dtype=np.float32), device=target_device)
        output_arr = wp.zeros(n, dtype=float, device=peer_device)

    test.assertEqual(type(input_arr._allocator).__name__, "CudaDefaultAllocator")

    wp.load_module(device=peer_device)

    peer_access_saved = wp.is_peer_access_enabled(target_device, peer_device)
    launch_verification_mode_saved = wp.config.launch_verification_mode
    try:
        wp.set_peer_access_enabled(target_device, peer_device, True)
        test.assertTrue(wp.is_peer_access_enabled(target_device, peer_device))

        wp.config.launch_verification_mode = wp.LaunchVerificationMode.CHECKED
        # The peer graph reads input_arr from target_device; wait for its H2D initialization.
        wp.synchronize_device(target_device)
        with wp.ScopedCapture(device=peer_device, force_module_load=False) as capture:
            wp.launch(scale_kernel, dim=n, inputs=[input_arr, output_arr, 2.0], device=peer_device)

        wp.capture_launch(capture.graph)
        np.testing.assert_allclose(output_arr.numpy(), np.arange(n, dtype=np.float32) * 2.0)
    finally:
        wp.config.launch_verification_mode = launch_verification_mode_saved
        wp.set_peer_access_enabled(target_device, peer_device, peer_access_saved)


@unittest.skipUnless(
    get_cuda_device_pair_with_mempool_access_support(graph_module_load_devices),
    "Requires devices with mempool access and CUDA graph module-load support",
)
def test_graph_launch_verification_mode_checked_mempool_access_cuda_capture(test, _):
    target_device, peer_device = get_cuda_device_pair_with_mempool_access_support(graph_module_load_devices)
    n = 64
    with wp.ScopedMempool(target_device, True):
        input_arr = wp.array(np.arange(n, dtype=np.float32), device=target_device)
    output_arr = wp.zeros(n, dtype=float, device=peer_device)

    test.assertEqual(type(input_arr._allocator).__name__, "CudaMempoolAllocator")

    wp.load_module(device=peer_device)

    mempool_access_saved = wp.is_mempool_access_enabled(target_device, peer_device)
    launch_verification_mode_saved = wp.config.launch_verification_mode
    try:
        wp.set_mempool_access_enabled(target_device, peer_device, True)
        test.assertTrue(wp.is_mempool_access_enabled(target_device, peer_device))

        wp.config.launch_verification_mode = wp.LaunchVerificationMode.CHECKED
        # The peer graph reads input_arr from target_device; wait for its H2D initialization.
        wp.synchronize_device(target_device)
        with wp.ScopedCapture(device=peer_device, force_module_load=False) as capture:
            wp.launch(scale_kernel, dim=n, inputs=[input_arr, output_arr, 2.0], device=peer_device)

        wp.capture_launch(capture.graph)
        np.testing.assert_allclose(output_arr.numpy(), np.arange(n, dtype=np.float32) * 2.0)
    finally:
        wp.config.launch_verification_mode = launch_verification_mode_saved
        wp.set_mempool_access_enabled(target_device, peer_device, mempool_access_saved)


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
    expected = (np.arange(n, dtype=np.float32) + 1.0) * 3.0
    np.testing.assert_allclose(output_arr.numpy(), expected)

    # Replay — should produce same result
    output_arr.zero_()
    wp.capture_launch(capture.graph)
    np.testing.assert_allclose(output_arr.numpy(), expected)


############################################################
#
# CUDA-only tests
#
############################################################


def test_cuda_graph_alloc_free_preserves_merged_frontier(test, device):
    """In-capture free preserves cross-stream dependencies merged via ``wait_stream``."""
    n = 1 << 14
    spin = 1 << 24

    with wp.ScopedDevice(device):
        wp.load_module(device=device)

        stream_a = wp.get_stream()
        stream_b = wp.Stream()

        warmup = wp.zeros(n, dtype=float)
        out = wp.zeros(n, dtype=float)
        seen = wp.zeros(n, dtype=float)
        wp.synchronize_device()

        with wp.ScopedCapture(stream=stream_a, force_module_load=False) as capture:
            with wp.ScopedStream(stream_a):
                wp.launch(warmup_kernel, dim=n, inputs=[warmup])

            with wp.ScopedStream(stream_b):
                wp.wait_stream(stream_a)
                wp.launch(slow_write_two_kernel, dim=n, inputs=[out, spin])

            with wp.ScopedStream(stream_a):
                ptr = wp.empty(n, dtype=float)
                wp.launch(write_tid_kernel, dim=n, inputs=[ptr])
                wp.wait_stream(stream_b)

                del ptr
                gc.collect()

                wp.launch(copy_kernel, dim=n, inputs=[out, seen])

        out.zero_()
        seen.zero_()
        wp.synchronize_device()

        wp.capture_launch(capture.graph, stream=stream_a)
        wp.synchronize_device()

        np.testing.assert_array_equal(seen.numpy(), np.full(n, 2.0, dtype=np.float32))


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


def test_cuda_graph_alloc_transient_stream(test, device):

    def foo(n, num_launches):
        # use temporary side stream
        with wp.ScopedStream(wp.Stream()):
            a = wp.zeros(n, dtype=float)
            b = wp.ones(n, dtype=float)
            for _ in range(num_launches):
                wp.launch(accum_kernel, dim=a.size, inputs=[a, b])
        # Array b goes out of scope here and is freed. If the free runs on
        # an incorrect stream, the memory could be released prematurely.
        # Other streams that are allocating memory could then reuse the memory
        # while it is still used on this stream, leading to data corruption.
        return a

    def bar(n, c_h):
        # use temporary side stream
        with wp.ScopedStream(wp.Stream()):
            c = wp.empty(n, dtype=float)
            wp.copy(c, c_h)
        return c

    with wp.ScopedDevice(device):
        n = 64 * 1024 * 1024
        num_launches = 100
        fill_value = 42

        c_h = wp.full(n, fill_value, dtype=float, device="cpu", pinned=True)

        wp.load_module(device=device)
        with wp.ScopedCapture(force_module_load=False) as capture:
            a = foo(n, num_launches)
            c = bar(n, c_h)

        wp.capture_launch(capture.graph)

        assert_np_equal(a.numpy(), np.full(n, num_launches, dtype=np.float32))
        assert_np_equal(c.numpy(), np.full(n, fill_value, dtype=np.float32))


# CUDA graph topology tests.
#
# Ensure that stream management (ScopedStream, wait_stream(), ...) and graph memory
# management are working as expected in CUDA graphs.
# These tests probe the graph topology rather than numerical computation results.
# Numerical results are unreliable with multi-stream computation, because it's possible
# to get "lucky" even if stream synchronization issues are present. Instead of launching
# kernels, we manually insert graph nodes on different streams and query their dependencies.
#
# Graph allocations are particularly tricky. Warp relies on Python GC to deallocate
# memory, which can trigger at unexpected times inside or outside of graph capture.
# We can't rely on any particular CUDA stream for ordering during GC, but we must ensure
# that all streams where an allocation is used complete their work before freeing the memory.
# The functions ``wp_alloc_device_async()`` and ``wp_free_device_async()`` do the
# heavy lifting. In particular, ``wp_free_device_async()`` is responsible for serializing
# the allocation-dependent streams by inserting a memory free node at their junction.
# This is tricky business, as we want to avoid unnecessary serialization of independent
# streams.
#
# Normally, to exercise wp_alloc_device_async() and wp_free_device_async() we would
# write tests that create and delete Warp arrays. However, the graph topology tests
# need graph node pointers, which those functions don't return by default. We thus wrap
# them in the utility functions _insert_alloc() and _insert_free() that return the
# corresponding alloc and free nodes.
# _insert_alloc() is equivalent to ``arr = wp.empty(...)`` during graph capture.
# _insert_free() is equivalent to ``del arr`` during graph capture.
#
# Graph construction:
# - Use ScopedDevice, ScopedStream, wait_stream(), etc. as usual.
# - _insert_node() adds an empty graph node that can be used for dependency queries.
# - _insert_alloc() invokes wp_alloc_device_async() and returns the alloc node.
# - _insert_free(alloc) invokes wp_free_device_async() and returns the free node.
#
# Node dependency queries:
# - _depends_on(A, B) checks if node A is a descendent of node B.
#   This means that node A always executes after B when the graph is launched.
# - _nodes_independent(A, B) checks if nodes A and B are independent of each other.
#   This means that node A doesn't depend on B and node B doesn't depend on A, so
#   they may execute concurrently when the graph is launched.
#
# Allocation queries:
# - _alloc_available(alloc, node) checks if a node can access an allocation. It checks
#   that the query node depends on the alloc node and that the allocation isn't freed
#   before the query node executes.
# - _alloc_freed(alloc, node) checks if an allocation is freed by the time execution
#   reaches the query node.
# - _alloc_inaccessible(alloc, node) checks if the query node is independent of an
#   allocation. This could be because the allocation executes later in the graph or
#   it executes in an independent (concurrent) section of the graph.


def _insert_alloc(device=None):
    device = wp.get_device(device)
    node = wp._src.context.runtime.core.wp_cuda_graph_insert_alloc_node(device.context, 1)
    if not node:
        raise RuntimeError("Failed to insert alloc node")
    return node


def _insert_free(alloc_node, device=None):
    device = wp.get_device(device)
    node = wp._src.context.runtime.core.wp_cuda_graph_insert_free_node(device.context, alloc_node)
    if not node:
        raise RuntimeError("Failed to insert free node")
    return node


def _insert_node(device=None):
    device = wp.get_device(device)
    node = wp._src.context.runtime.core.wp_cuda_graph_insert_empty_node(device.context)
    if not node:
        raise RuntimeError("Failed to insert empty node")
    return node


def _loopify(func, a, b):
    if not hasattr(a, "__len__"):
        a = (a,)
    if not hasattr(b, "__len__"):
        b = (b,)
    for i in a:
        for j in b:
            if not func(i, j):
                return False
    return True


class NodeDependencyResult(enum.IntEnum):
    DEPENDENT = 0  # argument node depends on referent node
    INDEPENDENT = 1  # argument node does not depend on referent node
    ERROR = -1  # an error occurred


def _depends_on(argument, referent):
    def impl(a, b):
        result = wp._src.context.runtime.core.wp_cuda_graph_node_depends_on(a, b)
        assert result != NodeDependencyResult.ERROR
        return result == NodeDependencyResult.DEPENDENT

    return _loopify(impl, argument, referent)


def _nodes_independent(node1, node2):
    def impl(a, b):
        return not _depends_on(a, b) and not _depends_on(b, a)

    return _loopify(impl, node1, node2)


class GraphAllocQueryResult(enum.IntEnum):
    AVAILABLE = 0  # query node can safely access the alloc
    FREED = 1  # alloc is freed before query node is reached
    INACCESSIBLE = 2  # alloc is not accessible by query node
    ERROR = -1  # an error occurred
    USE_AFTER_FREE = -2  # query node depends on the alloc but the free is independent of the query node (this indicates a critical failure of wp_free_device_async())


def _alloc_available(alloc, node):
    def impl(a, b):
        result = wp._src.context.runtime.core.wp_cuda_graph_alloc_query(a, b)
        assert result >= 0
        return result == GraphAllocQueryResult.AVAILABLE

    return _loopify(impl, alloc, node)


def _alloc_inaccessible(alloc, node):
    def impl(a, b):
        result = wp._src.context.runtime.core.wp_cuda_graph_alloc_query(a, b)
        assert result >= 0
        return result == GraphAllocQueryResult.INACCESSIBLE

    return _loopify(impl, alloc, node)


def _alloc_freed(alloc, node):
    def impl(a, b):
        result = wp._src.context.runtime.core.wp_cuda_graph_alloc_query(a, b)
        assert result >= 0
        return result == GraphAllocQueryResult.FREED

    return _loopify(impl, alloc, node)


# show graph for debugging
def _show_graph(graph):
    import shutil  # noqa: PLC0415
    import subprocess  # noqa: PLC0415

    basename = "graph"
    dotname = f"{basename}.dot"
    imgname = f"{basename}.png"
    print(f"Writing {dotname}")
    wp.capture_debug_dot_print(graph, f"{basename}.dot", verbose=False)
    if shutil.which("dot"):
        print(f"Writing {imgname}")
        subprocess.run(
            ["dot", "-Tpng", f"{basename}.dot", f"-o{basename}.png"], capture_output=True, text=True, check=True
        )
        if shutil.which("xdg-open"):
            subprocess.run(["xdg-open", f"{basename}.png"], capture_output=True, text=True, check=True)


def test_cuda_graph_topo_alloc_sequential(test, device):
    """Test availability of sequential allocs."""

    # Expected topology:
    #
    #   alloc1
    #     |
    #   node1
    #     |
    #   alloc2
    #     |
    #   node2

    with wp.ScopedDevice(device):
        with wp.ScopedCapture(force_module_load=False) as capture:
            alloc1 = _insert_alloc()
            node1 = _insert_node()
            alloc2 = _insert_alloc()
            node2 = _insert_node()

        wp.capture_launch(capture.graph)

        assert _alloc_available(alloc1, (node1, node2))
        assert _alloc_available(alloc2, node2)
        assert _alloc_inaccessible(alloc2, node1)


def test_cuda_graph_topo_alloc_sequential_free(test, device):
    """Test sequential alloc/free operations."""

    # Expected topology:
    #
    #   alloc1
    #     |
    #   node1
    #     |
    #   free1
    #     |
    #   alloc2
    #     |
    #   node2
    #     |
    #   free2
    #     |
    #   node3

    with wp.ScopedDevice(device):
        with wp.ScopedCapture(force_module_load=False) as capture:
            alloc1 = _insert_alloc()
            node1 = _insert_node()
            free1 = _insert_free(alloc1)
            alloc2 = _insert_alloc()
            node2 = _insert_node()
            free2 = _insert_free(alloc2)
            node3 = _insert_node()

        wp.capture_launch(capture.graph)

        assert _alloc_available(alloc1, node1)
        assert _alloc_available(alloc2, node2)
        assert _alloc_freed(alloc1, alloc2)
        assert _alloc_freed(alloc2, node3)
        assert _depends_on(free1, node1)
        assert _depends_on(alloc2, free1)
        assert _depends_on(free2, node2)


def test_cuda_graph_topo_alloc_side_stream_independent(test, device):
    """Alloc on an unjoined side stream is independent from the parent."""

    # Expected topology:
    #
    #      node1
    #      /   \
    #   alloc node3
    #     |
    #   node2

    with wp.ScopedDevice(device):
        with wp.ScopedCapture(force_module_load=False) as capture:
            node1 = _insert_node()
            with wp.ScopedStream(wp.Stream()):
                alloc = _insert_alloc()
                node2 = _insert_node()
            node3 = _insert_node()

        wp.capture_launch(capture.graph)

        assert _alloc_available(alloc, node2)
        assert _alloc_inaccessible(alloc, (node1, node3))
        assert _nodes_independent(node3, (alloc, node2))


def test_cuda_graph_topo_alloc_side_stream_independent_free(test, device):
    """Alloc/free on an unjoined side stream doesn't serialize parent."""

    # Expected topology:
    #
    #      node1
    #      /   \
    #   alloc node3
    #     |
    #   node2
    #     |
    #   free

    with wp.ScopedDevice(device):
        stream1 = wp.Stream()
        with wp.ScopedCapture(force_module_load=False) as capture:
            node1 = _insert_node()
            with wp.ScopedStream(stream1):
                alloc = _insert_alloc()
                node2 = _insert_node()
                free = _insert_free(alloc)
            node3 = _insert_node()

        wp.capture_launch(capture.graph)

        assert _alloc_available(alloc, node2)
        assert _alloc_inaccessible(alloc, (node1, node3))
        assert _depends_on(free, node2)
        assert _nodes_independent(node3, (alloc, node2, free))


def test_cuda_graph_topo_alloc_side_stream_joined(test, device):
    """Joining a side stream exposes its alloc to the parent."""

    # Expected topology:
    #
    #      node1
    #      /   \
    #   alloc   |
    #     |     |
    #   node2   |
    #      \   /
    #      node3

    with wp.ScopedDevice(device) as device:
        stream0 = device.stream
        stream1 = wp.Stream()
        with wp.ScopedCapture(force_module_load=False) as capture:
            node1 = _insert_node()
            with wp.ScopedStream(stream1):
                alloc = _insert_alloc()
                node2 = _insert_node()
            stream0.wait_stream(stream1)
            node3 = _insert_node()

        wp.capture_launch(capture.graph)

        assert _alloc_available(alloc, (node2, node3))
        assert _alloc_inaccessible(alloc, node1)
        assert _depends_on(node3, (node1, node2))


def test_cuda_graph_topo_alloc_fork(test, device):
    """A forked side stream inherits allocs from before the fork, but not from after the fork."""

    # Expected topology:
    #
    #      alloc1
    #        |
    #      node1
    #       / \
    #      /   \
    #  alloc2 node2
    #    |      |
    #  node3  node4

    with wp.ScopedDevice(device):
        stream1 = wp.Stream()
        with wp.ScopedCapture(force_module_load=False) as capture:
            alloc1 = _insert_alloc()
            node1 = _insert_node()
            # fork
            with wp.ScopedStream(stream1):
                node2 = _insert_node()
            alloc2 = _insert_alloc()
            node3 = _insert_node()
            # no fork
            with wp.ScopedStream(stream1, sync_enter=False):
                node4 = _insert_node()

        wp.capture_launch(capture.graph)

        assert _alloc_available(alloc1, (node1, node2, node3, node4))
        assert _alloc_available(alloc2, node3)
        assert _alloc_inaccessible(alloc2, (node1, node2, node4))
        assert _nodes_independent((alloc2, node3), (node2, node4))


def test_cuda_graph_topo_alloc_fork_free_on_main(test, device):
    """Forked allocs freed on main stream without sync."""

    # Expected topology:
    #
    #      alloc1
    #        |
    #      node1
    #       / \
    #      /   \
    #  alloc2 node2
    #    |      |
    #  node3    |
    #    |      |
    #  free2  node4
    #     \    /
    #      \  /
    #      free1

    with wp.ScopedDevice(device):
        stream1 = wp.Stream()
        with wp.ScopedCapture(force_module_load=False) as capture:
            alloc1 = _insert_alloc()
            node1 = _insert_node()
            # fork
            with wp.ScopedStream(stream1):
                node2 = _insert_node()
            alloc2 = _insert_alloc()
            node3 = _insert_node()
            # no fork
            with wp.ScopedStream(stream1, sync_enter=False):
                node4 = _insert_node()

            # free2 should not serialize stream1
            free2 = _insert_free(alloc2)
            # free1 should serialize stream1
            free1 = _insert_free(alloc1)

        wp.capture_launch(capture.graph)

        assert _alloc_available(alloc1, (node1, node2, node3, node4))
        assert _alloc_available(alloc2, node3)
        assert _alloc_inaccessible(alloc2, (node1, node2, node4))
        assert _nodes_independent((alloc2, node3, free2), (node2, node4))
        assert _depends_on(free2, node3)
        assert _depends_on(free1, (node1, node2, node3, node4))


def test_cuda_graph_topo_alloc_fork_free_on_side(test, device):
    """Forked allocs freed on side stream without sync."""

    # Expected topology:
    #
    #      alloc1
    #        |
    #      node1
    #       / \
    #      /   \
    #  alloc2 node2
    #    |      |
    #  node3    |
    #    |      |
    #  free2  node4
    #     \    /
    #      \  /
    #      free1

    with wp.ScopedDevice(device):
        stream1 = wp.Stream()
        with wp.ScopedCapture(force_module_load=False) as capture:
            alloc1 = _insert_alloc()
            node1 = _insert_node()
            # fork
            with wp.ScopedStream(stream1):
                node2 = _insert_node()
            alloc2 = _insert_alloc()
            node3 = _insert_node()
            # no fork
            with wp.ScopedStream(stream1, sync_enter=False):
                node4 = _insert_node()

                # free2 should not serialize stream1
                free2 = _insert_free(alloc2)
                # free1 should serialize stream1
                free1 = _insert_free(alloc1)

        wp.capture_launch(capture.graph)

        assert _alloc_available(alloc1, (node1, node2, node3, node4))
        assert _alloc_available(alloc2, node3)
        assert _alloc_inaccessible(alloc2, (node1, node2, node4))
        assert _nodes_independent((alloc2, node3, free2), (node2, node4))
        assert _depends_on(free2, node3)
        assert _depends_on(free1, (node1, node2, node3, node4))


def test_cuda_graph_topo_alloc_parallel_streams(test, device):
    """Parallel side streams' allocs are mutually independent."""

    # Expected topology:
    #
    #   alloc1   alloc2
    #     |        |
    #   node1    node2

    with wp.ScopedDevice(device):
        stream1 = wp.Stream()
        stream2 = wp.Stream()
        with wp.ScopedCapture(force_module_load=False) as capture:
            with wp.ScopedStream(stream1):
                alloc1 = _insert_alloc()
                node1 = _insert_node()
            with wp.ScopedStream(stream2):
                alloc2 = _insert_alloc()
                node2 = _insert_node()

        wp.capture_launch(capture.graph)

        assert _alloc_available(alloc1, node1)
        assert _alloc_available(alloc2, node2)
        assert _alloc_inaccessible(alloc1, node2)
        assert _alloc_inaccessible(alloc2, node1)
        assert _nodes_independent((alloc1, node1), (alloc2, node2))


def test_cuda_graph_topo_alloc_parallel_streams_free_on_sides(test, device):
    """Freeing side stream allocs does not serialize independent streams"""

    # Expected topology:
    #
    #   alloc1   alloc2
    #     |        |
    #   node1    node2
    #     |        |
    #   free1    free2

    with wp.ScopedDevice(device):
        stream1 = wp.Stream()
        stream2 = wp.Stream()
        with wp.ScopedCapture(force_module_load=False) as capture:
            with wp.ScopedStream(stream1):
                alloc1 = _insert_alloc()
                node1 = _insert_node()
                free1 = _insert_free(alloc1)
            with wp.ScopedStream(stream2):
                alloc2 = _insert_alloc()
                node2 = _insert_node()
                free2 = _insert_free(alloc2)

        wp.capture_launch(capture.graph)

        assert _alloc_available(alloc1, node1)
        assert _alloc_available(alloc2, node2)
        assert _alloc_inaccessible(alloc1, node2)
        assert _alloc_inaccessible(alloc2, node1)
        assert _nodes_independent((alloc1, node1, free1), (alloc2, node2, free2))
        assert _depends_on(free1, node1)
        assert _depends_on(free2, node2)


def test_cuda_graph_topo_alloc_parallel_streams_free_on_main(test, device):
    """Freeing side stream allocs does not serialize independent streams"""

    # Expected topology:
    #
    #   alloc1   alloc2
    #     |        |
    #   node1    node2
    #     |        |
    #   free1    free2

    with wp.ScopedDevice(device):
        stream1 = wp.Stream()
        stream2 = wp.Stream()
        with wp.ScopedCapture(force_module_load=False) as capture:
            with wp.ScopedStream(stream1):
                alloc1 = _insert_alloc()
                node1 = _insert_node()
            with wp.ScopedStream(stream2):
                alloc2 = _insert_alloc()
                node2 = _insert_node()
            free1 = _insert_free(alloc1)
            free2 = _insert_free(alloc2)

        wp.capture_launch(capture.graph)

        assert _alloc_available(alloc1, node1)
        assert _alloc_available(alloc2, node2)
        assert _alloc_inaccessible(alloc1, node2)
        assert _alloc_inaccessible(alloc2, node1)
        assert _nodes_independent((alloc1, node1, free1), (alloc2, node2, free2))
        assert _depends_on(free1, node1)
        assert _depends_on(free2, node2)


def test_cuda_graph_topo_alloc_parallel_streams_free_on_other(test, device):
    """Freeing side stream allocs does not serialize independent streams"""

    # Expected topology:
    #
    #   alloc1   alloc2
    #     |        |
    #   node1    node2
    #     |        |
    #   free1    free2

    with wp.ScopedDevice(device):
        stream1 = wp.Stream()
        stream2 = wp.Stream()
        stream3 = wp.Stream()
        with wp.ScopedCapture(force_module_load=False) as capture:
            with wp.ScopedStream(stream1):
                alloc1 = _insert_alloc()
                node1 = _insert_node()
            with wp.ScopedStream(stream2):
                alloc2 = _insert_alloc()
                node2 = _insert_node()

            # stream3 is not part of the capture
            with wp.ScopedStream(stream3, sync_enter=False):
                free1 = _insert_free(alloc1)
                free2 = _insert_free(alloc2)

        wp.capture_launch(capture.graph)

        assert _alloc_available(alloc1, node1)
        assert _alloc_available(alloc2, node2)
        assert _alloc_inaccessible(alloc1, node2)
        assert _alloc_inaccessible(alloc2, node1)
        assert _nodes_independent((alloc1, node1, free1), (alloc2, node2, free2))
        assert _depends_on(free1, node1)
        assert _depends_on(free2, node2)


def test_cuda_graph_topo_alloc_parallel_streams_joined(test, device):
    """Joining parallel side streams exposes both allocs to the parent."""

    # Expected topology:
    #
    #   alloc1  alloc2
    #     |       |
    #    node1  node2  node3
    #        \    |    /
    #         \   |   /
    #           node4
    #             |
    #           free1
    #             |
    #           free2
    #             |
    #           node5

    with wp.ScopedDevice(device) as device:
        stream0 = device.stream
        stream1 = wp.Stream()
        stream2 = wp.Stream()
        with wp.ScopedCapture(force_module_load=False) as capture:
            with wp.ScopedStream(stream1):
                alloc1 = _insert_alloc()
                node1 = _insert_node()
            with wp.ScopedStream(stream2):
                alloc2 = _insert_alloc()
                node2 = _insert_node()
            node3 = _insert_node()

            stream0.wait_stream(stream1)
            stream0.wait_stream(stream2)
            node4 = _insert_node()

            free1 = _insert_free(alloc1)
            free2 = _insert_free(alloc2)
            node5 = _insert_node()

        wp.capture_launch(capture.graph)

        assert _alloc_available(alloc1, node1)
        assert _alloc_available(alloc2, node2)
        assert _alloc_inaccessible(alloc1, node2)
        assert _alloc_inaccessible(alloc2, node1)
        assert _alloc_inaccessible((alloc1, alloc2), node3)
        assert _alloc_available((alloc1, alloc2), node4)
        assert _alloc_freed((alloc1, alloc2), node5)
        assert _nodes_independent((alloc1, node1), (alloc2, node2))
        assert _nodes_independent(node3, (alloc1, node1, alloc2, node2))
        assert _depends_on(free1, (node1, node2, node3, node4))
        assert _depends_on(free2, (node1, node2, node3, node4))


def test_cuda_graph_topo_alloc_nested_streams_chain(test, device):
    """Nested ``ScopedStream`` blocks chain alloc visibility."""

    # Expected topology:
    #
    #   alloc1
    #     |
    #   node1
    #     |
    #   alloc2
    #     |
    #   node2
    #     |
    #   alloc3
    #     |
    #   node3

    with wp.ScopedDevice(device):
        stream1 = wp.Stream()
        stream2 = wp.Stream()
        with wp.ScopedCapture(force_module_load=False) as capture:
            alloc1 = _insert_alloc()
            node1 = _insert_node()
            with wp.ScopedStream(stream1):
                alloc2 = _insert_alloc()
                node2 = _insert_node()
                with wp.ScopedStream(stream2):
                    alloc3 = _insert_alloc()
                    node3 = _insert_node()

        wp.capture_launch(capture.graph)

        assert _alloc_available(alloc1, (node1, node2, node3))
        assert _alloc_available(alloc2, (node2, node3))
        assert _alloc_available(alloc3, node3)
        assert _alloc_inaccessible(alloc2, node1)
        assert _alloc_inaccessible(alloc3, (node1, node2))


def test_cuda_graph_topo_alloc_nested_streams_chain_free(test, device):
    """Nested ``ScopedStream`` blocks chain alloc visibility."""

    # Expected topology:
    #
    #   alloc1
    #     |
    #   node1
    #     |
    #   alloc2
    #     |
    #   node2
    #     |
    #   alloc3
    #     |
    #   node3
    #     |
    #   free3
    #     |
    #   free2
    #     |
    #   free1

    with wp.ScopedDevice(device):
        stream1 = wp.Stream()
        stream2 = wp.Stream()
        with wp.ScopedCapture(force_module_load=False) as capture:
            alloc1 = _insert_alloc()
            node1 = _insert_node()
            with wp.ScopedStream(stream1):
                alloc2 = _insert_alloc()
                node2 = _insert_node()
                with wp.ScopedStream(stream2):
                    alloc3 = _insert_alloc()
                    node3 = _insert_node()
                    free3 = _insert_free(alloc3)
                free2 = _insert_free(alloc2)
            free1 = _insert_free(alloc1)

        wp.capture_launch(capture.graph)

        assert _alloc_available(alloc1, (node1, node2, node3))
        assert _alloc_available(alloc2, (node2, node3))
        assert _alloc_available(alloc3, node3)
        assert _alloc_inaccessible(alloc2, node1)
        assert _alloc_inaccessible(alloc3, (node1, node2))
        assert _depends_on((free1, free2, free3), (node1, node2, node3))


def test_cuda_graph_topo_alloc_free_serializes_dependent_streams_only(test, device):
    """Free serializes streams that depend on the allocation, independent streams are not affected."""

    # Expected topology:
    #
    #   node1  node2     alloc
    #     |      |        / \
    #     |      |       /   \
    #   node5  node6  node3 node4
    #                    \   /
    #                     \ /
    #                    free
    #                     / \
    #                    /   \
    #                 node7 node8

    with wp.ScopedDevice(device) as device:
        stream1 = wp.Stream()
        stream2 = wp.Stream()
        stream3 = wp.Stream()
        stream4 = wp.Stream()
        with wp.ScopedCapture(force_module_load=False) as capture:
            # stream1 and stream2 are independent of the alloc
            with wp.ScopedStream(stream1):
                node1 = _insert_node()
            with wp.ScopedStream(stream2):
                node2 = _insert_node()

            alloc = _insert_alloc()

            # stream3 and stream4 are dependent on the alloc
            with wp.ScopedStream(stream3):
                node3 = _insert_node()
            with wp.ScopedStream(stream4):
                node4 = _insert_node()

            free = _insert_free(alloc)

            with wp.ScopedStream(stream1, sync_enter=False):
                node5 = _insert_node()
            with wp.ScopedStream(stream2, sync_enter=False):
                node6 = _insert_node()

            with wp.ScopedStream(stream3, sync_enter=False):
                node7 = _insert_node()
            with wp.ScopedStream(stream4, sync_enter=False):
                node8 = _insert_node()

        wp.capture_launch(capture.graph)

        assert _alloc_available(alloc, (node3, node4))
        assert _alloc_inaccessible(alloc, (node1, node2))
        assert _alloc_inaccessible(alloc, (node5, node6))
        assert _alloc_freed(alloc, (node7, node8))
        assert _depends_on(free, (node3, node4))
        assert _nodes_independent((node1, node5), (node2, node6))
        assert _nodes_independent((node1, node5), (alloc, node3, node4, free, node7, node8))
        assert _nodes_independent((node2, node6), (alloc, node3, node4, free, node7, node8))
        assert _nodes_independent(node3, node4)
        assert _nodes_independent(node7, node8)


devices = get_test_devices()
devices_with_cuda_graph_module_load = get_test_devices_with_cuda_graph_module_load()
devices_with_mempool_and_cuda_graph_module_load = get_test_devices_with_mempool_and_cuda_graph_module_load()
cuda_devices_with_cuda_graph_module_load = [device for device in devices_with_cuda_graph_module_load if device.is_cuda]

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
    "test_graph_launch_verification_mode_checked_cuda_capture",
    test_graph_launch_verification_mode_checked_cuda_capture,
    devices=cuda_devices_with_cuda_graph_module_load,
)
add_function_test(
    TestGraph,
    "test_graph_launch_verification_mode_checked_peer_access_cuda_capture",
    test_graph_launch_verification_mode_checked_peer_access_cuda_capture,
)
add_function_test(
    TestGraph,
    "test_graph_launch_verification_mode_checked_mempool_access_cuda_capture",
    test_graph_launch_verification_mode_checked_mempool_access_cuda_capture,
)
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
    "test_cuda_graph_alloc_free_preserves_merged_frontier",
    test_cuda_graph_alloc_free_preserves_merged_frontier,
    devices=cuda_devices,
)
add_function_test(
    TestGraph,
    "test_cuda_graph_alloc_retained_release",
    test_cuda_graph_alloc_retained_release,
    devices=cuda_devices,
)
add_function_test(
    TestGraph,
    "test_cuda_graph_alloc_transient_stream",
    test_cuda_graph_alloc_transient_stream,
    devices=cuda_devices,
)

# CUDA graph topology tests.
_topo_tests = [
    test_cuda_graph_topo_alloc_sequential,
    test_cuda_graph_topo_alloc_sequential_free,
    test_cuda_graph_topo_alloc_side_stream_independent,
    test_cuda_graph_topo_alloc_side_stream_independent_free,
    test_cuda_graph_topo_alloc_side_stream_joined,
    test_cuda_graph_topo_alloc_fork,
    test_cuda_graph_topo_alloc_fork_free_on_main,
    test_cuda_graph_topo_alloc_fork_free_on_side,
    test_cuda_graph_topo_alloc_parallel_streams,
    test_cuda_graph_topo_alloc_parallel_streams_free_on_sides,
    test_cuda_graph_topo_alloc_parallel_streams_free_on_main,
    test_cuda_graph_topo_alloc_parallel_streams_free_on_other,
    test_cuda_graph_topo_alloc_parallel_streams_joined,
    test_cuda_graph_topo_alloc_nested_streams_chain,
    test_cuda_graph_topo_alloc_nested_streams_chain_free,
    test_cuda_graph_topo_alloc_free_serializes_dependent_streams_only,
]
for _topo_test in _topo_tests:
    add_function_test(TestGraph, _topo_test.__name__, _topo_test, devices=cuda_devices)

if __name__ == "__main__":
    unittest.main(verbosity=2)
