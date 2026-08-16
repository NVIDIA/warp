# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-capture ``wp.array`` deallocation against a shared ``cudaGraph_t``.

Driver-level setup (CUDASTF-style sequential captures into a single graph)::

    G = cudaGraphCreate()
    cudaStreamBeginCaptureToGraph(streamA, G, ...)
    wp.capture_begin(stream=streamA, external=True)
    a = wp.empty(...)  # MEM_ALLOC in G, alloc.capture_id = id_A
    wp.launch(consumer_kernel, ..., stream=streamA)
    wp.capture_end(stream=streamA)  # g_captures.erase(id_A); a still in g_graph_allocs
    cudaStreamEndCapture(streamA, ...)

    cudaStreamBeginCaptureToGraph(streamB, G, ...)
    wp.capture_begin(stream=streamB, external=True)
    del a  # wp_free_device_async on streamB's frontier
    wp.capture_end(stream=streamB)
    cudaStreamEndCapture(streamB, ...)

The driver assigns the **same** capture id to both sequential captures into the
shared graph. As a result, when ``wp_free_device_async`` looks up
``g_captures.find(alloc.capture_id)`` during ``del a`` (or a ``gc.collect()``
trigger of the same finalizer), it finds *capture B*'s entry (not capture A's,
which was erased at ``wp.capture_end`` for ``streamA``). The function then takes
the "capture still active" branch and adds a ``MEM_FREE`` node to ``G``, wired
against the calling stream's frontier (``streamB``).

That frontier contains no edge back to the consumer kernel in capture A, so the
resulting ``MEM_FREE`` node has *no* ``KERNEL`` ancestor. At replay, the runtime
is free to schedule the ``MEM_FREE`` before the consumer kernel completes; once
the mempool recycles the VA the kernel writes into a freed page and
``cudaErrorIllegalAddress`` surfaces from the next CUDA call. This is the same
class of bug as an unordered free on a forked substream within a single
capture, but reached through the shared-graph pattern that external
integrations (e.g. CUDASTF) use for sequential captures into one graph.

This test asserts the correctness invariant directly: every ``MEM_FREE`` node
produced by Warp during this scenario must have a ``KERNEL`` ancestor in its
predecessor closure. The captured topology is sensitive to the CUDA toolkit
and driver in use, so this serves as a regression guard for the invariant
across the toolkits exercised by CI.
"""

import ctypes
import ctypes.util
import gc
import unittest

import warp as wp
from warp.tests.unittest_utils import *

N = 4096

_CUDA_GRAPH_NODE_TYPE_KERNEL = 0x00
_CUDA_GRAPH_NODE_TYPE_MEM_ALLOC = 0x0A
_CUDA_GRAPH_NODE_TYPE_MEM_FREE = 0x0B

_CUDA_STREAM_CAPTURE_MODE_RELAXED = 2


@wp.kernel
def _touch(x: wp.array[wp.float32]):
    i = wp.tid()
    if i < x.shape[0]:
        x[i] = x[i] + 1.0


def _load_cudart():
    """Bind the CUDA runtime entry points we need to drive shared captures + walk the graph."""
    name = ctypes.util.find_library("cudart")
    if name is None:
        return None
    try:
        lib = ctypes.CDLL(name)
    except OSError:
        return None

    lib.cudaGetErrorString.restype = ctypes.c_char_p
    lib.cudaGetErrorString.argtypes = [ctypes.c_int]

    lib.cudaRuntimeGetVersion.restype = ctypes.c_int
    lib.cudaRuntimeGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]

    ver = ctypes.c_int(0)
    if lib.cudaRuntimeGetVersion(ctypes.byref(ver)) != 0:
        return None
    if ver.value < 12030:
        # cudaStreamBeginCaptureToGraph was introduced in CTK 12.3.
        return None
    lib._runtime_version = ver.value

    lib.cudaGraphCreate.restype = ctypes.c_int
    lib.cudaGraphCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]

    lib.cudaGraphDestroy.restype = ctypes.c_int
    lib.cudaGraphDestroy.argtypes = [ctypes.c_void_p]

    lib.cudaStreamBeginCaptureToGraph.restype = ctypes.c_int
    lib.cudaStreamBeginCaptureToGraph.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]

    lib.cudaStreamEndCapture.restype = ctypes.c_int
    lib.cudaStreamEndCapture.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]

    lib.cudaGraphGetNodes.restype = ctypes.c_int
    lib.cudaGraphGetNodes.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t),
    ]

    lib.cudaGraphNodeGetType.restype = ctypes.c_int
    lib.cudaGraphNodeGetType.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]

    lib.cudaGraphGetEdges.restype = ctypes.c_int
    if ver.value >= 13000:
        # CTK 13+: cudaGraphGetEdges(graph, from, to, edgeData, numEdges)
        lib.cudaGraphGetEdges.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
        ]
    else:
        # CTK 12 and earlier: cudaGraphGetEdges(graph, from, to, numEdges)
        lib.cudaGraphGetEdges.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
        ]

    return lib


CUDART = _load_cudart()


def _check(rt, rc, where):
    if rc != 0:
        msg = rt.cudaGetErrorString(rc).decode(errors="replace")
        raise RuntimeError(f"{where}: CUDA error {rc}: {msg}")


def _graph_nodes(rt, graph):
    n = ctypes.c_size_t(0)
    _check(rt, rt.cudaGraphGetNodes(graph, None, ctypes.byref(n)), "cudaGraphGetNodes(count)")
    if n.value == 0:
        return []
    arr = (ctypes.c_void_p * n.value)()
    _check(rt, rt.cudaGraphGetNodes(graph, arr, ctypes.byref(n)), "cudaGraphGetNodes(fill)")
    return [ctypes.c_void_p(arr[i]) for i in range(n.value)]


def _node_type(rt, node):
    t = ctypes.c_int(-1)
    _check(rt, rt.cudaGraphNodeGetType(node, ctypes.byref(t)), "cudaGraphNodeGetType")
    return t.value


def _graph_predecessors(rt, graph):
    """Return ``{to_addr: [from_addr, ...]}`` for every edge in the graph."""
    n = ctypes.c_size_t(0)
    if rt._runtime_version >= 13000:
        _check(rt, rt.cudaGraphGetEdges(graph, None, None, None, ctypes.byref(n)), "cudaGraphGetEdges(count)")
    else:
        _check(rt, rt.cudaGraphGetEdges(graph, None, None, ctypes.byref(n)), "cudaGraphGetEdges(count)")
    if n.value == 0:
        return {}
    from_arr = (ctypes.c_void_p * n.value)()
    to_arr = (ctypes.c_void_p * n.value)()
    if rt._runtime_version >= 13000:
        _check(
            rt,
            rt.cudaGraphGetEdges(graph, from_arr, to_arr, None, ctypes.byref(n)),
            "cudaGraphGetEdges(fill)",
        )
    else:
        _check(rt, rt.cudaGraphGetEdges(graph, from_arr, to_arr, ctypes.byref(n)), "cudaGraphGetEdges(fill)")
    preds: dict[int, list[int]] = {}
    for i in range(n.value):
        f = from_arr[i] or 0
        t = to_arr[i] or 0
        preds.setdefault(t, []).append(f)
    return preds


def _has_kernel_ancestor(node_addr, node_types, predecessors):
    seen = set()
    stack = list(predecessors.get(node_addr, ()))
    while stack:
        cur = stack.pop()
        if cur in seen or cur == 0:
            continue
        seen.add(cur)
        if node_types.get(cur) == _CUDA_GRAPH_NODE_TYPE_KERNEL:
            return True
        stack.extend(predecessors.get(cur, ()))
    return False


def _begin_external_capture_to_graph(stream, graph):
    _check(
        CUDART,
        CUDART.cudaStreamBeginCaptureToGraph(
            ctypes.c_void_p(stream.cuda_stream),
            graph,
            None,
            None,
            0,
            _CUDA_STREAM_CAPTURE_MODE_RELAXED,
        ),
        "cudaStreamBeginCaptureToGraph",
    )
    wp.capture_begin(stream=stream, external=True, force_module_load=False, capture_mode=wp.CaptureMode.RELAXED)


def _end_external_capture(stream):
    wp.capture_end(stream=stream)
    out = ctypes.c_void_p(0)
    _check(
        CUDART,
        CUDART.cudaStreamEndCapture(ctypes.c_void_p(stream.cuda_stream), ctypes.byref(out)),
        "cudaStreamEndCapture",
    )


def _build_shared_graph_alloc_in_A_free_in_B(device, *, release_via_gc: bool):
    """Drive the cross-capture alloc-in-A / free-in-B pattern; return the shared ``cudaGraph_t``."""
    stream_a = wp.Stream(device)
    stream_b = wp.Stream(device)

    # Warm up the CUDA memory pool outside of capture. Warp's own
    # ``capture_begin`` performs this warm-up for captures it starts, but this
    # capture is begun externally through the CUDA runtime, and the pool's
    # lazy first-allocation initialization is not capture-safe.
    warmup = wp.empty(N, dtype=wp.float32, device=device)
    del warmup

    wp.synchronize_device()

    g = ctypes.c_void_p(0)
    _check(CUDART, CUDART.cudaGraphCreate(ctypes.byref(g), 0), "cudaGraphCreate")

    # ----- Capture A: alloc + consume on stream_a -----
    # ``ScopedStream`` makes ``stream_a`` the device's current stream, so that
    # ``wp.empty`` (which calls ``wp_alloc_device_async`` against the context's
    # current stream) records its ``MEM_ALLOC`` node on the capturing stream.
    _begin_external_capture_to_graph(stream_a, g)
    with wp.ScopedStream(stream_a, sync_enter=False, sync_exit=False):
        a = wp.empty(N, dtype=wp.float32, device=device)
        wp.launch(_touch, dim=N, inputs=[a], stream=stream_a)
    _end_external_capture(stream_a)

    # ----- Capture B (sharing G): drop ``a`` -----
    # cudaStreamBeginCaptureToGraph reuses the **same capture id** across
    # sequential captures into the same graph. ``g_captures.find(alloc.capture_id)``
    # therefore finds capture B's entry, even though the alloc was recorded
    # under capture A. ``wp_free_device_async`` consequently adds a ``MEM_FREE``
    # node wired to stream_b's frontier (no edge back to ``_touch`` on stream_a).
    _begin_external_capture_to_graph(stream_b, g)
    with wp.ScopedStream(stream_b, sync_enter=False, sync_exit=False):
        if release_via_gc:
            # Stash ``a`` behind a function-scoped temporary that the collector
            # will reclaim, so ``gc.collect()`` deterministically runs the
            # wp.array finalizer. Exercises the GC-driven free path.
            holder = [a]
            del a
            holder.clear()
            del holder
            gc.collect()
        else:
            del a
    _end_external_capture(stream_b)

    return g


def _assert_mem_free_nodes_depend_on_consumer_kernel(test, graph):
    nodes = _graph_nodes(CUDART, graph)
    node_types = {n.value or 0: _node_type(CUDART, n) for n in nodes}
    predecessors = _graph_predecessors(CUDART, graph)

    free_node_addrs = [addr for addr, t in node_types.items() if t == _CUDA_GRAPH_NODE_TYPE_MEM_FREE]
    test.assertGreater(
        len(free_node_addrs),
        0,
        "expected wp_free_device_async to capture a MEM_FREE node for the cross-capture allocation",
    )
    for free_addr in free_node_addrs:
        test.assertTrue(
            _has_kernel_ancestor(free_addr, node_types, predecessors),
            "MEM_FREE node has no KERNEL ancestor: the free was wired against the "
            "calling stream's frontier in capture B (sharing the cudaGraph_t with capture A), "
            "but that frontier has no edge back to the consumer kernel in capture A. "
            "At replay the runtime may schedule MEM_FREE before the kernel completes; "
            "once the mempool recycles the VA this surfaces as cudaErrorIllegalAddress. "
            "Fix wp_free_device_async to order the MEM_FREE against the allocation's "
            "capture frontier (or against the original consumer) instead of the "
            "currently-active capture's frontier.",
        )


def test_cross_capture_explicit_del_orders_against_consumer_kernel(test, device):
    """``del a`` between sequential shared-graph captures must order MEM_FREE after the consumer kernel."""
    if CUDART is None:
        test.skipTest("libcudart >= 12.3 (cudaStreamBeginCaptureToGraph) not available")

    device = wp.get_device(device)
    wp.load_module(device=device)

    g = _build_shared_graph_alloc_in_A_free_in_B(device, release_via_gc=False)
    try:
        _assert_mem_free_nodes_depend_on_consumer_kernel(test, g)
    finally:
        _check(CUDART, CUDART.cudaGraphDestroy(g), "cudaGraphDestroy")


def test_cross_capture_garbage_collect_orders_against_consumer_kernel(test, device):
    """``gc.collect()`` reclaiming the wp.array exercises the same finalizer and must behave identically."""
    if CUDART is None:
        test.skipTest("libcudart >= 12.3 (cudaStreamBeginCaptureToGraph) not available")

    device = wp.get_device(device)
    wp.load_module(device=device)

    g = _build_shared_graph_alloc_in_A_free_in_B(device, release_via_gc=True)
    try:
        _assert_mem_free_nodes_depend_on_consumer_kernel(test, g)
    finally:
        _check(CUDART, CUDART.cudaGraphDestroy(g), "cudaGraphDestroy")


class TestGraphAllocCrossCaptureFree(unittest.TestCase):
    pass


devices = get_selected_cuda_test_devices()

add_function_test(
    TestGraphAllocCrossCaptureFree,
    "test_cross_capture_explicit_del_orders_against_consumer_kernel",
    test_cross_capture_explicit_del_orders_against_consumer_kernel,
    devices=devices,
)
add_function_test(
    TestGraphAllocCrossCaptureFree,
    "test_cross_capture_garbage_collect_orders_against_consumer_kernel",
    test_cross_capture_garbage_collect_orders_against_consumer_kernel,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
