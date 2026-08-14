# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct unit tests for ``wp.capture_end(skip_leaf_join=...)``.

``wp.capture_end()`` normally runs a broad leaf-join pass: every leaf node of
the captured graph that is not already a capture dependency of the main stream
is appended to the stream's dependency set, so forked capture streams that
were never joined back still end the capture cleanly (CUDA otherwise rejects
the capture with ``cudaErrorStreamCaptureUnjoined``).

Integrations that own the capture frontier themselves (CUDASTF stackable
contexts, ``torch.cuda.graph`` interop, ...) must be able to opt out: inside
an externally-owned capture the broad join would adopt leaves belonging to
OTHER branches of the outer graph and inject false serialization edges.
``skip_leaf_join=True`` disables the pass. These tests pin down both
behaviors without requiring ``cuda.stf``.
"""

import ctypes
import ctypes.util
import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

N = 1024


@wp.kernel
def _write_val(arr: wp.array[int], value: int):
    arr[wp.tid()] = value


def _begin_forked_capture(device, main, forked, a, b):
    """Begin a capture on ``main`` and fork ``forked`` into it.

    Launches one kernel on each stream and leaves the fork UNJOINED.
    """
    wp.capture_begin(device=device, stream=main, force_module_load=False)
    forked.wait_event(main.record_event())
    wp.launch(_write_val, dim=N, inputs=[a, 1], stream=main)
    wp.launch(_write_val, dim=N, inputs=[b, 2], stream=forked)


def test_leaf_join_rescues_unjoined_fork(test, device):
    """Check that the default leaf-join pass rescues an unjoined forked stream.

    The pass adopts the dangling forked branch, the capture succeeds, and the
    branch's work is part of the replayed graph.
    """
    main = wp.Stream(device)
    forked = wp.Stream(device)
    a = wp.zeros(N, dtype=int, device=device)
    b = wp.zeros(N, dtype=int, device=device)
    wp.load_module(device=device)

    _begin_forked_capture(device, main, forked, a, b)
    graph = wp.capture_end(device=device, stream=main)

    wp.capture_launch(graph, stream=main)
    wp.synchronize_stream(main)
    assert_np_equal(a.numpy(), np.full(N, 1, dtype=np.int32))
    assert_np_equal(b.numpy(), np.full(N, 2, dtype=np.int32))


def test_skip_leaf_join_rejects_unjoined_fork(test, device):
    """Check that ``skip_leaf_join=True`` refuses to adopt a dangling leaf.

    Warp does not join the forked branch, so CUDA rejects the capture as
    unjoined work and ``capture_end`` raises.
    """
    main = wp.Stream(device)
    forked = wp.Stream(device)
    a = wp.zeros(N, dtype=int, device=device)
    b = wp.zeros(N, dtype=int, device=device)
    wp.load_module(device=device)

    _begin_forked_capture(device, main, forked, a, b)
    with test.assertRaises(RuntimeError):
        wp.capture_end(device=device, stream=main, skip_leaf_join=True)


def test_skip_leaf_join_unjoined_fork_cleans_up_allocs(test, device):
    """Check allocation bookkeeping survives a rejected unjoined capture.

    The capture allocates an array (a graph allocation) before failing with
    unjoined forked work. The failed capture_end must clean up the graph
    allocation bookkeeping: freeing the array afterwards and running a
    subsequent capture must both work normally.
    """
    if not device.is_mempool_enabled:
        test.skipTest("Requires a mempool-enabled device")

    main = wp.Stream(device)
    forked = wp.Stream(device)
    a = wp.zeros(N, dtype=int, device=device)
    b = wp.zeros(N, dtype=int, device=device)
    wp.load_module(device=device)

    _begin_forked_capture(device, main, forked, a, b)
    # allocate inside the capture so the capture owns graph-alloc bookkeeping
    with wp.ScopedStream(main, sync_enter=False):
        c = wp.zeros(N, dtype=int, device=device)
        wp.launch(_write_val, dim=N, inputs=[c, 3], stream=main)
    with test.assertRaises(RuntimeError):
        wp.capture_end(device=device, stream=main, skip_leaf_join=True)

    # the failed capture must not wedge allocation tracking: dropping the
    # graph-allocated array and capturing again must work
    del c
    _begin_forked_capture(device, main, forked, a, b)
    main.wait_event(forked.record_event())
    graph = wp.capture_end(device=device, stream=main, skip_leaf_join=True)
    wp.capture_launch(graph, stream=main)
    wp.synchronize_stream(main)
    assert_np_equal(a.numpy(), np.full(N, 1, dtype=np.int32))
    assert_np_equal(b.numpy(), np.full(N, 2, dtype=np.int32))


def test_skip_leaf_join_explicit_join_succeeds(test, device):
    """Check that ``skip_leaf_join=True`` with a caller-managed join succeeds.

    With the capture frontier handled by the caller, joining the fork
    explicitly ends the capture cleanly and the graph replays correctly.
    """
    main = wp.Stream(device)
    forked = wp.Stream(device)
    a = wp.zeros(N, dtype=int, device=device)
    b = wp.zeros(N, dtype=int, device=device)
    wp.load_module(device=device)

    _begin_forked_capture(device, main, forked, a, b)
    # the integration owns the frontier discipline: join the fork ourselves
    main.wait_event(forked.record_event())
    graph = wp.capture_end(device=device, stream=main, skip_leaf_join=True)

    wp.capture_launch(graph, stream=main)
    wp.synchronize_stream(main)
    assert_np_equal(a.numpy(), np.full(N, 1, dtype=np.int32))
    assert_np_equal(b.numpy(), np.full(N, 2, dtype=np.int32))


def _load_cudart():
    """Load the CUDA runtime library for direct graph inspection."""
    candidates = []
    found = ctypes.util.find_library("cudart")
    if found:
        candidates.append(found)
    candidates += ["libcudart.so.13", "libcudart.so.12", "libcudart.so"]
    for name in candidates:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def _capture_with_guest_windows(test, device, skip_leaf_join):
    """Externally-owned capture with two guest ScopedCapture windows.

    The test owns the capture like an outer integration would: it begins the
    capture directly through the CUDA runtime (not through Warp), forks two
    streams into it, and adopts the ongoing capture from each fork with
    ``ScopedCapture(external=True)`` guest windows. Branch ``fork1`` is
    deliberately left dangling while the second window closes, then the owner
    extends ``fork2`` with one more kernel, joins its branches back, and ends
    the capture itself.

    With ``skip_leaf_join=True`` the second window's close leaves ``fork1``'s
    dangling leaf alone, so the owner's follow-up kernel depends only on its
    own branch. With the default broad join, the dangling leaf is swept into
    ``fork2``'s dependency set and the follow-up kernel gains a false
    serialization edge: exactly one extra edge in the captured graph.

    Returns the total edge count of the captured graph.
    """
    cudart = _load_cudart()
    if cudart is None:
        test.skipTest("libcudart is not available for graph inspection")

    main = wp.Stream(device)
    fork1 = wp.Stream(device)
    fork2 = wp.Stream(device)
    a = wp.zeros(N, dtype=int, device=device)
    b = wp.zeros(N, dtype=int, device=device)
    c = wp.zeros(N, dtype=int, device=device)
    wp.load_module(device=device)

    with wp.ScopedDevice(device):
        # the owner begins the capture through the CUDA runtime (thread-local mode)
        err = cudart.cudaStreamBeginCapture(ctypes.c_void_p(main.cuda_stream), 1)
        test.assertEqual(err, 0, "cudaStreamBeginCapture failed")

        graph = ctypes.c_void_p()
        try:
            ev = main.record_event()
            fork1.wait_event(ev)
            fork2.wait_event(ev)

            # guest window on fork1; the owner intends to extend this branch later
            with wp.ScopedCapture(stream=fork1, external=True, skip_leaf_join=skip_leaf_join):
                wp.launch(_write_val, dim=N, inputs=[a, 1], stream=fork1)

            # guest window on fork2, closed while fork1's leaf is still dangling
            with wp.ScopedCapture(stream=fork2, external=True, skip_leaf_join=skip_leaf_join):
                wp.launch(_write_val, dim=N, inputs=[b, 2], stream=fork2)

            # the owner extends fork2 after the guest windows closed
            wp.launch(_write_val, dim=N, inputs=[c, 3], stream=fork2)

            # the owner joins its branches and ends its capture
            main.wait_event(fork1.record_event())
            main.wait_event(fork2.record_event())
        finally:
            err = cudart.cudaStreamEndCapture(ctypes.c_void_p(main.cuda_stream), ctypes.byref(graph))

        test.assertEqual(err, 0, "cudaStreamEndCapture failed")
        test.assertTrue(graph.value, "capture produced no graph")

        try:
            # cudaGraphGetEdges changed signature in CUDA 13 (an edge-data
            # array parameter was inserted before numEdges), so dispatch on
            # the runtime version rather than on the returned error code.
            version = ctypes.c_int(0)
            err = cudart.cudaRuntimeGetVersion(ctypes.byref(version))
            test.assertEqual(err, 0, "cudaRuntimeGetVersion failed")

            num_edges = ctypes.c_size_t()
            if version.value >= 13000:
                # CUDA 13+: (graph, from, to, edgeData, numEdges). A NULL
                # edgeData count query is lossless here: the graph's edges
                # come from kernel launches and event order, which carry
                # default edge data.
                err = cudart.cudaGraphGetEdges(graph, None, None, None, ctypes.byref(num_edges))
            else:
                # CUDA 12: (graph, from, to, numEdges)
                err = cudart.cudaGraphGetEdges(graph, None, None, ctypes.byref(num_edges))
            test.assertEqual(err, 0, "cudaGraphGetEdges failed")
            return num_edges.value
        finally:
            cudart.cudaGraphDestroy(graph)


def test_scoped_capture_windows_preserve_owner_frontier(test, device):
    """Check that guest ScopedCapture windows preserve the owner's frontier.

    ``ScopedCapture(external=True, skip_leaf_join=True)`` guest windows must
    not adopt dangling leaves of the owner's other branches: the same capture
    runs with and without the opt-out, and the default's broad join shows up
    as exactly one extra (false) serialization edge.
    """
    edges_skip = _capture_with_guest_windows(test, device, skip_leaf_join=True)
    edges_join = _capture_with_guest_windows(test, device, skip_leaf_join=False)

    test.assertEqual(
        edges_join,
        edges_skip + 1,
        "the default broad leaf-join should inject exactly one false serialization "
        f"edge relative to skip_leaf_join=True (got {edges_join} vs {edges_skip})",
    )


class TestGraphSkipLeafJoin(unittest.TestCase):
    pass


devices = get_selected_cuda_test_devices()

add_function_test(
    TestGraphSkipLeafJoin,
    "test_leaf_join_rescues_unjoined_fork",
    test_leaf_join_rescues_unjoined_fork,
    devices=devices,
)
add_function_test(
    TestGraphSkipLeafJoin,
    "test_skip_leaf_join_rejects_unjoined_fork",
    test_skip_leaf_join_rejects_unjoined_fork,
    devices=devices,
)
add_function_test(
    TestGraphSkipLeafJoin,
    "test_skip_leaf_join_unjoined_fork_cleans_up_allocs",
    test_skip_leaf_join_unjoined_fork_cleans_up_allocs,
    devices=devices,
)
add_function_test(
    TestGraphSkipLeafJoin,
    "test_skip_leaf_join_explicit_join_succeeds",
    test_skip_leaf_join_explicit_join_succeeds,
    devices=devices,
)
add_function_test(
    TestGraphSkipLeafJoin,
    "test_scoped_capture_windows_preserve_owner_frontier",
    test_scoped_capture_windows_preserve_owner_frontier,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
