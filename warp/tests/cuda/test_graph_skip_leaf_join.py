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

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

N = 1024


@wp.kernel
def _write_val(arr: wp.array(dtype=int), value: int):
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
    """Default capture_end: the leaf-join pass adopts the dangling forked
    branch, the capture succeeds, and the branch's work is in the graph."""
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
    """skip_leaf_join=True: Warp does not adopt the dangling leaf, so CUDA
    rejects the capture (unjoined work) and capture_end raises."""
    main = wp.Stream(device)
    forked = wp.Stream(device)
    a = wp.zeros(N, dtype=int, device=device)
    b = wp.zeros(N, dtype=int, device=device)
    wp.load_module(device=device)

    _begin_forked_capture(device, main, forked, a, b)
    with test.assertRaises(RuntimeError):
        wp.capture_end(device=device, stream=main, skip_leaf_join=True)

    wp.synchronize_device(device)


def test_skip_leaf_join_explicit_join_succeeds(test, device):
    """skip_leaf_join=True with the frontier handled by the caller: joining
    the fork explicitly ends the capture cleanly and the graph replays."""
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
    "test_skip_leaf_join_explicit_join_succeeds",
    test_skip_leaf_join_explicit_join_succeeds,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
