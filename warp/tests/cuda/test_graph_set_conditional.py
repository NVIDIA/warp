# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the wp.graph_set_conditional() device builtin.

The builtin targets FOREIGN conditional scopes, where the caller owns the
condition handle: conditional nodes created by another framework (e.g. an
outer PyTorch capture) or by direct CUDA Graph API use when composing
captured code. Warp's own capture_while()/capture_if() scopes are driven by
their condition array instead -- Warp records its own evaluation kernel after
the loop body, overwriting any value set from the body, and their handles
are deliberately not exposed.

The test below plays the role of the foreign framework: it builds a graph
with a conditional while-node directly through the CUDA runtime API
(cudaGraphCreate / cudaGraphConditionalHandleCreate / cudaGraphAddNode),
populates the body with a Warp kernel via cudaStreamBeginCaptureToGraph, and
launches it. The kernel's wp.graph_set_conditional() call is the only thing
driving loop-back.
"""

import ctypes
import unittest

import warp as wp
from warp.tests.unittest_utils import *

# driver_types.h constants
_CUDA_GRAPH_COND_ASSIGN_DEFAULT = 1  # re-apply the default handle value at every graph launch
_CUDA_GRAPH_COND_TYPE_WHILE = 1
_CUDA_GRAPH_NODE_TYPE_CONDITIONAL = 0x0D
_CUDA_STREAM_CAPTURE_MODE_THREAD_LOCAL = 1


class _CondParams(ctypes.Structure):
    # struct cudaConditionalNodeParams. The trailing ctx field exists on
    # CUDA 13; on CUDA 12 those 8 zero bytes fall into the zero-initialized
    # padding of the enclosing union, so one layout serves both.
    _fields_ = [
        ("handle", ctypes.c_uint64),
        ("type", ctypes.c_int),
        ("size", ctypes.c_uint),
        ("phGraph_out", ctypes.POINTER(ctypes.c_void_p)),
        ("ctx", ctypes.c_void_p),
    ]


class _GraphNodeParams(ctypes.Structure):
    # struct cudaGraphNodeParams: node type + reserved ints + a 232-byte
    # union (long long reserved1[29]) + reserved2. Unused bytes must be zero,
    # which ctypes.Structure() guarantees.
    _fields_ = [
        ("type", ctypes.c_int),
        ("reserved0", ctypes.c_int * 3),
        ("conditional", _CondParams),
        ("reserved_pad", ctypes.c_byte * (232 - ctypes.sizeof(_CondParams))),
        ("reserved2", ctypes.c_longlong),
    ]


def _load_cudart():
    """dlopen the CUDA runtime, trying versioned sonames first (pip wheels ship no bare .so)."""
    for name in ("libcudart.so.13", "libcudart.so.12", "libcudart.so", "cudart64_13.dll", "cudart64_12.dll"):
        try:
            cudart = ctypes.CDLL(name)
        except OSError:
            continue
        cudart.cudaGraphCreate.argtypes = (ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint)
        cudart.cudaGraphConditionalHandleCreate.argtypes = (
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_uint,
        )
        cudart.cudaGraphAddNode.argtypes = (
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(_GraphNodeParams),
        )
        cudart.cudaStreamBeginCaptureToGraph.argtypes = (
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        )
        cudart.cudaStreamEndCapture.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p))
        cudart.cudaGraphInstantiate.argtypes = (
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_ulonglong,
        )
        cudart.cudaGraphLaunch.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
        cudart.cudaGraphExecDestroy.argtypes = (ctypes.c_void_p,)
        cudart.cudaGraphDestroy.argtypes = (ctypes.c_void_p,)
        return cudart
    return None


@wp.kernel
def _count_and_continue(counter: wp.array(dtype=int), handle: wp.graph_cond_handle, n_iters: int):
    count = counter[0] + 1
    counter[0] = count
    # Sole driver of the while-node's loop-back decision.
    wp.graph_set_conditional(handle, wp.where(count < n_iters, 1, 0))


@wp.kernel
def _noop_set(handle: wp.graph_cond_handle):
    wp.graph_set_conditional(handle, 0)


def test_user_owned_while_scope(test, device):
    """Drive a while-node built through the CUDA runtime API solely from a Warp kernel."""
    cudart = _load_cudart()
    if cudart is None:
        test.skipTest("CUDA runtime library not found on the loader path")

    def check(err):
        test.assertEqual(err, 0, f"CUDA runtime call failed with error {err}")

    n_iters = 7

    with wp.ScopedDevice(device):
        counter = wp.zeros(1, dtype=wp.int32, device=device)
        stream = wp.Stream(device)

        wp.load_module(device=device)

        # Build the foreign scope: a graph containing one conditional
        # while-node whose handle defaults to 1 (enter the loop) at every
        # graph launch.
        graph = ctypes.c_void_p()
        check(cudart.cudaGraphCreate(ctypes.byref(graph), 0))

        handle = ctypes.c_uint64()
        check(cudart.cudaGraphConditionalHandleCreate(ctypes.byref(handle), graph, 1, _CUDA_GRAPH_COND_ASSIGN_DEFAULT))

        params = _GraphNodeParams()
        params.type = _CUDA_GRAPH_NODE_TYPE_CONDITIONAL
        params.conditional.handle = handle.value
        params.conditional.type = _CUDA_GRAPH_COND_TYPE_WHILE
        params.conditional.size = 1
        node = ctypes.c_void_p()
        check(cudart.cudaGraphAddNode(ctypes.byref(node), graph, None, None, 0, ctypes.byref(params)))
        body_graph = params.conditional.phGraph_out[0]

        # Populate the body with a Warp kernel; its wp.graph_set_conditional()
        # call alone decides whether the loop runs another iteration.
        check(
            cudart.cudaStreamBeginCaptureToGraph(
                stream.cuda_stream, body_graph, None, None, 0, _CUDA_STREAM_CAPTURE_MODE_THREAD_LOCAL
            )
        )
        wp.launch(_count_and_continue, dim=1, inputs=[counter, handle.value, n_iters], stream=stream)
        out_graph = ctypes.c_void_p()
        check(cudart.cudaStreamEndCapture(stream.cuda_stream, ctypes.byref(out_graph)))

        graph_exec = ctypes.c_void_p()
        check(cudart.cudaGraphInstantiate(ctypes.byref(graph_exec), graph, 0))
        try:
            check(cudart.cudaGraphLaunch(graph_exec, stream.cuda_stream))
            wp.synchronize_stream(stream)
            test.assertEqual(counter.numpy()[0], n_iters)

            # cudaGraphCondAssignDefault re-arms the handle to 1 at every
            # launch, so replays are self-driving.
            counter.zero_()
            check(cudart.cudaGraphLaunch(graph_exec, stream.cuda_stream))
            wp.synchronize_stream(stream)
            test.assertEqual(counter.numpy()[0], n_iters)
        finally:
            cudart.cudaGraphExecDestroy(graph_exec)
            cudart.cudaGraphDestroy(graph)


def test_cpu_noop(test, device):
    """The builtin compiles and is a no-op on CPU devices."""
    with wp.ScopedDevice(device):
        wp.launch(_noop_set, dim=1, inputs=[0], device=device)
        wp.synchronize_device(device)


class TestGraphSetConditional(unittest.TestCase):
    pass


cuda_devices = get_selected_cuda_test_devices()
cpu_devices = [d for d in get_test_devices(mode="basic") if d.is_cpu]

if wp.is_conditional_graph_supported():
    add_function_test(
        TestGraphSetConditional,
        "test_user_owned_while_scope",
        test_user_owned_while_scope,
        devices=cuda_devices,
    )
add_function_test(TestGraphSetConditional, "test_cpu_noop", test_cpu_noop, devices=cpu_devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
