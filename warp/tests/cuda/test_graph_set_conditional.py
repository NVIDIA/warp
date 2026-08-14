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

The test below plays the role of the foreign framework: using cuda.bindings
(the cuda-python package), it builds a graph with a conditional while-node
(cudaGraphCreate / cudaGraphConditionalHandleCreate / cudaGraphAddNode),
populates the body with a Warp kernel via cudaStreamBeginCaptureToGraph, and
launches it. The kernel's wp.graph_set_conditional() call is the only thing
driving loop-back.
"""

import unittest

import warp as wp
from warp.tests.unittest_utils import *

try:
    from cuda.bindings import runtime as cudart
except ImportError:
    cudart = None


def _check_cuda(result):
    """Unpack a cuda.bindings result tuple, raising on any CUDA error."""
    err, *values = result
    if int(err) != 0:
        raise RuntimeError(f"CUDA runtime call failed: {err}")
    if len(values) == 1:
        return values[0]
    return values


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
    if cudart is None:
        test.skipTest("cuda-python (cuda.bindings) is not installed")

    n_iters = 7

    with wp.ScopedDevice(device):
        counter = wp.zeros(1, dtype=wp.int32, device=device)
        stream = wp.Stream(device)

        wp.load_module(device=device)

        # Build the foreign scope: a graph containing one conditional
        # while-node whose handle defaults to 1 (enter the loop) at every
        # graph launch.
        graph = _check_cuda(cudart.cudaGraphCreate(0))
        handle = _check_cuda(
            cudart.cudaGraphConditionalHandleCreate(
                graph, 1, cudart.cudaGraphConditionalHandleFlags.cudaGraphCondAssignDefault
            )
        )

        params = cudart.cudaGraphNodeParams()
        params.type = cudart.cudaGraphNodeType.cudaGraphNodeTypeConditional
        params.conditional.handle = handle
        params.conditional.type = cudart.cudaGraphConditionalNodeType.cudaGraphCondTypeWhile
        params.conditional.size = 1
        _check_cuda(cudart.cudaGraphAddNode(graph, None, None, 0, params))
        body_graph = params.conditional.phGraph_out[0]

        # Populate the body with a Warp kernel; its wp.graph_set_conditional()
        # call alone decides whether the loop runs another iteration.
        _check_cuda(
            cudart.cudaStreamBeginCaptureToGraph(
                stream.cuda_stream,
                body_graph,
                None,
                None,
                0,
                cudart.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal,
            )
        )
        wp.launch(_count_and_continue, dim=1, inputs=[counter, int(handle), n_iters], stream=stream)
        _check_cuda(cudart.cudaStreamEndCapture(stream.cuda_stream))

        graph_exec = _check_cuda(cudart.cudaGraphInstantiate(graph, 0))
        try:
            _check_cuda(cudart.cudaGraphLaunch(graph_exec, stream.cuda_stream))
            wp.synchronize_stream(stream)
            test.assertEqual(counter.numpy()[0], n_iters)

            # cudaGraphCondAssignDefault re-arms the handle to 1 at every
            # launch, so replays are self-driving.
            counter.zero_()
            _check_cuda(cudart.cudaGraphLaunch(graph_exec, stream.cuda_stream))
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
