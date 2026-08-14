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

To stay dependency-free, the foreign scope here is assembled manually with
Warp's low-level CUDA graph bindings (the same calls capture_while() uses),
but WITHOUT Warp's trailing condition-evaluation kernel: the user kernel's
wp.graph_set_conditional() call is the only thing driving loop-back, exactly
as in an externally owned scope.
"""

import ctypes
import unittest

import warp as wp
import warp._src.context as wp_context
from warp._src.context import capture_pause, capture_resume
from warp.tests.unittest_utils import *


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
    """Drive a manually assembled while-node solely from a user kernel."""
    n_iters = 7

    with wp.ScopedDevice(device):
        stream = device.stream

        # Loop-entry condition: non-zero so the body executes at least once.
        cond = wp.array([1], dtype=wp.int32, device=device)
        counter = wp.zeros(1, dtype=wp.int32, device=device)

        wp.load_module(device=device)

        wp.capture_begin(device, stream=stream)
        try:
            body_graph = ctypes.c_void_p()
            cond_handle = ctypes.c_uint64()
            # Insert the while-node exactly as capture_while() does...
            if not wp_context.runtime.core.wp_cuda_graph_insert_while(
                device.context,
                stream.cuda_stream,
                device.get_cuda_compile_arch(),
                device.get_cuda_output_format() == "ptx",
                ctypes.cast(cond.ptr, ctypes.POINTER(ctypes.c_int32)),
                ctypes.byref(body_graph),
                ctypes.byref(cond_handle),
            ):
                raise RuntimeError(wp_context.runtime.get_error_string())

            # ...capture the body into the child graph...
            main_graph = capture_pause(stream=stream)
            main_graph_ptr = main_graph.graph
            main_graph.graph = body_graph
            try:
                capture_resume(main_graph, stream=stream)
                wp.launch(
                    _count_and_continue,
                    dim=1,
                    inputs=[counter, cond_handle.value, n_iters],
                    stream=stream,
                )
                capture_pause(stream=stream)
            finally:
                main_graph.graph = main_graph_ptr

            if not wp_context.runtime.core.wp_cuda_graph_check_conditional_body(body_graph):
                raise RuntimeError(wp_context.runtime.get_error_string())

            # ...but do NOT record Warp's trailing condition-evaluation kernel:
            # the wp.graph_set_conditional() call in the body owns loop-back.
            capture_resume(main_graph, stream=stream)
        except Exception:
            capture_pause(stream=stream)
            raise

        graph = wp.capture_end(device, stream=stream)

        wp.capture_launch(graph, stream=stream)
        wp.synchronize_device(device)

        test.assertEqual(counter.numpy()[0], n_iters)

        # Replays are self-driving: the entry evaluation re-arms the handle
        # from the condition array, and the body's wp.graph_set_conditional()
        # again terminates the loop after n_iters iterations.
        counter.zero_()
        wp.capture_launch(graph, stream=stream)
        wp.synchronize_device(device)
        test.assertEqual(counter.numpy()[0], n_iters)


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
