# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import warp as wp
from warp import stf_experimental as wp_stf

N = 1024


@wp.kernel
def _fill_kernel(arr: wp.array[float], value: float):
    arr[wp.tid()] = value


@wp.kernel
def _add_kernel(out: wp.array[float], a: wp.array[float], b: wp.array[float]):
    i = wp.tid()
    out[i] = a[i] + b[i]


@wp.kernel
def _dec_cond(cond: wp.array[int]):
    cond[0] = cond[0] - 1


@wp.kernel
def _add_value(arr: wp.array[float], value: float):
    arr[wp.tid()] = arr[wp.tid()] + value


def _run_capture_while_with_local_stf(device) -> np.ndarray:
    result = wp.zeros(N, dtype=wp.float32, device=device)
    cond = wp.full((1,), 1, dtype=wp.int32, device=device)
    capture_stream = wp.Stream(device)

    # Scratch buffers must live outside the while-body: a conditional graph
    # body cannot contain cudaGraphNodeTypeMemAlloc / MemFree nodes, so we
    # pre-allocate once and reuse across iterations.
    a = wp.empty(N, dtype=wp.float32, device=device)
    b = wp.empty(N, dtype=wp.float32, device=device)

    def while_body():
        with wp_stf.context(stream=capture_stream) as ctx:
            tok_a = ctx.token()
            tok_b = ctx.token()
            tok_join = ctx.token()

            with ctx.task(tok_a.write()) as (s,):
                wp.launch(_fill_kernel, dim=N, inputs=[a, 3.0], stream=s)

            with ctx.task(tok_b.write()) as (s,):
                wp.launch(_fill_kernel, dim=N, inputs=[b, 4.0], stream=s)

            with ctx.task(tok_a.read(), tok_b.read(), tok_join.write()) as (s,):
                wp.launch(_add_kernel, dim=N, inputs=[result, a, b], stream=s)

        wp.launch(_dec_cond, dim=1, inputs=[cond], device=device, stream=capture_stream)

    wp.load_module(device=device)
    with wp.ScopedStream(capture_stream):
        with wp.ScopedCapture(
            device=device, stream=capture_stream, force_module_load=False, capture_mode=wp.CaptureMode.RELAXED
        ) as capture:
            wp.capture_while(cond, while_body)

    wp.capture_launch(capture.graph, stream=capture_stream)
    wp.synchronize_stream(capture_stream)
    return result.numpy()


def _run_stf_task_with_capture_while(device) -> np.ndarray:
    result = wp.zeros(N, dtype=wp.float32, device=device)
    cond = wp.full((1,), 3, dtype=wp.int32, device=device)

    graph = wp_stf.task_graph()
    ctx = graph.context
    tok_result = ctx.token()

    def while_body():
        wp.launch(_add_value, dim=N, inputs=[result, 2.0], device=device)
        wp.launch(_dec_cond, dim=1, inputs=[cond], device=device)

    wp.load_module(device=device)
    wp_stf.warmup(device=device)

    with graph:
        with ctx.task(tok_result.write()) as (_stream,):
            wp.capture_while(cond, while_body)

    graph.launch()
    wp.synchronize_device(device)
    graph.finalize()
    return result.numpy()


def _run_two_stf_tasks_with_capture_while(device) -> tuple[np.ndarray, np.ndarray]:
    result_a = wp.zeros(N, dtype=wp.float32, device=device)
    result_b = wp.zeros(N, dtype=wp.float32, device=device)
    cond_a = wp.full((1,), 2, dtype=wp.int32, device=device)
    cond_b = wp.full((1,), 4, dtype=wp.int32, device=device)

    graph = wp_stf.task_graph()
    ctx = graph.context
    tok_a = ctx.token()
    tok_b = ctx.token()

    wp.load_module(device=device)
    wp_stf.warmup(device=device)

    with graph:
        with ctx.task(tok_a.write()) as (stream,):

            def while_body_a():
                wp.launch(_add_value, dim=N, inputs=[result_a, 3.0], stream=stream)
                wp.launch(_dec_cond, dim=1, inputs=[cond_a], stream=stream)

            wp.capture_while(cond_a, while_body_a, stream=stream)

        with ctx.task(tok_b.write()) as (stream,):

            def while_body_b():
                wp.launch(_add_value, dim=N, inputs=[result_b, 5.0], stream=stream)
                wp.launch(_dec_cond, dim=1, inputs=[cond_b], stream=stream)

            wp.capture_while(cond_b, while_body_b, stream=stream)

    graph.launch()
    wp.synchronize_device(device)
    graph.finalize()
    return result_a.numpy(), result_b.numpy()


@unittest.skipUnless(
    wp.is_cuda_available() and wp.is_conditional_graph_supported() and wp_stf.is_available(),
    "CUDASTF or conditional CUDA graphs are not available",
)
class TestSTFExperimentalCaptureWhile(unittest.TestCase):
    def test_capture_while_body_with_local_stf(self):
        wp.init()
        device = wp.get_device("cuda:0")

        with wp.ScopedDevice(device):
            result = _run_capture_while_with_local_stf(device)

        np.testing.assert_allclose(result, np.full(N, 7.0, dtype=np.float32))

    def test_stf_task_body_with_capture_while(self):
        wp.init()
        device = wp.get_device("cuda:0")

        with wp.ScopedDevice(device):
            result = _run_stf_task_with_capture_while(device)

        np.testing.assert_allclose(result, np.full(N, 6.0, dtype=np.float32))

    def test_two_stf_tasks_with_capture_while(self):
        wp.init()
        device = wp.get_device("cuda:0")

        with wp.ScopedDevice(device):
            result_a, result_b = _run_two_stf_tasks_with_capture_while(device)

        np.testing.assert_allclose(result_a, np.full(N, 6.0, dtype=np.float32))
        np.testing.assert_allclose(result_b, np.full(N, 20.0, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
