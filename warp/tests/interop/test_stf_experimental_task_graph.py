# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
import unittest
from typing import Any

import numpy as np

import warp as wp
from warp import stf_experimental as wp_stf

N = 256


@wp.kernel
def _fill_kernel(arr: wp.array[int], value: int):
    arr[wp.tid()] = value


def _record_fill_graph(device: wp.Device, value: int) -> tuple[Any, wp.array]:
    result = wp.zeros(N, dtype=wp.int32, device=device)
    graph = wp_stf.task_graph()
    ctx = graph.context
    token = ctx.token()

    with graph:
        with ctx.task(token.write()) as (stream,):
            wp.launch(_fill_kernel, dim=N, inputs=[result, value], stream=stream)

    return graph, result


@unittest.skipUnless(wp.is_cuda_available() and wp_stf.is_available(), "CUDASTF is not available")
class TestSTFExperimentalTaskGraph(unittest.TestCase):
    def test_launch_before_recording_raises(self):
        wp.init()
        graph = wp_stf.task_graph()
        try:
            with self.assertRaises(RuntimeError):
                graph.launch()
            # reset() before recording is an idempotent no-op in cuda-stf.
            graph.reset()
        finally:
            graph.finalize()

    def test_records_launches_dumps_dot_and_finalizes(self):
        wp.init()
        device = wp.get_device("cuda:0")

        with wp.ScopedDevice(device):
            graph, result = _record_fill_graph(device, 11)
            graph.launch()

            np.testing.assert_array_equal(result.numpy(), np.full(N, 11, dtype=np.int32))

            with tempfile.TemporaryDirectory() as tmp_dir:
                dot_path = os.path.join(tmp_dir, "task_graph.dot")
                self.assertEqual(wp_stf.dump_dot(graph.raw, dot_path), dot_path)

            graph.finalize()
            graph.finalize()

            with self.assertRaises(RuntimeError):
                graph.launch()

    def test_task_rejected_outside_recording(self):
        wp.init()
        device = wp.get_device("cuda:0")

        with wp.ScopedDevice(device):
            graph = wp_stf.task_graph()
            ctx = graph.context
            token = ctx.token()

            try:
                with self.assertRaises(RuntimeError):
                    with ctx.task(token.write()):
                        pass

                with self.assertRaises(RuntimeError):
                    with wp_stf.task(ctx, token.write()):
                        pass
            finally:
                graph.finalize()

    def test_failed_record_locks_graph(self):
        wp.init()
        graph = wp_stf.task_graph()

        with self.assertRaises(ValueError):
            with graph:
                raise ValueError("record failed")

        with self.assertRaises(RuntimeError):
            with graph:
                pass

        with self.assertRaises(RuntimeError):
            graph.launch()

        with self.assertRaises(RuntimeError):
            _ = graph.raw
        graph.finalize()

    def test_finalize_before_recording_locks_graph(self):
        wp.init()
        graph = wp_stf.task_graph()
        graph.finalize()
        graph.finalize()

        with self.assertRaises(RuntimeError):
            with graph:
                pass

    def test_reset_prevents_future_launches(self):
        wp.init()
        device = wp.get_device("cuda:0")

        with wp.ScopedDevice(device):
            graph, _ = _record_fill_graph(device, 13)
            graph.launch()
            graph.reset()

            with self.assertRaises(RuntimeError):
                graph.launch()

            graph.finalize()

    def test_task_graph_rejects_device_kwarg(self):
        wp.init()
        device = wp.get_device("cuda:0")

        with self.assertRaises(TypeError):
            wp_stf.task_graph(device=device)


if __name__ == "__main__":
    unittest.main()
