# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import warp as wp
from warp import stf_experimental as wp_stf

N = 1 << 12
FRAMES = 3


@wp.kernel
def _robot_step(x: wp.array[float], dt: float):
    i = wp.tid()
    v = x[i]
    for _ in range(4):
        v = v + dt * (1.0 - v)
    x[i] = v


@wp.kernel
def _sand_step(x: wp.array[float], y: wp.array[float], dt: float):
    i = wp.tid()
    y[i] = y[i] + dt * x[i]


def _run_eager(device) -> tuple[np.ndarray, np.ndarray]:
    x = wp.zeros(N, dtype=wp.float32, device=device)
    y = wp.zeros(N, dtype=wp.float32, device=device)

    for _ in range(FRAMES):
        for _ in range(3):
            wp.launch(_robot_step, dim=N, inputs=[x, 0.1], device=device)

        for _ in range(2):
            wp.launch(_sand_step, dim=N, inputs=[x, y, 0.1], device=device)

    return x.numpy(), y.numpy()


def _run_stf_outer_dag(device) -> tuple[np.ndarray, np.ndarray]:
    x = wp.zeros(N, dtype=wp.float32, device=device)
    y = wp.zeros(N, dtype=wp.float32, device=device)

    graph = wp_stf.task_graph()
    ctx = graph.context
    token = ctx.token()

    with graph:
        with ctx.task(token.write()) as (stream,):
            for _ in range(3):
                wp.launch(_robot_step, dim=N, inputs=[x, 0.1], stream=stream)

        with ctx.task(token.read()) as (stream,):
            for _ in range(2):
                wp.launch(_sand_step, dim=N, inputs=[x, y, 0.1], stream=stream)

    for _ in range(FRAMES):
        graph.launch()

    graph.finalize()

    return x.numpy(), y.numpy()


@unittest.skipUnless(wp.is_cuda_available() and wp_stf.is_available(), "CUDASTF is not available")
class TestSTFExperimentalOuterDag(unittest.TestCase):
    def test_outer_dag_matches_eager(self):
        wp.init()
        device = wp.get_device("cuda:0")

        with wp.ScopedDevice(device):
            x_ref, y_ref = _run_eager(device)
            x_got, y_got = _run_stf_outer_dag(device)

        np.testing.assert_allclose(x_got, x_ref)
        np.testing.assert_allclose(y_got, y_ref)


if __name__ == "__main__":
    unittest.main()
