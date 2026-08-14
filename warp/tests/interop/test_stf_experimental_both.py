# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import warp as wp
from warp import stf_experimental as wp_stf

N = 1 << 12
FRAMES = 3

INIT_A1 = 1
INIT_A2 = 2
INIT_B1 = 4
INIT_B2 = 8


@wp.kernel
def _fill_kernel(arr: wp.array[int], value: int):
    arr[wp.tid()] = value


@wp.kernel
def _add_kernel(out: wp.array[int], a: wp.array[int], b: wp.array[int]):
    i = wp.tid()
    out[i] = a[i] + b[i]


def _run_eager(device) -> np.ndarray:
    a = wp.empty(N, dtype=wp.int32, device=device)
    b = wp.empty(N, dtype=wp.int32, device=device)
    c = wp.empty(N, dtype=wp.int32, device=device)
    v1 = wp.empty(N, dtype=wp.int32, device=device)
    v2 = wp.empty(N, dtype=wp.int32, device=device)

    for _ in range(FRAMES):
        wp.launch(_fill_kernel, dim=N, inputs=[v1, INIT_A1], device=device)
        wp.launch(_fill_kernel, dim=N, inputs=[v2, INIT_A2], device=device)
        wp.launch(_add_kernel, dim=N, inputs=[a, v1, v2], device=device)

        wp.launch(_fill_kernel, dim=N, inputs=[v1, INIT_B1], device=device)
        wp.launch(_fill_kernel, dim=N, inputs=[v2, INIT_B2], device=device)
        wp.launch(_add_kernel, dim=N, inputs=[b, v1, v2], device=device)

        wp.launch(_add_kernel, dim=N, inputs=[c, a, b], device=device)

    return c.numpy()


def _inner_fork_join(outer_stream: wp.Stream, device, *, dst: wp.array, val1: int, val2: int):
    v1 = wp.empty(N, dtype=wp.int32, device=device)
    v2 = wp.empty(N, dtype=wp.int32, device=device)

    with wp_stf.context(stream=outer_stream) as inner_ctx:
        tok_v1 = inner_ctx.token()
        tok_v2 = inner_ctx.token()
        tok_dst = inner_ctx.token()

        with inner_ctx.task(tok_v1.write()) as (s,):
            wp.launch(_fill_kernel, dim=N, inputs=[v1, val1], stream=s)

        with inner_ctx.task(tok_v2.write()) as (s,):
            wp.launch(_fill_kernel, dim=N, inputs=[v2, val2], stream=s)

        with inner_ctx.task(tok_v1.read(), tok_v2.read(), tok_dst.write()) as (s,):
            wp.launch(_add_kernel, dim=N, inputs=[dst, v1, v2], stream=s)


def _run_unified_with_local_stf(device) -> np.ndarray:
    a = wp.empty(N, dtype=wp.int32, device=device)
    b = wp.empty(N, dtype=wp.int32, device=device)
    c = wp.empty(N, dtype=wp.int32, device=device)

    step_graph = wp_stf.task_graph()
    outer_ctx = step_graph.context
    tok_a = outer_ctx.token()
    tok_b = outer_ctx.token()
    tok_c = outer_ctx.token()

    with step_graph:
        with outer_ctx.task(tok_a.write()) as (s,):
            _inner_fork_join(s, device, dst=a, val1=INIT_A1, val2=INIT_A2)

        with outer_ctx.task(tok_b.write()) as (s,):
            _inner_fork_join(s, device, dst=b, val1=INIT_B1, val2=INIT_B2)

        with outer_ctx.task(tok_a.read(), tok_b.read(), tok_c.write()) as (s,):
            wp.launch(_add_kernel, dim=N, inputs=[c, a, b], stream=s)

    for _ in range(FRAMES):
        step_graph.launch()

    step_graph.finalize()

    return c.numpy()


@unittest.skipUnless(wp.is_cuda_available() and wp_stf.is_available(), "CUDASTF is not available")
class TestSTFExperimentalBoth(unittest.TestCase):
    def test_unified_dag_with_local_stf_matches_eager(self):
        wp.init()
        device = wp.get_device("cuda:0")

        with wp.ScopedDevice(device):
            ref = _run_eager(device)
            got = _run_unified_with_local_stf(device)

        np.testing.assert_array_equal(ref, np.full(N, INIT_A1 + INIT_A2 + INIT_B1 + INIT_B2, dtype=np.int32))
        np.testing.assert_array_equal(got, ref)


if __name__ == "__main__":
    unittest.main()
