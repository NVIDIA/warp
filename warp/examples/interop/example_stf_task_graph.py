# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example CUDASTF Task Graph
#
# Demonstrates composing Warp kernels with cuda.stf through
# warp.stf_experimental. The outer CUDASTF context records a DAG of captured
# tasks, and each captured task opens a local CUDASTF context to express
# fork-join work inside the task.
###########################################################################

import numpy as np

import warp as wp
from warp import stf_experimental as wp_stf

N = 1 << 14


@wp.kernel
def fill_kernel(arr: wp.array[int], value: int):
    arr[wp.tid()] = value


@wp.kernel
def add_kernel(out: wp.array[int], a: wp.array[int], b: wp.array[int]):
    i = wp.tid()
    out[i] = a[i] + b[i]


def inner_fork_join(stream: wp.Stream, device, *, dst: wp.array, val1: int, val2: int):
    v1 = wp.empty(N, dtype=wp.int32, device=device)
    v2 = wp.empty(N, dtype=wp.int32, device=device)

    with wp_stf.context(stream=stream) as ctx:
        tok_v1 = ctx.token()
        tok_v2 = ctx.token()
        tok_dst = ctx.token()

        with ctx.task(tok_v1.write()) as (s,):
            wp.launch(fill_kernel, dim=N, inputs=[v1, val1], stream=s)

        with ctx.task(tok_v2.write()) as (s,):
            wp.launch(fill_kernel, dim=N, inputs=[v2, val2], stream=s)

        with ctx.task(tok_v1.read(), tok_v2.read(), tok_dst.write()) as (s,):
            wp.launch(add_kernel, dim=N, inputs=[dst, v1, v2], stream=s)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default="cuda:0", help="Override the default Warp device.")
    parser.add_argument("--stage-path", type=str, default=None, help="Unused; accepted for test harness compatibility.")
    args = parser.parse_known_args()[0]

    wp.init()

    if not wp.is_cuda_available() or not wp_stf.is_available():
        raise RuntimeError(
            "This example requires CUDA and cuda.stf. Install cuda-stf[cu12] or cuda-stf[cu13] for your CUDA toolkit."
        )

    device = wp.get_device(args.device)

    with wp.ScopedDevice(device):
        a = wp.empty(N, dtype=wp.int32, device=device)
        b = wp.empty(N, dtype=wp.int32, device=device)
        c = wp.empty(N, dtype=wp.int32, device=device)

        graph = wp_stf.task_graph()
        ctx = graph.context
        tok_a = ctx.token()
        tok_b = ctx.token()
        tok_c = ctx.token()

        with graph:
            with ctx.task(tok_a.write()) as (s,):
                inner_fork_join(s, device, dst=a, val1=1, val2=2)

            with ctx.task(tok_b.write()) as (s,):
                inner_fork_join(s, device, dst=b, val1=4, val2=8)

            with ctx.task(tok_a.read(), tok_b.read(), tok_c.write()) as (s,):
                wp.launch(add_kernel, dim=N, inputs=[c, a, b], stream=s)

        graph.launch()
        graph.finalize()

        expected = np.full(N, 15, dtype=np.int32)
        np.testing.assert_array_equal(c.numpy(), expected)
