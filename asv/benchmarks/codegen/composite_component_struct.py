# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Composite-component writes on a user ``@wp.struct`` array: populating
all four fields per element. Tracks the slot-level codegen fast-path
introduced for in-place composite-component writes.

Each sample replays a CUDA graph of back-to-back launches and reads its
duration from the GPU timeline.
"""

import warp as wp

from ..benchmarks_utils import LAUNCHES_PER_SAMPLE, CudaEventTimer, capture_launches, setup_once


@wp.struct
class StateStruct:
    position: wp.vec3
    velocity: wp.vec3
    rotation: wp.quatf
    mass: wp.float32


@wp.kernel
def state_multi_field(
    dst: wp.array(dtype=StateStruct),
    p: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    q: wp.array(dtype=wp.quatf),
    m: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    dst[i].position = p[i]
    dst[i].velocity = v[i]
    dst[i].rotation = q[i]
    dst[i].mass = m[i]


class RunKernel:
    timer = CudaEventTimer()
    repeat = 10
    number = 1

    @setup_once
    def setup(self):
        wp.init()
        wp.load_module(device="cuda:0")
        self.n = 1 << 20
        self.dst = wp.zeros(self.n, dtype=StateStruct, device="cuda:0", requires_grad=True)
        self.p = wp.ones(self.n, dtype=wp.vec3, device="cuda:0", requires_grad=True)
        self.v = wp.ones(self.n, dtype=wp.vec3, device="cuda:0", requires_grad=True)
        self.q = wp.ones(self.n, dtype=wp.quatf, device="cuda:0", requires_grad=True)
        self.m = wp.ones(self.n, dtype=wp.float32, device="cuda:0", requires_grad=True)
        cmd_fwd = wp.launch(
            state_multi_field,
            self.n,
            inputs=[self.dst, self.p, self.v, self.q, self.m],
            device="cuda:0",
            record_cmd=True,
        )
        self.forward_graph = capture_launches(cmd_fwd.launch, LAUNCHES_PER_SAMPLE)
        self.backward_graph = capture_launches(
            lambda: wp.launch(
                state_multi_field,
                self.n,
                inputs=[self.dst, self.p, self.v, self.q, self.m],
                adj_inputs=[self.dst.grad, self.p.grad, self.v.grad, self.q.grad, self.m.grad],
                adj_outputs=[],
                adjoint=True,
                device="cuda:0",
            ),
            LAUNCHES_PER_SAMPLE,
        )
        wp.capture_launch(self.forward_graph)
        wp.capture_launch(self.backward_graph)
        wp.synchronize_device("cuda:0")

    def time_forward_cuda(self):
        wp.capture_launch(self.forward_graph)

    def time_backward_cuda(self):
        wp.capture_launch(self.backward_graph)
