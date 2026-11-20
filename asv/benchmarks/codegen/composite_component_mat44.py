# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Composite-component writes on ``wp.array(dtype=wp.mat44)``: five
entries written per thread, mimicking constraint-jacobian row population.
Tracks the slot-level codegen fast-path introduced for in-place
composite-component writes.

Each sample replays a CUDA graph of back-to-back launches and reads its
duration from the GPU timeline.
"""

import warp as wp

from ..benchmarks_utils import LAUNCHES_PER_SAMPLE, CudaEventTimer, capture_launches, setup_once


@wp.kernel
def mat44_multi_elem(dst: wp.array(dtype=wp.mat44), src: wp.array(dtype=wp.float32)):
    i = wp.tid()
    dst[i][0, 0] = src[i] * 1.0
    dst[i][0, 3] = src[i] * 2.0
    dst[i][1, 1] = src[i] * 3.0
    dst[i][2, 2] = src[i] * 4.0
    dst[i][3, 3] = src[i] * 5.0


class RunKernel:
    timer = CudaEventTimer()
    repeat = 10
    number = 1

    @setup_once
    def setup(self):
        wp.init()
        wp.load_module(device="cuda:0")
        self.n = 1 << 20
        self.dst = wp.zeros(self.n, dtype=wp.mat44, device="cuda:0", requires_grad=True)
        self.src = wp.ones(self.n, dtype=wp.float32, device="cuda:0", requires_grad=True)
        cmd_fwd = wp.launch(mat44_multi_elem, self.n, inputs=[self.dst, self.src], device="cuda:0", record_cmd=True)
        self.forward_graph = capture_launches(cmd_fwd.launch, LAUNCHES_PER_SAMPLE)
        self.backward_graph = capture_launches(
            lambda: wp.launch(
                mat44_multi_elem,
                self.n,
                inputs=[self.dst, self.src],
                adj_inputs=[self.dst.grad, self.src.grad],
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
