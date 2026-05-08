# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Composite-component writes on ``wp.array(dtype=wp.vec3)``: all three
components written per element. Tracks the slot-level codegen fast-path
introduced for in-place composite-component writes.

10 000 iterations: per-launch cost is 9-100 µs, so a large inner loop
is needed to push the total runtime well above that variance floor and
stabilise the median reported by ASV.
"""

from statistics import median

import warp as wp


@wp.kernel
def vec3_triple_slot(dst: wp.array(dtype=wp.vec3), src: wp.array(dtype=wp.float32)):
    i = wp.tid()
    dst[i].x = src[i] * 2.0
    dst[i].y = src[i] * 3.0
    dst[i].z = src[i] * 4.0


class RunKernel:
    def setup(self):
        wp.init()
        wp.load_module(device="cuda:0")
        self.n = 1 << 20
        self.dst = wp.zeros(self.n, dtype=wp.vec3, device="cuda:0", requires_grad=True)
        self.src = wp.ones(self.n, dtype=wp.float32, device="cuda:0", requires_grad=True)
        self.cmd_fwd = wp.launch(
            vec3_triple_slot, self.n, inputs=[self.dst, self.src], device="cuda:0", record_cmd=True
        )
        wp.synchronize_device("cuda:0")

    def track_forward_cuda(self):
        with wp.ScopedTimer("bench", print=False, cuda_filter=wp.TIMING_KERNEL, synchronize=True) as timer:
            for _ in range(10000):
                self.cmd_fwd.launch()
        return median(r.elapsed for r in timer.timing_results) * 1e-3

    track_forward_cuda.unit = "seconds"

    def track_backward_cuda(self):
        with wp.ScopedTimer("bench", print=False, cuda_filter=wp.TIMING_KERNEL, synchronize=True) as timer:
            for _ in range(10000):
                wp.launch(
                    vec3_triple_slot,
                    self.n,
                    inputs=[self.dst, self.src],
                    adj_inputs=[self.dst.grad, self.src.grad],
                    adj_outputs=[],
                    adjoint=True,
                    device="cuda:0",
                )
        return median(r.elapsed for r in timer.timing_results) * 1e-3

    track_backward_cuda.unit = "seconds"
