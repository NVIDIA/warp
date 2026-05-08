# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Composite-component writes on ``wp.array(dtype=wp.mat44)``: five
entries written per thread, mimicking constraint-jacobian row population.
Tracks the slot-level codegen fast-path introduced for in-place
composite-component writes.

10 000 iterations: per-launch cost is 9-100 µs, so a large inner loop
is needed to push the total runtime well above that variance floor and
stabilise the median reported by ASV.
"""

from statistics import median

import warp as wp


@wp.kernel
def mat44_multi_elem(dst: wp.array(dtype=wp.mat44), src: wp.array(dtype=wp.float32)):
    i = wp.tid()
    dst[i][0, 0] = src[i] * 1.0
    dst[i][0, 3] = src[i] * 2.0
    dst[i][1, 1] = src[i] * 3.0
    dst[i][2, 2] = src[i] * 4.0
    dst[i][3, 3] = src[i] * 5.0


class RunKernel:
    def setup(self):
        wp.init()
        wp.load_module(device="cuda:0")
        self.n = 1 << 20
        self.dst = wp.zeros(self.n, dtype=wp.mat44, device="cuda:0", requires_grad=True)
        self.src = wp.ones(self.n, dtype=wp.float32, device="cuda:0", requires_grad=True)
        self.cmd_fwd = wp.launch(
            mat44_multi_elem, self.n, inputs=[self.dst, self.src], device="cuda:0", record_cmd=True
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
                    mat44_multi_elem,
                    self.n,
                    inputs=[self.dst, self.src],
                    adj_inputs=[self.dst.grad, self.src.grad],
                    adj_outputs=[],
                    adjoint=True,
                    device="cuda:0",
                )
        return median(r.elapsed for r in timer.timing_results) * 1e-3

    track_backward_cuda.unit = "seconds"
