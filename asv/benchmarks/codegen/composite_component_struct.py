# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Composite-component writes on a user ``@wp.struct`` array: populating
all four fields per element. Tracks the slot-level codegen fast-path
introduced for in-place composite-component writes.

10 000 iterations: per-launch cost is 9-100 µs, so a large inner loop
is needed to push the total runtime well above that variance floor and
stabilise the median reported by ASV.
"""

from statistics import median

import warp as wp


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
    def setup(self):
        wp.init()
        wp.load_module(device="cuda:0")
        self.n = 1 << 20
        self.dst = wp.zeros(self.n, dtype=StateStruct, device="cuda:0", requires_grad=True)
        self.p = wp.ones(self.n, dtype=wp.vec3, device="cuda:0", requires_grad=True)
        self.v = wp.ones(self.n, dtype=wp.vec3, device="cuda:0", requires_grad=True)
        self.q = wp.ones(self.n, dtype=wp.quatf, device="cuda:0", requires_grad=True)
        self.m = wp.ones(self.n, dtype=wp.float32, device="cuda:0", requires_grad=True)
        self.cmd_fwd = wp.launch(
            state_multi_field,
            self.n,
            inputs=[self.dst, self.p, self.v, self.q, self.m],
            device="cuda:0",
            record_cmd=True,
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
                    state_multi_field,
                    self.n,
                    inputs=[self.dst, self.p, self.v, self.q, self.m],
                    adj_inputs=[self.dst.grad, self.p.grad, self.v.grad, self.q.grad, self.m.grad],
                    adj_outputs=[],
                    adjoint=True,
                    device="cuda:0",
                )
        return median(r.elapsed for r in timer.timing_results) * 1e-3

    track_backward_cuda.unit = "seconds"
