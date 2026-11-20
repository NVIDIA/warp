# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warp as wp

from ..benchmarks_utils import (
    LAUNCHES_PER_SAMPLE,
    CudaEventTimer,
    capture_launches,
    clear_kernel_cache,
    setup_once,
)


@wp.kernel
def component_assignment(a: wp.array2d(dtype=wp.mat44), b: wp.array2d(dtype=wp.mat44)):
    i, j = wp.tid()

    m1 = wp.mat44()
    m2 = a[i, j]

    for ii in range(4):
        for jj in range(4):
            m1[ii, jj] = m2[ii, jj]

    b[i, j] = m1


class CompileModule:
    repeat = 20  # Number of samples to run
    number = 1  # Number of timed calls per sample

    def setup(self):
        wp.init()
        clear_kernel_cache()

    def teardown(self):
        component_assignment.module.unload()
        clear_kernel_cache()

    def time_cuda_codegen(self):
        wp.load_module(device="cuda:0")


class RunForwardKernel:
    timer = CudaEventTimer()
    repeat = 10
    number = 1

    @setup_once
    def setup(self):
        wp.init()
        wp.load_module(device="cuda:0")

        N = (1024, 1024)
        self.a = wp.ones(N, dtype=wp.mat44, device="cuda:0")
        self.b = wp.zeros(N, dtype=wp.mat44, device="cuda:0")
        cmd = wp.launch(component_assignment, self.a.shape, inputs=[self.a, self.b], device="cuda:0", record_cmd=True)
        self.graph = capture_launches(cmd.launch, LAUNCHES_PER_SAMPLE)
        wp.capture_launch(self.graph)
        wp.synchronize_device("cuda:0")

    def time_cuda(self):
        wp.capture_launch(self.graph)


class RunBackwardKernel:
    timer = CudaEventTimer()
    repeat = 10
    number = 1

    @setup_once
    def setup(self):
        wp.init()
        wp.load_module(device="cuda:0")

        N = (1024, 1024)
        self.a = wp.ones(N, dtype=wp.mat44, device="cuda:0", requires_grad=True)
        self.b = wp.zeros(N, dtype=wp.mat44, device="cuda:0", requires_grad=True)
        self.graph = capture_launches(
            lambda: wp.launch(
                component_assignment,
                self.a.shape,
                inputs=[self.a, self.b],
                adj_inputs=[self.a.grad, self.b.grad],
                adj_outputs=[],
                adjoint=True,
                device="cuda:0",
            ),
            LAUNCHES_PER_SAMPLE,
        )
        wp.capture_launch(self.graph)
        wp.synchronize_device("cuda:0")

    def time_cuda(self):
        wp.capture_launch(self.graph)
