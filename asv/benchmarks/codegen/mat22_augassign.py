# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
def matrix_augassign_kernel(a: wp.array2d(dtype=wp.mat22), b: wp.array2d(dtype=wp.mat22)):
    i, j = wp.tid()

    m1 = wp.mat22()
    m2 = b[i, j]

    m1[0, 0] += m2[0, 0]
    m1[0, 1] += m2[0, 1]
    m1[1, 0] += m2[1, 0]
    m1[1, 1] += m2[1, 1]

    a[i, j] = m1


class CompileModule:
    repeat = 20  # Number of samples to run
    number = 1  # Number of timed calls per sample

    def setup(self):
        wp.init()
        clear_kernel_cache()

    def teardown(self):
        matrix_augassign_kernel.module.unload()
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
        self.a = wp.zeros(N, dtype=wp.mat22, device="cuda:0")
        self.b = wp.ones(N, dtype=wp.mat22, device="cuda:0")
        self.graph = capture_launches(
            lambda: wp.launch(matrix_augassign_kernel, self.a.shape, inputs=[self.a, self.b], device="cuda:0"),
            LAUNCHES_PER_SAMPLE,
        )
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
        self.a = wp.zeros(N, dtype=wp.mat22, device="cuda:0", requires_grad=True)
        self.b = wp.ones(N, dtype=wp.mat22, device="cuda:0", requires_grad=True)
        self.graph = capture_launches(
            lambda: wp.launch(
                matrix_augassign_kernel,
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
