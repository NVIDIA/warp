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
def array2d_augassign_kernel(x: wp.array2d(dtype=float), y: wp.array2d(dtype=float)):
    i, j = wp.tid()
    x[i, j] += y[i, j]


class CompileModule:
    repeat = 10  # Number of samples to run
    number = 1  # Number of timed calls per sample

    def setup(self):
        wp.init()
        clear_kernel_cache()

    def teardown(self):
        array2d_augassign_kernel.module.unload()

    def time_cuda_codegen(self):
        wp.load_module(device="cuda:0")

    def time_cpu_codegen(self):
        wp.load_module(device="cpu")


class RunForwardKernel:
    timer = CudaEventTimer()
    repeat = 10
    number = 1

    @setup_once
    def setup(self):
        wp.init()
        wp.load_module(device="cuda:0")

        N = (1024, 1024)
        self.x = wp.ones(N, dtype=float, device="cuda:0")
        self.y = wp.ones(N, dtype=float, device="cuda:0")
        self.graph = capture_launches(
            lambda: wp.launch(array2d_augassign_kernel, self.x.shape, inputs=[self.x, self.y], device="cuda:0"),
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
        self.x = wp.ones(N, dtype=float, device="cuda:0", requires_grad=True)
        self.y = wp.ones(N, dtype=float, device="cuda:0", requires_grad=True)
        self.graph = capture_launches(
            lambda: wp.launch(
                array2d_augassign_kernel,
                self.x.shape,
                inputs=[self.x, self.y],
                adj_inputs=[self.x.grad, self.y.grad],
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
