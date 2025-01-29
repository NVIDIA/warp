# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import warp as wp


@wp.kernel
def matrix_augassign_kernel(
    a: wp.array2d(dtype=wp.mat22),
    b: wp.array2d(dtype=wp.mat22),
    c: wp.array2d(dtype=wp.mat22),
    d: wp.array2d(dtype=wp.mat22),
):
    i, j = wp.tid()

    m1 = wp.mat22()
    m2 = b[i, j]

    m1[0, 0] += m2[0, 0]
    m1[0, 1] += m2[0, 1]
    m1[1, 0] += m2[1, 0]
    m1[1, 1] += m2[1, 1]

    a[i, j] = m1

    m3 = wp.mat22()
    m4 = d[i, j]

    m3[0, 0] -= m4[0, 0]
    m3[0, 1] -= m4[0, 1]
    m3[1, 0] -= m4[1, 0]
    m3[1, 1] -= m4[1, 1]

    c[i, j] = m3


class CompileModule:
    repeat = 10  # Number of samples to run
    number = 1  # Number of measurements to make between a single setup and teardown

    def setup(self):
        wp.init()
        wp.build.clear_kernel_cache()

    def teardown(self):
        matrix_augassign_kernel.module.unload()

    def time_cuda_codegen(self):
        wp.load_module(device="cuda:0")

    def time_cpu_codegen(self):
        wp.load_module(device="cpu")


class RunForwardKernel:
    def setup(self):
        wp.init()
        wp.build.clear_kernel_cache()
        wp.load_module(device="cuda:0")

        N = (1024, 1024)
        self.a = wp.zeros(N, dtype=wp.mat22, device="cuda:0")
        self.b = wp.ones(N, dtype=wp.mat22, device="cuda:0")
        self.c = wp.zeros(N, dtype=wp.mat22, device="cuda:0")
        self.d = wp.ones(N, dtype=wp.mat22, device="cuda:0")

        wp.synchronize_device("cuda:0")

    def track_cuda(self):
        with wp.ScopedTimer("benchmark", cuda_filter=wp.TIMING_KERNEL, synchronize=True) as timer:
            for _ in range(1000):
                wp.launch(
                    matrix_augassign_kernel, self.a.shape, inputs=[self.a, self.b, self.c, self.d], device="cuda:0"
                )

        average = sum(result.elapsed for result in timer.timing_results) / len(timer.timing_results)

        return average * 1e-3

    track_cuda.unit = "seconds"


class RunBackwardKernel:
    def setup(self):
        wp.init()
        wp.build.clear_kernel_cache()
        wp.load_module(device="cuda:0")

        N = (1024, 1024)
        self.a = wp.zeros(N, dtype=wp.mat22, device="cuda:0", requires_grad=True)
        self.b = wp.ones(N, dtype=wp.mat22, device="cuda:0", requires_grad=True)
        self.c = wp.zeros(N, dtype=wp.mat22, device="cuda:0", requires_grad=True)
        self.d = wp.ones(N, dtype=wp.mat22, device="cuda:0", requires_grad=True)

        wp.synchronize_device("cuda:0")

    def track_cuda(self):
        with wp.ScopedTimer("benchmark", cuda_filter=wp.TIMING_KERNEL, synchronize=True) as timer:
            for _ in range(1000):
                wp.launch(
                    matrix_augassign_kernel,
                    self.a.shape,
                    inputs=[self.a, self.b, self.c, self.d],
                    adj_inputs=[self.a.grad, self.b.grad, self.c.grad, self.d.grad],
                    adj_outputs=[],
                    adjoint=True,
                    device="cuda:0",
                )

        average = sum(result.elapsed for result in timer.timing_results) / len(timer.timing_results)

        return average * 1e-3

    track_cuda.unit = "seconds"
