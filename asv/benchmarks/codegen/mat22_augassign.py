# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from statistics import median

import warp as wp


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
    number = 1  # Number of measurements to make between a single setup and teardown

    def setup(self):
        wp.init()
        wp.build.clear_kernel_cache()

    def teardown(self):
        matrix_augassign_kernel.module.unload()
        wp.build.clear_kernel_cache()

    def time_cuda_codegen(self):
        wp.load_module(device="cuda:0")


class RunForwardKernel:
    def setup(self):
        wp.init()
        wp.build.clear_kernel_cache()
        wp.load_module(device="cuda:0")

        N = (1024, 1024)
        self.a = wp.zeros(N, dtype=wp.mat22, device="cuda:0")
        self.b = wp.ones(N, dtype=wp.mat22, device="cuda:0")

        wp.synchronize_device("cuda:0")

    def track_cuda(self):
        with wp.ScopedTimer("benchmark", print=False, cuda_filter=wp.TIMING_KERNEL, synchronize=True) as timer:
            for _ in range(1000):
                wp.launch(matrix_augassign_kernel, self.a.shape, inputs=[self.a, self.b], device="cuda:0")

        return median(result.elapsed for result in timer.timing_results) * 1e-3

    track_cuda.unit = "seconds"


class RunBackwardKernel:
    def setup(self):
        wp.init()
        wp.build.clear_kernel_cache()
        wp.load_module(device="cuda:0")

        N = (1024, 1024)
        self.a = wp.zeros(N, dtype=wp.mat22, device="cuda:0", requires_grad=True)
        self.b = wp.ones(N, dtype=wp.mat22, device="cuda:0", requires_grad=True)

        wp.synchronize_device("cuda:0")

    def track_cuda(self):
        with wp.ScopedTimer("benchmark", print=False, cuda_filter=wp.TIMING_KERNEL, synchronize=True) as timer:
            for _ in range(1000):
                wp.launch(
                    matrix_augassign_kernel,
                    self.a.shape,
                    inputs=[self.a, self.b],
                    adj_inputs=[self.a.grad, self.b.grad],
                    adj_outputs=[],
                    adjoint=True,
                    device="cuda:0",
                )

        return median(result.elapsed for result in timer.timing_results) * 1e-3

    track_cuda.unit = "seconds"
