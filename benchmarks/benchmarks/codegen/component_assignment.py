# Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import warp as wp


@wp.kernel
def component_assignment(a: wp.array(dtype=wp.mat44)):
    tid = wp.tid()
    m = wp.mat44(0.0)
    for i in range(4):
        for j in range(4):
            if i == j:
                m[i, j] = 1.0
            else:
                m[i, j] = 0.0
    a[tid] = m


class CompileModule:
    repeat = 10  # Number of samples to run
    number = 1  # Number of measurements to make between a single setup and teardown

    def setup(self):
        wp.init()
        wp.build.clear_kernel_cache()

    def teardown(self):
        component_assignment.module.unload()

    def time_cuda_codegen(self):
        wp.load_module(device="cuda:0")

    def time_cpu_codegen(self):
        wp.load_module(device="cpu")
