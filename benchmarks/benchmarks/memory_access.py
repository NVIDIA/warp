# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import warp as wp

wp.set_module_options({"enable_backward": False})


NUM_COMPONENTS = 19
N = 256


@wp.kernel
def kernel_aos(test_array: wp.array4d(dtype=float)):
    i, j, k = wp.tid()

    for component_index in range(NUM_COMPONENTS):
        test_array[i, j, k, component_index] = 1.00001 * test_array[i, j, k, component_index]


@wp.kernel
def kernel_soa(test_array: wp.array4d(dtype=float)):
    i, j, k = wp.tid()

    for component_index in range(NUM_COMPONENTS):
        test_array[component_index, i, j, k] = 1.00001 * test_array[component_index, i, j, k]


class ArrayOfStructures:
    """Benchmark for measuring read-and-write to data organized as an array of structures."""

    number = 100

    def setup(self):
        wp.init()
        wp.build.clear_kernel_cache()
        wp.load_module("cuda:0")
        self.test_array = wp.ones((N, N, N, NUM_COMPONENTS), dtype=float, device="cuda:0")
        self.cmd = wp.launch(kernel_aos, (N, N, N), inputs=[self.test_array], record_cmd=True, device="cuda:0")

    def time_kernels(self):
        self.cmd.launch()
        wp.synchronize_device("cuda:0")


class StructureOfArrays:
    """Benchmark for measuring read-and-write to data organized as a structure of arrays."""

    number = 100

    def setup(self):
        wp.init()
        wp.build.clear_kernel_cache()
        wp.load_module("cuda:0")
        self.test_array = wp.ones((NUM_COMPONENTS, N, N, N), dtype=float, device="cuda:0")
        self.cmd = wp.launch(kernel_soa, (N, N, N), inputs=[self.test_array], record_cmd=True, device="cuda:0")

    def time_kernels(self):
        self.cmd.launch()
        wp.synchronize_device("cuda:0")
