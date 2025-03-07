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
N = 128


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
        self.device = wp.get_device("cuda:0")
        wp.load_module(device=self.device)
        self.test_array = wp.ones((N, N, N, NUM_COMPONENTS), dtype=float, device=self.device)
        self.cmd = wp.launch(kernel_aos, (N, N, N), inputs=[self.test_array], record_cmd=True, device=self.device)
        # Warmup
        self.cmd.launch()
        wp.synchronize_device(self.device)

    def time_kernels(self):
        self.cmd.launch()
        wp.synchronize_device(self.device)


class StructureOfArrays:
    """Benchmark for measuring read-and-write to data organized as a structure of arrays."""

    number = 100

    def setup(self):
        wp.init()
        wp.build.clear_kernel_cache()
        self.device = wp.get_device("cuda:0")
        wp.load_module(device=self.device)
        self.test_array = wp.ones((NUM_COMPONENTS, N, N, N), dtype=float, device=self.device)
        self.cmd = wp.launch(kernel_soa, (N, N, N), inputs=[self.test_array], record_cmd=True, device=self.device)
        # Warmup
        self.cmd.launch()
        wp.synchronize_device(self.device)

    def time_kernels(self):
        self.cmd.launch()
        wp.synchronize_device(self.device)


@wp.kernel
def load_store_ij(test_array_in: wp.array2d(dtype=float), test_array_out: wp.array2d(dtype=float)):
    i, j = wp.tid()

    test_array_out[i, j] = test_array_in[i, j]


@wp.kernel
def load_store_ji(test_array_in: wp.array2d(dtype=float), test_array_out: wp.array2d(dtype=float)):
    i, j = wp.tid()

    test_array_out[j, i] = test_array_in[j, i]


class LoadStoreIJ:
    """Benchmark for measuring the performance of loading and storing 2-D array data as [i,j]."""

    number = 100

    def setup(self):
        wp.init()
        wp.build.clear_kernel_cache()
        self.device = wp.get_device("cuda:0")
        wp.load_module(device=self.device)
        self.input_array = wp.ones((8192, 4096), dtype=float, device=self.device)
        self.output_array = wp.empty_like(self.input_array)

        self.cmd = wp.launch(
            load_store_ij,
            (8192, 4096),
            inputs=[self.input_array],
            outputs=[self.output_array],
            device=self.device,
            record_cmd=True,
        )
        # Warmup
        self.cmd.launch()
        wp.synchronize_device(self.device)

    def time_cuda(self):
        self.cmd.launch()
        wp.synchronize_device(self.device)


class LoadStoreJI:
    """Benchmark for measuring the performance of loading and storing 2-D array data as [j,i]."""

    number = 100

    def setup(self):
        wp.init()
        wp.build.clear_kernel_cache()
        self.device = wp.get_device("cuda:0")
        wp.load_module(device=self.device)
        self.input_array = wp.ones((8192, 4096), dtype=float, device=self.device)
        self.output_array = wp.empty_like(self.input_array)

        self.cmd = wp.launch(
            load_store_ji,
            (4096, 8192),
            inputs=[self.input_array],
            outputs=[self.output_array],
            device=self.device,
            record_cmd=True,
        )
        # Warmup
        self.cmd.launch()
        wp.synchronize_device(self.device)

    def time_cuda(self):
        self.cmd.launch()
        wp.synchronize_device(self.device)
