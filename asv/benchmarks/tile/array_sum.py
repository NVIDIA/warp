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

import numpy as np

import warp as wp


@wp.kernel
def array_sum_simt(
    a: wp.array2d(dtype=wp.float32),
    result: wp.array(dtype=wp.float32),
):
    i, j = wp.tid()

    local_val = 0.0

    if i < a.shape[0] and j < a.shape[1]:
        local_val = a[i, j]

    wp.atomic_add(result, 0, local_val)


@wp.kernel
def array_sum_tile(
    a: wp.array2d(dtype=wp.float32),
    result: wp.array(dtype=wp.float32),
):
    i, j = wp.tid()

    local_val = 0.0

    if i < a.shape[0] and j < a.shape[1]:
        local_val = a[i, j]

    t = wp.tile(local_val)
    s = wp.tile_sum(t)
    wp.tile_atomic_add(result, s)


class ArraySumSimt:
    """Atomically adds all array values using wp.atomic_add from all threads."""

    number = 100

    def setup(self):
        wp.init()
        wp.build.clear_kernel_cache()
        self.device = wp.get_device("cuda:0")
        wp.load_module(device=self.device)

        shape = (4096, 4096)

        rng = np.random.default_rng(42)

        self.a = wp.array(rng.random(shape, dtype=np.float32), dtype=wp.float32, device=self.device)
        self.result = wp.array([0.0], dtype=wp.float32, device=self.device)

        self.cmd = wp.launch(
            array_sum_simt,
            shape,
            inputs=[self.a],
            outputs=[self.result],
            device=self.device,
            record_cmd=True,
        )

        self.cmd.launch()
        wp.synchronize_device(self.device)

    def time_cuda(self):
        self.cmd.launch()
        wp.synchronize_device(self.device)


class ArraySumTile:
    """Atomically adds all array values using wp.tile_atomic_add with intermediate tile sum."""

    number = 100

    def setup(self):
        wp.init()
        wp.build.clear_kernel_cache()
        self.device = wp.get_device("cuda:0")
        wp.load_module(device=self.device)

        shape = (4096, 4096)

        rng = np.random.default_rng(42)

        self.a = wp.array(rng.random(shape, dtype=np.float32), dtype=wp.float32, device=self.device)
        self.result = wp.array([0.0], dtype=wp.float32, device=self.device)

        self.cmd = wp.launch(
            array_sum_tile,
            shape,
            inputs=[self.a],
            outputs=[self.result],
            device=self.device,
            record_cmd=True,
        )

        self.cmd.launch()

        wp.synchronize_device(self.device)

    def time_cuda(self):
        self.cmd.launch()
        wp.synchronize_device(self.device)
