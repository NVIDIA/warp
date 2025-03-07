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


def create_mlp_kernel(m, n, k):
    TILE_M = m
    TILE_N = n
    TILE_K = k

    @wp.kernel
    def mlp(x: wp.array2d(dtype=float), weights_wp: wp.array2d(dtype=float), n_k: int, output: wp.array2d(dtype=float)):
        i_m, i_n = wp.tid()
        sum = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=wp.float32)
        for count in range(n_k):
            feat = wp.tile_load(x, shape=(TILE_M, TILE_K), offset=(i_m * TILE_M, count * TILE_K))
            weight = wp.tile_load(weights_wp, shape=(TILE_K, TILE_N), offset=(count * TILE_K, i_n * TILE_N))
            wp.tile_matmul(feat, weight, sum)

        wp.tile_store(output, sum, offset=(i_m * TILE_M, i_n * TILE_N))

    return mlp


class Gemm256:
    """Benchmark performance of M=N=K=256 GEMM."""

    def setup(self):
        wp.init()
        wp.build.clear_kernel_cache()
        self.device = wp.get_device("cuda:0")
        wp.load_module(device=self.device)

        # Parameters found by auto-tuning for a 256x256 GEMM
        self.tile_m = 16
        self.tile_n = 16
        self.tile_k = 64
        self.block_dim = 64

        self.mlp = create_mlp_kernel(self.tile_m, self.tile_n, self.tile_k)

        rng = np.random.default_rng(42)

        self.output = wp.zeros((256, 256), dtype=float, device=self.device)
        self.a = wp.array(rng.random((256, 256), dtype=np.float32), dtype=wp.float32, device=self.device)
        self.b = wp.array(rng.random((256, 256), dtype=np.float32), dtype=wp.float32, device=self.device)

        self.cmd = wp.launch_tiled(
            kernel=self.mlp,
            dim=[256 // self.tile_m, 256 // self.tile_n],
            inputs=[self.a, self.b, 256 // self.tile_k, self.output],
            block_dim=self.block_dim,
            record_cmd=True,
            device=self.device,
        )
        # warm-up
        self.cmd.launch()
        wp.synchronize_device(self.device)

    def time_cuda(self):
        self.cmd.launch()
        wp.synchronize_device(self.device)


class Gemm1024:
    """Benchmark performance of M=N=K=1024 GEMM."""

    number = 1000

    def setup(self):
        wp.init()
        wp.build.clear_kernel_cache()
        wp.set_module_options({"fast_math": True, "enable_backward": False})
        self.device = wp.get_device("cuda:0")
        wp.load_module(device=self.device)

        self.tile_m = 64
        self.tile_n = 64
        self.tile_k = 64
        self.block_dim = 128

        self.mlp = create_mlp_kernel(self.tile_m, self.tile_n, self.tile_k)

        rng = np.random.default_rng(42)

        self.output = wp.zeros((1024, 1024), dtype=float, device=self.device)
        self.a = wp.array(rng.random((1024, 1024), dtype=np.float32), dtype=wp.float32, device=self.device)
        self.b = wp.array(rng.random((1024, 1024), dtype=np.float32), dtype=wp.float32, device=self.device)

        self.cmd = wp.launch_tiled(
            kernel=self.mlp,
            dim=[1024 // self.tile_m, 1024 // self.tile_n],
            inputs=[self.a, self.b, 1024 // self.tile_k, self.output],
            block_dim=self.block_dim,
            record_cmd=True,
            device=self.device,
        )

        # warm-up
        self.cmd.launch()

        wp.synchronize_device(self.device)

    def time_cuda(self):
        self.cmd.launch()
        wp.synchronize_device(self.device)
