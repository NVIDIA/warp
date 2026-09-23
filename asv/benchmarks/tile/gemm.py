# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import warp as wp

from ..benchmarks_utils import setup_once


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


def _setup_gemm(benchmark, size, tile_m, tile_n, tile_k, block_dim, device_name):
    benchmark.device = wp.get_device(device_name)
    benchmark.tile_m = tile_m
    benchmark.tile_n = tile_n
    benchmark.tile_k = tile_k
    benchmark.block_dim = block_dim
    benchmark.mlp = create_mlp_kernel(tile_m, tile_n, tile_k)

    rng = np.random.default_rng(42)

    benchmark.output = wp.zeros((size, size), dtype=float, device=benchmark.device)
    benchmark.a = wp.array(rng.random((size, size), dtype=np.float32), dtype=wp.float32, device=benchmark.device)
    benchmark.b = wp.array(rng.random((size, size), dtype=np.float32), dtype=wp.float32, device=benchmark.device)

    benchmark.cmd = wp.launch_tiled(
        kernel=benchmark.mlp,
        dim=[size // tile_m, size // tile_n],
        inputs=[benchmark.a, benchmark.b, size // tile_k, benchmark.output],
        block_dim=block_dim,
        record_cmd=True,
        device=benchmark.device,
    )
    benchmark.cmd.launch()
    wp.synchronize_device(benchmark.device)


class Gemm256CUDA:
    """Benchmark performance of M=N=K=256 GEMM."""

    @setup_once
    def setup(self):
        wp.init()
        _setup_gemm(self, 256, 16, 16, 64, 64, "cuda:0")

    def time_cuda(self):
        self.cmd.launch()
        wp.synchronize_device(self.device)


class Gemm256CPU:
    """Benchmark performance of M=N=K=256 GEMM on CPU."""

    number = 1

    @setup_once
    def setup(self):
        wp.init()
        wp.config.enable_cpu_blocks = True
        _setup_gemm(self, 256, 16, 16, 64, 64, "cpu")

    def time_cpu(self):
        self.cmd.launch()


class Gemm1024CUDA:
    """Benchmark performance of M=N=K=1024 GEMM."""

    number = 1000

    @setup_once
    def setup(self):
        wp.init()
        wp.set_module_options({"fast_math": True, "enable_backward": False})
        _setup_gemm(self, 1024, 64, 64, 64, 128, "cuda:0")

    def time_cuda(self):
        self.cmd.launch()
        wp.synchronize_device(self.device)


class Gemm1024CPU:
    """Benchmark performance of M=N=K=1024 GEMM on CPU."""

    number = 1

    @setup_once
    def setup(self):
        wp.init()
        wp.set_module_options({"fast_math": True, "enable_backward": False})
        wp.config.enable_cpu_blocks = True
        _setup_gemm(self, 1024, 64, 64, 64, 128, "cpu")

    def time_cpu(self):
        self.cmd.launch()
