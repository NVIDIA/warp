# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
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


class MlpGemm128:
    """Benchmark performance of M=N=K=128 GEMM."""

    def setup(self):
        wp.init()
        wp.build.clear_kernel_cache()
        wp.set_module_options({"fast_math": True, "enable_backward": False})
        wp.load_module(device="cuda:0")

        # Parameters found by auto-tuning for a 128x128 GEMM
        self.tile_m = 16
        self.tile_n = 16
        self.tile_k = 64
        self.block_dim = 256

        self.mlp = create_mlp_kernel(self.tile_m, self.tile_n, self.tile_k)

        rng = np.random.default_rng(42)

        self.output = wp.zeros((128, 128), dtype=float, device="cuda:0")
        self.a = wp.array(rng.random((128, 128), dtype=np.float32), dtype=wp.float32, device="cuda:0")
        self.b = wp.array(rng.random((128, 128), dtype=np.float32), dtype=wp.float32, device="cuda:0")

        self.cmd = wp.launch_tiled(
            kernel=self.mlp,
            dim=[128 // self.tile_m, 128 // self.tile_n],
            inputs=[self.a, self.b, 128 // self.tile_k, self.output],
            block_dim=self.block_dim,
            record_cmd=True,
            device="cuda:0",
        )

        # warm-up
        for _ in range(5):
            self.cmd.launch()

        wp.synchronize_device("cuda:0")

    def track_cuda(self):
        with wp.ScopedTimer("benchmark", print=False, cuda_filter=wp.TIMING_KERNEL, synchronize=True) as timer:
            for _ in range(100):
                self.cmd.launch()

        average = sum(result.elapsed for result in timer.timing_results) / len(timer.timing_results)

        return average * 1e-3

    track_cuda.unit = "seconds"


class MlpGemm128:
    """Benchmark performance of M=N=K=128 GEMM."""

    def setup(self):
        wp.init()
        wp.build.clear_kernel_cache()
        wp.set_module_options({"fast_math": True, "enable_backward": False})
        wp.load_module(device="cuda:0")

        # Parameters found by auto-tuning for a 128x128 GEMM
        self.tile_m = 16
        self.tile_n = 16
        self.tile_k = 64
        self.block_dim = 256

        self.mlp = create_mlp_kernel(self.tile_m, self.tile_n, self.tile_k)

        rng = np.random.default_rng(42)

        self.output = wp.zeros((128, 128), dtype=float, device="cuda:0")
        self.a = wp.array(rng.random((128, 128), dtype=np.float32), dtype=wp.float32, device="cuda:0")
        self.b = wp.array(rng.random((128, 128), dtype=np.float32), dtype=wp.float32, device="cuda:0")

        self.cmd = wp.launch_tiled(
            kernel=self.mlp,
            dim=[128 // self.tile_m, 128 // self.tile_n],
            inputs=[self.a, self.b, 128 // self.tile_k, self.output],
            block_dim=self.block_dim,
            record_cmd=True,
            device="cuda:0",
        )

        # warm-up
        for _ in range(5):
            self.cmd.launch()

        wp.synchronize_device("cuda:0")

    def track_cuda(self):
        with wp.ScopedTimer("benchmark", print=False, cuda_filter=wp.TIMING_KERNEL, synchronize=True) as timer:
            for _ in range(100):
                self.cmd.launch()

        average = sum(result.elapsed for result in timer.timing_results) / len(timer.timing_results)

        return average * 1e-3

    track_cuda.unit = "seconds"


class MlpGemm1024:
    """Benchmark performance of M=N=K=1024 GEMM."""

    def setup(self):
        wp.init()
        wp.build.clear_kernel_cache()
        wp.set_module_options({"fast_math": True, "enable_backward": False})
        wp.load_module(device="cuda:0")

        self.tile_m = 64
        self.tile_n = 64
        self.tile_k = 64
        self.block_dim = 128

        self.mlp = create_mlp_kernel(self.tile_m, self.tile_n, self.tile_k)

        rng = np.random.default_rng(42)

        self.output = wp.zeros((1024, 1024), dtype=float, device="cuda:0")
        self.a = wp.array(rng.random((1024, 1024), dtype=np.float32), dtype=wp.float32, device="cuda:0")
        self.b = wp.array(rng.random((1024, 1024), dtype=np.float32), dtype=wp.float32, device="cuda:0")

        self.cmd = wp.launch_tiled(
            kernel=self.mlp,
            dim=[1024 // self.tile_m, 1024 // self.tile_n],
            inputs=[self.a, self.b, 1024 // self.tile_k, self.output],
            block_dim=self.block_dim,
            record_cmd=True,
            device="cuda:0",
        )

        # warm-up
        for _ in range(5):
            self.cmd.launch()

        wp.synchronize_device("cuda:0")

    def track_cuda(self):
        with wp.ScopedTimer("benchmark", print=False, cuda_filter=wp.TIMING_KERNEL, synchronize=True) as timer:
            for _ in range(100):
                self.cmd.launch()

        average = sum(result.elapsed for result in timer.timing_results) / len(timer.timing_results)

        return average * 1e-3

    track_cuda.unit = "seconds"
