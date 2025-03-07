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

"""Compare GEMM performance between Torch and Warp (Tiled).

This script can be used to identify optimal tile parameters for a fixed-size
matrix multiplication.
"""

from itertools import product
from statistics import mean, stdev
from typing import List

import numpy as np
import torch

import warp as wp


# returns a kernel to compute a GEMM given m,n,k tile sizes
def create_gemm_kernel(m, n, k):
    TILE_M = m
    TILE_N = n
    TILE_K = k

    @wp.kernel
    def gemm(A: wp.array2d(dtype=float), B: wp.array2d(dtype=float), output: wp.array2d(dtype=float)):
        i, j = wp.tid()
        sum = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=wp.float32)

        count = A.shape[1] // TILE_K

        for k in range(count):
            a = wp.tile_load(A, shape=(TILE_M, TILE_K), offset=(i * TILE_M, k * TILE_K))
            b = wp.tile_load(B, shape=(TILE_K, TILE_N), offset=(k * TILE_K, j * TILE_N))

            wp.tile_matmul(a, b, sum)

        wp.tile_store(output, sum, offset=(i * TILE_M, j * TILE_N))

    return gemm


def benchmark_torch(A: torch.Tensor, B: torch.Tensor, warm_up: int, iterations: int):
    # warm-up
    for _ in range(warm_up):
        torch.matmul(A, B)

    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    timing_results = []

    for _i in range(iterations):
        start_event.record()
        torch.matmul(A, B)
        end_event.record()

        torch.cuda.synchronize()
        timing_results.append(start_event.elapsed_time(end_event))

    return mean(timing_results), stdev(timing_results)


def benchmark_warp(A: wp.array, B: wp.array, config: List[int], warm_up: int, iterations: int):
    TILE_M = config[0]
    TILE_N = config[1]
    TILE_K = config[2]
    BLOCK_DIM = config[3]

    mlp = create_gemm_kernel(TILE_M, TILE_N, TILE_K)

    M = A.shape[0]
    N = B.shape[1]

    output = wp.zeros((M, N), dtype=float)

    # create launch command
    cmd = wp.launch_tiled(
        kernel=mlp,
        dim=[M // TILE_M, N // TILE_N],
        inputs=[A, B, output],
        block_dim=BLOCK_DIM,
        record_cmd=True,
    )

    # warm-up
    for _ in range(warm_up):
        cmd.launch()

    # check output
    if warm_up > 0:
        try:
            np.testing.assert_allclose(output.numpy(), A.numpy() @ B.numpy(), atol=1e-3, rtol=1e-3)
        except AssertionError as e:
            print(f"Failed with {TILE_M=}, {TILE_N=}, {TILE_K=}, {BLOCK_DIM=}")
            raise e

    # benchmark
    with wp.ScopedTimer("warp", print=False, synchronize=True, cuda_filter=wp.TIMING_KERNEL) as timer:
        for _ in range(iterations):
            cmd.launch()

    timing_results = [result.elapsed for result in timer.timing_results]

    return mean(timing_results), stdev(timing_results)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False  # Disable TF32 for matrix multiplications
    torch.backends.cudnn.allow_tf32 = False  # Disable TF32 for cuDNN operations

    wp.init()
    wp.clear_kernel_cache()
    wp.set_module_options({"fast_math": True, "enable_backward": False})

    tile_m = [8, 16, 32, 64]
    tile_n = [8, 16, 32, 64]
    tile_k = [8, 16, 64]
    block = [32, 64, 128]

    M = 1024
    N = 1024
    K = 1024
    print(f"{M=}, {N=}, {K=}")

    A = torch.randn(M, K).cuda()
    B = torch.randn(K, N).cuda()

    iterations = 100
    warm_up = 5

    time_torch_mean, time_torch_std = benchmark_torch(A, B, warm_up, iterations)
    print(f"Torch: {time_torch_mean:.6g}±{time_torch_std:.2g} ms")

    configs = list(product(tile_m, tile_n, tile_k, block))

    wp.config.quiet = True

    # header
    print(
        f"{'TILE_M':<8s} {'TILE_N':<8s} {'TILE_K':<8s} {'BLOCK':<8s} {'Time (ms)':<10s} {'Std dev (ms)':<14s} {'Warp/Torch':<12s}"
    )
    print("-" * 79)

    for c in configs:
        time_mean, time_std = benchmark_warp(wp.from_torch(A), wp.from_torch(B), c, warm_up, iterations)
        print(
            f"{c[0]:<8d} {c[1]:<8d} {c[2]:<8d} {c[3]:<8d} {time_mean:<10.6g} {time_std:<#14.2g} {time_mean / time_torch_mean:<12.6g}"
        )
