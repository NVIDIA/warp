# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import warp as wp

wp.set_module_options({"enable_backward": False, "block_dim": 64})

# tile size
TILE_M = wp.constant(8)
TILE_N = wp.constant(4)
TILE_K = wp.constant(8)


@wp.kernel
def tile_gemm(A: wp.array2d(dtype=wp.float32), B: wp.array2d(dtype=wp.float16), C: wp.array2d(dtype=wp.float64)):
    # output tile index
    i, j = wp.tid()

    sum = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=wp.float64)

    _M = A.shape[0]
    _N = B.shape[1]
    K = A.shape[1]

    count = int(K / TILE_K)

    for k in range(0, count):
        a = wp.tile_load(A, shape=(TILE_M, TILE_K), offset=(i * TILE_M, k * TILE_K))
        b = wp.tile_load(B, shape=(TILE_K, TILE_N), offset=(k * TILE_K, j * TILE_N))

        # sum += a*b
        wp.tile_matmul(a, b, sum)

    wp.tile_store(C, sum, offset=(i * TILE_M, j * TILE_N))


class ColdCompileGemmLTO:
    """Benchmark GEMM compilation time from scratch."""

    repeat = 2  # Number of samples to run
    number = 1  # Number of measurements to make between a single setup and teardown
    timeout = 120.0

    def setup(self):
        wp.init()
        wp.build.clear_kernel_cache()
        wp.build.clear_lto_cache()

    def teardown(self):
        tile_gemm.module.unload()

    def time_cuda_codegen(self):
        wp.load_module(device="cuda:0")


class WarmCompileGemmLTO:
    """Benchmark GEMM compilation time with cached LTO."""

    repeat = 1  # Number of samples to run
    number = 10  # Number of measurements to make between a single setup and teardown
    timeout = 120.0

    def setup(self):
        wp.init()
        wp.build.clear_kernel_cache()
        wp.build.clear_lto_cache()
        wp.load_module(device="cuda:0")
        wp.build.clear_kernel_cache()
        tile_gemm.module.unload()

    def teardown(self):
        pass

    def time_cuda_codegen(self):
        wp.load_module(device="cuda:0")
        wp.build.clear_kernel_cache()
        tile_gemm.module.unload()
