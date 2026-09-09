# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: RUF059

import warp as wp

from ..benchmarks_utils import clear_kernel_cache, setup_once

wp.set_module_options({"enable_backward": False, "block_dim": 128})

TILE = 32


@wp.kernel
def cholesky(
    A: wp.array2d(dtype=wp.float64),
    L: wp.array2d(dtype=wp.float64),
    X: wp.array1d(dtype=wp.float64),
    Y: wp.array1d(dtype=wp.float64),
):
    k, j, _ = wp.tid()

    a = wp.tile_load(A, shape=(TILE, TILE))
    l = wp.tile_cholesky(a)
    wp.tile_store(L, l)

    x = wp.tile_load(X, shape=TILE)
    y = wp.tile_cholesky_solve(l, x)
    wp.tile_store(Y, y)


class ColdCompileCholeskyLTO:
    """Benchmark Cholesky solver compilation time from scratch."""

    repeat = 2  # Number of samples to run
    number = 1  # Number of timed calls per sample

    def setup(self):
        wp.init()
        clear_kernel_cache()
        wp.clear_lto_cache()

    def teardown(self):
        cholesky.module.unload()

    def time_cuda_codegen(self):
        wp.load_module(device="cuda:0")


class WarmCompileCholeskyLTO:
    """Benchmark Cholesky solver compilation time with cached LTO."""

    repeat = 1  # Number of samples to run
    number = 10  # Number of timed calls per sample

    @setup_once
    def setup(self):
        wp.init()
        clear_kernel_cache()
        wp.clear_lto_cache()
        wp.load_module(device="cuda:0")
        clear_kernel_cache()
        cholesky.module.unload()

    def teardown(self):
        pass

    def time_cuda_codegen(self):
        wp.load_module(device="cuda:0")
        clear_kernel_cache()
        cholesky.module.unload()
