# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: RUF059

import warp as wp

from ..benchmarks_utils import clear_kernel_cache, setup_once

wp.set_module_options({"enable_backward": False, "block_dim": 8})

TILE_M = 1
TILE_N = 32


@wp.kernel
def fft_tiled(x: wp.array2d(dtype=wp.vec2d), y: wp.array2d(dtype=wp.vec2d)):
    i, j, _ = wp.tid()
    a = wp.tile_load(x, shape=(TILE_M, TILE_N))
    wp.tile_fft(a)
    wp.tile_ifft(a)
    wp.tile_store(y, a)


class ColdCompileFftLTO:
    """Benchmark FFT compilation time from scratch."""

    repeat = 2  # Number of samples to run
    number = 1  # Number of timed calls per sample

    def setup(self):
        wp.init()
        clear_kernel_cache()
        wp.clear_lto_cache()

    def teardown(self):
        fft_tiled.module.unload()

    def time_cuda_codegen(self):
        wp.load_module(device="cuda:0")


class WarmCompileFftLTO:
    """Benchmark FFT compilation time with cached LTO."""

    repeat = 1  # Number of samples to run
    number = 10  # Number of timed calls per sample

    @setup_once
    def setup(self):
        wp.init()
        clear_kernel_cache()
        wp.clear_lto_cache()
        wp.load_module(device="cuda:0")
        clear_kernel_cache()
        fft_tiled.module.unload()

    def teardown(self):
        pass

    def time_cuda_codegen(self):
        wp.load_module(device="cuda:0")
        clear_kernel_cache()
        fft_tiled.module.unload()
