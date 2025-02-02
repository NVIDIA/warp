# Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import numpy as np

import warp as wp


def create_test_kernel(tile_dim: int, storage_type: str):
    TILE = tile_dim

    @wp.kernel
    def load_store(a: wp.array2d(dtype=wp.float32), b: wp.array2d(dtype=wp.float32)):
        i, j = wp.tid()

        if wp.static(storage_type == "shared"):
            a_tile = wp.tile_load(a, shape=(TILE, TILE), offset=(i * TILE, j * TILE), storage="shared")
        else:
            a_tile = wp.tile_load(a, shape=(TILE, TILE), offset=(i * TILE, j * TILE), storage="register")

        wp.tile_store(b, a_tile, offset=(i * TILE, j * TILE))

    return load_store


class LoadStore:
    """Load a tile from global memory and then store it."""

    params = (["shared", "register"], [512, 2048, 8192])
    param_names = ["storage", "size"]

    def setup(self, storage, size):
        wp.init()
        wp.build.clear_kernel_cache()
        wp.set_module_options({"fast_math": True, "enable_backward": False})
        wp.load_module(device="cuda:0")

        self.tile_dim = 32
        self.block_dim = 128

        self.load_store = create_test_kernel(self.tile_dim, storage)

        rng = np.random.default_rng(42)

        self.a = wp.array(rng.random((size, size), dtype=np.float32), dtype=wp.float32, device="cuda:0")
        self.b = wp.empty_like(self.a)

        self.cmd = wp.launch_tiled(
            self.load_store,
            dim=(self.a.shape[0] // self.tile_dim, self.a.shape[1] // self.tile_dim),
            inputs=[self.a],
            outputs=[self.b],
            block_dim=self.block_dim,
            record_cmd=True,
        )
        # Warmup
        for _ in range(5):
            self.cmd.launch()

        wp.synchronize_device("cuda:0")

    def track_cuda(self, storage, size):
        with wp.ScopedTimer("benchmark", print=False, cuda_filter=wp.TIMING_KERNEL, synchronize=True) as timer:
            for _ in range(1000):
                self.cmd.launch()

        average = sum(result.elapsed for result in timer.timing_results) / len(timer.timing_results)

        return 2.0 * (self.a.capacity / (1024 * 1024 * 1024)) / (1e-3 * average)

    track_cuda.unit = "GiB/s"
