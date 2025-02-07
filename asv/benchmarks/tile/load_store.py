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

    number = 500
    params = (["shared", "register"], [1024, 2048, 8192])
    param_names = ["storage", "size"]

    def setup(self, storage, size):
        wp.init()
        wp.build.clear_kernel_cache()
        wp.set_module_options({"fast_math": True, "enable_backward": False})
        self.device = wp.get_device("cuda:0")
        wp.load_module(device=self.device)

        self.tile_dim = 32
        self.block_dim = 128

        self.load_store = create_test_kernel(self.tile_dim, storage)

        rng = np.random.default_rng(42)

        self.a = wp.array(rng.random((size, size), dtype=np.float32), dtype=wp.float32, device=self.device)
        self.b = wp.empty_like(self.a)

        self.cmd = wp.launch_tiled(
            self.load_store,
            dim=(self.a.shape[0] // self.tile_dim, self.a.shape[1] // self.tile_dim),
            inputs=[self.a],
            outputs=[self.b],
            block_dim=self.block_dim,
            device=self.device,
            record_cmd=True,
        )
        # Warmup
        self.cmd.launch()

        wp.synchronize_device(self.device)

    def time_cuda(self, storage, size):
        self.cmd.launch()
        wp.synchronize_device(self.device)
