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


def create_test_kernel_2d(tile_dim: int, storage_type: str, dtype):
    TILE = tile_dim

    @wp.kernel
    def load_store(a: wp.array2d(dtype=dtype), b: wp.array2d(dtype=dtype)):
        i, j = wp.tid()

        if wp.static(storage_type == "shared"):
            a_tile = wp.tile_load(a, shape=(TILE, TILE), offset=(i * TILE, j * TILE), storage="shared")
        else:
            a_tile = wp.tile_load(a, shape=(TILE, TILE), offset=(i * TILE, j * TILE), storage="register")

        wp.tile_store(b, a_tile, offset=(i * TILE, j * TILE))

    return load_store


def create_test_kernel_1d(tile_dim: int, storage_type: str, dtype):
    TILE = tile_dim

    @wp.kernel
    def load_store(a: wp.array1d(dtype=dtype), b: wp.array1d(dtype=dtype)):
        i = wp.tid()

        if wp.static(storage_type == "shared"):
            a_tile = wp.tile_load(a, shape=(TILE,), offset=(i * TILE,), storage="shared")
        else:
            a_tile = wp.tile_load(a, shape=(TILE,), offset=(i * TILE,), storage="register")

        wp.tile_store(b, a_tile, offset=(i * TILE,))

    return load_store


def create_test_kernel_3d(tile_dim: int, storage_type: str, dtype):
    TILE = tile_dim

    @wp.kernel
    def load_store(a: wp.array3d(dtype=dtype), b: wp.array3d(dtype=dtype)):
        i, j, k = wp.tid()

        if wp.static(storage_type == "shared"):
            a_tile = wp.tile_load(a, shape=(TILE, TILE, TILE), offset=(i * TILE, j * TILE, k * TILE), storage="shared")
        else:
            a_tile = wp.tile_load(
                a, shape=(TILE, TILE, TILE), offset=(i * TILE, j * TILE, k * TILE), storage="register"
            )

        wp.tile_store(b, a_tile, offset=(i * TILE, j * TILE, k * TILE))

    return load_store


_DTYPE_MAP = {
    "float32": wp.float32,
    "vec2": wp.vec2,
    "vec4": wp.vec4f,
}


class LoadStore:
    """Load a tile from global memory and then store it."""

    number = 500
    params = (["shared", "register"], [1024, 2048], [1, 2, 3], ["float32", "vec2", "vec4"])
    param_names = ["storage", "size", "ndim", "dtype"]

    def setup(self, storage, size, ndim, dtype_name):
        wp.init()
        wp.set_module_options({"fast_math": True, "enable_backward": False})
        self.device = wp.get_device("cuda:0")
        wp.load_module(device=self.device)

        dtype = _DTYPE_MAP[dtype_name]
        self.tile_dim = 32 if ndim <= 2 else 8
        self.block_dim = 128

        rng = np.random.default_rng(42)
        elem_size = wp.types.type_size_in_bytes(dtype) // 4  # number of float32 per element

        if ndim == 1:
            n = size * size  # total elements to match 2D area
            shape = (n, elem_size) if elem_size > 1 else (n,)
            self.a = wp.array(rng.random(shape, dtype=np.float32), dtype=dtype, device=self.device)
            self.b = wp.empty_like(self.a)
            self.load_store = create_test_kernel_1d(self.tile_dim, storage, dtype)
            dim = (self.a.shape[0] // self.tile_dim,)
        elif ndim == 2:
            shape = (size, size, elem_size) if elem_size > 1 else (size, size)
            self.a = wp.array(rng.random(shape, dtype=np.float32), dtype=dtype, device=self.device)
            self.b = wp.empty_like(self.a)
            self.load_store = create_test_kernel_2d(self.tile_dim, storage, dtype)
            dim = (self.a.shape[0] // self.tile_dim, self.a.shape[1] // self.tile_dim)
        else:  # ndim == 3
            edge = int(round(size ** (2 / 3)))
            edge = max(edge, self.tile_dim)
            edge = (edge // self.tile_dim) * self.tile_dim
            shape = (edge, edge, edge, elem_size) if elem_size > 1 else (edge, edge, edge)
            self.a = wp.array(rng.random(shape, dtype=np.float32), dtype=dtype, device=self.device)
            self.b = wp.empty_like(self.a)
            self.load_store = create_test_kernel_3d(self.tile_dim, storage, dtype)
            dim = (edge // self.tile_dim, edge // self.tile_dim, edge // self.tile_dim)

        self.cmd = wp.launch_tiled(
            self.load_store,
            dim=dim,
            inputs=[self.a],
            outputs=[self.b],
            block_dim=self.block_dim,
            device=self.device,
            record_cmd=True,
        )
        # Warmup
        self.cmd.launch()

        wp.synchronize_device(self.device)

    def time_cuda(self, storage, size, ndim, dtype_name):
        self.cmd.launch()
        wp.synchronize_device(self.device)
