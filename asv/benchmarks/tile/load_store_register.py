# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Register tile load/store across dimensions and dtypes."""

import numpy as np

import warp as wp


def _create_kernel_1d(tile_dim, dtype):
    TILE = tile_dim

    @wp.kernel
    def k(a: wp.array1d(dtype=dtype), b: wp.array1d(dtype=dtype)):
        i = wp.tid()
        t = wp.tile_load(a, shape=(TILE,), offset=(i * TILE,), storage="register")
        wp.tile_store(b, t, offset=(i * TILE,))

    return k


def _create_kernel_2d(tile_dim, dtype):
    TILE = tile_dim

    @wp.kernel
    def k(a: wp.array2d(dtype=dtype), b: wp.array2d(dtype=dtype)):
        i, j = wp.tid()
        t = wp.tile_load(a, shape=(TILE, TILE), offset=(i * TILE, j * TILE), storage="register")
        wp.tile_store(b, t, offset=(i * TILE, j * TILE))

    return k


def _create_kernel_3d(tile_dim, dtype):
    TILE = tile_dim

    @wp.kernel
    def k(a: wp.array3d(dtype=dtype), b: wp.array3d(dtype=dtype)):
        i, j, kk = wp.tid()
        t = wp.tile_load(a, shape=(TILE, TILE, TILE), offset=(i * TILE, j * TILE, kk * TILE), storage="register")
        wp.tile_store(b, t, offset=(i * TILE, j * TILE, kk * TILE))

    return k


_DTYPE_MAP = {
    "float32": wp.float32,
    "vec3": wp.vec3,
    "mat22": wp.types.matrix(shape=(2, 2), dtype=float),
}

_TARGET_BYTES = 256 * 1024 * 1024  # 256 MiB


class LoadStoreRegister:
    """Register tile load+store across dimensions and dtypes."""

    number = 200
    params = ([1, 2, 3], ["float32", "vec3", "mat22"])
    param_names = ["ndim", "dtype"]

    def setup(self, ndim, dtype_name):
        wp.init()
        wp.set_module_options({"fast_math": True, "enable_backward": False})
        self.device = wp.get_device("cuda:0")
        wp.load_module(device=self.device)

        dtype = _DTYPE_MAP[dtype_name]
        elem_bytes = wp.types.type_size_in_bytes(dtype)
        elem_floats = elem_bytes // 4
        block_dim = 64
        rng = np.random.default_rng(42)

        tile_dim = 64 if ndim == 1 else (32 if ndim == 2 else 8)

        if ndim == 1:
            n = _TARGET_BYTES // elem_bytes
            n = (n // tile_dim) * tile_dim
            shape = (n, elem_floats) if elem_floats > 1 else (n,)
            self.a = wp.array(rng.random(shape, dtype=np.float32), dtype=dtype, device=self.device)
            self.b = wp.empty_like(self.a)
            kernel = _create_kernel_1d(tile_dim, dtype)
            dim = (n // tile_dim,)
        elif ndim == 2:
            edge = int(np.sqrt(_TARGET_BYTES / elem_bytes))
            edge = (edge // tile_dim) * tile_dim
            shape = (edge, edge, elem_floats) if elem_floats > 1 else (edge, edge)
            self.a = wp.array(rng.random(shape, dtype=np.float32), dtype=dtype, device=self.device)
            self.b = wp.empty_like(self.a)
            kernel = _create_kernel_2d(tile_dim, dtype)
            dim = (edge // tile_dim, edge // tile_dim)
        else:
            edge = int(round((_TARGET_BYTES / elem_bytes) ** (1 / 3)))
            edge = max(edge, tile_dim)
            edge = (edge // tile_dim) * tile_dim
            shape = (edge, edge, edge, elem_floats) if elem_floats > 1 else (edge, edge, edge)
            self.a = wp.array(rng.random(shape, dtype=np.float32), dtype=dtype, device=self.device)
            self.b = wp.empty_like(self.a)
            kernel = _create_kernel_3d(tile_dim, dtype)
            dim = (edge // tile_dim, edge // tile_dim, edge // tile_dim)

        self.cmd = wp.launch_tiled(
            kernel,
            dim=dim,
            inputs=[self.a],
            outputs=[self.b],
            block_dim=block_dim,
            device=self.device,
            record_cmd=True,
        )
        self.cmd.launch()
        wp.synchronize_device(self.device)

    def time_cuda(self, ndim, dtype_name):
        self.cmd.launch()
        wp.synchronize_device(self.device)
