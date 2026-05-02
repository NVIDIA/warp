# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared tile load/store via the vectorized float4 path (2D+ only).

Uses float4-aligned tile dimensions to ensure the vectorized code path is taken.
Compare against main with ``asv compare`` to measure 3D/4D improvements
(main had 2D vectorized; this branch extends to all dimensions).
"""

import numpy as np

import warp as wp


def _create_kernel_2d(tile_dim, dtype):
    TILE = tile_dim

    @wp.kernel
    def k(a: wp.array2d(dtype=dtype), b: wp.array2d(dtype=dtype)):
        i, j = wp.tid()
        t = wp.tile_load(a, shape=(TILE, TILE), offset=(i * TILE, j * TILE), storage="shared")
        wp.tile_store(b, t, offset=(i * TILE, j * TILE))

    return k


def _create_kernel_3d(tile_dim, dtype):
    TILE = tile_dim

    @wp.kernel
    def k(a: wp.array3d(dtype=dtype), b: wp.array3d(dtype=dtype)):
        i, j, kk = wp.tid()
        t = wp.tile_load(a, shape=(TILE, TILE, TILE), offset=(i * TILE, j * TILE, kk * TILE), storage="shared")
        wp.tile_store(b, t, offset=(i * TILE, j * TILE, kk * TILE))

    return k


def _align_unit(elem_bytes):
    """Smallest n such that n * elem_bytes % 16 == 0."""
    from math import gcd  # noqa: PLC0415

    return 16 // gcd(16, elem_bytes)


def _tile_dim(ndim, dtype_name):
    """Return a float4-aligned tile dimension for the given dtype."""
    elem_bytes = {"float32": 4, "vec3": 12, "mat22": 16}[dtype_name]
    au = _align_unit(elem_bytes)

    if ndim == 2:
        aligned_dim = max(128 // elem_bytes, au)
    else:
        aligned_dim = max(64 // elem_bytes, au)
    return ((aligned_dim + au - 1) // au) * au


_DTYPE_MAP = {
    "float32": wp.float32,
    "vec3": wp.vec3,
    "mat22": wp.types.matrix(shape=(2, 2), dtype=float),
}


class LoadStoreVectorized:
    """Shared tile load+store via the vectorized float4 path."""

    number = 500
    params = ([1024, 2048], [2, 3], ["float32", "vec3", "mat22"])
    param_names = ["size", "ndim", "dtype"]

    def setup(self, size, ndim, dtype_name):
        wp.init()
        wp.set_module_options({"fast_math": True, "enable_backward": False})
        self.device = wp.get_device("cuda:0")
        wp.load_module(device=self.device)

        dtype = _DTYPE_MAP[dtype_name]
        tile_dim = _tile_dim(ndim, dtype_name)
        elem_bytes = wp.types.type_size_in_bytes(dtype)
        elem_floats = elem_bytes // 4
        block_dim = 128
        rng = np.random.default_rng(42)

        if ndim == 2:
            edge = (size // tile_dim) * tile_dim
            shape = (edge, edge, elem_floats) if elem_floats > 1 else (edge, edge)
            self.a = wp.array(rng.random(shape, dtype=np.float32), dtype=dtype, device=self.device)
            self.b = wp.empty_like(self.a)
            kernel = _create_kernel_2d(tile_dim, dtype)
            dim = (edge // tile_dim, edge // tile_dim)
        else:
            edge = int(round(size ** (2 / 3)))
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

    def time_cuda(self, size, ndim, dtype_name):
        self.cmd.launch()
        wp.synchronize_device(self.device)
