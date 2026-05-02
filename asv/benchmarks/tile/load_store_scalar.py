# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared tile load/store via the scalar fallback path.

Forces the scalar path by using non-float4-aligned tile sizes (not divisible
by 16 bytes in the last dimension). Ensures no regressions in the fallback
path across dimensions and dtypes.
"""

import numpy as np

import warp as wp

# Non-POT tile sizes that are NOT float4-aligned for float32 (last_dim * 4 % 16 != 0)
_TILE_DIMS = {1: 17, 2: 17, 3: 7}

_DTYPE_MAP = {
    "float32": wp.float32,
    "vec3": wp.vec3,
}

_TARGET_BYTES = 16 * 1024 * 1024  # 16 MiB


def _create_kernel_1d(tile_dim, dtype):
    TILE = tile_dim

    @wp.kernel
    def k(a: wp.array1d(dtype=dtype), b: wp.array1d(dtype=dtype)):
        i = wp.tid()
        t = wp.tile_load(a, shape=(TILE,), offset=(i * TILE,), storage="shared")
        wp.tile_store(b, t, offset=(i * TILE,))

    return k


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


class LoadStoreScalar:
    """Shared tile load+store via scalar fallback (non-aligned tile sizes)."""

    number = 500
    params = ([1, 2, 3], ["float32", "vec3"])
    param_names = ["ndim", "dtype"]

    def setup(self, ndim, dtype_name):
        wp.init()
        wp.set_module_options({"fast_math": True, "enable_backward": False})
        self.device = wp.get_device("cuda:0")
        wp.load_module(device=self.device)

        dtype = _DTYPE_MAP[dtype_name]
        elem_bytes = wp.types.type_size_in_bytes(dtype)
        elem_floats = elem_bytes // 4
        tile_dim = _TILE_DIMS[ndim]
        block_dim = 128
        rng = np.random.default_rng(42)

        if ndim == 1:
            # Slice [::2] to create a strided (non-contiguous) view that forces
            # the scalar path. stride = 2*sizeof(T) != sizeof(T).
            n = _TARGET_BYTES // elem_bytes
            n = (n // tile_dim) * tile_dim
            shape = (n * 2, elem_floats) if elem_floats > 1 else (n * 2,)
            base = wp.array(rng.random(shape, dtype=np.float32), dtype=dtype, device=self.device)
            self.a = base[::2]
            base_out = wp.array(rng.random(shape, dtype=np.float32), dtype=dtype, device=self.device)
            self.b = base_out[::2]
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
