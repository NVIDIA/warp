# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared tile load/store via the coalesced byte-copy path.

Uses 1D tiles of large element types where coalescing provides the biggest
benefit. The coalesced path activates for contiguous, in-bounds tiles on CUDA.
Sweeps element size to show the crossover where coalescing helps (mat33+).
"""

import numpy as np

import warp as wp

_DTYPE_MAP = {
    "float32": wp.float32,
    "mat33": wp.mat33,
    "mat44": wp.mat44,
    "mat66": wp.types.matrix(shape=(6, 6), dtype=float),
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


class LoadStoreCoalesced:
    """Shared 1D tile load+store for large element types (coalesced path)."""

    number = 500
    params = ([8, 16, 32], ["float32", "mat33", "mat44", "mat66"])
    param_names = ["tile_size", "dtype"]

    def setup(self, tile_size, dtype_name):
        wp.init()
        wp.set_module_options({"fast_math": True, "enable_backward": False})
        self.device = wp.get_device("cuda:0")
        wp.load_module(device=self.device)

        dtype = _DTYPE_MAP[dtype_name]
        elem_bytes = wp.types.type_size_in_bytes(dtype)
        elem_floats = elem_bytes // 4
        block_dim = 128
        rng = np.random.default_rng(42)

        n = _TARGET_BYTES // elem_bytes
        n = (n // tile_size) * tile_size

        shape = (n, elem_floats) if elem_floats > 1 else (n,)
        self.a = wp.array(rng.random(shape, dtype=np.float32), dtype=dtype, device=self.device)
        self.b = wp.empty_like(self.a)
        kernel = _create_kernel_1d(tile_size, dtype)
        dim = (n // tile_size,)

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

    def time_cuda(self, tile_size, dtype_name):
        self.cmd.launch()
        wp.synchronize_device(self.device)
