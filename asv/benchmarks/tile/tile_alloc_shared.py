# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared tile allocation+store: ``tile_zeros`` vs ``tile_empty``.

Each kernel allocates a shared tile and stores it to global memory so the
shared allocation cannot be optimized away. The only difference between the
two kernels is the parallel block-wide zero-fill that ``tile_zeros`` emits
and ``tile_empty`` skips, so the measured delta attributes that cost.
"""

import warp as wp

# Output edge size in float32 elements. Each kernel launches a (EDGE / tile_dim)^2
# grid of blocks so the total store volume is constant across tile sizes (16 MiB).
_OUTPUT_EDGE = 2048
_BLOCK_DIM = 128


def _create_zeros_kernel(tile_dim):
    TILE = tile_dim

    @wp.kernel
    def k(out: wp.array2d(dtype=wp.float32)):
        i, j = wp.tid()
        t = wp.tile_zeros(shape=(TILE, TILE), dtype=wp.float32, storage="shared")
        wp.tile_store(out, t, offset=(i * TILE, j * TILE))

    return k


def _create_empty_kernel(tile_dim):
    TILE = tile_dim

    @wp.kernel
    def k(out: wp.array2d(dtype=wp.float32)):
        i, j = wp.tid()
        t = wp.tile_empty(shape=(TILE, TILE), dtype=wp.float32, storage="shared")
        wp.tile_store(out, t, offset=(i * TILE, j * TILE))

    return k


class TileZerosSharedStore:
    """Allocate a shared tile with ``tile_zeros`` and store it to global memory.

    Measures the cost of the parallel block-wide zero-fill into shared memory.
    """

    repeat = 50
    number = 100
    params = [64, 128]
    param_names = ["tile_dim"]

    def setup(self, tile_dim):
        wp.init()
        wp.set_module_options({"fast_math": True, "enable_backward": False})
        self.device = wp.get_device("cuda:0")
        wp.load_module(device=self.device)

        kernel = _create_zeros_kernel(tile_dim)
        grid_dim = _OUTPUT_EDGE // tile_dim
        self.out = wp.empty((_OUTPUT_EDGE, _OUTPUT_EDGE), dtype=wp.float32, device=self.device)

        self.cmd = wp.launch_tiled(
            kernel,
            dim=(grid_dim, grid_dim),
            inputs=[],
            outputs=[self.out],
            block_dim=_BLOCK_DIM,
            device=self.device,
            record_cmd=True,
        )
        self.cmd.launch()
        wp.synchronize_device(self.device)

    def time_cuda(self, tile_dim):
        self.cmd.launch()
        wp.synchronize_device(self.device)


class TileEmptySharedStore:
    """Allocate a shared tile with ``tile_empty`` and store it to global memory.

    Companion to :class:`TileZerosSharedStore`. The store work is identical;
    the difference is that ``tile_empty`` skips the shared zero-fill.
    """

    repeat = 50
    number = 100
    params = [64, 128]
    param_names = ["tile_dim"]

    def setup(self, tile_dim):
        wp.init()
        # Skip on commits where wp.tile_empty isn't registered yet (e.g.,
        # asv continuous comparing against a base ref that predates this MR).
        # Becomes inert after this MR merges; can be removed at that point.
        from warp._src.context import builtin_functions  # noqa: PLC0415

        if "tile_empty" not in builtin_functions:
            raise NotImplementedError("wp.tile_empty is not available in this Warp build")
        wp.set_module_options({"fast_math": True, "enable_backward": False})
        self.device = wp.get_device("cuda:0")
        wp.load_module(device=self.device)

        kernel = _create_empty_kernel(tile_dim)
        grid_dim = _OUTPUT_EDGE // tile_dim
        self.out = wp.empty((_OUTPUT_EDGE, _OUTPUT_EDGE), dtype=wp.float32, device=self.device)

        self.cmd = wp.launch_tiled(
            kernel,
            dim=(grid_dim, grid_dim),
            inputs=[],
            outputs=[self.out],
            block_dim=_BLOCK_DIM,
            device=self.device,
            record_cmd=True,
        )
        self.cmd.launch()
        wp.synchronize_device(self.device)

    def time_cuda(self, tile_dim):
        self.cmd.launch()
        wp.synchronize_device(self.device)
