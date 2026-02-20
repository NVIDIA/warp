# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Shared-memory tile load/store benchmarks for 1D, 2D, and 3D tiles.

Uses storage="shared" exclusively — this exercises the copy_from_global /
copy_to_global paths in tile.h (scalar fallback and vectorized float4).
Register-storage tiles use a different code path and are covered by
load_store.py.

Exercises power-of-two tile sizes (which enable the vectorized float4 path for 2D
tiles) and non-power-of-two tile sizes (scalar paths) across 1D, 2D, and 3D tiles,
multiple dtypes, and L2-cached vs DRAM-bound regimes.  The scalar path
uses flat base+i indexing for 1D and incremental coordinate iteration
for N-D (N >= 2).
"""

import warp as wp

DTYPE_MAP = {
    "float32": wp.float32,
    "float64": wp.float64,
    "vec3": wp.vec3,
}

DTYPE_BYTES = {
    "float32": 4,
    "float64": 8,
    "vec3": 12,
}

L2_TARGET_BYTES = 2 * 1024 * 1024  # ~2 MiB — fits in typical GPU L2 cache
DRAM_TARGET_BYTES = 256 * 1024 * 1024  # ~256 MiB — exceeds L2, exercises DRAM path

# L2 configs are ~10μs per iteration, DRAM configs are ~700μs.
# More iterations for L2 to improve signal-to-noise.
L2_NUMBER = 1000
DRAM_NUMBER = 200


def create_kernel_1d(tile_size, dtype):
    TILE = tile_size

    @wp.kernel
    def load_store_1d(a: wp.array(dtype=dtype), b: wp.array(dtype=dtype)):
        i = wp.tid()
        t = wp.tile_load(a, shape=TILE, offset=i * TILE, storage="shared")
        wp.tile_store(b, t, offset=i * TILE)

    return load_store_1d


def create_kernel_2d(tile_size, dtype):
    TILE = tile_size

    @wp.kernel
    def load_store_2d(a: wp.array2d(dtype=dtype), b: wp.array2d(dtype=dtype)):
        i, j = wp.tid()
        t = wp.tile_load(a, shape=(TILE, TILE), offset=(i * TILE, j * TILE), storage="shared")
        wp.tile_store(b, t, offset=(i * TILE, j * TILE))

    return load_store_2d


def create_kernel_3d(tile_size, dtype):
    TILE = tile_size

    @wp.kernel
    def load_store_3d(a: wp.array3d(dtype=dtype), b: wp.array3d(dtype=dtype)):
        i, j, k = wp.tid()
        t = wp.tile_load(a, shape=(TILE, TILE, TILE), offset=(i * TILE, j * TILE, k * TILE), storage="shared")
        wp.tile_store(b, t, offset=(i * TILE, j * TILE, k * TILE))

    return load_store_3d


def _compute_edge(ndim, tile_size, dtype_name, regime):
    """Compute the array edge length for a given dimensionality and memory regime.

    Returns an edge length (rounded down to a multiple of tile_size) such that
    the total array size in bytes approximates the target for the regime.
    """
    target = L2_TARGET_BYTES if regime == "L2" else DRAM_TARGET_BYTES
    total_elems = target // DTYPE_BYTES[dtype_name]
    edge = int(total_elems ** (1.0 / ndim))
    edge = (edge // tile_size) * tile_size
    return max(edge, tile_size)


def _setup_benchmark(self, ndim, tile_size, dtype_name, regime, create_kernel):
    """Common setup logic for all LoadStore benchmark classes."""
    self.number = L2_NUMBER if regime == "L2" else DRAM_NUMBER
    wp.init()
    wp.set_module_options({"fast_math": True, "enable_backward": False})
    self.device = wp.get_device("cuda:0")
    wp.load_module(device=self.device)

    dtype = DTYPE_MAP[dtype_name]
    edge = _compute_edge(ndim, tile_size, dtype_name, regime)

    shape = tuple(edge for _ in range(ndim))
    self.a = wp.zeros(shape if ndim > 1 else edge, dtype=dtype, device=self.device)
    self.b = wp.zeros(shape if ndim > 1 else edge, dtype=dtype, device=self.device)

    tiles_per_dim = edge // tile_size
    dim = tuple(tiles_per_dim for _ in range(ndim))
    dim = dim if ndim > 1 else tiles_per_dim

    kernel = create_kernel(tile_size, dtype)
    self.cmd = wp.launch_tiled(
        kernel,
        dim=dim,
        inputs=[self.a],
        outputs=[self.b],
        block_dim=64,
        device=self.device,
        record_cmd=True,
    )
    self.cmd.launch()
    wp.synchronize_device(self.device)


class LoadStore1D:
    """1D tile load/store: flat base+i scalar indexing for all tile sizes."""

    params = ([17, 47, 64], ["float32", "vec3"], ["L2", "DRAM"])
    param_names = ["tile_size", "dtype", "regime"]

    def setup(self, tile_size, dtype_name, regime):
        _setup_benchmark(self, 1, tile_size, dtype_name, regime, create_kernel_1d)

    def time_cuda(self, tile_size, dtype_name, regime):
        self.cmd.launch()
        wp.synchronize_device(self.device)


class LoadStore2D:
    """2D tile load/store: scalar incremental coords (non-power-of-two) and vectorized float4 (power-of-two)."""

    params = ([17, 23, 32], ["float32", "vec3"], ["L2", "DRAM"])
    param_names = ["tile_size", "dtype", "regime"]

    def setup(self, tile_size, dtype_name, regime):
        _setup_benchmark(self, 2, tile_size, dtype_name, regime, create_kernel_2d)

    def time_cuda(self, tile_size, dtype_name, regime):
        self.cmd.launch()
        wp.synchronize_device(self.device)


class LoadStore3D:
    """3D tile load/store: incremental coordinate iteration with carry propagation."""

    params = ([7, 11, 8], ["float32", "float64"], ["L2", "DRAM"])
    param_names = ["tile_size", "dtype", "regime"]

    def setup(self, tile_size, dtype_name, regime):
        _setup_benchmark(self, 3, tile_size, dtype_name, regime, create_kernel_3d)

    def time_cuda(self, tile_size, dtype_name, regime):
        self.cmd.launch()
        wp.synchronize_device(self.device)
