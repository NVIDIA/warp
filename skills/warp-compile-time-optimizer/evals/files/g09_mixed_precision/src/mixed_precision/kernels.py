# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tiled matrix kernels for calibration and serving.

Both stages live here so they share one set of compilation options.
"""

from __future__ import annotations

import warp as wp

TILE = 32

wp.set_module_options({"enable_mathdx_gemm": True})


@wp.kernel
def calibrate_kernel(
    samples: wp.array2d[wp.float32], basis: wp.array2d[wp.float32], fitted: wp.array2d[wp.float32]
) -> None:
    """Fit calibration coefficients once at startup, in full precision."""
    row, column = wp.tid()
    total = wp.tile_zeros(shape=(TILE, TILE), dtype=wp.float32)
    for step in range(samples.shape[1] // TILE):
        left = wp.tile_load(samples, shape=(TILE, TILE), offset=(row * TILE, step * TILE))
        right = wp.tile_load(basis, shape=(TILE, TILE), offset=(step * TILE, column * TILE))
        wp.tile_matmul(left, right, total)
    wp.tile_store(fitted, total, offset=(row * TILE, column * TILE))


@wp.kernel
def serve_kernel(
    activations: wp.array2d[wp.float16], weights: wp.array2d[wp.float16], output: wp.array2d[wp.float16]
) -> None:
    """Serve one request batch in half precision. Runs on every request."""
    row, column = wp.tid()
    total = wp.tile_zeros(shape=(TILE, TILE), dtype=wp.float16)
    for step in range(activations.shape[1] // TILE):
        left = wp.tile_load(activations, shape=(TILE, TILE), offset=(row * TILE, step * TILE))
        right = wp.tile_load(weights, shape=(TILE, TILE), offset=(step * TILE, column * TILE))
        wp.tile_matmul(left, right, total)
    wp.tile_store(output, total, offset=(row * TILE, column * TILE))
