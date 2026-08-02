# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tiled projection and reconstruction stages."""

from __future__ import annotations

import warp as wp

TILE = 8


@wp.kernel(module="tile_projection.encode")
def project_kernel(
    features: wp.array2d[wp.float32], basis: wp.array2d[wp.float32], coefficients: wp.array2d[wp.float32]
) -> None:
    """Project one feature tile onto the basis."""
    row, column = wp.tid()
    total = wp.tile_zeros(shape=(TILE, TILE), dtype=wp.float32)
    for step in range(features.shape[1] // TILE):
        left = wp.tile_load(features, shape=(TILE, TILE), offset=(row * TILE, step * TILE))
        right = wp.tile_load(basis, shape=(TILE, TILE), offset=(step * TILE, column * TILE))
        wp.tile_matmul(left, right, total)
    wp.tile_store(coefficients, total, offset=(row * TILE, column * TILE))


@wp.kernel(module="tile_projection.decode")
def reconstruct_kernel(
    coefficients: wp.array2d[wp.float32], basis: wp.array2d[wp.float32], rebuilt: wp.array2d[wp.float32]
) -> None:
    """Reconstruct one feature tile from its coefficients."""
    row, column = wp.tid()
    total = wp.tile_zeros(shape=(TILE, TILE), dtype=wp.float32)
    for step in range(coefficients.shape[1] // TILE):
        left = wp.tile_load(coefficients, shape=(TILE, TILE), offset=(row * TILE, step * TILE))
        right = wp.tile_load(basis, shape=(TILE, TILE), offset=(step * TILE, column * TILE))
        wp.tile_matmul(left, right, total)
    wp.tile_store(rebuilt, total, offset=(row * TILE, column * TILE))
