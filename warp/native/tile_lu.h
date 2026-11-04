// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tile.h"

namespace wp {

// Right-looking LU with partial pivoting. Row swaps include previously computed
// multipliers so the packed factors satisfy P*A = L*U. Each barrier also supports
// cooperative CPU execution, where threads need not advance in GPU lockstep.
template <typename TileLU, typename TilePivots, typename TileInfo>
CUDA_CALLABLE void tile_lu_native(TileLU& LU, TilePivots& pivots, TileInfo& info)
{
    using T = typename TileLU::Type;
    constexpr int n = TileLU::Layout::Shape::dim(0);

    for (int k = 0; k < n; ++k) {
        if (WP_TILE_THREAD_IDX == 0) {
            int pivot = k;
            T largest = wp::abs(LU.data(tile_coord(k, k)));
            for (int i = k + 1; i < n; ++i) {
                T candidate = wp::abs(LU.data(tile_coord(i, k)));
                if (candidate > largest) {
                    largest = candidate;
                    pivot = i;
                }
            }
            pivots.data(tile_coord(k)) = pivot;
            if (largest == T(0) && info.data(tile_coord(0)) == 0)
                info.data(tile_coord(0)) = k + 1;
        }
        WP_TILE_SYNC();

        const int pivot = pivots.data(tile_coord(k));
        if (pivot != k) {
            for (int j = WP_TILE_THREAD_IDX; j < n; j += WP_TILE_BLOCK_DIM) {
                T tmp = LU.data(tile_coord(k, j));
                LU.data(tile_coord(k, j)) = LU.data(tile_coord(pivot, j));
                LU.data(tile_coord(pivot, j)) = tmp;
            }
        }
        WP_TILE_SYNC();

        const T diagonal = LU.data(tile_coord(k, k));
        if (diagonal != T(0)) {
            for (int i = k + 1 + WP_TILE_THREAD_IDX; i < n; i += WP_TILE_BLOCK_DIM)
                LU.data(tile_coord(i, k)) /= diagonal;
        }
        WP_TILE_SYNC();

        const int remaining = n - k - 1;
        for (int index = WP_TILE_THREAD_IDX; index < remaining * remaining; index += WP_TILE_BLOCK_DIM) {
            const int i = k + 1 + index / remaining;
            const int j = k + 1 + index % remaining;
            LU.data(tile_coord(i, j)) -= LU.data(tile_coord(i, k)) * LU.data(tile_coord(k, j));
        }
        WP_TILE_SYNC();
    }
}

template <typename Fwd, typename TileA, typename TileLU, typename TilePivots, typename TileInfo>
CUDA_CALLABLE void tile_lu(Fwd fun_forward, TileA& A, TileLU& LU, TilePivots& pivots, TileInfo& info)
{
    LU = A;
    if (WP_TILE_THREAD_IDX == 0)
        info.data(tile_coord(0)) = 0;
    WP_TILE_SYNC();

#if defined(__CUDA_ARCH__) && WP_ENABLE_MATHDX
    if constexpr (!wp_is_null_func<Fwd>::value) {
        fun_forward(LU.data.ptr, pivots.data.ptr, info.data.ptr);
        WP_TILE_SYNC();
        // cuSOLVERDx uses LAPACK's one-based row-swap indices.
        for (int k = WP_TILE_THREAD_IDX; k < TileLU::Layout::Shape::dim(0); k += WP_TILE_BLOCK_DIM)
            pivots.data(tile_coord(k)) -= 1;
        WP_TILE_SYNC();
    } else
#endif
    {
        tile_lu_native(LU, pivots, info);
    }
}

}  // namespace wp
