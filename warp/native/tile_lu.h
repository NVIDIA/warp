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

template <
    typename Fwd,
    typename BkwdL,
    typename BkwdU,
    typename TileA,
    typename TileLU,
    typename TilePivots,
    typename TileInfo>
CUDA_CALLABLE void tile_lu(
    Fwd fun_forward,
    BkwdL fun_backward_l,
    BkwdU fun_backward_u,
    TileA& A,
    TileLU& LU,
    TilePivots& pivots,
    TileInfo& info
)
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

// Differentiate P*A = L*U with P fixed. For the packed-factor cotangent G,
// W = tril(L^T * tril(G, -1), -1) + triu(triu(G) * U^T), then
// adj_A += P^T * L^-T * W * U^-T. The unit diagonal of L has no cotangent.
template <
    typename Fwd,
    typename BkwdL,
    typename BkwdU,
    typename TileA,
    typename TileLU,
    typename TilePivots,
    typename TileInfo>
CUDA_CALLABLE void adj_tile_lu(
    Fwd fun_forward,
    BkwdL fun_backward_l,
    BkwdU fun_backward_u,
    TileA& A,
    TileLU& LU,
    TilePivots& pivots,
    TileInfo& info,
    Fwd adj_fun_forward,
    BkwdL adj_fun_backward_l,
    BkwdU adj_fun_backward_u,
    TileA& adj_A,
    TileLU& adj_LU,
    TilePivots& adj_pivots,
    TileInfo& adj_info
)
{
    using T = typename TileA::Type;
    constexpr int n = TileLU::Layout::Shape::dim(0);

#if defined(__CUDA_ARCH__)
    __shared__ T work[n * n];
#else
    T work_local[WP_TILE_BLOCK_DIM == 1 ? n * n : 1];
    T* work;
    if constexpr (WP_TILE_BLOCK_DIM == 1)
        work = work_local;
    else
        work = (T*)tile_shared_storage_t::alloc(int(sizeof(T) * n * n));
#endif
    WP_TILE_SYNC();

    // Construct the two masked products directly from packed storage.
    for (int index = WP_TILE_THREAD_IDX; index < n * n; index += WP_TILE_BLOCK_DIM) {
        const int i = index / n;
        const int j = index % n;
        T value = T(0);
        if (i > j) {
            value = adj_LU.grad(tile_coord(i, j));
            for (int k = i + 1; k < n; ++k)
                value += LU.data(tile_coord(k, i)) * adj_LU.grad(tile_coord(k, j));
        } else {
            for (int k = j; k < n; ++k)
                value += adj_LU.grad(tile_coord(i, k)) * LU.data(tile_coord(j, k));
        }
        work[index] = value;
    }
    WP_TILE_SYNC();

#if defined(__CUDA_ARCH__) && WP_ENABLE_MATHDX
    if constexpr (!wp_is_null_func<BkwdL>::value) {
        // Treat packed row-major LU as column-major to expose L^T and U^T.
        fun_backward_l(LU.data.ptr, work);
        WP_TILE_SYNC();
        fun_backward_u(LU.data.ptr, work);
        WP_TILE_SYNC();
    } else
#endif
    {
        // L^T * X = W: independent columns, implicit unit diagonal.
        for (int j = WP_TILE_THREAD_IDX; j < n; j += WP_TILE_BLOCK_DIM) {
            for (int i = n - 1; i >= 0; --i) {
                T value = work[i * n + j];
                for (int k = i + 1; k < n; ++k)
                    value -= LU.data(tile_coord(k, i)) * work[k * n + j];
                work[i * n + j] = value;
            }
        }
        WP_TILE_SYNC();

        // Y * U^T = X: independent rows, nonunit diagonal.
        for (int i = WP_TILE_THREAD_IDX; i < n; i += WP_TILE_BLOCK_DIM) {
            for (int j = n - 1; j >= 0; --j) {
                T value = work[i * n + j];
                for (int k = j + 1; k < n; ++k)
                    value -= work[i * n + k] * LU.data(tile_coord(j, k));
                work[i * n + j] = value / LU.data(tile_coord(j, j));
            }
        }
        WP_TILE_SYNC();
    }

    // Apply P^T by undoing the recorded swaps in reverse order.
    for (int k = n - 1; k >= 0; --k) {
        const int pivot = pivots.data(tile_coord(k));
        if (pivot != k) {
            for (int j = WP_TILE_THREAD_IDX; j < n; j += WP_TILE_BLOCK_DIM) {
                T value = work[k * n + j];
                work[k * n + j] = work[pivot * n + j];
                work[pivot * n + j] = value;
            }
        }
        WP_TILE_SYNC();
    }

    // User-function arguments may retain register storage. Use the common
    // accumulator, which also handles strided and overlapping shared views.
    auto contribution = tile_register_like<TileA>();
    using GradLayout = typename decltype(contribution)::Layout;
    for (int reg = 0; reg < GradLayout::NumRegs; ++reg) {
        const int linear = GradLayout::linear_from_register(reg);
        if (!GradLayout::valid(linear))
            break;
        contribution.data[reg] = work[linear];
    }
    adj_A.grad_add(contribution);
    WP_TILE_SYNC();

#if !defined(__CUDA_ARCH__)
    if constexpr (WP_TILE_BLOCK_DIM > 1)
        tile_shared_storage_t::alloc(-int(sizeof(T) * n * n));
#endif
}

}  // namespace wp
