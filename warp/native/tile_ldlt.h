// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tile.h"
#include "tile_reduce.h"

namespace wp {

// Bunch-Kaufman diagonal pivoting. Previously computed multiplier rows are
// interchanged too, so the packed triangular factor satisfies P*A*P^T=L*D*L^T.
// Upper stores L^T and reads the upper input triangle; elimination proceeds
// from the first column for either storage orientation.
template <bool Upper, typename TileA, typename TileLD, typename TilePivots, typename TileInfo>
CUDA_CALLABLE void tile_ldlt(TileA& A, TileLD& LD, TilePivots& pivots, TileInfo& info)
{
    using T = typename TileLD::Type;
    constexpr int n = TileLD::Layout::Shape::dim(0);
#if defined(__CUDA_ARCH__) && WP_TILE_BLOCK_DIM >= WP_TILE_WARP_SIZE
    constexpr int pivot_threads = n >= 16 ? WP_TILE_WARP_SIZE : 1;
#else
    constexpr int pivot_threads = 1;
#endif
    // FP32 column scans are cheap; reserve the warp reduction for the optional
    // row scan. FP64 benefits from parallelizing both searches.
    constexpr int column_threads = sizeof(T) == sizeof(double) ? pivot_threads : 1;
#if defined(__CUDA_ARCH__)
    constexpr bool swizzle = sizeof(T) == sizeof(double) && n >= 16 && (n & (n - 1)) == 0;
#else
    constexpr bool swizzle = false;
#endif
    const T alpha = T(0.6403882032022076);  // (1 + sqrt(17)) / 8
    auto idx = [](int i, int j) { return tile_coord(i, swizzle ? j ^ i : j); };
    // XOR is an involution: exchange each pair once to change storage without
    // another shared tile. Restore dense storage before returning the factors.
    auto swizzle_storage = [&]() {
        if constexpr (swizzle) {
            for (int index = WP_TILE_THREAD_IDX; index < n * n; index += WP_TILE_BLOCK_DIM) {
                const int i = index / n;
                const int j = index % n;
                const int other = j ^ i;
                if (j < other) {
                    const T value = LD.data(tile_coord(i, j));
                    LD.data(tile_coord(i, j)) = LD.data(tile_coord(i, other));
                    LD.data(tile_coord(i, other)) = value;
                }
            }
            WP_TILE_SYNC();
        }
    };
    auto magnitude_key = [](T value) {
#if defined(__CUDA_ARCH__)
        if constexpr (sizeof(T) == sizeof(double)) {
            // Nonnegative IEEE doubles have the same ordering as their bits.
            // Integer reductions preserve precision and avoid FP64 comparisons.
            const unsigned long long bits = __double_as_longlong(value) & 0x7fffffffffffffffULL;
            return bits <= 0x7ff0000000000000ULL ? bits : 0ULL;
        } else
#endif
        {
            return wp::abs(value);
        }
    };
    using Magnitude = decltype(magnitude_key(T(0)));
    auto magnitude_value = [](Magnitude value) {
#if defined(__CUDA_ARCH__)
        if constexpr (sizeof(T) == sizeof(double))
            return __longlong_as_double(value);
        else
#endif
            return T(value);
    };

#if defined(__CUDA_ARCH__)
    __shared__ T inverse[4];
#else
    T inverse_local[4];
    T* inverse = inverse_local;
    if constexpr (WP_TILE_BLOCK_DIM > 1)
        inverse = (T*)tile_shared_storage_t::alloc(int(4 * sizeof(T)));
#endif
    LD = A;
    // Work in lower row-major storage for either fill mode. Contiguous scratch
    // rows then avoid column-stride bank conflicts in the trailing updates.
    if constexpr (Upper) {
        for (int index = WP_TILE_THREAD_IDX; index < n * n; index += WP_TILE_BLOCK_DIM) {
            const int i = index / n;
            const int j = index % n;
            if (i > j)
                LD.data(tile_coord(i, j)) = LD.data(tile_coord(j, i));
        }
    }
    if (WP_TILE_THREAD_IDX == 0)
        info.data(tile_coord(0)) = 0;
    WP_TILE_SYNC();
    swizzle_storage();

    int k = 0;
    while (k < n) {
        if (WP_TILE_THREAD_IDX < pivot_threads) {
            int pivot = k;
            int block_size = 1;
            int largest_row = n;
            Magnitude column_key = Magnitude(0);
            const T diagonal = wp::abs(LD.data(idx(k, k)));
            if (WP_TILE_THREAD_IDX < column_threads) {
                for (int i = k + 1 + WP_TILE_THREAD_IDX; i < n; i += column_threads) {
                    const Magnitude candidate = magnitude_key(LD.data(idx(i, k)));
                    if (candidate > column_key) {
                        column_key = candidate;
                        largest_row = i;
                    }
                }
            }
#if defined(__CUDA_ARCH__) && WP_TILE_BLOCK_DIM >= WP_TILE_WARP_SIZE
            if constexpr (column_threads > 1) {
                // Reduce the first warp, breaking ties toward the earliest row
                // to match the serial search on CPU and on smaller tiles.
                WP_PRAGMA_UNROLL
                for (int offset = WP_TILE_WARP_SIZE / 2; offset > 0; offset /= 2) {
                    const Magnitude candidate = warp_shuffle_down(column_key, offset, WP_TILE_LANE_MASK_ALL);
                    const int row = warp_shuffle_down(largest_row, offset, WP_TILE_LANE_MASK_ALL);
                    if (candidate > column_key || (candidate == column_key && row < largest_row)) {
                        column_key = candidate;
                        largest_row = row;
                    }
                }
            }
            if constexpr (pivot_threads > 1) {
                largest_row = __shfl_sync(WP_TILE_LANE_MASK_ALL, largest_row, 0, WP_TILE_WARP_SIZE);
                column_key = largest_row < n ? magnitude_key(LD.data(idx(largest_row, k))) : Magnitude(0);
            }
#endif
            const T column_max = magnitude_value(column_key);
            int search_row = 0;
            if (WP_TILE_THREAD_IDX == 0) {
                if (diagonal == T(0) && column_max == T(0)) {
                    if (info.data(tile_coord(0)) == 0)
                        info.data(tile_coord(0)) = k + 1;
                } else if (diagonal == T(0) || diagonal < alpha * column_max) {
                    search_row = 1;
                }
            }
#if defined(__CUDA_ARCH__) && WP_TILE_BLOCK_DIM >= WP_TILE_WARP_SIZE
            if constexpr (pivot_threads > 1)
                search_row = __shfl_sync(WP_TILE_LANE_MASK_ALL, search_row, 0, WP_TILE_WARP_SIZE);
#endif
            if (search_row) {
                Magnitude row_key = Magnitude(0);
                for (int j = k + WP_TILE_THREAD_IDX; j < n; j += pivot_threads) {
                    if (j != largest_row) {
                        const Magnitude candidate
                            = magnitude_key(LD.data(j < largest_row ? idx(largest_row, j) : idx(j, largest_row)));
                        row_key = wp::max(row_key, candidate);
                    }
                }
#if defined(__CUDA_ARCH__) && WP_TILE_BLOCK_DIM >= WP_TILE_WARP_SIZE
                if constexpr (pivot_threads > 1) {
                    WP_PRAGMA_UNROLL
                    for (int offset = WP_TILE_WARP_SIZE / 2; offset > 0; offset /= 2)
                        row_key = wp::max(row_key, warp_shuffle_down(row_key, offset, WP_TILE_LANE_MASK_ALL));
                }
#endif
                const T row_max = magnitude_value(row_key);
                if (WP_TILE_THREAD_IDX == 0) {
                    if (diagonal == T(0) || diagonal < alpha * column_max * (column_max / row_max)) {
                        pivot = largest_row;
                        if (wp::abs(LD.data(idx(pivot, pivot))) < alpha * row_max)
                            block_size = 2;
                    }
                }
            }
            if (WP_TILE_THREAD_IDX == 0) {
                pivots.data(tile_coord(k)) = block_size == 1 ? pivot : -(pivot + 1);
                if (block_size == 2)
                    pivots.data(tile_coord(k + 1)) = -(pivot + 1);

                // Share inverse coefficients instead of repeating divisions in
                // every multiplier row. Very small pivots can overflow the inverse
                // even when L is finite; those use scaled division below.
                if (k + block_size == n) {
                    // No multipliers remain, so no inverse is needed.
                } else if (block_size == 1) {
                    const T value = LD.data(idx(pivot, pivot));
                    inverse[0] = value != T(0) ? T(1) / value : T(0);
                    inverse[3] = wp::isfinite(inverse[0]) ? T(1) : T(0);
                } else {
                    const T off_diagonal = LD.data(idx(pivot, k));
                    const T first_diagonal = LD.data(idx(k, k));
                    const T second_diagonal = LD.data(idx(pivot, pivot));
                    const T determinant = first_diagonal * second_diagonal - off_diagonal * off_diagonal;
                    inverse[3] = T(0);
                    if (determinant != T(0) && wp::isfinite(determinant)) {
                        const T reciprocal = T(1) / determinant;
                        inverse[0] = second_diagonal * reciprocal;
                        inverse[1] = -off_diagonal * reciprocal;
                        inverse[2] = first_diagonal * reciprocal;
                        inverse[3] = wp::isfinite(inverse[0]) && wp::isfinite(inverse[1]) && wp::isfinite(inverse[2])
                            ? T(1)
                            : T(0);
                    }
                }
            }
        }
        WP_TILE_SYNC();

        const int encoded = pivots.data(tile_coord(k));
        const int block_size = encoded < 0 ? 2 : 1;
        const int pivot = encoded < 0 ? -encoded - 1 : encoded;
        const int target = k + block_size - 1;
        if (pivot != target) {
            // Swap the selected symmetric rows/columns and earlier L rows.
            // Each address belongs to exactly one lane of this loop.
            for (int j = WP_TILE_THREAD_IDX; j < n; j += WP_TILE_BLOCK_DIM) {
                if (j == pivot)
                    continue;
                auto left = j <= target ? idx(target, j) : idx(j, target);
                auto right = j < target
                    ? idx(pivot, j)
                    : (j == target ? idx(pivot, pivot) : (j < pivot ? idx(pivot, j) : idx(j, pivot)));
                T value = LD.data(left);
                LD.data(left) = LD.data(right);
                LD.data(right) = value;
            }
            WP_TILE_SYNC();
        }

        const int first = k + block_size;
        if (first == n)
            break;
        // The unused triangle holds the original columns W=L*D while the
        // stored triangle receives L. This avoids an additional shared tile.
        for (int i = first + WP_TILE_THREAD_IDX; i < n; i += WP_TILE_BLOCK_DIM) {
            const T first_value = LD.data(idx(i, k));
            LD.data(idx(k, i)) = first_value;
            if (block_size == 1) {
                const T diagonal = LD.data(idx(k, k));
                LD.data(idx(i, k)) = inverse[3] != T(0) ? first_value * inverse[0] : first_value / diagonal;
            } else {
                const T second_value = LD.data(idx(i, k + 1));
                LD.data(idx(k + 1, i)) = second_value;
                if (inverse[3] != T(0)) {
                    LD.data(idx(i, k)) = first_value * inverse[0] + second_value * inverse[1];
                    LD.data(idx(i, k + 1)) = first_value * inverse[1] + second_value * inverse[2];
                } else {
                    const T off_diagonal = LD.data(idx(k + 1, k));
                    const T first_diagonal = LD.data(idx(k, k)) / off_diagonal;
                    const T second_diagonal = LD.data(idx(k + 1, k + 1)) / off_diagonal;
                    const T determinant = first_diagonal * second_diagonal - T(1);
                    const T first_scaled = first_value / off_diagonal;
                    const T second_scaled = second_value / off_diagonal;
                    LD.data(idx(i, k)) = (second_diagonal * first_scaled - second_scaled) / determinant;
                    LD.data(idx(i, k + 1)) = (first_diagonal * second_scaled - first_scaled) / determinant;
                }
            }
        }
        WP_TILE_SYNC();

        const int remaining = n - first;
        for (int index = WP_TILE_THREAD_IDX; index < remaining * (remaining + 1) / 2; index += WP_TILE_BLOCK_DIM) {
            // Fold a rectangle into the lower triangle, visiting each entry
            // exactly once without square roots or inactive upper-half lanes.
            int row = index / (remaining + 1);
            int col = index % (remaining + 1);
            if (col > row) {
                row = remaining - 1 - row;
                col = remaining - col;
            }
            const int i = first + row;
            const int j = first + col;
            T value = LD.data(idx(i, j)) - LD.data(idx(i, k)) * LD.data(idx(k, j));
            if (block_size == 2)
                value -= LD.data(idx(i, k + 1)) * LD.data(idx(k + 1, j));
            LD.data(idx(i, j)) = value;
        }
        WP_TILE_SYNC();
        k += block_size;
    }

    for (int index = WP_TILE_THREAD_IDX; index < n * n; index += WP_TILE_BLOCK_DIM) {
        const int i = index / n;
        const int j = index % n;
        if (i < j) {
            if constexpr (Upper) {
                LD.data(idx(i, j)) = LD.data(idx(j, i));
                LD.data(idx(j, i)) = T(0);
            } else {
                LD.data(idx(i, j)) = T(0);
            }
        }
    }
    WP_TILE_SYNC();
    swizzle_storage();
#if !defined(__CUDA_ARCH__)
    if constexpr (WP_TILE_BLOCK_DIM > 1)
        tile_shared_storage_t::alloc(-int(4 * sizeof(T)));
#endif
}

// Hold the recorded permutation and 1x1/2x2 block choices fixed. If
// E=L^-1*d(P*A*P^T)*L^-T and H=L^-1*dL, then E=H*D+dD+D*H^T.
// Off-diagonal blocks give H_ij=E_ij*D_j^-1. Pull back through this relation,
// two unit-triangular solves, and finally the symmetric permutation.
template <bool Upper, typename TileA, typename TileLD, typename TilePivots, typename TileInfo>
CUDA_CALLABLE void adj_tile_ldlt(
    TileA& A,
    TileLD& LD,
    TilePivots& pivots,
    TileInfo& info,
    TileA& adj_A,
    TileLD& adj_LD,
    TilePivots& adj_pivots,
    TileInfo& adj_info
)
{
    using T = typename TileLD::Type;
    constexpr int n = TileLD::Layout::Shape::dim(0);
    auto idx = [](int i, int j) { return Upper ? tile_coord(j, i) : tile_coord(i, j); };
#if defined(__CUDA_ARCH__)
    __shared__ T work[n * n];
    __shared__ int blocks[n];
    __shared__ int permutation[n];
#else
    T work_local[WP_TILE_BLOCK_DIM == 1 ? n * n : 1];
    int blocks_local[WP_TILE_BLOCK_DIM == 1 ? n : 1];
    int permutation_local[WP_TILE_BLOCK_DIM == 1 ? n : 1];
    T* work = work_local;
    int* blocks = blocks_local;
    int* permutation = permutation_local;
    if constexpr (WP_TILE_BLOCK_DIM > 1) {
        work = (T*)tile_shared_storage_t::alloc(int(n * n * sizeof(T)));
        blocks = (int*)tile_shared_storage_t::alloc(int(n * sizeof(int)));
        permutation = (int*)tile_shared_storage_t::alloc(int(n * sizeof(int)));
    }
#endif
    WP_TILE_SYNC();
    if (WP_TILE_THREAD_IDX == 0) {
        for (int i = 0; i < n; ++i)
            permutation[i] = i;
        for (int k = 0; k < n;) {
            const int encoded = pivots.data(tile_coord(k));
            const int size = encoded < 0 ? 2 : 1;
            const int pivot = encoded < 0 ? -encoded - 1 : encoded;
            const int target = k + size - 1;
            blocks[k] = k;
            if (size == 2)
                blocks[k + 1] = k;
            const int value = permutation[target];
            permutation[target] = permutation[pivot];
            permutation[pivot] = value;
            k += size;
        }
    }
    WP_TILE_SYNC();

    auto lower_product = [&](int i, int j) {
        T value = adj_LD.grad(idx(i, j));
        for (int row = i + 1; row < n; ++row) {
            if (blocks[row] != blocks[i])
                value += LD.data(idx(row, i)) * adj_LD.grad(idx(row, j));
        }
        return value;
    };

    for (int index = WP_TILE_THREAD_IDX; index < n * n; index += WP_TILE_BLOCK_DIM) {
        const int i = index / n;
        const int j = index % n;
        if (i >= j) {
            T value;
            if (i == j) {
                value = T(2) * adj_LD.grad(idx(i, j));
            } else if (blocks[i] == blocks[j]) {
                value = adj_LD.grad(idx(i, j));
            } else {
                const int first = blocks[j];
                const bool pair = first + 1 < n && blocks[first + 1] == first;
                if (!pair) {
                    value = lower_product(i, j) / LD.data(idx(j, j));
                } else {
                    const T off_diagonal = LD.data(idx(first + 1, first));
                    const T first_diagonal = LD.data(idx(first, first)) / off_diagonal;
                    const T second_diagonal = LD.data(idx(first + 1, first + 1)) / off_diagonal;
                    const T first_scaled = lower_product(i, first) / off_diagonal;
                    const T second_scaled = lower_product(i, first + 1) / off_diagonal;
                    const T determinant = first_diagonal * second_diagonal - T(1);
                    value = j == first ? (second_diagonal * first_scaled - second_scaled) / determinant
                                       : (first_diagonal * second_scaled - first_scaled) / determinant;
                }
            }
            work[i * n + j] = value;
            work[j * n + i] = value;
        }
    }
    WP_TILE_SYNC();

    // L^T * X = W. The diagonal blocks of L are identity matrices.
    for (int j = WP_TILE_THREAD_IDX; j < n; j += WP_TILE_BLOCK_DIM) {
        for (int i = n - 1; i >= 0; --i) {
            T value = work[i * n + j];
            for (int k = i + 1; k < n; ++k) {
                if (blocks[k] != blocks[i])
                    value -= LD.data(idx(k, i)) * work[k * n + j];
            }
            work[i * n + j] = value;
        }
    }
    WP_TILE_SYNC();
    // Y * L = X, with independent rows.
    for (int i = WP_TILE_THREAD_IDX; i < n; i += WP_TILE_BLOCK_DIM) {
        for (int j = n - 1; j >= 0; --j) {
            T value = work[i * n + j];
            for (int k = j + 1; k < n; ++k) {
                if (blocks[k] != blocks[j])
                    value -= work[i * n + k] * LD.data(idx(k, j));
            }
            work[i * n + j] = value;
        }
    }
    WP_TILE_SYNC();

    // Block metadata is no longer needed; reuse it for the inverse permutation.
    if (WP_TILE_THREAD_IDX == 0) {
        for (int i = 0; i < n; ++i)
            blocks[permutation[i]] = i;
    }
    WP_TILE_SYNC();
    auto contribution = tile_register_like<TileA>();
    using GradLayout = typename decltype(contribution)::Layout;
    for (int reg = 0; reg < GradLayout::NumRegs; ++reg) {
        const int linear = GradLayout::linear_from_register(reg);
        if (!GradLayout::valid(linear))
            break;
        const int i = linear / n;
        const int j = linear % n;
        const int row = blocks[i];
        const int col = blocks[j];
        T value = T(0);
        if (i == j)
            value = T(0.5) * work[row * n + row];
        else if (Upper ? i < j : i > j)
            value = T(0.5) * (work[row * n + col] + work[col * n + row]);
        contribution.data[reg] = value;
    }
    adj_A.grad_add(contribution);
    WP_TILE_SYNC();
#if !defined(__CUDA_ARCH__)
    if constexpr (WP_TILE_BLOCK_DIM > 1) {
        tile_shared_storage_t::alloc(-int(n * sizeof(int)));
        tile_shared_storage_t::alloc(-int(n * sizeof(int)));
        tile_shared_storage_t::alloc(-int(n * n * sizeof(T)));
    }
#endif
}

}  // namespace wp
