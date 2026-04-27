// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// tile_solve: cooperative scalar triangular solves (forward / back
// substitution) and the tile_lower_solve / tile_upper_solve entry templates
// (and inplace variants).
//
// Performance note: the cooperative scalar trsm path here is not solely a
// fallback for builds without libmathdx. cuSolverDx's `trsm` carries the same
// LTO compilation and per-launch setup overhead as cuBLASDx, so for small
// tile sizes the cooperative shared-memory implementation is expected to
// outperform it on the same dynamics matmul exhibits. Users can deliberately
// route a kernel through the scalar path on a libmathdx-enabled build by
// setting the module option `enable_mathdx_trsm=False` (or globally via
// `wp.config.enable_mathdx_trsm = False`). The crossover point is shape- and
// dtype-dependent -- benchmark your configuration.

#pragma once

#include "tile.h"

#ifdef __clang__
// disable warnings related to C++17 extensions on CPU JIT builds
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc++17-extensions"
#endif  // __clang__

namespace wp {

namespace partitioned_gemm {

// Writes into X (the result vector or matrix).
//
// Two paths gated on the rank of TileY:
//   - Vector RHS (Shape::N == 1): the inner i loop is sequential (each x[i]
//     depends on x[0..i-1]); only the inner dot product is parallel. We use a
//     cheap thread-0 gate + WP_TILE_SYNC. A per-row partial-sum reduction
//     across threads (n syncs total) was considered but for typical tile
//     sizes (n <= 64, block_dim = 32) the reduction overhead does not
//     obviously pay back -- defer to a benchmark-gated follow-up if profiling
//     shows vector-RHS solves are a bottleneck on GPU-without-mathdx.
//   - Matrix RHS (Shape::N == 2): the outer k loop iterates over m
//     independent columns; distribute that loop across threads. Each thread
//     runs the existing inner i loop sequentially.
//
// On CPU, WP_TILE_BLOCK_DIM == 1 collapses the matrix-RHS thread-strided loop
// to plain sequential, the vector-RHS thread-0 gate executes on the only
// thread, and WP_TILE_SYNC() is a no-op -- behaviour matches the prior
// single-threaded scalar fallback.
template <bool Upper, typename TileA, typename TileX, typename TileY>
inline CUDA_CALLABLE void scalar_cholesky_forward_substitution(TileA& A, TileX& X, TileY& Y)
{
    using T = typename TileA::Type;

    auto idx = [](int row, int col) { return Upper ? tile_coord(col, row) : tile_coord(row, col); };

    if constexpr (TileY::Layout::Shape::N == 1) {
        constexpr int n = TileA::Layout::Shape::dim(1);

        if (WP_TILE_THREAD_IDX == 0) {
            for (int i = 0; i < n; ++i) {
                T s = Y.data(tile_coord(i));

                for (int j = 0; j < i; ++j)
                    s -= A.data(idx(i, j)) * X.data(tile_coord(j));

                T diag = A.data(idx(i, i));
                X.data(tile_coord(i)) = (diag != T(0.0f)) ? s / diag : s;
            }
        }
        WP_TILE_SYNC();
    } else if constexpr (TileY::Layout::Shape::N == 2) {
        constexpr int n = TileA::Layout::Shape::dim(1);
        constexpr int m = TileY::Layout::Shape::dim(1);

        for (int k = WP_TILE_THREAD_IDX; k < m; k += WP_TILE_BLOCK_DIM) {
            for (int i = 0; i < n; ++i) {
                T s = Y.data(tile_coord(i, k));

                for (int j = 0; j < i; ++j)
                    s -= A.data(idx(i, j)) * X.data(tile_coord(j, k));

                T diag = A.data(idx(i, i));
                X.data(tile_coord(i, k)) = (diag != T(0.0f)) ? s / diag : s;
            }
        }
        WP_TILE_SYNC();
    }
}

// Reads and writes X.
//
// Same cooperative split as scalar_cholesky_forward_substitution: vector RHS
// uses a cheap thread-0 gate (the i loop is sequential and the inner dot
// product alone doesn't pay for reduction overhead at typical n); matrix RHS
// distributes the outer k loop across threads.
template <bool Upper, typename TileA, typename TileX>
inline CUDA_CALLABLE void scalar_cholesky_back_substitution(TileA& A, TileX& X)
{
    using T = typename TileA::Type;

    auto idx = [](int row, int col) { return Upper ? tile_coord(row, col) : tile_coord(col, row); };

    if constexpr (TileX::Layout::Shape::N == 1) {
        constexpr int n = TileA::Layout::Shape::dim(1);

        if (WP_TILE_THREAD_IDX == 0) {
            for (int i = n - 1; i >= 0; --i) {
                T s = X.data(tile_coord(i));

                for (int j = i + 1; j < n; ++j)
                    s -= A.data(idx(i, j)) * X.data(tile_coord(j));

                T diag = A.data(idx(i, i));
                X.data(tile_coord(i)) = (diag != T(0.0f)) ? s / diag : s;
            }
        }
        WP_TILE_SYNC();
    } else if constexpr (TileX::Layout::Shape::N == 2) {
        constexpr int n = TileA::Layout::Shape::dim(1);
        constexpr int m = TileX::Layout::Shape::dim(1);

        for (int k = WP_TILE_THREAD_IDX; k < m; k += WP_TILE_BLOCK_DIM) {
            for (int i = n - 1; i >= 0; --i) {
                T s = X.data(tile_coord(i, k));

                for (int j = i + 1; j < n; ++j)
                    s -= A.data(idx(i, j)) * X.data(tile_coord(j, k));

                T diag = A.data(idx(i, i));
                X.data(tile_coord(i, k)) = (diag != T(0.0f)) ? s / diag : s;
            }
        }
        WP_TILE_SYNC();
    }
}

template <bool Upper, typename TileA, typename TileX, typename TileY>
inline CUDA_CALLABLE void scalar_cholesky_solve(TileA& A, TileX& X, TileY& Y)
{
    scalar_cholesky_forward_substitution<Upper>(A, X, Y);
    scalar_cholesky_back_substitution<Upper>(A, X);
}


}  // namespace partitioned_gemm


template <typename Fwd, typename TileL, typename TileY, typename TileZ>
TileZ& tile_lower_solve(Fwd fun_forward, TileL& L, TileY& y, TileZ& z)
{
    // Copy y to z
    z = y;

#if !defined(__CUDA_ARCH__) || WP_ENABLE_MATHDX == 0
    partitioned_gemm::scalar_cholesky_forward_substitution<false>(L, z, y);
#else
    if constexpr (wp_is_null_func<Fwd>::value) {
        partitioned_gemm::scalar_cholesky_forward_substitution<false>(L, z, y);
    } else {
        WP_TILE_SYNC();
        fun_forward(L.data.ptr, z.data.ptr);
        WP_TILE_SYNC();
    }
#endif

    return z;
}

template <typename Fwd, typename TileL, typename TileY>
void tile_lower_solve_inplace(Fwd fun_forward, TileL& L, TileY& y)
{
#if !defined(__CUDA_ARCH__) || WP_ENABLE_MATHDX == 0
    partitioned_gemm::scalar_cholesky_forward_substitution<false>(L, y, y);
#else
    if constexpr (wp_is_null_func<Fwd>::value) {
        partitioned_gemm::scalar_cholesky_forward_substitution<false>(L, y, y);
    } else {
        WP_TILE_SYNC();
        fun_forward(L.data.ptr, y.data.ptr);
        WP_TILE_SYNC();
    }
#endif
}

#define adj_tile_lower_solve(function_name, L, y, z, adj_function_name, adj_L, adj_y, adj_z, adj_ret) \
     do { \
         assert(false); \
     } while (0)

#define adj_tile_lower_solve_inplace(function_name, L, y, adj_function_name, adj_L, adj_y) \
     do { \
         assert(false); \
     } while (0)


template <typename Fwd, typename TileU, typename TileZ, typename TileX>
TileX& tile_upper_solve(Fwd fun_forward, TileU& U, TileZ& z, TileX& x)
{
    // Copy z to x
    x = z;

#if !defined(__CUDA_ARCH__) || WP_ENABLE_MATHDX == 0
    {
        auto L = tile_transpose(U);
        partitioned_gemm::scalar_cholesky_back_substitution<false>(L, x);
    }
#else
    if constexpr (wp_is_null_func<Fwd>::value) {
        auto L = tile_transpose(U);
        partitioned_gemm::scalar_cholesky_back_substitution<false>(L, x);
    } else {
        WP_TILE_SYNC();
        fun_forward(U.data.ptr, x.data.ptr);
        WP_TILE_SYNC();
    }
#endif

    return x;
}

template <typename Fwd, typename TileU, typename TileZ>
void tile_upper_solve_inplace(Fwd fun_forward, TileU& U, TileZ& z)
{
#if !defined(__CUDA_ARCH__) || WP_ENABLE_MATHDX == 0
    {
        auto L = tile_transpose(U);
        partitioned_gemm::scalar_cholesky_back_substitution<false>(L, z);
    }
#else
    if constexpr (wp_is_null_func<Fwd>::value) {
        auto L = tile_transpose(U);
        partitioned_gemm::scalar_cholesky_back_substitution<false>(L, z);
    } else {
        WP_TILE_SYNC();
        fun_forward(U.data.ptr, z.data.ptr);
        WP_TILE_SYNC();
    }
#endif
}

#define adj_tile_upper_solve(function_name, U, z, x, adj_function_name, adj_U, adj_z, adj_x, adj_ret) \
     do { \
         assert(false); \
     } while (0)

#define adj_tile_upper_solve_inplace(function_name, U, z, adj_function_name, adj_U, adj_z) \
     do { \
         assert(false); \
     } while (0)


template <bool Upper, typename Fwd, typename TileA, typename TileY, typename TileX>
TileX& tile_cholesky_solve(Fwd fun_forward, TileA& A, TileY& Y, TileX& X)
{
    // Copy y to x

    X = Y;

#if !defined(__CUDA_ARCH__) || WP_ENABLE_MATHDX == 0
    partitioned_gemm::scalar_cholesky_solve<Upper>(A, X, Y);
#else
    if constexpr (wp_is_null_func<Fwd>::value) {
        partitioned_gemm::scalar_cholesky_solve<Upper>(A, X, Y);
    } else {
        WP_TILE_SYNC();
        fun_forward(A.data.ptr, X.data.ptr);
        WP_TILE_SYNC();
    }
#endif

    return X;
}

template <bool Upper, typename Fwd, typename TileA, typename TileY>
void tile_cholesky_solve_inplace(Fwd fun_forward, TileA& A, TileY& Y)
{
#if !defined(__CUDA_ARCH__) || WP_ENABLE_MATHDX == 0
    partitioned_gemm::scalar_cholesky_solve<Upper>(A, Y, Y);
#else
    if constexpr (wp_is_null_func<Fwd>::value) {
        partitioned_gemm::scalar_cholesky_solve<Upper>(A, Y, Y);
    } else {
        fun_forward(A.data.ptr, Y.data.ptr);
        WP_TILE_SYNC();
    }
#endif
}

#define adj_tile_cholesky_solve(function_name, A, Y, X, adj_function_name, adj_A, adj_Y, adj_X, adj_ret) \
     do { \
         assert(false); \
     } while (0)

#define adj_tile_cholesky_solve_inplace(function_name, A, Y, adj_function_name, adj_A, adj_Y) \
     do { \
         assert(false); \
     } while (0)


}  // namespace wp

#ifdef __clang__
#pragma clang diagnostic pop
#endif
