// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// tile_solve: cooperative scalar triangular solves (forward / back
// substitution) and the tile_lower_solve / tile_upper_solve entry templates
// (and inplace variants).
//
// Performance: cooperative scalar vs cuSolverDx (L40, sm_89, fp64,
// block_dim=32, m_rhs=8, batch=64)
//
//   tile_lower_solve  (matrix RHS)
//     n   mathdx (us)   scalar (us)   scalar/mathdx
//     4         16.2          16.6         1.02x
//     8         16.2          18.2         1.12x
//    16         16.9          22.9         1.35x
//    32         19.3          36.8         1.90x
//    64         26.8          83.3         3.11x
//
//   tile_cholesky_solve  (full LL^T solve = forward + back substitution)
//     n   mathdx (us)   scalar (us)   scalar/mathdx
//     4         16.1          17.2         1.07x
//     8         16.7          21.1         1.27x
//    16         18.4          30.3         1.65x
//    32         22.4          58.0         2.60x
//    64         34.7         153.3         4.42x
//
// cuSolverDx wins at every size we measured. The gap is within noise at n<=8,
// modest at n=16 (~1.4x for trsm, ~1.7x for cholesky_solve), and grows
// quickly past that. Triangular solves are inherently serial in the row
// dimension; cooperative parallelism only helps the inner dot product (vector
// RHS) or the outer column dimension (matrix RHS), neither of which catches
// the cuSolverDx LTO at the sizes measured here.
//
// The cooperative scalar path is therefore primarily a correctness fallback
// for builds without libmathdx and for users who want to skip the slow LTO
// compilation cost during development. Users can route a kernel through the
// scalar path on a libmathdx-enabled build by setting the module option
// `enable_mathdx_solver=False` (or globally via
// `wp.config.enable_mathdx_solver = False`).

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
//   - Vector RHS (Shape::N == 1): the outer i loop is sequential (each x[i]
//     depends on x[0..i-1]); the only plausible parallelization opportunity
//     is the inner dot product. We intentionally keep that sequential too --
//     gating the whole path on a thread-0 + WP_TILE_SYNC -- to avoid the
//     per-row reduction overhead (n syncs) that does not obviously pay back
//     at typical tile sizes (n <= 64, block_dim = 32). Defer a benchmark-
//     gated reduction follow-up if profiling shows vector-RHS solves are a
//     bottleneck on GPU-without-mathdx.
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
// is fully gated on thread 0 (the outer i loop is sequential and the inner
// dot product alone doesn't pay for the reduction overhead at typical n);
// matrix RHS distributes the outer k loop across threads.
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
        WP_TILE_SYNC();
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
