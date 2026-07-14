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


template <typename Fwd, typename Bkwd, typename TileL, typename TileY, typename TileZ>
TileZ& tile_lower_solve(Fwd fun_forward, Bkwd fun_bkwd, TileL& L, TileY& y, TileZ& z)
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

// Adjoint of the out-of-place lower solve L z = y.
//
// Two leading func params (Fwd, Bkwd) mirror the two-func-var dispatch tuple in
// builtins.py (var_fwd, var_bkwd); Bkwd is the backward TRSM that solves the
// transposed system L^T w = adj_ret.
template <
    typename Fwd,
    typename Bkwd,
    typename TileL,
    typename TileY,
    typename TileZ,
    typename AdjFwd,
    typename AdjBkwd,
    typename AdjTileL,
    typename AdjTileY,
    typename AdjTileZ,
    typename AdjRet>
CUDA_CALLABLE void adj_tile_lower_solve(
    Fwd fun_forward,
    Bkwd fun_bkwd,
    TileL& L,
    TileY& y,
    TileZ& z,
    AdjFwd adj_fun_forward,
    AdjBkwd adj_fun_bkwd,
    AdjTileL& adj_L,
    AdjTileY& adj_y,
    AdjTileZ& adj_z,
    AdjRet& adj_ret
)
{
    using T = typename AdjRet::Type;

    // n = matrix dimension, nrhs = number of right-hand sides (1 for a vector RHS).
    constexpr int n = TileL::Layout::Shape::dim(1);
    constexpr int nrhs = (TileZ::Layout::Shape::N == 1) ? 1 : TileZ::Layout::Shape::dim(1);

    // Raw scratch for the transposed solve L^T W = adj_ret.
#if defined(__CUDA_ARCH__)
    __shared__ T W[n * nrhs];
#else
    T W[n * nrhs];
#endif

    // Give the scalar fallback tile indexing without allocating tile storage.
    using WLayout = tile_layout_strided_t<typename TileZ::Layout::Shape>;
    tile_shared_t<T, WLayout, false> W_tile(W);

    // Preload the incoming gradient adj_ret into W (row-major, contiguous).
    for (int idx = WP_TILE_THREAD_IDX; idx < n * nrhs; idx += WP_TILE_BLOCK_DIM) {
        if constexpr (TileZ::Layout::Shape::N == 1) {
            W[idx] = adj_ret.grad(tile_coord(idx));
        } else {
            W[idx] = adj_ret.grad(tile_coord(idx / nrhs, idx % nrhs));
        }
    }
    WP_TILE_SYNC();

#if !defined(__CUDA_ARCH__) || WP_ENABLE_MATHDX == 0
    // scalar path
    partitioned_gemm::scalar_cholesky_back_substitution<false>(L, W_tile);
#else
    if constexpr (wp_is_null_func<Bkwd>::value) {
        partitioned_gemm::scalar_cholesky_back_substitution<false>(L, W_tile);
    } else {
        // MathDx path: in-place TRSM into the raw scratch.
        fun_bkwd(L.data.ptr, W);
        WP_TILE_SYNC();
    }
#endif

    // Accumulate the rhs gradient: adj_y += W (element-wise, any rank).
    for (int idx = WP_TILE_THREAD_IDX; idx < n * nrhs; idx += WP_TILE_BLOCK_DIM) {
        if constexpr (TileZ::Layout::Shape::N == 1) {
            adj_y.grad(tile_coord(idx)) += W[idx];
        } else {
            adj_y.grad(tile_coord(idx / nrhs, idx % nrhs)) += W[idx];
        }
    }
    WP_TILE_SYNC();

    // Accumulate the factor gradient: adj_L -= tril(W Z^T), lower triangle only.
    // Vector RHS (nrhs == 1) recovers adj_L[i, j] -= W[i] * z[j].
    for (int idx = WP_TILE_THREAD_IDX; idx < n * n; idx += WP_TILE_BLOCK_DIM) {
        int row = idx / n;
        int col = idx % n;
        if (row >= col) {
            T s = T(0);
            for (int k = 0; k < nrhs; ++k) {
                if constexpr (TileZ::Layout::Shape::N == 1) {
                    s += W[row] * z.data(tile_coord(col));
                } else {
                    s += W[row * nrhs + k] * z.data(tile_coord(col, k));
                }
            }
            adj_L.grad(tile_coord(row, col)) -= s;
        }
    }
    WP_TILE_SYNC();
}

template <typename Fwd, typename TileL, typename TileY, typename AdjFwd, typename AdjTileL, typename AdjTileY>
void adj_tile_lower_solve_inplace(
    Fwd fun_forward, TileL& L, TileY& y, AdjFwd adj_fun_forward, AdjTileL& adj_L, AdjTileY& adj_y
)
{
    // MISSINGADJOINT: same math as adj_tile_lower_solve but operating in place on
    // adj_y; adj_L -= outer(adj_y_new, y_pre_solve)
}


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

template <
    typename Fwd,
    typename TileU,
    typename TileZ,
    typename TileX,
    typename AdjFwd,
    typename AdjTileU,
    typename AdjTileZ,
    typename AdjTileX,
    typename AdjRet>
void adj_tile_upper_solve(
    Fwd fun_forward,
    TileU& U,
    TileZ& z,
    TileX& x,
    AdjFwd adj_fun_forward,
    AdjTileU& adj_U,
    AdjTileZ& adj_z,
    AdjTileX& adj_x,
    AdjRet& adj_ret
)
{
    // MISSINGADJOINT: adjoint is the transposed (lower) solve U^T y = adj_ret; then adj_z
    // += y and adj_U -= outer(x, y)
}

template <typename Fwd, typename TileU, typename TileZ, typename AdjFwd, typename AdjTileU, typename AdjTileZ>
void adj_tile_upper_solve_inplace(
    Fwd fun_forward, TileU& U, TileZ& z, AdjFwd adj_fun_forward, AdjTileU& adj_U, AdjTileZ& adj_z
)
{
    // MISSINGADJOINT: same math as adj_tile_upper_solve but operating in place on
    // adj_z; adj_U -= outer(z_post_solve, adj_z_new)
}


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

template <
    bool Upper,
    typename Fwd,
    typename TileA,
    typename TileY,
    typename TileX,
    typename AdjFwd,
    typename AdjTileA,
    typename AdjTileY,
    typename AdjTileX,
    typename AdjRet>
void adj_tile_cholesky_solve(
    Fwd fun_forward,
    TileA& A,
    TileY& Y,
    TileX& X,
    AdjFwd adj_fun_forward,
    AdjTileA& adj_A,
    AdjTileY& adj_Y,
    AdjTileX& adj_X,
    AdjRet& adj_ret
)
{
    // MISSINGADJOINT: implicit differentiation through A X = Y: solve A Z = adj_ret,
    // then adj_Y += Z and adj_A -= sym(outer(Z, X))
}

template <
    bool Upper,
    typename Fwd,
    typename TileA,
    typename TileY,
    typename AdjFwd,
    typename AdjTileA,
    typename AdjTileY>
void adj_tile_cholesky_solve_inplace(
    Fwd fun_forward, TileA& A, TileY& Y, AdjFwd adj_fun_forward, AdjTileA& adj_A, AdjTileY& adj_Y
)
{
    // MISSINGADJOINT: same math as adj_tile_cholesky_solve operating in place
    // on adj_Y; adj_A -= sym(outer(adj_Y_new, Y_post_solve))
}


}  // namespace wp

#ifdef __clang__
#pragma clang diagnostic pop
#endif
