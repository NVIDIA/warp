// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// tile_cholesky: cooperative scalar Cholesky factorization, solve helper, and
// adjoint, plus the tile_cholesky / tile_cholesky_inplace entry templates and
// the tile_cholesky adjoint dispatch.
//
// Performance: cooperative scalar vs cuSolverDx (L40, sm_89, fp64, block_dim=32)
//
//   tile_cholesky  (factorization)
//     n   mathdx (us)   scalar (us)   scalar/mathdx
//     4         15.0          15.9         1.06x
//     8         16.1          19.0         1.18x
//    16         18.1          27.5         1.52x
//    32         25.3          53.9         2.13x
//    64         54.3         159.4         2.94x
//
//   adj_tile_cholesky  (Murray 2016 derivative, gemm + 2 trsm composition)
//     measured indirectly via tile_cholesky_solve (n=8, m_rhs=8, batch=64):
//     mathdx 16.7us, scalar 21.1us, 1.27x; widens to 4.4x at n=64.
//
// cuSolverDx wins at every size we measured. The gap is within noise at n<=8,
// modest at n=16 (~1.5x), and grows quickly past that. Cholesky has more
// inherent serialization than matmul (sequential outer column loop in
// factorization, two sequential triangular solves in the adjoint), so the
// matmul-style "scalar wins at small tiles" pattern does not appear here.
//
// The cooperative scalar path is therefore primarily a correctness fallback
// for builds without libmathdx and for users who want to skip the slow LTO
// compilation cost during development. Users can route a kernel through the
// scalar path on a libmathdx-enabled build by setting the module option
// `enable_mathdx_solver=False` (or globally via
// `wp.config.enable_mathdx_solver = False`).

#pragma once

#include "tile.h"
#include "tile_matmul.h"
#include "tile_solve.h"

#ifdef __clang__
// disable warnings related to C++17 extensions on CPU JIT builds
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc++17-extensions"
#endif  // __clang__

namespace wp {

namespace partitioned_gemm {


// Scalar Cholesky factorization, cooperative across WP_TILE_BLOCK_DIM threads.
//
// Upper=false: A = L L^T, L is lower triangular
// Upper=true:  A = U^T U, U is upper triangular
//
// Cooperative structure:
//   - Outer j loop (column index): SEQUENTIAL -- column j depends on
//     columns 0..j-1. All threads execute it in lockstep.
//   - Diagonal element L[j,j] = sqrt(A[j,j] - sum_{k<j} L[j,k]^2):
//     ALL threads compute s and invS = 1/s redundantly into local registers.
//     Reads of A[j,j] and previously-written L[j,k] (k<j) are race-free since
//     this iteration is past the WP_TILE_SYNC at the end of iteration j-1.
//     Only thread 0 writes Out[j,j] = s. No shared-mem cell, no extra sync.
//
//     Tradeoff: a thread-0-broadcast variant via __shared__ T s_cell would
//     skip the redundant compute but plumb a shared-mem cell and add an extra
//     sync per j. A parallel-reduction variant of the diagonal sum is also
//     possible. Redundant compute is O(n^2/2) ops per factorization (~2000 ops
//     at n=64) -- negligible vs the row-update work, so we pay it for code
//     simplicity. Revisit if a benchmark shows the diagonal compute is hot.
//   - Inner i = j+1..n loop (row updates): distributed over threads. Each
//     thread owns a strided subset of i values.
//   - "Zero opposite triangle" pass: distributed over threads.
//   - WP_TILE_SYNC() at the end of each j so the next column sees finished
//     writes from this j.
//
// On CPU, WP_TILE_BLOCK_DIM == 1 collapses the thread-strided inner loops to
// plain sequential, the thread-0 diagonal write executes on the only thread,
// and WP_TILE_SYNC() is a no-op -- behaviour matches the prior single-threaded
// scalar fallback.
template <bool Upper, typename TileA, typename TileOut>
inline CUDA_CALLABLE void scalar_cholesky_impl(TileA& A, TileOut& Out)
{
    using T = typename TileA::Type;
    constexpr int n = TileA::Layout::Shape::dim(1);

    // Helper: index into the output triangle.
    // Lower: Out(row, col), Upper: Out(col, row)
    auto idx = [](int row, int col) { return Upper ? tile_coord(col, row) : tile_coord(row, col); };

    for (int j = 0; j < n; ++j) {
        // Diagonal: redundant compute on all threads.
        T s = A.data(tile_coord(j, j));

        for (int k = 0; k < j; ++k) {
            T r = Out.data(idx(j, k));
            s -= r * r;
        }

        s = wp::sqrt(s);
        T invS = 1.0 / s;

        // Only thread 0 writes the diagonal.
        if (WP_TILE_THREAD_IDX == 0) {
            Out.data(idx(j, j)) = s;
        }

        // Row updates below the diagonal -- distributed across threads.
        for (int i = j + 1 + WP_TILE_THREAD_IDX; i < n; i += WP_TILE_BLOCK_DIM) {
            T s_i = Upper ? A.data(tile_coord(j, i)) : A.data(tile_coord(i, j));

            for (int k = 0; k < j; ++k) {
                s_i -= Out.data(idx(i, k)) * Out.data(idx(j, k));
            }

            Out.data(idx(i, j)) = s_i * invS;
        }

        // Zero out the opposite triangle in column j -- distributed.
        for (int k = j + 1 + WP_TILE_THREAD_IDX; k < n; k += WP_TILE_BLOCK_DIM) {
            Out.data(idx(j, k)) = T {};
        }

        WP_TILE_SYNC();
    }
}


// Cooperative scalar Cholesky adjoint, mirrors the libmathdx adjoint
// structure but uses scalar inner kernels. Replaces the previous
// single-threaded scalar_cholesky_adj_impl (which gated to thread 0 at the
// call site -- correct but underutilized block_dim - 1 threads).
//
// Algorithm: Murray (2016) symmetric-matrix Cholesky derivative.
// Six thread-strided phases over __shared__ scratch buffers, mirroring
// adj_tile_cholesky_impl's libmathdx LTO path verbatim:
//   1. gemm into W1   = adj_Out @ Out^T (Upper) or Out^T @ adj_Out (Lower)
//   2. symmetrize W1, copy into W2 (mirror the stored triangle)
//   3. first triangular solve in-place on W2:  L^T X = W2 (Lower) or U X = W2 (Upper)
//   4. transpose W2 -> W1
//   5. second triangular solve in-place on W1:  L^T B = W1 (Lower) or U B = W1 (Upper)
//   6. accumulate W1 into adj_A.grad on the stored triangle, halving the diagonal.
//
// Each phase ends with WP_TILE_SYNC(). Intra-phase, every thread writes to a
// unique address.
//
// CPU compile: __shared__ is replaced with stack arrays. WP_TILE_BLOCK_DIM == 1
// makes thread-strided loops collapse to sequential. Behaviour matches the
// previous single-threaded adjoint on CPU.
//
// Upper=false: A = L L^T, Upper=true: A = U^T U
template <bool Upper, typename TileA, typename TileOut>
inline CUDA_CALLABLE void cooperative_scalar_cholesky_adj(TileA& adj_A, TileOut& adj_Out, TileOut& Out)
{
    using T = typename TileA::Type;
    constexpr int n = TileA::Layout::Shape::dim(1);

    // Helper: index into the output triangle.
    // Lower: Out(row, col), Upper: Out(col, row)
    auto idx = [](int row, int col) { return Upper ? tile_coord(col, row) : tile_coord(row, col); };

#if defined(__CUDA_ARCH__)
    __shared__ T W1[n * n];
    __shared__ T W2[n * n];
#else
    T W1[n * n];
    T W2[n * n];
#endif

    // Phase 1: gemm into W1.
    //   Upper: W1[i,j] = sum_k adj_Out[i,k] * Out[j,k]
    //   Lower: W1[i,j] = sum_k Out[k,i]    * adj_Out[k,j]
    for (int ij = WP_TILE_THREAD_IDX; ij < n * n; ij += WP_TILE_BLOCK_DIM) {
        int i = ij / n;
        int j = ij % n;
        T s = T(0);
        for (int k = 0; k < n; ++k) {
            if constexpr (Upper)
                s += adj_Out.grad(tile_coord(i, k)) * Out.data(tile_coord(j, k));
            else
                s += Out.data(tile_coord(k, i)) * adj_Out.grad(tile_coord(k, j));
        }
        W1[ij] = s;
    }
    WP_TILE_SYNC();

    // Phase 2: symmetrize W1 and copy into W2.
    //   Upper: keep triu, mirror to lower (W2[i,j] = W1[j,i] for i > j; else W1[i,j])
    //   Lower: keep tril, mirror to upper (W2[i,j] = W1[j,i] for i < j; else W1[i,j])
    for (int ij = WP_TILE_THREAD_IDX; ij < n * n; ij += WP_TILE_BLOCK_DIM) {
        int row = ij / n;
        int col = ij % n;
        bool mirror = Upper ? (row > col) : (row < col);
        W2[ij] = mirror ? W1[col * n + row] : W1[ij];
    }
    WP_TILE_SYNC();

    // Phase 3: solve L^T X = W2 (Lower) or U X = W2 (Upper) in-place into W2.
    //   Distribute over k columns; sequential descending i within column.
    for (int k = WP_TILE_THREAD_IDX; k < n; k += WP_TILE_BLOCK_DIM) {
        for (int i = n - 1; i >= 0; --i) {
            T s = W2[i * n + k];
            for (int j = i + 1; j < n; ++j)
                s -= Out.data(idx(j, i)) * W2[j * n + k];
            T diag = Out.data(tile_coord(i, i));
            W2[i * n + k] = (diag != T(0.0f)) ? s / diag : s;
        }
    }
    WP_TILE_SYNC();

    // Phase 4: transpose W2 into W1.
    for (int ij = WP_TILE_THREAD_IDX; ij < n * n; ij += WP_TILE_BLOCK_DIM) {
        int row = ij / n;
        int col = ij % n;
        W1[ij] = W2[col * n + row];
    }
    WP_TILE_SYNC();

    // Phase 5: solve L^T B = W1 (Lower) or U B = W1 (Upper) in-place into W1.
    for (int k = WP_TILE_THREAD_IDX; k < n; k += WP_TILE_BLOCK_DIM) {
        for (int i = n - 1; i >= 0; --i) {
            T s = W1[i * n + k];
            for (int j = i + 1; j < n; ++j)
                s -= Out.data(idx(j, i)) * W1[j * n + k];
            T diag = Out.data(tile_coord(i, i));
            W1[i * n + k] = (diag != T(0.0f)) ? s / diag : s;
        }
    }
    WP_TILE_SYNC();

    // Phase 6: accumulate W1 into adj_A.grad on the stored triangle.
    // Diagonal halved because B = A_bar + A_bar^T double-counts it. Writes via
    // tile_coord(row, col) directly (no Upper-swap) -- W1 and adj_A share the
    // same layout, so the gradient lands at the correct indices for both
    // Upper and Lower (the in_triangle predicate selects which half is
    // populated).
    for (int ij = WP_TILE_THREAD_IDX; ij < n * n; ij += WP_TILE_BLOCK_DIM) {
        int row = ij / n;
        int col = ij % n;
        bool in_triangle = Upper ? (row <= col) : (row >= col);
        if (in_triangle) {
            T scale = (row == col) ? T(0.5) : T(1);
            adj_A.grad(tile_coord(row, col)) += scale * W1[row * n + col];
        }
    }
    WP_TILE_SYNC();
}


}  // namespace partitioned_gemm


// Cholesky factorization (out-of-place) implementation.
// Upper=false: produces lower-triangular L s.t. A = L L^T, zeros upper triangle.
// Upper=true:  produces upper-triangular U s.t. A = U^T U, zeros lower triangle.
template <bool Upper, typename Fwd, typename TileA, typename TileOut>
CUDA_CALLABLE TileOut& tile_cholesky_impl(Fwd fun_forward, TileA& A, TileOut& Out)
{
    static_assert(TileA::Layout::Shape::N == 2, "Expected TileA::Layout::Shape::N == 2");
    static_assert(TileOut::Layout::Shape::N == 2, "Expected TileOut::Layout::Shape::N == 2");

    static_assert(TileA::Layout::Shape::dim(0) == TileA::Layout::Shape::dim(1), "Expected TileA to be square");
    static_assert(TileOut::Layout::Shape::dim(0) == TileOut::Layout::Shape::dim(1), "Expected TileOut to be square");
    static_assert(
        TileA::Layout::Shape::dim(0) == TileOut::Layout::Shape::dim(0),
        "Expected A and Out to have the same number of rows"
    );
    static_assert(
        TileA::Layout::Shape::dim(1) == TileOut::Layout::Shape::dim(1),
        "Expected A and Out to have the same number of columns"
    );

    Out = A;

#if !defined(__CUDA_ARCH__) || WP_ENABLE_MATHDX == 0
    partitioned_gemm::scalar_cholesky_impl<Upper>(A, Out);
#else
    if constexpr (wp_is_null_func<Fwd>::value) {
        partitioned_gemm::scalar_cholesky_impl<Upper>(A, Out);
    } else {
        // TODO: for batched Cholesky, need one info per batch
        __shared__ int info[1];

        if (WP_TILE_THREAD_IDX == 0) {
            info[0] = 0;
        }

        WP_TILE_SYNC();

        fun_forward(Out.data.ptr, info);

        WP_TILE_SYNC();

        // TODO: for batched Cholesky, check all batches
#if defined(_DEBUG)
        if (WP_TILE_THREAD_IDX == 0 && info[0] != 0) {
            printf("Non-zero status in Cholesky factorization, got %d\n", info[0]);
        }
#endif

        // Zero-out the opposite triangular part
        WP_PRAGMA_UNROLL
        for (int i = WP_TILE_THREAD_IDX; i < TileOut::Layout::Size; i += WP_TILE_BLOCK_DIM) {
            auto c = TileOut::Layout::coord_from_linear(i);

            if (Upper ? (c[0] > c[1]) : (c[0] < c[1]))
                Out.data(c) = 0.0;
        }

        WP_TILE_SYNC();
    }
#endif

    return Out;
}


template <bool Upper, typename BkwdGemm, typename BkwdTrsm, typename TileA, typename TileOut>
CUDA_CALLABLE void
adj_tile_cholesky_impl(BkwdGemm fun_bkwd_gemm, BkwdTrsm fun_bkwd_trsm, TileOut& Out, TileA& adj_A, TileOut& adj_Out)
{
    using T = typename TileA::Type;
    constexpr int n = TileA::Layout::Shape::dim(1);

#if !defined(__CUDA_ARCH__) || WP_ENABLE_MATHDX == 0

    // CPU (block_dim == 1) or GPU-without-mathdx: cooperative scalar adjoint.
    // Leading sync mirrors the libmathdx branch below: be defensive against
    // upstream hazards on Out / adj_Out before Phase 1 reads from them.
    WP_TILE_SYNC();
    partitioned_gemm::cooperative_scalar_cholesky_adj<Upper>(adj_A, adj_Out, Out);

#else

    if constexpr (wp_is_null_func<BkwdGemm>::value) {
        // GPU with mathdx build but backward LTOs absent (e.g. enable_backward
        // disabled, or enable_mathdx_solver=False at the module level):
        // cooperative scalar adjoint.
        WP_TILE_SYNC();
        partitioned_gemm::cooperative_scalar_cholesky_adj<Upper>(adj_A, adj_Out, Out);
    } else {
        __shared__ T W1[n * n];
        __shared__ T W2[n * n];

        T alpha_one = T(1);
        T beta_zero = T(0);
        WP_TILE_SYNC();

        // P = adj_Out @ Out^T (upper) or Out^T @ adj_Out (lower)
        if constexpr (Upper) {
            fun_bkwd_gemm(&alpha_one, adj_Out.grad.ptr, Out.data.ptr, &beta_zero, W1);
        } else {
            fun_bkwd_gemm(&alpha_one, Out.data.ptr, adj_Out.grad.ptr, &beta_zero, W1);
        }
        WP_TILE_SYNC();

        // Symmetrize P: mirror the stored triangle to the other side (preserving the diagonal).
        // Upper: keep triu, mirror to lower; Lower: keep tril, mirror to upper.
        for (int idx = WP_TILE_THREAD_IDX; idx < n * n; idx += WP_TILE_BLOCK_DIM) {
            int row = idx / n;
            int col = idx % n;
            bool mirror = Upper ? (row > col) : (row < col);
            if (mirror)
                W2[idx] = W1[col * n + row];
            else
                W2[idx] = W1[idx];
        }
        WP_TILE_SYNC();

        // Solve L^T X = S (lower) or U X = S (upper), in-place into W2
        fun_bkwd_trsm(Out.data.ptr, W2);
        WP_TILE_SYNC();

        // Transpose X into W1
        for (int idx = WP_TILE_THREAD_IDX; idx < n * n; idx += WP_TILE_BLOCK_DIM) {
            int row = idx / n;
            int col = idx % n;
            W1[idx] = W2[col * n + row];
        }
        WP_TILE_SYNC();

        // Solve L^T B = X^T (lower) or U B = X^T (upper), in-place into W1
        fun_bkwd_trsm(Out.data.ptr, W1);
        WP_TILE_SYNC();

        // Accumulate B into adj_A.grad (upper or lower triangle only).
        // Diagonal halved because B = A_bar + A_bar^T double-counts it.
        // W1 and adj_A share same layout so gradient accumulates at correct indices.
        for (int idx = WP_TILE_THREAD_IDX; idx < n * n; idx += WP_TILE_BLOCK_DIM) {
            int row = idx / n;
            int col = idx % n;
            bool in_triangle = Upper ? (row <= col) : (row >= col);
            if (in_triangle) {
                T scale = (row == col) ? T(0.5) : T(1);
                adj_A.grad(tile_coord(row, col)) += scale * W1[row * n + col];
            }
        }
    }

#endif

    WP_TILE_SYNC();
}

// Cholesky factorization (inplace) implementation.
template <bool Upper, typename Fwd, typename TileA>
CUDA_CALLABLE void tile_cholesky_inplace_impl(Fwd fun_forward, TileA& A)
{
    static_assert(TileA::Layout::Shape::N == 2, "Expected TileA::Layout::Shape::N == 2");
    static_assert(TileA::Layout::Shape::dim(0) == TileA::Layout::Shape::dim(1), "Expected TileA to be square");

#if !defined(__CUDA_ARCH__) || WP_ENABLE_MATHDX == 0
    partitioned_gemm::scalar_cholesky_impl<Upper>(A, A);
#else
    if constexpr (wp_is_null_func<Fwd>::value) {
        partitioned_gemm::scalar_cholesky_impl<Upper>(A, A);
    } else {
        // TODO: for batched Cholesky, need one info per batch
        __shared__ int info[1];

        if (WP_TILE_THREAD_IDX == 0) {
            info[0] = 0;
        }

        WP_TILE_SYNC();

        fun_forward(A.data.ptr, info);

        WP_TILE_SYNC();

        // TODO: for batched Cholesky, check all batches
#if defined(_DEBUG)
        if (WP_TILE_THREAD_IDX == 0 && info[0] != 0) {
            printf("Non-zero status in Cholesky factorization, got %d\n", info[0]);
        }
#endif

        // Zero-out the opposite triangular part
        WP_PRAGMA_UNROLL
        for (int i = WP_TILE_THREAD_IDX; i < TileA::Layout::Size; i += WP_TILE_BLOCK_DIM) {
            auto c = TileA::Layout::coord_from_linear(i);

            if (Upper ? (c[0] > c[1]) : (c[0] < c[1]))
                A.data(c) = 0.0;
        }

        WP_TILE_SYNC();
    }
#endif
}

// Cholesky (out-of-place): tile_cholesky<false>(...) for lower, tile_cholesky<true>(...) for upper
template <bool Upper, typename Fwd, typename BkwdGemm, typename BkwdTrsm, typename TileA, typename TileOut>
CUDA_CALLABLE TileOut&
tile_cholesky(Fwd fun_forward, BkwdGemm fun_bkwd_gemm, BkwdTrsm fun_bkwd_trsm, TileA& A, TileOut& Out)
{
    return tile_cholesky_impl<Upper>(fun_forward, A, Out);
}

// Adjoint of Cholesky (out-of-place, Murray 2016, "Differentiation of the Cholesky decomposition"):
// adj_tile_cholesky<false>(...) for lower, adj_tile_cholesky<true>(...) for upper
template <bool Upper, typename Fwd, typename BkwdGemm, typename BkwdTrsm, typename TileA, typename TileOut>
CUDA_CALLABLE void adj_tile_cholesky(
    Fwd fun_forward,
    BkwdGemm fun_bkwd_gemm,
    BkwdTrsm fun_bkwd_trsm,
    TileA& A,
    TileOut& Out,
    Fwd adj_fun_forward,
    BkwdGemm adj_fun_bkwd_gemm,
    BkwdTrsm adj_fun_bkwd_trsm,
    TileA& adj_A,
    TileOut& adj_Out,
    TileOut& adj_ret
)
{
    adj_tile_cholesky_impl<Upper>(fun_bkwd_gemm, fun_bkwd_trsm, Out, adj_A, adj_Out);
}

// Cholesky (inplace): tile_cholesky_inplace<false>(...) for lower, tile_cholesky_inplace<true>(...) for upper
template <bool Upper, typename Fwd, typename TileA> CUDA_CALLABLE void tile_cholesky_inplace(Fwd fun_forward, TileA& A)
{
    tile_cholesky_inplace_impl<Upper>(fun_forward, A);
}

#define adj_tile_cholesky_inplace(function_name, A, adj_function_name, adj_A) \
     do { \
         assert(false); \
     } while (0)


}  // namespace wp

#ifdef __clang__
#pragma clang diagnostic pop
#endif
