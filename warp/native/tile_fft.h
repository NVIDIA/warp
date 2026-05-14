// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "builtin.h"

#include "vec.h"

// Tile FFT / IFFT — three implementations selected at template-instantiation
// time to match cuFFTDx's unnormalized convention:
//
//   forward: X_k = sum_n x_n * exp(-2*pi*i*k*n/N)
//   inverse: X_k = sum_n x_n * exp(+2*pi*i*k*n/N)   (no 1/N scaling)
//
// 1. CPU sequential   — block_dim==1, single-thread iterative Cooley-Tukey or
//                       direct DFT for non-power-of-two sizes.
// 2. GPU cooperative  — no MathDx, block_dim>=2, shared-memory cooperative
//                       radix-2 with each thread owning a partition of the
//                       butterflies per stage. Power-of-two FFT sizes only.
// 3. GPU cuFFTDx LTO  — MathDx enabled; the LTO function pointer is supplied
//                       by the Python dispatch via build_lto_fft.
//
// Selection happens inside `tile_fft_entry`: when `Fwd` is the null-function
// placeholder type (literal `int`, set by the Python dispatch when MathDx is
// unavailable or disabled by `enable_mathdx_fft=False`), `if constexpr` picks
// the scalar branch. The macros at the bottom of this file are thin
// forwarders so the codegen call sites stay uniform across all three regimes.

namespace wp {

// =============================================================================
// FFT helpers (shared between CPU and GPU paths)
// =============================================================================

template <typename T> struct fft_consts {
    static constexpr T two_pi() { return T(6.28318530717958647692528676655900576839433879875021); }
};

// Returns log2(n) if n > 0 is a power of two, otherwise -1.
inline CUDA_CALLABLE int fft_log2_pow2(int n)
{
    if (n <= 0)
        return -1;
    int log_n = 0;
    int x = n;
    while ((x & 1) == 0) {
        x >>= 1;
        ++log_n;
    }
    return (x == 1) ? log_n : -1;
}

// Extract the scalar component type from a wp::vec_t<2, T>.
template <typename V> struct fft_vec2_component;
template <typename T> struct fft_vec2_component<vec_t<2, T>> {
    using type = T;
};

// =============================================================================
// CPU sequential primitives
// =============================================================================

// In-place bit-reverse permutation for size n (n must be power of 2).
// Uses the Gold-Rader incremental update: cheap on a single thread because j
// is computed incrementally from the previous step. The cooperative GPU path
// uses a different (closed-form) bit-reverse since this one is inherently
// serial.
template <typename T> inline CUDA_CALLABLE void fft_bit_reverse_permute(vec_t<2, T>* x, int n)
{
    int j = 0;
    for (int i = 1; i < n; ++i) {
        int bit = n >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) {
            vec_t<2, T> tmp = x[i];
            x[i] = x[j];
            x[j] = tmp;
        }
    }
}

// In-place radix-2 iterative Cooley-Tukey FFT of size n (n power of 2).
// direction_sign selects forward (-1) or inverse (+1).
// Twiddles accumulate via w *= wm to amortize trig calls; this is inherently
// serial across butterflies in a stage and would not survive parallelization.
template <typename T> inline CUDA_CALLABLE void fft_radix2_inplace(vec_t<2, T>* x, int n, int direction_sign)
{
    fft_bit_reverse_permute<T>(x, n);

    for (int len = 2; len <= n; len <<= 1) {
        const int half = len >> 1;
        const T theta = T(direction_sign) * fft_consts<T>::two_pi() / T(len);
        const T wm_re = wp::cos(theta);
        const T wm_im = wp::sin(theta);

        for (int k = 0; k < n; k += len) {
            T w_re = T(1);
            T w_im = T(0);
            for (int j = 0; j < half; ++j) {
                const T x_re = x[k + j + half].c[0];
                const T x_im = x[k + j + half].c[1];
                const T t_re = w_re * x_re - w_im * x_im;
                const T t_im = w_re * x_im + w_im * x_re;

                const T u_re = x[k + j].c[0];
                const T u_im = x[k + j].c[1];

                x[k + j].c[0] = u_re + t_re;
                x[k + j].c[1] = u_im + t_im;
                x[k + j + half].c[0] = u_re - t_re;
                x[k + j + half].c[1] = u_im - t_im;

                const T new_w_re = w_re * wm_re - w_im * wm_im;
                const T new_w_im = w_re * wm_im + w_im * wm_re;
                w_re = new_w_re;
                w_im = new_w_im;
            }
        }
    }
}

// Maximum supported FFT size for the non-power-of-2 DFT fallback.
// The fallback allocates two scratch arrays of this size on the stack
// (see fft_dft_inplace below), so 4096 fp64 elements = 64 KiB stack per call.
// Power-of-two sizes use the in-place radix-2 path and are not subject to this cap.
#define WP_FFT_CPU_MAX_DFT_SIZE 4096

// Naive O(N^2) DFT fallback for non-power-of-two sizes (CPU only).
template <typename T> inline CUDA_CALLABLE void fft_dft_inplace(vec_t<2, T>* x, int n, int direction_sign)
{
    assert(n <= WP_FFT_CPU_MAX_DFT_SIZE);

    T out_re[WP_FFT_CPU_MAX_DFT_SIZE];
    T out_im[WP_FFT_CPU_MAX_DFT_SIZE];

    const T base = T(direction_sign) * fft_consts<T>::two_pi() / T(n);

    for (int k = 0; k < n; ++k) {
        T acc_re = T(0);
        T acc_im = T(0);
        for (int j = 0; j < n; ++j) {
            const T theta = base * T(k) * T(j);
            const T c = wp::cos(theta);
            const T s = wp::sin(theta);
            const T xr = x[j].c[0];
            const T xi = x[j].c[1];
            acc_re += c * xr - s * xi;
            acc_im += c * xi + s * xr;
        }
        out_re[k] = acc_re;
        out_im[k] = acc_im;
    }

    for (int k = 0; k < n; ++k) {
        x[k].c[0] = out_re[k];
        x[k].c[1] = out_im[k];
    }
}

// Single-batch in-place FFT dispatcher used by the CPU implementation.
template <typename T> inline CUDA_CALLABLE void fft_inplace(vec_t<2, T>* x, int n, int direction_sign)
{
    if (fft_log2_pow2(n) >= 0) {
        fft_radix2_inplace<T>(x, n, direction_sign);
    } else {
        fft_dft_inplace<T>(x, n, direction_sign);
    }
}

// CPU entry point — block_dim==1, contiguous register-tile data.
// direction_sign: -1 = forward, +1 = inverse (both unnormalized).
template <int DirectionSign, typename Complex>
inline CUDA_CALLABLE void tile_fft_cpu_impl(int batch, int fft_size, Complex* data)
{
    using T = typename fft_vec2_component<Complex>::type;
    for (int b = 0; b < batch; ++b) {
        fft_inplace<T>(data + b * fft_size, fft_size, DirectionSign);
    }
}

// =============================================================================
// GPU cooperative primitives (no MathDx)
// =============================================================================

// Direct (closed-form) bit-reverse — independent per-i so it can be evaluated
// in parallel by every thread that owns an index.
inline CUDA_CALLABLE int fft_bitrev(int i, int log_n)
{
    int r = 0;
    for (int k = 0; k < log_n; ++k) {
        r = (r << 1) | (i & 1);
        i >>= 1;
    }
    return r;
}

// Cooperative in-place radix-2 FFT on a shared-memory buffer of size n
// (n power of 2). Every thread runs the same outer loop in lockstep and
// partitions the n/2 butterflies of each stage across WP_TILE_BLOCK_DIM.
// Twiddles are recomputed per butterfly because the incremental w *= wm
// accumulator used on CPU is inherently serial.
template <typename T>
inline CUDA_CALLABLE void cooperative_fft_radix2(vec_t<2, T>* x, int n, int log_n, int direction_sign)
{
    // Bit-reverse permutation — thread t handles indices i where i % BLOCK_DIM == t.
    // The `i < j` filter ensures each pair is swapped exactly once.
    for (int i = WP_TILE_THREAD_IDX; i < n; i += WP_TILE_BLOCK_DIM) {
        int j = fft_bitrev(i, log_n);
        if (i < j) {
            vec_t<2, T> tmp = x[i];
            x[i] = x[j];
            x[j] = tmp;
        }
    }
    WP_TILE_SYNC();

    // Butterfly stages.
    for (int len = 2; len <= n; len <<= 1) {
        const int half = len >> 1;
        const T theta_base = T(direction_sign) * fft_consts<T>::two_pi() / T(len);
        const int n_butterflies = n >> 1;

        for (int b_idx = WP_TILE_THREAD_IDX; b_idx < n_butterflies; b_idx += WP_TILE_BLOCK_DIM) {
            const int group = b_idx / half;
            const int pos = b_idx - group * half;
            const int k = group * len;
            const int idx_lo = k + pos;
            const int idx_hi = idx_lo + half;

            const T theta = theta_base * T(pos);
            const T w_re = wp::cos(theta);
            const T w_im = wp::sin(theta);

            const T x_re = x[idx_hi].c[0];
            const T x_im = x[idx_hi].c[1];
            const T t_re = w_re * x_re - w_im * x_im;
            const T t_im = w_re * x_im + w_im * x_re;

            const T u_re = x[idx_lo].c[0];
            const T u_im = x[idx_lo].c[1];

            x[idx_lo].c[0] = u_re + t_re;
            x[idx_lo].c[1] = u_im + t_im;
            x[idx_hi].c[0] = u_re - t_re;
            x[idx_hi].c[1] = u_im - t_im;
        }
        WP_TILE_SYNC();
    }
}

// GPU cooperative entry point — uses a shared-memory scratch of
// `shared_bytes` (== fft_size * sizeof(Complex)) per batch, scattering the
// strided register-tile layout into a contiguous batch, doing the cooperative
// FFT in place, then gathering back.
//
// Power-of-two fft_size only; the Python dispatch raises a clear error for
// other sizes and points at MathDx or the CPU path.
template <int DirectionSign, typename Complex, typename Tile>
inline CUDA_CALLABLE void tile_fft_gpu_impl(int batch, int ept, int shared_bytes, Tile& Xinout)
{
    using T = typename fft_vec2_component<Complex>::type;
    const int fft_size = ept * WP_TILE_BLOCK_DIM;
    const int log_n = fft_log2_pow2(fft_size);
    assert(log_n >= 0);  // dispatch should have rejected non-pow-2

    char* scratch_bytes = (char*)wp::tile_shared_storage_t::alloc(shared_bytes);
    Complex* scratch = reinterpret_cast<Complex*>(scratch_bytes);

    for (int b = 0; b < batch; ++b) {
        // Scatter strided register layout to contiguous shared memory.
        // Thread t's register `b*ept + r_in` maps to intra-batch position
        // `t + r_in*BLOCK_DIM`, so the scatter and gather access patterns
        // mirror each other.
        for (int r_in = 0; r_in < ept; ++r_in) {
            const int reg = b * ept + r_in;
            const int intra = WP_TILE_THREAD_IDX + r_in * WP_TILE_BLOCK_DIM;
            scratch[intra] = Xinout.data[reg];
        }
        WP_TILE_SYNC();

        cooperative_fft_radix2<T>(reinterpret_cast<vec_t<2, T>*>(scratch), fft_size, log_n, DirectionSign);

        for (int r_in = 0; r_in < ept; ++r_in) {
            const int reg = b * ept + r_in;
            const int intra = WP_TILE_THREAD_IDX + r_in * WP_TILE_BLOCK_DIM;
            Xinout.data[reg] = scratch[intra];
        }
        WP_TILE_SYNC();
    }

    wp::tile_shared_storage_t::alloc(-shared_bytes);
}

// =============================================================================
// Templated entry point
// =============================================================================
//
// Mirrors the matmul pattern: a single templated function selects between the
// scalar (CPU or GPU cooperative) and cuFFTDx LTO branches via if constexpr
// on `wp_is_null_func<Fwd>`. The Python dispatch passes a literal `0` for
// `fun_forward` when no LTO is generated (CPU, or when `enable_mathdx_fft` is
// false on a MathDx-enabled build), routing into the scalar branch.
//
// Ept is a non-type template parameter so the cuFFTDx branch can stack-allocate
// `Complex data[Ept]` without a VLA and so the call has a stable type.

template <int DirectionSign, typename Complex, int Ept, typename Fwd, typename Tile>
inline CUDA_CALLABLE void tile_fft_entry(Fwd fun_forward, int shared_bytes, int batch, Tile& Xinout)
{
    if constexpr (wp_is_null_func<Fwd>::value) {
#if !defined(__CUDA_ARCH__)
        // CPU sequential — block_dim==1 makes Xinout.data row-major contiguous.
        tile_fft_cpu_impl<DirectionSign, Complex>(batch, Ept, Xinout.data);
#else
        // GPU cooperative on a shared-memory scratch.
        tile_fft_gpu_impl<DirectionSign, Complex>(batch, Ept, shared_bytes, Xinout);
#endif
    } else {
        // GPU cuFFTDx LTO — call the per-batch LTO function with a per-thread
        // register staging buffer, exactly the original macro body.
        char* buffer = (char*)wp::tile_shared_storage_t::alloc(shared_bytes);
        // TODO(lcambier): use a properly overaligned complex type that matches
        // cuFFTDx's expectation and remove the need for alignas(16).
        alignas(16) Complex data[Ept];
        for (int b = 0; b < batch; ++b) {
            Complex* inout = Xinout.data + b * Ept;
            memcpy(data, inout, sizeof(Complex) * Ept);
            fun_forward(data, buffer);
            memcpy(inout, data, sizeof(Complex) * Ept);
            WP_TILE_SYNC();
        }
        wp::tile_shared_storage_t::alloc(-shared_bytes);
    }
}

}  // namespace wp

// =============================================================================
// Public dispatch macros (thin forwarders)
// =============================================================================
//
// `function_name` is the LTO symbol for this direction; `backward_function_name`
// is the LTO symbol for the *opposite* direction, used by the adjoint.
// The Python dispatch passes a literal `0` for both when no LTO is generated.

#define tile_fft(function_name, backward_function_name, dtype, shared_memory_size, batch_size, ept, Xinout) \
     wp::tile_fft_entry<-1, dtype, (int)(ept)>( \
         function_name, (int)(shared_memory_size), (int)(batch_size), Xinout)

#define tile_ifft(function_name, backward_function_name, dtype, shared_memory_size, batch_size, ept, Xinout) \
     wp::tile_fft_entry<+1, dtype, (int)(ept)>( \
         function_name, (int)(shared_memory_size), (int)(batch_size), Xinout)

// Adjoint of FFT is IFFT (unnormalized) on the output gradient — the +1
// direction is paired with `backward_function_name` (the inverse LTO).
#define adj_tile_fft(                                                                                                  \
    function_name, backward_function_name, dtype, shared_memory_size, batch_size, ept, Xinout, adj_function_name,      \
    adj_backward_function_name, adj_dtype, adj_shared_memory_size, adj_batch_size, adj_ept, adj_Xinout                 \
) \
     wp::tile_fft_entry<+1, dtype, (int)(ept)>( \
         backward_function_name, (int)(shared_memory_size), (int)(batch_size), adj_Xinout)

// Adjoint of IFFT is FFT (unnormalized) on the output gradient — the -1
// direction is paired with `backward_function_name` (the forward LTO).
#define adj_tile_ifft(                                                                                                 \
    function_name, backward_function_name, dtype, shared_memory_size, batch_size, ept, Xinout, adj_function_name,      \
    adj_backward_function_name, adj_dtype, adj_shared_memory_size, adj_batch_size, adj_ept, adj_Xinout                 \
) \
     wp::tile_fft_entry<-1, dtype, (int)(ept)>( \
         backward_function_name, (int)(shared_memory_size), (int)(batch_size), adj_Xinout)
