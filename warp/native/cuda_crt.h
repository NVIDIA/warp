/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

// Attributes otherwise defined by CUDA's crt/host_defines.h
#define __constant__ __attribute__((constant))
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __host__ __attribute__((host))
#define __shared__ __attribute__((shared))
#define __forceinline__ __attribute__((always_inline))

using size_t = unsigned long long;

// stdio.h
extern "C" __device__ int printf(const char* format, ... );

// assert.h
extern __host__ __device__ void __assertfail(const char * __assertion, 
                                             const char *__file,
                                             unsigned int __line,
                                             const char *__function,
                                             size_t charsize);

#if defined(NDEBUG)
#define assert(e) (static_cast<void>(0))
#else /* !NDEBUG */
#define __ASSERT_STR_HELPER(x) #x
#define assert(e) ((e) ? static_cast<void>(0)\
                       : __assertfail(__ASSERT_STR_HELPER(e), __FILE__,\
                                      __LINE__, __PRETTY_FUNCTION__,\
                                      sizeof(char)))
#endif

#define __device_forceinline__ static __device__ inline __forceinline__

// Implementations of CUDA builtin variables
struct __threadIdx_t
{
    __declspec(property(get = get_x)) unsigned int x;
    __declspec(property(get = get_y)) unsigned int y;
    __declspec(property(get = get_z)) unsigned int z;

    __device_forceinline__ unsigned int get_x() { return __nvvm_read_ptx_sreg_tid_x(); }
    __device_forceinline__ unsigned int get_y() { return __nvvm_read_ptx_sreg_tid_y(); }
    __device_forceinline__ unsigned int get_z() { return __nvvm_read_ptx_sreg_tid_z(); }
};

struct __blockIdx_t
{
    __declspec(property(get = get_x)) unsigned int x;
    __declspec(property(get = get_y)) unsigned int y;
    __declspec(property(get = get_z)) unsigned int z;

    __device_forceinline__ unsigned int get_x() { return __nvvm_read_ptx_sreg_ctaid_x(); }
    __device_forceinline__ unsigned int get_y() { return __nvvm_read_ptx_sreg_ctaid_y(); }
    __device_forceinline__ unsigned int get_z() { return __nvvm_read_ptx_sreg_ctaid_z(); }
};

struct __blockDim_t
{
    __declspec(property(get = get_x)) unsigned int x;
    __declspec(property(get = get_y)) unsigned int y;
    __declspec(property(get = get_z)) unsigned int z;

    __device_forceinline__ unsigned int get_x() { return __nvvm_read_ptx_sreg_ntid_x(); }
    __device_forceinline__ unsigned int get_y() { return __nvvm_read_ptx_sreg_ntid_y(); }
    __device_forceinline__ unsigned int get_z() { return __nvvm_read_ptx_sreg_ntid_z(); }
};

struct __gridDim_t
{
    __declspec(property(get = get_x)) unsigned int x;
    __declspec(property(get = get_y)) unsigned int y;
    __declspec(property(get = get_z)) unsigned int z;

    __device_forceinline__ unsigned int get_x() { return __nvvm_read_ptx_sreg_nctaid_x(); }
    __device_forceinline__ unsigned int get_y() { return __nvvm_read_ptx_sreg_nctaid_y(); }
    __device_forceinline__ unsigned int get_z() { return __nvvm_read_ptx_sreg_nctaid_z(); }
};

extern const __device__ __threadIdx_t threadIdx;
extern const __device__ __blockIdx_t blockIdx;
extern const __device__ __blockDim_t blockDim;
extern const __device__ __gridDim_t gridDim;

// Forward declarations of libdevice functions
extern "C" {

__device__ int __nv_abs(int a);
__device__ double __nv_acos(double a);
__device__ float __nv_acosf(float a);
__device__ double __nv_acosh(double a);
__device__ float __nv_acoshf(float a);
__device__ double __nv_asin(double a);
__device__ float __nv_asinf(float a);
__device__ double __nv_asinh(double a);
__device__ float __nv_asinhf(float a);
__device__ double __nv_atan2(double a, double b);
__device__ float __nv_atan2f(float a, float b);
__device__ double __nv_atan(double a);
__device__ float __nv_atanf(float a);
__device__ double __nv_atanh(double a);
__device__ float __nv_atanhf(float a);
__device__ int __nv_brev(int a);
__device__ long long __nv_brevll(long long a);
__device__ int __nv_byte_perm(int a, int b, int c);
__device__ double __nv_cbrt(double a);
__device__ float __nv_cbrtf(float a);
__device__ double __nv_ceil(double a);
__device__ float __nv_ceilf(float a);
__device__ int __nv_clz(int a);
__device__ int __nv_clzll(long long a);
__device__ double __nv_copysign(double a, double b);
__device__ float __nv_copysignf(float a, float b);
__device__ double __nv_cos(double a);
__device__ float __nv_cosf(float a);
__device__ double __nv_cosh(double a);
__device__ float __nv_coshf(float a);
__device__ double __nv_cospi(double a);
__device__ float __nv_cospif(float a);
__device__ double __nv_cyl_bessel_i0(double a);
__device__ float __nv_cyl_bessel_i0f(float a);
__device__ double __nv_cyl_bessel_i1(double a);
__device__ float __nv_cyl_bessel_i1f(float a);
__device__ double __nv_dadd_rd(double a, double b);
__device__ double __nv_dadd_rn(double a, double b);
__device__ double __nv_dadd_ru(double a, double b);
__device__ double __nv_dadd_rz(double a, double b);
__device__ double __nv_ddiv_rd(double a, double b);
__device__ double __nv_ddiv_rn(double a, double b);
__device__ double __nv_ddiv_ru(double a, double b);
__device__ double __nv_ddiv_rz(double a, double b);
__device__ double __nv_dmul_rd(double a, double b);
__device__ double __nv_dmul_rn(double a, double b);
__device__ double __nv_dmul_ru(double a, double b);
__device__ double __nv_dmul_rz(double a, double b);
__device__ float __nv_double2float_rd(double a);
__device__ float __nv_double2float_rn(double a);
__device__ float __nv_double2float_ru(double a);
__device__ float __nv_double2float_rz(double a);
__device__ int __nv_double2hiint(double a);
__device__ int __nv_double2int_rd(double a);
__device__ int __nv_double2int_rn(double a);
__device__ int __nv_double2int_ru(double a);
__device__ int __nv_double2int_rz(double a);
__device__ long long __nv_double2ll_rd(double a);
__device__ long long __nv_double2ll_rn(double a);
__device__ long long __nv_double2ll_ru(double a);
__device__ long long __nv_double2ll_rz(double a);
__device__ int __nv_double2loint(double a);
__device__ unsigned int __nv_double2uint_rd(double a);
__device__ unsigned int __nv_double2uint_rn(double a);
__device__ unsigned int __nv_double2uint_ru(double a);
__device__ unsigned int __nv_double2uint_rz(double a);
__device__ unsigned long long __nv_double2ull_rd(double a);
__device__ unsigned long long __nv_double2ull_rn(double a);
__device__ unsigned long long __nv_double2ull_ru(double a);
__device__ unsigned long long __nv_double2ull_rz(double a);
__device__ unsigned long long __nv_double_as_longlong(double a);
__device__ double __nv_drcp_rd(double a);
__device__ double __nv_drcp_rn(double a);
__device__ double __nv_drcp_ru(double a);
__device__ double __nv_drcp_rz(double a);
__device__ double __nv_dsqrt_rd(double a);
__device__ double __nv_dsqrt_rn(double a);
__device__ double __nv_dsqrt_ru(double a);
__device__ double __nv_dsqrt_rz(double a);
__device__ double __nv_dsub_rd(double a, double b);
__device__ double __nv_dsub_rn(double a, double b);
__device__ double __nv_dsub_ru(double a, double b);
__device__ double __nv_dsub_rz(double a, double b);
__device__ double __nv_erfc(double a);
__device__ float __nv_erfcf(float a);
__device__ double __nv_erfcinv(double a);
__device__ float __nv_erfcinvf(float a);
__device__ double __nv_erfcx(double a);
__device__ float __nv_erfcxf(float a);
__device__ double __nv_erf(double a);
__device__ float __nv_erff(float a);
__device__ double __nv_erfinv(double a);
__device__ float __nv_erfinvf(float a);
__device__ double __nv_exp10(double a);
__device__ float __nv_exp10f(float a);
__device__ double __nv_exp2(double a);
__device__ float __nv_exp2f(float a);
__device__ double __nv_exp(double a);
__device__ float __nv_expf(float a);
__device__ double __nv_expm1(double a);
__device__ float __nv_expm1f(float a);
__device__ double __nv_fabs(double a);
__device__ float __nv_fabsf(float a);
__device__ float __nv_fadd_rd(float a, float b);
__device__ float __nv_fadd_rn(float a, float b);
__device__ float __nv_fadd_ru(float a, float b);
__device__ float __nv_fadd_rz(float a, float b);
__device__ float __nv_fast_cosf(float a);
__device__ float __nv_fast_exp10f(float a);
__device__ float __nv_fast_expf(float a);
__device__ float __nv_fast_fdividef(float a, float b);
__device__ float __nv_fast_log10f(float a);
__device__ float __nv_fast_log2f(float a);
__device__ float __nv_fast_logf(float a);
__device__ float __nv_fast_powf(float a, float b);
__device__ void __nv_fast_sincosf(float a, float *s, float *c);
__device__ float __nv_fast_sinf(float a);
__device__ float __nv_fast_tanf(float a);
__device__ double __nv_fdim(double a, double b);
__device__ float __nv_fdimf(float a, float b);
__device__ float __nv_fdiv_rd(float a, float b);
__device__ float __nv_fdiv_rn(float a, float b);
__device__ float __nv_fdiv_ru(float a, float b);
__device__ float __nv_fdiv_rz(float a, float b);
__device__ int __nv_ffs(int a);
__device__ int __nv_ffsll(long long a);
__device__ int __nv_finitef(float a);
__device__ unsigned short __nv_float2half_rn(float a);
__device__ int __nv_float2int_rd(float a);
__device__ int __nv_float2int_rn(float a);
__device__ int __nv_float2int_ru(float a);
__device__ int __nv_float2int_rz(float a);
__device__ long long __nv_float2ll_rd(float a);
__device__ long long __nv_float2ll_rn(float a);
__device__ long long __nv_float2ll_ru(float a);
__device__ long long __nv_float2ll_rz(float a);
__device__ unsigned int __nv_float2uint_rd(float a);
__device__ unsigned int __nv_float2uint_rn(float a);
__device__ unsigned int __nv_float2uint_ru(float a);
__device__ unsigned int __nv_float2uint_rz(float a);
__device__ unsigned long long __nv_float2ull_rd(float a);
__device__ unsigned long long __nv_float2ull_rn(float a);
__device__ unsigned long long __nv_float2ull_ru(float a);
__device__ unsigned long long __nv_float2ull_rz(float a);
__device__ int __nv_float_as_int(float a);
__device__ unsigned int __nv_float_as_uint(float a);
__device__ double __nv_floor(double a);
__device__ float __nv_floorf(float a);
__device__ double __nv_fma(double a, double b, double c);
__device__ float __nv_fmaf(float a, float b, float c);
__device__ float __nv_fmaf_ieee_rd(float a, float b, float c);
__device__ float __nv_fmaf_ieee_rn(float a, float b, float c);
__device__ float __nv_fmaf_ieee_ru(float a, float b, float c);
__device__ float __nv_fmaf_ieee_rz(float a, float b, float c);
__device__ float __nv_fmaf_rd(float a, float b, float c);
__device__ float __nv_fmaf_rn(float a, float b, float c);
__device__ float __nv_fmaf_ru(float a, float b, float c);
__device__ float __nv_fmaf_rz(float a, float b, float c);
__device__ double __nv_fma_rd(double a, double b, double c);
__device__ double __nv_fma_rn(double a, double b, double c);
__device__ double __nv_fma_ru(double a, double b, double c);
__device__ double __nv_fma_rz(double a, double b, double c);
__device__ double __nv_fmax(double a, double b);
__device__ float __nv_fmaxf(float a, float b);
__device__ double __nv_fmin(double a, double b);
__device__ float __nv_fminf(float a, float b);
__device__ double __nv_fmod(double a, double b);
__device__ float __nv_fmodf(float a, float b);
__device__ float __nv_fmul_rd(float a, float b);
__device__ float __nv_fmul_rn(float a, float b);
__device__ float __nv_fmul_ru(float a, float b);
__device__ float __nv_fmul_rz(float a, float b);
__device__ float __nv_frcp_rd(float a);
__device__ float __nv_frcp_rn(float a);
__device__ float __nv_frcp_ru(float a);
__device__ float __nv_frcp_rz(float a);
__device__ double __nv_frexp(double a, int *b);
__device__ float __nv_frexpf(float a, int *b);
__device__ float __nv_frsqrt_rn(float a);
__device__ float __nv_fsqrt_rd(float a);
__device__ float __nv_fsqrt_rn(float a);
__device__ float __nv_fsqrt_ru(float a);
__device__ float __nv_fsqrt_rz(float a);
__device__ float __nv_fsub_rd(float a, float b);
__device__ float __nv_fsub_rn(float a, float b);
__device__ float __nv_fsub_ru(float a, float b);
__device__ float __nv_fsub_rz(float a, float b);
__device__ int __nv_hadd(int a, int b);
__device__ float __nv_half2float(unsigned short h);
__device__ double __nv_hiloint2double(int a, int b);
__device__ double __nv_hypot(double a, double b);
__device__ float __nv_hypotf(float a, float b);
__device__ int __nv_ilogb(double a);
__device__ int __nv_ilogbf(float a);
__device__ double __nv_int2double_rn(int a);
__device__ float __nv_int2float_rd(int a);
__device__ float __nv_int2float_rn(int a);
__device__ float __nv_int2float_ru(int a);
__device__ float __nv_int2float_rz(int a);
__device__ float __nv_int_as_float(int a);
__device__ int __nv_isfinited(double a);
__device__ int __nv_isinfd(double a);
__device__ int __nv_isinff(float a);
__device__ int __nv_isnand(double a);
__device__ int __nv_isnanf(float a);
__device__ double __nv_j0(double a);
__device__ float __nv_j0f(float a);
__device__ double __nv_j1(double a);
__device__ float __nv_j1f(float a);
__device__ float __nv_jnf(int a, float b);
__device__ double __nv_jn(int a, double b);
__device__ double __nv_ldexp(double a, int b);
__device__ float __nv_ldexpf(float a, int b);
__device__ double __nv_lgamma(double a);
__device__ float __nv_lgammaf(float a);
__device__ double __nv_ll2double_rd(long long a);
__device__ double __nv_ll2double_rn(long long a);
__device__ double __nv_ll2double_ru(long long a);
__device__ double __nv_ll2double_rz(long long a);
__device__ float __nv_ll2float_rd(long long a);
__device__ float __nv_ll2float_rn(long long a);
__device__ float __nv_ll2float_ru(long long a);
__device__ float __nv_ll2float_rz(long long a);
__device__ long long __nv_llabs(long long a);
__device__ long long __nv_llmax(long long a, long long b);
__device__ long long __nv_llmin(long long a, long long b);
__device__ long long __nv_llrint(double a);
__device__ long long __nv_llrintf(float a);
__device__ long long __nv_llround(double a);
__device__ long long __nv_llroundf(float a);
__device__ double __nv_log10(double a);
__device__ float __nv_log10f(float a);
__device__ double __nv_log1p(double a);
__device__ float __nv_log1pf(float a);
__device__ double __nv_log2(double a);
__device__ float __nv_log2f(float a);
__device__ double __nv_logb(double a);
__device__ float __nv_logbf(float a);
__device__ double __nv_log(double a);
__device__ float __nv_logf(float a);
__device__ double __nv_longlong_as_double(long long a);
__device__ int __nv_max(int a, int b);
__device__ int __nv_min(int a, int b);
__device__ double __nv_modf(double a, double *b);
__device__ float __nv_modff(float a, float *b);
__device__ int __nv_mul24(int a, int b);
__device__ long long __nv_mul64hi(long long a, long long b);
__device__ int __nv_mulhi(int a, int b);
__device__ double __nv_nan(const signed char *a);
__device__ float __nv_nanf(const signed char *a);
__device__ double __nv_nearbyint(double a);
__device__ float __nv_nearbyintf(float a);
__device__ double __nv_nextafter(double a, double b);
__device__ float __nv_nextafterf(float a, float b);
__device__ double __nv_norm3d(double a, double b, double c);
__device__ float __nv_norm3df(float a, float b, float c);
__device__ double __nv_norm4d(double a, double b, double c, double d);
__device__ float __nv_norm4df(float a, float b, float c, float d);
__device__ double __nv_normcdf(double a);
__device__ float __nv_normcdff(float a);
__device__ double __nv_normcdfinv(double a);
__device__ float __nv_normcdfinvf(float a);
__device__ float __nv_normf(int a, const float *b);
__device__ double __nv_norm(int a, const double *b);
__device__ int __nv_popc(int a);
__device__ int __nv_popcll(long long a);
__device__ double __nv_pow(double a, double b);
__device__ float __nv_powf(float a, float b);
__device__ double __nv_powi(double a, int b);
__device__ float __nv_powif(float a, int b);
__device__ double __nv_rcbrt(double a);
__device__ float __nv_rcbrtf(float a);
__device__ double __nv_rcp64h(double a);
__device__ double __nv_remainder(double a, double b);
__device__ float __nv_remainderf(float a, float b);
__device__ double __nv_remquo(double a, double b, int *c);
__device__ float __nv_remquof(float a, float b, int *c);
__device__ int __nv_rhadd(int a, int b);
__device__ double __nv_rhypot(double a, double b);
__device__ float __nv_rhypotf(float a, float b);
__device__ double __nv_rint(double a);
__device__ float __nv_rintf(float a);
__device__ double __nv_rnorm3d(double a, double b, double c);
__device__ float __nv_rnorm3df(float a, float b, float c);
__device__ double __nv_rnorm4d(double a, double b, double c, double d);
__device__ float __nv_rnorm4df(float a, float b, float c, float d);
__device__ float __nv_rnormf(int a, const float *b);
__device__ double __nv_rnorm(int a, const double *b);
__device__ double __nv_round(double a);
__device__ float __nv_roundf(float a);
__device__ double __nv_rsqrt(double a);
__device__ float __nv_rsqrtf(float a);
__device__ int __nv_sad(int a, int b, int c);
__device__ float __nv_saturatef(float a);
__device__ double __nv_scalbn(double a, int b);
__device__ float __nv_scalbnf(float a, int b);
__device__ int __nv_signbitd(double a);
__device__ int __nv_signbitf(float a);
__device__ void __nv_sincos(double a, double *b, double *c);
__device__ void __nv_sincosf(float a, float *b, float *c);
__device__ void __nv_sincospi(double a, double *b, double *c);
__device__ void __nv_sincospif(float a, float *b, float *c);
__device__ double __nv_sin(double a);
__device__ float __nv_sinf(float a);
__device__ double __nv_sinh(double a);
__device__ float __nv_sinhf(float a);
__device__ double __nv_sinpi(double a);
__device__ float __nv_sinpif(float a);
__device__ double __nv_sqrt(double a);
__device__ float __nv_sqrtf(float a);
__device__ double __nv_tan(double a);
__device__ float __nv_tanf(float a);
__device__ double __nv_tanh(double a);
__device__ float __nv_tanhf(float a);
__device__ double __nv_tgamma(double a);
__device__ float __nv_tgammaf(float a);
__device__ double __nv_trunc(double a);
__device__ float __nv_truncf(float a);
__device__ int __nv_uhadd(unsigned int a, unsigned int b);
__device__ double __nv_uint2double_rn(unsigned int i);
__device__ float __nv_uint2float_rd(unsigned int a);
__device__ float __nv_uint2float_rn(unsigned int a);
__device__ float __nv_uint2float_ru(unsigned int a);
__device__ float __nv_uint2float_rz(unsigned int a);
__device__ float __nv_uint_as_float(unsigned int a);
__device__ double __nv_ull2double_rd(unsigned long long a);
__device__ double __nv_ull2double_rn(unsigned long long a);
__device__ double __nv_ull2double_ru(unsigned long long a);
__device__ double __nv_ull2double_rz(unsigned long long a);
__device__ float __nv_ull2float_rd(unsigned long long a);
__device__ float __nv_ull2float_rn(unsigned long long a);
__device__ float __nv_ull2float_ru(unsigned long long a);
__device__ float __nv_ull2float_rz(unsigned long long a);
__device__ unsigned long long __nv_ullmax(unsigned long long a, unsigned long long b);
__device__ unsigned long long __nv_ullmin(unsigned long long a, unsigned long long b);
__device__ unsigned int __nv_umax(unsigned int a, unsigned int b);
__device__ unsigned int __nv_umin(unsigned int a, unsigned int b);
__device__ unsigned int __nv_umul24(unsigned int a, unsigned int b);
__device__ unsigned long long __nv_umul64hi(unsigned long long a, unsigned long long b);
__device__ unsigned int __nv_umulhi(unsigned int a, unsigned int b);
__device__ unsigned int __nv_urhadd(unsigned int a, unsigned int b);
__device__ unsigned int __nv_usad(unsigned int a, unsigned int b, unsigned int c);
__device__ double __nv_y0(double a);
__device__ float __nv_y0f(float a);
__device__ double __nv_y1(double a);
__device__ float __nv_y1f(float a);
__device__ float __nv_ynf(int a, float b);
__device__ double __nv_yn(int a, double b);

}  // extern "C"

// Implementation of CUDA intrinsics
__device_forceinline__ int __all(int a) { return __nvvm_vote_all(a); }
__device_forceinline__ int __any(int a) { return __nvvm_vote_any(a); }
__device_forceinline__ unsigned int __ballot(int a) { return __nvvm_vote_ballot(a); }
__device_forceinline__ unsigned int __brev(unsigned int a) { return __nv_brev(a); }
__device_forceinline__ unsigned long long __brevll(unsigned long long a) { return __nv_brevll(a); }
__device_forceinline__ void __brkpt() { __asm__ __volatile__("brkpt;"); }
__device_forceinline__ void __brkpt(int a) { __brkpt(); }
__device_forceinline__ unsigned int __byte_perm(unsigned int a, unsigned int b, unsigned int c) { return __nv_byte_perm(a, b, c); }
__device_forceinline__ int __clz(int a) { return __nv_clz(a); }
__device_forceinline__ int __clzll(long long a) { return __nv_clzll(a); }
__device_forceinline__ float __cosf(float a) { return __nv_fast_cosf(a); }
__device_forceinline__ double __dAtomicAdd(double *p, double v) { return __nvvm_atom_add_gen_d(p, v); }
__device_forceinline__ double __dAtomicAdd_block(double *p, double v) { return __nvvm_atom_cta_add_gen_d(p, v); }
__device_forceinline__ double __dAtomicAdd_system(double *p, double v) { return __nvvm_atom_sys_add_gen_d(p, v); }
__device_forceinline__ double __dadd_rd(double a, double b) { return __nv_dadd_rd(a, b); }
__device_forceinline__ double __dadd_rn(double a, double b) { return __nv_dadd_rn(a, b); }
__device_forceinline__ double __dadd_ru(double a, double b) { return __nv_dadd_ru(a, b); }
__device_forceinline__ double __dadd_rz(double a, double b) { return __nv_dadd_rz(a, b); }
__device_forceinline__ double __ddiv_rd(double a, double b) { return __nv_ddiv_rd(a, b); }
__device_forceinline__ double __ddiv_rn(double a, double b) { return __nv_ddiv_rn(a, b); }
__device_forceinline__ double __ddiv_ru(double a, double b) { return __nv_ddiv_ru(a, b); }
__device_forceinline__ double __ddiv_rz(double a, double b) { return __nv_ddiv_rz(a, b); }
__device_forceinline__ double __dmul_rd(double a, double b) { return __nv_dmul_rd(a, b); }
__device_forceinline__ double __dmul_rn(double a, double b) { return __nv_dmul_rn(a, b); }
__device_forceinline__ double __dmul_ru(double a, double b) { return __nv_dmul_ru(a, b); }
__device_forceinline__ double __dmul_rz(double a, double b) { return __nv_dmul_rz(a, b); }
__device_forceinline__ float __double2float_rd(double a) { return __nv_double2float_rd(a); }
__device_forceinline__ float __double2float_rn(double a) { return __nv_double2float_rn(a); }
__device_forceinline__ float __double2float_ru(double a) { return __nv_double2float_ru(a); }
__device_forceinline__ float __double2float_rz(double a) { return __nv_double2float_rz(a); }
__device_forceinline__ int __double2hiint(double a) { return __nv_double2hiint(a); }
__device_forceinline__ int __double2int_rd(double a) { return __nv_double2int_rd(a); }
__device_forceinline__ int __double2int_rn(double a) { return __nv_double2int_rn(a); }
__device_forceinline__ int __double2int_ru(double a) { return __nv_double2int_ru(a); }
__device_forceinline__ int __double2int_rz(double a) { return __nv_double2int_rz(a); }
__device_forceinline__ long long __double2ll_rd(double a) { return __nv_double2ll_rd(a); }
__device_forceinline__ long long __double2ll_rn(double a) { return __nv_double2ll_rn(a); }
__device_forceinline__ long long __double2ll_ru(double a) { return __nv_double2ll_ru(a); }
__device_forceinline__ long long __double2ll_rz(double a) { return __nv_double2ll_rz(a); }
__device_forceinline__ int __double2loint(double a) { return __nv_double2loint(a); }
__device_forceinline__ unsigned int __double2uint_rd(double a) { return __nv_double2uint_rd(a); }
__device_forceinline__ unsigned int __double2uint_rn(double a) { return __nv_double2uint_rn(a); }
__device_forceinline__ unsigned int __double2uint_ru(double a) { return __nv_double2uint_ru(a); }
__device_forceinline__ unsigned int __double2uint_rz(double a) { return __nv_double2uint_rz(a); }
__device_forceinline__ unsigned long long __double2ull_rd(double a) { return __nv_double2ull_rd(a); }
__device_forceinline__ unsigned long long __double2ull_rn(double a) { return __nv_double2ull_rn(a); }
__device_forceinline__ unsigned long long __double2ull_ru(double a) { return __nv_double2ull_ru(a); }
__device_forceinline__ unsigned long long __double2ull_rz(double a) { return __nv_double2ull_rz(a); }
__device_forceinline__ long long __double_as_longlong(double a) { return __nv_double_as_longlong(a); }
__device_forceinline__ double __drcp_rd(double a) { return __nv_drcp_rd(a); }
__device_forceinline__ double __drcp_rn(double a) { return __nv_drcp_rn(a); }
__device_forceinline__ double __drcp_ru(double a) { return __nv_drcp_ru(a); }
__device_forceinline__ double __drcp_rz(double a) { return __nv_drcp_rz(a); }
__device_forceinline__ double __dsqrt_rd(double a) { return __nv_dsqrt_rd(a); }
__device_forceinline__ double __dsqrt_rn(double a) { return __nv_dsqrt_rn(a); }
__device_forceinline__ double __dsqrt_ru(double a) { return __nv_dsqrt_ru(a); }
__device_forceinline__ double __dsqrt_rz(double a) { return __nv_dsqrt_rz(a); }
__device_forceinline__ double __dsub_rd(double a, double b) { return __nv_dsub_rd(a, b); }
__device_forceinline__ double __dsub_rn(double a, double b) { return __nv_dsub_rn(a, b); }
__device_forceinline__ double __dsub_ru(double a, double b) { return __nv_dsub_ru(a, b); }
__device_forceinline__ double __dsub_rz(double a, double b) { return __nv_dsub_rz(a, b); }
__device_forceinline__ float __exp10f(float a) { return __nv_fast_exp10f(a); }
__device_forceinline__ float __expf(float a) { return __nv_fast_expf(a); }
__device_forceinline__ float __fAtomicAdd(float *p, float v) { return __nvvm_atom_add_gen_f(p, v); }
__device_forceinline__ float __fAtomicAdd_block(float *p, float v) { return __nvvm_atom_cta_add_gen_f(p, v); }
__device_forceinline__ float __fAtomicAdd_system(float *p, float v) { return __nvvm_atom_sys_add_gen_f(p, v); }
__device_forceinline__ float __fAtomicExch(float *p, float v) { return __nv_int_as_float(__nvvm_atom_xchg_gen_i((int *)p, __nv_float_as_int(v))); }
__device_forceinline__ float __fAtomicExch_block(float *p, float v) { return __nv_int_as_float(__nvvm_atom_cta_xchg_gen_i((int *)p, __nv_float_as_int(v))); }
__device_forceinline__ float __fAtomicExch_system(float *p, float v) { return __nv_int_as_float(__nvvm_atom_sys_xchg_gen_i((int *)p, __nv_float_as_int(v))); }
__device_forceinline__ float __fadd_rd(float a, float b) { return __nv_fadd_rd(a, b); }
__device_forceinline__ float __fadd_rn(float a, float b) { return __nv_fadd_rn(a, b); }
__device_forceinline__ float __fadd_ru(float a, float b) { return __nv_fadd_ru(a, b); }
__device_forceinline__ float __fadd_rz(float a, float b) { return __nv_fadd_rz(a, b); }
__device_forceinline__ float __fdiv_rd(float a, float b) { return __nv_fdiv_rd(a, b); }
__device_forceinline__ float __fdiv_rn(float a, float b) { return __nv_fdiv_rn(a, b); }
__device_forceinline__ float __fdiv_ru(float a, float b) { return __nv_fdiv_ru(a, b); }
__device_forceinline__ float __fdiv_rz(float a, float b) { return __nv_fdiv_rz(a, b); }
__device_forceinline__ float __fdividef(float a, float b) { return __nv_fast_fdividef(a, b); }
__device_forceinline__ int __ffs(int a) { return __nv_ffs(a); }
__device_forceinline__ int __ffsll(long long a) { return __nv_ffsll(a); }
__device_forceinline__ int __finite(double a) { return __nv_isfinited(a); }
__device_forceinline__ int __finitef(float a) { return __nv_finitef(a); }
__device_forceinline__ int __float2int_rd(float a) { return __nv_float2int_rd(a); }
__device_forceinline__ int __float2int_rn(float a) { return __nv_float2int_rn(a); }
__device_forceinline__ int __float2int_ru(float a) { return __nv_float2int_ru(a); }
__device_forceinline__ int __float2int_rz(float a) { return __nv_float2int_rz(a); }
__device_forceinline__ long long __float2ll_rd(float a) { return __nv_float2ll_rd(a); }
__device_forceinline__ long long __float2ll_rn(float a) { return __nv_float2ll_rn(a); }
__device_forceinline__ long long __float2ll_ru(float a) { return __nv_float2ll_ru(a); }
__device_forceinline__ long long __float2ll_rz(float a) { return __nv_float2ll_rz(a); }
__device_forceinline__ unsigned int __float2uint_rd(float a) { return __nv_float2uint_rd(a); }
__device_forceinline__ unsigned int __float2uint_rn(float a) { return __nv_float2uint_rn(a); }
__device_forceinline__ unsigned int __float2uint_ru(float a) { return __nv_float2uint_ru(a); }
__device_forceinline__ unsigned int __float2uint_rz(float a) { return __nv_float2uint_rz(a); }
__device_forceinline__ unsigned long long __float2ull_rd(float a) { return __nv_float2ull_rd(a); }
__device_forceinline__ unsigned long long __float2ull_rn(float a) { return __nv_float2ull_rn(a); }
__device_forceinline__ unsigned long long __float2ull_ru(float a) { return __nv_float2ull_ru(a); }
__device_forceinline__ unsigned long long __float2ull_rz(float a) { return __nv_float2ull_rz(a); }
__device_forceinline__ int __float_as_int(float a) { return __nv_float_as_int(a); }
__device_forceinline__ unsigned int __float_as_uint(float a) { return __nv_float_as_uint(a); }
__device_forceinline__ double __fma_rd(double a, double b, double c) { return __nv_fma_rd(a, b, c); }
__device_forceinline__ double __fma_rn(double a, double b, double c) { return __nv_fma_rn(a, b, c); }
__device_forceinline__ double __fma_ru(double a, double b, double c) { return __nv_fma_ru(a, b, c); }
__device_forceinline__ double __fma_rz(double a, double b, double c) { return __nv_fma_rz(a, b, c); }
__device_forceinline__ float __fmaf_ieee_rd(float a, float b, float c) { return __nv_fmaf_ieee_rd(a, b, c); }
__device_forceinline__ float __fmaf_ieee_rn(float a, float b, float c) { return __nv_fmaf_ieee_rn(a, b, c); }
__device_forceinline__ float __fmaf_ieee_ru(float a, float b, float c) { return __nv_fmaf_ieee_ru(a, b, c); }
__device_forceinline__ float __fmaf_ieee_rz(float a, float b, float c) { return __nv_fmaf_ieee_rz(a, b, c); }
__device_forceinline__ float __fmaf_rd(float a, float b, float c) { return __nv_fmaf_rd(a, b, c); }
__device_forceinline__ float __fmaf_rn(float a, float b, float c) { return __nv_fmaf_rn(a, b, c); }
__device_forceinline__ float __fmaf_ru(float a, float b, float c) { return __nv_fmaf_ru(a, b, c); }
__device_forceinline__ float __fmaf_rz(float a, float b, float c) { return __nv_fmaf_rz(a, b, c); }
__device_forceinline__ float __fmul_rd(float a, float b) { return __nv_fmul_rd(a, b); }
__device_forceinline__ float __fmul_rn(float a, float b) { return __nv_fmul_rn(a, b); }
__device_forceinline__ float __fmul_ru(float a, float b) { return __nv_fmul_ru(a, b); }
__device_forceinline__ float __fmul_rz(float a, float b) { return __nv_fmul_rz(a, b); }
__device_forceinline__ float __frcp_rd(float a) { return __nv_frcp_rd(a); }
__device_forceinline__ float __frcp_rn(float a) { return __nv_frcp_rn(a); }
__device_forceinline__ float __frcp_ru(float a) { return __nv_frcp_ru(a); }
__device_forceinline__ float __frcp_rz(float a) { return __nv_frcp_rz(a); }
__device_forceinline__ float __frsqrt_rn(float a) { return __nv_frsqrt_rn(a); }
__device_forceinline__ float __fsqrt_rd(float a) { return __nv_fsqrt_rd(a); }
__device_forceinline__ float __fsqrt_rn(float a) { return __nv_fsqrt_rn(a); }
__device_forceinline__ float __fsqrt_ru(float a) { return __nv_fsqrt_ru(a); }
__device_forceinline__ float __fsqrt_rz(float a) { return __nv_fsqrt_rz(a); }
__device_forceinline__ float __fsub_rd(float a, float b) { return __nv_fsub_rd(a, b); }
__device_forceinline__ float __fsub_rn(float a, float b) { return __nv_fsub_rn(a, b); }
__device_forceinline__ float __fsub_ru(float a, float b) { return __nv_fsub_ru(a, b); }
__device_forceinline__ float __fsub_rz(float a, float b) { return __nv_fsub_rz(a, b); }
__device_forceinline__ int __hadd(int a, int b) { return __nv_hadd(a, b); }
__device_forceinline__ double __hiloint2double(int a, int b) { return __nv_hiloint2double(a, b); }
__device_forceinline__ int __iAtomicAdd(int *p, int v) { return __nvvm_atom_add_gen_i(p, v); }
__device_forceinline__ int __iAtomicAdd_block(int *p, int v) { return __nvvm_atom_cta_add_gen_i(p, v); }
__device_forceinline__ int __iAtomicAdd_system(int *p, int v) { return __nvvm_atom_sys_add_gen_i(p, v); }
__device_forceinline__ int __iAtomicAnd(int *p, int v) { return __nvvm_atom_and_gen_i(p, v); }
__device_forceinline__ int __iAtomicAnd_block(int *p, int v) { return __nvvm_atom_cta_and_gen_i(p, v); }
__device_forceinline__ int __iAtomicAnd_system(int *p, int v) { return __nvvm_atom_sys_and_gen_i(p, v); }
__device_forceinline__ int __iAtomicCAS(int *p, int cmp, int v) { return __nvvm_atom_cas_gen_i(p, cmp, v); }
__device_forceinline__ int __iAtomicCAS_block(int *p, int cmp, int v) { return __nvvm_atom_cta_cas_gen_i(p, cmp, v); }
__device_forceinline__ int __iAtomicCAS_system(int *p, int cmp, int v) { return __nvvm_atom_sys_cas_gen_i(p, cmp, v); }
__device_forceinline__ int __iAtomicExch(int *p, int v) { return __nvvm_atom_xchg_gen_i(p, v); }
__device_forceinline__ int __iAtomicExch_block(int *p, int v) { return __nvvm_atom_cta_xchg_gen_i(p, v); }
__device_forceinline__ int __iAtomicExch_system(int *p, int v) { return __nvvm_atom_sys_xchg_gen_i(p, v); }
__device_forceinline__ int __iAtomicMax(int *p, int v) { return __nvvm_atom_max_gen_i(p, v); }
__device_forceinline__ int __iAtomicMax_block(int *p, int v) { return __nvvm_atom_cta_max_gen_i(p, v); }
__device_forceinline__ int __iAtomicMax_system(int *p, int v) { return __nvvm_atom_sys_max_gen_i(p, v); }
__device_forceinline__ int __iAtomicMin(int *p, int v) { return __nvvm_atom_min_gen_i(p, v); }
__device_forceinline__ int __iAtomicMin_block(int *p, int v) { return __nvvm_atom_cta_min_gen_i(p, v); }
__device_forceinline__ int __iAtomicMin_system(int *p, int v) { return __nvvm_atom_sys_min_gen_i(p, v); }
__device_forceinline__ int __iAtomicOr(int *p, int v) { return __nvvm_atom_or_gen_i(p, v); }
__device_forceinline__ int __iAtomicOr_block(int *p, int v) { return __nvvm_atom_cta_or_gen_i(p, v); }
__device_forceinline__ int __iAtomicOr_system(int *p, int v) { return __nvvm_atom_sys_or_gen_i(p, v); }
__device_forceinline__ int __iAtomicXor(int *p, int v) { return __nvvm_atom_xor_gen_i(p, v); }
__device_forceinline__ int __iAtomicXor_block(int *p, int v) { return __nvvm_atom_cta_xor_gen_i(p, v); }
__device_forceinline__ int __iAtomicXor_system(int *p, int v) { return __nvvm_atom_sys_xor_gen_i(p, v); }
__device_forceinline__ long long __illAtomicMax(long long *p, long long v) { return __nvvm_atom_max_gen_ll(p, v); }
__device_forceinline__ long long __illAtomicMax_block(long long *p, long long v) { return __nvvm_atom_cta_max_gen_ll(p, v); }
__device_forceinline__ long long __illAtomicMax_system(long long *p, long long v) { return __nvvm_atom_sys_max_gen_ll(p, v); }
__device_forceinline__ long long __illAtomicMin(long long *p, long long v) { return __nvvm_atom_min_gen_ll(p, v); }
__device_forceinline__ long long __illAtomicMin_block(long long *p, long long v) { return __nvvm_atom_cta_min_gen_ll(p, v); }
__device_forceinline__ long long __illAtomicMin_system(long long *p, long long v) { return __nvvm_atom_sys_min_gen_ll(p, v); }
__device_forceinline__ double __int2double_rn(int a) { return __nv_int2double_rn(a); }
__device_forceinline__ float __int2float_rd(int a) { return __nv_int2float_rd(a); }
__device_forceinline__ float __int2float_rn(int a) { return __nv_int2float_rn(a); }
__device_forceinline__ float __int2float_ru(int a) { return __nv_int2float_ru(a); }
__device_forceinline__ float __int2float_rz(int a) { return __nv_int2float_rz(a); }
__device_forceinline__ float __int_as_float(int a) { return __nv_int_as_float(a); }
__device_forceinline__ int __isfinited(double a) { return __nv_isfinited(a); }
__device_forceinline__ int __isinf(double a) { return __nv_isinfd(a); }
__device_forceinline__ int __isinff(float a) { return __nv_isinff(a); }
__device_forceinline__ int __isnan(double a) { return __nv_isnand(a); }
__device_forceinline__ int __isnanf(float a) { return __nv_isnanf(a); }
__device_forceinline__ double __ll2double_rd(long long a) { return __nv_ll2double_rd(a); }
__device_forceinline__ double __ll2double_rn(long long a) { return __nv_ll2double_rn(a); }
__device_forceinline__ double __ll2double_ru(long long a) { return __nv_ll2double_ru(a); }
__device_forceinline__ double __ll2double_rz(long long a) { return __nv_ll2double_rz(a); }
__device_forceinline__ float __ll2float_rd(long long a) { return __nv_ll2float_rd(a); }
__device_forceinline__ float __ll2float_rn(long long a) { return __nv_ll2float_rn(a); }
__device_forceinline__ float __ll2float_ru(long long a) { return __nv_ll2float_ru(a); }
__device_forceinline__ float __ll2float_rz(long long a) { return __nv_ll2float_rz(a); }
__device_forceinline__ long long __llAtomicAnd(long long *p, long long v) { return __nvvm_atom_and_gen_ll(p, v); }
__device_forceinline__ long long __llAtomicAnd_block(long long *p, long long v) { return __nvvm_atom_cta_and_gen_ll(p, v); }
__device_forceinline__ long long __llAtomicAnd_system(long long *p, long long v) { return __nvvm_atom_sys_and_gen_ll(p, v); }
__device_forceinline__ long long __llAtomicOr(long long *p, long long v) { return __nvvm_atom_or_gen_ll(p, v); }
__device_forceinline__ long long __llAtomicOr_block(long long *p, long long v) { return __nvvm_atom_cta_or_gen_ll(p, v); }
__device_forceinline__ long long __llAtomicOr_system(long long *p, long long v) { return __nvvm_atom_sys_or_gen_ll(p, v); }
__device_forceinline__ long long __llAtomicXor(long long *p, long long v) { return __nvvm_atom_xor_gen_ll(p, v); }
__device_forceinline__ long long __llAtomicXor_block(long long *p, long long v) { return __nvvm_atom_cta_xor_gen_ll(p, v); }
__device_forceinline__ long long __llAtomicXor_system(long long *p, long long v) { return __nvvm_atom_sys_xor_gen_ll(p, v); }
__device_forceinline__ float __log10f(float a) { return __nv_fast_log10f(a); }
__device_forceinline__ float __log2f(float a) { return __nv_fast_log2f(a); }
__device_forceinline__ float __logf(float a) { return __nv_fast_logf(a); }
__device_forceinline__ double __longlong_as_double(long long a) { return __nv_longlong_as_double(a); }
__device_forceinline__ int __mul24(int a, int b) { return __nv_mul24(a, b); }
__device_forceinline__ long long __mul64hi(long long a, long long b) { return __nv_mul64hi(a, b); }
__device_forceinline__ int __mulhi(int a, int b) { return __nv_mulhi(a, b); }
__device_forceinline__ unsigned int __pm0(void) { return __nvvm_read_ptx_sreg_pm0(); }
__device_forceinline__ unsigned int __pm1(void) { return __nvvm_read_ptx_sreg_pm1(); }
__device_forceinline__ unsigned int __pm2(void) { return __nvvm_read_ptx_sreg_pm2(); }
__device_forceinline__ unsigned int __pm3(void) { return __nvvm_read_ptx_sreg_pm3(); }
__device_forceinline__ int __popc(int a) { return __nv_popc(a); }
__device_forceinline__ int __popcll(long long a) { return __nv_popcll(a); }
__device_forceinline__ float __powf(float a, float b) { return __nv_fast_powf(a, b); }
#define __prof_trigger(__counter) __asm__ __volatile__("pmevent \t%0;" ::"i"(__counter))
__device_forceinline__ int __rhadd(int a, int b) { return __nv_rhadd(a, b); }
__device_forceinline__ unsigned int __sad(int a, int b, unsigned int c) { return __nv_sad(a, b, c); }
__device_forceinline__ float __saturatef(float a) { return __nv_saturatef(a); }
__device_forceinline__ int __signbitd(double a) { return __nv_signbitd(a); }
__device_forceinline__ int __signbitf(float a) { return __nv_signbitf(a); }
__device_forceinline__ void __sincosf(float a, float *s, float *c) { return __nv_fast_sincosf(a, s, c); }
__device_forceinline__ float __sinf(float a) { return __nv_fast_sinf(a); }
__device_forceinline__ int __syncthreads_and(int a) { return __nvvm_bar0_and(a); }
__device_forceinline__ int __syncthreads_count(int a) { return __nvvm_bar0_popc(a); }
__device_forceinline__ int __syncthreads_or(int a) { return __nvvm_bar0_or(a); }
__device_forceinline__ float __tanf(float a) { return __nv_fast_tanf(a); }
__device_forceinline__ void __threadfence(void) { __nvvm_membar_gl(); }
__device_forceinline__ void __threadfence_block(void) { __nvvm_membar_cta(); };
__device_forceinline__ void __threadfence_system(void) { __nvvm_membar_sys(); };
__device_forceinline__ void __trap(void) { __asm__ __volatile__("trap;"); }
__device_forceinline__ unsigned int __uAtomicAdd(unsigned int *p, unsigned int v) { return __nvvm_atom_add_gen_i((int *)p, v); }
__device_forceinline__ unsigned int __uAtomicAdd_block(unsigned int *p, unsigned int v) { return __nvvm_atom_cta_add_gen_i((int *)p, v); }
__device_forceinline__ unsigned int __uAtomicAdd_system(unsigned int *p, unsigned int v) { return __nvvm_atom_sys_add_gen_i((int *)p, v); }
__device_forceinline__ unsigned int __uAtomicAnd(unsigned int *p, unsigned int v) { return __nvvm_atom_and_gen_i((int *)p, v); }
__device_forceinline__ unsigned int __uAtomicAnd_block(unsigned int *p, unsigned int v) { return __nvvm_atom_cta_and_gen_i((int *)p, v); }
__device_forceinline__ unsigned int __uAtomicAnd_system(unsigned int *p, unsigned int v) { return __nvvm_atom_sys_and_gen_i((int *)p, v); }
__device_forceinline__ unsigned int __uAtomicCAS(unsigned int *p, unsigned int cmp, unsigned int v) { return __nvvm_atom_cas_gen_i((int *)p, cmp, v); }
__device_forceinline__ unsigned int __uAtomicCAS_block(unsigned int *p, unsigned int cmp, unsigned int v) { return __nvvm_atom_cta_cas_gen_i((int *)p, cmp, v); }
__device_forceinline__ unsigned int __uAtomicCAS_system(unsigned int *p, unsigned int cmp, unsigned int v) { return __nvvm_atom_sys_cas_gen_i((int *)p, cmp, v); }
__device_forceinline__ unsigned int __uAtomicDec(unsigned int *p, unsigned int v) { return __nvvm_atom_dec_gen_ui(p, v); }
__device_forceinline__ unsigned int __uAtomicDec_block(unsigned int *p, unsigned int v) { return __nvvm_atom_cta_dec_gen_ui(p, v); }
__device_forceinline__ unsigned int __uAtomicDec_system(unsigned int *p, unsigned int v) { return __nvvm_atom_sys_dec_gen_ui(p, v); }
__device_forceinline__ unsigned int __uAtomicExch(unsigned int *p, unsigned int v) { return __nvvm_atom_xchg_gen_i((int *)p, v); }
__device_forceinline__ unsigned int __uAtomicExch_block(unsigned int *p, unsigned int v) { return __nvvm_atom_cta_xchg_gen_i((int *)p, v); }
__device_forceinline__ unsigned int __uAtomicExch_system(unsigned int *p, unsigned int v) { return __nvvm_atom_sys_xchg_gen_i((int *)p, v); }
__device_forceinline__ unsigned int __uAtomicInc(unsigned int *p, unsigned int v) { return __nvvm_atom_inc_gen_ui(p, v); }
__device_forceinline__ unsigned int __uAtomicInc_block(unsigned int *p, unsigned int v) { return __nvvm_atom_cta_inc_gen_ui(p, v); }
__device_forceinline__ unsigned int __uAtomicInc_system(unsigned int *p, unsigned int v) { return __nvvm_atom_sys_inc_gen_ui(p, v); }
__device_forceinline__ unsigned int __uAtomicMax(unsigned int *p, unsigned int v) { return __nvvm_atom_max_gen_ui(p, v); }
__device_forceinline__ unsigned int __uAtomicMax_block(unsigned int *p, unsigned int v) { return __nvvm_atom_cta_max_gen_ui(p, v); }
__device_forceinline__ unsigned int __uAtomicMax_system(unsigned int *p, unsigned int v) { return __nvvm_atom_sys_max_gen_ui(p, v); }
__device_forceinline__ unsigned int __uAtomicMin(unsigned int *p, unsigned int v) { return __nvvm_atom_min_gen_ui(p, v); }
__device_forceinline__ unsigned int __uAtomicMin_block(unsigned int *p, unsigned int v) { return __nvvm_atom_cta_min_gen_ui(p, v); }
__device_forceinline__ unsigned int __uAtomicMin_system(unsigned int *p, unsigned int v) { return __nvvm_atom_sys_min_gen_ui(p, v); }
__device_forceinline__ unsigned int __uAtomicOr(unsigned int *p, unsigned int v) { return __nvvm_atom_or_gen_i((int *)p, v); }
__device_forceinline__ unsigned int __uAtomicOr_block(unsigned int *p, unsigned int v) { return __nvvm_atom_cta_or_gen_i((int *)p, v); }
__device_forceinline__ unsigned int __uAtomicOr_system(unsigned int *p, unsigned int v) { return __nvvm_atom_sys_or_gen_i((int *)p, v); }
__device_forceinline__ unsigned int __uAtomicXor(unsigned int *p, unsigned int v) { return __nvvm_atom_xor_gen_i((int *)p, v); }
__device_forceinline__ unsigned int __uAtomicXor_block(unsigned int *p, unsigned int v) { return __nvvm_atom_cta_xor_gen_i((int *)p, v); }
__device_forceinline__ unsigned int __uAtomicXor_system(unsigned int *p, unsigned int v) { return __nvvm_atom_sys_xor_gen_i((int *)p, v); }
__device_forceinline__ unsigned int __uhadd(unsigned int a, unsigned int b) { return __nv_uhadd(a, b); }
__device_forceinline__ double __uint2double_rn(unsigned int a) { return __nv_uint2double_rn(a); }
__device_forceinline__ float __uint2float_rd(unsigned int a) { return __nv_uint2float_rd(a); }
__device_forceinline__ float __uint2float_rn(unsigned int a) { return __nv_uint2float_rn(a); }
__device_forceinline__ float __uint2float_ru(unsigned int a) { return __nv_uint2float_ru(a); }
__device_forceinline__ float __uint2float_rz(unsigned int a) { return __nv_uint2float_rz(a); }
__device_forceinline__ float __uint_as_float(unsigned int a) { return __nv_uint_as_float(a); }
__device_forceinline__ double __ull2double_rd(unsigned long long a) { return __nv_ull2double_rd(a); }
__device_forceinline__ double __ull2double_rn(unsigned long long a) { return __nv_ull2double_rn(a); }
__device_forceinline__ double __ull2double_ru(unsigned long long a) { return __nv_ull2double_ru(a); }
__device_forceinline__ double __ull2double_rz(unsigned long long a) { return __nv_ull2double_rz(a); }
__device_forceinline__ float __ull2float_rd(unsigned long long a) { return __nv_ull2float_rd(a); }
__device_forceinline__ float __ull2float_rn(unsigned long long a) { return __nv_ull2float_rn(a); }
__device_forceinline__ float __ull2float_ru(unsigned long long a) { return __nv_ull2float_ru(a); }
__device_forceinline__ float __ull2float_rz(unsigned long long a) { return __nv_ull2float_rz(a); }
__device_forceinline__ unsigned long long __ullAtomicAdd(unsigned long long *p, unsigned long long v) { return __nvvm_atom_add_gen_ll((long long *)p, v); }
__device_forceinline__ unsigned long long __ullAtomicAdd_block(unsigned long long *p, unsigned long long v) { return __nvvm_atom_cta_add_gen_ll((long long *)p, v); }
__device_forceinline__ unsigned long long __ullAtomicAdd_system(unsigned long long *p, unsigned long long v) { return __nvvm_atom_sys_add_gen_ll((long long *)p, v); }
__device_forceinline__ unsigned long long __ullAtomicAnd(unsigned long long *p, unsigned long long v) { return __nvvm_atom_and_gen_ll((long long *)p, v); }
__device_forceinline__ unsigned long long __ullAtomicAnd_block(unsigned long long *p, unsigned long long v) { return __nvvm_atom_cta_and_gen_ll((long long *)p, v); }
__device_forceinline__ unsigned long long __ullAtomicAnd_system(unsigned long long *p, unsigned long long v) { return __nvvm_atom_sys_and_gen_ll((long long *)p, v); }
__device_forceinline__ unsigned long long __ullAtomicCAS(unsigned long long *p, unsigned long long cmp, unsigned long long v) { return __nvvm_atom_cas_gen_ll((long long *)p, cmp, v); }
__device_forceinline__ unsigned long long __ullAtomicCAS_block(unsigned long long *p, unsigned long long cmp, unsigned long long v) { return __nvvm_atom_cta_cas_gen_ll((long long *)p, cmp, v); }
__device_forceinline__ unsigned long long __ullAtomicCAS_system(unsigned long long *p, unsigned long long cmp, unsigned long long v) { return __nvvm_atom_sys_cas_gen_ll((long long *)p, cmp, v); }
__device_forceinline__ unsigned long long __ullAtomicExch(unsigned long long *p, unsigned long long v) { return __nvvm_atom_xchg_gen_ll((long long *)p, v); }
__device_forceinline__ unsigned long long __ullAtomicExch_block(unsigned long long *p, unsigned long long v) { return __nvvm_atom_cta_xchg_gen_ll((long long *)p, v); }
__device_forceinline__ unsigned long long __ullAtomicExch_system(unsigned long long *p, unsigned long long v) { return __nvvm_atom_sys_xchg_gen_ll((long long *)p, v); }
__device_forceinline__ unsigned long long __ullAtomicMax(unsigned long long *p, unsigned long long v) { return __nvvm_atom_max_gen_ull(p, v); }
__device_forceinline__ unsigned long long __ullAtomicMax_block(unsigned long long *p, unsigned long long v) { return __nvvm_atom_cta_max_gen_ull(p, v); }
__device_forceinline__ unsigned long long __ullAtomicMax_system(unsigned long long *p, unsigned long long v) { return __nvvm_atom_sys_max_gen_ull(p, v); }
__device_forceinline__ unsigned long long __ullAtomicMin(unsigned long long *p, unsigned long long v) { return __nvvm_atom_min_gen_ull(p, v); }
__device_forceinline__ unsigned long long __ullAtomicMin_block(unsigned long long *p, unsigned long long v) { return __nvvm_atom_cta_min_gen_ull(p, v); }
__device_forceinline__ unsigned long long __ullAtomicMin_system(unsigned long long *p, unsigned long long v) { return __nvvm_atom_sys_min_gen_ull(p, v); }
__device_forceinline__ unsigned long long __ullAtomicOr(unsigned long long *p, unsigned long long v) { return __nvvm_atom_or_gen_ll((long long *)p, v); }
__device_forceinline__ unsigned long long __ullAtomicOr_block(unsigned long long *p, unsigned long long v) { return __nvvm_atom_cta_or_gen_ll((long long *)p, v); }
__device_forceinline__ unsigned long long __ullAtomicOr_system(unsigned long long *p, unsigned long long v) { return __nvvm_atom_sys_or_gen_ll((long long *)p, v); }
__device_forceinline__ unsigned long long __ullAtomicXor(unsigned long long *p, unsigned long long v) { return __nvvm_atom_xor_gen_ll((long long *)p, v); }
__device_forceinline__ unsigned long long __ullAtomicXor_block(unsigned long long *p, unsigned long long v) { return __nvvm_atom_cta_xor_gen_ll((long long *)p, v); }
__device_forceinline__ unsigned long long __ullAtomicXor_system(unsigned long long *p, unsigned long long v) { return __nvvm_atom_sys_xor_gen_ll((long long *)p, v); }
__device_forceinline__ unsigned int __umul24(unsigned int a, unsigned int b) { return __nv_umul24(a, b); }
__device_forceinline__ unsigned long long __umul64hi(unsigned long long a, unsigned long long b) { return __nv_umul64hi(a, b); }
__device_forceinline__ unsigned int __umulhi(unsigned int a, unsigned int b) { return __nv_umulhi(a, b); }
__device_forceinline__ unsigned int __urhadd(unsigned int a, unsigned int b) { return __nv_urhadd(a, b); }
__device_forceinline__ unsigned int __usad(unsigned int a, unsigned int b, unsigned int c) { return __nv_usad(a, b, c); }

__device_forceinline__ void *memcpy(void *a, const void *b, size_t c) { return __builtin_memcpy(a, b, c); }
__device_forceinline__ void *memset(void *a, int b, size_t c) { return __builtin_memset(a, b, c); }

#if defined(__FAST_MATH__)
#define __FAST_OR_SLOW(fast, slow) fast
#else
#define __FAST_OR_SLOW(fast, slow) slow
#endif

__device_forceinline__ int abs(int a) { return __nv_abs(a); }
__device_forceinline__ double fabs(double a) { return __nv_fabs(a); }
__device_forceinline__ double acos(double a) { return __nv_acos(a); }
__device_forceinline__ float acosf(float a) { return __nv_acosf(a); }
__device_forceinline__ double acosh(double a) { return __nv_acosh(a); }
__device_forceinline__ float acoshf(float a) { return __nv_acoshf(a); }
__device_forceinline__ double asin(double a) { return __nv_asin(a); }
__device_forceinline__ float asinf(float a) { return __nv_asinf(a); }
__device_forceinline__ double asinh(double a) { return __nv_asinh(a); }
__device_forceinline__ float asinhf(float a) { return __nv_asinhf(a); }
__device_forceinline__ double atan(double a) { return __nv_atan(a); }
__device_forceinline__ double atan2(double a, double b) { return __nv_atan2(a, b); }
__device_forceinline__ float atan2f(float a, float b) { return __nv_atan2f(a, b); }
__device_forceinline__ float atanf(float a) { return __nv_atanf(a); }
__device_forceinline__ double atanh(double a) { return __nv_atanh(a); }
__device_forceinline__ float atanhf(float a) { return __nv_atanhf(a); }
__device_forceinline__ double cbrt(double a) { return __nv_cbrt(a); }
__device_forceinline__ float cbrtf(float a) { return __nv_cbrtf(a); }
__device_forceinline__ double ceil(double a) { return __nv_ceil(a); }
__device_forceinline__ float ceilf(float a) { return __nv_ceilf(a); }
__device_forceinline__ double copysign(double a, double b) { return __nv_copysign(a, b); }
__device_forceinline__ float copysignf(float a, float b) { return __nv_copysignf(a, b); }
__device_forceinline__ double cos(double a) { return __nv_cos(a); }
__device_forceinline__ float cosf(float a) { return __FAST_OR_SLOW(__nv_fast_cosf, __nv_cosf)(a); }
__device_forceinline__ double cosh(double a) { return __nv_cosh(a); }
__device_forceinline__ float coshf(float a) { return __nv_coshf(a); }
__device_forceinline__ double cospi(double a) { return __nv_cospi(a); }
__device_forceinline__ float cospif(float a) { return __nv_cospif(a); }
__device_forceinline__ double cyl_bessel_i0(double a) { return __nv_cyl_bessel_i0(a); }
__device_forceinline__ float cyl_bessel_i0f(float a) { return __nv_cyl_bessel_i0f(a); }
__device_forceinline__ double cyl_bessel_i1(double a) { return __nv_cyl_bessel_i1(a); }
__device_forceinline__ float cyl_bessel_i1f(float a) { return __nv_cyl_bessel_i1f(a); }
__device_forceinline__ double erf(double a) { return __nv_erf(a); }
__device_forceinline__ double erfc(double a) { return __nv_erfc(a); }
__device_forceinline__ float erfcf(float a) { return __nv_erfcf(a); }
__device_forceinline__ double erfcinv(double a) { return __nv_erfcinv(a); }
__device_forceinline__ float erfcinvf(float a) { return __nv_erfcinvf(a); }
__device_forceinline__ double erfcx(double a) { return __nv_erfcx(a); }
__device_forceinline__ float erfcxf(float a) { return __nv_erfcxf(a); }
__device_forceinline__ float erff(float a) { return __nv_erff(a); }
__device_forceinline__ double erfinv(double a) { return __nv_erfinv(a); }
__device_forceinline__ float erfinvf(float a) { return __nv_erfinvf(a); }
__device_forceinline__ double exp(double a) { return __nv_exp(a); }
__device_forceinline__ double exp10(double a) { return __nv_exp10(a); }
__device_forceinline__ float exp10f(float a) { return __nv_exp10f(a); }
__device_forceinline__ double exp2(double a) { return __nv_exp2(a); }
__device_forceinline__ float exp2f(float a) { return __nv_exp2f(a); }
__device_forceinline__ float expf(float a) { return __nv_expf(a); }
__device_forceinline__ double expm1(double a) { return __nv_expm1(a); }
__device_forceinline__ float expm1f(float a) { return __nv_expm1f(a); }
__device_forceinline__ float fabsf(float a) { return __nv_fabsf(a); }
__device_forceinline__ double fdim(double a, double b) { return __nv_fdim(a, b); }
__device_forceinline__ float fdimf(float a, float b) { return __nv_fdimf(a, b); }
__device_forceinline__ double fdivide(double a, double b) { return a / b; }
__device_forceinline__ float fdividef(float a, float b) { return __FAST_OR_SLOW(__nv_fast_fdividef(a, b), a / b); }
__device_forceinline__ double floor(double f) { return __nv_floor(f); }
__device_forceinline__ float floorf(float f) { return __nv_floorf(f); }
__device_forceinline__ double fma(double a, double b, double c) { return __nv_fma(a, b, c); }
__device_forceinline__ float fmaf(float a, float b, float c) { return __nv_fmaf(a, b, c); }
__device_forceinline__ double fmax(double a, double b) { return __nv_fmax(a, b); }
__device_forceinline__ float fmaxf(float a, float b) { return __nv_fmaxf(a, b); }
__device_forceinline__ double fmin(double a, double b) { return __nv_fmin(a, b); }
__device_forceinline__ float fminf(float a, float b) { return __nv_fminf(a, b); }
__device_forceinline__ double fmod(double a, double b) { return __nv_fmod(a, b); }
__device_forceinline__ float fmodf(float a, float b) { return __nv_fmodf(a, b); }
__device_forceinline__ double frexp(double a, int *b) { return __nv_frexp(a, b); }
__device_forceinline__ float frexpf(float a, int *b) { return __nv_frexpf(a, b); }
__device_forceinline__ double hypot(double a, double b) { return __nv_hypot(a, b); }
__device_forceinline__ float hypotf(float a, float b) { return __nv_hypotf(a, b); }
__device_forceinline__ int ilogb(double a) { return __nv_ilogb(a); }
__device_forceinline__ int ilogbf(float a) { return __nv_ilogbf(a); }
__device_forceinline__ double j0(double a) { return __nv_j0(a); }
__device_forceinline__ float j0f(float a) { return __nv_j0f(a); }
__device_forceinline__ double j1(double a) { return __nv_j1(a); }
__device_forceinline__ float j1f(float a) { return __nv_j1f(a); }
__device_forceinline__ double jn(int n, double a) { return __nv_jn(n, a); }
__device_forceinline__ float jnf(int n, float a) { return __nv_jnf(n, a); }
#if defined(__LP64__)
__device_forceinline__ long labs(long a) { return __nv_llabs(a); };
#else
__device_forceinline__ long labs(long a) { return __nv_abs(a); };
#endif
__device_forceinline__ double ldexp(double a, int b) { return __nv_ldexp(a, b); }
__device_forceinline__ float ldexpf(float a, int b) { return __nv_ldexpf(a, b); }
__device_forceinline__ double lgamma(double a) { return __nv_lgamma(a); }
__device_forceinline__ float lgammaf(float a) { return __nv_lgammaf(a); }
__device_forceinline__ long long llabs(long long a) { return __nv_llabs(a); }
__device_forceinline__ long long llmax(long long a, long long b) { return __nv_llmax(a, b); }
__device_forceinline__ long long llmin(long long a, long long b) { return __nv_llmin(a, b); }
__device_forceinline__ long long llrint(double a) { return __nv_llrint(a); }
__device_forceinline__ long long llrintf(float a) { return __nv_llrintf(a); }
__device_forceinline__ long long llround(double a) { return __nv_llround(a); }
__device_forceinline__ long long llroundf(float a) { return __nv_llroundf(a); }
__device_forceinline__ double round(double a) { return __nv_round(a); }
__device_forceinline__ float roundf(float a) { return __nv_roundf(a); }
__device_forceinline__ double log(double a) { return __nv_log(a); }
__device_forceinline__ double log10(double a) { return __nv_log10(a); }
__device_forceinline__ float log10f(float a) { return __nv_log10f(a); }
__device_forceinline__ double log1p(double a) { return __nv_log1p(a); }
__device_forceinline__ float log1pf(float a) { return __nv_log1pf(a); }
__device_forceinline__ double log2(double a) { return __nv_log2(a); }
__device_forceinline__ float log2f(float a) { return __FAST_OR_SLOW(__nv_fast_log2f, __nv_log2f)(a); }
__device_forceinline__ double logb(double a) { return __nv_logb(a); }
__device_forceinline__ float logbf(float a) { return __nv_logbf(a); }
__device_forceinline__ float logf(float a) { return __FAST_OR_SLOW(__nv_fast_logf, __nv_logf)(a); }
#if defined(__LP64__)
__device_forceinline__ long lrint(double a) { return llrint(a); }
__device_forceinline__ long lrintf(float a) { return __float2ll_rn(a); }
__device_forceinline__ long lround(double a) { return llround(a); }
__device_forceinline__ long lroundf(float a) { return llroundf(a); }
#else
__device_forceinline__ long lrint(double a) { return (long)rint(a); }
__device_forceinline__ long lrintf(float a) { return __float2int_rn(a); }
__device_forceinline__ long lround(double a) { return round(a); }
__device_forceinline__ long lroundf(float a) { return roundf(a); }
#endif
__device_forceinline__ int max(int a, int b) { return __nv_max(a, b); }
__device_forceinline__ int min(int a, int b) { return __nv_min(a, b); }
__device_forceinline__ double modf(double a, double *b) { return __nv_modf(a, b); }
__device_forceinline__ float modff(float a, float *b) { return __nv_modff(a, b); }
__device_forceinline__ double nearbyint(double a) { return __builtin_nearbyint(a); }
__device_forceinline__ float nearbyintf(float a) { return __builtin_nearbyintf(a); }
__device_forceinline__ double nextafter(double a, double b) { return __nv_nextafter(a, b); }
__device_forceinline__ float nextafterf(float a, float b) { return __nv_nextafterf(a, b); }
__device_forceinline__ double norm(int dim, const double *t) { return __nv_norm(dim, t); }
__device_forceinline__ double norm3d(double a, double b, double c) { return __nv_norm3d(a, b, c); }
__device_forceinline__ float norm3df(float a, float b, float c) { return __nv_norm3df(a, b, c); }
__device_forceinline__ double norm4d(double a, double b, double c, double d) { return __nv_norm4d(a, b, c, d); }
__device_forceinline__ float norm4df(float a, float b, float c, float d) { return __nv_norm4df(a, b, c, d); }
__device_forceinline__ double normcdf(double a) { return __nv_normcdf(a); }
__device_forceinline__ float normcdff(float a) { return __nv_normcdff(a); }
__device_forceinline__ double normcdfinv(double a) { return __nv_normcdfinv(a); }
__device_forceinline__ float normcdfinvf(float a) { return __nv_normcdfinvf(a); }
__device_forceinline__ float normf(int dim, const float *t) { return __nv_normf(dim, t); }
__device_forceinline__ double pow(double a, double b) { return __nv_pow(a, b); }
__device_forceinline__ float powf(float a, float b) { return __nv_powf(a, b); }
__device_forceinline__ double powi(double a, int b) { return __nv_powi(a, b); }
__device_forceinline__ float powif(float a, int b) { return __nv_powif(a, b); }
__device_forceinline__ double rcbrt(double a) { return __nv_rcbrt(a); }
__device_forceinline__ float rcbrtf(float a) { return __nv_rcbrtf(a); }
__device_forceinline__ double remainder(double a, double b) { return __nv_remainder(a, b); }
__device_forceinline__ float remainderf(float a, float b) { return __nv_remainderf(a, b); }
__device_forceinline__ double remquo(double a, double b, int *c) { return __nv_remquo(a, b, c); }
__device_forceinline__ float remquof(float a, float b, int *c) { return __nv_remquof(a, b, c); }
__device_forceinline__ double rhypot(double a, double b) { return __nv_rhypot(a, b); }
__device_forceinline__ float rhypotf(float a, float b) { return __nv_rhypotf(a, b); }
__device_forceinline__ double rint(double a) { return __builtin_rint(a); }
__device_forceinline__ float rintf(float a) { return __builtin_rintf(a); }
__device_forceinline__ double rnorm(int a, const double *b) { return __nv_rnorm(a, b); }
__device_forceinline__ double rnorm3d(double a, double b, double c) { return __nv_rnorm3d(a, b, c); }
__device_forceinline__ float rnorm3df(float a, float b, float c) { return __nv_rnorm3df(a, b, c); }
__device_forceinline__ double rnorm4d(double a, double b, double c, double d) { return __nv_rnorm4d(a, b, c, d); }
__device_forceinline__ float rnorm4df(float a, float b, float c, float d) { return __nv_rnorm4df(a, b, c, d); }
__device_forceinline__ float rnormf(int dim, const float *t) { return __nv_rnormf(dim, t); }
__device_forceinline__ double rsqrt(double a) { return __nv_rsqrt(a); }
__device_forceinline__ float rsqrtf(float a) { return __nv_rsqrtf(a); }
__device_forceinline__ double scalbn(double a, int b) { return __nv_scalbn(a, b); }
__device_forceinline__ float scalbnf(float a, int b) { return __nv_scalbnf(a, b); }
__device_forceinline__ double scalbln(double a, long b) {
    if (b > INT_MAX) { return a > 0 ? INFINITY : -HUGE_VAL; }
    if (b < INT_MIN) { return a > 0 ? 0.0 : -0.0; }
    return scalbn(a, (int)b);
}
__device_forceinline__ float scalblnf(float a, long b) {
    if (b > INT_MAX) { return a > 0 ? HUGE_VALF : -HUGE_VALF; }
    if (b < INT_MIN) { return a > 0 ? 0.f : -0.f; }
    return scalbnf(a, (int)b);
}
__device_forceinline__ double sin(double a) { return __nv_sin(a); }
__device_forceinline__ void sincos(double a, double *s, double *c) { return __nv_sincos(a, s, c); }
__device_forceinline__ void sincosf(float a, float *s, float *c) { return __FAST_OR_SLOW(__nv_fast_sincosf, __nv_sincosf)(a, s, c); }
__device_forceinline__ void sincospi(double a, double *s, double *c) { return __nv_sincospi(a, s, c); }
__device_forceinline__ void sincospif(float a, float *s, float *c) { return __nv_sincospif(a, s, c); }
__device_forceinline__ float sinf(float a) { return __FAST_OR_SLOW(__nv_fast_sinf, __nv_sinf)(a); }
__device_forceinline__ double sinh(double a) { return __nv_sinh(a); }
__device_forceinline__ float sinhf(float a) { return __nv_sinhf(a); }
__device_forceinline__ double sinpi(double a) { return __nv_sinpi(a); }
__device_forceinline__ float sinpif(float a) { return __nv_sinpif(a); }
__device_forceinline__ double sqrt(double a) { return __nv_sqrt(a); }
__device_forceinline__ float sqrtf(float a) { return __nv_sqrtf(a); }
__device_forceinline__ double tan(double a) { return __nv_tan(a); }
__device_forceinline__ float tanf(float a) { return __nv_tanf(a); }
__device_forceinline__ double tanh(double a) { return __nv_tanh(a); }
__device_forceinline__ float tanhf(float a) { return __nv_tanhf(a); }
__device_forceinline__ double tgamma(double a) { return __nv_tgamma(a); }
__device_forceinline__ float tgammaf(float a) { return __nv_tgammaf(a); }
__device_forceinline__ double trunc(double a) { return __nv_trunc(a); }
__device_forceinline__ float truncf(float a) { return __nv_truncf(a); }
__device_forceinline__ unsigned long long ullmax(unsigned long long a, unsigned long long b) { return __nv_ullmax(a, b); }
__device_forceinline__ unsigned long long ullmin(unsigned long long a, unsigned long long b) { return __nv_ullmin(a, b); }
__device_forceinline__ unsigned int umax(unsigned int a, unsigned int b) { return __nv_umax(a, b); }
__device_forceinline__ unsigned int umin(unsigned int a, unsigned int b) { return __nv_umin(a, b); }
__device_forceinline__ double y0(double a) { return __nv_y0(a); }
__device_forceinline__ float y0f(float a) { return __nv_y0f(a); }
__device_forceinline__ double y1(double a) { return __nv_y1(a); }
__device_forceinline__ float y1f(float a) { return __nv_y1f(a); }
__device_forceinline__ double yn(int a, double b) { return __nv_yn(a, b); }
__device_forceinline__ float ynf(int a, float b) { return __nv_ynf(a, b); }

#undef __FAST_OR_SLOW

// Implementation of a subset of <cuda_fp16.h> functionality
struct __half
{
    unsigned short u;
};

__device_forceinline__ short __half_as_short(const __half h)
{
    short i;
    asm("{  mov.b16 %0, %1;}\n" : "=h"(i) : "h"(h));
    return i;
}

__device_forceinline__ __half __short_as_half(const short i)
{
    __half h;
    asm("{  mov.b16 %0, %1;}\n" : "=h"(h) : "h"(i));
    return h;
}

__device_forceinline__ __half __hadd(const __half a, const __half b)
{
    __half x;
    asm("{  add.f16 %0, %1, %2;}\n" : "=h"(x) : "h"(a), "h"(b));
    return x;
}

// Implementation of a subset of <cuda_runtime.h> functionality
__device_forceinline__ bool isfinite(double x) { return __nv_isfinited(x); }
__device_forceinline__ bool isfinite(float x) { return __nv_finitef(x); }

__device_forceinline__ unsigned short atomicCAS(unsigned short *address, unsigned short compare, unsigned short val)
{
    unsigned short r;

    asm volatile ("{atom.global.cas.b16 %0,[%1],%2,%3; }\n"
                    : "=h"(r)
                    : "l"(address), "h"(compare), "h"(val)
                    : "memory");

    return r;
}

__device_forceinline__ int atomicCAS(int *address, int compare, int val)
{
    return __iAtomicCAS(address, compare, val);
}

__device_forceinline__ __half atomicAdd(__half *const address, const __half val)
{
    unsigned short* address_as_us = (unsigned short*)address;
    unsigned short old = *address_as_us;
    unsigned short assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address_as_us, assumed, __half_as_short(__hadd(val, __short_as_half(assumed))));
    }
    while (assumed != old);

    return __short_as_half(old);
}

__device_forceinline__ double atomicAdd(double *const address, const double val)
{
    return __dAtomicAdd(address, val);
}

__device_forceinline__ float atomicAdd(float *const address, const float val)
{
    return __fAtomicAdd(address, val);
}

__device_forceinline__ int atomicAdd(int *const address, const int val)
{
    return __iAtomicAdd(address, val);
}

__device_forceinline__ unsigned int atomicAdd(unsigned int *const address, const unsigned int val)
{
    return __uAtomicAdd(address, val);
}

__device_forceinline__ unsigned int atomicAdd(unsigned long long *const address, const unsigned long long val)
{
    return __ullAtomicAdd(address, val);
}

__device_forceinline__ int atomicMin(int *const address, const int val)
{
    return __iAtomicMin(address, val);
}

__device_forceinline__ int atomicMax(int *const address, const int val)
{
    return __iAtomicMax(address, val);
}
