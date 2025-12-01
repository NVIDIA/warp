/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// This file declares a subset of the C runtime (CRT) functions and macros for
// use by compute kernel modules. There are three environments in which this
// file gets included:
// - CUDA kernel modules (WP_NO_CRT and __CUDACC__). CUDA already has implicitly
//   declared builtins for most functions. printf() and macro definitions are
//   the notable exceptions.
// - C++ kernel modules (WP_NO_CRT and !__CUDACC__). These can't use the CRT
//   directly when using a standalone compiler. The functions get obtained from
//   the compiler library instead (clang.dll).
// - Warp runtime (!WP_NO_CRT). When building warp.dll it's fine to include the
//   standard C library headers, and it avoids mismatched redefinitions.

#if !defined(__CUDA_ARCH__)
#if defined(_WIN32)
#define WP_API __declspec(dllexport)
#else
#define WP_API __attribute__ ((visibility ("default")))
#endif
#else
#define WP_API
#endif

#if !defined(__CUDA_ARCH__)

// Helper for implementing assert() macro
extern "C" WP_API void _wp_assert(const char* message, const char* file, unsigned int line);

// Helper for implementing isfinite()
extern "C" WP_API int _wp_isfinite(double);

// Helper for implementing isnan()
extern "C" WP_API int _wp_isnan(double);

// Helper for implementing isinf()
extern "C" WP_API int _wp_isinf(double);

#endif  // !__CUDA_ARCH__

#if !defined(WP_NO_CRT)

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#else

// These definitions are taken from Jitify: https://github.com/NVIDIA/jitify

/// float.h
#define FLT_RADIX       2
#define FLT_MANT_DIG    24
#define DBL_MANT_DIG    53
#define FLT_DIG         6
#define DBL_DIG         15
#define FLT_MIN_EXP     -125
#define DBL_MIN_EXP     -1021
#define FLT_MIN_10_EXP  -37
#define DBL_MIN_10_EXP  -307
#define FLT_MAX_EXP     128
#define DBL_MAX_EXP     1024
#define FLT_MAX_10_EXP  38
#define DBL_MAX_10_EXP  308
#define FLT_MAX         3.4028234e38f
#define DBL_MAX         1.7976931348623157e308
#define FLT_EPSILON     1.19209289e-7f
#define DBL_EPSILON     2.220440492503130e-16
#define FLT_MIN         1.1754943e-38f
#define DBL_MIN         2.2250738585072013e-308
#define FLT_ROUNDS      1
#if defined __cplusplus && __cplusplus >= 201103L
#define FLT_EVAL_METHOD 0
#define DECIMAL_DIG     21
#endif

/// limits.h
#if defined _WIN32 || defined _WIN64
#define __WORDSIZE 32
#else
#if defined __x86_64__ && !defined __ILP32__
#define __WORDSIZE 64
#else
#define __WORDSIZE 32
#endif
#endif
#define MB_LEN_MAX  16
#define CHAR_BIT    8
#define SCHAR_MIN   (-128)
#define SCHAR_MAX   127
#define UCHAR_MAX   255
#define _JITIFY_CHAR_IS_UNSIGNED ((char)-1 >= 0)
#define CHAR_MIN (_JITIFY_CHAR_IS_UNSIGNED ? 0 : SCHAR_MIN)
#define CHAR_MAX (_JITIFY_CHAR_IS_UNSIGNED ? UCHAR_MAX : SCHAR_MAX)
#define SHRT_MIN    (-32768)
#define SHRT_MAX    32767
#define USHRT_MAX   65535
#define INT_MIN     (-INT_MAX - 1)
#define INT_MAX     2147483647
#define UINT_MAX    4294967295U
#if __WORDSIZE == 64
#define LONG_MAX  9223372036854775807L
#else
#define LONG_MAX  2147483647L
#endif
#define LONG_MIN    (-LONG_MAX - 1L)
#if __WORDSIZE == 64
#define ULONG_MAX  18446744073709551615UL
#else
#define ULONG_MAX  4294967295UL
#endif
#define LLONG_MAX  9223372036854775807LL
#define LLONG_MIN  (-LLONG_MAX - 1LL)
#define ULLONG_MAX 18446744073709551615ULL

#define INFINITY   ((float)(DBL_MAX * DBL_MAX))
#define HUGE_VAL   ((double)INFINITY)
#define HUGE_VALF  ((float)INFINITY)
#define NAN        ((float)(0.0 / 0.0))

/// stdint.h
typedef signed char int8_t;
typedef signed short int16_t;
typedef signed int int32_t;
typedef signed long long int64_t;
// typedef signed char      int_fast8_t;
// typedef signed short     int_fast16_t;
// typedef signed int       int_fast32_t;
// typedef signed long long int_fast64_t;
// typedef signed char      int_least8_t;
// typedef signed short     int_least16_t;
// typedef signed int       int_least32_t;
// typedef signed long long int_least64_t;
// typedef signed long long intmax_t;
// typedef signed long      intptr_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
// typedef unsigned char      uint_fast8_t;
// typedef unsigned short     uint_fast16_t;
// typedef unsigned int       uint_fast32_t;
// typedef unsigned long long uint_fast64_t;
// typedef unsigned char      uint_least8_t;
// typedef unsigned short     uint_least16_t;
// typedef unsigned int       uint_least32_t;
// typedef unsigned long long uint_least64_t;
// typedef unsigned long long uintmax_t;


/// math.h

// #if __cplusplus >= 201103L
// #define DEFINE_MATH_UNARY_FUNC_WRAPPER(f) \
// 	inline double      f(double x)         { return ::f(x); } \
// 	inline float       f##f(float x)       { return ::f(x); } \
// 	/*inline long double f##l(long double x) { return ::f(x); }*/ \
// 	inline float       f(float x)          { return ::f(x); } \
// 	/*inline long double f(long double x)    { return ::f(x); }*/
// #else
// #define DEFINE_MATH_UNARY_FUNC_WRAPPER(f) \
// 	inline double      f(double x)         { return ::f(x); } \
// 	inline float       f##f(float x)       { return ::f(x); } \
// 	/*inline long double f##l(long double x) { return ::f(x); }*/
// #endif
// DEFINE_MATH_UNARY_FUNC_WRAPPER(cos)
// DEFINE_MATH_UNARY_FUNC_WRAPPER(sin)
// DEFINE_MATH_UNARY_FUNC_WRAPPER(tan)
// DEFINE_MATH_UNARY_FUNC_WRAPPER(acos)
// DEFINE_MATH_UNARY_FUNC_WRAPPER(asin)
// DEFINE_MATH_UNARY_FUNC_WRAPPER(atan)
// template<typename T> inline T atan2(T y, T x) { return ::atan2(y, x); }
// DEFINE_MATH_UNARY_FUNC_WRAPPER(cosh)
// DEFINE_MATH_UNARY_FUNC_WRAPPER(sinh)
// DEFINE_MATH_UNARY_FUNC_WRAPPER(tanh)
// DEFINE_MATH_UNARY_FUNC_WRAPPER(exp)
// template<typename T> inline T frexp(T x, int* exp) { return ::frexp(x, exp); }
// template<typename T> inline T ldexp(T x, int  exp) { return ::ldexp(x, exp); }
// DEFINE_MATH_UNARY_FUNC_WRAPPER(log)
// DEFINE_MATH_UNARY_FUNC_WRAPPER(log10)
// template<typename T> inline T modf(T x, T* intpart) { return ::modf(x, intpart); }
// template<typename T> inline T pow(T x, T y) { return ::pow(x, y); }
// DEFINE_MATH_UNARY_FUNC_WRAPPER(sqrt)
// template<typename T> inline T fmod(T n, T d) { return ::fmod(n, d); }
// DEFINE_MATH_UNARY_FUNC_WRAPPER(fabs)
// template<typename T> inline T abs(T x) { return ::abs(x); }
// #if __cplusplus >= 201103L
// DEFINE_MATH_UNARY_FUNC_WRAPPER(acosh)
// DEFINE_MATH_UNARY_FUNC_WRAPPER(asinh)
// DEFINE_MATH_UNARY_FUNC_WRAPPER(atanh)
// DEFINE_MATH_UNARY_FUNC_WRAPPER(exp2)
// DEFINE_MATH_UNARY_FUNC_WRAPPER(expm1)
// template<typename T> inline int ilogb(T x) { return ::ilogb(x); }
// DEFINE_MATH_UNARY_FUNC_WRAPPER(log1p)
// DEFINE_MATH_UNARY_FUNC_WRAPPER(log2)
// DEFINE_MATH_UNARY_FUNC_WRAPPER(logb)
// template<typename T> inline T scalbn (T x, int n)  { return ::scalbn(x, n); }
// template<typename T> inline T scalbln(T x, long n) { return ::scalbn(x, n); }
// DEFINE_MATH_UNARY_FUNC_WRAPPER(cbrt)
// template<typename T> inline T hypot(T x, T y) { return ::hypot(x, y); }
// DEFINE_MATH_UNARY_FUNC_WRAPPER(erf)
// DEFINE_MATH_UNARY_FUNC_WRAPPER(erfc)
// DEFINE_MATH_UNARY_FUNC_WRAPPER(tgamma)
// DEFINE_MATH_UNARY_FUNC_WRAPPER(lgamma)
// DEFINE_MATH_UNARY_FUNC_WRAPPER(round)
// DEFINE_MATH_UNARY_FUNC_WRAPPER(rint)
// DEFINE_MATH_UNARY_FUNC_WRAPPER(trunc)
// DEFINE_MATH_UNARY_FUNC_WRAPPER(floor)
// DEFINE_MATH_UNARY_FUNC_WRAPPER(ceil)
// template<typename T> inline long lround(T x) { return ::lround(x); }
// template<typename T> inline long long llround(T x) { return ::llround(x); }
// DEFINE_MATH_UNARY_FUNC_WRAPPER(rint)
// template<typename T> inline long lrint(T x) { return ::lrint(x); }
// template<typename T> inline long long llrint(T x) { return ::llrint(x); }
// DEFINE_MATH_UNARY_FUNC_WRAPPER(nearbyint)
// //DEFINE_MATH_UNARY_FUNC_WRAPPER(isfinite)
// // TODO: remainder, remquo, copysign, nan, nextafter, nexttoward, fdim,
// // fmax, fmin, fma
// #endif
// #undef DEFINE_MATH_UNARY_FUNC_WRAPPER

#define M_PI 3.14159265358979323846

#if defined(__CUDACC__)

#if defined(__clang__)
// When compiling CUDA with barebones Clang we need to define its builtins and runtime functions ourselves.
#include "cuda_crt.h"
#endif

#else

extern "C" {

// stdio.h
int printf(const char* format, ...);

// stdlib.h
int abs(int);
long long llabs(long long);

// math.h
float fmodf(float, float);
double fmod(double, double);
float logf(float);
double log(double);
float log2f(float);
double log2(double);
float log10f(float);
double log10(double);
float expf(float);
double exp(double);
float sqrtf(float);
double sqrt(double);
float cbrtf(float);
double cbrt(double);
float powf(float, float);
double pow(double, double);
float floorf(float);
double floor(double);
float ceilf(float);
double ceil(double);
float fabsf(float);
double fabs(double);
float roundf(float);
double round(double);
float truncf(float);
double trunc(double);
float rintf(float);
double rint(double);
float acosf(float);
double acos(double);
float asinf(float);
double asin(double);
float atanf(float);
double atan(double);
float atan2f(float, float);
double atan2(double, double);
float cosf(float);
double cos(double);
float sinf(float);
double sin(double);
float tanf(float);
double tan(double);
float sinhf(float);
double sinh(double);
float coshf(float);
double cosh(double);
float tanhf(float);
double tanh(double);
float fmaf(float, float, float);
double fma(double, double, double);
double erf(double);
float erff(float);
double erfc(double);
float erfcf(float);
double erfinv(double);
float erfinvf(float);
double erfcinv(double);
float erfcinvf(float);

// stddef.h
#if defined(_WIN32)
using size_t = unsigned __int64;
#else
using size_t = unsigned long;
#endif

// string.h
void* memset(void*, int, size_t);
void* memcpy(void*, const void*, size_t);

// stdlib.h
void* malloc(size_t);
void free(void*);

}  // extern "C"

// cmath
inline bool isfinite(double x) { return _wp_isfinite(x); }

inline bool isnan(double x) { return _wp_isnan(x); }

inline bool isinf(double x) { return _wp_isinf(x); }

// assert.h
#ifdef NDEBUG
#define assert(expression) ((void)0)
#else
#define assert(expression) (void)(                                    \
            (!!(expression)) ||                                           \
            (_wp_assert((#expression), (__FILE__), (unsigned)(__LINE__)), 0) \
        )
#endif

#endif  // !__CUDACC__

#endif  // WP_NO_CRT

#if !defined(__CUDACC__)

/*
 * From Cephes Library polevl.c
 * Original source: https://www.netlib.org/cephes/
 * Copyright (c) 1984 by Stephen L. Moshier.
 * All rights reserved.
 */
// evaluate polynomial using Horner's method
static inline double polevl(double x, const double* coefs, int N)
{
    double ans = coefs[0];
    for (int i = 1; i <= N; i++) {
        ans = ans * x + coefs[i];
    }
    return ans;
}

/*
 * From Cephes Library polevl.c
 * Original source: https://www.netlib.org/cephes/
 * Copyright (c) 1984 by Stephen L. Moshier.
 * All rights reserved.
 */
// evaluate polynomial assuming leading coef = 1, using Horner's method
static inline double p1evl(double x, const double* coefs, int N)
{
    double ans = x + coefs[0];
    for (int i = 1; i < N; i++) {
        ans = ans * x + coefs[i];
    }
    return ans;
}

/*
 * From Cephes Library ndtri.c
 * Original source: https://www.netlib.org/cephes/
 * Copyright (c) 1984 by Stephen L. Moshier.
 * All rights reserved.
 */
// inverse normal distribution function (ndtri)
static inline double ndtri(double y)
{
    // domain check
    if (y <= 0.0 || y >= 1.0) {
        return (y <= 0.0) ? -HUGE_VAL : HUGE_VAL;
    }

    // constants from Cephes
    const double s2pi = 2.50662827463100050242E0;  // sqrt(2*pi)
    const double exp_neg2 = 0.13533528323661269189;  // exp(-2)

    // approximation for 0 <= abs(z - 0.5) <= 3/8
    static const double P0[5] = { -5.99633501014107895267e1, 9.80010754185999661536e1, -5.66762857469070293439e1,
                                  1.39312609387279679503e1, -1.23916583867381258016e0 };

    static const double Q0[8]
        = { 1.95448858338141759834e0, 4.67627912898881538453e0,  8.63602421390890590575e1, -2.25462687854119370527e2,
            2.00260212380060660359e2, -8.20372256168333339912e1, 1.59056225126211695515e1, -1.18331621121330003142e0 };

    // approximation for interval z = sqrt(-2 log y) between 2 and 8
    static const double P1[9] = { 4.05544892305962419923e0,   3.15251094599893866154e1,   5.71628192246421288162e1,
                                  4.40805073893200834700e1,   1.46849561928858024014e1,   2.18663306850790267539e0,
                                  -1.40256079171354495875e-1, -3.50424626827848203418e-2, -8.57456785154685413611e-4 };

    static const double Q1[8] = { 1.57799883256466749731e1,   4.53907635128879210584e1,  4.13172038254672030440e1,
                                  1.50425385692907503408e1,   2.50464946208309415979e0,  -1.42182922854787788574e-1,
                                  -3.80806407691578277194e-2, -9.33259480895457427372e-4 };

    // approximation for interval z = sqrt(-2 log y) between 8 and 64
    static const double P2[9] = { 3.23774891776946035970e0,  6.91522889068984211695e0,  3.93881025292474443415e0,
                                  1.33303460815807542389e0,  2.01485389549179081538e-1, 1.23716634817820021358e-2,
                                  3.01581553508235416007e-4, 2.65806974686737550832e-6, 6.23974539184983293730e-9 };

    static const double Q2[8] = { 6.02427039364742014255e0,  3.67983563856160859403e0,  1.37702099489081330271e0,
                                  2.16236993594496635890e-1, 1.34204006088543189037e-2, 3.28014464682127739104e-4,
                                  2.89247864745380683936e-6, 6.79019408009981274425e-9 };

    int code = 1;
    double y_work = y;

    if (y_work > (1.0 - exp_neg2)) {
        y_work = 1.0 - y_work;
        code = 0;
    }

    // middle region: 0 <= |y - 0.5| <= 3/8
    if (y_work > exp_neg2) {
        y_work -= 0.5;
        double y2 = y_work * y_work;
        double x = y_work + y_work * (y2 * polevl(y2, P0, 4) / p1evl(y2, Q0, 8));
        x = x * s2pi;
        return x;
    }

    double x = ::sqrt(-2.0 * ::log(y_work));
    double x0 = x - ::log(x) / x;

    double z = 1.0 / x;
    double x1;
    if (x < 8.0) {
        x1 = z * polevl(z, P1, 8) / p1evl(z, Q1, 8);
    } else {
        x1 = z * polevl(z, P2, 8) / p1evl(z, Q2, 8);
    }

    x = x0 - x1;
    if (code != 0) {
        x = -x;
    }

    return x;
}

// inverse error function (not in standard C library)
// only compiled for non-CUDA builds - CUDA provides these in its math headers
inline double erfinv(double z)
{
    // handle special cases
    if (z == 0.0)
        return 0.0;
    if (z == 1.0)
        return HUGE_VAL;  // infinity
    if (z == -1.0)
        return -HUGE_VAL;  // -infinity
    if (z < -1.0 || z > 1.0)
        return NAN;  // outside valid range

    // erfinv(z) = ndtri((z + 1) / 2) / sqrt(2)
    return ndtri((z + 1.0) / 2.0) / ::sqrt(2.0);
}

inline float erfinvf(float x) { return (float)erfinv((double)x); }

inline double erfcinv(double x) { return erfinv(1.0 - x); }

inline float erfcinvf(float x) { return (float)erfcinv((double)x); }

#endif  // !defined(__CUDACC__)
