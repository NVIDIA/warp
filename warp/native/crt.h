/** Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#ifndef WP_NO_CRT

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <float.h>

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
enum {
  _JITIFY_CHAR_IS_UNSIGNED = (char)-1 >= 0,
  CHAR_MIN = _JITIFY_CHAR_IS_UNSIGNED ? 0 : SCHAR_MIN,
  CHAR_MAX = _JITIFY_CHAR_IS_UNSIGNED ? UCHAR_MAX : SCHAR_MAX,
};
#define SHRT_MIN    (-32768)
#define SHRT_MAX    32767
#define USHRT_MAX   65535
#define INT_MIN     (-INT_MAX - 1)
#define INT_MAX     2147483647
#define UINT_MAX    4294967295U
#if __WORDSIZE == 64
 # define LONG_MAX  9223372036854775807L
#else
 # define LONG_MAX  2147483647L
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


/// stdint
typedef signed char      int8_t;
typedef signed short     int16_t;
typedef signed int       int32_t;
typedef signed long long int64_t;
typedef signed char      int_fast8_t;
typedef signed short     int_fast16_t;
typedef signed int       int_fast32_t;
typedef signed long long int_fast64_t;
typedef signed char      int_least8_t;
typedef signed short     int_least16_t;
typedef signed int       int_least32_t;
typedef signed long long int_least64_t;
typedef signed long long intmax_t;
typedef signed long      intptr_t; 
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef unsigned char      uint_fast8_t;
typedef unsigned short     uint_fast16_t;
typedef unsigned int       uint_fast32_t;
typedef unsigned long long uint_fast64_t;
typedef unsigned char      uint_least8_t;
typedef unsigned short     uint_least16_t;
typedef unsigned int       uint_least32_t;
typedef unsigned long long uint_least64_t;
typedef unsigned long long uintmax_t;


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


// stdio
extern "C"
{
  __device__ int printf(const char * format, ... );
}

// assert.h
//#ifndef __CUDACC__
//#include <cassert>
//#endif
//#define assert(x) {}

#endif // WP_CUDA