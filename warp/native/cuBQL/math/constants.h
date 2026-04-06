// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <limits>
#include <limits.h>
#if defined(__CUDACC__) && !defined(CUDART_INF_F)
#include <math_constants.h>
#endif

#ifndef M_PI
#define M_PI 3.141593f
#endif

namespace cuBQL {

  static struct ZeroTy
  {
    __cubql_both operator          double   ( ) const { return 0; }
    __cubql_both operator          float    ( ) const { return 0; }
    __cubql_both operator          long long( ) const { return 0; }
    __cubql_both operator unsigned long long( ) const { return 0; }
    __cubql_both operator          long     ( ) const { return 0; }
    __cubql_both operator unsigned long     ( ) const { return 0; }
    __cubql_both operator          int      ( ) const { return 0; }
    __cubql_both operator unsigned int      ( ) const { return 0; }
    __cubql_both operator          short    ( ) const { return 0; }
    __cubql_both operator unsigned short    ( ) const { return 0; }
    __cubql_both operator          char     ( ) const { return 0; }
    __cubql_both operator unsigned char     ( ) const { return 0; }
  } zero MAYBE_UNUSED;

  static struct OneTy
  {
    __cubql_both operator          double   ( ) const { return 1; }
    __cubql_both operator          float    ( ) const { return 1; }
    __cubql_both operator          long long( ) const { return 1; }
    __cubql_both operator unsigned long long( ) const { return 1; }
    __cubql_both operator          long     ( ) const { return 1; }
    __cubql_both operator unsigned long     ( ) const { return 1; }
    __cubql_both operator          int      ( ) const { return 1; }
    __cubql_both operator unsigned int      ( ) const { return 1; }
    __cubql_both operator          short    ( ) const { return 1; }
    __cubql_both operator unsigned short    ( ) const { return 1; }
    __cubql_both operator          char     ( ) const { return 1; }
    __cubql_both operator unsigned char     ( ) const { return 1; }
  } one MAYBE_UNUSED;

  static struct NegInfTy
  {
#ifdef __CUDA_ARCH__
    __device__ operator          double   ( ) const { return -CUDART_INF; }
    __device__ operator          float    ( ) const { return -CUDART_INF_F; }
#else
    __cubql_both operator          double   ( ) const { return -std::numeric_limits<double>::infinity(); }
    __cubql_both operator          float    ( ) const { return -std::numeric_limits<float>::infinity(); }
    __cubql_both operator          long long( ) const { return std::numeric_limits<long long>::min(); }
    __cubql_both operator unsigned long long( ) const { return std::numeric_limits<unsigned long long>::min(); }
    __cubql_both operator          long     ( ) const { return std::numeric_limits<long>::min(); }
    __cubql_both operator unsigned long     ( ) const { return std::numeric_limits<unsigned long>::min(); }
    __cubql_both operator          int      ( ) const { return std::numeric_limits<int>::min(); }
    __cubql_both operator unsigned int      ( ) const { return std::numeric_limits<unsigned int>::min(); }
    __cubql_both operator          short    ( ) const { return std::numeric_limits<short>::min(); }
    __cubql_both operator unsigned short    ( ) const { return std::numeric_limits<unsigned short>::min(); }
    __cubql_both operator          char     ( ) const { return std::numeric_limits<char>::min(); }
    __cubql_both operator unsigned char     ( ) const { return std::numeric_limits<unsigned char>::min(); }
#endif
  } neg_inf MAYBE_UNUSED;

  inline __cubql_both float infty() {
#if defined(__CUDA_ARCH__)
    return CUDART_INF_F; 
#else
    return std::numeric_limits<float>::infinity(); 
#endif
  }
  
  static struct PosInfTy
  {
#ifdef __CUDA_ARCH__
    __device__ operator          double   ( ) const { return CUDART_INF; }
    __device__ operator          float    ( ) const { return CUDART_INF_F; }
#else
    __cubql_both operator          double   ( ) const { return std::numeric_limits<double>::infinity(); }
    __cubql_both operator          float    ( ) const { return std::numeric_limits<float>::infinity(); }
    __cubql_both operator          long long( ) const { return std::numeric_limits<long long>::max(); }
    __cubql_both operator unsigned long long( ) const { return std::numeric_limits<unsigned long long>::max(); }
    __cubql_both operator          long     ( ) const { return std::numeric_limits<long>::max(); }
    __cubql_both operator unsigned long     ( ) const { return std::numeric_limits<unsigned long>::max(); }
    __cubql_both operator          int      ( ) const { return std::numeric_limits<int>::max(); }
    __cubql_both operator unsigned int      ( ) const { return std::numeric_limits<unsigned int>::max(); }
    __cubql_both operator          short    ( ) const { return std::numeric_limits<short>::max(); }
    __cubql_both operator unsigned short    ( ) const { return std::numeric_limits<unsigned short>::max(); }
    __cubql_both operator          char     ( ) const { return std::numeric_limits<char>::max(); }
    __cubql_both operator unsigned char     ( ) const { return std::numeric_limits<unsigned char>::max(); }
#endif
  } inf MAYBE_UNUSED, pos_inf MAYBE_UNUSED;

  static struct NaNTy
  {
#ifdef __CUDA_ARCH__
    __device__ operator double( ) const { return CUDART_NAN; }
    __device__ operator float ( ) const { return CUDART_NAN_F; }
#else
    __cubql_both operator double( ) const { return std::numeric_limits<double>::quiet_NaN(); }
    __cubql_both operator float ( ) const { return std::numeric_limits<float>::quiet_NaN(); }
#endif
  } nan MAYBE_UNUSED;

  static struct UlpTy
  {
#ifdef __CUDA_ARCH__
    // todo
#else
    __cubql_both operator double( ) const { return std::numeric_limits<double>::epsilon(); }
    __cubql_both operator float ( ) const { return std::numeric_limits<float>::epsilon(); }
#endif
  } ulp MAYBE_UNUSED;



  template<bool is_integer>
  struct limits_traits;

  template<> struct limits_traits<true> {
    template<typename T> static inline __cubql_both T value_limits_lower(T) { return std::numeric_limits<T>::min(); }
    template<typename T> static inline __cubql_both T value_limits_upper(T) { return std::numeric_limits<T>::max(); }
  };
  template<> struct limits_traits<false> {
    template<typename T> static inline __cubql_both T value_limits_lower(T) { return (T)NegInfTy(); }//{ return -std::numeric_limits<T>::infinity(); }
    template<typename T> static inline __cubql_both T value_limits_upper(T) { return (T)PosInfTy(); }//{ return +std::numeric_limits<T>::infinity();  }
  };
  
  /*! lower value of a completely *empty* range [+inf..-inf] */
  template<typename T> inline __cubql_both T empty_bounds_lower()
  {
    return limits_traits<std::numeric_limits<T>::is_integer>::value_limits_upper(T());
  }
  
  /*! upper value of a completely *empty* range [+inf..-inf] */
  template<typename T> inline __cubql_both T empty_bounds_upper()
  {
    return limits_traits<std::numeric_limits<T>::is_integer>::value_limits_lower(T());
  }

  /*! lower value of a completely *empty* range [+inf..-inf] */
  template<typename T> inline __cubql_both T empty_range_lower()
  {
    return limits_traits<std::numeric_limits<T>::is_integer>::value_limits_upper(T());
  }
  
  /*! upper value of a completely *empty* range [+inf..-inf] */
  template<typename T> inline __cubql_both T empty_range_upper()
  {
    return limits_traits<std::numeric_limits<T>::is_integer>::value_limits_lower(T());
  }

  /*! lower value of a completely open range [-inf..+inf] */
  template<typename T> inline __cubql_both T open_range_lower()
  {
    return limits_traits<std::numeric_limits<T>::is_integer>::value_limits_lower(T());
  }

  /*! upper value of a completely open range [-inf..+inf] */
  template<typename T> inline __cubql_both T open_range_upper()
  {
    return limits_traits<std::numeric_limits<T>::is_integer>::value_limits_upper(T());
  }

  template<> inline __cubql_both uint32_t empty_bounds_lower<uint32_t>() 
  { return uint32_t(UINT_MAX); }
  template<> inline __cubql_both uint32_t empty_bounds_upper<uint32_t>() 
  { return uint32_t(0); }
  template<> inline __cubql_both uint32_t open_range_lower<uint32_t>() 
  { return uint32_t(0); }
  template<> inline __cubql_both uint32_t open_range_upper<uint32_t>() 
  { return uint32_t(UINT_MAX); }

  template<> inline __cubql_both int32_t empty_bounds_lower<int32_t>() 
  { return int32_t(INT_MAX); }
  template<> inline __cubql_both int32_t empty_bounds_upper<int32_t>() 
  { return int32_t(INT_MIN); }
  template<> inline __cubql_both int32_t open_range_lower<int32_t>() 
  { return int32_t(INT_MIN); }
  template<> inline __cubql_both int32_t open_range_upper<int32_t>() 
  { return int32_t(INT_MAX); }

  template<> inline __cubql_both uint64_t empty_bounds_lower<uint64_t>() 
  { return uint64_t(ULONG_MAX); }
  template<> inline __cubql_both uint64_t empty_bounds_upper<uint64_t>() 
  { return uint64_t(0); }
  template<> inline __cubql_both uint64_t open_range_lower<uint64_t>() 
  { return uint64_t(0); }
  template<> inline __cubql_both uint64_t open_range_upper<uint64_t>() 
  { return uint64_t(ULONG_MAX); }

  template<> inline __cubql_both int64_t empty_bounds_lower<int64_t>() 
  { return int64_t(LONG_MAX); }
  template<> inline __cubql_both int64_t empty_bounds_upper<int64_t>() 
  { return int64_t(LONG_MIN); }
  template<> inline __cubql_both int64_t open_range_lower<int64_t>() 
  { return int64_t(LONG_MIN); }
  template<> inline __cubql_both int64_t open_range_upper<int64_t>() 
  { return int64_t(LONG_MAX); }


  template<> inline __cubql_both uint16_t empty_bounds_lower<uint16_t>() 
  { return uint16_t(USHRT_MAX); }
  template<> inline __cubql_both uint16_t empty_bounds_upper<uint16_t>() 
  { return uint16_t(0); }
  template<> inline __cubql_both uint16_t open_range_lower<uint16_t>() 
  { return uint16_t(0); }
  template<> inline __cubql_both uint16_t open_range_upper<uint16_t>() 
  { return uint16_t(USHRT_MAX); }

  template<> inline __cubql_both int16_t empty_bounds_lower<int16_t>() 
  { return int16_t(SHRT_MAX); }
  template<> inline __cubql_both int16_t empty_bounds_upper<int16_t>() 
  { return int16_t(SHRT_MIN); }
  template<> inline __cubql_both int16_t open_range_lower<int16_t>() 
  { return int16_t(SHRT_MIN); }
  template<> inline __cubql_both int16_t open_range_upper<int16_t>() 
  { return int16_t(SHRT_MAX); }

  template<> inline __cubql_both uint8_t empty_bounds_lower<uint8_t>() 
  { return uint8_t(CHAR_MAX); }
  template<> inline __cubql_both uint8_t empty_bounds_upper<uint8_t>() 
  { return uint8_t(CHAR_MIN); }
  template<> inline __cubql_both uint8_t open_range_lower<uint8_t>() 
  { return uint8_t(CHAR_MIN); }
  template<> inline __cubql_both uint8_t open_range_upper<uint8_t>() 
  { return uint8_t(CHAR_MAX); }

  template<> inline __cubql_both int8_t empty_bounds_lower<int8_t>() 
  { return int8_t(SCHAR_MIN); }
  template<> inline __cubql_both int8_t empty_bounds_upper<int8_t>() 
  { return int8_t(SCHAR_MAX); }
  template<> inline __cubql_both int8_t open_range_lower<int8_t>() 
  { return int8_t(SCHAR_MAX); }
  template<> inline __cubql_both int8_t open_range_upper<int8_t>() 
  { return int8_t(SCHAR_MIN); }

} // ::cuBQL
