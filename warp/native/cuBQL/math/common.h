// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifndef _USE_MATH_DEFINES
#  define _USE_MATH_DEFINES
#endif
#include <math.h> // using cmath causes issues under Windows

#include <stdio.h>
#include <inttypes.h>
#include <iostream>
#include <stdexcept>
#include <memory>
#include <assert.h>
#include <string>
#include <math.h>
#include <cmath>
#include <cfloat>
#include <algorithm>
#include <sstream>
#ifdef __GNUC__
# include <execinfo.h>
# include <sys/time.h>
#endif
#if defined(__HIPCC__)
#  include <hip/hip_runtime.h>
#  include <hip/driver_types.h>
#  include <hip/hip_runtime.h>
#elif defined(__CUDACC__)
#  include <cuda_runtime.h>
#endif

#ifdef _WIN32
# ifndef WIN32_LEAN_AND_MEAN
#  define WIN32_LEAN_AND_MEAN
# endif
# include <Windows.h>
# ifdef min
#  undef min
# endif
# ifdef max
#  undef max
# endif
#endif

#if !defined(WIN32)
#include <signal.h>
#endif

#if defined(_MSC_VER)
#  define CUBQL_DLL_EXPORT __declspec(dllexport)
#  define CUBQL_DLL_IMPORT __declspec(dllimport)
#elif defined(__clang__) || defined(__GNUC__)
#  define CUBQL_DLL_EXPORT __attribute__((visibility("default")))
#  define CUBQL_DLL_IMPORT __attribute__((visibility("default")))
#else
#  define CUBQL_DLL_EXPORT
#  define CUBQL_DLL_IMPORT
#endif

# define CUBQL_INTERFACE /* nothing - currently not building any special 'cubql.dll' */

#ifndef __PRETTY_FUNCTION__
# if defined(__func__)
#  define __PRETTY_FUNCTION__ __func__
// #  define __PRETTY_FUNCTION__ __FILE__##"::"##__LINE__##": "##__FUNCTION__
# else
#  define __PRETTY_FUNCTION__ __FUNCTION__
# endif
#endif


#ifndef PRINT
# define PRINT(var) std::cout << #var << "=" << var << std::endl;
#ifdef __WIN32__
# define PING std::cout << __FILE__ << "::" << __LINE__ << ": " << __FUNCTION__ << std::endl;
#else
# define PING std::cout << __FILE__ << "::" << __LINE__ << ": " << __PRETTY_FUNCTION__ << std::endl;
#endif
#endif

#if defined(__CUDACC__)
# define __cubql_device   __device__
# define __cubql_host     __host__
#elif defined(__HIPCC__)
# define __cubql_device   __device__
# define __cubql_host     __host__
#else
# define __cubql_device   /* ignore */
# define __cubql_host     /* ignore */
#endif

# define __cubql_both   __cubql_host __cubql_device


#ifdef __GNUC__
#define MAYBE_UNUSED __attribute__((unused))
#else
#define MAYBE_UNUSED
#endif

namespace cuBQL {

  using longlong = int64_t;
  
  namespace detail {
    inline static std::string backtrace()
    {
#ifdef __GNUC__
      static const int max_frames = 16;

      void* buffer[max_frames] = { 0 };
      int cnt = ::backtrace(buffer,max_frames);

      char** symbols = backtrace_symbols(buffer,cnt);

      if (symbols) {
        std::stringstream str;
        for (int n = 1; n < cnt; ++n) // skip the 1st entry (address of this function)
          {
            str << symbols[n] << '\n';
          }
        free(symbols);
        return str.str();
      }
      return "";
#else
      return "not implemented yet";
#endif
    }

    inline void cubqlRaise_impl(std::string str)
    {
      fprintf(stderr,"%s\n",str.c_str());
#ifdef WIN32
      if (IsDebuggerPresent())
        DebugBreak();
      else
        throw std::runtime_error(str);
#else
#ifndef NDEBUG
      std::string bt = backtrace();
      fprintf(stderr,"%s\n",bt.c_str());
#endif
      raise(SIGINT);
#endif
    }
  }
}

#define CUBQL_NOTIMPLEMENTED throw std::runtime_error(std::string(__PRETTY_FUNCTION__)+" not implemented")

#ifdef WIN32
# define CUBQL_TERMINAL_RED ""
# define CUBQL_TERMINAL_GREEN ""
# define CUBQL_TERMINAL_LIGHT_GREEN ""
# define CUBQL_TERMINAL_YELLOW ""
# define CUBQL_TERMINAL_BLUE ""
# define CUBQL_TERMINAL_LIGHT_BLUE ""
# define CUBQL_TERMINAL_RESET ""
# define CUBQL_TERMINAL_DEFAULT CUBQL_TERMINAL_RESET
# define CUBQL_TERMINAL_BOLD ""

# define CUBQL_TERMINAL_MAGENTA ""
# define CUBQL_TERMINAL_LIGHT_MAGENTA ""
# define CUBQL_TERMINAL_CYAN ""
# define CUBQL_TERMINAL_LIGHT_RED ""
#else
# define CUBQL_TERMINAL_RED "\033[0;31m"
# define CUBQL_TERMINAL_GREEN "\033[0;32m"
# define CUBQL_TERMINAL_LIGHT_GREEN "\033[1;32m"
# define CUBQL_TERMINAL_YELLOW "\033[1;33m"
# define CUBQL_TERMINAL_BLUE "\033[0;34m"
# define CUBQL_TERMINAL_LIGHT_BLUE "\033[1;34m"
# define CUBQL_TERMINAL_RESET "\033[0m"
# define CUBQL_TERMINAL_DEFAULT CUBQL_TERMINAL_RESET
# define CUBQL_TERMINAL_BOLD "\033[1;1m"

# define CUBQL_TERMINAL_MAGENTA "\e[35m"
# define CUBQL_TERMINAL_LIGHT_MAGENTA "\e[95m"
# define CUBQL_TERMINAL_CYAN "\e[36m"
# define CUBQL_TERMINAL_LIGHT_RED "\033[1;31m"
#endif

#ifdef _MSC_VER
# define CUBQL_ALIGN(alignment) __declspec(align(alignment)) 
#else
# define CUBQL_ALIGN(alignment) __attribute__((aligned(alignment)))
#endif



namespace cuBQL {

  inline __cubql_both int32_t  divRoundUp(int32_t a, int32_t b) { return (a+b-1)/b; }
  inline __cubql_both uint32_t divRoundUp(uint32_t a, uint32_t b) { return (a+b-1)/b; }
  inline __cubql_both int64_t  divRoundUp(int64_t a, int64_t b) { return (a+b-1)/b; }
  inline __cubql_both uint64_t divRoundUp(uint64_t a, uint64_t b) { return (a+b-1)/b; }
  
#ifdef __WIN32__
#  define cubql_snprintf sprintf_s
#else
#  define cubql_snprintf snprintf
#endif
  
  /*! added pretty-print function for large numbers, printing 10000000 as "10M" instead */
  inline std::string prettyDouble(const double val) {
    const double absVal = abs(val);
    char result[1000];

    if      (absVal >= 1e+18f) cubql_snprintf(result,1000,"%.1f%c",float(val/1e18f),'E');
    else if (absVal >= 1e+15f) cubql_snprintf(result,1000,"%.1f%c",float(val/1e15f),'P');
    else if (absVal >= 1e+12f) cubql_snprintf(result,1000,"%.1f%c",float(val/1e12f),'T');
    else if (absVal >= 1e+09f) cubql_snprintf(result,1000,"%.1f%c",float(val/1e09f),'G');
    else if (absVal >= 1e+06f) cubql_snprintf(result,1000,"%.1f%c",float(val/1e06f),'M');
    else if (absVal >= 1e+03f) cubql_snprintf(result,1000,"%.1f%c",float(val/1e03f),'k');
    else if (absVal <= 1e-12f) cubql_snprintf(result,1000,"%.1f%c",float(val*1e15f),'f');
    else if (absVal <= 1e-09f) cubql_snprintf(result,1000,"%.1f%c",float(val*1e12f),'p');
    else if (absVal <= 1e-06f) cubql_snprintf(result,1000,"%.1f%c",float(val*1e09f),'n');
    else if (absVal <= 1e-03f) cubql_snprintf(result,1000,"%.1f%c",float(val*1e06f),'u');
    else if (absVal <= 1e-00f) cubql_snprintf(result,1000,"%.1f%c",float(val*1e03f),'m');
    else cubql_snprintf(result,1000,"%f",(float)val);

    return result;
  }
  

  /*! return a nicely formatted number as in "3.4M" instead of
    "3400000", etc, using mulitples of thousands (K), millions
    (M), etc. Ie, the value 64000 would be returned as 64K, and
    65536 would be 65.5K */
  inline std::string prettyNumber(const size_t s)
  {
    char buf[1000];
    if (s >= (1000LL*1000LL*1000LL*1000LL)) {
      cubql_snprintf(buf, 1000,"%.2fT",s/(1000.f*1000.f*1000.f*1000.f));
    } else if (s >= (1000LL*1000LL*1000LL)) {
      cubql_snprintf(buf, 1000, "%.2fG",s/(1000.f*1000.f*1000.f));
    } else if (s >= (1000LL*1000LL)) {
      cubql_snprintf(buf, 1000, "%.2fM",s/(1000.f*1000.f));
    } else if (s >= (1000LL)) {
      cubql_snprintf(buf, 1000, "%.2fK",s/(1000.f));
    } else {
      cubql_snprintf(buf,1000,"%zi",s);
    }
    return buf;
  }

  /*! return a nicely formatted number as in "3.4M" instead of
    "3400000", etc, using mulitples of 1024 as in kilobytes,
    etc. Ie, the value 65534 would be 64K, 64000 would be 63.8K */
  inline std::string prettyBytes(const size_t s)
  {
    char buf[1000];
    if (s >= (1024LL*1024LL*1024LL*1024LL)) {
      cubql_snprintf(buf, 1000,"%.2fT",s/(1024.f*1024.f*1024.f*1024.f));
    } else if (s >= (1024LL*1024LL*1024LL)) {
      cubql_snprintf(buf, 1000, "%.2fG",s/(1024.f*1024.f*1024.f));
    } else if (s >= (1024LL*1024LL)) {
      cubql_snprintf(buf, 1000, "%.2fM",s/(1024.f*1024.f));
    } else if (s >= (1024LL)) {
      cubql_snprintf(buf, 1000, "%.2fK",s/(1024.f));
    } else {
      cubql_snprintf(buf,1000,"%zi",s);
    }
    return buf;
  }
  
  inline double getCurrentTime()
  {
#ifdef _WIN32
    SYSTEMTIME tp; GetSystemTime(&tp);
    /*
      Please note: we are not handling the "leap year" issue.
    */
    size_t numSecsSince2020
      = tp.wSecond
      + (60ull) * tp.wMinute
      + (60ull * 60ull) * tp.wHour
      + (60ull * 60ul * 24ull) * tp.wDay
      + (60ull * 60ul * 24ull * 365ull) * (tp.wYear - 2020);
    return double(numSecsSince2020 + tp.wMilliseconds * 1e-3);
#else
    struct timeval tp; gettimeofday(&tp,nullptr);
    return double(tp.tv_sec) + double(tp.tv_usec)/1E6;
#endif
  }

  inline bool hasSuffix(const std::string &s, const std::string &suffix)
  {
    return s.substr(s.size()-suffix.size()) == suffix;
  }

  template<typename T> struct is_real { enum { value = false }; };
  template<> struct is_real<float> { enum { value = true }; };
  template<> struct is_real<double> { enum { value = true }; };
} // ::cubql



#if defined(__CUDACC__) || defined(__HIPCC__)

#define CUBQL_RAISE(MSG) ::cuBQL::detail::cubqlRaise_impl(MSG);

#define CUBQL_CUDA_CHECK( call )                                        \
  {                                                                     \
    cudaError_t rc = call;                                              \
    if (rc != cudaSuccess) {                                            \
      fprintf(stderr,                                                   \
              "CUDA call (%s) failed with code %d (line %d): %s\n",     \
              #call, rc, __LINE__, cudaGetErrorString(rc));             \
      CUBQL_RAISE("fatal cuda error");                                  \
    }                                                                   \
  }

#define CUBQL_CUDA_CALL(call) CUBQL_CUDA_CHECK(cuda##call)

#define CUBQL_CUDA_CHECK2( where, call )                                \
  {                                                                     \
    cudaError_t rc = call;                                              \
    if(rc != cudaSuccess) {                                             \
      if (where)                                                        \
        fprintf(stderr, "at %s: CUDA call (%s) "                        \
                "failed with code %d (line %d): %s\n",                  \
                where,#call, rc, __LINE__, cudaGetErrorString(rc));     \
      fprintf(stderr,                                                   \
              "CUDA call (%s) failed with code %d (line %d): %s\n",     \
              #call, rc, __LINE__, cudaGetErrorString(rc));             \
      CUBQL_RAISE("fatal cuda error");                                  \
    }                                                                   \
  }

#define CUBQL_CUDA_SYNC_CHECK()                                 \
  {                                                             \
    cudaDeviceSynchronize();                                    \
    cudaError_t rc = cudaGetLastError();                        \
    if (rc != cudaSuccess) {                                    \
      fprintf(stderr, "error (%s: line %d): %s\n",              \
              __FILE__, __LINE__, cudaGetErrorString(rc));      \
      CUBQL_RAISE("fatal cuda error");                          \
    }                                                           \
  }

#define CUBQL_CUDA_SYNC_CHECK_STREAM(s)                                 \
  {                                                                     \
    cudaError_t rc = cudaStreamSynchronize(s);                          \
    if (rc != cudaSuccess) {                                            \
      fprintf(stderr, "error (%s: line %d): %s\n",                      \
              __FILE__, __LINE__, cudaGetErrorString(rc));              \
      CUBQL_RAISE("fatal cuda error");                                  \
    }                                                                   \
  }



#define CUBQL_CUDA_CHECK_NOTHROW( call )                                \
  {                                                                     \
    cudaError_t rc = call;                                              \
    if (rc != cudaSuccess) {                                            \
      fprintf(stderr,                                                   \
              "CUDA call (%s) failed with code %d (line %d): %s\n",     \
              #call, rc, __LINE__, cudaGetErrorString(rc));             \
      exit(2);                                                          \
    }                                                                   \
  }

#define CUBQL_CUDA_CALL_NOTHROW(call) CUBQL_CUDA_CHECK_NOTHROW(cuda##call)

#define CUBQL_CUDA_CHECK2_NOTHROW( where, call )                        \
  {                                                                     \
    cudaError_t rc = call;                                              \
    if(rc != cudaSuccess) {                                             \
      if (where)                                                        \
        fprintf(stderr, "at %s: CUDA call (%s) "                        \
                "failed with code %d (line %d): %s\n",                  \
                where,#call, rc, __LINE__, cudaGetErrorString(rc));     \
      fprintf(stderr,                                                   \
              "CUDA call (%s) failed with code %d (line %d): %s\n",     \
              #call, rc, __LINE__, cudaGetErrorString(rc));             \
      exit(2);                                                          \
    }                                                                   \
  }
#endif



namespace cuBQL {
  struct dbgout {
    static constexpr const char *const endl = "\n";
  };
  static constexpr const char *const endl = "\n";
  static constexpr dbgout dout = {};
  inline __cubql_both dbgout operator<<(dbgout o, const char *s)
  { printf("%s",s); return o; }
  
  inline __cubql_both dbgout operator<<(dbgout o, int32_t i)
  { printf("%i",i); return o; }
  inline __cubql_both dbgout operator<<(dbgout o, uint32_t i)
  { printf("%u",i); return o; }
  inline __cubql_both dbgout operator<<(dbgout o, float f)
  { printf("%f",f); return o; }
  inline __cubql_both dbgout operator<<(dbgout o, double f)
  { printf("%lf",f); return o; }
  inline __cubql_both dbgout operator<<(dbgout o, uint64_t i)
  { printf("%" PRIu64, i); return o; }
  inline __cubql_both dbgout operator<<(dbgout o, int64_t i)
  { printf("%" PRId64, i); return o; }
  template <typename T>
  inline __cubql_both dbgout operator<<(dbgout o, T *ptr)
  { printf("%p",ptr); return o; }

  inline __cubql_both float abst(float f)   { return (f < 0.f) ? -f : f; }
  inline __cubql_both double abst(double f) { return (f < 0. ) ? -f : f; }
  
};


