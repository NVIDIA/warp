// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cuBQL/math/common.h"
#ifdef __CUDACC__
#include <cuda/std/limits>
#endif
#include <limits>

namespace cuBQL {
  
#ifdef __CUDACC__
  // make sure we use the built-in cuda functoins that use floats, not
  // the c-stdlib ones that use doubles.
  using ::min;
  using ::max;
#else
  using std::min;
  using std::max;
#endif

#ifdef __CUDA_ARCH__
# define CUBQL_INF ::cuda::std::numeric_limits<float>::infinity()
#else
# define CUBQL_INF std::numeric_limits<float>::infinity()
#endif

#ifdef __CUDA_ARCH__
#else
    inline float __int_as_float(int i) { return (const float &)i; }
    inline int __float_as_int(float f) { return (const int &)f; }
#endif

  inline __cubql_both float squareOf(float f) { return f*f; }
  
  
  template<int N> struct log_of { enum { value = -1 }; };
  template<> struct log_of< 2> { enum { value = 1 }; };
  template<> struct log_of< 4> { enum { value = 2 }; };
  template<> struct log_of< 8> { enum { value = 3 }; };
  template<> struct log_of<16> { enum { value = 4 }; };
  template<> struct log_of<32> { enum { value = 5 }; };

  /*! square of a value */
  inline __cubql_both float sqr(float f) { return f*f; }
  
  /*! unary functors on scalar types, so we can lift them to vector types later on */
  inline __cubql_both float  rcp(float f)     { return 1.f/f; }
  inline __cubql_both double rcp(double d)    { return 1./d; }

  template<typename T>
  inline __cubql_both T clamp(T t, T lo=T(0), T hi=T(1))
  { return min(max(t,lo),hi); }

  inline __cubql_both float saturate(float f) { return clamp(f,0.f,1.f); }
  inline __cubql_both double saturate(double f) { return clamp(f,0.,1.); }

  // inline __cubql_both float sqrt(float f) { return ::sqrtf(f); }
  // inline __cubql_both double sqrt(double d) { return ::sqrt(d); }
}

