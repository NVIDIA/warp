// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file cuBQL/math/conservativeDistances.h Implements various helper functions to compute distances with well defined conservative rounding modes.

  When traversing BVHes we often need to compute distances between,
  say, a query point and a given subtree's bounding box. Based on the
  type of data used, this can lead to all kind of numerical issues, in
  particular because of the squares used in the dot product used in L2
  distance computations. To mitigate this we define some helper
  functions (defined in this header file) that will always - no matter
  what input data type - compute distances in floats, and use well
  defined rounding modes that return conservative values. Ie, the
  `float fSqrDistance_rd(vec2i, box2i)` will return a float that is
  guaranteed to be _smaller or equal to_ than whatever the correct
  number would have been */
#pragma once

#include "cuBQL/math/box.h"

namespace cuBQL {

  /*! convert a _positive_ int64 to float, with round-down */
  inline __cubql_both float toFloat_pos_rd(int64_t v)
  {
#ifdef __CUDA_ARCH__
    return __ll2float_rd(v);
#else
    float f = (float)v;
    if ((int64_t)f >= v) f = nextafter(f,CUBQL_INF);
    return f;
#endif
  }
  
  /*! convert a _positive_ double to float, with round-down */
  inline __cubql_both float toFloat_pos_rd(double v)
  {
#ifdef __CUDA_ARCH__
    return __double2float_rd(v);
#else
    float f = (float)v;
    if ((double)f >= v) f = nextafter(f,CUBQL_INF);
    return f;
#endif
  }
  
  // ------------------------------------------------------------------
  inline __cubql_both float fSquare_rd(float v)
  { return v*v; }

  inline __cubql_both float fSquare_rd(double v)
  { return toFloat_pos_rd(v*v); }

  inline __cubql_both float fSquare_rd(int v)
  { return toFloat_pos_rd(v*(int64_t)v); }
  
  inline __cubql_both float fSquare_rd(int64_t v)
  { float f = toFloat_pos_rd(v < 0 ? -v : v); return f*f; }

  // ------------------------------------------------------------------
  template<typename T, int D>
  inline __cubql_both float fSqrLength_rd(vec_t<T,D> v)
  {
    float sum = 0.f;
#pragma unroll
    for (int i=0;i<D;i++)
      sum += fSquare_rd(v[i]);
    return sum;
  }

  template<typename T>
  inline __cubql_both float fSqrLength_rd(vec_t<T,2> v)
  { return fSquare_rd(v.x)+fSquare_rd(v.y); }
  
  template<typename T>
  inline __cubql_both float fSqrLength_rd(vec_t<T,3> v)
  { return fSquare_rd(v.x)+fSquare_rd(v.y)+fSquare_rd(v.z); }

  template<typename T>
  inline __cubql_both float fSqrLength_rd(vec_t<T,4> v)
  { return fSquare_rd(v.x)+fSquare_rd(v.y)+fSquare_rd(v.z)+fSquare_rd(v.w); }
  
  
  // ------------------------------------------------------------------
  template<typename T, int D>
  inline __cubql_both float fSqrDistance_rd(vec_t<T,D> a, vec_t<T,D> b)
  { return fSqrLength_rd(a-b); }

  /*! distance of point to box */
  template<typename T, int D>
  inline __cubql_both float fSqrDistance_rd(vec_t<T,D> a, box_t<T,D> b)
  { return fSqrDistance_rd(a,project(b,a)); }

  /*! distance of point to box */
  template<typename T, int D>
  inline __cubql_both float fSqrDistance_rd(box_t<T,D> b, vec_t<T,D> a)
  { return fSqrDistance_rd(a,project(b,a)); }

  /*! box-box distance */
  template<typename T, int D>
  inline __cubql_both float fSqrDistance_rd(box_t<T,D> a, box_t<T,D> b)
  {
    vec_t<T,D> lo = max(a.lower,b.lower);
    vec_t<T,D> hi = min(a.upper,b.upper);
    vec_t<T,D> diff = max(vec_t<T,D>((T)0),lo-hi);
    return fSqrLength_rd(diff);
  }


    /*! compute square distance with round-down where necessary. Use
      this in traversal query-to-node distance checks, because the
      rounding down will mean we'll only ever, if anything,
      _under_estimate the distance to the node, and thus never wrongly
      cull any subtrees) */
    inline __cubql_both
    float fSqrDistance_rd(vec2f point, box2f box)
    {
      vec2f projected = min(max(point,box.lower),box.upper);
      vec2f v = projected - point;
      return dot(v,v);
    }
    /*! compute square distance with round-down where necessary. Use
      this in traversal query-to-node distance checks, because the
      rounding down will mean we'll only ever, if anything,
      _under_estimate the distance to the node, and thus never wrongly
      cull any subtrees) */
    inline __cubql_both
    float fSqrDistance_rd(vec3f point, box3f box)
    {
      vec3f projected = min(max(point,box.lower),box.upper);
      vec3f v = projected - point;
      return dot(v,v);
    }
    /*! compute square distance with round-down where necessary. Use
      this in traversal query-to-node distance checks, because the
      rounding down will mean we'll only ever, if anything,
      _under_estimate the distance to the node, and thus never wrongly
      cull any subtrees) */
    inline __cubql_both
    float fSqrDistance_rd(vec4f point, box4f box)
    {
      vec4f projected = min(max(point,box.lower),box.upper);
      vec4f v = projected - point;
      return dot(v,v);
    }


    /*! compute square distance with round-down where necessary. Use
      this in traversal query-to-node distance checks, because the
      rounding down will mean we'll only ever, if anything,
      _under_estimate the distance to the node, and thus never wrongly
      cull any subtrees) */
    inline __cubql_both
    float fSqrDistance_rd(vec2i point, box2i box)
    {
      vec2i projected = min(max(point,box.lower),box.upper);
      vec2i v = projected - point;
#ifdef __CUDA_ARCH__
      return __ll2float_rd(dot(v,v));
#else
      return host::__ull2float_rd(dot(v,v));
#endif
    }
    /*! compute square distance with round-down where necessary. Use
      this in traversal query-to-node distance checks, because the
      rounding down will mean we'll only ever, if anything,
      _under_estimate the distance to the node, and thus never wrongly
      cull any subtrees) */
    inline __cubql_both
    float fSqrDistance_rd(vec3i point, box3i box)
    {
      vec3i projected = min(max(point,box.lower),box.upper);
      vec3i v = projected - point;
#ifdef __CUDA_ARCH__
      return __ll2float_rd(dot(v,v));
#else
      return host::__ull2float_rd(dot(v,v));
#endif
    }
    /*! compute square distance with round-down where necessary. Use
      this in traversal query-to-node distance checks, because the
      rounding down will mean we'll only ever, if anything,
      _under_estimate the distance to the node, and thus never wrongly
      cull any subtrees) */
    inline __cubql_both
    float fSqrDistance_rd(vec4i point, box4i box)
    {
      vec4i projected = min(max(point,box.lower),box.upper);
      vec4i v = projected - point;
#ifdef __CUDA_ARCH__
      return __ll2float_rd(dot(v,v));
#else
      return host::__ull2float_rd(dot(v,v));
#endif
    }



    /*! compute square distance with round-down where necessary. Use
      this in traversal query-to-node distance checks, because the
      rounding down will mean we'll only ever, if anything,
      _under_estimate the distance to the node, and thus never wrongly
      cull any subtrees) */
    inline __cubql_both
    float fSqrDistance_rd(vec2d point, box2d box)
    {
      vec2d projected = min(max(point,box.lower),box.upper);
      vec2d v = projected - point;
#ifdef __CUDA_ARCH__
      return __ll2float_rd(dot(v,v));
#else
      return host::__udouble2float_rd(dot(v,v));
#endif
    }
    
    /*! compute square distance with round-down where necessary. Use
      this in traversal query-to-node distance checks, because the
      rounding down will mean we'll only ever, if anything,
      _under_estimate the distance to the node, and thus never wrongly
      cull any subtrees) */
    inline __cubql_both
    float fSqrDistance_rd(vec3d point, box3d box)
    {
      vec3d projected = min(max(point,box.lower),box.upper);
      vec3d v = projected - point;
#ifdef __CUDA_ARCH__
      return __double2float_rd(dot(v,v));
#else
      return host::__udouble2float_rd(dot(v,v));
#endif
    }
    
    /*! compute square distance with round-down where necessary. Use
      this in traversal query-to-node distance checks, because the
      rounding down will mean we'll only ever, if anything,
      _under_estimate the distance to the node, and thus never wrongly
      cull any subtrees) */
    inline __cubql_both
    float fSqrDistance_rd(vec4d point, box4d box)
    {
      vec4d projected = min(max(point,box.lower),box.upper);
      vec4d v = projected - point;
#ifdef __CUDA_ARCH__
      return __double2float_rd(dot(v,v));
#else
      return host::__udouble2float_rd(dot(v,v));
#endif
    }

    template<int D>
    inline __cubql_both
    float fSqrDistance_rd(vec_t<long long int, D> a,
                          vec_t<long long int, D> b)
    {
      float sum = 0.f;
      for (int i=0;i<D;i++) {
        long long lo = min(a[i],b[i]);
        long long hi = max(a[i],b[i]);
        unsigned long long diff = hi - lo;
#ifdef __CUDA_ARCH__
        float fDiff = __ll2float_rd(diff);
#else
        float fDiff = host::__ull2float_rd(diff);
#endif
        sum += fDiff*fDiff;
      }
      return sum;
    }
    
  
}
