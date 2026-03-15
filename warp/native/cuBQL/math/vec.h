// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cuBQL/math/math.h"
#include <type_traits>
#include <limits>
#include "constants.h"
#ifdef __CUDACC__
# include <cuda.h>
#endif

#ifdef _MSC_VER
# define CUBQL_PRAGMA_UNROLL /* nothing */
#else
# define CUBQL_PRAGMA_UNROLL _Pragma("unroll")
#endif

namespace cuBQL {

#ifndef __CUDACC__
  using std::min;
  using std::max;
#endif

#ifndef CUBQL_SUPPORT_CUDA_VECTOR_TYPES
#define CUBQL_SUPPORT_CUDA_VECTOR_TYPES 0
#endif
  
  template<typename /* scalar type */T, int /*! dimension */D>
  struct vec_t_data {
    inline __cubql_both T  operator[](int i) const { return v[i]; }
    inline __cubql_both T &operator[](int i)       { return v[i]; }
    T v[D];
  };

  /*! defines a "invalid" type to allow for using as a parameter where
    no "actual" type for something exists. e.g., a vec<float,4> has
    a cuda equivalent type of float4, but a vec<float,5> does not */
  struct invalid_t {};
  
  /*! defines the "cuda equivalent type" for a given vector type; i.e.,
    a vec3f=vec_t<float,3> has a equivalent cuda built-in type of
    float3. to also allow vec_t's that do not have a cuda
    equivalent, let's also create a 'invalid_t' to be used by
    default */

#ifndef __CUDACC__
  struct float2 { float x, y; };
  struct float3 { float x, y, z; };
  struct CUBQL_ALIGN(16) float4 { float x, y, z, w; };

  struct int2 { int x, y; };
  struct int3 { int x, y, z; };
  struct CUBQL_ALIGN(16) int4 { int x, y, z, w; };

  struct double2 { double x, y; };
  struct double3 { double x, y, z; };
  struct CUBQL_ALIGN(16) double4 { double x, y, z, w; };
#endif
  
#if CUBQL_SUPPORT_CUDA_VECTOR_TYPES
  template<typename T, int D> struct cuda_eq_t { using type = invalid_t; };
  template<> struct cuda_eq_t<float,2> { using type = float2; };
  template<> struct cuda_eq_t<float,3> { using type = float3; };
  template<> struct cuda_eq_t<float,4> { using type = float4; };
  template<> struct cuda_eq_t<int,2> { using type = int2; };
  template<> struct cuda_eq_t<int,3> { using type = int3; };
  template<> struct cuda_eq_t<int,4> { using type = int4; };
  template<> struct cuda_eq_t<double,2> { using type = double2; };
  template<> struct cuda_eq_t<double,3> { using type = double3; };
  template<> struct cuda_eq_t<double,4> { using type = double4; }; 
#endif
  
  template<typename T>
  struct vec_t_data<T,2> {
    inline __cubql_both T         get(int i) const { return i?y:x; }
    inline __cubql_both T  operator[](int i) const { return i?y:x; }
    inline __cubql_both T &operator[](int i)       { return i?y:x; }
    /*! auto-cast to equivalent cuda type */
#if CUBQL_SUPPORT_CUDA_VECTOR_TYPES
    using cuda_t = typename cuda_eq_t<T,2>::type;
    inline __cubql_both operator cuda_t() { cuda_t t; t.x = x; t.y = y; return t; }
#endif
#ifdef __CUDACC__
    /*! allow to typecast that to a dim3, so it can be used as a cuda kernel launch dim */
    inline __cubql_both operator dim3() { dim3 t; t.x = x; t.y = y; t.z = 1; return t; }
#endif    
    T x, y;
  };
  template<typename T>
  struct vec_t_data<T,3> {
    inline __cubql_both T         get(int i) const { return (i==2)?z:(i?y:x); }
    inline __cubql_both T  operator[](int i) const { return (i==2)?z:(i?y:x); }
    inline __cubql_both T &operator[](int i)       { return (i==2)?z:(i?y:x); }
    /*! auto-cast to equivalent cuda type */
#ifdef __CUDACC__
    /*! allow to typecast that to a dim3, so it can be used as a cuda kernel launch dim */
    inline __cubql_both operator dim3() { dim3 t; t.x = x; t.y = y; t.z = z; return t; }
#endif    
#if CUBQL_SUPPORT_CUDA_VECTOR_TYPES
    using cuda_t = typename cuda_eq_t<T,3>::type;
    inline __cubql_both operator cuda_t() { cuda_t t; t.x = x; t.y = y; t.z = z; return t; }
#endif
    T x, y, z;
  };
  template<typename T>
  struct vec_t_data<T,4> {
    inline __cubql_both T         get(int i) const { return (i>=2)?(i==2?z:w):(i?y:x); }
    inline __cubql_both T  operator[](int i) const { return (i>=2)?(i==2?z:w):(i?y:x); }
    inline __cubql_both T &operator[](int i)       { return (i>=2)?(i==2?z:w):(i?y:x); }
    /*! auto-cast to equivalent cuda type */
#if CUBQL_SUPPORT_CUDA_VECTOR_TYPES
    using cuda_t = typename cuda_eq_t<T,4>::type;
    inline __cubql_both operator cuda_t() { cuda_t t; t.x = x; t.y = y; return t; }
#endif
    T x, y, z, w;
  };
  
  template<typename T, int D>
  struct vec_t : public vec_t_data<T,D> {
    enum { numDims = D };
    using scalar_t = T;

    inline __cubql_both vec_t() {}
    inline __cubql_both vec_t(const T &t)
    {
      CUBQL_PRAGMA_UNROLL
        for (int i=0;i<D;i++) (*this)[i] = t;
    }
    inline __cubql_both vec_t(T a, T b) { this->x = a; this->y = b; }
    inline __cubql_both vec_t(const vec_t_data<T,D> &o)
    {
      CUBQL_PRAGMA_UNROLL
        for (int i=0;i<D;i++) (*this)[i] = o[i];
    }
#if CUBQL_SUPPORT_CUDA_VECTOR_TYPES
    using cuda_t = typename cuda_eq_t<T,D>::type;
    inline __cubql_both vec_t(const cuda_t &o)
    {
      CUBQL_PRAGMA_UNROLL
        for (int i=0;i<D;i++) (*this)[i] = (&o.x)[i];
    }
    inline __cubql_both vec_t &operator=(cuda_t v)
    {
      CUBQL_PRAGMA_UNROLL
        for (int i=0;i<numDims;i++) (*this)[i] = (&v.x)[i]; 
      return *this;
    }
#endif
    template<typename OT>
    explicit __cubql_both vec_t(const vec_t_data<OT,D> &o)
    {
      CUBQL_PRAGMA_UNROLL
        for (int i=0;i<D;i++) (*this)[i] = (T)o[i];
    }
    
    
    inline __cubql_both T         get(int i) const { return (*this)[i]; }
    inline static std::string typeName();
  };

  template<typename T>
  struct vec_t<T,3> : public vec_t_data<T,3> {
    enum { numDims = 3 };
    using scalar_t = T;
    using vec_t_data<T,3>::x;
    using vec_t_data<T,3>::y;
    using vec_t_data<T,3>::z;

    inline vec_t() = default;
    inline vec_t(const vec_t &) = default;
    // inline __cubql_both vec_t() {}
    inline __cubql_both vec_t(const T &t) { x = y = z = t; }
    inline __cubql_both vec_t(T x, T y, T z)
    { this->x = x; this->y = y; this->z = z; }
    inline __cubql_both vec_t(const vec_t_data<T,3> &o)
    { this->x = (o.x); this->y = (o.y); this->z = (o.z); }
#if CUBQL_SUPPORT_CUDA_VECTOR_TYPES
    using cuda_t = typename cuda_eq_t<T,3>::type;
    inline __cubql_both vec_t(const cuda_t &o) 
    { this->x = (o.x); this->y = (o.y); this->z = (o.z); }
    inline __cubql_both vec_t &operator=(cuda_t o)
    { this->x = (o.x); this->y = (o.y); this->z = (o.z); return *this; }
#endif

    template<typename OT>
    explicit __cubql_both vec_t(const vec_t_data<OT,3> &o)
    { this->x = T(o.x); this->y = T(o.y); this->z = T(o.z); }
    
    inline static std::string typeName();
  };
  
  template<typename T>
  struct vec_t<T,4> : public vec_t_data<T,4> {
    enum { numDims = 4 };
    using scalar_t = T;
    using vec_t_data<T,4>::x;
    using vec_t_data<T,4>::y;
    using vec_t_data<T,4>::z;
    using vec_t_data<T,4>::w;

    inline __cubql_both vec_t() {}
    inline __cubql_both vec_t(const T &t) { x = y = z = w = t; }
    inline __cubql_both vec_t(T x, T y, T z, T w)
    { this->x = x; this->y = y; this->z = z; this->w = w; }
    inline __cubql_both vec_t(const vec_t_data<T,4> &o)
    { this->x = (o.x); this->y = (o.y); this->z = (o.z); this->w = (o.w); }

    inline __cubql_both vec_t(const vec_t_data<T,3> &o, T w)
    { this->x = (o.x); this->y = (o.y); this->z = (o.z); this->w = w; }

    template<typename OT>
    explicit __cubql_both vec_t(const vec_t_data<OT,4> &o)
    { this->x = T(o.x); this->y = T(o.y); this->z = T(o.z); this->w = T(o.w); }
    
#if CUBQL_SUPPORT_CUDA_VECTOR_TYPES
    using cuda_t = typename cuda_eq_t<T,4>::type;
    inline __cubql_both vec_t(const cuda_t &o) 
    { this->x = (o.x); this->y = (o.y); this->z = (o.z); this->w = (o.w); }
    inline __cubql_both vec_t &operator=(cuda_t o)
    { this->x = (o.x); this->y = (o.y); this->z = (o.z); this->w = (o.w); return *this; }
#endif    
    inline static std::string typeName();
  };
  
  using vec2f = vec_t<float,2>;
  using vec3f = vec_t<float,3>;
  using vec4f = vec_t<float,4>;

  using vec2d = vec_t<double,2>;
  using vec3d = vec_t<double,3>;
  using vec4d = vec_t<double,4>;

  using vec2i = vec_t<int32_t,2>;
  using vec3i = vec_t<int32_t,3>;
  using vec4i = vec_t<int32_t,4>;
  
  using vec2ui = vec_t<uint32_t,2>;
  using vec3ui = vec_t<uint32_t,3>;
  using vec4ui = vec_t<uint32_t,4>;


  using vec2l = vec_t<int64_t,2>;
  using vec3l = vec_t<int64_t,3>;
  using vec4l = vec_t<int64_t,4>;
  
  using vec2ul = vec_t<uint64_t,2>;
  using vec3ul = vec_t<uint64_t,3>;
  using vec4ul = vec_t<uint64_t,4>;


  template<typename T, int D>
  inline __cubql_both
  vec_t<T,D> madd(vec_t<T,D> a, vec_t<T,D> b, vec_t<T,D> c)
  { return a * b + c; }
  
  template<typename T>
  inline __cubql_both vec_t<T,3> cross(const vec_t<T,3> &a, const vec_t<T,3> &b)
  {
    return vec_t<T,3>(a.y*b.z-b.y*a.z,
                      a.z*b.x-b.z*a.x,
                      a.x*b.y-b.x*a.y);
  }

  /*! traits for a vec_t */
  template<typename vec_t>
  struct our_vec_t_traits {
    enum { numDims = vec_t::numDims };
    using scalar_t = typename vec_t::scalar_t;
  };


  /*! vec_traits<T> describe the scalar type and number of dimensions
    of whatever wants to get used as a vector/point type in
    cuBQL. By default cuBQL will use its own vec_t<T,D> for that,
    but this should allow to also describe the traits of external
    types such as CUDA's float3 */
  template<typename T> struct vec_traits : public our_vec_t_traits<T> {};
  
  template<> struct vec_traits<float3> { enum { numDims = 3 }; using scalar_t = float; };



  template<typename vec_t>
  inline __cubql_both vec_t make(typename vec_t::scalar_t v)
  {
    vec_t r;
    CUBQL_PRAGMA_UNROLL
      for (int i=0;i<vec_t::numDims;i++)
        r[i] = v;
    return r;
  }

#if CUBQL_SUPPORT_CUDA_VECTOR_TYPES
  template<typename vec_t>
  inline __cubql_both vec_t make(typename cuda_eq_t<typename vec_t::scalar_t,vec_t::numDims>::type v)
  {
    vec_t r;
    CUBQL_PRAGMA_UNROLL
      for (int i=0;i<vec_t::numDims;i++)
        r[i] = (&v.x)[i];
    return r;
  }
#endif

  template<typename T, int D>
  inline __cubql_both
  vec_t<T,D> operator-(vec_t<T,D> v)
  { return vec_t<T,D>(0)-v; }
  
  template<typename T, int D>
  inline __cubql_both
  vec_t<T,D> operator+(vec_t<T,D> v)
  { return v; }

  template<typename T, int D>
  inline __cubql_both
  vec_t<T,D> &operator-=(vec_t<T,D> &self, vec_t<T,D> v)
  { self = self-v; return self; }
  
  template<typename T, int D>
  inline __cubql_both
  vec_t<T,D> &operator+=(vec_t<T,D> &self, vec_t<T,D> v)
  { self = self+v; return self; }
  

#if CUBQL_SUPPORT_CUDA_VECTOR_TYPES
# define CUBQL_OPERATOR_CUDA_T(long_op, op)      \
    /* cudaVec:vec */                                                   \
    template<typename T, int D>                                         \
    inline __cubql_both                                                 \
    vec_t<T,D> long_op(typename cuda_eq_t<T,D>::type a, vec_t<T,D> b)   \
    {                                                                   \
      vec_t<T,D> r;                                                     \
      CUBQL_PRAGMA_UNROLL                                               \
        for (int i=0;i<D;i++) r[i] = (&a.x)[i] op b[i];                 \
      return r;                                                         \
    }                                                                   \
    /* vec:cudaVec */                                                   \
    template<typename T, int D>                                         \
    inline __cubql_both                                                 \
    vec_t<T,D> long_op(vec_t<T,D> a,  typename cuda_eq_t<T,D>::type b)  \
    {                                                                   \
      vec_t<T,D> r;                                                     \
      CUBQL_PRAGMA_UNROLL                                               \
        for (int i=0;i<D;i++) r[i] = a[i] op (&b.x)[i];                 \
      return r;                                                         \
    }                                                                   
#else
# define CUBQL_OPERATOR_CUDA_T(long_op, op)      /* ignore */
#endif
  
#define CUBQL_OPERATOR(long_op, op)                                     \
  /* vec:vec */                                                         \
  template<typename T, int D>                                           \
  inline __cubql_both                                                   \
  vec_t<T,D> long_op(const vec_t<T,D> &a, const vec_t<T,D> &b)          \
  {                                                                     \
    vec_t<T,D> r;                                                       \
    for (int i=0;i<D;i++) r[i] = a[i] op b[i];                          \
    return r;                                                           \
  }                                                                     \
  template<typename T>                                                  \
  inline __cubql_both                                                   \
  vec_t<T,2> long_op(const vec_t<T,2> &a, const vec_t<T,2> &b)          \
  {                                                                     \
    vec_t<T,2> r;                                                       \
    r.x = a.x op b.x;                                                   \
    r.y = a.y op b.y;                                                   \
    return r;                                                           \
  }                                                                     \
  template<typename T>                                                  \
  inline __cubql_both                                                   \
  vec_t<T,3> long_op(const vec_t<T,3> &a, const vec_t<T,3> &b)          \
  {                                                                     \
    vec_t<T,3> r;                                                       \
    r.x = a.x op b.x;                                                   \
    r.y = a.y op b.y;                                                   \
    r.z = a.z op b.z;                                                   \
    return r;                                                           \
  }                                                                     \
  template<typename T>                                                  \
  inline __cubql_both                                                   \
  vec_t<T,4> long_op(const vec_t<T,4> &a, const vec_t<T,4> &b)          \
  {                                                                     \
    vec_t<T,4> r;                                                       \
    r.x = a.x op b.x;                                                   \
    r.y = a.y op b.y;                                                   \
    r.z = a.z op b.z;                                                   \
    r.w = a.w op b.w;                                                   \
    return r;                                                           \
  }                                                                     \
  /* scalar-vec */                                                      \
  template<typename T, int D>                                           \
  inline __cubql_both                                                   \
  vec_t<T,D> long_op(T a, const vec_t<T,D> &b)                          \
  { return vec_t<T,D>(a) op b; }                                        \
  /* vec:scalar */                                                      \
  template<typename T, int D>                                           \
  inline __cubql_both                                                   \
  vec_t<T,D> long_op(const vec_t<T,D> &a, T b)                          \
  { return a op vec_t<T,D>(b); }                                        \
  CUBQL_OPERATOR_CUDA_T(long_op,op)

  CUBQL_OPERATOR(operator+,+)
  CUBQL_OPERATOR(operator-,-)
  CUBQL_OPERATOR(operator*,*)
  CUBQL_OPERATOR(operator/,/)
#undef CUBQL_OPERATOR

#define CUBQL_EQ_OPERATOR(long_op, op)                  \
  /* vec:vec */                                         \
  template<typename T, int D>                           \
  inline __cubql_both                                   \
  vec_t<T,D> &long_op(vec_t<T,D> &a, vec_t<T,D> b)      \
  {                                                     \
    CUBQL_PRAGMA_UNROLL                                 \
      for (int i=0;i<D;i++) a[i] op b[i];               \
    return a;                                           \
  }                                                     \
  /* vec:scalar */                                      \
  template<typename T, int D>                           \
  inline __cubql_both                                   \
  vec_t<T,D> &long_op(vec_t<T,D> &a, T b)               \
  {                                                     \
    CUBQL_PRAGMA_UNROLL                                 \
      for (int i=0;i<D;i++) a[i] op b;                  \
    return a;                                           \
  }                                                     \
  
  CUBQL_EQ_OPERATOR(operator*=,*=)
#undef CUBQL_EQ_OPERATOR

  // --------- vec << int -------------
  inline __cubql_both vec_t<int,2> operator<<(vec_t<int,2> v, int b)
  { return vec_t<int,2>( v.x << b, v.y << b ); }
  inline __cubql_both vec_t<int,3> operator<<(vec_t<int,3> v, int b)
  { return vec_t<int,3>( v.x << b, v.y << b, v.z << b ); }
  inline __cubql_both vec_t<int,4> operator<<(vec_t<int,4> v, int b)
  { return vec_t<int,4>( v.x << b, v.y << b, v.z << b, v.w << b ); }

  inline __cubql_both vec_t<uint32_t,2> operator<<(vec_t<uint32_t,2> v, int b)
  { return vec_t<uint32_t,2>( v.x << b, v.y << b ); }
  inline __cubql_both vec_t<uint32_t,3> operator<<(vec_t<uint32_t,3> v, int b)
  { return vec_t<uint32_t,3>( v.x << b, v.y << b, v.z << b ); }
  inline __cubql_both vec_t<uint32_t,4> operator<<(vec_t<uint32_t,4> v, int b)
  { return vec_t<uint32_t,4>( v.x << b, v.y << b, v.z << b, v.w << b ); }
  
  inline __cubql_both vec_t<longlong,2> operator<<(vec_t<longlong,2> v, int b)
  { return vec_t<longlong,2>( v.x << b, v.y << b ); }
  inline __cubql_both vec_t<longlong,3> operator<<(vec_t<longlong,3> v, int b)
  { return vec_t<longlong,3>( v.x << b, v.y << b, v.z << b ); }
  inline __cubql_both vec_t<longlong,4> operator<<(vec_t<longlong,4> v, int b)
  { return vec_t<longlong,4>( v.x << b, v.y << b, v.z << b, v.w << b ); }
  
  inline __cubql_both vec_t<uint64_t,2> operator<<(vec_t<uint64_t,2> v, int b)
  { return vec_t<uint64_t,2>( v.x << b, v.y << b ); }
  inline __cubql_both vec_t<uint64_t,3> operator<<(vec_t<uint64_t,3> v, int b)
  { return vec_t<uint64_t,3>( v.x << b, v.y << b, v.z << b ); }
  inline __cubql_both vec_t<uint64_t,4> operator<<(vec_t<uint64_t,4> v, int b)
  { return vec_t<uint64_t,4>( v.x << b, v.y << b, v.z << b, v.w << b ); }
  
  // --------- vec >> int -------------
  inline __cubql_both vec_t<int,2> operator>>(vec_t<int,2> v, int b)
  { return vec_t<int,2>( v.x >> b, v.y >> b ); }
  inline __cubql_both vec_t<int,3> operator>>(vec_t<int,3> v, int b)
  { return vec_t<int,3>( v.x >> b, v.y >> b, v.z >> b ); }
  inline __cubql_both vec_t<int,4> operator>>(vec_t<int,4> v, int b)
  { return vec_t<int,4>( v.x >> b, v.y >> b, v.z >> b, v.w >> b ); }

  inline __cubql_both vec_t<uint32_t,2> operator>>(vec_t<uint32_t,2> v, int b)
  { return vec_t<uint32_t,2>( v.x >> b, v.y >> b ); }
  inline __cubql_both vec_t<uint32_t,3> operator>>(vec_t<uint32_t,3> v, int b)
  { return vec_t<uint32_t,3>( v.x >> b, v.y >> b, v.z >> b ); }
  inline __cubql_both vec_t<uint32_t,4> operator>>(vec_t<uint32_t,4> v, int b)
  { return vec_t<uint32_t,4>( v.x >> b, v.y >> b, v.z >> b, v.w >> b ); }
  
  inline __cubql_both vec_t<longlong,2> operator>>(vec_t<longlong,2> v, int b)
  { return vec_t<longlong,2>( v.x >> b, v.y >> b ); }
  inline __cubql_both vec_t<longlong,3> operator>>(vec_t<longlong,3> v, int b)
  { return vec_t<longlong,3>( v.x >> b, v.y >> b, v.z >> b ); }
  inline __cubql_both vec_t<longlong,4> operator>>(vec_t<longlong,4> v, int b)
  { return vec_t<longlong,4>( v.x >> b, v.y >> b, v.z >> b, v.w >> b ); }
  
  inline __cubql_both vec_t<uint64_t,2> operator>>(vec_t<uint64_t,2> v, int b)
  { return vec_t<uint64_t,2>( v.x >> b, v.y >> b ); }
  inline __cubql_both vec_t<uint64_t,3> operator>>(vec_t<uint64_t,3> v, int b)
  { return vec_t<uint64_t,3>( v.x >> b, v.y >> b, v.z >> b ); }
  inline __cubql_both vec_t<uint64_t,4> operator>>(vec_t<uint64_t,4> v, int b)
  { return vec_t<uint64_t,4>( v.x >> b, v.y >> b, v.z >> b, v.w >> b ); }
  
  inline __cubql_both double abs(double d) {
#ifdef __CUDA_ARCH__
    return ::abs(d);
#else
    return std::abs(d);
#endif
  }
  inline __cubql_both float abs(float d) {
#ifdef __CUDA_ARCH__
    return ::abs(d);
#else
    return std::abs(d);
#endif
  }
  // inline __cubql_both double abs(double d) { return absf(d); }
  
#define CUBQL_UNARY(op)                         \
  template<typename T, int D>                   \
  inline __cubql_both                           \
  vec_t<T,D> op(vec_t<T,D> a)                  \
  {                                             \
    vec_t<T,D> r;                               \
    CUBQL_PRAGMA_UNROLL                         \
      for (int i=0;i<D;i++) r[i] = ::cuBQL:: op(a[i]);   \
    return r;                                   \
  }

  CUBQL_UNARY(rcp)
  CUBQL_UNARY(abs)
#undef CUBQL_FUNCTOR
  
#define CUBQL_BINARY(op)                                \
  template<typename T, int D>                           \
  inline __cubql_both                                   \
  vec_t<T,D> op(vec_t<T,D> a, vec_t<T,D> b)             \
  {                                                     \
    vec_t<T,D> r;                                       \
    CUBQL_PRAGMA_UNROLL                                 \
      for (int i=0;i<D;i++) r[i] = op(a[i],b[i]);       \
    return r;                                           \
  }

  CUBQL_BINARY(min)
  CUBQL_BINARY(max)
#undef CUBQL_FUNCTOR

  template<typename T, int D>
  vec_t<T,D> divRoundUp(vec_t<T,D> a, vec_t<T,D> b)
  {
    vec_t<T,D> r;
    for (int i=0;i<D;i++)
      r[i] = divRoundUp(a[i],b[i]);
    return r;
  }
  
  /*! host-side equivalent(s) of various cuda functions */
  namespace host {
    inline float __ull2float_rd(uint64_t ul) {
      float f = float(ul);
      if ((uint64_t)f > ul)
        f = nextafterf(f,-CUBQL_INF);
      return f;
    }
    inline float __udouble2float_rd(double ud) {
      float f = float(ud);
      if ((uint64_t)f > ud)
        f = nextafterf(f,-CUBQL_INF);
      return f;
    }
  }
  
  template<typename T> struct dot_result_t;
  template<> struct dot_result_t<double> { using type = double; };
  template<> struct dot_result_t<float> { using type = float; };
  template<> struct dot_result_t<int32_t> { using type = int64_t; };
  template<> struct dot_result_t<long long int> { using type = int64_t; };

  template<typename T, int D> inline __cubql_both
  typename dot_result_t<T>::type dot(vec_t<T,D> a, vec_t<T,D> b)
  {
    typename dot_result_t<T>::type result = 0;
    CUBQL_PRAGMA_UNROLL
      for (int i=0;i<D;i++)
        result += a[i]*(typename dot_result_t<T>::type)b[i];
    return result;
  }


  /*! accurate square-length of a vector; due to the 'square' involved
    in computing the distance this may need to change the type from
    int to long, etc - so a bit less rounding issues, but a bit more
    complicated to use with the right typenames */
  template<typename T, int D> inline __cubql_both
  typename dot_result_t<T>::type sqrLength(vec_t<T,D> v)
  {
    return dot(v,v);
  }


  // ------------------------------------------------------------------
  // *square* distance between two points (can always be computed w/o
  // a square root, so makes sense even for non-float types)
  // ------------------------------------------------------------------

  /*! accurate square-distance between two points; due to the 'square'
    involved in computing the distance this may need to change the
    type from int to long, etc - so a bit less rounding issues, but
    a bit more complicated to use with the right typenames */
  template<typename T, int D> inline __cubql_both
  typename dot_result_t<T>::type sqrDistance(vec_t<T,D> a, vec_t<T,D> b)
  {
    return sqrLength(a-b);
  }

  // /*! approximate-conservative square distance between two
  //   points. whatever type the points are, the result will be
  //   returned in floats, including whatever rounding error that might
  //   incur. we will, however, always round downwars, so if this is
  //   used for culling it will, if anything, under-estiamte the
  //   distance to a subtree (and thus, still traverse it) rather than
  //   wrongly skipping it*/
  // template<typename T, int D> inline __cubql_both
  // float fSqrDistance(vec_t<T,D> a, vec_t<T,D> b)
  // {
  //   float sum = 0.f;
  //   CUBQL_PRAGMA_UNROLL
  //     for (int i=0;i<D;i++)
  //       sum += fSqrLength(a[i]-b[i]);
  //   return sum;
  // }

  
  // ------------------------------------------------------------------
  // 'length' of a vector - may only make sense for certain types
  // ------------------------------------------------------------------
  template<int D>
  inline __cubql_both float length(const vec_t<float,D> &v)
  { return sqrtf(dot(v,v)); }
    
  template<int D>
  inline __cubql_both double length(const vec_t<double,D> &v)
  { return sqrt(dot(v,v)); }
    
  
  template<typename T, int D>
  inline std::ostream &operator<<(std::ostream &o,
                                  const vec_t_data<T,D> &v)
  {
    o << "(";
    for (int i=0;i<D;i++) {
      if (i) o << ",";
      o << v[i];
    }
    o << ")";
    return o;
  }



  template<typename /* scalar type */T>
  inline __cubql_both bool operator==(const vec_t_data<T,2> &a,
                                      const vec_t_data<T,2> &b)
  { return a.x==b.x && a.y==b.y; }
  
  template<typename /* scalar type */T>
  inline __cubql_both bool operator==(const vec_t_data<T,3> &a,
                                      const vec_t_data<T,3> &b)
  { return a.x==b.x && a.y==b.y && a.z==b.z; }
  
  template<typename /* scalar type */T>
  inline __cubql_both bool operator==(const vec_t_data<T,4> &a,
                                      const vec_t_data<T,4> &b)
  { return a.x==b.x && a.y==b.y && a.z==b.z && a.w==b.w; }
  
  template<typename /* scalar type */T, int /*! dimensoins */D>
  inline __cubql_both bool operator==(const vec_t_data<T,D> &a,
                                      const vec_t_data<T,D> &b)
  {
#pragma unroll
    for (int i=0;i<D;i++)
      if (a[i] != b[i]) return false;
    return true;
  }
  template<typename /* scalar type */T, int /*! dimensoins */D>
  inline __cubql_both bool operator!=(const vec_t_data<T,D> &a,
                                      const vec_t_data<T,D> &b)
  {
    return !(a == b);
  }
  

  template<typename T>
  inline __cubql_both
  T reduce_max(vec_t<T,2> v) { return max(v.x,v.y); }
    
  template<typename T>
  inline __cubql_both
  T reduce_min(vec_t<T,2> v) { return min(v.x,v.y); }
    

  template<typename T>
  inline __cubql_both
  T reduce_max(vec_t<T,3> v) { return max(max(v.x,v.y),v.z); }
    
  template<typename T>
  inline __cubql_both
  T reduce_min(vec_t<T,3> v) { return min(min(v.x,v.y),v.z); }
    

  template<typename T>
  inline __cubql_both
  T reduce_max(vec_t<T,4> v) { return max(max(v.x,v.y),max(v.z,v.w)); }
    
  template<typename T>
  inline __cubql_both
  T reduce_min(vec_t<T,4> v) { return min(min(v.x,v.y),min(v.z,v.w)); }
    

  template<typename T, int N>
  inline __cubql_both
  vec_t<T,N> normalize(vec_t<T,N> v) { return v * (T(1)/sqrt(dot(v,v))); }


  // ------------------------------------------------------------------
  /*! @{ returns if _any_ component of vector a is lower than its
    corresponding component in vector b */
  template<typename T, int D>
  inline __cubql_both
  bool any_less_than(const vec_t<T,D> &a, const vec_t<T,D> &b)
  {
    for (int i=0;i<D;i++) if (a[i] < b[i]) return true;
    return false;
  }
  
  template<typename T>
  inline __cubql_both
  bool any_less_than(const vec_t<T,2> &a, const vec_t<T,2> &b)
  { return (a.x < b.x) | (a.y < b.y); }

  template<typename T>
  inline __cubql_both
  bool any_less_than(const vec_t<T,3> &a, const vec_t<T,3> &b)
  { return (a.x < b.x) | (a.y < b.y) | (a.z < b.z); }
  
  template<typename T>
  inline __cubql_both
  bool any_less_than(const vec_t<T,4> &a, const vec_t<T,4> &b)
  { return (a.x < b.x) | (a.y < b.y) | (a.z < b.z) | (a.w < b.w); }

  template<typename T, int D>
  std::ostream &operator<<(std::ostream &o, const vec_t<T,D> &v)
  {
    o << "(";
    for (int i=0;i<D;i++) {
      if (i) o << ",";
      o << v[i];
    }
    o << ")";
    return o;
  }

  // ------------------------------------------------------------------
  template<typename T, int D>
  inline int arg_max(vec_t<T,D> v)
  {
    int maxVal = v[0];
    int maxDim = 0;
    for (int i=1;i<D;i++)
      if (v[i] > maxVal) { maxVal = v[i]; maxDim = i; };
    return maxDim;
  }

  template<typename T>
  inline int arg_max(vec_t<T,2> v)
  {
    return v.y > v.x ? 1 : 0;
  }
  template<typename T>
  inline int arg_max(vec_t<T,3> v)
  {
    return v.z > max(v.x,v.y) ? 2 : (v.y > v.x ? 1 : 0);
  }
  template<typename T>
  inline int arg_max(vec_t<T,4> v)
  {
    T mm = max(max(v.x,v.y),max(v.z,v.w));
    int d = 3;
    d = (mm == v.z) ? 2 : d;
    d = (mm == v.y) ? 1 : d;
    d = (mm == v.x) ? 0 : d;
    return d;
  }
  // ------------------------------------------------------------------
  template<typename T>
  inline std::string toString();
  template<> inline std::string toString<float>()    { return "float"; }
  template<> inline std::string toString<int>()      { return "int"; }
  template<> inline std::string toString<double>()   { return "double"; }
  template<> inline std::string toString<longlong>() { return "long"; }

  template<typename T, int D>
  std::string vec_t<T,D>::typeName()
  { return cuBQL::toString<T>()+std::to_string(D); }
  template<typename T>
  std::string vec_t<T,3>::typeName()
  { return cuBQL::toString<T>()+std::to_string(3); }
  template<typename T>
  std::string vec_t<T,4>::typeName()
  { return cuBQL::toString<T>()+std::to_string(4); }

  template<typename T, int D>
  inline T distance(vec_t<T,D> a, vec_t<T,D> b)
  { return sqrtf(dot(b-a,b-a)); }


  template<typename T>
  inline __cubql_both dbgout operator<<(dbgout o, vec_t<T,2> v)
  { o << "(" << v.x << "," << v.y << ")"; return o; }
  template<typename T>
  inline __cubql_both dbgout operator<<(dbgout o, vec_t<T,3> v)
  { o << "(" << v.x << "," << v.y << "," << v.z << ")"; return o; }
  template<typename T>
  inline __cubql_both dbgout operator<<(dbgout o, vec_t<T,4> v)
  { o << "(" << v.x << "," << v.y << "," << v.z << "," << v.w << ")"; return o; }
  
  /*! @} */
  // ------------------------------------------------------------------
  
}

