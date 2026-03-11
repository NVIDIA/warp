// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/* this file contains the entire builder; this should never be included directly */
#pragma once

#include "cuBQL/bvh.h"
// #include "cuBQL/builder/cuda.h"
#ifdef __HIPCC__
# include <hipcub/hipcub.hpp>
#else
# include <cub/cub.cuh>
#endif
#include <float.h>
#include <limits.h>

#ifdef __HIPCC__
namespace cub {
  using namespace hipcub;
}
#endif

namespace cuBQL {
  namespace gpuBuilder_impl {

    inline __device__ void atomic_min(int32_t *v, int32_t vv)
    { atomicMin((int *)v,(int)vv); }
    inline __device__ void atomic_min(int64_t *v, int64_t vv)
    { atomicMin((long long *)v,(long long)vv); }
    inline __device__ void atomic_min(uint32_t *v, uint32_t vv)
    { atomicMin((unsigned int *)v,(unsigned int)vv); }
    inline __device__ void atomic_min(uint64_t *v, uint64_t vv)
    { atomicMin((unsigned long long *)v,(unsigned long long)vv); }

    inline __device__ void atomic_max(int32_t *v, int32_t vv)
    { atomicMax((int *)v,(int)vv); }
    inline __device__ void atomic_max(int64_t *v, int64_t vv)
    { atomicMax((long long *)v,(long long)vv); }
    inline __device__ void atomic_max(uint32_t *v, uint32_t vv)
    { atomicMax((unsigned int *)v,(unsigned int)vv); }
    inline __device__ void atomic_max(uint64_t *v, uint64_t vv)
    { atomicMax((unsigned long long *)v,(unsigned long long)vv); }
    
    template<typename T, typename count_t>
    inline void _ALLOC(T *&ptr, count_t count, cudaStream_t s,
                       GpuMemoryResource &mem_resource)
    { mem_resource.malloc((void**)&ptr,count*sizeof(T),s); }
    
    template<typename T>
    inline void _FREE(T *&ptr, cudaStream_t s, GpuMemoryResource &mem_resource)
    { mem_resource.free((void*)ptr,s); ptr = 0; }
    
    typedef enum : int8_t { OPEN_BRANCH, OPEN_NODE, DONE_NODE } NodeState;
    
    template<typename T> inline __device__ T empty_box_lower();
    template<typename T> inline __device__ T empty_box_upper();

    template<> inline __device__
    float empty_box_lower<float>() { return +FLT_MAX; };
    
    template<> inline __device__
    float empty_box_upper<float>() { return -FLT_MAX; };
    
    template<> inline __device__
    double empty_box_lower<double>() { return +DBL_MAX; };
    
    template<> inline __device__
    double empty_box_upper<double>() { return -DBL_MAX; };

    template<> inline __device__
    int empty_box_lower<int>() { return INT_MAX; };
    
    template<> inline __device__
    int empty_box_upper<int>() { return INT_MIN; };

    template<> inline __device__
    int64_t empty_box_lower<int64_t>() { return LLONG_MAX; };
    
    template<> inline __device__
    int64_t empty_box_upper<int64_t>() { return LLONG_MIN; };

    template<typename T> struct int_type_of;
    template<> struct int_type_of<float> { typedef int32_t type; };
    template<> struct int_type_of<double> { typedef int64_t type; };
    template<> struct int_type_of<int32_t> { typedef int32_t type; };
    template<> struct int_type_of<int64_t> { typedef int64_t type; };
    template<> struct int_type_of<uint32_t> { typedef uint32_t type; };
    template<> struct int_type_of<uint64_t> { typedef uint64_t type; };

    template<typename T> inline __device__
    typename int_type_of<T>::type encode(T v);
    
    template<> inline __device__
    int32_t encode(float f)
    {
      const int32_t sign = 0x80000000;
      int32_t bits = __float_as_int(f);
      if (bits & sign) bits ^= 0x7fffffff;
      return bits;
    }
    template<> inline __device__
    int64_t encode(double f)
    {
      const int64_t sign = 0x8000000000000000LL;
      int64_t bits = __double_as_longlong(f);
      if (bits & sign) bits ^= 0x7fffffffffffffffLL;
      return bits;
    }
    template<> inline __device__
    int32_t encode(int32_t bits)
    {
      return bits;
    }
    template<> inline __device__
    int64_t encode(int64_t bits)
    {
      return bits;
    }

    template<typename T> inline __device__
    T decode(int32_t v);
    template<typename T> inline __device__
    T decode(uint32_t v);
    template<typename T> inline __device__
    T decode(int64_t v);
    template<typename T> inline __device__
    T decode(uint64_t v);

    template<> inline __device__
    float decode<float>(int32_t bits)
    {
      const int32_t sign = 0x80000000;
      if (bits & sign) bits ^= 0x7fffffff;
      return __int_as_float(bits);
    }
    template<> inline __device__
    int32_t decode<int32_t>(int32_t bits)
    { return bits; }
    template<> inline __device__
    int64_t decode<int64_t>(int64_t bits)
    { return bits; }

    template<> inline __device__
    double decode<double>(int64_t bits)
    {
      const int64_t sign = 0x8000000000000000LL;
      if (bits & sign) bits ^= 0x7fffffffffffffffLL;
      return __longlong_as_double(bits);
    }
    
    template<typename box_t>
    struct CUBQL_ALIGN(8) AtomicBox {
      using scalar_t = typename box_t::scalar_t;
      inline __device__ bool is_empty() const { return lower[0] > upper[0]; }
      inline __device__ void  set_empty();
      // set_empty, in owl::common-style naming
      inline __device__ void  clear() { set_empty(); }
      inline __device__ scalar_t get_center(int dim) const;
      inline __device__ box_t make_box() const;

      inline __device__ scalar_t get_lower(int dim) const {
        if (box_t::numDims>4) 
          return decode<scalar_t>(lower[dim]);
        else if (box_t::numDims==4) {
          return decode<scalar_t>(dim>1
                                  ?((dim>2)?lower[3]:lower[2])
                                  :((dim  )?lower[1]:lower[0]));
        } else if (box_t::numDims==3) {
          return decode<scalar_t>(dim>1
                                  ?lower[2]
                                  :((dim  )?lower[1]:lower[0]));
        } else
          return decode<scalar_t>(lower[dim]);
      }
      inline __device__ scalar_t get_upper(int dim) const {
        if (box_t::numDims>4) 
          return decode<scalar_t>(upper[dim]);
        else if (box_t::numDims==4) {
          return decode<scalar_t>(dim>1
                                  ?((dim>2)?upper[3]:upper[2])
                                  :((dim  )?upper[1]:upper[0]));
        } else if (box_t::numDims==3)
          return decode<scalar_t>(dim>1
                                  ?upper[2]
                                  :((dim  )?upper[1]:upper[0]));
        else
          return decode<scalar_t>(upper[dim]);
      }
      
      typename int_type_of<scalar_t>::type lower[box_t::numDims];
      typename int_type_of<scalar_t>::type upper[box_t::numDims];
      // int32_t lower[box_t::numDims];
      // int32_t upper[box_t::numDims];
      
      // inline static __device__ int32_t encode(float f);
      // inline static __device__ float   decode(int32_t bits);
    };
    
    template<typename box_t>
    inline __device__ typename AtomicBox<box_t>::scalar_t
    AtomicBox<box_t>::get_center(int dim) const
    {
      return (get_lower(dim)+get_upper(dim))/(AtomicBox<box_t>::scalar_t)2;
      // return 0.5f*(decode(lower[dim])+decode(upper[dim]));
    }
    // template<typename box_t>
    // inline __device__ float AtomicBox<box_t>::get_center(int dim) const
    // {
    //   return 0.5f*(get_lower(dim)+get_upper(dim));
    //   // return 0.5f*(decode(lower[dim])+decode(upper[dim]));
    // }

    template<typename box_t>
    inline __device__ box_t AtomicBox<box_t>::make_box() const
    {
      using scalar_t = typename box_t::scalar_t;
      box_t box;
#pragma unroll
      for (int d=0;d<box_t::numDims;d++) {
        box.lower[d] = decode<scalar_t>(lower[d]);
        box.upper[d] = decode<scalar_t>(upper[d]);
      }
      return box;
    }
    
    // template<typename box_t>
    // inline __device__ int32_t AtomicBox<box_t>::encode(float f)
    // {
    //   const int32_t sign = 0x80000000;
    //   int32_t bits = __float_as_int(f);
    //   if (bits & sign) bits ^= 0x7fffffff;
    //   return bits;
    // }
      
    // template<typename box_t>
    // inline __device__ float AtomicBox<box_t>::decode(int32_t bits)
    // {
    //   const int32_t sign = 0x80000000;
    //   if (bits & sign) bits ^= 0x7fffffff;
    //   return __int_as_float(bits);
    // }
    
    template<typename box_t>
    inline __device__ void AtomicBox<box_t>::set_empty()
    {
      using scalar_t = typename box_t::scalar_t;
#pragma unroll
      for (int d=0;d<box_t::numDims;d++) {
        lower[d] = encode(empty_box_lower<scalar_t>());//encode(+FLT_MAX);
        upper[d] = encode(empty_box_upper<scalar_t>());//encode(-FLT_MAX);
      }
    }

    template<typename box_t> inline __device__
    void atomic_grow(AtomicBox<box_t> &abox, const typename box_t::vec_t &other)
    {
      using scalar_t = typename AtomicBox<box_t>::scalar_t;
#pragma unroll
      for (int d=0;d<box_t::numDims;d++) {
        const typename int_type_of<scalar_t>::type enc
          = //AtomicBox<box_t>::
          encode(other[d]);//get(other,d));
        if (enc < abox.lower[d])
          atomic_min(&abox.lower[d],enc);
        if (enc > abox.upper[d])
          atomic_max(&abox.upper[d],enc);
      }
    } 
    
    template<typename box_t>
    inline __device__ void atomic_grow(AtomicBox<box_t> &abox, const box_t &other)
    {
      using scalar_t = typename AtomicBox<box_t>::scalar_t;
#pragma unroll
      for (int d=0;d<box_t::numDims;d++) {
        const typename int_type_of<scalar_t>::type 
          enc_lower = //AtomicBox<box_t>::
          encode(other.get_lower(d));
        const typename int_type_of<scalar_t>::type 
          enc_upper = //AtomicBox<box_t>::
          encode(other.get_upper(d));
        if (enc_lower < abox.lower[d]) atomic_min(&abox.lower[d],enc_lower);
        if (enc_upper > abox.upper[d]) atomic_max(&abox.upper[d],enc_upper);
      }
    }

    template<typename box_t>
    inline __device__ void atomic_grow(AtomicBox<box_t> &abox, const AtomicBox<box_t> &other)
    {
      using scalar_t = typename AtomicBox<box_t>::scalar_t;
#pragma unroll
      for (int d=0;d<box_t::numDims;d++) {
        const typename int_type_of<scalar_t>::type 
          enc_lower = other.lower[d];
        const typename int_type_of<scalar_t>::type 
          enc_upper = other.upper[d];
        if (enc_lower < abox.lower[d]) atomic_min(&abox.lower[d],enc_lower);
        if (enc_upper > abox.upper[d]) atomic_max(&abox.upper[d],enc_upper);
      }
    }
    
    struct BuildState {
      uint32_t  numNodes;
    };

  } // ::cuBQL::gpuBuilder_impl
} // ::cuBQL

