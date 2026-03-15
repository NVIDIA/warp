// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cuBQL/bvh.h"
#include <omp.h>
#include <atomic>

namespace cuBQL {
  namespace omp {
    
    struct Context {
      Context(int gpuID);

      void *alloc(size_t numBytes);
      
      template<typename T>
      void alloc(T *&d_data, size_t Nelements);
      
      template<typename T>
      void alloc_and_upload(T *&d_data, const T *h_data, size_t Nelements);
      
      template<typename T>
      void upload(T *d_data, const T *h_data, size_t Nelements);
      
      template<typename T>
      void alloc_and_upload(T *&d_data, const std::vector<T> &h_vector);

      template<typename T>
      std::vector<T> download_vector(const T *d_data, size_t N);

      template<typename T>
      void download(T &h_value, T *d_value);

      void free(void *);
      
      int gpuID;
      int hostID;
    };
    
    struct Kernel {
      inline int workIdx() const { return _workIdx; }
      int _workIdx;
    };

    inline uint32_t atomicAdd(uint32_t *ptr, uint32_t inc)
    {
#ifdef __NVCOMPILER
      return (uint32_t)((std::atomic<int> *)ptr)->fetch_add((int)inc);
#else
      uint32_t t;
#pragma omp atomic capture
      { t = *ptr; *ptr += inc; }
      // return ((std::atomic<int> *)p_value)->fetch_add(inc);
      return t;
#endif
    }
    

    // ##################################################################
    // IMPLEMENTATION SECTION
    // ##################################################################
    Context::Context(int gpuID)
      : gpuID(gpuID),
        hostID(omp_get_initial_device())
    {
      assert(gpuID < omp_get_num_devices());
      printf("#cuBQL:omp:Context(gpu=%i/%i,host=%i)\n",
             gpuID,omp_get_num_devices(),hostID);
    }

    void *Context::alloc(size_t numBytes)
    { return omp_target_alloc(numBytes,gpuID); }
    
    template<typename T> inline
    void Context::upload(T *d_data,
                         const T *h_data,
                         size_t N)
    {
      assert(d_data);
      omp_target_memcpy(d_data,h_data,N*sizeof(T),
                        0,0,gpuID,hostID);
    }
      
    template<typename T> inline
    void Context::alloc_and_upload(T *&d_data,
                                   const T *h_data,
                                   size_t N)
    {
      printf("target_alloc N %li gpu %i\n",N,gpuID);
      d_data = (T *)omp_target_alloc(N*sizeof(T),gpuID);
      printf("ptr %p\n",d_data);
      upload(d_data,h_data,N);
    }
      
    template<typename T> inline
    void Context::alloc_and_upload(T *&d_data,
                                   const std::vector<T> &h_vector)
    { alloc_and_upload(d_data,h_vector.data(),h_vector.size()); }

    template<typename T>
    std::vector<T> Context::download_vector(const T *d_data, size_t N)
    {
      PRINT(N);
      PRINT(d_data);
      
      std::vector<T> out(N);
      PRINT(out.data());
      PRINT(sizeof(T));
      omp_target_memcpy(out.data(),d_data,N*sizeof(T),
                        0,0,hostID,gpuID);
      return out;
    }

    inline void Context::free(void *ptr)
    { omp_target_free(ptr,gpuID); }

    template<typename T> inline
    void Context::alloc(T *&d_data, size_t N)
    {
      d_data = (T*)omp_target_alloc(N*sizeof(T),gpuID);
    }
      
    // template<typename T> inline
    // void Context::alloc_and_upload(T *&d_data,
    //                                const T *h_data,
    //                                size_t N)
    // {
    //   alloc(d_data,N);
    //   upload(d_data,h_data,N);
    // }
      
    // template<typename T> inline
    // void Context::alloc_and_upload(T *&d_data,
    //                                const std::vector<T> &h_vector)
    // {
    //   alloc(d_data,h_vector.size());
    //   upload(d_data,h_vector);
    // }

    // template<typename T> inline
    // std::vector<T> Context::download_vector(const T *d_data,
    //                                         size_t N)
    // {
    //   std::vector<T> vec(N);
    //   omp_target_memcpy(vec.data(),d_data,N*sizeof(T),
    //                     0,0,hostID,gpuID);
    //   return vec;
    // }

    template<typename T>
    inline void Context::download(T &h_value, T *d_value)
    {
      omp_target_memcpy(&h_value,d_value,sizeof(T),
                        0,0,hostID,gpuID);
    }

    
  } // ::cuBQL::omp
} // ::cuBQL
