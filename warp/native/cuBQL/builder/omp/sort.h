// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// imported from github.com:ingowald/openmp_target_sort under this license:

// SPDX-FileCopyrightText: Copyright (c) Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <omp.h>

namespace omp {
  namespace bitonic {

#pragma omp declare target
    template<typename key_t>
    inline
    void g_orderSegmentPairs(uint32_t tid,
                             int logSegLen,
                             key_t *const d_values,
                             int numValues)
    {
      uint32_t segLen    = 1<<logSegLen;
      uint32_t pairIdx   = tid>>logSegLen;
      uint32_t pairRank  = tid-(pairIdx<<logSegLen);
      uint32_t pairBegin = 2*(pairIdx<<logSegLen);

      uint32_t r = pairBegin+segLen+pairRank;
      uint32_t l = pairBegin+segLen-1-pairRank;
                        
      if (r >= numValues) return;
        
      key_t lv = d_values[l];
      key_t rv = d_values[r];
      if (rv < lv) {
        d_values[r] = lv;
        d_values[l] = rv;
      }
    }
#pragma omp end declare target
      
    template<typename key_t, typename value_t>
    void g_orderSegmentPairs(uint32_t tid,
                             int logSegLen,
                             key_t *const d_keys,
                             value_t *const d_values,
                             int numValues)
    {
      uint32_t segLen    = 1<<logSegLen;
      uint32_t pairIdx   = tid>>logSegLen;
      uint32_t pairRank  = tid-(pairIdx<<logSegLen);
      uint32_t pairBegin = 2*(pairIdx<<logSegLen);


      uint32_t r = pairBegin+segLen+pairRank;
      uint32_t l = pairBegin+segLen-1-pairRank;
                        
      if (r >= numValues) return;
        
      key_t lk = d_keys[l];
      key_t rk = d_keys[r];
      key_t lv = d_values[l];
      key_t rv = d_values[r];
      if (rk < lk) {
        d_values[r] = lv;
        d_values[l] = rv;
        d_keys[r]   = lk;
        d_keys[l]   = rk;
      }
    }
      
    template<typename key_t>
    void orderSegmentPairs(int logSegLen,
                           key_t *const d_values,
                           int numValues,
                           uint32_t deviceID)
    {
#if 0
#pragma omp target device(deviceID) teams is_device_ptr(d_values) num_teams(128)
      {
        int bs = 
      }
#else
#pragma omp target device(deviceID) is_device_ptr(d_values) nowait
#pragma omp teams distribute parallel for 
      for (int i=0;i<numValues;i++)
        g_orderSegmentPairs(i,logSegLen,
                            d_values,numValues);
#endif
    }
      
    template<typename key_t, typename value_t>
    void orderSegmentPairs(int logSegLen,
                           key_t *const d_keys,
                           value_t *const d_values,
                           int numValues,
                           uint32_t deviceID)
    {
// #pragma omp target nowait device(deviceID) is_device_ptr(d_keys) is_device_ptr(d_values)
// #pragma omp target teams distribute parallel for
#pragma omp target nowait device(deviceID) is_device_ptr(d_keys) is_device_ptr(d_values) 
      for (int i=0;i<numValues;i++)
        g_orderSegmentPairs(i,logSegLen,
                            d_keys,d_values,numValues);
    }
      
    template<typename key_t>
    void sortSegments(int logSegLen,
                      key_t *const d_keys,
                      int numValues,
                      uint32_t deviceID)
    {
      if (logSegLen == 0) return;

      sortSegments(logSegLen-1,d_keys,numValues,deviceID);
      orderSegmentPairs(logSegLen-1,d_keys,numValues,deviceID);
      sortSegments(logSegLen-1,d_keys,numValues,deviceID);
    }
      
    template<typename key_t, typename value_t>
    void sortSegments(int logSegLen,
                      key_t *const d_keys,
                      value_t *const d_values,
                      int numValues,
                      uint32_t deviceID)
    {
      if (logSegLen == 0) return;

      sortSegments(logSegLen-1,d_keys,d_values,numValues,deviceID);
      orderSegmentPairs(logSegLen-1,d_keys,d_values,numValues,deviceID);
      sortSegments(logSegLen-1,d_keys,d_values,numValues,deviceID);
    }
      
    template<typename key_t>
    void sort(key_t *const d_values,
              int numValues,
              uint32_t deviceID)
    {
      uint32_t logSegLen = 0;
      while (1<<logSegLen < numValues)
        logSegLen++;
      sortSegments(logSegLen,d_values,numValues,deviceID);
    }
      
    template<typename key_t, typename value_t>
    void sort(key_t *const d_keys,
              value_t *const d_values,
              int numValues,
              uint32_t deviceID)
    {
      uint32_t logSegLen = 0;
      while ((1<<logSegLen) < numValues)
        logSegLen++;
      sortSegments(logSegLen,d_keys,d_values,numValues,deviceID);
    }
      
  }
  
  template<typename key_t>
  void omp_target_sort(key_t *const d_values,
                       size_t numValues,
                       uint32_t deviceID)
  {
    assert(numValues < (1ull<<31));
    bitonic::sort(d_values,(int)numValues,deviceID);
  }
    
  template<typename key_t, typename value_t>
  void omp_target_sort(key_t *const d_keys,
                       value_t *const d_values,
                       size_t numValues,
                       uint32_t deviceID)
  {
    assert(numValues < (1ull<<31));
    bitonic::sort(d_keys,d_values,(int)numValues,deviceID);
  }
    
}
