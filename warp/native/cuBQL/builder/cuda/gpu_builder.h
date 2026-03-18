// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cuBQL/builder/cuda/builder_common.h"
#include "cuBQL/builder/cuda/sm_builder.h"
#include "cuBQL/builder/cuda/sah_builder.h"
#include "cuBQL/builder/cuda/elh_builder.h"

namespace cuBQL {

  template<typename T, int D>
  struct is3f { enum { value = false }; };
  template<>
  struct is3f<float,3> { enum { value = true }; };
  
  template<typename T, int D>
  void gpuBuilder(BinaryBVH<T,D>    &bvh,
                  const box_t<T,D>  *boxes,
                  uint32_t           numBoxes,
                  BuildConfig        buildConfig,
                  cudaStream_t       s,
                  GpuMemoryResource &memResource)
  {
    if (numBoxes == 0) return;

    int devID;
    CUBQL_CUDA_CALL(GetDevice(&devID));
    
    if (buildConfig.buildMethod == BuildConfig::SAH) {
      if (buildConfig.makeLeafThreshold == 0)
        // unless explicitly specified, use default for spatial median
        // builder:
        buildConfig.makeLeafThreshold = 1;
      if (is3f<T,D>::value) {
        /* for D == 3 these typecasts won't do anything; for D != 3
           they'd be invalid, but won't ever happen */
        sahBuilder_impl::sahBuilder((BinaryBVH<float,3>&)bvh,(const box_t<float,3> *)boxes,
                                    numBoxes,buildConfig,s,memResource);
      } else
        throw std::runtime_error("SAH builder not supported for this type of BVH");
    } else if (buildConfig.buildMethod == BuildConfig::ELH) {
      /* edge-length-heurstic; splits based on sum of the lengths of
         the edges of the bounding box - not as good as sah for
         tracing rays, but often somewhat better than spatial median
         for kNN style queries */
      elhBuilder_impl::elhBuilder(bvh,boxes,numBoxes,buildConfig,s,memResource);
    } else {
      if (buildConfig.makeLeafThreshold == 0)
        // unless explicitly specified, use default for spatial median
        // builder:
        buildConfig.makeLeafThreshold = 1;
      gpuBuilder_impl::build(bvh,boxes,numBoxes,buildConfig,s,memResource);
    }
    cuBQL::cuda::refit(bvh,boxes,s,memResource);
  }

  namespace cuda {
    template<typename T, int D>
    void free(BinaryBVH<T,D>    &bvh,
              cudaStream_t       s,
              GpuMemoryResource &memResource)
    {
      CUBQL_CUDA_CALL(StreamSynchronize(s));  // Sync BEFORE freeing
      gpuBuilder_impl::_FREE(bvh.primIDs,s,memResource);
      gpuBuilder_impl::_FREE(bvh.nodes,s,memResource);
      CUBQL_CUDA_CALL(StreamSynchronize(s));  // And AFTER
      bvh.primIDs = 0;
    }
  }
}



