// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! instantiates the GPU builder(s) */
#define CUBQL_GPU_BUILDER_IMPLEMENTATION 1
#include "cuBQL/bvh.h"
#include "cuBQL/builder/cuda.h"
#include "cuBQL/builder/cuda/radix.h"
#include "cuBQL/builder/cuda/rebinMortonBuilder.h"
#include "cuBQL/builder/cuda/wide_gpu_builder.h"


#define CUBQL_INSTANTIATE_BINARY_BVH(T,D)                               \
  namespace cuBQL {                                                     \
  namespace radixBuilder_impl {                                         \
    template                                                            \
    void build(BinaryBVH<T,D>    &bvh,                                  \
               const box_t<T,D>  *boxes,                                \
               uint32_t           numPrims,                             \
               BuildConfig        buildConfig,                          \
               cudaStream_t       s,                                    \
               GpuMemoryResource &memResource);                         \
  }                                                                     \
  template void gpuBuilder(BinaryBVH<T,D>    &bvh,                      \
                           const box_t<T,D>  *boxes,                    \
                           uint32_t           numBoxes,                 \
                           BuildConfig        buildConfig,              \
                           cudaStream_t       s,                        \
                           GpuMemoryResource &mem_resource);            \
  namespace cuda {                                                      \
    template                                                            \
    void radixBuilder<T,D>(BinaryBVH<T,D>    &bvh,                      \
                           const box_t<T,D>  *boxes,                    \
                           uint32_t           numBoxes,                 \
                           BuildConfig        buildConfig,              \
                           cudaStream_t       s,                        \
                           GpuMemoryResource &mem_resource);            \
    template                                                            \
    void rebinRadixBuilder<T,D>(BinaryBVH<T,D>    &bvh,                 \
                                const box_t<T,D>  *boxes,               \
                                uint32_t           numBoxes,            \
                                BuildConfig        buildConfig,         \
                                cudaStream_t       s,                   \
                                GpuMemoryResource &mem_resource);       \
    template                                                            \
    void sahBuilder<T,D>(BinaryBVH<T,D>    &bvh,                        \
                         const box_t<T,D>  *boxes,                      \
                         uint32_t           numBoxes,                   \
                         BuildConfig        buildConfig,                \
                         cudaStream_t       s,                          \
                         GpuMemoryResource &mem_resource);              \
    template                                                            \
    void free(BinaryBVH<T,D>    &bvh,                                   \
              cudaStream_t       s,                                     \
              GpuMemoryResource &mem_resource);                         \
  }                                                                     \
  }                                                                     \
  
#define CUBQL_INSTANTIATE_WIDE_BVH(T,D,N)                       \
  namespace cuBQL {                                             \
    template void gpuBuilder(WideBVH<T,D,N>    &bvh,            \
                             const box_t<T,D>  *boxes,          \
                             uint32_t           numBoxes,       \
                             BuildConfig        buildConfig,    \
                             cudaStream_t       s,              \
                             GpuMemoryResource &mem_resource);  \
    namespace cuda {                                            \
      template void free(WideBVH<T,D,N>  &bvh,                  \
                         cudaStream_t s,                        \
                         GpuMemoryResource& mem_resource);      \
    }                                                           \
  }


// CUBQL_INSTANTIATE_BINARY_BVH(CUBQL_INSTANTIATE_T,CUBQL_INSTANTIATE_D)

#ifdef CUBQL_INSTANTIATE_T
// instantiate an explict type and dimension
CUBQL_INSTANTIATE_BINARY_BVH(CUBQL_INSTANTIATE_T,CUBQL_INSTANTIATE_D)
CUBQL_INSTANTIATE_WIDE_BVH(CUBQL_INSTANTIATE_T,CUBQL_INSTANTIATE_D,4)
CUBQL_INSTANTIATE_WIDE_BVH(CUBQL_INSTANTIATE_T,CUBQL_INSTANTIATE_D,8)
CUBQL_INSTANTIATE_WIDE_BVH(CUBQL_INSTANTIATE_T,CUBQL_INSTANTIATE_D,16)
#else
// default instantiation(s) for float3 only
CUBQL_INSTANTIATE_BINARY_BVH(float,3)
CUBQL_INSTANTIATE_WIDE_BVH(float,3,4)
CUBQL_INSTANTIATE_WIDE_BVH(float,3,8)
CUBQL_INSTANTIATE_WIDE_BVH(float,3,16)
#endif
  
 
