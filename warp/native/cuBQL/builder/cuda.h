// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef __HIPCC__
# include <hip/hip_runtime.h>
#else
# include <cuda_runtime_api.h>
#endif
# include "cuBQL/math/box.h"
# include <mutex>

namespace cuBQL {
  // ------------------------------------------------------------------
  /*! defines a 'memory resource' that can be used for allocating gpu
    memory; this allows the user to switch between usign
    cudaMallocAsync (where avialable) vs regular cudaMalloc (where
    not), or to use their own memory pool, to use managed memory,
    etc. All memory allocatoins done during construction will use
    the memory resource passed to the respective build function. */
  struct GpuMemoryResource {
    virtual void malloc(void** ptr, size_t size, cudaStream_t s) = 0;
    virtual void free(void* ptr, cudaStream_t s) = 0;
  };
  
  struct ManagedMemMemoryResource : public GpuMemoryResource {
    void malloc(void** ptr, size_t size, cudaStream_t s) override
    {
      CUBQL_CUDA_CALL(StreamSynchronize(s));
      CUBQL_CUDA_CALL(MallocManaged(ptr,size));
    }
    void free(void* ptr, cudaStream_t s) override
    {
      CUBQL_CUDA_CALL(StreamSynchronize(s));
      CUBQL_CUDA_CALL(Free(ptr));
    }
  };

  /*! allocator that uses regular cudaMalloc to allocate memory. can
    be a bit slower than using aync mallocs, but in case of
    multi-gpu system is easier to control which device the memory
    gets allocated on */
  struct DeviceMemoryResource final : GpuMemoryResource {
    DeviceMemoryResource()
    {}
    void malloc(void** ptr, size_t size, cudaStream_t s) override {
      CUBQL_CUDA_CALL(StreamSynchronize(s));
      CUBQL_CUDA_CALL(Malloc(ptr, size));
    }
    void free(void* ptr, cudaStream_t s) override
    {
      CUBQL_CUDA_CALL(Free(ptr));
    }
  };
  
// hipMallocAsync and the hipMemPool API are available on ROCm; on the HIP
// toolchain CUDART_VERSION is undefined (0), so select the async path
// explicitly there. The CUDA arm keeps its >= 11020 floor unchanged.
#if defined(__HIPCC__) || CUDART_VERSION >= 11020
  /* Allocator that uses cudaMallocAsync to allocate memory. This can
     be much faster than cudaMalloc because it doesn't require a
     device sync for each malloc; but .. CAREFUL: to get memory to be
     allocated on a given GPU it is NOT sufficient to just do a
     cudaSetDevice() to the given GPU - async memory gets allocated on
     the GPU to which the given stream passed to the builder is
     associated. If you pass the default stream 0, your mallocs will
     always happen on the first device! */
  struct AsyncGpuMemoryResource final : GpuMemoryResource {
    AsyncGpuMemoryResource()
    {
      // Configure the default memory pool on all visible devices the first
      // time an instance of this object is created. The operation is thread-safe.
      static std::once_flag s_initializedFlag;

      std::call_once(s_initializedFlag, [this]() {
        CUBQL_CUDA_CALL(GetDeviceCount(&s_numDevices));

        for (int iDevice = 0; iDevice < s_numDevices; iDevice++) {
          cudaMemPool_t mempool;
          CUBQL_CUDA_CALL(DeviceGetDefaultMemPool(&mempool, iDevice));
          uint64_t threshold = UINT64_MAX;
          CUBQL_CUDA_CALL
            (MemPoolSetAttribute(mempool,
                                 cudaMemPoolAttrReleaseThreshold,
                                 &threshold));
        }
      } );
    }
    void malloc(void** ptr, size_t size, cudaStream_t s) override {
#ifndef NDEBUG
      if (s_numDevices > 1 && s == 0)
        std::cerr << "@cuBQL: warning; async memory allocator used with default stream."
                  << std::endl;
#endif
      CUBQL_CUDA_CALL(MallocAsync(ptr, size, s));
    }
    void free(void* ptr, cudaStream_t s) override
    {
      CUBQL_CUDA_CALL(FreeAsync(ptr, s));
    }

  private:
    static inline int s_numDevices = 0;
  };

  /* by default let's use cuda malloc async, which is much better and
     faster than regular malloc; but that's available on cuda 11, so
     let's add a fall back for older cuda's, too */
  inline GpuMemoryResource &defaultGpuMemResource() {
    static AsyncGpuMemoryResource memResource;
    return memResource;
  }
#else
  inline GpuMemoryResource &defaultGpuMemResource() {
    static ManagedMemMemoryResource memResource;
    return memResource;
  }
#endif

  // ------------------------------------------------------------------  
  /*! builds a wide-bvh over a given set of primitive bounding boxes.

    builder runs on the GPU; boxes[] must be a device-readable array
    (managed or device mem); bvh arrays will be allocated in device mem 

    input primitives may be marked as "inactive/invalid" by using a
    bounding box whose lower/upper coordinates are inverted; such
    primitmives will be ignored, and will thus neither be visited
    during traversal nor mess up the tree in any way, shape, or form
  */
  template<typename T, int D>
  void gpuBuilder(BinaryBVH<T,D>   &bvh,
                  /*! array of bounding boxes to build BVH over, must
                    be in device memory */
                  const box_t<T,D> *boxes,
                  uint32_t          numBoxes,
                  BuildConfig       buildConfig=BuildConfig(),
                  cudaStream_t      s=0,
                  GpuMemoryResource &memResource=defaultGpuMemResource());
  
  /*! builds a BinaryBVH over the given set of boxes (using the given
    stream), using a simple adaptive spatial median builder (ie,
    each subtree will be split by first computing the bounding box
    of all its contained primitives' spatial centers, then choosing
    a split plane that splits this cntroid bounds in the center,
    along the widest dimension. Leaves will be created once the size
    of a subtree get to or below buildConfig.makeLeafThreshold */
  template<typename /*scalar type*/T, int /*dims*/D, int /*branching factor*/N>
  void gpuBuilder(WideBVH<T,D,N> &bvh,
                  /*! array of bounding boxes to build BVH over, must
                    be in device memory */
                  const box_t<T,D>  *boxes,
                  uint32_t          numBoxes,
                  BuildConfig       buildConfig=BuildConfig(),
                  cudaStream_t      s=0,
                  GpuMemoryResource& memResource=defaultGpuMemResource());


  // ------------------------------------------------------------------
  /*! SAH(surface area heuristic) based builder, only alowed for float3 */
  // ------------------------------------------------------------------
  namespace cuda {
    template<typename T, int D>
    void sahBuilder(BinaryBVH<T,D>    &bvh,
                    const box_t<T,D>  *boxes,
                    uint32_t           numPrims,
                    BuildConfig        buildConfig,
                    cudaStream_t       s=0,
                    GpuMemoryResource &memResource=defaultGpuMemResource());
  
    // ------------------------------------------------------------------
    /*! fast radix/morton builder */
    // ------------------------------------------------------------------
    template<typename T, int D>
    void radixBuilder(BinaryBVH<T,D>    &bvh,
                      const box_t<T,D>  *boxes,
                      uint32_t           numPrims,
                      BuildConfig        buildConfig,
                      cudaStream_t       s=0,
                      GpuMemoryResource &memResource=defaultGpuMemResource());
  
    // ------------------------------------------------------------------
    /*! fast radix/morton builder with automatic rebinning where
      required (better for certain numerically challenging data
      distributions) */
    // ------------------------------------------------------------------
    template<typename T, int D>
    void rebinRadixBuilder(BinaryBVH<T,D>    &bvh,
                           const box_t<T,D>  *boxes,
                           uint32_t           numPrims,
                           BuildConfig        buildConfig,
                           cudaStream_t       s=0,
                           GpuMemoryResource &memResource=defaultGpuMemResource());
  
    // ------------------------------------------------------------------
    /*! refit a previously built boxes to a new set of bounding
        boxes. The order of boxes in the array boxes[] has to
        correspond to that used when building the tree. */
    // ------------------------------------------------------------------
    template<typename T, int D>
    void refit(BinaryBVH<T,D>    &bvh,
               const box_t<T,D>  *boxes,
               cudaStream_t       s=0,
               GpuMemoryResource &memResource=defaultGpuMemResource());
    
    // ------------------------------------------------------------------
    /*! frees the bvh.nodes[] and bvh.primIDs[] memory allocated when
      building the BVH. this assumes that the 'memResource' provided
      here was the same that was used during building */
    // ------------------------------------------------------------------
    template<typename T, int D>
    void free(BinaryBVH<T,D> &bvh,
              cudaStream_t      s=0,
              GpuMemoryResource& memResource=defaultGpuMemResource());
    
    // ------------------------------------------------------------------
    /*! frees the bvh.nodes[] and bvh.primIDs[] memory allocated when
      building the BVH. this assumes that the 'memResource' provided
      here was the same that was used during building */
    // ------------------------------------------------------------------
    template<typename T, int D, int W>
    void free(WideBVH<T,D,W> &bvh,
              cudaStream_t      s=0,
              GpuMemoryResource& memResource=defaultGpuMemResource());
  }

  // ------------------------------------------------------------------
  /*! frees the bvh.nodes[] and bvh.primIDs[] memory allocated when
    building the BVH. this assumes that the 'memResource' provided
    here was the same that was used during building. This function is
    deprecated; it should be replaced by a call to
    cuBQL::cuda::free(..) */
  template<typename T, int D>
  inline void free(BinaryBVH<T,D> &bvh,
                   cudaStream_t      s=0,
                   GpuMemoryResource& memResource=defaultGpuMemResource())
  { cuda::free(bvh,s,memResource); }

  /*! frees the bvh.nodes[] and bvh.primIDs[] memory allocated when
    building the BVH. this assumes that the 'memResource' provided
    here was the same that was used during building. This function is
    deprecated; it should be replaced by a call to
    cuBQL::cuda::free(..) */
  template<typename T, int D, int N>
  inline void free(WideBVH<T,D,N> &bvh,
                   cudaStream_t      s=0,
                   GpuMemoryResource& memResource=defaultGpuMemResource())
  { cuda::free(bvh,s,memResource); }

}

#if defined(__CUDACC__) || defined(__HIPCC__)
# ifdef CUBQL_GPU_BUILDER_IMPLEMENTATION
#  include "cuBQL/builder/cuda/gpu_builder.h"  
#  include "cuBQL/builder/cuda/sm_builder.h"  
#  include "cuBQL/builder/cuda/sah_builder.h"  
#  include "cuBQL/builder/cuda/elh_builder.h"  
#  include "cuBQL/builder/cuda/radix.h"  
#  include "cuBQL/builder/cuda/wide_gpu_builder.h"  
# endif
#endif




