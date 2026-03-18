// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cuBQL/builder/cuda/builder_common.h"

namespace cuBQL {
  namespace gpuBuilder_impl {
    
    struct CollapseInfo {
      // careful: 'isWideRoot' and ''binaryRoot' get written to in
      // parallel by differnet threads; they must be in different atomic
      // words.
      struct {
        int32_t  parent:31;
        uint32_t isWideRoot:1;
      };
      /*! for *wide* nodes: the ID of the binary node that is the root
          of the treelet that this node maps to */
      int32_t binaryRoot;
      
      /*! for *binary* nodes that re treelet root nodes: the ID of the
        wide node that it maps to */
      int     wideNodeID;
    };
  
    template<typename T, int D>
    __global__
    void collapseInit(int *d_numWideNodes,
                      CollapseInfo *d_infos,
                      BinaryBVH<T,D> bvh)
    {
      int tid = threadIdx.x+blockIdx.x*blockDim.x;
      if (tid >= bvh.numNodes) return;

      if (tid == 0) {
        *d_numWideNodes  =  1;
        d_infos[0].parent = -1;
        d_infos[0].isWideRoot = 1;
        d_infos[0].wideNodeID = 0;
        d_infos[0].binaryRoot = -1;
      }

      auto &node = bvh.nodes[tid];
      if (node.admin.count > 0)
        // leaf node
        return;
    
      // _could_ write this as a int4 ... we know it'll have to be
      // 128-bit aligned
      d_infos[node.admin.offset+0].isWideRoot = 0;
      d_infos[node.admin.offset+0].parent = tid;
      d_infos[node.admin.offset+0].binaryRoot = -1;
      d_infos[node.admin.offset+1].isWideRoot = 0;
      d_infos[node.admin.offset+1].parent = tid;
      d_infos[node.admin.offset+1].binaryRoot = -1;
    }

    template<typename T, int D, int N>
    __global__
    void collapseSummarize(int *d_numWideNodes,
                           CollapseInfo *d_infos,
                           BinaryBVH<T,D> bvh)
    {
      int tid = threadIdx.x+blockIdx.x*blockDim.x;
      if (tid >= bvh.numNodes) return;
      if (tid == 1)
        // bvh.node[1] is always unused
        return;
    
      int depth  = 0;
      {
        int nodeID = tid;
        while (nodeID > 0) {
          depth++;
          nodeID = d_infos[nodeID].parent;
        }
      }

      const bool isWideNodeRoot
        =  /* inner node: */
        (bvh.nodes[tid].admin.count == 0)
        && /* on right level*/
        ((depth % (log_of<N>::value)) == 0)

        || /* special case: single-node BVH */
        (bvh.numNodes == 1);

      if (!isWideNodeRoot) 
        return;

      const int wideNodeID
        = (tid == 0)
        ? 0
        : atomicAdd(d_numWideNodes,1);
      d_infos[wideNodeID].binaryRoot = tid;
      d_infos[tid].isWideRoot = true;
      d_infos[tid].wideNodeID = wideNodeID;
    }


    template<typename T, int D, int N>
    __global__
    void collapseExecute(CollapseInfo *d_infos,
                         WideBVH<T,D,N> wideBVH,
                         BinaryBVH<T,D>  binary)
    {
      int tid = threadIdx.x+blockIdx.x*blockDim.x;
      if (tid >= wideBVH.numNodes)
        return;

      int nodeStack[5], *stackPtr = nodeStack;
      int binaryRoot = d_infos[tid].binaryRoot;
      *stackPtr++ = binaryRoot;
      
      typename WideBVH<T,D,N>::Node &target = wideBVH.nodes[tid];
      int numWritten = 0;
      while (stackPtr > nodeStack) {
        int nodeID = *--stackPtr;
        auto &node = binary.nodes[nodeID];
        if ((node.admin.count > 0) ||
            ((nodeID != binaryRoot) && d_infos[nodeID].isWideRoot)) {
          target.children[numWritten].bounds = node.bounds;
          if (node.admin.count) {
            target.children[numWritten].offset = node.admin.offset;
          } else {
            target.children[numWritten].offset = d_infos[nodeID].wideNodeID;
          }
          target.children[numWritten].count  = node.admin.count;
          target.children[numWritten].valid  = 1;
          numWritten++;
        } else {
          *stackPtr++ = node.admin.offset+0;
          *stackPtr++ = node.admin.offset+1;
        }
      }
      while (numWritten < N) {
        target.children[numWritten].bounds.set_empty();
        // lower
        //   = make_float3(+INFINITY,+INFINITY,+INFINITY);
        // target.children[numWritten].bounds.upper
        //   = make_float3(-INFINITY,-INFINITY,-INFINITY);
        target.children[numWritten].offset = (uint32_t)-1;
        target.children[numWritten].count  = (uint32_t)-1;
        target.children[numWritten].valid  = 0;
        ++numWritten;
      }
    }

    template<typename T, int D, int N>
    void gpuBuilder(WideBVH<T,D,N>    &wideBVH,
                    const box_t<T,D>  *boxes,
                    uint32_t           numBoxes,
                    BuildConfig        buildConfig,
                    cudaStream_t       s,
                    GpuMemoryResource &memResource)
    {
      BinaryBVH<T,D> binaryBVH;
      gpuBuilder(binaryBVH,boxes,numBoxes,buildConfig,s,memResource);

      int          *d_numWideNodes;
      CollapseInfo *d_infos;
      _ALLOC(d_numWideNodes,1,s,memResource);
      _ALLOC(d_infos,binaryBVH.numNodes,s,memResource);
      // cudaMemset(d_infos,0,binaryBVH.numNodes*sizeof(*d_infos));
      collapseInit<<<divRoundUp((int)binaryBVH.numNodes,1024),1024,0,s>>>
        (d_numWideNodes,d_infos,binaryBVH);
      collapseSummarize<T,D,N><<<divRoundUp((int)binaryBVH.numNodes,1024),1024,0,s>>>
        (d_numWideNodes,d_infos,binaryBVH);
      CUBQL_CUDA_CALL(StreamSynchronize(s));

      CUBQL_CUDA_CALL(MemcpyAsync(&wideBVH.numNodes,d_numWideNodes,
                                  sizeof(int),cudaMemcpyDefault,s));
      CUBQL_CUDA_CALL(StreamSynchronize(s));
      _ALLOC(wideBVH.nodes,wideBVH.numNodes,s,memResource);

      collapseExecute<<<divRoundUp((int)wideBVH.numNodes,1024),1024,0,s>>>
        (d_infos,wideBVH,binaryBVH);

      wideBVH.numPrims  = binaryBVH.numPrims;
      wideBVH.primIDs   = binaryBVH.primIDs;
      binaryBVH.primIDs = 0;
    
      CUBQL_CUDA_CALL(StreamSynchronize(s));
      _FREE(d_infos,s,memResource);
      _FREE(d_numWideNodes,s,memResource);
      free(binaryBVH,s,memResource);
    }

  } // ::cuBQL::gpuBuilder_impl

  template<typename T, int D, int N>
  void gpuBuilder(WideBVH<T,D,N>    &bvh,
                  const box_t<T,D>  *boxes,
                  uint32_t           numBoxes,
                  BuildConfig        buildConfig,
                  cudaStream_t       s,
                  GpuMemoryResource &memResource)
  {
    gpuBuilder_impl::gpuBuilder(bvh,boxes,numBoxes,buildConfig,s,memResource);
  }

  namespace cuda {
    template<typename T, int D, int N>
    void free(WideBVH<T,D,N>   &bvh,
              cudaStream_t s,
              GpuMemoryResource &memResource)
    {
      CUBQL_CUDA_CALL(StreamSynchronize(s));
      gpuBuilder_impl::_FREE(bvh.primIDs,s,memResource);
      gpuBuilder_impl::_FREE(bvh.nodes,s,memResource);
      // CUBQL_CUDA_CALL(FreeAsync(bvh.primIDs,s));
      // CUBQL_CUDA_CALL(FreeAsync(bvh.nodes,s));
      CUBQL_CUDA_CALL(StreamSynchronize(s));
      bvh.primIDs = 0;
    }
  }    
} // :: cuBQL

