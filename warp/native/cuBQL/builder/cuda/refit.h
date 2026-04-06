// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cuBQL/builder/cuda/builder_common.h"

namespace cuBQL {
  namespace cuda {

    template<typename T, int D>
    __global__ void
    refit_init(const typename BinaryBVH<T,D>::Node *nodes,
               uint32_t              *refitData,
               int numNodes)
    {
      const int nodeID = threadIdx.x+blockIdx.x*blockDim.x;
      if (nodeID == 1 || nodeID >= numNodes) return;
      if (nodeID < 2)
        refitData[0] = 0;
      const auto &node = nodes[nodeID];
      if (node.admin.count) return;

      refitData[node.admin.offset+0] = nodeID << 1;
      refitData[node.admin.offset+1] = nodeID << 1;
    }
    
    template<typename T, int D>
    __global__
    void refit_run(BinaryBVH<T,D> bvh,
                   uint32_t *refitData,
                   const box_t<T,D> *boxes)
    {
      int nodeID = threadIdx.x+blockIdx.x*blockDim.x;
      if (nodeID == 1 || nodeID >= bvh.numNodes) return;
      
      typename BinaryBVH<T,D>::Node *node = &bvh.nodes[nodeID];
      if (node->admin.count == 0)
        // this is a inner node - exit
        return;

      box_t<T,D> bounds; bounds.set_empty();
      for (int i=0;i<node->admin.count;i++) {
        const box_t<T,D> primBox = boxes[bvh.primIDs[node->admin.offset+i]];
        bounds.lower = min(bounds.lower,primBox.lower);
        bounds.upper = max(bounds.upper,primBox.upper);
      }

      int parentID = (refitData[nodeID] >> 1);
      while (true) {
        node->bounds = bounds;
        __threadfence();
        if (node == bvh.nodes)
          break;

        uint32_t refitBits = atomicAdd(&refitData[parentID],1u);
        if ((refitBits & 1) == 0)
          // we're the first one - let other one do it
          break;

        nodeID   = parentID;
        node     = &bvh.nodes[parentID];
        parentID = (refitBits >> 1);
        
        typename BinaryBVH<T,D>::Node l = bvh.nodes[node->admin.offset+0];
        typename BinaryBVH<T,D>::Node r = bvh.nodes[node->admin.offset+1];
        bounds.lower = min(l.bounds.lower,r.bounds.lower);
        bounds.upper = max(l.bounds.upper,r.bounds.upper);
      }
    }

    template<typename T, int D>
    void refit(BinaryBVH<T,D>    &bvh,
               const box_t<T,D>  *boxes,
               cudaStream_t       s,
               GpuMemoryResource &memResource)
    {
      int numNodes = bvh.numNodes;
      
      uint32_t *refitData = 0;
      memResource.malloc((void**)&refitData,numNodes*sizeof(*refitData),s);
      
      refit_init<T,D><<<divRoundUp(numNodes,1024),1024,0,s>>>
        (bvh.nodes,refitData,numNodes);
      refit_run<<<divRoundUp(numNodes,32),32,0,s>>>
        (bvh,refitData,boxes);
      memResource.free((void*)refitData,s);
      // we're not syncing here - let APP do that
    }
    
  } // ::cuBQL::gpuBuilder_impl
} // ::cuBQL
