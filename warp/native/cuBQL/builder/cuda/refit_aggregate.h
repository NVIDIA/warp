// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cuBQL/builder/cuda/builder_common.h"
#include "cuBQL/builder/cuda/refit.h"

namespace cuBQL {
  namespace cuda {

    // ------------------------------------------------------------------
    // INTERFACE
    // ------------------------------------------------------------------
    
    template<
      typename T,
      int D,
      typename AggregateNodeData
      // ,
      // typename AggregateFct
      >
    void refit_aggregate(BinaryBVH<T,D> bvh,
                         AggregateNodeData *d_aggregateNodeData,
                         void (*aggregateFct)(BinaryBVH<T,D>,
                                              AggregateNodeData[],
                                              int),
                         cudaStream_t       s =0,
                         GpuMemoryResource &memResource
                         =defaultGpuMemResource());
    
    template<typename T, int D,
             typename AggregateNodeData>
    __global__
    void refit_aggregate_run(BinaryBVH<T,D> bvh,
                             AggregateNodeData *aggregateNodeData,
                         void (*aggregateFct)(BinaryBVH<T,D>,
                                              AggregateNodeData[],
                                              int),
                             uint32_t *refitData)
    {
      int nodeID = threadIdx.x+blockIdx.x*blockDim.x;
      if (nodeID == 1 || nodeID >= bvh.numNodes) return;
      
      typename BinaryBVH<T,D>::Node *node = &bvh.nodes[nodeID];
      if (node->admin.count == 0)
        // this is a inner node - exit
        return;
      
      int parentID = (refitData[nodeID] >> 1);
      while (true) {
        aggregateFct(bvh,aggregateNodeData,nodeID);
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
      }
    }

    
    
    // ------------------------------------------------------------------
    // IMPLEMENTATION
    // ------------------------------------------------------------------
    template<
      typename T,
      int D,
      typename AggregateNodeData>
    void refit_aggregate(BinaryBVH<T,D> bvh,
                         AggregateNodeData *d_aggregateNodeData,
                         void (*aggregateFct)(BinaryBVH<T,D>,
                                              AggregateNodeData[],
                                              int),
                         cudaStream_t       s,
                         GpuMemoryResource &memResource)
    {
      int numNodes = bvh.numNodes;
      
      uint32_t *refitData = 0;
      memResource.malloc((void**)&refitData,numNodes*sizeof(*refitData),s);
      refit_init<T,D><<<divRoundUp(numNodes,1024),1024,0,s>>>
        (bvh.nodes,refitData,numNodes);
      refit_aggregate_run<<<divRoundUp(numNodes,32),32,0,s>>>
        (bvh,d_aggregateNodeData,aggregateFct,refitData);
      memResource.free((void*)refitData,s);
      // we're not syncing here - let APP do that
    }
  }
}
