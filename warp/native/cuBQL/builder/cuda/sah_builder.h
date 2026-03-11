// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cuBQL/builder/cuda/sm_builder.h"

namespace cuBQL {
  namespace sahBuilder_impl {
    using ::cuBQL::gpuBuilder_impl::AtomicBox;
    using ::cuBQL::gpuBuilder_impl::PrimState;
    using ::cuBQL::gpuBuilder_impl::NodeState;
    using ::cuBQL::gpuBuilder_impl::BuildState;
    using ::cuBQL::gpuBuilder_impl::OPEN_NODE;
    using ::cuBQL::gpuBuilder_impl::DONE_NODE;
    using ::cuBQL::gpuBuilder_impl::OPEN_BRANCH;
    using ::cuBQL::gpuBuilder_impl::_ALLOC;
    using ::cuBQL::gpuBuilder_impl::_FREE;

    template<typename box_t>
    struct CUBQL_ALIGN(16) TempNode {
      union {
        struct {
          AtomicBox<box_t> centBounds;
          uint32_t         count;
          uint32_t         unused;
        } openBranch;
        struct {
          AtomicBox<box_t> centBounds;
          uint32_t offset;
#if 1
          int8_t dim;
          int8_t bin;
          int32_t atomicSideDeciderBits;
#else
          int8_t   dim;
          int8_t   bin;
#endif
        } openNode;
        struct {
          uint32_t offset;
          uint32_t count;
          uint32_t unused[2];
        } doneNode;
      };
    };

    template<typename box_t>
    struct SAHBins {
      enum { numBins = 16 };
      struct {
        struct CUBQL_ALIGN(16) {
          AtomicBox<box_t> bounds;
          int   count;
        } bins[numBins];
      } dims[3];
    };

    template<typename box_t>
    inline __device__
    void evaluateSAH(int &splitDim,
                     int &splitBin,
                     const SAHBins<box_t> &sah,
                     int maxAllowedLeafSize)
    {
      float bestCost = CUBQL_INF;
      
      float rAreas[sah.numBins];
      for (int d=0;d<3;d++) {
        box_t box; box.set_empty();
        int   rCount = 0;
        for (int b=sah.numBins-1;b>=0;--b) {
          auto bin = sah.dims[d].bins[b];
          grow(box,bin.bounds.make_box());
          rCount += bin.count;
          rAreas[b] = surfaceArea(box);
        }
        const float leafCost = rAreas[0] * rCount;
        if (leafCost < bestCost && rCount <= maxAllowedLeafSize) {
          bestCost = leafCost;
          splitDim = -1;
        }
        box.set_empty();
        int lCount = 0;
        for (int b=0;b<sah.numBins;b++) {
          float rArea = rAreas[b];
          float lArea = surfaceArea(box);
          if (lCount>0 && rCount>0) {
            float cost = lArea*lCount+rArea*rCount;
            if (cost < bestCost) {
              bestCost = cost;
              splitDim = d;
              splitBin = b;
            }
          }
          auto bin = sah.dims[d].bins[b];
          grow(box,bin.bounds.make_box());
          lCount += bin.count;
          rCount -= bin.count;
        }
      }
    }
    
    template<typename T, int D>
    __global__
    void initState(BuildState *buildState,
                   NodeState  *nodeStates,
                   TempNode<box_t<T,D>>   *nodes)
    {
      // buildState->nodes = nodes;
      buildState->numNodes = 2;
      
      nodeStates[0]             = OPEN_BRANCH;
      nodes[0].openBranch.count = 0;
      nodes[0].openBranch.centBounds.set_empty();

      nodeStates[1]            = DONE_NODE;
      nodes[1].doneNode.offset = 0;
      nodes[1].doneNode.count  = 0;
    }

    template<typename T, int D>
    __global__
    void initPrims(TempNode<box_t<T,D>>    *nodes,
                   PrimState   *primState,
                   const box_t<T,D> *primBoxes,
                   uint32_t     numPrims)
    {
      const int primID = threadIdx.x+blockIdx.x*blockDim.x;
      if (primID >= numPrims) return;
      
      auto &me = primState[primID];
      me.primID = primID;
                                                    
      const box_t<T,D> box = primBoxes[primID];
      if (box.get_lower(0) <= box.get_upper(0)) {
        me.nodeID = 0;
        me.done   = false;
        // this could be made faster by block-reducing ...
        atomicAdd(&nodes[0].openBranch.count,1);
        atomic_grow(nodes[0].openBranch.centBounds,box.center());
      } else {
        me.nodeID = (uint32_t)-1;
        me.done   = true;
      }
    }

    template<typename T, int D>
    __global__
    void binPrims(SAHBins<box_t<T,D>>          *sahBins,
                  int               sahNodeBegin,
                  int               sahNodeEnd,
                  TempNode<box_t<T,D>>         *nodes,
                  PrimState        *primState,
                  const box_t<T,D>      *primBoxes,
                  uint32_t          numPrims)
    {
      const int primID = threadIdx.x+blockIdx.x*blockDim.x;
      if (primID >= numPrims) return;

      auto ps = primState[primID];
      if (ps.done) return;

      int nodeID = ps.nodeID;
      if (nodeID < sahNodeBegin || nodeID >= sahNodeEnd)
        return;

      const box_t<T,D> primBox = primBoxes[primID];

      auto &sah = sahBins[nodeID-sahNodeBegin];
      box_t<T,D> centBounds = nodes[nodeID].openBranch.centBounds.make_box();
#pragma unroll 3 
      for (int d=0;d<3;d++) {
        int bin = 0;
        T lo = centBounds.get_lower(d);
        T hi = centBounds.get_upper(d);
        if (hi > lo) {
          T prim_d = T(0.5)*(primBox.get_lower(d)+primBox.get_upper(d));
          T rel
            = (prim_d - centBounds.get_lower(d))
            / (centBounds.get_upper(d)-centBounds.get_lower(d)+1e-20f);
          bin = int(rel*(int)SAHBins<box_t<T,D>>::numBins);
          bin = max(0,min((int)SAHBins<box_t<T,D>>::numBins-1,bin));
          // printf("prim %i in node %i, pos %f %f %f in cent %f %f %f - %f %f %f; dim %i: rel %f bin %i\n",
          //        primID,nodeID,
          //        primBox.lower.x,
          //        primBox.lower.y,
          //        primBox.lower.z,
          //        centBounds.lower.x,
          //        centBounds.lower.y,
          //        centBounds.lower.z,
          //        centBounds.upper.x,
          //        centBounds.upper.y,
          //        centBounds.upper.z,
          //        d,rel,bin);
        }
        auto &myBin = sah.dims[d].bins[bin];
        atomic_grow(myBin.bounds,primBox);
        atomicAdd(&myBin.count,1);
      }
    }
    
    template<typename T, int D>
    __global__
    void closeOpenNodes(BuildState *buildState,
                        NodeState  *nodeStates,
                        TempNode<box_t<T,D>>   *nodes,
                        int numNodes)
    {
      const int nodeID = threadIdx.x+blockIdx.x*blockDim.x;
      if (nodeID >= numNodes)
        return;
      
      NodeState &nodeState = nodeStates[nodeID];
      if (nodeState == DONE_NODE)
        // this node was already closed before
        return;
      if (nodeState == OPEN_NODE) {
        // this node was open in the last pass, can close it.
        nodeState   = DONE_NODE;
        int offset  = nodes[nodeID].openNode.offset;
        auto &done  = nodes[nodeID].doneNode;
        done.count  = 0;
        done.offset = offset;
        return;
      }
      // cannot be anything else...
    }
    
    template<typename T, int D>
    __global__
    void selectSplits(BuildState *buildState,
                      SAHBins<box_t<T,D>>    *sahBins,
                      int         sahNodeBegin,
                      int         sahNodeEnd,
                      NodeState  *nodeStates,
                      TempNode<box_t<T,D>>   *nodes,
                      BuildConfig buildConfig)
    {
      const int nodeID = sahNodeBegin + threadIdx.x+blockIdx.x*blockDim.x;
      if (nodeID == 1) return;
      
      if (nodeID < sahNodeBegin || nodeID >= sahNodeEnd) return;
      
      NodeState &nodeState = nodeStates[nodeID];
      auto in = nodes[nodeID].openBranch;
      auto &sah = sahBins[nodeID-sahNodeBegin];
      int   splitDim = -1;
      int   splitBin;
      if (in.count > buildConfig.makeLeafThreshold) {
        evaluateSAH(splitDim,splitBin,sah,buildConfig.maxAllowedLeafSize);
      }
      bool makeLeaf
        =  (splitDim < 0)
        && (in.count <= buildConfig.maxAllowedLeafSize);
      
      if (makeLeaf) {
        nodeState   = DONE_NODE;
        auto &done  = nodes[nodeID].doneNode;
        done.count  = in.count;

        // set this to max-value, so the prims can later do atomicMin
        // with their position ion the leaf list; this value is
        // greater than any prim position.
        done.offset = (uint32_t)-1;
      } else {
        nodeState = OPEN_NODE;
        
        auto &open      = nodes[nodeID].openNode;
        open.dim = splitDim;
        open.bin = splitBin;
        open.centBounds = in.centBounds;
        open.offset = atomicAdd(&buildState->numNodes,2);
#pragma unroll
        for (int side=0;side<2;side++) {
          const int childID = open.offset+side;
          auto &child = nodes[childID].openBranch;
          child.centBounds.set_empty();
          child.count         = 0;
          nodeStates[childID] = OPEN_BRANCH;
        }
        // printf("#ss node %i making inner, offset %i, dim %i bin %i\n",
        //        nodeID,open.offset,open.dim,open.bin);
      }
    }

    template<typename T, int D>
    __global__
    void updatePrims(NodeState       *nodeStates,
                     TempNode<box_t<T,D>>        *nodes,
                     PrimState       *primStates,
                     const box_t<T,D>     *primBoxes,
                     int numPrims)
    {
      const int primID = threadIdx.x+blockIdx.x*blockDim.x;
      if (primID >= numPrims) return;

      auto &me = primStates[primID];
      if (me.done) return;
      
      auto ns = nodeStates[me.nodeID];
      if (ns == DONE_NODE) {
        // node became a leaf, we're done.
        me.done = true;
        return;
      }

      auto open = nodes[me.nodeID].openNode;
      const box_t<T,D> primBox = primBoxes[me.primID];
      int side = -1;
      if (open.dim == -1) {
        // // use top bit to atomically 'sort' prims left vs right
        // side = atomicAdd(&nodes[me.nodeID].openNode.atomicSideDeciderBits,1u<<31)>>31;
        side = atomicAdd(&nodes[me.nodeID].openNode.atomicSideDeciderBits,1u)&1;
      } else {

        const int d = open.dim;
        T lo = open.centBounds.get_lower(d);
        T hi = open.centBounds.get_upper(d);
      
        T prim_d = T(0.5)*(primBox.get_lower(d)+primBox.get_upper(d));
        T rel
          = (prim_d - lo)
          / (hi - lo + 1e-20f);
        int prim_bin = int(rel*(int)SAHBins<box_t<T,D>>::numBins);
        prim_bin = max(0,min((int)SAHBins<box_t<T,D>>::numBins-1,prim_bin));
      
        side = (prim_bin >= open.bin);
      }
      int newNodeID = open.offset+side;
      auto &myBranch = nodes[newNodeID].openBranch;
      atomicAdd(&myBranch.count,1);
      atomic_grow(myBranch.centBounds,primBox.center());
      me.nodeID = newNodeID;
    }

    /* given a sorted list of {nodeID,primID} pairs, this kernel does
       two things: a) it extracts the 'primID's and puts them into the
       bvh's primIDs[] array; and b) it writes, for each leaf nod ein
       the nodes[] array, the node.offset value to point to the first
       of this nodes' items in that bvh.primIDs[] list. */
    template<typename T, int D>
    __global__
    void writePrimsAndLeafOffsets(TempNode<box_t<T,D>>        *nodes,
                                  uint32_t        *bvhItemList,
                                  PrimState       *primStates,
                                  int              numPrims)
    {
      const int offset = threadIdx.x+blockIdx.x*blockDim.x;
      if (offset >= numPrims) return;

      auto &ps = primStates[offset];
      bvhItemList[offset] = ps.primID;
      
      if ((int)ps.nodeID < 0)
        /* invalid prim, just skip here */
        return;
      auto &node = nodes[ps.nodeID];
      atomicMin(&node.doneNode.offset,offset);
    }

    template<typename T, int D>
    __global__
    void clearBins(SAHBins<box_t<T,D>> *sahBins, int numActive)
    {
      const int tid = threadIdx.x+blockIdx.x*blockDim.x;
      if (tid >= numActive) return;

      for (int d=0;d<3;d++)
        for (int b=0;b<SAHBins<box_t<T,D>>::numBins;b++) {
          auto &mine = sahBins[tid].dims[d].bins[b];
          mine.count = 0;
          mine.bounds.set_empty();
        }
    }
    
    /* writes main phase's temp nodes into final bvh.nodes[]
       layout. actual bounds of that will NOT yet bewritten */
    template<typename T, int D>
    __global__
    void writeNodes(typename BinaryBVH<T,D>::Node *finalNodes,
                    TempNode<box_t<T,D>>  *tempNodes,
                    int        numNodes)
    {
      const int nodeID = threadIdx.x+blockIdx.x*blockDim.x;
      if (nodeID >= numNodes) return;

      finalNodes[nodeID].admin.offset = tempNodes[nodeID].doneNode.offset;
      finalNodes[nodeID].admin.count  = tempNodes[nodeID].doneNode.count;
    }

    template<typename T, int D>
    inline void sahBuilder(BinaryBVH<T,D>  &bvh,
                    const box_t<T,D> *boxes,
                    uint32_t     _numPrims,
                    BuildConfig  buildConfig,
                    cudaStream_t s,
                    GpuMemoryResource& memResource)
    { throw std::runtime_error("cuBQL::cuda::sahBuilder() only makes sense for float3 data"); }

    template<typename T>
    inline void real3_sahBuilder(BinaryBVH<T,3>  &bvh,
                          const box_t<T,3> *boxes,
                          uint32_t     _numPrims,
                          BuildConfig  buildConfig,
                          cudaStream_t s,
                          GpuMemoryResource& memResource)
    {
      enum { D=3 };
      // ==================================================================
      // do build on temp nodes
      // ==================================================================
      int         numPrims   = _numPrims;
      TempNode<box_t<T,D>>   *tempNodes  = 0;
      NodeState  *nodeStates = 0;
      PrimState  *primStates = 0;
      BuildState *buildState = 0;
      SAHBins<box_t<T,D>>    *sahBins    = 0;
      int maxActiveSAHs = 4+numPrims/(8*SAHBins<box_t<T,D>>::numBins);
      _ALLOC(tempNodes,2*numPrims,s,memResource);
      _ALLOC(nodeStates,2*numPrims,s,memResource);
      _ALLOC(primStates,numPrims,s,memResource);
      _ALLOC(buildState,1,s,memResource);
      _ALLOC(sahBins,maxActiveSAHs,s,memResource);
      
      initState<T,D>
        <<<1,1,0,s>>>(buildState,
                      nodeStates,
                      tempNodes);
      initPrims<T,D>
        <<<divRoundUp(numPrims,1024),1024,0,s>>>
        (tempNodes,
         primStates,boxes,numPrims);
      
      int numDone = 0;
      int numNodes = 0;

      while (true) {
        CUBQL_CUDA_CALL(MemcpyAsync(&numNodes,&buildState->numNodes,
                                    sizeof(numNodes),cudaMemcpyDeviceToHost,s));
        CUBQL_CUDA_CALL(StreamSynchronize(s));
        if (numNodes == numDone)
          break;

        // close all nodes that might still be open in last round
        if (numDone > 0)
          closeOpenNodes<T,D><<<divRoundUp(numDone,1024),1024,0,s>>>
            (buildState,nodeStates,tempNodes,numDone);

        // compute which nodes (by defintion, at the of the array) are
        // currently open and need binning/sah plane selection
        const int openBegin = numDone;
        const int openEnd   = numNodes;

        // go over these nodes in blocks of 'maxActiveSAH' nodes
        // (because that's all we have SAH bin storage for), and do
        // these sah bin-and-select steps
        for (int sahBegin=openBegin;sahBegin<openEnd;sahBegin+=maxActiveSAHs) {
          const int sahEnd = std::min(sahBegin+maxActiveSAHs,openEnd);
          const int numSAH = sahEnd-sahBegin;

          // clear as many of our current set of bins as we might need.
          clearBins<T,D><<<divRoundUp(numSAH,32),32,0,s>>>
            (sahBins,numSAH);

          // bin all prims into those bins; note this will
          // automatically do an immediate return/no-op for all prims
          // that are not in an yof the currently processed nodes.
          binPrims<T,D><<<divRoundUp(numPrims,128),128,0,s>>>
            (sahBins,sahBegin,sahEnd,
             tempNodes,
             primStates,boxes,numPrims);

          // now that we have SAH bin information for all those nodes,
          // go over those active nodes and select their split plane
          // (or make them into a leaf)
          selectSplits<T,D><<<divRoundUp(numSAH,32),32,0,s>>>
            (buildState,
             sahBins,sahBegin,sahEnd,
             nodeStates,tempNodes,
             buildConfig);
        }

        // done with this wave; all those nodes we had looked at so
        // far are not done. there's possibly some additional set of
        // active nodes on the device that got created in the last few
        // steps, but we first have to download that counter to know
        // how many those are - and we'll do that in the next step.
        numDone = numNodes;

        // now last step as in the simple algorithm as well: go over
        // all prims, and "move" them to their respect left or right
        // subtree (or mark as done if their node just became a leaf)
        updatePrims<T,D><<<divRoundUp(numPrims,1024),1024,0,s>>>
          (nodeStates,tempNodes,
           primStates,boxes,numPrims);
      }
      // ==================================================================
      // sort {item,nodeID} list
      // ==================================================================
      
      // set up sorting of prims
      uint8_t   *d_temp_storage = NULL;
      size_t     temp_storage_bytes = 0;
      PrimState *sortedPrimStates;
      _ALLOC(sortedPrimStates,numPrims,s,memResource);
      auto rc =
      cub::DeviceRadixSort::SortKeys((void*&)d_temp_storage, temp_storage_bytes,
                                     (uint64_t*)primStates,
                                     (uint64_t*)sortedPrimStates,
                                     numPrims,32,64,s);
      _ALLOC(d_temp_storage,temp_storage_bytes,s,memResource);
      rc =
      cub::DeviceRadixSort::SortKeys((void*&)d_temp_storage, temp_storage_bytes,
                                     (uint64_t*)primStates,
                                     (uint64_t*)sortedPrimStates,
                                     numPrims,32,64,s);
      rc = rc; // ignore 'unused' warning
      CUBQL_CUDA_CALL(StreamSynchronize(s));
      _FREE(d_temp_storage,s,memResource);
      // ==================================================================
      // allocate and write BVH item list, and write offsets of leaf nodes
      // ==================================================================

      bvh.numPrims = numPrims;
      _ALLOC(bvh.primIDs,numPrims,s,memResource);
      writePrimsAndLeafOffsets<T,D><<<divRoundUp(numPrims,1024),1024,0,s>>>
        (tempNodes,bvh.primIDs,sortedPrimStates,numPrims);

      // ==================================================================
      // allocate and write final nodes
      // ==================================================================
      bvh.numNodes = numNodes;
      _ALLOC(bvh.nodes,numNodes,s,memResource);
      writeNodes<T,D><<<divRoundUp(numNodes,1024),1024,0,s>>>
        (bvh.nodes,tempNodes,numNodes);
      CUBQL_CUDA_CALL(StreamSynchronize(s));
      _FREE(sortedPrimStates,s,memResource);
      _FREE(tempNodes,s,memResource);
      _FREE(nodeStates,s,memResource);
      _FREE(primStates,s,memResource);
      _FREE(buildState,s,memResource);
      _FREE(sahBins,s,memResource);

      cuBQL::cuda::refit(bvh,boxes,s,memResource);
    }

    template<>
    inline void sahBuilder(BinaryBVH<float,3>  &bvh,
                    const box_t<float,3> *boxes,
                    uint32_t     _numPrims,
                    BuildConfig  buildConfig,
                    cudaStream_t s,
                    GpuMemoryResource& memResource)
    { real3_sahBuilder<float>(bvh,boxes,_numPrims,buildConfig,s,memResource); }
    
    template<>
    inline void sahBuilder(BinaryBVH<double,3>  &bvh,
                    const box_t<double,3> *boxes,
                    uint32_t     _numPrims,
                    BuildConfig  buildConfig,
                    cudaStream_t s,
                    GpuMemoryResource& memResource)
    { real3_sahBuilder<double>(bvh,boxes,_numPrims,buildConfig,s,memResource); }
  } // ::cuBQL::sahBuilder_impl

  namespace cuda {
    template<typename T, int D>
    inline void sahBuilder(BinaryBVH<T,D>    &bvh,
                    const box_t<T,D>  *boxes,
                    uint32_t           numPrims,
                    BuildConfig        buildConfig,
                    cudaStream_t       s,
                    GpuMemoryResource &memResource)
    {
      ::cuBQL::sahBuilder_impl::sahBuilder(bvh,boxes,numPrims,buildConfig,s,memResource);
    }
  } // ::cuBQL::cuda
} // ::cuBQL

