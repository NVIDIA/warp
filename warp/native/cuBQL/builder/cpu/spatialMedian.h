// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cuBQL/bvh.h"
#if CUBQL_CPU_BUILDER_IMPLEMENTATION
#include <vector>
#endif

namespace cuBQL {
  namespace cpu {
    // ******************************************************************
    // INTERFACE
    // (which functionality this header file provides)
    // ******************************************************************

    /*! a simple (and currently non parallel) recursive spatial median
        builder */
    template<typename T, int D>
    void spatialMedian(BinaryBVH<T,D>   &bvh,
                       const box_t<T,D> *boxes,
                       uint32_t          numPrims,
                       BuildConfig       buildConfig);

    // ******************************************************************
    // IMPLEMENTATION
    // ******************************************************************
    
#if CUBQL_CPU_BUILDER_IMPLEMENTATION
    namespace spatialMedian_impl {
      struct Topo {
        struct {
          int offset;
          int count;
        } admin;
      };

      inline void makeLeaf(int nodeID, int begin, int end,
                    std::vector<Topo> &topo)
      {
        auto &node = topo[nodeID];
        node.admin.count = end-begin;
        node.admin.offset = begin;
      }
      
      inline int makeInner(int nodeID,
                    std::vector<Topo> &topo)
      {
        int childID = (int)topo.size();
        topo.push_back({});
        topo.push_back({});
        auto &node = topo[nodeID];
        node.admin.count = 0;
        node.admin.offset = childID;
        return childID;
      }
      
      template<typename T, int D>
      void buildRec(int nodeID, int begin, int end,
                    std::vector<Topo> &topo,
                    std::vector<int>  &primIDs,
                    std::vector<int>  &altPrimIDs,
                    const box_t<T,D>  *boxes,
                    BuildConfig        buildConfig)
      {
        if (end-begin <= buildConfig.makeLeafThreshold)
          return makeLeaf(nodeID,begin,end,topo);
        
        using box_t = ::cuBQL::box_t<T,D>;
        
        box_t centBounds;
        for (int i=begin;i<end;i++)
          centBounds.extend(boxes[primIDs[i]].center());
        
        int dim = arg_max(centBounds.size());
        T   pos = centBounds.center()[dim];
        int Nl = 0, Nr = 0;
        for (int i=begin;i<end;i++) {
          int primID = primIDs[i];
          if (boxes[primID].center()[dim] < pos) {
            altPrimIDs[begin + Nl++] = primID;
          } else {
            altPrimIDs[end   - ++Nr] = primID;
          }
        }
        int mid = -1;
        if (Nl && Nr) 
          mid = begin+Nl;
        else if (end - begin <= std::max(1,buildConfig.makeLeafThreshold)/*maxAllowedLeafSize*/)
          return makeLeaf(nodeID,begin,end,topo);
        else
          mid = (begin+end)/2;
        
        for (int i=begin;i<end;i++)
          primIDs[i] = altPrimIDs[i];
        
        int childID = makeInner(nodeID,topo);
        buildRec(childID+0,begin,mid,topo,primIDs,altPrimIDs,boxes,buildConfig);
        buildRec(childID+1,mid,  end,topo,primIDs,altPrimIDs,boxes,buildConfig);
      }
      
      template<typename T, int D>
      void refit(uint64_t nodeID,
                 BinaryBVH<T,D>   &bvh,
                 const box_t<T,D> *boxes)
      {
        auto &node = bvh.nodes[nodeID];
        if (node.admin.count == 0) {
          refit(node.admin.offset+0,bvh,boxes);
          refit(node.admin.offset+1,bvh,boxes);
          node.bounds = box_t<T,D>()
            .including(bvh.nodes[node.admin.offset+0].bounds)
            .including(bvh.nodes[node.admin.offset+1].bounds);
        } else {
          node.bounds.clear();
          for (int i=0;i<node.admin.count;i++)
            node.bounds.extend(boxes[bvh.primIDs[node.admin.offset+i]]);
        }
      }
                         
      template<typename T, int D>
      void spatialMedian(BinaryBVH<T,D>   &bvh,
                         const box_t<T,D> *boxes,
                         int                   numPrims,
                         BuildConfig           buildConfig)
      {
        using box_t = ::cuBQL::box_t<T,D>;
        std::vector<int> primIDs;
        for (int i=0;i<numPrims;i++) {
          box_t box = boxes[i];
          if (box.empty()) continue;
          primIDs.push_back(i);
        }
        std::vector<int>  altPrimIDs(primIDs.size());
        std::vector<Topo> topo(1);
        
        buildRec(0,0,(int)primIDs.size(),
                 topo,primIDs,altPrimIDs,boxes,buildConfig);
        altPrimIDs.clear();
        bvh.primIDs = new uint32_t[primIDs.size()];
        bvh.numPrims = (uint32_t)primIDs.size();
        std::copy(primIDs.begin(),primIDs.end(),bvh.primIDs);
        primIDs.clear();

        bvh.nodes = new typename BinaryBVH<T,D>::Node[topo.size()];
        bvh.numNodes = (uint32_t)topo.size();
        for (int i=0;i<(int)topo.size();i++) {
          bvh.nodes[i].admin.count = topo[i].admin.count;
          bvh.nodes[i].admin.offset = topo[i].admin.offset;
        }
        topo.clear();
        refit(0,bvh,boxes);
      }
    } // spatialMedian_impl
    
    /*! a simple (and currently non parallel) recursive spatial median
      builder */
    template<typename T, int D>
    void spatialMedian(BinaryBVH<T,D>   &bvh,
                       const box_t<T,D> *boxes,
                       uint32_t              numPrims,
                       BuildConfig           buildConfig)
    {
      spatialMedian_impl::spatialMedian(bvh,boxes,numPrims,buildConfig);
    }

    
    template<typename T, int D, int W>
    void spatialMedian(WideBVH<T,D,W>   &bvh,
                       const box_t<T,D> *boxes,
                       uint32_t              numPrims,
                       BuildConfig           buildConfig)
    { throw std::runtime_error("not yet implemented"); }
    
#endif
  }
}

#define CUBQL_CPU_INSTANTIATE_BINARY_BVH(T,D)                      \
  namespace cuBQL {                                                 \
    namespace cpu {                                                \
      template void spatialMedian(BinaryBVH<T,D>   &bvh,            \
                                  const box_t<T,D> *boxes,          \
                                  uint32_t          numPrims,       \
                                  BuildConfig       buildConfig);   \
    }                                                               \
  }                                                                 \


#define CUBQL_CPU_INSTANTIATE_WIDE_BVH(T,D,W)                       \
  namespace cuBQL {                                                 \
    namespace cpu {                                                \
      template void spatialMedian(WideBVH<T,D,W>   &bvh,            \
                                  const box_t<T,D> *boxes,          \
                                  uint32_t          numPrims,       \
                                  BuildConfig       buildConfig);   \
    }                                                               \
  }                                                                 \


