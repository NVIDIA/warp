// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cuBQL/math/common.h"
#include "cuBQL/math/vec.h"
#include "cuBQL/math/box.h"
#ifdef __CUDACC__
# include <cuda.h>
#endif

namespace cuBQL {

  /*! struct used to control how exactly the builder is supposed to
      build the tree; in particular, at which threshold to make a
      leaf */
  struct BuildConfig {
    BuildConfig(int makeLeafThreshold=0)
      : makeLeafThreshold(makeLeafThreshold)
    {}
    inline BuildConfig &enableSAH() { buildMethod = SAH; return *this; }
    inline BuildConfig &enableELH() { buildMethod = ELH; return *this; }
    typedef enum
      {
       /*! simple 'adaptive spatial median' strategy. When splitting a
         subtree, this first computes the centroid of each input
         primitive in that subtree, then computes the bounding box of
         those centroids, then creates a split plane along the widest
         dimension of that centroid boundig box, right through the
         middle */
       SPATIAL_MEDIAN=0,
       /*! use good old surface area heurstic. In theory that only
         makes sense for BVHes that are used for tracing rays
         (theoretic motivation is a bit wobbly for other sorts of
         queries), but it seems to help even for other queries. Much
         more expensive to build, though */
       SAH,
       /*! edge-length heuristic - experimental */
       ELH
    } BuildMethod;
    
    /*! what leaf size the builder is _allowed_ to make; no matter
        what input is specified, the builder may never produce leaves
        larger than this value */
    int maxAllowedLeafSize = 1<<15;

    /*! threshold below which the builder should make a leaf, no
        matter what the prims in the subtree look like. A value of 0
        means "leave it to the builder" */
    int makeLeafThreshold = 0;

    BuildMethod buildMethod = SPATIAL_MEDIAN;
  };

  /*! the most basic type of BVH where each BVH::Node is either a leaf
      (and contains Node::count primitives), or is a inner node (and
      points to a pair of child nodes). Node 0 is the root node; node
      1 is always unused (so all other node pairs start on n even
      index) */
  template<typename _scalar_t, int _numDims>
  struct BinaryBVH {
    using scalar_t = _scalar_t;
    enum { numDims = _numDims };
    using vec_t = cuBQL::vec_t<scalar_t,numDims>;
    using box_t = cuBQL::box_t<scalar_t,numDims>;

    static constexpr int const node_width = 1;

    struct CUBQL_ALIGN(16) Node {
      enum { count_bits = 16, offset_bits = 64-count_bits };

      box_t    bounds;

      struct Admin {
      /*! For inner nodes, this points into the nodes[] array, with
        left child at nodes.offset+0, and right child at
        nodes.offset+1. For leaf nodes, this points into the
        primIDs[] array, which first prim beign primIDs[offset],
        next one primIDs[offset+1], etc. */
        union {
          struct {
            uint64_t offset : offset_bits;
            /* number of primitives in this leaf, if a leaf; 0 for inner
               nodes. */
            uint64_t count  : count_bits;
          };
          // the same as a single int64, so we can read/write with a
          // single op
          uint64_t offsetAndCountBits;
        };
      };
      Admin admin;
    };

    enum { maxLeafSize=((1<<Node::count_bits)-1) };
    
    using node_t       = Node;
    node_t   *nodes    = 0;
    uint32_t  numNodes = 0;
    uint32_t *primIDs  = 0;
    uint32_t  numPrims = 0;
  };

  /*! a 'wide' BVH in which each node has a fixed number of
    `BVH_WIDTH` children (some of those children can be un-used) */
  template<typename _scalar_t, int _numDims, int BVH_WIDTH>
  struct WideBVH {
    using scalar_t = _scalar_t;

    enum { numDims = _numDims };
    using vec_t = cuBQL::vec_t<scalar_t,numDims>;
    using box_t = cuBQL::box_t<scalar_t,numDims>;

    static constexpr int const node_width = BVH_WIDTH;

    /*! a n-wide node of this BVH; note that unlike BinaryBVH::Node
      this is not a "single" node, but actually N nodes merged
      together */
    struct CUBQL_ALIGN(16) Node {
      struct CUBQL_ALIGN(16) Child {
        box_t    bounds;
        struct {
          uint64_t valid  :  1;
          uint64_t offset : 45;
          uint64_t count  : 16;
        };
      } children[BVH_WIDTH];
    };

    using node_t = Node;
    node_t   *nodes    = 0;
    //! number of (multi-)nodes on this WideBVH
    uint32_t  numNodes = 0;
    uint32_t *primIDs  = 0;
    uint32_t  numPrims = 0;
  };


  template<typename T, int D>
  using bvh_t = BinaryBVH<T,D>;

  // easy short-hand - though cubql also supports other types of bvhs,
  // scalars, etc, this will likely be the most commonly used one.
  using bvh3f = BinaryBVH<float,3>;
  using bvh3d = BinaryBVH<double,3>;

#ifdef __CUDACC__
  typedef BinaryBVH<float,2> bvh_float2;
  typedef BinaryBVH<float,3> bvh_float3;
  typedef BinaryBVH<float,4> bvh_float4;
#endif
  
} // ::cuBQL

#ifdef __CUDACC__
# include "cuBQL/builder/cuda.h"
#endif
# include "cuBQL/builder/cpu.h"



  
