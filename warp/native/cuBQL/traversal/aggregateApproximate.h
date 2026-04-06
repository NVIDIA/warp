// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cuBQL/bvh.h"
#include <cuBQL/math/vec.h>
#include <cuBQL/math/box.h>
#include <cuBQL/math/affine.h>
#include <cuBQL/math/conservativeDistances.h>

/* Defines and implements cuBQL "approximate/aggregate" style
   traversals that can, for example, be used for N-body style
   problems.

   The core idea of these types of queries is that the user provides
   three things:

   - one, some per-subtree 'aggregate data' (of the user's choosing,
     and computed, for example, via refit_aggregate()). For an n-body
     style problem this could, for example, be the sum of all
     planets/bodies/masses in a subtree.

   - second, a callback function that checks if a given query can be
     approximately fulfilled with the subtree's aggregate data; i.e.,
     _without_ having to traverse that subtree's children. If so, this
     helper function can accumulate this partial result (in whatever
     way it chooses - it's user code, after all), and returns 'true'
     to tell cuBQL that this subtree is 'done' and does not require
     further processing. Otherwise, it returns 'false' and cuBQL will
     process the children

   - third, a second callback function that operates on individual
     primitmives, and gets called by cuBQL if traversal reaches a leaf
     without ever having decided to approximate in any of that child
     dnoe's parent nodes

   Obviously both callback functions need additional data to do their
   job: the bvh to be traversed (eg to get a node's bounding box), the
   (tempalted) aggregate data (obviously), the (templated) query_t for
   which the query is performed, and some (templated) result_t in
   which both callbacks can accumulate their partial results (eg for
   an n-body style, this could be the sum of all forces)

   Note that "approximate/aggregate" refers to the two key concepts
   required to realize these kind of traversals: the idea to avoid a
   "full" tree traversal by "approimating" certain subtrees (instead
   of just traversing both children); and the idea that one needs some
   sort of "aggregate data" for a subtree to even decide whether
   that's possible or not.
*/ 
namespace cuBQL {
  namespace aggregateApproximate {

    // ------------------------------------------------------------------
    // INTERFACE
    // ------------------------------------------------------------------
    
    /*! implements a approximate/aggregate traversal (see above for
        the core idea). Note this function is heavily templated, so to
        allow template matchign to do its magic the order of
        parameters is pretty important.

        `approximateSubtreeFct_t` is a lambda with signature
        inline __device__
        bool approximateSubtree(bvh_t,
                                aggregateNodeData_t [],
                                int nodeID,
                                result_t &,
                                query_t)
        
        `perPrimFct_t` is a lambda with signature
        inline __device__
        void processPrim(result_t &result,
                         const query_t &queryPoint,
                         int primID,
                         const primitive_t prims[])

    */
    template
    < /*! T/D that describes the BVH data/dimensionality */
      typename T, int D,
      /*! the aggregate data we store per node (to base the user's
          culling test on) */
      typename aggregateNodeData_t,
      /* the type of primitmives stored in leaves, for when traversal
         reaches leaves */
      typename primitive_t,
      /*! the result type of the query, for when traversal decides to
          either aprpxoimate a subtree (and has to udpate the result
          with that subtree's approximateion), or when prim tests are
          done */
      typename result_t,
      /*! the type of query primitive that the traversal operates
          on */
      typename query_t,
      /*! the lambda function that tests a subtree on whether it can
          be solved with an approximation */
      typename approximateSubtreeFct_t,
      /*! the lambda function that gets executed for each prim (those
          that did not get approximates higher up the tree */
      typename perPrimFct_t>
    inline __device__
    void traverse(bvh_t<T,D> bvh,
                  aggregateNodeData_t aggregateData[],
                  primitive_t primitives[],
                  result_t result,
                  query_t queryPoint,
                  const approximateSubtreeFct_t &approximateSubtreeFct,
                  const perPrimFct_t perPrimFct);


    // ------------------------------------------------------------------
    // IMPLEMENTATION
    // ------------------------------------------------------------------
    template
    </* T/D that describes the BVH data/dimensionality */
      typename T, int D,
      typename aggregateNodeData_t,
      typename primitive_t,
      typename result_t,
      typename query_t,
      typename approximateSubtreeFct_t,
      typename perPrimFct_t>
    inline __device__
    void traverse(bvh_t<T,D> bvh,
                  aggregateNodeData_t aggregateData[],
                  primitive_t primitives[],
                  result_t result,
                  query_t queryPrim,
                  const approximateSubtreeFct_t &approximateSubtreeFct,
                  const perPrimFct_t perPrimFct)
    {
      struct StackEntry {
        uint32_t idx;
      };
      bvh3f::node_t::Admin traversalStack[64], *stackPtr = traversalStack;
      bvh3f::node_t::Admin node = bvh.nodes[0].admin;
      // ------------------------------------------------------------------
      // traverse until there's nothing left to traverse:
      // ------------------------------------------------------------------
      while (true) {

        // ------------------------------------------------------------------
        // traverse INNER nodes downward; breaking out if we either
        // find a leaf, or a encounter subtrees that can be either
        // approximated or culled with the approximateSubtreeFct()
        // ------------------------------------------------------------------
        while (true) {
          if (node.count != 0)
            // it's a boy! - seriously: this is not a inner node, step
            // out of down-travesal and let leaf code pop in.
            break;

          uint32_t n0Idx = (uint32_t)node.offset+0;
          uint32_t n1Idx = (uint32_t)node.offset+1;
          bvh3f::node_t n0 = bvh.nodes[n0Idx];
          bvh3f::node_t n1 = bvh.nodes[n1Idx];
          bool done0 = approximateSubtreeFct(bvh,aggregateData,n0Idx,
                                             result,queryPrim);
          bool done1 = approximateSubtreeFct(bvh,aggregateData,n1Idx,
                                             result,queryPrim);
          bool o0 = !done0;
          bool o1 = !done1;
          if (o0) {
            if (o1) {
              *stackPtr++ = n1.admin;
            } else {
            }
            node = n0.admin;
          } else {
            if (o1) {
              node = n1.admin;
            } else {
              // both children are too far away; this is a dead end
              node.count = 0;
              break;
            }
          }
        }

        if (node.count != 0) {
          for (int i=0;i<node.count;i++)
            perPrimFct(result,
                       queryPrim,
                       bvh.primIDs[node.offset+i],
                       primitives);
        }
        // ------------------------------------------------------------------
        // pop next un-traversed node from stack, discarding any nodes
        // that are more distant than whatever query radius we now have
        // ------------------------------------------------------------------
        if (stackPtr == traversalStack)
          return;
        node = *--stackPtr;
      }
    }
    
  } // ::cuBQL::aggregateApproximate
} // ::cuBQL

