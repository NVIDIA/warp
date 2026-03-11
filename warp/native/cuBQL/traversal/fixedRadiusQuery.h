// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file fixedRadiusQuery.h implements the traversal fromworks for
    various traversals the perofrm a *fixed* radius query. Ie, these
    traversals will traverse a BVH and call the provided lambdas with
    any candidate (leaf or prim, depending on traversal) that might
    yield an intersection with the query ball defiend by query center
    and query radius. Being a *fixed* radius query the user's
    per-prim/per-leaf lambda are supposed to return either
    CUBQL_TERMINATE_TRAVERSAL or CUBQL_CONTINUE_TRAVERSAL */

#pragma once

#include <cuBQL/math/conservativeDistances.h>
#include <cuBQL/traversal/fixedAnyShapeQuery.h>

namespace cuBQL {
  namespace fixedRadiusQuery {
    
    // ******************************************************************
    // INTERFACE
    // (which functions this header file provides)
    // ******************************************************************

    /*! This query finds all BVH *leaves* within a given fixed (ie,
        never either moving or shrinking) query ball, and calls a
        user-provided lambda for each bvh leaf within this query
        ball. The ball is given by its center point and its *square*
        radius, the lambda is expected to return either
        CUBQL_TERMINATE_TRAVERAL or CUBQL_CONTINUE_TRAVERSAL.
    
        For this variant the lambda has signature
        
    
        [](int primID)->int
        
        with the returned int being either CUBQL_TERMINATE_TRAVERSAL or
        CUBQL_CONTINUE_TRAVERSAL
    */
    template<typename T, int D,
             typename PerPrimLambda>
    inline __cubql_both
    void forEachPrim(/*! the SQUARE of the maximum query radius to
                       which we want to restrict the search; can be
                       CUBQL_INF for unrestricted searches */
                     PerPrimLambda lambdaToCallForEachPrimInSphere,
                     /* the bvh we're querying into */
                     bvh_t<T,D> bvh,
                     /* the center of the query ball we're querying */
                     vec_t<T,D> queryBallCenter,
                     /*! the SQUARE of the maximum query radius to
                       which we want to restrict the search; can be
                       CUBQL_INF for unrestricted searches */
                     T queryBallRadiusSquare,
                     bool dbg = false);

    /*! This query finds all BVH *leaves* within a given fixed (ie,
        never either moving or shrinking) query ball, and calls a
        user-provided lambda for each bvh leaf within this query
        ball. The ball is given by its center point and its *square*
        radius, the lambda is expected to return either
        CUBQL_TERMINATE_TRAVERAL or CUBQL_CONTINUE_TRAVERSAL.
    
        For this variant the lambda has signature
        
        [](const uint32_t *primIDs, size_t numPrims)->int
        
        with the returned int being either CUBQL_TERMINATE_TRAVERSAL or
        CUBQL_CONTINUE_TRAVERSAL
    */
    template<typename T, int D, typename PerLeafLambda>
    inline __cubql_both
    void forEachLeaf(/*! the user-provided lambda that we'll call for
                         each leaf that overlaps the query ball */
                     PerLeafLambda lambdaToCallForEachLeaf,
                     /* the bvh we're querying into */
                     bvh_t<T,D> bvh,
                     /* the center of the query ball we're querying */
                     vec_t<T,D> queryBallCenter,
                     /*! the SQUARE of the maximum query radius to
                       which we want to restrict the search; can be
                       CUBQL_INF for unrestricted searches */
                     T queryBallRadiusSquare,
                     bool dbg = false);
  
    // ******************************************************************
    // IMPLEMENTATION
    // ******************************************************************
    
    /*! This query finds all BVH *leaves* within a given fixed (ie,
        never either moving or shrinking) query ball, and calls a
        user-provided lambda for each bvh leaf within this query
        ball. The ball is given by its center point and its *square*
        radius, the lambda is expected to return either
        CUBQL_TERMINATE_TRAVERAL or CUBQL_CONTINUE_TRAVERSAL.
    
        For this variant the lambda has signature
        
        [](const uint32_t *primIDs, size_t numPrims)->int
        
        with the returned int being either CUBQL_TERMINATE_TRAVERSAL or
        CUBQL_CONTINUE_TRAVERSAL
    */
    template<typename T, int D, typename PerLeafLambda>
    inline __cubql_both
    void forEachLeaf(/*! the user-provided lambda that we'll call for
                         each leaf that overlaps the query ball */
                     PerLeafLambda lambdaToCallForEachLeaf,
                     /* the bvh we're querying into */
                     bvh_t<T,D> bvh,
                     /* the center of the query ball we're querying */
                     vec_t<T,D> queryBallCenter,
                     /*! the SQUARE of the maximum query radius to
                       which we want to restrict the search; can be
                       CUBQL_INF for unrestricted searches */
                     T queryBallRadiusSquare,
                     bool dbg)
    {
      const int stackSize = 64;
      typename bvh_t<T, D>::node_t::Admin traversalStack[stackSize], *stackPtr = traversalStack;
      typename bvh_t<T,D>::node_t::Admin node = bvh.nodes[0].admin;
      // ------------------------------------------------------------------
      // traverse until there's nothing left to traverse:
      // ------------------------------------------------------------------
      while (true) {

        // ------------------------------------------------------------------
        // traverse INNER nodes downward; breaking out if we either find
        // a leaf within the current search radius, or found a dead-end
        // at which we need to pop
        // ------------------------------------------------------------------
        while (true) {
          if (node.count != 0)
            // it's a boy! - seriously: this is not a inner node, step
            // out of down-travesal and let leaf code pop in.
            break;

          uint32_t n0Idx = (uint32_t)node.offset+0;
          uint32_t n1Idx = (uint32_t)node.offset+1;
          typename bvh_t<T,D>::node_t n0 = bvh.nodes[n0Idx];
          typename bvh_t<T,D>::node_t n1 = bvh.nodes[n1Idx];
          float d0 = fSqrDistance_rd(queryBallCenter,n0.bounds);
          float d1 = fSqrDistance_rd(queryBallCenter,n1.bounds);
          bool o0 = (d0 < queryBallRadiusSquare);
          bool o1 = (d1 < queryBallRadiusSquare);

          // if (dbg) {
          //   dout << "at node " << node.offset << endl;
          //   dout << "w/ query ball " << queryBallCenter << "," << queryBallRadiusSquare << endl;
          //   dout << "  " << n0.bounds << " -> " << (int)o0 << endl;
          //   dout << "  " << n1.bounds << " -> " << (int)o1 << endl;
          // }
          
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
          // we're at a valid leaf: call the lambda and see if that gave
          // us a enw, closer cull radius
          int leafResult
            = lambdaToCallForEachLeaf(bvh.primIDs+node.offset,(uint32_t)node.count);
          if (leafResult == CUBQL_TERMINATE_TRAVERSAL)
            return;
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
    
    /*! This query finds all *primitives* that are possibly within a
        given fixed (ie, never either moving or shrinking) query ball,
        and calls a user-provided lambda for each bvh leaf within this
        query ball. The ball is given by its center point and its
        *square* radius, the lambda is expected to return either
        CUBQL_TERMINATE_TRAVERAL or CUBQL_CONTINUE_TRAVERSAL.
    
        For this variant the lambda has signature
        
    
        [](int primID)->int
        
        with the returned int being either CUBQL_TERMINATE_TRAVERSAL or
        CUBQL_CONTINUE_TRAVERSAL
    */
    template<typename T, int D,
             typename PerPrimLambda>
    inline __cubql_both
    void forEachPrim(/*! the SQUARE of the maximum query radius to
                       which we want to restrict the search; can be
                       CUBQL_INF for unrestricted searches */
                     PerPrimLambda lambdaToCallForEachPrimInSphere,
                     /* the bvh we're querying into */
                     bvh_t<T,D> bvh,
                     /* the query ball we're querying - note SQUARE radius */
                     vec_t<T,D> queryBallCenter,
                     T queryBallRadiusSquare,
                     bool dbg)
    {
      /* the code we want to have executed for each leaf that may
         contain candidates. we loop over each prim in a given leaf,
         and return the minimum culling distance returned by any of
         the per-prim lambdas */
      auto lambdaToCallForEachVisitedNode
        = [lambdaToCallForEachPrimInSphere](const uint32_t *leafPrims,
                                            size_t numPrims)->float
        {
          for (size_t i=0;i<numPrims;i++)
            if (lambdaToCallForEachPrimInSphere(leafPrims[i]) == CUBQL_TERMINATE_TRAVERSAL)
              return CUBQL_TERMINATE_TRAVERSAL;
          return CUBQL_CONTINUE_TRAVERSAL;
        };
      forEachLeaf(lambdaToCallForEachVisitedNode,bvh,
                  queryBallCenter,queryBallRadiusSquare,dbg);
    }
    
  }
}

