// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cuBQL/traversal/fixedAnyShapeQuery.h"

namespace cuBQL {
  namespace fixedBoxQuery {
    
    /*! This query finds all primitives within a given fixed (ie, never
      changing) axis-aligned cartesian box, and calls the provided
      callback-lambda for each such prim. The provided lambda can do
      with the provided prim as it pleases, and is to report either
      CUBQL_TERMINATE_TRAVERSAL (in which case traversal will
      immediately terminate), or CUBQL_CONTINUE_TRAVERSAL (in which
      case traversal will continue to the next respective primitmive
      within the box, if such exists. 
    
      for this "for each prim' variant, the lambda should have a signature of
    
      [](int primID)->int

      with the returned int being either CUBQL_TERMINATE_TRAVERSAL or
      CUBQL_CONTINUE_TRAVERSAL
    */
    template<typename T, int D, typename Lambda>
    inline __cubql_both
    void forEachPrim(const Lambda &lambdaToCallOnEachPrim,
                     const BinaryBVH<T,D> bvh,
                     const box3f queryBox,
                     bool dbg=false);
  
    template<typename T, int D, typename Lambda>
    inline __cubql_both
    void forEachLeaf(const Lambda &lambdaToCallOnEachPrim,
                     const BinaryBVH<T,D> bvh,
                     const box3f queryBox,
                     bool dbg=false);
  
    template<typename T, int D, int W, typename Lambda>
    inline __cubql_both
    void forEachPrim(const Lambda &lambdaToCallOnEachPrim,
                     const WideBVH<T,D,W> bvh,
                     const box3f queryBox,
                     bool dbg=false);
  
    template<typename T, int D, int W, typename Lambda>
    inline __cubql_both
    void forEachLeaf(const Lambda &lambdaToCallOnEachPrim,
                     const WideBVH<T,D,W> bvh,
                     const box3f queryBox,
                     bool dbg=false);
  



    // ==================================================================
    // IMPLEMENTATION
    // ==================================================================

    template<typename T, int D, typename Lambda>
    inline __cubql_both
    void forEachLeaf(const Lambda &lambdaToCallOnEachLeaf,
                     const BinaryBVH<T,D> bvh,
                     const box3f queryBox,
                     bool dbg)
    {
      bvh3f::node_t::Admin traversalStack[64], *stackPtr = traversalStack;
      bvh3f::node_t::Admin node = bvh.nodes[0].admin;
      // ------------------------------------------------------------------
      // traverse until there's nothing left to traverse:
      // ------------------------------------------------------------------
      // if (dbg) dout << "fixedBoxQuery::traverse" << endl;
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
          bvh3f::node_t n0 = bvh.nodes[n0Idx];
          bvh3f::node_t n1 = bvh.nodes[n1Idx];
          bool o0 = queryBox.overlaps(n0.bounds);
          bool o1 = queryBox.overlaps(n1.bounds);

          // if (dbg) {
          //   dout << "at node " << node.offset << endl;
          //   dout << "w/ query box " << queryBox << endl;
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

        // if (dbg)
        //   dout << "at leaf ofs " << (int)node.offset << " cnt " << node.count << endl;
        if (node.count != 0) {
          // we're at a valid leaf: call the lambda and see if that gave
          // us a enw, closer cull radius
          int leafResult
            = lambdaToCallOnEachLeaf(bvh.primIDs+node.offset,(uint32_t)node.count);
          // if (dbg)
          //   dout << "leaf returned " << leafResult << endl;
          if (leafResult == CUBQL_TERMINATE_TRAVERSAL)
            return;
        }
        // ------------------------------------------------------------------
        // pop next un-traversed node from stack, discarding any nodes
        // that are more distant than whatever query radius we now have
        // ------------------------------------------------------------------
        // if (dbg) dout << "rem stack depth " << (stackPtr-traversalStack) << endl;
        if (stackPtr == traversalStack)
          return;
        node = *--stackPtr;
      }
    }

    template<typename T, int D, typename Lambda>
    inline __cubql_both
    void forEachPrim(const Lambda &lambdaToCallOnEachPrim,
                     const BinaryBVH<T,D> bvh,
                     const box3f queryBox,
                     bool dbg)
    {
      auto leafCode = [&lambdaToCallOnEachPrim]
        (const uint32_t *primIDs, size_t numPrims) -> int
      {
        for (int i=0;i<(int)numPrims;i++)
          if (lambdaToCallOnEachPrim(primIDs[i]) == CUBQL_TERMINATE_TRAVERSAL)
            return CUBQL_TERMINATE_TRAVERSAL;
        return CUBQL_CONTINUE_TRAVERSAL;
      };
      forEachLeaf(leafCode,bvh,queryBox,dbg);
    }




    template<typename T, int D, int W, typename Lambda>
    inline __cubql_both
    void forEachLeaf(const Lambda &lambdaToCallOnEachLeaf,
                     const WideBVH<T,D,W> bvh,
                     const box3f queryBox,
                     bool dbg)
    {
      struct StackEntry {
        uint64_t nodeID:48;
        uint64_t childID:16;
      };
      StackEntry traversalStack[64], *stackPtr = traversalStack;
      StackEntry current;
      current.nodeID = 0;
      current.childID = 0;
#if 1
      typename WideBVH<T,D,W>::Node::Child child;
      while (/* while there may be more leaves to find */true) {
        while (/* while we have not reached a leaf */true) {
          child = bvh.nodes[current.nodeID].children[current.childID];
          if (!child.valid
              ||
              !queryBox.overlaps(child.bounds)) {
            if (current.childID+1 < W)
              current.childID = current.childID+1;
            else if (stackPtr > traversalStack)
              current = *--stackPtr;
            else
              return;
          } else if (child.count == 0) {
            if (current.childID+1 < W) {
              current.childID++;
              *stackPtr++ = current;
            }
            // .. then point current to first child of down-node
            current.nodeID = child.offset;
            current.childID = 0;
          } else
            break;
        }

        // we DID reach a leaf!
        int leafResult
          = lambdaToCallOnEachLeaf(bvh.primIDs+child.offset,(uint32_t)child.count);
        if (leafResult == CUBQL_TERMINATE_TRAVERSAL)
          return;
        if (current.childID+1 < W)
          current.childID = current.childID+1;
        else if (stackPtr > traversalStack)
          current = *--stackPtr;
        else
          /// nothing more to do!
          return;
      }
#else
      // ------------------------------------------------------------------
      // relatively straight-forward traversal; careful there's some
      // good degree of divergence here because leaves get processes
      // the moment they get reached, no matter how many other threads
      // decided to skip _their_ current node
      // ------------------------------------------------------------------
      while (true) {
        auto child = bvh.nodes[current.nodeID].children[current.childID];

        if (!child.valid)
          /* nothing to do here, then skip to next */;
        else if (!queryBox.overlaps(child.bounds))
          /* nothing to do here, then skip to next */;
        else if (child.count != 0) {
          // it's a boy! do the leaf...
          int leafResult
            = lambdaToCallOnEachLeaf(bvh.primIDs+child.offset,(uint32_t)child.count);
          if (leafResult == CUBQL_TERMINATE_TRAVERSAL)
            return;
          // ... then skip to next
        } else {
          // we need to go down - first save check if we do have a
          // right neighbor, and if so, push it:
          if (current.childID+1 < W) {
            current.childID++;
            *stackPtr++ = current;
          }
          // .. then point current to first child of down-node
          current.nodeID = child.offset;
          current.childID = 0;
          // do NOT skip to next-node/pop-from-stack, this IS the next
          // node to taverse!
          continue;
        }
        // we do NOT yet have a next node to go to - find one!
        if (current.childID+1 < W)
          current.childID = current.childID+1;
        else if (stackPtr > traversalStack)
          current = *--stackPtr;
        else
          /// nothing more to do!
          return;
      }
#endif
    }

    template<typename T, int D, int W, typename Lambda>
    inline __cubql_both
    void forEachPrim(const Lambda &lambdaToCallOnEachPrim,
                     const WideBVH<T,D,W> bvh,
                     const box3f queryBox,
                     bool dbg)
    {
      auto leafCode = [&lambdaToCallOnEachPrim]
        (const uint32_t *primIDs, size_t numPrims) -> int
      {
        for (int i=0;i<(int)numPrims;i++)
          if (lambdaToCallOnEachPrim(primIDs[i]) == CUBQL_TERMINATE_TRAVERSAL)
            return CUBQL_TERMINATE_TRAVERSAL;
        return CUBQL_CONTINUE_TRAVERSAL;
      };
      forEachLeaf(leafCode,bvh,queryBox,dbg);
    }
    
  } // ::cubql::fixedBoxQuery
} // ::cubql
