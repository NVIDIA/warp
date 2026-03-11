// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cuBQL/math/vec.h"
#include "cuBQL/math/box.h"
#include "cuBQL/math/Ray.h"
#include "cuBQL/traversal/fixedBoxQuery.h"

namespace cuBQL {

  // ******************************************************************
  // INTERFACE
  // (which functions this header file provides)
  // ******************************************************************

  /*! \namespace fixedRayQuery *Fixed* ray queries are queries on rays
    for whom the query interval [ray.tMax,ray.tMax] can never change
    during traversal. The per-prim/per-leaf lambdas can at any point
    _terminate_ a traveral, but ordering child nodes is not required
    because ordering shouldn't matter */
  namespace fixedRayQuery {
    template<typename Lambda, typename T, int D>
    inline __cubql_both
    void forEachLeaf(const Lambda &lambdaToExecuteForEachCandidate,
                     cuBQL::bvh_t<T, D> bvh,
                     cuBQL::ray3f ray,
                     bool dbg=false);

    template<typename Lambda, typename T, int D, int W>
    inline __cubql_both
    void forEachLeaf(const Lambda &lambdaToExecuteForEachCandidate,
                     cuBQL::WideBVH<T, D, W> bvh,
                     cuBQL::ray3f ray,
                     bool dbg=false);
    
    template<typename Lambda, typename bvh_t>
    inline __cubql_both
    void forEachPrim(const Lambda &lambdaToExecuteForEachCandidate,
                     bvh_t bvh,
                     cuBQL::ray3f ray,
                     bool dbg=false);
    
    /*! traverse BVH with given fixed-length, axis-aligned ray, and
      call lambda for each prim encounterd.
      
      Traversal is UNORDERED (meaning it will NOT try to traverse
      front-to-back) and FIXED-SHAPE (ray will not shrink during
      traversal).
      
      Lambda is expected to return CUBQL_{CONTINUE|TERMINATE}_TRAVERSAL 
    */
    template<int axis, int sign, typename Lambda>
    inline __cubql_both
    void forEachLeaf(const Lambda &lambdaToExecuteForEachCandidate,
                     cuBQL::bvh3f bvh,
                     AxisAlignedRay<axis,sign> ray,
                     bool dbg=false);
    
    /*! traverse BVH with given fixed-length, axis-aligned ray, and
      call lambda for each prim encounterd.
      
      Traversal is UNORDERED (meaning it will NOT try to traverse
      front-to-back) and FIXED-SHAPE (ray will not shrink during
      traversal).
      
      Lambda is expected to return CUBQL_{CONTINUE|TERMINATE}_TRAVERSAL 
    */
    template<int axis, int sign, typename Lambda>
    inline __cubql_both
    void forEachPrim(const Lambda &lambdaToExecuteForEachCandidate,
                     cuBQL::bvh3f bvh,
                     AxisAlignedRay<axis,sign> ray,
                     bool dbg=false);
  }

  /*! \namespace shrinkingRayQuery *Shrinking* ray queries are queries
    where ray.tMax can shrink during traversal, so hits found in one
    subtree can shrink ray.tmax such that other subtrees may then
    later get skipped */
  namespace shrinkingRayQuery {
    
    /*! single level BVH ray traversal, provided lambda covers what
      happens when a ray wants to intersect a given prim within that
      bvh */
    template<typename Lambda, typename T, int D, typename ray_t>
    inline __cubql_both
    float forEachLeaf(const Lambda &lambdaToCallOnEachLeaf,
                      bvh_t<T, D> bvh,
                      ray_t ray,
                      bool dbg=false);

    /*! single level BVH ray traversal, provided lambda covers what
      happens when a ray wants to intersect a given prim within that
      bvh */
    template<typename Lambda, typename T, int D, int W, typename ray_t>
    inline __cubql_both
    float forEachLeaf(const Lambda &lambdaToCallOnEachLeaf,
                      WideBVH<T, D, W> bvh,
                      ray_t ray,
                      bool dbg=false);
    
    /*! single level BVH ray traversal, provided lambda covers what
      happens when a ray wants to intersect a given prim within that
      bvh */
    template<typename Lambda, typename bvh_t, typename ray_t>
    inline __cubql_both
    void forEachPrim(const Lambda &lambdaToExecuteForEachCandidate,
                     bvh_t bvh,
                     ray_t &ray,
                     bool dbg=false);
    
    namespace twoLevel {
      /*! two-level BVH ray traversal, where the BVH is made up of a
        "TLAS" (top-level acceleration structure) that itself contains
        objects with "BLAS"es (bottom-level acceleration
        structures). One of the lambdas describes what happens when a
        ray enters a leaf in a BLAS (just like in the single-level BVH
        traversal; the other describes what happens when a ray needs
        to transition from TLAS to BLAS. That second lambda can modify
        the current ray's org and dir to transform it into the BLASes
        coordinate frame where required (transforming back is not
        required, cubql will save/restore the original ray as
        required), and is supposed to return a new bvh_t to be
        traversed in the BLAS */
      template<typename EnterBlasLambda,
               typename LeaveBlasLambda,
               typename ProcessLeafLambda,
               typename bvh_t, typename ray_t>
      inline __cubql_both
      void forEachLeaf(const EnterBlasLambda   &enterBlas,
                       const LeaveBlasLambda   &leaveBlas,
                       const ProcessLeafLambda &processLeaf,
                       bvh_t bvh,
                       /*! REFERENCE to a ray, so 'enterBlas()' can modify it */
                       ray_t &ray,
                       bool dbg=false);
      
      /*! two-level BVH ray traversal, where the BVH is made up of a
        "TLAS" (top-level acceleration structure) that itself contains
        objects with "BLAS"es (bottom-level acceleration
        structures). One of the lambdas describes what happens when a
        ray enters a leaf in a BLAS (just like in the single-level BVH
        traversal; the other describes what happens when a ray needs
        to transition from TLAS to BLAS. That second lambda can modify
        the current ray's org and dir to transform it into the BLASes
        coordinate frame where required (transforming back is not
        required, cubql will save/restore the original ray as
        required), and is supposed to return a new bvh_t to be
        traversed in the BLAS */
      template<typename EnterBlasLambda,
               typename LeaveBlasLambda,
               typename IntersectPrimLambda,
               typename bvh_t, typename ray_t>
      inline __cubql_both
      void forEachPrim(const EnterBlasLambda     &enterBlas,
                       const LeaveBlasLambda     &leaveBlas,
                       const IntersectPrimLambda &intersectPrim,
                       bvh_t bvh,
                       /*! REFERENCE to a ray, so 'enterBlas()' can modify it */
                       ray_t &ray,
                       bool dbg=false);
    }

  } // ::cuBQL::shrinkingRayQuery


  // =========================================================================
  // *** IMPLEMENTATION ***
  // =========================================================================

  template<typename T>
  inline __cubql_both
  bool rayIntersectsBox(ray_t<T> ray, box_t<T,3> box)
  {
    using vec3 = vec_t<T,3>;
    vec3 inv = rcp(ray.direction);
    vec3 lo = (box.lower - ray.origin) * inv;
    vec3 hi = (box.upper - ray.origin) * inv;
    vec3 nr = min(lo,hi);
    vec3 fr = max(lo,hi);
    T tin  = max(ray.tMin,reduce_max(nr));
    T tout = min(ray.tMax,reduce_min(fr));
    return tin <= tout;
  }

  template<typename T>
  inline __cubql_both
  void rayBoxTest(T &tin, T &tout,
                  ray_t<T> ray, box_t<T,3> box)
  {
    using vec3 = vec_t<T,3>;
    vec3 inv = rcp(ray.direction);
    vec3 lo = (box.lower - ray.origin) * inv;
    vec3 hi = (box.upper - ray.origin) * inv;
    vec3 nr = min(lo,hi);
    vec3 fr = max(lo,hi);
    tin  = max(ray.tMin,reduce_max(nr));
    tout = min(ray.tMax,reduce_min(fr));
  }

  template<typename T>
  inline __cubql_both
  bool rayIntersectsBox(float &ret_t0,
                        ray_t<T> ray, vec_t<T,3> rcp_dir, box_t<T,3> box)
  {
    using vec3 = vec_t<T,3>;
    vec3 lo = (box.lower - ray.origin) * rcp_dir;
    vec3 hi = (box.upper - ray.origin) * rcp_dir;
    vec3 nr = min(lo,hi);
    vec3 fr = max(lo,hi);
    T tin  = max(ray.tMin,reduce_max(nr));
    T tout = min(ray.tMax,reduce_min(fr));
    ret_t0 = tin;
    return tin <= tout;
  }


  template<int axis, int sign, typename Lambda>
  inline __cubql_both
  void fixedRayQuery::forEachLeaf(const Lambda &lambdaToExecuteForEachCandidate,
                                  cuBQL::bvh3f bvh,
                                  AxisAlignedRay<axis,sign> ray,
                                  bool dbg)
  {
    /* for an axis-aligned ray, we can just convert that ray to a
       box, and traverse that instad */
    vec3f A = ray.origin + ray.tMin * ray.direction();
    vec3f B = ray.origin + ray.tMax * ray.direction();
    box3f rayAsBox { min(A,B), max(A,B) };
    // if (dbg) dout << "asbox " << rayAsBox << dout.endl;
    cuBQL::fixedBoxQuery::forEachLeaf(lambdaToExecuteForEachCandidate,bvh,rayAsBox,dbg);
  }

  /*! this query assumes lambads that return CUBQL_CONTINUE_TRAVERSAL
    or CUBQL_TERMINATE_TRAVERSAL */
  template<int axis, int sign, typename Lambda>
  inline __cubql_both
  void fixedRayQuery::forEachPrim(const Lambda &lambdaToExecuteForEachCandidate,
                                  cuBQL::bvh3f bvh,
                                  AxisAlignedRay<axis,sign> ray,
                                  bool dbg)
  {
    /* the code we want to have executed for each leaf that may
       contain candidates. we loop over each prim in a given leaf,
       and return the minimum culling distance returned by any of
       the per-prim lambdas */
    auto leafCode
      = [lambdaToExecuteForEachCandidate,dbg](const uint32_t *leafPrims,
                                              size_t numPrims)->int
      {
        // if (dbg) dout << "fixedRayQuery::forEachPrim leaf " << numPrims << endl;
        for (int i=0;i<numPrims;i++) 
          if (lambdaToExecuteForEachCandidate(leafPrims[i])
              == CUBQL_TERMINATE_TRAVERSAL)
            return CUBQL_TERMINATE_TRAVERSAL;
        return CUBQL_CONTINUE_TRAVERSAL;
      };
    forEachLeaf(leafCode,bvh,ray,dbg);
  }

  template<typename Lambda, typename T, int D>
  inline __cubql_both
  void fixedRayQuery::forEachLeaf(const Lambda &lambdaToCallOnEachLeaf,
                                  cuBQL::bvh_t<T, D> bvh,
                                  cuBQL::ray3f ray,
                                  bool dbg)
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
        bool o0 = rayIntersectsBox(ray,n0.bounds);
        bool o1 = rayIntersectsBox(ray,n1.bounds);
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
          = lambdaToCallOnEachLeaf(bvh.primIDs+node.offset,node.count);
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

  template<int N>
  struct ChildOrder {
    inline __cubql_both void clear(int i) { v[i] = (uint64_t)-1; }
    inline __cubql_both void set(int i, float dist, uint32_t payload) {
        v[i] = (uint64_t(__float_as_int(dist)) << 32) | payload;
    }
    uint64_t v[N];
  };

  template<int N>
  inline __cubql_both void sort(ChildOrder<N>& children)
  {
#pragma unroll
      for (int i = N - 1; i > 0; --i) {
#pragma unroll
          for (int j = 0; j < i; j++) {
              uint64_t c0 = children.v[j + 0];
              uint64_t c1 = children.v[j + 1];
              children.v[j + 0] = min(c0, c1);
              children.v[j + 1] = max(c0, c1);
          }
      }
  }

  template<typename Lambda, typename T, int D, int W>
  inline __cubql_both
  void fixedRayQuery::forEachLeaf(const Lambda& lambdaToCallOnEachLeaf,
                                  cuBQL::WideBVH<T, D, W> bvh,
                                  cuBQL::ray3f ray,
                                  bool dbg)
  {
      using node_t = typename WideBVH<T, D, W>::node_t;

      int traversalStack[64], * stackPtr = traversalStack;
      int nodeID = 0;

      if (ray.direction.x == (T)0) ray.direction.x = T(1e-20);
      if (ray.direction.y == (T)0) ray.direction.y = T(1e-20);
      if (ray.direction.z == (T)0) ray.direction.z = T(1e-20);
      vec_t<T, 3> rcp_dir = rcp(ray.direction);

      ChildOrder<W> childOrder;

      // ------------------------------------------------------------------
      // traverse until there's nothing left to traverse:
      // ------------------------------------------------------------------
      while (true) {
          while (true) {
              while (nodeID == -1) {
                  if (stackPtr == traversalStack)
                      return;
                  nodeID = *--stackPtr;
                  // pop....
              }
              if (nodeID & (1 << 31))
                  break;

              node_t const& node = bvh.nodes[nodeID];
#pragma unroll
              for (int c = 0; c < W; c++) {
                  const auto child = node.children[c];
                  if (!node.children[c].valid)
                      childOrder.clear(c);
                  else {
                      float dist2;
                      bool o = rayIntersectsBox(dist2, ray, rcp_dir, node.children[c].bounds);
                      if (!o)
                          childOrder.clear(c);
                      else {
                          uint32_t payload = child.count ?
                              ((1 << 31) | (nodeID << log_of<W>::value) | c) : child.offset;
                          childOrder.set(c, dist2, payload);
                      }
                  }
              }
              sort(childOrder);
#pragma unroll
              for (int c = W - 1; c > 0; --c) {
                  uint64_t coc = childOrder.v[c];
                  if (coc != uint64_t(-1)) {
                      *stackPtr++ = coc;
                      // if (stackPtr - stackBase == stackSize)
                      //   printf("stack overrun!\n");
                  }
              }
              if (childOrder.v[0] == uint64_t(-1)) {
                  nodeID = -1;
                  continue;
              }
              nodeID = uint32_t(childOrder.v[0]);
          }

          int c = nodeID & ((1 << log_of<W>::value) - 1);
          int n = (nodeID & 0x7fffffff) >> log_of<W>::value;
          int offset = bvh.nodes[n].children[c].offset;
          int count = bvh.nodes[n].children[c].count;

          if (count != 0) {
              // we're at a valid leaf: call the lambda and see if that gave
              // us a new, closer cull radius
              int leafResult
                  = lambdaToCallOnEachLeaf(bvh.primIDs + offset, count);
              if (leafResult == CUBQL_TERMINATE_TRAVERSAL)
                  return;
          }
          nodeID = -1;
      }
  }

  /*! this query assumes lambads that return CUBQL_CONTINUE_TRAVERSAL
    or CUBQL_TERMINATE_TRAVERSAL */
  template<typename Lambda, typename bvh_t>
  inline __cubql_both
  void fixedRayQuery::forEachPrim(const Lambda &lambdaToExecuteForEachCandidate,
                                  bvh_t bvh,
                                  cuBQL::ray3f ray,
                                  bool dbg)
  {
    /* the code we want to have executed for each leaf that may
       contain candidates. we loop over each prim in a given leaf,
       and return the minimum culling distance returned by any of
       the per-prim lambdas */
    auto leafCode
      = [lambdaToExecuteForEachCandidate,dbg](const uint32_t *leafPrims,
                                              size_t numPrims)->int
      {
        // if (dbg) dout << "fixedRayQuery::forEachPrim leaf " << numPrims << endl;
        for (int i=0;i<numPrims;i++) 
          if (lambdaToExecuteForEachCandidate(leafPrims[i])
              == CUBQL_TERMINATE_TRAVERSAL)
            return CUBQL_TERMINATE_TRAVERSAL;
        return CUBQL_CONTINUE_TRAVERSAL;
      };
    forEachLeaf(leafCode,bvh,ray,dbg);
  }
    
  template<typename Lambda, typename T, int D, typename ray_t>
  inline __cubql_both
  float shrinkingRayQuery::forEachLeaf(const Lambda &lambdaToCallOnEachLeaf,
                                       bvh_t<T, D> bvh,
                                       ray_t ray,
                                       bool dbg)
  {
    using node_t = typename bvh_t<T, D>::node_t;
    struct StackEntry {
      uint32_t idx;
    };
    typename node_t::Admin traversalStack[64], *stackPtr = traversalStack;
    typename node_t::Admin node = bvh.nodes[0].admin;

    if (ray.direction.x == (T)0) ray.direction.x = T(1e-20);
    if (ray.direction.y == (T)0) ray.direction.y = T(1e-20);
    if (ray.direction.z == (T)0) ray.direction.z = T(1e-20);
    vec_t<T,3> rcp_dir = rcp(ray.direction);
      
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
        node_t n0 = bvh.nodes[n0Idx];
        node_t n1 = bvh.nodes[n1Idx];
        float node_t0 = 0.f, node_t1 = 0.f;
        bool o0 = rayIntersectsBox(node_t0,ray,rcp_dir,n0.bounds);
        bool o1 = rayIntersectsBox(node_t1,ray,rcp_dir,n1.bounds);

        if (o0) {
          if (o1) {
            *stackPtr++ = (node_t0 < node_t1) ? n1.admin : n0.admin;
            node = (node_t0 < node_t1) ? n0.admin : n1.admin;
          } else {
            node = n0.admin;
          }
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
        ray.tMax
          = lambdaToCallOnEachLeaf(bvh.primIDs+node.offset,node.count);
      }
      // ------------------------------------------------------------------
      // pop next un-traversed node from stack, discarding any nodes
      // that are more distant than whatever query radius we now have
      // ------------------------------------------------------------------
      if (stackPtr == traversalStack)
        return ray.tMax;
      node = *--stackPtr;
    }
  }

  template<typename Lambda, typename T, int D, int W, typename ray_t>
  inline __cubql_both
  float shrinkingRayQuery::forEachLeaf(const Lambda& lambdaToCallOnEachLeaf,
                                       WideBVH<T, D, W> bvh,
                                       ray_t ray,
                                       bool dbg)
  {
      using node_t = typename WideBVH<T, D, W>::node_t;

      int traversalStack[64], * stackPtr = traversalStack;
      int nodeID = 0;

      if (ray.direction.x == (T)0) ray.direction.x = T(1e-20);
      if (ray.direction.y == (T)0) ray.direction.y = T(1e-20);
      if (ray.direction.z == (T)0) ray.direction.z = T(1e-20);
      vec_t<T, 3> rcp_dir = rcp(ray.direction);

      ChildOrder<W> childOrder;

      // ------------------------------------------------------------------
      // traverse until there's nothing left to traverse:
      // ------------------------------------------------------------------
      while (true) {
          while (true) {
              while (nodeID == -1) {
                  if (stackPtr == traversalStack)
                      return ray.tMax;
                  nodeID = *--stackPtr;
                  // pop....
              }
              if (nodeID & (1 << 31))
                  break;

              node_t const& node = bvh.nodes[nodeID];
#pragma unroll
              for (int c = 0; c < W; c++) {
                  const auto child = node.children[c];
                  if (!node.children[c].valid)
                      childOrder.clear(c);
                  else {
                      float dist2;
                      bool o = rayIntersectsBox(dist2, ray, rcp_dir, node.children[c].bounds);
                      if (!o)
                          childOrder.clear(c);
                      else {
                          uint32_t payload = child.count ?
                              ((1 << 31) | (nodeID << log_of<W>::value) | c) : child.offset;
                          childOrder.set(c, dist2, payload);
                      }
                  }
              }
              sort(childOrder);
#pragma unroll
              for (int c = W - 1; c > 0; --c) {
                  uint64_t coc = childOrder.v[c];
                  if (coc != uint64_t(-1)) {
                      *stackPtr++ = coc;
                      // if (stackPtr - stackBase == stackSize)
                      //   printf("stack overrun!\n");
                  }
              }
              if (childOrder.v[0] == uint64_t(-1)) {
                  nodeID = -1;
                  continue;
              }
              nodeID = uint32_t(childOrder.v[0]);
          }

          int c = nodeID & ((1 << log_of<W>::value) - 1);
          int n = (nodeID & 0x7fffffff) >> log_of<W>::value;
          int offset = bvh.nodes[n].children[c].offset;
          int count = bvh.nodes[n].children[c].count;

          if (count != 0) {
              // we're at a valid leaf: call the lambda and see if that gave
              // us a new, closer cull radius
              ray.tMax
                  = lambdaToCallOnEachLeaf(bvh.primIDs + offset, count);
          }
          nodeID = -1;
      }
      return T(CUBQL_INF);
  }

  template<typename Lambda, typename bvh_t, typename ray_t>
  inline __cubql_both
  void shrinkingRayQuery::forEachPrim(const Lambda &lambdaToExecuteForEachCandidate,
                                      bvh_t bvh,
                                      ray_t &ray,
                                      bool dbg)
  {
    auto perLeaf = [dbg,bvh,&ray,lambdaToExecuteForEachCandidate]
      (const uint32_t *leaf, int count) {
      for (int i=0;i<count;i++)
        ray.tMax = lambdaToExecuteForEachCandidate(leaf[i]);
      return ray.tMax;
    };
    shrinkingRayQuery::forEachLeaf(perLeaf,bvh,ray,dbg);
  }



  /*! two-level BVH ray traversal, where the BVH is made up of a
    "TLAS" (top-level acceleration structure) that itself contains
    objects with "BLAS"es (bottom-level acceleration
    structures). One of the lambdas describes what happens when a
    ray enters a leaf in a BLAS (just like in the single-level BVH
    traversal; the other describes what happens when a ray needs
    to transition from TLAS to BLAS. That second lambda can modify
    the current ray's org and dir to transform it into the BLASes
    coordinate frame where required (transforming back is not
    required, cubql will save/restore the original ray as
    required), and is supposed to return a new bvh_t to be
    traversed in the BLAS */
  template<typename EnterBlasLambda,
           typename LeaveBlasLambda,
           typename ProcessLeafLambda,
           typename bvh_t, typename ray_t>
  inline __cubql_both
  void shrinkingRayQuery::twoLevel::
  forEachLeaf(const EnterBlasLambda   &enterBlas,
              const LeaveBlasLambda   &leaveBlas,
              const ProcessLeafLambda &processLeaf,
              bvh_t bvh,
              /*! REFERENCE to a ray, so 'enterBlas()' can modify it */
              ray_t &ray,
              bool dbg)
  {
    using node_t = typename bvh_t::node_t;
    using T = typename bvh_t::scalar_t;
    struct StackEntry {
      uint32_t idx;
    };
    enum { STACK_DEPTH=128 };
    typename node_t::Admin
      traversalStack[STACK_DEPTH],
      *stackPtr = traversalStack,
      *blasStackBase = nullptr;
    typename node_t::Admin node = bvh.nodes[0].admin;

    node_t   *tlasSavedNodePtr = 0;
    uint32_t *tlasSavedPrimIDs = 0;
    vec_t<T,3> saved_dir, saved_org;
    
    if (ray.direction.x == (T)0) ray.direction.x = T(1e-20);
    if (ray.direction.y == (T)0) ray.direction.y = T(1e-20);
    if (ray.direction.z == (T)0) ray.direction.z = T(1e-20);
    vec_t<T,3> rcp_dir = rcp(ray.direction);
      
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
        if (dbg) printf("node %i.%i\n",(int)node.offset,(int)node.count);
        if (node.count != 0) {
          // it's a boy! - seriously: this is not a inner node; so
          // we're either at a final leaf, or at an instance node
          if (blasStackBase != nullptr)
            // it's a real leaf, in a blas; break out here and let
            // leaf code trigger.
            break;
          // it's not a real leaf, so this must be a instance node
          tlasSavedNodePtr = bvh.nodes;
          tlasSavedPrimIDs = bvh.primIDs;
          if (node.count != 1)
            printf("TWO-LEVEL BVH MUST BE BUILT WITH 1 PRIM PER LEAF!\n");
          if (dbg)
            printf("inner-leaf primIDs %p ofs %i count %i\n",
                   bvh.primIDs,
                   (int)node.offset,
                   (int)node.count);
              
          int instID
            = bvh.primIDs
            ? bvh.primIDs[node.offset]
            : node.offset;

          saved_dir = ray.direction;
          saved_org = ray.origin;
          bvh_t blas;
          ray_t transformed_ray = ray;
          enterBlas(transformed_ray,blas,instID);
          ray.origin = transformed_ray.origin;
          ray.direction = transformed_ray.direction;
          if (ray.direction.x == (T)0) ray.direction.x = T(1e-20);
          if (ray.direction.y == (T)0) ray.direction.y = T(1e-20);
          if (ray.direction.z == (T)0) ray.direction.z = T(1e-20);
          rcp_dir = rcp(ray.direction);
          bvh.nodes     = blas.nodes;
          bvh.primIDs   = blas.primIDs;
          blasStackBase = stackPtr;
          node          = bvh.nodes[0].admin;
          // now check if those blas root node is _also_ a leaf:
          if (node.count != 0)
            break;
          if (dbg) printf("new node %i.%i\n",(int)node.offset,(int)node.count);
        }          

        uint32_t n0Idx = (uint32_t)node.offset+0;
        uint32_t n1Idx = (uint32_t)node.offset+1;
        node_t n0 = bvh.nodes[n0Idx];
        node_t n1 = bvh.nodes[n1Idx];
        float node_t0 = 0.f, node_t1 = 0.f;
        bool o0 = rayIntersectsBox(node_t0,ray,rcp_dir,n0.bounds);
        bool o1 = rayIntersectsBox(node_t1,ray,rcp_dir,n1.bounds);

        if (dbg) {
          // dout << " node L " << n0.bounds << "\n";
          // dout << " node R " << n1.bounds << "\n";
          printf("children L hit %i dist %f R hit %i dist %f\n",
                 int(o0),node_t0,
                 int(o1),node_t1);
        }
        if (o0) {
          if (o1) {
            if (stackPtr-traversalStack >= STACK_DEPTH) {
              return;
            }
            
            *stackPtr++ = (node_t0 < node_t1) ? n1.admin : n0.admin;
            node = (node_t0 < node_t1) ? n0.admin : n1.admin;
          } else {
            node = n0.admin;
          }
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
 
      if (node.count != 0 && blasStackBase != nullptr) {
        // we're at a valid leaf: call the lambda and see if that gave
        // us a new, closer cull radius
        if (dbg)
          printf("trav leaf-leaf primIDs %p offset %i count %i\n",
                 bvh.primIDs,(int)node.offset,(int)node.count);
        ray.tMax
          = processLeaf(bvh.primIDs,(int)node.offset,(int)node.count);
      }
      // ------------------------------------------------------------------
      // pop next un-traversed node from stack, discarding any nodes
      // that are more distant than whatever query radius we now have
      // ------------------------------------------------------------------
      if (stackPtr == blasStackBase) {
        leaveBlas();
        ray.direction = saved_dir;
        ray.origin    = saved_org;
        rcp_dir = rcp(ray.direction);
        blasStackBase = nullptr;
        bvh.nodes   = tlasSavedNodePtr;
        bvh.primIDs = tlasSavedPrimIDs;
      }
      if (stackPtr == traversalStack)
        return;// ray.tMax;
      node = *--stackPtr;
    }
  }
  
  /*! two-level BVH ray traversal, where the BVH is made up of a
    "TLAS" (top-level acceleration structure) that itself contains
    objects with "BLAS"es (bottom-level acceleration
    structures). One of the lambdas describes what happens when a
    ray enters a leaf in a BLAS (just like in the single-level BVH
    traversal; the other describes what happens when a ray needs
    to transition from TLAS to BLAS. That second lambda can modify
    the current ray's org and dir to transform it into the BLASes
    coordinate frame where required (transforming back is not
    required, cubql will save/restore the original ray as
    required), and is supposed to return a new bvh_t to be
    traversed in the BLAS */
  template<typename EnterBlasLambda,
           typename LeaveBlasLambda,
           typename IntersectPrimLambda,
           typename bvh_t, typename ray_t>
  inline __cubql_both
  void shrinkingRayQuery::twoLevel::
  forEachPrim(const EnterBlasLambda     &enterBlas,
              const LeaveBlasLambda     &leaveBlas,
              const IntersectPrimLambda &intersectPrim,
              bvh_t bvh,
              /*! REFERENCE to a ray, so 'enterBlas()' can modify it */
              ray_t &ray,
              bool dbg)
  {
    auto perLeaf = [dbg,&bvh,&ray,
                    enterBlas,
                    leaveBlas,
                    intersectPrim]
      (const uint32_t *primIDs, int offset, int count) {
      if (dbg) printf("AT LEAF!, primIDs %p, count %i\n",bvh.primIDs,count);
      for (int i=0;i<count;i++) { 
        int primIdx = offset+i;
        if (primIDs) primIdx = primIDs[primIdx];
        ray.tMax = min(ray.tMax,intersectPrim(primIdx));
      }
      if (dbg) printf("LEAVING LEAF! t = %f\n",ray.tMax);
      return ray.tMax;
    };
    shrinkingRayQuery::twoLevel::forEachLeaf
      (enterBlas,leaveBlas,perLeaf,bvh,ray,dbg);
  }
  
} // ::cuBQL
