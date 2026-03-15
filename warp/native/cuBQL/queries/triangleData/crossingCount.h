// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file queries/triangles/crossingCount Implement a "ray-triangle
  crossing count" query

  In this query, the data model is a triangle mesh (with a cuBQL BVH
  built over it, obviously), and the query is a list of ray segments
  (given by origin point and direction vector, respectively. The job
  of the query is to perform a 'crossing count', where each ray is
  traced against the triangles, and for every triangle it
  intersects, increases or decreses a given per-ray counter: -1 for
  crossing _into_ a surface (ie, the ray hits the triangle on its
  "front" side), and +1 for every crossing _out of_ the surface (if
  ray intersects triangle's back side).

*/

#pragma once

// for 'fixedRayQuery'
#include "cuBQL/traversal/rayQueries.h"
// the kind of model data we operate on
#include "cuBQL/queries/triangleData/math/rayTriangleIntersections.h"

/*! \namespace cuBQL - *cu*BQL based geometric *q*ueries */
namespace cuBQL {
  namespace triangles {

    // =============================================================================
    // *** INTERFACE ***
    // =============================================================================

    /*! returns (absolute) crossing count (ie, counting each triangle
      once, no matter how it is oriented wrt the query line) for a
      axis-aligned line that goes through point queryOrigin and is
      aligend to the axis'th coordinate axis.

      In theory, for any properly closed and outward-oriented surface
      mesh any given ray shot from a point should have a crossing
      count of +1 if that point was inside the mesh (the ray 'left'
      once more than it entered), and 0 it it was outside the mesh
      (every time it entered it also left). Note however that due to
      'funny siutations' like rays like double-counting triangles if a
      ray happens to just hit a edge or vertex this cannot be
      absoltely relied on for any single ray.

      getTriangle is a lambda getTriangle(uint32_t primID)->Triangle
    */
    template<int axis, int sign, typename GetTriangleLambda>
    inline __cubql_both
    int signedCrossingCount(bvh3f bvh,
                            GetTriangleLambda getTriangle,
                            AxisAlignedRay<axis,sign> ray);
    
    /*! defines a crossing count kernel. The struct itself defines the
      return values computed for this query, the 'compute()' method
      provides a device-side implementation of that kernel for a given
      set of inputs */
    struct CrossingCount {
      // ====================== COMPUTED VALUES ======================
      
      /* sum of all ray-triangle crossings, using "-1" for crossing
         _into_ a surface, and "+1" for crossing _out of_ a surface. in
         theory for a closed and properly outside-oriented surface and
         infinite-length query rays a point not inside the object should
         have value 0, no point should ever have values < 0 (because
         that would require a ray to enter an object and never leave
         it), and points inside the object should hae a value of exactly
         1 (because it should cross out exactly once more than it
         crosses in). Caveat: for query rays hitting edges, vertices, or
         just numerically fancy configurations this theory will probably
         not match practice :-/ */
      int crossingCount = 0;
      
      /*! total number of ray-triangle intersections, no matter which
        sign. note this *may* count certain surfaces twice if the ray
        happens to hit on an edge or vertex */
      int totalCount = 0;
      
      // ====================== ACTUAL QUERIES ======================
      
      /*! runs one complete crossing-count query; will compute
        crossing count for every triangle whose bounding box
        intersects the given ray */
      inline __cubql_both
      void runQuery(const cuBQL::TriangleMesh mesh,
                    const cuBQL::bvh3f        bvh,
                    const cuBQL::Ray          queryRay);
      
      /*! runs one complete crossing-count query; will compute
        crossing count for every triangle whose bounding box
        intersects the given ray

        'cuBQL::Triangle getTriangle(int triangleIdx)` is a lambda
        returning the triangle with given index, and allows the app to
        choose however it wants to store the triangles as long as it
        can return one on request..
      */
      template<int axis, int sign, typename GetTriangleLambda>
      inline __cubql_both
      void runQuery(const cuBQL::bvh3f        bvh,
                    const GetTriangleLambda  &getTriangle/* Triangle(*)(int) */,
                    const cuBQL::AxisAlignedRay<axis,sign> queryRay,
                    bool dbg=false);
      
      /*! runs one complete crossing-count query; will compute
        crossing count for every triangle whose bounding box
        intersects the given ray */
      template<int axis, int sign>
      inline __cubql_both
      void runQuery(const cuBQL::TriangleMesh mesh,
                    const cuBQL::bvh3f        bvh,
                    const cuBQL::AxisAlignedRay<axis,sign> queryRay,
                    bool dbg=false);
    };

    // =============================================================================
    // *** IMPLEMENTATION ***
    // =============================================================================

    /*! runs one complete crossing-count query; will compute
      crossing count for every triangle whose bounding box
      intersects the given ray

      'cuBQL::Triangle getTriangle(int triangleIdx)` is a lambda
      returning the triangle with given index, and allows the app to
      choose however it wants to store the triangles as long as it
      can return one on request..
    */
    template<int axis, int sign, typename GetTriangleLambda>
    inline __cubql_both
    void CrossingCount::runQuery(const cuBQL::bvh3f        bvh,
                                 const GetTriangleLambda  &getTriangle,
                                 const cuBQL::AxisAlignedRay<axis,sign> queryRay,
                                 bool dbg)
    {
      // reset to defaults
      *this = {0,0};
      if (dbg)
        printf("#####################\ncrossing count query axis %i sign %i\n",axis,sign);
      auto perPrimCode = [getTriangle,this,queryRay,dbg](uint32_t triangleIdx)->int {
        const Triangle triangle = getTriangle(triangleIdx);
#if 1
        if (rayIntersectsTriangle(queryRay,triangle,dbg)) {
          this->totalCount++;
          this->crossingCount
            += (dot(triangle.normal(),queryRay.direction()) > 0.f ? +1 : -1);
        }
#else
        const Ray ray = queryRay.makeRay();
        RayTriangleIntersection isec;
        if (isec.compute(ray,triangle)) {
          this->totalCount++;
          this->crossingCount
            += (dot(isec.N,ray.direction) > 0.f ? +1 : -1);
        }
#endif
        return CUBQL_CONTINUE_TRAVERSAL;
      };
      cuBQL::fixedRayQuery::forEachPrim(perPrimCode,bvh,queryRay,dbg);
    }

    
    template<int axis, int sign, typename GetTriangleLambda>
    inline __cubql_both
    int signedCrossingCount(bvh3f bvh,
                            GetTriangleLambda getTriangle,
                            AxisAlignedRay<axis,sign> queryRay,
                            bool dbg=false)
    {
      CrossingCount cc;
      // AxisAlignedRay<axis,sign> queryRay(queryPoint,0.f,+CUBQL_INF);
      cc.runQuery(bvh,getTriangle,queryRay,dbg);
      return cc.crossingCount;
    }
    
  } // ::cuBQL::triangles
} // ::cuBQL
