// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cuBQL/bvh.h"
// the kind of model data we operate on
#include "cuBQL/queries/triangleData/math/pointToTriangleDistance.h"
// the kind of traversal we need for this query
#include "cuBQL/traversal/shrinkingRadiusQuery.h"

namespace cuBQL {
  /*! \namespace triangles for any queries operating on triangle model data */
  namespace triangles {
    
    /*! result of a cpat query (cpat =
      closest-point-on-any-triangle); if no result was found within
      the specified max seach distance, triangleIex will be returned
      as -1 */
    struct CPAT : public PointToTriangleTestResult
    {
      /* INHERITED: float sqrDist = INFINITY; */
      /* INHERITED: vec3f P; */
      
      /*! index of triangle that had closest hit; -1 means 'none found
        that was closer than cut-off distance */
      int   triangleIdx = -1;
      
      /*! performs one complete query, starting with an empty CPAT
        result, traversing the BVH for the givne mesh, and processing
        every triangle that needs consideration. Only intersections
        that are < maxQueryRadius will get accepted */
      inline __cubql_both
      void runQuery(const cuBQL::vec3f       *mesh_vertices,
                    const cuBQL::vec3i       *mesh_indices,
                    const cuBQL::bvh3f        bvh,
                    const cuBQL::vec3f        queryPoint,
                    float maxQueryRadius = CUBQL_INF);
      
      /*! performs one complete query, starting with an empty CPAT
        result, traversing the BVH for the givne mesh, and processing
        every triangle that needs consideration. Only intersections
        that are < maxQueryRadius will get accepted */
      inline __cubql_both
      void runQuery(const cuBQL::Triangle    *triangles,
                    const cuBQL::bvh3f        bvh,
                    const cuBQL::vec3f        queryPoint,
                    float maxQueryRadius = CUBQL_INF);
      
    };


    // =============================================================================
    // *** IMPLEMENTATION ***
    // =============================================================================
    
    /*! performs one complete query, starting with an empty CPAT
      result, traversing the BVH for the givne mesh, and processing
      every triangle that needs consideration. Only intersections
      that are < maxQueryRadius will get accepted */
    inline __cubql_both
    void CPAT::runQuery(const cuBQL::Triangle *triangles,
                        const cuBQL::bvh3f     bvh,
                        const cuBQL::vec3f     queryPoint,
                        float                  maxQueryRadius)
    {
      triangleIdx = -1;
      sqrDist     = maxQueryRadius*maxQueryRadius;
      auto perPrimitiveCode
        = [bvh,triangles,queryPoint,this]
        (uint32_t triangleIdx)->float
        {
          const Triangle triangle = triangles[triangleIdx];
          if (cuBQL::triangles::computeClosestPoint(*this,triangle,queryPoint))
            this->triangleIdx = triangleIdx;
          /*! the (possibly new?) max cut-off radius (squared, as
              traversals operate on square distances!) */
          return this->sqrDist;
        };
      // careful: traversals operate on the SQUARE radii
      const float maxQueryRadiusDistance
        = maxQueryRadius * maxQueryRadius;
      cuBQL::shrinkingRadiusQuery::forEachPrim
        (/* what we want to execute for each candidate: */perPrimitiveCode,
         /* what we're querying into*/bvh,
         /* where we're querying */queryPoint,
         /* initial maximum search radius */maxQueryRadiusDistance
         );
    }
    
  } // ::cuBQL::triangles
} // ::cuBQL
