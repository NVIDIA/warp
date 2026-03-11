// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// for 'fixedRayQuery'
#include "cuBQL/traversal/rayQueries.h"
// for 'rayIntersectsTriangle()'
#include "cuBQL/queries/triangleData/math/rayTriangleIntersections.h"

namespace cuBQL {
  namespace triangles {

    // =============================================================================
    // *** INTERFACE ***
    // =============================================================================
    
    /*! performs line-of-sight comutation between two points, and
      returns true if both points are mutually visible (ie, _not_
      occluded by any triangle), or false if the line of sightis
      blocked by at least one triangle
        
      getTriangle is lambda getTriangle(uint32_t triID)->Triangle
    */
    template<typename GetTriangleLambda>
    inline __cubql_both
    bool pointsMutuallyVisible(bvh3f bvh,
                               GetTriangleLambda getTriangle,
                               const vec3f pointA,
                               const vec3f pointB);
    
    // =============================================================================
    // *** IMPLEMENTATION ***
    // =============================================================================
    
    template<typename GetTriangleLambda>
    inline __cubql_both
    bool pointsMutuallyVisible(bvh3f bvh,
                               GetTriangleLambda getTriangle,
                               const vec3f pointA,
                               const vec3f pointB)
    {
      bool mutuallyVisible = true;
      Ray queryRay(pointA,pointB-pointA,0.f,1.f);
      auto perTriangle=[&mutuallyVisible,getTriangle,queryRay](uint32_t primID)
      {
        Triangle triangle = getTriangle(primID);
        if (rayIntersectsTriangle(queryRay,triangle)) {
          mutuallyVisible = false;
          return CUBQL_TERMINATE_TRAVERSAL;
        }
        return CUBQL_CONTINUE_TRAVERSAL;
      };
      cuBQL::fixedRayQuery::forEachPrim(perTriangle,bvh,queryRay);
      return mutuallyVisible;
    }
  
  } // ::cuBQL::triangles
} // ::cuBQL
