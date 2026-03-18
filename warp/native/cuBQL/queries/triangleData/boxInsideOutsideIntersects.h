// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file queries/triangleData/pointInsideOutside.h computes whether a
  3D point is inside/outside a (supposedly closed) triangle
  mesh. Relies on crossing-count kernel to compute odd/equal
  crossing counts for semi-infinite ray(s) starting at that point */
#pragma once

#include "cuBQL/bvh.h"
#include "cuBQL/queries/triangleData/pointInsideOutside.h"
#include "cuBQL/queries/triangleData/trianglesInBox.h"

namespace cuBQL {
  namespace triangles {
    namespace boxInsideOutsideIntersects {
      // =============================================================================
      // *** INTERFACE ***
      // =============================================================================
      
      typedef enum { INSIDE=0, OUTSIDE, INTERSECTS } result_t;
      
      template<typename GetTriangleLambda>
      inline __cubql_both
      result_t queryVsActualTriangles(bvh3f bvh,
                                      GetTriangleLambda getTriangle,
                                      box3f queryBox);
      
      template<typename GetTriangleLambda>
      inline __cubql_both
      result_t queryVsTriangleBoundingBoxes(bvh3f bvh,
                                            GetTriangleLambda getTriangle,
                                            box3f queryBox);
      
      // =============================================================================
      // *** IMPLEMENTATION ***
      // =============================================================================
      
      template<typename GetTriangleLambda>
      inline __cubql_both
      result_t queryVsActualTriangles(bvh3f bvh,
                                      GetTriangleLambda getTriangle,
                                      box3f queryBox)
      {
        if (countTrianglesIntersectingQueryBox
            (bvh,getTriangle,queryBox,/* at most */1) > 0)
          return INTERSECTS;
        return pointIsInsideSurface(bvh,getTriangle,queryBox.center())
          ? INSIDE
          : OUTSIDE;
      }
    
      template<typename GetTriangleLambda>
      inline __cubql_both
      result_t queryVsTriangleBoundingBoxes(bvh3f bvh,
                                            GetTriangleLambda getTriangle,
                                            box3f queryBox)
      {
        if (countTrianglesWhoseBoundsOverlapQueryBox
            (bvh,getTriangle,queryBox,/* at most */1) > 0)
          return INTERSECTS;
        return pointIsInsideSurface(bvh,getTriangle,queryBox.center())
          ? INSIDE
          : OUTSIDE;
      }

    } // ::cuBQL::triangles::boxInsideOutsideIntersect
  } // ::cuBQL::triangles
} // ::cuBQL
