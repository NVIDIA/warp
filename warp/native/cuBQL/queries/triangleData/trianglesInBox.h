// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file queries/triangles/trianglesInBox.h implements a kernel that
    allows for checking whether a given query box intersects any
    triangle of a given model */

#pragma once

#include "cuBQL/traversal/rayQueries.h"
// the kind of model data we operate on
#include "cuBQL/queries/triangleData/Triangle.h"
#include "cuBQL/queries/triangleData/math/boxTriangleIntersections.h"

/*! \namespace cuBQL - *cu*BQL based geometric *q*ueries */
namespace cuBQL {
  namespace triangles {

    // =============================================================================
    // *** INTERFACE ***
    // =============================================================================

    /*! given a bvh built over a triangle mesh ('bvh'), and a lambda
        that let's us retrieve a triangle by its triangle index
        ('lambda'), this kernel counts how many of the underlying
        mesh's triangles are actually intersecting the query box (ie,
        not just overlapping that box with their bounding box, but
        where some actual point or portion of the triangle lies within
        the actual box).

      'maxIntersectionsToLookFor' allows for an early exit; once the
      number of triangles found reaches that value we stop any further
      queries and return this value. In particular, this allows for
      checking if *any* triangle intersects this box by calling this
      kernel with maxIntersectionsToLookFor==1

      getTriangle is lambda getTriangle(uint32_t triID)->Triangle
    */
    template<typename GetTrianglesLambda>
    inline __cubql_both
    int countTrianglesIntersectingQueryBox(const bvh3f bvh,
                                           const GetTrianglesLambda getTriangle,
                                           const box3f queryBox,
                                           int maxIntersectionsToLookFor=INT_MAX);

    /*! similar to \see countTrianglesIntersectingQueryBox, but
        doesn't perform actual trianle-box tests, and instead only
        uses cheaper (and conservative) test against the triangles'
        bounding boxes

        getTriangle is lambda getTriangle(uint32_t triID)->Triangle
    */
    template<typename GetTrianglesLambda>
    inline __cubql_both
    int countTrianglesWhoseBoundsOverlapQueryBox(const bvh3f bvh,
                                                 const GetTrianglesLambda getTriangle,
                                                 const box3f queryBox,
                                                 int maxIntersectionsToLookFor=INT_MAX);
                                            
    
    // =============================================================================
    // *** IMPLEMENTATION ***
    // =============================================================================

    template<typename GetTrianglesLambda>
    inline __cubql_both
    int countTrianglesIntersectingQueryBox(const bvh3f bvh,
                                           const GetTrianglesLambda getTriangle,
                                           const box3f queryBox,
                                           int maxIntersectionsToLookFor)
    {
      int count = 0;
      auto perTriangle
        = [&count,getTriangle,queryBox,maxIntersectionsToLookFor](uint32_t primID)
        {
          if (triangles::triangleIntersectsBox(getTriangle(primID),queryBox))
            ++count;
          return (count >= maxIntersectionsToLookFor)
            ? CUBQL_TERMINATE_TRAVERSAL
            : CUBQL_CONTINUE_TRAVERSAL;
        };
      fixedBoxQuery::forEachPrim(perTriangle,bvh,queryBox,/*dbg*/false);
      return count;
    }

    template<typename GetTrianglesLambda>
    inline __cubql_both
    int countTrianglesWhoseBoundsOverlapQueryBox(const bvh3f bvh,
                                                 const GetTrianglesLambda getTriangle,
                                                 const box3f queryBox,
                                                 int maxIntersectionsToLookFor)
    {
      int count = 0;
      auto perTriangle
        = [&count,getTriangle,queryBox,maxIntersectionsToLookFor](uint32_t primID)->int
        {
          if (getTriangle(primID).bounds().overlaps(queryBox))
            ++count;
          return (count >= maxIntersectionsToLookFor)
            ? CUBQL_TERMINATE_TRAVERSAL
            : CUBQL_CONTINUE_TRAVERSAL;
        };
      fixedBoxQuery::forEachPrim(perTriangle,bvh,queryBox,/*dbg*/false);
      return count;
    }
    
  } // ::cuBQL::triangles
} // ::cuBQL
