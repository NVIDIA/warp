// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cuBQL/queries/triangleData/Triangle.h"

namespace cuBQL {
  /*! \namespace triangles for any queries operating on triangle model data */
  namespace triangles {

    // =============================================================================
    // *** INTERFACE ***
    // =============================================================================

    struct PointToTriangleTestResult {
      /*! (square!) distance between query point and closest point on triangle.*/
      float sqrDist = PosInfTy();
      /*! the actual 3D point that's closest on the triangle */
      vec3f P;
    };

    /*! compute one point-triangle distance test, fill in the result
        value, and return it*/
    inline __cubql_both
    PointToTriangleTestResult computeClosestPoint(Triangle triangle,
                                                  vec3f queryPoint,
                                                  bool dbg=false);
    /*! compute one point-triangle distance test, fill in the result
        value, and return it*/
    inline __cubql_both
    PointToTriangleTestResult computeClosestPoint(vec3f queryPoint,
                                                  Triangle triangle,
                                                  bool dbg=false);

    /*! given a pre-initialized 'PointToTriangleTestResult' struct -
        that may already contain some other triangle's distance test -
        compute a distance test, and check if is closer than the hit
        already stored. If not, 'existingResult' will remain
        unmodified and this fct returns false; if true, the result
        will be updated to the new hit, and this fct return true */
    inline __cubql_both
    bool computeClosestPoint(PointToTriangleTestResult &existingResult,
                             Triangle triangle,
                             vec3f queryPoint,
                             bool dbg=false);
    
    
    // =============================================================================
    // *** IMPLEMENTATION ***
    // =============================================================================
    
    namespace pointToTriangleTest {
      /*! helper struct for a edge with double coordinates; mainly
        exists for the Edge::closestPoint test method */
      struct Edge {
        inline __cubql_both
        Edge(vec3f a, vec3f b) : a(a), b(b) {}
        
        /*! compute point-to-distance for this triangle; returns true if the
          result struct was updated with a closer point than what it
          previously contained */
        inline __cubql_both
        bool closestPoint(PointToTriangleTestResult &result,
                          const vec3f &referencePointToComputeDistanceTo,
                          bool dbg=0) const;
        
        const vec3f a, b; 
      };
      
      
      /*! compute point-to-distance for this edge; returns true if the
        result struct was updated with a closer point than what it
        previously contained */
      inline __cubql_both
      bool Edge::closestPoint(PointToTriangleTestResult &result,
                              const vec3f &p,
                              bool dbg) const
      {
        float t = dot(p-a,b-a) / dot(b-a,b-a);
        t = clamp(t);
        vec3f cp = a + t * (b-a);
        float sqrDist = dot(cp-p,cp-p);
        if (sqrDist >= result.sqrDist) 
          return false;
        
        result.sqrDist = sqrDist;
        result.P       = cp;
        return true;
      }
      
      /*! computes the querypoint-triangle test for a given pair of
        triangle and query point; returns true if this _was_ closer
        than what 'this' stored before (and if so, 'this' was
        updated); if this returns false the computed distance was
        greater than the already stored distance, and 'this' was
        left unmodified */
      inline __cubql_both
      bool computeOneIntersection(PointToTriangleTestResult &result,
                                  const cuBQL::Triangle triangle,
                                  const cuBQL::vec3f    queryPoint,
                                  bool dbg)
      {
        if (dbg) printf("testing triangle ...\n");
        const vec3f a = triangle.a;
        const vec3f b = triangle.b;
        const vec3f c = triangle.c;
        vec3f N = cross(b-a,c-a);
        bool projectsOutside
          =  (N == vec3f(0.f,0.f,0.f))
          || (dot(queryPoint-a,cross(b-a,N)) >= 0.f)
          || (dot(queryPoint-b,cross(c-b,N)) >= 0.f)
          || (dot(queryPoint-c,cross(a-c,N)) >= 0.f);
        if (projectsOutside) {
          return
            Edge(a,b).closestPoint(result,queryPoint) |
            Edge(b,c).closestPoint(result,queryPoint) |
            Edge(c,a).closestPoint(result,queryPoint);
        } else {
          N = normalize(N);
          float signed_dist = dot(queryPoint-a,N);
          float sqrDist = signed_dist*signed_dist;
          if (sqrDist >= result.sqrDist) return false;
          result.sqrDist = sqrDist;
          result.P       = queryPoint - signed_dist * N;
          return true;
        }
      }

    } // ::cuBQL::triangles::pointToTriangleTest

    inline __cubql_both
    PointToTriangleTestResult computeClosestPoint(Triangle triangle,
                                                  vec3f queryPoint,
                                                  bool dbg)
    {
      PointToTriangleTestResult result;
      pointToTriangleTest::computeOneIntersection(result,triangle,queryPoint,dbg);
      return result;
    }

    inline __cubql_both
    PointToTriangleTestResult computeClosestPoint(vec3f queryPoint,
                                                  Triangle triangle,
                                                  bool dbg)
    { return computeClosestPoint(triangle,queryPoint,dbg); }
    
    inline __cubql_both
    bool computeClosestPoint(PointToTriangleTestResult &existingResult,
                             Triangle triangle,
                             vec3f queryPoint,
                             bool dbg)
    {
      return pointToTriangleTest::computeOneIntersection
        (existingResult,triangle,queryPoint,dbg);
    }
    
  }
}
