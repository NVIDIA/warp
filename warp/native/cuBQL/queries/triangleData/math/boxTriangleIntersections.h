// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cuBQL/queries/triangleData/Triangle.h"
#include "cuBQL/math/box.h"

namespace cuBQL {
  namespace triangles {
    // =============================================================================
    // *** INTERFACE ***
    // =============================================================================

    /*! checks if a given triangle A-B-C *intersects* (ie, _not_ only
        "overlaps its bounding box) a given box. Only returns
        true/false, does not produce any actual intersection shape */
    inline __cubql_both
    bool triangleIntersectsBox(/*! the query box to check against */
                               box3f queryBox,
                               /*! the four vertices of the triangle
                                 to test */
                               Triangle triangle);
    inline __cubql_both
    bool triangleIntersectsBox(/*! the four vertices of the triangle
                                 to test */
                               Triangle triangle,
                               /*! the query box to check against */
                               box3f queryBox);
      
    /*! checks if a given triangle A-B-C *intersects* (ie, _not_ only
        "overlaps its bounding box) a given box. Only returns
        true/false, does not produce any actual intersection shape */
    inline __cubql_both
    bool triangleIntersectsBox(/*! the query box to check against */
                               box3f queryBox,
                               /*! the four vertices of the triangle
                                 to test */
                               vec3f a, vec3f b, vec3f c);
    
    // =============================================================================
    // *** IMPLEMENTATION ***
    // =============================================================================

    /*! given a line segment A-B, and a parameter interval [t0,t1]
      along that line (t=0 being at A, and t=1 being at B), clip
      that interval against the given DIM'th dimensional "slab" of
      the given query box. This modifies the [t0,t1] interval in
      place such that after returning from his function [t0,t1] is
      the intersection of the input interval and this dimension's
      slab interval */
    template</*! dimension to perform this clip in */int DIM>
    inline __cubql_both
    void clipLineSegmentToSlabOfBox(/*! the query box to clip
                                      against */
                                    box3f queryBox,
                                    /*! the line segment to
                                      consider */
                                    vec3f a, vec3f b,
                                    /*! the input (and output)
                                      interval along a-b */
                                    float &t0, float &t1) 
    {
      float lower_k, upper_k, a_k, b_k;
      if (DIM==0) {
        lower_k = queryBox.lower.x;
        upper_k = queryBox.upper.x;
        a_k = a.x;
        b_k = b.x;
      } else if (DIM==1) {
        lower_k = queryBox.lower.y;
        upper_k = queryBox.upper.y;
        a_k = a.y;
        b_k = b.y;
      } else {
        lower_k = queryBox.lower.z;
        upper_k = queryBox.upper.z;
        a_k = a.z;
        b_k = b.z;
      }

      if (a_k == b_k) {
        /* if a_k == b_k, then the line segment is exactly parallel to
           the dimension we're testing; in this case it's either
           entirely within that slab (iff lower_k <= a_k <= upper_k),
           or entirely outside */
        if (a_k > upper_k || a_k < lower_k) {
          t0 = PosInfTy(); t1 = NegInfTy();
        }
      } else {
        /* line segment is not parallel to the slab planes, so we can
           compute (t_lower,t_upper) distances to the two planes
           bounding this slab ... */
        float t_lower = (lower_k - a_k) / (b_k - a_k);
        float t_upper = (upper_k - a_k) / (b_k - a_k);
        /* ... and then clip this slab-segment against [t0,t1] segment */
        t0 = max(t0,min(t_lower,t_upper));
        t1 = min(t1,max(t_lower,t_upper));
      }
    }
       

    /*! checks if a given line segment A-B intersects with a box */
    inline __cubql_both
    bool lineSegmentIntersectsBox(/*! the query box to check against */
                                  box3f queryBox,
                                  /*! the end points of the line
                                    segment to test against */
                                  vec3f a,
                                  vec3f b) 
    {
      /* initial t-range of segment to clip to box, staring with
         [0,1] */
      float t0 = 0.f;
      float t1 = 1.f;
      
      /* clip that segment to X-slab of box */
      clipLineSegmentToSlabOfBox<0>(queryBox,a,b,t0,t1);
      
      /* clip that segment to Y-slab of box */
      clipLineSegmentToSlabOfBox<1>(queryBox,a,b,t0,t1);
      
      /* clip that segment to Z-slab of box */
      clipLineSegmentToSlabOfBox<2>(queryBox,a,b,t0,t1);
      
      /* segment intersects if and only if the clipped range is still
         a non-inverted interval */
      return t0 <= t1;
    }


      
    template<int X, int Y>
    inline __cubql_both
    bool diagonalIntersectsTriangle(/*! the query box being tested */
                                    box3f box,
                                    /* the triangle we're testing against */
                                    vec3f a, vec3f b, vec3f c) 
    {
      // compute corners of given diagonal; one on front side and
      // one on back side
      vec3f front(X?box.lower.x:box.upper.x,
                  Y?box.lower.y:box.upper.y,
                  box.lower.z);
      vec3f back(X?box.upper.x:box.lower.x,
                 Y?box.upper.y:box.lower.y,
                 box.upper.z);

      // iw - MIGHT want to subtract 'front' from all of
      // a,b,c,front,and back, to get better numerical accuracy
        
      // compute plane equation of triangle, and check if front and
      // back points lie on opposing signs of that plane (because if
      // not, it can't intersect)
      vec3f n = cross(b-a,c-a);
      if (n == vec3f(0.f)) {
        return false;
      }
        
      float p_front = dot(front-a,n);
      float p_back  = dot(back-a,n);
      if (p_front * p_back > 0.f) {
        /* both on same side */ 
        return false;
      }

      // finally, pluecker test of winding order
      vec3f org = front;
      vec3f dir = back - front;
        
      auto pluecker=[](vec3f a0, vec3f a1, vec3f b0, vec3f b1) 
      { return dot(a1-a0,cross(b1,b0))+dot(b1-b0,cross(a1,a0)); };
        
      // compute pluecker coordinates dot product of all edges wrt x
      // axis ray. since the ray is mostly 0es and 1es, this should all
      // evaluate to some fairly simple expressions
      float sx = pluecker(org,org+dir,a,b);
      float sy = pluecker(org,org+dir,b,c);
      float sz = pluecker(org,org+dir,c,a);
        
      // for ray to be inside edges it must have all positive or all
      // negative pluecker winding order
      auto min3=[](float x, float y, float z)
      { return min(min(x,y),z); };
      auto max3=[](float x, float y, float z)
      { return max(max(x,y),z); };

      float lo = min3(sx,sy,sz);
      float hi = max3(sx,sy,sz);
      bool result = (min3(sx,sy,sz) >= 0.f || max3(sx,sy,sz) <= 0.f);
      return result;
    }
    
    /*! checks if a given triangle A-B-C overlaps a given box */
    inline __cubql_both
    bool triangleIntersectsBox(/*! the query box to check against */
                               box3f queryBox,
                               /*! the four vertices of the triangle
                                 to test */
                               vec3f a, vec3f b, vec3f c) 
    {
      /*! check if any of the triangle lineSegments is at least partially
        inside the box; if any one is - even partially - the
        triangle is inside the box */
      if (lineSegmentIntersectsBox(queryBox,a,b)) return true;
      if (lineSegmentIntersectsBox(queryBox,b,c)) return true;
      if (lineSegmentIntersectsBox(queryBox,c,a)) return true;
        
      /*! check the four possible box diagonals on whether they
        intersect the triangle - this catches the case where a
        triangles' edges "surround" the box, but some of the
        triangle's inside is within the box */
      if (diagonalIntersectsTriangle<0,0>(queryBox,a,b,c)) return true;
      if (diagonalIntersectsTriangle<0,1>(queryBox,a,b,c)) return true;
      if (diagonalIntersectsTriangle<1,0>(queryBox,a,b,c)) return true;
      if (diagonalIntersectsTriangle<1,1>(queryBox,a,b,c)) return true;

      /*! if neither of the above two cases occurred, the triangle
        must be outside the box */
      return false;
    }

    /*! checks if a given triangle A-B-C overlaps a given box */
    inline __cubql_both
    bool triangleIntersectsBox(/*! the query box to check against */
                               box3f queryBox,
                               /*! the four vertices of the triangle
                                 to test */
                               Triangle triangle)
    { return triangleIntersectsBox(queryBox,triangle.a,triangle.b,triangle.c); }
    
    /*! checks if a given triangle A-B-C overlaps a given box */
    inline __cubql_both
    bool triangleIntersectsBox(/*! the four vertices of the triangle
                                 to test */
                               Triangle triangle,
                               /*! the query box to check against */
                               box3f queryBox)
    { return triangleIntersectsBox(queryBox,triangle.a,triangle.b,triangle.c); }
    
  } // ::cuBQL::triangles  
} // ::cuBQL

  
