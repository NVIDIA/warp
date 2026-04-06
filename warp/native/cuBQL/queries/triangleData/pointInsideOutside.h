// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file queries/triangleData/pointInsideOutside.h computes whether a
  3D point is inside/outside a (supposedly closed) triangle
  mesh. Relies on crossing-count kernel to compute odd/equal
  crossing counts for semi-infinite ray(s) starting at that point */
#pragma once

#include "cuBQL/bvh.h"
#include "cuBQL/queries/triangleData/crossingCount.h"

namespace cuBQL {
  namespace triangles {

    // =============================================================================
    // *** INTERFACE ***
    // =============================================================================

    /*! given a bvh ('bvh') built over a supposedly closed triangle
      mesh (with a triangle accessor function
      getTriangle()->Triangle), compute whether a given point
      `queryPoint` is inside or outside the surface mesh
        
      getTriangle is lambda getTriangle(uint32_t triID)->Triangle
    */
    template<typename GetTriangleLambda>
    inline __cubql_both
    bool pointIsInsideSurface(bvh3f bvh,
                              const GetTriangleLambda getTriangle,
                              vec3f queryPoint,
                              bool dbg=false);

    // =============================================================================
    // *** IMPLEMENTATION ***
    // =============================================================================

    template<typename GetTriangleLambda>
    inline __cubql_both
    bool pointIsInsideSurface(bvh3f bvh,
                              const GetTriangleLambda getTriangle,
                              vec3f queryPoint,
                              bool dbg)
    {
      /*! we trace 6 rays - one per principle axis - using the
        AxisAlignedRay rayquery. In theory, if the mesh is closed then
        these 6 calls should all agree; but in practice there's always
        some holes or double counting when rays going right through
        vertices or edges, so we just trace one ray in each direction
        and take a majority vote. */
      int n0 = signedCrossingCount(bvh,getTriangle,AxisAlignedRay<0,-1>(queryPoint),dbg);
      int p0 = signedCrossingCount(bvh,getTriangle,AxisAlignedRay<0,+1>(queryPoint),dbg);
      int n1 = signedCrossingCount(bvh,getTriangle,AxisAlignedRay<1,-1>(queryPoint),dbg);
      int p1 = signedCrossingCount(bvh,getTriangle,AxisAlignedRay<1,+1>(queryPoint),dbg);
      int n2 = signedCrossingCount(bvh,getTriangle,AxisAlignedRay<2,-1>(queryPoint),dbg);
      int p2 = signedCrossingCount(bvh,getTriangle,AxisAlignedRay<2,+1>(queryPoint),dbg);
      int numIn = p0+p1+p2+n0+n1+n2;
      if (dbg) printf("inside results %i %i %i %i %i %i\n",
                      n0,n1,n2,p0,p1,p2);

      // if (numIn != 0 && numIn != 6) {
      //   printf("disagreement: inside results %i %i %i %i %i %i\n",
      //          n0,n1,n2,p0,p1,p2);
      // }
      return /* take a majority vote ... */numIn > 3; // == 3;//> 3;
    }
    
  } // ::cuBQL::triangles
} // ::cuBQL
