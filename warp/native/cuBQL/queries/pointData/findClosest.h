// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file cuBQL/queries/points/findClosest Provides kernels for
  finding closest point(s) on point data */

#pragma once

#include "cuBQL/bvh.h"
#include "cuBQL/traversal/shrinkingRadiusQuery.h"

namespace cuBQL {
  namespace points {

    // ******************************************************************
    // INTERFACE
    // (which functions this header file provides)
    // ******************************************************************
    
    /*! given a bvh build over a set of float<N> points, perform a
      closest-point query that returns the index of the input point
      closest to the query point (if one exists within the given max
      query radius), or -1 (if not). 

      \returns Index of point in points[] array that is closest to
      query point, or -1 if no point exists within provided max query
      range

      \note If more than one point with similar closest distance
      exist, then this function will not make any guarantees as to
      which of them will be returned (though we can expect that
      succesuve such queries on the _same_ bvh will return the same
      result, different BVHs built even over the same input data may
      not)
    */
    template<typename T, int D>
    inline __cubql_both
    int findClosest(/*! binary bvh built over the given points[]
                      specfied below */
                    BinaryBVH<T,D> bvhOverPoints,
                    /*! data points that the bvh was built over */
                    const vec_t<T,D> *points,
                    /*! the query point for which we want to know the
                      result */
                    vec_t<T,D> queryPoint,
                    /*! square of the maximum query distance in which
                      this query is to look for candidates. note
                      this is the SQUARE distance */
                    float squareOfMaxQueryDistance=PosInfTy());

    /*! given a (W-wide) bvh build over a set of float<N> points,
      perform a closest-point query that returns the index of the
      input point closest to the query point (if one exists within the
      given max query radius), or -1 (if not).

      \returns Index of point in points[] array that is closest to
      query point, or -1 if no point exists within provided max query
      range

      \note If more than one point with similar closest distance
      exist, then this function will not make any guarantees as to
      which of them will be returned (though we can expect that
      succesuve such queries on the _same_ bvh will return the same
      result, different BVHs built even over the same input data may
      not)
    */
    template<typename T, int D, int W>
    inline __cubql_both
    int findClosest(/*! binary bvh built over the given points[]
                      specfied below */
                    WideBVH<T,D,W> bvhOverPoints,
                    /*! data points that the bvh was built over */
                    const vec_t<T,D> *points,
                    /*! the query point for which we want to know the
                      result */
                    vec_t<T,D> queryPoint,
                    /*! square of the maximum query distance in which
                      this query is to look for candidates. note
                      this is the SQUARE distance */
                    float squareOfMaxQueryDistance=PosInfTy());


    template<typename T, int D>
    inline __cubql_both
    int findClosest_exludeID(/*! primitive ID to _exclude_ from queries */
                             int idOfPointtoExclude,
                             /*! binary bvh built over the given points[]
                               specfied below */
                             BinaryBVH<T,D> bvhOverPoints,
                             /*! data points that the bvh was built over */
                             const vec_t<T,D> *points,
                             /*! the query point for which we want to know the
                               result */
                             vec_t<T,D> queryPoint,
                             /*! square of the maximum query distance in which
                               this query is to look for candidates. note
                               this is the SQUARE distance */
                             float squareOfMaxQueryDistance=PosInfTy());
    
    template<typename T, int D>
    /*! same as regular points::closestPoint, but excluding all data
      points that are at the query position itself */ 
    inline __cubql_device
    int findClosest_exludeSelf(/*! binary bvh built over the given points[]
                                 specfied below */
                               BinaryBVH<T,D> bvhOverPoints,
                               /*! data points that the bvh was built over */
                               const vec_t<T,D> *points,
                               /*! the query point for which we want to know the
                                 result */
                               vec_t<T,D> queryPoint,
                               /*! square of the maximum query distance in which
                                 this query is to look for candidates. note
                                 this is the SQUARE distance */
                               float squareOfMaxQueryDistance=PosInfTy());

#ifdef __CUDACC__
    // ******************************************************************
    /*! variant of cuBQL::point::findClosest() that's specialized for
      CUDA float2 type, just for convenience */
    inline __cubql_both
    int findClosest(/*! binary bvh built over the given points[]
                      specfied below */
                    bvh_float2 bvhOverPoints,
                    /*! data points that the bvh was built over */
                    const float2 *points,
                    float2 queryPoint,
                    /*! square of the maximum query distance in which
                      this query is to look for candidates. note
                      this is the SQUARE distance */
                    float squareOfMaxQueryDistance=PosInfTy());

    // ******************************************************************
    /*! variant of cuBQL::point::findClosest() that's specialized for
      CUDA float3 type, just for convenience */
    inline __cubql_both
    int findClosest(/*! binary bvh built over the given points[]
                      specfied below */
                    bvh_float3 bvhOverPoints,
                    /*! data points that the bvh was built over */
                    const float3 *points,
                    float3 queryPoint,
                    /*! square of the maximum query distance in which
                      this query is to look for candidates. note
                      this is the SQUARE distance */
                    float squareOfMaxQueryDistance=PosInfTy());

    // ******************************************************************
    /*! variant of cuBQL::point::findClosest() that's specialized for
      CUDA float4 type, just for convenience */
    inline __cubql_both
    int findClosest(/*! binary bvh built over the given points[]
                      specfied below */
                    bvh_float4 bvhOverPoints,
                    /*! data points that the bvh was built over */
                    const float4 *points,
                    float4 queryPoint,
                    /*! square of the maximum query distance in which
                      this query is to look for candidates. note
                      this is the SQUARE distance */
                    float squareOfMaxQueryDistance=PosInfTy());
#endif
    
    // ******************************************************************
    // IMPLEMENTATION
    // ******************************************************************

    template<typename T, int D, typename BlackListLambda>
    inline __cubql_device
    int findClosest_withBlackList(const BlackListLambda blackListed,
                                  /*! binary bvh built over the given points[]
                                    specfied below */
                                  BinaryBVH<T,D> bvhOverPoints,
                                  /*! data points that the bvh was built over */
                                  const vec_t<T,D> *points,
                                  /*! the query point for which we want to know the
                                    result */
                                  vec_t<T,D> queryPoint,
                                  /*! square of the maximum query distance in which
                                    this query is to look for candidates. note
                                    this is the SQUARE distance */
                                  float squareOfMaxQueryDistance=PosInfTy())
    {
      int closestID = -1;
      float closestSqrDist = squareOfMaxQueryDistance;
      // callback that processes each candidate, and checks if its
      // closer than current best
      auto candidateLambda
        = [blackListed,&closestID,&closestSqrDist,points,queryPoint]
        (int pointID)->float
        {
          if (blackListed(pointID))
            // caller explicitly blacklisted this point, do not process
            return closestSqrDist;
          
          // compute (square distance)
          float sqrDist = fSqrDistance_rd(points[pointID],queryPoint);
          if (sqrDist >= closestSqrDist)
            // candidate is further away than what we already have
            return closestSqrDist;

          // candidate is closer - accept and update search distance
          closestSqrDist = sqrDist;
          closestID      = pointID;
          return closestSqrDist;
        };

      cuBQL::shrinkingRadiusQuery::forEachPrim(candidateLambda,
                                               bvhOverPoints,
                                               queryPoint,
                                               squareOfMaxQueryDistance);
      return closestID;
    }


    template<typename T, int D, int W, typename BlackListLambda>
    inline __cubql_device
    int findClosest_withBlackList(const BlackListLambda blackListed,
                                  /*! binary bvh built over the given points[]
                                    specfied below */
                                  WideBVH<T,D,W> bvhOverPoints,
                                  /*! data points that the bvh was built over */
                                  const vec_t<T,D> *points,
                                  /*! the query point for which we want to know the
                                    result */
                                  vec_t<T,D> queryPoint,
                                  /*! square of the maximum query distance in which
                                    this query is to look for candidates. note
                                    this is the SQUARE distance */
                                  float squareOfMaxQueryDistance=PosInfTy())
    {
      int closestID = -1;
      float closestSqrDist = squareOfMaxQueryDistance;
      // callback that processes each candidate, and checks if its
      // closer than current best
      auto candidateLambda
        = [blackListed,&closestID,&closestSqrDist,points,queryPoint]
        (int pointID)->float
        {
          if (blackListed(pointID))
            // caller explicitly blacklisted this point, do not process
            return closestSqrDist;
          
          // compute (square distance)
          float sqrDist = fSqrDistance_rd(points[pointID],queryPoint);
          if (sqrDist >= closestSqrDist)
            // candidate is further away than what we already have
            return closestSqrDist;

          // candidate is closer - accept and update search distance
          closestSqrDist = sqrDist;
          closestID      = pointID;
          return closestSqrDist;
        };

      cuBQL::shrinkingRadiusQuery::forEachPrim(candidateLambda,
                                               bvhOverPoints,
                                               queryPoint,
                                               squareOfMaxQueryDistance);
      return closestID;
    }



    template<typename T, int D>
    inline __cubql_device
    int findClosest_exludeID(/*! primitive ID to _exclude_ from queries */
                             int idOfPointToExclude,
                             /*! binary bvh built over the given points[]
                               specfied below */
                             BinaryBVH<T,D> bvhOverPoints,
                             /*! data points that the bvh was built over */
                             const vec_t<T,D> *points,
                             /*! the query point for which we want to know the
                               result */
                             vec_t<T,D> queryPoint,
                             /*! square of the maximum query distance in which
                               this query is to look for candidates. note
                               this is the SQUARE distance */
                             float squareOfMaxQueryDistance)
    {
      /* blacklist the ID itself, then call
         `findClosest_withBlackList()` */
      auto blackList = [idOfPointToExclude](int pointID)->bool {
        return pointID == idOfPointToExclude;
      };
      return findClosest_withBlackList(blackList,
                                       bvhOverPoints,
                                       points,
                                       queryPoint,
                                       squareOfMaxQueryDistance);
    }

    /*! same as regular points::closestPoint, but excluding all data
      points that are at the query position itself */
    template<typename T, int D>
    inline __cubql_device
    int findClosest_exludeSelf(/*! binary bvh built over the given points[]
                                 specfied below */
                               BinaryBVH<T,D> bvhOverPoints,
                               /*! data points that the bvh was built over */
                               const vec_t<T,D> *points,
                               /*! the query point for which we want to know the
                                 result */
                               vec_t<T,D> queryPoint,
                               /*! square of the maximum query distance in which
                                 this query is to look for candidates. note
                                 this is the SQUARE distance */
                               float squareOfMaxQueryDistance)
    {
      /* blacklist any point at same position as query point, then
         call `findClosest_withBlackList()` */
      auto blackList = [points,queryPoint](int pointID)->bool {
        return points[pointID] == queryPoint;
      };
      return findClosest_withBlackList(blackList,
                                       bvhOverPoints,
                                       points,
                                       queryPoint,
                                       squareOfMaxQueryDistance);
    }

    template<typename T, int D>
    inline __cubql_device
    int findClosest(/*! binary bvh built over the given points[]
                      specfied below */
                    BinaryBVH<T,D> bvhOverPoints,
                    /*! data points that the bvh was built over */
                    const vec_t<T,D> *points,
                    /*! the query point for which we want to know the
                      result */
                    vec_t<T,D> queryPoint,
                    /*! square of the maximum query distance in which
                      this query is to look for candidates. note
                      this is the SQUARE distance */
                    float squareOfMaxQueryDistance)
    {
      /* no blacklist for 'general' find-closest */
      auto blackList = [](int pointID)->bool {
        return false;
      };
      return findClosest_withBlackList(blackList,
                                       bvhOverPoints,
                                       points,
                                       queryPoint,
                                       squareOfMaxQueryDistance);
    }


    template<typename T, int D, int W>
    inline __cubql_device
    int findClosest(/*! binary bvh built over the given points[]
                      specfied below */
                    WideBVH<T,D,W> bvhOverPoints,
                    /*! data points that the bvh was built over */
                    const vec_t<T,D> *points,
                    /*! the query point for which we want to know the
                      result */
                    vec_t<T,D> queryPoint,
                    /*! square of the maximum query distance in which
                      this query is to look for candidates. note
                      this is the SQUARE distance */
                    float squareOfMaxQueryDistance)
    {
      /* no blacklist for 'general' find-closest */
      auto blackList = [](int pointID)->bool {
        return false;
      };
      return findClosest_withBlackList(blackList,
                                       bvhOverPoints,
                                       points,
                                       queryPoint,
                                       squareOfMaxQueryDistance);
    }
    
#ifdef __CUDACC__
    // ******************************************************************
    /*! variant of cuBQL::point::findClosest() that's specialized for
      CUDA float2 type, just for convenience */
    inline __cubql_device
    int findClosest(/*! binary bvh built over the given points[]
                      specfied below */
                    bvh_float2 bvhOverPoints,
                    /*! data points that the bvh was built over */
                    const float2 *points,
                    float2 queryPoint,
                    /*! square of the maximum query distance in which
                      this query is to look for candidates. note
                      this is the SQUARE distance */
                    float squareOfMaxQueryDistance)
    {
      return findClosest(bvhOverPoints,
                         (const vec_t<float,2> *)points,
                         (const vec_t<float,2> &)queryPoint,
                         squareOfMaxQueryDistance);
    }

    // ******************************************************************
    /*! variant of cuBQL::point::findClosest() that's specialized for
      CUDA float3 type, just for convenience */
    inline __cubql_device
    int findClosest(/*! binary bvh built over the given points[]
                      specfied below */
                    bvh_float3 bvhOverPoints,
                    /*! data points that the bvh was built over */
                    const float3 *points,
                    float3 queryPoint,
                    /*! square of the maximum query distance in which
                      this query is to look for candidates. note
                      this is the SQUARE distance */
                    float squareOfMaxQueryDistance)
    { return findClosest(bvhOverPoints,
                         (const vec_t<float,3> *)points,
                         (const vec_t<float,3> &)queryPoint,
                         squareOfMaxQueryDistance);
    }

    // ******************************************************************
    /*! variant of cuBQL::point::findClosest() that's specialized for
      CUDA float4 type, just for convenience */
    inline __cubql_device
    int findClosest(/*! binary bvh built over the given points[]
                      specfied below */
                    bvh_float4 bvhOverPoints,
                    /*! data points that the bvh was built over */
                    const float4 *points,
                    float4 queryPoint,
                    /*! square of the maximum query distance in which
                      this query is to look for candidates. note
                      this is the SQUARE distance */
                    float squareOfMaxQueryDistance)
    { return findClosest(bvhOverPoints,
                         (const vec_t<float,4> *)points,
                         (const vec_t<float,4> &)queryPoint,
                         squareOfMaxQueryDistance);
    }
#endif
  }
}
