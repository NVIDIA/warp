// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file cuBQL/queries/points/findClosest Provides kernels for
  finding closest point(s) on point data */

#pragma once

#include "cuBQL/traversal/shrinkingRadiusQuery.h"
#include "cuBQL/queries/common/knn.h"

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
    cuBQL::knn::Result findKNN(/*! memory to return the list of found knn candidates in */
                               cuBQL::knn::Candidate *foundCandidates,
                               /*! number of knn candidates we're
                                   looking for (ie, the "N" in
                                   N-neighbors) */
                               int maxN,
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
    
    
    // ******************************************************************
    // IMPLEMENTATION
    // ******************************************************************

    template<typename T, int D>
    inline __cubql_both
    cuBQL::knn::Result findKNN(/*! memory to return the list of found knn candidates in */
                               cuBQL::knn::Candidate *foundCandidates,
                               /*! number of knn candidates we're
                                   looking for (ie, the "N" in
                                   N-neighbors) */
                               int maxN,
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
      cuBQL::knn::Result result = { 0, squareOfMaxQueryDistance };
      // callback that processes each candidate, and checks if its
      // closer than current best
      auto candidateLambda
        = [&result,foundCandidates,maxN,points,queryPoint]
        (int pointID)->float
        {
          // compute (square distance)
          float sqrDist = fSqrDistance_rd(points[pointID],queryPoint);
          cuBQL::knn::Candidate thisCandidate = { pointID, sqrDist };
          cuBQL::knn::insert_linear(foundCandidates,maxN,thisCandidate,result);
          // return the maximum query distance from here on out:
          return result.sqrDistMax;
        };

      cuBQL::shrinkingRadiusQuery::forEachPrim(candidateLambda,
                                               bvhOverPoints,
                                               queryPoint,
                                               squareOfMaxQueryDistance);
      return result;
    }
    
  } // ::cuBQL::points
} // ::cuBQL
