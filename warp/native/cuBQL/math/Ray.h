// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cuBQL/math/vec.h"
#include "cuBQL/math/box.h"

namespace cuBQL {

  // =============================================================================
  // *** INTERFACE ***
  // =============================================================================

  template<typename T=float>
  struct ray_t {
    using vec3 = vec_t<T,3>;
    
    inline ray_t() = default;
    inline ray_t(const ray_t &) = default;
    __cubql_both ray_t(vec3 org, vec3 dir, T tMin, T tMax);
    __cubql_both ray_t(vec3 org, vec3 dir);
    vec3 origin;
    vec3 direction;
    T tMin = T(0);
    T tMax = T(CUBQL_INF);
  };

  using ray3f = ray_t<float>;
  using ray3d = ray_t<double>;
  using Ray   = ray_t<float>;

  template<int /*! 0, 1, or 2 */axis, int /* +1 or -1 */sign>
  struct AxisAlignedRay {
    __cubql_both AxisAlignedRay(const vec3f origin);
    __cubql_both AxisAlignedRay(const vec3f origin, float tMin, float tMax);
    
    vec3f origin;
    float tMin=0.f, tMax=CUBQL_INF;

    inline __cubql_both vec3f direction() const;
    inline __cubql_both Ray   makeRay() const;
  };

  template<typename T>
  inline __cubql_both
  bool rayIntersectsBox(ray_t<T> ray, box_t<T,3> box);
  
  // ========================================================================
  // *** IMPLEMENTATION ***
  // ========================================================================

  template<typename T>
  inline __cubql_both ray_t<T>::ray_t(typename ray_t<T>::vec3 org,
                                      typename ray_t<T>::vec3 dir,
                                      T tMin, T tMax)
    : origin(org), direction(dir), tMin(tMin), tMax(tMax)
  {}
  
  template<typename T>
  inline __cubql_both ray_t<T>::ray_t(typename ray_t<T>::vec3 org,
                                      typename ray_t<T>::vec3 dir)
    : origin(org), direction(dir)
  {}
  
  template<int /*! 0, 1, or 2 */axis, int /* +1 or -1 */sign>
  inline __cubql_both
  AxisAlignedRay<axis,sign>::AxisAlignedRay(const vec3f origin,
                                            float tMin, float tMax)
    : origin(origin), tMin(tMin), tMax(tMax)
  {}
  
  template<int /*! 0, 1, or 2 */axis, int /* +1 or -1 */sign>
  inline __cubql_both
  AxisAlignedRay<axis,sign>::AxisAlignedRay(const vec3f origin)
    : origin(origin)
  {}
  
  template<int /*! 0, 1, or 2 */axis, int /* +1 or -1 */sign>
  inline __cubql_both vec3f AxisAlignedRay<axis,sign>::direction() const
  {
    return {
      (axis == 0) ? (sign > 0 ? +1.f : -1.f) : 0.f,
      (axis == 1) ? (sign > 0 ? +1.f : -1.f) : 0.f,
      (axis == 2) ? (sign > 0 ? +1.f : -1.f) : 0.f
    };
  }

  template<int /*! 0, 1, or 2 */axis, int /* +1 or -1 */sign>
  inline __cubql_both ray_t<float> AxisAlignedRay<axis,sign>::makeRay() const
  {
    return { origin, tMin, direction(), tMax };
  }

  template<int /*! 0, 1, or 2 */axis, int /* +1 or -1 */sign>
  inline __cubql_both dbgout operator<<(dbgout o, AxisAlignedRay<axis,sign> ray)
  {
    o << "AARay<"<<axis<<","<<sign<<">("<<ray.origin<<",["<<ray.tMin<<","<<ray.tMax<<"])";
    return o;
  }

  template<typename T>
  inline __cubql_both dbgout operator<<(dbgout o, ray_t<T> ray)
  {
    o << "Ray{"<<ray.origin<<"+["<<ray.tMin<<","<<ray.tMax<<"]*"<<ray.direction<<"}";
    return o;
  }

}



