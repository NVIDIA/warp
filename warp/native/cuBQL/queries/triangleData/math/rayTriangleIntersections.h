// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cuBQL/queries/triangleData/Triangle.h"
#include "cuBQL/math/Ray.h"

namespace cuBQL {

  // ========================================================================
  // *** INTERFACE ***
  // ========================================================================
  
  // struct RayTriangleIntersection {
  //   vec3f N;
  //   float t,u,v;
    
  //   inline __cubql_both bool compute(Ray ray, Triangle tri);
  // };

  template<typename T>
  struct RayTriangleIntersection_t {
    using vec3 = vec_t<T,3>;
    T t=0,u=0,v=0;
    vec3 N;
    
    inline __cubql_both bool compute(const ray_t<T> &ray,
                                     const triangle_t<T> &tri,
                                     bool dbg=false);
  };

  using RayTriangleIntersection = RayTriangleIntersection_t<float>;
  
  // ========================================================================
  // *** IMPLEMENTATION ***
  // ========================================================================

  template<typename T>
  inline __cubql_both
  bool RayTriangleIntersection_t<T>::compute(const ray_t<T> &ray,
                                             const triangle_t<T> &tri,
                                             bool dbg)
  {
    using vec3 = vec_t<T,3>;
    const vec3 v0(tri.a);
    const vec3 v1(tri.b);
    const vec3 v2(tri.c);

    const vec3 e1 = v1-v0;
    const vec3 e2 = v2-v0;

    N = cross(e1,e2);
    if (N == vec3(T(0)))
      return false;

    if (abst(dot(ray.direction,N)) < T(1e-12)) return false;
    
    // P = o+td
    // dot(P-v0,N) = 0
    // dot(o+td-v0,N) = 0
    // dot(td,N)+dot(o-v0,N)=0
    // t*dot(d,N) = -dot(o-v0,N)
    // t = -dot(o-v0,N)/dot(d,N)
    t = -dot(ray.origin-v0,N)/dot(ray.direction,N);
    if (t <= ray.tMin || t >= ray.tMax) return false;
    
    vec3 P = (ray.origin - v0) + t*ray.direction;
    
    T e1u,e2u,Pu;
    T e1v,e2v,Pv;
    if (abst(N.x) >= max(abst(N.y),abst(N.z))) {
      e1u = e1.y; e2u = e2.y; Pu = P.y;
      e1v = e1.z; e2v = e2.z; Pv = P.z;
    } else if (abst(N.y) > abst(N.z)) {
      e1u = e1.x; e2u = e2.x; Pu = P.x;
      e1v = e1.z; e2v = e2.z; Pv = P.z;
    } else {
      e1u = e1.x; e2u = e2.x; Pu = P.x;
      e1v = e1.y; e2v = e2.y; Pv = P.y;
    }
    auto det = [](T a, T b, T c, T d) -> T
    { return a*d - c*b; };
    
    // P = v0 + u * e1 + v * e2 + h * N
    // (P-v0) = [e1,e2]*(u,v,h)
    if (det(e1u,e1v,e2u,e2v) == T(0)) return false;

#if 0
    T den = det(e1u,e2u,e1v,e2v);
    T sign = den < T(0) ? T(-1):T(1);
    den *= sign;
    T den_u = sign*det(Pu,e2u,Pv,e2v);
    if (den_u < T(0)) return false;
    T den_v = sign*det(e1u,Pu,e1v,Pv);
    if (den_v < T(0)) return false;
    if (den_u + den_v > den) return false;
    T rcp_den = rcp(den);
    u = den_u * rcp_den;
    v = den_v * rcp_den;
#else
    u = det(Pu,e2u,Pv,e2v)/det(e1u,e2u,e1v,e2v);
    v = det(e1u,Pu,e1v,Pv)/det(e1u,e2u,e1v,e2v);

    if ((u < T(0)) || (v < T(0)) || ((u+v) > T(1))) return false;
#endif
    return true;
  }
  
  // inline __cubql_both
  // bool RayTriangleIntersection::compute(Ray ray, Triangle tri)
  // {
  //   const vec3f v0 = tri.a;
  //   const vec3f v1 = tri.b;
  //   const vec3f v2 = tri.c;
    
  //   const vec3f e1 = v1-v0;
  //   const vec3f e2 = v2-v0;
    
  //   vec3f N = cross(e1,e2);
  //   if (fabsf(dot(ray.direction,N)) < 1e-12f) return false;
    
  //   t = -dot(ray.origin-v0,N)/dot(ray.direction,N);
    
  //   if (t <= 0.f || t >= ray.tMax) return false;
    
  //   vec3f P = ray.origin - v0 + t*ray.direction;
    
  //   float e1u,e2u,Pu;
  //   float e1v,e2v,Pv;
  //   if (fabsf(N.x) >= max(fabsf(N.y),fabsf(N.z))) {
  //     e1u = e1.y; e2u = e2.y; Pu = P.y;
  //     e1v = e1.z; e2v = e2.z; Pv = P.z;
  //   } else if (fabsf(N.y) > fabsf(N.z)) {
  //     e1u = e1.x; e2u = e2.x; Pu = P.x;
  //     e1v = e1.z; e2v = e2.z; Pv = P.z;
  //   } else {
  //     e1u = e1.x; e2u = e2.x; Pu = P.x;
  //     e1v = e1.y; e2v = e2.y; Pv = P.y;
  //   }
  //   auto det = [](float a, float b, float c, float d) -> float
  //   { return a*d - c*b; };
    
  //   // P = v0 + u * e1 + v * e2 + h * N
  //   // (P-v0) = [e1,e2]*(u,v,h)
  //   if (det(e1u,e1v,e2u,e2v) == 0.f) return false;
    
  //   u = det(Pu,e2u,Pv,e2v)/det(e1u,e2u,e1v,e2v);
  //   v = det(e1u,Pu,e1v,Pv)/det(e1u,e2u,e1v,e2v);
  //   if ((u < 0.f) || (v < 0.f) || ((u+v) >= 1.f)) return false;
    
  //   return true;
  // }




  template<int /*! 0, 1, or 2 */axis, int /* +1 or -1 */sign>
  inline __cubql_both
  bool rayIntersectsTriangle(AxisAlignedRay<axis,sign> ray,
                             Triangle triangle,
                             bool dbg=false)
  {
    const vec3f dir = ray.direction();
    const vec3f org = ray.origin;
    
    if (dbg) {
      dout << "-----------\ntriangle " << triangle << "\n";
    }
    using cuBQL::dot;
    using cuBQL::cross;

    vec3f n = triangle.normal();
    if (dbg) dout << "normal " << n << endl;
    if (dbg) dout << "dir " << dir << endl;

    float cosND = dot(n,dir);
    if (cosND == 0.f)
      /* iw - this is debatable - a perfectly parallel triangle may
         still have the ray 'intersect' if its in the same 2D plane */
      return false;

    float t = -dot(org-triangle.a,n)/cosND;
    if (t <= ray.tMin || t >= ray.tMax) {
      if (dbg) dout << " -> not in interval" << endl;
      return false;
    }
    
    vec3f a = triangle.a;
    vec3f b = triangle.b;
    vec3f c = triangle.c;
    // transform triangle into space centered aorund line origin
    a = a - org;
    b = b - org;
    c = c - org;

    auto pluecker=[](vec3f a0, vec3f a1, vec3f b0, vec3f b1) 
    { return dot(a1-a0,cross(b1,b0))+dot(b1-b0,cross(a1,a0)); };

    // compute pluecker coordinates dot product of all edges wrt x
    // axis ray. since the ray is mostly 0es and 1es, this shold all
    // evaluate to some fairly simple expressions
    float sx = pluecker(vec3f(0.f),dir,a,b);
    float sy = pluecker(vec3f(0.f),dir,b,c);
    float sz = pluecker(vec3f(0.f),dir,c,a);
    if (dbg) dout << "pluecker " << sx << " " << sy << " " << sz << endl;
    // for ray to be inside edges it must have all positive or all
    // negative pluecker winding order
    auto min3=[](float x, float y, float z)
    { return min(min(x,y),z); };
    auto max3=[](float x, float y, float z)
    { return max(max(x,y),z); };
    if (min3(sx,sy,sz) >= 0.f || max3(sx,sy,sz) <= 0.f) {
      if (dbg) dout << " -> HIT\n";
      return true;
    }
      
      if (dbg) dout << " -> MISS\n";
    return false;
  }
  

  inline __cubql_both
  bool rayIntersectsTriangle(Ray ray,
                             Triangle triangle,
                             bool dbg=false)
  {
    vec3f org = ray.origin;
    vec3f dir = ray.direction;
    
    if (dbg) {
      dout << "-----------\ntriangle " << triangle << "\n";
    }
    using cuBQL::dot;
    using cuBQL::cross;

    vec3f n = triangle.normal();
    if (dbg) dout << "normal " << n << endl;
    if (dbg) dout << "dir " << dir << endl;

    float cosND = dot(n,dir);
    if (cosND == 0.f)
      /* iw - this is debatable - a perfectly parallel triangle may
         still have the ray 'intersect' if its in the same 2D plane */
      return false;

    float t = -dot(org-triangle.a,n)/cosND;
    if (t <= ray.tMin || t >= ray.tMax) {
      if (dbg) dout << " -> not in interval" << endl;
      return false;
    }
    
    vec3f a = triangle.a;
    vec3f b = triangle.b;
    vec3f c = triangle.c;
    // transform triangle into space centered aorund line origin
    a = a - org;
    b = b - org;
    c = c - org;

    auto pluecker=[](vec3f a0, vec3f a1, vec3f b0, vec3f b1) 
    { return dot(a1-a0,cross(b1,b0))+dot(b1-b0,cross(a1,a0)); };

    // compute pluecker coordinates dot product of all edges wrt x
    // axis ray. since the ray is mostly 0es and 1es, this shold all
    // evaluate to some fairly simple expressions
    float sx = pluecker(vec3f(0.f),dir,a,b);
    float sy = pluecker(vec3f(0.f),dir,b,c);
    float sz = pluecker(vec3f(0.f),dir,c,a);
    if (dbg) dout << "pluecker " << sx << " " << sy << " " << sz << endl;
    // float sx = pluecker(beg,dir,a,b-a);
    // float sy = pluecker(beg,dir,b,c-b);
    // float sz = pluecker(beg,dir,c,a-c);
    // for ray to be inside edges it must have all positive or all
    // negative pluecker winding order
    auto min3=[](float x, float y, float z)
    { return min(min(x,y),z); };
    auto max3=[](float x, float y, float z)
    { return max(max(x,y),z); };
    if (min3(sx,sy,sz) >= 0.f || max3(sx,sy,sz) <= 0.f) {
      if (dbg) dout << " -> HIT\n";
      return true;
    }
      
      if (dbg) dout << " -> MISS\n";
    return false;
  }
  
}

