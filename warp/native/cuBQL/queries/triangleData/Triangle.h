// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! \file cuBQL/triangles/Triangle.h Defines a generic triangle type and
  some operations thereon, that various queries can then build on */

#pragma once

#include "cuBQL/math/vec.h"
#include "cuBQL/math/box.h"

namespace cuBQL {

  // =========================================================================
  // *** INTERFACE ***
  // =========================================================================
  
  // /*! a simple triangle consisting of three vertices. In order to not
  //   overload this class with too many functions the actual
  //   operations on triangles - such as intersectin with a ray,
  //   computing distance to a point, etc - will be defined in the
  //   respective queries */
  // struct Triangle {
  //   /*! returns an axis aligned bounding box enclosing this triangle */
  //   inline __cubql_both box3f bounds() const;
  //   inline __cubql_both vec3f sample(float u, float v) const;
  //   inline __cubql_both vec3f normal() const;
    
  //   vec3f a, b, c;
  // };

  template<typename T>
  struct triangle_t
  {
    using vec3 = vec_t<T,3>;
    using box3 = box_t<T,3>;

    inline triangle_t() = default;
    inline triangle_t(const triangle_t &) = default;
    inline __cubql_both triangle_t(vec3 a, vec3 b, vec3 c)
      : a(a), b(b), c(c)
    {}
    inline __cubql_both box3 bounds() const;
    inline __cubql_both vec3 sample(float u, float v) const;
    inline __cubql_both vec3 normal() const;
    
    vec3 a;
    vec3 b;
    vec3 c;
  };

  using Triangle = triangle_t<float>;

  /*! a typical triangle mesh, with array of vertices and
      indices. This class will NOT do any allocation/deallocation, not
      use smart pointers - it's just a 'view' on what whoever else
      might own and manage, and may thus be used exactly the same on
      device as well as on host. */
  struct TriangleMesh {
    inline __cubql_both Triangle getTriangle(int i) const;
    
    /*! pointer to array of vertices; must be in same memory space as
        the operations performed on it (eg, if passed to a gpu builder
        it has to be gepu memory */
    vec3f *vertices;
    
    /*! pointer to array of vertices; must be in same memory space as
        the operations performed on it (eg, if passed to a gpu builder
        it has to be gepu memory */
    vec3i *indices;

    int numVertices;
    int numIndices;
  };

  
  // ========================================================================
  // *** IMPLEMENTATION ***
  // ========================================================================

  // ---------------------- TriangleMesh ----------------------
  inline __cubql_both Triangle TriangleMesh::getTriangle(int i) const
  {
    vec3i index = indices[i];
    return { vertices[index.x],vertices[index.y],vertices[index.z] };
  }
  
  // ---------------------- Triangle ----------------------
  template<typename T>
  inline __cubql_both vec_t<T,3> triangle_t<T>::normal() const
  { return cross(b-a,c-a); }
  
  template<typename T>
  inline __cubql_both box_t<T,3> triangle_t<T>::bounds() const
  { return box_t<T,3>().including(a).including(b).including(c); }

  template<typename T>
  inline __cubql_both float area(triangle_t<T> tri)
  { return length(cross(tri.b-tri.a,tri.c-tri.a)); }

  template<typename T>
  inline __cubql_both vec_t<T,3>
  triangle_t<T>::sample(float u, float v) const
  {
    if (u+v >= 1.f) { u = 1.f-u; v = 1.f-v; }
    return (1.f-u-v)*a + u * b + v * c;
  }
  // inline __cubql_both vec3f Triangle::sample(float u, float v) const
  // {
  //   if (u+v >= 1.f) { u = 1.f-u; v = 1.f-v; }
  //   return (1.f-u-v)*a + u * b + v * c;
  // }

  template<typename T>
  inline __cubql_both
  dbgout operator<<(dbgout o, const triangle_t<T> &triangle)
  {
    o << "{" << triangle.a << "," << triangle.b << "," << triangle.c << "}";
    return o;
  }
  
  
} // ::cuBQL

