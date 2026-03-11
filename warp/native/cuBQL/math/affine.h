// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "linear.h"

namespace cuBQL {

  ////////////////////////////////////////////////////////////////////////////////
  // Affine Space
  ////////////////////////////////////////////////////////////////////////////////

  template<typename L>
  struct AffineSpaceT
  {
    using vector_t = typename L::vector_t;
    using linear_t = L;
    using scalar_t = typename vector_t::scalar_t;

    linear_t l; /*< linear part of affine space */
    vector_t p; /*< affine part of affine space */

    ////////////////////////////////////////////////////////////////////////////////
    // Constructors, Assignment, Cast, Copy Operations
    ////////////////////////////////////////////////////////////////////////////////

    inline __cubql_both AffineSpaceT()
      : l(OneTy()),
        p(ZeroTy())
    {}

    inline AffineSpaceT(const AffineSpaceT &other) = default;
    
    inline __cubql_both AffineSpaceT(const L &other)
    {
      l = other  ;
      p = vector_t(ZeroTy());
    }
    
    inline __cubql_both AffineSpaceT& operator=(const AffineSpaceT& other)
    {
      l = other.l;
      p = other.p;
      return *this;
    }
    
    inline __cubql_both AffineSpaceT(const vector_t& vx,
                                     const vector_t& vy,
                                     const vector_t& vz,
                                     const vector_t& p)
      : l(vx,vy,vz),
        p(p)
    {}
    
    inline __cubql_both AffineSpaceT(const L& l,
                                     const vector_t& p)
      : l(l),
        p(p)
    {}

    template<typename L1> inline __cubql_both AffineSpaceT( const AffineSpaceT<L1>& s )
      : l(s.l),
        p(s.p)
    {}

    ////////////////////////////////////////////////////////////////////////////////
    // Constants
    ////////////////////////////////////////////////////////////////////////////////

    inline AffineSpaceT( ZeroTy ) : l(ZeroTy()), p(ZeroTy()) {}
    inline AffineSpaceT( OneTy )  : l(OneTy()),  p(ZeroTy()) {}

    /*! return matrix for scaling */
    static inline AffineSpaceT scale(const vector_t& s) { return L::scale(s); }

    /*! return matrix for translation */
    static inline AffineSpaceT translate(const vector_t& p) { return AffineSpaceT(OneTy(),p); }

    /*! return matrix for rotation, only in 2D */
    static inline AffineSpaceT rotate(const scalar_t &r) { return L::rotate(r); }

    /*! return matrix for rotation around arbitrary point (2D) or axis (3D) */
    static inline AffineSpaceT rotate(const vector_t &u,
                                      const scalar_t &r)
    { return L::rotate(u,r);}

    /*! return matrix for rotation around arbitrary axis and point, only in 3D */
    static inline AffineSpaceT rotate(const vector_t &p,
                                      const vector_t &u,
                                      const scalar_t &r)
    { return translate(+p) * rotate(u,r) * translate(-p);  }

    /*! return matrix for looking at given point, only in 3D; right-handed coordinate system */
    static inline AffineSpaceT lookat(const vector_t& eye,
                                      const vector_t& point,
                                      const vector_t& up)
    {
      vector_t Z = normalize(point-eye);
      vector_t U = normalize(cross(Z,up));
      vector_t V = cross(U,Z);
      return AffineSpaceT(L(U,V,Z),eye);
    }

  };

  ////////////////////////////////////////////////////////////////////////////////
  // Unary Operators
  ////////////////////////////////////////////////////////////////////////////////

  template<typename L> inline AffineSpaceT<L> operator -( const AffineSpaceT<L>& a ) { return AffineSpaceT<L>(-a.l,-a.p); }
  template<typename L> inline AffineSpaceT<L> operator +( const AffineSpaceT<L>& a ) { return AffineSpaceT<L>(+a.l,+a.p); }
  template<typename L>
  inline __cubql_both
  AffineSpaceT<L> rcp( const AffineSpaceT<L>& a ) {
    L il = rcp(a.l);
    return AffineSpaceT<L>(il,-(il*a.p));
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Binary Operators
  ////////////////////////////////////////////////////////////////////////////////

  template<typename L> inline AffineSpaceT<L> operator +( const AffineSpaceT<L>& a, const AffineSpaceT<L>& b ) { return AffineSpaceT<L>(a.l+b.l,a.p+b.p); }
  template<typename L> inline AffineSpaceT<L> operator -( const AffineSpaceT<L>& a, const AffineSpaceT<L>& b ) { return AffineSpaceT<L>(a.l-b.l,a.p-b.p); }

  template<typename L> inline __cubql_both
  AffineSpaceT<L> operator *(const typename AffineSpaceT<L>::scalar_t &a,
                             const AffineSpaceT<L>                   &b )
  { return AffineSpaceT<L>(a*b.l,a*b.p); }
  
  template<typename L> inline __cubql_both
  AffineSpaceT<L> operator *( const AffineSpaceT<L>& a, const AffineSpaceT<L>& b )
  { return AffineSpaceT<L>(a.l*b.l,a.l*b.p+a.p); }
  
  template<typename L> inline
  AffineSpaceT<L> operator /( const AffineSpaceT<L>& a, const AffineSpaceT<L>& b )
  { return a * rcp(b); }
  
  template<typename L> inline
  AffineSpaceT<L> operator/(const AffineSpaceT<L>                   &a,
                            const typename AffineSpaceT<L>::scalar_t &b)
  { return a * rcp(b); }
  
  template<typename L> inline
  AffineSpaceT<L>& operator *=( AffineSpaceT<L>& a, const AffineSpaceT<L>& b )
  { return a = a * b; }
  
  template<typename L> inline
  AffineSpaceT<L> &operator*=(AffineSpaceT<L>                         &a,
                              const typename AffineSpaceT<L>::scalar_t &b)
  { return a = a * b; }
  
  template<typename L> inline
  AffineSpaceT<L>& operator /=( AffineSpaceT<L>& a, const AffineSpaceT<L>& b )
  { return a = a / b; }
  
  template<typename L> inline
  AffineSpaceT<L> &operator/=(AffineSpaceT<L>                         &a,
                              const typename AffineSpaceT<L>::scalar_t &b)
  { return a = a / b; }

  template<typename L> inline __cubql_both 
  typename AffineSpaceT<L>::vector_t xfmPoint(const AffineSpaceT<L>& m,
                                             const typename AffineSpaceT<L>::vector_t &p)
  {
    using vector_t = typename AffineSpaceT<L>::vector_t;
    return madd(vector_t(p.x),m.l.vx,
                madd(vector_t(p.y),m.l.vy,
                     madd(vector_t(p.z),m.l.vz,
                          m.p)));
  }
  
  template<typename L> inline __cubql_both
  typename AffineSpaceT<L>::vector_t xfmVector(const AffineSpaceT<L>& m,
                                              const typename AffineSpaceT<L>::vector_t& v)
  { return xfmVector(m.l,v); }
  
  template<typename L> inline __cubql_both
  typename AffineSpaceT<L>::vector_t xfmNormal(const AffineSpaceT<L>& m,
                                              const typename AffineSpaceT<L>::vector_t& n)
  { return xfmNormal(m.l,n); }


  ////////////////////////////////////////////////////////////////////////////////
  /// Comparison Operators
  ////////////////////////////////////////////////////////////////////////////////

  template<typename L> inline __cubql_both
  bool operator ==( const AffineSpaceT<L>& a, const AffineSpaceT<L>& b )
  { return a.l == b.l && a.p == b.p; }
  
  template<typename L> inline __cubql_both
  bool operator !=( const AffineSpaceT<L>& a, const AffineSpaceT<L>& b )
  { return a.l != b.l || a.p != b.p; }

  ////////////////////////////////////////////////////////////////////////////////
  // Output Operators
  ////////////////////////////////////////////////////////////////////////////////

  template<typename L> inline std::ostream& operator<<(std::ostream& cout, const AffineSpaceT<L>& m) {
    return cout << "{ l = " << m.l << ", p = " << m.p << " }";
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Type Aliases
  ////////////////////////////////////////////////////////////////////////////////

  using AffineSpace2f = AffineSpaceT<linear2f>;
  using AffineSpace3f = AffineSpaceT<linear3f>;
  using AffineSpace2d = AffineSpaceT<linear2d>;
  using AffineSpace3d = AffineSpaceT<linear3d>;

  ////////////////////////////////////////////////////////////////////////////////
  /*! Template Specialization for 2D: return matrix for rotation around point (rotation around arbitrarty vector is not meaningful in 2D) */
  template<> inline AffineSpace2f AffineSpace2f::rotate(const vec2f& p, const float& r)
  { return translate(+p) * AffineSpace2f(LinearSpace2f::rotate(r)) * translate(-p); }


  using affine2f = AffineSpace2f;
  using affine3f = AffineSpace3f;
  using affine2d = AffineSpace2d;
  using affine3d = AffineSpace3d;
} // ::cuBQL
