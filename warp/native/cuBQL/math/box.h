// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cuBQL/math/vec.h"

namespace cuBQL {
  using int64 = ::int64_t;
  
  /*! @{ defines what box.lower/box.upper coordinate values to use for
    indicating an "empty box" of the given type */
  template<typename scalar_t>
  inline __cubql_both scalar_t empty_box_lower_value();
  template<typename scalar_t>
  inline __cubql_both scalar_t empty_box_upper_value();

  template<> inline __cubql_both float empty_box_lower_value<float>() { return +CUBQL_INF; }
  template<> inline __cubql_both float empty_box_upper_value<float>() { return -CUBQL_INF; }
  template<> inline __cubql_both double empty_box_lower_value<double>() { return +CUBQL_INF; }
  template<> inline __cubql_both double empty_box_upper_value<double>() { return -CUBQL_INF; }
  template<> inline __cubql_both int empty_box_lower_value<int>() { return INT_MAX; }
  template<> inline __cubql_both int empty_box_upper_value<int>() { return INT_MIN; }
  template<> inline __cubql_both int64 empty_box_lower_value<int64>() { return LONG_MAX; }
  template<> inline __cubql_both int64 empty_box_upper_value<int64>() { return LONG_MIN; }
  /*! @} */
  
  /*! data-only part of a axis-aligned bounding box, made up of a
    'lower' and 'upper' vector bounding it. any box with any
    lower[k] > upper[k] is to be treated as a "empty box" (e.g., for
    a bvh that might indicate a primitive that we want to ignore
    from the build) */
  template<typename _scalar_t, int _numDims>
  struct box_t_pod {
    enum { numDims = _numDims };
    using scalar_t = _scalar_t;
    using vec_t = cuBQL::vec_t<scalar_t,numDims>;
    
    vec_t lower, upper;
  };


  /*! a axis-aligned bounding box, made up of a 'lower' and 'upper'
    vector bounding it. any box with any lower[k] > upper[k] is to
    be treated as a "empty box" (e.g., for a bvh that might indicate
    a primitive that we want to ignore from the build) */
  template<typename T, int D>
  struct CUBQL_ALIGN(8) box_t : public box_t_pod<T,D> {
    enum { numDims = D };
    using scalar_t = T;
    using vec_t = cuBQL::vec_t<T,D>;

    using box_t_pod<T,D>::lower;
    using box_t_pod<T,D>::upper;

    /*! default constructor - creates an "empty" box (ie, not an
      un-initialized one like a POD version, but one that is
      explicitly marked as inverted b yhavin lower > upper) */
    inline __cubql_both box_t()
    {
      lower = vec_t(empty_box_lower_value<T>());
      upper = vec_t(empty_box_upper_value<T>());
    };
    
    /*! copy-constructor - create a box from another box (or
      POD-version of such) of same type */
    inline explicit __cubql_both box_t(const box_t_pod<T,D> &ob);

    template<typename OT>
    /*! copy-constructor - create a box from another box (or
      POD-version of such) of same type */
    inline __cubql_both box_t(const box_t_pod<OT,D> &ob)
    { lower = cuBQL::vec_t<T,D>(ob.lower); upper = cuBQL::vec_t<T,D>(ob.upper); }


    /*! create a box containing a single point */
    inline __cubql_both box_t(const vec_t_data<T,D> &v) { lower = upper = v; }

    /*! create a box from two points. note this will NOT make sure
      that a<b; if this is an empty box it will just create an empty
      box! */
    inline __cubql_both box_t(const vec_t_data<T,D> &a, const vec_t_data<T,D> &b)
    {
      lower = vec_t(a);
      upper = vec_t(b);
    }

    /*! returns a box that bounds both 'this' and another point 'v';
      this does not get modified */
    inline __cubql_both box_t including(const vec_t &v) const
    { return box_t{min(lower,v),max(upper,v)}; }
    /*! returns a box that bounds both 'this' and another box 'b';
      this does not get modified */
    inline __cubql_both box_t including(const box_t &b) const
    { return box_t{min(lower,b.lower),max(upper,b.upper)}; }
    
    inline __cubql_both box_t &grow(const vec_t &v)
    { lower = min(lower,v); upper = max(upper,v); return *this; }
    inline __cubql_both box_t &grow(const box_t &other)
    { lower = min(lower,other.lower); upper = max(upper,other.upper); return *this; }

    /*! "extend" an existing box such that it also encompasses v; this
      is the same as 'grow()', but with naming that is compatible to
      the owl project */
    inline __cubql_both box_t &extend(const vec_t &v)
    { lower = min(lower,v); upper = max(upper,v); return *this; }

    /*! "extend" an existing box such that it also encompasses this
      other box; this is the same as 'grow()', but with naming that
      is compatible to the owl project */
    inline __cubql_both box_t &extend(const box_t &other)
    { lower = min(lower,other.lower); upper = max(upper,other.upper); return *this; }
    inline __cubql_both box_t &set_empty()
    {
      this->lower = make<vec_t>(empty_box_lower_value<scalar_t>());
      this->upper = make<vec_t>(empty_box_upper_value<scalar_t>());
      return *this;
    }
    inline __cubql_both void clear() { set_empty(); }
    inline __cubql_both bool empty() const { return get_lower(0) > get_upper(0); }

#if CUBQL_SUPPORT_CUDA_VECTOR_TYPES
    using cuda_vec_t = typename cuda_eq_t<scalar_t,numDims>::type;
    inline __cubql_both box_t &grow(cuda_vec_t other)
    {
      this->lower = min(this->lower,make<vec_t>(other));
      this->upper = max(this->upper,make<vec_t>(other)); return *this;
    }
#endif

    /*! returns the center of the box, up to rounding errors. (i.e. on
      its, the center of a box with lower=2 and upper=3 is 2, not
      2.5! */
    inline __cubql_both vec_t center() const
    { return (this->lower+this->upper)/scalar_t(2); }

    /*! return a vector that give the size of the box in x,y,...etc */
    inline __cubql_both vec_t size() const
    { return this->upper - this->lower; }

    /*! return a vector that give the size of the box in x,y,...etc
        (also known as the 'span' of a box, due to the it being the
        vector that spans the box from lower to upper) */
    inline __cubql_both vec_t span() const
    { return this->upper - this->lower; }

    /*! returns TWICE the center (which happens to be the SUM of lower
      an dupper). Note this 'conceptually' the same as 'center()',
      but without the nasty division that may lead to rounding
      errors for int types; it's obviously not the center but twice
      the center - but as long as all routines that expect centers
      use that same 'times 2' this will still work out */
    inline __cubql_both vec_t twice_center() const
    { return (this->lower+this->upper); }

    /*! for convenience's sake, get_lower(i) := lower[i] */
    inline __cubql_both T get_lower(int i) const { return lower.get(i); }

    /*! for convenience's sake, get_upper(i) := upper[i] */
    inline __cubql_both T get_upper(int i) const { return upper.get(i); };

    /*! linearly interpolate given box, and return point that
        corresponds to these interpolation weights */
    inline __cubql_both
      cuBQL::vec_t<float,D> lerp(cuBQL::vec_t<float,D> f) const
    { return
        (cuBQL::vec_t<float,D>(1.f) - f)*cuBQL::vec_t<float,D>(lower)
        +
        f                               *cuBQL::vec_t<float,D>(upper);
    }
    
    inline __cubql_both cuBQL::vec_t<double,D> lerp(cuBQL::vec_t<double,D> f) const;
    inline __cubql_both bool overlaps(const box_t<T,D> &otherBox) const;
  };

  /*! float32 */
  using box2f = box_t<float,2>;
  using box3f = box_t<float,3>;
  using box4f = box_t<float,4>;
  /* (signed) int (ie, int32) */
  using box2i = box_t<int,2>;
  using box3i = box_t<int,3>;
  using box4i = box_t<int,4>;
  /* (signed) long long (ie, int64) */
  using box2l = box_t<long long,2>;
  using box3l = box_t<long long,3>;
  using box4l = box_t<long long,4>;
  /* double */
  using box2d = box_t<double,2>;
  using box3d = box_t<double,3>;
  using box4d = box_t<double,4>;

  inline __cubql_both
  float surfaceArea(box3f box)
  {
    const float sx = box.upper[0]-box.lower[0];
    const float sy = box.upper[1]-box.lower[1];
    const float sz = box.upper[2]-box.lower[2];
    return sx*sy + sx*sz + sy*sz;
  }

  template<typename T, int D> inline __cubql_both
  bool box_t<T,D>::overlaps(const box_t<T,D> &other) const
  {
    return !(any_less_than(this->upper,other.lower) ||
             any_less_than(other.upper,this->lower));
  }
  
  template<typename T, int D> inline __cubql_both
  typename dot_result_t<T>::type sqrDistance(box_t<T,D> box, vec_t<T,D> point)
  {
    vec_t<T,D> closestPoint = min(max(point,box.lower),box.upper);
    return sqrDistance(closestPoint,point);
  }

  /*! grow the box such that it contains the given point, and return
    reference to that grown box */
  template<typename T, int D> inline __cubql_both
  box_t<T,D> &grow(box_t<T,D> &b, vec_t<T,D> v)
  { b.grow(v); return b; }

  /*! grow the first box such that it contains the given second box,
    and return reference to that grown box */
  template<typename T, int D> inline __cubql_both
  box_t<T,D> &grow(box_t<T,D> &b, box_t<T,D> ob)
  { b.grow(ob); return b; }

  /*! construct a new box spanning a single point */
  template<typename T, int D> inline __cubql_both
  box_t<T,D> make_box(vec_t<T,D> v) { return box_t<T,D>(v); }


  /*! projects a vector into a box, in the sense that the point
    returned is the point within the box that's closest to the point
    provided; if input point is inside the box it will return
    itself, otherwise it'll be a point on the outer shell of the
    box */
  template<typename T, int D>
  inline __cubql_both vec_t<T,D> project(box_t<T,D> box, vec_t<T,D> point)
  {
    return min(max(point,box.lower),box.upper);
  }

  template<typename T, int D>
  bool operator==(const box_t<T,D> &a, const box_t<T,D> &b)
  { return a.lower == b.lower && a.upper == b.upper; }

  template<typename T, int D>
  bool operator!=(const box_t<T,D> &a, const box_t<T,D> &b)
  { return !(a==b); }

  template<typename T, int D>
  std::ostream &operator<<(std::ostream &o, const box_t<T,D> &box)
  { o << "{" << box.lower << "," << box.upper << "}"; return o; }

  template<typename T, int D>
  inline __cubql_both dbgout operator<<(dbgout o, const box_t<T,D> &box)
  { o << "{" << box.lower << "," << box.upper << "}"; return o; }
  
}

