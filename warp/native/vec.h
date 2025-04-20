/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "initializer_array.h"

namespace wp
{

template<unsigned Length, typename Type>
struct vec_t
{
    Type c[Length];

    inline CUDA_CALLABLE vec_t()
        : c()
    {}
    
    inline CUDA_CALLABLE vec_t(Type s)
    {
        for( unsigned i=0; i < Length; ++i )
        {
            c[i] = s;
        }
    }
    
    template <typename OtherType>
    inline explicit CUDA_CALLABLE vec_t(const vec_t<Length, OtherType>& other)
    {
        for( unsigned i=0; i < Length; ++i )
        {
            c[i] = static_cast<Type>(other[i]);
        }
    }

    inline CUDA_CALLABLE vec_t(Type x, Type y)
    {
        assert(Length == 2);
        c[0]=x;
        c[1]=y;
    }

    inline CUDA_CALLABLE vec_t(Type x, Type y, Type z)
    {
        assert(Length == 3);
        c[0]=x;
        c[1]=y;
        c[2]=z;
    }


    inline CUDA_CALLABLE vec_t(Type x, Type y, Type z, Type w)
    {
        assert(Length == 4);
        c[0]=x;
        c[1]=y;
        c[2]=z;
        c[3]=w;
    }
    
    inline CUDA_CALLABLE vec_t(const initializer_array<Length, Type> &l)
    {
        for( unsigned i=0; i < Length; ++i )
        {
            c[i] = l[i];
        }
    }
    
    // special screw vector constructor for spatial_vectors:
    inline CUDA_CALLABLE vec_t(vec_t<3,Type> w, vec_t<3,Type> v)
    {
        c[0] = w[0];
        c[1] = w[1];
        c[2] = w[2];
        c[3] = v[0];
        c[4] = v[1];
        c[5] = v[2];
    }

    inline CUDA_CALLABLE Type operator[](int index) const
    {
        assert(index < Length);
        return c[index];
    }

    inline CUDA_CALLABLE Type& operator[](int index)
    {
        assert(index < Length);
        return c[index];
    }
};

using vec2b = vec_t<2,int8>;
using vec3b = vec_t<3,int8>;
using vec4b = vec_t<4,int8>;
using vec2ub = vec_t<2,uint8>;
using vec3ub = vec_t<3,uint8>;
using vec4ub = vec_t<4,uint8>;

using vec2s = vec_t<2,int16>;
using vec3s = vec_t<3,int16>;
using vec4s = vec_t<4,int16>;
using vec2us = vec_t<2,uint16>;
using vec3us = vec_t<3,uint16>;
using vec4us = vec_t<4,uint16>;

using vec2i = vec_t<2,int32>;
using vec3i = vec_t<3,int32>;
using vec4i = vec_t<4,int32>;
using vec2ui = vec_t<2,uint32>;
using vec3ui = vec_t<3,uint32>;
using vec4ui = vec_t<4,uint32>;

using vec2l = vec_t<2,int64>;
using vec3l = vec_t<3,int64>;
using vec4l = vec_t<4,int64>;
using vec2ul = vec_t<2,uint64>;
using vec3ul = vec_t<3,uint64>;
using vec4ul = vec_t<4,uint64>;

using vec2h = vec_t<2,half>;
using vec3h = vec_t<3,half>;
using vec4h = vec_t<4,half>;

using vec2 = vec_t<2,float>;
using vec3 = vec_t<3,float>;
using vec4 = vec_t<4,float>;

using vec2f = vec_t<2,float>;
using vec3f = vec_t<3,float>;
using vec4f = vec_t<4,float>;

using vec2d = vec_t<2,double>;
using vec3d = vec_t<3,double>;
using vec4d = vec_t<4,double>;

template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> operator - (const vec_t<Length, Type>& x)
{
    vec_t<Length, Type> ret;
    for(unsigned i=0; i < Length; ++i)
    {
        ret[i] = -x[i];
    }

    return ret;
}

template<unsigned Length, typename Type>
CUDA_CALLABLE inline vec_t<Length, Type> pos(const vec_t<Length, Type>& x)
{
    return x;
}

template<unsigned Length, typename Type>
CUDA_CALLABLE inline vec_t<Length, Type> neg(const vec_t<Length, Type>& x)
{
    return -x;
}

template<typename Type>
CUDA_CALLABLE inline vec_t<3, Type> neg(const vec_t<3, Type>& x)
{
    return vec_t<3, Type>(-x.c[0], -x.c[1], -x.c[2]);
}

template<typename Type>
CUDA_CALLABLE inline vec_t<2, Type> neg(const vec_t<2, Type>& x)
{
    return vec_t<2, Type>(-x.c[0], -x.c[1]);
}

template<unsigned Length, typename Type>
CUDA_CALLABLE inline void adj_neg(const vec_t<Length, Type>& x, vec_t<Length, Type>& adj_x, const vec_t<Length, Type>& adj_ret)
{
    adj_x -= adj_ret;
}

// equality:
template<unsigned Length, typename Type>
inline CUDA_CALLABLE bool operator ==(const vec_t<Length, Type>& a, const vec_t<Length, Type>& b)
{
    for( unsigned i=0; i < Length; ++i )
    {
        if(a[i] != b[i])
        {
            return false;
        }
    }
    return true;
}

// scalar multiplication:
template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> mul(vec_t<Length, Type> a, Type s)
{
    vec_t<Length, Type> ret;
    for( unsigned i=0; i < Length; ++i )
    {
        ret[i] = a[i] * s;
    }
    return ret;
}

template<typename Type>
inline CUDA_CALLABLE vec_t<3, Type> mul(vec_t<3, Type> a, Type s)
{
    return vec_t<3, Type>(a.c[0]*s,a.c[1]*s,a.c[2]*s);
}

template<typename Type>
inline CUDA_CALLABLE vec_t<2, Type> mul(vec_t<2, Type> a, Type s)
{
    return vec_t<2, Type>(a.c[0]*s,a.c[1]*s);
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> mul(Type s, vec_t<Length, Type> a)
{
    return mul(a, s);
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> operator*(Type s, vec_t<Length, Type> a)
{
    return mul(a, s);
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> operator*(vec_t<Length, Type> a, Type s)
{
    return mul(a, s);
}


// component wise multiplication:
template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> cw_mul(vec_t<Length, Type> a, vec_t<Length, Type> b)
{
    vec_t<Length, Type> ret;
    for( unsigned i=0; i < Length; ++i )
    {
        ret[i] = a[i] * b[i];
    }
    return ret;
}

// division
template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> div(vec_t<Length, Type> a, Type s)
{
    vec_t<Length, Type> ret;
    for( unsigned i=0; i < Length; ++i )
    {
        ret[i] = a[i] / s;
    }
    return ret;
}

template<typename Type>
inline CUDA_CALLABLE vec_t<3, Type> div(vec_t<3, Type> a, Type s)
{
    return vec_t<3, Type>(a.c[0]/s,a.c[1]/s,a.c[2]/s);
}

template<typename Type>
inline CUDA_CALLABLE vec_t<2, Type> div(vec_t<2, Type> a, Type s)
{
    return vec_t<2, Type>(a.c[0]/s,a.c[1]/s);
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> div(Type s, vec_t<Length, Type> a)
{
    vec_t<Length, Type> ret;
    for (unsigned i=0; i < Length; ++i)
    {
        ret[i] = s / a[i];
    }
    return ret;
}

template<typename Type>
inline CUDA_CALLABLE vec_t<3, Type> div(Type s, vec_t<3, Type> a)
{
    return vec_t<3, Type>(s/a.c[0],s/a.c[1],s/a.c[2]);
}

template<typename Type>
inline CUDA_CALLABLE vec_t<2, Type> div(Type s, vec_t<2, Type> a)
{
    return vec_t<2, Type>(s/a.c[0],s/a.c[1]);
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> operator / (vec_t<Length, Type> a, Type s)
{
    return div(a,s);
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> operator / (Type s, vec_t<Length, Type> a)
{
    return div(s, a);
}

// component wise division
template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> cw_div(vec_t<Length, Type> a, vec_t<Length, Type> b)
{
    vec_t<Length, Type> ret;
    for( unsigned i=0; i < Length; ++i )
    {
        ret[i] = a[i] / b[i];
    }
    return ret;
}

// addition
template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> add(vec_t<Length, Type> a, vec_t<Length, Type> b)
{
    vec_t<Length, Type> ret;
    for( unsigned i=0; i < Length; ++i )
    {
        ret[i] = a[i] + b[i];
    }
    return ret;
}

template<typename Type>
inline CUDA_CALLABLE vec_t<2, Type> add(vec_t<2, Type> a, vec_t<2, Type> b)
{
    return vec_t<2, Type>( a.c[0] + b.c[0], a.c[1] + b.c[1]);
}

template<typename Type>
inline CUDA_CALLABLE vec_t<3, Type> add(vec_t<3, Type> a, vec_t<3, Type> b)
{
    return vec_t<3, Type>( a.c[0] + b.c[0], a.c[1] + b.c[1], a.c[2] + b.c[2]);
}

// subtraction
template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> sub(vec_t<Length, Type> a, vec_t<Length, Type> b)
{
    vec_t<Length, Type> ret;
    for( unsigned i=0; i < Length; ++i )
    {
        ret[i] = Type(a[i] - b[i]);
    }
    return ret;
}

template<typename Type>
inline CUDA_CALLABLE vec_t<2, Type> sub(vec_t<2, Type> a, vec_t<2, Type> b)
{
    return vec_t<2, Type>( a.c[0] - b.c[0], a.c[1] - b.c[1]);
}

template<typename Type>
inline CUDA_CALLABLE vec_t<3, Type> sub(vec_t<3, Type> a, vec_t<3, Type> b)
{
    return vec_t<3, Type>( a.c[0] - b.c[0], a.c[1] - b.c[1], a.c[2] - b.c[2]);
}

// modulo
template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> mod(vec_t<Length, Type> a, vec_t<Length, Type> b)
{
    vec_t<Length, Type> ret;
    for (unsigned i=0; i < Length; ++i)
    {
        ret[i] = mod(a[i], b[i]);
    }

    return ret;
}

template<typename Type>
inline CUDA_CALLABLE vec_t<2, Type> mod(vec_t<2, Type> a, vec_t<2, Type> b)
{
    return vec_t<2, Type>(mod(a.c[0], b.c[0]), mod(a.c[1], b.c[1]));
}

template<typename Type>
inline CUDA_CALLABLE vec_t<3, Type> mod(vec_t<3, Type> a, vec_t<3, Type> b)
{
    return vec_t<3, Type>(mod(a.c[0], b.c[0]), mod(a.c[1], b.c[1]), mod(a.c[2], b.c[2]));
}

// dot product:
template<unsigned Length, typename Type>
inline CUDA_CALLABLE Type dot(vec_t<Length, Type> a, vec_t<Length, Type> b)
{
    Type ret(0);
    for( unsigned i=0; i < Length; ++i )
    {
        ret += a[i] * b[i];
    }
    return ret;
}

template<typename Type>
inline CUDA_CALLABLE Type dot(vec_t<2, Type> a, vec_t<2, Type> b)
{
    return a.c[0] * b.c[0] + a.c[1] * b.c[1];
}

template<typename Type>
inline CUDA_CALLABLE Type dot(vec_t<3, Type> a, vec_t<3, Type> b)
{
    return a.c[0] * b.c[0] + a.c[1] * b.c[1] + a.c[2] * b.c[2];
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE Type tensordot(vec_t<Length, Type> a, vec_t<Length, Type> b)
{
    // corresponds to `np.tensordot()` with all axes being contracted
    return dot(a, b);
}


template<unsigned Length, typename Type>
inline CUDA_CALLABLE Type extract(const vec_t<Length, Type> & a, int idx)
{
#ifndef NDEBUG
    if (idx < 0 || idx >= Length)
    {
        printf("vec index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    return a[idx];        
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE Type* index(vec_t<Length, Type>& v, int idx)
{
#ifndef NDEBUG
    if (idx < 0 || idx >= Length)
    {
        printf("vec index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    return &v[idx];
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE Type* indexref(vec_t<Length, Type>* v, int idx)
{
#ifndef NDEBUG
    if (idx < 0 || idx >= Length)
    {
        printf("vec store %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    return &((*v)[idx]);
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_index(vec_t<Length, Type>& v, int idx,
                                       vec_t<Length, Type>& adj_v, int adj_idx, const Type& adj_value)
{
    // nop
}


template<unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_indexref(vec_t<Length, Type>* v, int idx, 
                                       vec_t<Length, Type>& adj_v, int adj_idx, const Type& adj_value)
{
    // nop
}


template<unsigned Length, typename Type>
inline CUDA_CALLABLE void add_inplace(vec_t<Length, Type>& v, int idx, Type value)
{
#ifndef NDEBUG
    if (idx < 0 || idx >= Length)
    {
        printf("vec index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    v[idx] += value;
}


template<unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_add_inplace(vec_t<Length, Type>& v, int idx, Type value,
                                        vec_t<Length, Type>& adj_v, int adj_idx, Type& adj_value)
{
#ifndef NDEBUG
    if (idx < 0 || idx >= Length)
    {
        printf("vec index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    adj_value += adj_v[idx];
}


template<unsigned Length, typename Type>
inline CUDA_CALLABLE void sub_inplace(vec_t<Length, Type>& v, int idx, Type value)
{
#ifndef NDEBUG
    if (idx < 0 || idx >= Length)
    {
        printf("vec index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    v[idx] -= value;
}


template<unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_sub_inplace(vec_t<Length, Type>& v, int idx, Type value,
                                        vec_t<Length, Type>& adj_v, int adj_idx, Type& adj_value)
{
#ifndef NDEBUG
    if (idx < 0 || idx >= Length)
    {
        printf("vec index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    adj_value -= adj_v[idx];
}


template<unsigned Length, typename Type>
inline CUDA_CALLABLE void assign_inplace(vec_t<Length, Type>& v, int idx, Type value)
{
#ifndef NDEBUG
    if (idx < 0 || idx >= Length)
    {
        printf("vec index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    v[idx] = value;
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_assign_inplace(vec_t<Length, Type>& v, int idx, Type value, vec_t<Length, Type>& adj_v, int& adj_idx, Type& adj_value)
{
#ifndef NDEBUG
    if (idx < 0 || idx >= Length)
    {
        printf("vec index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    adj_value += adj_v[idx];
}


template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> assign_copy(vec_t<Length, Type>& v, int idx, Type value)
{
#ifndef NDEBUG
    if (idx < 0 || idx >= Length)
    {
        printf("vec index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    vec_t<Length, Type> ret(v);
    ret[idx] = value;
    return ret;
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_assign_copy(vec_t<Length, Type>& v, int idx, Type value, vec_t<Length, Type>& adj_v, int& adj_idx, Type& adj_value, const vec_t<Length, Type>& adj_ret)
{
#ifndef NDEBUG
    if (idx < 0 || idx >= Length)
    {
        printf("vec index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    adj_value += adj_ret[idx];
    for(unsigned i=0; i < Length; ++i)
    {
        if (i != idx)
            adj_v[i] += adj_ret[i];
    }
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE Type length(vec_t<Length, Type> a)
{
    return sqrt(dot(a, a));
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE Type length_sq(vec_t<Length, Type> a)
{
    return dot(a, a);
}


template<typename Type>
inline CUDA_CALLABLE Type length(vec_t<2, Type> a)
{
    return sqrt(a.c[0] * a.c[0] + a.c[1] * a.c[1]);
}

template<typename Type>
inline CUDA_CALLABLE Type length(vec_t<3, Type> a)
{
    return sqrt(a.c[0] * a.c[0] + a.c[1] * a.c[1] + a.c[2] * a.c[2]);
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> normalize(vec_t<Length, Type> a)
{
    Type l = length(a);
    if (l > Type(kEps))
        return div(a,l);
    else
        return vec_t<Length, Type>();
}

template<typename Type>
inline CUDA_CALLABLE vec_t<2, Type> normalize(vec_t<2, Type> a)
{
    Type l = sqrt(a.c[0] * a.c[0] + a.c[1] * a.c[1]);
    if (l > Type(kEps))
        return vec_t<2, Type>(a.c[0]/l,a.c[1]/l);
    else
        return vec_t<2, Type>();
}

template<typename Type>
inline CUDA_CALLABLE vec_t<3, Type> normalize(vec_t<3, Type> a)
{
    Type l = sqrt(a.c[0] * a.c[0] + a.c[1] * a.c[1] + a.c[2] * a.c[2]);
    if (l > Type(kEps))
        return vec_t<3, Type>(a.c[0]/l,a.c[1]/l,a.c[2]/l);
    else
        return vec_t<3, Type>();
}


template<typename Type>
inline CUDA_CALLABLE vec_t<3,Type> cross(vec_t<3,Type> a, vec_t<3,Type> b)
{
    return {
        Type(a[1]*b[2] - a[2]*b[1]),
        Type(a[2]*b[0] - a[0]*b[2]),
        Type(a[0]*b[1] - a[1]*b[0])
    };
}


template<unsigned Length, typename Type>
inline bool CUDA_CALLABLE isfinite(vec_t<Length, Type> x)
{
    for( unsigned i=0; i < Length; ++i )
    {
        if(!isfinite(x[i]))
        {
            return false;
        }
    }
    return true;
}

template<unsigned Length, typename Type>
inline bool CUDA_CALLABLE isnan(vec_t<Length, Type> x)
{
    for( unsigned i=0; i < Length; ++i )
    {
        if(isnan(x[i]))
        {
            return true;
        }
    }
    return false;
}

template<unsigned Length, typename Type>
inline bool CUDA_CALLABLE isinf(vec_t<Length, Type> x)
{
    for( unsigned i=0; i < Length; ++i )
    {
        if(isinf(x[i]))
        {
            return true;
        }
    }
    return false;
}

// These two functions seem to compile very slowly
template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length,Type> min(vec_t<Length,Type> a, vec_t<Length,Type> b)
{
    vec_t<Length,Type> ret;
    for( unsigned i=0; i < Length; ++i )
    {
        ret[i] = a[i] < b[i] ? a[i] : b[i];
    }
    return ret;
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length,Type> max(vec_t<Length,Type> a, vec_t<Length,Type> b)
{
    vec_t<Length,Type> ret;
    for( unsigned i=0; i < Length; ++i )
    {
        ret[i] = a[i] > b[i] ? a[i] : b[i];
    }
    return ret;
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE Type min(vec_t<Length,Type> v)
{
    Type ret = v[0];
    for( unsigned i=1; i < Length; ++i )
    {
        if (v[i] < ret)
            ret = v[i];
    }
    return ret;
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE Type max(vec_t<Length,Type> v)
{
    Type ret = v[0];
    for( unsigned i=1; i < Length; ++i )
    {
        if (v[i] > ret)
            ret = v[i];
    }
    return ret;
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE unsigned argmin(vec_t<Length,Type> v)
{
    unsigned ret = 0;
    for( unsigned i=1; i < Length; ++i )
    {
        if (v[i] < v[ret])
            ret = i;
    }
    return ret;
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE unsigned argmax(vec_t<Length,Type> v)
{
    unsigned ret = 0;
    for( unsigned i=1; i < Length; ++i )
    {
        if (v[i] > v[ret])
            ret = i;
    }
    return ret;
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length,Type> abs(vec_t<Length,Type> v)
{
    vec_t<Length,Type> ret;
    for (unsigned i=0; i < Length; ++i)
    {
        ret[i] = abs(v[i]);
    }

    return ret;
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length,Type> sign(vec_t<Length,Type> v)
{
    vec_t<Length,Type> ret;
    for (unsigned i=0; i < Length; ++i)
    {
        ret[i] = v[i] < Type(0) ? Type(-1) : Type(1);
    }

    return ret;
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE void expect_near(const vec_t<Length, Type>& actual, const vec_t<Length, Type>& expected, const Type& tolerance)
{
    Type diff(0);
    for(size_t i=0; i<Length; ++i)
    {
        diff = max(diff,abs(actual[i] - expected[i]));
    }
    if (diff > tolerance)
    {
        printf("Error, expect_near() failed with tolerance "); print(tolerance);
        printf("    Expected: "); print(expected); 
        printf("    Actual: "); print(actual);
        printf("    Max absolute difference: "); print(diff);
    }
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_expect_near(const vec_t<Length, Type>& actual, const vec_t<Length, Type>& expected, Type tolerance, vec_t<Length, Type>& adj_actual, vec_t<Length, Type>& adj_expected, Type adj_tolerance)
{
    // nop
}

// adjoint for the initializer_array constructor:
template<unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_vec_t(const initializer_array<Length, Type> &cmps, const initializer_array<Length, Type*> &adj_cmps, const vec_t<Length, Type>& adj_ret)
{
    for(unsigned i=0; i < Length; ++i)
    {
        *(adj_cmps[i]) += adj_ret[i];
    }
}


// adjoint for the component constructors:
template<typename Type>
inline CUDA_CALLABLE void adj_vec_t(Type cmpx, Type cmpy, Type &adj_cmpx, Type &adj_cmpy, const vec_t<2, Type>& adj_ret)
{
    adj_cmpx += adj_ret.c[0];
    adj_cmpy += adj_ret.c[1];
}

template<typename Type>
inline CUDA_CALLABLE void adj_vec_t(Type cmpx, Type cmpy, Type cmpz, Type &adj_cmpx, Type &adj_cmpy, Type &adj_cmpz, const vec_t<3, Type>& adj_ret)
{
    adj_cmpx += adj_ret.c[0];
    adj_cmpy += adj_ret.c[1];
    adj_cmpz += adj_ret.c[2];
}

template<typename Type>
inline CUDA_CALLABLE void adj_vec_t(Type cmpx, Type cmpy, Type cmpz, Type cmpw, Type &adj_cmpx, Type &adj_cmpy, Type &adj_cmpz, Type &adj_cmpw, const vec_t<4, Type>& adj_ret)
{
    adj_cmpx += adj_ret.c[0];
    adj_cmpy += adj_ret.c[1];
    adj_cmpz += adj_ret.c[2];
    adj_cmpw += adj_ret.c[3];
}

// adjoint for the constant constructor:
template<unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_vec_t(Type s, Type& adj_s, const vec_t<Length, Type>& adj_ret)
{
    for( unsigned i=0; i < Length; ++i )
    {
        adj_s += adj_ret[i];
    }
}

// adjoint for the casting constructor
template<unsigned Length, typename Type, typename OtherType>
inline CUDA_CALLABLE void adj_vec_t(const vec_t<Length, OtherType>& other, vec_t<Length, OtherType>& adj_other, const vec_t<Length, Type>& adj_ret)
{
    for( unsigned i=0; i < Length; ++i )
    {
        adj_other[i] += static_cast<OtherType>(adj_ret[i]);
    }
}

template<typename Type>
CUDA_CALLABLE inline void adj_vec_t(const vec_t<3,Type>& w, const vec_t<3,Type>& v, vec_t<3,Type>& adj_w, vec_t<3,Type>& adj_v, const vec_t<6,Type>& adj_ret)
{
    adj_w[0] += adj_ret[0];
    adj_w[1] += adj_ret[1];
    adj_w[2] += adj_ret[2];
    adj_v[0] += adj_ret[3];
    adj_v[1] += adj_ret[4];
    adj_v[2] += adj_ret[5];
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_mul(vec_t<Length, Type> a, Type s, vec_t<Length, Type>& adj_a, Type& adj_s, const vec_t<Length, Type>& adj_ret)
{
    for( unsigned i=0; i < Length; ++i )
    {
        adj_a[i] += s*adj_ret[i];
    }

    adj_s += dot(a, adj_ret);

#if FP_CHECK
    if (!isfinite(a) || !isfinite(s) || !isfinite(adj_a) || !isfinite(adj_s) || !isfinite(adj_ret))
    {
        // \TODO: How shall we implement this error message?
        //printf("adj_mul((%f %f %f %f), %f, (%f %f %f %f), %f, (%f %f %f %f)\n", a.x, a.y, a.z, a.w, s, adj_a.x, adj_a.y, adj_a.z, adj_a.w, adj_s, adj_ret.x, adj_ret.y, adj_ret.z, adj_ret.w);
        assert(0);
    }
#endif
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_mul(Type s, vec_t<Length, Type> a, Type& adj_s, vec_t<Length, Type>& adj_a, const vec_t<Length, Type>& adj_ret)
{
    adj_mul(a, s, adj_a, adj_s, adj_ret);
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_cw_mul(vec_t<Length, Type> a, vec_t<Length, Type> b, vec_t<Length, Type>& adj_a, vec_t<Length, Type>& adj_b, const vec_t<Length, Type>& adj_ret)
{
  adj_a += cw_mul(b, adj_ret);
  adj_b += cw_mul(a, adj_ret);
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_div(vec_t<Length, Type> a, Type s, vec_t<Length, Type>& adj_a, Type& adj_s, const vec_t<Length, Type>& adj_ret)
{

    adj_s -= dot(a , adj_ret)/ (s * s); // - a / s^2

    for( unsigned i=0; i < Length; ++i )
    {
        adj_a[i] += adj_ret[i] / s;
    }

#if FP_CHECK
    if (!isfinite(a) || !isfinite(s) || !isfinite(adj_a) || !isfinite(adj_s) || !isfinite(adj_ret))
    {
        // \TODO: How shall we implement this error message?
        // printf("adj_div((%f %f %f %f), %f, (%f %f %f %f), %f, (%f %f %f %f)\n", a.x, a.y, a.z, a.w, s, adj_a.x, adj_a.y, adj_a.z, adj_a.w, adj_s, adj_ret.x, adj_ret.y, adj_ret.z, adj_ret.w);
        assert(0);
    }
#endif
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_div(Type s, vec_t<Length, Type> a, Type& adj_s, vec_t<Length, Type>& adj_a, const vec_t<Length, Type>& adj_ret)
{

    adj_s -= dot(a , adj_ret)/ (s * s); // - a / s^2

    for( unsigned i=0; i < Length; ++i )
    {
        adj_a[i] += s / adj_ret[i];
    }

#if FP_CHECK
    if (!isfinite(a) || !isfinite(s) || !isfinite(adj_a) || !isfinite(adj_s) || !isfinite(adj_ret))
    {
        // \TODO: How shall we implement this error message?
        // printf("adj_div((%f %f %f %f), %f, (%f %f %f %f), %f, (%f %f %f %f)\n", a.x, a.y, a.z, a.w, s, adj_a.x, adj_a.y, adj_a.z, adj_a.w, adj_s, adj_ret.x, adj_ret.y, adj_ret.z, adj_ret.w);
        assert(0);
    }
#endif
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_cw_div(vec_t<Length, Type> a, vec_t<Length, Type> b, vec_t<Length, Type>& ret, vec_t<Length, Type>& adj_a, vec_t<Length, Type>& adj_b, const vec_t<Length, Type>& adj_ret) {
  adj_a += cw_div(adj_ret, b);
  adj_b -= cw_mul(adj_ret, cw_div(ret, b));
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_add(vec_t<Length, Type> a, vec_t<Length, Type> b, vec_t<Length, Type>& adj_a, vec_t<Length, Type>& adj_b, const vec_t<Length, Type>& adj_ret)
{
    adj_a += adj_ret;
    adj_b += adj_ret;
}

template<typename Type>
inline CUDA_CALLABLE void adj_add(vec_t<2, Type> a, vec_t<2, Type> b, vec_t<2, Type>& adj_a, vec_t<2, Type>& adj_b, const vec_t<2, Type>& adj_ret)
{
    adj_a.c[0] += adj_ret.c[0];
    adj_a.c[1] += adj_ret.c[1];
    adj_b.c[0] += adj_ret.c[0];
    adj_b.c[1] += adj_ret.c[1];
}

template<typename Type>
inline CUDA_CALLABLE void adj_add(vec_t<3, Type> a, vec_t<3, Type> b, vec_t<3, Type>& adj_a, vec_t<3, Type>& adj_b, const vec_t<3, Type>& adj_ret)
{
    adj_a.c[0] += adj_ret.c[0];
    adj_a.c[1] += adj_ret.c[1];
    adj_a.c[2] += adj_ret.c[2];
    adj_b.c[0] += adj_ret.c[0];
    adj_b.c[1] += adj_ret.c[1];
    adj_b.c[2] += adj_ret.c[2];
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_sub(vec_t<Length, Type> a, vec_t<Length, Type> b, vec_t<Length, Type>& adj_a, vec_t<Length, Type>& adj_b, const vec_t<Length, Type>& adj_ret)
{
    adj_a += adj_ret;
    adj_b -= adj_ret;
}

template<typename Type>
inline CUDA_CALLABLE void adj_sub(vec_t<2, Type> a, vec_t<2, Type> b, vec_t<2, Type>& adj_a, vec_t<2, Type>& adj_b, const vec_t<2, Type>& adj_ret)
{
    adj_a.c[0] += adj_ret.c[0];
    adj_a.c[1] += adj_ret.c[1];
    adj_b.c[0] -= adj_ret.c[0];
    adj_b.c[1] -= adj_ret.c[1];
}

template<typename Type>
inline CUDA_CALLABLE void adj_sub(vec_t<3, Type> a, vec_t<3, Type> b, vec_t<3, Type>& adj_a, vec_t<3, Type>& adj_b, const vec_t<3, Type>& adj_ret)
{
    adj_a.c[0] += adj_ret.c[0];
    adj_a.c[1] += adj_ret.c[1];
    adj_a.c[2] += adj_ret.c[2];
    adj_b.c[0] -= adj_ret.c[0];
    adj_b.c[1] -= adj_ret.c[1];
    adj_b.c[2] -= adj_ret.c[2];
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_mod(vec_t<Length, Type> a, vec_t<Length, Type> b, vec_t<Length, Type>& adj_a, vec_t<Length, Type>& adj_b, const vec_t<Length, Type>& adj_ret)
{
}

template<typename Type>
inline CUDA_CALLABLE void adj_mod(vec_t<2, Type> a, vec_t<2, Type> b, vec_t<2, Type>& adj_a, vec_t<2, Type>& adj_b, const vec_t<2, Type>& adj_ret)
{
}

template<typename Type>
inline CUDA_CALLABLE void adj_mod(vec_t<3, Type> a, vec_t<3, Type> b, vec_t<3, Type>& adj_a, vec_t<3, Type>& adj_b, const vec_t<3, Type>& adj_ret)
{
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_dot(vec_t<Length, Type> a, vec_t<Length, Type> b, vec_t<Length, Type>& adj_a, vec_t<Length, Type>& adj_b, const Type adj_ret)
{
    adj_a += b*adj_ret;
    adj_b += a*adj_ret;

#if FP_CHECK
    if (!isfinite(a) || !isfinite(b) || !isfinite(adj_a) || !isfinite(adj_b) || !isfinite(adj_ret))
    {
        // \TODO: How shall we implement this error message?
        //printf("adj_dot((%f %f %f %f), (%f %f %f %f), (%f %f %f %f), (%f %f %f %f), %f)\n", a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w, adj_a.x, adj_a.y, adj_a.z, adj_a.w, adj_b.x, adj_b.y, adj_b.z, adj_b.w, adj_ret);
        assert(0);
    }
#endif
}



template<typename Type>
inline CUDA_CALLABLE void adj_dot(vec_t<2, Type> a, vec_t<2, Type> b, vec_t<2, Type>& adj_a, vec_t<2, Type>& adj_b, const Type adj_ret)
{
    adj_a.c[0] += b.c[0]*adj_ret;
    adj_a.c[1] += b.c[1]*adj_ret;

    adj_b.c[0] += a.c[0]*adj_ret;
    adj_b.c[1] += a.c[1]*adj_ret;
}

template<typename Type>
inline CUDA_CALLABLE void adj_dot(vec_t<3, Type> a, vec_t<3, Type> b, vec_t<3, Type>& adj_a, vec_t<3, Type>& adj_b, const Type adj_ret)
{
    adj_a.c[0] += b.c[0]*adj_ret;
    adj_a.c[1] += b.c[1]*adj_ret;
    adj_a.c[2] += b.c[2]*adj_ret;

    adj_b.c[0] += a.c[0]*adj_ret;
    adj_b.c[1] += a.c[1]*adj_ret;
    adj_b.c[2] += a.c[2]*adj_ret;
}


template<unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_extract(const vec_t<Length, Type> & a, int idx, vec_t<Length, Type> & adj_a, int & adj_idx, Type & adj_ret)
{
#ifndef NDEBUG
    if (idx < 0 || idx > Length)
    {
        printf("Tvec2<Scalar> index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    adj_a[idx] += adj_ret;
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_length(vec_t<Length, Type> a, Type ret, vec_t<Length, Type>& adj_a, const Type adj_ret)
{
    if (ret > Type(kEps))
    {
        adj_a += div(a, ret) * adj_ret;
    }

#if FP_CHECK
    if (!isfinite(adj_a))
    {
        // \TODO: How shall we implement this error message?
        //printf("%s:%d - adj_length((%f %f %f %f), (%f %f %f %f), (%f))\n", __FILE__, __LINE__, a.x, a.y, a.z, a.w, adj_a.x, adj_a.y, adj_a.z, adj_a.w, adj_ret);
        assert(0);
    }
#endif
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_length_sq(vec_t<Length, Type> a, vec_t<Length, Type>& adj_a, const Type adj_ret)
{
    adj_a += Type(2.0)*a*adj_ret;

#if FP_CHECK
    if (!isfinite(adj_a))
    {
        // \TODO: How shall we implement this error message?
        //printf("%s:%d - adj_length((%f %f %f %f), (%f %f %f %f), (%f))\n", __FILE__, __LINE__, a.x, a.y, a.z, a.w, adj_a.x, adj_a.y, adj_a.z, adj_a.w, adj_ret);
        assert(0);
    }
#endif
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_normalize(vec_t<Length, Type> a, vec_t<Length, Type>& ret, vec_t<Length, Type>& adj_a, const vec_t<Length, Type>& adj_ret)
{
    Type d = length(a);
    
    if (d > Type(kEps))
    {
        Type invd = Type(1.0f)/d;

        adj_a += (adj_ret*invd - ret*(dot(ret, adj_ret))*invd);

#if FP_CHECK
        if (!isfinite(adj_a))
        {
            // \TODO: How shall we implement this error message?
            //printf("%s:%d - adj_normalize((%f %f %f %f), (%f %f %f %f), (%f, %f, %f, %f))\n", __FILE__, __LINE__, a.x, a.y, a.z, a.w, adj_a.x, adj_a.y, adj_a.z, adj_a.w, adj_ret.x, adj_ret.y, adj_ret.z, adj_ret.w);
            assert(0);
        }
#endif
    }
}

template<typename Type>
inline CUDA_CALLABLE void adj_cross(vec_t<3,Type> a, vec_t<3,Type> b, vec_t<3,Type>& adj_a, vec_t<3,Type>& adj_b, const vec_t<3,Type>& adj_ret)
{
    // todo: sign check
    adj_a += cross(b, adj_ret);
    adj_b -= cross(a, adj_ret);
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_isfinite(const vec_t<Length, Type> &x, vec_t<Length,Type>& adj_x, const bool &adj_ret)
{

}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_isnan(const vec_t<Length, Type> &x, vec_t<Length,Type>& adj_x, const bool &adj_ret)
{

}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_isinf(const vec_t<Length, Type> &x, vec_t<Length,Type>& adj_x, const bool &adj_ret)
{

}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_min(const vec_t<Length,Type> &a, const vec_t<Length,Type> &b, vec_t<Length,Type>& adj_a, vec_t<Length,Type>& adj_b, const vec_t<Length,Type> &adj_ret)
{
    for( unsigned i=0; i < Length; ++i )
    {
        if (a[i] < b[i])
            adj_a[i] += adj_ret[i];
        else
            adj_b[i] += adj_ret[i];
    }
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_max(const vec_t<Length,Type> &a, const vec_t<Length,Type> &b, vec_t<Length,Type>& adj_a, vec_t<Length,Type>& adj_b, const vec_t<Length,Type> &adj_ret)
{
    for( unsigned i=0; i < Length; ++i )
    {
        if (a[i] > b[i])
            adj_a[i] += adj_ret[i];
        else
            adj_b[i] += adj_ret[i];
    }
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_min(const vec_t<Length,Type> &v, vec_t<Length,Type>& adj_v, const Type &adj_ret)
{
    unsigned i = argmin(v);
    adj_v[i] += adj_ret;
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_max(const vec_t<Length,Type> &v, vec_t<Length,Type>& adj_v, const Type &adj_ret)
{
    unsigned i = argmax(v);
    adj_v[i] += adj_ret;
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_abs(
    const vec_t<Length,Type>& v,
    vec_t<Length,Type>& adj_v,
    const vec_t<Length,Type>& adj_ret
)
{
    for (unsigned i=0; i < Length; ++i)
    {
        if (v[i] < Type(0))
        {
            adj_v[i] -= adj_ret[i];
        }
        else
        {
            adj_v[i] += adj_ret[i];
        }
    }
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_sign(
    const vec_t<Length,Type>& v,
    vec_t<Length,Type>& adj_v,
    const vec_t<Length,Type>& adj_ret
)
{
    for (unsigned i=0; i < Length; ++i)
    {
        if (v[i] < Type(0))
        {
            adj_v[i] -= adj_ret[i];
        }
        else
        {
            adj_v[i] += adj_ret[i];
        }
    }
}

// Do I need to specialize these for different lengths?
template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> atomic_add(vec_t<Length, Type> * addr, vec_t<Length, Type> value)
{
    vec_t<Length, Type> ret;
    for( unsigned i=0; i < Length; ++i )
    {
        ret[i] = atomic_add(&(addr -> c[i]), value[i]);
    }

    return ret;
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> atomic_min(vec_t<Length, Type> * addr, vec_t<Length, Type> value)
{
    vec_t<Length, Type> ret;
    for( unsigned i=0; i < Length; ++i )
    {
        ret[i] = atomic_min(&(addr -> c[i]), value[i]);
    }

    return ret;
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE vec_t<Length, Type> atomic_max(vec_t<Length, Type> * addr, vec_t<Length, Type> value)
{
    vec_t<Length, Type> ret;
    for( unsigned i=0; i < Length; ++i )
    {
        ret[i] = atomic_max(&(addr -> c[i]), value[i]);
    }

    return ret;
}

template<unsigned Length, typename Type>
inline CUDA_CALLABLE void adj_atomic_minmax(
    vec_t<Length,Type> *addr,
    vec_t<Length,Type> *adj_addr,
    const vec_t<Length,Type> &value,
    vec_t<Length,Type> &adj_value)
{
    for (unsigned i=0; i < Length; ++i)
        adj_atomic_minmax(&(addr->c[i]), &(adj_addr->c[i]), value[i], adj_value[i]);
}

// ok, the original implementation of this didn't take the absolute values.
// I wouldn't consider this expected behavior. It looks like it's only
// being used for bounding boxes at the moment, where this doesn't matter,
// but you often use it for ray tracing where it does. Not sure if the
// fabs() incurs a performance hit...
template<unsigned Length, typename Type>
CUDA_CALLABLE inline int longest_axis(const vec_t<Length, Type>& v)
{
    Type lmax = abs(v[0]);
    int ret(0);
    for( unsigned i=1; i < Length; ++i )
    {
        Type l = abs(v[i]);
        if( l > lmax )
        {
            ret = i;
            lmax = l;
        }
    }
    return ret;
}

template<unsigned Length, typename Type>
CUDA_CALLABLE inline vec_t<Length,Type> lerp(const vec_t<Length,Type>& a, const vec_t<Length,Type>& b, Type t)
{
    return a*(Type(1)-t) + b*t;
}

template<unsigned Length, typename Type>
CUDA_CALLABLE inline void adj_lerp(const vec_t<Length,Type>& a, const vec_t<Length,Type>& b, Type t, vec_t<Length,Type>& adj_a, vec_t<Length,Type>& adj_b, Type& adj_t, const vec_t<Length,Type>& adj_ret)
{
    adj_a += adj_ret*(Type(1)-t);
    adj_b += adj_ret*t;
    adj_t += tensordot(b, adj_ret) - tensordot(a, adj_ret);
}

// for integral types we do not accumulate gradients
template<unsigned Length> CUDA_CALLABLE inline void adj_atomic_add(vec_t<Length, int8>* buf, const vec_t<Length, int8> &value) { }
template<unsigned Length> CUDA_CALLABLE inline void adj_atomic_add(vec_t<Length, uint8>* buf, const vec_t<Length, uint8> &value) { }
template<unsigned Length> CUDA_CALLABLE inline void adj_atomic_add(vec_t<Length, int16>* buf, const vec_t<Length, int16> &value) { }
template<unsigned Length> CUDA_CALLABLE inline void adj_atomic_add(vec_t<Length, uint16>* buf, const vec_t<Length, uint16> &value) { }
template<unsigned Length> CUDA_CALLABLE inline void adj_atomic_add(vec_t<Length, int32>* buf, const vec_t<Length, int32> &value) { }
template<unsigned Length> CUDA_CALLABLE inline void adj_atomic_add(vec_t<Length, uint32>* buf, const vec_t<Length, uint32> &value) { }
template<unsigned Length> CUDA_CALLABLE inline void adj_atomic_add(vec_t<Length, int64>* buf, const vec_t<Length, int64> &value) { }
template<unsigned Length> CUDA_CALLABLE inline void adj_atomic_add(vec_t<Length, uint64>* buf, const vec_t<Length, uint64> &value) { }


// adjoints for some of the constructors, used in intersect.h
inline CUDA_CALLABLE void adj_vec2(float x, float y, float& adj_x, float& adj_y, const vec2& adj_ret)
{
    adj_x += adj_ret[0];
    adj_y += adj_ret[1];
}

inline CUDA_CALLABLE void adj_vec3(float x, float y, float z, float& adj_x, float& adj_y, float& adj_z, const vec3& adj_ret)
{
    adj_x += adj_ret[0];
    adj_y += adj_ret[1];
    adj_z += adj_ret[2];
}

inline CUDA_CALLABLE void adj_vec4(float x, float y, float z, float w, float& adj_x, float& adj_y, float& adj_z, float& adj_w, const vec4& adj_ret)
{
    adj_x += adj_ret[0];
    adj_y += adj_ret[1];
    adj_z += adj_ret[2];
    adj_w += adj_ret[3];
}

inline CUDA_CALLABLE void adj_vec3(float s, float& adj_s, const vec3& adj_ret)
{
    adj_vec_t(s, adj_s, adj_ret);
}

inline CUDA_CALLABLE void adj_vec4(float s, float& adj_s, const vec4& adj_ret)
{
    adj_vec_t(s, adj_s, adj_ret);
}

template<unsigned Length, typename Type>
CUDA_CALLABLE inline int len(const vec_t<Length, Type>& x)
{
    return Length;
}

template<unsigned Length, typename Type>
CUDA_CALLABLE inline void adj_len(const vec_t<Length, Type>& x, vec_t<Length, Type>& adj_x, const int& adj_ret)
{
}

} // namespace wp
