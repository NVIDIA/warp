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

#include "mat.h"

namespace wp
{

template<typename Type>
struct quat_t
{
    // zero constructor for adjoint variable initialization
    inline CUDA_CALLABLE quat_t(Type x=Type(0), Type y=Type(0), Type z=Type(0), Type w=Type(0)) : x(x), y(y), z(z), w(w) {}    
    explicit inline CUDA_CALLABLE quat_t(const vec_t<3,Type>& v, Type w=Type(0)) : x(v[0]), y(v[1]), z(v[2]), w(w) {}
    
    template<typename OtherType>
    explicit inline CUDA_CALLABLE quat_t(const quat_t<OtherType>& other)
    {
        x = static_cast<Type>(other.x);
        y = static_cast<Type>(other.y);
        z = static_cast<Type>(other.z);
        w = static_cast<Type>(other.w);
    }

    inline CUDA_CALLABLE quat_t(const initializer_array<4, Type> &l)
    {
        x = l[0];
        y = l[1];
        z = l[2];
        w = l[3];
    }

    // imaginary part
    Type x;
    Type y;
    Type z;
    
    // real part
    Type w;

    inline CUDA_CALLABLE Type operator[](int index) const
    {
        switch (index)
        {
            case 0:
                return x;
            case 1:
                return y;
            case 2:
                return z;
            case 3:
                return w;
            default:
                assert(0);
                return x;
        }
    }

    inline CUDA_CALLABLE Type& operator[](int index)
    {
        switch (index)
        {
            case 0:
                return x;
            case 1:
                return y;
            case 2:
                return z;
            case 3:
                return w;
            default:
                assert(0);
                return x;
        }
    }
};

using quat = quat_t<float>;
using quath = quat_t<half>;
using quatf = quat_t<float>;
using quatd = quat_t<double>;


template<typename Type>
inline CUDA_CALLABLE bool operator==(const quat_t<Type>& a, const quat_t<Type>& b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

template<typename Type>
inline bool CUDA_CALLABLE isfinite(const quat_t<Type>& q)
{
    return isfinite(q.x) && isfinite(q.y) && isfinite(q.z) && isfinite(q.w);
}

template<typename Type>
inline void CUDA_CALLABLE adj_isfinite(const quat_t<Type>& q, quat_t<Type>& adj_q, const bool &adj_ret)
{
}

template<typename Type>
inline bool CUDA_CALLABLE isnan(const quat_t<Type>& q)
{
    return isnan(q.x) || isnan(q.y) || isnan(q.z) || isnan(q.w);
}

template<typename Type>
inline void CUDA_CALLABLE adj_isnan(const quat_t<Type>& q, quat_t<Type>& adj_q, const bool &adj_ret)
{
}

template<typename Type>
inline bool CUDA_CALLABLE isinf(const quat_t<Type>& q)
{
    return isinf(q.x) || isinf(q.y) || isinf(q.z) || isinf(q.w);
}

template<typename Type>
inline void CUDA_CALLABLE adj_isinf(const quat_t<Type>& q, quat_t<Type>& adj_q, const bool &adj_ret)
{
}

template<typename Type>
inline CUDA_CALLABLE quat_t<Type> atomic_add(quat_t<Type> * addr, quat_t<Type> value) 
{
    Type x = atomic_add(&(addr -> x), value.x);
    Type y = atomic_add(&(addr -> y), value.y);
    Type z = atomic_add(&(addr -> z), value.z);
    Type w = atomic_add(&(addr -> w), value.w);

    return quat_t<Type>(x, y, z, w);
}

template<typename Type>
inline CUDA_CALLABLE void adj_quat_t(Type x, Type y, Type z, Type w, Type& adj_x, Type& adj_y, Type& adj_z, Type& adj_w, quat_t<Type> adj_ret)
{
    adj_x += adj_ret.x;
    adj_y += adj_ret.y;
    adj_z += adj_ret.z;
    adj_w += adj_ret.w;
}

template<typename Type>
inline CUDA_CALLABLE void adj_quat_t(const vec_t<3,Type>& v, Type w, vec_t<3,Type>& adj_v, Type& adj_w, quat_t<Type> adj_ret)
{
    adj_v[0] += adj_ret.x;
    adj_v[1] += adj_ret.y;
    adj_v[2] += adj_ret.z;
    adj_w    += adj_ret.w;
}

// casting constructor adjoint
template<typename Type, typename OtherType>
inline CUDA_CALLABLE void adj_quat_t(const quat_t<OtherType>& other, quat_t<OtherType>& adj_other, const quat_t<Type>& adj_ret)
{
    adj_other.x += static_cast<OtherType>(adj_ret.x);
    adj_other.y += static_cast<OtherType>(adj_ret.y);
    adj_other.z += static_cast<OtherType>(adj_ret.z);
    adj_other.w += static_cast<OtherType>(adj_ret.w);
}

// forward methods

template<typename Type>
inline CUDA_CALLABLE quat_t<Type> quat_from_axis_angle(const vec_t<3,Type>& axis, Type angle)
{
    Type half = angle*Type(Type(0.5));
    Type w = cos(half);

    Type sin_theta_over_two = sin(half);
    vec_t<3,Type> v = axis*sin_theta_over_two;

    return quat_t<Type>(v[0], v[1], v[2], w);
}

template<typename Type>
inline CUDA_CALLABLE void quat_to_axis_angle(const quat_t<Type>& q, vec_t<3,Type>& axis, Type& angle)
{
    vec_t<3,Type> v = vec_t<3,Type>(q.x, q.y, q.z);
    axis = q.w < Type(0) ? -normalize(v) : normalize(v);
    angle = Type(2) * atan2(length(v), abs(q.w));
}

template<typename Type>
inline CUDA_CALLABLE quat_t<Type> quat_rpy(Type roll, Type pitch, Type yaw)
{
    Type cy = cos(yaw * Type(0.5));
    Type sy = sin(yaw * Type(0.5));
    Type cr = cos(roll * Type(0.5));
    Type sr = sin(roll * Type(0.5));
    Type cp = cos(pitch * Type(0.5));
    Type sp = sin(pitch * Type(0.5));

    Type w = (cy * cr * cp + sy * sr * sp);
    Type x = (cy * sr * cp - sy * cr * sp);
    Type y = (cy * cr * sp + sy * sr * cp);
    Type z = (sy * cr * cp - cy * sr * sp);

    return quat_t<Type>(x, y, z, w);
}



template<typename Type>
inline CUDA_CALLABLE quat_t<Type> quat_inverse(const quat_t<Type>& q)
{
    return quat_t<Type>(-q.x, -q.y, -q.z, q.w);
}


template<typename Type>
inline CUDA_CALLABLE Type dot(const quat_t<Type>& a, const quat_t<Type>& b)
{
    return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
}

template<typename Type>
inline CUDA_CALLABLE Type tensordot(const quat_t<Type>& a, const quat_t<Type>& b)
{
    // corresponds to `np.tensordot()` with all axes being contracted
    return dot(a, b);
}

template<typename Type>
inline CUDA_CALLABLE Type length(const quat_t<Type>& q)
{
    return sqrt(dot(q, q));
}

template<typename Type>
inline CUDA_CALLABLE Type length_sq(const quat_t<Type>& q)
{
    return dot(q, q);
}

template<typename Type>
inline CUDA_CALLABLE quat_t<Type> normalize(const quat_t<Type>& q)
{
    Type l = length(q);
    if (l > Type(kEps))
    {
        Type inv_l = Type(1)/l;

        return quat_t<Type>(q.x*inv_l, q.y*inv_l, q.z*inv_l, q.w*inv_l);
    }
    else
    {
        return quat_t<Type>(Type(0), Type(0), Type(0), Type(1));
    }
}

template<typename Type>
inline CUDA_CALLABLE quat_t<Type> add(const quat_t<Type>& a, const quat_t<Type>& b)
{
    return quat_t<Type>(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);
}

template<typename Type>
inline CUDA_CALLABLE quat_t<Type> sub(const quat_t<Type>& a, const quat_t<Type>& b)
{
    return quat_t<Type>(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w);
}

template<typename Type>
inline CUDA_CALLABLE quat_t<Type> operator - (const quat_t<Type>& q)
{
    return quat_t<Type>(-q.x, -q.y, -q.z, -q.w);
}

template<typename Type>
CUDA_CALLABLE inline quat_t<Type> pos(const quat_t<Type>& q)
{
    return q;
}

template<typename Type>
CUDA_CALLABLE inline quat_t<Type> neg(const quat_t<Type>& q)
{
    return -q;
}

template<typename Type>
CUDA_CALLABLE inline void adj_neg(const quat_t<Type>& q, quat_t<Type>& adj_q, const quat_t<Type>& adj_ret)
{
    adj_q -= adj_ret;
}

template<typename Type>
inline CUDA_CALLABLE quat_t<Type> mul(const quat_t<Type>& a, const quat_t<Type>& b)
{
    return quat_t<Type>(a.w*b.x + b.w*a.x + a.y*b.z - b.y*a.z,
                a.w*b.y + b.w*a.y + a.z*b.x - b.z*a.x,
                a.w*b.z + b.w*a.z + a.x*b.y - b.x*a.y,
                a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z);
}

template<typename Type>
inline CUDA_CALLABLE quat_t<Type> mul(const quat_t<Type>& a, Type s)
{
    return quat_t<Type>(a.x*s, a.y*s, a.z*s, a.w*s);    
}

template<typename Type>
inline CUDA_CALLABLE quat_t<Type> mul(Type s, const quat_t<Type>& a)
{
    return mul(a, s);
}

template<typename Type>
inline CUDA_CALLABLE quat_t<Type> div(quat_t<Type> q, Type s)
{
    return quat_t<Type>(q.x/s, q.y/s, q.z/s, q.w/s);
}

template<typename Type>
inline CUDA_CALLABLE quat_t<Type> div(Type s, quat_t<Type> q)
{
    return quat_t<Type>(s/q.x, s/q.y, s/q.z, s/q.w);
}

template<typename Type>
inline CUDA_CALLABLE quat_t<Type> operator / (quat_t<Type> a, Type s)
{
    return div(a,s);
}

template<typename Type>
inline CUDA_CALLABLE quat_t<Type> operator / (Type s, quat_t<Type> a)
{
    return div(s,a);
}

template<typename Type>
inline CUDA_CALLABLE quat_t<Type> operator*(Type s, const quat_t<Type>& a)
{
    return mul(a, s);
}

template<typename Type>
inline CUDA_CALLABLE quat_t<Type> operator*(const quat_t<Type>& a, Type s)
{
    return mul(a, s);
}

template<typename Type>
inline CUDA_CALLABLE vec_t<3,Type> quat_rotate(const quat_t<Type>& q, const vec_t<3,Type>& x)
{
    Type c = (Type(2)*q.w*q.w-Type(1));
    Type d = Type(2)*(q.x*x.c[0] + q.y*x.c[1] + q.z*x.c[2]);
    return vec_t<3,Type>(
        x.c[0]*c + q.x*d + (q.y * x[2] - q.z * x[1])*q.w*Type(2),
        x.c[1]*c + q.y*d + (q.z * x[0] - q.x * x[2])*q.w*Type(2),
        x.c[2]*c + q.z*d + (q.x * x[1] - q.y * x[0])*q.w*Type(2)
    );
}

template<typename Type>
inline CUDA_CALLABLE vec_t<3,Type> quat_rotate_inv(const quat_t<Type>& q, const vec_t<3,Type>& x)
{
    Type c = (Type(2)*q.w*q.w-Type(1));
    Type d = Type(2)*(q.x*x.c[0] + q.y*x.c[1] + q.z*x.c[2]);
    return vec_t<3,Type>(
        x.c[0]*c + q.x*d - (q.y * x[2] - q.z * x[1])*q.w*Type(2),
        x.c[1]*c + q.y*d - (q.z * x[0] - q.x * x[2])*q.w*Type(2),
        x.c[2]*c + q.z*d - (q.x * x[1] - q.y * x[0])*q.w*Type(2)
    );
}

template<typename Type>
inline CUDA_CALLABLE quat_t<Type> quat_slerp(const quat_t<Type>& q0, const quat_t<Type>& q1, Type t)
{
    vec_t<3,Type> axis;
    Type angle;
    quat_to_axis_angle(mul(quat_inverse(q0), q1), axis, angle);
    return mul(q0, quat_from_axis_angle(axis, t * angle));
}

template<typename Type>
inline CUDA_CALLABLE mat_t<3,3,Type> quat_to_matrix(const quat_t<Type>& q)
{
    vec_t<3,Type> c1 = quat_rotate(q, vec_t<3,Type>(1.0, 0.0, 0.0));
    vec_t<3,Type> c2 = quat_rotate(q, vec_t<3,Type>(0.0, 1.0, 0.0));
    vec_t<3,Type> c3 = quat_rotate(q, vec_t<3,Type>(0.0, 0.0, 1.0));

    return matrix_from_cols<Type>(c1, c2, c3);
}

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE quat_t<Type> quat_from_matrix(const mat_t<Rows,Cols,Type>& m)
{
    static_assert((Rows == 3 && Cols == 3) || (Rows == 4 && Cols == 4), "Non-square matrix");

    const Type tr = m.data[0][0] + m.data[1][1] + m.data[2][2];
    Type x, y, z, w, h = Type(0);

    if (tr >= Type(0)) {
        h = sqrt(tr + Type(1));
        w = Type(0.5) * h;
        h = Type(0.5) / h;

        x = (m.data[2][1] - m.data[1][2]) * h;
        y = (m.data[0][2] - m.data[2][0]) * h;
        z = (m.data[1][0] - m.data[0][1]) * h;
    } else {
        size_t max_diag = 0;
        if (m.data[1][1] > m.data[0][0]) {
            max_diag = 1;
        }
        if (m.data[2][2] > m.data[max_diag][max_diag]) {
            max_diag = 2;
        }

        if (max_diag == 0) {
            h = sqrt((m.data[0][0] - (m.data[1][1] + m.data[2][2])) + Type(1));
            x = Type(0.5) * h;
            h = Type(0.5) / h;

            y = (m.data[0][1] + m.data[1][0]) * h;
            z = (m.data[2][0] + m.data[0][2]) * h;
            w = (m.data[2][1] - m.data[1][2]) * h;
        } else if (max_diag == 1) {
            h = sqrt((m.data[1][1] - (m.data[2][2] + m.data[0][0])) + Type(1));
            y = Type(0.5) * h;
            h = Type(0.5) / h;

            z = (m.data[1][2] + m.data[2][1]) * h;
            x = (m.data[0][1] + m.data[1][0]) * h;
            w = (m.data[0][2] - m.data[2][0]) * h;
        } if (max_diag == 2) {
            h = sqrt((m.data[2][2] - (m.data[0][0] + m.data[1][1])) + Type(1));
            z = Type(0.5) * h;
            h = Type(0.5) / h;

            x = (m.data[2][0] + m.data[0][2]) * h;
            y = (m.data[1][2] + m.data[2][1]) * h;
            w = (m.data[1][0] - m.data[0][1]) * h;
        }
    }

    return normalize(quat_t<Type>(x, y, z, w));
}

template<typename Type>
inline CUDA_CALLABLE Type extract(const quat_t<Type>& a, int idx)
{
#if FP_CHECK
    if (idx < 0 || idx > 3)
    {
        printf("quat_t index %d out of bounds at %s %d", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    /*
    * Because quat data is not stored in an array, we index the quaternion by checking all possible idx values.
    * (&a.x)[idx] would be the preferred access strategy, but this results in undefined behavior in the clang compiler
    * at optimization level 3.
    */ 
    if (idx == 0)       {return a.x;}
    else if (idx == 1)  {return a.y;}
    else if (idx == 2)  {return a.z;}
    else                {return a.w;}
}

template<typename Type>
inline CUDA_CALLABLE Type* index(quat_t<Type>& q, int idx)
{
#ifndef NDEBUG
    if (idx < 0 || idx > 3)
    {
        printf("quat index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    return &q[idx];
}

template<typename Type>
inline CUDA_CALLABLE Type* indexref(quat_t<Type>* q, int idx)
{
#ifndef NDEBUG
    if (idx < 0 || idx > 3)
    {
        printf("quat store %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    return &((*q)[idx]);
}

template<typename Type>
inline CUDA_CALLABLE void adj_index(quat_t<Type>& q, int idx,
                                    quat_t<Type>& adj_q, int adj_idx, const Type& adj_value)
{
    // nop
}


template<typename Type>
inline CUDA_CALLABLE void adj_indexref(quat_t<Type>* q, int idx,
                                       quat_t<Type>& adj_q, int adj_idx, const Type& adj_value)
{
    // nop
}


template<typename Type>
inline CUDA_CALLABLE void add_inplace(quat_t<Type>& q, int idx, Type value)
{
#ifndef NDEBUG
    if (idx < 0 || idx > 3)
    {
        printf("quat index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    q[idx] += value;
}


template<typename Type>
inline CUDA_CALLABLE void adj_add_inplace(quat_t<Type>& q, int idx, Type value,
                                        quat_t<Type>& adj_q, int adj_idx, Type& adj_value)
{
#ifndef NDEBUG
    if (idx < 0 || idx > 3)
    {
        printf("quat index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    adj_value += adj_q[idx];
}


template<typename Type>
inline CUDA_CALLABLE void sub_inplace(quat_t<Type>& q, int idx, Type value)
{
#ifndef NDEBUG
    if (idx < 0 || idx > 3)
    {
        printf("quat index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    q[idx] -= value;
}


template<typename Type>
inline CUDA_CALLABLE void adj_sub_inplace(quat_t<Type>& q, int idx, Type value,
                                        quat_t<Type>& adj_q, int adj_idx, Type& adj_value)
{
#ifndef NDEBUG
    if (idx < 0 || idx > 3)
    {
        printf("quat index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    adj_value -= adj_q[idx];
}


template<typename Type>
inline CUDA_CALLABLE void assign_inplace(quat_t<Type>& q, int idx, Type value)
{
#ifndef NDEBUG
    if (idx < 0 || idx > 3)
    {
        printf("quat index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    q[idx] = value;
}

template<typename Type>
inline CUDA_CALLABLE void adj_assign_inplace(quat_t<Type>& q, int idx, Type value, quat_t<Type>& adj_q, int& adj_idx, Type& adj_value)
{
#ifndef NDEBUG
    if (idx < 0 || idx > 3)
    {
        printf("quat index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    adj_value += adj_q[idx];
}


template<typename Type>
inline CUDA_CALLABLE quat_t<Type> assign_copy(quat_t<Type>& q, int idx, Type value)
{
#ifndef NDEBUG
    if (idx < 0 || idx > 3)
    {
        printf("quat index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    quat_t<Type> ret(q);
    ret[idx] = value;
    return ret;
}

template<typename Type>
inline CUDA_CALLABLE void adj_assign_copy(quat_t<Type>& q, int idx, Type value, quat_t<Type>& adj_q, int& adj_idx, Type& adj_value, const quat_t<Type>& adj_ret)
{
#ifndef NDEBUG
    if (idx < 0 || idx > 3)
    {
        printf("quat index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    adj_value += adj_ret[idx];
    for(unsigned i=0; i < 4; ++i)
    {
        if(i != idx)
            adj_q[i] += adj_ret[i];
    }
}


template<typename Type>
CUDA_CALLABLE inline quat_t<Type> lerp(const quat_t<Type>& a, const quat_t<Type>& b, Type t)
{
    return a*(Type(1)-t) + b*t;
}

template<typename Type>
CUDA_CALLABLE inline void adj_lerp(const quat_t<Type>& a, const quat_t<Type>& b, Type t, quat_t<Type>& adj_a, quat_t<Type>& adj_b, Type& adj_t, const quat_t<Type>& adj_ret)
{
    adj_a += adj_ret*(Type(1)-t);
    adj_b += adj_ret*t;
    adj_t += tensordot(b, adj_ret) - tensordot(a, adj_ret);
}

template<typename Type>
inline CUDA_CALLABLE void adj_extract(const quat_t<Type>& a, int idx, quat_t<Type>& adj_a, int & adj_idx, Type & adj_ret)
{
#if FP_CHECK
    if (idx < 0 || idx > 3)
    {
        printf("quat_t index %d out of bounds at %s %d", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    // See wp::extract(const quat_t<Type>& a, int idx) note
    if (idx == 0)       {adj_a.x += adj_ret;}
    else if (idx == 1)  {adj_a.y += adj_ret;}
    else if (idx == 2)  {adj_a.z += adj_ret;}
    else                {adj_a.w += adj_ret;}
}


// backward methods
template<typename Type>
inline CUDA_CALLABLE void adj_quat_from_axis_angle(const vec_t<3,Type>& axis, Type angle, vec_t<3,Type>& adj_axis, Type& adj_angle, const quat_t<Type>& adj_ret)
{
    vec_t<3,Type> v = vec_t<3,Type>(adj_ret.x, adj_ret.y, adj_ret.z);

    Type s = sin(angle*Type(0.5));
    Type c = cos(angle*Type(0.5));

    quat_t<Type> dqda = quat_t<Type>(axis[0]*c, axis[1]*c, axis[2]*c, -s)*Type(0.5);

    adj_axis += v*s;
    adj_angle += dot(dqda, adj_ret);
}

template<typename Type>
inline CUDA_CALLABLE void adj_quat_to_axis_angle(const quat_t<Type>& q, vec_t<3,Type>& axis, Type& angle, quat_t<Type>& adj_q, const vec_t<3,Type>& adj_axis, const Type& adj_angle)
{   
    Type l = length(vec_t<3,Type>(q.x, q.y, q.z));

    Type ax_qx = Type(0);
    Type ax_qy = Type(0);
    Type ax_qz = Type(0);
    Type ay_qx = Type(0);
    Type ay_qy = Type(0);
    Type ay_qz = Type(0);
    Type az_qx = Type(0);
    Type az_qy = Type(0);
    Type az_qz = Type(0);

    Type t_qx = Type(0);
    Type t_qy = Type(0);
    Type t_qz = Type(0);
    Type t_qw = Type(0);

    Type flip = q.w < Type(0) ? -1.0 : 1.0;

    if (l > Type(0))
    {
        Type l_sq = l*l;
        Type l_inv = Type(1) / l;
        Type l_inv_sq = l_inv * l_inv;
        Type l_inv_cu = l_inv_sq * l_inv;
        
        Type C = flip * l_inv_cu;
        ax_qx = C * (q.y*q.y + q.z*q.z);
        ax_qy = -C * q.x*q.y;
        ax_qz = -C * q.x*q.z;
        ay_qx = -C * q.y*q.x;
        ay_qy = C * (q.x*q.x + q.z*q.z);
        ay_qz = -C * q.y*q.z;
        az_qx = -C * q.z*q.x;
        az_qy = -C * q.z*q.y;
        az_qz = C * (q.x*q.x + q.y*q.y);

        Type D = Type(2) * flip / (l_sq + q.w*q.w);
        t_qx = D * l_inv * q.x * q.w;
        t_qy = D * l_inv * q.y * q.w;
        t_qz = D * l_inv * q.z * q.w;
        t_qw = -D * l;
    }
    else
    {
        if (abs(q.w) > Type(kEps))
        {
            Type t_qx = Type(2) / (sqrt(Type(3)) * abs(q.w));
            Type t_qy = Type(2) / (sqrt(Type(3)) * abs(q.w));
            Type t_qz = Type(2) / (sqrt(Type(3)) * abs(q.w));
        }
        // o/w we have a null quat_t which cannot backpropagate 
    }

    adj_q.x += ax_qx * adj_axis[0] + ay_qx * adj_axis[1] + az_qx * adj_axis[2] + t_qx * adj_angle;
    adj_q.y += ax_qy * adj_axis[0] + ay_qy * adj_axis[1] + az_qy * adj_axis[2] + t_qy * adj_angle;
    adj_q.z += ax_qz * adj_axis[0] + ay_qz * adj_axis[1] + az_qz * adj_axis[2] + t_qz * adj_angle;
    adj_q.w += t_qw * adj_angle;
}

template<typename Type>
inline CUDA_CALLABLE void adj_quat_rpy(Type roll, Type pitch, Type yaw, Type& adj_roll, Type& adj_pitch, Type& adj_yaw, const quat_t<Type>& adj_ret)
{
    Type cy = cos(yaw * Type(0.5));
    Type sy = sin(yaw * Type(0.5));
    Type cr = cos(roll * Type(0.5));
    Type sr = sin(roll * Type(0.5));
    Type cp = cos(pitch * Type(0.5));
    Type sp = sin(pitch * Type(0.5));

    Type w = (cy * cr * cp + sy * sr * sp);
    Type x = (cy * sr * cp - sy * cr * sp);
    Type y = (cy * cr * sp + sy * sr * cp);
    Type z = (sy * cr * cp - cy * sr * sp);

    Type dx_dr = Type(0.5) * w;
    Type dx_dp = -Type(0.5) * cy * sr * sp - Type(0.5) * sy * cr * cp;
    Type dx_dy = -Type(0.5) * y;

    Type dy_dr = Type(0.5) * z;
    Type dy_dp = Type(0.5) * cy * cr * cp - Type(0.5) * sy * sr * sp;
    Type dy_dy = Type(0.5) * x;

    Type dz_dr = -Type(0.5) * y;
    Type dz_dp = -Type(0.5) * sy * cr * sp - Type(0.5) * cy * sr * cp;
    Type dz_dy = Type(0.5) * w;

    Type dw_dr = -Type(0.5) * x;
    Type dw_dp = -Type(0.5) * cy * cr * sp + Type(0.5) * sy * sr * cp;
    Type dw_dy = -Type(0.5) * z;

    adj_roll += dot(quat_t<Type>(dx_dr, dy_dr, dz_dr, dw_dr), adj_ret);
    adj_pitch += dot(quat_t<Type>(dx_dp, dy_dp, dz_dp, dw_dp), adj_ret);
    adj_yaw += dot(quat_t<Type>(dx_dy, dy_dy, dz_dy, dw_dy), adj_ret);
}


template<typename Type>
inline CUDA_CALLABLE void adj_dot(const quat_t<Type>& a, const quat_t<Type>& b, quat_t<Type>& adj_a, quat_t<Type>& adj_b, const Type adj_ret)
{
    adj_a += b*adj_ret;
    adj_b += a*adj_ret;    
}

template<typename Type>
inline CUDA_CALLABLE void tensordot(const quat_t<Type>& a, const quat_t<Type>& b, quat_t<Type>& adj_a, quat_t<Type>& adj_b, const Type adj_ret)
{
    adj_dot(a, b, adj_a, adj_b, adj_ret);
}

template<typename Type>
inline CUDA_CALLABLE void adj_length(const quat_t<Type>& a, Type ret, quat_t<Type>& adj_a, const Type adj_ret)
{
    if (ret > Type(kEps))
    {
        Type inv_l = Type(1)/ret;

        adj_a += quat_t<Type>(a.x*inv_l, a.y*inv_l, a.z*inv_l, a.w*inv_l) * adj_ret;
    }
}

template<typename Type>
inline CUDA_CALLABLE void adj_length_sq(const quat_t<Type>& a, quat_t<Type>& adj_a, const Type adj_ret)
{
    adj_a += Type(2)*a*adj_ret;
}

template<typename Type>
inline CUDA_CALLABLE void adj_normalize(const quat_t<Type>& q, quat_t<Type>& adj_q, const quat_t<Type>& adj_ret)
{
    Type l = length(q);

    if (l > Type(kEps))
    {
        Type l_inv = Type(1)/l;

        adj_q += adj_ret*l_inv - q*(l_inv*l_inv*l_inv*dot(q, adj_ret));
    }
}

template<typename Type>
inline CUDA_CALLABLE void adj_quat_inverse(const quat_t<Type>& q, quat_t<Type>& adj_q, const quat_t<Type>& adj_ret)
{
    adj_q.x -= adj_ret.x;
    adj_q.y -= adj_ret.y;
    adj_q.z -= adj_ret.z;
    adj_q.w += adj_ret.w;
}

template<typename Type>
inline CUDA_CALLABLE void adj_add(const quat_t<Type>& a, const quat_t<Type>& b, quat_t<Type>& adj_a, quat_t<Type>& adj_b, const quat_t<Type>& adj_ret)
{
    adj_a += adj_ret;
    adj_b += adj_ret;
}

template<typename Type>
inline CUDA_CALLABLE void adj_sub(const quat_t<Type>& a, const quat_t<Type>& b, quat_t<Type>& adj_a, quat_t<Type>& adj_b, const quat_t<Type>& adj_ret)
{
    adj_a += adj_ret;
    adj_b -= adj_ret;
}

template<typename Type>
inline CUDA_CALLABLE void adj_mul(const quat_t<Type>& a, const quat_t<Type>& b, quat_t<Type>& adj_a, quat_t<Type>& adj_b, const quat_t<Type>& adj_ret)
{
    // shorthand
    const quat_t<Type>& r = adj_ret;

    adj_a += quat_t<Type>(b.w*r.x - b.x*r.w + b.y*r.z - b.z*r.y,
                  b.w*r.y - b.y*r.w - b.x*r.z + b.z*r.x,
                  b.w*r.z + b.x*r.y - b.y*r.x - b.z*r.w,
                  b.w*r.w + b.x*r.x + b.y*r.y + b.z*r.z);

    adj_b += quat_t<Type>(a.w*r.x - a.x*r.w - a.y*r.z + a.z*r.y,
                  a.w*r.y - a.y*r.w + a.x*r.z - a.z*r.x,
                  a.w*r.z - a.x*r.y + a.y*r.x - a.z*r.w,
                  a.w*r.w + a.x*r.x + a.y*r.y + a.z*r.z);

}

template<typename Type>
inline CUDA_CALLABLE void adj_mul(const quat_t<Type>& a, Type s, quat_t<Type>& adj_a, Type& adj_s, const quat_t<Type>& adj_ret)
{
    adj_a += adj_ret*s;
    adj_s += dot(a, adj_ret);
}

template<typename Type>
inline CUDA_CALLABLE void adj_mul(Type s, const quat_t<Type>& a, Type& adj_s, quat_t<Type>& adj_a, const quat_t<Type>& adj_ret)
{
    adj_mul(a, s, adj_a, adj_s, adj_ret);
}

template<typename Type>
inline CUDA_CALLABLE void adj_div(quat_t<Type> a, Type s, quat_t<Type>& adj_a, Type& adj_s, const quat_t<Type>& adj_ret)
{
    adj_s -= dot(a, adj_ret)/ (s * s); // - a / s^2
    adj_a += adj_ret / s;
}

template<typename Type>
inline CUDA_CALLABLE void adj_div(Type s, quat_t<Type> a, Type& adj_s, quat_t<Type>& adj_a, const quat_t<Type>& adj_ret)
{
    adj_s -= dot(a, adj_ret)/ (s * s); // - a / s^2
    adj_a += s / adj_ret;
}

template<typename Type>
inline CUDA_CALLABLE void adj_quat_rotate(const quat_t<Type>& q, const vec_t<3,Type>& p, quat_t<Type>& adj_q, vec_t<3,Type>& adj_p, const vec_t<3,Type>& adj_ret)
{

    {
        Type t2 = p[2]*q.z*Type(2);
        Type t3 = p[1]*q.w*Type(2);
        Type t4 = p[0]*q.w*Type(2);
        Type t5 = p[0]*q.x*Type(2);
        Type t6 = p[1]*q.y*Type(2);
        Type t7 = p[2]*q.y*Type(2);
        Type t8 = p[0]*q.z*Type(2);
        Type t9 = p[0]*q.y*Type(2);
        Type t10 = p[1]*q.x*Type(2);
        adj_q.x += adj_ret[2]*(t3+t8)+adj_ret[0]*(t2+t6+p[0]*q.x*Type(4))+adj_ret[1]*(t9-p[2]*q.w*Type(2));
        adj_q.y += adj_ret[1]*(t2+t5+p[1]*q.y*Type(4))+adj_ret[0]*(t10+p[2]*q.w*Type(2))-adj_ret[2]*(t4-p[1]*q.z*Type(2));
        adj_q.z += adj_ret[1]*(t4+t7)+adj_ret[2]*(t5+t6+p[2]*q.z*Type(4))-adj_ret[0]*(t3-p[2]*q.x*Type(2));
        adj_q.w += adj_ret[0]*(t7+p[0]*q.w*Type(4)-p[1]*q.z*Type(2))+adj_ret[1]*(t8+p[1]*q.w*Type(4)-p[2]*q.x*Type(2))+adj_ret[2]*(-t9+t10+p[2]*q.w*Type(4));
    }

    {
        Type t2 = q.w*q.w;
        Type t3 = t2*Type(2);
        Type t4 = q.w*q.z*Type(2);
        Type t5 = q.x*q.y*Type(2);
        Type t6 = q.w*q.y*Type(2);
        Type t7 = q.w*q.x*Type(2);
        Type t8 = q.y*q.z*Type(2);
        adj_p[0] += adj_ret[1]*(t4+t5)+adj_ret[0]*(t3+(q.x*q.x)*Type(2)-Type(1))-adj_ret[2]*(t6-q.x*q.z*Type(2));
        adj_p[1] += adj_ret[2]*(t7+t8)-adj_ret[0]*(t4-t5)+adj_ret[1]*(t3+(q.y*q.y)*Type(2)-Type(1));
        adj_p[2] += -adj_ret[1]*(t7-t8)+adj_ret[2]*(t3+(q.z*q.z)*Type(2)-Type(1))+adj_ret[0]*(t6+q.x*q.z*Type(2));
    }
}

template<typename Type>
inline CUDA_CALLABLE void adj_quat_rotate_inv(const quat_t<Type>& q, const vec_t<3,Type>& p, quat_t<Type>& adj_q, vec_t<3,Type>& adj_p, const vec_t<3,Type>& adj_ret)
{
    const vec_t<3,Type>& r = adj_ret;

    {
        Type t2 = p[2]*q.w*Type(2);
        Type t3 = p[2]*q.z*Type(2);
        Type t4 = p[1]*q.w*Type(2);
        Type t5 = p[0]*q.w*Type(2);
        Type t6 = p[0]*q.x*Type(2);
        Type t7 = p[1]*q.y*Type(2);
        Type t8 = p[1]*q.z*Type(2);
        Type t9 = p[2]*q.x*Type(2);
        Type t10 = p[0]*q.y*Type(2);
        adj_q.x += r[1]*(t2+t10)+r[0]*(t3+t7+p[0]*q.x*Type(4))-r[2]*(t4-p[0]*q.z*Type(2));
        adj_q.y += r[2]*(t5+t8)+r[1]*(t3+t6+p[1]*q.y*Type(4))-r[0]*(t2-p[1]*q.x*Type(2));
        adj_q.z += r[0]*(t4+t9)+r[2]*(t6+t7+p[2]*q.z*Type(4))-r[1]*(t5-p[2]*q.y*Type(2));
        adj_q.w += r[0]*(t8+p[0]*q.w*Type(4)-p[2]*q.y*Type(2))+r[1]*(t9+p[1]*q.w*Type(4)-p[0]*q.z*Type(2))+r[2]*(t10-p[1]*q.x*Type(2)+p[2]*q.w*Type(4));
    }

    {
        Type t2 = q.w*q.w;
        Type t3 = t2*Type(2);
        Type t4 = q.w*q.z*Type(2);
        Type t5 = q.w*q.y*Type(2);
        Type t6 = q.x*q.z*Type(2);
        Type t7 = q.w*q.x*Type(2);
        adj_p[0] += r[2]*(t5+t6)+r[0]*(t3+(q.x*q.x)*Type(2)-Type(1))-r[1]*(t4-q.x*q.y*Type(2));
        adj_p[1] += r[1]*(t3+(q.y*q.y)*Type(2)-Type(1))+r[0]*(t4+q.x*q.y*Type(2))-r[2]*(t7-q.y*q.z*Type(2));
        adj_p[2] += -r[0]*(t5-t6)+r[2]*(t3+(q.z*q.z)*Type(2)-Type(1))+r[1]*(t7+q.y*q.z*Type(2));
    }
}

template<typename Type>
inline CUDA_CALLABLE void adj_quat_slerp(const quat_t<Type>& q0, const quat_t<Type>& q1, Type t, quat_t<Type>& ret, quat_t<Type>& adj_q0, quat_t<Type>& adj_q1, Type& adj_t, const quat_t<Type>& adj_ret)
{
    vec_t<3,Type> axis;
    Type angle;
    quat_t<Type> q0_inv = quat_inverse(q0);
    quat_t<Type> q_inc = mul(q0_inv, q1);
    quat_to_axis_angle(q_inc, axis, angle);
    quat_t<Type> qt = quat_from_axis_angle(axis, angle * t);
    angle = angle * 0.5;
    
    // adj_t
    adj_t += dot(mul(ret, quat_t<Type>(angle*axis[0], angle*axis[1], angle*axis[2], Type(0))), adj_ret);

    // adj_q0
    quat_t<Type> q_inc_x_q0;
    quat_t<Type> q_inc_y_q0;
    quat_t<Type> q_inc_z_q0;
    quat_t<Type> q_inc_w_q0;
    
    quat_t<Type> q_inc_x_q1;
    quat_t<Type> q_inc_y_q1;
    quat_t<Type> q_inc_z_q1;
    quat_t<Type> q_inc_w_q1;

    adj_mul(q0_inv, q1, q_inc_x_q0, q_inc_x_q1, quat_t<Type>(1.f, Type(0), Type(0), Type(0)));
    adj_mul(q0_inv, q1, q_inc_y_q0, q_inc_y_q1, quat_t<Type>(Type(0), 1.f, Type(0), Type(0)));
    adj_mul(q0_inv, q1, q_inc_z_q0, q_inc_z_q1, quat_t<Type>(Type(0), Type(0), 1.f, Type(0)));
    adj_mul(q0_inv, q1, q_inc_w_q0, q_inc_w_q1, quat_t<Type>(Type(0), Type(0), Type(0), 1.f));

    quat_t<Type> a_x_q_inc;
    quat_t<Type> a_y_q_inc;
    quat_t<Type> a_z_q_inc;
    quat_t<Type> t_q_inc;

    adj_quat_to_axis_angle(q_inc, axis, angle, a_x_q_inc, vec_t<3,Type>(1.f, Type(0), Type(0)), Type(0));
    adj_quat_to_axis_angle(q_inc, axis, angle, a_y_q_inc, vec_t<3,Type>(Type(0), 1.f, Type(0)), Type(0));
    adj_quat_to_axis_angle(q_inc, axis, angle, a_z_q_inc, vec_t<3,Type>(Type(0), Type(0), 1.f), Type(0));
    adj_quat_to_axis_angle(q_inc, axis, angle, t_q_inc, vec_t<3,Type>(Type(0), Type(0), Type(0)), Type(1));
    
    Type cs = cos(angle*t);
    Type sn = sin(angle*t);

    quat_t<Type> q_inc_q0_x = quat_t<Type>(-q_inc_x_q0.x, -q_inc_y_q0.x, -q_inc_z_q0.x, -q_inc_w_q0.x);
    quat_t<Type> q_inc_q0_y = quat_t<Type>(-q_inc_x_q0.y, -q_inc_y_q0.y, -q_inc_z_q0.y, -q_inc_w_q0.y);
    quat_t<Type> q_inc_q0_z = quat_t<Type>(-q_inc_x_q0.z, -q_inc_y_q0.z, -q_inc_z_q0.z, -q_inc_w_q0.z);
    quat_t<Type> q_inc_q0_w = quat_t<Type>(q_inc_x_q0.w, q_inc_y_q0.w, q_inc_z_q0.w, q_inc_w_q0.w);

    Type a_x_q0_x = dot(a_x_q_inc, q_inc_q0_x);
    Type a_x_q0_y = dot(a_x_q_inc, q_inc_q0_y);
    Type a_x_q0_z = dot(a_x_q_inc, q_inc_q0_z);
    Type a_x_q0_w = dot(a_x_q_inc, q_inc_q0_w);
    Type a_y_q0_x = dot(a_y_q_inc, q_inc_q0_x);
    Type a_y_q0_y = dot(a_y_q_inc, q_inc_q0_y);
    Type a_y_q0_z = dot(a_y_q_inc, q_inc_q0_z);
    Type a_y_q0_w = dot(a_y_q_inc, q_inc_q0_w);
    Type a_z_q0_x = dot(a_z_q_inc, q_inc_q0_x);
    Type a_z_q0_y = dot(a_z_q_inc, q_inc_q0_y);
    Type a_z_q0_z = dot(a_z_q_inc, q_inc_q0_z);
    Type a_z_q0_w = dot(a_z_q_inc, q_inc_q0_w);
    Type t_q0_x = dot(t_q_inc, q_inc_q0_x);
    Type t_q0_y = dot(t_q_inc, q_inc_q0_y);
    Type t_q0_z = dot(t_q_inc, q_inc_q0_z);
    Type t_q0_w = dot(t_q_inc, q_inc_q0_w);

    quat_t<Type> q_s_q0_x = mul(quat_t<Type>(1.f, Type(0), Type(0), Type(0)), qt) + mul(q0, quat_t<Type>(
        0.5 * t * axis[0] * t_q0_x * cs + a_x_q0_x * sn,
        0.5 * t * axis[1] * t_q0_x * cs + a_y_q0_x * sn,
        0.5 * t * axis[2] * t_q0_x * cs + a_z_q0_x * sn,
        -0.5 * t * t_q0_x * sn));

    quat_t<Type> q_s_q0_y = mul(quat_t<Type>(Type(0), 1.f, Type(0), Type(0)), qt) + mul(q0, quat_t<Type>(
        0.5 * t * axis[0] * t_q0_y * cs + a_x_q0_y * sn,
        0.5 * t * axis[1] * t_q0_y * cs + a_y_q0_y * sn,
        0.5 * t * axis[2] * t_q0_y * cs + a_z_q0_y * sn,
        -0.5 * t * t_q0_y * sn));

    quat_t<Type> q_s_q0_z = mul(quat_t<Type>(Type(0), Type(0), 1.f, Type(0)), qt) + mul(q0, quat_t<Type>(
        0.5 * t * axis[0] * t_q0_z * cs + a_x_q0_z * sn,
        0.5 * t * axis[1] * t_q0_z * cs + a_y_q0_z * sn,
        0.5 * t * axis[2] * t_q0_z * cs + a_z_q0_z * sn,
        -0.5 * t * t_q0_z * sn));

    quat_t<Type> q_s_q0_w = mul(quat_t<Type>(Type(0), Type(0), Type(0), 1.f), qt) + mul(q0, quat_t<Type>(
        0.5 * t * axis[0] * t_q0_w * cs + a_x_q0_w * sn,
        0.5 * t * axis[1] * t_q0_w * cs + a_y_q0_w * sn,
        0.5 * t * axis[2] * t_q0_w * cs + a_z_q0_w * sn,
        -0.5 * t * t_q0_w * sn));

    adj_q0.x += dot(q_s_q0_x, adj_ret);
    adj_q0.y += dot(q_s_q0_y, adj_ret);
    adj_q0.z += dot(q_s_q0_z, adj_ret);
    adj_q0.w += dot(q_s_q0_w, adj_ret);

    // adj_q1
    quat_t<Type> q_inc_q1_x = quat_t<Type>(q_inc_x_q1.x, q_inc_y_q1.x, q_inc_z_q1.x, q_inc_w_q1.x);
    quat_t<Type> q_inc_q1_y = quat_t<Type>(q_inc_x_q1.y, q_inc_y_q1.y, q_inc_z_q1.y, q_inc_w_q1.y);
    quat_t<Type> q_inc_q1_z = quat_t<Type>(q_inc_x_q1.z, q_inc_y_q1.z, q_inc_z_q1.z, q_inc_w_q1.z);
    quat_t<Type> q_inc_q1_w = quat_t<Type>(q_inc_x_q1.w, q_inc_y_q1.w, q_inc_z_q1.w, q_inc_w_q1.w);

    Type a_x_q1_x = dot(a_x_q_inc, q_inc_q1_x);
    Type a_x_q1_y = dot(a_x_q_inc, q_inc_q1_y);
    Type a_x_q1_z = dot(a_x_q_inc, q_inc_q1_z);
    Type a_x_q1_w = dot(a_x_q_inc, q_inc_q1_w);
    Type a_y_q1_x = dot(a_y_q_inc, q_inc_q1_x);
    Type a_y_q1_y = dot(a_y_q_inc, q_inc_q1_y);
    Type a_y_q1_z = dot(a_y_q_inc, q_inc_q1_z);
    Type a_y_q1_w = dot(a_y_q_inc, q_inc_q1_w);
    Type a_z_q1_x = dot(a_z_q_inc, q_inc_q1_x);
    Type a_z_q1_y = dot(a_z_q_inc, q_inc_q1_y);
    Type a_z_q1_z = dot(a_z_q_inc, q_inc_q1_z);
    Type a_z_q1_w = dot(a_z_q_inc, q_inc_q1_w);
    Type t_q1_x = dot(t_q_inc, q_inc_q1_x);
    Type t_q1_y = dot(t_q_inc, q_inc_q1_y);
    Type t_q1_z = dot(t_q_inc, q_inc_q1_z);
    Type t_q1_w = dot(t_q_inc, q_inc_q1_w);

    quat_t<Type> q_s_q1_x = mul(q0, quat_t<Type>(
        0.5 * t * axis[0] * t_q1_x * cs + a_x_q1_x * sn,
        0.5 * t * axis[1] * t_q1_x * cs + a_y_q1_x * sn,
        0.5 * t * axis[2] * t_q1_x * cs + a_z_q1_x * sn,
        -0.5 * t * t_q1_x * sn));

    quat_t<Type> q_s_q1_y = mul(q0, quat_t<Type>(
        0.5 * t * axis[0] * t_q1_y * cs + a_x_q1_y * sn,
        0.5 * t * axis[1] * t_q1_y * cs + a_y_q1_y * sn,
        0.5 * t * axis[2] * t_q1_y * cs + a_z_q1_y * sn,
        -0.5 * t * t_q1_y * sn));

    quat_t<Type> q_s_q1_z = mul(q0, quat_t<Type>(
        0.5 * t * axis[0] * t_q1_z * cs + a_x_q1_z * sn,
        0.5 * t * axis[1] * t_q1_z * cs + a_y_q1_z * sn,
        0.5 * t * axis[2] * t_q1_z * cs + a_z_q1_z * sn,
        -0.5 * t * t_q1_z * sn));

    quat_t<Type> q_s_q1_w = mul(q0, quat_t<Type>(
        0.5 * t * axis[0] * t_q1_w * cs + a_x_q1_w * sn,
        0.5 * t * axis[1] * t_q1_w * cs + a_y_q1_w * sn,
        0.5 * t * axis[2] * t_q1_w * cs + a_z_q1_w * sn,
        -0.5 * t * t_q1_w * sn));

    adj_q1.x += dot(q_s_q1_x, adj_ret);
    adj_q1.y += dot(q_s_q1_y, adj_ret);
    adj_q1.z += dot(q_s_q1_z, adj_ret);
    adj_q1.w += dot(q_s_q1_w, adj_ret);    

}

template<typename Type>
inline CUDA_CALLABLE void adj_quat_to_matrix(const quat_t<Type>& q, quat_t<Type>& adj_q, mat_t<3,3,Type>& adj_ret)
{
    // we don't care about adjoint w.r.t. constant identity matrix
    vec_t<3,Type> t;

    adj_quat_rotate(q, vec_t<3,Type>(1.0, 0.0, 0.0), adj_q, t, adj_ret.get_col(0));
    adj_quat_rotate(q, vec_t<3,Type>(0.0, 1.0, 0.0), adj_q, t, adj_ret.get_col(1));
    adj_quat_rotate(q, vec_t<3,Type>(0.0, 0.0, 1.0), adj_q, t, adj_ret.get_col(2));
}

template<unsigned Rows, unsigned Cols, typename Type>
inline CUDA_CALLABLE void adj_quat_from_matrix(const mat_t<Rows,Cols,Type>& m, mat_t<Rows,Cols,Type>& adj_m, const quat_t<Type>& adj_ret)
{
    static_assert((Rows == 3 && Cols == 3) || (Rows == 4 && Cols == 4), "Non-square matrix");

    const Type tr = m.data[0][0] + m.data[1][1] + m.data[2][2];
    Type x, y, z, w, h = Type(0);

    Type dx_dm00 = Type(0), dx_dm01 = Type(0), dx_dm02 = Type(0);
    Type dx_dm10 = Type(0), dx_dm11 = Type(0), dx_dm12 = Type(0);
    Type dx_dm20 = Type(0), dx_dm21 = Type(0), dx_dm22 = Type(0);
    Type dy_dm00 = Type(0), dy_dm01 = Type(0), dy_dm02 = Type(0);
    Type dy_dm10 = Type(0), dy_dm11 = Type(0), dy_dm12 = Type(0);
    Type dy_dm20 = Type(0), dy_dm21 = Type(0), dy_dm22 = Type(0);
    Type dz_dm00 = Type(0), dz_dm01 = Type(0), dz_dm02 = Type(0);
    Type dz_dm10 = Type(0), dz_dm11 = Type(0), dz_dm12 = Type(0);
    Type dz_dm20 = Type(0), dz_dm21 = Type(0), dz_dm22 = Type(0);
    Type dw_dm00 = Type(0), dw_dm01 = Type(0), dw_dm02 = Type(0);
    Type dw_dm10 = Type(0), dw_dm11 = Type(0), dw_dm12 = Type(0);
    Type dw_dm20 = Type(0), dw_dm21 = Type(0), dw_dm22 = Type(0);

    if (tr >= Type(0)) {
        h = sqrt(tr + Type(1));
        w = Type(0.5) * h;
        h = Type(0.5) / h;

        x = (m.data[2][1] - m.data[1][2]) * h;
        y = (m.data[0][2] - m.data[2][0]) * h;
        z = (m.data[1][0] - m.data[0][1]) * h;

        dw_dm00 = Type(0.5) * h;
        dw_dm11 = Type(0.5) * h;
        dw_dm22 = Type(0.5) * h;
        dx_dm21 = h;
        dx_dm12 = -h;
        dx_dm00 = Type(2) * h*h*h * (m.data[1][2] - m.data[2][1]);
        dx_dm11 = Type(2) * h*h*h * (m.data[1][2] - m.data[2][1]);
        dx_dm22 = Type(2) * h*h*h * (m.data[1][2] - m.data[2][1]);
        dy_dm02 = h;
        dy_dm20 = -h;
        dy_dm00 = Type(2) * h*h*h * (m.data[2][0] - m.data[0][2]);
        dy_dm11 = Type(2) * h*h*h * (m.data[2][0] - m.data[0][2]);
        dy_dm22 = Type(2) * h*h*h * (m.data[2][0] - m.data[0][2]);
        dz_dm10 = h;
        dz_dm01 = -h;
        dz_dm00 = Type(2) * h*h*h * (m.data[0][1] - m.data[1][0]);
        dz_dm11 = Type(2) * h*h*h * (m.data[0][1] - m.data[1][0]);
        dz_dm22 = Type(2) * h*h*h * (m.data[0][1] - m.data[1][0]);
    } else {
        size_t max_diag = 0;
        if (m.data[1][1] > m.data[0][0]) {
            max_diag = 1;
        }
        if (m.data[2][2] > m.data[max_diag][max_diag]) {
            max_diag = 2;
        }
        
        if (max_diag == 0) {
            h = sqrt((m.data[0][0] - (m.data[1][1] + m.data[2][2])) + Type(1));
            x = Type(0.5) * h;
            h = Type(0.5) / h;

            y = (m.data[0][1] + m.data[1][0]) * h;
            z = (m.data[2][0] + m.data[0][2]) * h;
            w = (m.data[2][1] - m.data[1][2]) * h;

            dx_dm00 = Type(0.5) * h;
            dx_dm11 = -Type(0.5) * h;
            dx_dm22 = -Type(0.5) * h;
            dy_dm01 = h;
            dy_dm10 = h;
            dy_dm00 = -Type(2) * h*h*h * (m.data[0][1] + m.data[1][0]);
            dy_dm11 = Type(2) * h*h*h * (m.data[0][1] + m.data[1][0]);
            dy_dm22 = Type(2) * h*h*h * (m.data[0][1] + m.data[1][0]);
            dz_dm20 = h;
            dz_dm02 = h;
            dz_dm00 = -Type(2) * h*h*h * (m.data[2][0] + m.data[0][2]);
            dz_dm11 = Type(2) * h*h*h * (m.data[2][0] + m.data[0][2]);
            dz_dm22 = Type(2) * h*h*h * (m.data[2][0] + m.data[0][2]);
            dw_dm21 = h;
            dw_dm12 = -h;
            dw_dm00 = Type(2) * h*h*h * (m.data[1][2] - m.data[2][1]);
            dw_dm11 = Type(2) * h*h*h * (m.data[2][1] - m.data[1][2]);
            dw_dm22 = Type(2) * h*h*h * (m.data[2][1] - m.data[1][2]);
        } else if (max_diag == 1) {
            h = sqrt((m.data[1][1] - (m.data[2][2] + m.data[0][0])) + Type(1));
            y = Type(0.5) * h;
            h = Type(0.5) / h;

            z = (m.data[1][2] + m.data[2][1]) * h;
            x = (m.data[0][1] + m.data[1][0]) * h;
            w = (m.data[0][2] - m.data[2][0]) * h;

            dy_dm00 = -Type(0.5) * h;
            dy_dm11 = Type(0.5) * h;
            dy_dm22 = -Type(0.5) * h;
            dz_dm12 = h;
            dz_dm21 = h;
            dz_dm00 = Type(2) * h*h*h * (m.data[1][2] + m.data[2][1]);
            dz_dm11 = -Type(2) * h*h*h * (m.data[1][2] + m.data[2][1]);
            dz_dm22 = Type(2) * h*h*h * (m.data[1][2] + m.data[2][1]);
            dx_dm01 = h;
            dx_dm10 = h;
            dx_dm00 = Type(2) * h*h*h * (m.data[0][1] + m.data[1][0]);
            dx_dm11 = -Type(2) * h*h*h * (m.data[0][1] + m.data[1][0]);
            dx_dm22 = Type(2) * h*h*h * (m.data[0][1] + m.data[1][0]);
            dw_dm02 = h;
            dw_dm20 = -h;
            dw_dm00 = Type(2) * h*h*h * (m.data[0][2] - m.data[2][0]);
            dw_dm11 = Type(2) * h*h*h * (m.data[2][0] - m.data[0][2]);
            dw_dm22 = Type(2) * h*h*h * (m.data[0][2] - m.data[2][0]);
        } if (max_diag == 2) {
            h = sqrt((m.data[2][2] - (m.data[0][0] + m.data[1][1])) + Type(1));
            z = Type(0.5) * h;
            h = Type(0.5) / h;

            x = (m.data[2][0] + m.data[0][2]) * h;
            y = (m.data[1][2] + m.data[2][1]) * h;
            w = (m.data[1][0] - m.data[0][1]) * h;

            dz_dm00 = -Type(0.5) * h;
            dz_dm11 = -Type(0.5) * h;
            dz_dm22 = Type(0.5) * h;
            dx_dm20 = h;
            dx_dm02 = h;
            dx_dm00 = Type(2) * h*h*h * (m.data[2][0] + m.data[0][2]);
            dx_dm11 = Type(2) * h*h*h * (m.data[2][0] + m.data[0][2]);
            dx_dm22 = -Type(2) * h*h*h * (m.data[2][0] + m.data[0][2]);
            dy_dm12 = h;
            dy_dm21 = h;
            dy_dm00 = Type(2) * h*h*h * (m.data[1][2] + m.data[2][1]);
            dy_dm11 = Type(2) * h*h*h * (m.data[1][2] + m.data[2][1]);
            dy_dm22 = -Type(2) * h*h*h * (m.data[1][2] + m.data[2][1]);
            dw_dm10 = h;
            dw_dm01 = -h;
            dw_dm00 = Type(2) * h*h*h * (m.data[1][0] - m.data[0][1]);
            dw_dm11 = Type(2) * h*h*h * (m.data[1][0] - m.data[0][1]);
            dw_dm22 = Type(2) * h*h*h * (m.data[0][1] - m.data[1][0]);
        }
    }

    quat_t<Type> dq_dm00 = quat_t<Type>(dx_dm00, dy_dm00, dz_dm00, dw_dm00);
    quat_t<Type> dq_dm01 = quat_t<Type>(dx_dm01, dy_dm01, dz_dm01, dw_dm01);
    quat_t<Type> dq_dm02 = quat_t<Type>(dx_dm02, dy_dm02, dz_dm02, dw_dm02);
    quat_t<Type> dq_dm10 = quat_t<Type>(dx_dm10, dy_dm10, dz_dm10, dw_dm10);
    quat_t<Type> dq_dm11 = quat_t<Type>(dx_dm11, dy_dm11, dz_dm11, dw_dm11);
    quat_t<Type> dq_dm12 = quat_t<Type>(dx_dm12, dy_dm12, dz_dm12, dw_dm12);
    quat_t<Type> dq_dm20 = quat_t<Type>(dx_dm20, dy_dm20, dz_dm20, dw_dm20);
    quat_t<Type> dq_dm21 = quat_t<Type>(dx_dm21, dy_dm21, dz_dm21, dw_dm21);
    quat_t<Type> dq_dm22 = quat_t<Type>(dx_dm22, dy_dm22, dz_dm22, dw_dm22);

    quat_t<Type> adj_q;
    adj_normalize(quat_t<Type>(x, y, z, w), adj_q, adj_ret);

    adj_m.data[0][0] += dot(dq_dm00, adj_q);
    adj_m.data[0][1] += dot(dq_dm01, adj_q);
    adj_m.data[0][2] += dot(dq_dm02, adj_q);
    adj_m.data[1][0] += dot(dq_dm10, adj_q);
    adj_m.data[1][1] += dot(dq_dm11, adj_q);
    adj_m.data[1][2] += dot(dq_dm12, adj_q);
    adj_m.data[2][0] += dot(dq_dm20, adj_q);
    adj_m.data[2][1] += dot(dq_dm21, adj_q);
    adj_m.data[2][2] += dot(dq_dm22, adj_q);
}

template<typename Type>
inline CUDA_CALLABLE void adj_mat_t(const vec_t<3,Type>& pos, const quat_t<Type>& rot, const vec_t<3,Type>& scale,
                                          vec_t<3,Type>& adj_pos, quat_t<Type>& adj_rot, vec_t<3,Type>& adj_scale, const mat_t<4,4,Type>& adj_ret)
{
    mat_t<3,3,Type> R = quat_to_matrix(rot);
    mat_t<3,3,Type> adj_R(0);

    adj_pos[0] += adj_ret.data[0][3];
    adj_pos[1] += adj_ret.data[1][3];
    adj_pos[2] += adj_ret.data[2][3];

    adj_mul(R.data[0][0], scale[0], adj_R.data[0][0], adj_scale[0], adj_ret.data[0][0]);
    adj_mul(R.data[1][0], scale[0], adj_R.data[1][0], adj_scale[0], adj_ret.data[1][0]);
    adj_mul(R.data[2][0], scale[0], adj_R.data[2][0], adj_scale[0], adj_ret.data[2][0]);

    adj_mul(R.data[0][1], scale[1], adj_R.data[0][1], adj_scale[1], adj_ret.data[0][1]);
    adj_mul(R.data[1][1], scale[1], adj_R.data[1][1], adj_scale[1], adj_ret.data[1][1]);
    adj_mul(R.data[2][1], scale[1], adj_R.data[2][1], adj_scale[1], adj_ret.data[2][1]);

    adj_mul(R.data[0][2], scale[2], adj_R.data[0][2], adj_scale[2], adj_ret.data[0][2]);
    adj_mul(R.data[1][2], scale[2], adj_R.data[1][2], adj_scale[2], adj_ret.data[1][2]);
    adj_mul(R.data[2][2], scale[2], adj_R.data[2][2], adj_scale[2], adj_ret.data[2][2]);

    adj_quat_to_matrix(rot, adj_rot, adj_R);
}

template<unsigned Rows, unsigned Cols, typename Type>                
inline CUDA_CALLABLE mat_t<Rows,Cols,Type>::mat_t(const vec_t<3,Type>& pos, const quat_t<Type>& rot, const vec_t<3,Type>& scale)
{
    mat_t<3,3,Type> R = quat_to_matrix(rot);

    data[0][0] = R.data[0][0]*scale[0];
    data[1][0] = R.data[1][0]*scale[0];
    data[2][0] = R.data[2][0]*scale[0];
    data[3][0] = Type(0);

    data[0][1] = R.data[0][1]*scale[1];
    data[1][1] = R.data[1][1]*scale[1];
    data[2][1] = R.data[2][1]*scale[1];
    data[3][1] = Type(0);

    data[0][2] = R.data[0][2]*scale[2];
    data[1][2] = R.data[1][2]*scale[2];
    data[2][2] = R.data[2][2]*scale[2];
    data[3][2] = Type(0);

    data[0][3] = pos[0];
    data[1][3] = pos[1];
    data[2][3] = pos[2];
    data[3][3] = Type(1);
}

template<typename Type=float32>
inline CUDA_CALLABLE quat_t<Type> quat_identity()
{
    return quat_t<Type>(Type(0), Type(0), Type(0), Type(1));
}

template<typename Type>
CUDA_CALLABLE inline int len(const quat_t<Type>& x)
{
    return 4;
}

template<typename Type>
CUDA_CALLABLE inline void adj_len(const quat_t<Type>& x, quat_t<Type>& adj_x, const int& adj_ret)
{
}

template<typename Type>
inline CUDA_CALLABLE void expect_near(const quat_t<Type>& actual, const quat_t<Type>& expected, const Type& tolerance)
{
    Type diff(0);
    for(size_t i = 0; i < 4; ++i)
    {
        diff = max(diff, abs(actual[i] - expected[i]));
    }
    if (diff > tolerance)
    {
        printf("Error, expect_near() failed with tolerance "); print(tolerance);
        printf("    Expected: "); print(expected);
        printf("    Actual: "); print(actual);
        printf("    Max absolute difference: "); print(diff);
    }
}

template<typename Type>
inline CUDA_CALLABLE void adj_expect_near(const quat_t<Type>& actual, const quat_t<Type>& expected, Type tolerance, quat_t<Type>& adj_actual, quat_t<Type>& adj_expected, Type adj_tolerance)
{
    // nop
}

} // namespace wp
