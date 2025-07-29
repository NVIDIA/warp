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

namespace wp
{

//---------------------------------------------------------------------------------
// Represents a twist in se(3)
template <typename Type>
using spatial_vector_t = vec_t<6,Type>;

template<typename Type>
CUDA_CALLABLE inline Type spatial_dot(const spatial_vector_t<Type>& a, const spatial_vector_t<Type>& b)
{
    return dot(a, b);
}

template<typename Type>
CUDA_CALLABLE inline vec_t<3,Type> &w_vec( spatial_vector_t<Type>& a )
{
    return *(vec_t<3,Type>*)(&a);
}

template<typename Type>
CUDA_CALLABLE inline vec_t<3,Type> &v_vec( spatial_vector_t<Type>& a )
{
    return *(vec_t<3,Type>*)(&a.c[3]);
}

template<typename Type>
CUDA_CALLABLE inline const vec_t<3,Type> &w_vec( const spatial_vector_t<Type>& a )
{
    spatial_vector_t<Type> &non_const_vec = *(spatial_vector_t<Type>*)(const_cast<Type*>(&a.c[0]));
    return w_vec(non_const_vec);
}

template<typename Type>
CUDA_CALLABLE inline const vec_t<3,Type> &v_vec( const spatial_vector_t<Type>& a )
{
    spatial_vector_t<Type> &non_const_vec = *(spatial_vector_t<Type>*)(const_cast<Type*>(&a.c[0]));
    return v_vec(non_const_vec);
}

template<typename Type>
CUDA_CALLABLE inline spatial_vector_t<Type> spatial_cross(const spatial_vector_t<Type>& a,  const spatial_vector_t<Type>& b)
{
    vec_t<3,Type> w = cross(w_vec(a), w_vec(b));
    vec_t<3,Type> v = cross(v_vec(a), w_vec(b)) + cross(w_vec(a), v_vec(b));
    
    return spatial_vector_t<Type>({w[0], w[1], w[2], v[0], v[1], v[2]});
}

template<typename Type>
CUDA_CALLABLE inline spatial_vector_t<Type> spatial_cross_dual(const spatial_vector_t<Type>& a,  const spatial_vector_t<Type>& b)
{
    vec_t<3,Type> w = cross(w_vec(a), w_vec(b)) + cross(v_vec(a), v_vec(b));
    vec_t<3,Type> v = cross(w_vec(a), v_vec(b));

    return spatial_vector_t<Type>({w[0], w[1], w[2], v[0], v[1], v[2]});
}

template<typename Type>
CUDA_CALLABLE inline vec_t<3,Type> spatial_top(const spatial_vector_t<Type>& a)
{
    return w_vec(a);
}

template<typename Type>
CUDA_CALLABLE inline vec_t<3,Type> spatial_bottom(const spatial_vector_t<Type>& a)
{
    return v_vec(a);
}

template<typename Type>
CUDA_CALLABLE inline void adj_spatial_dot(const spatial_vector_t<Type>& a, const spatial_vector_t<Type>& b, spatial_vector_t<Type>& adj_a, spatial_vector_t<Type>& adj_b, const Type& adj_ret)
{
    adj_dot(a, b, adj_a, adj_b, adj_ret);
}

template<typename Type>
CUDA_CALLABLE inline void adj_spatial_cross(const spatial_vector_t<Type>& a,  const spatial_vector_t<Type>& b, spatial_vector_t<Type>& adj_a,  spatial_vector_t<Type>& adj_b, const spatial_vector_t<Type>& adj_ret)
{
    adj_cross(w_vec(a), w_vec(b), w_vec(adj_a), w_vec(adj_b), w_vec(adj_ret));
    
    adj_cross(v_vec(a), w_vec(b), v_vec(adj_a), w_vec(adj_b), v_vec(adj_ret));
    adj_cross(w_vec(a), v_vec(b), w_vec(adj_a), v_vec(adj_b), v_vec(adj_ret));
}

template<typename Type>
CUDA_CALLABLE inline void adj_spatial_cross_dual(const spatial_vector_t<Type>& a,  const spatial_vector_t<Type>& b, spatial_vector_t<Type>& adj_a, spatial_vector_t<Type>& adj_b, const spatial_vector_t<Type>& adj_ret)
{
    adj_cross(w_vec(a), w_vec(b), w_vec(adj_a), w_vec(adj_b), w_vec(adj_ret));
    adj_cross(v_vec(a), v_vec(b), v_vec(adj_a), v_vec(adj_b), w_vec(adj_ret));

    adj_cross(w_vec(a), v_vec(b), w_vec(adj_a), v_vec(adj_b), v_vec(adj_ret));
}

template<typename Type>
CUDA_CALLABLE inline void adj_spatial_top(const spatial_vector_t<Type>& a, spatial_vector_t<Type>& adj_a, const vec_t<3,Type>& adj_ret)
{
    w_vec(adj_a) += adj_ret;
}

template<typename Type>
CUDA_CALLABLE inline void adj_spatial_bottom(const spatial_vector_t<Type>& a, spatial_vector_t<Type>& adj_a, const vec_t<3,Type>& adj_ret)
{
    v_vec(adj_a) += adj_ret;
}


//---------------------------------------------------------------------------------
// Represents a rigid body transform<Type>ation

template<typename Type>
struct transform_t
{
    vec_t<3,Type> p;
    quat_t<Type> q;

    CUDA_CALLABLE inline transform_t(vec_t<3,Type> p=vec_t<3,Type>(), quat_t<Type> q=quat_t<Type>()) : p(p), q(q) {}
    CUDA_CALLABLE inline transform_t(Type)  {}  // helps uniform initialization

    template<typename OtherType>
    inline explicit CUDA_CALLABLE transform_t(const transform_t<OtherType>& other)
    {
        p = other.p;
        q = other.q;
    }

    CUDA_CALLABLE inline transform_t(const initializer_array<7, Type> &l)
    {
        p = vec_t<3,Type>(l[0], l[1], l[2]);
        q = quat_t<Type>(l[3], l[4], l[5], l[6]);
    }

    CUDA_CALLABLE inline Type operator[](int index) const
    {
        assert(index < 7);

        return p.c[index];
    }

    CUDA_CALLABLE inline Type& operator[](int index)
    {
        assert(index < 7);

        return p.c[index];
    }    
};

template<typename Type=float32>
CUDA_CALLABLE inline transform_t<Type> transform_identity()
{
    return transform_t<Type>(vec_t<3,Type>(), quat_identity<Type>());
}

template<typename Type>
inline CUDA_CALLABLE transform_t<Type> operator - (const transform_t<Type>& x)
{
    transform_t<Type> ret;

    ret.p = -x.p;
    ret.q = -x.q;

    return ret;
}

template<typename Type>
CUDA_CALLABLE inline transform_t<Type> pos(const transform_t<Type>& x)
{
    return x;
}

template<typename Type>
CUDA_CALLABLE inline transform_t<Type> neg(const transform_t<Type>& x)
{
    return -x;
}

template<typename Type>
CUDA_CALLABLE inline void adj_neg(const transform_t<Type>& x, transform_t<Type>& adj_x, const transform_t<Type>& adj_ret)
{
    adj_x -= adj_ret;
}

template<typename Type>
inline CUDA_CALLABLE bool operator==(const transform_t<Type>& a, const transform_t<Type>& b)
{
    return a.p == b.p && a.q == b.q;
}


template<typename Type>
inline bool CUDA_CALLABLE isfinite(const transform_t<Type>& t)
{
    return isfinite(t.p) && isfinite(t.q);
}

template<typename Type>
CUDA_CALLABLE inline vec_t<3,Type> transform_get_translation(const transform_t<Type>& t)
{
    return t.p;
}

template<typename Type>
CUDA_CALLABLE inline quat_t<Type> transform_get_rotation(const transform_t<Type>& t)
{
    return t.q;
}

template<typename Type>
CUDA_CALLABLE inline void adj_transform_get_translation(const transform_t<Type>& t, transform_t<Type>& adj_t, const vec_t<3,Type>& adj_ret)
{
    adj_t.p += adj_ret;
}

template<typename Type>
CUDA_CALLABLE inline void adj_transform_get_rotation(const transform_t<Type>& t, transform_t<Type>& adj_t, const quat_t<Type>& adj_ret)
{
    adj_t.q += adj_ret;
}

template<typename Type>
CUDA_CALLABLE inline void transform_set_translation(transform_t<Type>& t, const vec_t<3, Type>& p)
{
    t.p = p;
}

template<typename Type>
CUDA_CALLABLE inline void transform_set_rotation(transform_t<Type>& t, const quat_t<Type>& q)
{
    t.q = q;
}

template<typename Type>
CUDA_CALLABLE inline transform_t<Type> transform_set_translation_copy(transform_t<Type>& t, const vec_t<3, Type>& p)
{
    transform_t<Type> ret(t);
    ret.p = p;
    return ret;
}

template<typename Type>
CUDA_CALLABLE inline transform_t<Type> transform_set_rotation_copy(transform_t<Type>& t, const quat_t<Type>& q)
{
    transform_t<Type> ret(t);
    ret.q = q;
    return ret;
}

template<typename Type>
CUDA_CALLABLE inline void adj_transform_set_translation(transform_t<Type>& t, const vec_t<3, Type>& p, const transform_t<Type>& adj_t, vec_t<3, Type>& adj_p)
{
    adj_p += adj_t.p;
}

template<typename Type>
CUDA_CALLABLE inline void adj_transform_set_rotation(transform_t<Type>& t, const quat_t<Type>& q, const transform_t<Type>& adj_t, quat_t<Type>& adj_q)
{
    adj_q += adj_t.q;
}

template<typename Type>
CUDA_CALLABLE inline void adj_transform_set_translation_copy(transform_t<Type>& t, const vec_t<3, Type>& p, transform_t<Type>& adj_t, vec_t<3, Type>& adj_p, const transform_t<Type>& adj_ret)
{
    adj_p += adj_ret.p;
    adj_t.q += adj_ret.q;
}

template<typename Type>
CUDA_CALLABLE inline void adj_transform_set_rotation_copy(transform_t<Type>& t, const quat_t<Type>& q, transform_t<Type>& adj_t, quat_t<Type>& adj_q, const transform_t<Type>& adj_ret)
{
    adj_q += adj_ret.q;
    adj_t.p += adj_ret.p;
}

template<typename Type>
inline CUDA_CALLABLE void transform_add_inplace(transform_t<Type>& t, const vec_t<3, Type>& p)
{
    t.p += p;
}

template<typename Type>
inline CUDA_CALLABLE void transform_sub_inplace(transform_t<Type>& t, const vec_t<3, Type>& p)
{
    t.p -= p;
}

template<typename Type>
inline CUDA_CALLABLE void adj_transform_add_inplace(transform_t<Type>& t, const vec_t<3, Type>& p, transform_t<Type>& adj_t, vec_t<3, Type>& adj_p)
{
    adj_p += adj_t.p;
}

template<typename Type>
inline CUDA_CALLABLE void adj_transform_sub_inplace(transform_t<Type>& t, const vec_t<3, Type>& p, transform_t<Type>& adj_t, vec_t<3, Type>& adj_p)
{
    adj_p -= adj_t.p;
}

template<typename Type>
CUDA_CALLABLE inline transform_t<Type> transform_multiply(const transform_t<Type>& a, const transform_t<Type>& b)
{
    return { quat_rotate(a.q, b.p) + a.p, mul(a.q, b.q) };
}

template<typename Type>
CUDA_CALLABLE inline void adj_transform_multiply(const transform_t<Type>& a, const transform_t<Type>& b, transform_t<Type>& adj_a, transform_t<Type>& adj_b, const transform_t<Type>& adj_ret)
{
    // translational part
    adj_quat_rotate(a.q, b.p, adj_a.q, adj_b.p, adj_ret.p);
    adj_a.p += adj_ret.p;

    // rotational part
    adj_mul(a.q, b.q, adj_a.q, adj_b.q, adj_ret.q);
}


template<typename Type>
CUDA_CALLABLE inline transform_t<Type> transform_inverse(const transform_t<Type>& t)
{
    quat_t<Type> q_inv = quat_inverse(t.q);
    return transform_t<Type>(-quat_rotate(q_inv, t.p), q_inv);
}

    
template<typename Type>
CUDA_CALLABLE inline vec_t<3,Type> transform_vector(const transform_t<Type>& t, const vec_t<3,Type>& x)
{
    return quat_rotate(t.q, x);
}

template<typename Type>
CUDA_CALLABLE inline vec_t<3,Type> transform_point(const transform_t<Type>& t, const vec_t<3,Type>& x)
{
    return t.p + quat_rotate(t.q, x);
}

// not totally sure why you'd want to do this seeing as adding/subtracting two rotation
// quats doesn't seem to do anything meaningful
template<typename Type>
CUDA_CALLABLE inline transform_t<Type> add(const transform_t<Type>& a, const transform_t<Type>& b)
{
    return { a.p + b.p, a.q + b.q };
}

template<typename Type>
CUDA_CALLABLE inline transform_t<Type> sub(const transform_t<Type>& a, const transform_t<Type>& b)
{
    return { a.p - b.p, a.q - b.q };
}

// also not sure why you'd want to do this seeing as the quat would end up unnormalized
template<typename Type>
CUDA_CALLABLE inline transform_t<Type> mul(const transform_t<Type>& a, Type s)
{
    return { a.p*s, a.q*s };
}

template<typename Type>
CUDA_CALLABLE inline transform_t<Type> mul(Type s, const transform_t<Type>& a)
{
    return mul(a, s);
}

template<typename Type>
CUDA_CALLABLE inline transform_t<Type> mul(const transform_t<Type>& a, const transform_t<Type>& b)
{
    return transform_multiply(a, b);
}

template<typename Type>
CUDA_CALLABLE inline transform_t<Type> operator*(const transform_t<Type>& a, Type s)
{
    return mul(a, s);
}

template<typename Type>
CUDA_CALLABLE inline transform_t<Type> operator*(Type s, const transform_t<Type>& a)
{
    return mul(a, s);
}

template<typename Type>
inline CUDA_CALLABLE Type tensordot(const transform_t<Type>& a, const transform_t<Type>& b)
{
    // corresponds to `np.tensordot()` with all axes being contracted
    return tensordot(a.p, b.p) + tensordot(a.q, b.q);
}

template<typename Type>
inline CUDA_CALLABLE Type extract(const transform_t<Type>& t, int idx)
{
#ifndef NDEBUG
    if (idx < -7 || idx >= 7)
    {
        printf("transformation index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    if (idx < 0)
    {
        idx += 7;
    }

    return t[idx];
}

template<unsigned SliceLength, typename Type>
inline CUDA_CALLABLE vec_t<SliceLength, Type> extract(const transform_t<Type> & t, slice_t slice)
{
    vec_t<SliceLength, Type> ret;

    assert(slice.start >= 0 && slice.start <= 7);
    assert(slice.stop >= -1 && slice.stop <= 7);
    assert(slice.step != 0 && slice.step < 0 ? slice.start >= slice.stop : slice.start <= slice.stop);
    assert(slice_get_length(slice) == SliceLength);

    bool is_reversed = slice.step < 0;

    int ii = 0;
    for (
        int i = slice.start;
        is_reversed ? (i > slice.stop) : (i < slice.stop);
        i += slice.step
    )
    {
        ret[ii] = t[i];
        ++ii;
    }

    assert(ii == SliceLength);
    return ret;
}

template<typename Type>
inline CUDA_CALLABLE Type* index(transform_t<Type>& t, int idx)
{
#ifndef NDEBUG
    if (idx < -7 || idx >= 7)
    {
        printf("transformation index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    if (idx < 0)
    {
        idx += 7;
    }

    return &t[idx];
}

template<typename Type>
inline CUDA_CALLABLE Type* indexref(transform_t<Type>* t, int idx)
{
#ifndef NDEBUG
    if (idx < -7 || idx >= 7)
    {
        printf("transformation index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    if (idx < 0)
    {
        idx += 7;
    }

    return &((*t)[idx]);
}

template<typename Type>
inline void CUDA_CALLABLE adj_extract(const transform_t<Type>& t, int idx, transform_t<Type>& adj_t, int& adj_idx, Type adj_ret)
{
    adj_t[idx] += adj_ret;
}

template<unsigned SliceLength, typename Type>
inline CUDA_CALLABLE void adj_extract(
    const transform_t<Type>& t, slice_t slice,
    transform_t<Type>& adj_t, slice_t& adj_slice,
    const vec_t<SliceLength, Type>& adj_ret
)
{
    assert(slice.start >= 0 && slice.start <= 7);
    assert(slice.stop >= -1 && slice.stop <= 7);
    assert(slice.step != 0 && slice.step < 0 ? slice.start >= slice.stop : slice.start <= slice.stop);
    assert(slice_get_length(slice) == SliceLength);

    bool is_reversed = slice.step < 0;

    int ii = 0;
    for (
        int i = slice.start;
        is_reversed ? (i > slice.stop) : (i < slice.stop);
        i += slice.step
    )
    {
        adj_t[i] += adj_ret[ii];
        ++ii;
    }

    assert(ii == SliceLength);
}

template<typename Type>
inline CUDA_CALLABLE void adj_index(transform_t<Type>& t, int idx,
                                       transform_t<Type>& adj_t, int adj_idx, const Type& adj_value)
{
    // nop
}

template<typename Type>
inline CUDA_CALLABLE void adj_indexref(transform_t<Type>* t, int idx, 
                                       transform_t<Type>& adj_t, int adj_idx, const Type& adj_value)
{
    // nop
}

template<typename Type>
inline CUDA_CALLABLE void add_inplace(transform_t<Type>& t, int idx, Type value)
{
#ifndef NDEBUG
    if (idx < -7 || idx >= 7)
    {
        printf("transformation index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    if (idx < 0)
    {
        idx += 7;
    }

    t[idx] += value;
}


template<typename Type>
inline CUDA_CALLABLE void add_inplace(transform_t<Type>& t, slice_t slice, Type value)
{
    assert(slice.start >= 0 && slice.start <= 7);
    assert(slice.stop >= -1 && slice.stop <= 7);
    assert(slice.step != 0 && slice.step < 0 ? slice.start >= slice.stop : slice.start <= slice.stop);

    bool is_reversed = slice.step < 0;

    for (
        int i = slice.start;
        is_reversed ? (i > slice.stop) : (i < slice.stop);
        i += slice.step
    )
    {
        t[i] += value;
    }
}


template<unsigned SliceLength, typename Type>
inline CUDA_CALLABLE void add_inplace(transform_t<Type>& t, slice_t slice, const vec_t<SliceLength, Type> &a)
{
    assert(slice.start >= 0 && slice.start <= 7);
    assert(slice.stop >= -1 && slice.stop <= 7);
    assert(slice.step != 0 && slice.step < 0 ? slice.start >= slice.stop : slice.start <= slice.stop);
    assert(slice_get_length(slice) == SliceLength);

    bool is_reversed = slice.step < 0;

    int ii = 0;
    for (
        int i = slice.start;
        is_reversed ? (i > slice.stop) : (i < slice.stop);
        i += slice.step
    )
    {
        t[i] += a[ii];
        ++ii;
    }

    assert(ii == SliceLength);
}


template<typename Type>
inline CUDA_CALLABLE void adj_add_inplace(transform_t<Type>& t, int idx, Type value,
                                        transform_t<Type>& adj_t, int adj_idx, Type& adj_value)
{
#ifndef NDEBUG
    if (idx < -7 || idx >= 7)
    {
        printf("transformation index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    if (idx < 0)
    {
        idx += 7;
    }

    adj_value += adj_t[idx];
}


template<typename Type>
inline CUDA_CALLABLE void adj_add_inplace(
    const transform_t<Type>& t, slice_t slice, Type value,
    transform_t<Type>& adj_t, slice_t& adj_slice, Type& adj_value
)
{
    assert(slice.start >= 0 && slice.start <= 7);
    assert(slice.stop >= -1 && slice.stop <= 7);
    assert(slice.step != 0 && slice.step < 0 ? slice.start >= slice.stop : slice.start <= slice.stop);

    bool is_reversed = slice.step < 0;

    for (
        int i = slice.start;
        is_reversed ? (i > slice.stop) : (i < slice.stop);
        i += slice.step
    )
    {
        adj_value += adj_t[i];
    }
}


template<unsigned SliceLength, typename Type>
inline CUDA_CALLABLE void adj_add_inplace(
    const transform_t<Type>& t, slice_t slice, const vec_t<SliceLength, Type> &a,
    transform_t<Type>& adj_t, slice_t& adj_slice, vec_t<SliceLength, Type>& adj_a
)
{
    assert(slice.start >= 0 && slice.start <= 7);
    assert(slice.stop >= -1 && slice.stop <= 7);
    assert(slice.step != 0 && slice.step < 0 ? slice.start >= slice.stop : slice.start <= slice.stop);
    assert(slice_get_length(slice) == SliceLength);

    bool is_reversed = slice.step < 0;

    int ii = 0;
    for (
        int i = slice.start;
        is_reversed ? (i > slice.stop) : (i < slice.stop);
        i += slice.step
    )
    {
        adj_a[ii] += adj_t[i];
        ++ii;
    }

    assert(ii == SliceLength);
}


template<typename Type>
inline CUDA_CALLABLE void sub_inplace(transform_t<Type>& t, int idx, Type value)
{
#ifndef NDEBUG
    if (idx < -7 || idx >= 7)
    {
        printf("transformation index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    if (idx < 0)
    {
        idx += 7;
    }

    t[idx] -= value;
}


template<typename Type>
inline CUDA_CALLABLE void sub_inplace(transform_t<Type>& t, slice_t slice, Type value)
{
    assert(slice.start >= 0 && slice.start <= 7);
    assert(slice.stop >= -1 && slice.stop <= 7);
    assert(slice.step != 0 && slice.step < 0 ? slice.start >= slice.stop : slice.start <= slice.stop);

    bool is_reversed = slice.step < 0;

    for (
        int i = slice.start;
        is_reversed ? (i > slice.stop) : (i < slice.stop);
        i += slice.step
    )
    {
        t[i] -= value;
    }
}


template<unsigned SliceLength, typename Type>
inline CUDA_CALLABLE void sub_inplace(transform_t<Type>& t, slice_t slice, const vec_t<SliceLength, Type> &a)
{
    assert(slice.start >= 0 && slice.start <= 7);
    assert(slice.stop >= -1 && slice.stop <= 7);
    assert(slice.step != 0 && slice.step < 0 ? slice.start >= slice.stop : slice.start <= slice.stop);
    assert(slice_get_length(slice) == SliceLength);

    bool is_reversed = slice.step < 0;

    int ii = 0;
    for (
        int i = slice.start;
        is_reversed ? (i > slice.stop) : (i < slice.stop);
        i += slice.step
    )
    {
        t[i] -= a[ii];
        ++ii;
    }

    assert(ii == SliceLength);
}


template<typename Type>
inline CUDA_CALLABLE void adj_sub_inplace(transform_t<Type>& t, int idx, Type value,
                                        transform_t<Type>& adj_t, int adj_idx, Type& adj_value)
{
#ifndef NDEBUG
    if (idx < -7 || idx >= 7)
    {
        printf("transformation index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    if (idx < 0)
    {
        idx += 7;
    }

    adj_value -= adj_t[idx];
}


template<typename Type>
inline CUDA_CALLABLE void adj_sub_inplace(
    const transform_t<Type>& t, slice_t slice, Type value,
    transform_t<Type>& adj_t, slice_t& adj_slice, Type& adj_value
)
{
    assert(slice.start >= 0 && slice.start <= 7);
    assert(slice.stop >= -1 && slice.stop <= 7);
    assert(slice.step != 0 && slice.step < 0 ? slice.start >= slice.stop : slice.start <= slice.stop);

    bool is_reversed = slice.step < 0;
    for (
        int i = slice.start;
        is_reversed ? (i > slice.stop) : (i < slice.stop);
        i += slice.step
    )
    {
        adj_value -= adj_t[i];
    }
}


template<unsigned SliceLength, typename Type>
inline CUDA_CALLABLE void adj_sub_inplace(
    const transform_t<Type>& t, slice_t slice, const vec_t<SliceLength, Type> &a,
    transform_t<Type>& adj_t, slice_t& adj_slice, vec_t<SliceLength, Type>& adj_a
)
{
    assert(slice.start >= 0 && slice.start <= 7);
    assert(slice.stop >= -1 && slice.stop <= 7);
    assert(slice.step != 0 && slice.step < 0 ? slice.start >= slice.stop : slice.start <= slice.stop);
    assert(slice_get_length(slice) == SliceLength);

    bool is_reversed = slice.step < 0;

    int ii = 0;
    for (
        int i = slice.start;
        is_reversed ? (i > slice.stop) : (i < slice.stop);
        i += slice.step
    )
    {
        adj_a[ii] -= adj_t[i];
        ++ii;
    }

    assert(ii == SliceLength);
}


template<typename Type>
inline CUDA_CALLABLE void assign_inplace(transform_t<Type>& t, int idx, Type value)
{
#ifndef NDEBUG
    if (idx < -7 || idx >= 7)
    {
        printf("transformation index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    if (idx < 0)
    {
        idx += 7;
    }

    t[idx] = value;
}

template<typename Type>
inline CUDA_CALLABLE void assign_inplace(transform_t<Type>& t, slice_t slice, Type value)
{
    assert(slice.start >= 0 && slice.start <= 7);
    assert(slice.stop >= -1 && slice.stop <= 7);
    assert(slice.step != 0 && slice.step < 0 ? slice.start >= slice.stop : slice.start <= slice.stop);

    bool is_reversed = slice.step < 0;

    for (
        int i = slice.start;
        is_reversed ? (i > slice.stop) : (i < slice.stop);
        i += slice.step
    )
    {
        t[i] = value;
    }
}

template<unsigned SliceLength, typename Type>
inline CUDA_CALLABLE void assign_inplace(transform_t<Type>& t, slice_t slice, const vec_t<SliceLength, Type> &a)
{
    assert(slice.start >= 0 && slice.start <= 7);
    assert(slice.stop >= -1 && slice.stop <= 7);
    assert(slice.step != 0 && slice.step < 0 ? slice.start >= slice.stop : slice.start <= slice.stop);
    assert(slice_get_length(slice) == SliceLength);

    bool is_reversed = slice.step < 0;

    int ii = 0;
    for (
        int i = slice.start;
        is_reversed ? (i > slice.stop) : (i < slice.stop);
        i += slice.step
    )
    {
        t[i] = a[ii];
        ++ii;
    }

    assert(ii == SliceLength);
}

template<typename Type>
inline CUDA_CALLABLE void adj_assign_inplace(transform_t<Type>& t, int idx, Type value, transform_t<Type>& adj_t, int& adj_idx, Type& adj_value)
{
#ifndef NDEBUG
    if (idx < -7 || idx >= 7)
    {
        printf("transformation index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    if (idx < 0)
    {
        idx += 7;
    }

    adj_value += adj_t[idx];
}


template<typename Type>
inline CUDA_CALLABLE void adj_assign_inplace(
    const transform_t<Type>& t, slice_t slice, Type value,
    transform_t<Type>& adj_t, slice_t& adj_slice, Type& adj_value
)
{
    assert(slice.start >= 0 && slice.start <= 7);
    assert(slice.stop >= -1 && slice.stop <= 7);
    assert(slice.step != 0 && slice.step < 0 ? slice.start >= slice.stop : slice.start <= slice.stop);

    bool is_reversed = slice.step < 0;

    for (
        int i = slice.start;
        is_reversed ? (i > slice.stop) : (i < slice.stop);
        i += slice.step
    )
    {
        adj_value += adj_t[i];
    }
}

template<unsigned SliceLength, typename Type>
inline CUDA_CALLABLE void adj_assign_inplace(
    const transform_t<Type>& t, slice_t slice, const vec_t<SliceLength, Type> &a,
    transform_t<Type>& adj_t, slice_t& adj_slice, vec_t<SliceLength, Type>& adj_a
)
{
    assert(slice.start >= 0 && slice.start <= 7);
    assert(slice.stop >= -1 && slice.stop <= 7);
    assert(slice.step != 0 && slice.step < 0 ? slice.start >= slice.stop : slice.start <= slice.stop);
    assert(slice_get_length(slice) == SliceLength);

    bool is_reversed = slice.step < 0;

    int ii = 0;
    for (
        int i = slice.start;
        is_reversed ? (i > slice.stop) : (i < slice.stop);
        i += slice.step
    )
    {
        adj_a[ii] += adj_t[i];
        ++ii;
    }

    assert(ii == SliceLength);
}


template<typename Type>
inline CUDA_CALLABLE transform_t<Type> assign_copy(transform_t<Type>& t, int idx, Type value)
{
#ifndef NDEBUG
    if (idx < -7 || idx >= 7)
    {
        printf("transformation index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    if (idx < 0)
    {
        idx += 7;
    }

    transform_t<Type> ret(t);
    ret[idx] = value;
    return ret;
}

template<typename Type>
inline CUDA_CALLABLE transform_t<Type> assign_copy(transform_t<Type>& t, slice_t slice, Type value)
{
    transform_t<Type> ret(t);
    assign_inplace(ret, slice, value);
    return ret;
}

template<unsigned SliceLength, typename Type>
inline CUDA_CALLABLE transform_t<Type> assign_copy(transform_t<Type>& t, slice_t slice, const vec_t<SliceLength, Type> &a)
{
    transform_t<Type> ret(t);
    assign_inplace<SliceLength>(ret, slice, a);
    return ret;
}

template<typename Type>
inline CUDA_CALLABLE void adj_assign_copy(transform_t<Type>& t, int idx, Type value, transform_t<Type>& adj_t, int& adj_idx, Type& adj_value, const transform_t<Type>& adj_ret)
{
#ifndef NDEBUG
    if (idx < -7 || idx >= 7)
    {
        printf("transformation index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    if (idx < 0)
    {
        idx += 7;
    }

    adj_value += adj_ret[idx];
    for(unsigned i=0; i < 7; ++i)
    {
        if (i != idx)
            adj_t[i] += adj_ret[i];
    }
}

template<typename Type>
inline CUDA_CALLABLE void adj_assign_copy(
    transform_t<Type>& t, slice_t slice, Type value,
    transform_t<Type>& adj_t, slice_t& adj_slice, Type& adj_value,
    const transform_t<Type>& adj_ret
)
{
    assert(slice.start >= 0 && slice.start <= 7);
    assert(slice.stop >= -1 && slice.stop <= 7);
    assert(slice.step != 0 && slice.step < 0 ? slice.start >= slice.stop : slice.start <= slice.stop);

    bool is_reversed = slice.step < 0;

    for (int i = 0; i < 7; ++i)
    {
        bool in_slice = is_reversed
            ? (i <= slice.start && i > slice.stop && (slice.start - i) % (-slice.step) == 0)
            : (i >= slice.start && i < slice.stop && (i - slice.start) % slice.step == 0);

        if (!in_slice)
        {
            adj_t[i] += adj_ret[i];
        }
        else
        {
            adj_value += adj_ret[i];
        }
    }
}

template<unsigned SliceLength, typename Type>
inline CUDA_CALLABLE void adj_assign_copy(
    transform_t<Type>& t, slice_t slice, const vec_t<SliceLength, Type> &a,
    transform_t<Type>& adj_t, slice_t& adj_slice, vec_t<SliceLength, Type>& adj_a,
    const transform_t<Type>& adj_ret
)
{
    assert(slice.start >= 0 && slice.start <= 7);
    assert(slice.stop >= -1 && slice.stop <= 7);
    assert(slice.step != 0 && slice.step < 0 ? slice.start >= slice.stop : slice.start <= slice.stop);
    assert(slice_get_length(slice) == SliceLength);

    bool is_reversed = slice.step < 0;

    int ii = 0;
    for (int i = 0; i < 7; ++i)
    {
        bool in_slice = is_reversed
            ? (i <= slice.start && i > slice.stop && (slice.start - i) % (-slice.step) == 0)
            : (i >= slice.start && i < slice.stop && (i - slice.start) % slice.step == 0);

        if (!in_slice)
        {
            adj_t[i] += adj_ret[i];
        }
        else
        {
            adj_a[ii] += adj_ret[i];
            ++ii;
        }
    }

    assert(ii == SliceLength);
}


// adjoint methods
template<typename Type>
CUDA_CALLABLE inline void adj_add(const transform_t<Type>& a, const transform_t<Type>& b, transform_t<Type>& adj_a, transform_t<Type>& adj_b, const transform_t<Type>& adj_ret)
{
    adj_add(a.p, b.p, adj_a.p, adj_b.p, adj_ret.p);
    adj_add(a.q, b.q, adj_a.q, adj_b.q, adj_ret.q);
}

template<typename Type>
CUDA_CALLABLE inline void adj_add(
    const transform_t<Type>& a, Type b,
    transform_t<Type>& adj_a, Type& adj_b,
    const transform_t<Type>& adj_ret
)
{
    adj_a += adj_ret;

    adj_b += adj_ret.p[0];
    adj_b += adj_ret.p[1];
    adj_b += adj_ret.p[2];

    adj_b += adj_ret.q[0];
    adj_b += adj_ret.q[1];
    adj_b += adj_ret.q[2];
    adj_b += adj_ret.q[3];
}

template<typename Type>
CUDA_CALLABLE inline void adj_sub(const transform_t<Type>& a, const transform_t<Type>& b, transform_t<Type>& adj_a, transform_t<Type>& adj_b, const transform_t<Type>& adj_ret)
{
    adj_sub(a.p, b.p, adj_a.p, adj_b.p, adj_ret.p);
    adj_sub(a.q, b.q, adj_a.q, adj_b.q, adj_ret.q);
}

template<typename Type>
CUDA_CALLABLE inline void adj_sub(
    const transform_t<Type>& a, Type b,
    transform_t<Type>& adj_a, Type& adj_b,
    const transform_t<Type>& adj_ret
)
{
    adj_a -= adj_ret;

    adj_b -= adj_ret.p[0];
    adj_b -= adj_ret.p[1];
    adj_b -= adj_ret.p[2];

    adj_b -= adj_ret.q[0];
    adj_b -= adj_ret.q[1];
    adj_b -= adj_ret.q[2];
    adj_b -= adj_ret.q[3];
}

template<typename Type>
CUDA_CALLABLE inline void adj_mul(const transform_t<Type>& a, Type s, transform_t<Type>& adj_a, Type& adj_s, const transform_t<Type>& adj_ret)
{
    adj_mul(a.p, s, adj_a.p, adj_s, adj_ret.p);
    adj_mul(a.q, s, adj_a.q, adj_s, adj_ret.q);
}

template<typename Type>
CUDA_CALLABLE inline void adj_mul(Type s, const transform_t<Type>& a, Type& adj_s, transform_t<Type>& adj_a, const transform_t<Type>& adj_ret)
{
    adj_mul(a, s, adj_a, adj_s, adj_ret);
}

template<typename Type>
CUDA_CALLABLE inline void adj_mul(const transform_t<Type>& a, const transform_t<Type>& b, transform_t<Type>& adj_a, transform_t<Type>& adj_b, const transform_t<Type>& adj_ret)
{
    adj_transform_multiply(a, b, adj_a, adj_b, adj_ret);
}


template<typename Type>
inline CUDA_CALLABLE transform_t<Type> atomic_add(transform_t<Type>* addr, const transform_t<Type>& value) 
{   
    vec_t<3,Type> p = atomic_add(&addr->p, value.p);
    quat_t<Type> q = atomic_add(&addr->q, value.q);

    return transform_t<Type>(p, q);
}

template<typename Type>
CUDA_CALLABLE inline void adj_transform_t(const vec_t<3,Type>& p, const quat_t<Type>& q, vec_t<3,Type>& adj_p, quat_t<Type>& adj_q, const transform_t<Type>& adj_ret)
{
    adj_p += adj_ret.p;
    adj_q += adj_ret.q;
}

template<typename Type>
CUDA_CALLABLE inline void adj_transform_inverse(const transform_t<Type>& t, transform_t<Type>& adj_t, const transform_t<Type>& adj_ret)
{

    // forward
    quat_t<Type> q_inv = quat_inverse(t.q); 
    vec_t<3,Type> p = quat_rotate(q_inv, t.p);
    vec_t<3,Type> np = -p;
    // transform<Type> t = transform<Type>(np, q_inv)

    // backward
    quat_t<Type> adj_q_inv(0.0f);
    quat_t<Type> adj_q(0.0f);
    vec_t<3,Type> adj_p(0.0f);
    vec_t<3,Type> adj_np(0.0f);

    adj_transform_t(np, q_inv, adj_np, adj_q_inv, adj_ret);
    adj_p = -adj_np;
    adj_quat_rotate(q_inv, t.p, adj_q_inv, adj_t.p, adj_p);
    adj_quat_inverse(t.q, adj_t.q, adj_q_inv);
    
}

template<typename Type>
CUDA_CALLABLE inline void adj_transform_vector(const transform_t<Type>& t, const vec_t<3,Type>& x, transform_t<Type>& adj_t, vec_t<3,Type>& adj_x, const vec_t<3,Type>& adj_ret)
{
    adj_quat_rotate(t.q, x, adj_t.q, adj_x, adj_ret);
}

template<typename Type>
CUDA_CALLABLE inline void adj_transform_point(const transform_t<Type>& t, const vec_t<3,Type>& x, transform_t<Type>& adj_t, vec_t<3,Type>& adj_x, const vec_t<3,Type>& adj_ret)
{
    adj_quat_rotate(t.q, x, adj_t.q, adj_x, adj_ret);
    adj_t.p += adj_ret;
}


template<typename Type>
CUDA_CALLABLE void print(transform_t<Type> t);

template<typename Type>
CUDA_CALLABLE inline transform_t<Type> lerp(const transform_t<Type>& a, const transform_t<Type>& b, Type t)
{
    return a*(Type(1)-t) + b*t;
}

template<typename Type>
CUDA_CALLABLE inline void adj_lerp(const transform_t<Type>& a, const transform_t<Type>& b, Type t, transform_t<Type>& adj_a, transform_t<Type>& adj_b, Type& adj_t, const transform_t<Type>& adj_ret)
{
    adj_a += adj_ret*(Type(1)-t);
    adj_b += adj_ret*t;
    adj_t += tensordot(b, adj_ret) - tensordot(a, adj_ret);
}

template<typename Type>
CUDA_CALLABLE inline int len(const transform_t<Type>& t)
{
    return 7;
}

template<typename Type>
CUDA_CALLABLE inline void adj_len(const transform_t<Type>& t, transform_t<Type>& adj_t, const int& adj_ret)
{
}

template<typename Type>
using spatial_matrix_t = mat_t<6,6,Type>;

template<typename Type>
inline CUDA_CALLABLE spatial_matrix_t<Type> spatial_adjoint(const mat_t<3,3,Type>& R, const mat_t<3,3,Type>& S)
{    
    spatial_matrix_t<Type> adT;

    // T = [Rah,   0]
    //     [S  R]

    // diagonal blocks    
    for (int i=0; i < 3; ++i)
    {
        for (int j=0; j < 3; ++j)
        {
            adT.data[i][j] = R.data[i][j];
            adT.data[i+3][j+3] = R.data[i][j];
        }
    }

    // lower off diagonal
    for (int i=0; i < 3; ++i)
    {
        for (int j=0; j < 3; ++j)
        {
            adT.data[i+3][j] = S.data[i][j];
        }
    }

    return adT;
}

template<typename Type>
inline CUDA_CALLABLE void adj_spatial_adjoint(const mat_t<3,3,Type>& R, const mat_t<3,3,Type>& S, mat_t<3,3,Type>& adj_R, mat_t<3,3,Type>& adj_S, const spatial_matrix_t<Type>& adj_ret)
{
    // diagonal blocks    
    for (int i=0; i < 3; ++i)
    {
        for (int j=0; j < 3; ++j)
        {
            adj_R.data[i][j] += adj_ret.data[i][j];
            adj_R.data[i][j] += adj_ret.data[i+3][j+3];
        }
    }

    // lower off diagonal
    for (int i=0; i < 3; ++i)
    {
        for (int j=0; j < 3; ++j)
        {
            adj_S.data[i][j] += adj_ret.data[i+3][j];
        }
    }
}


CUDA_CALLABLE inline int row_index(int stride, int i, int j)
{
    return i*stride + j;
}

// builds spatial Jacobian J which is an (joint_count*6)x(dof_count) matrix
template<typename Type>
CUDA_CALLABLE inline void spatial_jacobian(
    const spatial_vector_t<Type>* S,
    const int* joint_parents, 
    const int* joint_qd_start, 
    int joint_start,    // offset of the first joint for the articulation
    int joint_count,    
    int J_start,
    Type* J)
{
    const int articulation_dof_start = joint_qd_start[joint_start];
    const int articulation_dof_end = joint_qd_start[joint_start + joint_count];
    const int articulation_dof_count = articulation_dof_end-articulation_dof_start;

	// shift output pointers
	const int S_start = articulation_dof_start;

	S += S_start;
	J += J_start;
	
    for (int i=0; i < joint_count; ++i)
    {
        const int row_start = i * 6;

        int j = joint_start + i;
        while (j != -1)
        {
            const int joint_dof_start = joint_qd_start[j];
            const int joint_dof_end = joint_qd_start[j+1];
            const int joint_dof_count = joint_dof_end-joint_dof_start;

            // fill out each row of the Jacobian walking up the tree
            //for (int col=dof_start; col < dof_end; ++col)
            for (int dof=0; dof < joint_dof_count; ++dof)
            {
                const int col = (joint_dof_start-articulation_dof_start) + dof;

                J[row_index(articulation_dof_count, row_start+0, col)] = S[col].w[0];
                J[row_index(articulation_dof_count, row_start+1, col)] = S[col].w[1];
                J[row_index(articulation_dof_count, row_start+2, col)] = S[col].w[2];
                J[row_index(articulation_dof_count, row_start+3, col)] = S[col].v[0];
                J[row_index(articulation_dof_count, row_start+4, col)] = S[col].v[1];
                J[row_index(articulation_dof_count, row_start+5, col)] = S[col].v[2];
            }

            j = joint_parents[j];
        }
    }
}

template<typename Type>
CUDA_CALLABLE inline void adj_spatial_jacobian(
    const spatial_vector_t<Type>* S, 
    const int* joint_parents, 
    const int* joint_qd_start, 
    const int joint_start,
    const int joint_count, 
    const int J_start, 
    const Type* J,
    // adjs
    spatial_vector_t<Type>* adj_S, 
    int* adj_joint_parents, 
    int* adj_joint_qd_start, 
    int& adj_joint_start,
    int& adj_joint_count, 
    int& adj_J_start, 
    const Type* adj_J)
{   
    const int articulation_dof_start = joint_qd_start[joint_start];
    const int articulation_dof_end = joint_qd_start[joint_start + joint_count];
    const int articulation_dof_count = articulation_dof_end-articulation_dof_start;

	// shift output pointers
	const int S_start = articulation_dof_start;

	S += S_start;
	J += J_start;

    adj_S += S_start;
    adj_J += J_start;
	
    for (int i=0; i < joint_count; ++i)
    {
        const int row_start = i * 6;

        int j = joint_start + i;
        while (j != -1)
        {
            const int joint_dof_start = joint_qd_start[j];
            const int joint_dof_end = joint_qd_start[j+1];
            const int joint_dof_count = joint_dof_end-joint_dof_start;

            // fill out each row of the Jacobian walking up the tree
            //for (int col=dof_start; col < dof_end; ++col)
            for (int dof=0; dof < joint_dof_count; ++dof)
            {
                const int col = (joint_dof_start-articulation_dof_start) + dof;

                adj_S[col].w[0] += adj_J[row_index(articulation_dof_count, row_start+0, col)];
                adj_S[col].w[1] += adj_J[row_index(articulation_dof_count, row_start+1, col)];
                adj_S[col].w[2] += adj_J[row_index(articulation_dof_count, row_start+2, col)];
                adj_S[col].v[0] += adj_J[row_index(articulation_dof_count, row_start+3, col)];
                adj_S[col].v[1] += adj_J[row_index(articulation_dof_count, row_start+4, col)];
                adj_S[col].v[2] += adj_J[row_index(articulation_dof_count, row_start+5, col)];
            }

            j = joint_parents[j];
        }
    }
}


template<typename Type>
CUDA_CALLABLE inline void spatial_mass(const spatial_matrix_t<Type>* I_s, int joint_start, int joint_count, int M_start, Type* M)
{
    const int stride = joint_count*6;

    for (int l=0; l < joint_count; ++l)
    {
        for (int i=0; i < 6; ++i)
        {
            for (int j=0; j < 6; ++j)
            {
                M[M_start + row_index(stride, l*6 + i, l*6 + j)] = I_s[joint_start + l].data[i][j];
            }
        }
    } 
}

template<typename Type>
CUDA_CALLABLE inline void adj_spatial_mass(
    const spatial_matrix_t<Type>* I_s, 
    const int joint_start,
    const int joint_count, 
    const int M_start,
    const Type* M,
    spatial_matrix_t<Type>* adj_I_s, 
    int& adj_joint_start,
    int& adj_joint_count, 
    int& adj_M_start,
    const Type* adj_M)
{
    const int stride = joint_count*6;

    for (int l=0; l < joint_count; ++l)
    {
        for (int i=0; i < 6; ++i)
        {
            for (int j=0; j < 6; ++j)
            {
                adj_I_s[joint_start + l].data[i][j] += adj_M[M_start + row_index(stride, l*6 + i, l*6 + j)];
            }
        }
    } 
}

using transform = transform_t<float>;
using transformh = transform_t<half>;
using transformf = transform_t<float>;
using transformd = transform_t<double>;

using spatial_vector = spatial_vector_t<float>;
using spatial_vectorh = spatial_vector_t<half>;
using spatial_vectorf = spatial_vector_t<float>;
using spatial_vectord = spatial_vector_t<double>;

using spatial_matrix = spatial_matrix_t<float>;
using spatial_matrixh = spatial_matrix_t<half>;
using spatial_matrixf = spatial_matrix_t<float>;
using spatial_matrixd = spatial_matrix_t<double>;

 } // namespace wp