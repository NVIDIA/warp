/** Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

namespace wp
{

//---------------------------------------------------------------------------------
// Represents a twist in se(3)
template<typename Type>
struct spatial_vector_t
{
    vec<3,Type> w;
    vec<3,Type> v;

    CUDA_CALLABLE inline spatial_vector_t(Type a, Type b, Type c, Type d, Type e, Type f) : w(a, b, c), v(d, e, f) {}
    CUDA_CALLABLE inline spatial_vector_t(vec<3,Type> w=vec<3,Type>(), vec<3,Type> v=vec<3,Type>()) : w(w), v(v) {}
    CUDA_CALLABLE inline spatial_vector_t(Type a) : w(a, a, a), v(a, a, a) {}

    CUDA_CALLABLE inline Type operator[](int index) const
    {
        assert(index < 6);

        return (&w.c[0])[index];
    }

    CUDA_CALLABLE inline Type& operator[](int index)
    {
        assert(index < 6);

        return (&w.c[0])[index];
    }
};

template<typename Type>
inline CUDA_CALLABLE bool operator==(const spatial_vector_t<Type>& a, const spatial_vector_t<Type>& b)
{
    return a.w == b.w && a.v == b.v;
}

template<typename Type>
inline bool CUDA_CALLABLE isfinite(const spatial_vector_t<Type>& s)
{
    return isfinite(s.w) && isfinite(s.v);
}

template<typename Type>
CUDA_CALLABLE inline spatial_vector_t<Type> operator - (spatial_vector_t<Type> a)
{
    return spatial_vector_t<Type>(-a.w, -a.v);
}

template<typename Type>
CUDA_CALLABLE inline spatial_vector_t<Type> add(const spatial_vector_t<Type>& a, const spatial_vector_t<Type>& b)
{
    return { a.w + b.w, a.v + b.v };
}

template<typename Type>
CUDA_CALLABLE inline spatial_vector_t<Type> sub(const spatial_vector_t<Type>& a, const spatial_vector_t<Type>& b)
{
    return { a.w - b.w, a.v - b.v };
}

template<typename Type>
CUDA_CALLABLE inline spatial_vector_t<Type> mul(const spatial_vector_t<Type>& a, Type s)
{
    return { a.w*s, a.v*s };
}

template<typename Type>
CUDA_CALLABLE inline spatial_vector_t<Type> mul(Type s, const spatial_vector_t<Type>& a)
{
    return mul(a, s);
}

template<typename Type>
CUDA_CALLABLE inline Type spatial_dot(const spatial_vector_t<Type>& a, const spatial_vector_t<Type>& b)
{
    return dot(a.w, b.w) + dot(a.v, b.v);
}

template<typename Type>
CUDA_CALLABLE inline spatial_vector_t<Type> spatial_cross(const spatial_vector_t<Type>& a,  const spatial_vector_t<Type>& b)
{
    vec<3,Type> w = cross(a.w, b.w);
    vec<3,Type> v = cross(a.v, b.w) + cross(a.w, b.v);
    
    return spatial_vector_t<Type>(w, v);
}

template<typename Type>
CUDA_CALLABLE inline spatial_vector_t<Type> spatial_cross_dual(const spatial_vector_t<Type>& a,  const spatial_vector_t<Type>& b)
{
    vec<3,Type> w = cross(a.w, b.w) + cross(a.v, b.v);
    vec<3,Type> v = cross(a.w, b.v);

    return spatial_vector_t<Type>(w, v);
}

template<typename Type>
CUDA_CALLABLE inline vec<3,Type> spatial_top(const spatial_vector_t<Type>& a)
{
    return a.w;
}

template<typename Type>
CUDA_CALLABLE inline vec<3,Type> spatial_bottom(const spatial_vector_t<Type>& a)
{
    return a.v;
}

template<typename Type>
inline CUDA_CALLABLE Type tensordot(const spatial_vector_t<Type>& a, const spatial_vector_t<Type>& b)
{
    // corresponds to `np.tensordot()` with all axes being contracted
    return tensordot(a.v, b.v) + tensordot(a.w, b.w);
}

template<typename Type>
inline CUDA_CALLABLE Type index(const spatial_vector_t<Type>& v, int i)
{
#if FP_CHECK
    if (i < 0 || i > 5)
    {
        printf("spatial_vector_t<Type> index %d out of bounds at %s %d\n", i, __FILE__, __LINE__);
        assert(0);
    }
#endif
    return v[i];
}

template<typename Type>
inline void CUDA_CALLABLE adj_index(const spatial_vector_t<Type>& m, int i, spatial_vector_t<Type>& adj_v, int& adj_i, Type adj_ret)
{
#if FP_CHECK
    if (i < 0 || i > 5)
    {
        printf("spatial_vector_t<Type> index %d out of bounds at %s %d\n", i, __FILE__, __LINE__);
        assert(0);
    }
#endif
    adj_v[i] += adj_ret;
}


// adjoint methods
template<typename Type>
CUDA_CALLABLE inline void adj_spatial_vector_t(
    Type a, Type b, Type c, 
    Type d, Type e, Type f, 
    Type& adj_a, Type& adj_b, Type& adj_c,
    Type& adj_d, Type& adj_e,Type& adj_f, 
    const spatial_vector_t<Type>& adj_ret)
{
    adj_a += adj_ret.w[0];
    adj_b += adj_ret.w[1];
    adj_c += adj_ret.w[2];
    
    adj_d += adj_ret.v[0];
    adj_e += adj_ret.v[1];
    adj_f += adj_ret.v[2];
}

template<typename Type>
CUDA_CALLABLE inline void adj_spatial_vector_t(const vec<3,Type>& w, const vec<3,Type>& v, vec<3,Type>& adj_w, vec<3,Type>& adj_v, const spatial_vector_t<Type>& adj_ret)
{
    adj_w += adj_ret.w;
    adj_v += adj_ret.v;
}

template<typename Type>
CUDA_CALLABLE inline void adj_spatial_vector_t(Type a, Type& adj_a, const spatial_vector_t<Type>& adj_ret)
{
    adj_a += spatial_dot(adj_ret, spatial_vector_t<Type>(1.0));
}

template<typename Type>
CUDA_CALLABLE inline void adj_add(const spatial_vector_t<Type>& a, const spatial_vector_t<Type>& b, spatial_vector_t<Type>& adj_a, spatial_vector_t<Type>& adj_b, const spatial_vector_t<Type>& adj_ret)
{
    adj_add(a.w, b.w, adj_a.w, adj_b.w, adj_ret.w);
    adj_add(a.v, b.v, adj_a.v, adj_b.v, adj_ret.v);
}

template<typename Type>
CUDA_CALLABLE inline void adj_sub(const spatial_vector_t<Type>& a, const spatial_vector_t<Type>& b, spatial_vector_t<Type>& adj_a, spatial_vector_t<Type>& adj_b, const spatial_vector_t<Type>& adj_ret)
{
    adj_sub(a.w, b.w, adj_a.w, adj_b.w, adj_ret.w);
    adj_sub(a.v, b.v, adj_a.v, adj_b.v, adj_ret.v);
}

template<typename Type>
CUDA_CALLABLE inline void adj_mul(const spatial_vector_t<Type>& a, Type s, spatial_vector_t<Type>& adj_a, Type& adj_s, const spatial_vector_t<Type>& adj_ret)
{
    adj_mul(a.w, s, adj_a.w, adj_s, adj_ret.w);
    adj_mul(a.v, s, adj_a.v, adj_s, adj_ret.v);
}

template<typename Type>
CUDA_CALLABLE inline void adj_mul(Type s, const spatial_vector_t<Type>& a, Type& adj_s, spatial_vector_t<Type>& adj_a, const spatial_vector_t<Type>& adj_ret)
{
    adj_mul(a, s, adj_a, adj_s, adj_ret);
}

template<typename Type>
CUDA_CALLABLE inline void adj_spatial_dot(const spatial_vector_t<Type>& a, const spatial_vector_t<Type>& b, spatial_vector_t<Type>& adj_a, spatial_vector_t<Type>& adj_b, const Type& adj_ret)
{
    adj_dot(a.w, b.w, adj_a.w, adj_b.w, adj_ret);
    adj_dot(a.v, b.v, adj_a.v, adj_b.v, adj_ret);
}

template<typename Type>
CUDA_CALLABLE inline void adj_spatial_cross(const spatial_vector_t<Type>& a,  const spatial_vector_t<Type>& b, spatial_vector_t<Type>& adj_a,  spatial_vector_t<Type>& adj_b, const spatial_vector_t<Type>& adj_ret)
{
    adj_cross(a.w, b.w, adj_a.w, adj_b.w, adj_ret.w);
    
    adj_cross(a.v, b.w, adj_a.v, adj_b.w, adj_ret.v);
    adj_cross(a.w, b.v, adj_a.w, adj_b.v, adj_ret.v);
}

template<typename Type>
CUDA_CALLABLE inline void adj_spatial_cross_dual(const spatial_vector_t<Type>& a,  const spatial_vector_t<Type>& b, spatial_vector_t<Type>& adj_a, spatial_vector_t<Type>& adj_b, const spatial_vector_t<Type>& adj_ret)
{
    adj_cross(a.w, b.w, adj_a.w, adj_b.w, adj_ret.w);
    adj_cross(a.v, b.v, adj_a.v, adj_b.v, adj_ret.w);

    adj_cross(a.w, b.v, adj_a.w, adj_b.v, adj_ret.v);
}

template<typename Type>
CUDA_CALLABLE inline void adj_spatial_top(const spatial_vector_t<Type>& a, spatial_vector_t<Type>& adj_a, const vec<3,Type>& adj_ret)
{
    adj_a.w += adj_ret;
}

template<typename Type>
CUDA_CALLABLE inline void adj_spatial_bottom(const spatial_vector_t<Type>& a, spatial_vector_t<Type>& adj_a, const vec<3,Type>& adj_ret)
{
    adj_a.v += adj_ret;
}

template<typename Type>
inline CUDA_CALLABLE spatial_vector_t<Type> atomic_add(spatial_vector_t<Type>* addr, const spatial_vector_t<Type>& value) 
{   
    vec<3,Type> w = atomic_add(&(addr->w), value.w);
    vec<3,Type> v = atomic_add(&(addr->v), value.v);

    return spatial_vector_t<Type>(w, v);
}


//---------------------------------------------------------------------------------
// Represents a rigid body transform_t<Type>ation

template<typename Type>
struct transform_t
{
    vec<3,Type> p;
    quaternion<Type> q;

    CUDA_CALLABLE inline transform_t(vec<3,Type> p=vec<3,Type>(), quaternion<Type> q=quaternion<Type>()) : p(p), q(q) {}
    CUDA_CALLABLE inline transform_t(Type)  {}  // helps uniform initialization

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
CUDA_CALLABLE inline vec<3,Type> transform_get_translation(const transform_t<Type>& t)
{
    return t.p;
}

template<typename Type>
CUDA_CALLABLE inline quaternion<Type> transform_get_rotation(const transform_t<Type>& t)
{
    return t.q;
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
    quaternion<Type> q_inv = quat_inverse(t.q);
    return transform_t<Type>(-quat_rotate(q_inv, t.p), q_inv);
}

    
template<typename Type>
CUDA_CALLABLE inline vec<3,Type> transform_vector(const transform_t<Type>& t, const vec<3,Type>& x)
{
    return quat_rotate(t.q, x);
}

template<typename Type>
CUDA_CALLABLE inline vec<3,Type> transform_point(const transform_t<Type>& t, const vec<3,Type>& x)
{
    return t.p + quat_rotate(t.q, x);
}
/*
// Frank & Park definition 3.20, pg 100
CUDA_CALLABLE inline spatial_vector_t<Type> transform_t<Type>_twist(const transform_t<Type>& t, const spatial_vector_t<Type>& x)
{
    vec<3,Type> w = quat_rotate(t.q, x.w);
    vec<3,Type> v = quat_rotate(t.q, x.v) + cross(t.p, w);

    return spatial_vector_t<Type>(w, v);
}

CUDA_CALLABLE inline spatial_vector_t<Type> transform_t<Type>_wrench(const transform_t<Type>& t, const spatial_vector_t<Type>& x)
{
    vec<3,Type> v = quat_rotate(t.q, x.v);
    vec<3,Type> w = quat_rotate(t.q, x.w) + cross(t.p, v);

    return spatial_vector_t<Type>(w, v);
}
*/

// not totally sure why you'd want to do this seeing as adding/subtracting two rotation
// quaternions doesn't seem to do anything meaningful
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

// also not sure why you'd want to do this seeing as the quaternion would end up unnormalized
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
inline CUDA_CALLABLE Type index(const transform_t<Type>& t, int i)
{
    return t[i];
}

template<typename Type>
inline void CUDA_CALLABLE adj_index(const transform_t<Type>& t, int i, transform_t<Type>& adj_t, int& adj_i, Type adj_ret)
{
    adj_t[i] += adj_ret;
}


// adjoint methods
template<typename Type>
CUDA_CALLABLE inline void adj_add(const transform_t<Type>& a, const transform_t<Type>& b, transform_t<Type>& adj_a, transform_t<Type>& adj_b, const transform_t<Type>& adj_ret)
{
    adj_add(a.p, b.p, adj_a.p, adj_b.p, adj_ret.p);
    adj_add(a.q, b.q, adj_a.q, adj_b.q, adj_ret.q);
}

template<typename Type>
CUDA_CALLABLE inline void adj_sub(const transform_t<Type>& a, const transform_t<Type>& b, transform_t<Type>& adj_a, transform_t<Type>& adj_b, const transform_t<Type>& adj_ret)
{
    adj_sub(a.p, b.p, adj_a.p, adj_b.p, adj_ret.p);
    adj_sub(a.q, b.q, adj_a.q, adj_b.q, adj_ret.q);
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
    vec<3,Type> p = atomic_add(&addr->p, value.p);
    quaternion<Type> q = atomic_add(&addr->q, value.q);

    return transform_t<Type>(p, q);
}

template<typename Type>
CUDA_CALLABLE inline void adj_transform_t(const vec<3,Type>& p, const quaternion<Type>& q, vec<3,Type>& adj_p, quaternion<Type>& adj_q, const transform_t<Type>& adj_ret)
{
    adj_p += adj_ret.p;
    adj_q += adj_ret.q;
}


template<typename Type>
CUDA_CALLABLE inline void adj_transform_get_translation(const transform_t<Type>& t, transform_t<Type>& adj_t, const vec<3,Type>& adj_ret)
{
    adj_t.p += adj_ret;
}

template<typename Type>
CUDA_CALLABLE inline void adj_transform_get_rotation(const transform_t<Type>& t, transform_t<Type>& adj_t, const quaternion<Type>& adj_ret)
{
    adj_t.q += adj_ret;
}

template<typename Type>
CUDA_CALLABLE inline void adj_transform_inverse(const transform_t<Type>& t, transform_t<Type>& adj_t, const transform_t<Type>& adj_ret)
{

    // forward
    quaternion<Type> q_inv = quat_inverse(t.q); 
    vec<3,Type> p = quat_rotate(q_inv, t.p);
    vec<3,Type> np = -p;
    // transform_t<Type> t = transform_t<Type>(np, q_inv)

    // backward
    quaternion<Type> adj_q_inv(0.0f);
    quaternion<Type> adj_q(0.0f);
    vec<3,Type> adj_p(0.0f);
    vec<3,Type> adj_np(0.0f);

    adj_transform_t(np, q_inv, adj_np, adj_q_inv, adj_ret);
    adj_p = -adj_np;
    adj_quat_rotate(q_inv, t.p, adj_q_inv, adj_t.p, adj_p);
    adj_quat_inverse(t.q, adj_t.q, adj_q_inv);
    
}

template<typename Type>
CUDA_CALLABLE inline void adj_transform_vector(const transform_t<Type>& t, const vec<3,Type>& x, transform_t<Type>& adj_t, vec<3,Type>& adj_x, const vec<3,Type>& adj_ret)
{
    adj_quat_rotate(t.q, x, adj_t.q, adj_x, adj_ret);
}

template<typename Type>
CUDA_CALLABLE inline void adj_transform_point(const transform_t<Type>& t, const vec<3,Type>& x, transform_t<Type>& adj_t, vec<3,Type>& adj_x, const vec<3,Type>& adj_ret)
{
    adj_quat_rotate(t.q, x, adj_t.q, adj_x, adj_ret);
    adj_t.p += adj_ret;
}

/*
CUDA_CALLABLE inline void adj_transform_twist(const transform_t<Type>& a, const spatial_vector_t<Type>& s, transform_t<Type>& adj_a, spatial_vector_t<Type>& adj_s, const spatial_vector_t<Type>& adj_ret)
{
    printf("todo, %s, %d\n", __FILE__, __LINE__);

    // vec<3,Type> w = quat_rotate(t.q, x.w);
    // vec<3,Type> v = quat_rotate(t.q, x.v) + cross(t.p, w);

    // return spatial_vector_t<Type>(w, v);    
}

CUDA_CALLABLE inline void adj_transform_wrench(const transform_t<Type>& t, const spatial_vector_t<Type>& x, transform_t<Type>& adj_t, spatial_vector_t<Type>& adj_x, const spatial_vector_t<Type>& adj_ret)
{
    printf("todo, %s, %d\n", __FILE__, __LINE__);
    // vec<3,Type> v = quat_rotate(t.q, x.v);
    // vec<3,Type> w = quat_rotate(t.q, x.w) + cross(t.p, v);

    // return spatial_vector_t<Type>(w, v);
}
*/

/*
// should match model.py
#define JOINT_PRISMATIC 0
#define JOINT_REVOLUTE 1
#define JOINT_FIXED 2
#define JOINT_FREE 3


CUDA_CALLABLE inline transform_t<Type> spatial_jcalc(int type, Type* joint_q, vec<3,Type> axis, int start)
{
    if (type == JOINT_REVOLUTE)
    {
        Type q = joint_q[start];
        transform_t<Type> X_jc = transform_t<Type>(vec<3,Type>(), quat_from_axis_angle(axis, q));
        return X_jc;
    }
    else if (type == JOINT_PRISMATIC)
    {
        Type q = joint_q[start];
        transform_t<Type> X_jc = transform_t<Type>(axis*q, quat_identity());
        return X_jc;
    }
    else if (type == JOINT_FREE)
    {
        Type px = joint_q[start+0];
        Type py = joint_q[start+1];
        Type pz = joint_q[start+2];
        
        Type qx = joint_q[start+3];
        Type qy = joint_q[start+4];
        Type qz = joint_q[start+5];
        Type qw = joint_q[start+6];
        
        transform_t<Type> X_jc = transform_t<Type>(vec<3,Type>(px, py, pz), quaternion<Type>(qx, qy, qz, qw));
        return X_jc;
    }

    // JOINT_FIXED
    return transform_t<Type>(vec<3,Type>(), quat_identity());
}

CUDA_CALLABLE inline void adj_spatial_jcalc(int type, Type* q, vec<3,Type> axis, int start, int& adj_type, Type* adj_q, vec<3,Type>& adj_axis, int& adj_start, const transform_t<Type>& adj_ret)
{
    if (type == JOINT_REVOLUTE)
    {
        adj_quat_from_axis_angle(axis, q[start], adj_axis, adj_q[start], adj_ret.q);
    }
    else if (type == JOINT_PRISMATIC)
    {
        adj_mul(axis, q[start], adj_axis, adj_q[start], adj_ret.p);
    }
    else if (type == JOINT_FREE)
    {
        adj_q[start+0] += adj_ret.p[0];
        adj_q[start+1] += adj_ret.p[1];
        adj_q[start+2] += adj_ret.p[2];
        
        adj_q[start+3] += adj_ret.q[0];
        adj_q[start+4] += adj_ret.q[1];
        adj_q[start+5] += adj_ret.q[2];
        adj_q[start+6] += adj_ret.q.w;
    }
}
*/

template<typename Type>
struct spatial_matrix_t
{
    Type data[6][6];

    CUDA_CALLABLE inline spatial_matrix_t(Type f=0.0f)
    {
        for (unsigned i=0; i < 6; ++i)
            for (unsigned j=0; j < 6; ++j)
                data[i][j] = f;
    }

    CUDA_CALLABLE inline spatial_matrix_t(
        Type a00, Type a01, Type a02, Type a03, Type a04, Type a05,
        Type a10, Type a11, Type a12, Type a13, Type a14, Type a15,
        Type a20, Type a21, Type a22, Type a23, Type a24, Type a25,
        Type a30, Type a31, Type a32, Type a33, Type a34, Type a35,
        Type a40, Type a41, Type a42, Type a43, Type a44, Type a45,
        Type a50, Type a51, Type a52, Type a53, Type a54, Type a55)
    {
        data[0][0] = a00;
        data[0][1] = a01;
        data[0][2] = a02;
        data[0][3] = a03;
        data[0][4] = a04;
        data[0][5] = a05;

        data[1][0] = a10;
        data[1][1] = a11;
        data[1][2] = a12;
        data[1][3] = a13;
        data[1][4] = a14;
        data[1][5] = a15;

        data[2][0] = a20;
        data[2][1] = a21;
        data[2][2] = a22;
        data[2][3] = a23;
        data[2][4] = a24;
        data[2][5] = a25;

        data[3][0] = a30;
        data[3][1] = a31;
        data[3][2] = a32;
        data[3][3] = a33;
        data[3][4] = a34;
        data[3][5] = a35;

        data[4][0] = a40;
        data[4][1] = a41;
        data[4][2] = a42;
        data[4][3] = a43;
        data[4][4] = a44;
        data[4][5] = a45;

        data[5][0] = a50;
        data[5][1] = a51;
        data[5][2] = a52;
        data[5][3] = a53;
        data[5][4] = a54;
        data[5][5] = a55;
    }

};


template<typename Type>
inline CUDA_CALLABLE bool operator==(const spatial_matrix_t<Type>& a, const spatial_matrix_t<Type>& b)
{
    for (int i=0; i < 6; ++i)
        for (int j=0; j < 6; ++j)
            if (a.data[i][j] != b.data[i][j])
                return false;

    return true;
}

template<typename Type>
inline bool CUDA_CALLABLE isfinite(const spatial_matrix_t<Type>& m)
{
    for (int i=0; i < 6; ++i)
        for (int j=0; j < 6; ++j)
            if (!::isfinite(m.data[i][j]))
                return false;
    return true;
}

template<typename Type>
inline CUDA_CALLABLE Type index(const spatial_matrix_t<Type>& m, int row, int col)
{
#if FP_CHECK
    if (row < 0 || row > 5)
    {
        printf("spatial_matrix_t<Type> row index %d out of bounds at %s %d\n", row, __FILE__, __LINE__);
        assert(0);
    }
    if (col < 0 || col > 5)
    {
        printf("spatial_matrix_t<Type> col index %d out of bounds at %s %d\n", col, __FILE__, __LINE__);
        assert(0);
    }
#endif
    return m.data[row][col];
}


template<typename Type>
inline CUDA_CALLABLE spatial_matrix_t<Type> add(const spatial_matrix_t<Type>& a, const spatial_matrix_t<Type>& b)
{
    spatial_matrix_t<Type> out;

    for (int i=0; i < 6; ++i)
        for (int j=0; j < 6; ++j)
            out.data[i][j] = a.data[i][j] + b.data[i][j];

    return out;
}

template<typename Type>
inline CUDA_CALLABLE spatial_matrix_t<Type> sub(const spatial_matrix_t<Type>& a, const spatial_matrix_t<Type>& b)
{
    spatial_matrix_t<Type> out;

    for (int i=0; i < 6; ++i)
        for (int j=0; j < 6; ++j)
            out.data[i][j] = a.data[i][j] - b.data[i][j];

    return out;
}

template<typename Type>
inline CUDA_CALLABLE spatial_matrix_t<Type> mul(const spatial_matrix_t<Type>& a, Type b)
{
    spatial_matrix_t<Type> out;

    for (int i=0; i < 6; ++i)
    {
        for (int j=0; j < 6; ++j)
        {
            out.data[i][j] += a.data[i][j]*b;
        }
    }
    return out;
}

template<typename Type>
inline CUDA_CALLABLE spatial_matrix_t<Type> mul(Type b, const spatial_matrix_t<Type>& a)
{
    return mul(a,b);
}

template<typename Type>
inline CUDA_CALLABLE spatial_vector_t<Type> mul(const spatial_matrix_t<Type>& a, const spatial_vector_t<Type>& b)
{
    spatial_vector_t<Type> out;

    for (int i=0; i < 6; ++i)
        for (int j=0; j < 6; ++j)
            out[i] += a.data[i][j]*b[j];

    return out;
}


template<typename Type>
CUDA_CALLABLE inline spatial_vector_t<Type> operator*(const spatial_vector_t<Type>& a, Type s)
{
    return mul(a, s);
}

template<typename Type>
CUDA_CALLABLE inline spatial_vector_t<Type> operator*(Type s, const spatial_vector_t<Type>& a)
{
    return mul(a, s);
}

template<typename Type>
inline CUDA_CALLABLE spatial_matrix_t<Type> mul(const spatial_matrix_t<Type>& a, const spatial_matrix_t<Type>& b)
{
    spatial_matrix_t<Type> out;

    for (int i=0; i < 6; ++i)
    {
        for (int j=0; j < 6; ++j)
        {
            for (int k=0; k < 6; ++k)
            {
                out.data[i][j] += a.data[i][k]*b.data[k][j];
            }
        }
    }
    return out;
}

template<typename Type>
CUDA_CALLABLE inline spatial_matrix_t<Type> operator*(const spatial_matrix_t<Type>& a, Type s)
{
    return mul(a, s);
}

template<typename Type>
CUDA_CALLABLE inline spatial_matrix_t<Type> operator*(Type s, const spatial_matrix_t<Type>& a)
{
    return mul(a, s);
}

template<typename Type>
inline CUDA_CALLABLE Type tensordot(const spatial_matrix_t<Type>& a, const spatial_matrix_t<Type>& b)
{
    // corresponds to `np.tensordot()` with all axes being contracted
    return
          a.data[0][0] * b.data[0][0] + a.data[0][1] * b.data[0][1] + a.data[0][2] * b.data[0][2] + a.data[0][3] * b.data[0][3] + a.data[0][4] * b.data[0][4] + a.data[0][5] * b.data[0][5]
        + a.data[1][0] * b.data[1][0] + a.data[1][1] * b.data[1][1] + a.data[1][2] * b.data[1][2] + a.data[1][3] * b.data[1][3] + a.data[1][4] * b.data[1][4] + a.data[1][5] * b.data[1][5]
        + a.data[2][0] * b.data[2][0] + a.data[2][1] * b.data[2][1] + a.data[2][2] * b.data[2][2] + a.data[2][3] * b.data[2][3] + a.data[2][4] * b.data[2][4] + a.data[2][5] * b.data[2][5]
        + a.data[3][0] * b.data[3][0] + a.data[3][1] * b.data[3][1] + a.data[3][2] * b.data[3][2] + a.data[3][3] * b.data[3][3] + a.data[3][4] * b.data[3][4] + a.data[3][5] * b.data[3][5]
        + a.data[4][0] * b.data[4][0] + a.data[4][1] * b.data[4][1] + a.data[4][2] * b.data[4][2] + a.data[4][3] * b.data[4][3] + a.data[4][4] * b.data[4][4] + a.data[4][5] * b.data[4][5]
        + a.data[5][0] * b.data[5][0] + a.data[5][1] * b.data[5][1] + a.data[5][2] * b.data[5][2] + a.data[5][3] * b.data[5][3] + a.data[5][4] * b.data[5][4] + a.data[5][5] * b.data[5][5];
}

template<typename Type>
inline CUDA_CALLABLE spatial_matrix_t<Type> transpose(const spatial_matrix_t<Type>& a)
{
    spatial_matrix_t<Type> out;

    for (int i=0; i < 6; i++)
        for (int j=0; j < 6; j++)
            out.data[i][j] = a.data[j][i];

    return out;
}

template<typename Type>
inline CUDA_CALLABLE spatial_matrix_t<Type> outer(const spatial_vector_t<Type>& a, const spatial_vector_t<Type>& b)
{
    spatial_matrix_t<Type> out;

    for (int i=0; i < 6; i++)
        for (int j=0; j < 6; j++)
            out.data[i][j] = a[i]*b[j];

    return out;
}

template<typename Type>
CUDA_CALLABLE void print(transform_t<Type> t);
template<typename Type>
CUDA_CALLABLE void print(spatial_matrix_t<Type> m);

template<typename Type>
CUDA_CALLABLE inline spatial_matrix_t<Type> lerp(const spatial_matrix_t<Type>& a, const spatial_matrix_t<Type>& b, Type t)
{
    return a*(Type(1)-t) + b*t;
}

template<typename Type>
CUDA_CALLABLE inline void adj_lerp(const spatial_matrix_t<Type>& a, const spatial_matrix_t<Type>& b, Type t, spatial_matrix_t<Type>& adj_a, spatial_matrix_t<Type>& adj_b, Type& adj_t, const spatial_matrix_t<Type>& adj_ret)
{
    adj_a += adj_ret*(Type(1)-t);
    adj_b += adj_ret*t;
    adj_t += tensordot(b, adj_ret) - tensordot(a, adj_ret);
}

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
CUDA_CALLABLE inline spatial_vector_t<Type> lerp(const spatial_vector_t<Type>& a, const spatial_vector_t<Type>& b, Type t)
{
    return a*(Type(1)-t) + b*t;
}

template<typename Type>
CUDA_CALLABLE inline void adj_lerp(const spatial_vector_t<Type>& a, const spatial_vector_t<Type>& b, Type t, spatial_vector_t<Type>& adj_a, spatial_vector_t<Type>& adj_b, Type& adj_t, const spatial_vector_t<Type>& adj_ret)
{
    adj_a += adj_ret*(Type(1)-t);
    adj_b += adj_ret*t;
    adj_t += tensordot(b, adj_ret) - tensordot(a, adj_ret);
}

template<typename Type>
CUDA_CALLABLE inline void adj_spatial_matrix_t(
    Type a00, Type a01, Type a02, Type a03, Type a04, Type a05,
    Type a10, Type a11, Type a12, Type a13, Type a14, Type a15,
    Type a20, Type a21, Type a22, Type a23, Type a24, Type a25,
    Type a30, Type a31, Type a32, Type a33, Type a34, Type a35,
    Type a40, Type a41, Type a42, Type a43, Type a44, Type a45,
    Type a50, Type a51, Type a52, Type a53, Type a54, Type a55,
    Type &adj_a00, Type &adj_a01, Type &adj_a02, Type &adj_a03, Type &adj_a04, Type &adj_a05,
    Type &adj_a10, Type &adj_a11, Type &adj_a12, Type &adj_a13, Type &adj_a14, Type &adj_a15,
    Type &adj_a20, Type &adj_a21, Type &adj_a22, Type &adj_a23, Type &adj_a24, Type &adj_a25,
    Type &adj_a30, Type &adj_a31, Type &adj_a32, Type &adj_a33, Type &adj_a34, Type &adj_a35,
    Type &adj_a40, Type &adj_a41, Type &adj_a42, Type &adj_a43, Type &adj_a44, Type &adj_a45,
    Type &adj_a50, Type &adj_a51, Type &adj_a52, Type &adj_a53, Type &adj_a54, Type &adj_a55,
    const spatial_matrix_t<Type>& adj_ret)
{
    adj_a00 += adj_ret.data[0][0];
    adj_a01 += adj_ret.data[0][1];
    adj_a02 += adj_ret.data[0][2];
    adj_a03 += adj_ret.data[0][3];
    adj_a04 += adj_ret.data[0][4];
    adj_a05 += adj_ret.data[0][5];
    adj_a10 += adj_ret.data[1][0];
    adj_a11 += adj_ret.data[1][1];
    adj_a12 += adj_ret.data[1][2];
    adj_a13 += adj_ret.data[1][3];
    adj_a14 += adj_ret.data[1][4];
    adj_a15 += adj_ret.data[1][5];
    adj_a20 += adj_ret.data[2][0];
    adj_a21 += adj_ret.data[2][1];
    adj_a22 += adj_ret.data[2][2];
    adj_a23 += adj_ret.data[2][3];
    adj_a24 += adj_ret.data[2][4];
    adj_a25 += adj_ret.data[2][5];
    adj_a30 += adj_ret.data[3][0];
    adj_a31 += adj_ret.data[3][1];
    adj_a32 += adj_ret.data[3][2];
    adj_a33 += adj_ret.data[3][3];
    adj_a34 += adj_ret.data[3][4];
    adj_a35 += adj_ret.data[3][5];
    adj_a40 += adj_ret.data[4][0];
    adj_a41 += adj_ret.data[4][1];
    adj_a42 += adj_ret.data[4][2];
    adj_a43 += adj_ret.data[4][3];
    adj_a44 += adj_ret.data[4][4];
    adj_a45 += adj_ret.data[4][5];
    adj_a50 += adj_ret.data[5][0];
    adj_a51 += adj_ret.data[5][1];
    adj_a52 += adj_ret.data[5][2];
    adj_a53 += adj_ret.data[5][3];
    adj_a54 += adj_ret.data[5][4];
    adj_a55 += adj_ret.data[5][5];
}

template<typename Type>
inline CUDA_CALLABLE void adj_outer(const spatial_vector_t<Type>& a, const spatial_vector_t<Type>& b, spatial_vector_t<Type>& adj_a, spatial_vector_t<Type>& adj_b, const spatial_matrix_t<Type>& adj_ret)
{
  adj_a += mul(adj_ret, b);
  adj_b += mul(transpose(adj_ret), a);
}

template<typename Type>
inline CUDA_CALLABLE spatial_matrix_t<Type> spatial_adjoint(const mat<3,3,Type>& R, const mat<3,3,Type>& S)
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
inline CUDA_CALLABLE void adj_spatial_adjoint(const mat<3,3,Type>& R, const mat<3,3,Type>& S, mat<3,3,Type>& adj_R, mat<3,3,Type>& adj_S, const spatial_matrix_t<Type>& adj_ret)
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

/*
// computes adj_t^-T*I*adj_t^-1 (tensor change of coordinates), Frank & Park, section 8.2.3, pg 290
inline CUDA_CALLABLE spatial_matrix_t<Type> transform_t<Type>_inertia(const transform_t<Type>& t, const spatial_matrix_t<Type>& I)
{
    transform_t<Type> t_inv = transform_t<Type>_inverse(t);

    vec<3,Type> r1 = quat_rotate(t_inv.q, vec<3,Type>(1.0, 0.0, 0.0));
    vec<3,Type> r2 = quat_rotate(t_inv.q, vec<3,Type>(0.0, 1.0, 0.0));
    vec<3,Type> r3 = quat_rotate(t_inv.q, vec<3,Type>(0.0, 0.0, 1.0));

    mat<3,3,Type> R(r1, r2, r3);    
    mat<3,3,Type> S = mul(skew(t_inv.p), R);

    spatial_matrix_t<Type> T = spatial_adjoint(R, S);

    // first quadratic form, for derivation of the adjoint see https://people.maths.ox.ac.uk/gilesm/files/AD2008.pwp, section 2.3.2
    return mul(mul(transpose(T), I), T);
}
*/



template<typename Type>
inline CUDA_CALLABLE void adj_add(const spatial_matrix_t<Type>& a, const spatial_matrix_t<Type>& b, spatial_matrix_t<Type>& adj_a, spatial_matrix_t<Type>& adj_b, const spatial_matrix_t<Type>& adj_ret)
{
    adj_a += adj_ret;
    adj_b += adj_ret;
}

template<typename Type>
inline CUDA_CALLABLE void adj_sub(const spatial_matrix_t<Type>& a, const spatial_matrix_t<Type>& b, spatial_matrix_t<Type>& adj_a, spatial_matrix_t<Type>& adj_b, const spatial_matrix_t<Type>& adj_ret)
{
    adj_a += adj_ret;
    adj_b -= adj_ret;
}


template<typename Type>
inline CUDA_CALLABLE void adj_mul(const spatial_matrix_t<Type>& a, const spatial_vector_t<Type>& b, spatial_matrix_t<Type>& adj_a, spatial_vector_t<Type>& adj_b, const spatial_vector_t<Type>& adj_ret)
{
    adj_a += outer(adj_ret, b);
    adj_b += mul(transpose(a), adj_ret);
}

template<typename Type>
inline CUDA_CALLABLE void adj_mul(const spatial_matrix_t<Type>& a, const spatial_matrix_t<Type>& b, spatial_matrix_t<Type>& adj_a, spatial_matrix_t<Type>& adj_b, const spatial_matrix_t<Type>& adj_ret)
{
    adj_a += mul(adj_ret, transpose(b));
    adj_b += mul(transpose(a), adj_ret);
}

template<typename Type>
inline CUDA_CALLABLE void adj_mul(const spatial_matrix_t<Type>& a, Type b, spatial_matrix_t<Type>& adj_a, Type& adj_b, const spatial_matrix_t<Type>& adj_ret)
{
    for (unsigned i=0; i < 6; ++i)
    {
        for (unsigned j=0; j < 6; ++j)
        {
            adj_a.data[i][j] += b*adj_ret.data[i][j];
            adj_b += a.data[i][j]*adj_ret.data[i][j];
        }
    }
}

template<typename Type>
inline CUDA_CALLABLE void adj_mul( Type b, const spatial_matrix_t<Type>& a, Type& adj_b, spatial_matrix_t<Type>& adj_a, const spatial_matrix_t<Type>& adj_ret)
{
    adj_mul(a, b, adj_a, adj_b, adj_ret);
}

template<typename Type>
inline CUDA_CALLABLE void adj_transpose(const spatial_matrix_t<Type>& a, spatial_matrix_t<Type>& adj_a, const spatial_matrix_t<Type>& adj_ret)
{
    adj_a += transpose(adj_ret);
}



template<typename Type>
inline CUDA_CALLABLE void adj_transform_inertia(
    const transform_t<Type>& xform, const spatial_matrix_t<Type>& I,
    const transform_t<Type>& adj_xform, const spatial_matrix_t<Type>& adj_I,
    spatial_matrix_t<Type>& adj_ret)
{
    //printf("todo, %s, %d\n", __FILE__, __LINE__);
}


template<typename Type>
inline void CUDA_CALLABLE adj_index(const spatial_matrix_t<Type>& m, int row, int col, spatial_matrix_t<Type>& adj_m, int& adj_row, int& adj_col, Type adj_ret)
{
#if FP_CHECK
    if (row < 0 || row > 5)
    {
        printf("spatial_matrix_t<Type> row index %d out of bounds at %s %d\n", row, __FILE__, __LINE__);
        assert(0);
    }
    if (col < 0 || col > 5)
    {
        printf("spatial_matrix_t<Type> col index %d out of bounds at %s %d\n", col, __FILE__, __LINE__);
        assert(0);
    }
#endif
    adj_m.data[row][col] += adj_ret;
}

template<typename Type>
inline CUDA_CALLABLE spatial_matrix_t<Type> atomic_add(spatial_matrix_t<Type>* addr, const spatial_matrix_t<Type>& value) 
{
    spatial_matrix_t<Type> m;

    for (int i=0; i < 6; ++i)
    {
        for (int j=0; j < 6; ++j)
        {
            m.data[i][j] = atomic_add(&addr->data[i][j], value.data[i][j]);
        }
    }   

    return m;
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

using transformh = transform_t<half>;
using transform = transform_t<float>;
using transformf = transform_t<float>;
using transformd = transform_t<double>;

using spatial_vectorh = spatial_vector_t<half>;
using spatial_vector = spatial_vector_t<float>;
using spatial_vectorf = spatial_vector_t<float>;
using spatial_vectord = spatial_vector_t<double>;

using spatial_matrixh = spatial_matrix_t<half>;
using spatial_matrix = spatial_matrix_t<float>;
using spatial_matrixf = spatial_matrix_t<float>;
using spatial_matrixd = spatial_matrix_t<double>;



CUDA_CALLABLE inline transformh transformh_identity()
{
    return transformh(vec3h(), quath_identity());
}

CUDA_CALLABLE inline void adj_transformh_identity(const transformh& adj_ret)
{
    // nop
}

CUDA_CALLABLE inline transform transform_identity()
{
    return transform(vec3(), quat_identity());
}

CUDA_CALLABLE inline void adj_transform_identity(const transform& adj_ret)
{
    // nop
}

CUDA_CALLABLE inline transformd transformd_identity()
{
    return transformd(vec3d(), quatd_identity());
}

CUDA_CALLABLE inline void adj_transformd_identity(const transformd& adj_ret)
{
    // nop
}

CUDA_CALLABLE inline void adj_transform(const vec3& p, const quat& q, vec3& adj_p, quat& adj_q, const transform& adj_ret)
{
    adj_transform_t(p, q, adj_p, adj_q, adj_ret);
}

CUDA_CALLABLE inline void adj_spatial_vector(
    float a, float b, float c, 
    float d, float e, float f, 
    float& adj_a, float& adj_b, float& adj_c,
    float& adj_d, float& adj_e,float& adj_f, 
    const spatial_vector& adj_ret)
{
    adj_spatial_vector_t(
        a, b, c, 
        d, e, f, 
        adj_a, adj_b, adj_c,
        adj_d, adj_e, adj_f, 
        adj_ret
    );
}

CUDA_CALLABLE inline void adj_spatial_vector(const vec3& w, const vec3& v, vec3& adj_w, vec3& adj_v, const spatial_vector& adj_ret)
{
    adj_spatial_vector_t(w, v, adj_w, adj_v, adj_ret);
}

CUDA_CALLABLE inline void adj_spatial_vector(float a, float& adj_a, const spatial_vector& adj_ret)
{
    adj_spatial_vector_t(a, adj_a, adj_ret);
}

CUDA_CALLABLE inline void adj_spatial_matrix(
    float a00, float a01, float a02, float a03, float a04, float a05,
    float a10, float a11, float a12, float a13, float a14, float a15,
    float a20, float a21, float a22, float a23, float a24, float a25,
    float a30, float a31, float a32, float a33, float a34, float a35,
    float a40, float a41, float a42, float a43, float a44, float a45,
    float a50, float a51, float a52, float a53, float a54, float a55,
    float adj_a00, float adj_a01, float adj_a02, float adj_a03, float adj_a04, float adj_a05,
    float adj_a10, float adj_a11, float adj_a12, float adj_a13, float adj_a14, float adj_a15,
    float adj_a20, float adj_a21, float adj_a22, float adj_a23, float adj_a24, float adj_a25,
    float adj_a30, float adj_a31, float adj_a32, float adj_a33, float adj_a34, float adj_a35,
    float adj_a40, float adj_a41, float adj_a42, float adj_a43, float adj_a44, float adj_a45,
    float adj_a50, float adj_a51, float adj_a52, float adj_a53, float adj_a54, float adj_a55,
    const spatial_matrix& adj_ret)
{
    adj_spatial_matrix_t(
        a00, a01, a02, a03, a04, a05,
        a10, a11, a12, a13, a14, a15,
        a20, a21, a22, a23, a24, a25,
        a30, a31, a32, a33, a34, a35,
        a40, a41, a42, a43, a44, a45,
        a50, a51, a52, a53, a54, a55,
        adj_a00, adj_a01, adj_a02, adj_a03, adj_a04, adj_a05,
        adj_a10, adj_a11, adj_a12, adj_a13, adj_a14, adj_a15,
        adj_a20, adj_a21, adj_a22, adj_a23, adj_a24, adj_a25,
        adj_a30, adj_a31, adj_a32, adj_a33, adj_a34, adj_a35,
        adj_a40, adj_a41, adj_a42, adj_a43, adj_a44, adj_a45,
        adj_a50, adj_a51, adj_a52, adj_a53, adj_a54, adj_a55,
        adj_ret
    );
}


 } // namespace wp