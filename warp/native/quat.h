/** Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include "mat33.h"

namespace wp
{

struct quat
{
    // imaginary part
    float x;
    float y;
    float z;
    
    // real part
    float w;

    inline CUDA_CALLABLE quat(float x=0.0f, float y=0.0f, float z=0.0f, float w=0.0) : x(x), y(y), z(z), w(w) {}    
    explicit inline CUDA_CALLABLE quat(const vec3& v, float w=0.0f) : x(v.x), y(v.y), z(v.z), w(w) {}

};

inline CUDA_CALLABLE bool operator==(const quat& a, const quat& b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

inline bool CUDA_CALLABLE isfinite(const quat& q)
{
    return ::isfinite(q.x) && ::isfinite(q.y) && ::isfinite(q.z) && ::isfinite(q.w);
}

inline CUDA_CALLABLE quat atomic_add(quat * addr, quat value) 
{
    float x = atomic_add(&(addr -> x), value.x);
    float y = atomic_add(&(addr -> y), value.y);
    float z = atomic_add(&(addr -> z), value.z);
    float w = atomic_add(&(addr -> w), value.w);

    return quat(x, y, z, w);
}


inline CUDA_CALLABLE void adj_quat(float x, float y, float z, float w, float& adj_x, float& adj_y, float& adj_z, float& adj_w, quat adj_ret)
{
    adj_x += adj_ret.x;
    adj_y += adj_ret.y;
    adj_z += adj_ret.z;
    adj_w += adj_ret.w;
}

inline CUDA_CALLABLE void adj_quat(const vec3& v, float w, vec3& adj_v, float& adj_w, quat adj_ret)
{
    adj_v.x += adj_ret.x;
    adj_v.y += adj_ret.y;
    adj_v.z += adj_ret.z;
    adj_w   += adj_ret.w;
}

// forward methods

inline CUDA_CALLABLE quat quat_from_axis_angle(const vec3& axis, float angle)
{
    float half = angle*0.5f;
    float w = cos(half);

    float sin_theta_over_two = sin(half);
    vec3 v = axis*sin_theta_over_two;

    return quat(v.x, v.y, v.z, w);
}

inline CUDA_CALLABLE quat quat_rpy(float roll, float pitch, float yaw)
{
    float cy = cos(yaw * 0.5);
    float sy = sin(yaw * 0.5);
    float cr = cos(roll * 0.5);
    float sr = sin(roll * 0.5);
    float cp = cos(pitch * 0.5);
    float sp = sin(pitch * 0.5);

    float w = (cy * cr * cp + sy * sr * sp);
    float x = (cy * sr * cp - sy * cr * sp);
    float y = (cy * cr * sp + sy * sr * cp);
    float z = (sy * cr * cp - cy * sr * sp);

    return quat(x, y, z, w);
}


inline CUDA_CALLABLE quat quat_identity()
{
    return quat(0.0f, 0.0f, 0.0f, 1.0f);
}

inline CUDA_CALLABLE quat quat_inverse(const quat& q)
{
    return quat(-q.x, -q.y, -q.z, q.w);
}


inline CUDA_CALLABLE float dot(const quat& a, const quat& b)
{
    return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
}

inline CUDA_CALLABLE float length(const quat& q)
{
    return sqrt(dot(q, q));
}

inline CUDA_CALLABLE quat normalize(const quat& q)
{
    float l = length(q);
    if (l > kEps)
    {
        float inv_l = 1.0f/l;

        return quat(q.x*inv_l, q.y*inv_l, q.z*inv_l, q.w*inv_l);
    }
    else
    {
        return quat(0.0f, 0.0f, 0.0f, 1.0f);
    }
}


inline CUDA_CALLABLE quat add(const quat& a, const quat& b)
{
    return quat(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w);
}

inline CUDA_CALLABLE quat sub(const quat& a, const quat& b)
{
    return quat(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w);}


inline CUDA_CALLABLE quat mul(const quat& a, const quat& b)
{
    return quat(a.w*b.x + b.w*a.x + a.y*b.z - b.y*a.z,
                a.w*b.y + b.w*a.y + a.z*b.x - b.z*a.x,
                a.w*b.z + b.w*a.z + a.x*b.y - b.x*a.y,
                a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z);
}

inline CUDA_CALLABLE quat mul(const quat& a, float s)
{
    return quat(a.x*s, a.y*s, a.z*s, a.w*s);    
}

inline CUDA_CALLABLE quat mul(float s, const quat& a)
{
    return mul(a, s);
}

inline CUDA_CALLABLE vec3 quat_rotate(const quat& q, const vec3& x)
{
    return x*(2.0f*q.w*q.w-1.0f) + cross(vec3(&q.x), x)*q.w*2.0f + vec3(&q.x)*dot(vec3(&q.x), x)*2.0f;
}

inline CUDA_CALLABLE vec3 quat_rotate_inv(const quat& q, const vec3& x)
{
    return x*(2.0f*q.w*q.w-1.0f) - cross(vec3(&q.x), x)*q.w*2.0f + vec3(&q.x)*dot(vec3(&q.x), x)*2.0f;
}

inline CUDA_CALLABLE mat33 quat_to_matrix(const quat& q)
{
    vec3 c1 = quat_rotate(q, vec3(1.0, 0.0, 0.0));
    vec3 c2 = quat_rotate(q, vec3(0.0, 1.0, 0.0));
    vec3 c3 = quat_rotate(q, vec3(0.0, 0.0, 1.0));

    return mat33(c1, c2, c3);
}

inline CUDA_CALLABLE quat quat_from_matrix(const mat33& m)
{
    const float tr = m.data[0][0] + m.data[1][1] + m.data[2][2];
    float x, y, z, w, h = 0.0f;

    if (tr >= 0.0f) {
        h = sqrt(tr + 1.0f);
        w = 0.5f * h;
        h = 0.5f / h;

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
            h = sqrt((m.data[0][0] - (m.data[1][1] + m.data[2][2])) + 1.0f);
            x = 0.5f * h;
            h = 0.5f / h;

            y = (m.data[0][1] + m.data[1][0]) * h;
            z = (m.data[2][0] + m.data[0][2]) * h;
            w = (m.data[2][1] - m.data[1][2]) * h;
        } else if (max_diag == 1) {
            h = sqrt((m.data[1][1] - (m.data[2][2] + m.data[0][0])) + 1.0f);
            y = 0.5f * h;
            h = 0.5f / h;

            z = (m.data[1][2] + m.data[2][1]) * h;
            x = (m.data[0][1] + m.data[1][0]) * h;
            w = (m.data[0][2] - m.data[2][0]) * h;
        } if (max_diag == 2) {
            h = sqrt((m.data[2][2] - (m.data[0][0] + m.data[1][1])) + 1.0f);
            z = 0.5f * h;
            h = 0.5f / h;

            x = (m.data[2][0] + m.data[0][2]) * h;
            y = (m.data[1][2] + m.data[2][1]) * h;
            w = (m.data[1][0] - m.data[0][1]) * h;
        }
    }

    return normalize(quat(x, y, z, w));
}

inline CUDA_CALLABLE float index(const quat& a, int idx)
{
#if FP_CHECK
    if (idx < 0 || idx > 3)
    {
        printf("quat index %d out of bounds at %s %d", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    return (&a.x)[idx];
}

inline CUDA_CALLABLE void adj_index(const quat& a, int idx, quat& adj_a, int & adj_idx, float & adj_ret)
{
#if FP_CHECK
    if (idx < 0 || idx > 3)
    {
        printf("quat index %d out of bounds at %s %d", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    (&adj_a.x)[idx] += adj_ret;
}


// backward methods
inline CUDA_CALLABLE void adj_quat_from_axis_angle(const vec3& axis, float angle, vec3& adj_axis, float& adj_angle, const quat& adj_ret)
{
    vec3 v = vec3(adj_ret.x, adj_ret.y, adj_ret.z);

    float s = sin(angle*0.5f);
    float c = cos(angle*0.5f);

    quat dqda = quat(axis.x*c, axis.y*c, axis.z*c, -s)*0.5f;

    adj_axis += v*s;
    adj_angle += dot(dqda, adj_ret);
}

inline CUDA_CALLABLE void adj_quat_rpy(float roll, float pitch, float yaw, float& adj_roll, float& adj_pitch, float& adj_yaw, const quat& adj_ret)
{
    // todo
}

inline CUDA_CALLABLE void adj_quat_identity(const quat& adj_ret)
{
    // nop
}

inline CUDA_CALLABLE void adj_dot(const quat& a, const quat& b, quat& adj_a, quat& adj_b, const float adj_ret)
{
    adj_a += b*adj_ret;
    adj_b += a*adj_ret;    
}

inline CUDA_CALLABLE void adj_length(const quat& a, quat& adj_a, const float adj_ret)
{
    adj_a += normalize(a)*adj_ret;
}

inline CUDA_CALLABLE void adj_normalize(const quat& q, quat& adj_q, const quat& adj_ret)
{
    float l = length(q);

    if (l > kEps)
    {
        float l_inv = 1.0f/l;

        adj_q += adj_ret*l_inv - q*(l_inv*l_inv*l_inv*dot(q, adj_ret));
    }
}

inline CUDA_CALLABLE void adj_quat_inverse(const quat& q, quat& adj_q, const quat& adj_ret)
{
    adj_q.x -= adj_ret.x;
    adj_q.y -= adj_ret.y;
    adj_q.z -= adj_ret.z;
    adj_q.w += adj_ret.w;
}

// inline void adj_normalize(const quat& a, quat& adj_a, const quat& adj_ret)
// {
//     float d = length(a);
    
//     if (d > kEps)
//     {
//         float invd = 1.0f/d;

//         quat ahat = normalize(a);

//         adj_a += (adj_ret - ahat*(dot(ahat, adj_ret))*invd);

//         //if (!isfinite(adj_a))
//         //    printf("%s:%d - adj_normalize((%f %f %f), (%f %f %f), (%f, %f, %f))\n", __FILE__, __LINE__, a.x, a.y, a.z, adj_a.x, adj_a.y, adj_a.z, adj_ret.x, adj_ret.y, adj_ret.z);
//     }
// }

inline CUDA_CALLABLE void adj_add(const quat& a, const quat& b, quat& adj_a, quat& adj_b, const quat& adj_ret)
{
    adj_a += adj_ret;
    adj_b += adj_ret;
}

inline CUDA_CALLABLE void adj_sub(const quat& a, const quat& b, quat& adj_a, quat& adj_b, const quat& adj_ret)
{
    adj_a += adj_ret;
    adj_b -= adj_ret;
}

inline CUDA_CALLABLE void adj_mul(const quat& a, const quat& b, quat& adj_a, quat& adj_b, const quat& adj_ret)
{
    // shorthand
    const quat& r = adj_ret;

    adj_a += quat(b.w*r.x - b.x*r.w + b.y*r.z - b.z*r.y,
                  b.w*r.y - b.y*r.w - b.x*r.z + b.z*r.x,
                  b.w*r.z + b.x*r.y - b.y*r.x - b.z*r.w,
                  b.w*r.w + b.x*r.x + b.y*r.y + b.z*r.z);

    adj_b += quat(a.w*r.x - a.x*r.w - a.y*r.z + a.z*r.y,
                  a.w*r.y - a.y*r.w + a.x*r.z - a.z*r.x,
                  a.w*r.z - a.x*r.y + a.y*r.x - a.z*r.w,
                  a.w*r.w + a.x*r.x + a.y*r.y + a.z*r.z);

}

inline CUDA_CALLABLE void adj_mul(const quat& a, float s, quat& adj_a, float& adj_s, const quat& adj_ret)
{
    adj_a += adj_ret*s;
    adj_s += dot(a, adj_ret);
}

inline CUDA_CALLABLE void adj_mul(float s, const quat& a, float& adj_s, quat& adj_a, const quat& adj_ret)
{
    adj_mul(a, s, adj_a, adj_s, adj_ret);
}

inline CUDA_CALLABLE void adj_quat_rotate(const quat& q, const vec3& p, quat& adj_q, vec3& adj_p, const vec3& adj_ret)
{
    const vec3& r = adj_ret;

    {
        float t2 = p.z*q.z*2.0f;
        float t3 = p.y*q.w*2.0f;
        float t4 = p.x*q.w*2.0f;
        float t5 = p.x*q.x*2.0f;
        float t6 = p.y*q.y*2.0f;
        float t7 = p.z*q.y*2.0f;
        float t8 = p.x*q.z*2.0f;
        float t9 = p.x*q.y*2.0f;
        float t10 = p.y*q.x*2.0f;
        adj_q.x += r.z*(t3+t8)+r.x*(t2+t6+p.x*q.x*4.0f)+r.y*(t9-p.z*q.w*2.0f);
        adj_q.y += r.y*(t2+t5+p.y*q.y*4.0f)+r.x*(t10+p.z*q.w*2.0f)-r.z*(t4-p.y*q.z*2.0f);
        adj_q.z += r.y*(t4+t7)+r.z*(t5+t6+p.z*q.z*4.0f)-r.x*(t3-p.z*q.x*2.0f);
        adj_q.w += r.x*(t7+p.x*q.w*4.0f-p.y*q.z*2.0f)+r.y*(t8+p.y*q.w*4.0f-p.z*q.x*2.0f)+r.z*(-t9+t10+p.z*q.w*4.0f);
    }

    {
        float t2 = q.w*q.w;
        float t3 = t2*2.0f;
        float t4 = q.w*q.z*2.0f;
        float t5 = q.x*q.y*2.0f;
        float t6 = q.w*q.y*2.0f;
        float t7 = q.w*q.x*2.0f;
        float t8 = q.y*q.z*2.0f;
        adj_p.x += r.y*(t4+t5)+r.x*(t3+(q.x*q.x)*2.0f-1.0f)-r.z*(t6-q.x*q.z*2.0f);
        adj_p.y += r.z*(t7+t8)-r.x*(t4-t5)+r.y*(t3+(q.y*q.y)*2.0f-1.0f);
        adj_p.z += -r.y*(t7-t8)+r.z*(t3+(q.z*q.z)*2.0f-1.0f)+r.x*(t6+q.x*q.z*2.0f);
    }
}

inline CUDA_CALLABLE void adj_quat_rotate_inv(const quat& q, const vec3& p, quat& adj_q, vec3& adj_p, const vec3& adj_ret)
{
    const vec3& r = adj_ret;

    {
        float t2 = p.z*q.w*2.0f;
        float t3 = p.z*q.z*2.0f;
        float t4 = p.y*q.w*2.0f;
        float t5 = p.x*q.w*2.0f;
        float t6 = p.x*q.x*2.0f;
        float t7 = p.y*q.y*2.0f;
        float t8 = p.y*q.z*2.0f;
        float t9 = p.z*q.x*2.0f;
        float t10 = p.x*q.y*2.0f;
        adj_q.x += r.y*(t2+t10)+r.x*(t3+t7+p.x*q.x*4.0f)-r.z*(t4-p.x*q.z*2.0f);
        adj_q.y += r.z*(t5+t8)+r.y*(t3+t6+p.y*q.y*4.0f)-r.x*(t2-p.y*q.x*2.0f);
        adj_q.z += r.x*(t4+t9)+r.z*(t6+t7+p.z*q.z*4.0f)-r.y*(t5-p.z*q.y*2.0f);
        adj_q.w += r.x*(t8+p.x*q.w*4.0f-p.z*q.y*2.0f)+r.y*(t9+p.y*q.w*4.0f-p.x*q.z*2.0f)+r.z*(t10-p.y*q.x*2.0f+p.z*q.w*4.0f);
    }

    {
        float t2 = q.w*q.w;
        float t3 = t2*2.0f;
        float t4 = q.w*q.z*2.0f;
        float t5 = q.w*q.y*2.0f;
        float t6 = q.x*q.z*2.0f;
        float t7 = q.w*q.x*2.0f;
        adj_p.x += r.z*(t5+t6)+r.x*(t3+(q.x*q.x)*2.0f-1.0f)-r.y*(t4-q.x*q.y*2.0f);
        adj_p.y += r.y*(t3+(q.y*q.y)*2.0f-1.0f)+r.x*(t4+q.x*q.y*2.0f)-r.z*(t7-q.y*q.z*2.0f);
        adj_p.z += -r.x*(t5-t6)+r.z*(t3+(q.z*q.z)*2.0f-1.0f)+r.y*(t7+q.y*q.z*2.0f);
    }
}

inline CUDA_CALLABLE void adj_quat_to_matrix(const quat& q, quat& adj_q, mat33& adj_ret)
{
    // we don't care about adjoint w.r.t. constant identity matrix
    vec3 t;

    adj_quat_rotate(q, vec3(1.0, 0.0, 0.0), adj_q, t, adj_ret.get_col(0));
    adj_quat_rotate(q, vec3(0.0, 1.0, 0.0), adj_q, t, adj_ret.get_col(1));
    adj_quat_rotate(q, vec3(0.0, 0.0, 1.0), adj_q, t, adj_ret.get_col(2));
}


inline CUDA_CALLABLE void adj_quat_from_matrix(const mat33& m, mat33& adj_m, const quat& adj_ret)
{
    // TODO
}

} // namespace wp