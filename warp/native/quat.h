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

inline CUDA_CALLABLE void quat_to_axis_angle(const quat& q, vec3& axis, float& angle)
{
    vec3 v = vec3(q.x, q.y, q.z);
    axis = q.w < 0.f ? -normalize(v) : normalize(v);
    angle = 2.f * atan2(length(v), fabs(q.w));
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

inline CUDA_CALLABLE float tensordot(const quat& a, const quat& b)
{
    // corresponds to `np.tensordot()` with all axes being contracted
    return dot(a, b);
}

inline CUDA_CALLABLE float length(const quat& q)
{
    return sqrt(dot(q, q));
}

inline CUDA_CALLABLE float length_sq(const quat& q)
{
    return dot(q, q);
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

inline CUDA_CALLABLE vec3 rotate_rodriguez(const vec3& r, const vec3& x)
{
    float angle = length(r);
    if (angle > kEps || angle < kEps)
    {
        vec3 axis = r / angle;
        return x * cos(angle) + cross(axis, x) * sin(angle) + axis * dot(axis, x) * (1.f - cos(angle));
    }
    else
    {
        return x;
    }
}

inline CUDA_CALLABLE quat quat_slerp(const quat& q0, const quat& q1, float t)
{
    vec3 axis;
    float angle;
    quat_to_axis_angle(mul(quat_inverse(q0), q1), axis, angle);
    return mul(q0, quat_from_axis_angle(axis, t * angle));
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

inline CUDA_CALLABLE void adj_quat_to_axis_angle(const quat& q, vec3& axis, float& angle, quat& adj_q, const vec3& adj_axis, const float& adj_angle)
{   
    float l = length(vec3(q.x, q.y, q.z));

    float ax_qx = 0.f;
    float ax_qy = 0.f;
    float ax_qz = 0.f;
    float ay_qx = 0.f;
    float ay_qy = 0.f;
    float ay_qz = 0.f;
    float az_qx = 0.f;
    float az_qy = 0.f;
    float az_qz = 0.f;

    float t_qx = 0.f;
    float t_qy = 0.f;
    float t_qz = 0.f;
    float t_qw = 0.f;

    float flip = q.w < 0.f ? -1.0 : 1.0;

    if (l > 0.f)
    {
        float l_sq = l*l;
        float l_inv = 1.f / l;
        float l_inv_sq = l_inv * l_inv;
        float l_inv_cu = l_inv_sq * l_inv;
        
        float C = flip * l_inv_cu;
        ax_qx = C * (q.y*q.y + q.z*q.z);
        ax_qy = -C * q.x*q.y;
        ax_qz = -C * q.x*q.z;
        ay_qx = -C * q.y*q.x;
        ay_qy = C * (q.x*q.x + q.z*q.z);
        ay_qz = -C * q.y*q.z;
        az_qx = -C * q.z*q.x;
        az_qy = -C * q.z*q.y;
        az_qz = C * (q.x*q.x + q.y*q.y);

        float D = 2.f * flip / (l_sq + q.w*q.w);
        t_qx = D * l_inv * q.x * q.w;
        t_qy = D * l_inv * q.y * q.w;
        t_qz = D * l_inv * q.z * q.w;
        t_qw = -D * l;
    }
    else
    {
        if (abs(q.w) > kEps)
        {
            float t_qx = 2.f / (sqrt(3.f) * abs(q.w));
            float t_qy = 2.f / (sqrt(3.f) * abs(q.w));
            float t_qz = 2.f / (sqrt(3.f) * abs(q.w));
        }
        // o/w we have a null quaternion which cannot backpropagate 
    }

    adj_q.x += ax_qx * adj_axis.x + ay_qx * adj_axis.y + az_qx * adj_axis.z + t_qx * adj_angle;
    adj_q.y += ax_qy * adj_axis.x + ay_qy * adj_axis.y + az_qy * adj_axis.z + t_qy * adj_angle;
    adj_q.z += ax_qz * adj_axis.x + ay_qz * adj_axis.y + az_qz * adj_axis.z + t_qz * adj_angle;
    adj_q.w += t_qw * adj_angle;
}

inline CUDA_CALLABLE void adj_quat_rpy(float roll, float pitch, float yaw, float& adj_roll, float& adj_pitch, float& adj_yaw, const quat& adj_ret)
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

    float dx_dr = 0.5 * w;
    float dx_dp = -0.5 * cy * sr * sp - 0.5 * sy * cr * cp;
    float dx_dy = -0.5 * y;

    float dy_dr = 0.5 * z;
    float dy_dp = 0.5 * cy * cr * cp - 0.5 * sy * sr * sp;
    float dy_dy = 0.5 * x;

    float dz_dr = -0.5 * y;
    float dz_dp = -0.5 * sy * cr * sp - 0.5 * cy * sr * cp;
    float dz_dy = 0.5 * w;

    float dw_dr = -0.5 * x;
    float dw_dp = -0.5 * cy * cr * sp + 0.5 * sy * sr * cp;
    float dw_dy = -0.5 * z;

    adj_roll += dot(quat(dx_dr, dy_dr, dz_dr, dw_dr), adj_ret);
    adj_pitch += dot(quat(dx_dp, dy_dp, dz_dp, dw_dp), adj_ret);
    adj_yaw += dot(quat(dx_dy, dy_dy, dz_dy, dw_dy), adj_ret);
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

inline CUDA_CALLABLE void tensordot(const quat& a, const quat& b, quat& adj_a, quat& adj_b, const float adj_ret)
{
    adj_dot(a, b, adj_a, adj_b, adj_ret);
}

inline CUDA_CALLABLE void adj_length(const quat& a, quat& adj_a, const float adj_ret)
{
    adj_a += normalize(a)*adj_ret;
}

inline CUDA_CALLABLE void adj_length_sq(const quat& a, quat& adj_a, const float adj_ret)
{
    adj_a += 2.0f*a*adj_ret;
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

inline CUDA_CALLABLE void adj_rotate_rodriguez(const vec3& r, const vec3& x, vec3& adj_r, vec3& adj_x, const vec3& adj_ret)
{
    float angle = length(r);
    float angle_squared = angle * angle;

    vec3 axis = r / angle;
    mat33 rotation_matrix = quat_to_matrix(quat_from_axis_angle(axis, angle));
    mat33 inverse_rotation_matrix = transpose(rotation_matrix);
    mat33 A = mul(mul(rotation_matrix, skew(x)), -1.0);

    if (!angle_squared)
    {
        adj_r += mul(transpose(A), adj_ret);
    }
    else{
        float inv_angle_squared = 1.f / angle_squared;
        mat33 B = mul(add(outer(r,r), mul(sub(inverse_rotation_matrix, diag(vec3(1.f))), skew(r))), inv_angle_squared);
        adj_r += mul(transpose(mul(A, B)), adj_ret);
    }
    
    // todo: add adj_x
}

inline CUDA_CALLABLE void adj_quat_slerp(const quat& q0, const quat& q1, float t, quat& adj_q0, quat& adj_q1, float& adj_t, const quat& adj_ret)
{
    vec3 axis;
    float angle;
    quat q0_inv = quat_inverse(q0);
    quat q_inc = mul(q0_inv, q1);
    quat_to_axis_angle(q_inc, axis, angle);
    angle = angle * 0.5;
    
    // adj_t
    adj_t += dot(mul(quat_slerp(q0, q1, t), quat(angle*axis.x, angle*axis.y, angle*axis.z, 0.f)), adj_ret);

    // adj_q0
    quat q_inc_x_q0;
    quat q_inc_y_q0;
    quat q_inc_z_q0;
    quat q_inc_w_q0;
    
    quat q_inc_x_q1;
    quat q_inc_y_q1;
    quat q_inc_z_q1;
    quat q_inc_w_q1;

    adj_mul(q0_inv, q1, q_inc_x_q0, q_inc_x_q1, quat(1.f, 0.f, 0.f, 0.f));
    adj_mul(q0_inv, q1, q_inc_y_q0, q_inc_y_q1, quat(0.f, 1.f, 0.f, 0.f));
    adj_mul(q0_inv, q1, q_inc_z_q0, q_inc_z_q1, quat(0.f, 0.f, 1.f, 0.f));
    adj_mul(q0_inv, q1, q_inc_w_q0, q_inc_w_q1, quat(0.f, 0.f, 0.f, 1.f));

    quat q_inc_q0_x = quat(-q_inc_x_q0.x, -q_inc_y_q0.x, -q_inc_z_q0.x, -q_inc_w_q0.x);
    quat q_inc_q0_y = quat(-q_inc_x_q0.y, -q_inc_y_q0.y, -q_inc_z_q0.y, -q_inc_w_q0.y);
    quat q_inc_q0_z = quat(-q_inc_x_q0.z, -q_inc_y_q0.z, -q_inc_z_q0.z, -q_inc_w_q0.z);
    quat q_inc_q0_w = quat(q_inc_x_q0.w, q_inc_y_q0.w, q_inc_z_q0.w, q_inc_w_q0.w);

    quat a_x_q_inc;
    quat a_y_q_inc;
    quat a_z_q_inc;
    quat t_q_inc;

    adj_quat_to_axis_angle(q_inc, axis, angle, a_x_q_inc, vec3(1.f, 0.f, 0.f), 0.f);
    adj_quat_to_axis_angle(q_inc, axis, angle, a_y_q_inc, vec3(0.f, 1.f, 0.f), 0.f);
    adj_quat_to_axis_angle(q_inc, axis, angle, a_z_q_inc, vec3(0.f, 0.f, 1.f), 0.f);
    adj_quat_to_axis_angle(q_inc, axis, angle, t_q_inc, vec3(0.f, 0.f, 0.f), 1.f);

    float a_x_q0_x = dot(a_x_q_inc, q_inc_q0_x);
    float a_x_q0_y = dot(a_x_q_inc, q_inc_q0_y);
    float a_x_q0_z = dot(a_x_q_inc, q_inc_q0_z);
    float a_x_q0_w = dot(a_x_q_inc, q_inc_q0_w);
    float a_y_q0_x = dot(a_y_q_inc, q_inc_q0_x);
    float a_y_q0_y = dot(a_y_q_inc, q_inc_q0_y);
    float a_y_q0_z = dot(a_y_q_inc, q_inc_q0_z);
    float a_y_q0_w = dot(a_y_q_inc, q_inc_q0_w);
    float a_z_q0_x = dot(a_z_q_inc, q_inc_q0_x);
    float a_z_q0_y = dot(a_z_q_inc, q_inc_q0_y);
    float a_z_q0_z = dot(a_z_q_inc, q_inc_q0_z);
    float a_z_q0_w = dot(a_z_q_inc, q_inc_q0_w);
    float t_q0_x = dot(t_q_inc, q_inc_q0_x);
    float t_q0_y = dot(t_q_inc, q_inc_q0_y);
    float t_q0_z = dot(t_q_inc, q_inc_q0_z);
    float t_q0_w = dot(t_q_inc, q_inc_q0_w);

    float cs = cos(angle*t);
    float sn = sin(angle*t);

    quat q_s_q0_x = mul(quat(1.f, 0.f, 0.f, 0.f), q_inc) + mul(q0, quat(
        0.5 * t * axis.x * t_q0_x * cs + a_x_q0_x * sn,
        0.5 * t * axis.y * t_q0_x * cs + a_y_q0_x * sn,
        0.5 * t * axis.z * t_q0_x * cs + a_z_q0_x * sn,
        -0.5 * t * t_q0_x * sn));

    quat q_s_q0_y = mul(quat(0.f, 1.f, 0.f, 0.f), q_inc) + mul(q0, quat(
        0.5 * t * axis.x * t_q0_y * cs + a_x_q0_y * sn,
        0.5 * t * axis.y * t_q0_y * cs + a_y_q0_y * sn,
        0.5 * t * axis.z * t_q0_y * cs + a_z_q0_y * sn,
        -0.5 * t * t_q0_y * sn));

    quat q_s_q0_z = mul(quat(0.f, 0.f, 1.f, 0.f), q_inc) + mul(q0, quat(
        0.5 * t * axis.x * t_q0_z * cs + a_x_q0_z * sn,
        0.5 * t * axis.y * t_q0_z * cs + a_y_q0_z * sn,
        0.5 * t * axis.z * t_q0_z * cs + a_z_q0_z * sn,
        -0.5 * t * t_q0_z * sn));

    quat q_s_q0_w = mul(quat(0.f, 0.f, 0.f, 1.f), q_inc) + mul(q0, quat(
        0.5 * t * axis.x * t_q0_w * cs + a_x_q0_w * sn,
        0.5 * t * axis.y * t_q0_w * cs + a_y_q0_w * sn,
        0.5 * t * axis.z * t_q0_w * cs + a_z_q0_w * sn,
        -0.5 * t * t_q0_w * sn));

    adj_q0.x += dot(q_s_q0_x, adj_ret);
    adj_q0.y += dot(q_s_q0_y, adj_ret);
    adj_q0.z += dot(q_s_q0_z, adj_ret);
    adj_q0.w += dot(q_s_q0_w, adj_ret);

    // adj_q1
    quat q_inc_q1_x = quat(q_inc_x_q1.x, q_inc_y_q1.x, q_inc_z_q1.x, q_inc_w_q1.x);
    quat q_inc_q1_y = quat(q_inc_x_q1.y, q_inc_y_q1.y, q_inc_z_q1.y, q_inc_w_q1.y);
    quat q_inc_q1_z = quat(q_inc_x_q1.z, q_inc_y_q1.z, q_inc_z_q1.z, q_inc_w_q1.z);
    quat q_inc_q1_w = quat(q_inc_x_q1.w, q_inc_y_q1.w, q_inc_z_q1.w, q_inc_w_q1.w);

    float a_x_q1_x = dot(a_x_q_inc, q_inc_q1_x);
    float a_x_q1_y = dot(a_x_q_inc, q_inc_q1_y);
    float a_x_q1_z = dot(a_x_q_inc, q_inc_q1_z);
    float a_x_q1_w = dot(a_x_q_inc, q_inc_q1_w);
    float a_y_q1_x = dot(a_y_q_inc, q_inc_q1_x);
    float a_y_q1_y = dot(a_y_q_inc, q_inc_q1_y);
    float a_y_q1_z = dot(a_y_q_inc, q_inc_q1_z);
    float a_y_q1_w = dot(a_y_q_inc, q_inc_q1_w);
    float a_z_q1_x = dot(a_z_q_inc, q_inc_q1_x);
    float a_z_q1_y = dot(a_z_q_inc, q_inc_q1_y);
    float a_z_q1_z = dot(a_z_q_inc, q_inc_q1_z);
    float a_z_q1_w = dot(a_z_q_inc, q_inc_q1_w);
    float t_q1_x = dot(t_q_inc, q_inc_q1_x);
    float t_q1_y = dot(t_q_inc, q_inc_q1_y);
    float t_q1_z = dot(t_q_inc, q_inc_q1_z);
    float t_q1_w = dot(t_q_inc, q_inc_q1_w);

    quat q_s_q1_x = mul(q0, quat(
        0.5 * t * axis.x * t_q1_x * cs + a_x_q1_x * sn,
        0.5 * t * axis.y * t_q1_x * cs + a_y_q1_x * sn,
        0.5 * t * axis.z * t_q1_x * cs + a_z_q1_x * sn,
        -0.5 * t * t_q1_x * sn));

    quat q_s_q1_y = mul(q0, quat(
        0.5 * t * axis.x * t_q1_y * cs + a_x_q1_y * sn,
        0.5 * t * axis.y * t_q1_y * cs + a_y_q1_y * sn,
        0.5 * t * axis.z * t_q1_y * cs + a_z_q1_y * sn,
        -0.5 * t * t_q1_y * sn));

    quat q_s_q1_z = mul(q0, quat(
        0.5 * t * axis.x * t_q1_z * cs + a_x_q1_z * sn,
        0.5 * t * axis.y * t_q1_z * cs + a_y_q1_z * sn,
        0.5 * t * axis.z * t_q1_z * cs + a_z_q1_z * sn,
        -0.5 * t * t_q1_z * sn));

    quat q_s_q1_w = mul(q0, quat(
        0.5 * t * axis.x * t_q1_w * cs + a_x_q1_w * sn,
        0.5 * t * axis.y * t_q1_w * cs + a_y_q1_w * sn,
        0.5 * t * axis.z * t_q1_w * cs + a_z_q1_w * sn,
        -0.5 * t * t_q1_w * sn));

    adj_q1.x += dot(q_s_q1_x, adj_ret);
    adj_q1.y += dot(q_s_q1_y, adj_ret);
    adj_q1.z += dot(q_s_q1_z, adj_ret);
    adj_q1.w += dot(q_s_q1_w, adj_ret);    

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
    const float tr = m.data[0][0] + m.data[1][1] + m.data[2][2];
    float x, y, z, w, h = 0.0f;

    float dx_dm00 = 0.f, dx_dm01 = 0.f, dx_dm02 = 0.f;
    float dx_dm10 = 0.f, dx_dm11 = 0.f, dx_dm12 = 0.f;
    float dx_dm20 = 0.f, dx_dm21 = 0.f, dx_dm22 = 0.f;
    float dy_dm00 = 0.f, dy_dm01 = 0.f, dy_dm02 = 0.f;
    float dy_dm10 = 0.f, dy_dm11 = 0.f, dy_dm12 = 0.f;
    float dy_dm20 = 0.f, dy_dm21 = 0.f, dy_dm22 = 0.f;
    float dz_dm00 = 0.f, dz_dm01 = 0.f, dz_dm02 = 0.f;
    float dz_dm10 = 0.f, dz_dm11 = 0.f, dz_dm12 = 0.f;
    float dz_dm20 = 0.f, dz_dm21 = 0.f, dz_dm22 = 0.f;
    float dw_dm00 = 0.f, dw_dm01 = 0.f, dw_dm02 = 0.f;
    float dw_dm10 = 0.f, dw_dm11 = 0.f, dw_dm12 = 0.f;
    float dw_dm20 = 0.f, dw_dm21 = 0.f, dw_dm22 = 0.f;

    if (tr >= 0.0f) {
        h = sqrt(tr + 1.0f);
        w = 0.5 * h;
        h = 0.5f / h;

        x = (m.data[2][1] - m.data[1][2]) * h;
        y = (m.data[0][2] - m.data[2][0]) * h;
        z = (m.data[1][0] - m.data[0][1]) * h;

        dw_dm00 = 0.5f * h;
        dw_dm11 = 0.5f * h;
        dw_dm22 = 0.5f * h;
        dx_dm21 = h;
        dx_dm12 = -h;
        dx_dm00 = 2.f * h*h*h * (m.data[1][2] - m.data[2][1]);
        dx_dm11 = 2.f * h*h*h * (m.data[1][2] - m.data[2][1]);
        dx_dm22 = 2.f * h*h*h * (m.data[1][2] - m.data[2][1]);
        dy_dm02 = h;
        dy_dm20 = -h;
        dy_dm00 = 2.f * h*h*h * (m.data[2][0] - m.data[0][2]);
        dy_dm11 = 2.f * h*h*h * (m.data[2][0] - m.data[0][2]);
        dy_dm22 = 2.f * h*h*h * (m.data[2][0] - m.data[0][2]);
        dz_dm10 = h;
        dz_dm01 = -h;
        dz_dm00 = 2.f * h*h*h * (m.data[0][1] - m.data[1][0]);
        dz_dm11 = 2.f * h*h*h * (m.data[0][1] - m.data[1][0]);
        dz_dm22 = 2.f * h*h*h * (m.data[0][1] - m.data[1][0]);
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

            dx_dm00 = 0.5f * h;
            dx_dm11 = -0.5f * h;
            dx_dm22 = -0.5f * h;
            dy_dm01 = h;
            dy_dm10 = h;
            dy_dm00 = -2.f * h*h*h * (m.data[0][1] + m.data[1][0]);
            dy_dm11 = 2.f * h*h*h * (m.data[0][1] + m.data[1][0]);
            dy_dm22 = 2.f * h*h*h * (m.data[0][1] + m.data[1][0]);
            dz_dm20 = h;
            dz_dm02 = h;
            dz_dm00 = -2.f * h*h*h * (m.data[2][0] + m.data[0][2]);
            dz_dm11 = 2.f * h*h*h * (m.data[2][0] + m.data[0][2]);
            dz_dm22 = 2.f * h*h*h * (m.data[2][0] + m.data[0][2]);
            dw_dm21 = h;
            dw_dm12 = -h;
            dw_dm00 = 2.f * h*h*h * (m.data[1][2] - m.data[2][1]);
            dw_dm11 = 2.f * h*h*h * (m.data[2][1] - m.data[1][2]);
            dw_dm22 = 2.f * h*h*h * (m.data[2][1] - m.data[1][2]);
        } else if (max_diag == 1) {
            h = sqrt((m.data[1][1] - (m.data[2][2] + m.data[0][0])) + 1.0f);
            y = 0.5f * h;
            h = 0.5f / h;

            z = (m.data[1][2] + m.data[2][1]) * h;
            x = (m.data[0][1] + m.data[1][0]) * h;
            w = (m.data[0][2] - m.data[2][0]) * h;

            dy_dm00 = -0.5f * h;
            dy_dm11 = 0.5f * h;
            dy_dm22 = -0.5f * h;
            dz_dm12 = h;
            dz_dm21 = h;
            dz_dm00 = 2.f * h*h*h * (m.data[1][2] + m.data[2][1]);
            dz_dm11 = -2.f * h*h*h * (m.data[1][2] + m.data[2][1]);
            dz_dm22 = 2.f * h*h*h * (m.data[1][2] + m.data[2][1]);
            dx_dm01 = h;
            dx_dm10 = h;
            dx_dm00 = 2.f * h*h*h * (m.data[0][1] + m.data[1][0]);
            dx_dm11 = -2.f * h*h*h * (m.data[0][1] + m.data[1][0]);
            dx_dm22 = 2.f * h*h*h * (m.data[0][1] + m.data[1][0]);
            dw_dm02 = h;
            dw_dm20 = -h;
            dw_dm00 = 2.f * h*h*h * (m.data[0][2] - m.data[2][0]);
            dw_dm11 = 2.f * h*h*h * (m.data[2][0] - m.data[0][2]);
            dw_dm22 = 2.f * h*h*h * (m.data[0][2] - m.data[2][0]);
        } if (max_diag == 2) {
            h = sqrt((m.data[2][2] - (m.data[0][0] + m.data[1][1])) + 1.0f);
            z = 0.5f * h;
            h = 0.5f / h;

            x = (m.data[2][0] + m.data[0][2]) * h;
            y = (m.data[1][2] + m.data[2][1]) * h;
            w = (m.data[1][0] - m.data[0][1]) * h;

            dz_dm00 = -0.5f * h;
            dz_dm11 = -0.5f * h;
            dz_dm22 = 0.5f * h;
            dx_dm20 = h;
            dx_dm02 = h;
            dx_dm00 = 2.f * h*h*h * (m.data[2][0] + m.data[0][2]);
            dx_dm11 = 2.f * h*h*h * (m.data[2][0] + m.data[0][2]);
            dx_dm22 = -2.f * h*h*h * (m.data[2][0] + m.data[0][2]);
            dy_dm12 = h;
            dy_dm21 = h;
            dy_dm00 = 2.f * h*h*h * (m.data[1][2] + m.data[2][1]);
            dy_dm11 = 2.f * h*h*h * (m.data[1][2] + m.data[2][1]);
            dy_dm22 = -2.f * h*h*h * (m.data[1][2] + m.data[2][1]);
            dw_dm10 = h;
            dw_dm01 = -h;
            dw_dm00 = 2.f * h*h*h * (m.data[1][0] - m.data[0][1]);
            dw_dm11 = 2.f * h*h*h * (m.data[1][0] - m.data[0][1]);
            dw_dm22 = 2.f * h*h*h * (m.data[0][1] - m.data[1][0]);
        }
    }

    quat dq_dm00 = quat(dx_dm00, dy_dm00, dz_dm00, dw_dm00);
    quat dq_dm01 = quat(dx_dm01, dy_dm01, dz_dm01, dw_dm01);
    quat dq_dm02 = quat(dx_dm02, dy_dm02, dz_dm02, dw_dm02);
    quat dq_dm10 = quat(dx_dm10, dy_dm10, dz_dm10, dw_dm10);
    quat dq_dm11 = quat(dx_dm11, dy_dm11, dz_dm11, dw_dm11);
    quat dq_dm12 = quat(dx_dm12, dy_dm12, dz_dm12, dw_dm12);
    quat dq_dm20 = quat(dx_dm20, dy_dm20, dz_dm20, dw_dm20);
    quat dq_dm21 = quat(dx_dm21, dy_dm21, dz_dm21, dw_dm21);
    quat dq_dm22 = quat(dx_dm22, dy_dm22, dz_dm22, dw_dm22);

    quat adj_q;
    adj_normalize(quat(x, y, z, w), adj_q, adj_ret);

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

} // namespace wp