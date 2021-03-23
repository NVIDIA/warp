#pragma once

struct quat
{
    // imaginary part
    float x;
    float y;
    float z;
    
    // real part
    float w;

    inline CUDA_CALLABLE quat(float x=0.0f, float y=0.0f, float z=0.0f, float w=0.0) : x(x), y(y), z(z), w(w) {}    
    explicit inline CUDA_CALLABLE quat(const float3& v, float w=0.0f) : x(v.x), y(v.y), z(v.z), w(w) {}
};

#ifdef CUDA
inline __device__ void atomic_add(quat * addr, quat value) {
    atomicAdd(&(addr -> x), value.x);
    atomicAdd(&(addr -> y), value.y);
    atomicAdd(&(addr -> z), value.z);
    atomicAdd(&(addr -> w), value.w);
}
#endif

inline CUDA_CALLABLE void adj_quat(float x, float y, float z, float w, float& adj_x, float& adj_y, float& adj_z, float& adj_w, quat adj_ret)
{
    adj_x += adj_ret.x;
    adj_y += adj_ret.y;
    adj_z += adj_ret.z;
    adj_w += adj_ret.w;
}

inline CUDA_CALLABLE void adj_quat(const float3& v, float w, float3& adj_v, float& adj_w, quat adj_ret)
{
    adj_v.x += adj_ret.x;
    adj_v.y += adj_ret.y;
    adj_v.z += adj_ret.z;
    adj_w   += adj_ret.w;
}

// foward methods

inline CUDA_CALLABLE quat quat_from_axis_angle(const float3& axis, float angle)
{
    float half = angle*0.5f;
    float w = cosf(half);

    float sin_theta_over_two = sinf(half);
    float3 v = axis*sin_theta_over_two;

    return quat(v.x, v.y, v.z, w);
}

inline CUDA_CALLABLE quat quat_identity()
{
    return quat(0.0f, 0.0f, 0.0f, 1.0f);
}

inline CUDA_CALLABLE float dot(const quat& a, const quat& b)
{
    return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
}

inline CUDA_CALLABLE float length(const quat& q)
{
    return sqrtf(dot(q, q));
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

inline CUDA_CALLABLE quat inverse(const quat& q)
{
    return quat(-q.x, -q.y, -q.z, q.w);
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

inline CUDA_CALLABLE float3 rotate(const quat& q, const float3& x)
{
    return x*(2.0f*q.w*q.w-1.0f) + cross(float3(&q.x), x)*q.w*2.0f + float3(&q.x)*dot(float3(&q.x), x)*2.0f;
}

inline CUDA_CALLABLE float3 rotate_inv(const quat& q, const float3& x)
{
    return x*(2.0f*q.w*q.w-1.0f) - cross(float3(&q.x), x)*q.w*2.0f + float3(&q.x)*dot(float3(&q.x), x)*2.0f;
}


inline CUDA_CALLABLE float index(const quat& a, int idx)
{
#if FP_CHECK
    if (idx < 0 || idx > 3)
    {
        printf("quat index %d out of bounds at %s %d", idx, __FILE__, __LINE__);
        exit(1);
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
        exit(1);
    }
#endif

    (&adj_a.x)[idx] += adj_ret;
}


// backward methods
inline CUDA_CALLABLE void adj_quat_from_axis_angle(const float3& axis, float angle, float3& adj_axis, float& adj_angle, const quat& adj_ret)
{
    float3 v = float3(adj_ret.x, adj_ret.y, adj_ret.z);

    float s = sinf(angle*0.5f);
    float c = cosf(angle*0.5f);

    quat dqda = quat(axis.x*c, axis.y*c, axis.z*c, -s)*0.5f;

    adj_axis += v*s;
    adj_angle += dot(dqda, adj_ret);
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

inline CUDA_CALLABLE void adj_inverse(const quat& q, quat& adj_q, const quat& adj_ret)
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


inline CUDA_CALLABLE void adj_rotate(const quat& q, const float3& p, quat& adj_q, float3& adj_p, const float3& adj_ret)
{
    const float3& r = adj_ret;

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

inline CUDA_CALLABLE void adj_rotate_inv(const quat& q, const float3& p, quat& adj_q, float3& adj_p, const float3& adj_ret)
{
    const float3& r = adj_ret;

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