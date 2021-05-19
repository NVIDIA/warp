#pragma once

struct vec4
{
    float x;
    float y;
    float z;
    float w;

    inline CUDA_CALLABLE vec4() : x(0.0f), y(0.0f), z(0.0f), w(0.0f) {}
    inline CUDA_CALLABLE vec4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}
    explicit inline CUDA_CALLABLE vec4(const float* p) : x(p[0]), y(p[1]), z(p[2]), w(p[3]) {}

    float operator[](int index) const
    {
        assert(index < 4);
        return (&x)[index];
    }

    float& operator[](int index)
    {
        assert(index < 4);
        return (&x)[index];
    }    
};


//--------------
// vec4 methods

inline CUDA_CALLABLE vec4 operator - (vec4 a)
{
    return { -a.x, -a.y, -a.z, -a.w };
}


inline CUDA_CALLABLE vec4 mul(vec4 a, float s)
{
    return { a.x*s, a.y*s, a.z*s, a.w*s };
}

inline CUDA_CALLABLE vec4 mul(float s, vec4 a)
{
    return mul(a, s);
}


inline CUDA_CALLABLE vec4 div(vec4 a, float s)
{
    return { a.x/s, a.y/s, a.z/s, a.w/s };
}

inline CUDA_CALLABLE vec4 add(vec4 a, vec4 b)
{
    return { a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w};
}


inline CUDA_CALLABLE vec4 sub(vec4 a, vec4 b)
{
    return { a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w };
}

inline CUDA_CALLABLE float dot(vec4 a, vec4 b)
{
    return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
}

inline CUDA_CALLABLE float index(const vec4 & a, int idx)
{
#if FP_CHECK
    if (idx < 0 || idx > 3)
    {
        printf("vec4 index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        exit(1);
    }
#endif

    return (&a.x)[idx];
        
}

inline CUDA_CALLABLE void adj_index(const vec4 & a, int idx, vec4 & adj_a, int & adj_idx, float & adj_ret)
{
#if FP_CHECK
    if (idx < 0 || idx > 3)
    {
        printf("vec4 index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        exit(1);
    }
#endif

    (&adj_a.x)[idx] += adj_ret;
}

inline CUDA_CALLABLE float length(vec4 a)
{
    return sqrtf(dot(a, a));
}

inline CUDA_CALLABLE vec4 normalize(vec4 a)
{
    float l = length(a);
    if (l > kEps)
        return div(a,l);
    else
        return vec4();
}



inline bool CUDA_CALLABLE isfinite(vec4 x)
{
    return ::isfinite(x.x) && ::isfinite(x.y) && ::isfinite(x.z) && ::isfinite(x.w);
}

// adjoint vec4 constructor
inline CUDA_CALLABLE void adj_vec4(float x, float y, float z, float& adj_x, float& adj_y, float& adj_z, float& adj_w, const vec4& adj_ret)
{
    adj_x += adj_ret.x;
    adj_y += adj_ret.y;
    adj_z += adj_ret.z;
    adj_w += adj_ret.w;
}

inline CUDA_CALLABLE void adj_mul(vec4 a, float s, vec4& adj_a, float& adj_s, const vec4& adj_ret)
{
    adj_a.x += s*adj_ret.x;
    adj_a.y += s*adj_ret.y;
    adj_a.z += s*adj_ret.z;
    adj_a.w += s*adj_ret.w;

    adj_s += dot(a, adj_ret);

#if FP_CHECK
    if (!isfinite(a) || !isfinite(s) || !isfinite(adj_a) || !isfinite(adj_s) || !isfinite(adj_ret))
        printf("adj_mul((%f %f %f), %f, (%f %f %f), %f, (%f %f %f)\n", a.x, a.y, a.z, s, adj_a.x, adj_a.y, adj_a.z, adj_s, adj_ret.x, adj_ret.y, adj_ret.z);
#endif
}

inline CUDA_CALLABLE void adj_div(vec4 a, float s, vec4& adj_a, float& adj_s, const vec4& adj_ret)
{
    adj_s += dot(- a / (s * s), adj_ret); // - a / s^2

    adj_a.x += adj_ret.x / s;
    adj_a.y += adj_ret.y / s;
    adj_a.z += adj_ret.z / s;
    adj_a.w += adj_ret.w / s;

#if FP_CHECK
    if (!isfinite(a) || !isfinite(s) || !isfinite(adj_a) || !isfinite(adj_s) || !isfinite(adj_ret))
        printf("adj_div((%f %f %f), %f, (%f %f %f), %f, (%f %f %f)\n", a.x, a.y, a.z, s, adj_a.x, adj_a.y, adj_a.z, adj_s, adj_ret.x, adj_ret.y, adj_ret.z);
#endif
}

inline CUDA_CALLABLE void adj_add(vec4 a, vec4 b, vec4& adj_a, vec4& adj_b, const vec4& adj_ret)
{
    adj_a += adj_ret;
    adj_b += adj_ret;
}

inline CUDA_CALLABLE void adj_sub(vec4 a, vec4 b, vec4& adj_a, vec4& adj_b, const vec4& adj_ret)
{
    adj_a += adj_ret;
    adj_b -= adj_ret;
}

inline CUDA_CALLABLE void adj_dot(vec4 a, vec4 b, vec4& adj_a, vec4& adj_b, const float adj_ret)
{
    adj_a += b*adj_ret;
    adj_b += a*adj_ret;

#if FP_CHECK
    if (!isfinite(a) || !isfinite(b) || !isfinite(adj_a) || !isfinite(adj_b) || !isfinite(adj_ret))
        printf("adj_dot((%f %f %f), (%f %f %f), (%f %f %f), (%f %f %f), %f)\n", a.x, a.y, a.z, b.x, b.y, b.z, adj_a.x, adj_a.y, adj_a.z, adj_b.x, adj_b.y, adj_b.z, adj_ret);
#endif

}

#ifdef CUDA
inline __device__ void atomic_add(vec4 * addr, vec4 value) {
    // *addr += value;
    atomicAdd(&(addr -> x), value.x);
    atomicAdd(&(addr -> y), value.y);
    atomicAdd(&(addr -> z), value.z);
    atomicAdd(&(addr -> w), value.w);
}
#endif

inline CUDA_CALLABLE void adj_length(vec4 a, vec4& adj_a, const float adj_ret)
{
    adj_a += normalize(a)*adj_ret;

#if FP_CHECK
    if (!isfinite(adj_a))
        printf("%s:%d - adj_length((%f %f %f), (%f %f %f), (%f))\n", __FILE__, __LINE__, a.x, a.y, a.z, adj_a.x, adj_a.y, adj_a.z, adj_ret);
#endif
}

inline CUDA_CALLABLE void adj_normalize(vec4 a, vec4& adj_a, const vec4& adj_ret)
{
    float d = length(a);
    
    if (d > kEps)
    {
        float invd = 1.0f/d;

        vec4 ahat = normalize(a);

        adj_a += (adj_ret - ahat*(dot(ahat, adj_ret))*invd);

#if FP_CHECK
        if (!isfinite(adj_a))
            printf("%s:%d - adj_normalize((%f %f %f), (%f %f %f), (%f, %f, %f))\n", __FILE__, __LINE__, a.x, a.y, a.z, adj_a.x, adj_a.y, adj_a.z, adj_ret.x, adj_ret.y, adj_ret.z);

#endif
    }
}

