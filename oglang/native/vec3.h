#pragma once

struct vec3
{
    float x;
    float y;
    float z;

    inline CUDA_CALLABLE vec3() : x(0.0f), y(0.0f), z(0.0f) {}
    inline CUDA_CALLABLE vec3(float x, float y, float z) : x(x), y(y), z(z) {}
    inline CUDA_CALLABLE vec3(float s) : x(s), y(s), z(s) {}

    explicit inline CUDA_CALLABLE vec3(const float* p) : x(p[0]), y(p[1]), z(p[2]) {}

    CUDA_CALLABLE float operator[](int index) const
    {
        assert(index < 3);        
        return (&x)[index];
    }

    CUDA_CALLABLE float& operator[](int index)
    {
        assert(index < 3);
        return (&x)[index];
    }    
};


//--------------
// vec3 methods

inline CUDA_CALLABLE vec3 operator - (vec3 a)
{
    return { -a.x, -a.y, -a.z };
}


inline CUDA_CALLABLE vec3 mul(vec3 a, float s)
{
    return { a.x*s, a.y*s, a.z*s };
}

inline CUDA_CALLABLE vec3 mul(float s, vec3 a)
{
    return mul(a, s);
}

inline CUDA_CALLABLE vec3 mul(vec3 a, vec3 b)
{
    return { a.x*b.x, a.y*b.y, a.z*b.z };
}

inline CUDA_CALLABLE vec3 div(vec3 a, float s)
{
    return { a.x/s, a.y/s, a.z/s };
}

inline CUDA_CALLABLE vec3 div(vec3 a, vec3 b)
{
    return { a.x/b.x, a.y/b.y, a.z/b.z };
}

inline CUDA_CALLABLE vec3 add(vec3 a, vec3 b)
{
    return { a.x+b.x, a.y+b.y, a.z+b.z };
}

inline CUDA_CALLABLE vec3 add(vec3 a, float s)
{
    return { a.x + s, a.y + s, a.z + s };
}


inline CUDA_CALLABLE vec3 sub(vec3 a, vec3 b)
{
    return { a.x-b.x, a.y-b.y, a.z-b.z };
}

inline CUDA_CALLABLE float dot(vec3 a, vec3 b)
{
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

inline CUDA_CALLABLE vec3 cross(vec3 a, vec3 b)
{
    vec3 c;
    c.x = a.y*b.z - a.z*b.y;
    c.y = a.z*b.x - a.x*b.z;
    c.z = a.x*b.y - a.y*b.x;

    return c;
}

inline CUDA_CALLABLE vec3 min(vec3 a, vec3 b)
{
    return vec3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}

inline CUDA_CALLABLE vec3 max(vec3 a, vec3 b)
{
    return vec3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}

inline CUDA_CALLABLE float index(const vec3 & a, int idx)
{
#if FP_CHECK
    if (idx < 0 || idx > 2)
    {
        printf("vec3 index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        exit(1);
    }
#endif

    return (&a.x)[idx];
        
}

inline CUDA_CALLABLE void adj_index(const vec3 & a, int idx, vec3 & adj_a, int & adj_idx, float & adj_ret)
{
#if FP_CHECK
    if (idx < 0 || idx > 2)
    {
        printf("vec3 index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        exit(1);
    }
#endif

    (&adj_a.x)[idx] += adj_ret;
}

inline CUDA_CALLABLE float length(vec3 a)
{
    return sqrtf(dot(a, a));
}

inline CUDA_CALLABLE float length_sq(vec3 a)
{
    return dot(a, a);
}

inline CUDA_CALLABLE vec3 normalize(vec3 a)
{
    float l = length(a);
    if (l > kEps)
        return div(a,l);
    else
        return vec3();
}



inline bool CUDA_CALLABLE isfinite(vec3 x)
{
    return ::isfinite(x.x) && ::isfinite(x.y) && ::isfinite(x.z);
}

// adjoint vec3 constructor
inline CUDA_CALLABLE void adj_vec3(float x, float y, float z, float& adj_x, float& adj_y, float& adj_z, const vec3& adj_ret)
{
    adj_x += adj_ret.x;
    adj_y += adj_ret.y;
    adj_z += adj_ret.z;
}

inline CUDA_CALLABLE void adj_mul(vec3 a, float s, vec3& adj_a, float& adj_s, const vec3& adj_ret)
{
    adj_a.x += s*adj_ret.x;
    adj_a.y += s*adj_ret.y;
    adj_a.z += s*adj_ret.z;
    adj_s += dot(a, adj_ret);

#if FP_CHECK
    if (!isfinite(a) || !isfinite(s) || !isfinite(adj_a) || !isfinite(adj_s) || !isfinite(adj_ret))
        printf("adj_mul((%f %f %f), %f, (%f %f %f), %f, (%f %f %f)\n", a.x, a.y, a.z, s, adj_a.x, adj_a.y, adj_a.z, adj_s, adj_ret.x, adj_ret.y, adj_ret.z);
#endif
}



inline CUDA_CALLABLE void adj_div(vec3 a, float s, vec3& adj_a, float& adj_s, const vec3& adj_ret)
{
    adj_s += dot(- a / (s * s), adj_ret); // - a / s^2

    adj_a.x += adj_ret.x / s;
    adj_a.y += adj_ret.y / s;
    adj_a.z += adj_ret.z / s;

#if FP_CHECK
    if (!isfinite(a) || !isfinite(s) || !isfinite(adj_a) || !isfinite(adj_s) || !isfinite(adj_ret))
        printf("adj_div((%f %f %f), %f, (%f %f %f), %f, (%f %f %f)\n", a.x, a.y, a.z, s, adj_a.x, adj_a.y, adj_a.z, adj_s, adj_ret.x, adj_ret.y, adj_ret.z);
#endif
}

inline CUDA_CALLABLE void adj_add(vec3 a, vec3 b, vec3& adj_a, vec3& adj_b, const vec3& adj_ret)
{
    adj_a += adj_ret;
    adj_b += adj_ret;
}

inline CUDA_CALLABLE void adj_add(vec3 a, float s, vec3& adj_a, float& adj_s, const vec3& adj_ret)
{
    adj_a += adj_ret;
    adj_s += adj_ret.x + adj_ret.y + adj_ret.z;
}

inline CUDA_CALLABLE void adj_sub(vec3 a, vec3 b, vec3& adj_a, vec3& adj_b, const vec3& adj_ret)
{
    adj_a += adj_ret;
    adj_b -= adj_ret;
}

inline CUDA_CALLABLE void adj_dot(vec3 a, vec3 b, vec3& adj_a, vec3& adj_b, const float adj_ret)
{
    adj_a += b*adj_ret;
    adj_b += a*adj_ret;

#if FP_CHECK
    if (!isfinite(a) || !isfinite(b) || !isfinite(adj_a) || !isfinite(adj_b) || !isfinite(adj_ret))
        printf("adj_dot((%f %f %f), (%f %f %f), (%f %f %f), (%f %f %f), %f)\n", a.x, a.y, a.z, b.x, b.y, b.z, adj_a.x, adj_a.y, adj_a.z, adj_b.x, adj_b.y, adj_b.z, adj_ret);
#endif

}

inline CUDA_CALLABLE void adj_cross(vec3 a, vec3 b, vec3& adj_a, vec3& adj_b, const vec3& adj_ret)
{
    // todo: sign check
    adj_a += cross(b, adj_ret);
    adj_b -= cross(a, adj_ret);
}


#ifdef CUDA
inline __device__ void atomic_add(vec3 * addr, vec3 value) {
    // *addr += value;
    atomicAdd(&(addr -> x), value.x);
    atomicAdd(&(addr -> y), value.y);
    atomicAdd(&(addr -> z), value.z);
}
#endif

inline CUDA_CALLABLE void adj_length(vec3 a, vec3& adj_a, const float adj_ret)
{
    adj_a += normalize(a)*adj_ret;

#if FP_CHECK
    if (!isfinite(adj_a))
        printf("%s:%d - adj_length((%f %f %f), (%f %f %f), (%f))\n", __FILE__, __LINE__, a.x, a.y, a.z, adj_a.x, adj_a.y, adj_a.z, adj_ret);
#endif
}

inline CUDA_CALLABLE void adj_normalize(vec3 a, vec3& adj_a, const vec3& adj_ret)
{
    float d = length(a);
    
    if (d > kEps)
    {
        float invd = 1.0f/d;

        vec3 ahat = normalize(a);

        adj_a += (adj_ret - ahat*(dot(ahat, adj_ret))*invd);

#if FP_CHECK
        if (!isfinite(adj_a))
            printf("%s:%d - adj_normalize((%f %f %f), (%f %f %f), (%f, %f, %f))\n", __FILE__, __LINE__, a.x, a.y, a.z, adj_a.x, adj_a.y, adj_a.z, adj_ret.x, adj_ret.y, adj_ret.z);

#endif
    }
}


CUDA_CALLABLE inline int longest_axis(const vec3& v)
{    
	if (v.x > v.y && v.x > v.z)
		return 0;
	else
		return (v.y > v.z) ? 1 : 2;
}

