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

struct vec2
{
    float x;
    float y;

    inline CUDA_CALLABLE vec2() : x(0.0f), y(0.0f) {}
    inline CUDA_CALLABLE vec2(float x, float y) : x(x), y(y){}
    inline CUDA_CALLABLE vec2(float s) : x(s), y(s) {}

    explicit inline CUDA_CALLABLE vec2(const float* p) : x(p[0]), y(p[1]) {}

    CUDA_CALLABLE float operator[](int index) const
    {
        assert(index < 2);        
        return (&x)[index];
    }

    CUDA_CALLABLE float& operator[](int index)
    {
        assert(index < 2);
        return (&x)[index];
    }    
};


//--------------
// vec2 methods

inline CUDA_CALLABLE vec2 operator-(vec2 a)
{
    return { -a.x, -a.y };
}

inline CUDA_CALLABLE bool operator==(const vec2& a, const vec2& b)
{
    return a.x == b.x && a.y == b.y;
}

inline CUDA_CALLABLE vec2 mul(vec2 a, float s)
{
    return { a.x*s, a.y*s };
}

inline CUDA_CALLABLE vec2 mul(float s, vec2 a)
{
    return mul(a, s);
}

inline CUDA_CALLABLE vec2 cw_mul(vec2 a, vec2 b)
{
    return { a.x*b.x, a.y*b.y };
}

inline CUDA_CALLABLE vec2 div(vec2 a, float s)
{
    return { a.x/s, a.y/s };
}

inline CUDA_CALLABLE vec2 cw_div(vec2 a, vec2 b)
{
    return { a.x/b.x, a.y/b.y };
}

inline CUDA_CALLABLE vec2 add(vec2 a, vec2 b)
{
    return { a.x+b.x, a.y+b.y };
}

inline CUDA_CALLABLE vec2 add(vec2 a, float s)
{
    return { a.x + s, a.y + s };
}


inline CUDA_CALLABLE vec2 sub(vec2 a, vec2 b)
{
    return { a.x-b.x, a.y-b.y };
}

inline CUDA_CALLABLE vec2 sub(vec2 a, float s)
{
  return { a.x - s, a.y - s };
}

inline CUDA_CALLABLE vec2 log(vec2 a)
{
  return { logf(a.x), logf(a.y) };
}

inline CUDA_CALLABLE vec2 exp(vec2 a)
{
  return { expf(a.x), expf(a.y) };
}

inline CUDA_CALLABLE vec2 pow(vec2 a, float b)
{
    return { powf(a.x, b), powf(a.y, b) };
}

inline CUDA_CALLABLE float dot(vec2 a, vec2 b)
{
    return a.x*b.x + a.y*b.y;
}

inline CUDA_CALLABLE vec2 min(vec2 a, vec2 b)
{
    return vec2(min(a.x, b.x), min(a.y, b.y));
}

inline CUDA_CALLABLE vec2 max(vec2 a, vec2 b)
{
    return vec2(max(a.x, b.x), max(a.y, b.y));
}

inline CUDA_CALLABLE float index(const vec2 & a, int idx)
{
#if FP_CHECK
    if (idx < 0 || idx > 1)
    {
        printf("vec2 index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    return (&a.x)[idx];
        
}

inline CUDA_CALLABLE void adj_index(const vec2 & a, int idx, vec2 & adj_a, int & adj_idx, float & adj_ret)
{
#if FP_CHECK
    if (idx < 0 || idx > 1)
    {
        printf("vec2 index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    (&adj_a.x)[idx] += adj_ret;
}

inline CUDA_CALLABLE float length(vec2 a)
{
    return sqrt(dot(a, a));
}

inline CUDA_CALLABLE float length_sq(vec2 a)
{
    return dot(a, a);
}

inline CUDA_CALLABLE vec2 normalize(vec2 a)
{
    float l = length(a);
    if (l > kEps)
        return div(a,l);
    else
        return vec2();
}



inline bool CUDA_CALLABLE isfinite(vec2 x)
{
    return ::isfinite(x.x) && ::isfinite(x.y);
}

// adjoint vec2 constructors
inline CUDA_CALLABLE void adj_vec2(float x, float y, float& adj_x, float& adj_y, const vec2& adj_ret)
{
    adj_x += adj_ret.x;
    adj_y += adj_ret.y;
}

inline CUDA_CALLABLE void adj_vec2(float s, float& adj_s, const vec2& adj_ret)
{
    adj_s += adj_ret.x + adj_ret.y;
}


inline CUDA_CALLABLE void adj_mul(vec2 a, float s, vec2& adj_a, float& adj_s, const vec2& adj_ret)
{
    adj_a.x += s*adj_ret.x;
    adj_a.y += s*adj_ret.y;
    adj_s += dot(a, adj_ret);

#if FP_CHECK
    if (!isfinite(a) || !isfinite(s) || !isfinite(adj_a) || !isfinite(adj_s) || !isfinite(adj_ret))
    {
        printf("adj_mul((%f %f), %f, (%f %f), %f, (%f %f)\n", a.x, a.y, s, adj_a.x, adj_a.y, adj_s, adj_ret.x, adj_ret.y);
        assert(0);
    }
#endif
}

inline CUDA_CALLABLE void adj_mul(float s, vec2 a, float& adj_s, vec2& adj_a, const vec2& adj_ret)
{
    adj_mul(a, s, adj_a, adj_s, adj_ret);
}

inline CUDA_CALLABLE void adj_cw_mul(vec2 a, vec2 b, vec2& adj_a, vec2& adj_b, const vec2& adj_ret)
{
  adj_a += cw_mul(b, adj_ret);
  adj_b += cw_mul(a, adj_ret);
}

inline CUDA_CALLABLE void adj_div(vec2 a, float s, vec2& adj_a, float& adj_s, const vec2& adj_ret)
{
    adj_s += dot(- a / (s * s), adj_ret); // - a / s^2

    adj_a.x += adj_ret.x / s;
    adj_a.y += adj_ret.y / s;

#if FP_CHECK
    if (!isfinite(a) || !isfinite(s) || !isfinite(adj_a) || !isfinite(adj_s) || !isfinite(adj_ret))
    {
        printf("adj_div((%f %f), %f, (%f %f), %f, (%f %f)\n", a.x, a.y, s, adj_a.x, adj_a.y, adj_s, adj_ret.x, adj_ret.y);
        assert(0);
    }
#endif
}

inline CUDA_CALLABLE void adj_cw_div(vec2 a, vec2 b, vec2& adj_a, vec2& adj_b, const vec2& adj_ret) {
  adj_a += cw_div(adj_ret, b);
  adj_b -= cw_mul(adj_ret, cw_div(cw_div(a, b), b));
}

inline CUDA_CALLABLE void adj_add(vec2 a, vec2 b, vec2& adj_a, vec2& adj_b, const vec2& adj_ret)
{
    adj_a += adj_ret;
    adj_b += adj_ret;
}

inline CUDA_CALLABLE void adj_add(vec2 a, float s, vec2& adj_a, float& adj_s, const vec2& adj_ret)
{
    adj_a += adj_ret;
    adj_s += adj_ret.x + adj_ret.y;
}

inline CUDA_CALLABLE void adj_sub(vec2 a, vec2 b, vec2& adj_a, vec2& adj_b, const vec2& adj_ret)
{
    adj_a += adj_ret;
    adj_b -= adj_ret;
}

inline CUDA_CALLABLE void adj_sub(vec2 a, float s, vec2& adj_a, float& adj_s, const vec2& adj_ret)
{
  adj_a += adj_ret;
  adj_s -= adj_ret.x + adj_ret.y;
}

// TODO: test
inline CUDA_CALLABLE void adj_log(vec2 a, vec2& adj_a, const vec2& adj_ret)
{
  adj_a += vec2(adj_ret.x / a.x, adj_ret.y / a.y);
}

// TODO: test
inline CUDA_CALLABLE void adj_exp(vec2 a, vec2& adj_a, const vec2& adj_ret)
{
  adj_a += vec2(adj_ret.x * expf(a.x), adj_ret.y * expf(a.y));
}

// TODO: test
inline CUDA_CALLABLE void adj_pow(vec2 a, float b, vec2& adj_a, float& adj_b, const vec2& adj_ret)
{
    adj_a += vec2(b*powf(a.x, b-1.f) * adj_ret.x, b*powf(a.y, b-1.f) * adj_ret.y);
    adj_b += logf(a.x)*powf(a.x, b) * adj_ret.x + logf(a.y)*powf(a.y, b) * adj_ret.y;
}

inline CUDA_CALLABLE void adj_dot(vec2 a, vec2 b, vec2& adj_a, vec2& adj_b, const float adj_ret)
{
    adj_a += b*adj_ret;
    adj_b += a*adj_ret;

#if FP_CHECK
    if (!isfinite(a) || !isfinite(b) || !isfinite(adj_a) || !isfinite(adj_b) || !isfinite(adj_ret))
    {
        printf("adj_dot((%f %f), (%f %f), (%f %f), (%f %f), %f)\n", a.x, a.y, b.x, b.y, adj_a.x, adj_a.y, adj_b.x, adj_b.y, adj_ret);
        assert(0);
    }
#endif

}


inline CUDA_CALLABLE vec2 atomic_add(vec2 * addr, vec2 value) {

    float x = atomic_add(&(addr -> x), value.x);
    float y = atomic_add(&(addr -> y), value.y);

    return vec2(x, y);
}

inline CUDA_CALLABLE void adj_length(vec2 a, vec2& adj_a, const float adj_ret)
{
    adj_a += normalize(a)*adj_ret;

#if FP_CHECK
    if (!isfinite(adj_a))
    {
        printf("%s:%d - adj_length((%f %f), (%f %f), (%f))\n", __FILE__, __LINE__, a.x, a.y, adj_a.x, adj_a.y, adj_ret);
        assert(0);
    }
#endif
}

inline CUDA_CALLABLE void adj_normalize(vec2 a, vec2& adj_a, const vec2& adj_ret)
{
    float d = length(a);
    
    if (d > kEps)
    {
        float invd = 1.0f/d;

        vec2 ahat = normalize(a);

        adj_a += (adj_ret*invd - ahat*(dot(ahat, adj_ret))*invd);

#if FP_CHECK
        if (!isfinite(adj_a))
        {
            printf("%s:%d - adj_normalize((%f %f), (%f %f), (%f, %f))\n", __FILE__, __LINE__, a.x, a.y, adj_a.x, adj_a.y, adj_ret.x, adj_ret.y);
            assert(0);
        }
#endif
    }
}


} // namespace wp
