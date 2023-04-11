/** Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

# pragma once
#include "array.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

namespace wp
{

inline CUDA_CALLABLE uint32 rand_pcg(uint32 state)
{
    uint32 b = state * 747796405u + 2891336453u;
    uint32 c = ((b >> ((b >> 28u) + 4u)) ^ b) * 277803737u;
    return (c >> 22u) ^ c;
}

inline CUDA_CALLABLE uint32 rand_init(int seed) { return rand_pcg(uint32(seed)); }
inline CUDA_CALLABLE uint32 rand_init(int seed, int offset) { return rand_pcg(uint32(seed) + rand_pcg(uint32(offset))); }

inline CUDA_CALLABLE int randi(uint32& state) { state = rand_pcg(state); return int(state); }
inline CUDA_CALLABLE int randi(uint32& state, int min, int max) { state = rand_pcg(state); return state % (max - min) + min; }

inline CUDA_CALLABLE float randf(uint32& state) { state = rand_pcg(state); return (state >> 8) * (1.0f / 16777216.0f); }
inline CUDA_CALLABLE float randf(uint32& state, float min, float max) { return (max - min) * randf(state) + min; }

// Box-Muller method
inline CUDA_CALLABLE float randn(uint32& state) { return sqrt(-2.f * log(randf(state))) * cos(2.f * M_PI * randf(state)); }

inline CUDA_CALLABLE void adj_rand_init(int seed, int& adj_seed, float adj_ret) {}
inline CUDA_CALLABLE void adj_rand_init(int seed, int offset, int& adj_seed, int& adj_offset, float adj_ret) {}

inline CUDA_CALLABLE void adj_randi(uint32& state, uint32& adj_state, float adj_ret) {}
inline CUDA_CALLABLE void adj_randi(uint32& state, int min, int max, uint32& adj_state, int& adj_min, int& adj_max, float adj_ret) {}

inline CUDA_CALLABLE void adj_randf(uint32& state, uint32& adj_state, float adj_ret) {}
inline CUDA_CALLABLE void adj_randf(uint32& state, float min, float max, uint32& adj_state, float& adj_min, float& adj_max, float adj_ret) {}

inline CUDA_CALLABLE void adj_randn(uint32& state, uint32& adj_state, float adj_ret) {}

inline CUDA_CALLABLE int sample_cdf(uint32& state, const array_t<float>& cdf)
{
    float u = randf(state);
    return lower_bound<float>(cdf, u);
}

inline CUDA_CALLABLE vec2 sample_triangle(uint32& state)
{
    float r = sqrt(randf(state));
    float u = 1.0 - r;
    float v = randf(state) * r;
    return vec2(u, v);
}

inline CUDA_CALLABLE vec2 sample_unit_ring(uint32& state)
{
    float theta = randf(state, 0.f, 2.f*M_PI);
    float x = cos(theta);
    float y = sin(theta);
    return vec2(x, y);
}

inline CUDA_CALLABLE vec2 sample_unit_disk(uint32& state)
{
    float r = sqrt(randf(state));
    float theta = randf(state, 0.f, 2.f*M_PI);
    float x = r * cos(theta);
    float y = r * sin(theta);
    return vec2(x, y);
}

inline CUDA_CALLABLE vec3 sample_unit_sphere_surface(uint32& state)
{
    float phi = acos(1.f - 2.f * randf(state));
    float theta = randf(state, 0.f, 2.f*M_PI);
    float x = cos(theta) * sin(phi);
    float y = sin(theta) * sin(phi);
    float z = cos(phi);
    return vec3(x, y, z);
}

inline CUDA_CALLABLE vec3 sample_unit_sphere(uint32& state)
{
    float phi = acos(1.f  - 2.f * randf(state));
    float theta = randf(state, 0.f, 2.f*M_PI);
    float r = pow(randf(state), 1.f/3.f);
    float x = r * cos(theta) * sin(phi);
    float y = r * sin(theta) * sin(phi);
    float z = r * cos(phi);
    return vec3(x, y, z);
}

inline CUDA_CALLABLE vec3 sample_unit_hemisphere_surface(uint32& state)
{
    float phi = acos(1.f - randf(state));
    float theta = randf(state, 0.f, 2.f*M_PI);
    float x = cos(theta) * sin(phi);
    float y = sin(theta) * sin(phi);
    float z = cos(phi);
    return vec3(x, y, z);
}

inline CUDA_CALLABLE vec3 sample_unit_hemisphere(uint32& state)
{
    float phi = acos(1.f - randf(state));
    float theta = randf(state, 0.f, 2.f*M_PI);
    float r = pow(randf(state), 1.f/3.f);
    float x = r * cos(theta) * sin(phi);
    float y = r * sin(theta) * sin(phi);
    float z = r * cos(phi);
    return vec3(x, y, z);
}

inline CUDA_CALLABLE vec2 sample_unit_square(uint32& state)
{
    float x = randf(state) - 0.5f;
    float y = randf(state) - 0.5f;
    return vec2(x, y);
}

inline CUDA_CALLABLE vec3 sample_unit_cube(uint32& state)
{
    float x = randf(state) - 0.5f;
    float y = randf(state) - 0.5f;
    float z = randf(state) - 0.5f;
    return vec3(x, y, z);
}

inline CUDA_CALLABLE void adj_sample_cdf(uint32& state, const array_t<float>& cdf, uint32& adj_state, array_t<float>& adj_cdf, const int& adj_ret) {}
inline CUDA_CALLABLE void adj_sample_triangle(uint32& state, uint32& adj_state, const vec2& adj_ret) {}
inline CUDA_CALLABLE void adj_sample_unit_ring(uint32& state, uint32& adj_state, const vec2& adj_ret) {}
inline CUDA_CALLABLE void adj_sample_unit_disk(uint32& state, uint32& adj_state, const vec2& adj_ret) {}
inline CUDA_CALLABLE void adj_sample_unit_sphere_surface(uint32& state, uint32& adj_state, const vec3& adj_ret) {}
inline CUDA_CALLABLE void adj_sample_unit_sphere(uint32& state, uint32& adj_state, const vec3& adj_ret) {}
inline CUDA_CALLABLE void adj_sample_unit_hemisphere_surface(uint32& state, uint32& adj_state, const vec3& adj_ret) {}
inline CUDA_CALLABLE void adj_sample_unit_hemisphere(uint32& state, uint32& adj_state, const vec3& adj_ret) {}
inline CUDA_CALLABLE void adj_sample_unit_square(uint32& state, uint32& adj_state, const vec2& adj_ret) {}
inline CUDA_CALLABLE void adj_sample_unit_cube(uint32& state, uint32& adj_state, const vec3& adj_ret) {}

/*
 * log-gamma function to support some of these distributions. The
 * algorithm comes from SPECFUN by Shanjie Zhang and Jianming Jin and their
 * book "Computation of Special Functions", 1996, John Wiley & Sons, Inc.
 *
 * If random_loggam(k+1) is being used to compute log(k!) for an integer k, consider
 * using logfactorial(k) instead.
 */
inline CUDA_CALLABLE float random_loggam(float x)
{
    float x0, x2, lg2pi, gl, gl0;
    uint32 n;

    const float a[10] = {8.333333333333333e-02, -2.777777777777778e-03,
                        7.936507936507937e-04, -5.952380952380952e-04,
                        8.417508417508418e-04, -1.917526917526918e-03,
                        6.410256410256410e-03, -2.955065359477124e-02,
                        1.796443723688307e-01, -1.39243221690590e+00};

    if ((x == 1.0) || (x == 2.0))
    {
        return 0.0;
    }
    else if (x < 7.0)
    {
        n = uint32((7 - x));
    }
    else
    {
        n = 0;
    }

    x0 = x + float(n);
    x2 = (1.0 / x0) * (1.0 / x0);
    // log(2 * M_PI)
    lg2pi = 1.8378770664093453f;
    gl0 = a[9];
    for (int i = 8; i >= 0; i--)
    {
        gl0 *= x2;
        gl0 += a[i];
    }
    gl = gl0 / x0 + 0.5 * lg2pi + (x0 - 0.5) * log(x0) - x0;
    if (x < 7.0)
    {
        for (uint32 k = 1; k <= n; k++)
        {
            gl -= log(x0 - 1.0);
            x0 -= 1.0;
        }
    }
    return gl;
}

inline CUDA_CALLABLE uint32 random_poisson_mult(uint32& state, float lam) {
    uint32 X;
    float prod, U, enlam;

    enlam = exp(-lam);
    X = 0;
    prod = 1.0;

    while (1)
    {
        U = randf(state);
        prod *= U;
        if (prod > enlam)
        {
            X += 1;
        }
        else
        {
            return X;
        }
    }
}

/*
 * The transformed rejection method for generating Poisson random variables
 * W. Hoermann
 * Insurance: Mathematics and Economics 12, 39-45 (1993)
 */
inline CUDA_CALLABLE uint32 random_poisson(uint32& state, float lam)
{
    uint32 k;
    float U, V, slam, loglam, a, b, invalpha, vr, us;

    slam = sqrt(lam);
    loglam = log(lam);
    b = 0.931 + 2.53 * slam;
    a = -0.059 + 0.02483 * b;
    invalpha = 1.1239 + 1.1328 / (b - 3.4);
    vr = 0.9277 - 3.6224 / (b - 2.0);

    while (1)
    {
        U = randf(state) - 0.5;
        V = randf(state);
        us = 0.5 - abs(U);
        k = uint32(floor((2 * a / us + b) * U + lam + 0.43));
        if ((us >= 0.07) && (V <= vr))
        {
            return k;
        }
        if ((us < 0.013) && (V > us))
        {
            continue;
        }
        if ((log(V) + log(invalpha) - log(a / (us * us) + b)) <= (-lam + k * loglam - random_loggam(k + 1)))
        {
            return k;
        }
    }
}

/*
* Adpated from NumPy's implementation
* Warp's state variable is half the precision of NumPy's so
* poisson implementation uses half the precision used in NumPy's implementation
* both precisions appear to converge in the statistical limit
*/
inline CUDA_CALLABLE uint32 poisson(uint32& state, float lam)
{
    if (lam >= 10.f)
    {
        return random_poisson(state, lam);
    }
    else if (lam == 0.f)
    {
        return 0;
    }
    else
    {
        return random_poisson_mult(state, lam);
    }
}

inline CUDA_CALLABLE void adj_random_loggam(float x, float& adj_x, const float adj_ret) {}
inline CUDA_CALLABLE void random_poisson_mult(uint32& state, float lam, uint32& adj_state, float& adj_lam, const uint32& adj_ret) {}
inline CUDA_CALLABLE void adj_random_poisson(uint32& state, float lam, uint32& adj_state, float& adj_lam, const uint32& adj_ret) {}
inline CUDA_CALLABLE void adj_poisson(uint32& state, float lam, uint32& adj_state, float& adj_lam, const uint32& adj_ret) {}

} // namespace wp