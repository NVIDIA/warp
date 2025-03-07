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

# pragma once
#include "array.h"

#ifndef M_PI_F
#define M_PI_F 3.14159265358979323846f
#endif

/*
 * Please first read the randf comment. randf returns values uniformly distributed in the range [0.f, 1.f - 2.^-24] in equal intervals of size 2.^-24.
 * randn computes sqrt(-2.f * log(x)). For this to return a real value, log(x) < 0.f (we exclude 0.f as a precaution) and therefore x < 1.f. 
 * For it to be finite, x > 0.f. So x must be in (0.f, 1.f). We define RANDN_EPSILON to be 2^-24 truncated to 5.96e-8f and add it to the range of randf,
 * giving the domain [RANDN_EPSILON, 1.f - 2.^-24 + RAND_EPSILON] which satisfies the requirement that x is in (0.f, 1.f).
 */

#ifndef RANDN_EPSILON
#define RANDN_EPSILON 5.96e-8f
#endif

namespace wp
{

/*
 * Mark Jarzynski and Marc Olano, Hash Functions for GPU Rendering, Journal of Computer
 * Graphics Techniques (JCGT), vol. 9, no. 3, 20–38, 2020
 */
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

inline CUDA_CALLABLE uint32 randu(uint32& state) { state = rand_pcg(state); return state; }
inline CUDA_CALLABLE uint32 randu(uint32& state, uint32 min, uint32 max) { state = rand_pcg(state); return state % (max - min) + min; }

/*
 * We want to ensure randf adheres to a uniform distribution over [0,1). The set of all possible float32 (IEEE 754 standard) values is not uniformly distributed however.
 * On the other hand, for a given sign and exponent, the mantissa of the float32 representation is uniformly distributed.
 * Fixing an exponent of -1, we can craft a uniform distribution using the sign bit and 23-bit mantissa that spans the domain [0, 1) in 2^24 equal intervals.
 * We can map 2^24 unique unsigned integers to these 2^24 intervals, so if our random number generator returns values in the range [0, 2^24) without bias,
 * we can ensure that our float distribution in the range [0, 1) is also without bias.
 * Our random number generator returns values in the range [0, 2^32), so we bit shift a random unsigned int 8 places, and then make the assumption that the remaining bit strings
 * are uniformly distributed. After dividing by 2.^24, randf returns values uniformly distributed in the range [0.f, 1.f - 2.^-24].
 */
inline CUDA_CALLABLE float randf(uint32& state) { state = rand_pcg(state); return (state >> 8) * (1.0f / 16777216.0f); }
inline CUDA_CALLABLE float randf(uint32& state, float min, float max) { return (max - min) * randf(state) + min; }

// Box-Muller method
inline CUDA_CALLABLE float randn(uint32& state) { return sqrt(-2.f * log(randf(state) + RANDN_EPSILON)) * cos(2.f * M_PI_F * randf(state)); }

inline CUDA_CALLABLE void adj_rand_init(int seed, int& adj_seed, float adj_ret) {}
inline CUDA_CALLABLE void adj_rand_init(int seed, int offset, int& adj_seed, int& adj_offset, float adj_ret) {}

inline CUDA_CALLABLE void adj_randi(uint32& state, uint32& adj_state, float adj_ret) {}
inline CUDA_CALLABLE void adj_randi(uint32& state, int min, int max, uint32& adj_state, int& adj_min, int& adj_max, float adj_ret) {}

inline CUDA_CALLABLE void adj_randu(uint32& state, uint32& adj_state, float adj_ret) {}
inline CUDA_CALLABLE void adj_randu(uint32& state, uint32 min, uint32 max, uint32& adj_state, uint32& adj_min, uint32& adj_max, float adj_ret) {}

inline CUDA_CALLABLE void adj_randf(uint32& state, uint32& adj_state, float adj_ret) {}
inline CUDA_CALLABLE void adj_randf(uint32& state, float min, float max, uint32& adj_state, float& adj_min, float& adj_max, float adj_ret) {}

inline CUDA_CALLABLE void adj_randn(uint32& state, uint32& adj_state, float adj_ret) {}

inline CUDA_CALLABLE int sample_cdf(uint32& state, const array_t<float>& cdf)
{
    float u = randf(state);
    return lower_bound<float>(cdf, u);
}

/*
 * uniform sampling methods for various geometries
 */
inline CUDA_CALLABLE vec2 sample_triangle(uint32& state)
{
    float r = sqrt(randf(state));
    float u = 1.f - r;
    float v = randf(state) * r;
    return vec2(u, v);
}

inline CUDA_CALLABLE vec2 sample_unit_ring(uint32& state)
{
    float theta = randf(state, 0.f, 2.f*M_PI_F);
    float x = cos(theta);
    float y = sin(theta);
    return vec2(x, y);
}

inline CUDA_CALLABLE vec2 sample_unit_disk(uint32& state)
{
    float r = sqrt(randf(state));
    float theta = randf(state, 0.f, 2.f*M_PI_F);
    float x = r * cos(theta);
    float y = r * sin(theta);
    return vec2(x, y);
}

inline CUDA_CALLABLE vec3 sample_unit_sphere_surface(uint32& state)
{
    float phi = acos(1.f - 2.f * randf(state));
    float theta = randf(state, 0.f, 2.f*M_PI_F);
    float x = cos(theta) * sin(phi);
    float y = sin(theta) * sin(phi);
    float z = cos(phi);
    return vec3(x, y, z);
}

inline CUDA_CALLABLE vec3 sample_unit_sphere(uint32& state)
{
    float phi = acos(1.f  - 2.f * randf(state));
    float theta = randf(state, 0.f, 2.f*M_PI_F);
    float r = pow(randf(state), 1.f/3.f);
    float x = r * cos(theta) * sin(phi);
    float y = r * sin(theta) * sin(phi);
    float z = r * cos(phi);
    return vec3(x, y, z);
}

inline CUDA_CALLABLE vec3 sample_unit_hemisphere_surface(uint32& state)
{
    float phi = acos(1.f - randf(state));
    float theta = randf(state, 0.f, 2.f*M_PI_F);
    float x = cos(theta) * sin(phi);
    float y = sin(theta) * sin(phi);
    float z = cos(phi);
    return vec3(x, y, z);
}

inline CUDA_CALLABLE vec3 sample_unit_hemisphere(uint32& state)
{
    float phi = acos(1.f - randf(state));
    float theta = randf(state, 0.f, 2.f*M_PI_F);
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

inline CUDA_CALLABLE vec4 sample_unit_hypercube(uint32& state)
{
    float a = randf(state) - 0.5f;
    float b = randf(state) - 0.5f;
    float c = randf(state) - 0.5f;
    float d = randf(state) - 0.5f;
    return vec4(a, b, c, d);
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
inline CUDA_CALLABLE void adj_sample_unit_hypercube(uint32& state, uint32& adj_state, const vec3& adj_ret) {}

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

    const float a[10] = {8.333333333333333e-02f, -2.777777777777778e-03f,
                        7.936507936507937e-04f, -5.952380952380952e-04f,
                        8.417508417508418e-04f, -1.917526917526918e-03f,
                        6.410256410256410e-03f, -2.955065359477124e-02f,
                        1.796443723688307e-01f, -1.39243221690590e+00f};

    if ((x == 1.f) || (x == 2.f))
    {
        return 0.f;
    }
    else if (x < 7.f)
    {
        n = uint32((7 - x));
    }
    else
    {
        n = 0;
    }

    x0 = x + float(n);
    x2 = (1.f / x0) * (1.f / x0);
    // log(2 * M_PI_F)
    lg2pi = 1.8378770664093453f;
    gl0 = a[9];
    for (int i = 8; i >= 0; i--)
    {
        gl0 *= x2;
        gl0 += a[i];
    }
    gl = gl0 / x0 + 0.5f * lg2pi + (x0 - 0.5f) * log(x0) - x0;
    if (x < 7.f)
    {
        for (uint32 k = 1; k <= n; k++)
        {
            gl -= log(x0 - 1.f);
            x0 -= 1.f;
        }
    }
    return gl;
}

inline CUDA_CALLABLE uint32 random_poisson_mult(uint32& state, float lam) {
    uint32 X;
    float prod, U, enlam;

    enlam = exp(-lam);
    X = 0;
    prod = 1.f;

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
    b = 0.931f + 2.53f * slam;
    a = -0.059f + 0.02483f * b;
    invalpha = 1.1239f + 1.1328f / (b - 3.4f);
    vr = 0.9277f - 3.6224f / (b - 2.f);

    while (1)
    {
        U = randf(state) - 0.5f;
        V = randf(state);
        us = 0.5f - abs(U);
        k = uint32(floor((2.f * a / us + b) * U + lam + 0.43f));
        if ((us >= 0.07f) && (V <= vr))
        {
            return k;
        }
        if ((us < 0.013f) && (V > us))
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
* Adapted from NumPy's implementation
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
