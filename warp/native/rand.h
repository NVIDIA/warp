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

inline CUDA_CALLABLE float randf(uint32& state) { state = rand_pcg(state); return float(state) / 0xffffffff; }
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

} // namespace wp