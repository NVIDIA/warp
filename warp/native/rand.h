/** Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

# pragma once

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

inline CUDA_CALLABLE int sample_cdf(uint32& state, const array_t<float>& cdf, int n)
{
    float u = randf(state);
    int i = 0, j = n, m = 0;

    // assume cdf is ordered, binary search closest value, return index
    while (i < j)
    {
        m = (i + j) / 2;
        if (u == cdf[m])
            return m;
        
        if (u < cdf[m])
        {
            if (m > 0 && u > cdf[m - 1])
                return u - cdf[m - 1] >= cdf[m] - u ? m : m - 1;
            j = m;
        }
        else
        {
            if (m < n - 1 && u < cdf[m + 1])
                return u - cdf[m] >= cdf[m + 1] - u ? m + 1 : m;
            i = m + 1;
        }
    }
    return m;
}

inline CUDA_CALLABLE vec3 sample_unit_sphere_surface(uint32& state)
{
    float u = randf(state);
    float phi = acos(1.0 - 2.0 * u);
    float theta = randf(state, 0.0, 2.0*M_PI);
    float x = cos(theta) * sin(phi);
    float y = sin(theta) * sin(phi);
    float z = cos(phi);
    return vec3(x, y, z);
}

inline CUDA_CALLABLE vec3 sample_unit_sphere(uint32& state)
{
    float u = randf(state);
    float phi = acos(1.0  - 2.0 * u);
    float theta = randf(state, 0.0, 2.0*M_PI);
    float x = cos(theta) * sin(phi);
    float y = sin(theta) * sin(phi);
    float z = cos(phi);
    return vec3(x, y, z);
}

inline CUDA_CALLABLE void adj_sample_cdf(uint32& state, const array_t<float>& cdf, int n, uint32& adj_state, const array_t<float>& adj_cdf, int adj_int) {}
inline CUDA_CALLABLE void adj_sample_unit_sphere_surface(uint32& state, uint32& adj_state) {}

} // namespace wp