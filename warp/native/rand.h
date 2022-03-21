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

} // namespace wp