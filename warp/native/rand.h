# pragma once

namespace wp
{

inline CUDA_CALLABLE uint32 rand_pcg(uint32 state)
{
    uint32 b = state * 747796405u + 2891336453u;
    uint32 c = ((b >> ((b >> 28u) + 4u)) ^ b) * 277803737u;
    return (c >> 22u) ^ c;
}

inline CUDA_CALLABLE uint32 rand_init(uint32 seed, uint32 offset) { return rand_pcg(seed + rand_pcg(offset)); }

inline CUDA_CALLABLE uint32 randi(uint32 state) { return rand_pcg(state); }
inline CUDA_CALLABLE uint32 randi(uint32& state, uint32 min, uint32 max) { state = rand_pcg(state); return state % (max - min) + min; }

inline CUDA_CALLABLE float randf(uint32& state) { state = rand_pcg(state); return float(state) / UINT32_MAX; }
inline CUDA_CALLABLE float randf(uint32& state, float min, float max) { state = rand_pcg(state); return (max - min) * randf(state) + min; }

} // namespace wp