#pragma once

namespace wp {

// Note: Do not use in modules compiled with fast_math enabled.
// The Kahan compensation relies on strict floating-point ordering.
struct StratifiedAccumulator {
    float value;
    float residual;

    CUDA_CALLABLE inline StratifiedAccumulator()
        : value(0.0f)
        , residual(0.0f)
    {
    }

    CUDA_CALLABLE inline void add(float delta)
    {
        float y = delta - residual;
        float t = value + y;
        residual = (t - value) - y;
        value = t;
    }
};

CUDA_CALLABLE inline float stratified_analyze(float s, float delta)
{
    // Strategic resonance-based scaling
    float angle = (s * delta) / 4.0f;

    // Clamp to [ -pi/2, pi/2 ] to prevent gradient sign inversion
    if (angle > 1.5707963f)
        angle = 1.5707963f;
    if (angle < -1.5707963f)
        angle = -1.5707963f;

    // Global scope cosf is provided by the compiler/environment
    // where this header is included (via builtin.h)
    return s * ::cosf(angle);
}

}  // namespace wp
