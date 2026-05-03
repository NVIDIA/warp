#pragma once

#include "builtin.h"

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
    // Strategic resonance-based projection
    float angle = (s * delta) / 4.0f;

    // Clamp to [ -pi/2, pi/2 ] to prevent gradient sign inversion
    if (angle > 1.5707963f)
        angle = 1.5707963f;
    if (angle < -1.5707963f)
        angle = -1.5707963f;

    // Using standard cosf with explicit global scope for portability
    return s * ::cosf(angle);
}

}  // namespace wp
