#pragma once

namespace wp {

/// Kahan compensated summation accumulator for high-precision floating-point addition.
/// Tracks a residual correction term to minimize accumulated rounding error.
/// Note: Do not use in modules compiled with fast_math enabled.
/// The Kahan compensation relies on strict floating-point ordering.
struct StratifiedAccumulator {
    float value;
    float residual;

    /// Default constructor initializes the accumulator to zero.
    CUDA_CALLABLE inline StratifiedAccumulator()
        : value(0.0f)
        , residual(0.0f)
    {
    }

    /// Adds a value using Kahan summation to minimize floating-point error.
    /// @param delta The value to accumulate.
    CUDA_CALLABLE inline void add(float delta)
    {
        float y = delta - residual;
        float t = value + y;
        residual = (t - value) - y;
        value = t;
    }
};

/// Applies a resonance-based cosine scaling to a gradient value.
/// Computes a stratified angle from the input and delta, clamps it to [-pi/2, pi/2]
/// to prevent gradient sign inversion, and returns the cosine-scaled result.
/// @param s The input gradient value.
/// @param delta The scaling factor.
/// @return The cosine-scaled gradient value.
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
