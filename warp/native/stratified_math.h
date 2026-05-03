#pragma once

namespace wp {

struct StratifiedAccumulator {
    float value;
    float residual;

    CUDA_CALLABLE StratifiedAccumulator()
        : value(0.0f)
        , residual(0.0f)
    {
    }

    CUDA_CALLABLE void add(float delta)
    {
        float y = delta - residual;
        float t = value + y;
        residual = (t - value) - y;
        value = t;
    }
};

CUDA_CALLABLE inline float stratified_analyze(float s, float delta)
{
    float angle = (s * delta) / 4.0f;

    if (angle > 1.5707963f)
        angle = 1.5707963f;
    if (angle < -1.5707963f)
        angle = -1.5707963f;

    // Using standard cosf which is available via builtin.h scope
    return s * ::cosf(angle);
}

}  // namespace wp
