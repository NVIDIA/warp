#pragma once

#include "builtin.h"

namespace wp {

struct StratifiedAccumulator {
    float value = 0.0f;      
    float phase_debt = 0.0f; 

    CUDA_CALLABLE inline void add(float input) {
        // Standard Kahan-Babuska-Neumaier logic
        // Ensures the residual "debt" is reinjected into the next operation
        float t = value + input;
        if (wp::abs(value) >= wp::abs(input)) {
            phase_debt += (value - t) + input;
        } else {
            phase_debt += (input - t) + value;
        }
        value = t;
    }

    CUDA_CALLABLE inline float get() const {
        return value + phase_debt;
    }
};

} // namespace wp
