#pragma once
#include <warp/native/builtin.h>

namespace wp {

struct StratifiedAccumulator {
    float value = 0.0f;      // The running total
    float phase_debt = 0.0f; // The compensation term (low-order bits)

    WP_DEVICE inline void add(float input) {
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

    WP_DEVICE inline float get() const {
        return value + phase_debt;
    }
};

} // namespace wp
