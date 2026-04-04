#ifndef STRATIFIED_MATH_H
#define STRATIFIED_MATH_H

#include <cuda_runtime.h>
#include <math.h>

struct StratifiedAccumulator {
    float value;      // Tier 1+: Visible State
    float phase_debt; // Tier 0: Foundational Remainder

    __device__ __forceinline__ void add(float increment) {
        // Capture the standard floating-point error
        float total = value + increment;
        float v_error = total - value;
        float i_error = increment - v_error;
        
        // Accumulate the debt in the Tier 0 field
        phase_debt += i_error;
        
        // Re-inject when the debt reaches machine epsilon
        if (fabsf(phase_debt) >= 1.1920929e-7f) {
            total += phase_debt;
            phase_debt = 0.0f;
        }
        
        value = total;
    }
};

#endif
