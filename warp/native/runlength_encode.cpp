// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "warp.h"

#include "apic.h"
#include "apic_internal.h"
#include "apic_types.h"

#include <cstdint>

// Record a host run-length-encode into the active APIC byte stream; returns
// true if recorded (and therefore should NOT execute now). Mirrors the sort
// try-record helpers: under CPU graph capture the host call is otherwise
// invisible to the byte stream, so replay would leave the outputs at their
// capture-time values.
static bool
apic_capture_runlength_encode(uint64_t values, uint64_t run_values, uint64_t run_lengths, uint64_t run_count, int n)
{
    APICState* state = wp_apic_get_recording_state();
    if (!state)
        return false;
    if (n < 0)
        return false;
    if (n == 0) {
        // Nothing to encode; write run_count=0 now so capture_save snapshots
        // the correct value and replay initializes from it without a separate op.
        if (run_count)
            *reinterpret_cast<int32_t*>(run_count) = 0;
        return true;
    }
    uint64_t io_bytes = static_cast<uint64_t>(n) * sizeof(int32_t);
    APICAddress values_addr = apic_resolve_host_ptr(state, values, io_bytes);
    APICAddress run_values_addr = apic_resolve_host_ptr(state, run_values, io_bytes);
    APICAddress run_lengths_addr = apic_resolve_host_ptr(state, run_lengths, io_bytes);
    APICAddress run_count_addr = apic_resolve_host_ptr(state, run_count, sizeof(int32_t));
    apic_record_runlength_encode(
        state, values_addr.region_id, values_addr.offset, run_values_addr.region_id, run_values_addr.offset,
        run_lengths_addr.region_id, run_lengths_addr.offset, run_count_addr.region_id, run_count_addr.offset,
        static_cast<uint32_t>(n)
    );
    return true;
}

template <typename T>
void runlength_encode_host(int n, const T* values, T* run_values, int* run_lengths, int* run_count)
{
    if (n == 0) {
        *run_count = 0;
        return;
    }

    const T* end = values + n;

    *run_count = 1;
    *run_lengths = 1;
    *run_values = *values;

    while (++values != end) {
        if (*values == *run_values) {
            ++*run_lengths;
        } else {
            ++*run_count;
            *(++run_lengths) = 1;
            *(++run_values) = *values;
        }
    }
}

void wp_runlength_encode_int_host(uint64_t values, uint64_t run_values, uint64_t run_lengths, uint64_t run_count, int n)
{
    if (apic_capture_runlength_encode(values, run_values, run_lengths, run_count, n))
        return;
    runlength_encode_host<int>(
        n, reinterpret_cast<const int*>(values), reinterpret_cast<int*>(run_values),
        reinterpret_cast<int*>(run_lengths), reinterpret_cast<int*>(run_count)
    );
}

#if !WP_ENABLE_CUDA
void wp_runlength_encode_int_device(
    uint64_t values, uint64_t run_values, uint64_t run_lengths, uint64_t run_count, int n
)
{
}
#endif
