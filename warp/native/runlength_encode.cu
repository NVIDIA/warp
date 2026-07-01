// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "warp.h"

#include "apic.h"
#include "apic_internal.h"
#include "cuda_util.h"

#define THRUST_IGNORE_CUB_VERSION_CHECK
#include <cub/device/device_run_length_encode.cuh>

template <typename T>
void runlength_encode_device(int n, const T* values, T* run_values, int* run_lengths, int* run_count)
{
    ContextGuard guard(wp_cuda_context_get_current());
    cudaStream_t stream = static_cast<cudaStream_t>(wp_cuda_stream_get_current());

    size_t buff_size = 0;
    check_cuda(
        cub::DeviceRunLengthEncode::Encode(nullptr, buff_size, values, run_values, run_lengths, run_count, n, stream)
    );

    void* temp_buffer = wp_alloc_device(WP_CURRENT_CONTEXT, buff_size, "(native:runlength_encode)");

    check_cuda(
        cub::DeviceRunLengthEncode::Encode(
            temp_buffer, buff_size, values, run_values, run_lengths, run_count, n, stream
        )
    );

    wp_free_device(WP_CURRENT_CONTEXT, temp_buffer);
}

// Record-and-execute a run-length-encode under CUDA APIC capture: record params
// into the byte stream, then fall through so the live encode issues onto the
// captured stream. Mirror of apic_capture_runlength_encode in
// runlength_encode.cpp, device-scoped and non-skipping. The CUDA path requires
// an explicit device run_count (the Python wrapper rejects the host-return form
// under capture), so run_count is always a real device pointer. No-op outside a
// CUDA APIC capture.
static void apic_capture_runlength_encode_device(
    uint64_t values, uint64_t run_values, uint64_t run_lengths, uint64_t run_count, int n
)
{
    APICState* state = wp_apic_get_cuda_recording_state();
    if (!state || n <= 0)
        return;
    uint64_t io_bytes = static_cast<uint64_t>(n) * sizeof(int32_t);
    APICAddress values_addr = apic_resolve_live_ptr(state, values, io_bytes);
    APICAddress run_values_addr = apic_resolve_live_ptr(state, run_values, io_bytes);
    APICAddress run_lengths_addr = apic_resolve_live_ptr(state, run_lengths, io_bytes);
    APICAddress run_count_addr = apic_resolve_live_ptr(state, run_count, sizeof(int32_t));
    apic_record_runlength_encode(
        state, values_addr.region_id, values_addr.offset, run_values_addr.region_id, run_values_addr.offset,
        run_lengths_addr.region_id, run_lengths_addr.offset, run_count_addr.region_id, run_count_addr.offset,
        static_cast<uint32_t>(n)
    );
}

void wp_runlength_encode_int_device(
    uint64_t values, uint64_t run_values, uint64_t run_lengths, uint64_t run_count, int n
)
{
    // A negative count is invalid; reject it before the device fallback so a bad
    // length can't drive CUB DeviceRunLengthEncode with a negative num_items.
    // Mirrors wp_runlength_encode_int_host().
    if (n < 0)
        return;
    apic_capture_runlength_encode_device(values, run_values, run_lengths, run_count, n);
    return runlength_encode_device<int>(
        n, reinterpret_cast<const int*>(values), reinterpret_cast<int*>(run_values),
        reinterpret_cast<int*>(run_lengths), reinterpret_cast<int*>(run_count)
    );
}
