// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "warp.h"

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

    void* temp_buffer = wp_alloc_device(WP_CURRENT_CONTEXT, buff_size);

    check_cuda(
        cub::DeviceRunLengthEncode::Encode(
            temp_buffer, buff_size, values, run_values, run_lengths, run_count, n, stream
        )
    );

    wp_free_device(WP_CURRENT_CONTEXT, temp_buffer);
}

void wp_runlength_encode_int_device(
    uint64_t values, uint64_t run_values, uint64_t run_lengths, uint64_t run_count, int n
)
{
    return runlength_encode_device<int>(
        n, reinterpret_cast<const int*>(values), reinterpret_cast<int*>(run_values),
        reinterpret_cast<int*>(run_lengths), reinterpret_cast<int*>(run_count)
    );
}
