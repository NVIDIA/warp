/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "warp.h"
#include "scan.h"

#define THRUST_IGNORE_CUB_VERSION_CHECK

#include <cub/device/device_scan.cuh>

template<typename T>
void scan_device(const T* values_in, T* values_out, int n, bool inclusive)
{
    ContextGuard guard(wp_cuda_context_get_current());

    cudaStream_t stream = static_cast<cudaStream_t>(wp_cuda_stream_get_current());

    // compute temporary memory required
	size_t scan_temp_size;
    if (inclusive) {
        check_cuda(cub::DeviceScan::InclusiveSum(NULL, scan_temp_size, values_in, values_out, n));
    } else {
        check_cuda(cub::DeviceScan::ExclusiveSum(NULL, scan_temp_size, values_in, values_out, n));
    }

    void* temp_buffer = wp_alloc_device(WP_CURRENT_CONTEXT, scan_temp_size);

    // scan
    if (inclusive) {
        check_cuda(cub::DeviceScan::InclusiveSum(temp_buffer, scan_temp_size, values_in, values_out, n, stream));
    } else {
        check_cuda(cub::DeviceScan::ExclusiveSum(temp_buffer, scan_temp_size, values_in, values_out, n, stream));
    }

    wp_free_device(WP_CURRENT_CONTEXT, temp_buffer);
}

template void scan_device(const int*, int*, int, bool);
template void scan_device(const float*, float*, int, bool);
