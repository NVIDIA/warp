/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cstdint>

template <typename T>
void runlength_encode_host(int n,
                           const T *values,
                           T *run_values,
                           int *run_lengths,
                           int *run_count)
{
    if (n == 0)
    {
        *run_count = 0;
        return;
    }

    const T *end = values + n;

    *run_count = 1;
    *run_lengths = 1;
    *run_values = *values;

    while (++values != end)
    {
        if (*values == *run_values)
        {
            ++*run_lengths;
        }
        else
        {
            ++*run_count;
            *(++run_lengths) = 1;
            *(++run_values) = *values;
        }
    }
}

void wp_runlength_encode_int_host(
    uint64_t values,
    uint64_t run_values,
    uint64_t run_lengths,
    uint64_t run_count,
    int n)
{
    runlength_encode_host<int>(n,
                               reinterpret_cast<const int *>(values),
                               reinterpret_cast<int *>(run_values),
                               reinterpret_cast<int *>(run_lengths),
                               reinterpret_cast<int *>(run_count));
}

#if !WP_ENABLE_CUDA
void wp_runlength_encode_int_device(
    uint64_t values,
    uint64_t run_values,
    uint64_t run_lengths,
    uint64_t run_count,
    int n)
{
}
#endif
