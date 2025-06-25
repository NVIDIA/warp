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

#include "scan.h"

#include <numeric>

template<typename T>
void scan_host(const T* values_in, T* values_out, int n, bool inclusive)
{
    static void* scan_temp_memory = NULL;
    static size_t scan_temp_max_size = 0;

    // compute temporary memory required
    if (!inclusive && n > scan_temp_max_size)
    {
	    wp_free_host(scan_temp_memory);
        scan_temp_memory = wp_alloc_host(sizeof(T) * n);
        scan_temp_max_size = n;
    }

    T* result = inclusive ? values_out : static_cast<T*>(scan_temp_memory);

    // scan
    std::partial_sum(values_in, values_in + n, result);
    if (!inclusive) {
        values_out[0] = (T)0;
        wp_memcpy_h2h(values_out + 1, result, sizeof(T) * (n - 1));
    }
}

template void scan_host(const int*, int*, int, bool);
template void scan_host(const float*, float*, int, bool);
