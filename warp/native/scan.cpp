// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "warp.h"

#include "scan.h"

#include <cstring>
#include <numeric>

template <typename T> void scan_host(const T* values_in, T* values_out, int n, bool inclusive)
{
    if (n <= 0)
        return;

    static void* scan_temp_memory = NULL;
    static size_t scan_temp_max_size = 0;

    // compute temporary memory required
    if (!inclusive && n > scan_temp_max_size) {
        wp_free_host(scan_temp_memory);
        scan_temp_memory = wp_alloc_host(sizeof(T) * n, "(native:scan)");
        scan_temp_max_size = n;
    }

    T* result = inclusive ? values_out : static_cast<T*>(scan_temp_memory);

    // scan
    std::partial_sum(values_in, values_in + n, result);
    if (!inclusive) {
        values_out[0] = (T)0;
        // Internal scratch -> output copy. Use plain memcpy so it doesn't
        // interact with APIC recording (the scan itself isn't a captured op;
        // its memcpys are implementation details).
        if (n > 1)
            std::memcpy(values_out + 1, result, sizeof(T) * (n - 1));
    }
}

template void scan_host(const int*, int*, int, bool);
template void scan_host(const float*, float*, int, bool);
