// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "warp.h"

#include "scan.h"

template <typename T>
void scan_host(
    const T* values_in, T* values_out, int n, int in_byte_stride, int out_byte_stride, int type_length, bool inclusive
)
{
    assert((in_byte_stride % sizeof(T)) == 0);
    assert((out_byte_stride % sizeof(T)) == 0);

    if (n <= 0)
        return;

    const int in_stride = in_byte_stride / sizeof(T);
    const int out_stride = out_byte_stride / sizeof(T);

    for (int k = 0; k < type_length; ++k) {
        T sum = T(0);

        for (int i = 0; i < n; ++i) {
            const T value = values_in[i * in_stride + k];

            if (inclusive) {
                sum += value;
                values_out[i * out_stride + k] = sum;
            } else {
                values_out[i * out_stride + k] = sum;
                sum += value;
            }
        }
    }
}

template <typename T> void scan_host(const T* values_in, T* values_out, int n, bool inclusive)
{
    scan_host(values_in, values_out, n, sizeof(T), sizeof(T), 1, inclusive);
}

template void scan_host(const int*, int*, int, bool);
template void scan_host(const int64_t*, int64_t*, int, bool);
template void scan_host(const float*, float*, int, bool);
template void scan_host(const double*, double*, int, bool);

template void scan_host(const int*, int*, int, int, int, int, bool);
template void scan_host(const int64_t*, int64_t*, int, int, int, int, bool);
template void scan_host(const float*, float*, int, int, int, int, bool);
template void scan_host(const double*, double*, int, int, int, int, bool);
