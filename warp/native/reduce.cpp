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

namespace
{

// Specialized accumulation functions for common type sizes
template <int N, typename T> void fixed_len_sum(const T *val, T *sum, int value_size)
{
    for (int i = 0; i < N; ++i, ++val, ++sum)
    {
        *sum += *val;
    }
}

template <typename T> void dyn_len_sum(const T *val, T *sum, int value_size)
{
    for (int i = 0; i < value_size; ++i, ++val, ++sum)
    {
        *sum += *val;
    }
}

template <int N, typename T> void fixed_len_inner(const T *a, const T *b, T *dot, int value_size)
{
    for (int i = 0; i < N; ++i, ++a, ++b)
    {
        *dot += *a * *b;
    }
}

template <typename T> void dyn_len_inner(const T *a, const T *b, T *dot, int value_size)
{
    for (int i = 0; i < value_size; ++i, ++a, ++b)
    {
        *dot += *a * *b;
    }
}

} // namespace

template <typename T>
void array_inner_host(const T *ptr_a, const T *ptr_b, T *ptr_out, int count, int byte_stride_a, int byte_stride_b,
                      int type_length)
{
    assert((byte_stride_a % sizeof(T)) == 0);
    assert((byte_stride_b % sizeof(T)) == 0);
    const int stride_a = byte_stride_a / sizeof(T);
    const int stride_b = byte_stride_b / sizeof(T);

    void (*inner_func)(const T *, const T *, T *, int);
    switch (type_length)
    {
    case 1:
        inner_func = fixed_len_inner<1, T>;
        break;
    case 2:
        inner_func = fixed_len_inner<2, T>;
        break;
    case 3:
        inner_func = fixed_len_inner<3, T>;
        break;
    case 4:
        inner_func = fixed_len_inner<4, T>;
        break;
    default:
        inner_func = dyn_len_inner<T>;
    }

    *ptr_out = 0.0f;
    for (int i = 0; i < count; ++i)
    {
        inner_func(ptr_a + i * stride_a, ptr_b + i * stride_b, ptr_out, type_length);
    }
}

template <typename T> void array_sum_host(const T *ptr_a, T *ptr_out, int count, int byte_stride, int type_length)
{
    assert((byte_stride % sizeof(T)) == 0);
    const int stride = byte_stride / sizeof(T);

    void (*accumulate_func)(const T *, T *, int);
    switch (type_length)
    {
    case 1:
        accumulate_func = fixed_len_sum<1, T>;
        break;
    case 2:
        accumulate_func = fixed_len_sum<2, T>;
        break;
    case 3:
        accumulate_func = fixed_len_sum<3, T>;
        break;
    case 4:
        accumulate_func = fixed_len_sum<4, T>;
        break;
    default:
        accumulate_func = dyn_len_sum<T>;
    }

    memset(ptr_out, 0, sizeof(T)*type_length);
    for (int i = 0; i < count; ++i)
        accumulate_func(ptr_a + i * stride, ptr_out, type_length);
}

void array_inner_float_host(uint64_t a, uint64_t b, uint64_t out, int count, int byte_stride_a, int byte_stride_b,
                            int type_length)
{
    const float *ptr_a = (const float *)(a);
    const float *ptr_b = (const float *)(b);
    float *ptr_out = (float *)(out);

    array_inner_host(ptr_a, ptr_b, ptr_out, count, byte_stride_a, byte_stride_b, type_length);
}

void array_inner_double_host(uint64_t a, uint64_t b, uint64_t out, int count, int byte_stride_a, int byte_stride_b,
                             int type_length)
{
    const double *ptr_a = (const double *)(a);
    const double *ptr_b = (const double *)(b);
    double *ptr_out = (double *)(out);

    array_inner_host(ptr_a, ptr_b, ptr_out, count, byte_stride_a, byte_stride_b, type_length);
}

void array_sum_float_host(uint64_t a, uint64_t out, int count, int byte_stride_a, int type_length)
{
    const float *ptr_a = (const float *)(a);
    float *ptr_out = (float *)(out);
    array_sum_host(ptr_a, ptr_out, count, byte_stride_a, type_length);
}

void array_sum_double_host(uint64_t a, uint64_t out, int count, int byte_stride_a, int type_length)
{
    const double *ptr_a = (const double *)(a);
    double *ptr_out = (double *)(out);
    array_sum_host(ptr_a, ptr_out, count, byte_stride_a, type_length);
}

#if !WP_ENABLE_CUDA
void array_inner_float_device(uint64_t a, uint64_t b, uint64_t out, int count, int byte_stride_a, int byte_stride_b,
                              int type_length)
{
}

void array_inner_double_device(uint64_t a, uint64_t b, uint64_t out, int count, int byte_stride_a, int byte_stride_b,
                               int type_length)
{
}

void array_sum_float_device(uint64_t a, uint64_t out, int count, int byte_stride_a, int type_length)
{
}

void array_sum_double_device(uint64_t a, uint64_t out, int count, int byte_stride_a, int type_length)
{
}
#endif
