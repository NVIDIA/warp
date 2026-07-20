// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "warp.h"

#include "apic.h"
#include "apic_internal.h"

namespace {

// Specialized accumulation functions for common type sizes
template <int N, typename T> void fixed_len_sum(const T* val, T* sum, int value_size)
{
    for (int i = 0; i < N; ++i, ++val, ++sum) {
        *sum += *val;
    }
}

template <typename T> void dyn_len_sum(const T* val, T* sum, int value_size)
{
    for (int i = 0; i < value_size; ++i, ++val, ++sum) {
        *sum += *val;
    }
}

template <int N, typename T> void fixed_len_inner(const T* a, const T* b, T* dot, int value_size)
{
    for (int i = 0; i < N; ++i, ++a, ++b) {
        *dot += *a * *b;
    }
}

template <typename T> void dyn_len_inner(const T* a, const T* b, T* dot, int value_size)
{
    for (int i = 0; i < value_size; ++i, ++a, ++b) {
        *dot += *a * *b;
    }
}

static bool apic_capture_reduction_host(
    uint64_t input_a,
    uint64_t input_b,
    uint64_t output,
    int count,
    int input_a_stride,
    int input_b_stride,
    int type_len,
    uint8_t kind,
    uint8_t dtype
)
{
    APICState* state = wp_apic_get_recording_state();
    if (!state)
        return false;

    if (!apic_reduction_metadata_valid(
            input_a, input_b, output, count, input_a_stride, input_b_stride, type_len, kind, dtype
        )) {
        // Poison the byte stream so capture validation fails closed. Returning
        // false here would execute the malformed reduction eagerly.
        apic_record_reduction(
            state, -1, 0, -1, 0, -1, 0, static_cast<uint32_t>(count), input_a_stride, input_b_stride, type_len, kind,
            dtype
        );
        return true;
    }

    size_t scalar_size = apic_type_size(dtype);
    size_t input_a_bytes = apic_reduction_input_bytes(count, input_a_stride, type_len, scalar_size);
    size_t input_b_bytes
        = kind == APIC_REDUCTION_INNER ? apic_reduction_input_bytes(count, input_b_stride, type_len, scalar_size) : 0;
    size_t output_bytes = kind == APIC_REDUCTION_SUM ? static_cast<size_t>(type_len) * scalar_size : scalar_size;
    if (input_a_bytes == 0 || output_bytes == 0 || (kind == APIC_REDUCTION_INNER && input_b_bytes == 0)) {
        apic_record_reduction(
            state, -1, 0, -1, 0, -1, 0, static_cast<uint32_t>(count), input_a_stride, input_b_stride, type_len, kind,
            dtype
        );
        return true;
    }

    APICAddress input_a_addr = apic_resolve_host_ptr(state, input_a, input_a_bytes);
    APICAddress input_b_addr
        = kind == APIC_REDUCTION_INNER ? apic_resolve_host_ptr(state, input_b, input_b_bytes) : APICAddress { -1, 0 };
    APICAddress output_addr = apic_resolve_host_ptr(state, output, output_bytes);
    apic_record_reduction(
        state, input_a_addr.region_id, input_a_addr.offset, input_b_addr.region_id, input_b_addr.offset,
        output_addr.region_id, output_addr.offset, static_cast<uint32_t>(count), input_a_stride, input_b_stride,
        type_len, kind, dtype
    );
    return true;
}

}  // namespace

template <typename T>
void array_inner_host(
    const T* ptr_a, const T* ptr_b, T* ptr_out, int count, int byte_stride_a, int byte_stride_b, int type_length
)
{
    assert((byte_stride_a % sizeof(T)) == 0);
    assert((byte_stride_b % sizeof(T)) == 0);
    const int stride_a = byte_stride_a / sizeof(T);
    const int stride_b = byte_stride_b / sizeof(T);

    void (*inner_func)(const T*, const T*, T*, int);
    switch (type_length) {
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
    for (int i = 0; i < count; ++i) {
        inner_func(ptr_a + i * stride_a, ptr_b + i * stride_b, ptr_out, type_length);
    }
}

template <typename T> void array_sum_host(const T* ptr_a, T* ptr_out, int count, int byte_stride, int type_length)
{
    assert((byte_stride % sizeof(T)) == 0);
    const int stride = byte_stride / sizeof(T);

    void (*accumulate_func)(const T*, T*, int);
    switch (type_length) {
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

    memset(ptr_out, 0, sizeof(T) * type_length);
    for (int i = 0; i < count; ++i)
        accumulate_func(ptr_a + i * stride, ptr_out, type_length);
}

void wp_array_inner_float_host(
    uint64_t a, uint64_t b, uint64_t out, int count, int byte_stride_a, int byte_stride_b, int type_length
)
{
    if (apic_capture_reduction_host(
            a, b, out, count, byte_stride_a, byte_stride_b, type_length, APIC_REDUCTION_INNER, APIC_TYPE_FLOAT32
        ))
        return;

    const float* ptr_a = (const float*)(a);
    const float* ptr_b = (const float*)(b);
    float* ptr_out = (float*)(out);

    array_inner_host(ptr_a, ptr_b, ptr_out, count, byte_stride_a, byte_stride_b, type_length);
}

void wp_array_inner_double_host(
    uint64_t a, uint64_t b, uint64_t out, int count, int byte_stride_a, int byte_stride_b, int type_length
)
{
    if (apic_capture_reduction_host(
            a, b, out, count, byte_stride_a, byte_stride_b, type_length, APIC_REDUCTION_INNER, APIC_TYPE_FLOAT64
        ))
        return;

    const double* ptr_a = (const double*)(a);
    const double* ptr_b = (const double*)(b);
    double* ptr_out = (double*)(out);

    array_inner_host(ptr_a, ptr_b, ptr_out, count, byte_stride_a, byte_stride_b, type_length);
}

void wp_array_sum_float_host(uint64_t a, uint64_t out, int count, int byte_stride_a, int type_length)
{
    if (apic_capture_reduction_host(
            a, 0, out, count, byte_stride_a, 0, type_length, APIC_REDUCTION_SUM, APIC_TYPE_FLOAT32
        ))
        return;

    const float* ptr_a = (const float*)(a);
    float* ptr_out = (float*)(out);
    array_sum_host(ptr_a, ptr_out, count, byte_stride_a, type_length);
}

void wp_array_sum_double_host(uint64_t a, uint64_t out, int count, int byte_stride_a, int type_length)
{
    if (apic_capture_reduction_host(
            a, 0, out, count, byte_stride_a, 0, type_length, APIC_REDUCTION_SUM, APIC_TYPE_FLOAT64
        ))
        return;

    const double* ptr_a = (const double*)(a);
    double* ptr_out = (double*)(out);
    array_sum_host(ptr_a, ptr_out, count, byte_stride_a, type_length);
}

#if !WP_ENABLE_CUDA
void wp_array_inner_float_device(
    uint64_t a, uint64_t b, uint64_t out, int count, int byte_stride_a, int byte_stride_b, int type_length
)
{
}

void wp_array_inner_double_device(
    uint64_t a, uint64_t b, uint64_t out, int count, int byte_stride_a, int byte_stride_b, int type_length
)
{
}

void wp_array_sum_float_device(uint64_t a, uint64_t out, int count, int byte_stride_a, int type_length) { }

void wp_array_sum_double_device(uint64_t a, uint64_t out, int count, int byte_stride_a, int type_length) { }
#endif
