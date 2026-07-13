// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <stddef.h>

void radix_sort_reserve(
    void* context, int n, void** mem_out = NULL, size_t* size_out = NULL, int begin_bit = 0, int end_bit = 32
);
void radix_sort_reserve_u64(
    void* context, int n, void** mem_out = NULL, size_t* size_out = NULL, int begin_bit = 0, int end_bit = 64
);
void radix_sort_release(void* context, void* stream);

void radix_sort_pairs_host(int* keys, int* values, int n, int begin_bit = 0, int end_bit = 32);
void radix_sort_pairs_host(uint32_t* keys, int* values, int n, int begin_bit = 0, int end_bit = 32);
void radix_sort_pairs_host(float* keys, int* values, int n, int begin_bit = 0, int end_bit = 32);
void radix_sort_pairs_host(int64_t* keys, int* values, int n, int begin_bit = 0, int end_bit = 64);
void radix_sort_pairs_host(uint64_t* keys, int* values, int n, int begin_bit = 0, int end_bit = 64);
void radix_sort_pairs_host(double* keys, int* values, int n, int begin_bit = 0, int end_bit = 64);
void radix_sort_pairs_device(void* context, int* keys, int* values, int n, int begin_bit = 0, int end_bit = 32);
void radix_sort_pairs_device(void* context, uint32_t* keys, int* values, int n, int begin_bit = 0, int end_bit = 32);
void radix_sort_pairs_device(void* context, float* keys, int* values, int n, int begin_bit = 0, int end_bit = 32);
void radix_sort_pairs_device(void* context, int64_t* keys, int* values, int n, int begin_bit = 0, int end_bit = 64);
void radix_sort_pairs_device(void* context, uint64_t* keys, int* values, int n, int begin_bit = 0, int end_bit = 64);
void radix_sort_pairs_device(void* context, double* keys, int* values, int n, int begin_bit = 0, int end_bit = 64);

void segmented_sort_pairs_host(
    float* keys, int* values, int n, int* segment_start_indices, int* segment_end_indices, int num_segments
);
void segmented_sort_pairs_device(
    void* context,
    float* keys,
    int* values,
    int n,
    int* segment_start_indices,
    int* segment_end_indices,
    int num_segments
);
void segmented_sort_pairs_host(
    int* keys, int* values, int n, int* segment_start_indices, int* segment_end_indices, int num_segments
);
void segmented_sort_pairs_device(
    void* context, int* keys, int* values, int n, int* segment_start_indices, int* segment_end_indices, int num_segments
);
