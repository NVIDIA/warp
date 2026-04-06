// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stddef.h>

void radix_sort_reserve(void* context, int n, void** mem_out = NULL, size_t* size_out = NULL);
void radix_sort_release(void* context, void* stream);

void radix_sort_pairs_host(int* keys, int* values, int n);
void radix_sort_pairs_host(float* keys, int* values, int n);
void radix_sort_pairs_host(int64_t* keys, int* values, int n);
void radix_sort_pairs_device(void* context, int* keys, int* values, int n);
void radix_sort_pairs_device(void* context, float* keys, int* values, int n);
void radix_sort_pairs_device(void* context, int64_t* keys, int* values, int n);
void radix_sort_pairs_device(void* context, uint64_t* keys, int* values, int n);

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
