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

#pragma once

#include <stddef.h>

void radix_sort_reserve(void* context, int n, void** mem_out=NULL, size_t* size_out=NULL);
void radix_sort_pairs_host(int* keys, int* values, int n);
void radix_sort_pairs_host(float* keys, int* values, int n);
void radix_sort_pairs_host(int64_t* keys, int* values, int n);
void radix_sort_pairs_device(void* context, int* keys, int* values, int n);
void radix_sort_pairs_device(void* context, float* keys, int* values, int n);
void radix_sort_pairs_device(void* context, int64_t* keys, int* values, int n);

void segmented_sort_pairs_host(float* keys, int* values, int n, int* segment_start_indices, int* segment_end_indices, int num_segments);
void segmented_sort_pairs_device(void* context, float* keys, int* values, int n, int* segment_start_indices, int* segment_end_indices, int num_segments);
void segmented_sort_pairs_host(void* context, int* keys, int* values, int n, int* segment_start_indices, int* segment_end_indices, int num_segments);
void segmented_sort_pairs_device(void* context, int* keys, int* values, int n, int* segment_start_indices, int* segment_end_indices, int num_segments);
