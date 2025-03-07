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
#include "cuda_util.h"
#include "sort.h"

#define THRUST_IGNORE_CUB_VERSION_CHECK

#include <cub/cub.cuh>

#include <map>

// temporary buffer for radix sort
struct RadixSortTemp
{
    void* mem = NULL;
    size_t size = 0;
};

// map temp buffers to CUDA contexts
static std::map<void*, RadixSortTemp> g_radix_sort_temp_map;


template <typename KeyType>
void radix_sort_reserve_internal(void* context, int n, void** mem_out, size_t* size_out)
{
    ContextGuard guard(context);

    cub::DoubleBuffer<KeyType> d_keys;
	cub::DoubleBuffer<int> d_values;

    // compute temporary memory required
	size_t sort_temp_size;
    check_cuda(cub::DeviceRadixSort::SortPairs(
        NULL,
        sort_temp_size,
        d_keys,
        d_values,
        n, 0, sizeof(KeyType)*8,
        (cudaStream_t)cuda_stream_get_current()));

    if (!context)
        context = cuda_context_get_current();

    RadixSortTemp& temp = g_radix_sort_temp_map[context];

    if (sort_temp_size > temp.size)
    {
	    free_device(WP_CURRENT_CONTEXT, temp.mem);
        temp.mem = alloc_device(WP_CURRENT_CONTEXT, sort_temp_size);
        temp.size = sort_temp_size;
    }
    
    if (mem_out)
        *mem_out = temp.mem;
    if (size_out)
        *size_out = temp.size;
}

void radix_sort_reserve(void* context, int n, void** mem_out, size_t* size_out)
{
    radix_sort_reserve_internal<int>(context, n, mem_out, size_out);
}

template <typename KeyType>
void radix_sort_pairs_device(void* context, KeyType* keys, int* values, int n)
{
    ContextGuard guard(context);

    cub::DoubleBuffer<KeyType> d_keys(keys, keys + n);
	cub::DoubleBuffer<int> d_values(values, values + n);

    RadixSortTemp temp;
    radix_sort_reserve_internal<KeyType>(WP_CURRENT_CONTEXT, n, &temp.mem, &temp.size);

    // sort
    check_cuda(cub::DeviceRadixSort::SortPairs(
        temp.mem,
        temp.size,
        d_keys, 
        d_values, 
        n, 0, sizeof(KeyType)*8, 
        (cudaStream_t)cuda_stream_get_current()));

	if (d_keys.Current() != keys)
		memcpy_d2d(WP_CURRENT_CONTEXT, keys, d_keys.Current(), sizeof(KeyType)*n);

	if (d_values.Current() != values)
		memcpy_d2d(WP_CURRENT_CONTEXT, values, d_values.Current(), sizeof(int)*n);
}

void radix_sort_pairs_device(void* context, int* keys, int* values, int n)
{
    radix_sort_pairs_device<int>(context, keys, values, n);
}

void radix_sort_pairs_device(void* context, float* keys, int* values, int n)
{
    radix_sort_pairs_device<float>(context, keys, values, n);
}

void radix_sort_pairs_device(void* context, int64_t* keys, int* values, int n)
{
    radix_sort_pairs_device<int64_t>(context, keys, values, n);
}

void radix_sort_pairs_int_device(uint64_t keys, uint64_t values, int n)
{
    radix_sort_pairs_device(
        WP_CURRENT_CONTEXT,
        reinterpret_cast<int *>(keys),
        reinterpret_cast<int *>(values), n);
}

void radix_sort_pairs_float_device(uint64_t keys, uint64_t values, int n)
{
    radix_sort_pairs_device(
        WP_CURRENT_CONTEXT,
        reinterpret_cast<float *>(keys),
        reinterpret_cast<int *>(values), n);
}

void radix_sort_pairs_int64_device(uint64_t keys, uint64_t values, int n)
{
    radix_sort_pairs_device(
        WP_CURRENT_CONTEXT,
        reinterpret_cast<int64_t *>(keys),
        reinterpret_cast<int *>(values), n);
}

void segmented_sort_reserve(void* context, int n, int num_segments, void** mem_out, size_t* size_out)
{
    ContextGuard guard(context);

    cub::DoubleBuffer<int> d_keys;
	cub::DoubleBuffer<int> d_values;

    int* start_indices = NULL;
    int* end_indices = NULL;

    // compute temporary memory required
	size_t sort_temp_size;
    check_cuda(cub::DeviceSegmentedRadixSort::SortPairs(
        NULL,
        sort_temp_size,
        d_keys,
        d_values,
        n, 
        num_segments,
        start_indices,
        end_indices,
        0,
        32,
        (cudaStream_t)cuda_stream_get_current()));

    if (!context)
        context = cuda_context_get_current();

    RadixSortTemp& temp = g_radix_sort_temp_map[context];

    if (sort_temp_size > temp.size)
    {
	    free_device(WP_CURRENT_CONTEXT, temp.mem);
        temp.mem = alloc_device(WP_CURRENT_CONTEXT, sort_temp_size);
        temp.size = sort_temp_size;
    }
    
    if (mem_out)
        *mem_out = temp.mem;
    if (size_out)
        *size_out = temp.size;
}

// segment_start_indices and segment_end_indices are arrays of length num_segments, where segment_start_indices[i] is the index of the first element 
// in the i-th segment and segment_end_indices[i] is the index after the last element in the i-th segment
// https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceSegmentedRadixSort.html
void segmented_sort_pairs_device(void* context, float* keys, int* values, int n, int* segment_start_indices, int* segment_end_indices, int num_segments)
{
    ContextGuard guard(context);

    cub::DoubleBuffer<float> d_keys(keys, keys + n);
	cub::DoubleBuffer<int> d_values(values, values + n);

    RadixSortTemp temp;
    segmented_sort_reserve(WP_CURRENT_CONTEXT, n, num_segments, &temp.mem, &temp.size);

    // sort
    check_cuda(cub::DeviceSegmentedRadixSort::SortPairs(
        temp.mem,
        temp.size,
        d_keys, 
        d_values, 
        n,
        num_segments,
        segment_start_indices,
        segment_end_indices,
        0,
        32,
        (cudaStream_t)cuda_stream_get_current()));

	if (d_keys.Current() != keys)
		memcpy_d2d(WP_CURRENT_CONTEXT, keys, d_keys.Current(), sizeof(float)*n);

	if (d_values.Current() != values)
		memcpy_d2d(WP_CURRENT_CONTEXT, values, d_values.Current(), sizeof(int)*n);
}

void segmented_sort_pairs_float_device(uint64_t keys, uint64_t values, int n, uint64_t segment_start_indices, uint64_t segment_end_indices, int num_segments)
{
    segmented_sort_pairs_device(
        WP_CURRENT_CONTEXT,
        reinterpret_cast<float *>(keys),
        reinterpret_cast<int *>(values), n,
        reinterpret_cast<int *>(segment_start_indices),
        reinterpret_cast<int *>(segment_end_indices),
        num_segments);
}

// segment_indices is an array of length num_segments + 1, where segment_indices[i] is the index of the first element in the i-th segment
// The end of a segment is given by segment_indices[i+1]
// https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceSegmentedSort.html#a-simple-example
void segmented_sort_pairs_device(void* context, int* keys, int* values, int n, int* segment_start_indices, int* segment_end_indices, int num_segments)
{
    ContextGuard guard(context);

    cub::DoubleBuffer<int> d_keys(keys, keys + n);
	cub::DoubleBuffer<int> d_values(values, values + n);

    RadixSortTemp temp;
    segmented_sort_reserve(WP_CURRENT_CONTEXT, n, num_segments, &temp.mem, &temp.size);

    // sort
    check_cuda(cub::DeviceSegmentedRadixSort::SortPairs(
        temp.mem,
        temp.size,
        d_keys, 
        d_values, 
        n,
        num_segments,
        segment_start_indices,
        segment_end_indices,
        0,
        32,
        (cudaStream_t)cuda_stream_get_current()));

	if (d_keys.Current() != keys)
		memcpy_d2d(WP_CURRENT_CONTEXT, keys, d_keys.Current(), sizeof(float)*n);

	if (d_values.Current() != values)
		memcpy_d2d(WP_CURRENT_CONTEXT, values, d_values.Current(), sizeof(int)*n);
}

void segmented_sort_pairs_int_device(uint64_t keys, uint64_t values, int n, uint64_t segment_start_indices, uint64_t segment_end_indices, int num_segments)
{
    segmented_sort_pairs_device(
        WP_CURRENT_CONTEXT,
        reinterpret_cast<int *>(keys),
        reinterpret_cast<int *>(values), n,
        reinterpret_cast<int *>(segment_start_indices),
        reinterpret_cast<int *>(segment_end_indices),
        num_segments);
}
