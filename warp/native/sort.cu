/** Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "warp.h"
#include "sort.h"

#define THRUST_IGNORE_CUB_VERSION_CHECK

#include <cub/cub.cuh>

static void* radix_sort_temp_memory = NULL;
static size_t radix_sort_temp_max_size = 0;

void radix_sort_reserve(int n)
{
    cub::DoubleBuffer<int> d_keys;
	cub::DoubleBuffer<int> d_values;

    // compute temporary memory required
	size_t sort_temp_size;
	cub::DeviceRadixSort::SortPairs(NULL, sort_temp_size, d_keys, d_values, int(n), 0, 32, (cudaStream_t)cuda_get_stream());

    if (sort_temp_size > radix_sort_temp_max_size)
    {
	    free_device(radix_sort_temp_memory);
        radix_sort_temp_memory = alloc_device(sort_temp_size);
        radix_sort_temp_max_size = sort_temp_size;
    }
}

void radix_sort_pairs_device(int* keys, int* values, int n)
{
    cub::DoubleBuffer<int> d_keys(keys, keys + n);
	cub::DoubleBuffer<int> d_values(values, values + n);

    radix_sort_reserve(n);

    // sort
    cub::DeviceRadixSort::SortPairs(
        radix_sort_temp_memory, 
        radix_sort_temp_max_size, 
        d_keys, 
        d_values, 
        n, 0, 32, 
        (cudaStream_t)cuda_get_stream());

	if (d_keys.Current() != keys)
		memcpy_d2d(keys, d_keys.Current(), sizeof(int)*n);

	if (d_values.Current() != values)
		memcpy_d2d(values, d_values.Current(), sizeof(int)*n);
}