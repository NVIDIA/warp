/** Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
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


void radix_sort_reserve(void* context, int n, void** mem_out, size_t* size_out)
{
    ContextGuard guard(context);

    cub::DoubleBuffer<int> d_keys;
	cub::DoubleBuffer<int> d_values;

    // compute temporary memory required
	size_t sort_temp_size;
    check_cuda(cub::DeviceRadixSort::SortPairs(
        NULL,
        sort_temp_size,
        d_keys,
        d_values,
        n, 0, 32,
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

void radix_sort_pairs_device(void* context, int* keys, int* values, int n)
{
    ContextGuard guard(context);

    cub::DoubleBuffer<int> d_keys(keys, keys + n);
	cub::DoubleBuffer<int> d_values(values, values + n);

    RadixSortTemp temp;
    radix_sort_reserve(WP_CURRENT_CONTEXT, n, &temp.mem, &temp.size);

    // sort
    check_cuda(cub::DeviceRadixSort::SortPairs(
        temp.mem,
        temp.size,
        d_keys, 
        d_values, 
        n, 0, 32, 
        (cudaStream_t)cuda_stream_get_current()));

	if (d_keys.Current() != keys)
		memcpy_d2d(WP_CURRENT_CONTEXT, keys, d_keys.Current(), sizeof(int)*n);

	if (d_values.Current() != values)
		memcpy_d2d(WP_CURRENT_CONTEXT, values, d_values.Current(), sizeof(int)*n);
}

void radix_sort_pairs_int_device(uint64_t keys, uint64_t values, int n)
{
    radix_sort_pairs_device(
        WP_CURRENT_CONTEXT,
        reinterpret_cast<int *>(keys),
        reinterpret_cast<int *>(values), n);
}
