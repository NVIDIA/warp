#include "warp.h"
#include "sort.h"

#define THRUST_IGNORE_CUB_VERSION_CHECK

#include <cub/cub.cuh>

void radix_sort_pairs_device(int* keys, int* values, int n)
{
    static void* sort_temp_memory = NULL;
    static size_t sort_temp_max_size = 0;

    cub::DoubleBuffer<int> d_keys(keys, keys + n);
	cub::DoubleBuffer<int> d_values(values, values + n);

    // compute temporary memory required
	size_t sort_temp_size;
	cub::DeviceRadixSort::SortPairs(NULL, sort_temp_size, d_keys, d_values, int(n), 0, 32);

    if (sort_temp_size > sort_temp_max_size)
    {
	    free_device(sort_temp_memory);
        sort_temp_memory = alloc_device(sort_temp_size);
        sort_temp_max_size = sort_temp_size;
    }

    // sort
    cub::DeviceRadixSort::SortPairs(sort_temp_memory, sort_temp_size, d_keys, d_values, n, 0, 32, (cudaStream_t)cuda_get_stream());

	if (d_keys.Current() != keys)
		memcpy_d2d(keys, d_keys.Current(), sizeof(int)*n);

	if (d_values.Current() != values)
		memcpy_d2d(values, d_values.Current(), sizeof(int)*n);
}