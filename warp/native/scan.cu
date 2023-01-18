#include "warp.h"
#include "scan.h"

#define THRUST_IGNORE_CUB_VERSION_CHECK

#include <cub/cub.cuh>

template<typename T>
void scan_device(const T* values_in, T* values_out, int n, bool inclusive)
{
    static void* scan_temp_memory = NULL;
    static size_t scan_temp_max_size = 0;

    // compute temporary memory required
	size_t scan_temp_size;
    if (inclusive) {
        cub::DeviceScan::InclusiveSum(NULL, scan_temp_size, values_in, values_out, n);
    } else {
        cub::DeviceScan::ExclusiveSum(NULL, scan_temp_size, values_in, values_out, n);
    }

    if (scan_temp_size > scan_temp_max_size)
    {
	    free_device(WP_CURRENT_CONTEXT, scan_temp_memory);
        scan_temp_memory = alloc_device(WP_CURRENT_CONTEXT, scan_temp_size);
        scan_temp_max_size = scan_temp_size;
    }

    // scan
    if (inclusive) {
        cub::DeviceScan::InclusiveSum(scan_temp_memory, scan_temp_size, values_in, values_out, n, (cudaStream_t)cuda_stream_get_current());
    } else {
        cub::DeviceScan::ExclusiveSum(scan_temp_memory, scan_temp_size, values_in, values_out, n, (cudaStream_t)cuda_stream_get_current());
    }
}

template void scan_device(const int*, int*, int, bool);
template void scan_device(const float*, float*, int, bool);
