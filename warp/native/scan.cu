#include "warp.h"
#include "scan.h"

#include "temp_buffer.h"

#define THRUST_IGNORE_CUB_VERSION_CHECK

#include <cub/device/device_scan.cuh>

template<typename T>
void scan_device(const T* values_in, T* values_out, int n, bool inclusive)
{
    void *context = cuda_context_get_current();
    TemporaryBuffer &cub_temp = g_temp_buffer_map[context];

    ContextGuard guard(context);

    cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream_get_current());

    // compute temporary memory required
	size_t scan_temp_size;
    if (inclusive) {
        cub::DeviceScan::InclusiveSum(NULL, scan_temp_size, values_in, values_out, n);
    } else {
        cub::DeviceScan::ExclusiveSum(NULL, scan_temp_size, values_in, values_out, n);
    }

    cub_temp.ensure_fits(scan_temp_size);

    // scan
    if (inclusive) {
        cub::DeviceScan::InclusiveSum(cub_temp.buffer, scan_temp_size, values_in, values_out, n, (cudaStream_t)cuda_stream_get_current());
    } else {
        cub::DeviceScan::ExclusiveSum(cub_temp.buffer, scan_temp_size, values_in, values_out, n, (cudaStream_t)cuda_stream_get_current());
    }
}

template void scan_device(const int*, int*, int, bool);
template void scan_device(const float*, float*, int, bool);
