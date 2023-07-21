

#include "warp.h"
#include "cuda_util.h"

#define THRUST_IGNORE_CUB_VERSION_CHECK
#include <cub/device/device_run_length_encode.cuh>

template <typename T>
void runlength_encode_device(int n,
                             const T *values,
                             T *run_values,
                             int *run_lengths,
                             int *run_count)
{
    ContextGuard guard(cuda_context_get_current());
    cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream_get_current());

    size_t buff_size = 0;
    check_cuda(cub::DeviceRunLengthEncode::Encode(
        nullptr, buff_size, values, run_values, run_lengths, run_count,
        n, stream));

    void* temp_buffer = alloc_temp_device(WP_CURRENT_CONTEXT, buff_size);

    check_cuda(cub::DeviceRunLengthEncode::Encode(
        temp_buffer, buff_size, values, run_values, run_lengths, run_count,
        n, stream));

    free_temp_device(WP_CURRENT_CONTEXT, temp_buffer);
}

void runlength_encode_int_device(
    uint64_t values,
    uint64_t run_values,
    uint64_t run_lengths,
    uint64_t run_count,
    int n)
{
    return runlength_encode_device<int>(
        n,
        reinterpret_cast<const int *>(values),
        reinterpret_cast<int *>(run_values),
        reinterpret_cast<int *>(run_lengths),
        reinterpret_cast<int *>(run_count));
}
