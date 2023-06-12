

#include "warp.h"
#include "cuda_util.h"

#include "temp_buffer.h"

#define THRUST_IGNORE_CUB_VERSION_CHECK
#include <cub/device/device_run_length_encode.cuh>

template <typename T>
void runlength_encode_device(int n,
                             const T *values,
                             T *run_values,
                             int *run_lengths,
                             int *run_count)
{
    void *context = cuda_context_get_current();

    TemporaryBuffer &cub_temp = g_temp_buffer_map[context];

    ContextGuard guard(context);
    cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream_get_current());

    size_t buff_size = 0;
    check_cuda(cub::DeviceRunLengthEncode::Encode(
        nullptr, buff_size, values, run_values, run_lengths, run_count,
        n, stream));

    cub_temp.ensure_fits(buff_size);

    check_cuda(cub::DeviceRunLengthEncode::Encode(
        cub_temp.buffer, buff_size, values, run_values, run_lengths, run_count,
        n, stream));
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