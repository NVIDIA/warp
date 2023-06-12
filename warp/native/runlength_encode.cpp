#include "warp.h"

#include <cstdint>

template <typename T>
void runlength_encode_host(int n,
                           const T *values,
                           T *run_values,
                           int *run_lengths,
                           int *run_count)
{
    if (n == 0)
    {
        *run_count = 0;
        return;
    }

    const T *end = values + n;

    *run_count = 1;
    *run_lengths = 1;
    *run_values = *values;

    while (++values != end)
    {
        if (*values == *run_values)
        {
            ++*run_lengths;
        }
        else
        {
            ++*run_count;
            *(++run_lengths) = 1;
            *(++run_values) = *values;
        }
    }
}

void runlength_encode_int_host(
    uint64_t values,
    uint64_t run_values,
    uint64_t run_lengths,
    uint64_t run_count,
    int n)
{
    runlength_encode_host<int>(n,
                               reinterpret_cast<const int *>(values),
                               reinterpret_cast<int *>(run_values),
                               reinterpret_cast<int *>(run_lengths),
                               reinterpret_cast<int *>(run_count));
}

#if !WP_ENABLE_CUDA
void runlength_encode_int_device(
    uint64_t values,
    uint64_t run_values,
    uint64_t run_lengths,
    uint64_t run_count,
    int n)
{
}
#endif