// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "warp.h"

#include "cuda_util.h"
#include "error.h"
#include "scan.h"

#define THRUST_IGNORE_CUB_VERSION_CHECK

#include <iterator>
#include <type_traits>

#include <cub/device/device_scan.cuh>

namespace {

template <typename T> struct cub_strided_iterator {
    typedef cub_strided_iterator<T> self_type;
    typedef std::ptrdiff_t difference_type;
    typedef typename std::remove_cv<T>::type value_type;
    typedef T* pointer;
    typedef T& reference;

    typedef std::random_access_iterator_tag iterator_category;

    T* ptr = nullptr;
    int stride = 1;

    CUDA_CALLABLE self_type operator++(int)
    {
        self_type old(*this);
        ++(*this);
        return old;
    }

    CUDA_CALLABLE self_type& operator++()
    {
        ptr += stride;
        return *this;
    }

    CUDA_CALLABLE self_type operator+(difference_type n) const { return self_type(*this) += n; }

    CUDA_CALLABLE self_type& operator+=(difference_type n)
    {
        ptr += n * stride;
        return *this;
    }

    CUDA_CALLABLE self_type operator-(difference_type n) const { return self_type(*this) -= n; }

    CUDA_CALLABLE self_type& operator-=(difference_type n)
    {
        ptr -= n * stride;
        return *this;
    }

    CUDA_CALLABLE difference_type operator-(const self_type& other) const { return (ptr - other.ptr) / stride; }

    CUDA_CALLABLE reference operator*() const { return *ptr; }

    CUDA_CALLABLE reference operator[](difference_type n) const { return *(ptr + n * stride); }

    CUDA_CALLABLE pointer operator->() const { return ptr; }

    CUDA_CALLABLE bool operator==(const self_type& rhs) const { return (ptr == rhs.ptr); }

    CUDA_CALLABLE bool operator!=(const self_type& rhs) const { return (ptr != rhs.ptr); }
};

}  // anonymous namespace

namespace {

// Queries the CUB temporary storage size for a strided sum scan of n elements on the
// current CUDA context. The strided iterator type is used for both contiguous and
// strided arrays, so the size depends only on T, n, the scan kind, and the device.
template <typename T> bool scan_query_temp_size(int n, bool inclusive, cudaStream_t stream, size_t& temp_size)
{
    cub_strided_iterator<const T> values_in_iter { nullptr, 1 };
    cub_strided_iterator<T> values_out_iter { nullptr, 1 };

    temp_size = 0;
    if (inclusive)
        return check_cuda(cub::DeviceScan::InclusiveSum(NULL, temp_size, values_in_iter, values_out_iter, n, stream));
    else
        return check_cuda(cub::DeviceScan::ExclusiveSum(NULL, temp_size, values_in_iter, values_out_iter, n, stream));
}

}  // anonymous namespace

template <typename T> bool scan_device_temp_storage_size(void* context, int n, size_t* temp_storage_bytes)
{
    *temp_storage_bytes = 0;
    if (n <= 0)
        return true;

    ContextGuard guard(context);

    // Report the larger of the inclusive and exclusive requirements so one buffer
    // serves both scan kinds.
    size_t inclusive_size = 0;
    size_t exclusive_size = 0;
    if (!scan_query_temp_size<T>(n, true, nullptr, inclusive_size))
        return false;
    if (!scan_query_temp_size<T>(n, false, nullptr, exclusive_size))
        return false;

    *temp_storage_bytes = inclusive_size > exclusive_size ? inclusive_size : exclusive_size;
    return true;
}

template <typename T>
bool scan_device(
    const T* values_in,
    T* values_out,
    int n,
    int in_byte_stride,
    int out_byte_stride,
    int type_length,
    bool inclusive,
    void* temp_storage,
    size_t temp_storage_bytes
)
{
    assert((in_byte_stride % sizeof(T)) == 0);
    assert((out_byte_stride % sizeof(T)) == 0);

    const int in_stride = in_byte_stride / sizeof(T);
    const int out_stride = out_byte_stride / sizeof(T);

    ContextGuard guard(wp_cuda_context_get_current());

    cudaStream_t stream = static_cast<cudaStream_t>(wp_cuda_stream_get_current());

    // Stop if CUB cannot size its temporary storage; otherwise the scan could
    // continue without valid temporary storage.
    size_t scan_temp_size = 0;
    if (!scan_query_temp_size<T>(n, inclusive, stream, scan_temp_size))
        return false;

    // Use caller-provided storage when given so the scan neither allocates nor
    // frees memory. That keeps it valid in conditional graph bodies and in
    // captures that Warp did not start.
    const bool owns_temp = temp_storage == nullptr;
    void* temp_buffer = temp_storage;
    if (owns_temp) {
        temp_buffer = wp_alloc_device(WP_CURRENT_CONTEXT, scan_temp_size, "(native:scan)");
        // wp_alloc_device() already records the CUDA error.
        if (scan_temp_size > 0 && !temp_buffer)
            return false;
    } else if (temp_storage_bytes < scan_temp_size) {
        wp::set_error_string(
            "Warp error: array_scan() temporary storage is too small: %zu bytes required, %zu bytes provided",
            scan_temp_size, temp_storage_bytes
        );
        return false;
    }

    // scan each scalar component independently, reusing the same temporary storage
    bool success = true;
    for (int k = 0; k < type_length && success; ++k) {
        cub_strided_iterator<const T> values_in_iter { values_in + k, in_stride };
        cub_strided_iterator<T> values_out_iter { values_out + k, out_stride };
        size_t temp_bytes = owns_temp ? scan_temp_size : temp_storage_bytes;

        if (inclusive) {
            success = check_cuda(
                cub::DeviceScan::InclusiveSum(temp_buffer, temp_bytes, values_in_iter, values_out_iter, n, stream)
            );
        } else {
            success = check_cuda(
                cub::DeviceScan::ExclusiveSum(temp_buffer, temp_bytes, values_in_iter, values_out_iter, n, stream)
            );
        }
    }

    if (owns_temp)
        free_device_on_stream(WP_CURRENT_CONTEXT, temp_buffer, stream);
    return success;
}

template <typename T> bool scan_device(const T* values_in, T* values_out, int n, bool inclusive)
{
    return scan_device(values_in, values_out, n, sizeof(T), sizeof(T), 1, inclusive);
}

template bool scan_device(const int*, int*, int, bool);
template bool scan_device(const int64_t*, int64_t*, int, bool);
template bool scan_device(const float*, float*, int, bool);
template bool scan_device(const double*, double*, int, bool);

template bool scan_device(const int*, int*, int, int, int, int, bool, void*, size_t);
template bool scan_device(const int64_t*, int64_t*, int, int, int, int, bool, void*, size_t);
template bool scan_device(const float*, float*, int, int, int, int, bool, void*, size_t);
template bool scan_device(const double*, double*, int, int, int, int, bool, void*, size_t);

template bool scan_device_temp_storage_size<int>(void*, int, size_t*);
template bool scan_device_temp_storage_size<int64_t>(void*, int, size_t*);
template bool scan_device_temp_storage_size<float>(void*, int, size_t*);
template bool scan_device_temp_storage_size<double>(void*, int, size_t*);
