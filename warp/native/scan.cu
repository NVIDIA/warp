// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "warp.h"

#include "cuda_util.h"
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

template <typename T>
void scan_device(
    const T* values_in, T* values_out, int n, int in_byte_stride, int out_byte_stride, int type_length, bool inclusive
)
{
    assert((in_byte_stride % sizeof(T)) == 0);
    assert((out_byte_stride % sizeof(T)) == 0);

    const int in_stride = in_byte_stride / sizeof(T);
    const int out_stride = out_byte_stride / sizeof(T);

    ContextGuard guard(wp_cuda_context_get_current());

    cudaStream_t stream = static_cast<cudaStream_t>(wp_cuda_stream_get_current());

    // compute temporary memory required
    size_t scan_temp_size = 0;
    cub_strided_iterator<const T> values_in_iter { values_in, in_stride };
    cub_strided_iterator<T> values_out_iter { values_out, out_stride };

    if (inclusive) {
        check_cuda(cub::DeviceScan::InclusiveSum(NULL, scan_temp_size, values_in_iter, values_out_iter, n, stream));
    } else {
        check_cuda(cub::DeviceScan::ExclusiveSum(NULL, scan_temp_size, values_in_iter, values_out_iter, n, stream));
    }

    void* temp_buffer = wp_alloc_device(WP_CURRENT_CONTEXT, scan_temp_size, "(native:scan)");

    // scan each scalar component independently
    for (int k = 0; k < type_length; ++k) {
        cub_strided_iterator<const T> values_in_iter { values_in + k, in_stride };
        cub_strided_iterator<T> values_out_iter { values_out + k, out_stride };
        size_t temp_storage_bytes = scan_temp_size;

        if (inclusive) {
            check_cuda(
                cub::DeviceScan::InclusiveSum(
                    temp_buffer, temp_storage_bytes, values_in_iter, values_out_iter, n, stream
                )
            );
        } else {
            check_cuda(
                cub::DeviceScan::ExclusiveSum(
                    temp_buffer, temp_storage_bytes, values_in_iter, values_out_iter, n, stream
                )
            );
        }
    }

    wp_free_device(WP_CURRENT_CONTEXT, temp_buffer);
}

template <typename T> void scan_device(const T* values_in, T* values_out, int n, bool inclusive)
{
    scan_device(values_in, values_out, n, sizeof(T), sizeof(T), 1, inclusive);
}

template void scan_device(const int*, int*, int, bool);
template void scan_device(const int64_t*, int64_t*, int, bool);
template void scan_device(const float*, float*, int, bool);
template void scan_device(const double*, double*, int, bool);

template void scan_device(const int*, int*, int, int, int, int, bool);
template void scan_device(const int64_t*, int64_t*, int, int, int, int, bool);
template void scan_device(const float*, float*, int, int, int, int, bool);
template void scan_device(const double*, double*, int, int, int, int, bool);
