// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Deterministic mode post-kernel sort-reduce.
//
// After a kernel scatters (key, value) records into a temporary buffer, this
// module sorts them by key (CUB radix sort) and then applies a deterministic
// segmented reduction: records with the same destination index are accumulated
// left-to-right in thread-id order, and the result is applied to the target
// array.

#include "warp.h"

#include "cuda_util.h"
#include "temp_buffer.h"

#define THRUST_IGNORE_CUB_VERSION_CHECK

#include <cub/cub.cuh>

namespace {

// Extract the destination index (upper 32 bits) from a sort key.
__host__ __device__ __forceinline__ int dest_index_from_key(int64_t key)
{
    return static_cast<int>(static_cast<uint64_t>(key) >> 32);
}

// Reduction op identifiers (must match the Python-side constants).
enum ReduceOp {
    REDUCE_ADD = 0,
    REDUCE_MIN = 1,
    REDUCE_MAX = 2,
};

enum ScalarType {
    SCALAR_HALF = 0,
    SCALAR_FLOAT = 1,
    SCALAR_DOUBLE = 2,
    SCALAR_INT = 3,
    SCALAR_UINT = 4,
    SCALAR_INT64 = 5,
    SCALAR_UINT64 = 6,
};

__global__ void init_record_indices_kernel(int* indices, int count)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < count) {
        indices[tid] = tid;
    }
}

template <typename T> struct ReduceByKeyOp {
    int op;

    __host__ __device__ T operator()(const T& a, const T& b) const
    {
        switch (op) {
        case REDUCE_ADD:
            return a + b;
        case REDUCE_MIN:
            return wp::min(a, b);
        case REDUCE_MAX:
            return wp::max(a, b);
        default:
            return a;
        }
    }
};

struct DestIndexTransform {
    __host__ __device__ int operator()(const int64_t& key) const { return dest_index_from_key(key); }
};

template <typename T>
__global__ void apply_reduced_runs_kernel(
    const int* __restrict__ unique_dests,
    const T* __restrict__ aggregates,
    const int* __restrict__ num_runs,
    T* __restrict__ dest_array,
    int dest_size,
    int op
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= *num_runs)
        return;

    int dest = unique_dests[tid];
    if (dest < 0 || dest >= dest_size)
        return;

    switch (op) {
    case REDUCE_ADD:
        dest_array[dest] = dest_array[dest] + aggregates[tid];
        break;
    case REDUCE_MIN:
        dest_array[dest] = wp::min(dest_array[dest], aggregates[tid]);
        break;
    case REDUCE_MAX:
        dest_array[dest] = wp::max(dest_array[dest], aggregates[tid]);
        break;
    }
}

// Kernel that walks the sorted scatter records and applies a deterministic
// segmented reduction.  Each thread handles one contiguous run of records that
// target the same destination index.
//
// We identify segment boundaries by comparing adjacent keys' upper 32 bits.
// A prefix "segment-head" scan finds the start of each segment, and each
// segment-head thread accumulates its segment sequentially.
template <typename T>
__global__ void deterministic_reduce_kernel(
    const int64_t* __restrict__ sorted_keys,
    const int* __restrict__ sorted_indices,
    const T* __restrict__ values,
    int num_records,
    int components,
    T* __restrict__ dest_array,
    int dest_size,
    int op
)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_records)
        return;

    // A thread is a segment head if it is the first record, or its dest index
    // differs from the previous record's.
    int my_dest = dest_index_from_key(sorted_keys[tid]);
    bool is_head = (tid == 0) || (dest_index_from_key(sorted_keys[tid - 1]) != my_dest);

    if (!is_head)
        return;

    if (my_dest < 0 || my_dest >= dest_size)
        return;

    int base = sorted_indices[tid] * components;
    int dest_base = my_dest * components;

    // Accumulate the segment sequentially (deterministic left-to-right order).
    for (int c = 0; c < components; ++c) {
        T accum = values[base + c];
        for (int i = tid + 1; i < num_records; ++i) {
            if (dest_index_from_key(sorted_keys[i]) != my_dest)
                break;
            T val = values[sorted_indices[i] * components + c];
            switch (op) {
            case REDUCE_ADD:
                accum = accum + val;
                break;
            case REDUCE_MIN:
                accum = wp::min(accum, val);
                break;
            case REDUCE_MAX:
                accum = wp::max(accum, val);
                break;
            }
        }

        switch (op) {
        case REDUCE_ADD:
            dest_array[dest_base + c] = dest_array[dest_base + c] + accum;
            break;
        case REDUCE_MIN:
            dest_array[dest_base + c] = wp::min(dest_array[dest_base + c], accum);
            break;
        case REDUCE_MAX:
            dest_array[dest_base + c] = wp::max(dest_array[dest_base + c], accum);
            break;
        }
    }
}

template <typename T>
void deterministic_sort_reduce_device_scalar(int64_t* keys, T* values, int count, T* dest_array, int dest_size, int op)
{
    if (count <= 0)
        return;

    ContextGuard guard(wp_cuda_context_get_current());
    cudaStream_t stream = static_cast<cudaStream_t>(wp_cuda_stream_get_current());

    ScopedTemporary<int64_t> alt_keys(WP_CURRENT_CONTEXT, count);
    ScopedTemporary<T> alt_values(WP_CURRENT_CONTEXT, count);

    cub::DoubleBuffer<int64_t> d_keys(keys, alt_keys.buffer());
    cub::DoubleBuffer<T> d_values(values, alt_values.buffer());

    size_t sort_temp_size = 0;
    check_cuda(
        cub::DeviceRadixSort::SortPairs(
            nullptr, sort_temp_size, d_keys, d_values, count, 0, sizeof(int64_t) * 8, stream
        )
    );

    void* sort_temp = wp_alloc_device(WP_CURRENT_CONTEXT, sort_temp_size);
    check_cuda(
        cub::DeviceRadixSort::SortPairs(
            sort_temp, sort_temp_size, d_keys, d_values, count, 0, sizeof(int64_t) * 8, stream
        )
    );
    wp_free_device(WP_CURRENT_CONTEXT, sort_temp);

    ScopedTemporary<int> unique_dests(WP_CURRENT_CONTEXT, count);
    ScopedTemporary<T> aggregates(WP_CURRENT_CONTEXT, count);
    ScopedTemporary<int> num_runs(WP_CURRENT_CONTEXT, 1);
    auto dest_keys
        = cub::TransformInputIterator<int, DestIndexTransform, const int64_t*>(d_keys.Current(), DestIndexTransform {});

    size_t reduce_temp_size = 0;
    ReduceByKeyOp<T> reduce_op { op };
    check_cuda(
        cub::DeviceReduce::ReduceByKey(
            nullptr, reduce_temp_size, dest_keys, unique_dests.buffer(), d_values.Current(), aggregates.buffer(),
            num_runs.buffer(), reduce_op, count, stream
        )
    );

    void* reduce_temp = wp_alloc_device(WP_CURRENT_CONTEXT, reduce_temp_size);
    check_cuda(
        cub::DeviceReduce::ReduceByKey(
            reduce_temp, reduce_temp_size, dest_keys, unique_dests.buffer(), d_values.Current(), aggregates.buffer(),
            num_runs.buffer(), reduce_op, count, stream
        )
    );
    wp_free_device(WP_CURRENT_CONTEXT, reduce_temp);

    const int block_size = 256;
    const int num_blocks = (count + block_size - 1) / block_size;
    apply_reduced_runs_kernel<T><<<num_blocks, block_size, 0, stream>>>(
        unique_dests.buffer(), aggregates.buffer(), num_runs.buffer(), dest_array, dest_size, op
    );
}

// Sort the scatter buffer by key using CUB radix sort, then launch the
// reduce kernel.
template <typename T>
void deterministic_sort_reduce_device(
    int64_t* keys, T* values, int count, T* dest_array, int dest_size, int op, int components
)
{
    if (count <= 0)
        return;

    if (components == 1) {
        deterministic_sort_reduce_device_scalar(keys, values, count, dest_array, dest_size, op);
        return;
    }

    ContextGuard guard(wp_cuda_context_get_current());
    cudaStream_t stream = static_cast<cudaStream_t>(wp_cuda_stream_get_current());

    // --- Sort by key ---
    // The input buffers have a fixed capacity. Unused slots are initialized
    // with key == -1, which sorts to the end and is ignored by the reduce
    // kernel.
    // Sort keys together with record indices, then reduce from the original
    // value buffer component-wise.
    ScopedTemporary<int64_t> alt_keys(WP_CURRENT_CONTEXT, count);
    ScopedTemporary<int> record_indices(WP_CURRENT_CONTEXT, count);
    ScopedTemporary<int> alt_record_indices(WP_CURRENT_CONTEXT, count);

    const int block_size = 256;
    const int num_blocks = (count + block_size - 1) / block_size;
    init_record_indices_kernel<<<num_blocks, block_size, 0, stream>>>(record_indices.buffer(), count);

    cub::DoubleBuffer<int64_t> d_keys(keys, alt_keys.buffer());
    cub::DoubleBuffer<int> d_indices(record_indices.buffer(), alt_record_indices.buffer());

    size_t sort_temp_size = 0;
    check_cuda(
        cub::DeviceRadixSort::SortPairs(
            nullptr, sort_temp_size, d_keys, d_indices, count, 0, sizeof(int64_t) * 8, stream
        )
    );

    void* sort_temp = wp_alloc_device(WP_CURRENT_CONTEXT, sort_temp_size);
    check_cuda(
        cub::DeviceRadixSort::SortPairs(
            sort_temp, sort_temp_size, d_keys, d_indices, count, 0, sizeof(int64_t) * 8, stream
        )
    );
    wp_free_device(WP_CURRENT_CONTEXT, sort_temp);

    // --- Segmented reduce ---
    deterministic_reduce_kernel<T><<<num_blocks, block_size, 0, stream>>>(
        d_keys.Current(), d_indices.Current(), values, count, components, dest_array, dest_size, op
    );
}

}  // anonymous namespace


// Public API entry points called from the Python runtime via ctypes.
// Arguments are passed as uint64_t pointers (matching the Warp convention).

void wp_deterministic_sort_reduce_device(
    uint64_t keys,
    uint64_t values,
    int count,
    uint64_t dest_array,
    int dest_size,
    int op,
    int scalar_type,
    int components
)
{
    switch (scalar_type) {
    case SCALAR_HALF:
        deterministic_sort_reduce_device<wp::half>(
            reinterpret_cast<int64_t*>(keys), reinterpret_cast<wp::half*>(values), count,
            reinterpret_cast<wp::half*>(dest_array), dest_size, op, components
        );
        break;
    case SCALAR_FLOAT:
        deterministic_sort_reduce_device<float>(
            reinterpret_cast<int64_t*>(keys), reinterpret_cast<float*>(values), count,
            reinterpret_cast<float*>(dest_array), dest_size, op, components
        );
        break;
    case SCALAR_DOUBLE:
        deterministic_sort_reduce_device<double>(
            reinterpret_cast<int64_t*>(keys), reinterpret_cast<double*>(values), count,
            reinterpret_cast<double*>(dest_array), dest_size, op, components
        );
        break;
    case SCALAR_INT:
        deterministic_sort_reduce_device<int>(
            reinterpret_cast<int64_t*>(keys), reinterpret_cast<int*>(values), count, reinterpret_cast<int*>(dest_array),
            dest_size, op, components
        );
        break;
    case SCALAR_UINT:
        deterministic_sort_reduce_device<unsigned int>(
            reinterpret_cast<int64_t*>(keys), reinterpret_cast<unsigned int*>(values), count,
            reinterpret_cast<unsigned int*>(dest_array), dest_size, op, components
        );
        break;
    case SCALAR_INT64:
        deterministic_sort_reduce_device<int64_t>(
            reinterpret_cast<int64_t*>(keys), reinterpret_cast<int64_t*>(values), count,
            reinterpret_cast<int64_t*>(dest_array), dest_size, op, components
        );
        break;
    case SCALAR_UINT64:
        deterministic_sort_reduce_device<uint64_t>(
            reinterpret_cast<int64_t*>(keys), reinterpret_cast<uint64_t*>(values), count,
            reinterpret_cast<uint64_t*>(dest_array), dest_size, op, components
        );
        break;
    default:
        break;
    }
}
