// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#define THRUST_IGNORE_CUB_VERSION_CHECK

#include <cub/cub.cuh>
#include <thrust/iterator/transform_iterator.h>

namespace {

// Extract the destination index (upper 32 bits) from a sort key.
__host__ __device__ __forceinline__ int dest_index_from_key(int64_t key)
{
    return static_cast<int>(static_cast<uint64_t>(key) >> 32);
}

// Reduction op identifiers (must match the Python-side constants).
enum ReduceOp {
    REDUCE_OP_ADD = 0,
    REDUCE_OP_MIN = 1,
    REDUCE_OP_MAX = 2,
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

enum DeterminismLevel {
    DETERMINISTIC_NOT_GUARANTEED = 0,
    DETERMINISTIC_RUN_TO_RUN = 1,
    DETERMINISTIC_GPU_TO_GPU = 2,
};

constexpr int DETERMINISTIC_BLOCK_SIZE = 256;

inline int deterministic_num_blocks(int count)
{
    return static_cast<int>((static_cast<int64_t>(count) + DETERMINISTIC_BLOCK_SIZE - 1) / DETERMINISTIC_BLOCK_SIZE);
}

__device__ __forceinline__ bool deterministic_thread_index(int count, int& tid)
{
    int64_t linear_tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linear_tid >= count)
        return false;

    tid = static_cast<int>(linear_tid);
    return true;
}

inline bool check_kernel_launch() { return check_cuda(cudaGetLastError()); }

inline size_t align_offset(size_t offset, size_t alignment) { return (offset + alignment - 1) & ~(alignment - 1); }

template <typename T> T* layout_array(char* base, size_t& offset, size_t count)
{
    offset = align_offset(offset, alignof(T));
    T* ptr = base ? reinterpret_cast<T*>(base + offset) : nullptr;
    offset += count * sizeof(T);
    return ptr;
}

char* layout_bytes(char* base, size_t& offset, size_t count)
{
    offset = align_offset(offset, 256);
    char* ptr = base ? base + offset : nullptr;
    offset += count;
    return ptr;
}

__global__ void init_record_indices_kernel(int* indices, int count)
{
    int tid;
    if (!deterministic_thread_index(count, tid))
        return;

    indices[tid] = tid;
}

struct CounterPrefixState {
    int dest;
    int sum;
};

struct CounterPrefixOp {
    __host__ __device__ CounterPrefixState operator()(const CounterPrefixState& a, const CounterPrefixState& b) const
    {
        if (a.dest == b.dest) {
            return CounterPrefixState { b.dest, a.sum + b.sum };
        }
        return b;
    }
};

__global__ void make_counter_prefix_states_kernel(
    const int64_t* __restrict__ sorted_keys,
    const int* __restrict__ sorted_indices,
    const int* __restrict__ values,
    const int* __restrict__ counter_bases,
    CounterPrefixState* __restrict__ states,
    int counter_size,
    int count
)
{
    int tid;
    if (!deterministic_thread_index(count, tid))
        return;

    int dest = dest_index_from_key(sorted_keys[tid]);
    int original_slot = sorted_indices[tid];
    int value = dest < 0 ? 0 : values[original_slot];

    if (dest >= 0 && dest < counter_size) {
        bool is_head = (tid == 0) || (dest_index_from_key(sorted_keys[tid - 1]) != dest);
        if (is_head) {
            value += counter_bases[dest];
        }
    }

    states[tid] = CounterPrefixState { dest, value };
}

__global__ void scatter_counter_prefixes_kernel(
    const int64_t* __restrict__ sorted_keys,
    const int* __restrict__ sorted_indices,
    const int* __restrict__ values,
    const CounterPrefixState* __restrict__ inclusive_prefixes,
    int* __restrict__ prefixes,
    int* __restrict__ counter_totals,
    int counter_size,
    int count
)
{
    int tid;
    if (!deterministic_thread_index(count, tid))
        return;

    int dest = dest_index_from_key(sorted_keys[tid]);
    int original_slot = sorted_indices[tid];
    if (dest < 0 || original_slot < 0)
        return;

    int value = values[original_slot];
    int inclusive = inclusive_prefixes[tid].sum;
    prefixes[original_slot] = inclusive - value;

    bool is_tail = (tid == count - 1) || (dest_index_from_key(sorted_keys[tid + 1]) != dest);
    if (is_tail && dest < counter_size) {
        counter_totals[dest] = inclusive;
    }
}

__global__ void writeback_counter_totals_kernel(
    const int64_t* __restrict__ keys,
    int count,
    const int* __restrict__ counter_totals,
    int* __restrict__ counters,
    int counter_size
)
{
    int tid;
    if (!deterministic_thread_index(count, tid))
        return;

    int dest = dest_index_from_key(keys[tid]);
    if (dest >= 0 && dest < counter_size) {
        counters[dest] = counter_totals[dest];
    }
}

template <typename T> struct ReduceByKeyOp {
    int op;

    __host__ __device__ T operator()(const T& a, const T& b) const
    {
        switch (op) {
        case REDUCE_OP_ADD:
            return a + b;
        case REDUCE_OP_MIN:
            return wp::min(a, b);
        case REDUCE_OP_MAX:
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
void query_scalar_temp_sizes(int count, int op, cudaStream_t stream, size_t& sort_temp_size, size_t& reduce_temp_size)
{
    sort_temp_size = 0;
    reduce_temp_size = 0;

    cub::DoubleBuffer<int64_t> d_keys(nullptr, nullptr);
    cub::DoubleBuffer<T> d_values(nullptr, nullptr);
    check_cuda(
        cub::DeviceRadixSort::SortPairs(
            nullptr, sort_temp_size, d_keys, d_values, count, 0, sizeof(int64_t) * 8, stream
        )
    );

    auto dest_keys = thrust::make_transform_iterator(static_cast<int64_t*>(nullptr), DestIndexTransform {});
    ReduceByKeyOp<T> reduce_op { op };
    check_cuda(
        cub::DeviceReduce::ReduceByKey(
            nullptr, reduce_temp_size, dest_keys, static_cast<int*>(nullptr), static_cast<T*>(nullptr),
            static_cast<T*>(nullptr), static_cast<int*>(nullptr), reduce_op, count, stream
        )
    );
}

template <typename T> size_t scalar_run_to_run_workspace_size(int count, int op, cudaStream_t stream)
{
    if (count <= 0)
        return 0;

    size_t sort_temp_size = 0;
    size_t reduce_temp_size = 0;
    query_scalar_temp_sizes<T>(count, op, stream, sort_temp_size, reduce_temp_size);

    size_t offset = 0;
    layout_array<int64_t>(nullptr, offset, count);
    layout_array<T>(nullptr, offset, count);
    layout_bytes(nullptr, offset, sort_temp_size);
    layout_array<int>(nullptr, offset, count);
    layout_array<T>(nullptr, offset, count);
    layout_array<int>(nullptr, offset, 1);
    layout_bytes(nullptr, offset, reduce_temp_size);
    return offset;
}

void query_generic_sort_temp_size(int count, cudaStream_t stream, size_t& sort_temp_size)
{
    sort_temp_size = 0;

    cub::DoubleBuffer<int64_t> d_keys(nullptr, nullptr);
    cub::DoubleBuffer<int> d_indices(nullptr, nullptr);
    check_cuda(
        cub::DeviceRadixSort::SortPairs(
            nullptr, sort_temp_size, d_keys, d_indices, count, 0, sizeof(int64_t) * 8, stream
        )
    );
}

size_t generic_workspace_size(int count, cudaStream_t stream)
{
    if (count <= 0)
        return 0;

    size_t sort_temp_size = 0;
    query_generic_sort_temp_size(count, stream, sort_temp_size);

    size_t offset = 0;
    layout_array<int64_t>(nullptr, offset, count);
    layout_array<int>(nullptr, offset, count);
    layout_array<int>(nullptr, offset, count);
    layout_bytes(nullptr, offset, sort_temp_size);
    return offset;
}

void query_counter_scan_temp_size(int count, cudaStream_t stream, size_t& scan_temp_size)
{
    scan_temp_size = 0;

    CounterPrefixState* states = nullptr;
    CounterPrefixState* prefixes = nullptr;
    CounterPrefixOp op {};
    check_cuda(cub::DeviceScan::InclusiveScan(nullptr, scan_temp_size, states, prefixes, op, count, stream));
}

size_t counter_workspace_size(int count, cudaStream_t stream)
{
    if (count <= 0)
        return 0;

    size_t sort_temp_size = 0;
    size_t scan_temp_size = 0;
    query_generic_sort_temp_size(count, stream, sort_temp_size);
    query_counter_scan_temp_size(count, stream, scan_temp_size);

    size_t offset = 0;
    layout_array<int64_t>(nullptr, offset, count);
    layout_array<int>(nullptr, offset, count);
    layout_array<int>(nullptr, offset, count);
    layout_bytes(nullptr, offset, sort_temp_size);
    layout_array<CounterPrefixState>(nullptr, offset, count);
    layout_array<CounterPrefixState>(nullptr, offset, count);
    layout_bytes(nullptr, offset, scan_temp_size);
    return offset;
}

template <typename T>
size_t deterministic_workspace_size(int count, int op, int components, int determinism_level, cudaStream_t stream)
{
    if (components == 1 && determinism_level == DETERMINISTIC_RUN_TO_RUN) {
        return scalar_run_to_run_workspace_size<T>(count, op, stream);
    }
    return generic_workspace_size(count, stream);
}

// Scalar-only (components == 1); widen indexing to int64_t if extended to
// multi-component types (see deterministic_reduce_kernel).
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
    int tid;
    if (!deterministic_thread_index(*num_runs, tid))
        return;

    int dest = unique_dests[tid];
    if (dest < 0 || dest >= dest_size)
        return;

    switch (op) {
    case REDUCE_OP_ADD:
        dest_array[dest] = dest_array[dest] + aggregates[tid];
        break;
    case REDUCE_OP_MIN:
        dest_array[dest] = wp::min(dest_array[dest], aggregates[tid]);
        break;
    case REDUCE_OP_MAX:
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
    int tid;
    if (!deterministic_thread_index(num_records, tid))
        return;

    // A thread is a segment head if it is the first record, or its dest index
    // differs from the previous record's.
    int my_dest = dest_index_from_key(sorted_keys[tid]);
    bool is_head = (tid == 0) || (dest_index_from_key(sorted_keys[tid - 1]) != my_dest);

    if (!is_head)
        return;

    if (my_dest < 0 || my_dest >= dest_size)
        return;

    int64_t base = static_cast<int64_t>(sorted_indices[tid]) * components;
    int64_t dest_base = static_cast<int64_t>(my_dest) * components;

    // Accumulate each segment sequentially to preserve a deterministic
    // left-to-right order. This intentionally favors reproducibility over
    // throughput for highly contended destinations in gpu_to_gpu mode.
    for (int c = 0; c < components; ++c) {
        T accum = values[base + c];
        for (int i = tid + 1; i < num_records; ++i) {
            if (dest_index_from_key(sorted_keys[i]) != my_dest)
                break;
            T val = values[static_cast<int64_t>(sorted_indices[i]) * components + c];
            switch (op) {
            case REDUCE_OP_ADD:
                accum = accum + val;
                break;
            case REDUCE_OP_MIN:
                accum = wp::min(accum, val);
                break;
            case REDUCE_OP_MAX:
                accum = wp::max(accum, val);
                break;
            }
        }

        switch (op) {
        case REDUCE_OP_ADD:
            dest_array[dest_base + c] = dest_array[dest_base + c] + accum;
            break;
        case REDUCE_OP_MIN:
            dest_array[dest_base + c] = wp::min(dest_array[dest_base + c], accum);
            break;
        case REDUCE_OP_MAX:
            dest_array[dest_base + c] = wp::max(dest_array[dest_base + c], accum);
            break;
        }
    }
}

template <typename T>
void deterministic_sort_reduce_device_scalar_run_to_run(
    int64_t* keys, T* values, int count, T* dest_array, int dest_size, int op, void* workspace, size_t workspace_size
)
{
    if (count <= 0)
        return;

    ContextGuard guard(wp_cuda_context_get_current());
    cudaStream_t stream = static_cast<cudaStream_t>(wp_cuda_stream_get_current());

    size_t required_workspace_size = scalar_run_to_run_workspace_size<T>(count, op, stream);
    if (workspace == nullptr || workspace_size < required_workspace_size) {
        check_cuda(cudaErrorInvalidValue);
        return;
    }

    size_t sort_temp_size = 0;
    size_t reduce_temp_size = 0;
    query_scalar_temp_sizes<T>(count, op, stream, sort_temp_size, reduce_temp_size);

    char* workspace_bytes = static_cast<char*>(workspace);
    size_t offset = 0;
    int64_t* alt_keys = layout_array<int64_t>(workspace_bytes, offset, count);
    T* alt_values = layout_array<T>(workspace_bytes, offset, count);
    void* sort_temp = layout_bytes(workspace_bytes, offset, sort_temp_size);
    int* unique_dests = layout_array<int>(workspace_bytes, offset, count);
    T* aggregates = layout_array<T>(workspace_bytes, offset, count);
    int* num_runs = layout_array<int>(workspace_bytes, offset, 1);
    void* reduce_temp = layout_bytes(workspace_bytes, offset, reduce_temp_size);

    cub::DoubleBuffer<int64_t> d_keys(keys, alt_keys);
    cub::DoubleBuffer<T> d_values(values, alt_values);

    check_cuda(
        cub::DeviceRadixSort::SortPairs(
            sort_temp, sort_temp_size, d_keys, d_values, count, 0, sizeof(int64_t) * 8, stream
        )
    );

    auto dest_keys = thrust::make_transform_iterator(d_keys.Current(), DestIndexTransform {});

    ReduceByKeyOp<T> reduce_op { op };
    check_cuda(
        cub::DeviceReduce::ReduceByKey(
            reduce_temp, reduce_temp_size, dest_keys, unique_dests, d_values.Current(), aggregates, num_runs, reduce_op,
            count, stream
        )
    );

    const int num_blocks = deterministic_num_blocks(count);
    apply_reduced_runs_kernel<T><<<num_blocks, DETERMINISTIC_BLOCK_SIZE, 0, stream>>>(
        unique_dests, aggregates, num_runs, dest_array, dest_size, op
    );
    check_kernel_launch();
}

// Sort the scatter buffer by key using CUB radix sort, then launch the
// reduce kernel.
template <typename T>
void deterministic_sort_reduce_device(
    int64_t* keys,
    T* values,
    int count,
    T* dest_array,
    int dest_size,
    int op,
    int components,
    int determinism_level,
    void* workspace,
    size_t workspace_size
)
{
    if (count <= 0)
        return;

    if (components == 1 && determinism_level == DETERMINISTIC_RUN_TO_RUN) {
        deterministic_sort_reduce_device_scalar_run_to_run(
            keys, values, count, dest_array, dest_size, op, workspace, workspace_size
        );
        return;
    }

    ContextGuard guard(wp_cuda_context_get_current());
    cudaStream_t stream = static_cast<cudaStream_t>(wp_cuda_stream_get_current());

    size_t required_workspace_size = generic_workspace_size(count, stream);
    if (workspace == nullptr || workspace_size < required_workspace_size) {
        check_cuda(cudaErrorInvalidValue);
        return;
    }

    size_t sort_temp_size = 0;
    query_generic_sort_temp_size(count, stream, sort_temp_size);

    // --- Sort by key ---
    // The input buffers have a fixed capacity. Unused slots are initialized
    // with key == -1. CUB orders that signed key before valid non-negative
    // destination keys, and the reduce kernels ignore invalid destinations.
    // Sort keys together with record indices, then reduce from the original
    // value buffer component-wise.
    char* workspace_bytes = static_cast<char*>(workspace);
    size_t offset = 0;
    int64_t* alt_keys = layout_array<int64_t>(workspace_bytes, offset, count);
    int* record_indices = layout_array<int>(workspace_bytes, offset, count);
    int* alt_record_indices = layout_array<int>(workspace_bytes, offset, count);
    void* sort_temp = layout_bytes(workspace_bytes, offset, sort_temp_size);

    const int num_blocks = deterministic_num_blocks(count);
    init_record_indices_kernel<<<num_blocks, DETERMINISTIC_BLOCK_SIZE, 0, stream>>>(record_indices, count);
    if (!check_kernel_launch())
        return;

    cub::DoubleBuffer<int64_t> d_keys(keys, alt_keys);
    cub::DoubleBuffer<int> d_indices(record_indices, alt_record_indices);

    check_cuda(
        cub::DeviceRadixSort::SortPairs(
            sort_temp, sort_temp_size, d_keys, d_indices, count, 0, sizeof(int64_t) * 8, stream
        )
    );

    // --- Segmented reduce ---
    deterministic_reduce_kernel<T><<<num_blocks, DETERMINISTIC_BLOCK_SIZE, 0, stream>>>(
        d_keys.Current(), d_indices.Current(), values, count, components, dest_array, dest_size, op
    );
    check_kernel_launch();
}

void deterministic_counter_scan_device(
    int64_t* keys,
    int* values,
    int count,
    int* prefixes,
    const int* counter_bases,
    int* counter_totals,
    int counter_size,
    void* workspace,
    size_t workspace_size
)
{
    if (count <= 0)
        return;

    ContextGuard guard(wp_cuda_context_get_current());
    cudaStream_t stream = static_cast<cudaStream_t>(wp_cuda_stream_get_current());

    size_t required_workspace_size = counter_workspace_size(count, stream);
    if (workspace == nullptr || workspace_size < required_workspace_size) {
        check_cuda(cudaErrorInvalidValue);
        return;
    }

    size_t sort_temp_size = 0;
    size_t scan_temp_size = 0;
    query_generic_sort_temp_size(count, stream, sort_temp_size);
    query_counter_scan_temp_size(count, stream, scan_temp_size);

    char* workspace_bytes = static_cast<char*>(workspace);
    size_t offset = 0;
    int64_t* alt_keys = layout_array<int64_t>(workspace_bytes, offset, count);
    int* record_indices = layout_array<int>(workspace_bytes, offset, count);
    int* alt_record_indices = layout_array<int>(workspace_bytes, offset, count);
    void* sort_temp = layout_bytes(workspace_bytes, offset, sort_temp_size);
    CounterPrefixState* states = layout_array<CounterPrefixState>(workspace_bytes, offset, count);
    CounterPrefixState* inclusive_prefixes = layout_array<CounterPrefixState>(workspace_bytes, offset, count);
    void* scan_temp = layout_bytes(workspace_bytes, offset, scan_temp_size);

    const int num_blocks = deterministic_num_blocks(count);
    init_record_indices_kernel<<<num_blocks, DETERMINISTIC_BLOCK_SIZE, 0, stream>>>(record_indices, count);
    if (!check_kernel_launch())
        return;

    cub::DoubleBuffer<int64_t> d_keys(keys, alt_keys);
    cub::DoubleBuffer<int> d_indices(record_indices, alt_record_indices);

    check_cuda(
        cub::DeviceRadixSort::SortPairs(
            sort_temp, sort_temp_size, d_keys, d_indices, count, 0, sizeof(int64_t) * 8, stream
        )
    );

    make_counter_prefix_states_kernel<<<num_blocks, DETERMINISTIC_BLOCK_SIZE, 0, stream>>>(
        d_keys.Current(), d_indices.Current(), values, counter_bases, states, counter_size, count
    );
    if (!check_kernel_launch())
        return;

    CounterPrefixOp op {};
    check_cuda(
        cub::DeviceScan::InclusiveScan(scan_temp, scan_temp_size, states, inclusive_prefixes, op, count, stream)
    );

    scatter_counter_prefixes_kernel<<<num_blocks, DETERMINISTIC_BLOCK_SIZE, 0, stream>>>(
        d_keys.Current(), d_indices.Current(), values, inclusive_prefixes, prefixes, counter_totals, counter_size, count
    );
    check_kernel_launch();
}

void deterministic_counter_writeback_device(
    int64_t* keys, int count, const int* counter_totals, int* counters, int counter_size
)
{
    if (count <= 0)
        return;

    ContextGuard guard(wp_cuda_context_get_current());
    cudaStream_t stream = static_cast<cudaStream_t>(wp_cuda_stream_get_current());

    const int num_blocks = deterministic_num_blocks(count);
    writeback_counter_totals_kernel<<<num_blocks, DETERMINISTIC_BLOCK_SIZE, 0, stream>>>(
        keys, count, counter_totals, counters, counter_size
    );
    check_kernel_launch();
}

}  // anonymous namespace


// Public API entry points called from the Python runtime via ctypes.
// Arguments are passed as uint64_t pointers (matching the Warp convention).

size_t
wp_deterministic_sort_reduce_workspace_size(int count, int op, int scalar_type, int components, int determinism_level)
{
    ContextGuard guard(wp_cuda_context_get_current());
    cudaStream_t stream = static_cast<cudaStream_t>(wp_cuda_stream_get_current());

    switch (scalar_type) {
    case SCALAR_HALF:
        return deterministic_workspace_size<wp::half>(count, op, components, determinism_level, stream);
    case SCALAR_FLOAT:
        return deterministic_workspace_size<float>(count, op, components, determinism_level, stream);
    case SCALAR_DOUBLE:
        return deterministic_workspace_size<double>(count, op, components, determinism_level, stream);
    case SCALAR_INT:
        return deterministic_workspace_size<int>(count, op, components, determinism_level, stream);
    case SCALAR_UINT:
        return deterministic_workspace_size<unsigned int>(count, op, components, determinism_level, stream);
    case SCALAR_INT64:
        return deterministic_workspace_size<int64_t>(count, op, components, determinism_level, stream);
    case SCALAR_UINT64:
        return deterministic_workspace_size<uint64_t>(count, op, components, determinism_level, stream);
    default:
        return 0;
    }
}

size_t wp_deterministic_counter_scan_workspace_size(int count)
{
    ContextGuard guard(wp_cuda_context_get_current());
    cudaStream_t stream = static_cast<cudaStream_t>(wp_cuda_stream_get_current());
    return counter_workspace_size(count, stream);
}

void wp_deterministic_sort_reduce_device(
    uint64_t keys,
    uint64_t values,
    int count,
    uint64_t dest_array,
    int dest_size,
    int op,
    int scalar_type,
    int components,
    int determinism_level,
    uint64_t workspace,
    size_t workspace_size
)
{
    switch (scalar_type) {
    case SCALAR_HALF:
        deterministic_sort_reduce_device<wp::half>(
            reinterpret_cast<int64_t*>(keys), reinterpret_cast<wp::half*>(values), count,
            reinterpret_cast<wp::half*>(dest_array), dest_size, op, components, determinism_level,
            reinterpret_cast<void*>(workspace), workspace_size
        );
        break;
    case SCALAR_FLOAT:
        deterministic_sort_reduce_device<float>(
            reinterpret_cast<int64_t*>(keys), reinterpret_cast<float*>(values), count,
            reinterpret_cast<float*>(dest_array), dest_size, op, components, determinism_level,
            reinterpret_cast<void*>(workspace), workspace_size
        );
        break;
    case SCALAR_DOUBLE:
        deterministic_sort_reduce_device<double>(
            reinterpret_cast<int64_t*>(keys), reinterpret_cast<double*>(values), count,
            reinterpret_cast<double*>(dest_array), dest_size, op, components, determinism_level,
            reinterpret_cast<void*>(workspace), workspace_size
        );
        break;
    case SCALAR_INT:
        deterministic_sort_reduce_device<int>(
            reinterpret_cast<int64_t*>(keys), reinterpret_cast<int*>(values), count, reinterpret_cast<int*>(dest_array),
            dest_size, op, components, determinism_level, reinterpret_cast<void*>(workspace), workspace_size
        );
        break;
    case SCALAR_UINT:
        deterministic_sort_reduce_device<unsigned int>(
            reinterpret_cast<int64_t*>(keys), reinterpret_cast<unsigned int*>(values), count,
            reinterpret_cast<unsigned int*>(dest_array), dest_size, op, components, determinism_level,
            reinterpret_cast<void*>(workspace), workspace_size
        );
        break;
    case SCALAR_INT64:
        deterministic_sort_reduce_device<int64_t>(
            reinterpret_cast<int64_t*>(keys), reinterpret_cast<int64_t*>(values), count,
            reinterpret_cast<int64_t*>(dest_array), dest_size, op, components, determinism_level,
            reinterpret_cast<void*>(workspace), workspace_size
        );
        break;
    case SCALAR_UINT64:
        deterministic_sort_reduce_device<uint64_t>(
            reinterpret_cast<int64_t*>(keys), reinterpret_cast<uint64_t*>(values), count,
            reinterpret_cast<uint64_t*>(dest_array), dest_size, op, components, determinism_level,
            reinterpret_cast<void*>(workspace), workspace_size
        );
        break;
    default:
        break;
    }
}

void wp_deterministic_counter_scan_device(
    uint64_t keys,
    uint64_t values,
    int count,
    uint64_t prefixes,
    uint64_t counter_bases,
    uint64_t counter_totals,
    int counter_size,
    uint64_t workspace,
    size_t workspace_size
)
{
    deterministic_counter_scan_device(
        reinterpret_cast<int64_t*>(keys), reinterpret_cast<int*>(values), count, reinterpret_cast<int*>(prefixes),
        reinterpret_cast<int*>(counter_bases), reinterpret_cast<int*>(counter_totals), counter_size,
        reinterpret_cast<void*>(workspace), workspace_size
    );
}

void wp_deterministic_counter_writeback_device(
    uint64_t keys, int count, uint64_t counter_totals, uint64_t counters, int counter_size
)
{
    deterministic_counter_writeback_device(
        reinterpret_cast<int64_t*>(keys), count, reinterpret_cast<int*>(counter_totals),
        reinterpret_cast<int*>(counters), counter_size
    );
}
