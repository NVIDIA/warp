// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0

#include "builtin.h"
#include "warp.h"

#include "apic.h"
#include "apic_internal.h"
#include "cuda_util.h"
#include "sparse_util.h"
#include "temp_buffer.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>

#define THRUST_IGNORE_CUB_VERSION_CHECK

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>

extern CUcontext get_current_context();

namespace {

// Combined row+column value that can be radix-sorted with CUB
using BsrRowCol = uint64_t;

static constexpr BsrRowCol PRUNED_ROWCOL = ~BsrRowCol(0);

template <typename T, int BlockSize> struct BlockValue {
    T values[BlockSize];
};

template <typename T, int BlockSize> struct BlockReference {
    T* ptr;

    CUDA_CALLABLE operator BlockValue<T, BlockSize>() const
    {
        BlockValue<T, BlockSize> value;
        for (int i = 0; i < BlockSize; ++i) {
            value.values[i] = ptr[i];
        }
        return value;
    }

    CUDA_CALLABLE BlockReference& operator=(const BlockValue<T, BlockSize>& value)
    {
        for (int i = 0; i < BlockSize; ++i) {
            ptr[i] = value.values[i];
        }
        return *this;
    }

    CUDA_CALLABLE BlockReference& operator=(const BlockReference& other)
    {
        BlockValue<T, BlockSize> value = other;
        return *this = value;
    }
};

template <typename T, int BlockSize> struct BlockIterator {
    using difference_type = ptrdiff_t;
    using value_type = BlockValue<T, BlockSize>;
    using pointer = BlockValue<T, BlockSize>*;
    using reference = BlockReference<T, BlockSize>;
    using iterator_category = std::random_access_iterator_tag;

    T* ptr;
    int stride;

    CUDA_CALLABLE BlockIterator()
        : ptr(nullptr)
        , stride(BlockSize)
    {
    }

    CUDA_CALLABLE BlockIterator(T* ptr, int stride)
        : ptr(ptr)
        , stride(stride)
    {
    }

    CUDA_CALLABLE reference operator*() const { return { ptr }; }

    CUDA_CALLABLE reference operator[](difference_type offset) const { return { ptr + offset * stride }; }

    CUDA_CALLABLE BlockIterator& operator++()
    {
        ptr += stride;
        return *this;
    }

    CUDA_CALLABLE BlockIterator operator++(int)
    {
        BlockIterator tmp = *this;
        ptr += stride;
        return tmp;
    }

    CUDA_CALLABLE BlockIterator operator+(difference_type offset) const
    {
        return BlockIterator(ptr + offset * stride, stride);
    }

    CUDA_CALLABLE difference_type operator-(const BlockIterator& other) const { return (ptr - other.ptr) / stride; }

    CUDA_CALLABLE bool operator==(const BlockIterator& other) const { return ptr == other.ptr; }
    CUDA_CALLABLE bool operator!=(const BlockIterator& other) const { return ptr != other.ptr; }
};

struct BsrColumnIsActive {
    CUDA_CALLABLE bool operator()(const int& col) const { return col >= 0; }
};

template <typename T, int BlockSize>
void bsr_select_compact_value_chunk_impl(
    void* temp,
    size_t& temp_bytes,
    T* values,
    int block_size,
    int component_offset,
    const int* bsr_columns,
    int* selected_count,
    int nnz_upper_bound,
    cudaStream_t stream
)
{
    check_cuda(
        cub::DeviceSelect::FlaggedIf(
            temp, temp_bytes, BlockIterator<T, BlockSize>(values + component_offset, block_size), bsr_columns,
            selected_count, nnz_upper_bound, BsrColumnIsActive(), stream
        )
    );
}

template <typename T>
void bsr_select_compact_value_chunk(
    void* temp,
    size_t& temp_bytes,
    T* values,
    int block_size,
    int component_offset,
    int chunk_size,
    const int* bsr_columns,
    int* selected_count,
    int nnz_upper_bound,
    cudaStream_t stream
)
{
    switch (chunk_size) {
    case 1:
        bsr_select_compact_value_chunk_impl<T, 1>(
            temp, temp_bytes, values, block_size, component_offset, bsr_columns, selected_count, nnz_upper_bound, stream
        );
        break;
    case 2:
        bsr_select_compact_value_chunk_impl<T, 2>(
            temp, temp_bytes, values, block_size, component_offset, bsr_columns, selected_count, nnz_upper_bound, stream
        );
        break;
    case 4:
        bsr_select_compact_value_chunk_impl<T, 4>(
            temp, temp_bytes, values, block_size, component_offset, bsr_columns, selected_count, nnz_upper_bound, stream
        );
        break;
    case 8:
        bsr_select_compact_value_chunk_impl<T, 8>(
            temp, temp_bytes, values, block_size, component_offset, bsr_columns, selected_count, nnz_upper_bound, stream
        );
        break;
    case 16:
        bsr_select_compact_value_chunk_impl<T, 16>(
            temp, temp_bytes, values, block_size, component_offset, bsr_columns, selected_count, nnz_upper_bound, stream
        );
        break;
    default:
        check_cuda(cudaErrorInvalidValue);
        break;
    }
}

inline int bsr_select_compact_value_chunk_size(int remaining)
{
    if (remaining >= 16)
        return 16;
    if (remaining >= 8)
        return 8;
    if (remaining >= 4)
        return 4;
    if (remaining >= 2)
        return 2;
    return 1;
}

template <typename T>
void bsr_select_compact_value_chunks(
    void* temp,
    size_t& temp_bytes,
    T* values,
    int block_size,
    const int* bsr_columns,
    int* selected_count,
    int nnz_upper_bound,
    cudaStream_t stream
)
{
    const size_t available_temp_bytes = temp_bytes;
    size_t max_temp_bytes = 0;

    for (int comp = 0; comp < block_size;) {
        const int chunk_size = bsr_select_compact_value_chunk_size(block_size - comp);
        size_t chunk_temp_bytes = temp ? available_temp_bytes : 0;
        bsr_select_compact_value_chunk<T>(
            temp, chunk_temp_bytes, values, block_size, comp, chunk_size, bsr_columns, selected_count, nnz_upper_bound,
            stream
        );

        if (!temp && chunk_temp_bytes > max_temp_bytes) {
            max_temp_bytes = chunk_temp_bytes;
        }

        comp += chunk_size;
    }

    if (!temp) {
        temp_bytes = max_temp_bytes;
    }
}

CUDA_CALLABLE BsrRowCol bsr_combine_row_col(uint32_t row, uint32_t col)
{
    return (static_cast<uint64_t>(row) << 32) | col;
}

CUDA_CALLABLE uint32_t bsr_get_row(const BsrRowCol& row_col) { return row_col >> 32; }

CUDA_CALLABLE uint32_t bsr_get_col(const BsrRowCol& row_col) { return row_col & INT_MAX; }

template <typename T> struct BsrBlockIsNotZero {
    int block_size;
    const T* values;
    T zero_mask;

    BsrBlockIsNotZero(int block_size, const void* values, const uint64_t zero_mask)
        : block_size(block_size)
        , values(static_cast<const T*>(values))
        , zero_mask(static_cast<T>(zero_mask))
    {
    }

    CUDA_CALLABLE_DEVICE bool operator()(int block) const
    {
        if (!values)
            return true;

        const T* val = values + block * block_size;
        for (int i = 0; i < block_size; ++i, ++val) {
            if ((*val & zero_mask) != 0)
                return true;
        }
        return false;
    }
};

template <> struct BsrBlockIsNotZero<void> {
    BsrBlockIsNotZero(int block_size, const void* values, const uint64_t zero_mask) { }

    CUDA_CALLABLE_DEVICE bool operator()(int block) const { return true; }
};

struct BsrBlockInMask {
    const int nrow;
    const int ncol;
    const int* bsr_offsets;
    const int* bsr_row_counts;
    const int* bsr_columns;
    const int* device_nnz;

    CUDA_CALLABLE_DEVICE bool operator()(int index, int row, int col) const
    {
        if (device_nnz != nullptr && index >= *device_nnz)
            return false;

        if (row < 0 || row >= nrow || col < 0 || col >= ncol) {
            return false;
        }

        if (bsr_offsets == nullptr)
            return true;

        int lower = bsr_offsets[row];
        int upper = bsr_active_row_end(bsr_offsets, bsr_row_counts, row) - 1;

        while (lower < upper) {
            const int mid = lower + (upper - lower) / 2;

            if (bsr_columns[mid] < col) {
                lower = mid + 1;
            } else {
                upper = mid;
            }
        }

        return lower == upper && (bsr_columns[lower] == col);
    }
};

template <typename T>
__global__ void bsr_fill_triplet_key_values(
    const int nnz,
    const int* tpl_rows,
    const int* tpl_columns,
    const BsrBlockIsNotZero<T> nonZero,
    const BsrBlockInMask mask,
    int* block_indices,
    BsrRowCol* tpl_row_col
)
{
    int block = blockIdx.x * blockDim.x + threadIdx.x;
    if (block >= nnz)
        return;

    const int row = tpl_rows[block];
    const int col = tpl_columns[block];

    const BsrRowCol row_col = mask(block, row, col) && nonZero(block) ? bsr_combine_row_col(row, col) : PRUNED_ROWCOL;

    tpl_row_col[block] = row_col;
    block_indices[block] = block;
}

template <typename T>
__global__ void
bsr_find_row_offsets(uint32_t row_count, const T* d_nnz, const BsrRowCol* unique_row_col, int* row_offsets)
{
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row > row_count)
        return;

    const uint32_t nnz = *d_nnz;
    if (row == 0 || nnz == 0) {
        row_offsets[row] = 0;
        return;
    }

    if (bsr_get_row(unique_row_col[nnz - 1]) < row) {
        row_offsets[row] = nnz;
        return;
    }

    // binary search for row start
    uint32_t lower = 0;
    uint32_t upper = nnz - 1;
    while (lower < upper) {
        uint32_t mid = lower + (upper - lower) / 2;

        if (bsr_get_row(unique_row_col[mid]) < row) {
            lower = mid + 1;
        } else {
            upper = mid;
        }
    }

    row_offsets[row] = lower;
}

__global__ void bsr_set_column(const int* d_nnz, const BsrRowCol* unique_row_cols, int* bsr_cols)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *d_nnz)
        return;
    const BsrRowCol row_col = unique_row_cols[i];
    bsr_cols[i] = bsr_get_col(row_col);
}

__global__ void bsr_transpose_scatter_padded(
    const int col_count,
    const int* compact_offsets,
    const BsrRowCol* transposed_row_col,
    const int* sorted_src_block_indices,
    const int* transposed_bsr_offsets,
    int* transposed_bsr_row_counts,
    int* transposed_bsr_columns,
    int* src_block_indices,
    int* status
)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= col_count) {
        return;
    }

    const int compact_beg = compact_offsets[row];
    const int compact_end = compact_offsets[row + 1];
    const int block_count = compact_end - compact_beg;

    const int dest_beg = transposed_bsr_offsets[row];
    const int dest_capacity_end = transposed_bsr_offsets[row + 1];

    if (dest_beg + block_count > dest_capacity_end) {
        transposed_bsr_row_counts[row] = 0;
        if (status != nullptr) {
            atomicMax(status, BSR_STATUS_ROW_CAPACITY_EXCEEDED);
        }
        for (int block = dest_beg; block < dest_capacity_end; ++block) {
            src_block_indices[block] = -1;
        }
        return;
    }

    transposed_bsr_row_counts[row] = block_count;
    for (int block = dest_beg; block < dest_capacity_end; ++block) {
        const int local_block = block - dest_beg;
        if (local_block < block_count) {
            const int compact_block = compact_beg + local_block;
            transposed_bsr_columns[block] = bsr_get_col(transposed_row_col[compact_block]);
            src_block_indices[block] = sorted_src_block_indices[compact_block];
        } else {
            src_block_indices[block] = -1;
        }
    }
}

template <typename T>
void launch_bsr_fill_triplet_key_values(
    const int block_size,
    const int nnz,
    const BsrBlockInMask& mask,
    const int* tpl_rows,
    const int* tpl_columns,
    const void* tpl_values,
    const uint64_t scalar_zero_mask,
    int* block_indices,
    BsrRowCol* row_col
)
{
    BsrBlockIsNotZero<T> isNotZero { block_size, tpl_values, scalar_zero_mask };
    wp_launch_device(
        WP_CURRENT_CONTEXT, bsr_fill_triplet_key_values, nnz,
        (nnz, tpl_rows, tpl_columns, isNotZero, mask, block_indices, row_col)
    );
}

__global__ void bsr_transpose_fill_row_col(
    const int nnz_upper_bound,
    const int row_count,
    const int* bsr_offsets,
    const int* bsr_row_counts,
    const int* bsr_columns,
    int* block_indices,
    BsrRowCol* transposed_row_col
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= nnz_upper_bound) {
        // Outside of allocated bounds, do nothing
        return;
    }

    block_indices[i] = i;

    if (i >= bsr_offsets[row_count]) {
        // Below upper bound but above actual nnz count, mark as invalid
        transposed_row_col[i] = PRUNED_ROWCOL;
        return;
    }

    // Binary search for row
    int lower = 0;
    int upper = row_count - 1;

    while (lower < upper) {
        int mid = lower + (upper - lower) / 2;

        if (bsr_offsets[mid + 1] <= i) {
            lower = mid + 1;
        } else {
            upper = mid;
        }
    }

    const int row = lower;
    if (i >= bsr_active_row_end(bsr_offsets, bsr_row_counts, row)) {
        transposed_row_col[i] = PRUNED_ROWCOL;
        return;
    }

    const int col = bsr_columns[i];
    BsrRowCol row_col = bsr_combine_row_col(col, row);
    transposed_row_col[i] = row_col;
}

__global__ void
bsr_count_active_blocks(const int row_count, const int* bsr_offsets, const int* bsr_row_counts, int* active_count)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count)
        return;

    atomicAdd(active_count, bsr_active_row_end(bsr_offsets, bsr_row_counts, row) - bsr_offsets[row]);
}

static constexpr int BSR_COMPRESS_MIN_INPLACE_THREADS = 64;
static constexpr int BSR_COMPRESS_MAX_INPLACE_THREADS = 1024;
static constexpr int BSR_COMPRESS_MAX_INPLACE_WARPS = BSR_COMPRESS_MAX_INPLACE_THREADS / WP_TILE_WARP_SIZE;
static constexpr int BSR_COMPRESS_SHORT_ROW_CAPACITY = 16;
static constexpr int BSR_COMPRESS_DEFAULT_BLOCKS_PER_THREAD = 4;
static constexpr int BSR_COMPRESS_FLOAT64_BLOCKS_PER_THREAD = 2;
static constexpr int BSR_COMPRESS_STAGED_ROW_SUM_CAPACITY = 16;
static constexpr int BSR_COMPRESS_TOPOLOGY_INTS_PER_SLOT = 3;
static constexpr int BSR_COMPRESS_INVALID_COLUMN = 0x7fffffff;

CUDA_CALLABLE_DEVICE int bsr_compress_column_key(int col) { return col >= 0 ? col : BSR_COMPRESS_INVALID_COLUMN; }

template <typename T, bool CompressValues>
__global__ void bsr_compress_inplace_rows_serial_kernel(
    const int row_count,
    const int block_size,
    const int* bsr_offsets,
    const int* bsr_row_counts,
    int* out_row_counts,
    int* bsr_columns,
    T* bsr_values,
    const T* prune_values,
    const uint64_t prune_zero_mask
)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) {
        return;
    }

    bsr_compress_inplace_row<T, CompressValues>(
        row, block_size, bsr_offsets, bsr_row_counts, out_row_counts, bsr_columns, bsr_values, prune_values,
        prune_zero_mask
    );
}

CUDA_CALLABLE int bsr_next_power_of_two(int value)
{
    if (value <= 1) {
        return 1;
    }

    const uint32_t bits = uint32_t(value - 1);
#if defined(__CUDA_ARCH__)
    return int(uint32_t(1) << (32 - __clz(int(bits))));
#elif defined(__GNUC__) || defined(__clang__)
    return int(uint32_t(1) << (32 - __builtin_clz(static_cast<unsigned int>(bits))));
#else
    uint32_t power = 1;
    while (power <= bits) {
        power <<= 1;
    }
    return int(power);
#endif
}

template <typename T, bool CompressValues>
CUDA_CALLABLE_DEVICE int
bsr_compress_row_sum_capacity(const size_t shared_memory_bytes, const int sort_capacity, const int block_size)
{
    if constexpr (!CompressValues) {
        return 0;
    } else {
        const size_t topology_bytes = size_t(sort_capacity) * BSR_COMPRESS_TOPOLOGY_INTS_PER_SLOT * sizeof(int);
        if (topology_bytes >= shared_memory_bytes) {
            return 0;
        }

        const size_t value_bytes = shared_memory_bytes - topology_bytes;
        const size_t staged_capacity = value_bytes / (sizeof(T) * size_t(block_size));
        return int(
            staged_capacity < size_t(BSR_COMPRESS_STAGED_ROW_SUM_CAPACITY)
                ? staged_capacity
                : size_t(BSR_COMPRESS_STAGED_ROW_SUM_CAPACITY)
        );
    }
}

CUDA_CALLABLE_DEVICE int bsr_compress_select_sorted_runs(
    const int count,
    const int* sorted_columns,
    const int previous_column,
    int* selected_columns,
    int* selected_run_starts
)
{
    constexpr unsigned int full_warp_mask = 0xffffffffu;

    const int tid = threadIdx.x;
    const int lane = tid & (WP_TILE_WARP_SIZE - 1);
    const int warp = tid / WP_TILE_WARP_SIZE;
    const int warp_count = blockDim.x / WP_TILE_WARP_SIZE;

    __shared__ int warp_offsets[BSR_COMPRESS_MAX_INPLACE_WARPS];
    __shared__ int tile_run_count;

    int selected_count = 0;

    for (int base = 0; base < count; base += blockDim.x) {
        const int block = base + tid;
        int col = BSR_COMPRESS_INVALID_COLUMN;
        bool run_start = false;
        if (block < count) {
            col = sorted_columns[block];
            const int prev_col = block == 0 ? previous_column : sorted_columns[block - 1];
            run_start = col >= 0 && col != BSR_COMPRESS_INVALID_COLUMN && prev_col != col;
        }

        const unsigned int keep_mask = __ballot_sync(full_warp_mask, run_start);
        const int warp_total = __popc(keep_mask);
        const int lane_prefix = __popc(keep_mask & ((1u << lane) - 1u));

        if (lane == WP_TILE_WARP_SIZE - 1) {
            warp_offsets[warp] = warp_total;
        }
        __syncthreads();

        if (warp == 0) {
            int warp_prefix = lane < warp_count ? warp_offsets[lane] : 0;
#pragma unroll
            for (int offset = 1; offset < WP_TILE_WARP_SIZE; offset <<= 1) {
                const int other = __shfl_up_sync(full_warp_mask, warp_prefix, offset, WP_TILE_WARP_SIZE);
                if (lane >= offset) {
                    warp_prefix += other;
                }
            }

            if (lane < warp_count) {
                warp_offsets[lane] = warp_prefix - warp_offsets[lane];
            }
            if (lane == warp_count - 1) {
                tile_run_count = warp_prefix;
            }
        }
        __syncthreads();

        const int tile_offset = selected_count + warp_offsets[warp] + lane_prefix;
        if (run_start) {
            selected_columns[tile_offset] = col;
            selected_run_starts[tile_offset] = block;
        }

        selected_count += tile_run_count;
        __syncthreads();
    }

    return selected_count;
}

CUDA_CALLABLE_DEVICE void bsr_compress_load_indexed_row(
    const int row_beg,
    const int candidate_count,
    const int sort_count,
    const int* bsr_columns,
    int* shared_columns,
    int* shared_indices
)
{
    const int tid = threadIdx.x;

    for (int block = tid; block < sort_count; block += blockDim.x) {
        int col = BSR_COMPRESS_INVALID_COLUMN;
        int src = -1;
        if (block < candidate_count) {
            const int candidate = row_beg + block;
            col = bsr_compress_column_key(bsr_columns[candidate]);
            if (col != BSR_COMPRESS_INVALID_COLUMN) {
                src = block;
            }
        }

        shared_columns[block] = col;
        shared_indices[block] = src;
    }
}

CUDA_CALLABLE_DEVICE void bsr_compress_sort_indexed_row(const int sort_count, int* shared_columns, int* shared_indices)
{
    const int tid = threadIdx.x;

    for (int k = 2; k <= sort_count; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int block = tid; block < sort_count; block += blockDim.x) {
                const int partner = block ^ j;
                if (partner > block) {
                    const bool ascending = (block & k) == 0;
                    const int col_a = shared_columns[block];
                    const int col_b = shared_columns[partner];
                    const bool should_swap = ascending ? (col_a > col_b) : (col_a < col_b);
                    if (should_swap) {
                        shared_columns[block] = col_b;
                        shared_columns[partner] = col_a;

                        const int index_a = shared_indices[block];
                        shared_indices[block] = shared_indices[partner];
                        shared_indices[partner] = index_a;
                    }
                }
            }
            __syncthreads();
        }
    }
}

template <typename T>
CUDA_CALLABLE_DEVICE void bsr_compress_stage_indexed_row_sums(
    const int row_beg,
    const int sort_count,
    const int kept_count,
    const int block_size,
    const int* shared_columns,
    const int* shared_indices,
    const int* shared_run_starts,
    T* bsr_values,
    T* shared_value_scratch
)
{
    const int tid = threadIdx.x;

    // Stage sums before writing compact values because output slots may alias
    // unread source values.
    for (int comp = tid; comp < block_size; comp += blockDim.x) {
        for (int block = 0; block < kept_count; ++block) {
            const int run_start = shared_run_starts[block];
            const int col = shared_columns[run_start];
            T sum = T(0);
            int read = run_start;
            while (read < sort_count && shared_columns[read] == col) {
                sum += bsr_values[(row_beg + shared_indices[read]) * block_size + comp];
                ++read;
            }
            shared_value_scratch[block * block_size + comp] = sum;
        }

        for (int block = 0; block < kept_count; ++block) {
            const int out = row_beg + block;
            bsr_values[out * block_size + comp] = shared_value_scratch[block * block_size + comp];
        }
    }
}

template <typename T>
CUDA_CALLABLE_DEVICE void bsr_compress_stream_indexed_row_sums(
    const int row_beg,
    const int candidate_count,
    const int sort_count,
    const int kept_count,
    const int block_size,
    const int* shared_columns,
    const int* shared_indices,
    const int* shared_run_starts,
    T* bsr_values,
    T* shared_value_scratch
)
{
    const int tid = threadIdx.x;

    // Cache one component at a time, then write compact sums directly.
    for (int comp = 0; comp < block_size; ++comp) {
        for (int block = tid; block < candidate_count; block += blockDim.x) {
            shared_value_scratch[block] = bsr_values[(row_beg + block) * block_size + comp];
        }
        __syncthreads();

        for (int block = tid; block < kept_count; block += blockDim.x) {
            const int run_start = shared_run_starts[block];
            const int col = shared_columns[run_start];
            T sum = T(0);
            int read = run_start;
            while (read < sort_count && shared_columns[read] == col) {
                sum += shared_value_scratch[shared_indices[read]];
                ++read;
            }
            const int out = row_beg + block;
            bsr_values[out * block_size + comp] = sum;
        }
        if (comp + 1 < block_size) {
            __syncthreads();
        }
    }
}

template <typename T, bool CompressValues>
CUDA_CALLABLE_DEVICE int bsr_compress_inplace_row_indexed_shared(
    const int row_beg,
    const int row_end,
    const int block_size,
    int* bsr_columns,
    T* bsr_values,
    int* shared_columns,
    int* shared_indices,
    int* shared_run_starts,
    T* shared_value_scratch,
    const int row_sum_capacity
)
{
    const int candidate_count = row_end - row_beg;
    const int sort_count = bsr_next_power_of_two(candidate_count);

    bsr_compress_load_indexed_row(row_beg, candidate_count, sort_count, bsr_columns, shared_columns, shared_indices);
    __syncthreads();

    bsr_compress_sort_indexed_row(sort_count, shared_columns, shared_indices);

    const int kept_count = bsr_compress_select_sorted_runs(
        sort_count, shared_columns, BSR_COMPRESS_INVALID_COLUMN, bsr_columns + row_beg, shared_run_starts
    );

    if constexpr (CompressValues) {
        if (sort_count <= row_sum_capacity) {
            bsr_compress_stage_indexed_row_sums<T>(
                row_beg, sort_count, kept_count, block_size, shared_columns, shared_indices, shared_run_starts,
                bsr_values, shared_value_scratch
            );
        } else {
            bsr_compress_stream_indexed_row_sums<T>(
                row_beg, candidate_count, sort_count, kept_count, block_size, shared_columns, shared_indices,
                shared_run_starts, bsr_values, shared_value_scratch
            );
        }
    }

    return kept_count;
}

template <typename T, bool CompressValues>
CUDA_CALLABLE_DEVICE void bsr_compress_swap_blocks(int a, int b, int block_size, int* bsr_columns, T* bsr_values)
{
    if (a == b) {
        return;
    }

    const int col = bsr_columns[a];
    bsr_columns[a] = bsr_columns[b];
    bsr_columns[b] = col;

    if constexpr (CompressValues) {
        T* values_a = bsr_values + a * block_size;
        T* values_b = bsr_values + b * block_size;
        for (int comp = 0; comp < block_size; ++comp, ++values_a, ++values_b) {
            const T value = *values_a;
            *values_a = *values_b;
            *values_b = value;
        }
    }
}

template <typename T, bool CompressValues>
CUDA_CALLABLE_DEVICE void
bsr_compress_sift_down_global(int row_beg, int root, int count, int block_size, int* bsr_columns, T* bsr_values)
{
    while (true) {
        int child = 2 * root + 1;
        if (child >= count) {
            break;
        }

        const int right = child + 1;
        if (right < count
            && bsr_compress_column_key(bsr_columns[row_beg + child])
                < bsr_compress_column_key(bsr_columns[row_beg + right])) {
            child = right;
        }

        if (bsr_compress_column_key(bsr_columns[row_beg + root])
            >= bsr_compress_column_key(bsr_columns[row_beg + child])) {
            break;
        }

        bsr_compress_swap_blocks<T, CompressValues>(
            row_beg + root, row_beg + child, block_size, bsr_columns, bsr_values
        );
        root = child;
    }
}

template <typename T, bool CompressValues>
CUDA_CALLABLE_DEVICE void
bsr_compress_sort_row_global(int row_beg, int count, int block_size, int* bsr_columns, T* bsr_values)
{
    for (int start = (count - 2) / 2; start >= 0; --start) {
        bsr_compress_sift_down_global<T, CompressValues>(row_beg, start, count, block_size, bsr_columns, bsr_values);
    }

    for (int end = count - 1; end > 0; --end) {
        bsr_compress_swap_blocks<T, CompressValues>(row_beg, row_beg + end, block_size, bsr_columns, bsr_values);
        bsr_compress_sift_down_global<T, CompressValues>(row_beg, 0, end, block_size, bsr_columns, bsr_values);
    }
}

template <typename T, bool CompressValues>
CUDA_CALLABLE_DEVICE int bsr_compress_sorted_row(
    const int row_beg,
    const int active_count,
    const int block_size,
    int* bsr_columns,
    T* bsr_values,
    int* shared_columns,
    int* shared_run_starts,
    T* shared_value_scratch,
    const int shared_sort_capacity
)
{
    const int tid = threadIdx.x;

    if constexpr (CompressValues) {
        for (int block = tid; block < active_count; block += blockDim.x) {
            const int col = bsr_columns[row_beg + block];
            const bool run_start = col >= 0 && (block == 0 || bsr_columns[row_beg + block - 1] != col);
            if (run_start) {
                for (int comp = 0; comp < block_size; ++comp) {
                    T sum = T(0);
                    int read = block;
                    while (read < active_count && bsr_columns[row_beg + read] == col) {
                        sum += bsr_values[(row_beg + read) * block_size + comp];
                        ++read;
                    }
                    bsr_values[(row_beg + block) * block_size + comp] = sum;
                }
            }
        }
        __syncthreads();
    }

    int kept_count = 0;
    for (int chunk_beg = 0; chunk_beg < active_count; chunk_beg += shared_sort_capacity) {
        const int remaining_count = active_count - chunk_beg;
        const int chunk_count = shared_sort_capacity < remaining_count ? shared_sort_capacity : remaining_count;
        const int previous_column = chunk_beg == 0 ? BSR_COMPRESS_INVALID_COLUMN : bsr_columns[row_beg + chunk_beg - 1];
        const int chunk_kept_count = bsr_compress_select_sorted_runs(
            chunk_count, bsr_columns + row_beg + chunk_beg, previous_column, shared_columns, shared_run_starts
        );

        if constexpr (CompressValues) {
            for (int comp = 0; comp < block_size; ++comp) {
                for (int block = tid; block < chunk_kept_count; block += blockDim.x) {
                    const int src_offset = shared_run_starts[block];
                    shared_value_scratch[block] = bsr_values[(row_beg + chunk_beg + src_offset) * block_size + comp];
                }
                __syncthreads();

                for (int block = tid; block < chunk_kept_count; block += blockDim.x) {
                    bsr_values[(row_beg + kept_count + block) * block_size + comp] = shared_value_scratch[block];
                }
                __syncthreads();
            }
        }

        for (int block = tid; block < chunk_kept_count; block += blockDim.x) {
            bsr_columns[row_beg + kept_count + block] = shared_columns[block];
        }
        __syncthreads();

        kept_count += chunk_kept_count;
    }
    return kept_count;
}

template <typename T, bool CompressValues>
CUDA_CALLABLE_DEVICE int bsr_compress_inplace_row_global(
    const int row_beg,
    const int row_end,
    const int block_size,
    int* bsr_columns,
    T* bsr_values,
    int* shared_columns,
    int* shared_indices,
    int* shared_run_starts,
    T* shared_value_scratch,
    const int row_sum_capacity,
    int shared_sort_capacity
)
{
    const int tid = threadIdx.x;
    const int candidate_count = row_end - row_beg;

    int active_count = candidate_count;
    bool use_heap_fallback = false;
    while (active_count > shared_sort_capacity) {
        int write_count = 0;
        for (int chunk_beg = 0; chunk_beg < active_count; chunk_beg += shared_sort_capacity) {
            const int remaining_count = active_count - chunk_beg;
            const int chunk_count = shared_sort_capacity < remaining_count ? shared_sort_capacity : remaining_count;
            const int kept_count = bsr_compress_inplace_row_indexed_shared<T, CompressValues>(
                row_beg + chunk_beg, row_beg + chunk_beg + chunk_count, block_size, bsr_columns, bsr_values,
                shared_columns, shared_indices, shared_run_starts, shared_value_scratch, row_sum_capacity
            );
            __syncthreads();

            if (write_count != chunk_beg) {
                for (int block = tid; block < kept_count; block += blockDim.x) {
                    const int src = row_beg + chunk_beg + block;
                    shared_columns[block] = bsr_columns[src];
                }
                __syncthreads();

                for (int block = tid; block < kept_count; block += blockDim.x) {
                    const int dst = row_beg + write_count + block;
                    bsr_columns[dst] = shared_columns[block];
                }
                __syncthreads();

                if constexpr (CompressValues) {
                    for (int comp = 0; comp < block_size; ++comp) {
                        for (int block = tid; block < kept_count; block += blockDim.x) {
                            const int src = row_beg + chunk_beg + block;
                            shared_value_scratch[block] = bsr_values[src * block_size + comp];
                        }
                        __syncthreads();

                        for (int block = tid; block < kept_count; block += blockDim.x) {
                            const int dst = row_beg + write_count + block;
                            bsr_values[dst * block_size + comp] = shared_value_scratch[block];
                        }
                        __syncthreads();
                    }
                }
            }
            write_count += kept_count;
        }

        if (write_count >= active_count) {
            use_heap_fallback = true;
            break;
        }
        active_count = write_count;
    }

    if (use_heap_fallback) {
        if (tid == 0) {
            bsr_compress_sort_row_global<T, CompressValues>(row_beg, active_count, block_size, bsr_columns, bsr_values);
        }
        __syncthreads();

        return bsr_compress_sorted_row<T, CompressValues>(
            row_beg, active_count, block_size, bsr_columns, bsr_values, shared_columns, shared_run_starts,
            shared_value_scratch, shared_sort_capacity
        );
    } else {
        return bsr_compress_inplace_row_indexed_shared<T, CompressValues>(
            row_beg, row_beg + active_count, block_size, bsr_columns, bsr_values, shared_columns, shared_indices,
            shared_run_starts, shared_value_scratch, row_sum_capacity
        );
    }
}

template <typename T, bool CompressValues>
__global__ void bsr_compress_inplace_rows_indexed_kernel(
    const int row_count,
    const int block_size,
    const int shared_sort_capacity,
    const size_t shared_memory_bytes,
    const int* bsr_offsets,
    const int* bsr_row_counts,
    int* out_row_counts,
    int* bsr_columns,
    T* bsr_values,
    const T* prune_values,
    const uint64_t prune_zero_mask
)
{
    extern __shared__ int shared_ints[];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= row_count) {
        return;
    }

    const int row_beg = bsr_offsets[row];
    const int row_end = bsr_active_row_end(bsr_offsets, bsr_row_counts, row);
    const int candidate_count = row_end - row_beg;

    if (prune_values != nullptr) {
        for (int block = tid; block < candidate_count; block += blockDim.x) {
            const int candidate = row_beg + block;
            if (bsr_columns[candidate] >= 0
                && bsr_typed_block_is_zero(candidate, block_size, prune_values, prune_zero_mask)) {
                bsr_columns[candidate] = -1;
            }
        }
        __syncthreads();
    }

    if (candidate_count <= 1) {
        if (tid == 0) {
            out_row_counts[row] = candidate_count == 1 && bsr_columns[row_beg] >= 0 ? 1 : 0;
        }
        return;
    }

    int kept_count = 0;

    const int row_sort_capacity = bsr_next_power_of_two(candidate_count);
    const bool use_shared_sort = row_sort_capacity <= shared_sort_capacity;
    const int layout_sort_capacity = use_shared_sort ? row_sort_capacity : shared_sort_capacity;

    int* shared_columns = shared_ints;
    int* shared_indices = shared_columns + layout_sort_capacity;
    int* shared_run_starts = shared_indices + layout_sort_capacity;
    T* shared_value_scratch = reinterpret_cast<T*>(shared_run_starts + layout_sort_capacity);

    const int row_sum_capacity
        = bsr_compress_row_sum_capacity<T, CompressValues>(shared_memory_bytes, layout_sort_capacity, block_size);

    if (use_shared_sort) {
        kept_count = bsr_compress_inplace_row_indexed_shared<T, CompressValues>(
            row_beg, row_end, block_size, bsr_columns, bsr_values, shared_columns, shared_indices, shared_run_starts,
            shared_value_scratch, row_sum_capacity
        );
    } else {
        kept_count = bsr_compress_inplace_row_global<T, CompressValues>(
            row_beg, row_end, block_size, bsr_columns, bsr_values, shared_columns, shared_indices, shared_run_starts,
            shared_value_scratch, row_sum_capacity, shared_sort_capacity
        );
    }

    if (tid == 0) {
        out_row_counts[row] = kept_count;
    }
}

__global__ void bsr_compress_inplace_compact_counts_kernel(
    const int row_count, const int* bsr_offsets, int* row_counts, int* bsr_columns
)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    if (row == 0 && tid == 0) {
        row_counts[row_count] = 0;
    }
    if (row >= row_count) {
        return;
    }

    const int row_beg = bsr_offsets[row];
    const int row_end = row_beg + row_counts[row];
    const int row_capacity_end = bsr_offsets[row + 1];

    for (int block = row_end + tid; block < row_capacity_end; block += blockDim.x) {
        bsr_columns[block] = -1;
    }
}

}  // namespace

WP_API void wp_bsr_matrix_from_triplets_device(
    const int block_size,
    int scalar_size,
    const int row_count,
    const int col_count,
    const int nnz,
    const int* tpl_nnz,
    const int* tpl_rows,
    const int* tpl_columns,
    const void* tpl_values,
    const uint64_t scalar_zero_mask,
    const bool masked_topology,
    int* tpl_block_offsets,
    int* tpl_block_indices,
    int* bsr_offsets,
    const int* bsr_row_counts,
    int* bsr_columns,
    int* bsr_nnz,
    void* bsr_nnz_event
)
{
    void* context = wp_cuda_context_get_current();
    ContextGuard guard(context);

    cudaStream_t stream = static_cast<cudaStream_t>(wp_cuda_stream_get_current());

    ScopedTemporary<BsrRowCol> combined_row_col(context, 2 * size_t(nnz));
    ScopedTemporary<int> unique_triplet_count(context, 1);

    bool return_summed_blocks = tpl_block_offsets != nullptr && tpl_block_indices != nullptr;
    if (!return_summed_blocks) {
        // if not provided, allocate temporary offset and indices buffers
        tpl_block_offsets = static_cast<int*>(wp_alloc_device(context, size_t(nnz) * sizeof(int), "(native:sparse)"));
        tpl_block_indices = static_cast<int*>(wp_alloc_device(context, size_t(nnz) * sizeof(int), "(native:sparse)"));
    }

    cub::DoubleBuffer<int> d_keys(tpl_block_indices, tpl_block_offsets);
    cub::DoubleBuffer<BsrRowCol> d_values(combined_row_col.buffer(), combined_row_col.buffer() + nnz);

    // Combine rows and columns so we can sort on them both,
    // ensuring that blocks that should be pruned are moved to the end
    BsrBlockInMask mask {
        row_count,   col_count, masked_topology ? bsr_offsets : nullptr, masked_topology ? bsr_row_counts : nullptr,
        bsr_columns, tpl_nnz,
    };
    if (scalar_zero_mask == 0 || tpl_values == nullptr)
        scalar_size = 0;
    switch (scalar_size) {
    case sizeof(uint8_t):
        launch_bsr_fill_triplet_key_values<uint8_t>(
            block_size, nnz, mask, tpl_rows, tpl_columns, tpl_values, scalar_zero_mask, d_keys.Current(),
            d_values.Current()
        );
        break;
    case sizeof(uint16_t):
        launch_bsr_fill_triplet_key_values<uint16_t>(
            block_size, nnz, mask, tpl_rows, tpl_columns, tpl_values, scalar_zero_mask, d_keys.Current(),
            d_values.Current()
        );
        break;
    case sizeof(uint32_t):
        launch_bsr_fill_triplet_key_values<uint32_t>(
            block_size, nnz, mask, tpl_rows, tpl_columns, tpl_values, scalar_zero_mask, d_keys.Current(),
            d_values.Current()
        );
        break;
    case sizeof(uint64_t):
        launch_bsr_fill_triplet_key_values<uint64_t>(
            block_size, nnz, mask, tpl_rows, tpl_columns, tpl_values, scalar_zero_mask, d_keys.Current(),
            d_values.Current()
        );
        break;
    default:
        // no scalar-level pruning
        launch_bsr_fill_triplet_key_values<void>(
            block_size, nnz, mask, tpl_rows, tpl_columns, tpl_values, scalar_zero_mask, d_keys.Current(),
            d_values.Current()
        );
        break;
    }

    // Sort
    {
        size_t buff_size = 0;
        check_cuda(cub::DeviceRadixSort::SortPairs(nullptr, buff_size, d_values, d_keys, nnz, 0, 64, stream));
        ScopedTemporary<> temp(context, buff_size);
        check_cuda(cub::DeviceRadixSort::SortPairs(temp.buffer(), buff_size, d_values, d_keys, nnz, 0, 64, stream));

        // Depending on data size and GPU architecture buffers may have been swapped
        // or not Ensures the sorted keys are available in summed_block_indices if
        // needed
        if (return_summed_blocks && d_keys.Current() != tpl_block_indices) {
            check_cuda(cudaMemcpyAsync(
                tpl_block_indices, d_keys.Current(), nnz * sizeof(int), cudaMemcpyDeviceToDevice, stream
            ));
        }
    }

    // Runlength encode row-col sequences
    {
        size_t buff_size = 0;
        check_cuda(
            cub::DeviceRunLengthEncode::Encode(
                nullptr, buff_size, d_values.Current(), d_values.Alternate(), tpl_block_offsets,
                unique_triplet_count.buffer(), nnz, stream
            )
        );
        ScopedTemporary<> temp(context, buff_size);
        check_cuda(
            cub::DeviceRunLengthEncode::Encode(
                temp.buffer(), buff_size, d_values.Current(), d_values.Alternate(), tpl_block_offsets,
                unique_triplet_count.buffer(), nnz, stream
            )
        );
    }

    // Compute row offsets from sorted unique blocks
    wp_launch_device(
        WP_CURRENT_CONTEXT, bsr_find_row_offsets, row_count + 1,
        (row_count, unique_triplet_count.buffer(), d_values.Alternate(), bsr_offsets)
    );

    if (bsr_nnz) {
        // Copy nnz to host, and record an event for the completed transfer if
        // desired

        wp_memcpy_d2h(WP_CURRENT_CONTEXT, bsr_nnz, bsr_offsets + row_count, sizeof(int), stream);

        if (bsr_nnz_event) {
            const bool external = true;
            wp_cuda_event_record(bsr_nnz_event, stream, external);
        }
    }

    // Set column indices
    wp_launch_device(
        WP_CURRENT_CONTEXT, bsr_set_column, nnz, (bsr_offsets + row_count, d_values.Alternate(), bsr_columns)
    );

    // Scan repeated block counts
    if (return_summed_blocks) {
        size_t buff_size = 0;
        check_cuda(
            cub::DeviceScan::InclusiveSum(nullptr, buff_size, tpl_block_offsets, tpl_block_offsets, nnz, stream)
        );
        ScopedTemporary<> temp(context, buff_size);
        check_cuda(
            cub::DeviceScan::InclusiveSum(temp.buffer(), buff_size, tpl_block_offsets, tpl_block_offsets, nnz, stream)
        );
    } else {
        // free our temporary buffers
        wp_free_device(context, tpl_block_offsets);
        wp_free_device(context, tpl_block_indices);
    }
}

// Record-and-execute a BSR transpose under CUDA APIC capture: record the source
// and (output) transposed region addresses into the byte stream, then fall
// through so the live transpose issues onto the captured stream. Mirror of
// apic_capture_bsr_transpose in sparse.cpp, device-scoped and non-skipping. The
// transpose does only D2D copies + a CUB sort + temp allocs (no host readback),
// so it is fully capturable. No-op outside a CUDA APIC capture.
static void apic_capture_bsr_transpose_device(
    int row_count,
    int col_count,
    int nnz,
    const int* bsr_offsets,
    const int* bsr_row_counts,
    const int* bsr_columns,
    int* transposed_bsr_offsets,
    int* transposed_bsr_row_counts,
    int* transposed_bsr_columns,
    int* src_block_indices,
    int* status
)
{
    APICState* state = wp_apic_get_cuda_recording_state();
    if (!state || nnz <= 0)
        return;
    if (!bsr_offsets || !bsr_columns || !transposed_bsr_offsets || !transposed_bsr_columns || !src_block_indices)
        return;

    uint64_t int_bytes = static_cast<uint64_t>(nnz) * sizeof(int32_t);
    uint64_t rowp1_bytes = (static_cast<uint64_t>(row_count) + 1) * sizeof(int32_t);
    uint64_t colp1_bytes = (static_cast<uint64_t>(col_count) + 1) * sizeof(int32_t);
    // transposed_bsr_offsets is a device buffer holding the padded capacity (for
    // padded destinations) that the live CUDA replay re-reads in place, so no
    // host-side capacity snapshot is recorded here (unlike the CPU path, which
    // snapshots it for byte-stream replay -- GH-1587).
    //
    // transposed_bsr_columns spans the destination's capacity, which is also
    // device-resident (transposed_bsr_offsets[col_count]) and cannot be read
    // during capture. Resolve it with a minimal span: Python tracks the array's
    // real extent before the native call, so the pointer resolves into that
    // region. Claiming the source's nnz upper bound instead would grow the
    // region past the real allocation of a smaller padded destination,
    // corrupting pointer resolution for the rest of the capture.
    uint64_t transposed_columns_bytes = sizeof(int32_t);
    // src_block_indices doubles as CUB sort scratch: the DoubleBuffer key
    // halves span 2*nnz entries.
    uint64_t block_indices_bytes = 2 * int_bytes;

    APICAddress bo_addr = apic_resolve_live_ptr(state, reinterpret_cast<uint64_t>(bsr_offsets), rowp1_bytes);
    APICAddress brc_addr;
    if (bsr_row_counts)
        brc_addr = apic_resolve_live_ptr(
            state, reinterpret_cast<uint64_t>(bsr_row_counts), static_cast<uint64_t>(row_count) * sizeof(int32_t)
        );
    APICAddress bc_addr = apic_resolve_live_ptr(state, reinterpret_cast<uint64_t>(bsr_columns), int_bytes);
    APICAddress to_addr = apic_resolve_live_ptr(state, reinterpret_cast<uint64_t>(transposed_bsr_offsets), colp1_bytes);
    APICAddress trc_addr;
    if (transposed_bsr_row_counts)
        trc_addr = apic_resolve_live_ptr(
            state, reinterpret_cast<uint64_t>(transposed_bsr_row_counts),
            static_cast<uint64_t>(col_count) * sizeof(int32_t)
        );
    APICAddress tc_addr
        = apic_resolve_live_ptr(state, reinterpret_cast<uint64_t>(transposed_bsr_columns), transposed_columns_bytes);
    APICAddress bi_addr
        = apic_resolve_live_ptr(state, reinterpret_cast<uint64_t>(src_block_indices), block_indices_bytes);
    APICAddress status_addr;
    if (status)
        status_addr = apic_resolve_live_ptr(state, reinterpret_cast<uint64_t>(status), sizeof(int32_t));

    apic_record_bsr_transpose(
        state, row_count, col_count, nnz, bo_addr.region_id, bo_addr.offset, brc_addr.region_id, brc_addr.offset,
        bc_addr.region_id, bc_addr.offset, to_addr.region_id, to_addr.offset, trc_addr.region_id, trc_addr.offset,
        tc_addr.region_id, tc_addr.offset, bi_addr.region_id, bi_addr.offset, status_addr.region_id, status_addr.offset,
        nullptr, 0
    );
}

WP_API void wp_bsr_transpose_device(
    int row_count,
    int col_count,
    int nnz,
    const int* bsr_offsets,
    const int* bsr_row_counts,
    const int* bsr_columns,
    int* transposed_bsr_offsets,
    int* transposed_bsr_row_counts,
    int* transposed_bsr_columns,
    int* src_block_indices,
    int* status
)
{
    void* context = wp_cuda_context_get_current();
    ContextGuard guard(context);

    apic_capture_bsr_transpose_device(
        row_count, col_count, nnz, bsr_offsets, bsr_row_counts, bsr_columns, transposed_bsr_offsets,
        transposed_bsr_row_counts, transposed_bsr_columns, src_block_indices, status
    );

    cudaStream_t stream = static_cast<cudaStream_t>(wp_cuda_stream_get_current());
    const bool padded = transposed_bsr_row_counts != nullptr;

    ScopedTemporary<BsrRowCol> combined_row_col(context, 2 * nnz);
    ScopedTemporary<int> active_count(context, 1);
    check_cuda(cudaMemsetAsync(active_count.buffer(), 0, sizeof(int), stream));

    cub::DoubleBuffer<int> d_keys(src_block_indices + nnz, src_block_indices);
    cub::DoubleBuffer<BsrRowCol> d_values(combined_row_col.buffer(), combined_row_col.buffer() + nnz);

    wp_launch_device(
        WP_CURRENT_CONTEXT, bsr_count_active_blocks, row_count,
        (row_count, bsr_offsets, bsr_row_counts, active_count.buffer())
    );

    wp_launch_device(
        WP_CURRENT_CONTEXT, bsr_transpose_fill_row_col, nnz,
        (nnz, row_count, bsr_offsets, bsr_row_counts, bsr_columns, d_keys.Current(), d_values.Current())
    );

    // Sort blocks
    {
        size_t buff_size = 0;
        check_cuda(cub::DeviceRadixSort::SortPairs(nullptr, buff_size, d_values, d_keys, nnz, 0, 64, stream));
        ScopedTemporary<> temp(context, buff_size);
        check_cuda(cub::DeviceRadixSort::SortPairs(temp.buffer(), buff_size, d_values, d_keys, nnz, 0, 64, stream));

        // Depending on data size and GPU architecture buffers may have been swapped
        // or not. For compact output, ensure the sorted keys are available in
        // src_block_indices.
        if (!padded && d_keys.Current() != src_block_indices) {
            check_cuda(cudaMemcpyAsync(
                src_block_indices, src_block_indices + nnz, size_t(nnz) * sizeof(int), cudaMemcpyDeviceToDevice, stream
            ));
        }
    }

    ScopedTemporary<int> sorted_src_block_indices(context, padded ? size_t(nnz) : size_t(1));
    const int* sorted_src_blocks = d_keys.Current();
    if (padded) {
        sorted_src_blocks = sorted_src_block_indices.buffer();
        check_cuda(cudaMemcpyAsync(
            sorted_src_block_indices.buffer(), d_keys.Current(), size_t(nnz) * sizeof(int), cudaMemcpyDeviceToDevice,
            stream
        ));
    } else {
        sorted_src_blocks = src_block_indices;
    }

    ScopedTemporary<int> padded_offsets(context, padded ? size_t(col_count + 1) : size_t(1));
    int* compact_offsets = padded ? padded_offsets.buffer() : transposed_bsr_offsets;

    // Compute row offsets from sorted unique blocks
    wp_launch_device(
        WP_CURRENT_CONTEXT, bsr_find_row_offsets, col_count + 1,
        (col_count, active_count.buffer(), d_values.Current(), compact_offsets)
    );

    if (padded) {
        wp_launch_device(
            WP_CURRENT_CONTEXT, bsr_transpose_scatter_padded, col_count,
            (col_count, compact_offsets, d_values.Current(), sorted_src_blocks, transposed_bsr_offsets,
             transposed_bsr_row_counts, transposed_bsr_columns, src_block_indices, status)
        );
        return;
    }

    wp_launch_device(
        WP_CURRENT_CONTEXT, bsr_set_column, nnz, (active_count.buffer(), d_values.Current(), transposed_bsr_columns)
    );
}

template <typename T, bool CompressValues>
void wp_bsr_compress_inplace_device_impl(
    int row_count,
    int block_size,
    int nnz_upper_bound,
    bool prune_numerical_zeros,
    bool make_compact,
    int* bsr_offsets,
    int* bsr_row_counts,
    int* bsr_columns,
    void* bsr_values,
    uint64_t scalar_zero_mask,
    int* bsr_nnz,
    void* bsr_nnz_event
)
{
    T* values_to_compress = nullptr;
    const T* values_for_pruning = nullptr;
    if constexpr (CompressValues) {
        values_to_compress = static_cast<T*>(bsr_values);
    }
    if (prune_numerical_zeros) {
        values_for_pruning = static_cast<const T*>(bsr_values);
    }
    if (row_count <= 0) {
        return;
    }

    void* context = wp_cuda_context_get_current();
    ContextGuard guard(context);
    cudaStream_t stream = static_cast<cudaStream_t>(wp_cuda_stream_get_current());
    std::optional<ScopedTemporary<int>> compact_row_counts;
    int* out_row_counts = bsr_row_counts;
    if (make_compact) {
        compact_row_counts.emplace(context, size_t(row_count) + 1);
        out_row_counts = compact_row_counts->buffer();
    }

    {
        const int avg_row_count = (int64_t(nnz_upper_bound) + int64_t(row_count) - 1) / int64_t(row_count);
        const bool serial_small_blocks
            = int64_t(block_size) * avg_row_count <= int64_t(BSR_COMPRESS_SHORT_ROW_CAPACITY);
        if (serial_small_blocks) {
            wp_launch_device(
                WP_CURRENT_CONTEXT, (bsr_compress_inplace_rows_serial_kernel<T, CompressValues>), row_count,
                (row_count, block_size, bsr_offsets, bsr_row_counts, out_row_counts, bsr_columns, values_to_compress,
                 values_for_pruning, scalar_zero_mask)
            );
        } else {
            const int threads = wp::clamp(
                bsr_next_power_of_two(avg_row_count), BSR_COMPRESS_MIN_INPLACE_THREADS, BSR_COMPRESS_MAX_INPLACE_THREADS
            );
            // Keep dynamic shared memory within the conservative 48 KiB limit.
            const int slots_per_thread = (CompressValues && sizeof(T) > 4) || threads >= 1024
                ? BSR_COMPRESS_FLOAT64_BLOCKS_PER_THREAD
                : BSR_COMPRESS_DEFAULT_BLOCKS_PER_THREAD;
            const int shared_sort_capacity = threads * slots_per_thread;
            const size_t shared_topology_bytes
                = size_t(shared_sort_capacity) * BSR_COMPRESS_TOPOLOGY_INTS_PER_SLOT * sizeof(int);

            size_t shared_value_bytes = 0;
            if constexpr (CompressValues) {
                shared_value_bytes = size_t(shared_sort_capacity) * sizeof(T);
            }

            const size_t shared_bytes = shared_topology_bytes + shared_value_bytes;
#ifndef NDEBUG
            static constexpr size_t max_dynamic_shared_bytes = 48 * 1024;
            assert(shared_bytes <= max_dynamic_shared_bytes);
#endif

            begin_cuda_range(WP_TIMING_KERNEL_BUILTIN, stream, context, "bsr_compress_inplace_rows_indexed_kernel");
            bsr_compress_inplace_rows_indexed_kernel<T, CompressValues><<<row_count, threads, shared_bytes, stream>>>(
                row_count, block_size, shared_sort_capacity, shared_bytes, bsr_offsets, bsr_row_counts, out_row_counts,
                bsr_columns, values_to_compress, values_for_pruning, scalar_zero_mask
            );
#if _DEBUG
            check_cuda(wp_cuda_context_check(WP_CURRENT_CONTEXT));
#endif
            end_cuda_range(WP_TIMING_KERNEL_BUILTIN, stream);
        }
    }

    if (!make_compact) {
        return;
    }

    int* row_counts = compact_row_counts->buffer();

    constexpr int threads = 128;
    begin_cuda_range(WP_TIMING_KERNEL_BUILTIN, stream, context, "bsr_compress_inplace_compact_counts_kernel");
    bsr_compress_inplace_compact_counts_kernel<<<row_count, threads, 0, stream>>>(
        row_count, bsr_offsets, row_counts, bsr_columns
    );
#if _DEBUG
    check_cuda(wp_cuda_context_check(WP_CURRENT_CONTEXT));
#endif
    end_cuda_range(WP_TIMING_KERNEL_BUILTIN, stream);

    {
        size_t buff_size = 0;
        check_cuda(cub::DeviceScan::ExclusiveSum(nullptr, buff_size, row_counts, bsr_offsets, row_count + 1, stream));
        ScopedTemporary<> temp(context, buff_size);
        check_cuda(
            cub::DeviceScan::ExclusiveSum(temp.buffer(), buff_size, row_counts, bsr_offsets, row_count + 1, stream)
        );
    }

    if (nnz_upper_bound > 0) {
        size_t column_select_bytes = 0;
        check_cuda(
            cub::DeviceSelect::If(
                nullptr, column_select_bytes, bsr_columns, bsr_offsets + row_count, nnz_upper_bound,
                BsrColumnIsActive(), stream
            )
        );

        size_t value_select_bytes = 0;
        if constexpr (CompressValues) {
            bsr_select_compact_value_chunks<T>(
                nullptr, value_select_bytes, values_to_compress, block_size, bsr_columns, bsr_offsets + row_count,
                nnz_upper_bound, stream
            );
        }

        const size_t select_bytes = column_select_bytes > value_select_bytes ? column_select_bytes : value_select_bytes;
        ScopedTemporary<> select_temp(context, select_bytes);

        if constexpr (CompressValues) {
            begin_cuda_range(
                WP_TIMING_KERNEL_BUILTIN, stream, context, "cub::DeviceSelect::FlaggedIf(bsr_value_blocks)"
            );
            size_t temp_bytes = select_bytes;
            bsr_select_compact_value_chunks<T>(
                select_temp.buffer(), temp_bytes, values_to_compress, block_size, bsr_columns, bsr_offsets + row_count,
                nnz_upper_bound, stream
            );
            end_cuda_range(WP_TIMING_KERNEL_BUILTIN, stream);
        }

        {
            begin_cuda_range(WP_TIMING_KERNEL_BUILTIN, stream, context, "cub::DeviceSelect::If(bsr_columns)");
            size_t temp_bytes = select_bytes;
            check_cuda(
                cub::DeviceSelect::If(
                    select_temp.buffer(), temp_bytes, bsr_columns, bsr_offsets + row_count, nnz_upper_bound,
                    BsrColumnIsActive(), stream
                )
            );
            end_cuda_range(WP_TIMING_KERNEL_BUILTIN, stream);
        }
    }

    if (bsr_nnz) {
        wp_memcpy_d2h(WP_CURRENT_CONTEXT, bsr_nnz, bsr_offsets + row_count, sizeof(int), stream);

        if (bsr_nnz_event) {
            const bool external = true;
            wp_cuda_event_record(bsr_nnz_event, stream, external);
        }
    }
}

WP_API void wp_bsr_compress_inplace_device(
    int row_count,
    int block_size,
    int scalar_size_in_bytes,
    int scalar_type,
    int nnz_upper_bound,
    bool prune_numerical_zeros,
    uint64_t scalar_zero_mask,
    bool make_compact,
    int* bsr_offsets,
    int* bsr_row_counts,
    int* bsr_columns,
    void* bsr_values,
    bool compress_values,
    int* bsr_nnz,
    void* bsr_nnz_event
)
{
    const bool prune_from_input_values = !compress_values && prune_numerical_zeros && bsr_values != nullptr;
    void* values_to_write = compress_values ? bsr_values : nullptr;

    if (values_to_write == nullptr && !prune_from_input_values) {
        wp_bsr_compress_inplace_device_impl<wp::float32, false>(
            row_count, 0, nnz_upper_bound, false, make_compact, bsr_offsets, bsr_row_counts, bsr_columns, nullptr, 0,
            bsr_nnz, bsr_nnz_event
        );
        return;
    }

    if (!compress_values) {
        switch (scalar_size_in_bytes) {
        case sizeof(uint8_t):
            wp_bsr_compress_inplace_device_impl<uint8_t, false>(
                row_count, block_size, nnz_upper_bound, true, make_compact, bsr_offsets, bsr_row_counts, bsr_columns,
                bsr_values, scalar_zero_mask, bsr_nnz, bsr_nnz_event
            );
            break;
        case sizeof(uint16_t):
            wp_bsr_compress_inplace_device_impl<uint16_t, false>(
                row_count, block_size, nnz_upper_bound, true, make_compact, bsr_offsets, bsr_row_counts, bsr_columns,
                bsr_values, scalar_zero_mask, bsr_nnz, bsr_nnz_event
            );
            break;
        case sizeof(uint32_t):
            wp_bsr_compress_inplace_device_impl<uint32_t, false>(
                row_count, block_size, nnz_upper_bound, true, make_compact, bsr_offsets, bsr_row_counts, bsr_columns,
                bsr_values, scalar_zero_mask, bsr_nnz, bsr_nnz_event
            );
            break;
        case sizeof(uint64_t):
            wp_bsr_compress_inplace_device_impl<uint64_t, false>(
                row_count, block_size, nnz_upper_bound, true, make_compact, bsr_offsets, bsr_row_counts, bsr_columns,
                bsr_values, scalar_zero_mask, bsr_nnz, bsr_nnz_event
            );
            break;
        }
        return;
    }

    switch (scalar_type) {
    case BSR_SCALAR_FLOAT32:
        if (scalar_size_in_bytes == sizeof(wp::float32))
            wp_bsr_compress_inplace_device_impl<wp::float32, true>(
                row_count, block_size, nnz_upper_bound, prune_numerical_zeros, make_compact, bsr_offsets,
                bsr_row_counts, bsr_columns, values_to_write, 0, bsr_nnz, bsr_nnz_event
            );
        break;
    case BSR_SCALAR_FLOAT64:
        if (scalar_size_in_bytes == sizeof(wp::float64))
            wp_bsr_compress_inplace_device_impl<wp::float64, true>(
                row_count, block_size, nnz_upper_bound, prune_numerical_zeros, make_compact, bsr_offsets,
                bsr_row_counts, bsr_columns, values_to_write, 0, bsr_nnz, bsr_nnz_event
            );
        break;
    }
}
