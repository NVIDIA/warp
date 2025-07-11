/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "cuda_util.h"
#include "warp.h"
#include "stdint.h"
#include <cstdint>

#define THRUST_IGNORE_CUB_VERSION_CHECK

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>

namespace
{

// Combined row+column value that can be radix-sorted with CUB
using BsrRowCol = uint64_t;

static constexpr BsrRowCol PRUNED_ROWCOL = ~BsrRowCol(0);

CUDA_CALLABLE BsrRowCol bsr_combine_row_col(uint32_t row, uint32_t col)
{
    return (static_cast<uint64_t>(row) << 32) | col;
}

CUDA_CALLABLE uint32_t bsr_get_row(const BsrRowCol& row_col) { return row_col >> 32; }

CUDA_CALLABLE uint32_t bsr_get_col(const BsrRowCol& row_col) { return row_col & INT_MAX; }

template <typename T> struct BsrBlockIsNotZero
{
    int block_size;
    const T* values;
    T zero_mask;

    BsrBlockIsNotZero(int block_size, const void* values, const uint64_t zero_mask)
        : block_size(block_size), values(static_cast<const T*>(values)), zero_mask(static_cast<const T>(zero_mask)) 
        {}

    CUDA_CALLABLE_DEVICE bool operator()(int block) const
    {
        if (!values)
            return true;

        const T* val = values + block * block_size;
        for (int i = 0; i < block_size; ++i, ++val)
        {
            if ((*val & zero_mask) != 0)
                return true;
        }
        return false;
    }
};

template <> struct BsrBlockIsNotZero<void>
{
    BsrBlockIsNotZero(int block_size, const void* values, const uint64_t zero_mask)
    {}

    CUDA_CALLABLE_DEVICE bool operator()(int block) const
    {
        return true;
    }
};

struct BsrBlockInMask
{
    const int nrow;
    const int ncol;
    const int* bsr_offsets;
    const int* bsr_columns;
    const int* device_nnz;

    CUDA_CALLABLE_DEVICE bool operator()(int index, int row, int col) const
    {
        if (device_nnz != nullptr && index >= *device_nnz)
            return false;
    
        if (row < 0 || row >= nrow || col < 0 || col >= ncol){
            return false;
        }

        if (bsr_offsets == nullptr)
            return true;

        int lower = bsr_offsets[row];
        int upper = bsr_offsets[row + 1] - 1;

        while (lower < upper)
        {
            const int mid = lower + (upper - lower) / 2;

            if (bsr_columns[mid] < col)
            {
                lower = mid + 1;
            }
            else
            {
                upper = mid;
            }
        }

        return lower == upper && (bsr_columns[lower] == col);
    }
};

template <typename T>
__global__ void bsr_fill_triplet_key_values(const int nnz, const int* tpl_rows, const int* tpl_columns,
                                            const BsrBlockIsNotZero<T> nonZero, const BsrBlockInMask mask,
                                            int* block_indices, BsrRowCol* tpl_row_col)
{
    int block = blockIdx.x * blockDim.x + threadIdx.x;
    if (block >= nnz)
        return;

    const int row = tpl_rows[block];
    const int col = tpl_columns[block];

    const BsrRowCol row_col =
        mask(block, row, col) && nonZero(block) ? bsr_combine_row_col(row, col) : PRUNED_ROWCOL;

    tpl_row_col[block] = row_col;
    block_indices[block] = block;
}

template <typename T>
__global__ void bsr_find_row_offsets(uint32_t row_count, const T* d_nnz, const BsrRowCol* unique_row_col,
                                     int* row_offsets)
{
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row > row_count)
        return;

    const uint32_t nnz = *d_nnz;
    if (row == 0 || nnz == 0)
    {
        row_offsets[row] = 0;
        return;
    }

    if (bsr_get_row(unique_row_col[nnz - 1]) < row)
    {
        row_offsets[row] = nnz;
        return;
    }

    // binary search for row start
    uint32_t lower = 0;
    uint32_t upper = nnz - 1;
    while (lower < upper)
    {
        uint32_t mid = lower + (upper - lower) / 2;

        if (bsr_get_row(unique_row_col[mid]) < row)
        {
            lower = mid + 1;
        }
        else
        {
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
    BsrBlockIsNotZero<T> isNotZero{block_size, tpl_values, scalar_zero_mask};
    wp_launch_device(WP_CURRENT_CONTEXT, bsr_fill_triplet_key_values, nnz,
                     (nnz, tpl_rows, tpl_columns, isNotZero, mask, block_indices, row_col   ));
}


__global__ void bsr_transpose_fill_row_col(const int nnz_upper_bound, const int row_count, const int* bsr_offsets,
                                           const int* bsr_columns, int* block_indices, BsrRowCol* transposed_row_col)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= nnz_upper_bound)
    {
        // Outside of allocated bounds, do nothing
        return;
    }
    
    block_indices[i] = i;

    if (i >= bsr_offsets[row_count])
    {
        // Below upper bound but above actual nnz count, mark as invalid
        transposed_row_col[i] = PRUNED_ROWCOL;
        return;
    }

    // Binary search for row
    int lower = 0;
    int upper = row_count - 1;

    while (lower < upper)
    {
        int mid = lower + (upper - lower) / 2;

        if (bsr_offsets[mid + 1] <= i)
        {
            lower = mid + 1;
        }
        else
        {
            upper = mid;
        }
    }

    const int row = lower;
    const int col = bsr_columns[i];
    BsrRowCol row_col = bsr_combine_row_col(col, row);
    transposed_row_col[i] = row_col;
}

} // namespace


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
    int* bsr_columns,
    int* bsr_nnz, void* bsr_nnz_event)
{
    void* context = wp_cuda_context_get_current();
    ContextGuard guard(context);

    // Per-context cached temporary buffers
    // BsrFromTripletsTemp& bsr_temp = g_bsr_from_triplets_temp_map[context];

    cudaStream_t stream = static_cast<cudaStream_t>(wp_cuda_stream_get_current());

    ScopedTemporary<BsrRowCol> combined_row_col(context, 2 * size_t(nnz));
    ScopedTemporary<int> unique_triplet_count(context, 1);

    bool return_summed_blocks = tpl_block_offsets != nullptr && tpl_block_indices != nullptr;
    if(!return_summed_blocks)
    {
        // if not provided, allocate temporary offset and indices buffers
        tpl_block_offsets = static_cast<int*>(wp_alloc_device(context, size_t(nnz) * sizeof(int)));
        tpl_block_indices = static_cast<int*>(wp_alloc_device(context,  size_t(nnz) * sizeof(int)));
    }


    cub::DoubleBuffer<int> d_keys(tpl_block_indices, tpl_block_offsets);
    cub::DoubleBuffer<BsrRowCol> d_values(combined_row_col.buffer(), combined_row_col.buffer() + nnz);

    // Combine rows and columns so we can sort on them both,
    // ensuring that blocks that should be pruned are moved to the end 
    BsrBlockInMask mask{row_count, col_count, masked_topology ? bsr_offsets : nullptr, bsr_columns, tpl_nnz};
    if (scalar_zero_mask == 0 || tpl_values == nullptr)
        scalar_size = 0;
    switch(scalar_size)
    {
        case sizeof(uint8_t):
            launch_bsr_fill_triplet_key_values<uint8_t>(block_size, nnz, mask, tpl_rows, tpl_columns, tpl_values, scalar_zero_mask, d_keys.Current(), d_values.Current());
            break;
        case sizeof(uint16_t):
            launch_bsr_fill_triplet_key_values<uint16_t>(block_size, nnz, mask, tpl_rows, tpl_columns, tpl_values, scalar_zero_mask, d_keys.Current(), d_values.Current());
            break;
        case sizeof(uint32_t):
            launch_bsr_fill_triplet_key_values<uint32_t>(block_size, nnz, mask, tpl_rows, tpl_columns, tpl_values, scalar_zero_mask, d_keys.Current(), d_values.Current());
            break;
        case sizeof(uint64_t):
            launch_bsr_fill_triplet_key_values<uint64_t>(block_size, nnz, mask, tpl_rows, tpl_columns, tpl_values, scalar_zero_mask, d_keys.Current(), d_values.Current());
            break;
        default:
            // no scalar-level pruning
            launch_bsr_fill_triplet_key_values<void>(block_size, nnz, mask, tpl_rows, tpl_columns, tpl_values, scalar_zero_mask, d_keys.Current(), d_values.Current());
            break;
    }


    // Sort
    {
        size_t buff_size = 0;
        check_cuda(cub::DeviceRadixSort::SortPairs(nullptr, buff_size, d_values, d_keys, nnz, 0, 64, stream));
        ScopedTemporary<> temp(context, buff_size);
        check_cuda(cub::DeviceRadixSort::SortPairs(temp.buffer(), buff_size, d_values, d_keys, nnz, 0, 64, stream));

        // Depending on data size and GPU architecture buffers may have been swapped or not
        // Ensures the sorted keys are available in summed_block_indices if needed
        if(return_summed_blocks && d_keys.Current() != tpl_block_indices)
        {
            check_cuda(cudaMemcpyAsync(tpl_block_indices, d_keys.Current(), nnz * sizeof(int), cudaMemcpyDeviceToDevice, stream));
        }
    }

    // Runlength encode row-col sequences
    {
        size_t buff_size = 0;
        check_cuda(cub::DeviceRunLengthEncode::Encode(nullptr, buff_size, d_values.Current(), d_values.Alternate(),
                                                      tpl_block_offsets, unique_triplet_count.buffer(), nnz, stream));
        ScopedTemporary<> temp(context, buff_size);
        check_cuda(cub::DeviceRunLengthEncode::Encode(temp.buffer(), buff_size, d_values.Current(),
                                                      d_values.Alternate(), tpl_block_offsets, unique_triplet_count.buffer(),
                                                      nnz, stream));
    }

    // Compute row offsets from sorted unique blocks
    wp_launch_device(WP_CURRENT_CONTEXT, bsr_find_row_offsets, row_count + 1,
                     (row_count, unique_triplet_count.buffer(), d_values.Alternate(), bsr_offsets));

    if (bsr_nnz)
    {
        // Copy nnz to host, and record an event for the completed transfer if desired

        wp_memcpy_d2h(WP_CURRENT_CONTEXT, bsr_nnz, bsr_offsets + row_count, sizeof(int), stream);

        if (bsr_nnz_event)
        {
            wp_cuda_event_record(bsr_nnz_event, stream);
        }
    }

    // Set column indices
    wp_launch_device(WP_CURRENT_CONTEXT, bsr_set_column, nnz,
                     (bsr_offsets + row_count, d_values.Alternate(),
                      bsr_columns));

    // Scan repeated block counts
    if(return_summed_blocks)
    {
        size_t buff_size = 0;
        check_cuda(
            cub::DeviceScan::InclusiveSum(nullptr, buff_size, tpl_block_offsets, tpl_block_offsets, nnz, stream));
        ScopedTemporary<> temp(context, buff_size);
        check_cuda(cub::DeviceScan::InclusiveSum(temp.buffer(), buff_size, tpl_block_offsets, tpl_block_offsets, nnz,
                                                 stream));
    } else {
        // free our temporary buffers
        wp_free_device(context, tpl_block_offsets);
        wp_free_device(context, tpl_block_indices);
     }
}


WP_API void wp_bsr_transpose_device(int row_count, int col_count, int nnz,
                          const int* bsr_offsets, const int* bsr_columns, 
                          int* transposed_bsr_offsets, int* transposed_bsr_columns,
                          int* src_block_indices)
{
    void* context = wp_cuda_context_get_current();
    ContextGuard guard(context);

    cudaStream_t stream = static_cast<cudaStream_t>(wp_cuda_stream_get_current());

    ScopedTemporary<BsrRowCol> combined_row_col(context, 2 * nnz);

    cub::DoubleBuffer<int> d_keys(src_block_indices + nnz, src_block_indices);
    cub::DoubleBuffer<BsrRowCol> d_values(combined_row_col.buffer(), combined_row_col.buffer() + nnz);

    wp_launch_device(WP_CURRENT_CONTEXT, bsr_transpose_fill_row_col, nnz,
                     (nnz, row_count, bsr_offsets, bsr_columns, d_keys.Current(), d_values.Current()));

    // Sort blocks
    {
        size_t buff_size = 0;
        check_cuda(cub::DeviceRadixSort::SortPairs(nullptr, buff_size, d_values, d_keys, nnz, 0, 64, stream));
        ScopedTemporary<> temp(context, buff_size);
        check_cuda(cub::DeviceRadixSort::SortPairs(temp.buffer(), buff_size, d_values, d_keys, nnz, 0, 64, stream));

        // Depending on data size and GPU architecture buffers may have been swapped or not
        // Ensures the sorted keys are available in summed_block_indices if needed
        if(d_keys.Current() != src_block_indices)
        {
            check_cuda(cudaMemcpy(src_block_indices, src_block_indices+nnz, size_t(nnz) * sizeof(int), cudaMemcpyDeviceToDevice));
        }
    }

    // Compute row offsets from sorted unique blocks
    wp_launch_device(WP_CURRENT_CONTEXT, bsr_find_row_offsets, col_count + 1,
                     (col_count, bsr_offsets + row_count, d_values.Current(), transposed_bsr_offsets));


    wp_launch_device(WP_CURRENT_CONTEXT, bsr_set_column, nnz,
                     (bsr_offsets + row_count, d_values.Current(),
                      transposed_bsr_columns));
}
