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

    CUDA_CALLABLE_DEVICE bool operator()(int i) const
    {
        if (!values)
            return true;

        const T* val = values + i * block_size;
        for (int i = 0; i < block_size; ++i, ++val)
        {
            if (*val != T(0))
                return true;
        }
        return false;
    }
};

struct BsrBlockInMask
{
    const int* bsr_offsets;
    const int* bsr_columns;

    CUDA_CALLABLE_DEVICE bool operator()(int row, int col) const
    {
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
__global__ void bsr_fill_triplet_key_values(const int nnz, const int nrow, const int* tpl_rows, const int* tpl_columns,
                                            const BsrBlockIsNotZero<T> nonZero, const BsrBlockInMask mask,
                                            uint32_t* block_indices, BsrRowCol* tpl_row_col)
{
    int block = blockIdx.x * blockDim.x + threadIdx.x;
    if (block >= nnz)
        return;

    const int row = tpl_rows[block];
    const int col = tpl_columns[block];
    const bool is_valid = row >= 0 && row < nrow;

    const BsrRowCol row_col =
        is_valid && nonZero(block) && mask(row, col) ? bsr_combine_row_col(row, col) : PRUNED_ROWCOL;
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

template <typename T>
__global__ void bsr_merge_blocks(const int* d_nnz, int block_size, const uint32_t* block_offsets,
                                 const uint32_t* sorted_block_indices, const BsrRowCol* unique_row_cols,
                                 const T* tpl_values, int* bsr_cols, T* bsr_values)

{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= *d_nnz)
        return;

    const BsrRowCol row_col = unique_row_cols[i];
    bsr_cols[i] = bsr_get_col(row_col);

    // Accumulate merged block values
    if (row_col == PRUNED_ROWCOL || bsr_values == nullptr)
        return;

    const uint32_t beg = i ? block_offsets[i - 1] : 0;
    const uint32_t end = block_offsets[i];

    T* bsr_val = bsr_values + i * block_size;
    const T* tpl_val = tpl_values + sorted_block_indices[beg] * block_size;

    for (int k = 0; k < block_size; ++k)
    {
        bsr_val[k] = tpl_val[k];
    }

    for (uint32_t cur = beg + 1; cur != end; ++cur)
    {
        const T* tpl_val = tpl_values + sorted_block_indices[cur] * block_size;
        for (int k = 0; k < block_size; ++k)
        {
            bsr_val[k] += tpl_val[k];
        }
    }
}

template <typename T>
void bsr_matrix_from_triplets_device(const int rows_per_block, const int cols_per_block, const int row_count,
                                     const int nnz, const int* tpl_rows, const int* tpl_columns, const T* tpl_values,
                                     const bool prune_numerical_zeros, const bool masked, int* bsr_offsets,
                                     int* bsr_columns, T* bsr_values, int* bsr_nnz, void* bsr_nnz_event)
{
    const int block_size = rows_per_block * cols_per_block;

    void* context = cuda_context_get_current();
    ContextGuard guard(context);

    // Per-context cached temporary buffers
    // BsrFromTripletsTemp& bsr_temp = g_bsr_from_triplets_temp_map[context];

    cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream_get_current());

    ScopedTemporary<uint32_t> block_indices(context, 2 * nnz + 1);
    ScopedTemporary<BsrRowCol> combined_row_col(context, 2 * nnz);

    cub::DoubleBuffer<uint32_t> d_keys(block_indices.buffer(), block_indices.buffer() + nnz);
    cub::DoubleBuffer<BsrRowCol> d_values(combined_row_col.buffer(), combined_row_col.buffer() + nnz);

    uint32_t* unique_triplet_count = block_indices.buffer() + 2 * nnz;

    // Combine rows and columns so we can sort on them both
    BsrBlockIsNotZero<T> isNotZero{block_size, prune_numerical_zeros ? tpl_values : nullptr};
    BsrBlockInMask mask{masked ? bsr_offsets : nullptr, bsr_columns};
    wp_launch_device(WP_CURRENT_CONTEXT, bsr_fill_triplet_key_values, nnz,
                     (nnz, row_count, tpl_rows, tpl_columns, isNotZero, mask, d_keys.Current(), d_values.Current()));

    // Sort
    {
        size_t buff_size = 0;
        check_cuda(cub::DeviceRadixSort::SortPairs(nullptr, buff_size, d_values, d_keys, nnz, 0, 64, stream));
        ScopedTemporary<> temp(context, buff_size);
        check_cuda(cub::DeviceRadixSort::SortPairs(temp.buffer(), buff_size, d_values, d_keys, nnz, 0, 64, stream));
    }

    // Runlength encode row-col sequences
    {
        size_t buff_size = 0;
        check_cuda(cub::DeviceRunLengthEncode::Encode(nullptr, buff_size, d_values.Current(), d_values.Alternate(),
                                                      d_keys.Alternate(), unique_triplet_count, nnz, stream));
        ScopedTemporary<> temp(context, buff_size);
        check_cuda(cub::DeviceRunLengthEncode::Encode(temp.buffer(), buff_size, d_values.Current(),
                                                      d_values.Alternate(), d_keys.Alternate(), unique_triplet_count,
                                                      nnz, stream));
    }

    // Compute row offsets from sorted unique blocks
    wp_launch_device(WP_CURRENT_CONTEXT, bsr_find_row_offsets, row_count + 1,
                     (row_count, unique_triplet_count, d_values.Alternate(), bsr_offsets));

    if (bsr_nnz)
    {
        // Copy nnz to host, and record an event for the completed transfer if desired

        memcpy_d2h(WP_CURRENT_CONTEXT, bsr_nnz, bsr_offsets + row_count, sizeof(int), stream);

        if (bsr_nnz_event)
        {
            cuda_event_record(bsr_nnz_event, stream);
        }
    }

    // Scan repeated block counts
    {
        size_t buff_size = 0;
        check_cuda(
            cub::DeviceScan::InclusiveSum(nullptr, buff_size, d_keys.Alternate(), d_keys.Alternate(), nnz, stream));
        ScopedTemporary<> temp(context, buff_size);
        check_cuda(cub::DeviceScan::InclusiveSum(temp.buffer(), buff_size, d_keys.Alternate(), d_keys.Alternate(), nnz,
                                                 stream));
    }

    // Accumulate repeated blocks and set column indices
    wp_launch_device(WP_CURRENT_CONTEXT, bsr_merge_blocks, nnz,
                     (bsr_offsets + row_count, block_size, d_keys.Alternate(), d_keys.Current(), d_values.Alternate(),
                      tpl_values, bsr_columns, bsr_values));
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

    if (i >= bsr_offsets[row_count])
    {
        // Below upper bound but above actual nnz count, mark as invalid
        transposed_row_col[i] = PRUNED_ROWCOL;
        return;
    }

    block_indices[i] = i;

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

template <int Rows, int Cols, typename T> struct BsrBlockTransposer
{
    void CUDA_CALLABLE_DEVICE operator()(const T* src, T* dest) const
    {
        for (int r = 0; r < Rows; ++r)
        {
            for (int c = 0; c < Cols; ++c)
            {
                dest[c * Rows + r] = src[r * Cols + c];
            }
        }
    }
};

template <typename T> struct BsrBlockTransposer<-1, -1, T>
{

    int row_count;
    int col_count;

    void CUDA_CALLABLE_DEVICE operator()(const T* src, T* dest) const
    {
        for (int r = 0; r < row_count; ++r)
        {
            for (int c = 0; c < col_count; ++c)
            {
                dest[c * row_count + r] = src[r * col_count + c];
            }
        }
    }
};

template <int Rows, int Cols, typename T>
__global__ void bsr_transpose_blocks(const int* nnz, const int block_size, BsrBlockTransposer<Rows, Cols, T> transposer,
                                     const int* block_indices, const BsrRowCol* transposed_indices, const T* bsr_values,
                                     int* transposed_bsr_columns, T* transposed_bsr_values)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= *nnz)
        return;

    const int src_idx = block_indices[i];

    transposer(bsr_values + src_idx * block_size, transposed_bsr_values + i * block_size);

    transposed_bsr_columns[i] = bsr_get_col(transposed_indices[i]);
}

template <typename T>
void launch_bsr_transpose_blocks(int nnz, const int* d_nnz, const int block_size, const int rows_per_block,
                                 const int cols_per_block, const int* block_indices,
                                 const BsrRowCol* transposed_indices, const T* bsr_values, int* transposed_bsr_columns,
                                 T* transposed_bsr_values)
{

    switch (rows_per_block)
    {
    case 1:
        switch (cols_per_block)
        {
        case 1:
            wp_launch_device(WP_CURRENT_CONTEXT, bsr_transpose_blocks, nnz,
                             (d_nnz, block_size, BsrBlockTransposer<1, 1, T>{}, block_indices, transposed_indices,
                              bsr_values, transposed_bsr_columns, transposed_bsr_values));
            return;
        case 2:
            wp_launch_device(WP_CURRENT_CONTEXT, bsr_transpose_blocks, nnz,
                             (d_nnz, block_size, BsrBlockTransposer<1, 2, T>{}, block_indices, transposed_indices,
                              bsr_values, transposed_bsr_columns, transposed_bsr_values));
            return;
        case 3:
            wp_launch_device(WP_CURRENT_CONTEXT, bsr_transpose_blocks, nnz,
                             (d_nnz, block_size, BsrBlockTransposer<1, 3, T>{}, block_indices, transposed_indices,
                              bsr_values, transposed_bsr_columns, transposed_bsr_values));
            return;
        }
    case 2:
        switch (cols_per_block)
        {
        case 1:
            wp_launch_device(WP_CURRENT_CONTEXT, bsr_transpose_blocks, nnz,
                             (d_nnz, block_size, BsrBlockTransposer<2, 1, T>{}, block_indices, transposed_indices,
                              bsr_values, transposed_bsr_columns, transposed_bsr_values));
            return;
        case 2:
            wp_launch_device(WP_CURRENT_CONTEXT, bsr_transpose_blocks, nnz,
                             (d_nnz, block_size, BsrBlockTransposer<2, 2, T>{}, block_indices, transposed_indices,
                              bsr_values, transposed_bsr_columns, transposed_bsr_values));
            return;
        case 3:
            wp_launch_device(WP_CURRENT_CONTEXT, bsr_transpose_blocks, nnz,
                             (d_nnz, block_size, BsrBlockTransposer<2, 3, T>{}, block_indices, transposed_indices,
                              bsr_values, transposed_bsr_columns, transposed_bsr_values));
            return;
        }
    case 3:
        switch (cols_per_block)
        {
        case 1:
            wp_launch_device(WP_CURRENT_CONTEXT, bsr_transpose_blocks, nnz,
                             (d_nnz, block_size, BsrBlockTransposer<3, 1, T>{}, block_indices, transposed_indices,
                              bsr_values, transposed_bsr_columns, transposed_bsr_values));
            return;
        case 2:
            wp_launch_device(WP_CURRENT_CONTEXT, bsr_transpose_blocks, nnz,
                             (d_nnz, block_size, BsrBlockTransposer<3, 2, T>{}, block_indices, transposed_indices,
                              bsr_values, transposed_bsr_columns, transposed_bsr_values));
            return;
        case 3:
            wp_launch_device(WP_CURRENT_CONTEXT, bsr_transpose_blocks, nnz,
                             (d_nnz, block_size, BsrBlockTransposer<3, 3, T>{}, block_indices, transposed_indices,
                              bsr_values, transposed_bsr_columns, transposed_bsr_values));
            return;
        }
    }

    wp_launch_device(WP_CURRENT_CONTEXT, bsr_transpose_blocks, nnz,
                     (d_nnz, block_size, BsrBlockTransposer<-1, -1, T>{rows_per_block, cols_per_block}, block_indices,
                      transposed_indices, bsr_values, transposed_bsr_columns, transposed_bsr_values));
}

template <typename T>
void bsr_transpose_device(int rows_per_block, int cols_per_block, int row_count, int col_count, int nnz,
                          const int* bsr_offsets, const int* bsr_columns, const T* bsr_values,
                          int* transposed_bsr_offsets, int* transposed_bsr_columns, T* transposed_bsr_values)
{

    const int block_size = rows_per_block * cols_per_block;

    void* context = cuda_context_get_current();
    ContextGuard guard(context);

    cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream_get_current());

    ScopedTemporary<int> block_indices(context, 2 * nnz);
    ScopedTemporary<BsrRowCol> combined_row_col(context, 2 * nnz);

    cub::DoubleBuffer<int> d_keys(block_indices.buffer(), block_indices.buffer() + nnz);
    cub::DoubleBuffer<BsrRowCol> d_values(combined_row_col.buffer(), combined_row_col.buffer() + nnz);

    wp_launch_device(WP_CURRENT_CONTEXT, bsr_transpose_fill_row_col, nnz,
                     (nnz, row_count, bsr_offsets, bsr_columns, d_keys.Current(), d_values.Current()));

    // Sort blocks
    {
        size_t buff_size = 0;
        check_cuda(cub::DeviceRadixSort::SortPairs(nullptr, buff_size, d_values, d_keys, nnz, 0, 64, stream));
        ScopedTemporary<> temp(context, buff_size);
        check_cuda(cub::DeviceRadixSort::SortPairs(temp.buffer(), buff_size, d_values, d_keys, nnz, 0, 64, stream));
    }

    // Compute row offsets from sorted unique blocks
    wp_launch_device(WP_CURRENT_CONTEXT, bsr_find_row_offsets, col_count + 1,
                     (col_count, bsr_offsets + row_count, d_values.Current(), transposed_bsr_offsets));

    // Move and transpose individual blocks
    if (transposed_bsr_values != nullptr)
    {
        launch_bsr_transpose_blocks(nnz, bsr_offsets + row_count, block_size, rows_per_block, cols_per_block,
                                    d_keys.Current(), d_values.Current(), bsr_values, transposed_bsr_columns,
                                    transposed_bsr_values);
    }
}

} // namespace

void bsr_matrix_from_triplets_float_device(int rows_per_block, int cols_per_block, int row_count, int nnz,
                                           int* tpl_rows, int* tpl_columns, void* tpl_values,
                                           bool prune_numerical_zeros, bool masked, int* bsr_offsets, int* bsr_columns,
                                           void* bsr_values, int* bsr_nnz, void* bsr_nnz_event)
{
    return bsr_matrix_from_triplets_device<float>(rows_per_block, cols_per_block, row_count, nnz, tpl_rows, tpl_columns,
                                                  static_cast<const float*>(tpl_values), prune_numerical_zeros, masked,
                                                  bsr_offsets, bsr_columns, static_cast<float*>(bsr_values), bsr_nnz,
                                                  bsr_nnz_event);
}

void bsr_matrix_from_triplets_double_device(int rows_per_block, int cols_per_block, int row_count, int nnz,
                                            int* tpl_rows, int* tpl_columns, void* tpl_values,
                                            bool prune_numerical_zeros, bool masked, int* bsr_offsets, int* bsr_columns,
                                            void* bsr_values, int* bsr_nnz, void* bsr_nnz_event)
{
    return bsr_matrix_from_triplets_device<double>(rows_per_block, cols_per_block, row_count, nnz, tpl_rows,
                                                   tpl_columns, static_cast<const double*>(tpl_values),
                                                   prune_numerical_zeros, masked, bsr_offsets, bsr_columns,
                                                   static_cast<double*>(bsr_values), bsr_nnz, bsr_nnz_event);
}

void bsr_transpose_float_device(int rows_per_block, int cols_per_block, int row_count, int col_count, int nnz,
                                int* bsr_offsets, int* bsr_columns, void* bsr_values, int* transposed_bsr_offsets,
                                int* transposed_bsr_columns, void* transposed_bsr_values)
{
    bsr_transpose_device(rows_per_block, cols_per_block, row_count, col_count, nnz, bsr_offsets, bsr_columns,
                         static_cast<const float*>(bsr_values), transposed_bsr_offsets, transposed_bsr_columns,
                         static_cast<float*>(transposed_bsr_values));
}

void bsr_transpose_double_device(int rows_per_block, int cols_per_block, int row_count, int col_count, int nnz,
                                 int* bsr_offsets, int* bsr_columns, void* bsr_values, int* transposed_bsr_offsets,
                                 int* transposed_bsr_columns, void* transposed_bsr_values)
{
    bsr_transpose_device(rows_per_block, cols_per_block, row_count, col_count, nnz, bsr_offsets, bsr_columns,
                         static_cast<const double*>(bsr_values), transposed_bsr_offsets, transposed_bsr_columns,
                         static_cast<double*>(transposed_bsr_values));
}
