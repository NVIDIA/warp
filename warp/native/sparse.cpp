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

#include "warp.h"

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <vector>

namespace
{

template <typename T> bool bsr_block_is_zero(int block_idx, int block_size, const void* values, const uint64_t scalar_zero_mask)
{
    const T* block_values = static_cast<const T*>(values) + block_idx * block_size;
    const T zero_mask = static_cast<T>(scalar_zero_mask);

    return std::all_of(block_values, block_values + block_size, [zero_mask](T v) { return (v & zero_mask) == T(0); });
}

} // namespace


WP_API void wp_bsr_matrix_from_triplets_host(
    int block_size,
    int scalar_size_in_bytes,
    int row_count,
    int col_count,
    int nnz,
    const int* tpl_nnz,
    const int* tpl_rows,
    const int* tpl_columns,
    const void* tpl_values,
    const uint64_t scalar_zero_mask,
    bool masked_topology,
    int* tpl_block_offsets,
    int* tpl_block_indices,
    int* bsr_offsets,
    int* bsr_columns,
    int* bsr_nnz,
    void* bsr_nnz_event)
{
    if (tpl_nnz != nullptr)
    {
        nnz = *tpl_nnz;
    }

    // allocate temporary buffers if not provided
    bool return_summed_blocks = tpl_block_offsets != nullptr && tpl_block_indices != nullptr;
    if (!return_summed_blocks)
    {
        tpl_block_offsets = static_cast<int*>(wp_alloc_host(size_t(nnz) * sizeof(int)));
        tpl_block_indices = static_cast<int*>(wp_alloc_host(size_t(nnz) * sizeof(int)));
    }

    std::iota(tpl_block_indices, tpl_block_indices + nnz, 0);

    // remove invalid indices / indices not in mask
    auto discard_invalid_block = [&](int i) -> bool
    {
        const int row = tpl_rows[i];
        const int col = tpl_columns[i];
        if (row < 0 || row >= row_count || col < 0 || col >= col_count)
        {
            return true;
        }

        if (!masked_topology)
        {
            return false;
        }

        const int* beg = bsr_columns + bsr_offsets[row];
        const int* end = bsr_columns + bsr_offsets[row + 1];
        const int* block = std::lower_bound(beg, end, col);
        return block == end || *block != col;
    };

    int* valid_indices_end = std::remove_if(tpl_block_indices, tpl_block_indices + nnz, discard_invalid_block);

    // remove zero blocks
    if (tpl_values != nullptr && scalar_zero_mask != 0)
    {
        switch (scalar_size_in_bytes)
        {
            case sizeof(uint8_t):
                valid_indices_end = std::remove_if(tpl_block_indices, valid_indices_end, [block_size, tpl_values, scalar_zero_mask](uint32_t i) { return bsr_block_is_zero<uint8_t>(i, block_size, tpl_values, scalar_zero_mask); });
                break;
            case sizeof(uint16_t):
                valid_indices_end = std::remove_if(tpl_block_indices, valid_indices_end, [block_size, tpl_values, scalar_zero_mask](uint32_t i) { return bsr_block_is_zero<uint16_t>(i, block_size, tpl_values, scalar_zero_mask); });
                break;
            case sizeof(uint32_t):
                valid_indices_end = std::remove_if(tpl_block_indices, valid_indices_end, [block_size, tpl_values, scalar_zero_mask](uint32_t i) { return bsr_block_is_zero<uint32_t>(i, block_size, tpl_values, scalar_zero_mask); });
                break;
            case sizeof(uint64_t):
                valid_indices_end = std::remove_if(tpl_block_indices, valid_indices_end, [block_size, tpl_values, scalar_zero_mask](uint32_t i) { return bsr_block_is_zero<uint64_t>(i, block_size, tpl_values, scalar_zero_mask); });
                break;
        }
    }
    
    // sort block indices according to lexico order
    std::sort(tpl_block_indices, valid_indices_end, [tpl_rows, tpl_columns](int i, int j) -> bool
              { return tpl_rows[i] < tpl_rows[j] || (tpl_rows[i] == tpl_rows[j] && tpl_columns[i] < tpl_columns[j]); });

    // accumulate blocks at same locations, count blocks per row
    std::fill_n(bsr_offsets, row_count + 1, 0);

    int current_row = -1;
    int current_col = -1;
    int current_block_idx = -1;

    for (int *block = tpl_block_indices, *block_offset = tpl_block_offsets ; block != valid_indices_end ; ++ block)
    {
        int32_t idx = *block;
        int row = tpl_rows[idx];
        int col = tpl_columns[idx];

        if (row != current_row || col != current_col)
        {
            *(bsr_columns++) = col;

            ++bsr_offsets[row + 1];

            if(current_row == -1) {
                *block_offset = 0;
            } else {
                *(block_offset+1) = *block_offset;
                ++block_offset;
            }

            current_row = row;
            current_col = col;
        }

        ++(*block_offset);
    }

    // build postfix sum of row counts
    std::partial_sum(bsr_offsets, bsr_offsets + row_count + 1, bsr_offsets);

    if(!return_summed_blocks)
    {
        // free our temporary buffers
        wp_free_host(tpl_block_offsets);
        wp_free_host(tpl_block_indices);
    }

    if (bsr_nnz != nullptr)
    {
        *bsr_nnz = bsr_offsets[row_count];
    }
}

WP_API void wp_bsr_transpose_host(
    int row_count, int col_count, int nnz,
    const int* bsr_offsets, const int* bsr_columns,
    int* transposed_bsr_offsets,
    int* transposed_bsr_columns,
    int* block_indices
    )
{
    nnz = bsr_offsets[row_count];

    std::vector<int> bsr_rows(nnz);
    std::iota(block_indices, block_indices + nnz, 0);

    // Fill row indices from offsets
    for (int row = 0; row < row_count; ++row)
    {
        std::fill(bsr_rows.begin() + bsr_offsets[row], bsr_rows.begin() + bsr_offsets[row + 1], row);
    }

    // sort block indices according to (transposed) lexico order
    std::sort(
        block_indices, block_indices + nnz, [&bsr_rows, bsr_columns](int i, int j) -> bool
        { return bsr_columns[i] < bsr_columns[j] || (bsr_columns[i] == bsr_columns[j] && bsr_rows[i] < bsr_rows[j]); });

    // Count blocks per column and transpose blocks
    std::fill_n(transposed_bsr_offsets, col_count + 1, 0);

    for (int i = 0; i < nnz; ++i)
    {
        int idx = block_indices[i];
        int row = bsr_rows[idx];
        int col = bsr_columns[idx];

        ++transposed_bsr_offsets[col + 1];
        transposed_bsr_columns[i] = row;
    }

    // build postfix sum of column counts
    std::partial_sum(transposed_bsr_offsets, transposed_bsr_offsets + col_count + 1, transposed_bsr_offsets);

}

#if !WP_ENABLE_CUDA
WP_API void wp_bsr_matrix_from_triplets_device(
    int block_size,
    int scalar_size_in_bytes,
    int row_count,
    int col_count,
    int tpl_nnz_upper_bound,
    const int* tpl_nnz,
    const int* tpl_rows,
    const int* tpl_columns,
    const void* tpl_values,
    const uint64_t scalar_zero_mask,
    bool masked_topology,
    int* summed_block_offsets,
    int* summed_block_indices,
    int* bsr_offsets,
    int* bsr_columns,
    int* bsr_nnz,
    void* bsr_nnz_event) {}


WP_API void wp_bsr_transpose_device(
    int row_count, int col_count, int nnz,
    const int* bsr_offsets, const int* bsr_columns,
    int* transposed_bsr_offsets,
    int* transposed_bsr_columns,
    int* src_block_indices) {}



#endif
