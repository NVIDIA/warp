// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "builtin.h"

#include <cstdint>
#include <type_traits>

namespace {

static constexpr int BSR_STATUS_ROW_CAPACITY_EXCEEDED = 1;

enum BsrScalarType {
    BSR_SCALAR_FLOAT32 = 0,
    BSR_SCALAR_FLOAT64 = 1,
};

template <typename T>
CUDA_CALLABLE inline bool bsr_typed_block_is_zero(int block, int block_size, const T* values, uint64_t zero_mask = 0)
{
    for (int comp = 0; comp < block_size; ++comp) {
        const T value = values[block * block_size + comp];
        if constexpr (std::is_integral<T>::value) {
            if ((value & static_cast<T>(zero_mask)) != T(0)) {
                return false;
            }
        } else {
            if (value != T(0)) {
                return false;
            }
        }
    }
    return true;
}

template <typename T> CUDA_CALLABLE inline void bsr_copy_block(int dst, int src, int block_size, T* values)
{
    for (int comp = 0; comp < block_size; ++comp) {
        values[dst * block_size + comp] = values[src * block_size + comp];
    }
}

template <typename T> CUDA_CALLABLE inline void bsr_swap_blocks(int a, int b, int block_size, T* values)
{
    for (int comp = 0; comp < block_size; ++comp) {
        T& va = values[a * block_size + comp];
        T& vb = values[b * block_size + comp];
        const T tmp = va;
        va = vb;
        vb = tmp;
    }
}

template <typename T> CUDA_CALLABLE inline void bsr_add_block(int dst, int src, int block_size, T* values)
{
    for (int comp = 0; comp < block_size; ++comp) {
        values[dst * block_size + comp] += values[src * block_size + comp];
    }
}

template <typename T> CUDA_CALLABLE inline void bsr_zero_block(int block, int block_size, T* values)
{
    for (int comp = 0; comp < block_size; ++comp) {
        values[block * block_size + comp] = T(0);
    }
}

CUDA_CALLABLE inline int bsr_active_row_end(const int* offsets, const int* row_counts, int row)
{
    if (row_counts == nullptr) {
        return offsets[row + 1];
    }
    return offsets[row] + row_counts[row];
}

// Row-local in-place compression: tag zero input blocks via prune_values (if provided),
// insertion-sort columns, and coalesce runs of equal columns.
// Shared by the host serial pass and the CUDA serial fallback kernel.
template <typename T, bool CompressValues>
CUDA_CALLABLE inline void bsr_compress_inplace_row(
    int row,
    int block_size,
    const int* bsr_offsets,
    const int* bsr_row_counts,
    int* out_row_counts,
    int* bsr_columns,
    T* bsr_values,
    const T* prune_values,
    uint64_t prune_zero_mask
)
{
    const int row_beg = bsr_offsets[row];
    const int row_end = bsr_active_row_end(bsr_offsets, bsr_row_counts, row);

    if (prune_values != nullptr) {
        for (int block = row_beg; block < row_end; ++block) {
            if (bsr_columns[block] >= 0 && bsr_typed_block_is_zero(block, block_size, prune_values, prune_zero_mask)) {
                bsr_columns[block] = -1;
            }
        }
    }

    for (int block = row_beg + 1; block < row_end; ++block) {
        int scan = block;
        while (scan > row_beg && bsr_columns[scan] < bsr_columns[scan - 1]) {
            const int col = bsr_columns[scan];
            bsr_columns[scan] = bsr_columns[scan - 1];
            bsr_columns[scan - 1] = col;
            if constexpr (CompressValues) {
                bsr_swap_blocks(scan, scan - 1, block_size, bsr_values);
            }
            --scan;
        }
    }

    int write = row_beg;
    int read = row_beg;
    while (read < row_end) {
        const int col = bsr_columns[read];
        if (col < 0) {
            ++read;
            continue;
        }

        if (write != read) {
            bsr_columns[write] = col;
            if constexpr (CompressValues) {
                bsr_copy_block(write, read, block_size, bsr_values);
            }
        }

        ++read;
        while (read < row_end && bsr_columns[read] == col) {
            if constexpr (CompressValues) {
                bsr_add_block(write, read, block_size, bsr_values);
            }
            ++read;
        }

        ++write;
    }

    out_row_counts[row] = write - row_beg;
}

}  // namespace
