// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "warp.h"

#include "sparse_util.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <numeric>
#include <type_traits>
#include <vector>

namespace {

template <typename T>
bool bsr_block_is_zero(int block_idx, int block_size, const void* values, const uint64_t scalar_zero_mask)
{
    const T* block_values = static_cast<const T*>(values) + block_idx * block_size;
    const T zero_mask = static_cast<T>(scalar_zero_mask);

    return std::all_of(block_values, block_values + block_size, [zero_mask](T v) { return (v & zero_mask) == T(0); });
}

template <typename T, bool CompressValues>
void bsr_compress_inplace_host_impl(
    int row_count,
    int block_size,
    int nnz_upper_bound,
    bool prune_numerical_zeros,
    bool make_compact,
    int* bsr_offsets,
    int* bsr_row_counts,
    int* bsr_columns,
    void* bsr_values,
    uint64_t prune_zero_mask,
    int* bsr_nnz
)
{
    T* values = nullptr;
    const T* values_to_prune = nullptr;
    if constexpr (CompressValues) {
        values = static_cast<T*>(bsr_values);
    }
    if (prune_numerical_zeros) {
        values_to_prune = static_cast<const T*>(bsr_values);
    }
    std::vector<int> compact_row_counts;
    int* out_row_counts = bsr_row_counts;
    if (make_compact) {
        compact_row_counts.resize(row_count);
        out_row_counts = compact_row_counts.data();
    }

    for (int row = 0; row < row_count; ++row) {
        bsr_compress_inplace_row<T, CompressValues>(
            row, block_size, bsr_offsets, bsr_row_counts, out_row_counts, bsr_columns, values, values_to_prune,
            prune_zero_mask
        );
    }

    if (!make_compact) {
        return;
    }

    std::vector<int> old_offsets(bsr_offsets, bsr_offsets + row_count + 1);
    int write = 0;
    for (int row = 0; row < row_count; ++row) {
        const int row_beg = old_offsets[row];
        const int count = out_row_counts[row];

        if (count > 0 && write != row_beg) {
            std::memmove(bsr_columns + write, bsr_columns + row_beg, size_t(count) * sizeof(int));
            if constexpr (CompressValues) {
                std::memmove(
                    values + write * block_size, values + row_beg * block_size, size_t(count) * block_size * sizeof(T)
                );
            }
        }

        bsr_offsets[row] = write;
        write += count;
    }
    bsr_offsets[row_count] = write;

    for (int block = write; block < nnz_upper_bound; ++block) {
        bsr_columns[block] = -1;
        if constexpr (CompressValues) {
            bsr_zero_block(block, block_size, values);
        }
    }

    if (bsr_nnz != nullptr) {
        *bsr_nnz = write;
    }
}

}  // namespace


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
    const int* bsr_row_counts,
    int* bsr_columns,
    int* bsr_nnz,
    void* bsr_nnz_event
)
{
    if (tpl_nnz != nullptr) {
        nnz = *tpl_nnz;
    }

    // allocate temporary buffers if not provided
    bool return_summed_blocks = tpl_block_offsets != nullptr && tpl_block_indices != nullptr;
    if (!return_summed_blocks) {
        tpl_block_offsets = static_cast<int*>(wp_alloc_host(size_t(nnz) * sizeof(int), "(native:sparse)"));
        tpl_block_indices = static_cast<int*>(wp_alloc_host(size_t(nnz) * sizeof(int), "(native:sparse)"));
    }

    std::iota(tpl_block_indices, tpl_block_indices + nnz, 0);

    // remove invalid indices / indices not in mask
    auto discard_invalid_block = [&](int i) -> bool {
        const int row = tpl_rows[i];
        const int col = tpl_columns[i];
        if (row < 0 || row >= row_count || col < 0 || col >= col_count) {
            return true;
        }

        if (!masked_topology) {
            return false;
        }

        const int* beg = bsr_columns + bsr_offsets[row];
        const int* end = bsr_columns + bsr_active_row_end(bsr_offsets, bsr_row_counts, row);
        const int* block = std::lower_bound(beg, end, col);
        return block == end || *block != col;
    };

    int* valid_indices_end = std::remove_if(tpl_block_indices, tpl_block_indices + nnz, discard_invalid_block);

    // remove zero blocks
    if (tpl_values != nullptr && scalar_zero_mask != 0) {
        switch (scalar_size_in_bytes) {
        case sizeof(uint8_t):
            valid_indices_end = std::remove_if(
                tpl_block_indices, valid_indices_end, [block_size, tpl_values, scalar_zero_mask](uint32_t i) {
                    return bsr_block_is_zero<uint8_t>(i, block_size, tpl_values, scalar_zero_mask);
                }
            );
            break;
        case sizeof(uint16_t):
            valid_indices_end = std::remove_if(
                tpl_block_indices, valid_indices_end, [block_size, tpl_values, scalar_zero_mask](uint32_t i) {
                    return bsr_block_is_zero<uint16_t>(i, block_size, tpl_values, scalar_zero_mask);
                }
            );
            break;
        case sizeof(uint32_t):
            valid_indices_end = std::remove_if(
                tpl_block_indices, valid_indices_end, [block_size, tpl_values, scalar_zero_mask](uint32_t i) {
                    return bsr_block_is_zero<uint32_t>(i, block_size, tpl_values, scalar_zero_mask);
                }
            );
            break;
        case sizeof(uint64_t):
            valid_indices_end = std::remove_if(
                tpl_block_indices, valid_indices_end, [block_size, tpl_values, scalar_zero_mask](uint32_t i) {
                    return bsr_block_is_zero<uint64_t>(i, block_size, tpl_values, scalar_zero_mask);
                }
            );
            break;
        }
    }

    // sort block indices according to lexico order
    std::sort(tpl_block_indices, valid_indices_end, [tpl_rows, tpl_columns](int i, int j) -> bool {
        return tpl_rows[i] < tpl_rows[j] || (tpl_rows[i] == tpl_rows[j] && tpl_columns[i] < tpl_columns[j]);
    });

    // accumulate blocks at same locations, count blocks per row
    std::fill_n(bsr_offsets, row_count + 1, 0);

    int current_row = -1;
    int current_col = -1;
    int current_block_idx = -1;

    for (int *block = tpl_block_indices, *block_offset = tpl_block_offsets; block != valid_indices_end; ++block) {
        int32_t idx = *block;
        int row = tpl_rows[idx];
        int col = tpl_columns[idx];

        if (row != current_row || col != current_col) {
            *(bsr_columns++) = col;

            ++bsr_offsets[row + 1];

            if (current_row == -1) {
                *block_offset = 0;
            } else {
                *(block_offset + 1) = *block_offset;
                ++block_offset;
            }

            current_row = row;
            current_col = col;
        }

        ++(*block_offset);
    }

    // build postfix sum of row counts
    std::partial_sum(bsr_offsets, bsr_offsets + row_count + 1, bsr_offsets);

    if (!return_summed_blocks) {
        // free our temporary buffers
        wp_free_host(tpl_block_offsets);
        wp_free_host(tpl_block_indices);
    }

    if (bsr_nnz != nullptr) {
        *bsr_nnz = bsr_offsets[row_count];
    }
}

WP_API void wp_bsr_transpose_host(
    int row_count,
    int col_count,
    int nnz,
    const int* bsr_offsets,
    const int* bsr_row_counts,
    const int* bsr_columns,
    int* transposed_bsr_offsets,
    int* transposed_bsr_row_counts,
    int* transposed_bsr_columns,
    int* block_indices,
    int* status
)
{
    const int capacity = std::min(nnz, bsr_offsets[row_count]);
    const bool padded = transposed_bsr_row_counts != nullptr;

    std::vector<int> bsr_rows(capacity);

    // Fill row indices from active row ranges only.
    int active_nnz = 0;
    for (int row = 0; row < row_count; ++row) {
        const int row_end = std::min(bsr_active_row_end(bsr_offsets, bsr_row_counts, row), bsr_offsets[row + 1]);
        for (int block = bsr_offsets[row]; block < row_end; ++block) {
            block_indices[active_nnz++] = block;
            bsr_rows[block] = row;
        }
    }

    // sort block indices according to (transposed) lexico order
    std::sort(block_indices, block_indices + active_nnz, [&bsr_rows, bsr_columns](int i, int j) -> bool {
        return bsr_columns[i] < bsr_columns[j] || (bsr_columns[i] == bsr_columns[j] && bsr_rows[i] < bsr_rows[j]);
    });

    if (padded) {
        const std::vector<int> sorted_blocks(block_indices, block_indices + active_nnz);
        int sorted_beg = 0;

        for (int dest_row = 0; dest_row < col_count; ++dest_row) {
            int sorted_end = sorted_beg;
            while (sorted_end < active_nnz && bsr_columns[sorted_blocks[sorted_end]] == dest_row) {
                ++sorted_end;
            }

            const int block_count = sorted_end - sorted_beg;
            const int dest_beg = transposed_bsr_offsets[dest_row];
            const int dest_capacity_end = transposed_bsr_offsets[dest_row + 1];

            if (dest_beg + block_count > dest_capacity_end) {
                transposed_bsr_row_counts[dest_row] = 0;
                if (status != nullptr) {
                    *status = std::max(*status, BSR_STATUS_ROW_CAPACITY_EXCEEDED);
                }
                for (int block = dest_beg; block < dest_capacity_end; ++block) {
                    block_indices[block] = -1;
                }
                sorted_beg = sorted_end;
                continue;
            }

            transposed_bsr_row_counts[dest_row] = block_count;
            for (int block = dest_beg; block < dest_capacity_end; ++block) {
                const int local_block = block - dest_beg;
                if (local_block < block_count) {
                    const int src_block = sorted_blocks[sorted_beg + local_block];
                    transposed_bsr_columns[block] = bsr_rows[src_block];
                    block_indices[block] = src_block;
                } else {
                    block_indices[block] = -1;
                }
            }

            sorted_beg = sorted_end;
        }

        return;
    }

    // Count blocks per column and transpose blocks
    std::fill_n(transposed_bsr_offsets, col_count + 1, 0);

    for (int i = 0; i < active_nnz; ++i) {
        int idx = block_indices[i];
        int row = bsr_rows[idx];
        int col = bsr_columns[idx];

        ++transposed_bsr_offsets[col + 1];
        transposed_bsr_columns[i] = row;
    }

    // build postfix sum of column counts
    std::partial_sum(transposed_bsr_offsets, transposed_bsr_offsets + col_count + 1, transposed_bsr_offsets);

    std::fill(transposed_bsr_columns + active_nnz, transposed_bsr_columns + nnz, -1);
}

WP_API void wp_bsr_compress_inplace_host(
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
    (void)bsr_nnz_event;

    if (row_count <= 0) {
        if (make_compact && bsr_nnz != nullptr) {
            *bsr_nnz = 0;
        }
        return;
    }

    const bool prune_from_input_values = !compress_values && prune_numerical_zeros && bsr_values != nullptr;
    void* values_to_write = compress_values ? bsr_values : nullptr;

    if (values_to_write == nullptr && !prune_from_input_values) {
        bsr_compress_inplace_host_impl<wp::float32, false>(
            row_count, 0, nnz_upper_bound, false, make_compact, bsr_offsets, bsr_row_counts, bsr_columns, nullptr, 0,
            bsr_nnz
        );
        return;
    }

    if (!compress_values) {
        switch (scalar_size_in_bytes) {
        case sizeof(uint8_t):
            bsr_compress_inplace_host_impl<uint8_t, false>(
                row_count, block_size, nnz_upper_bound, prune_numerical_zeros, make_compact, bsr_offsets,
                bsr_row_counts, bsr_columns, bsr_values, scalar_zero_mask, bsr_nnz
            );
            break;
        case sizeof(uint16_t):
            bsr_compress_inplace_host_impl<uint16_t, false>(
                row_count, block_size, nnz_upper_bound, prune_numerical_zeros, make_compact, bsr_offsets,
                bsr_row_counts, bsr_columns, bsr_values, scalar_zero_mask, bsr_nnz
            );
            break;
        case sizeof(uint32_t):
            bsr_compress_inplace_host_impl<uint32_t, false>(
                row_count, block_size, nnz_upper_bound, prune_numerical_zeros, make_compact, bsr_offsets,
                bsr_row_counts, bsr_columns, bsr_values, scalar_zero_mask, bsr_nnz
            );
            break;
        case sizeof(uint64_t):
            bsr_compress_inplace_host_impl<uint64_t, false>(
                row_count, block_size, nnz_upper_bound, prune_numerical_zeros, make_compact, bsr_offsets,
                bsr_row_counts, bsr_columns, bsr_values, scalar_zero_mask, bsr_nnz
            );
            break;
        }
        return;
    }

    switch (scalar_type) {
    case BSR_SCALAR_FLOAT32:
        if (scalar_size_in_bytes == sizeof(wp::float32))
            bsr_compress_inplace_host_impl<wp::float32, true>(
                row_count, block_size, nnz_upper_bound, prune_numerical_zeros, make_compact, bsr_offsets,
                bsr_row_counts, bsr_columns, values_to_write, 0, bsr_nnz
            );
        break;
    case BSR_SCALAR_FLOAT64:
        if (scalar_size_in_bytes == sizeof(wp::float64))
            bsr_compress_inplace_host_impl<wp::float64, true>(
                row_count, block_size, nnz_upper_bound, prune_numerical_zeros, make_compact, bsr_offsets,
                bsr_row_counts, bsr_columns, values_to_write, 0, bsr_nnz
            );
        break;
    }
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
    const int* bsr_row_counts,
    int* bsr_columns,
    int* bsr_nnz,
    void* bsr_nnz_event
)
{
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
}


#endif
