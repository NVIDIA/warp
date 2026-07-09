// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "warp.h"

#include "apic.h"
#include "apic_internal.h"
#include "apic_types.h"
#include "sparse_util.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
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

// Record a host BSR-from-triplets topology build into the active APIC byte
// stream; returns true if recorded (and therefore should NOT execute now).
// Mirrors the sort / runlength-encode try-record helpers: under CPU graph
// capture the host call is otherwise invisible to the byte stream, so replay
// would leave the matrix topology frozen at the capture-time (deferred/empty)
// triplet data.
bool apic_capture_bsr_from_triplets(
    int block_size,
    int scalar_size_in_bytes,
    int row_count,
    int col_count,
    int nnz,
    const int* tpl_nnz,
    const int* tpl_rows,
    const int* tpl_columns,
    const void* tpl_values,
    uint64_t scalar_zero_mask,
    bool masked_topology,
    int* summed_block_offsets,
    int* summed_block_indices,
    int* bsr_offsets,
    const int* bsr_row_counts,
    int* bsr_columns,
    int* bsr_nnz
)
{
    APICState* state = wp_apic_get_recording_state();
    if (!state)
        return false;
    if (nnz <= 0)
        return false;  // empty topology: cheap and free of deferred-data hazards; execute normally
    // Only record the recordable shape (caller-provided summed-block buffers, as
    // wp.sparse.bsr_set_from_triplets always passes). If a caller relies on the
    // internally-allocated scratch path, fall through to normal execution.
    if (!tpl_rows || !tpl_columns || !summed_block_offsets || !summed_block_indices || !bsr_offsets || !bsr_columns)
        return false;

    uint64_t int_bytes = static_cast<uint64_t>(nnz) * sizeof(int32_t);
    uint64_t rowp1_bytes = (static_cast<uint64_t>(row_count) + 1) * sizeof(int32_t);
    uint64_t values_bytes
        = static_cast<uint64_t>(nnz) * static_cast<uint64_t>(block_size) * static_cast<uint64_t>(scalar_size_in_bytes);
    uint64_t bsr_columns_count = static_cast<uint64_t>(nnz);
    if (masked_topology && bsr_offsets)
        bsr_columns_count = std::max(bsr_columns_count, static_cast<uint64_t>(bsr_offsets[row_count]));
    uint64_t bsr_columns_bytes = bsr_columns_count * sizeof(int32_t);

    // Optional pointers: preserve APICAddress{} (region_id == -1) when null.
    // Do NOT call the resolver with a null pointer — it would auto-register
    // address 0 as a region, silently changing semantics.
    APICAddress tpl_nnz_addr;
    if (tpl_nnz)
        tpl_nnz_addr = apic_resolve_host_ptr(state, reinterpret_cast<uint64_t>(tpl_nnz), sizeof(int32_t));
    APICAddress tpl_rows_addr = apic_resolve_host_ptr(state, reinterpret_cast<uint64_t>(tpl_rows), int_bytes);
    APICAddress tpl_columns_addr = apic_resolve_host_ptr(state, reinterpret_cast<uint64_t>(tpl_columns), int_bytes);
    APICAddress tpl_values_addr;
    if (tpl_values)
        tpl_values_addr = apic_resolve_host_ptr(state, reinterpret_cast<uint64_t>(tpl_values), values_bytes);
    APICAddress sbo_addr = apic_resolve_host_ptr(state, reinterpret_cast<uint64_t>(summed_block_offsets), int_bytes);
    APICAddress sbi_addr = apic_resolve_host_ptr(state, reinterpret_cast<uint64_t>(summed_block_indices), int_bytes);
    APICAddress bo_addr = apic_resolve_host_ptr(state, reinterpret_cast<uint64_t>(bsr_offsets), rowp1_bytes);
    APICAddress brc_addr;
    if (bsr_row_counts)
        brc_addr = apic_resolve_host_ptr(
            state, reinterpret_cast<uint64_t>(bsr_row_counts), static_cast<uint64_t>(row_count) * sizeof(int32_t)
        );
    APICAddress bc_addr = apic_resolve_host_ptr(state, reinterpret_cast<uint64_t>(bsr_columns), bsr_columns_bytes);
    APICAddress bnnz_addr;
    if (bsr_nnz)
        bnnz_addr = apic_resolve_host_ptr(state, reinterpret_cast<uint64_t>(bsr_nnz), sizeof(int32_t));

    apic_record_bsr_from_triplets(
        state, block_size, scalar_size_in_bytes, row_count, col_count, nnz, scalar_zero_mask, masked_topology ? 1 : 0,
        tpl_nnz_addr.region_id, tpl_nnz_addr.offset, tpl_rows_addr.region_id, tpl_rows_addr.offset,
        tpl_columns_addr.region_id, tpl_columns_addr.offset, tpl_values_addr.region_id, tpl_values_addr.offset,
        sbo_addr.region_id, sbo_addr.offset, sbi_addr.region_id, sbi_addr.offset, bo_addr.region_id, bo_addr.offset,
        brc_addr.region_id, brc_addr.offset, bc_addr.region_id, bc_addr.offset, bnnz_addr.region_id, bnnz_addr.offset
    );
    return true;
}

// Record a host BSR-transpose into the active APIC byte stream; returns true if
// recorded (skip execution). ``nnz`` is the upper bound passed by Python; the
// replayed host call re-reads the exact nnz from the (replay-time) offsets.
bool apic_capture_bsr_transpose(
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
    APICState* state = wp_apic_get_recording_state();
    if (!state)
        return false;
    if (nnz <= 0)
        return false;
    if (!bsr_offsets || !bsr_columns || !transposed_bsr_offsets || !transposed_bsr_columns || !src_block_indices)
        return false;

    uint64_t int_bytes = static_cast<uint64_t>(nnz) * sizeof(int32_t);
    uint64_t rowp1_bytes = (static_cast<uint64_t>(row_count) + 1) * sizeof(int32_t);
    uint64_t colp1_bytes = (static_cast<uint64_t>(col_count) + 1) * sizeof(int32_t);
    // For a padded destination, transposed_bsr_offsets holds the fixed row-capacity
    // layout (a read-only input the transpose never writes). Snapshot it so replay
    // can restore it even if the caller resets the destination offsets before
    // capture_launch (GH-1587). For a compact destination it is an output the
    // transpose recomputes, so no snapshot is taken.
    const bool padded = transposed_bsr_row_counts != nullptr;
    // The transpose touches transposed_bsr_columns only within the destination's
    // capacity: offsets[col_count] blocks for a padded destination (which may be
    // smaller than the source's nnz upper bound — that is the capacity-overflow
    // case the padded API reports via status), nnz blocks for a compact one (the
    // caller guarantees fit). Claiming nnz on a smaller padded destination would
    // grow the region past the real allocation, corrupting pointer resolution
    // (and the host snapshot) for the rest of the capture.
    uint64_t transposed_capacity
        = padded ? static_cast<uint64_t>(std::max(transposed_bsr_offsets[col_count], 0)) : static_cast<uint64_t>(nnz);
    uint64_t transposed_columns_bytes = transposed_capacity * sizeof(int32_t);
    // block_indices carries the sorted source blocks (up to nnz entries) and, for
    // padded destinations, per-slot gap markers up to the destination capacity.
    uint64_t block_indices_bytes = std::max(int_bytes, transposed_columns_bytes);

    APICAddress bo_addr = apic_resolve_host_ptr(state, reinterpret_cast<uint64_t>(bsr_offsets), rowp1_bytes);
    APICAddress brc_addr;
    if (bsr_row_counts)
        brc_addr = apic_resolve_host_ptr(
            state, reinterpret_cast<uint64_t>(bsr_row_counts), static_cast<uint64_t>(row_count) * sizeof(int32_t)
        );
    APICAddress bc_addr = apic_resolve_host_ptr(state, reinterpret_cast<uint64_t>(bsr_columns), int_bytes);
    APICAddress to_addr = apic_resolve_host_ptr(state, reinterpret_cast<uint64_t>(transposed_bsr_offsets), colp1_bytes);
    APICAddress trc_addr;
    if (transposed_bsr_row_counts)
        trc_addr = apic_resolve_host_ptr(
            state, reinterpret_cast<uint64_t>(transposed_bsr_row_counts),
            static_cast<uint64_t>(col_count) * sizeof(int32_t)
        );
    APICAddress tc_addr
        = apic_resolve_host_ptr(state, reinterpret_cast<uint64_t>(transposed_bsr_columns), transposed_columns_bytes);
    APICAddress bi_addr
        = apic_resolve_host_ptr(state, reinterpret_cast<uint64_t>(src_block_indices), block_indices_bytes);
    APICAddress status_addr;
    if (status)
        status_addr = apic_resolve_host_ptr(state, reinterpret_cast<uint64_t>(status), sizeof(int32_t));

    const int32_t* padded_capacity_offsets = padded ? transposed_bsr_offsets : nullptr;
    int32_t padded_capacity_offset_count = padded ? (col_count + 1) : 0;
    apic_record_bsr_transpose(
        state, row_count, col_count, nnz, bo_addr.region_id, bo_addr.offset, brc_addr.region_id, brc_addr.offset,
        bc_addr.region_id, bc_addr.offset, to_addr.region_id, to_addr.offset, trc_addr.region_id, trc_addr.offset,
        tc_addr.region_id, tc_addr.offset, bi_addr.region_id, bi_addr.offset, status_addr.region_id, status_addr.offset,
        padded_capacity_offsets, padded_capacity_offset_count
    );
    return true;
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
    // Under CPU graph capture, record the topology build (so it replays with
    // fresh triplets) instead of executing it now on deferred/stale data.
    // ``nnz`` here is the triplet upper bound used for buffer sizing.
    if (apic_capture_bsr_from_triplets(
            block_size, scalar_size_in_bytes, row_count, col_count, nnz, tpl_nnz, tpl_rows, tpl_columns, tpl_values,
            scalar_zero_mask, masked_topology, tpl_block_offsets, tpl_block_indices, bsr_offsets, bsr_row_counts,
            bsr_columns, bsr_nnz
        ))
        return;

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
    // Under CPU graph capture, record the transpose (so it replays with the
    // fresh source topology) instead of executing it now on deferred/stale
    // offsets. ``nnz`` here is the upper bound (the exact nnz is re-read from
    // the offsets at replay time).
    if (apic_capture_bsr_transpose(
            row_count, col_count, nnz, bsr_offsets, bsr_row_counts, bsr_columns, transposed_bsr_offsets,
            transposed_bsr_row_counts, transposed_bsr_columns, block_indices, status
        ))
        return;

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
