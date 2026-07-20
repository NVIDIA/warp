// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stddef.h>
#include <stdint.h>

// =============================================================================
// APIC Format Constants
// =============================================================================

#define APIC_FORMAT_VERSION 15
#define APIC_MIN_SUPPORTED_FORMAT_VERSION 13
#define APIC_MAGIC "WRP1"
#define APIC_MAGIC_VALUE 0x31505257  // "WRP1" as little-endian uint32

// Maximum array dimensions (matches Warp's ARRAY_MAX_DIMS)
#define APIC_MAX_DIMS 4

// Maximum launch dimensions (must match LAUNCH_MAX_DIMS in builtin.h)
#define APIC_LAUNCH_MAX_DIMS 4

// =============================================================================
// Enums
// =============================================================================

// Operation types
enum APICOpType : uint32_t {
    APIC_OP_KERNEL_LAUNCH = 1,
    APIC_OP_MEMCPY_H2D = 2,
    APIC_OP_MEMCPY_D2H = 3,  // Reserved for future
    APIC_OP_MEMCPY_D2D = 4,
    APIC_OP_MEMSET = 5,
    APIC_OP_ALLOC = 6,  // In-graph allocation
    APIC_OP_IF = 7,  // wp.capture_if
    APIC_OP_WHILE = 8,  // wp.capture_while
    APIC_OP_SCAN = 9,  // wp.utils.array_scan
    APIC_OP_MEMTILE = 10,  // wp_memtile_host (multi-byte arr.fill_)
    APIC_OP_SEGMENTED_SORT = 11,  // wp.utils.segmented_sort_pairs
    APIC_OP_RADIX_SORT = 12,  // wp.utils.radix_sort_pairs
    APIC_OP_RUNLENGTH_ENCODE = 13,  // wp.utils.runlength_encode
    APIC_OP_BSR_FROM_TRIPLETS = 14,  // wp_bsr_matrix_from_triplets_host
    APIC_OP_BSR_TRANSPOSE = 15,  // wp_bsr_transpose_host
    APIC_OP_REDUCTION = 16,
};

enum APICReductionKind : uint8_t {
    APIC_REDUCTION_SUM = 1,
    APIC_REDUCTION_INNER = 2,
};

// Scalar element types
enum APICType : uint8_t {
    APIC_TYPE_INT32 = 1,
    APIC_TYPE_UINT32 = 2,
    APIC_TYPE_INT64 = 3,
    APIC_TYPE_UINT64 = 4,
    APIC_TYPE_FLOAT32 = 5,
    APIC_TYPE_FLOAT64 = 6,
};

// Per-value-blob relocation kinds. A relocation patches one 8-byte slot
// (pointer-sized) inside a kernel-launch parameter's value blob at replay
// time so the kernel sees a live process-local pointer instead of the stale
// pointer captured at recording time.
enum APICRelocKind : uint8_t {
    APIC_RELOC_DATA_PTR = 1,  // Region pointer: write resolve_ptr(region_id, region_offset).
    APIC_RELOC_HANDLE = 2,  // wp.handle / mesh id: write handle_ptr_remap[region_offset].
    APIC_RELOC_NULL = 3,  // Explicit zero (null array data/grad, absent indexedarray dim).
};

// =============================================================================
// WRP File Header
// =============================================================================

// All serialization structs are packed to ensure binary compatibility with Python
#pragma pack(push, 1)

struct APICFileHeader {
    uint8_t magic[4];  // "WRP1" -- Warp Recorded Program
    uint32_t version;  // APIC_FORMAT_VERSION
    uint32_t flags;  // Reserved flags
    uint32_t num_sections;  // Number of sections
    uint64_t section_table_offset;  // Offset to section table
    uint8_t device_type;  // APICDeviceType value (reserved for future; see wp_apic_load_graph)
    uint8_t _reserved_dt[3];
    uint32_t target_arch;  // CUDA SM version (e.g., 86) or 0 (CPU)
    uint32_t _reserved[8];  // Reserved for future use
};  // 64 bytes

// Device types (runtime APICGraph::device_type and the wire-format
// byte APICFileHeader::device_type use the same integer values).
enum APICDeviceType : uint32_t {
    APIC_DEVICE_CUDA = 0,
    APIC_DEVICE_CPU = 1,
};

// Section types
enum APICSectionType : uint32_t {
    APIC_SECTION_METADATA = 0x01,
    APIC_SECTION_MEMORY = 0x02,
    APIC_SECTION_OPERATIONS = 0x03,
};

struct APICSectionEntry {
    APICSectionType type;
    uint32_t flags;  // Section flags
    uint64_t offset;  // Offset from file start
    uint64_t size;  // Section size (compressed)
    uint64_t uncompressed_size;  // Uncompressed size
};  // 32 bytes

// =============================================================================
// Mesh Serialization Records
// =============================================================================

struct APICMeshRecord {
    int32_t num_points;
    int32_t num_tris;
    uint8_t support_winding_number;
    uint8_t bvh_constructor;
    uint16_t bvh_leaf_size;
    uint32_t points_region_id;
    uint32_t indices_region_id;
    uint32_t velocities_region_id;  // 0 if absent
    uint64_t original_ptr;
};  // 32 bytes

// =============================================================================
// Operation Records
// =============================================================================

// Common header for all operations
struct APICOpHeader {
    APICOpType op_type;
    uint32_t total_size;  // Total bytes including header and variable data
};  // 8 bytes

// Kernel launch record (fixed part).
//
// Every kernel argument is serialized as a "value blob" — the exact bytes the
// live launch would pass to the kernel — plus zero or more relocations that
// patch pointer-sized fields inside the blob at replay time. This is one
// uniform path for scalars, vec/mat, by-value structs, wp.array (the blob is
// the array_t descriptor with relocs on data + grad), wp.indexedarray (blob
// is the indexedarray_t with relocs on the nested array_t pointers and
// indices[]), wp.handle (blob is a uint64 handle with a HANDLE reloc), and
// composites of all of the above. No inline-vs-pool split, no special-case
// array record kind.
//
// Variable data layout follows the fixed part in this order:
//   1. char kernel_key[kernel_key_len]
//   2. char module_hash[module_hash_len]
//   3. APICLaunchParamRecord[num_params]              (forward bindings)
//   4. APICLaunchParamRecord[adj_count]                (adjoint bindings; only
//                                                       present when
//                                                       is_forward == 0, with
//                                                       adj_count == num_params)
//   5. APICLaunchPtrLocation[num_relocs]                     (flat reloc table;
//                                                      forward then adjoint,
//                                                      sliced per-param via
//                                                      each record's
//                                                      num_relocs counter)
//   6. uint8_t value_data[value_data_size]            (concatenated value
//                                                      blobs; each binding
//                                                      points at its slice
//                                                      via value_offset)
struct APICLaunchRecord {
    APICOpHeader header;  // op_type = APIC_OP_KERNEL_LAUNCH

    // Generated kernel launch bounds (embedded). Stores the shape and dimensionality
    // used by launch_bounds_t<N>, not necessarily the original Python wp.launch() rank.
    int32_t shape[APIC_LAUNCH_MAX_DIMS];  // Shape passed to the generated kernel entry point
    int32_t ndim;  // Kernel dimensionality used to select launch_bounds_t<N>
    uint64_t size;  // Total threads, including any folded launch axes

    // Launch parameters
    uint64_t dim;  // Total threads
    int32_t max_blocks;  // Maximum blocks
    int32_t block_dim;  // Threads per block
    int32_t smem_bytes;  // Shared memory bytes
    int32_t grid_stride;  // 1 = grid-stride loop kernel, 0 = lean 3D kernel
    uint8_t is_forward;  // 1 for forward pass, 0 for backward
    uint8_t cluster_dim;  // 1D CTA cluster size, or 0 for older records
    uint8_t _pad1[2];

    // Variable data sizes
    uint16_t kernel_key_len;  // Length of kernel_key string
    uint16_t module_hash_len;  // Length of module_hash string
    uint16_t num_params;  // Forward parameter bindings. Backward kernels carry
                          // an adjoint block of the same size after the forward
                          // block; readers compute adj_count = is_forward ? 0 :
                          // num_params.
    uint16_t _pad2;  // Reserved, must be 0.

    uint32_t num_relocs;  // Total entries in the trailing reloc table
                          // (forward + adjoint, summed).
    uint32_t value_data_size;  // Bytes of trailing value_data section.
};

// One entry per kernel argument (1-based; param_index = 0 is reserved for
// launch_bounds). Describes a slice [value_offset, value_offset + value_size)
// of the per-launch value_data section, plus the count of relocations that
// patch pointer fields inside the slice.
struct APICLaunchParamRecord {
    uint16_t param_index;  // Parameter index in kernel signature (1-based)
    uint16_t num_relocs;  // Count of relocations in the flat reloc table
                          // that belong to this binding (consumed in order).
    uint32_t value_offset;  // Byte offset of this binding's value blob within
                            // the per-launch value_data section.
    uint32_t value_size;  // Size of the value blob in bytes.
    uint32_t value_align;  // Natural C++ alignof for the kernel args-struct
                           // field this binding feeds (so the replay-side
                           // packer reproduces the live args-struct layout).
};  // 16 bytes

// One entry per pointer field inside a value blob — the launch-time analogue
// of APICMemoryPtrLocation (defined in apic_internal.h, which records pointer
// offsets inside captured memory regions for handle fixup at load time). The
// two structs share the same idea ("where the pointer lives, plus what it
// should be patched with") but the lifetimes are very different: this one is
// consumed on every kernel launch to rewrite the args buffer, while
// APICMemoryPtrLocation is applied once at graph-load time to fix up handles
// in registered regions.
//
// At replay time we copy each binding's value_data slice into the kernel
// args buffer and then, for each relocation, overwrite the 8 bytes at
// [args_buf_field_offset, +8) with the kind-specific patched pointer:
//   - DATA_PTR: resolve_ptr(region_id, region_offset).
//   - HANDLE:   handle_ptr_remap[region_offset]  (region_offset carries the
//               original captured handle id; region_id is unused / -1).
//   - NULL:     literal zero (e.g. null array.data, absent indexedarray dim).
struct APICLaunchPtrLocation {
    uint32_t value_byte_offset;  // Offset within the value blob.
    int32_t region_id;  // For DATA_PTR; -1 for HANDLE / NULL.
    uint64_t region_offset;  // DATA_PTR: byte offset within region.
                             // HANDLE:   original captured handle id (uint64).
                             // NULL:     unused.
    uint8_t kind;  // APICRelocKind
    uint8_t _pad[7];  // Reserved, must be 0.
};  // 24 bytes

// Memcpy Host-to-Device (variable: has inline data)
struct APICMemcpyH2DRecord {
    APICOpHeader header;  // op_type = APIC_OP_MEMCPY_H2D
    int32_t dst_region_id;
    uint32_t _pad;
    uint64_t dst_offset;
    uint64_t size;
    // uint8_t data[size] follows
};  // 32 bytes fixed

// Memcpy Device-to-Device (fixed size)
struct APICMemcpyD2DRecord {
    APICOpHeader header;  // op_type = APIC_OP_MEMCPY_D2D
    int32_t dst_region_id;
    int32_t src_region_id;
    uint64_t dst_offset;
    uint64_t src_offset;
    uint64_t size;
};  // 40 bytes

// Memset (fixed size)
struct APICMemsetRecord {
    APICOpHeader header;  // op_type = APIC_OP_MEMSET
    int32_t region_id;
    int32_t value;
    uint64_t offset;
    uint64_t size;
};  // 32 bytes

// Memtile fill (variable: trailing inline value blob of srcsize bytes).
// Records a contiguous wp_memtile_host call that writes ``count`` copies of
// the ``srcsize``-byte value into the destination region starting at
// ``offset``. Used by ``arr.fill_(value)`` for multi-byte element types,
// which wp_memset cannot express because memset only repeats a single byte.
struct APICMemtileRecord {
    APICOpHeader header;  // op_type = APIC_OP_MEMTILE
    int32_t region_id;
    uint32_t srcsize;  // bytes per element
    uint64_t offset;
    uint64_t count;  // number of elements
    // followed by srcsize bytes of inline value data
};  // 32 bytes fixed

// In-graph allocation (fixed size)
struct APICAllocRecord {
    APICOpHeader header;  // op_type = APIC_OP_ALLOC
    int32_t region_id;
    uint32_t _pad;
    uint64_t size;
};  // 24 bytes

// Scan op (fixed size). Records a wp.utils.array_scan() call so replay
// recomputes from the current input region instead of returning the
// capture-time output. ``dtype`` selects the scalar host entry point.
struct APICScanRecord {
    APICOpHeader header;  // op_type = APIC_OP_SCAN
    int32_t dst_region_id;
    int32_t src_region_id;
    uint64_t dst_offset;
    uint64_t src_offset;
    uint32_t length;  // element count
    int32_t in_stride;
    int32_t out_stride;
    int32_t type_len;
    uint8_t dtype;  // APICType
    uint8_t inclusive;  // 1 = inclusive, 0 = exclusive
    uint8_t _pad[2];
};

struct APICReductionRecord {
    APICOpHeader header;  // op_type = APIC_OP_REDUCTION
    int32_t input_a_region_id;
    int32_t input_b_region_id;  // -1 for APIC_REDUCTION_SUM
    int32_t output_region_id;
    uint32_t count;
    uint64_t input_a_offset;
    uint64_t input_b_offset;
    uint64_t output_offset;
    int32_t input_a_stride;
    int32_t input_b_stride;  // 0 for APIC_REDUCTION_SUM
    int32_t type_len;
    uint8_t kind;  // APICReductionKind
    uint8_t dtype;  // APICType: FLOAT32 or FLOAT64
    uint8_t _pad[2];
};  // 64 bytes
static_assert(sizeof(APICReductionRecord) == 64, "APICReductionRecord must remain 64 bytes");

// Segmented-sort op (fixed size). Records a wp.utils.segmented_sort_pairs()
// call so replay re-sorts the current (replay-time) keys/values instead of
// leaving them in capture-time order. On CPU this op exists because the sort
// dispatches to a host function (wp_segmented_sort_pairs_*_host) that, unlike a
// kernel launch, is otherwise invisible to the byte stream. ``dtype`` is the
// key dtype (APICType: INT32 or FLOAT32); values are always int32. The
// keys/values buffers must hold 2*count elements (sort scratch).
struct APICSegmentedSortRecord {
    APICOpHeader header;  // op_type = APIC_OP_SEGMENTED_SORT
    int32_t keys_region_id;
    int32_t values_region_id;
    int32_t segstart_region_id;
    int32_t segend_region_id;
    uint64_t keys_offset;
    uint64_t values_offset;
    uint64_t segstart_offset;
    uint64_t segend_offset;
    uint32_t count;  // n (number of elements to sort)
    uint32_t num_segments;
    uint8_t dtype;  // APICType (key dtype)
    uint8_t _pad[3];
};

// Radix-sort op (fixed size). Records a wp.utils.radix_sort_pairs() call (the
// non-segmented host sort). Same rationale as APICSegmentedSortRecord: on CPU
// the sort dispatches to a host function invisible to the byte stream.
// ``dtype`` is the key APICType and ``value_size`` records whether each
// value element is 4 or 8 bytes. keys/values buffers hold 2*count elements
// (sort scratch).
struct APICRadixSortRecord {
    APICOpHeader header;  // op_type = APIC_OP_RADIX_SORT
    int32_t keys_region_id;
    int32_t values_region_id;
    uint64_t keys_offset;
    uint64_t values_offset;
    uint32_t count;
    int32_t begin_bit;
    int32_t end_bit;
    int32_t value_size;
    uint8_t dtype;  // APICType
    uint8_t _pad[3];
};

// Run-length-encode op (fixed size). Records a wp.utils.runlength_encode() call
// (the int32 host path). Same rationale as the sorts: on CPU it dispatches to a
// host function invisible to the byte stream. Inputs/outputs are all int32:
// ``values`` (value_count elements) -> ``run_values`` / ``run_lengths`` (up to
// value_count) and a single-element ``run_count``.
struct APICRunlengthEncodeRecord {
    APICOpHeader header;  // op_type = APIC_OP_RUNLENGTH_ENCODE
    int32_t values_region_id;
    int32_t run_values_region_id;
    int32_t run_lengths_region_id;
    int32_t run_count_region_id;
    uint64_t values_offset;
    uint64_t run_values_offset;
    uint64_t run_lengths_offset;
    uint64_t run_count_offset;
    uint32_t value_count;
    uint8_t _pad[4];
};

// BSR-from-triplets op. Records a wp_bsr_matrix_from_triplets_host() call so
// replay recomputes the matrix topology (bsr_offsets/bsr_columns and the
// per-triplet summed_block offsets/indices that the recorded
// _bsr_accumulate_triplet_values kernel reads) from the current (replay-time)
// triplets. Same rationale as the sorts/runlength-encode: on CPU the topology
// is computed by a host function that, unlike a kernel launch, is otherwise
// invisible to the byte stream — so without this op the topology stays frozen
// at the capture-time (deferred/empty) triplet data. Region ids are -1 for the
// optional pointers (tpl_nnz, tpl_values, bsr_row_counts, bsr_nnz) when absent.
// Buffer sizes derive from ``nnz_upper_bound`` (the triplet count) and the
// scalar fields.
struct APICBsrFromTripletsRecord {
    APICOpHeader header;  // op_type = APIC_OP_BSR_FROM_TRIPLETS
    int32_t block_size;
    int32_t scalar_size_in_bytes;
    int32_t row_count;
    int32_t col_count;
    int32_t nnz_upper_bound;
    uint8_t masked_topology;
    uint8_t _pad[3];
    uint64_t scalar_zero_mask;
    int32_t tpl_nnz_region_id;  // -1 if absent
    int32_t tpl_rows_region_id;
    int32_t tpl_columns_region_id;
    int32_t tpl_values_region_id;  // -1 if absent
    int32_t summed_block_offsets_region_id;
    int32_t summed_block_indices_region_id;
    int32_t bsr_offsets_region_id;
    int32_t bsr_row_counts_region_id;  // -1 if absent
    int32_t bsr_columns_region_id;
    int32_t bsr_nnz_region_id;  // -1 if absent
    uint64_t tpl_nnz_offset;
    uint64_t tpl_rows_offset;
    uint64_t tpl_columns_offset;
    uint64_t tpl_values_offset;
    uint64_t summed_block_offsets_offset;
    uint64_t summed_block_indices_offset;
    uint64_t bsr_offsets_offset;
    uint64_t bsr_row_counts_offset;
    uint64_t bsr_columns_offset;
    uint64_t bsr_nnz_offset;
};

// BSR-transpose op. Records a wp_bsr_transpose_host() call so replay recomputes
// the transposed topology (transposed_offsets/columns and block_indices, which
// the recorded _bsr_transpose_values kernel reads) from the current source
// topology. Same rationale as APICBsrFromTripletsRecord. Buffer sizes derive
// from ``nnz_upper_bound`` and row/col counts.
//
// For padded destinations (``transposed_row_counts_region_id >= 0``) the
// transpose reads ``transposed_offsets`` as the fixed row-capacity layout but
// never writes it, so a ``col_count + 1`` int32 snapshot of that capacity is
// appended after the fixed record (format version >= 14). Replay restores it
// into the destination offsets before executing, so the captured graph rebuilds
// the destination even if the caller reset its offsets buffer before replay
// (GH-1587). Compact destinations recompute the offsets and append no tail.
struct APICBsrTransposeRecord {
    APICOpHeader header;  // op_type = APIC_OP_BSR_TRANSPOSE
    int32_t row_count;
    int32_t col_count;
    int32_t nnz_upper_bound;
    int32_t bsr_offsets_region_id;
    int32_t bsr_row_counts_region_id;  // -1 if absent
    int32_t bsr_columns_region_id;
    int32_t transposed_offsets_region_id;
    int32_t transposed_row_counts_region_id;  // -1 if absent
    int32_t transposed_columns_region_id;
    int32_t block_indices_region_id;
    int32_t status_region_id;  // -1 if absent
    uint64_t bsr_offsets_offset;
    uint64_t bsr_row_counts_offset;
    uint64_t bsr_columns_offset;
    uint64_t transposed_offsets_offset;
    uint64_t transposed_row_counts_offset;
    uint64_t transposed_columns_offset;
    uint64_t block_indices_offset;
    uint64_t status_offset;
};

// Conditional / loop op (variable: trailing per-branch op-stream blocks).
// Used by both APIC_OP_IF (then + else branches) and APIC_OP_WHILE
// (body in branch_a, branch_b empty). Replay reads the int32 condition
// from `cond_region_id + cond_offset`, treats nonzero as "execute
// branch_a / repeat body" and zero as "execute branch_b / exit loop".
struct APICCondRecord {
    APICOpHeader header;  // op_type = APIC_OP_IF or APIC_OP_WHILE
    int32_t cond_region_id;
    uint32_t _pad;
    uint64_t cond_offset;
    uint32_t branch_a_size;  // bytes
    uint32_t branch_a_op_count;
    uint32_t branch_b_size;  // bytes (0 for APIC_OP_WHILE)
    uint32_t branch_b_op_count;
    // followed by branch_a_size bytes of inner ops
    // followed by branch_b_size bytes of inner ops
};  // 40 bytes fixed

// =============================================================================
// Memory Section Records
// =============================================================================

struct APICMemoryRegionRecord {
    uint32_t region_id;
    uint32_t element_size;
    uint64_t size;  // Size in bytes
    uint8_t has_initial_data;  // 1 if initial_data follows
    uint8_t _pad[7];
    // If has_initial_data: uint8_t initial_data[size] follows
};  // 24 bytes fixed

#pragma pack(pop)

// =============================================================================
// Recording API Structures (passed from Python to C++, naturally aligned)
// =============================================================================

// Launch info passed to wp_cuda_launch_kernel() / wp_cpu_launch_kernel()
// Not packed — contains pointers that must be naturally aligned.
struct APICLaunchInfo {
    const char* kernel_key;  // Kernel identifier string
    const char* module_hash;  // Module hash string
    uint8_t is_forward;  // 1 for forward, 0 for backward
    uint8_t _pad[7];  // Align params to 8 bytes
    const APICLaunchParamRecord* params;  // Forward parameter bindings
    int32_t num_params;  // Number of forward parameter bindings
    int32_t kernel_dim;  // Kernel launch dimensionality (1-4), from kernel.adj.kernel_dim
    uint32_t value_data_size;  // Size in bytes of value_data (0 if none)
    const uint8_t* value_data;  // Per-launch value blobs (concatenated, sliced
                                // by each binding's value_offset / value_size)
    // Adjoint parameter bindings (NULL for forward, num_params entries for
    // backward).
    const APICLaunchParamRecord* adj_params;
    uint32_t num_relocs;  // Total entries in `relocs` (forward + adjoint)
    uint32_t _pad2;
    const APICLaunchPtrLocation* relocs;  // Flat per-launch relocation table
};

// =============================================================================
// Execution Structures (must match runtime types in builtin.h and array.h)
// =============================================================================

// Array descriptor - must match layout of array_t<T> in array.h
struct apic_array_t {
    uint64_t data;  // Device/host pointer
    uint64_t grad;  // Gradient pointer (usually 0)
    int shape[APIC_MAX_DIMS];
    int strides[APIC_MAX_DIMS];
    uint16_t ndim;
    uint16_t flags;
};
