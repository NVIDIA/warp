// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Internal APIC structures and cross-file helpers shared between apic.cpp
// (pure C++) and apic.cu (CUDA). This header must compile with both nvcc and
// standard C++ compilers, so it must not expose CUDA types outside the
// __CUDACC__ / WP_ENABLE_CUDA guards.

#include "builtin.h"

#include "apic_types.h"

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef __CUDACC__
#include <cudaTypedefs.h>
#endif

// Byte size of a scalar APICType value: 4 for the 32-bit members, 8 for the
// 64-bit members, 0 for an unrecognized/unset value. Shared by the array_scan
// and radix/segmented sort capture and replay paths (apic.cpp and warp.cpp).
inline uint32_t apic_type_size(uint8_t dtype)
{
    switch (dtype) {
    case APIC_TYPE_INT32:
    case APIC_TYPE_UINT32:
    case APIC_TYPE_FLOAT32:
        return 4;
    case APIC_TYPE_INT64:
    case APIC_TYPE_UINT64:
    case APIC_TYPE_FLOAT64:
        return 8;
    default:
        return 0;
    }
}

// Byte span touched by a strided array_scan operand: the last element starts at
// (length-1)*stride scalars and spans type_len scalars. Returns 0 for an empty
// or malformed descriptor (length 0, negative stride, non-positive type_len, or
// unknown scalar size). Shared by the scan capture/replay paths (apic.cpp) and
// the CUDA rebuild (apic.cu).
inline size_t apic_strided_access_bytes(uint32_t length, int32_t stride, int32_t type_len, size_t scalar_size)
{
    if (length == 0 || stride < 0 || type_len <= 0 || scalar_size == 0)
        return 0;
    return static_cast<size_t>(length - 1) * static_cast<size_t>(stride) + static_cast<size_t>(type_len) * scalar_size;
}

inline size_t apic_reduction_input_bytes(uint32_t count, int32_t stride, int32_t type_len, size_t scalar_size)
{
    if (count == 0 || stride < 0 || type_len <= 0 || scalar_size == 0)
        return 0;

    size_t count_minus_one = static_cast<size_t>(count - 1);
    size_t byte_stride = static_cast<size_t>(stride);
    size_t lanes = static_cast<size_t>(type_len);
    if ((byte_stride != 0 && count_minus_one > SIZE_MAX / byte_stride) || lanes > SIZE_MAX / scalar_size)
        return 0;

    size_t strided_bytes = count_minus_one * byte_stride;
    size_t value_bytes = lanes * scalar_size;
    if (strided_bytes > SIZE_MAX - value_bytes)
        return 0;
    return strided_bytes + value_bytes;
}

inline bool apic_reduction_metadata_valid(
    uint64_t input_a,
    uint64_t input_b,
    uint64_t output,
    int32_t count,
    int32_t input_a_stride,
    int32_t input_b_stride,
    int32_t type_len,
    uint8_t kind,
    uint8_t dtype
)
{
    bool is_sum = kind == APIC_REDUCTION_SUM;
    bool is_inner = kind == APIC_REDUCTION_INNER;
    size_t scalar_size = apic_type_size(dtype);
    return (is_sum || is_inner) && scalar_size > 0 && count > 0 && type_len > 0 && input_a != 0 && output != 0
        && input_a_stride >= 0 && input_a % scalar_size == 0 && output % scalar_size == 0
        && input_a_stride % scalar_size == 0
        && ((is_sum && input_b == 0 && input_b_stride == 0)
            || (is_inner && input_b != 0 && input_b_stride >= 0 && input_b % scalar_size == 0
                && input_b_stride % scalar_size == 0));
}

namespace apic_detail {

constexpr size_t launch_bounds_align8(size_t offset) { return (offset + 7) & ~size_t(7); }

constexpr size_t launch_bounds_size_offset(int ndim)
{
    return launch_bounds_align8(static_cast<size_t>(ndim) * sizeof(int));
}

constexpr size_t launch_bounds_coord_mult_offset(int ndim) { return launch_bounds_size_offset(ndim) + sizeof(size_t); }

constexpr size_t launch_bounds_storage_size(int ndim)
{
    return launch_bounds_align8(launch_bounds_coord_mult_offset(ndim) + sizeof(size_t));
}

template <int N> constexpr bool launch_bounds_layout_matches_apic_buffer()
{
    using Bounds = wp::launch_bounds_t<N>;
    static_assert(offsetof(Bounds, shape) == 0, "APIC replay expects launch_bounds_t<N>::shape at offset 0");
    static_assert(
        offsetof(Bounds, size) == launch_bounds_size_offset(N),
        "APIC replay size offset must match launch_bounds_t<N>::size"
    );
    static_assert(
        offsetof(Bounds, coord_mult) == launch_bounds_coord_mult_offset(N),
        "APIC replay coord_mult offset must match launch_bounds_t<N>::coord_mult"
    );
    static_assert(
        sizeof(Bounds) == launch_bounds_storage_size(N),
        "APIC replay launch-bounds buffer size must match launch_bounds_t<N>"
    );
    return true;
}

static_assert(
    APIC_LAUNCH_MAX_DIMS == wp::LAUNCH_MAX_DIMS,
    "APIC_LAUNCH_MAX_DIMS (apic_types.h) must match wp::LAUNCH_MAX_DIMS (builtin.h)"
);
static_assert(launch_bounds_layout_matches_apic_buffer<1>(), "APIC launch_bounds_t<1> layout mismatch");
static_assert(launch_bounds_layout_matches_apic_buffer<2>(), "APIC launch_bounds_t<2> layout mismatch");
static_assert(launch_bounds_layout_matches_apic_buffer<3>(), "APIC launch_bounds_t<3> layout mismatch");
static_assert(launch_bounds_layout_matches_apic_buffer<4>(), "APIC launch_bounds_t<4> layout mismatch");

}  // namespace apic_detail

// ============================================================================
// APIC Internal Structures
// ============================================================================

struct APICModule {
    std::string module_hash;
    std::string module_name;
    std::string cubin_filename;
    int target_arch = 0;
#ifdef __CUDACC__
    CUmodule cuda_module = nullptr;  // Set after loading
#else
    void* cuda_module = nullptr;  // CUmodule, opaque on non-CUDA compilers
#endif
};

struct APICKernel {
    std::string kernel_key;
    std::string module_hash;
    std::string forward_name;
    std::string backward_name;
    int forward_smem_bytes = 0;
    int backward_smem_bytes = 0;
    int block_dim = 0;
};

// Include the module hash anywhere APIC keys kernel metadata or CPU function
// pointers. Same-key kernels can exist in distinct unique modules, and \x1f
// cannot appear in a hex hash or qualified kernel name.
static inline std::string apic_kernel_map_key(const std::string& module_hash, const std::string& kernel_key)
{
    return module_hash + '\x1f' + kernel_key;
}

// Abstract APIC address: a (region_id, byte_offset) pair that fully specifies
// a byte location in APIC's region-based memory model. Default-constructed
// value (region_id == -1) means "no region" / null address.
struct APICAddress {
    int32_t region_id = -1;
    uint64_t offset = 0;
};

struct APICMemory {
    uint32_t region_id = 0;
    uint64_t base_ptr = 0;
    uint64_t size = 0;
    uint32_t element_size = 0;
    std::vector<uint8_t> initial_data;
    void* ptr = nullptr;  // Allocated pointer (device or host depending on context)
};

struct APICMemoryPtrLocation {
    uint32_t region_id;
    uint64_t offset;
    uint64_t stride;
};

// CPU kernel function pointers, resolved during capture / load and consumed
// by CPU replay. Stored both in APICState (for live capture) and
// APICGraph (for loaded .wrp graphs), keyed by (module_hash, kernel_key).
struct APICCPUKernel {
    void* forward_fn = nullptr;
    void* backward_fn = nullptr;
};

// ============================================================================
// APICState — recording state
// ============================================================================

struct APICState {
    bool recording = false;

    // Device class of the active capture (set by wp_apic_begin_recording). Host
    // capture hooks record only when this is true and CUDA device hooks only
    // when it is false, so a device-mismatched op invoked during a capture is
    // not recorded into the wrong byte stream.
    bool is_cpu = false;

    // Set by wp_apic_end_recording after apic_validate_operation_stream passes.
    // Replay paths require it; see apic_validate_operation_stream.
    bool operations_validated = false;

    // Contiguous operation byte stream (serializable)
    std::vector<uint8_t> operation_stream;
    uint32_t operation_count = 0;

    // Memory regions keyed by region_id (registered from Python)
    struct RegionInfo {
        uint32_t region_id = 0;
        uint64_t size = 0;
        uint32_t element_size = 0;
        std::vector<uint8_t> initial_data;
    };
    std::map<uint32_t, RegionInfo> memory_regions;

    // Active region-id entries used for pointer resolution. region_ids are
    // dense (from next_region_id), so a vector indexed by region_id gives O(1)
    // lookup. Each entry stores the current base pointer and recorded size, not
    // the region's byte contents. All writes go through apic_set_region_entry().
    // Critical for replay of launch-dense graphs.
    struct RegionEntry {
        uint64_t base_ptr = 0;
        uint64_t size = 0;
        bool valid = false;
    };
    std::vector<RegionEntry> regions_by_id;
    uint32_t next_region_id = 1;

    // Module and kernel metadata (registered from Python for serialization).
    // Kernels are keyed by (module_hash, kernel_key) so same-key kernels from
    // distinct unique modules do not overwrite each other.
    std::unordered_map<std::string, APICModule> modules;
    // Keyed by (module_hash, kernel_key), matching serialized kernel metadata.
    std::unordered_map<std::string, APICKernel> kernels;

    // Named bindings (name -> region_id)
    std::vector<std::pair<std::string, uint32_t>> bindings;

    // Handle pointer locations
    std::vector<APICMemoryPtrLocation> ptr_locations;

    // Mesh records (registered from Python during capture_save)
    std::vector<APICMeshRecord> mesh_records;

    // CPU kernel function pointers (registered during capture for replay)
    std::unordered_map<std::string, APICCPUKernel> cpu_kernels;

    void append_bytes(const void* data, size_t sz)
    {
        size_t offset = operation_stream.size();
        operation_stream.resize(offset + sz);
        memcpy(operation_stream.data() + offset, data, sz);
    }
};

// ============================================================================
// Operation Recording (called from the kernel-launch / memcpy / memset hooks in
// warp.cu and warp.cpp when a capture is active)
// ============================================================================

// Records a kernel launch into the byte stream. For backward kernels
// (is_forward == 0) ``adj_params`` must be non-NULL and is read as
// ``num_params`` adjoint bindings — there is no separate adjoint count,
// since the forward and adjoint blocks always have the same shape.
void apic_record_kernel_launch(
    APICState* state,
    const char* kernel_key,
    const char* module_hash,
    int is_forward,
    const int* shape,
    int ndim,
    uint64_t size,
    int max_blocks,
    int block_dim,
    int grid_stride,
    int cluster_dim,
    int smem_bytes,
    const APICLaunchParamRecord* params,
    int num_params,
    const APICLaunchParamRecord* adj_params,
    const APICLaunchPtrLocation* relocs,
    uint32_t num_relocs,
    const uint8_t* value_data,
    uint32_t value_data_size
);

void apic_record_memcpy_d2d(
    APICState* state,
    int32_t dst_region_id,
    uint64_t dst_offset,
    int32_t src_region_id,
    uint64_t src_offset,
    uint64_t size
);

void apic_record_memset(APICState* state, int32_t region_id, uint64_t offset, uint64_t size, int32_t value);

// Records a wp_memtile_host call. ``srcsize`` bytes from ``value_ptr``
// are copied into the byte stream verbatim, then repeated ``count`` times
// into the destination region at replay time.
void apic_record_memtile(
    APICState* state, int32_t region_id, uint64_t offset, uint32_t srcsize, const void* value_ptr, uint64_t count
);

void apic_record_alloc(APICState* state, int32_t region_id, uint64_t size);

// Records a wp.utils.array_scan() call. dtype is APICType.
void apic_record_scan(
    APICState* state,
    int32_t dst_region_id,
    uint64_t dst_offset,
    int32_t src_region_id,
    uint64_t src_offset,
    uint32_t length,
    int32_t in_stride,
    int32_t out_stride,
    int32_t type_len,
    uint8_t dtype,
    uint8_t inclusive
);

void apic_record_reduction(
    APICState* state,
    int32_t input_a_region_id,
    uint64_t input_a_offset,
    int32_t input_b_region_id,
    uint64_t input_b_offset,
    int32_t output_region_id,
    uint64_t output_offset,
    uint32_t count,
    int32_t input_a_stride,
    int32_t input_b_stride,
    int32_t type_len,
    uint8_t kind,
    uint8_t dtype
);

// Records a wp.utils.segmented_sort_pairs() call. dtype is the key
// APICType (INT32 or FLOAT32); values are always int32.
void apic_record_segmented_sort(
    APICState* state,
    int32_t keys_region_id,
    uint64_t keys_offset,
    int32_t values_region_id,
    uint64_t values_offset,
    int32_t segstart_region_id,
    uint64_t segstart_offset,
    int32_t segend_region_id,
    uint64_t segend_offset,
    uint32_t count,
    uint32_t num_segments,
    uint8_t dtype
);

// Records a wp.utils.radix_sort_pairs() call. dtype is the key
// APICType; value_size is 4 or 8 bytes.
void apic_record_radix_sort(
    APICState* state,
    int32_t keys_region_id,
    uint64_t keys_offset,
    int32_t values_region_id,
    uint64_t values_offset,
    uint32_t count,
    int32_t begin_bit,
    int32_t end_bit,
    int32_t value_size,
    uint8_t dtype
);

// Records a wp.utils.runlength_encode() call. All arrays are int32.
void apic_record_runlength_encode(
    APICState* state,
    int32_t values_region_id,
    uint64_t values_offset,
    int32_t run_values_region_id,
    uint64_t run_values_offset,
    int32_t run_lengths_region_id,
    uint64_t run_lengths_offset,
    int32_t run_count_region_id,
    uint64_t run_count_offset,
    uint32_t value_count
);

// Records a wp_bsr_matrix_from_triplets_host() call. Optional regions
// (tpl_nnz, tpl_values, bsr_nnz) carry region_id == -1 when absent.
void apic_record_bsr_from_triplets(
    APICState* state,
    int32_t block_size,
    int32_t scalar_size_in_bytes,
    int32_t row_count,
    int32_t col_count,
    int32_t nnz_upper_bound,
    uint64_t scalar_zero_mask,
    uint8_t masked_topology,
    int32_t tpl_nnz_region_id,
    uint64_t tpl_nnz_offset,
    int32_t tpl_rows_region_id,
    uint64_t tpl_rows_offset,
    int32_t tpl_columns_region_id,
    uint64_t tpl_columns_offset,
    int32_t tpl_values_region_id,
    uint64_t tpl_values_offset,
    int32_t summed_block_offsets_region_id,
    uint64_t summed_block_offsets_offset,
    int32_t summed_block_indices_region_id,
    uint64_t summed_block_indices_offset,
    int32_t bsr_offsets_region_id,
    uint64_t bsr_offsets_offset,
    int32_t bsr_row_counts_region_id,
    uint64_t bsr_row_counts_offset,
    int32_t bsr_columns_region_id,
    uint64_t bsr_columns_offset,
    int32_t bsr_nnz_region_id,
    uint64_t bsr_nnz_offset
);

// Records a wp_bsr_transpose_host() call. For padded destinations,
// ``padded_capacity_offsets`` carries a ``padded_capacity_offset_count`` int32
// snapshot of the destination row-capacity layout so replay can restore it
// (pass nullptr / 0 for compact destinations, which recompute the offsets).
void apic_record_bsr_transpose(
    APICState* state,
    int32_t row_count,
    int32_t col_count,
    int32_t nnz_upper_bound,
    int32_t bsr_offsets_region_id,
    uint64_t bsr_offsets_offset,
    int32_t bsr_row_counts_region_id,
    uint64_t bsr_row_counts_offset,
    int32_t bsr_columns_region_id,
    uint64_t bsr_columns_offset,
    int32_t transposed_offsets_region_id,
    uint64_t transposed_offsets_offset,
    int32_t transposed_row_counts_region_id,
    uint64_t transposed_row_counts_offset,
    int32_t transposed_columns_region_id,
    uint64_t transposed_columns_offset,
    int32_t block_indices_region_id,
    uint64_t block_indices_offset,
    int32_t status_region_id,
    uint64_t status_offset,
    const int32_t* padded_capacity_offsets,
    int32_t padded_capacity_offset_count
);

// ----- Pointer resolution (for memcpy/memset hooks that receive raw pointers) -----

// Register a live replay region for `ptr` and return its APIC address.
// If `ptr` falls inside an existing region that region is reused; if the
// access would overshoot the recorded size the region is grown in place.
// If no existing region contains `ptr` a fresh region is created with
// base_ptr = ptr and size = max(access_size, 1). Always succeeds —
// region_id == -1 only when `state` is null.
// Does not snapshot host bytes or update state->memory_regions.
// Use for CUDA / device pointers where host memcpy would be unsafe.
APICAddress apic_resolve_live_ptr(APICState* state, uint64_t ptr, uint64_t access_size);

// Same as apic_resolve_live_ptr, but additionally syncs the serialized
// region metadata (state->memory_regions) on grow / fresh-register and
// snapshots the live host bytes into initial_data, so a saved .wrp carries
// the region's current contents and extent.
// Passing a non-host-accessible pointer is undefined behavior.
APICAddress apic_resolve_host_ptr(APICState* state, uint64_t ptr, uint64_t access_size);

// ============================================================================
// APICGraph — loaded .wrp graph state
// ============================================================================

struct APICGraph {
    void* cuda_context = nullptr;
    int target_arch = 0;
    APICDeviceType device_type = APIC_DEVICE_CUDA;

    std::unordered_map<std::string, APICModule> modules;
    std::unordered_map<std::string, APICKernel> kernels;
    std::unordered_map<uint32_t, APICMemory> regions;
    std::unordered_map<std::string, uint32_t> bindings;
    std::vector<std::string> binding_names;

    std::vector<uint8_t> operation_stream;
    uint32_t operation_count = 0;

    // Set by wp_apic_load_graph after apic_validate_operation_stream passes.
    // Replay paths require it; see apic_validate_operation_stream.
    bool operations_validated = false;

    std::unordered_map<uint64_t, uint64_t> handle_ptr_remap;
    std::vector<APICMemoryPtrLocation> ptr_locations;
    std::vector<APICMeshRecord> mesh_records;
    std::vector<uint64_t> created_mesh_ids;

    std::unordered_map<std::string, APICCPUKernel> cpu_kernels;

#ifdef __CUDACC__
    CUgraph cuda_graph = nullptr;
    CUgraphExec cuda_graph_exec = nullptr;
#else
    void* cuda_graph = nullptr;  // CUgraph, opaque on non-CUDA compilers
    void* cuda_graph_exec = nullptr;  // CUgraphExec, opaque on non-CUDA compilers
#endif
    std::string base_path;

    // Destructor defined in apic.cu (CUDA builds) / apic.cpp (non-CUDA builds)
    ~APICGraph();
};

// ============================================================================
// Operation stream validation
// ============================================================================

// Walk the operation byte stream once and verify that every record header,
// variable-length payload, and op_type is within bounds. Prints a diagnostic
// and returns false on first inconsistency. Defined in apic.cpp.
bool apic_validate_operation_stream(const uint8_t* data, size_t size, uint32_t operation_count, uint32_t depth = 0);

// ============================================================================
// .wrp file reading helpers (pure C++, defined in apic.cpp)
// ============================================================================

bool apic_read_file(const char* path, std::vector<uint8_t>& data);

// .wrp parsing — pure C++ (defined in apic.cpp). Populate fields on `graph`.
bool apic_parse_metadata(const uint8_t* data, size_t size, APICGraph* graph);
bool apic_parse_operations(const uint8_t* data, size_t size, APICGraph* graph);
bool apic_parse_memory_regions(const uint8_t* data, size_t size, APICGraph* graph);

// Reconstruct mesh objects from serialized mesh records. Dispatches to
// wp_mesh_create_host or wp_mesh_create_device based on graph->device_type.
bool apic_create_meshes(APICGraph* graph);

// Free CPU-side graph resources: destroy host meshes, free host region memory.
// Defined in apic.cpp; called from the CPU branch of APICGraph's
// destructor (on both CUDA and non-CUDA builds).
void apic_destroy_cpu_graph_resources(APICGraph* graph);

// Allocate host memory for each region and initialize from memory_ptr (the
// raw bytes of the memory section of a .wrp file). Called from the CPU branch
// of wp_apic_load_graph. Returns false on allocation failure.
bool apic_init_cpu_graph_memory(APICGraph* graph, const uint8_t* memory_ptr, size_t memory_size);

// Register a CPU mesh (host pointers) with the APIC state: registers
// points/indices/(velocities) as memory regions with initial data and appends
// an APICMeshRecord. mesh_id here is the address of a wp::Mesh struct.
// Used by both the non-CUDA wp_apic_register_mesh (in apic.cpp) and the
// CUDA-build version (in apic.cu) when the descriptor lookup fails.
void apic_register_cpu_mesh(APICState* state, uint64_t mesh_id);

// Fix up handle pointers (mesh IDs etc.) in memory regions after meshes are
// recreated during graph load. Defined in apic.cpp; the CPU path is fully
// implemented there, and in CUDA builds it forwards device handles through a
// private helper defined in apic.cu.
void apic_fixup_ptr_locations(APICGraph* graph);

// ============================================================================
// CUDA helpers (declared here so apic.cpp can call them without including
// CUDA headers; defined in apic.cu and present only in CUDA builds).
// ============================================================================

#if WP_ENABLE_CUDA
// Handle-pointer fixup for a single offset within a CUDA region.
// Reads uint64_t at (base + offset), remaps via graph->handle_ptr_remap,
// writes back. Returns false on cudaMemcpy failure.
bool apic_fixup_handle_cuda(APICGraph* graph, uint8_t* base, uint64_t offset);

// D2H-snapshot auto-registered device regions (present in regions_by_id but
// missing from memory_regions) into memory_regions before serialization, so a
// saved CUDA graph carries their initial data. Called from wp_apic_state_save.
// Returns false (with an error string set) if a region's device-to-host copy
// fails, so the save is aborted rather than emit a region missing initial data.
bool apic_snapshot_device_regions(APICState* state, void* context);

// Host-to-device copy for wp_apic_set_param on CUDA graphs.
bool apic_set_param_cuda(APICGraph* graph, void* dst, const void* src, size_t size);

// Device-to-host copy for wp_apic_get_param on CUDA graphs.
bool apic_get_param_cuda(APICGraph* graph, void* dst, const void* src, size_t size);

// CUDA-side setup during wp_apic_load_graph: load CUmodules from the
// modules_dir and cudaMalloc each region, then apic_init_memory (H2D) copies
// initial data from the .wrp file into device memory. Returns false on
// failure; the caller is responsible for deleting the graph.
bool apic_load_graph_cuda_setup(
    APICGraph* graph, void* context, const std::string& modules_dir, const uint8_t* memory_ptr, size_t memory_size
);
#endif
