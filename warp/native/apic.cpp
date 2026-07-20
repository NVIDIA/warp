// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Pure-C++ APIC (API Capture) implementation. Compiled in all builds and
// owns most of APIC: recording state, memory-region / metadata registration,
// operation recording, CPU graph replay, .wrp serialization and load,
// operation-stream validation, and the public Graph API dispatchers
// (wp_apic_load_graph, wp_apic_set_param / wp_apic_get_param, etc.).
//
// CUDA-specific code lives in apic.cu and is reached via a small set of
// helpers declared in apic_internal.h under WP_ENABLE_CUDA
// (apic_load_graph_cuda_setup, apic_set_param_cuda, apic_get_param_cuda,
// apic_fixup_handle_cuda). apic.cu also owns the CUDA build of the
// APICGraph destructor, the device-mesh registration path, and
// CUDA graph rebuild / launch.

#include "warp.h"

#include "apic.h"
#include "apic_internal.h"
#include "error.h"
#include "mesh.h"

#include <algorithm>
#include <cassert>
#include <climits>
#include <cstddef>  // offsetof
#include <cstdint>  // SIZE_MAX
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

// ============================================================================
// Thread-local Recording State
// ============================================================================

static thread_local APICState* g_apic_state = nullptr;

APICState* wp_apic_create_state() { return new APICState(); }

void wp_apic_destroy_state(APICState* state) { delete state; }

void wp_apic_begin_recording(APICState* state, int is_cpu)
{
    if (state) {
        // Nested captures are not supported (matches CUDA runtime behavior).
        if (g_apic_state) {
            fprintf(stderr, "Warp error: nested APIC capture is not supported\n");
            return;
        }
        state->recording = true;
        // Remember the capture's device class so device-mismatched host/device
        // hooks can decline to record into the wrong byte stream.
        state->is_cpu = (is_cpu != 0);
        // The stream is about to be (re)built; any prior validation is stale.
        state->operations_validated = false;
        g_apic_state = state;
    }
}

void wp_apic_end_recording(APICState* state)
{
    if (state) {
        state->recording = false;
        // Validate the recorded byte stream once, at the point it becomes final.
        // Downstream replay paths gate on operations_validated and skip per-op
        // bounds checks.
        state->operations_validated = apic_validate_operation_stream(
            state->operation_stream.data(), state->operation_stream.size(), state->operation_count
        );
        g_apic_state = nullptr;
    }
}

// Host (CPU) capture hooks: return the active state only for a CPU capture, so
// a host op invoked during a CUDA APIC capture runs live instead of recording
// into the CUDA byte stream.
APICState* wp_apic_get_recording_state() { return (g_apic_state && g_apic_state->is_cpu) ? g_apic_state : nullptr; }

// Device (CUDA) capture hooks: mirror of the above for the CUDA side.
APICState* wp_apic_get_cuda_recording_state()
{
    return (g_apic_state && !g_apic_state->is_cpu) ? g_apic_state : nullptr;
}

uint32_t wp_apic_get_operation_count(APICState* state) { return state ? state->operation_count : 0; }

// ============================================================================
// Memory Region Registration
// ============================================================================

static void apic_set_region_entry(APICState* state, uint32_t region_id, uint64_t base_ptr, uint64_t size)
{
    if (state->regions_by_id.size() <= region_id)
        state->regions_by_id.resize(region_id + 1);
    auto& slot = state->regions_by_id[region_id];
    slot.base_ptr = base_ptr;
    slot.size = size;
    slot.valid = true;
}

uint32_t
wp_apic_register_memory_region_by_ptr(APICState* state, uint64_t base_ptr, uint64_t size, uint32_t element_size)
{
    if (!state)
        return 0;
    for (size_t i = 1; i < state->regions_by_id.size(); i++) {
        if (state->regions_by_id[i].valid && state->regions_by_id[i].base_ptr == base_ptr)
            return (uint32_t)i;
    }
    uint32_t region_id = state->next_region_id++;
    apic_set_region_entry(state, region_id, base_ptr, size);
    return region_id;
}

void wp_apic_register_memory_region(APICState* state, uint32_t region_id, uint64_t size, uint32_t es, const void* data)
{
    if (!state)
        return;
    APICState::RegionInfo info;
    info.region_id = region_id;
    info.size = size;
    info.element_size = es;
    if (data && size > 0) {
        info.initial_data.resize(size);
        memcpy(info.initial_data.data(), data, size);
    }
    state->memory_regions[region_id] = std::move(info);
}

static void apic_sync_serialized_host_region(APICState* state, uint32_t region_id, uint64_t base_ptr, uint64_t size)
{
    if (!state || region_id == 0)
        return;

    APICState::RegionInfo& info = state->memory_regions[region_id];
    info.region_id = region_id;
    if (info.element_size == 0)
        info.element_size = 1;
    if (size > info.size)
        info.size = size;

    if (base_ptr == 0 || size == 0)
        return;

    size_t old_size = info.initial_data.size();
    size_t new_size = static_cast<size_t>(size);
    if (new_size <= old_size)
        return;

    info.initial_data.resize(new_size);
    memcpy(
        info.initial_data.data() + old_size, reinterpret_cast<const void*>(base_ptr + old_size), new_size - old_size
    );
}

enum class APICPtrRegistrationMode {
    LiveOnly,
    SnapshotHostInitialData,
};

static APICAddress apic_resolve_ptr(APICState* state, uint64_t ptr, uint64_t access_size, APICPtrRegistrationMode mode)
{
    if (!state)
        return APICAddress {};

    // First pass: scan regions_by_id for a region containing ptr. Grow in
    // place if the access would overshoot the recorded size.
    for (size_t i = 1; i < state->regions_by_id.size(); i++) {
        auto& region = state->regions_by_id[i];
        if (!region.valid)
            continue;
        if (ptr >= region.base_ptr && ptr < region.base_ptr + region.size) {
            uint64_t needed = (ptr - region.base_ptr) + access_size;
            if (needed > region.size)
                apic_set_region_entry(state, (uint32_t)i, region.base_ptr, needed);
            if (mode == APICPtrRegistrationMode::SnapshotHostInitialData)
                apic_sync_serialized_host_region(state, (uint32_t)i, region.base_ptr, region.size);
            return APICAddress { (int32_t)i, ptr - region.base_ptr };
        }
    }

    // No existing region contains `ptr` — auto-register it with the
    // pointer itself as the region base. Subsequent ops in
    // [ptr, ptr + access_size) will resolve into this region; an op at
    // a different base creates another region.
    uint32_t region_id = state->next_region_id++;
    uint64_t size = access_size > 0 ? access_size : 1;
    apic_set_region_entry(state, region_id, ptr, size);
    if (mode == APICPtrRegistrationMode::SnapshotHostInitialData)
        apic_sync_serialized_host_region(state, region_id, ptr, size);
    return APICAddress { (int32_t)region_id, 0 };
}

APICAddress apic_resolve_live_ptr(APICState* state, uint64_t ptr, uint64_t access_size)
{
    return apic_resolve_ptr(state, ptr, access_size, APICPtrRegistrationMode::LiveOnly);
}

APICAddress apic_resolve_host_ptr(APICState* state, uint64_t ptr, uint64_t access_size)
{
    return apic_resolve_ptr(state, ptr, access_size, APICPtrRegistrationMode::SnapshotHostInitialData);
}

// Resolve a region pointer from regions_by_id.
// `access_size` is the number of bytes the caller intends to read or write
// starting at the resolved pointer; the resolver returns nullptr unless the
// range [offset, offset + access_size) fits inside the region. Pass 0 when
// only the pointer value is needed (e.g. a DATA_PTR relocation captured as
// a kernel argument — the kernel's element accesses are governed by the
// captured array shape/strides, not by this resolver).
static void* apic_resolve_state_region_ptr(APICState* state, int32_t region_id, uint64_t offset, size_t access_size)
{
    if (region_id < 0)
        return nullptr;
    if (static_cast<size_t>(region_id) >= state->regions_by_id.size())
        return nullptr;
    const auto& entry = state->regions_by_id[region_id];
    if (!entry.valid)
        return nullptr;
    uint64_t end = offset + access_size;
    if (end < offset || end > entry.size || entry.base_ptr > UINTPTR_MAX - offset)
        return nullptr;
    return reinterpret_cast<void*>(entry.base_ptr + offset);
}

// ============================================================================
// Metadata Registration (for serialization)
// ============================================================================

void wp_apic_register_module(
    APICState* state, const char* module_hash, const char* module_name, const char* bf, int arch
)
{
    if (!state || !module_hash)
        return;
    std::string hash_str(module_hash);
    if (state->modules.find(hash_str) == state->modules.end()) {
        APICModule mod;
        mod.module_hash = hash_str;
        mod.module_name = module_name ? module_name : "";
        mod.cubin_filename = bf ? bf : "";
        mod.target_arch = arch;
        state->modules[hash_str] = mod;
    }
}

void wp_apic_register_kernel(
    APICState* state,
    const char* kernel_key,
    const char* module_hash,
    const char* forward_name,
    const char* backward_name,
    int forward_smem_bytes,
    int backward_smem_bytes,
    int block_dim
)
{
    if (!state || !kernel_key)
        return;
    std::string key_str(kernel_key);
    std::string hash_str = module_hash ? module_hash : "";
    std::string map_key = apic_kernel_map_key(hash_str, key_str);
    if (state->kernels.find(map_key) == state->kernels.end()) {
        APICKernel kern;
        kern.kernel_key = key_str;
        kern.module_hash = hash_str;
        kern.forward_name = forward_name ? forward_name : "";
        kern.backward_name = backward_name ? backward_name : "";
        kern.forward_smem_bytes = forward_smem_bytes;
        kern.backward_smem_bytes = backward_smem_bytes;
        kern.block_dim = block_dim;
        state->kernels[map_key] = kern;
    }
}

void wp_apic_register_binding(APICState* state, const char* name, uint32_t region_id)
{
    if (!state || !name)
        return;
    state->bindings.push_back({ std::string(name), region_id });
}

void wp_apic_register_ptr_location(APICState* state, uint32_t region_id, uint64_t offset, uint64_t stride)
{
    if (!state)
        return;
    APICMemoryPtrLocation loc;
    loc.region_id = region_id;
    loc.offset = offset;
    loc.stride = stride;
    state->ptr_locations.push_back(loc);
}

void wp_apic_register_cpu_kernel(
    APICState* state, const char* kernel_key, const char* module_hash, void* forward_fn, void* backward_fn
)
{
    if (!state || !kernel_key)
        return;
    APICCPUKernel entry;
    entry.forward_fn = forward_fn;
    entry.backward_fn = backward_fn;
    state->cpu_kernels[apic_kernel_map_key(module_hash ? module_hash : "", kernel_key)] = entry;
}

// ============================================================================
// Operation Recording
// ============================================================================

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
)
{
    if (!state)
        return;
    size_t key_len = kernel_key ? strlen(kernel_key) : 0;
    size_t hash_len = module_hash ? strlen(module_hash) : 0;
    size_t params_size = num_params * sizeof(APICLaunchParamRecord);
    // Backward kernels carry an adjoint block of the same shape as the
    // forward block; everything else is forward-only.
    int num_adj_params = (is_forward || !adj_params) ? 0 : num_params;
    size_t adj_params_size = num_adj_params * sizeof(APICLaunchParamRecord);
    size_t relocs_count = relocs ? static_cast<size_t>(num_relocs) : 0;
    size_t relocs_size = relocs_count * sizeof(APICLaunchPtrLocation);
    size_t value_size = value_data ? static_cast<size_t>(value_data_size) : 0;
    uint32_t total_size = (uint32_t)(sizeof(APICLaunchRecord) + key_len + hash_len + params_size + adj_params_size
                                     + relocs_size + value_size);

    APICLaunchRecord rec = {};
    rec.header.op_type = APIC_OP_KERNEL_LAUNCH;
    rec.header.total_size = total_size;
    rec.ndim = ndim;
    rec.size = size;
    for (int d = 0; d < ndim && d < APIC_LAUNCH_MAX_DIMS; d++)
        rec.shape[d] = shape[d];
    rec.dim = size;
    rec.max_blocks = max_blocks;
    rec.block_dim = block_dim;
    rec.grid_stride = grid_stride;
    rec.smem_bytes = smem_bytes;
    rec.is_forward = is_forward ? 1 : 0;
    rec.cluster_dim = cluster_dim > 0 ? static_cast<uint8_t>(cluster_dim) : 1;
    rec.kernel_key_len = static_cast<uint16_t>(key_len);
    rec.module_hash_len = static_cast<uint16_t>(hash_len);
    rec.num_params = static_cast<uint16_t>(num_params);
    rec._pad2 = 0;
    rec.num_relocs = static_cast<uint32_t>(relocs_count);
    rec.value_data_size = static_cast<uint32_t>(value_size);

    state->append_bytes(&rec, sizeof(rec));
    if (key_len > 0)
        state->append_bytes(kernel_key, key_len);
    if (hash_len > 0)
        state->append_bytes(module_hash, hash_len);
    if (params_size > 0)
        state->append_bytes(params, params_size);
    if (adj_params_size > 0)
        state->append_bytes(adj_params, adj_params_size);
    if (relocs_size > 0)
        state->append_bytes(relocs, relocs_size);
    if (value_size > 0)
        state->append_bytes(value_data, value_size);
    state->operation_count++;
}

void apic_record_memcpy_d2d(
    APICState* state,
    int32_t dst_region_id,
    uint64_t dst_offset,
    int32_t src_region_id,
    uint64_t src_offset,
    uint64_t size
)
{
    if (!state)
        return;
    APICMemcpyD2DRecord rec = {};
    rec.header.op_type = APIC_OP_MEMCPY_D2D;
    rec.header.total_size = sizeof(rec);
    rec.dst_region_id = dst_region_id;
    rec.src_region_id = src_region_id;
    rec.dst_offset = dst_offset;
    rec.src_offset = src_offset;
    rec.size = size;
    state->append_bytes(&rec, sizeof(rec));
    state->operation_count++;
}

void apic_record_memset(APICState* state, int32_t region_id, uint64_t offset, uint64_t size, int32_t value)
{
    if (!state)
        return;
    APICMemsetRecord rec = {};
    rec.header.op_type = APIC_OP_MEMSET;
    rec.header.total_size = sizeof(rec);
    rec.region_id = region_id;
    rec.value = value;
    rec.offset = offset;
    rec.size = size;
    state->append_bytes(&rec, sizeof(rec));
    state->operation_count++;
}

void apic_record_memtile(
    APICState* state, int32_t region_id, uint64_t offset, uint32_t srcsize, const void* value_ptr, uint64_t count
)
{
    if (!state)
        return;
    APICMemtileRecord rec = {};
    rec.header.op_type = APIC_OP_MEMTILE;
    rec.header.total_size = static_cast<uint32_t>(sizeof(rec) + srcsize);
    rec.region_id = region_id;
    rec.srcsize = srcsize;
    rec.offset = offset;
    rec.count = count;
    state->append_bytes(&rec, sizeof(rec));
    if (srcsize > 0 && value_ptr)
        state->append_bytes(value_ptr, srcsize);
    state->operation_count++;
}

void apic_record_alloc(APICState* state, int32_t region_id, uint64_t size)
{
    if (!state)
        return;
    APICAllocRecord rec = {};
    rec.header.op_type = APIC_OP_ALLOC;
    rec.header.total_size = sizeof(rec);
    rec.region_id = region_id;
    rec.size = size;
    state->append_bytes(&rec, sizeof(rec));
    state->operation_count++;
}

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
)
{
    if (!state)
        return;
    APICScanRecord rec = {};
    rec.header.op_type = APIC_OP_SCAN;
    rec.header.total_size = sizeof(rec);
    rec.dst_region_id = dst_region_id;
    rec.src_region_id = src_region_id;
    rec.dst_offset = dst_offset;
    rec.src_offset = src_offset;
    rec.length = length;
    rec.in_stride = in_stride;
    rec.out_stride = out_stride;
    rec.type_len = type_len;
    rec.dtype = dtype;
    rec.inclusive = inclusive;
    state->append_bytes(&rec, sizeof(rec));
    state->operation_count++;
}

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
)
{
    if (!state)
        return;
    APICReductionRecord rec = {};
    rec.header.op_type = APIC_OP_REDUCTION;
    rec.header.total_size = sizeof(rec);
    rec.input_a_region_id = input_a_region_id;
    rec.input_b_region_id = input_b_region_id;
    rec.output_region_id = output_region_id;
    rec.count = count;
    rec.input_a_offset = input_a_offset;
    rec.input_b_offset = input_b_offset;
    rec.output_offset = output_offset;
    rec.input_a_stride = input_a_stride;
    rec.input_b_stride = input_b_stride;
    rec.type_len = type_len;
    rec.kind = kind;
    rec.dtype = dtype;
    state->append_bytes(&rec, sizeof(rec));
    state->operation_count++;
}

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
)
{
    if (!state)
        return;
    APICSegmentedSortRecord rec = {};
    rec.header.op_type = APIC_OP_SEGMENTED_SORT;
    rec.header.total_size = sizeof(rec);
    rec.keys_region_id = keys_region_id;
    rec.values_region_id = values_region_id;
    rec.segstart_region_id = segstart_region_id;
    rec.segend_region_id = segend_region_id;
    rec.keys_offset = keys_offset;
    rec.values_offset = values_offset;
    rec.segstart_offset = segstart_offset;
    rec.segend_offset = segend_offset;
    rec.count = count;
    rec.num_segments = num_segments;
    rec.dtype = dtype;
    state->append_bytes(&rec, sizeof(rec));
    state->operation_count++;
}

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
)
{
    if (!state)
        return;
    APICRadixSortRecord rec = {};
    rec.header.op_type = APIC_OP_RADIX_SORT;
    rec.header.total_size = sizeof(rec);
    rec.keys_region_id = keys_region_id;
    rec.values_region_id = values_region_id;
    rec.keys_offset = keys_offset;
    rec.values_offset = values_offset;
    rec.count = count;
    rec.begin_bit = begin_bit;
    rec.end_bit = end_bit;
    rec.value_size = value_size;
    rec.dtype = dtype;
    state->append_bytes(&rec, sizeof(rec));
    state->operation_count++;
}

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
)
{
    if (!state)
        return;
    APICRunlengthEncodeRecord rec = {};
    rec.header.op_type = APIC_OP_RUNLENGTH_ENCODE;
    rec.header.total_size = sizeof(rec);
    rec.values_region_id = values_region_id;
    rec.run_values_region_id = run_values_region_id;
    rec.run_lengths_region_id = run_lengths_region_id;
    rec.run_count_region_id = run_count_region_id;
    rec.values_offset = values_offset;
    rec.run_values_offset = run_values_offset;
    rec.run_lengths_offset = run_lengths_offset;
    rec.run_count_offset = run_count_offset;
    rec.value_count = value_count;
    state->append_bytes(&rec, sizeof(rec));
    state->operation_count++;
}

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
)
{
    if (!state)
        return;
    APICBsrFromTripletsRecord rec = {};
    rec.header.op_type = APIC_OP_BSR_FROM_TRIPLETS;
    rec.header.total_size = sizeof(rec);
    rec.block_size = block_size;
    rec.scalar_size_in_bytes = scalar_size_in_bytes;
    rec.row_count = row_count;
    rec.col_count = col_count;
    rec.nnz_upper_bound = nnz_upper_bound;
    rec.masked_topology = masked_topology;
    rec.scalar_zero_mask = scalar_zero_mask;
    rec.tpl_nnz_region_id = tpl_nnz_region_id;
    rec.tpl_rows_region_id = tpl_rows_region_id;
    rec.tpl_columns_region_id = tpl_columns_region_id;
    rec.tpl_values_region_id = tpl_values_region_id;
    rec.summed_block_offsets_region_id = summed_block_offsets_region_id;
    rec.summed_block_indices_region_id = summed_block_indices_region_id;
    rec.bsr_offsets_region_id = bsr_offsets_region_id;
    rec.bsr_row_counts_region_id = bsr_row_counts_region_id;
    rec.bsr_columns_region_id = bsr_columns_region_id;
    rec.bsr_nnz_region_id = bsr_nnz_region_id;
    rec.tpl_nnz_offset = tpl_nnz_offset;
    rec.tpl_rows_offset = tpl_rows_offset;
    rec.tpl_columns_offset = tpl_columns_offset;
    rec.tpl_values_offset = tpl_values_offset;
    rec.summed_block_offsets_offset = summed_block_offsets_offset;
    rec.summed_block_indices_offset = summed_block_indices_offset;
    rec.bsr_offsets_offset = bsr_offsets_offset;
    rec.bsr_row_counts_offset = bsr_row_counts_offset;
    rec.bsr_columns_offset = bsr_columns_offset;
    rec.bsr_nnz_offset = bsr_nnz_offset;
    state->append_bytes(&rec, sizeof(rec));
    state->operation_count++;
}

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
)
{
    if (!state)
        return;
    // Snapshot the padded destination's row-capacity offsets (read-only input
    // the padded transpose never writes) so replay can restore them; compact
    // destinations recompute the offsets and append no tail (GH-1587).
    uint32_t tail_bytes = (padded_capacity_offsets && padded_capacity_offset_count > 0)
        ? static_cast<uint32_t>(padded_capacity_offset_count) * static_cast<uint32_t>(sizeof(int32_t))
        : 0;
    APICBsrTransposeRecord rec = {};
    rec.header.op_type = APIC_OP_BSR_TRANSPOSE;
    rec.header.total_size = static_cast<uint32_t>(sizeof(rec)) + tail_bytes;
    rec.row_count = row_count;
    rec.col_count = col_count;
    rec.nnz_upper_bound = nnz_upper_bound;
    rec.bsr_offsets_region_id = bsr_offsets_region_id;
    rec.bsr_row_counts_region_id = bsr_row_counts_region_id;
    rec.bsr_columns_region_id = bsr_columns_region_id;
    rec.transposed_offsets_region_id = transposed_offsets_region_id;
    rec.transposed_row_counts_region_id = transposed_row_counts_region_id;
    rec.transposed_columns_region_id = transposed_columns_region_id;
    rec.block_indices_region_id = block_indices_region_id;
    rec.status_region_id = status_region_id;
    rec.bsr_offsets_offset = bsr_offsets_offset;
    rec.bsr_row_counts_offset = bsr_row_counts_offset;
    rec.bsr_columns_offset = bsr_columns_offset;
    rec.transposed_offsets_offset = transposed_offsets_offset;
    rec.transposed_row_counts_offset = transposed_row_counts_offset;
    rec.transposed_columns_offset = transposed_columns_offset;
    rec.block_indices_offset = block_indices_offset;
    rec.status_offset = status_offset;
    state->append_bytes(&rec, sizeof(rec));
    if (tail_bytes)
        state->append_bytes(padded_capacity_offsets, tail_bytes);
    state->operation_count++;
}

// ============================================================================
// Conditional / Loop branch capture (APIC_OP_IF / APIC_OP_WHILE)
// ============================================================================

struct APICBranchStart {
    APICState* owner;  // The state that issued this token; end_branch refuses any other state.
    uint64_t offset;
    uint32_t op_count;
};

struct APICBranchBody {
    std::vector<uint8_t> data;
    uint32_t op_count = 0;
};

APICBranchStart* wp_apic_begin_branch(APICState* state)
{
    if (!state)
        return nullptr;
    auto* start = new APICBranchStart();
    start->owner = state;
    start->offset = static_cast<uint64_t>(state->operation_stream.size());
    start->op_count = state->operation_count;
    return start;
}

APICBranchBody* wp_apic_end_branch(APICState* state, APICBranchStart* start)
{
    if (!state || !start)
        return nullptr;
    // Refuse tokens from a different state or whose state has been rolled
    // back beneath the captured start position — otherwise we'd underflow
    // operation_count or truncate the wrong stream.
    if (start->owner != state || state->operation_count < start->op_count
        || state->operation_stream.size() < start->offset) {
        fprintf(stderr, "APIC: Error - branch start/state mismatch\n");
        delete start;
        return nullptr;
    }
    auto* body = new APICBranchBody();
    size_t end_pos = state->operation_stream.size();
    if (end_pos > start->offset) {
        body->data.assign(state->operation_stream.begin() + start->offset, state->operation_stream.begin() + end_pos);
        state->operation_stream.resize(start->offset);
    }
    body->op_count = state->operation_count - start->op_count;
    state->operation_count = start->op_count;
    delete start;
    return body;
}

void wp_apic_free_branch_body(APICBranchBody* body) { delete body; }

void wp_apic_record_conditional(
    APICState* state, int op_type, int32_t cond_region_id, uint64_t cond_offset, APICBranchBody* a, APICBranchBody* b
)
{
    if (!state) {
        wp_apic_free_branch_body(a);
        wp_apic_free_branch_body(b);
        return;
    }
    uint64_t a_size = a ? static_cast<uint64_t>(a->data.size()) : 0;
    uint64_t b_size = b ? static_cast<uint64_t>(b->data.size()) : 0;
    uint32_t a_op_count = a ? a->op_count : 0;
    uint32_t b_op_count = b ? b->op_count : 0;
    uint64_t total = sizeof(APICCondRecord) + a_size + b_size;

    APICCondRecord rec = {};
    rec.header.op_type = static_cast<APICOpType>(op_type);
    rec.header.total_size = static_cast<uint32_t>(total);
    rec.cond_region_id = cond_region_id;
    rec.cond_offset = cond_offset;
    rec.branch_a_size = static_cast<uint32_t>(a_size);
    rec.branch_a_op_count = a_op_count;
    rec.branch_b_size = static_cast<uint32_t>(b_size);
    rec.branch_b_op_count = b_op_count;

    state->append_bytes(&rec, sizeof(rec));
    if (a_size > 0)
        state->append_bytes(a->data.data(), a_size);
    if (b_size > 0)
        state->append_bytes(b->data.data(), b_size);
    state->operation_count++;

    delete a;
    delete b;
}

// ============================================================================
// Mesh Registration
//
// Two entry points to wp_apic_register_mesh, selected at build time:
//   - Non-CUDA build: trivial forwarder below that calls apic_register_cpu_mesh.
//   - CUDA build: a version in apic.cu that tries the device-mesh descriptor
//     table first and falls back to apic_register_cpu_mesh when mesh_id is a
//     host pointer.
// apic_register_cpu_mesh (defined here) treats mesh_id as the host address of
// a wp::Mesh and copies the point / index / velocity arrays via host pointers.
// ============================================================================

void apic_register_cpu_mesh(APICState* state, uint64_t mesh_id)
{
    if (!state || mesh_id == 0)
        return;

    for (const auto& rec : state->mesh_records) {
        if (rec.original_ptr == mesh_id)
            return;
    }

    wp::Mesh* mesh_ptr = reinterpret_cast<wp::Mesh*>(mesh_id);
    if (!mesh_ptr) {
        fprintf(stderr, "APIC: Error - null mesh pointer for mesh_id 0x%llx\n", (unsigned long long)mesh_id);
        return;
    }
    wp::Mesh mesh = *mesh_ptr;

    uint64_t points_size = mesh.num_points * sizeof(wp::vec3);
    uint64_t indices_size = mesh.num_tris * (3 * sizeof(int));

    uint32_t points_region_id = wp_apic_register_memory_region_by_ptr(
        state, reinterpret_cast<uint64_t>(mesh.points.data), points_size, sizeof(wp::vec3)
    );
    wp_apic_register_memory_region(state, points_region_id, points_size, sizeof(wp::vec3), mesh.points.data);

    uint32_t indices_region_id = wp_apic_register_memory_region_by_ptr(
        state, reinterpret_cast<uint64_t>(mesh.indices.data), indices_size, sizeof(int)
    );
    wp_apic_register_memory_region(state, indices_region_id, indices_size, sizeof(int), mesh.indices.data);

    uint32_t velocities_region_id = 0;
    if (mesh.velocities.data) {
        uint64_t vel_size = mesh.num_points * sizeof(wp::vec3);
        velocities_region_id = wp_apic_register_memory_region_by_ptr(
            state, reinterpret_cast<uint64_t>(mesh.velocities.data), vel_size, sizeof(wp::vec3)
        );
        wp_apic_register_memory_region(state, velocities_region_id, vel_size, sizeof(wp::vec3), mesh.velocities.data);
    }

    APICMeshRecord rec = {};
    rec.num_points = mesh.num_points;
    rec.num_tris = mesh.num_tris;
    rec.support_winding_number = mesh.solid_angle_props ? 1 : 0;
    rec.bvh_constructor = 0;
    rec.bvh_leaf_size = 1;
    rec.points_region_id = points_region_id;
    rec.indices_region_id = indices_region_id;
    rec.velocities_region_id = velocities_region_id;
    rec.original_ptr = mesh_id;

    state->mesh_records.push_back(rec);
}

#if !WP_ENABLE_CUDA
// Non-CUDA build: mesh_id is always a host pointer (no descriptor table).
bool wp_apic_register_mesh(APICState* state, uint64_t mesh_id)
{
    apic_register_cpu_mesh(state, mesh_id);
    return true;
}
#endif  // !WP_ENABLE_CUDA

// ============================================================================
// Graph-side helpers (shared between CUDA and non-CUDA builds)
// ============================================================================

// See apic_resolve_state_region_ptr for the `access_size` contract.
void* apic_resolve_region_ptr(APICGraph* graph, int32_t region_id, uint64_t offset, size_t access_size)
{
    if (region_id < 0)
        return nullptr;
    auto it = graph->regions.find(region_id);
    if (it != graph->regions.end() && it->second.ptr) {
        uint64_t end = offset + access_size;
        if (end < offset || end > it->second.size)
            return nullptr;
        return static_cast<void*>(static_cast<uint8_t*>(it->second.ptr) + offset);
    }
    return nullptr;
}

// Free host-side mesh and region memory for a CPU graph. Called from:
//   - Non-CUDA build: APICGraph destructor (below).
//   - CUDA build: apic.cu's destructor calls this for APIC_DEVICE_CPU graphs.
void apic_destroy_cpu_graph_resources(APICGraph* graph)
{
    for (uint64_t mesh_id : graph->created_mesh_ids) {
        wp_mesh_destroy_host(mesh_id);
    }
    graph->created_mesh_ids.clear();
    for (auto& pair : graph->regions) {
        free(pair.second.ptr);
    }
}

// Allocate host memory for each region, zero it, and (if memory_ptr is
// non-null) copy initial data from the memory section of the .wrp file.
// Called from the CPU branch of wp_apic_load_graph.
bool apic_init_cpu_graph_memory(APICGraph* graph, const uint8_t* memory_ptr, size_t memory_size)
{
    for (auto& pair : graph->regions) {
        pair.second.ptr = malloc(pair.second.size);
        if (!pair.second.ptr) {
            fprintf(stderr, "APIC: Error - failed to allocate %llu bytes\n", (unsigned long long)pair.second.size);
            return false;
        }
        memset(pair.second.ptr, 0, pair.second.size);
    }

    if (!memory_ptr || memory_size < 4)
        return true;

    const uint8_t* ptr = memory_ptr;
    const uint8_t* end = memory_ptr + memory_size;
    uint32_t region_count = 0;
    memcpy(&region_count, ptr, sizeof(uint32_t));
    ptr += sizeof(uint32_t);

    for (uint32_t i = 0; i < region_count; i++) {
        if (ptr + sizeof(APICMemoryRegionRecord) > end) {
            fprintf(stderr, "APIC: Error - truncated memory region record\n");
            return false;
        }
        const APICMemoryRegionRecord* rec = reinterpret_cast<const APICMemoryRegionRecord*>(ptr);
        ptr += sizeof(APICMemoryRegionRecord);
        if (rec->has_initial_data) {
            if (ptr + rec->size > end) {
                fprintf(stderr, "APIC: Error - truncated memory region data\n");
                return false;
            }
            auto it = graph->regions.find(rec->region_id);
            if (it != graph->regions.end() && it->second.ptr) {
                memcpy(it->second.ptr, ptr, rec->size);
            }
            ptr += rec->size;
        }
    }
    return true;
}

#if !WP_ENABLE_CUDA
// Non-CUDA build: APICGraph destructor only frees host memory.
// (CUDA builds define the destructor in apic.cu so it can use cudaFree etc.)
APICGraph::~APICGraph() { apic_destroy_cpu_graph_resources(this); }
#endif

// ============================================================================
// .wrp File Reading and Parsing (pure C++, used by wp_apic_load_graph)
// ============================================================================

bool apic_read_file(const char* path, std::vector<uint8_t>& data)
{
    FILE* f = fopen(path, "rb");
    if (!f)
        return false;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    data.resize(size);
    size_t read = fread(data.data(), 1, size, f);
    fclose(f);
    return read == static_cast<size_t>(size);
}

template <typename T> static T apic_read_value(const uint8_t*& ptr)
{
    T value;
    memcpy(&value, ptr, sizeof(T));
    ptr += sizeof(T);
    return value;
}

static std::string apic_read_lp_string(const uint8_t*& ptr)
{
    uint32_t len = apic_read_value<uint32_t>(ptr);
    std::string s(reinterpret_cast<const char*>(ptr), len);
    ptr += len;
    return s;
}

bool apic_parse_metadata(const uint8_t* data, size_t size, APICGraph* graph)
{
    if (!data || size < 28)
        return false;
    const uint8_t* ptr = data;

    /*uint32_t version =*/apic_read_value<uint32_t>(ptr);
    graph->target_arch = apic_read_value<uint32_t>(ptr);
    uint32_t num_modules = apic_read_value<uint32_t>(ptr);
    uint32_t num_kernels = apic_read_value<uint32_t>(ptr);
    uint32_t num_params = apic_read_value<uint32_t>(ptr);
    uint32_t num_meshes = apic_read_value<uint32_t>(ptr);
    uint32_t num_ptr_locations = apic_read_value<uint32_t>(ptr);

    for (uint32_t i = 0; i < num_modules; i++) {
        APICModule mod;
        mod.module_hash = apic_read_lp_string(ptr);
        mod.module_name = apic_read_lp_string(ptr);
        mod.cubin_filename = apic_read_lp_string(ptr);
        mod.target_arch = apic_read_value<uint32_t>(ptr);
        graph->modules[mod.module_hash] = mod;
    }

    for (uint32_t i = 0; i < num_kernels; i++) {
        APICKernel info;
        info.kernel_key = apic_read_lp_string(ptr);
        info.module_hash = apic_read_lp_string(ptr);
        info.forward_name = apic_read_lp_string(ptr);
        info.backward_name = apic_read_lp_string(ptr);
        info.forward_smem_bytes = apic_read_value<uint32_t>(ptr);
        info.backward_smem_bytes = apic_read_value<uint32_t>(ptr);
        info.block_dim = apic_read_value<uint32_t>(ptr);
        graph->kernels[apic_kernel_map_key(info.module_hash, info.kernel_key)] = info;
    }

    for (uint32_t i = 0; i < num_params; i++) {
        std::string name = apic_read_lp_string(ptr);
        uint32_t region_id = apic_read_value<uint32_t>(ptr);
        graph->bindings[name] = region_id;
        graph->binding_names.push_back(name);
    }

    for (uint32_t i = 0; i < num_meshes; i++) {
        APICMeshRecord rec;
        memcpy(&rec, ptr, sizeof(APICMeshRecord));
        ptr += sizeof(APICMeshRecord);
        graph->mesh_records.push_back(rec);
    }

    for (uint32_t i = 0; i < num_ptr_locations; i++) {
        APICMemoryPtrLocation loc;
        loc.region_id = apic_read_value<uint32_t>(ptr);
        loc.offset = apic_read_value<uint64_t>(ptr);
        loc.stride = apic_read_value<uint64_t>(ptr);
        graph->ptr_locations.push_back(loc);
    }

    return true;
}

bool apic_parse_operations(const uint8_t* data, size_t size, APICGraph* graph)
{
    if (!data || size < 4)
        return false;
    const uint8_t* ptr = data;
    graph->operation_count = apic_read_value<uint32_t>(ptr);
    size_t stream_size = size - 4;
    if (stream_size > 0) {
        graph->operation_stream.resize(stream_size);
        memcpy(graph->operation_stream.data(), ptr, stream_size);
    }
    return true;
}

bool apic_parse_memory_regions(const uint8_t* data, size_t size, APICGraph* graph)
{
    if (!data || size < 4)
        return true;
    const uint8_t* ptr = data;
    const uint8_t* end = data + size;
    uint32_t region_count = apic_read_value<uint32_t>(ptr);

    for (uint32_t i = 0; i < region_count; i++) {
        if (ptr + sizeof(APICMemoryRegionRecord) > end)
            return false;
        const APICMemoryRegionRecord* rec = reinterpret_cast<const APICMemoryRegionRecord*>(ptr);
        ptr += sizeof(APICMemoryRegionRecord);

        if (graph->regions.find(rec->region_id) == graph->regions.end()) {
            APICMemory region;
            region.region_id = rec->region_id;
            region.size = rec->size;
            region.element_size = rec->element_size;
            graph->regions[rec->region_id] = region;
        }

        if (rec->has_initial_data) {
            ptr += rec->size;
        }
    }
    return true;
}

bool apic_create_meshes(APICGraph* graph)
{
    for (const APICMeshRecord& rec : graph->mesh_records) {
        auto points_it = graph->regions.find(rec.points_region_id);
        auto indices_it = graph->regions.find(rec.indices_region_id);

        if (points_it == graph->regions.end() || !points_it->second.ptr) {
            fprintf(stderr, "APIC: Error - mesh points region %u not found\n", rec.points_region_id);
            return false;
        }
        if (indices_it == graph->regions.end() || !indices_it->second.ptr) {
            fprintf(stderr, "APIC: Error - mesh indices region %u not found\n", rec.indices_region_id);
            return false;
        }

        wp::array_t<wp::vec3> points = {};
        points.data = static_cast<wp::vec3*>(points_it->second.ptr);
        points.shape[0] = rec.num_points;
        points.strides[0] = sizeof(wp::vec3);
        points.ndim = 1;

        wp::array_t<int> indices = {};
        indices.data = static_cast<int*>(indices_it->second.ptr);
        indices.shape[0] = rec.num_tris * 3;
        indices.strides[0] = sizeof(int);
        indices.ndim = 1;

        wp::array_t<wp::vec3> velocities = {};
        if (rec.velocities_region_id != 0) {
            auto vel_it = graph->regions.find(rec.velocities_region_id);
            if (vel_it != graph->regions.end() && vel_it->second.ptr) {
                velocities.data = static_cast<wp::vec3*>(vel_it->second.ptr);
                velocities.shape[0] = rec.num_points;
                velocities.strides[0] = sizeof(wp::vec3);
                velocities.ndim = 1;
            }
        }

        uint64_t new_mesh_id;
        if (graph->device_type == APIC_DEVICE_CPU) {
            new_mesh_id = wp_mesh_create_host(
                points, velocities, indices, rec.num_points, rec.num_tris, rec.support_winding_number,
                rec.bvh_constructor, nullptr, rec.bvh_leaf_size
            );
        } else {
            new_mesh_id = wp_mesh_create_device(
                graph->cuda_context, points, velocities, indices, rec.num_points, rec.num_tris,
                rec.support_winding_number, rec.bvh_constructor, nullptr, rec.bvh_leaf_size
            );
        }

        if (new_mesh_id == 0) {
            fprintf(stderr, "APIC: Error - failed to create mesh from serialized data\n");
            return false;
        }

        graph->created_mesh_ids.push_back(new_mesh_id);
        graph->handle_ptr_remap[rec.original_ptr] = new_mesh_id;
    }

    return true;
}

// Walk the graph's ptr_locations and remap any old handle values found at
// those offsets to their new values in graph->handle_ptr_remap. The CPU path
// is pure memcpy; the CUDA path delegates per-offset to apic_fixup_handle_cuda.
void apic_fixup_ptr_locations(APICGraph* graph)
{
    if (graph->handle_ptr_remap.empty() || graph->ptr_locations.empty())
        return;

    bool is_cpu = (graph->device_type == APIC_DEVICE_CPU);

    for (const auto& loc : graph->ptr_locations) {
        auto region_it = graph->regions.find(loc.region_id);
        if (region_it == graph->regions.end() || !region_it->second.ptr)
            continue;

        uint8_t* base = static_cast<uint8_t*>(region_it->second.ptr);
        uint64_t region_size = region_it->second.size;

        auto fixup = [&](uint64_t off) -> bool {
            if (is_cpu) {
                uint64_t old_val;
                memcpy(&old_val, base + off, sizeof(uint64_t));
                auto remap_it = graph->handle_ptr_remap.find(old_val);
                if (remap_it != graph->handle_ptr_remap.end()) {
                    uint64_t new_val = remap_it->second;
                    memcpy(base + off, &new_val, sizeof(uint64_t));
                }
                return true;
            }
#if WP_ENABLE_CUDA
            return apic_fixup_handle_cuda(graph, base, off);
#else
            (void)base;
            (void)off;
            return false;
#endif
        };

        if (loc.stride == 0) {
            if (loc.offset + sizeof(uint64_t) <= region_size)
                fixup(loc.offset);
        } else {
            for (uint64_t off = loc.offset; off + sizeof(uint64_t) <= region_size; off += loc.stride) {
                if (!fixup(off))
                    break;  // Stop this region; continue with next to avoid cascade
            }
        }
    }
}

// ============================================================================
// Operation Stream Validation
// ============================================================================

static bool apic_is_scan_dtype(uint8_t dtype)
{
    return dtype == APIC_TYPE_INT32 || dtype == APIC_TYPE_FLOAT32 || dtype == APIC_TYPE_INT64
        || dtype == APIC_TYPE_FLOAT64;
}

static bool apic_is_radix_dtype(uint8_t dtype)
{
    return dtype == APIC_TYPE_INT32 || dtype == APIC_TYPE_UINT32 || dtype == APIC_TYPE_FLOAT32
        || dtype == APIC_TYPE_INT64 || dtype == APIC_TYPE_UINT64 || dtype == APIC_TYPE_FLOAT64;
}

// Checked multiply: writes a*b to *out and returns true, or returns false if
// the product would overflow size_t. Replay computes byte spans as products of
// record fields (count, stride, nnz, block size, scalar size) and trusts the
// validator, so a corrupt/oversized .wrp could otherwise wrap a span to a small
// value that passes resolve_ptr's bounds check while the host op reads/writes
// far more. The validator uses this to reject such records up front.
static bool apic_mul_check(uint64_t a, uint64_t b, uint64_t* out)
{
    if (b != 0 && a > (SIZE_MAX / b))
        return false;
    *out = a * b;
    return true;
}

static bool apic_add_check(uint64_t a, uint64_t b, uint64_t* out)
{
    if (a > UINT64_MAX - b)
        return false;
    *out = a + b;
    return true;
}

// Walks the byte stream once and verifies every record fits in bounds, its
// variable-length fields don't overflow its declared total_size, and the
// op_type is recognized. Called once at stream-close time (end of recording
// for live capture; after .wrp load for deserialized graphs). Replay paths
// skip per-op bounds checks once this returns true.
bool apic_validate_operation_stream(const uint8_t* data, size_t size, uint32_t operation_count, uint32_t depth)
{
    // Cap conditional nesting before recursion can stack-overflow on a
    // malformed .wrp. 1024 levels is far past anything a real program will
    // ever produce and far below any platform's default stack budget.
    constexpr uint32_t kMaxBranchDepth = 1024;
    if (depth > kMaxBranchDepth) {
        fprintf(stderr, "APIC: Error - conditional nesting exceeds %u levels\n", kMaxBranchDepth);
        return false;
    }
    if (operation_count == 0)
        return true;
    if (!data) {
        fprintf(stderr, "APIC: Error - null operation stream with %u operations declared\n", operation_count);
        return false;
    }

    const uint8_t* ptr = data;
    const uint8_t* end = ptr + size;

    for (uint32_t i = 0; i < operation_count; i++) {
        if (ptr + sizeof(APICOpHeader) > end) {
            fprintf(stderr, "APIC: Error - truncated op header at operation %u\n", i);
            return false;
        }
        const APICOpHeader* header = reinterpret_cast<const APICOpHeader*>(ptr);
        const uint8_t* op_start = ptr;

        if (header->total_size < sizeof(APICOpHeader) || op_start + header->total_size > end) {
            fprintf(stderr, "APIC: Error - invalid op size %u at operation %u\n", header->total_size, i);
            return false;
        }
        const uint8_t* op_end = op_start + header->total_size;

        switch (header->op_type) {
        case APIC_OP_KERNEL_LAUNCH: {
            if (op_end < op_start + sizeof(APICLaunchRecord)) {
                fprintf(stderr, "APIC: Error - kernel launch record overflow at operation %u\n", i);
                return false;
            }
            const APICLaunchRecord* rec = reinterpret_cast<const APICLaunchRecord*>(ptr);
            const uint8_t* var_data = ptr + sizeof(APICLaunchRecord);
            if (var_data + rec->kernel_key_len + rec->module_hash_len > op_end) {
                fprintf(stderr, "APIC: Error - kernel launch strings overflow at operation %u\n", i);
                return false;
            }
            const uint8_t* params_start = var_data + rec->kernel_key_len + rec->module_hash_len;
            if (params_start + rec->num_params * sizeof(APICLaunchParamRecord) > op_end) {
                fprintf(stderr, "APIC: Error - kernel launch params overflow at operation %u\n", i);
                return false;
            }
            const uint8_t* adj_params_start = params_start + rec->num_params * sizeof(APICLaunchParamRecord);
            // Backward kernels carry an adjoint block of num_params bindings;
            // forward kernels carry none.
            const uint32_t adj_count = rec->is_forward ? 0u : rec->num_params;
            if (adj_params_start + adj_count * sizeof(APICLaunchParamRecord) > op_end) {
                fprintf(stderr, "APIC: Error - kernel launch adj params overflow at operation %u\n", i);
                return false;
            }
            const uint8_t* relocs_start = adj_params_start + adj_count * sizeof(APICLaunchParamRecord);
            if (relocs_start + rec->num_relocs * sizeof(APICLaunchPtrLocation) > op_end) {
                fprintf(stderr, "APIC: Error - kernel launch relocs overflow at operation %u\n", i);
                return false;
            }
            const uint8_t* value_data = relocs_start + rec->num_relocs * sizeof(APICLaunchPtrLocation);
            if (value_data + rec->value_data_size > op_end) {
                fprintf(stderr, "APIC: Error - kernel launch value_data overflow at operation %u\n", i);
                return false;
            }
            // Walk the bindings (forward followed by adjoint), bounds-check each
            // binding's value slice and sum its num_relocs. The reloc cursor is
            // advanced per-binding in apic_pack_args_buf at replay; total
            // num_relocs must exactly equal the sum across all bindings.
            uint32_t reloc_cursor = 0;
            const APICLaunchPtrLocation* relocs_table = reinterpret_cast<const APICLaunchPtrLocation*>(relocs_start);
            uint32_t total_bindings = static_cast<uint32_t>(rec->num_params) + adj_count;
            for (uint32_t j = 0; j < total_bindings; j++) {
                const APICLaunchParamRecord* binding;
                if (j < rec->num_params) {
                    binding = reinterpret_cast<const APICLaunchParamRecord*>(
                        params_start + j * sizeof(APICLaunchParamRecord)
                    );
                } else {
                    binding = reinterpret_cast<const APICLaunchParamRecord*>(
                        adj_params_start + (j - rec->num_params) * sizeof(APICLaunchParamRecord)
                    );
                }
                if (binding->value_offset > rec->value_data_size
                    || binding->value_size > rec->value_data_size - binding->value_offset) {
                    fprintf(stderr, "APIC: Error - value_data offset out of range at operation %u (param %u)\n", i, j);
                    return false;
                }
                // Each reloc must point at a valid 8-byte slot inside the blob.
                for (uint16_t r = 0; r < binding->num_relocs; r++) {
                    if (reloc_cursor + r >= rec->num_relocs) {
                        fprintf(stderr, "APIC: Error - reloc count exceeds table at operation %u (param %u)\n", i, j);
                        return false;
                    }
                    const APICLaunchPtrLocation* reloc = &relocs_table[reloc_cursor + r];
                    if (reloc->kind != APIC_RELOC_DATA_PTR && reloc->kind != APIC_RELOC_HANDLE
                        && reloc->kind != APIC_RELOC_NULL) {
                        fprintf(
                            stderr, "APIC: Error - unknown reloc kind %u at operation %u (param %u reloc %u)\n",
                            reloc->kind, i, j, r
                        );
                        return false;
                    }
                    if (reloc->value_byte_offset + sizeof(uint64_t) > binding->value_size) {
                        fprintf(
                            stderr,
                            "APIC: Error - reloc target out of value blob at operation %u (param %u reloc %u)\n", i, j,
                            r
                        );
                        return false;
                    }
                }
                reloc_cursor += binding->num_relocs;
            }
            if (reloc_cursor != rec->num_relocs) {
                fprintf(
                    stderr, "APIC: Error - reloc table size mismatch at operation %u (sum=%u, header=%u)\n", i,
                    reloc_cursor, rec->num_relocs
                );
                return false;
            }
            break;
        }
        case APIC_OP_MEMCPY_H2D: {
            if (op_end < op_start + sizeof(APICMemcpyH2DRecord)) {
                fprintf(stderr, "APIC: Error - H2D memcpy record overflow at operation %u\n", i);
                return false;
            }
            const APICMemcpyH2DRecord* rec = reinterpret_cast<const APICMemcpyH2DRecord*>(ptr);
            if (op_start + sizeof(APICMemcpyH2DRecord) + rec->size > op_end) {
                fprintf(stderr, "APIC: Error - H2D memcpy inline data overflow at operation %u\n", i);
                return false;
            }
            break;
        }
        case APIC_OP_MEMCPY_D2D:
            if (op_end < op_start + sizeof(APICMemcpyD2DRecord)) {
                fprintf(stderr, "APIC: Error - D2D memcpy record overflow at operation %u\n", i);
                return false;
            }
            break;
        case APIC_OP_MEMSET:
            if (op_end < op_start + sizeof(APICMemsetRecord)) {
                fprintf(stderr, "APIC: Error - memset record overflow at operation %u\n", i);
                return false;
            }
            break;
        case APIC_OP_MEMTILE: {
            if (op_end < op_start + sizeof(APICMemtileRecord)) {
                fprintf(stderr, "APIC: Error - memtile record overflow at operation %u\n", i);
                return false;
            }
            const APICMemtileRecord* rec = reinterpret_cast<const APICMemtileRecord*>(ptr);
            if (op_start + sizeof(APICMemtileRecord) + rec->srcsize > op_end) {
                fprintf(stderr, "APIC: Error - memtile inline data overflow at operation %u\n", i);
                return false;
            }
            break;
        }
        case APIC_OP_ALLOC:
            if (op_end < op_start + sizeof(APICAllocRecord)) {
                fprintf(stderr, "APIC: Error - alloc record overflow at operation %u\n", i);
                return false;
            }
            break;
        case APIC_OP_SCAN: {
            if (op_end < op_start + sizeof(APICScanRecord)) {
                fprintf(stderr, "APIC: Error - scan record overflow at operation %u\n", i);
                return false;
            }
            const APICScanRecord* rec = reinterpret_cast<const APICScanRecord*>(ptr);
            if (!apic_is_scan_dtype(rec->dtype)) {
                fprintf(stderr, "APIC: Error - unknown scan dtype %u at operation %u\n", rec->dtype, i);
                return false;
            }
            if (rec->inclusive > 1) {
                fprintf(stderr, "APIC: Error - scan inclusive flag out of range at operation %u\n", i);
                return false;
            }
            if (rec->length > 0 && (rec->in_stride < 0 || rec->out_stride < 0 || rec->type_len <= 0)) {
                fprintf(stderr, "APIC: Error - invalid scan strides/type length at operation %u\n", i);
                return false;
            }
            {
                // Replay accesses (length-1)*stride + type_len*scalar_size bytes.
                uint64_t scan_stride
                    = static_cast<uint64_t>(rec->in_stride > rec->out_stride ? rec->in_stride : rec->out_stride);
                uint64_t span_tmp;
                if (rec->length > 0
                    && (!apic_mul_check(static_cast<uint64_t>(rec->length) - 1, scan_stride, &span_tmp)
                        || !apic_mul_check(
                            static_cast<uint64_t>(rec->type_len), apic_type_size(rec->dtype), &span_tmp
                        ))) {
                    fprintf(stderr, "APIC: Error - scan span overflow at operation %u\n", i);
                    return false;
                }
            }
            break;
        }
        case APIC_OP_REDUCTION: {
            if (header->total_size != sizeof(APICReductionRecord) || op_end < op_start + sizeof(APICReductionRecord)) {
                fprintf(stderr, "APIC: Error - reduction record size invalid at operation %u\n", i);
                return false;
            }
            const APICReductionRecord* rec = reinterpret_cast<const APICReductionRecord*>(ptr);
            bool is_sum = rec->kind == APIC_REDUCTION_SUM;
            bool is_inner = rec->kind == APIC_REDUCTION_INNER;
            if ((!is_sum && !is_inner) || (rec->dtype != APIC_TYPE_FLOAT32 && rec->dtype != APIC_TYPE_FLOAT64)
                || rec->count == 0 || rec->count > static_cast<uint32_t>(INT_MAX) || rec->type_len <= 0
                || rec->input_a_stride < 0 || rec->input_a_region_id < 0 || rec->output_region_id < 0
                || (is_sum && (rec->input_b_region_id != -1 || rec->input_b_offset != 0 || rec->input_b_stride != 0))
                || (is_inner && (rec->input_b_region_id < 0 || rec->input_b_stride < 0))) {
                fprintf(stderr, "APIC: Error - invalid reduction metadata at operation %u\n", i);
                return false;
            }

            uint64_t scalar_size = apic_type_size(rec->dtype);
            if (rec->input_a_offset % scalar_size != 0 || rec->output_offset % scalar_size != 0
                || rec->input_a_stride % scalar_size != 0
                || (is_inner && (rec->input_b_offset % scalar_size != 0 || rec->input_b_stride % scalar_size != 0))) {
                fprintf(stderr, "APIC: Error - misaligned reduction metadata at operation %u\n", i);
                return false;
            }
            uint64_t input_a_bytes
                = apic_reduction_input_bytes(rec->count, rec->input_a_stride, rec->type_len, scalar_size);
            uint64_t input_b_bytes = is_inner
                ? apic_reduction_input_bytes(rec->count, rec->input_b_stride, rec->type_len, scalar_size)
                : 0;
            uint64_t output_bytes = is_sum ? static_cast<uint64_t>(rec->type_len) * scalar_size : scalar_size;
            uint64_t span_end;
            if (input_a_bytes == 0 || output_bytes == 0 || (is_inner && input_b_bytes == 0)
                || !apic_add_check(rec->input_a_offset, input_a_bytes, &span_end)
                || !apic_add_check(rec->output_offset, output_bytes, &span_end)
                || (is_inner && !apic_add_check(rec->input_b_offset, input_b_bytes, &span_end))) {
                fprintf(stderr, "APIC: Error - reduction span overflow at operation %u\n", i);
                return false;
            }
            break;
        }
        case APIC_OP_SEGMENTED_SORT: {
            if (op_end < op_start + sizeof(APICSegmentedSortRecord)) {
                fprintf(stderr, "APIC: Error - segmented-sort record overflow at operation %u\n", i);
                return false;
            }
            const APICSegmentedSortRecord* rec = reinterpret_cast<const APICSegmentedSortRecord*>(ptr);
            if (rec->dtype != APIC_TYPE_INT32 && rec->dtype != APIC_TYPE_FLOAT32) {
                fprintf(stderr, "APIC: Error - unknown segmented-sort dtype %u at operation %u\n", rec->dtype, i);
                return false;
            }
            {
                // Replay accesses 2*count keys/values and num_segments+1 start indices.
                uint64_t span_tmp;
                if (!apic_mul_check(static_cast<uint64_t>(rec->count) * 2u, sizeof(uint32_t), &span_tmp)
                    || !apic_mul_check(static_cast<uint64_t>(rec->num_segments) + 1u, sizeof(int32_t), &span_tmp)) {
                    fprintf(stderr, "APIC: Error - segmented-sort span overflow at operation %u\n", i);
                    return false;
                }
            }
            break;
        }
        case APIC_OP_RADIX_SORT: {
            if (op_end < op_start + sizeof(APICRadixSortRecord)) {
                fprintf(stderr, "APIC: Error - radix-sort record overflow at operation %u\n", i);
                return false;
            }
            const APICRadixSortRecord* rec = reinterpret_cast<const APICRadixSortRecord*>(ptr);
            if (!apic_is_radix_dtype(rec->dtype)) {
                fprintf(stderr, "APIC: Error - unknown radix-sort dtype %u at operation %u\n", rec->dtype, i);
                return false;
            }
            int32_t key_bits = static_cast<int32_t>(apic_type_size(rec->dtype) * 8);
            if ((rec->value_size != 4 && rec->value_size != 8) || rec->begin_bit < 0 || rec->end_bit < rec->begin_bit
                || rec->end_bit > key_bits) {
                fprintf(stderr, "APIC: Error - invalid radix-sort metadata at operation %u\n", i);
                return false;
            }
            {
                // Replay accesses 2*count keys (key_size) and 2*count values (value_size).
                uint64_t span_tmp;
                if (!apic_mul_check(static_cast<uint64_t>(rec->count) * 2u, apic_type_size(rec->dtype), &span_tmp)
                    || !apic_mul_check(
                        static_cast<uint64_t>(rec->count) * 2u, static_cast<uint64_t>(rec->value_size), &span_tmp
                    )) {
                    fprintf(stderr, "APIC: Error - radix-sort span overflow at operation %u\n", i);
                    return false;
                }
            }
            break;
        }
        case APIC_OP_RUNLENGTH_ENCODE: {
            if (op_end < op_start + sizeof(APICRunlengthEncodeRecord)) {
                fprintf(stderr, "APIC: Error - runlength-encode record overflow at operation %u\n", i);
                return false;
            }
            {
                const APICRunlengthEncodeRecord* rec = reinterpret_cast<const APICRunlengthEncodeRecord*>(ptr);
                uint64_t span_tmp;
                if (!apic_mul_check(static_cast<uint64_t>(rec->value_count), sizeof(int32_t), &span_tmp)) {
                    fprintf(stderr, "APIC: Error - runlength-encode span overflow at operation %u\n", i);
                    return false;
                }
            }
            break;
        }
        case APIC_OP_BSR_FROM_TRIPLETS: {
            if (op_end < op_start + sizeof(APICBsrFromTripletsRecord)) {
                fprintf(stderr, "APIC: Error - bsr-from-triplets record overflow at operation %u\n", i);
                return false;
            }
            {
                const APICBsrFromTripletsRecord* rec = reinterpret_cast<const APICBsrFromTripletsRecord*>(ptr);
                // Replay accesses nnz*int32 topology and nnz*block_size*scalar_size block data;
                // the three int32 factors can overflow size_t for a corrupt record.
                uint64_t span_tmp;
                if (rec->nnz_upper_bound < 0 || rec->row_count < 0 || rec->block_size < 0
                    || rec->scalar_size_in_bytes < 0
                    || !apic_mul_check(static_cast<uint64_t>(rec->row_count) + 1u, sizeof(int32_t), &span_tmp)
                    || !apic_mul_check(
                        static_cast<uint64_t>(rec->nnz_upper_bound), static_cast<uint64_t>(rec->block_size), &span_tmp
                    )
                    || !apic_mul_check(span_tmp, static_cast<uint64_t>(rec->scalar_size_in_bytes), &span_tmp)) {
                    fprintf(stderr, "APIC: Error - bsr-from-triplets span overflow at operation %u\n", i);
                    return false;
                }
            }
            break;
        }
        case APIC_OP_BSR_TRANSPOSE: {
            if (op_end < op_start + sizeof(APICBsrTransposeRecord)) {
                fprintf(stderr, "APIC: Error - bsr-transpose record overflow at operation %u\n", i);
                return false;
            }
            {
                const APICBsrTransposeRecord* rec = reinterpret_cast<const APICBsrTransposeRecord*>(ptr);
                uint64_t span_tmp;
                if (rec->nnz_upper_bound < 0 || rec->row_count < 0 || rec->col_count < 0
                    || !apic_mul_check(static_cast<uint64_t>(rec->nnz_upper_bound), sizeof(int32_t), &span_tmp)
                    || !apic_mul_check(static_cast<uint64_t>(rec->row_count) + 1u, sizeof(int32_t), &span_tmp)
                    || !apic_mul_check(static_cast<uint64_t>(rec->col_count) + 1u, sizeof(int32_t), &span_tmp)) {
                    fprintf(stderr, "APIC: Error - bsr-transpose span overflow at operation %u\n", i);
                    return false;
                }
            }
            break;
        }
        case APIC_OP_IF:
        case APIC_OP_WHILE: {
            if (op_end < op_start + sizeof(APICCondRecord)) {
                fprintf(stderr, "APIC: Error - cond record overflow at operation %u\n", i);
                return false;
            }
            const APICCondRecord* rec = reinterpret_cast<const APICCondRecord*>(ptr);
            if (sizeof(APICCondRecord) + (uint64_t)rec->branch_a_size + (uint64_t)rec->branch_b_size
                > header->total_size) {
                fprintf(stderr, "APIC: Error - cond branch sizes overflow at operation %u\n", i);
                return false;
            }
            const uint8_t* branch_a = op_start + sizeof(APICCondRecord);
            const uint8_t* branch_b = branch_a + rec->branch_a_size;
            // Recurse into the inner sub-streams.
            if (rec->branch_a_size > 0
                && !apic_validate_operation_stream(branch_a, rec->branch_a_size, rec->branch_a_op_count, depth + 1)) {
                fprintf(stderr, "APIC: Error - branch_a invalid at operation %u\n", i);
                return false;
            }
            if (header->op_type == APIC_OP_WHILE && rec->branch_b_size != 0) {
                fprintf(stderr, "APIC: Error - APIC_OP_WHILE with non-empty branch_b at operation %u\n", i);
                return false;
            }
            if (rec->branch_b_size > 0
                && !apic_validate_operation_stream(branch_b, rec->branch_b_size, rec->branch_b_op_count, depth + 1)) {
                fprintf(stderr, "APIC: Error - branch_b invalid at operation %u\n", i);
                return false;
            }
            break;
        }
        default:
            fprintf(stderr, "APIC: Error - unknown op type %u at operation %u\n", unsigned(header->op_type), i);
            return false;
        }

        ptr = op_end;
    }

    if (ptr != end) {
        fprintf(stderr, "APIC: Warning - %td trailing bytes in operation stream\n", end - ptr);
    }

    return true;
}

// ============================================================================
// CPU Graph Replay
// ============================================================================

// Pack the param array `bindings` into `args_buf` by copying each binding's
// value blob from `value_data + value_offset` and patching the pointer fields
// it contains via the per-binding relocation slice. Caller pre-allocates
// `args_buf` to the size returned by apic_args_buf_size. Returns false (and
// emits a diagnostic) if a DATA_PTR relocation cannot be resolved, so the
// caller can abort replay rather than launching with a NULL pointer arg.
//
// `relocs` points at the slice of the per-launch reloc table corresponding
// to this binding group (forward or adjoint). Within the loop the cursor
// advances by `binding->num_relocs` per binding.
//
// `resolve_ptr(region_id, region_offset, access_size) -> void*` resolves
// DATA_PTR relocations to a live host (or device) pointer. `access_size` is
// the number of bytes the caller will read/write at the resolved pointer;
// pass 0 for "pointer-only" use (e.g. capturing the .data field of an
// array_t — the kernel handles element-level access). `remap_handle(uint64)
// -> uint64` translates HANDLE relocations through the load-time handle
// remap table (identity for live recording, real lookup for loaded graphs).
template <typename ResolvePtrFn, typename RemapHandleFn>
static bool apic_pack_args_buf(
    uint8_t* args_buf,
    size_t args_buf_size,
    const APICLaunchParamRecord* bindings,
    uint16_t num_bindings,
    const uint8_t* value_data,
    const APICLaunchPtrLocation* relocs,
    ResolvePtrFn resolve_ptr,
    RemapHandleFn remap_handle
)
{
    memset(args_buf, 0, args_buf_size);
    size_t args_offset = 0;
    const APICLaunchPtrLocation* reloc_cursor = relocs;
    for (uint16_t j = 0; j < num_bindings; j++) {
        const APICLaunchParamRecord* binding = &bindings[j];
        size_t align = binding->value_align > 0 ? static_cast<size_t>(binding->value_align) : alignof(void*);
        args_offset = (args_offset + align - 1) & ~(align - 1);

        // Copy the captured value bytes verbatim.
        memcpy(args_buf + args_offset, value_data + binding->value_offset, binding->value_size);

        // Patch pointer fields inside the blob.
        for (uint16_t r = 0; r < binding->num_relocs; r++) {
            const APICLaunchPtrLocation* reloc = &reloc_cursor[r];
            uint64_t patched = 0;
            switch (reloc->kind) {
            case APIC_RELOC_DATA_PTR: {
                // Pass access_size=0: we only need the pointer value to drop
                // into the value blob; the kernel's element access is bounded
                // by the captured array shape/strides, not by this resolver.
                void* resolved = resolve_ptr(reloc->region_id, reloc->region_offset, 0);
                if (!resolved) {
                    fprintf(
                        stderr,
                        "APIC: Error - unresolved DATA_PTR relocation (region=%d offset=%llu) at param %u reloc %u\n",
                        reloc->region_id, static_cast<unsigned long long>(reloc->region_offset), j, r
                    );
                    return false;
                }
                patched = reinterpret_cast<uint64_t>(resolved);
                break;
            }
            case APIC_RELOC_HANDLE:
                patched = remap_handle(reloc->region_offset);
                break;
            case APIC_RELOC_NULL:
                patched = 0;
                break;
            }
            memcpy(args_buf + args_offset + reloc->value_byte_offset, &patched, sizeof(uint64_t));
        }
        reloc_cursor += binding->num_relocs;
        args_offset += binding->value_size;
    }
    // Catches divergence between apic_args_buf_size (used to size args_buf)
    // and the alignment / sizing rules in the loop above.
    assert(args_offset <= args_buf_size);
    return true;
}

static size_t apic_args_buf_size(const APICLaunchParamRecord* bindings, uint16_t num_bindings)
{
    size_t total = 0;
    for (uint16_t j = 0; j < num_bindings; j++) {
        const APICLaunchParamRecord* binding = &bindings[j];
        size_t align = binding->value_align > 0 ? static_cast<size_t>(binding->value_align) : alignof(void*);
        total = (total + align - 1) & ~(align - 1);
        total += binding->value_size;
    }
    return total;
}

// Walk an APIC byte stream and execute CPU operations. Assumes the stream
// has already passed apic_validate_operation_stream — no per-op bounds
// checks are performed here.
// resolve_ptr:   (int32_t region_id, uint64_t offset) -> void*
// remap_handle:  (uint64_t old_id) -> uint64_t (identity for live recording,
//                                                load-time remap for loaded graphs)
// find_kernel:   (const std::string& key, uint8_t is_forward) -> void*
template <typename ResolvePtrFn, typename RemapHandleFn, typename FindKernelFn>
static bool apic_cpu_replay_stream(
    const uint8_t* stream_data,
    size_t stream_size,
    uint32_t operation_count,
    ResolvePtrFn resolve_ptr,
    RemapHandleFn remap_handle,
    FindKernelFn find_kernel
)
{
    // Stream was validated at close time (wp_apic_end_recording or
    // wp_apic_load_graph) via apic_validate_operation_stream — no per-op
    // bounds checks needed here.
    const uint8_t* ptr = stream_data;

    // Grow-only heap scratch reused across all kernel-launch ops in this stream,
    // replacing per-op malloc/free of the args buffers. Heap-backed (not stack)
    // to keep the stack shallow for the kernel's own tile storage; reused to
    // avoid per-launch allocator churn on launch-dense graphs (e.g. diffsim_bear
    // replays ~10^5 launches per iteration).
    std::vector<uint8_t> fwd_scratch;
    std::vector<uint8_t> adj_scratch;

    for (uint32_t i = 0; i < operation_count; i++) {
        const APICOpHeader* header = reinterpret_cast<const APICOpHeader*>(ptr);

        switch (header->op_type) {
        case APIC_OP_KERNEL_LAUNCH: {
            const APICLaunchRecord* rec = reinterpret_cast<const APICLaunchRecord*>(ptr);
            const uint8_t* var_data = ptr + sizeof(APICLaunchRecord);
            std::string key_str(reinterpret_cast<const char*>(var_data), rec->kernel_key_len);
            // The module hash recorded alongside the key disambiguates same-key
            // kernels compiled into distinct modules (see apic_kernel_map_key).
            std::string module_hash_str(
                reinterpret_cast<const char*>(var_data + rec->kernel_key_len), rec->module_hash_len
            );

            const uint16_t adj_count = rec->is_forward ? 0u : rec->num_params;
            const APICLaunchParamRecord* fwd_bindings
                = reinterpret_cast<const APICLaunchParamRecord*>(var_data + rec->kernel_key_len + rec->module_hash_len);
            const APICLaunchParamRecord* adj_bindings = reinterpret_cast<const APICLaunchParamRecord*>(
                reinterpret_cast<const uint8_t*>(fwd_bindings) + rec->num_params * sizeof(APICLaunchParamRecord)
            );
            const APICLaunchPtrLocation* relocs = reinterpret_cast<const APICLaunchPtrLocation*>(
                reinterpret_cast<const uint8_t*>(adj_bindings) + adj_count * sizeof(APICLaunchParamRecord)
            );
            const uint8_t* value_data
                = reinterpret_cast<const uint8_t*>(relocs) + rec->num_relocs * sizeof(APICLaunchPtrLocation);

            // Sum forward num_relocs so the adjoint pack call can start its
            // reloc cursor in the right place. The validator guarantees
            // fwd_relocs + adj_relocs == rec->num_relocs.
            uint32_t fwd_reloc_count = 0;
            for (uint16_t j = 0; j < rec->num_params; j++)
                fwd_reloc_count += fwd_bindings[j].num_relocs;

            void* func = find_kernel(key_str, module_hash_str, rec->is_forward);
            if (!func) {
                fprintf(stderr, "APIC: Error - CPU kernel not found: %s\n", key_str.c_str());
                return false;
            }

            // Build launch_bounds_t<N> with the per-N layout expected by the
            // generated kernel entry point.
            int ndim = rec->ndim;
            if (ndim < 1)
                ndim = 1;
            if (ndim > APIC_LAUNCH_MAX_DIMS)
                ndim = APIC_LAUNCH_MAX_DIMS;

            size_t size_offset = apic_detail::launch_bounds_size_offset(ndim);
            size_t coord_mult_offset = apic_detail::launch_bounds_coord_mult_offset(ndim);
            size_t bounds_size = apic_detail::launch_bounds_storage_size(ndim);

            constexpr size_t bounds_capacity = apic_detail::launch_bounds_storage_size(APIC_LAUNCH_MAX_DIMS);
            alignas(wp::launch_bounds_t<APIC_LAUNCH_MAX_DIMS>) uint8_t bounds_buf[bounds_capacity];
            memset(bounds_buf, 0, bounds_size);
            uint64_t shape_size = 1;
            for (int d = 0; d < ndim; d++)
                reinterpret_cast<int*>(bounds_buf)[d] = rec->shape[d];
            for (int d = 0; d < ndim; d++)
                shape_size *= static_cast<uint64_t>(rec->shape[d]);
            *reinterpret_cast<size_t*>(bounds_buf + size_offset) = rec->size;
            size_t coord_mult = 1;
            if (shape_size > 0 && rec->size > shape_size) {
                uint64_t mult = rec->size / shape_size;
                coord_mult = static_cast<size_t>(mult);
            }
            *reinterpret_cast<size_t*>(bounds_buf + coord_mult_offset) = coord_mult;

            // Build forward args buffer in the reused heap scratch (grow-only).
            // Heap-backed (not stack) to keep the stack shallow before calling
            // into the kernel — the kernel itself may allocate a 256 KB
            // tile_shared_storage_t on its own stack frame.
            size_t fwd_total = apic_args_buf_size(fwd_bindings, rec->num_params);
            size_t fwd_need = fwd_total > 0 ? fwd_total : size_t(1);
            if (fwd_scratch.size() < fwd_need)
                fwd_scratch.resize(fwd_need);
            uint8_t* fwd_buf = fwd_scratch.data();
            if (!apic_pack_args_buf(
                    fwd_buf, fwd_total, fwd_bindings, rec->num_params, value_data, relocs, resolve_ptr, remap_handle
                )) {
                fprintf(stderr, "APIC: Error - forward arg packing failed at operation %u\n", i);
                return false;
            }

            // Build adjoint args buffer when this is a backward kernel.
            // wp_cpu_launch_kernel selects the backward ABI by adj_args != nullptr,
            // so a zero-param backward kernel still needs a (1-byte sentinel) adj_buf.
            size_t adj_total = 0;
            uint8_t* adj_buf = nullptr;
            if (!rec->is_forward) {
                adj_total = apic_args_buf_size(adj_bindings, adj_count);
                size_t adj_need = adj_total > 0 ? adj_total : size_t(1);
                if (adj_scratch.size() < adj_need)
                    adj_scratch.resize(adj_need);
                adj_buf = adj_scratch.data();
                if (adj_total > 0
                    && !apic_pack_args_buf(
                        adj_buf, adj_total, adj_bindings, adj_count, value_data, relocs + fwd_reloc_count, resolve_ptr,
                        remap_handle
                    )) {
                    fprintf(stderr, "APIC: Error - adjoint arg packing failed at operation %u\n", i);
                    return false;
                }
            }

            // Replay via the same wp_cpu_launch_kernel that captured this op.
            // apic_info=nullptr is safe: g_apic_state is null during replay, so
            // the recording branch in wp_cpu_launch_kernel is a no-op and the
            // execute branch fires.
            wp_cpu_launch_kernel(func, bounds_buf, fwd_buf, adj_buf, /*apic_info=*/nullptr);
            break;
        }

        case APIC_OP_MEMCPY_D2D: {
            const APICMemcpyD2DRecord* rec = reinterpret_cast<const APICMemcpyD2DRecord*>(ptr);
            void* dst = resolve_ptr(rec->dst_region_id, rec->dst_offset, rec->size);
            const void* src = resolve_ptr(rec->src_region_id, rec->src_offset, rec->size);
            if (!dst || !src) {
                fprintf(stderr, "APIC: Error - memcpy pointer resolution failed at operation %u\n", i);
                return false;
            }
            // "D2D" on CPU is host-to-host — replay via wp_memcpy_h2h.
            if (!wp_memcpy_h2h(dst, const_cast<void*>(src), rec->size))
                return false;
            break;
        }

        case APIC_OP_MEMSET: {
            const APICMemsetRecord* rec = reinterpret_cast<const APICMemsetRecord*>(ptr);
            void* dst = resolve_ptr(rec->region_id, rec->offset, rec->size);
            if (!dst) {
                fprintf(stderr, "APIC: Error - memset pointer resolution failed at operation %u\n", i);
                return false;
            }
            if (!wp_memset_host(dst, rec->value, rec->size))
                return false;
            break;
        }

        case APIC_OP_MEMTILE: {
            const APICMemtileRecord* rec = reinterpret_cast<const APICMemtileRecord*>(ptr);
            const void* value = ptr + sizeof(APICMemtileRecord);
            uint64_t total_bytes = rec->count * rec->srcsize;
            void* dst = resolve_ptr(rec->region_id, rec->offset, total_bytes);
            if (!dst) {
                fprintf(stderr, "APIC: Error - memtile pointer resolution failed at operation %u\n", i);
                return false;
            }
            // g_apic_state is null during replay, so this call executes
            // the actual memtile instead of re-recording it.
            wp_memtile_host(dst, value, rec->srcsize, rec->count);
            break;
        }

        case APIC_OP_ALLOC:
            break;

        case APIC_OP_SCAN: {
            const APICScanRecord* rec = reinterpret_cast<const APICScanRecord*>(ptr);
            size_t scalar_size = apic_type_size(rec->dtype);
            size_t src_bytes = apic_strided_access_bytes(rec->length, rec->in_stride, rec->type_len, scalar_size);
            size_t dst_bytes = apic_strided_access_bytes(rec->length, rec->out_stride, rec->type_len, scalar_size);
            void* dst = resolve_ptr(rec->dst_region_id, rec->dst_offset, dst_bytes);
            const void* src = resolve_ptr(rec->src_region_id, rec->src_offset, src_bytes);
            if (!dst || !src) {
                fprintf(stderr, "APIC: Error - scan pointer resolution failed at operation %u\n", i);
                return false;
            }
            // g_apic_state is null during replay, so these calls execute
            // (not record). Same entry points the user-facing
            // wp.utils.array_scan() dispatches to.
            if (rec->dtype == APIC_TYPE_INT32) {
                wp_array_scan_int_host(
                    reinterpret_cast<uint64_t>(src), reinterpret_cast<uint64_t>(dst), static_cast<int>(rec->length),
                    rec->in_stride, rec->out_stride, rec->type_len, rec->inclusive != 0
                );
            } else if (rec->dtype == APIC_TYPE_INT64) {
                wp_array_scan_int64_host(
                    reinterpret_cast<uint64_t>(src), reinterpret_cast<uint64_t>(dst), static_cast<int>(rec->length),
                    rec->in_stride, rec->out_stride, rec->type_len, rec->inclusive != 0
                );
            } else if (rec->dtype == APIC_TYPE_FLOAT32) {
                wp_array_scan_float_host(
                    reinterpret_cast<uint64_t>(src), reinterpret_cast<uint64_t>(dst), static_cast<int>(rec->length),
                    rec->in_stride, rec->out_stride, rec->type_len, rec->inclusive != 0
                );
            } else {
                wp_array_scan_double_host(
                    reinterpret_cast<uint64_t>(src), reinterpret_cast<uint64_t>(dst), static_cast<int>(rec->length),
                    rec->in_stride, rec->out_stride, rec->type_len, rec->inclusive != 0
                );
            }
            break;
        }

        case APIC_OP_REDUCTION: {
            const APICReductionRecord* rec = reinterpret_cast<const APICReductionRecord*>(ptr);
            size_t scalar_size = apic_type_size(rec->dtype);
            size_t input_a_bytes
                = apic_reduction_input_bytes(rec->count, rec->input_a_stride, rec->type_len, scalar_size);
            size_t input_b_bytes = rec->kind == APIC_REDUCTION_INNER
                ? apic_reduction_input_bytes(rec->count, rec->input_b_stride, rec->type_len, scalar_size)
                : 0;
            size_t output_bytes
                = rec->kind == APIC_REDUCTION_SUM ? static_cast<size_t>(rec->type_len) * scalar_size : scalar_size;

            const void* input_a = resolve_ptr(rec->input_a_region_id, rec->input_a_offset, input_a_bytes);
            const void* input_b = rec->kind == APIC_REDUCTION_INNER
                ? resolve_ptr(rec->input_b_region_id, rec->input_b_offset, input_b_bytes)
                : nullptr;
            void* output = resolve_ptr(rec->output_region_id, rec->output_offset, output_bytes);
            if (!input_a || !output || (rec->kind == APIC_REDUCTION_INNER && !input_b)) {
                fprintf(stderr, "APIC: Error - reduction pointer resolution failed at operation %u\n", i);
                return false;
            }

            if (rec->kind == APIC_REDUCTION_SUM) {
                if (rec->dtype == APIC_TYPE_FLOAT32)
                    wp_array_sum_float_host(
                        reinterpret_cast<uint64_t>(input_a), reinterpret_cast<uint64_t>(output),
                        static_cast<int>(rec->count), rec->input_a_stride, rec->type_len
                    );
                else
                    wp_array_sum_double_host(
                        reinterpret_cast<uint64_t>(input_a), reinterpret_cast<uint64_t>(output),
                        static_cast<int>(rec->count), rec->input_a_stride, rec->type_len
                    );
            } else {
                if (rec->dtype == APIC_TYPE_FLOAT32)
                    wp_array_inner_float_host(
                        reinterpret_cast<uint64_t>(input_a), reinterpret_cast<uint64_t>(input_b),
                        reinterpret_cast<uint64_t>(output), static_cast<int>(rec->count), rec->input_a_stride,
                        rec->input_b_stride, rec->type_len
                    );
                else
                    wp_array_inner_double_host(
                        reinterpret_cast<uint64_t>(input_a), reinterpret_cast<uint64_t>(input_b),
                        reinterpret_cast<uint64_t>(output), static_cast<int>(rec->count), rec->input_a_stride,
                        rec->input_b_stride, rec->type_len
                    );
            }
            break;
        }

        case APIC_OP_SEGMENTED_SORT: {
            const APICSegmentedSortRecord* rec = reinterpret_cast<const APICSegmentedSortRecord*>(ptr);
            size_t key_size = (rec->dtype == APIC_TYPE_INT32 ? sizeof(int32_t) : sizeof(float));
            // keys/values buffers span 2*count elements (sort scratch).
            size_t keys_bytes = static_cast<size_t>(2) * rec->count * key_size;
            size_t values_bytes = static_cast<size_t>(2) * rec->count * sizeof(int32_t);
            // Inferred-end captures alias segment_end into the start array (same
            // region), so the start region spans num_segments+1 entries; explicit-end
            // captures use two separate num_segments-entry arrays (distinct regions).
            // Match the span recorded at capture.
            bool segments_inferred_end = (rec->segstart_region_id == rec->segend_region_id);
            size_t segstart_bytes
                = (static_cast<size_t>(rec->num_segments) + (segments_inferred_end ? 1 : 0)) * sizeof(int32_t);
            size_t segend_bytes = static_cast<size_t>(rec->num_segments) * sizeof(int32_t);
            void* keys = resolve_ptr(rec->keys_region_id, rec->keys_offset, keys_bytes);
            void* values = resolve_ptr(rec->values_region_id, rec->values_offset, values_bytes);
            void* segstart = resolve_ptr(rec->segstart_region_id, rec->segstart_offset, segstart_bytes);
            void* segend = resolve_ptr(rec->segend_region_id, rec->segend_offset, segend_bytes);
            if (!keys || !values || !segstart || !segend) {
                fprintf(stderr, "APIC: Error - segmented-sort pointer resolution failed at operation %u\n", i);
                return false;
            }
            // g_apic_state is null during replay, so these calls execute the
            // real sort instead of re-recording. Same entry points the
            // user-facing wp.utils.segmented_sort_pairs() dispatches to.
            if (rec->dtype == APIC_TYPE_INT32) {
                wp_segmented_sort_pairs_int_host(
                    reinterpret_cast<uint64_t>(keys), reinterpret_cast<uint64_t>(values), static_cast<int>(rec->count),
                    reinterpret_cast<uint64_t>(segstart), reinterpret_cast<uint64_t>(segend),
                    static_cast<int>(rec->num_segments)
                );
            } else {
                wp_segmented_sort_pairs_float_host(
                    reinterpret_cast<uint64_t>(keys), reinterpret_cast<uint64_t>(values), static_cast<int>(rec->count),
                    reinterpret_cast<uint64_t>(segstart), reinterpret_cast<uint64_t>(segend),
                    static_cast<int>(rec->num_segments)
                );
            }
            break;
        }

        case APIC_OP_RADIX_SORT: {
            const APICRadixSortRecord* rec = reinterpret_cast<const APICRadixSortRecord*>(ptr);
            size_t key_size = apic_type_size(rec->dtype);
            size_t keys_bytes = static_cast<size_t>(2) * rec->count * key_size;
            size_t values_bytes = static_cast<size_t>(2) * rec->count * static_cast<size_t>(rec->value_size);
            void* keys = resolve_ptr(rec->keys_region_id, rec->keys_offset, keys_bytes);
            void* values = resolve_ptr(rec->values_region_id, rec->values_offset, values_bytes);
            if (!keys || !values) {
                fprintf(stderr, "APIC: Error - radix-sort pointer resolution failed at operation %u\n", i);
                return false;
            }
            // g_apic_state is null during replay, so these calls execute the
            // real sort. Same entry points wp.utils.radix_sort_pairs() uses.
            if (rec->dtype == APIC_TYPE_INT32) {
                wp_radix_sort_pairs_int_host(
                    reinterpret_cast<uint64_t>(keys), reinterpret_cast<uint64_t>(values), static_cast<int>(rec->count),
                    rec->begin_bit, rec->end_bit, rec->value_size
                );
            } else if (rec->dtype == APIC_TYPE_UINT32) {
                wp_radix_sort_pairs_uint_host(
                    reinterpret_cast<uint64_t>(keys), reinterpret_cast<uint64_t>(values), static_cast<int>(rec->count),
                    rec->begin_bit, rec->end_bit, rec->value_size
                );
            } else if (rec->dtype == APIC_TYPE_INT64) {
                wp_radix_sort_pairs_int64_host(
                    reinterpret_cast<uint64_t>(keys), reinterpret_cast<uint64_t>(values), static_cast<int>(rec->count),
                    rec->begin_bit, rec->end_bit, rec->value_size
                );
            } else if (rec->dtype == APIC_TYPE_UINT64) {
                wp_radix_sort_pairs_uint64_host(
                    reinterpret_cast<uint64_t>(keys), reinterpret_cast<uint64_t>(values), static_cast<int>(rec->count),
                    rec->begin_bit, rec->end_bit, rec->value_size
                );
            } else if (rec->dtype == APIC_TYPE_FLOAT32) {
                wp_radix_sort_pairs_float_host(
                    reinterpret_cast<uint64_t>(keys), reinterpret_cast<uint64_t>(values), static_cast<int>(rec->count),
                    rec->begin_bit, rec->end_bit, rec->value_size
                );
            } else {
                wp_radix_sort_pairs_double_host(
                    reinterpret_cast<uint64_t>(keys), reinterpret_cast<uint64_t>(values), static_cast<int>(rec->count),
                    rec->begin_bit, rec->end_bit, rec->value_size
                );
            }
            break;
        }

        case APIC_OP_RUNLENGTH_ENCODE: {
            const APICRunlengthEncodeRecord* rec = reinterpret_cast<const APICRunlengthEncodeRecord*>(ptr);
            size_t n = rec->value_count;
            size_t in_bytes = n * sizeof(int32_t);
            void* values = resolve_ptr(rec->values_region_id, rec->values_offset, in_bytes);
            // run_values / run_lengths hold up to value_count entries.
            void* run_values = resolve_ptr(rec->run_values_region_id, rec->run_values_offset, in_bytes);
            void* run_lengths = resolve_ptr(rec->run_lengths_region_id, rec->run_lengths_offset, in_bytes);
            void* run_count = resolve_ptr(rec->run_count_region_id, rec->run_count_offset, sizeof(int32_t));
            if (!values || !run_values || !run_lengths || !run_count) {
                fprintf(stderr, "APIC: Error - runlength-encode pointer resolution failed at operation %u\n", i);
                return false;
            }
            // g_apic_state is null during replay, so this executes the real
            // encode. Same entry point wp.utils.runlength_encode() uses.
            wp_runlength_encode_int_host(
                reinterpret_cast<uint64_t>(values), reinterpret_cast<uint64_t>(run_values),
                reinterpret_cast<uint64_t>(run_lengths), reinterpret_cast<uint64_t>(run_count),
                static_cast<int>(rec->value_count)
            );
            break;
        }

        case APIC_OP_BSR_FROM_TRIPLETS: {
            const APICBsrFromTripletsRecord* rec = reinterpret_cast<const APICBsrFromTripletsRecord*>(ptr);
            size_t nnz = static_cast<size_t>(rec->nnz_upper_bound);
            size_t int_bytes = nnz * sizeof(int32_t);
            size_t rowp1_bytes = (static_cast<size_t>(rec->row_count) + 1) * sizeof(int32_t);
            size_t values_bytes
                = nnz * static_cast<size_t>(rec->block_size) * static_cast<size_t>(rec->scalar_size_in_bytes);

            void* tpl_nnz = rec->tpl_nnz_region_id >= 0
                ? resolve_ptr(rec->tpl_nnz_region_id, rec->tpl_nnz_offset, sizeof(int32_t))
                : nullptr;
            void* tpl_rows = resolve_ptr(rec->tpl_rows_region_id, rec->tpl_rows_offset, int_bytes);
            void* tpl_columns = resolve_ptr(rec->tpl_columns_region_id, rec->tpl_columns_offset, int_bytes);
            void* tpl_values = rec->tpl_values_region_id >= 0
                ? resolve_ptr(rec->tpl_values_region_id, rec->tpl_values_offset, values_bytes)
                : nullptr;
            void* summed_block_offsets
                = resolve_ptr(rec->summed_block_offsets_region_id, rec->summed_block_offsets_offset, int_bytes);
            void* summed_block_indices
                = resolve_ptr(rec->summed_block_indices_region_id, rec->summed_block_indices_offset, int_bytes);
            void* bsr_offsets = resolve_ptr(rec->bsr_offsets_region_id, rec->bsr_offsets_offset, rowp1_bytes);
            void* bsr_row_counts = rec->bsr_row_counts_region_id >= 0
                ? resolve_ptr(
                      rec->bsr_row_counts_region_id, rec->bsr_row_counts_offset,
                      static_cast<size_t>(rec->row_count) * sizeof(int32_t)
                  )
                : nullptr;
            size_t bsr_columns_bytes = int_bytes;
            if (bsr_offsets && rec->masked_topology != 0)
                bsr_columns_bytes = std::max(
                    bsr_columns_bytes,
                    static_cast<size_t>(reinterpret_cast<const int*>(bsr_offsets)[rec->row_count]) * sizeof(int32_t)
                );
            void* bsr_columns = resolve_ptr(rec->bsr_columns_region_id, rec->bsr_columns_offset, bsr_columns_bytes);
            void* bsr_nnz = rec->bsr_nnz_region_id >= 0
                ? resolve_ptr(rec->bsr_nnz_region_id, rec->bsr_nnz_offset, sizeof(int32_t))
                : nullptr;

            if (!tpl_rows || !tpl_columns || !summed_block_offsets || !summed_block_indices || !bsr_offsets
                || !bsr_columns || (rec->tpl_nnz_region_id >= 0 && !tpl_nnz)
                || (rec->tpl_values_region_id >= 0 && !tpl_values)
                || (rec->bsr_row_counts_region_id >= 0 && !bsr_row_counts)
                || (rec->bsr_nnz_region_id >= 0 && !bsr_nnz)) {
                fprintf(stderr, "APIC: Error - bsr-from-triplets pointer resolution failed at operation %u\n", i);
                return false;
            }
            // g_apic_state is null during replay, so this executes the real
            // topology build. Same entry point wp.sparse.bsr_set_from_triplets uses.
            wp_bsr_matrix_from_triplets_host(
                rec->block_size, rec->scalar_size_in_bytes, rec->row_count, rec->col_count, rec->nnz_upper_bound,
                reinterpret_cast<const int*>(tpl_nnz), reinterpret_cast<const int*>(tpl_rows),
                reinterpret_cast<const int*>(tpl_columns), tpl_values, rec->scalar_zero_mask, rec->masked_topology != 0,
                reinterpret_cast<int*>(summed_block_offsets), reinterpret_cast<int*>(summed_block_indices),
                reinterpret_cast<int*>(bsr_offsets), reinterpret_cast<const int*>(bsr_row_counts),
                reinterpret_cast<int*>(bsr_columns), reinterpret_cast<int*>(bsr_nnz), nullptr
            );
            break;
        }

        case APIC_OP_BSR_TRANSPOSE: {
            const APICBsrTransposeRecord* rec = reinterpret_cast<const APICBsrTransposeRecord*>(ptr);
            size_t nnz = static_cast<size_t>(rec->nnz_upper_bound);
            size_t int_bytes = nnz * sizeof(int32_t);
            size_t rowp1_bytes = (static_cast<size_t>(rec->row_count) + 1) * sizeof(int32_t);
            size_t colp1_bytes = (static_cast<size_t>(rec->col_count) + 1) * sizeof(int32_t);
            void* bsr_offsets = resolve_ptr(rec->bsr_offsets_region_id, rec->bsr_offsets_offset, rowp1_bytes);
            void* bsr_row_counts = rec->bsr_row_counts_region_id >= 0
                ? resolve_ptr(
                      rec->bsr_row_counts_region_id, rec->bsr_row_counts_offset,
                      static_cast<size_t>(rec->row_count) * sizeof(int32_t)
                  )
                : nullptr;
            void* bsr_columns = resolve_ptr(rec->bsr_columns_region_id, rec->bsr_columns_offset, int_bytes);
            void* t_offsets
                = resolve_ptr(rec->transposed_offsets_region_id, rec->transposed_offsets_offset, colp1_bytes);
            void* t_row_counts = rec->transposed_row_counts_region_id >= 0
                ? resolve_ptr(
                      rec->transposed_row_counts_region_id, rec->transposed_row_counts_offset,
                      static_cast<size_t>(rec->col_count) * sizeof(int32_t)
                  )
                : nullptr;
            if (!bsr_offsets || !bsr_columns || !t_offsets || (rec->bsr_row_counts_region_id >= 0 && !bsr_row_counts)
                || (rec->transposed_row_counts_region_id >= 0 && !t_row_counts)) {
                fprintf(stderr, "APIC: Error - bsr-transpose pointer resolution failed at operation %u\n", i);
                return false;
            }
            // Restore the padded destination's row-capacity offsets recorded at
            // capture time, so replay reconstructs the destination even if the
            // caller reset its offsets buffer before capture_launch. Compact
            // transposes recompute the offsets and carry no tail (GH-1587).
            // This must happen before sizing the destination spans below: the
            // recorded tail, not the (possibly reset) live offsets, is the
            // authoritative capacity layout.
            size_t transposed_capacity_bytes = int_bytes;
            if (rec->transposed_row_counts_region_id >= 0) {
                size_t tail_bytes = static_cast<size_t>(rec->header.total_size) - sizeof(APICBsrTransposeRecord);
                if (tail_bytes >= colp1_bytes) {
                    memcpy(t_offsets, ptr + sizeof(APICBsrTransposeRecord), colp1_bytes);
                    // The transpose touches the destination columns only within
                    // the padded capacity (offsets[col_count] blocks), which may
                    // be smaller than the source's nnz upper bound. Match the
                    // span claimed at capture so resolution is bounds-checked
                    // against the destination's real extent.
                    int32_t capacity = reinterpret_cast<const int32_t*>(t_offsets)[rec->col_count];
                    transposed_capacity_bytes = capacity > 0 ? static_cast<size_t>(capacity) * sizeof(int32_t) : 0;
                }
            }
            void* t_columns = resolve_ptr(
                rec->transposed_columns_region_id, rec->transposed_columns_offset, transposed_capacity_bytes
            );
            // block_indices carries the sorted source blocks (up to nnz entries)
            // and, for padded destinations, per-slot gap markers up to the
            // destination capacity.
            void* block_indices = resolve_ptr(
                rec->block_indices_region_id, rec->block_indices_offset, std::max(int_bytes, transposed_capacity_bytes)
            );
            void* status = rec->status_region_id >= 0
                ? resolve_ptr(rec->status_region_id, rec->status_offset, sizeof(int32_t))
                : nullptr;
            if (!t_columns || !block_indices || (rec->status_region_id >= 0 && !status)) {
                fprintf(stderr, "APIC: Error - bsr-transpose pointer resolution failed at operation %u\n", i);
                return false;
            }
            // g_apic_state is null during replay, so this executes the real
            // transpose. Same entry point wp.sparse.bsr_set_transpose uses.
            wp_bsr_transpose_host(
                rec->row_count, rec->col_count, rec->nnz_upper_bound, reinterpret_cast<const int*>(bsr_offsets),
                reinterpret_cast<const int*>(bsr_row_counts), reinterpret_cast<const int*>(bsr_columns),
                reinterpret_cast<int*>(t_offsets), reinterpret_cast<int*>(t_row_counts),
                reinterpret_cast<int*>(t_columns), reinterpret_cast<int*>(block_indices), reinterpret_cast<int*>(status)
            );
            break;
        }

        case APIC_OP_IF: {
            const APICCondRecord* rec = reinterpret_cast<const APICCondRecord*>(ptr);
            const uint8_t* branch_a = ptr + sizeof(APICCondRecord);
            const uint8_t* branch_b = branch_a + rec->branch_a_size;

            // Read condition value (int32) from the captured region. The
            // resolver verifies the full int32_t fits at cond_offset, so a
            // malformed .wrp cannot drive an out-of-bounds read here.
            void* cond_ptr = resolve_ptr(rec->cond_region_id, rec->cond_offset, sizeof(int32_t));
            if (!cond_ptr) {
                fprintf(stderr, "APIC: Error - cond pointer resolution failed at operation %u\n", i);
                return false;
            }
            int32_t cond_value = *reinterpret_cast<const int32_t*>(cond_ptr);

            if (cond_value && rec->branch_a_size > 0) {
                if (!apic_cpu_replay_stream(
                        branch_a, rec->branch_a_size, rec->branch_a_op_count, resolve_ptr, remap_handle, find_kernel
                    ))
                    return false;
            } else if (!cond_value && rec->branch_b_size > 0) {
                if (!apic_cpu_replay_stream(
                        branch_b, rec->branch_b_size, rec->branch_b_op_count, resolve_ptr, remap_handle, find_kernel
                    ))
                    return false;
            }
            break;
        }

        case APIC_OP_WHILE: {
            const APICCondRecord* rec = reinterpret_cast<const APICCondRecord*>(ptr);
            const uint8_t* body = ptr + sizeof(APICCondRecord);

            void* cond_ptr = resolve_ptr(rec->cond_region_id, rec->cond_offset, sizeof(int32_t));
            if (!cond_ptr) {
                fprintf(stderr, "APIC: Error - cond pointer resolution failed at operation %u\n", i);
                return false;
            }
            // The body's first kernel is responsible for updating the
            // condition int32; we re-read it after each iteration.
            uint32_t guard = 0;
            const uint32_t guard_limit = 1u << 24;  // sanity bound to break runaway loops
            while (*reinterpret_cast<const volatile int32_t*>(cond_ptr)) {
                if (rec->branch_a_size == 0)
                    break;
                if (!apic_cpu_replay_stream(
                        body, rec->branch_a_size, rec->branch_a_op_count, resolve_ptr, remap_handle, find_kernel
                    ))
                    return false;
                if (++guard >= guard_limit) {
                    fprintf(stderr, "APIC: Error - APIC_OP_WHILE exceeded guard limit at operation %u\n", i);
                    return false;
                }
            }
            break;
        }

        default:
            fprintf(stderr, "APIC: Error - unsupported CPU replay op type %u\n", unsigned(header->op_type));
            break;
        }

        ptr += header->total_size;
    }

    return true;
}

// Shared body for the two public CPU replay entry points. Container is either
// APICState or APICGraph; both expose operation_stream,
// operation_count, and cpu_kernels (map<string, APICCPUKernel>). The only
// per-container difference is how region_id is resolved to a host pointer,
// which the caller supplies as `resolve`.
template <typename Container, typename Resolver, typename HandleRemap>
static bool apic_cpu_replay_container(Container* c, Resolver resolve, HandleRemap remap_handle)
{
    auto find_kernel = [c](const std::string& key, const std::string& module_hash, uint8_t is_forward) -> void* {
        // Prefer the exact (module_hash, key) match so same-key kernels from
        // distinct modules are dispatched correctly. The plain-key fallback is
        // retained for older in-memory states/graphs registered before module
        // hash disambiguation was available.
        auto it = c->cpu_kernels.find(apic_kernel_map_key(module_hash, key));
        if (it == c->cpu_kernels.end())
            it = c->cpu_kernels.find(key);
        if (it == c->cpu_kernels.end())
            return nullptr;
        return is_forward ? it->second.forward_fn : it->second.backward_fn;
    };
    return apic_cpu_replay_stream(
        c->operation_stream.data(), c->operation_stream.size(), c->operation_count, resolve, remap_handle, find_kernel
    );
}

bool wp_apic_cpu_replay_state(APICState* state)
{
    if (!state)
        return false;
    if (state->operation_stream.empty())
        return true;
    if (!state->operations_validated) {
        fprintf(stderr, "APIC: Error - replay called on unvalidated state; was wp_apic_end_recording called?\n");
        return false;
    }
    // Live recording: handles captured this session are still valid, so the
    // remap is identity.
    return apic_cpu_replay_container(
        state,
        [state](int32_t region_id, uint64_t offset, size_t access_size) {
            return apic_resolve_state_region_ptr(state, region_id, offset, access_size);
        },
        [](uint64_t handle) -> uint64_t { return handle; }
    );
}

bool wp_apic_cpu_replay_graph(APICGraph* graph)
{
    if (!graph)
        return false;
    if (graph->operation_stream.empty())
        return true;
    // Precondition: graph->operations_validated is true. Enforced by
    // wp_apic_load_graph, which is the only path that produces an APICGraph.
    return apic_cpu_replay_container(
        graph,
        [graph](int32_t region_id, uint64_t offset, size_t access_size) {
            return apic_resolve_region_ptr(graph, region_id, offset, access_size);
        },
        [graph](uint64_t handle) -> uint64_t {
            auto it = graph->handle_ptr_remap.find(handle);
            return (it != graph->handle_ptr_remap.end()) ? it->second : handle;
        }
    );
}

// ============================================================================
// Serialization: wp_apic_state_save
// ============================================================================

// Append a trivially-copyable value to the buffer (not naturally aligned).
template <typename T> static void apic_write_nc(std::vector<uint8_t>& buf, T val)
{
    size_t off = buf.size();
    buf.resize(off + sizeof(T));
    memcpy(buf.data() + off, &val, sizeof(T));
}

static void apic_write_string_nc(std::vector<uint8_t>& buf, const std::string& s)
{
    apic_write_nc<uint32_t>(buf, static_cast<uint32_t>(s.size()));
    if (!s.empty()) {
        size_t off = buf.size();
        buf.resize(off + s.size());
        memcpy(buf.data() + off, s.data(), s.size());
    }
}

bool wp_apic_state_save(APICState* state, const char* path, int target_arch, void* context)
{
    if (!state || !path) {
        fprintf(stderr, "APIC: Null %s passed to wp_apic_state_save\n", !state ? "state" : "path");
        return false;
    }

#if WP_ENABLE_CUDA
    // Snapshot device regions auto-registered by native hooks (and thus absent
    // from Python's capture_save snapshot of apic_capture._regions) so a saved
    // CUDA graph carries their initial data. CPU saves (target_arch == 0) skip this.
    // Abort the save if a snapshot fails rather than emit a region missing its data.
    if (target_arch != 0 && !apic_snapshot_device_regions(state, context))
        return false;  // error string set by apic_snapshot_device_regions
#else
    (void)context;
#endif

    // Build metadata
    std::vector<uint8_t> metadata_section;
    apic_write_nc<uint32_t>(metadata_section, APIC_FORMAT_VERSION);
    apic_write_nc<uint32_t>(metadata_section, target_arch);
    apic_write_nc<uint32_t>(metadata_section, static_cast<uint32_t>(state->modules.size()));
    apic_write_nc<uint32_t>(metadata_section, static_cast<uint32_t>(state->kernels.size()));
    apic_write_nc<uint32_t>(metadata_section, static_cast<uint32_t>(state->bindings.size()));
    apic_write_nc<uint32_t>(metadata_section, static_cast<uint32_t>(state->mesh_records.size()));
    apic_write_nc<uint32_t>(metadata_section, static_cast<uint32_t>(state->ptr_locations.size()));

    for (const auto& kv : state->modules) {
        const APICModule& m = kv.second;
        apic_write_string_nc(metadata_section, m.module_hash);
        apic_write_string_nc(metadata_section, m.module_name);
        apic_write_string_nc(metadata_section, m.cubin_filename);
        apic_write_nc<uint32_t>(metadata_section, static_cast<uint32_t>(m.target_arch));
    }
    for (const auto& kv : state->kernels) {
        const APICKernel& k = kv.second;
        apic_write_string_nc(metadata_section, k.kernel_key);
        apic_write_string_nc(metadata_section, k.module_hash);
        apic_write_string_nc(metadata_section, k.forward_name);
        apic_write_string_nc(metadata_section, k.backward_name);
        apic_write_nc<uint32_t>(metadata_section, static_cast<uint32_t>(k.forward_smem_bytes));
        apic_write_nc<uint32_t>(metadata_section, static_cast<uint32_t>(k.backward_smem_bytes));
        apic_write_nc<uint32_t>(metadata_section, static_cast<uint32_t>(k.block_dim));
    }
    for (const auto& b : state->bindings) {
        apic_write_string_nc(metadata_section, b.first);
        apic_write_nc<uint32_t>(metadata_section, b.second);
    }
    for (const auto& rec : state->mesh_records) {
        size_t off = metadata_section.size();
        metadata_section.resize(off + sizeof(APICMeshRecord));
        memcpy(metadata_section.data() + off, &rec, sizeof(APICMeshRecord));
    }
    for (const auto& loc : state->ptr_locations) {
        apic_write_nc<uint32_t>(metadata_section, loc.region_id);
        apic_write_nc<uint64_t>(metadata_section, loc.offset);
        apic_write_nc<uint64_t>(metadata_section, loc.stride);
    }

    // Build memory section
    std::vector<uint8_t> memory_section;
    uint32_t region_count = static_cast<uint32_t>(state->memory_regions.size());
    memory_section.resize(4);
    memcpy(memory_section.data(), &region_count, 4);
    for (const auto& kv : state->memory_regions) {
        const APICState::RegionInfo& region = kv.second;
        APICMemoryRegionRecord rec = {};
        rec.region_id = region.region_id;
        rec.element_size = region.element_size;
        rec.size = region.size;
        rec.has_initial_data = region.initial_data.empty() ? 0 : 1;
        size_t offset = memory_section.size();
        size_t data_size = rec.has_initial_data ? region.initial_data.size() : 0;
        memory_section.resize(offset + sizeof(rec) + data_size);
        memcpy(memory_section.data() + offset, &rec, sizeof(rec));
        if (rec.has_initial_data)
            memcpy(
                memory_section.data() + offset + sizeof(rec), region.initial_data.data(), region.initial_data.size()
            );
    }

    // Build operations section
    std::vector<uint8_t> ops_section;
    ops_section.resize(4 + state->operation_stream.size());
    memcpy(ops_section.data(), &state->operation_count, 4);
    if (!state->operation_stream.empty())
        memcpy(ops_section.data() + 4, state->operation_stream.data(), state->operation_stream.size());

    // Write file
    FILE* f = fopen(path, "wb");
    if (!f)
        return false;

    const uint32_t HEADER_SIZE = 64;
    const uint32_t SECTION_ENTRY_SIZE = 32;
    uint32_t num_sections = 3;
    uint64_t section_table_offset = HEADER_SIZE;
    uint64_t data_offset = section_table_offset + num_sections * SECTION_ENTRY_SIZE;
    uint64_t metadata_offset = data_offset;
    uint64_t memory_offset = metadata_offset + metadata_section.size();
    uint64_t operations_offset = memory_offset + memory_section.size();

    APICFileHeader header = {};
    header.magic[0] = 'W';
    header.magic[1] = 'R';
    header.magic[2] = 'P';
    header.magic[3] = '1';
    header.version = APIC_FORMAT_VERSION;
    header.num_sections = num_sections;
    header.section_table_offset = section_table_offset;
    header.target_arch = target_arch;
    header.device_type = (target_arch == 0) ? APIC_DEVICE_CPU : APIC_DEVICE_CUDA;

    if (fwrite(&header, sizeof(header), 1, f) != 1) {
        fclose(f);
        return false;
    }

    APICSectionEntry entries[3] = {};
    entries[0].type = APIC_SECTION_METADATA;
    entries[0].offset = metadata_offset;
    entries[0].size = entries[0].uncompressed_size = metadata_section.size();
    entries[1].type = APIC_SECTION_MEMORY;
    entries[1].offset = memory_offset;
    entries[1].size = entries[1].uncompressed_size = memory_section.size();
    entries[2].type = APIC_SECTION_OPERATIONS;
    entries[2].offset = operations_offset;
    entries[2].size = entries[2].uncompressed_size = ops_section.size();

    if (fwrite(entries, sizeof(APICSectionEntry), 3, f) != 3) {
        fclose(f);
        return false;
    }
    if (!metadata_section.empty()
        && fwrite(metadata_section.data(), 1, metadata_section.size(), f) != metadata_section.size()) {
        fclose(f);
        return false;
    }
    if (!memory_section.empty()
        && fwrite(memory_section.data(), 1, memory_section.size(), f) != memory_section.size()) {
        fclose(f);
        return false;
    }
    if (!ops_section.empty() && fwrite(ops_section.data(), 1, ops_section.size(), f) != ops_section.size()) {
        fclose(f);
        return false;
    }

    fclose(f);
    return true;
}

// ============================================================================
// Graph API — pure-C++ implementations (work on both CPU and CUDA graphs)
// ============================================================================

void wp_apic_destroy_graph(APICGraph* graph) { delete graph; }

int wp_apic_get_num_params(APICGraph* graph) { return graph ? static_cast<int>(graph->binding_names.size()) : 0; }

const char* wp_apic_get_param_name(APICGraph* graph, int index)
{
    if (!graph || index < 0 || index >= static_cast<int>(graph->binding_names.size()))
        return nullptr;
    return graph->binding_names[index].c_str();
}

size_t wp_apic_get_param_size(APICGraph* graph, const char* name)
{
    if (!graph || !name)
        return 0;
    auto it = graph->bindings.find(name);
    if (it == graph->bindings.end())
        return 0;
    auto region_it = graph->regions.find(it->second);
    if (region_it == graph->regions.end())
        return 0;
    return region_it->second.size;
}

void* wp_apic_get_param_ptr(APICGraph* graph, const char* name)
{
    if (!graph || !name)
        return nullptr;
    auto param_it = graph->bindings.find(name);
    if (param_it == graph->bindings.end())
        return nullptr;
    auto region_it = graph->regions.find(param_it->second);
    if (region_it == graph->regions.end())
        return nullptr;
    return region_it->second.ptr;
}

int wp_apic_get_num_kernels(APICGraph* graph) { return graph ? static_cast<int>(graph->kernels.size()) : 0; }

static const APICKernel* apic_get_kernel_by_index(APICGraph* graph, int index)
{
    if (!graph || index < 0 || index >= static_cast<int>(graph->kernels.size()))
        return nullptr;
    auto it = graph->kernels.begin();
    std::advance(it, index);
    return &it->second;
}

const char* wp_apic_get_kernel_key(APICGraph* graph, int index)
{
    const APICKernel* kernel = apic_get_kernel_by_index(graph, index);
    return kernel ? kernel->kernel_key.c_str() : nullptr;
}

const char* wp_apic_get_kernel_module_hash(APICGraph* graph, int index)
{
    const APICKernel* kernel = apic_get_kernel_by_index(graph, index);
    return kernel ? kernel->module_hash.c_str() : nullptr;
}

const char* wp_apic_get_kernel_module_binary_filename(APICGraph* graph, int index)
{
    const APICKernel* kernel = apic_get_kernel_by_index(graph, index);
    if (!kernel)
        return nullptr;

    auto it = graph->modules.find(kernel->module_hash);
    if (it == graph->modules.end())
        return nullptr;

    return it->second.cubin_filename.c_str();
}

const char* wp_apic_get_kernel_forward_name(APICGraph* graph, int index)
{
    const APICKernel* kernel = apic_get_kernel_by_index(graph, index);
    return kernel ? kernel->forward_name.c_str() : nullptr;
}

const char* wp_apic_get_kernel_backward_name(APICGraph* graph, int index)
{
    const APICKernel* kernel = apic_get_kernel_by_index(graph, index);
    return kernel ? kernel->backward_name.c_str() : nullptr;
}

void wp_apic_register_loaded_cpu_kernel(
    APICGraph* graph, const char* kernel_key, const char* module_hash, void* forward_fn, void* backward_fn
)
{
    if (!graph || !kernel_key)
        return;
    APICCPUKernel info;
    info.forward_fn = forward_fn;
    info.backward_fn = backward_fn;
    graph->cpu_kernels[apic_kernel_map_key(module_hash ? module_hash : "", kernel_key)] = info;
}

// ============================================================================
// Graph parameter set/get — common dispatcher, CUDA path lives in apic.cu
// ============================================================================

static bool apic_lookup_param_region(APICGraph* graph, const char* name, void** out_ptr, size_t expected_size)
{
    if (!graph || !name)
        return false;

    auto param_it = graph->bindings.find(name);
    if (param_it == graph->bindings.end()) {
        fprintf(stderr, "APIC: Error - unknown parameter: %s\n", name);
        return false;
    }

    auto region_it = graph->regions.find(param_it->second);
    if (region_it == graph->regions.end() || !region_it->second.ptr) {
        fprintf(stderr, "APIC: Error - parameter region not found: %s\n", name);
        return false;
    }

    if (expected_size != region_it->second.size) {
        fprintf(
            stderr, "APIC: Error - size mismatch for parameter %s: expected %llu, got %llu\n", name,
            (unsigned long long)region_it->second.size, (unsigned long long)expected_size
        );
        return false;
    }

    *out_ptr = region_it->second.ptr;
    return true;
}

bool wp_apic_set_param(APICGraph* graph, const char* name, const void* data, size_t size)
{
    if (!data)
        return false;
    void* region_ptr = nullptr;
    if (!apic_lookup_param_region(graph, name, &region_ptr, size))
        return false;

    if (graph->device_type == APIC_DEVICE_CPU) {
        memcpy(region_ptr, data, size);
        return true;
    }
#if WP_ENABLE_CUDA
    return apic_set_param_cuda(graph, region_ptr, data, size);
#else
    return false;
#endif
}

bool wp_apic_get_param(APICGraph* graph, const char* name, void* data, size_t size)
{
    if (!data)
        return false;
    void* region_ptr = nullptr;
    if (!apic_lookup_param_region(graph, name, &region_ptr, size))
        return false;

    if (graph->device_type == APIC_DEVICE_CPU) {
        memcpy(data, region_ptr, size);
        return true;
    }
#if WP_ENABLE_CUDA
    return apic_get_param_cuda(graph, data, region_ptr, size);
#else
    return false;
#endif
}

// ============================================================================
// Graph load dispatcher
// ============================================================================

APICGraph* wp_apic_load_graph(void* context, const char* path, int device_type)
{
    if (!path) {
        wp::set_error_string("Path is null");
        return nullptr;
    }

#if !WP_ENABLE_CUDA
    if (device_type != APIC_DEVICE_CPU) {
        wp::set_error_string("CUDA graph load requested in a non-CUDA build");
        return nullptr;
    }
#endif

    std::string path_str(path);
    std::string wgf_path = path_str;
    std::string base_name = path_str;

    if (path_str.length() < 4 || path_str.substr(path_str.length() - 4) != ".wrp") {
        wgf_path = path_str + ".wrp";
    } else {
        base_name = path_str.substr(0, path_str.length() - 4);
    }

    size_t last_sep = base_name.find_last_of("/\\");
    std::string dir_path = (last_sep != std::string::npos) ? base_name.substr(0, last_sep + 1) : "";
    std::string name_only = (last_sep != std::string::npos) ? base_name.substr(last_sep + 1) : base_name;
    std::string modules_dir = dir_path + name_only + "_modules";

    std::vector<uint8_t> file_data;
    if (!apic_read_file(wgf_path.c_str(), file_data)) {
        wp::set_error_string("Failed to read file: %s", wgf_path.c_str());
        return nullptr;
    }

    if (file_data.size() < sizeof(APICFileHeader)) {
        wp::set_error_string("Invalid WRP file: too small");
        return nullptr;
    }

    const APICFileHeader* header = reinterpret_cast<const APICFileHeader*>(file_data.data());
    if (memcmp(header->magic, APIC_MAGIC, 4) != 0) {
        wp::set_error_string("Invalid WRP file: bad magic");
        return nullptr;
    }
    if (header->version < APIC_MIN_SUPPORTED_FORMAT_VERSION || header->version > APIC_FORMAT_VERSION) {
        wp::set_error_string(
            "Unsupported WRP version: %u (supported range %u-%u)", header->version,
            (unsigned)APIC_MIN_SUPPORTED_FORMAT_VERSION, (unsigned)APIC_FORMAT_VERSION
        );
        return nullptr;
    }

    APICGraph* graph = new APICGraph();
    graph->cuda_context = context;
    graph->target_arch = header->target_arch;
    graph->device_type = static_cast<APICDeviceType>(device_type);
    graph->base_path = base_name;

    const APICSectionEntry* sections
        = reinterpret_cast<const APICSectionEntry*>(file_data.data() + header->section_table_offset);

    const uint8_t* metadata_ptr = nullptr;
    size_t metadata_size = 0;
    const uint8_t* memory_ptr = nullptr;
    size_t memory_size = 0;
    const uint8_t* operations_ptr = nullptr;
    size_t operations_size = 0;

    for (uint32_t i = 0; i < header->num_sections; i++) {
        if (sections[i].type == APIC_SECTION_METADATA) {
            metadata_ptr = file_data.data() + sections[i].offset;
            metadata_size = sections[i].size;
        } else if (sections[i].type == APIC_SECTION_MEMORY) {
            memory_ptr = file_data.data() + sections[i].offset;
            memory_size = sections[i].size;
        } else if (sections[i].type == APIC_SECTION_OPERATIONS) {
            operations_ptr = file_data.data() + sections[i].offset;
            operations_size = sections[i].size;
        }
    }

    if (metadata_ptr && metadata_size > 0) {
        if (!apic_parse_metadata(metadata_ptr, metadata_size, graph)) {
            wp::set_error_string("Failed to parse metadata");
            delete graph;
            return nullptr;
        }
    }

    if (memory_ptr && !apic_parse_memory_regions(memory_ptr, memory_size, graph)) {
        wp::set_error_string("Failed to parse memory regions");
        delete graph;
        return nullptr;
    }

    if (device_type == APIC_DEVICE_CPU) {
        // CPU: allocate + initialize regions. Module loading is handled by
        // Python (LLVM) after this call returns, via wp_apic_register_loaded_cpu_kernel.
        if (!apic_init_cpu_graph_memory(graph, memory_ptr, memory_size)) {
            delete graph;
            return nullptr;
        }
    } else {
#if WP_ENABLE_CUDA
        if (!apic_load_graph_cuda_setup(graph, context, modules_dir, memory_ptr, memory_size)) {
            delete graph;
            return nullptr;
        }
#else
        // Unreachable — rejected above.
        delete graph;
        return nullptr;
#endif
    }

    if (!apic_create_meshes(graph)) {
        delete graph;
        return nullptr;
    }
    apic_fixup_ptr_locations(graph);

    if (operations_ptr && !apic_parse_operations(operations_ptr, operations_size, graph)) {
        wp::set_error_string("Failed to parse operations");
        delete graph;
        return nullptr;
    }

    // Validate the operation byte stream once, before any replay/rebuild consumes it.
    // Downstream paths (apic_rebuild_cuda_graph, wp_apic_cpu_replay_graph) gate on
    // graph->operations_validated and skip per-op bounds checks.
    graph->operations_validated = apic_validate_operation_stream(
        graph->operation_stream.data(), graph->operation_stream.size(), graph->operation_count
    );
    if (!graph->operations_validated) {
        wp::set_error_string("APIC operation stream failed validation");
        delete graph;
        return nullptr;
    }

    return graph;
}

// ============================================================================
// CUDA-only API stubs for non-CUDA builds
// ============================================================================

#if !WP_ENABLE_CUDA
void* wp_apic_get_cuda_graph(APICGraph*) { return nullptr; }
void* wp_apic_get_cuda_graph_exec(APICGraph*) { return nullptr; }
bool wp_apic_launch(APICGraph*, void*) { return false; }
#endif  // !WP_ENABLE_CUDA
