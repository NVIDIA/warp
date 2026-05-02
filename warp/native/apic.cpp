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
// APICGraphInternal destructor, the device-mesh registration path, and
// CUDA graph rebuild / launch.

#include "warp.h"

#include "apic.h"
#include "apic_internal.h"
#include "error.h"
#include "mesh.h"

#include <cstddef>  // offsetof
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

// The APIC record and replay paths (in this file, apic.cu, warp.cpp, and
// warp.cu) construct launch_bounds_t from serialized fields (shape/ndim/size)
// and pass it as args[0] to kernel entry points. Lock the expected layout so
// any future struct change fails at compile time rather than silently breaking
// the generated kernel's ABI on replay.
static_assert(sizeof(wp::launch_bounds_t) == 32, "APIC record/replay assumes 32-byte launch_bounds_t; see builtin.h");
static_assert(
    offsetof(wp::launch_bounds_t, shape) == 0, "APIC record/replay expects launch_bounds_t::shape at offset 0"
);
static_assert(
    offsetof(wp::launch_bounds_t, ndim) == 16, "APIC record/replay expects launch_bounds_t::ndim at offset 16"
);
static_assert(
    offsetof(wp::launch_bounds_t, size) == 24, "APIC record/replay expects launch_bounds_t::size at offset 24"
);
static_assert(
    APIC_LAUNCH_MAX_DIMS == wp::LAUNCH_MAX_DIMS,
    "APIC_LAUNCH_MAX_DIMS (apic_types.h) must match wp::LAUNCH_MAX_DIMS (builtin.h)"
);

// ============================================================================
// Thread-local Recording State
// ============================================================================

static thread_local APICState g_apic_state = nullptr;

APICState wp_apic_create_state() { return new APICStateInternal(); }

void wp_apic_destroy_state(APICState state) { delete state; }

void wp_apic_begin_recording(APICState state)
{
    if (state) {
        // Nested captures are not supported (matches CUDA runtime behavior).
        if (g_apic_state) {
            fprintf(stderr, "Warp error: nested APIC capture is not supported\n");
            return;
        }
        state->recording = true;
        // The stream is about to be (re)built; any prior validation is stale.
        state->operations_validated = false;
        g_apic_state = state;
    }
}

void wp_apic_end_recording(APICState state)
{
    if (state) {
        state->recording = false;
        // Validate the recorded byte stream once, at the point it becomes final.
        // Downstream replay paths gate on operations_validated and skip per-op
        // bounds checks.
        state->operations_validated = apic_validate_operation_stream(
            state->operation_stream.data(), state->operation_stream.size(), state->operation_count
        );
    }
    g_apic_state = nullptr;
}

APICState wp_apic_get_recording_state() { return g_apic_state; }

uint32_t wp_apic_get_operation_count(APICState state) { return state ? state->operation_count : 0; }

// ============================================================================
// Memory Region Registration
// ============================================================================

uint32_t wp_apic_register_memory_region_by_ptr(APICState state, uint64_t base_ptr, uint64_t size, uint32_t element_size)
{
    if (!state)
        return 0;
    for (size_t i = 0; i < state->ptr_region_table.size(); i++) {
        if (state->ptr_region_table[i].base_ptr == base_ptr)
            return state->ptr_region_table[i].region_id;
    }
    uint32_t region_id = state->next_region_id++;
    APICStateInternal::PtrRegionEntry entry;
    entry.base_ptr = base_ptr;
    entry.size = size;
    entry.region_id = region_id;
    state->ptr_region_table.push_back(entry);
    return region_id;
}

void wp_apic_register_memory_region(APICState state, uint32_t region_id, uint64_t size, uint32_t es, const void* data)
{
    if (!state)
        return;
    APICStateInternal::RegionInfo info;
    info.region_id = region_id;
    info.size = size;
    info.element_size = es;
    if (data && size > 0) {
        info.initial_data.resize(size);
        memcpy(info.initial_data.data(), data, size);
    }
    state->memory_regions[region_id] = std::move(info);
}

bool apic_resolve_ptr(APICState state, uint64_t ptr, int32_t* out_region_id, uint64_t* out_offset)
{
    if (!state)
        return false;
    for (size_t i = 0; i < state->ptr_region_table.size(); i++) {
        uint64_t base = state->ptr_region_table[i].base_ptr;
        uint64_t sz = state->ptr_region_table[i].size;
        if (ptr >= base && ptr < base + sz) {
            *out_region_id = (int32_t)state->ptr_region_table[i].region_id;
            *out_offset = ptr - base;
            return true;
        }
    }
    return false;
}

// Resolve a region pointer from an APICStateInternal's ptr_region_table.
static void* apic_resolve_state_region_ptr(APICStateInternal* state, int32_t region_id, uint64_t offset)
{
    if (region_id < 0)
        return nullptr;
    for (size_t i = 0; i < state->ptr_region_table.size(); i++) {
        auto& entry = state->ptr_region_table[i];
        if (entry.region_id == static_cast<uint32_t>(region_id)) {
            if (offset > entry.size || entry.base_ptr > UINTPTR_MAX - offset)
                return nullptr;
            return reinterpret_cast<void*>(entry.base_ptr + offset);
        }
    }
    return nullptr;
}

// ============================================================================
// Metadata Registration (for serialization)
// ============================================================================

void wp_apic_register_module(
    APICState state, const char* module_hash, const char* module_name, const char* bf, int arch
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
    APICState state,
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
    if (state->kernels.find(key_str) == state->kernels.end()) {
        APICKernel kern;
        kern.kernel_key = key_str;
        kern.module_hash = module_hash ? module_hash : "";
        kern.forward_name = forward_name ? forward_name : "";
        kern.backward_name = backward_name ? backward_name : "";
        kern.forward_smem_bytes = forward_smem_bytes;
        kern.backward_smem_bytes = backward_smem_bytes;
        kern.block_dim = block_dim;
        state->kernels[key_str] = kern;
    }
}

void wp_apic_register_binding(APICState state, const char* name, uint32_t region_id)
{
    if (!state || !name)
        return;
    state->bindings.push_back({ std::string(name), region_id });
}

void wp_apic_register_ptr_location(APICState state, uint32_t region_id, uint64_t offset, uint64_t stride)
{
    if (!state)
        return;
    APICPtrLocation loc;
    loc.region_id = region_id;
    loc.offset = offset;
    loc.stride = stride;
    state->ptr_locations.push_back(loc);
}

void wp_apic_register_cpu_kernel(APICState state, const char* kernel_key, void* forward_fn, void* backward_fn)
{
    if (!state || !kernel_key)
        return;
    APICCPUKernel entry;
    entry.forward_fn = forward_fn;
    entry.backward_fn = backward_fn;
    state->cpu_kernels[std::string(kernel_key)] = entry;
}

// ============================================================================
// Operation Recording
// ============================================================================

void apic_record_kernel_launch(
    APICState state,
    const char* kernel_key,
    const char* module_hash,
    int is_forward,
    const int* shape,
    int ndim,
    uint64_t size,
    int max_blocks,
    int block_dim,
    int smem_bytes,
    const APICLaunchParamRecord* params,
    int num_params
)
{
    if (!state)
        return;
    size_t key_len = kernel_key ? strlen(kernel_key) : 0;
    size_t hash_len = module_hash ? strlen(module_hash) : 0;
    size_t params_size = num_params * sizeof(APICLaunchParamRecord);
    uint32_t total_size = (uint32_t)(sizeof(APICLaunchRecord) + key_len + hash_len + params_size);

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
    rec.smem_bytes = smem_bytes;
    rec.is_forward = is_forward ? 1 : 0;
    rec.kernel_key_len = static_cast<uint16_t>(key_len);
    rec.module_hash_len = static_cast<uint16_t>(hash_len);
    rec.num_params = static_cast<uint16_t>(num_params);
    rec.num_handle_offsets = 0;

    state->append_bytes(&rec, sizeof(rec));
    if (key_len > 0)
        state->append_bytes(kernel_key, key_len);
    if (hash_len > 0)
        state->append_bytes(module_hash, hash_len);
    if (params_size > 0)
        state->append_bytes(params, params_size);
    state->operation_count++;
}

void apic_record_memcpy_d2d(
    APICState state,
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

void apic_record_memset(APICState state, int32_t region_id, uint64_t offset, uint64_t size, int32_t value)
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

void apic_record_alloc(APICState state, int32_t region_id, uint64_t size)
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

void apic_register_cpu_mesh(APICState state, uint64_t mesh_id)
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
void wp_apic_register_mesh(APICState state, uint64_t mesh_id) { apic_register_cpu_mesh(state, mesh_id); }
#endif  // !WP_ENABLE_CUDA

// ============================================================================
// Graph-side helpers (shared between CUDA and non-CUDA builds)
// ============================================================================

void* apic_resolve_region_ptr(APICGraphInternal* graph, int32_t region_id, uint64_t offset)
{
    if (region_id < 0)
        return nullptr;
    auto it = graph->regions.find(region_id);
    if (it != graph->regions.end() && it->second.ptr) {
        if (offset > it->second.size)
            return nullptr;
        return static_cast<void*>(static_cast<uint8_t*>(it->second.ptr) + offset);
    }
    return nullptr;
}

// Free host-side mesh and region memory for a CPU graph. Called from:
//   - Non-CUDA build: APICGraphInternal destructor (below).
//   - CUDA build: apic.cu's destructor calls this for APIC_DEVICE_CPU graphs.
void apic_destroy_cpu_graph_resources(APICGraphInternal* graph)
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
bool apic_init_cpu_graph_memory(APICGraphInternal* graph, const uint8_t* memory_ptr, size_t memory_size)
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
// Non-CUDA build: APICGraphInternal destructor only frees host memory.
// (CUDA builds define the destructor in apic.cu so it can use cudaFree etc.)
APICGraphInternal::~APICGraphInternal() { apic_destroy_cpu_graph_resources(this); }
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

bool apic_parse_metadata(const uint8_t* data, size_t size, APICGraphInternal* graph)
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
        graph->kernels[info.kernel_key] = info;
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
        APICPtrLocation loc;
        loc.region_id = apic_read_value<uint32_t>(ptr);
        loc.offset = apic_read_value<uint64_t>(ptr);
        loc.stride = apic_read_value<uint64_t>(ptr);
        graph->ptr_locations.push_back(loc);
    }

    return true;
}

bool apic_parse_operations(const uint8_t* data, size_t size, APICGraphInternal* graph)
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

bool apic_parse_memory_regions(const uint8_t* data, size_t size, APICGraphInternal* graph)
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
            APICRegion region;
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

bool apic_create_meshes(APICGraphInternal* graph)
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
void apic_fixup_ptr_locations(APICGraphInternal* graph)
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

// Walks the byte stream once and verifies every record fits in bounds, its
// variable-length fields don't overflow its declared total_size, and the
// op_type is recognized. Called once at stream-close time (end of recording
// for live capture; after .wrp load for deserialized graphs). Replay paths
// skip per-op bounds checks once this returns true.
bool apic_validate_operation_stream(const uint8_t* data, size_t size, uint32_t operation_count)
{
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
        case APIC_OP_ALLOC:
            if (op_end < op_start + sizeof(APICAllocRecord)) {
                fprintf(stderr, "APIC: Error - alloc record overflow at operation %u\n", i);
                return false;
            }
            break;
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

// Walk an APIC byte stream and execute CPU operations. Assumes the stream
// has already passed apic_validate_operation_stream — no per-op bounds
// checks are performed here.
// resolve_ptr: callback (int32_t region_id, uint64_t offset) -> void*
// find_kernel: callback (const std::string& key, uint8_t is_forward) -> void*
template <typename ResolvePtrFn, typename FindKernelFn>
static bool apic_cpu_replay_stream(
    const uint8_t* stream_data,
    size_t stream_size,
    uint32_t operation_count,
    ResolvePtrFn resolve_ptr,
    FindKernelFn find_kernel
)
{
    // Stream was validated at close time (wp_apic_end_recording or
    // wp_apic_load_graph) via apic_validate_operation_stream — no per-op
    // bounds checks needed here.
    const uint8_t* ptr = stream_data;

    for (uint32_t i = 0; i < operation_count; i++) {
        const APICOpHeader* header = reinterpret_cast<const APICOpHeader*>(ptr);

        switch (header->op_type) {
        case APIC_OP_KERNEL_LAUNCH: {
            const APICLaunchRecord* rec = reinterpret_cast<const APICLaunchRecord*>(ptr);
            const uint8_t* var_data = ptr + sizeof(APICLaunchRecord);
            std::string key_str(reinterpret_cast<const char*>(var_data), rec->kernel_key_len);

            const uint8_t* params_start = var_data + rec->kernel_key_len + rec->module_hash_len;

            void* func = find_kernel(key_str, rec->is_forward);
            if (!func) {
                fprintf(stderr, "APIC: Error - CPU kernel not found: %s\n", key_str.c_str());
                return false;
            }

            // Build launch_bounds_t (see builtin.h) to pass as the kernel's
            // first argument, matching wp_cpu_launch_kernel's signature.
            int ndim = rec->ndim;
            if (ndim < 1)
                ndim = 1;
            if (ndim > APIC_LAUNCH_MAX_DIMS)
                ndim = APIC_LAUNCH_MAX_DIMS;

            wp::launch_bounds_t bounds = {};
            for (int d = 0; d < ndim; d++)
                bounds.shape[d] = rec->shape[d];
            bounds.ndim = ndim;
            bounds.size = rec->size;

            // Build args from param bindings
            const uint8_t* params_ptr = params_start;

            // Compute total args size with alignment
            size_t args_total = 0;
            for (uint16_t j = 0; j < rec->num_params; j++) {
                const APICLaunchParamRecord* binding
                    = reinterpret_cast<const APICLaunchParamRecord*>(params_ptr + j * sizeof(APICLaunchParamRecord));
                size_t param_size
                    = binding->is_array ? sizeof(apic_array_t) : static_cast<size_t>(binding->byte_offset);
                size_t align = binding->is_array ? alignof(void*) : (param_size >= 8 ? 8 : (param_size >= 4 ? 4 : 1));
                args_total = (args_total + align - 1) & ~(align - 1);
                args_total += param_size;
            }

            // Stack-allocate args buffer (typical kernel args are small)
            uint8_t args_stack[512];
            uint8_t* args_buf
                = (args_total <= sizeof(args_stack)) ? args_stack : static_cast<uint8_t*>(malloc(args_total));
            if (!args_buf) {
                fprintf(stderr, "APIC: Error - failed to allocate %zu bytes for kernel args\n", args_total);
                return false;
            }
            memset(args_buf, 0, args_total > 0 ? args_total : 1);
            size_t args_offset = 0;

            for (uint16_t j = 0; j < rec->num_params; j++) {
                const APICLaunchParamRecord* binding
                    = reinterpret_cast<const APICLaunchParamRecord*>(params_ptr + j * sizeof(APICLaunchParamRecord));

                if (binding->is_array) {
                    size_t align = alignof(void*);
                    args_offset = (args_offset + align - 1) & ~(align - 1);

                    apic_array_t* arr = reinterpret_cast<apic_array_t*>(args_buf + args_offset);
                    void* resolved = resolve_ptr(binding->region_id, binding->byte_offset);
                    arr->data = reinterpret_cast<uint64_t>(resolved);
                    arr->grad = 0;
                    arr->ndim = binding->ndim;
                    for (int d = 0; d < binding->ndim && d < APIC_MAX_DIMS; d++) {
                        arr->shape[d] = static_cast<int>(binding->shape[d]);
                        arr->strides[d] = static_cast<int>(binding->strides[d]);
                    }
                    args_offset += sizeof(apic_array_t);
                } else {
                    size_t scalar_size = static_cast<size_t>(binding->byte_offset);
                    size_t align = scalar_size >= 8 ? 8 : (scalar_size >= 4 ? 4 : 1);
                    args_offset = (args_offset + align - 1) & ~(align - 1);

                    const uint8_t* shape_bytes = reinterpret_cast<const uint8_t*>(binding->shape);
                    const uint8_t* strides_bytes = reinterpret_cast<const uint8_t*>(binding->strides);
                    size_t shape_bytes_size = static_cast<size_t>(APIC_MAX_DIMS * sizeof(int64_t));
                    size_t copy_shape = scalar_size < shape_bytes_size ? scalar_size : shape_bytes_size;
                    size_t copy_strides = (scalar_size > copy_shape) ? scalar_size - copy_shape : 0;
                    if (copy_strides > shape_bytes_size)
                        copy_strides = shape_bytes_size;
                    memcpy(args_buf + args_offset, shape_bytes, copy_shape);
                    if (copy_strides > 0)
                        memcpy(args_buf + args_offset + copy_shape, strides_bytes, copy_strides);
                    args_offset += scalar_size;
                }
            }

            // Replay via the same wp_cpu_launch_kernel that captured this op.
            // apic_info=nullptr is safe: g_apic_state is null during replay, so
            // the recording branch in wp_cpu_launch_kernel is a no-op and the
            // execute branch fires.
            wp_cpu_launch_kernel(func, &bounds, args_buf, /*adj_args=*/nullptr, /*apic_info=*/nullptr);

            if (args_buf != args_stack)
                free(args_buf);
            break;
        }

        case APIC_OP_MEMCPY_D2D: {
            const APICMemcpyD2DRecord* rec = reinterpret_cast<const APICMemcpyD2DRecord*>(ptr);
            void* dst = resolve_ptr(rec->dst_region_id, rec->dst_offset);
            const void* src = resolve_ptr(rec->src_region_id, rec->src_offset);
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
            void* dst = resolve_ptr(rec->region_id, rec->offset);
            if (!dst) {
                fprintf(stderr, "APIC: Error - memset pointer resolution failed at operation %u\n", i);
                return false;
            }
            if (!wp_memset_host(dst, rec->value, rec->size))
                return false;
            break;
        }

        case APIC_OP_ALLOC:
            break;
        default:
            fprintf(stderr, "APIC: Error - unsupported CPU replay op type %u\n", unsigned(header->op_type));
            break;
        }

        ptr += header->total_size;
    }

    return true;
}

// Shared body for the two public CPU replay entry points. Container is either
// APICStateInternal or APICGraphInternal; both expose operation_stream,
// operation_count, and cpu_kernels (map<string, APICCPUKernel>). The only
// per-container difference is how region_id is resolved to a host pointer,
// which the caller supplies as `resolve`.
template <typename Container, typename Resolver> static bool apic_cpu_replay_container(Container* c, Resolver resolve)
{
    auto find_kernel = [c](const std::string& key, uint8_t is_forward) -> void* {
        auto it = c->cpu_kernels.find(key);
        if (it == c->cpu_kernels.end())
            return nullptr;
        return is_forward ? it->second.forward_fn : it->second.backward_fn;
    };
    return apic_cpu_replay_stream(
        c->operation_stream.data(), c->operation_stream.size(), c->operation_count, resolve, find_kernel
    );
}

bool wp_apic_cpu_replay_state(APICState state)
{
    if (!state)
        return false;
    if (state->operation_stream.empty())
        return true;
    if (!state->operations_validated) {
        fprintf(stderr, "APIC: Error - replay called on unvalidated state; was wp_apic_end_recording called?\n");
        return false;
    }
    return apic_cpu_replay_container(state, [state](int32_t region_id, uint64_t offset) {
        return apic_resolve_state_region_ptr(state, region_id, offset);
    });
}

bool wp_apic_cpu_replay_graph(APICGraph graph)
{
    if (!graph)
        return false;
    if (graph->operation_stream.empty())
        return true;
    // Precondition: graph->operations_validated is true. Enforced by
    // wp_apic_load_graph, which is the only path that produces an APICGraph.
    return apic_cpu_replay_container(graph, [graph](int32_t region_id, uint64_t offset) {
        return apic_resolve_region_ptr(graph, region_id, offset);
    });
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

bool wp_apic_state_save(APICState state, const char* path, int target_arch)
{
    if (!state || !path) {
        fprintf(stderr, "APIC: Null %s passed to wp_apic_state_save\n", !state ? "state" : "path");
        return false;
    }

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
        const APICStateInternal::RegionInfo& region = kv.second;
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

void wp_apic_destroy_graph(APICGraph graph) { delete graph; }

int wp_apic_get_num_params(APICGraph graph) { return graph ? static_cast<int>(graph->binding_names.size()) : 0; }

const char* wp_apic_get_param_name(APICGraph graph, int index)
{
    if (!graph || index < 0 || index >= static_cast<int>(graph->binding_names.size()))
        return nullptr;
    return graph->binding_names[index].c_str();
}

size_t wp_apic_get_param_size(APICGraph graph, const char* name)
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

void* wp_apic_get_param_ptr(APICGraph graph, const char* name)
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

int wp_apic_get_num_kernels(APICGraph graph) { return graph ? static_cast<int>(graph->kernels.size()) : 0; }

const char* wp_apic_get_kernel_key(APICGraph graph, int index)
{
    if (!graph || index < 0 || index >= static_cast<int>(graph->kernels.size()))
        return nullptr;
    auto it = graph->kernels.begin();
    std::advance(it, index);
    return it->first.c_str();
}

const char* wp_apic_get_kernel_forward_name(APICGraph graph, const char* kernel_key)
{
    if (!graph || !kernel_key)
        return nullptr;
    auto it = graph->kernels.find(kernel_key);
    if (it == graph->kernels.end())
        return nullptr;
    return it->second.forward_name.c_str();
}

const char* wp_apic_get_kernel_backward_name(APICGraph graph, const char* kernel_key)
{
    if (!graph || !kernel_key)
        return nullptr;
    auto it = graph->kernels.find(kernel_key);
    if (it == graph->kernels.end())
        return nullptr;
    return it->second.backward_name.c_str();
}

const char* wp_apic_get_kernel_module_hash(APICGraph graph, const char* kernel_key)
{
    if (!graph || !kernel_key)
        return nullptr;
    auto it = graph->kernels.find(kernel_key);
    if (it == graph->kernels.end())
        return nullptr;
    return it->second.module_hash.c_str();
}

void wp_apic_register_loaded_cpu_kernel(APICGraph graph, const char* kernel_key, void* forward_fn, void* backward_fn)
{
    if (!graph || !kernel_key)
        return;
    APICCPUKernel info;
    info.forward_fn = forward_fn;
    info.backward_fn = backward_fn;
    graph->cpu_kernels[std::string(kernel_key)] = info;
}

// ============================================================================
// Graph parameter set/get — common dispatcher, CUDA path lives in apic.cu
// ============================================================================

static bool apic_lookup_param_region(APICGraph graph, const char* name, void** out_ptr, size_t expected_size)
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

bool wp_apic_set_param(APICGraph graph, const char* name, const void* data, size_t size)
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

bool wp_apic_get_param(APICGraph graph, const char* name, void* data, size_t size)
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

APICGraph wp_apic_load_graph(void* context, const char* path, int device_type)
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
    if (header->version > APIC_FORMAT_VERSION) {
        wp::set_error_string(
            "Unsupported WRP version: %u (expected %u)", header->version, (unsigned)APIC_FORMAT_VERSION
        );
        return nullptr;
    }

    APICGraphInternal* graph = new APICGraphInternal();
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
void* wp_apic_get_cuda_graph(APICGraph) { return nullptr; }
void* wp_apic_get_cuda_graph_exec(APICGraph) { return nullptr; }
bool wp_apic_launch(APICGraph, void*) { return false; }
#endif  // !WP_ENABLE_CUDA
