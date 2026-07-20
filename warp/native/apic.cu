// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// This file is #included at the end of warp.cu and contains only the
// CUDA-specific parts of APIC (API Capture): device-mesh registration with
// descriptor lookup, CUDA-side graph load setup (module loading + cudaMalloc
// + H2D init), graph rebuild, and the APICGraph destructor (which
// needs cudaFree / cudaGraphDestroy). Everything else — pure-C++ recording,
// serialization, loading, and replay — lives in apic.cpp.
//
// Operations are recorded directly in C++ via hooks in wp_cuda_launch_kernel()
// and the memcpy/memset functions. The byte stream is the single source of
// truth for serialization.

#include "apic.h"
#include "apic_internal.h"
#include "mesh.h"

#include <algorithm>

// Register a memory region, copying data from device memory to host for serialization.
// Returns false (with an error string set) if the device-to-host copy fails, so the
// caller can abort the save rather than emit a region missing its initial data.
static bool apic_register_region_from_device(
    APICState* state, uint32_t region_id, uint64_t size, uint32_t element_size, const void* device_ptr
)
{
    if (!device_ptr || size == 0) {
        wp_apic_register_memory_region(state, region_id, size, element_size, nullptr);
        return true;
    }

    // Copy device memory to host buffer for serialization
    std::vector<uint8_t> host_buf(size);
    cudaError_t err = cudaMemcpy(host_buf.data(), device_ptr, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        wp::set_error_string(
            "APIC: device-to-host copy failed for region %u (CUDA error %d); the saved graph "
            "would be missing this region's initial data",
            region_id, err
        );
        return false;
    }
    wp_apic_register_memory_region(state, region_id, size, element_size, host_buf.data());
    return true;
}

// D2H-snapshot device regions that native hooks auto-registered during capture
// (memset/memcpy/scan/sort/...) into memory_regions so a saved CUDA graph
// carries their initial data. Python's capture_save snapshots only the regions
// it tracks in apic_capture._regions; internal inputs Python never sees -- e.g. a
// segmented sort's segment-index array -- live only in regions_by_id. Regions
// already present in memory_regions (snapshotted by Python) are left untouched;
// only the gaps are filled. Called from wp_apic_state_save for CUDA saves.
// Returns false (with an error string set) if any gap region's device-to-host copy
// fails; the caller aborts the save rather than emit a referenced region with no
// initial data. Transient (size-only) regions are pre-registered in memory_regions
// by Python's capture_save and skipped here, so a failure is always a content region.
bool apic_snapshot_device_regions(APICState* state, void* context)
{
    if (!state)
        return true;

    ContextGuard guard(context);
    for (uint32_t region_id = 0; region_id < state->regions_by_id.size(); region_id++) {
        const APICState::RegionEntry& entry = state->regions_by_id[region_id];
        if (!entry.valid || entry.base_ptr == 0 || entry.size == 0)
            continue;
        if (state->memory_regions.find(region_id) != state->memory_regions.end())
            continue;  // already serialized (e.g. a Python-tracked region)

        std::vector<uint8_t> host_buf(entry.size);
        cudaError_t err = cudaMemcpy(
            host_buf.data(), reinterpret_cast<const void*>(entry.base_ptr), entry.size, cudaMemcpyDeviceToHost
        );
        if (err != cudaSuccess) {
            wp::set_error_string(
                "APIC: device-region D2H snapshot failed for region %u (CUDA error %d); the saved "
                "graph would be missing this region's initial data",
                region_id, err
            );
            return false;
        }
        wp_apic_register_memory_region(state, region_id, entry.size, 1, host_buf.data());
    }
    return true;
}

bool wp_apic_register_mesh(APICState* state, uint64_t mesh_id)
{
    if (!state || mesh_id == 0)
        return true;

    // Try descriptor table first. If the mesh is not a device mesh, delegate to
    // the host-mesh helper in apic.cpp (mesh_id is a host pointer to wp::Mesh).
    wp::Mesh mesh;
    if (!wp::mesh_get_descriptor(mesh_id, mesh)) {
        apic_register_cpu_mesh(state, mesh_id);
        return true;
    }

    // Device mesh: check-if-registered and register device arrays (cudaMemcpy D2H).
    for (const auto& rec : state->mesh_records) {
        if (rec.original_ptr == mesh_id)
            return true;
    }

    bool ok = true;
    auto register_region = [&](uint64_t ptr, uint64_t size, uint32_t elem_size) -> uint32_t {
        uint32_t rid = wp_apic_register_memory_region_by_ptr(state, ptr, size, elem_size);
        if (!apic_register_region_from_device(state, rid, size, elem_size, reinterpret_cast<void*>(ptr)))
            ok = false;
        return rid;
    };

    uint64_t points_size = mesh.num_points * sizeof(wp::vec3);
    uint64_t indices_size = mesh.num_tris * (3 * sizeof(int));

    uint32_t points_region_id
        = register_region(reinterpret_cast<uint64_t>(mesh.points.data), points_size, sizeof(wp::vec3));
    uint32_t indices_region_id
        = register_region(reinterpret_cast<uint64_t>(mesh.indices.data), indices_size, sizeof(int));

    uint32_t velocities_region_id = 0;
    if (mesh.velocities.data) {
        uint64_t vel_size = mesh.num_points * sizeof(wp::vec3);
        velocities_region_id
            = register_region(reinterpret_cast<uint64_t>(mesh.velocities.data), vel_size, sizeof(wp::vec3));
    }

    // A D2H snapshot of one of the mesh arrays failed; abort before recording the
    // mesh so capture_save fails loudly instead of saving a mesh missing its data.
    if (!ok)
        return false;

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
    return true;
}


// APICGraph is defined in apic_internal.h. The destructor is defined
// here (CUDA builds) because it uses CUDA APIs to free device resources.
APICGraph::~APICGraph()
{
    if (device_type == APIC_DEVICE_CPU) {
        apic_destroy_cpu_graph_resources(this);
        return;
    }

    ContextGuard guard(cuda_context);

    for (uint64_t mesh_id : created_mesh_ids) {
        wp_mesh_destroy_device(mesh_id);
    }
    if (cuda_graph_exec) {
        cudaGraphExecDestroy((cudaGraphExec_t)cuda_graph_exec);
    }
    if (cuda_graph) {
        cudaGraphDestroy((cudaGraph_t)cuda_graph);
    }
    for (auto& pair : regions) {
        if (pair.second.ptr) {
            cudaFree(pair.second.ptr);
        }
    }
    for (auto& pair : modules) {
        if (pair.second.cuda_module) {
            cuModuleUnload_f(pair.second.cuda_module);
        }
    }
}

// apic_read_value<T> is a small inline helper used only by apic_init_memory below.
template <typename T> static T apic_read_value(const uint8_t*& ptr)
{
    T value;
    memcpy(&value, ptr, sizeof(T));
    ptr += sizeof(T);
    return value;
}

static bool apic_init_memory(const uint8_t* data, size_t size, APICGraph* graph)
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

        if (rec->has_initial_data) {
            auto it = graph->regions.find(rec->region_id);
            if (it != graph->regions.end() && it->second.ptr) {
                cudaError_t err = cudaMemcpy(it->second.ptr, ptr, rec->size, cudaMemcpyHostToDevice);
                if (err != cudaSuccess) {
                    wp::set_error_string("Failed to initialize memory region %u", rec->region_id);
                    return false;
                }
            }
            ptr += rec->size;
        }
    }
    return true;
}

// Per-offset handle fixup for a CUDA region: read the uint64 handle D2H,
// look it up in handle_ptr_remap, write the new value back H2D. The outer
// loop (apic_fixup_ptr_locations) lives in apic.cpp.
bool apic_fixup_handle_cuda(APICGraph* graph, uint8_t* base, uint64_t offset)
{
    uint64_t old_val;
    cudaError_t err = cudaMemcpy(&old_val, base + offset, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "APIC: Error - handle fixup D2H cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        return false;
    }
    auto remap_it = graph->handle_ptr_remap.find(old_val);
    if (remap_it != graph->handle_ptr_remap.end()) {
        uint64_t new_val = remap_it->second;
        err = cudaMemcpy(base + offset, &new_val, sizeof(uint64_t), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "APIC: Error - handle fixup H2D cudaMemcpy failed: %s\n", cudaGetErrorString(err));
            return false;
        }
    }
    return true;
}

// Forward declaration for a helper defined in apic.cpp; used by the
// kernel-launch / memcpy / memset dispatch in apic_rebuild_cuda_graph below.
void* apic_resolve_region_ptr(APICGraph* graph, int32_t region_id, uint64_t offset, size_t access_size);

static CUfunction apic_get_kernel_function(
    APICGraph* graph, const char* module_hash, size_t hash_len, const char* kernel_key, size_t key_len, bool is_forward
)
{
    std::string hash_str(module_hash, hash_len);
    std::string key_str(kernel_key, key_len);

    auto mod_it = graph->modules.find(hash_str);
    if (mod_it == graph->modules.end() || !mod_it->second.cuda_module) {
        wp::set_error_string("Module not loaded: %s", hash_str.c_str());
        return nullptr;
    }

    auto kern_it = graph->kernels.find(apic_kernel_map_key(hash_str, key_str));
    if (kern_it == graph->kernels.end()) {
        wp::set_error_string("Kernel not found: %s", key_str.c_str());
        return nullptr;
    }

    const std::string& kernel_name = is_forward ? kern_it->second.forward_name : kern_it->second.backward_name;
    CUfunction kernel;
    CUresult err = cuModuleGetFunction_f(&kernel, mod_it->second.cuda_module, kernel_name.c_str());
    if (err != CUDA_SUCCESS) {
        wp::set_error_string("Failed to get kernel function %s: %d", kernel_name.c_str(), err);
        return nullptr;
    }
    return kernel;
}

static bool apic_configure_kernel_cluster_attrs(APICGraph* graph)
{
    if (!graph)
        return true;
    if (graph->operation_count == 0 || graph->operation_stream.empty())
        return true;

    const uint8_t* ptr = graph->operation_stream.data();
    const uint8_t* end = ptr + graph->operation_stream.size();

    for (uint32_t i = 0; i < graph->operation_count && ptr < end; i++) {
        const APICOpHeader* header = reinterpret_cast<const APICOpHeader*>(ptr);

        if (header->op_type == APIC_OP_KERNEL_LAUNCH) {
            const APICLaunchRecord* rec = reinterpret_cast<const APICLaunchRecord*>(ptr);
            int cluster_dim = rec->cluster_dim > 0 ? rec->cluster_dim : 1;

            if (cluster_dim > 1) {
                const uint8_t* var_data = ptr + sizeof(APICLaunchRecord);
                const char* kernel_key = reinterpret_cast<const char*>(var_data);
                const char* module_hash = reinterpret_cast<const char*>(var_data + rec->kernel_key_len);
                CUfunction kernel = apic_get_kernel_function(
                    graph, module_hash, rec->module_hash_len, kernel_key, rec->kernel_key_len, rec->is_forward != 0
                );
                if (!kernel)
                    return false;

                if (!wp_cuda_set_kernel_cluster_attrs(kernel, cluster_dim, 1, 1)) {
                    wp::set_error_string("Failed to set cluster attributes for APIC kernel launch");
                    return false;
                }
            }
        }

        ptr += header->total_size;
    }

    return true;
}

// Replay a (sub-)stream of ops onto an already-capturing `stream`. No begin/end
// capture here: the caller (apic_rebuild_cuda_graph) owns the capture lifetime,
// so this function can recurse into IF/WHILE branch sub-streams via the same
// pause/resume mechanism the live capture_if/while path uses. `arch`/`use_ptx`
// are forwarded to the conditional helpers, which JIT the set-condition kernels.
static bool apic_replay_ops_into_cuda_capture(
    APICGraph* graph, CUstream stream, const uint8_t* op_data, size_t op_size, uint32_t op_count, int arch, bool use_ptx
)
{
    bool success = true;

    const uint8_t* ptr = op_data;
    const uint8_t* end = ptr + op_size;

    for (uint32_t i = 0; i < op_count && ptr < end && success; i++) {
        const APICOpHeader* header = reinterpret_cast<const APICOpHeader*>(ptr);
        const uint8_t* op_start = ptr;

        switch (header->op_type) {
        case APIC_OP_KERNEL_LAUNCH: {
            const APICLaunchRecord* rec = reinterpret_cast<const APICLaunchRecord*>(ptr);
            const uint8_t* var_data = ptr + sizeof(APICLaunchRecord);
            const char* kernel_key = reinterpret_cast<const char*>(var_data);
            const char* module_hash = reinterpret_cast<const char*>(var_data + rec->kernel_key_len);

            CUfunction kernel = apic_get_kernel_function(
                graph, module_hash, rec->module_hash_len, kernel_key, rec->kernel_key_len, rec->is_forward != 0
            );
            if (!kernel) {
                success = false;
                break;
            }

            const uint8_t* params_ptr = var_data + rec->kernel_key_len + rec->module_hash_len;
            const uint8_t* adj_params_ptr = params_ptr + rec->num_params * sizeof(APICLaunchParamRecord);
            // Backward kernels carry an adjoint block of the same shape; forward kernels none.
            const uint16_t adj_count = rec->is_forward ? 0u : rec->num_params;
            const APICLaunchPtrLocation* relocs = reinterpret_cast<const APICLaunchPtrLocation*>(
                adj_params_ptr + adj_count * sizeof(APICLaunchParamRecord)
            );
            const uint8_t* value_data
                = reinterpret_cast<const uint8_t*>(relocs) + rec->num_relocs * sizeof(APICLaunchPtrLocation);

            std::vector<void*> args;
            std::vector<uint8_t*> arg_storage;

            // Build launch_bounds_t<N> as args[0]. The buffer is owned by
            // arg_storage (uint8_t*) for delete[] after launch.
            {
                int ndim = rec->ndim;
                if (ndim < 1)
                    ndim = 1;
                if (ndim > APIC_LAUNCH_MAX_DIMS)
                    ndim = APIC_LAUNCH_MAX_DIMS;

                size_t size_offset = apic_detail::launch_bounds_size_offset(ndim);
                size_t coord_mult_offset = apic_detail::launch_bounds_coord_mult_offset(ndim);
                size_t bounds_alloc_size = apic_detail::launch_bounds_storage_size(ndim);

                uint8_t* bounds_buf = new uint8_t[bounds_alloc_size];
                memset(bounds_buf, 0, bounds_alloc_size);

                uint64_t shape_size = 1;
                int* shape_ptr = reinterpret_cast<int*>(bounds_buf);
                for (int d = 0; d < ndim; d++) {
                    shape_ptr[d] = rec->shape[d];
                    shape_size *= static_cast<uint64_t>(rec->shape[d]);
                }
                *reinterpret_cast<size_t*>(bounds_buf + size_offset) = rec->size;

                size_t coord_mult = 1;
                if (shape_size > 0 && rec->size > shape_size) {
                    uint64_t mult = rec->size / shape_size;
                    coord_mult = static_cast<size_t>(mult);
                }
                *reinterpret_cast<size_t*>(bounds_buf + coord_mult_offset) = coord_mult;

                args.push_back(bounds_buf);
                arg_storage.push_back(bounds_buf);
            }

            // Parse param bindings. Each binding's value blob is copied
            // verbatim from value_data, then relocations patch its pointer
            // fields with live device pointers and remapped handles. Called
            // once for forward params and once for adjoint params; the reloc
            // cursor advances naturally across both.
            const APICLaunchPtrLocation* reloc_cursor = relocs;
            auto append_bindings = [&](const uint8_t*& bindings_ptr, uint32_t count) -> bool {
                for (uint32_t j = 0; j < count; j++) {
                    const APICLaunchParamRecord* binding = reinterpret_cast<const APICLaunchParamRecord*>(bindings_ptr);
                    bindings_ptr += sizeof(APICLaunchParamRecord);

                    uint8_t* buf = new uint8_t[binding->value_size > 0 ? binding->value_size : 1]();
                    if (binding->value_size > 0)
                        memcpy(buf, value_data + binding->value_offset, binding->value_size);

                    for (uint16_t r = 0; r < binding->num_relocs; r++) {
                        const APICLaunchPtrLocation* reloc = &reloc_cursor[r];
                        uint64_t patched = 0;
                        switch (reloc->kind) {
                        case APIC_RELOC_DATA_PTR: {
                            void* resolved = apic_resolve_region_ptr(graph, reloc->region_id, reloc->region_offset, 0);
                            if (!resolved) {
                                delete[] buf;
                                wp::set_error_string(
                                    "APIC: unresolved DATA_PTR relocation (region_id=%d offset=%llu)", reloc->region_id,
                                    (unsigned long long)reloc->region_offset
                                );
                                return false;
                            }
                            patched = reinterpret_cast<uint64_t>(resolved);
                            break;
                        }
                        case APIC_RELOC_HANDLE: {
                            auto it = graph->handle_ptr_remap.find(reloc->region_offset);
                            patched = (it != graph->handle_ptr_remap.end()) ? it->second : reloc->region_offset;
                            break;
                        }
                        case APIC_RELOC_NULL:
                            patched = 0;
                            break;
                        }
                        memcpy(buf + reloc->value_byte_offset, &patched, sizeof(uint64_t));
                    }
                    reloc_cursor += binding->num_relocs;

                    args.push_back(buf);
                    arg_storage.push_back(buf);
                }
                return true;
            };

            if (!append_bindings(params_ptr, rec->num_params)
                || (!rec->is_forward && !append_bindings(adj_params_ptr, adj_count))) {
                for (uint8_t* p : arg_storage)
                    delete[] p;
                success = false;
                break;
            }

            // Replay via the same wp_cuda_launch_kernel that captured this op.
            // apic_info=nullptr is safe: g_apic_state is null during replay, so
            // the recording branch in wp_cuda_launch_kernel is a no-op.
            size_t launch_res = wp_cuda_launch_kernel(
                graph->cuda_context, kernel, rec->dim, rec->max_blocks, rec->block_dim, rec->grid_stride,
                rec->cluster_dim > 0 ? rec->cluster_dim : 1, rec->smem_bytes, args.data(), stream, /*apic_info=*/nullptr
            );
            if (launch_res != CUDA_SUCCESS)
                success = false;

            for (uint8_t* p : arg_storage)
                delete[] p;
            if (!success)
                break;
            break;
        }

        case APIC_OP_MEMCPY_H2D: {
            const APICMemcpyH2DRecord* rec = reinterpret_cast<const APICMemcpyH2DRecord*>(ptr);
            const uint8_t* src_data = ptr + sizeof(APICMemcpyH2DRecord);
            void* dst_ptr = apic_resolve_region_ptr(graph, rec->dst_region_id, rec->dst_offset, rec->size);
            if (!dst_ptr) {
                wp::set_error_string(
                    "H2D memcpy: failed to resolve dst region_id=%d offset=%llu", rec->dst_region_id,
                    (unsigned long long)rec->dst_offset
                );
                success = false;
                break;
            }
            if (!wp_memcpy_h2d(graph->cuda_context, dst_ptr, (void*)src_data, rec->size, stream))
                success = false;
            break;
        }

        case APIC_OP_MEMCPY_D2D: {
            const APICMemcpyD2DRecord* rec = reinterpret_cast<const APICMemcpyD2DRecord*>(ptr);
            void* dst = apic_resolve_region_ptr(graph, rec->dst_region_id, rec->dst_offset, rec->size);
            void* src = apic_resolve_region_ptr(graph, rec->src_region_id, rec->src_offset, rec->size);
            if (!dst || !src) {
                wp::set_error_string(
                    "D2D memcpy: failed to resolve dst region_id=%d offset=%llu / src region_id=%d offset=%llu",
                    rec->dst_region_id, (unsigned long long)rec->dst_offset, rec->src_region_id,
                    (unsigned long long)rec->src_offset
                );
                success = false;
                break;
            }
            if (!wp_memcpy_d2d(graph->cuda_context, dst, src, rec->size, stream))
                success = false;
            break;
        }

        case APIC_OP_MEMSET: {
            const APICMemsetRecord* rec = reinterpret_cast<const APICMemsetRecord*>(ptr);
            void* dst = apic_resolve_region_ptr(graph, rec->region_id, rec->offset, rec->size);
            if (!dst) {
                wp::set_error_string(
                    "memset: failed to resolve region_id=%d offset=%llu", rec->region_id,
                    (unsigned long long)rec->offset
                );
                success = false;
                break;
            }
            if (!wp_memset_device(graph->cuda_context, dst, rec->value, rec->size, stream))
                success = false;
            break;
        }

        case APIC_OP_ALLOC:
            break;  // Allocations handled by region setup

        case APIC_OP_MEMTILE: {
            const APICMemtileRecord* rec = reinterpret_cast<const APICMemtileRecord*>(ptr);
            const void* value = ptr + sizeof(APICMemtileRecord);
            uint64_t total_bytes = rec->count * rec->srcsize;
            void* dst = apic_resolve_region_ptr(graph, rec->region_id, rec->offset, total_bytes);
            if (!dst) {
                wp::set_error_string("APIC memtile: failed to resolve region (op %u)", i);
                success = false;
                break;
            }
            // g_apic_state is null during rebuild, so this executes (not record)
            // on the capture stream set as current above.
            wp_memtile_device(graph->cuda_context, dst, value, rec->srcsize, rec->count);
            break;
        }

        case APIC_OP_SCAN: {
            const APICScanRecord* rec = reinterpret_cast<const APICScanRecord*>(ptr);
            size_t scalar_size = apic_type_size(rec->dtype);
            size_t src_bytes = apic_strided_access_bytes(rec->length, rec->in_stride, rec->type_len, scalar_size);
            size_t dst_bytes = apic_strided_access_bytes(rec->length, rec->out_stride, rec->type_len, scalar_size);
            void* dst = apic_resolve_region_ptr(graph, rec->dst_region_id, rec->dst_offset, dst_bytes);
            void* src = apic_resolve_region_ptr(graph, rec->src_region_id, rec->src_offset, src_bytes);
            if (!dst || !src) {
                wp::set_error_string("APIC scan: failed to resolve region (op %u)", i);
                success = false;
                break;
            }
            // g_apic_state is null during rebuild, so these execute (not record)
            // on the capture stream set as current above. Same entry points
            // wp.utils.array_scan() dispatches to.
            if (rec->dtype == APIC_TYPE_INT32)
                wp_array_scan_int_device(
                    reinterpret_cast<uint64_t>(src), reinterpret_cast<uint64_t>(dst), static_cast<int>(rec->length),
                    rec->in_stride, rec->out_stride, rec->type_len, rec->inclusive != 0
                );
            else if (rec->dtype == APIC_TYPE_INT64)
                wp_array_scan_int64_device(
                    reinterpret_cast<uint64_t>(src), reinterpret_cast<uint64_t>(dst), static_cast<int>(rec->length),
                    rec->in_stride, rec->out_stride, rec->type_len, rec->inclusive != 0
                );
            else if (rec->dtype == APIC_TYPE_FLOAT32)
                wp_array_scan_float_device(
                    reinterpret_cast<uint64_t>(src), reinterpret_cast<uint64_t>(dst), static_cast<int>(rec->length),
                    rec->in_stride, rec->out_stride, rec->type_len, rec->inclusive != 0
                );
            else
                wp_array_scan_double_device(
                    reinterpret_cast<uint64_t>(src), reinterpret_cast<uint64_t>(dst), static_cast<int>(rec->length),
                    rec->in_stride, rec->out_stride, rec->type_len, rec->inclusive != 0
                );
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

            void* input_a = apic_resolve_region_ptr(graph, rec->input_a_region_id, rec->input_a_offset, input_a_bytes);
            void* input_b = rec->kind == APIC_REDUCTION_INNER
                ? apic_resolve_region_ptr(graph, rec->input_b_region_id, rec->input_b_offset, input_b_bytes)
                : nullptr;
            void* output = apic_resolve_region_ptr(graph, rec->output_region_id, rec->output_offset, output_bytes);
            if (!input_a || !output || (rec->kind == APIC_REDUCTION_INNER && !input_b)) {
                wp::set_error_string("APIC reduction: failed to resolve region (op %u)", i);
                success = false;
                break;
            }

            // Reissue the CUB entry points while stream capture is active to
            // record native graph nodes. GPU work runs only at graph launch.
            if (rec->kind == APIC_REDUCTION_SUM) {
                if (rec->dtype == APIC_TYPE_FLOAT32)
                    wp_array_sum_float_device(
                        reinterpret_cast<uint64_t>(input_a), reinterpret_cast<uint64_t>(output),
                        static_cast<int>(rec->count), rec->input_a_stride, rec->type_len
                    );
                else
                    wp_array_sum_double_device(
                        reinterpret_cast<uint64_t>(input_a), reinterpret_cast<uint64_t>(output),
                        static_cast<int>(rec->count), rec->input_a_stride, rec->type_len
                    );
            } else {
                if (rec->dtype == APIC_TYPE_FLOAT32)
                    wp_array_inner_float_device(
                        reinterpret_cast<uint64_t>(input_a), reinterpret_cast<uint64_t>(input_b),
                        reinterpret_cast<uint64_t>(output), static_cast<int>(rec->count), rec->input_a_stride,
                        rec->input_b_stride, rec->type_len
                    );
                else
                    wp_array_inner_double_device(
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
            // captures use two separate num_segments-entry arrays. Match the
            // span recorded at capture.
            bool segments_inferred_end = (rec->segstart_region_id == rec->segend_region_id);
            size_t segstart_bytes
                = (static_cast<size_t>(rec->num_segments) + (segments_inferred_end ? 1 : 0)) * sizeof(int32_t);
            size_t segend_bytes = static_cast<size_t>(rec->num_segments) * sizeof(int32_t);
            void* keys = apic_resolve_region_ptr(graph, rec->keys_region_id, rec->keys_offset, keys_bytes);
            void* values = apic_resolve_region_ptr(graph, rec->values_region_id, rec->values_offset, values_bytes);
            void* segstart
                = apic_resolve_region_ptr(graph, rec->segstart_region_id, rec->segstart_offset, segstart_bytes);
            void* segend = apic_resolve_region_ptr(graph, rec->segend_region_id, rec->segend_offset, segend_bytes);
            if (!keys || !values || !segstart || !segend) {
                wp::set_error_string("APIC segmented-sort: failed to resolve region (op %u)", i);
                success = false;
                break;
            }
            // g_apic_state is null during rebuild, so these execute (not record)
            // on the capture stream set as current above.
            if (rec->dtype == APIC_TYPE_INT32)
                wp_segmented_sort_pairs_int_device(
                    reinterpret_cast<uint64_t>(keys), reinterpret_cast<uint64_t>(values), static_cast<int>(rec->count),
                    reinterpret_cast<uint64_t>(segstart), reinterpret_cast<uint64_t>(segend),
                    static_cast<int>(rec->num_segments)
                );
            else
                wp_segmented_sort_pairs_float_device(
                    reinterpret_cast<uint64_t>(keys), reinterpret_cast<uint64_t>(values), static_cast<int>(rec->count),
                    reinterpret_cast<uint64_t>(segstart), reinterpret_cast<uint64_t>(segend),
                    static_cast<int>(rec->num_segments)
                );
            break;
        }

        case APIC_OP_RADIX_SORT: {
            const APICRadixSortRecord* rec = reinterpret_cast<const APICRadixSortRecord*>(ptr);
            size_t key_size = apic_type_size(rec->dtype);
            size_t keys_bytes = static_cast<size_t>(2) * rec->count * key_size;
            size_t values_bytes = static_cast<size_t>(2) * rec->count * static_cast<size_t>(rec->value_size);
            void* keys = apic_resolve_region_ptr(graph, rec->keys_region_id, rec->keys_offset, keys_bytes);
            void* values = apic_resolve_region_ptr(graph, rec->values_region_id, rec->values_offset, values_bytes);
            if (!keys || !values) {
                wp::set_error_string("APIC radix-sort: failed to resolve region (op %u)", i);
                success = false;
                break;
            }
            // g_apic_state is null during rebuild, so these execute (not record)
            // on the capture stream set as current above.
            if (rec->dtype == APIC_TYPE_INT32)
                wp_radix_sort_pairs_int_device(
                    reinterpret_cast<uint64_t>(keys), reinterpret_cast<uint64_t>(values), static_cast<int>(rec->count),
                    rec->begin_bit, rec->end_bit, rec->value_size
                );
            else if (rec->dtype == APIC_TYPE_UINT32)
                wp_radix_sort_pairs_uint_device(
                    reinterpret_cast<uint64_t>(keys), reinterpret_cast<uint64_t>(values), static_cast<int>(rec->count),
                    rec->begin_bit, rec->end_bit, rec->value_size
                );
            else if (rec->dtype == APIC_TYPE_INT64)
                wp_radix_sort_pairs_int64_device(
                    reinterpret_cast<uint64_t>(keys), reinterpret_cast<uint64_t>(values), static_cast<int>(rec->count),
                    rec->begin_bit, rec->end_bit, rec->value_size
                );
            else if (rec->dtype == APIC_TYPE_UINT64)
                wp_radix_sort_pairs_uint64_device(
                    reinterpret_cast<uint64_t>(keys), reinterpret_cast<uint64_t>(values), static_cast<int>(rec->count),
                    rec->begin_bit, rec->end_bit, rec->value_size
                );
            else if (rec->dtype == APIC_TYPE_FLOAT32)
                wp_radix_sort_pairs_float_device(
                    reinterpret_cast<uint64_t>(keys), reinterpret_cast<uint64_t>(values), static_cast<int>(rec->count),
                    rec->begin_bit, rec->end_bit, rec->value_size
                );
            else
                wp_radix_sort_pairs_double_device(
                    reinterpret_cast<uint64_t>(keys), reinterpret_cast<uint64_t>(values), static_cast<int>(rec->count),
                    rec->begin_bit, rec->end_bit, rec->value_size
                );
            break;
        }

        case APIC_OP_RUNLENGTH_ENCODE: {
            const APICRunlengthEncodeRecord* rec = reinterpret_cast<const APICRunlengthEncodeRecord*>(ptr);
            size_t n = rec->value_count;
            size_t in_bytes = n * sizeof(int32_t);
            void* values = apic_resolve_region_ptr(graph, rec->values_region_id, rec->values_offset, in_bytes);
            // run_values / run_lengths hold up to value_count entries.
            void* run_values
                = apic_resolve_region_ptr(graph, rec->run_values_region_id, rec->run_values_offset, in_bytes);
            void* run_lengths
                = apic_resolve_region_ptr(graph, rec->run_lengths_region_id, rec->run_lengths_offset, in_bytes);
            void* run_count
                = apic_resolve_region_ptr(graph, rec->run_count_region_id, rec->run_count_offset, sizeof(int32_t));
            if (!values || !run_values || !run_lengths || !run_count) {
                wp::set_error_string("APIC runlength-encode: failed to resolve region (op %u)", i);
                success = false;
                break;
            }
            // g_apic_state is null during rebuild, so this executes (not record)
            // on the capture stream set as current above.
            wp_runlength_encode_int_device(
                reinterpret_cast<uint64_t>(values), reinterpret_cast<uint64_t>(run_values),
                reinterpret_cast<uint64_t>(run_lengths), reinterpret_cast<uint64_t>(run_count),
                static_cast<int>(rec->value_count)
            );
            break;
        }

        case APIC_OP_BSR_TRANSPOSE: {
            const APICBsrTransposeRecord* rec = reinterpret_cast<const APICBsrTransposeRecord*>(ptr);
            size_t nnz = static_cast<size_t>(rec->nnz_upper_bound);
            size_t int_bytes = nnz * sizeof(int32_t);
            size_t rowp1_bytes = (static_cast<size_t>(rec->row_count) + 1) * sizeof(int32_t);
            size_t colp1_bytes = (static_cast<size_t>(rec->col_count) + 1) * sizeof(int32_t);
            void* bsr_offsets
                = apic_resolve_region_ptr(graph, rec->bsr_offsets_region_id, rec->bsr_offsets_offset, rowp1_bytes);
            void* bsr_row_counts = rec->bsr_row_counts_region_id >= 0
                ? apic_resolve_region_ptr(
                      graph, rec->bsr_row_counts_region_id, rec->bsr_row_counts_offset,
                      static_cast<size_t>(rec->row_count) * sizeof(int32_t)
                  )
                : nullptr;
            void* bsr_columns
                = apic_resolve_region_ptr(graph, rec->bsr_columns_region_id, rec->bsr_columns_offset, int_bytes);
            void* t_offsets = apic_resolve_region_ptr(
                graph, rec->transposed_offsets_region_id, rec->transposed_offsets_offset, colp1_bytes
            );
            void* t_row_counts = rec->transposed_row_counts_region_id >= 0
                ? apic_resolve_region_ptr(
                      graph, rec->transposed_row_counts_region_id, rec->transposed_row_counts_offset,
                      static_cast<size_t>(rec->col_count) * sizeof(int32_t)
                  )
                : nullptr;
            // The transposed-columns span is the destination's capacity
            // (t_offsets[col_count]), which is device memory that must NOT be
            // dereferenced here during the active rebuild capture (GH-1431), and
            // which may be smaller than the source's nnz upper bound. Resolve it
            // with a minimal span; the device transpose bounds its own accesses
            // by the capacity offsets. The block-index buffer doubles as CUB
            // sort scratch (DoubleBuffer key halves span 2*nnz entries), so
            // resolve it with the full scratch span.
            void* t_columns = apic_resolve_region_ptr(
                graph, rec->transposed_columns_region_id, rec->transposed_columns_offset, sizeof(int32_t)
            );
            void* block_indices = apic_resolve_region_ptr(
                graph, rec->block_indices_region_id, rec->block_indices_offset, 2 * int_bytes
            );
            void* status = rec->status_region_id >= 0
                ? apic_resolve_region_ptr(graph, rec->status_region_id, rec->status_offset, sizeof(int32_t))
                : nullptr;
            if (!bsr_offsets || !bsr_columns || !t_offsets || !t_columns || !block_indices
                || (rec->bsr_row_counts_region_id >= 0 && !bsr_row_counts)
                || (rec->transposed_row_counts_region_id >= 0 && !t_row_counts)
                || (rec->status_region_id >= 0 && !status)) {
                wp::set_error_string("APIC bsr-transpose: failed to resolve region (op %u)", i);
                success = false;
                break;
            }
            // g_apic_state is null during rebuild, so this executes (not record)
            // on the capture stream set as current above.
            wp_bsr_transpose_device(
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

            void* cond_ptr = apic_resolve_region_ptr(graph, rec->cond_region_id, rec->cond_offset, sizeof(int32_t));
            if (!cond_ptr) {
                wp::set_error_string("APIC if: failed to resolve condition region (op %u)", i);
                success = false;
                break;
            }

            // Insert the conditional node(s), then populate each branch body by
            // redirecting the same stream's capture into it -- the exact mechanism
            // the live capture_if uses (pause the outer capture, resume into the
            // body graph, replay the branch sub-stream, pause, check, then resume
            // the outer capture).
            void* graph_on_true = nullptr;
            void* graph_on_false = nullptr;
            if (!wp_cuda_graph_insert_if_else(
                    graph->cuda_context, (void*)stream, arch, use_ptx, reinterpret_cast<int*>(cond_ptr),
                    rec->branch_a_size > 0 ? &graph_on_true : nullptr,
                    rec->branch_b_size > 0 ? &graph_on_false : nullptr
                )) {
                success = false;
                break;
            }

            void* outer_graph = nullptr;
            if (!wp_cuda_graph_pause_capture(graph->cuda_context, (void*)stream, &outer_graph)) {
                success = false;
                break;
            }

            void* tmp = nullptr;
            if (success && rec->branch_a_size > 0) {
                if (!wp_cuda_graph_resume_capture(graph->cuda_context, (void*)stream, graph_on_true)
                    || !apic_replay_ops_into_cuda_capture(
                        graph, stream, branch_a, rec->branch_a_size, rec->branch_a_op_count, arch, use_ptx
                    )
                    || !wp_cuda_graph_pause_capture(graph->cuda_context, (void*)stream, &tmp)
                    || !wp_cuda_graph_check_conditional_body(graph_on_true)) {
                    success = false;
                }
            }
            if (success && rec->branch_b_size > 0) {
                if (!wp_cuda_graph_resume_capture(graph->cuda_context, (void*)stream, graph_on_false)
                    || !apic_replay_ops_into_cuda_capture(
                        graph, stream, branch_b, rec->branch_b_size, rec->branch_b_op_count, arch, use_ptx
                    )
                    || !wp_cuda_graph_pause_capture(graph->cuda_context, (void*)stream, &tmp)
                    || !wp_cuda_graph_check_conditional_body(graph_on_false)) {
                    success = false;
                }
            }

            // Always resume the outer capture so the caller's end_capture has a
            // capturing stream to close, even on a mid-branch failure.
            if (!wp_cuda_graph_resume_capture(graph->cuda_context, (void*)stream, outer_graph))
                success = false;
            break;
        }

        case APIC_OP_WHILE: {
            const APICCondRecord* rec = reinterpret_cast<const APICCondRecord*>(ptr);
            const uint8_t* body = ptr + sizeof(APICCondRecord);

            void* cond_ptr = apic_resolve_region_ptr(graph, rec->cond_region_id, rec->cond_offset, sizeof(int32_t));
            if (!cond_ptr) {
                wp::set_error_string("APIC while: failed to resolve condition region (op %u)", i);
                success = false;
                break;
            }

            // Mirror the live capture_while: insert the while node, capture the
            // body into its body graph, then set_condition re-evaluates the
            // condition after each iteration (the body updates it).
            void* body_graph = nullptr;
            uint64_t cond_handle = 0;
            if (!wp_cuda_graph_insert_while(
                    graph->cuda_context, (void*)stream, arch, use_ptx, reinterpret_cast<int*>(cond_ptr), &body_graph,
                    &cond_handle
                )) {
                success = false;
                break;
            }

            void* outer_graph = nullptr;
            if (!wp_cuda_graph_pause_capture(graph->cuda_context, (void*)stream, &outer_graph)) {
                success = false;
                break;
            }

            void* tmp = nullptr;
            if (success && rec->branch_a_size > 0) {
                if (!wp_cuda_graph_resume_capture(graph->cuda_context, (void*)stream, body_graph)
                    || !apic_replay_ops_into_cuda_capture(
                        graph, stream, body, rec->branch_a_size, rec->branch_a_op_count, arch, use_ptx
                    )
                    || !wp_cuda_graph_set_condition(
                        graph->cuda_context, (void*)stream, arch, use_ptx, reinterpret_cast<int*>(cond_ptr), cond_handle
                    )
                    || !wp_cuda_graph_pause_capture(graph->cuda_context, (void*)stream, &tmp)
                    || !wp_cuda_graph_check_conditional_body(body_graph)) {
                    success = false;
                }
            }

            if (!wp_cuda_graph_resume_capture(graph->cuda_context, (void*)stream, outer_graph))
                success = false;
            break;
        }

        default:
            wp::set_error_string("Unknown operation type: %d", header->op_type);
            success = false;
            break;
        }

        ptr = op_start + header->total_size;
    }

    return success;
}

static bool apic_rebuild_cuda_graph(APICGraph* graph, CUstream stream)
{
    // Precondition: graph->operations_validated is true. Enforced by
    // wp_apic_load_graph, which is the only path that produces an APICGraph.
    if (graph->cuda_graph_exec) {
        cudaGraphExecDestroy((cudaGraphExec_t)graph->cuda_graph_exec);
        graph->cuda_graph_exec = nullptr;
    }
    if (graph->cuda_graph) {
        cudaGraphDestroy((cudaGraph_t)graph->cuda_graph);
        graph->cuda_graph = nullptr;
    }

    if (!apic_configure_kernel_cluster_attrs(graph))
        return false;

    // Use Warp's capture management (not a raw cudaStreamBeginCapture) so the
    // allocator knows a capture is active. Extended ops (reduction/scan/sort/bsr) allocate
    // CUB scratch during capture; the allocator only routes those allocs/frees
    // to graph memory nodes when it sees an active Warp capture, otherwise it
    // would cudaFreeAsync on the null stream and invalidate the capture.
    if (!wp_cuda_graph_begin_capture(graph->cuda_context, (void*)stream, 0, WP_CUDA_GRAPH_CAPTURE_MODE_THREAD_LOCAL)) {
        return false;
    }

    // Extended ops (REDUCTION/SCAN/SORT/RUNLENGTH/BSR/MEMTILE) issue their kernels on the
    // context's current stream rather than the passed stream. Point the current
    // stream at the capture stream so they are recorded into the graph, then
    // restore it after the replay loop.
    void* prev_current_stream = wp_cuda_context_get_stream(graph->cuda_context);
    wp_cuda_context_set_stream(graph->cuda_context, (void*)stream, 0);

    // use_ptx is only consulted for IF/WHILE conditional kernels; loaded .wrp
    // graphs use the cubin path for the JIT-compiled set-condition kernels.
    bool success = apic_replay_ops_into_cuda_capture(
        graph, stream, graph->operation_stream.data(), graph->operation_stream.size(), graph->operation_count,
        graph->target_arch, false
    );

    wp_cuda_context_set_stream(graph->cuda_context, prev_current_stream, 0);

    cudaGraph_t captured_graph = nullptr;
    if (!wp_cuda_graph_end_capture(graph->cuda_context, (void*)stream, (void**)&captured_graph)) {
        // wp_cuda_graph_end_capture terminates the capture on failure.
        return false;
    }

    if (!success) {
        if (captured_graph)
            cudaGraphDestroy(captured_graph);
        return false;
    }

    graph->cuda_graph = (CUgraph)captured_graph;
    return true;
}

// ============================================================================
// CUDA-side graph load setup (called from wp_apic_load_graph in apic.cpp)
// ============================================================================

bool apic_load_graph_cuda_setup(
    APICGraph* graph, void* context, const std::string& modules_dir, const uint8_t* memory_ptr, size_t memory_size
)
{
    for (auto& pair : graph->modules) {
        std::string cubin_path = modules_dir + "/" + pair.second.cubin_filename;
#ifdef _WIN32
        std::replace(cubin_path.begin(), cubin_path.end(), '/', '\\');
#endif
        CUmodule cuda_module = (CUmodule)wp_cuda_load_module(context, cubin_path.c_str());
        if (!cuda_module) {
            wp::set_error_string("Failed to load module %s", cubin_path.c_str());
            return false;
        }
        pair.second.cuda_module = cuda_module;
    }

    ContextGuard guard(context);

    for (auto& pair : graph->regions) {
        void* device_ptr = nullptr;
        cudaError_t err = cudaMalloc(&device_ptr, pair.second.size);
        if (err != cudaSuccess) {
            wp::set_error_string("Failed to allocate %llu bytes: %d", (unsigned long long)pair.second.size, err);
            return false;
        }
        pair.second.ptr = device_ptr;
    }

    if (memory_ptr && !apic_init_memory(memory_ptr, memory_size, graph)) {
        return false;
    }

    cudaDeviceSynchronize();
    return true;
}

// wp_apic_set_param / wp_apic_get_param are defined in apic.cpp; they
// dispatch to apic_set_param_cuda / apic_get_param_cuda (below) when
// graph->device_type != APIC_DEVICE_CPU.

bool apic_set_param_cuda(APICGraph* graph, void* dst, const void* src, size_t size)
{
    ContextGuard guard(graph->cuda_context);
    cudaError_t err = cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, 0);
    if (err != cudaSuccess) {
        wp::set_error_string("Failed to copy parameter data: %d", err);
        return false;
    }
    return true;
}

bool apic_get_param_cuda(APICGraph* graph, void* dst, const void* src, size_t size)
{
    ContextGuard guard(graph->cuda_context);
    cudaError_t err = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, 0);
    if (err != cudaSuccess) {
        wp::set_error_string("Failed to copy parameter data: %d", err);
        return false;
    }
    // get_param returns immediately and the caller reads dst right after; ensure
    // the async copy has actually populated the destination before we return.
    err = cudaStreamSynchronize(0);
    if (err != cudaSuccess) {
        wp::set_error_string("Failed to synchronize get_param copy: %d", err);
        return false;
    }
    return true;
}

void* wp_apic_get_cuda_graph(APICGraph* graph)
{
    if (!graph)
        return nullptr;
    ContextGuard guard(graph->cuda_context);

    if (!graph->cuda_graph) {
        CUstream stream = (CUstream)wp_cuda_stream_create(graph->cuda_context, 0);
        if (!stream) {
            wp::set_error_string("Failed to create CUDA stream for graph rebuild");
            return nullptr;
        }
        bool success = apic_rebuild_cuda_graph(graph, stream);
        wp_cuda_stream_destroy(graph->cuda_context, stream);
        if (!success)
            return nullptr;
    }
    return graph->cuda_graph;
}

void* wp_apic_get_cuda_graph_exec(APICGraph* graph)
{
    if (!graph)
        return nullptr;
    ContextGuard guard(graph->cuda_context);

    if (!wp_apic_get_cuda_graph(graph))
        return nullptr;

    if (!graph->cuda_graph_exec) {
        cudaError_t err = cudaGraphInstantiateWithFlags(
            (cudaGraphExec_t*)&graph->cuda_graph_exec, (cudaGraph_t)graph->cuda_graph, 0
        );
        if (err != cudaSuccess) {
            wp::set_error_string("Failed to instantiate graph: %d", err);
            return nullptr;
        }
    }
    return graph->cuda_graph_exec;
}

bool wp_apic_launch(APICGraph* graph, void* stream)
{
    if (!graph)
        return false;

    void* exec = wp_apic_get_cuda_graph_exec(graph);
    if (!exec)
        return false;

    ContextGuard guard(graph->cuda_context);
    cudaError_t err = cudaGraphLaunch((cudaGraphExec_t)exec, (cudaStream_t)stream);
    if (err != cudaSuccess) {
        wp::set_error_string("Failed to launch graph: %d", err);
        return false;
    }
    return true;
}
