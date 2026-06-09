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
static void apic_register_region_from_device(
    APICState* state, uint32_t region_id, uint64_t size, uint32_t element_size, const void* device_ptr
)
{
    if (!device_ptr || size == 0) {
        wp_apic_register_memory_region(state, region_id, size, element_size, nullptr);
        return;
    }

    // Copy device memory to host buffer for serialization
    std::vector<uint8_t> host_buf(size);
    cudaError_t err = cudaMemcpy(host_buf.data(), device_ptr, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "APIC: Error - cudaMemcpy D2H failed for region %u: %d\n", region_id, err);
        return;
    }
    wp_apic_register_memory_region(state, region_id, size, element_size, host_buf.data());
}

void wp_apic_register_mesh(APICState* state, uint64_t mesh_id)
{
    if (!state || mesh_id == 0)
        return;

    // Try descriptor table first. If the mesh is not a device mesh, delegate to
    // the host-mesh helper in apic.cpp (mesh_id is a host pointer to wp::Mesh).
    wp::Mesh mesh;
    if (!wp::mesh_get_descriptor(mesh_id, mesh)) {
        apic_register_cpu_mesh(state, mesh_id);
        return;
    }

    // Device mesh: check-if-registered and register device arrays (cudaMemcpy D2H).
    for (const auto& rec : state->mesh_records) {
        if (rec.original_ptr == mesh_id)
            return;
    }

    auto register_region = [&](uint64_t ptr, uint64_t size, uint32_t elem_size) -> uint32_t {
        uint32_t rid = wp_apic_register_memory_region_by_ptr(state, ptr, size, elem_size);
        apic_register_region_from_device(state, rid, size, elem_size, reinterpret_cast<void*>(ptr));
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

    auto kern_it = graph->kernels.find(key_str);
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

    cudaError_t cuda_err = cudaStreamBeginCapture((cudaStream_t)stream, cudaStreamCaptureModeThreadLocal);
    if (cuda_err != cudaSuccess) {
        wp::set_error_string("Failed to begin graph capture: %d", cuda_err);
        return false;
    }

    bool success = true;

    const uint8_t* ptr = graph->operation_stream.data();
    const uint8_t* end = ptr + graph->operation_stream.size();

    for (uint32_t i = 0; i < graph->operation_count && ptr < end && success; i++) {
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
            size_t launch_result = wp_cuda_launch_kernel(
                graph->cuda_context, kernel, rec->dim, rec->max_blocks, rec->block_dim, rec->smem_bytes, args.data(),
                stream, /*apic_info=*/nullptr
            );

            for (uint8_t* p : arg_storage)
                delete[] p;
            if (launch_result) {
                success = false;
                break;
            }
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

        case APIC_OP_IF:
        case APIC_OP_WHILE:
            // CUDA rebuild of conditional ops is not yet supported (CPU only
            // for now). Reject instead of silently skipping.
            wp::set_error_string("APIC conditional ops (IF / WHILE) are not yet supported on CUDA loaded graphs");
            success = false;
            break;

        default:
            wp::set_error_string("Unknown operation type: %d", header->op_type);
            success = false;
            break;
        }

        ptr = op_start + header->total_size;
    }

    cudaGraph_t captured_graph;
    cuda_err = cudaStreamEndCapture((cudaStream_t)stream, &captured_graph);
    if (cuda_err != cudaSuccess) {
        wp::set_error_string("Failed to end graph capture: %d", cuda_err);
        return false;
    }

    if (!success) {
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
        CUstream stream;
        CUresult res = cuStreamCreate_f(&stream, CU_STREAM_DEFAULT);
        if (res != CUDA_SUCCESS) {
            wp::set_error_string("Failed to create CUDA stream for graph rebuild: %d", res);
            return nullptr;
        }
        bool success = apic_rebuild_cuda_graph(graph, stream);
        cuStreamDestroy_f(stream);
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
