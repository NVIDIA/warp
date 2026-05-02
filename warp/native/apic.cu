// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// This file is #included at the end of warp.cu and contains only the
// CUDA-specific parts of APIC (API Capture): device-mesh registration with
// descriptor lookup, CUDA-side graph load setup (module loading + cudaMalloc
// + H2D init), graph rebuild, and the APICGraphInternal destructor (which
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
    APICState state, uint32_t region_id, uint64_t size, uint32_t element_size, const void* device_ptr
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

void wp_apic_register_mesh(APICState state, uint64_t mesh_id)
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


// APICGraphInternal is defined in apic_internal.h. The destructor is defined
// here (CUDA builds) because it uses CUDA APIs to free device resources.
APICGraphInternal::~APICGraphInternal()
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

static bool apic_init_memory(const uint8_t* data, size_t size, APICGraphInternal* graph)
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
bool apic_fixup_handle_cuda(APICGraphInternal* graph, uint8_t* base, uint64_t offset)
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
void* apic_resolve_region_ptr(APICGraphInternal* graph, int32_t region_id, uint64_t offset);

static CUfunction apic_get_kernel_function(
    APICGraphInternal* graph,
    const char* module_hash,
    size_t hash_len,
    const char* kernel_key,
    size_t key_len,
    bool is_forward
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

static bool apic_rebuild_cuda_graph(APICGraphInternal* graph, CUstream stream)
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

            std::vector<void*> args;
            std::vector<uint8_t*> arg_storage;

            // Build launch_bounds_t as args[0] (see builtin.h). The buffer is
            // owned by arg_storage (uint8_t*) for delete[] after launch.
            {
                int ndim = rec->ndim;
                if (ndim < 1)
                    ndim = 1;
                if (ndim > APIC_LAUNCH_MAX_DIMS)
                    ndim = APIC_LAUNCH_MAX_DIMS;

                uint8_t* bounds_buf = new uint8_t[sizeof(wp::launch_bounds_t)]();
                auto* bounds = reinterpret_cast<wp::launch_bounds_t*>(bounds_buf);
                for (int d = 0; d < ndim; d++)
                    bounds->shape[d] = rec->shape[d];
                bounds->ndim = ndim;
                bounds->size = rec->size;

                args.push_back(bounds_buf);
                arg_storage.push_back(bounds_buf);
            }

            // Parse param bindings
            for (uint16_t j = 0; j < rec->num_params; j++) {
                const APICLaunchParamRecord* binding = reinterpret_cast<const APICLaunchParamRecord*>(params_ptr);
                params_ptr += sizeof(APICLaunchParamRecord);

                if (binding->is_array) {
                    uint8_t* arr_buf = new uint8_t[sizeof(apic_array_t)];
                    apic_array_t* arr_ptr = reinterpret_cast<apic_array_t*>(arr_buf);
                    memset(arr_ptr, 0, sizeof(apic_array_t));

                    void* resolved = apic_resolve_region_ptr(graph, binding->region_id, binding->byte_offset);
                    arr_ptr->data = reinterpret_cast<uint64_t>(resolved);
                    arr_ptr->grad = 0;
                    arr_ptr->ndim = binding->ndim;
                    for (int d = 0; d < binding->ndim && d < APIC_MAX_DIMS; d++) {
                        arr_ptr->shape[d] = static_cast<int>(binding->shape[d]);
                        arr_ptr->strides[d] = static_cast<int>(binding->strides[d]);
                    }

                    args.push_back(arr_ptr);
                    arg_storage.push_back(arr_buf);
                } else {
                    size_t scalar_size = static_cast<size_t>(binding->byte_offset);
                    uint8_t* scalar_buf = new uint8_t[scalar_size]();
                    const uint8_t* shape_bytes = reinterpret_cast<const uint8_t*>(binding->shape);
                    const uint8_t* strides_bytes = reinterpret_cast<const uint8_t*>(binding->strides);
                    size_t shape_bytes_size = static_cast<size_t>(APIC_MAX_DIMS * sizeof(int64_t));
                    size_t copy_shape = scalar_size < shape_bytes_size ? scalar_size : shape_bytes_size;
                    size_t copy_strides = (scalar_size > copy_shape) ? scalar_size - copy_shape : 0;
                    if (copy_strides > shape_bytes_size)
                        copy_strides = shape_bytes_size;
                    memcpy(scalar_buf, shape_bytes, copy_shape);
                    if (copy_strides > 0)
                        memcpy(scalar_buf + copy_shape, strides_bytes, copy_strides);

                    args.push_back(scalar_buf);
                    arg_storage.push_back(scalar_buf);
                }
            }

            // Replay via the same wp_cuda_launch_kernel that captured this op.
            // apic_info=nullptr is safe: g_apic_state is null during replay, so
            // the recording branch in wp_cuda_launch_kernel is a no-op.
            wp_cuda_launch_kernel(
                graph->cuda_context, kernel, rec->dim, rec->max_blocks, rec->block_dim, rec->smem_bytes, args.data(),
                stream, /*apic_info=*/nullptr
            );

            for (uint8_t* p : arg_storage)
                delete[] p;
            break;
        }

        case APIC_OP_MEMCPY_H2D: {
            const APICMemcpyH2DRecord* rec = reinterpret_cast<const APICMemcpyH2DRecord*>(ptr);
            const uint8_t* src_data = ptr + sizeof(APICMemcpyH2DRecord);
            void* dst_ptr = apic_resolve_region_ptr(graph, rec->dst_region_id, rec->dst_offset);
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
            void* dst = apic_resolve_region_ptr(graph, rec->dst_region_id, rec->dst_offset);
            void* src = apic_resolve_region_ptr(graph, rec->src_region_id, rec->src_offset);
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
            void* dst = apic_resolve_region_ptr(graph, rec->region_id, rec->offset);
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
    APICGraphInternal* graph,
    void* context,
    const std::string& modules_dir,
    const uint8_t* memory_ptr,
    size_t memory_size
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

bool apic_set_param_cuda(APICGraphInternal* graph, void* dst, const void* src, size_t size)
{
    ContextGuard guard(graph->cuda_context);
    cudaError_t err = cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, 0);
    if (err != cudaSuccess) {
        wp::set_error_string("Failed to copy parameter data: %d", err);
        return false;
    }
    return true;
}

bool apic_get_param_cuda(APICGraphInternal* graph, void* dst, const void* src, size_t size)
{
    ContextGuard guard(graph->cuda_context);
    cudaError_t err = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, 0);
    if (err != cudaSuccess) {
        wp::set_error_string("Failed to copy parameter data: %d", err);
        return false;
    }
    return true;
}

void* wp_apic_get_cuda_graph(APICGraph graph)
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

void* wp_apic_get_cuda_graph_exec(APICGraph graph)
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

bool wp_apic_launch(APICGraph graph, void* stream)
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
