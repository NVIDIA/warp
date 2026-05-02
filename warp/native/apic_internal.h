// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Internal APIC structures and cross-file helpers shared between apic.cpp
// (pure C++) and apic.cu (CUDA). This header must compile with both nvcc and
// standard C++ compilers, so it must not expose CUDA types outside the
// __CUDACC__ / WP_ENABLE_CUDA guards.

#include "apic_types.h"

#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

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

struct APICRegion {
    uint32_t region_id = 0;
    uint64_t base_ptr = 0;
    uint64_t size = 0;
    uint32_t element_size = 0;
    std::vector<uint8_t> initial_data;
    void* ptr = nullptr;  // Allocated pointer (device or host depending on context)
};

struct APICPtrLocation {
    uint32_t region_id;
    uint64_t offset;
    uint64_t stride;
};

// CPU kernel function pointers, resolved during capture / load and consumed
// by CPU replay. Stored both in APICStateInternal (for live capture) and
// APICGraphInternal (for loaded .wrp graphs), keyed by kernel name.
struct APICCPUKernel {
    void* forward_fn = nullptr;
    void* backward_fn = nullptr;
};

// ============================================================================
// APICStateInternal — recording state
// ============================================================================

struct APICStateInternal {
    bool recording = false;

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

    // Pointer-to-region lookup (for memcpy/memset hooks that receive raw pointers)
    struct PtrRegionEntry {
        uint64_t base_ptr = 0;
        uint64_t size = 0;
        uint32_t region_id = 0;
    };
    std::vector<PtrRegionEntry> ptr_region_table;
    uint32_t next_region_id = 1;

    // Module and kernel metadata (registered from Python for serialization)
    std::unordered_map<std::string, APICModule> modules;
    std::unordered_map<std::string, APICKernel> kernels;

    // Named bindings (name -> region_id)
    std::vector<std::pair<std::string, uint32_t>> bindings;

    // Handle pointer locations
    std::vector<APICPtrLocation> ptr_locations;

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
// APICGraphInternal — loaded .wrp graph state
// ============================================================================

struct APICGraphInternal {
    void* cuda_context = nullptr;
    int target_arch = 0;
    APICDeviceType device_type = APIC_DEVICE_CUDA;

    std::unordered_map<std::string, APICModule> modules;
    std::unordered_map<std::string, APICKernel> kernels;
    std::unordered_map<uint32_t, APICRegion> regions;
    std::unordered_map<std::string, uint32_t> bindings;
    std::vector<std::string> binding_names;

    std::vector<uint8_t> operation_stream;
    uint32_t operation_count = 0;

    // Set by wp_apic_load_graph after apic_validate_operation_stream passes.
    // Replay paths require it; see apic_validate_operation_stream.
    bool operations_validated = false;

    std::unordered_map<uint64_t, uint64_t> handle_ptr_remap;
    std::vector<APICPtrLocation> ptr_locations;
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
    ~APICGraphInternal();
};

// ============================================================================
// Operation stream validation
// ============================================================================

// Walk the operation byte stream once and verify that every record header,
// variable-length payload, and op_type is within bounds. Prints a diagnostic
// and returns false on first inconsistency. Defined in apic.cpp.
bool apic_validate_operation_stream(const uint8_t* data, size_t size, uint32_t operation_count);

// ============================================================================
// .wrp file reading helpers (pure C++, defined in apic.cpp)
// ============================================================================

bool apic_read_file(const char* path, std::vector<uint8_t>& data);

// .wrp parsing — pure C++ (defined in apic.cpp). Populate fields on `graph`.
bool apic_parse_metadata(const uint8_t* data, size_t size, APICGraphInternal* graph);
bool apic_parse_operations(const uint8_t* data, size_t size, APICGraphInternal* graph);
bool apic_parse_memory_regions(const uint8_t* data, size_t size, APICGraphInternal* graph);

// Reconstruct mesh objects from serialized mesh records. Dispatches to
// wp_mesh_create_host or wp_mesh_create_device based on graph->device_type.
bool apic_create_meshes(APICGraphInternal* graph);

// Free CPU-side graph resources: destroy host meshes, free host region memory.
// Defined in apic.cpp; called from the CPU branch of APICGraphInternal's
// destructor (on both CUDA and non-CUDA builds).
void apic_destroy_cpu_graph_resources(APICGraphInternal* graph);

// Allocate host memory for each region and initialize from memory_ptr (the
// raw bytes of the memory section of a .wrp file). Called from the CPU branch
// of wp_apic_load_graph. Returns false on allocation failure.
bool apic_init_cpu_graph_memory(APICGraphInternal* graph, const uint8_t* memory_ptr, size_t memory_size);

// Register a CPU mesh (host pointers) with the APIC state: registers
// points/indices/(velocities) as memory regions with initial data and appends
// an APICMeshRecord. mesh_id here is the address of a wp::Mesh struct.
// Used by both the non-CUDA wp_apic_register_mesh (in apic.cpp) and the
// CUDA-build version (in apic.cu) when the descriptor lookup fails.
void apic_register_cpu_mesh(APICState state, uint64_t mesh_id);

// Fix up handle pointers (mesh IDs etc.) in memory regions after meshes are
// recreated during graph load. Defined in apic.cpp; the CPU path is fully
// implemented there, and in CUDA builds it forwards device handles through a
// private helper defined in apic.cu.
void apic_fixup_ptr_locations(APICGraphInternal* graph);

// ============================================================================
// CUDA helpers (declared here so apic.cpp can call them without including
// CUDA headers; defined in apic.cu and present only in CUDA builds).
// ============================================================================

#if WP_ENABLE_CUDA
// Handle-pointer fixup for a single offset within a CUDA region.
// Reads uint64_t at (base + offset), remaps via graph->handle_ptr_remap,
// writes back. Returns false on cudaMemcpy failure.
bool apic_fixup_handle_cuda(APICGraphInternal* graph, uint8_t* base, uint64_t offset);

// Host-to-device copy for wp_apic_set_param on CUDA graphs.
bool apic_set_param_cuda(APICGraphInternal* graph, void* dst, const void* src, size_t size);

// Device-to-host copy for wp_apic_get_param on CUDA graphs.
bool apic_get_param_cuda(APICGraphInternal* graph, void* dst, const void* src, size_t size);

// CUDA-side setup during wp_apic_load_graph: load CUmodules from the
// modules_dir and cudaMalloc each region, then apic_init_memory (H2D) copies
// initial data from the .wrp file into device memory. Returns false on
// failure; the caller is responsible for deleting the graph.
bool apic_load_graph_cuda_setup(
    APICGraphInternal* graph,
    void* context,
    const std::string& modules_dir,
    const uint8_t* memory_ptr,
    size_t memory_size
);
#endif
