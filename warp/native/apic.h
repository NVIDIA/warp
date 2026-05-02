// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// APIC (API Capture) - Graph recording, serialization, and replay.
//
// Operations are recorded directly in C++ via hooks in wp_cuda_launch_kernel(),
// wp_cpu_launch_kernel(), and the memcpy/memset functions. Python builds
// APICLaunchInfo structs per kernel launch but does NOT maintain its own
// recording. The C++ byte stream is the single source of truth.
//
// Three consumption paths from the byte stream:
// 1. wp_apic_state_save()         -> .wrp file (serialization)
// 2. wp_apic_cpu_replay_state()   -> executes a live CPU capture
// 3. wp_apic_cpu_replay_graph() / wp_apic_get_cuda_graph()
//                                 -> executes a loaded .wrp graph (CPU / CUDA)

#include "apic_types.h"

#include <stddef.h>
#include <stdint.h>

#ifndef WP_API
#ifdef _WIN32
#define WP_API __declspec(dllexport)
#else
#define WP_API __attribute__((visibility("default")))
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Recording State Management
// =============================================================================

// Opaque handle to APIC recording/serialization state
typedef struct APICStateInternal* APICState;

WP_API APICState wp_apic_create_state();
WP_API void wp_apic_destroy_state(APICState state);

// Begin/end recording: sets/clears thread-local g_apic_state
WP_API void wp_apic_begin_recording(APICState state);
WP_API void wp_apic_end_recording(APICState state);

// Access current thread-local recording state. Called from the launch /
// memcpy / memset hooks in warp.cu and warp.cpp to check whether a capture
// is active; returns nullptr when not capturing.
WP_API APICState wp_apic_get_recording_state();

// =============================================================================
// Memory Region Registration (called from Python)
//
// Region IDs are uint32_t in the registration API. Valid IDs start from 1
// (assigned by a monotonic counter); 0 indicates failure or absence.
// In the byte stream format and recording API, region_id is int32_t so that
// -1 can serve as a sentinel for null/absent arrays (see
// APICLaunchParamRecord::region_id). The recording functions should only be
// called with valid (non-zero) region IDs.
// =============================================================================

// Register a memory region by base pointer. Returns region_id.
// If the pointer is already registered, returns the existing region_id.
WP_API uint32_t
wp_apic_register_memory_region_by_ptr(APICState state, uint64_t base_ptr, uint64_t size, uint32_t element_size);

// Register a memory region with initial data (for serialization).
// Called from Python during capture_save().
WP_API void wp_apic_register_memory_region(
    APICState state, uint32_t region_id, uint64_t size, uint32_t element_size, const void* initial_data
);

// =============================================================================
// Metadata Registration (called from Python for serialization)
// =============================================================================

WP_API void wp_apic_register_module(
    APICState state, const char* module_hash, const char* module_name, const char* binary_filename, int target_arch
);

WP_API void wp_apic_register_kernel(
    APICState state,
    const char* kernel_key,
    const char* module_hash,
    const char* forward_name,
    const char* backward_name,
    int forward_smem_bytes,
    int backward_smem_bytes,
    int block_dim
);

WP_API void wp_apic_register_binding(APICState state, const char* name, uint32_t region_id);

WP_API void wp_apic_register_ptr_location(APICState state, uint32_t region_id, uint64_t offset, uint64_t stride);

// Register a mesh for serialization. Looks up the mesh by ID in the global
// mesh descriptor table, registers its arrays as memory regions with initial
// data, and stores an APICMeshRecord for the save path.
WP_API void wp_apic_register_mesh(APICState state, uint64_t mesh_id);

// =============================================================================
// Operation Recording (called from the kernel-launch / memcpy / memset hooks
// in warp.cu and warp.cpp when a capture is active)
// =============================================================================

WP_API void apic_record_kernel_launch(
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
);

WP_API void apic_record_memcpy_d2d(
    APICState state,
    int32_t dst_region_id,
    uint64_t dst_offset,
    int32_t src_region_id,
    uint64_t src_offset,
    uint64_t size
);

WP_API void apic_record_memset(APICState state, int32_t region_id, uint64_t offset, uint64_t size, int32_t value);

WP_API void apic_record_alloc(APICState state, int32_t region_id, uint64_t size);

// =============================================================================
// Pointer Resolution (for memcpy/memset hooks that receive raw pointers)
// =============================================================================

// Resolve a raw pointer to (region_id, byte_offset).
// Returns true on success, false if the pointer is not in any registered region.
WP_API bool apic_resolve_ptr(APICState state, uint64_t ptr, int32_t* out_region_id, uint64_t* out_offset);

// Opaque handle to a loaded .wrp graph (CPU or CUDA; see wp_apic_load_graph).
typedef struct APICGraphInternal* APICGraph;

// =============================================================================
// CPU Graph Replay (execute operations directly from the APIC byte stream)
// =============================================================================

// Register a CPU kernel function pointer during live capture
WP_API void wp_apic_register_cpu_kernel(APICState state, const char* kernel_key, void* forward_fn, void* backward_fn);

// Replay CPU operations from a live capture's byte stream
WP_API bool wp_apic_cpu_replay_state(APICState state);

// Replay CPU operations from a loaded graph's byte stream
WP_API bool wp_apic_cpu_replay_graph(APICGraph graph);

// =============================================================================
// Serialization: Save to .wrp file
// =============================================================================

// Returns true on success, false on failure
WP_API bool wp_apic_state_save(APICState state, const char* path, int target_arch);

// State queries
WP_API uint32_t wp_apic_get_operation_count(APICState state);

// =============================================================================
// Loading and Execution (standalone, no Python needed)
// =============================================================================

// device_type: APICDeviceType value (APIC_DEVICE_CUDA=0, APIC_DEVICE_CPU=1).
// context is ignored for APIC_DEVICE_CPU.
WP_API APICGraph wp_apic_load_graph(void* context, const char* path, int device_type);
WP_API void wp_apic_destroy_graph(APICGraph graph);

WP_API bool wp_apic_set_param(APICGraph graph, const char* name, const void* data, size_t size);
WP_API bool wp_apic_get_param(APICGraph graph, const char* name, void* data, size_t size);
WP_API void* wp_apic_get_param_ptr(APICGraph graph, const char* name);

WP_API void* wp_apic_get_cuda_graph(APICGraph graph);
WP_API void* wp_apic_get_cuda_graph_exec(APICGraph graph);

WP_API int wp_apic_get_num_params(APICGraph graph);
WP_API const char* wp_apic_get_param_name(APICGraph graph, int index);
WP_API size_t wp_apic_get_param_size(APICGraph graph, const char* name);

WP_API bool wp_apic_launch(APICGraph graph, void* stream);

// CPU kernel resolution for loaded graphs (called from Python after loading .o modules)
WP_API void
wp_apic_register_loaded_cpu_kernel(APICGraph graph, const char* kernel_key, void* forward_fn, void* backward_fn);

// Query loaded graph kernel metadata (for CPU module loading)
WP_API int wp_apic_get_num_kernels(APICGraph graph);
WP_API const char* wp_apic_get_kernel_key(APICGraph graph, int index);
WP_API const char* wp_apic_get_kernel_forward_name(APICGraph graph, const char* kernel_key);
WP_API const char* wp_apic_get_kernel_backward_name(APICGraph graph, const char* kernel_key);
WP_API const char* wp_apic_get_kernel_module_hash(APICGraph graph, const char* kernel_key);

#ifdef __cplusplus
}  // extern "C"
#endif
