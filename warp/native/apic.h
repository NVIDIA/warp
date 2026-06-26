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
//
// This header declares APIC's DLL-exported C API. Public functions use the
// WP_API export macro and `wp_` name prefix; internal helpers live in
// apic_internal.h.

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

extern "C" {

// =============================================================================
// Recording State Management
// =============================================================================

// Opaque APIC recording/serialization state. Defined in apic_internal.h.
struct APICState;

WP_API APICState* wp_apic_create_state();
WP_API void wp_apic_destroy_state(APICState* state);

// Begin/end recording: sets/clears thread-local g_apic_state. ``is_cpu`` marks
// whether the capture targets a CPU device, so device-mismatched host/device
// hooks can avoid recording into the wrong byte stream.
WP_API void wp_apic_begin_recording(APICState* state, int is_cpu);
WP_API void wp_apic_end_recording(APICState* state);

// Access the active recording state for HOST (CPU) capture hooks. Returns
// nullptr when not capturing OR when the active capture targets CUDA, so a
// host op invoked during a CUDA APIC capture is executed live instead of being
// recorded into the CUDA byte stream. Called from the host hooks in warp.cpp,
// sort.cpp, runlength_encode.cpp, and sparse.cpp.
WP_API APICState* wp_apic_get_recording_state();

// Access the active recording state for DEVICE (CUDA) capture hooks. Mirror of
// wp_apic_get_recording_state() for the CUDA side: returns nullptr when not
// capturing OR when the active capture targets CPU. Called from warp.cu.
WP_API APICState* wp_apic_get_cuda_recording_state();

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
wp_apic_register_memory_region_by_ptr(APICState* state, uint64_t base_ptr, uint64_t size, uint32_t element_size);

// Register a memory region with initial data (for serialization).
// Called from Python during capture_save().
WP_API void wp_apic_register_memory_region(
    APICState* state, uint32_t region_id, uint64_t size, uint32_t element_size, const void* initial_data
);

// =============================================================================
// Metadata Registration (called from Python for serialization)
// =============================================================================

WP_API void wp_apic_register_module(
    APICState* state, const char* module_hash, const char* module_name, const char* binary_filename, int target_arch
);

WP_API void wp_apic_register_kernel(
    APICState* state,
    const char* kernel_key,
    const char* module_hash,
    const char* forward_name,
    const char* backward_name,
    int forward_smem_bytes,
    int backward_smem_bytes,
    int block_dim
);

WP_API void wp_apic_register_binding(APICState* state, const char* name, uint32_t region_id);

WP_API void wp_apic_register_ptr_location(APICState* state, uint32_t region_id, uint64_t offset, uint64_t stride);

// Register a mesh for serialization. Looks up the mesh by ID in the global
// mesh descriptor table, registers its arrays as memory regions with initial
// data, and stores an APICMeshRecord for the save path.
WP_API void wp_apic_register_mesh(APICState* state, uint64_t mesh_id);

// =============================================================================
// Conditional / Loop ops (APIC_OP_IF / APIC_OP_WHILE)
// =============================================================================
//
// Python's wp.capture_if / wp.capture_while record conditional structure
// during APIC capture by:
//   1. Calling wp_apic_begin_branch(state) before the branch callback to
//      remember the current stream position.
//   2. Running the user callback (appends ops to the main stream).
//   3. Calling wp_apic_end_branch(state, start) to carve out the bytes
//      appended during the callback into an APICBranchBody, truncating the
//      main stream back to the saved position.
//   4. Optionally repeating for the else branch.
//   5. Calling wp_apic_record_conditional(...) to emit the APIC_OP_IF /
//      APIC_OP_WHILE record with the branch bodies embedded as branch_a /
//      branch_b. The bodies are consumed (freed) by the call.
//
// Replay reads the int32 condition from cond_region_id+cond_offset and
// executes the appropriate branch (or repeats branch_a for OP_WHILE).

// Opaque structs returned by begin_branch / end_branch. Defined in apic.cpp.
struct APICBranchStart;  // begin-time stream position
struct APICBranchBody;  // captured branch sub-stream

WP_API APICBranchStart* wp_apic_begin_branch(APICState* state);
WP_API APICBranchBody* wp_apic_end_branch(APICState* state, APICBranchStart* start);
WP_API void wp_apic_free_branch_body(APICBranchBody* body);
WP_API void wp_apic_record_conditional(
    APICState* state,
    int op_type,
    int32_t cond_region_id,
    uint64_t cond_offset,
    APICBranchBody* branch_a,  // consumed (freed)
    APICBranchBody* branch_b  // consumed; may be NULL for OP_WHILE / IF without else
);

// Opaque loaded .wrp graph state (CPU or CUDA; see wp_apic_load_graph).
// Defined in apic_internal.h.
struct APICGraph;

// =============================================================================
// CPU Graph Replay (execute operations directly from the APIC byte stream)
// =============================================================================

// Register a CPU kernel function pointer during live capture. module_hash
// disambiguates same-key kernels compiled into distinct modules (module="unique").
WP_API void wp_apic_register_cpu_kernel(
    APICState* state, const char* kernel_key, const char* module_hash, void* forward_fn, void* backward_fn
);

// Replay CPU operations from a live capture's byte stream
WP_API bool wp_apic_cpu_replay_state(APICState* state);

// Replay CPU operations from a loaded graph's byte stream
WP_API bool wp_apic_cpu_replay_graph(APICGraph* graph);

// =============================================================================
// Serialization: Save to .wrp file
// =============================================================================

// Returns true on success, false on failure
WP_API bool wp_apic_state_save(APICState* state, const char* path, int target_arch);

// State queries
WP_API uint32_t wp_apic_get_operation_count(APICState* state);

// =============================================================================
// Loading and Execution (standalone, no Python needed)
// =============================================================================

// device_type: APICDeviceType value (APIC_DEVICE_CUDA=0, APIC_DEVICE_CPU=1).
// context is ignored for APIC_DEVICE_CPU.
WP_API APICGraph* wp_apic_load_graph(void* context, const char* path, int device_type);
WP_API void wp_apic_destroy_graph(APICGraph* graph);

WP_API bool wp_apic_set_param(APICGraph* graph, const char* name, const void* data, size_t size);
WP_API bool wp_apic_get_param(APICGraph* graph, const char* name, void* data, size_t size);
WP_API void* wp_apic_get_param_ptr(APICGraph* graph, const char* name);

WP_API void* wp_apic_get_cuda_graph(APICGraph* graph);
WP_API void* wp_apic_get_cuda_graph_exec(APICGraph* graph);

WP_API int wp_apic_get_num_params(APICGraph* graph);
WP_API const char* wp_apic_get_param_name(APICGraph* graph, int index);
WP_API size_t wp_apic_get_param_size(APICGraph* graph, const char* name);

WP_API bool wp_apic_launch(APICGraph* graph, void* stream);

// CPU kernel resolution for loaded graphs (called from Python after loading .o modules)
WP_API void wp_apic_register_loaded_cpu_kernel(
    APICGraph* graph, const char* kernel_key, const char* module_hash, void* forward_fn, void* backward_fn
);

// Query loaded graph kernel metadata (for CPU module loading)
WP_API int wp_apic_get_num_kernels(APICGraph* graph);
WP_API const char* wp_apic_get_kernel_key(APICGraph* graph, int index);
WP_API const char* wp_apic_get_kernel_module_hash(APICGraph* graph, int index);
WP_API const char* wp_apic_get_kernel_module_binary_filename(APICGraph* graph, int index);
WP_API const char* wp_apic_get_kernel_forward_name(APICGraph* graph, int index);
WP_API const char* wp_apic_get_kernel_backward_name(APICGraph* graph, int index);

}  // extern "C"
