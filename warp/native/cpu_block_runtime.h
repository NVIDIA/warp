// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "crt.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*wp_cpu_block_lane_fn)(void* dim, size_t block_id, int lane, void* args);

typedef struct wp_cpu_block_runtime_api {
    int (*get_thread_idx)();
    int (*get_active_count)();
    int (*get_first_active_lane)();
    void (*tile_sync)();
    int (*run_block)(
        int block_dim, int active_count, wp_cpu_block_lane_fn kernel_fn, void* dim, size_t block_id, void* args
    );
} wp_cpu_block_runtime_api;

// Fiber-only lane state. Generated one-lane kernels never reference these
// functions, allowing their accesses and barriers to disappear completely.
int wp_cpu_get_thread_idx();
int wp_cpu_get_active_count();
// Elect the lowest-numbered lane that has not returned. Cooperative callers
// should query this after a barrier before publishing a shared result. Returns
// zero outside a cooperative block.
int wp_cpu_get_first_active_lane();
void wp_cpu_tile_sync();

// Run the active prefix of one logical CPU block. Returns nonzero on success
// and records an error through Warp's native error channel on failure.
int wp_cpu_run_block(
    int block_dim, int active_count, wp_cpu_block_lane_fn kernel_fn, void* dim, size_t block_id, void* args
);

// Consume the calling thread's recoverable block-dispatch error.
// Native/Python launch bridges call this before and after invoking the
// generated void kernel ABI.
WP_API const char* wp_take_cpu_block_error();

// Return the core runtime entry points that warp-clang binds into JIT-compiled
// cooperative kernels. Keeping the scheduler in one library also keeps its
// thread-local lane, fiber-pool, and error state in one place.
WP_API const wp_cpu_block_runtime_api* wp_cpu_block_runtime_get_api();

#ifdef __cplusplus
}  // extern "C"
#endif
