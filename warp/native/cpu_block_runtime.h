// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "crt.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*wp_cpu_block_lane_fn)(void* dim, size_t block_id, int lane, void* args);

// Fiber-only lane state. Generated one-lane kernels never reference these
// functions, allowing their accesses and barriers to disappear completely.
WP_API int wp_cpu_get_thread_idx();
WP_API int wp_cpu_get_active_count();
// Elect the lowest-numbered lane that has not returned. Cooperative callers
// should query this after a barrier before publishing a shared result. Returns
// zero outside a cooperative block.
WP_API int wp_cpu_get_first_active_lane();
WP_API void wp_cpu_tile_sync();

// Run the active prefix of one logical CPU block. Returns nonzero on success
// and records an error through Warp's native error channel on failure.
WP_API int wp_cpu_run_block(
    int block_dim, int active_count, wp_cpu_block_lane_fn kernel_fn, void* dim, size_t block_id, void* args
);

// Clear or consume the calling thread's recoverable block-dispatch error.
// Native/Python launch bridges use these around the generated void kernel ABI.
WP_API void wp_cpu_block_error_clear();
WP_API const char* wp_cpu_block_error_take();

// Return the number of reusable worker fibers allocated by this thread.
// This is an internal diagnostic used by the CPU block runtime tests.
WP_API size_t wp_cpu_block_pool_size();

// Make the next worker-pool growth fail. Internal test hook only.
WP_API void wp_cpu_test_fail_next_worker_allocation();

// Native scheduler probe used by ``warp/tests/test_cpu_block_runtime.py``.
// Barrier arrivals are encoded as ``lane`` and lane completion as
// ``active_count + lane`` in ``events``.
WP_API int wp_cpu_test_schedule(
    int block_dim,
    int active_count,
    const uint32_t* barrier_counts,
    int* events,
    size_t event_capacity,
    size_t* event_count
);

#ifdef __cplusplus
}  // extern "C"
#endif
