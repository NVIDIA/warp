// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Stackful user-space fibers for cooperative scheduling within a CPU kernel
// block. Each logical "thread" in a `block_dim`-sized block is a fiber; they
// run on a single OS thread, swapping at `WP_TILE_SYNC` barrier points.
//
// API is libco-shaped: `wp_fiber_create` allocates a fiber with its own stack;
// `wp_fiber_switch` saves the caller's registers and resumes the target fiber.
// Per-arch context-switch trampolines live in `cpu_fiber_<arch>.S`.
//
// This header is included by `cpu_block_runtime.cpp`. JIT-compiled kernels do
// NOT see fibers directly ? they call `wp_cpu_tile_sync()` which the runtime
// turns into a fiber yield.

#pragma once

#include "crt.h"  // WP_API, size_t

#ifdef __cplusplus
extern "C" {
#endif

// Opaque fiber handle. Allocated by `wp_fiber_create`, freed by
// `wp_fiber_destroy`. Never dereferenced outside the fiber implementation.
typedef struct wp_fiber wp_fiber_t;

// Create a new fiber with its own stack. When the fiber is first switched to,
// `entry(arg)` is called on the fiber's stack. If `entry` returns, the fiber
// switches to the OS thread's main fiber and never resumes ? it must be
// destroyed via `wp_fiber_destroy`. Returning does not resume the fiber that
// most recently switched to this one.
//
// `stack_size` is rounded up to a page; a leading guard page (PROT_NONE) is
// added so stack overflow segfaults loudly instead of corrupting an adjacent
// fiber. Returns NULL on allocation failure.
wp_fiber_t* wp_fiber_create(void (*entry)(void* arg), void* arg, size_t stack_size);

// Free a fiber's stack and bookkeeping. Passing the currently-running fiber
// or the OS thread's implicit main fiber is invalid and has no effect.
void wp_fiber_destroy(wp_fiber_t* f);

// Save the calling context to the active fiber, switch to `to`. The caller is
// resumed when some other fiber switches back to it.
//
// Calling `wp_fiber_switch` from the OS thread's main context (before any
// fiber has been switched in) is fine ? the implementation lazily allocates a
// "main fiber" for the OS thread on first switch. On Windows, the switch has
// no effect if converting the calling thread to a fiber fails.
void wp_fiber_switch(wp_fiber_t* to);

// Returns the currently-running fiber. On first call from an OS thread that
// has never switched, returns the lazily-allocated "main" fiber for that
// thread. Returns NULL on Windows if converting the calling thread to a fiber
// fails.
wp_fiber_t* wp_fiber_active(void);

// Returns 1 if the fiber's `entry` has returned (it's been "consumed"), else 0.
// Useful for the scheduler to know when to stop scheduling a fiber.
int wp_fiber_finished(wp_fiber_t* f);

#ifdef __cplusplus
}  // extern "C"
#endif
