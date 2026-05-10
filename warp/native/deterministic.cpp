// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// CPU stubs for deterministic mode.
// CPU kernels execute sequentially so atomics are already deterministic.
// These stubs satisfy the linker when building without CUDA.

#include "warp.h"

#if !WP_ENABLE_CUDA

size_t
wp_deterministic_sort_reduce_workspace_size(int count, int op, int scalar_type, int components, int determinism_level)
{
    return 0;
}

void wp_deterministic_sort_reduce_device(
    uint64_t keys,
    uint64_t values,
    int count,
    uint64_t dest_array,
    int dest_size,
    int op,
    int scalar_type,
    int components,
    int determinism_level,
    uint64_t workspace,
    size_t workspace_size
)
{
}

void wp_deterministic_counter_total_device(uint64_t contrib, uint64_t prefix, int count, uint64_t counter) { }

size_t wp_deterministic_counter_scan_workspace_size(int count) { return 0; }

void wp_deterministic_counter_scan_device(
    uint64_t keys,
    uint64_t values,
    int count,
    uint64_t prefixes,
    uint64_t counters,
    int counter_size,
    uint64_t workspace,
    size_t workspace_size
)
{
}

#endif  // !WP_ENABLE_CUDA
