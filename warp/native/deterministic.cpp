// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// CPU stubs for deterministic mode.
// CPU kernels execute sequentially so atomics are already deterministic.
// These stubs satisfy the linker when building without CUDA.

#include "warp.h"

#if !WP_ENABLE_CUDA

void wp_deterministic_sort_reduce_device(
    uint64_t keys,
    uint64_t values,
    int count,
    uint64_t dest_array,
    int dest_size,
    int op,
    int scalar_type,
    int components
)
{
}

#endif  // !WP_ENABLE_CUDA
