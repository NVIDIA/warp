# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA kernels used to check tile-reduction compiler diagnostics."""

import warp as wp


@wp.kernel(enable_backward=False)
def reduce_vec3_tile(values: wp.array2d[wp.vec3], out: wp.array[wp.vec3]):
    tile = wp.tile_load(values, shape=(8, 8))
    wp.tile_store(out, wp.tile_sum(tile))


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
    __shared__ wp::vec_t<3, wp::float32> scratch;
    if (threadIdx.x == 0)
        scratch = wp::vec_t<3, wp::float32>(1.0f);
    __syncthreads();
    return scratch;
#else
    return wp::vec_t<3, wp::float32>(1.0f);
#endif
"""
)
def unsupported_dynamic_shared_value() -> wp.vec3:
    """Emit a dynamic shared-variable initialization as a diagnostic control."""
    ...


@wp.kernel(enable_backward=False, module="unique")
def compiler_warning_control(out: wp.array[wp.vec3]):
    out[wp.tid()] = unsupported_dynamic_shared_value()
