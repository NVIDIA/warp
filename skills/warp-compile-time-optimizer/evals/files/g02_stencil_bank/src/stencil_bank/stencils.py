# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Factory-created Warp kernels for documented stencil radii."""

from __future__ import annotations

import warp as wp

DOCUMENTED_RADII = (1, 2, 4, 8, 24)


def make_stencil_kernel(radius: int, *, max_unroll: int = 16, enable_backward: bool = True) -> object:
    """Create one clamped box-stencil kernel for a documented ``radius``."""
    if radius not in DOCUMENTED_RADII:
        raise ValueError(f"radius must be one of {DOCUMENTED_RADII}")

    def stencil_kernel(values: wp.array[wp.float32], filtered: wp.array[wp.float32]) -> None:
        index = wp.tid()
        total = wp.float32(0.0)
        count = int(0)
        for offset in range(-radius, radius + 1):
            sample = index + offset
            if sample >= 0 and sample < values.shape[0]:
                total += values[sample]
                count += 1
        filtered[index] = total / wp.float32(count)

    kernel_name = f"stencil_radius_{radius}_kernel"
    stencil_kernel.__name__ = kernel_name
    stencil_kernel.__qualname__ = kernel_name
    return wp.kernel(
        stencil_kernel,
        module="unique",
        module_options={"max_unroll": max_unroll, "enable_backward": enable_backward},
    )
