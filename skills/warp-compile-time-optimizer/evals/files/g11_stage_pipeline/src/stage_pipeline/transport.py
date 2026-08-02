# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transport stage: advection and relaxation on a one-dimensional grid."""

from __future__ import annotations

import warp as wp

# This module is always launched at BLOCK_DIM, so declare it as the module
# default too: a preload that does not name a block dimension then builds the
# variant this stage actually uses instead of a second one.
wp.set_module_options({"enable_backward": False, "block_dim": 512})

BLOCK_DIM = 512


@wp.kernel
def upwind_advect(
    field: wp.array[wp.float32],
    velocity: wp.array[wp.float32],
    result: wp.array[wp.float32],
    step: wp.float32,
) -> None:
    """First-order upwind advection."""
    index = wp.tid()
    last = field.shape[0] - 1
    speed = velocity[index]
    if speed > 0.0:
        upstream = field[wp.max(index - 1, 0)]
    else:
        upstream = field[wp.min(index + 1, last)]
    result[index] = field[index] - step * speed * (field[index] - upstream)


@wp.kernel
def relax_field(field: wp.array[wp.float32], result: wp.array[wp.float32], weight: wp.float32) -> None:
    """Jacobi relaxation toward the neighbour average."""
    index = wp.tid()
    last = field.shape[0] - 1
    lower = field[wp.max(index - 1, 0)]
    upper = field[wp.min(index + 1, last)]
    average = wp.float32(0.5) * (lower + upper)
    result[index] = field[index] * (wp.float32(1.0) - weight) + average * weight


@wp.kernel
def clamp_flux(field: wp.array[wp.float32], result: wp.array[wp.float32], limit: wp.float32) -> None:
    """Limit the field to a physical range."""
    index = wp.tid()
    result[index] = wp.clamp(field[index], -limit, limit)


@wp.kernel
def accumulate_flux(field: wp.array[wp.float32], total: wp.array[wp.float32]) -> None:
    """Sum the absolute flux across the grid."""
    index = wp.tid()
    wp.atomic_add(total, 0, wp.abs(field[index]))
