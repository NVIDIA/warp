# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Forward-only inspection report over a scalar field.

Nothing here differentiates: the report is a set of summary statistics. This is
the path that runs in batch and pays the startup cost.
"""

from __future__ import annotations

import numpy as np

import warp as wp

from .kernels import (
    STENCIL_TAPS,
    advect_field,
    apply_stencil,
    clamp_field,
    curvature_field,
    diffuse_field,
    energy_of_field,
    gradient_field,
    rescale_field,
    seed_field,
    smooth_field,
)

BLOCK_DIM = 256


def inspect_field(size: int = 4096, device: str = "cuda:0") -> dict[str, float]:
    """Return summary statistics for a generated field.

    Args:
        size: Number of samples in the field.
        device: Warp device to run on.

    Returns:
        Mapping of statistic name to value.
    """
    if size <= 0:
        raise ValueError("size must be positive")

    field = wp.zeros(size, dtype=wp.float32, device=device)
    scratch = wp.zeros(size, dtype=wp.float32, device=device)
    velocity = wp.zeros(size, dtype=wp.float32, device=device)
    energy = wp.zeros(1, dtype=wp.float32, device=device)
    taps = wp.array(
        np.full(STENCIL_TAPS, 1.0 / STENCIL_TAPS, dtype=np.float32),
        dtype=wp.float32,
        device=device,
    )

    launch = {"device": device, "block_dim": BLOCK_DIM}
    wp.launch(seed_field, dim=size, inputs=[field, wp.float32(0.05)], **launch)
    wp.launch(smooth_field, dim=size, inputs=[field, scratch, 2], **launch)
    wp.launch(gradient_field, dim=size, inputs=[scratch, velocity, wp.float32(1.0)], **launch)
    wp.launch(advect_field, dim=size, inputs=[scratch, velocity, field, wp.float32(0.5)], **launch)
    wp.launch(diffuse_field, dim=size, inputs=[field, scratch, wp.float32(0.1), wp.float32(1.0)], **launch)
    wp.launch(apply_stencil, dim=size, inputs=[scratch, taps, field], **launch)
    wp.launch(curvature_field, dim=size, inputs=[field, scratch, wp.float32(1.0)], **launch)
    wp.launch(rescale_field, dim=size, inputs=[scratch, wp.float32(1.5), wp.float32(0.0)], **launch)
    wp.launch(clamp_field, dim=size, inputs=[scratch, field, wp.float32(2.0)], **launch)
    wp.launch(energy_of_field, dim=size, inputs=[field, energy], **launch)

    values = field.numpy()
    return {
        "energy": float(energy.numpy()[0]),
        "mean": float(values.mean()),
        "peak": float(np.abs(values).max()),
    }
