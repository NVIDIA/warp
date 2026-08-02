# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image smoothing for varying image sizes."""

from __future__ import annotations

import numpy as np

import warp as wp


@wp.kernel(module="image_pyramid.filters")
def adaptive_filter_kernel(values: wp.array2d[wp.float32], filtered: wp.array2d[wp.float32]) -> None:
    """Smooth every pixel with clamped image borders."""
    row, column = wp.tid()
    total = wp.float32(0.0)
    for row_offset in range(-1, 2):
        for column_offset in range(-1, 2):
            sample_row = wp.max(0, wp.min(row + row_offset, values.shape[0] - 1))
            sample_column = wp.max(0, wp.min(column + column_offset, values.shape[1] - 1))
            total += values[sample_row, sample_column]
    filtered[row, column] = total / wp.float32(9.0)


def adaptive_filter(image: np.ndarray, device: str = "cuda:0") -> np.ndarray:
    """Return a smoothed float32 image."""
    source = np.asarray(image)
    if source.ndim != 2 or source.size == 0:
        raise ValueError("image must be a nonempty two-dimensional array")
    if source.dtype != np.float32:
        raise TypeError("image must use float32")
    values = wp.array(np.ascontiguousarray(source), dtype=wp.float32, device=device)
    filtered = wp.empty_like(values)
    block_dim = 64 if source.size <= 256 else 192
    wp.launch(
        adaptive_filter_kernel,
        dim=source.shape,
        inputs=[values],
        outputs=[filtered],
        device=device,
        block_dim=block_dim,
    )
    return filtered.numpy()
