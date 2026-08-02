# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integer polarity diagnostics for analytics series."""

from __future__ import annotations

import numpy as np

import warp as wp


@wp.kernel(module="gpu_analytics.diagnostics", enable_backward=False)
def polarity_kernel(values: wp.array[wp.int16], counts: wp.array[wp.int32]) -> None:
    """Count negative, zero, and positive values."""
    index = wp.tid()
    if values[index] < 0:
        wp.atomic_add(counts, 0, 1)
    elif values[index] == 0:
        wp.atomic_add(counts, 1, 1)
    else:
        wp.atomic_add(counts, 2, 1)


def diagnose(values: np.ndarray, device: str = "cuda:0") -> np.ndarray:
    """Return negative, zero, and positive counts for int16 values."""
    source = np.asarray(values)
    if source.ndim != 1 or source.size == 0 or source.dtype != np.int16:
        raise TypeError("diagnostic values must be a nonempty one-dimensional int16 array")
    integer_values = wp.array(np.ascontiguousarray(source), dtype=wp.int16, device=device)
    counts = wp.zeros(3, dtype=wp.int32, device=device)
    wp.launch(
        polarity_kernel,
        dim=source.size,
        inputs=[integer_values],
        outputs=[counts],
        device=device,
        block_dim=128,
    )
    return counts.numpy()
