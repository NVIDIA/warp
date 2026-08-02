# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Signal conditioning kernels and batch orchestration."""

from __future__ import annotations

import numpy as np

import warp as wp

from .runtime import configure_runtime

_BLOCK_DIM = 256


@wp.kernel(module="batch_signals.conditioning")
def detrend_kernel(samples: wp.array[wp.float32], mean: wp.float32) -> None:
    """Remove the batch mean from every sample."""
    index = wp.tid()
    samples[index] = samples[index] - mean


@wp.kernel(module="batch_signals.conditioning")
def deadband_kernel(samples: wp.array[wp.float32], threshold: wp.float32) -> None:
    """Zero samples whose magnitude sits inside the sensor deadband."""
    index = wp.tid()
    if wp.abs(samples[index]) < threshold:
        samples[index] = wp.float32(0.0)


@wp.kernel(module="batch_signals.conditioning")
def rescale_kernel(samples: wp.array[wp.float32], gain: wp.float32) -> None:
    """Apply the calibrated channel gain."""
    index = wp.tid()
    samples[index] = samples[index] * gain


def condition_batch(samples: np.ndarray, *, threshold: float, gain: float, device: str = "cpu") -> np.ndarray:
    """Condition one recorded batch and return a float32 NumPy array."""
    values = np.asarray(samples)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("batch must be a nonempty one-dimensional array")
    if values.dtype != np.float32:
        raise TypeError("batch must use float32 samples")

    configure_runtime()

    mean = np.float32(values.mean())
    buffer = wp.array(np.ascontiguousarray(values), dtype=wp.float32, device=device)
    wp.launch(detrend_kernel, dim=values.size, inputs=[buffer, mean], device=device, block_dim=_BLOCK_DIM)
    wp.launch(
        deadband_kernel,
        dim=values.size,
        inputs=[buffer, np.float32(threshold)],
        device=device,
        block_dim=_BLOCK_DIM,
    )
    wp.launch(rescale_kernel, dim=values.size, inputs=[buffer, np.float32(gain)], device=device, block_dim=_BLOCK_DIM)
    return buffer.numpy()
