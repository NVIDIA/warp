# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Calibration and serving entry points."""

from __future__ import annotations

import numpy as np

import warp as wp

from .kernels import TILE, calibrate_kernel, serve_kernel


def _check(matrix: np.ndarray, dtype: type, name: str) -> np.ndarray:
    values = np.asarray(matrix)
    if values.ndim != 2 or any(extent % TILE for extent in values.shape):
        raise ValueError(f"{name} must be two-dimensional with extents divisible by {TILE}")
    if values.dtype != dtype:
        raise TypeError(f"{name} must use {np.dtype(dtype).name}")
    return np.ascontiguousarray(values)


def calibrate(samples: np.ndarray, basis: np.ndarray, device: str = "cuda:0") -> np.ndarray:
    """Fit calibration coefficients. Runs once when the service starts."""
    left = _check(samples, np.float32, "samples")
    right = _check(basis, np.float32, "basis")
    if left.shape[1] != right.shape[0]:
        raise ValueError("sample columns must match basis rows")
    a = wp.array(left, dtype=wp.float32, device=device)
    b = wp.array(right, dtype=wp.float32, device=device)
    fitted = wp.empty((left.shape[0], right.shape[1]), dtype=wp.float32, device=device)
    wp.launch_tiled(
        calibrate_kernel,
        dim=(left.shape[0] // TILE, right.shape[1] // TILE),
        inputs=[a, b, fitted],
        device=device,
        block_dim=128,
    )
    return fitted.numpy()


def serve(activations: np.ndarray, weights: np.ndarray, device: str = "cuda:0") -> np.ndarray:
    """Serve one request batch. This is the hot path; it runs per request."""
    left = _check(activations, np.float16, "activations")
    right = _check(weights, np.float16, "weights")
    if left.shape[1] != right.shape[0]:
        raise ValueError("activation columns must match weight rows")
    a = wp.array(left, dtype=wp.float16, device=device)
    b = wp.array(right, dtype=wp.float16, device=device)
    output = wp.empty((left.shape[0], right.shape[1]), dtype=wp.float16, device=device)
    wp.launch_tiled(
        serve_kernel,
        dim=(left.shape[0] // TILE, right.shape[1] // TILE),
        inputs=[a, b, output],
        device=device,
        block_dim=128,
    )
    return output.numpy()
