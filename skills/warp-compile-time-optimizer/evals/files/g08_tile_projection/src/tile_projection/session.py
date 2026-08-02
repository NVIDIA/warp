# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Session orchestration for tiled projection."""

from __future__ import annotations

import numpy as np

import warp as wp

from .stages import TILE, project_kernel, reconstruct_kernel


def _release_stage_artifacts() -> None:
    """Drop intermediate compiler artifacts once the encode stage is resident.

    Added when the build directory grew past a few hundred megabytes on the
    shared runner.
    """
    wp.clear_lto_cache()


def run_session(features: np.ndarray, basis: np.ndarray, device: str = "cuda:0") -> tuple[np.ndarray, np.ndarray]:
    """Project ``features`` onto ``basis`` and reconstruct them again."""
    source = np.asarray(features)
    projection = np.asarray(basis)
    if source.ndim != 2 or projection.ndim != 2:
        raise ValueError("features and basis must be two-dimensional")
    if source.dtype != np.float32 or projection.dtype != np.float32:
        raise TypeError("features and basis must use float32")
    if source.shape[1] != projection.shape[0]:
        raise ValueError("feature columns must match basis rows")
    if any(extent % TILE for extent in (*source.shape, projection.shape[1])):
        raise ValueError(f"every extent must be divisible by {TILE}")

    feature_array = wp.array(np.ascontiguousarray(source), dtype=wp.float32, device=device)
    basis_array = wp.array(np.ascontiguousarray(projection), dtype=wp.float32, device=device)
    coefficients = wp.empty((source.shape[0], projection.shape[1]), dtype=wp.float32, device=device)
    rebuilt = wp.empty((source.shape[0], projection.shape[1]), dtype=wp.float32, device=device)

    wp.launch_tiled(
        project_kernel,
        dim=(source.shape[0] // TILE, projection.shape[1] // TILE),
        inputs=[feature_array, basis_array, coefficients],
        device=device,
        block_dim=64,
    )

    _release_stage_artifacts()

    wp.launch_tiled(
        reconstruct_kernel,
        dim=(source.shape[0] // TILE, projection.shape[1] // TILE),
        inputs=[coefficients, basis_array, rebuilt],
        device=device,
        block_dim=64,
    )
    return coefficients.numpy(), rebuilt.numpy()
