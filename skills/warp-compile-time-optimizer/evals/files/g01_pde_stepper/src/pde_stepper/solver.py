# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public orchestration for two-dimensional finite-difference stepping."""

from __future__ import annotations

import numpy as np

import warp as wp

from .boundaries import boundary_kernel
from .operators import (
    approximate_step_kernel,
    configure_step_mode,
    exact_step_kernel,
    finalize_kernel,
    preprocess_kernel,
)

_BLOCK_DIM = 64


class PDEStepper:
    """Advance a float32 two-dimensional diffusion field on a Warp device.

    Args:
        mode: ``"exact"`` selects strict evaluation; ``"approximate"`` permits
            relaxed numerical evaluation.
        boundary: Either ``"zero"`` for a Dirichlet ring or ``"periodic"``
            for a periodic ghost ring.
        device: Warp device for every finite-difference launch.
    """

    def __init__(self, mode: str, boundary: str, device: str = "cuda:0") -> None:
        if mode not in {"exact", "approximate"}:
            raise ValueError("mode must be 'exact' or 'approximate'")
        self.mode = mode
        self.boundary = boundary
        self.device = device
        self._boundary_kernel = boundary_kernel(boundary)

    def _launch(self, kernel: object, source: object, result: object) -> None:
        wp.launch(
            kernel,
            dim=source.shape,
            inputs=[source],
            outputs=[result],
            device=self.device,
            block_dim=_BLOCK_DIM,
        )

    def _apply_boundary(self, values: object) -> None:
        wp.launch(
            self._boundary_kernel,
            dim=values.shape,
            outputs=[values],
            device=self.device,
            block_dim=_BLOCK_DIM,
        )

    def step(self, field: np.ndarray, iterations: int) -> np.ndarray:
        """Advance ``field`` for ``iterations`` and return a float32 NumPy array."""
        values = np.asarray(field)
        if values.ndim != 2 or min(values.shape) < 3:
            raise ValueError("field must be a two-dimensional array with both dimensions at least three")
        if values.dtype != np.float32:
            raise TypeError("field must use float32 values")
        if iterations < 0:
            raise ValueError("iterations must be nonnegative")
        source = wp.array(np.ascontiguousarray(values), dtype=wp.float32, device=self.device)
        current = wp.empty_like(source)
        advanced = wp.empty_like(source)
        result = wp.empty_like(source)
        self._launch(preprocess_kernel, source, current)
        configure_step_mode(self.mode == "approximate")
        step_kernel = exact_step_kernel if self.mode == "exact" else approximate_step_kernel
        for _ in range(iterations):
            self._launch(step_kernel, current, advanced)
            self._apply_boundary(advanced)
            current, advanced = advanced, current
        self._launch(finalize_kernel, current, result)
        return result.numpy()
