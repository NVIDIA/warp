# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public orchestration for incremental procedural stencil analysis."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

import warp as wp

from .stencils import DOCUMENTED_RADII, make_stencil_kernel

_BLOCK_DIM = 64


class StencilBank:
    """Apply documented one-dimensional box stencils in the supplied order.

    Args:
        radii: One or more documented stencil radii.
        device: CUDA device used for every analysis launch.
    """

    def __init__(self, radii: Iterable[int], device: str = "cuda:0") -> None:
        self.radii = tuple(radii)
        if not self.radii or any(radius not in DOCUMENTED_RADII for radius in self.radii):
            raise ValueError(f"radii must be a nonempty subset of {DOCUMENTED_RADII}")
        self.device = device

    def _launch(self, kernel: object, values: object, filtered: object) -> None:
        wp.launch(
            kernel,
            dim=values.shape[0],
            inputs=[values],
            outputs=[filtered],
            device=self.device,
            block_dim=_BLOCK_DIM,
        )

    def analyze(self, values: np.ndarray) -> np.ndarray:
        """Apply each radius incrementally and return float32 host values."""
        source_values = np.asarray(values)
        if source_values.ndim != 1 or source_values.size == 0:
            raise ValueError("values must be a nonempty one-dimensional array")
        if source_values.dtype != np.float32:
            raise TypeError("values must use float32")
        current = wp.array(np.ascontiguousarray(source_values), dtype=wp.float32, device=self.device)
        filtered = wp.empty_like(current)
        for radius in self.radii:
            kernel = make_stencil_kernel(radius, max_unroll=16, enable_backward=True)
            self._launch(kernel, current, filtered)
            current, filtered = filtered, current
        return current.numpy()
