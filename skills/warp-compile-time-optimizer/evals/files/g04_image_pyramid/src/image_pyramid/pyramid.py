# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public image pyramid assembly."""

from __future__ import annotations

import numpy as np

import warp as wp

from .filters import (
    configure_filtering,
    level_128_kernel,
    level_256_kernel,
    level_512_kernel,
    prepare_image_kernel,
)


class ImagePyramid:
    """Build three filtered image levels.

    Args:
        device: Device used for image processing.
    """

    def __init__(self, device: str = "cuda:0") -> None:
        self.device = device

    def _launch(self, kernel: object, values: object, filtered: object, *, block_dim: int) -> None:
        wp.launch(
            kernel,
            dim=filtered.shape,
            inputs=[values],
            outputs=[filtered],
            device=self.device,
            block_dim=block_dim,
        )

    def build(self, image: np.ndarray) -> tuple[np.ndarray, ...]:
        """Build and return three filtered, downsampled image levels."""
        source = np.asarray(image)
        if source.ndim != 2 or source.size == 0:
            raise ValueError("image must be a nonempty two-dimensional array")
        if source.dtype != np.float32:
            raise TypeError("image must use float32")
        values = wp.array(np.ascontiguousarray(source), dtype=wp.float32, device=self.device)
        prepared = wp.empty_like(values)
        self._launch(prepare_image_kernel, values, prepared, block_dim=128)
        configure_filtering()
        levels = []
        current = prepared
        shape = source.shape
        for kernel, block_dim in ((level_128_kernel, 128), (level_256_kernel, 256), (level_512_kernel, 512)):
            shape = ((shape[0] + 1) // 2, (shape[1] + 1) // 2)
            filtered = wp.empty(shape, dtype=wp.float32, device=self.device)
            self._launch(kernel, current, filtered, block_dim=block_dim)
            levels.append(filtered)
            current = filtered
        return tuple(level.numpy() for level in levels)
