# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public pipeline orchestration for particle preprocessing."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

import warp as wp

from .plugins import Plugin
from .stages import configure_stages, make_center_kernel, make_clip_kernel, make_scale_kernel


class ParticlePipeline:
    """Apply preprocessing stages followed by optional transforms.

    Args:
        center: Value subtracted from each input particle value.
        scale: Factor applied after centering.
        lower: Inclusive lower clipping bound.
        upper: Inclusive upper clipping bound.
        plugins: Additional transforms applied after clipping.
        device: Warp device used for the preprocessing launches.
    """

    def __init__(
        self,
        center: float,
        scale: float,
        lower: float,
        upper: float,
        plugins: Iterable[Plugin] = (),
        device: str = "cpu",
    ) -> None:
        if lower > upper:
            raise ValueError("lower bound must not exceed upper bound")
        self.center = float(center)
        self.scale = float(scale)
        self.lower = float(lower)
        self.upper = float(upper)
        self.plugins = tuple(plugins)
        self.device = device

    def _launch(self, kernel: object, values: object, *parameters: float) -> None:
        wp.launch(kernel, dim=values.shape[0], inputs=[values, *parameters], device=self.device)

    def run(self, values: np.ndarray) -> np.ndarray:
        """Run the pipeline and return a float32 NumPy array."""
        particle_values = wp.array(np.asarray(values, dtype=np.float32), dtype=wp.float32, device=self.device)
        center = make_center_kernel()
        self._launch(center, particle_values, self.center)
        configure_stages()
        scale = make_scale_kernel()
        self._launch(scale, particle_values, self.scale)
        clip = make_clip_kernel()
        self._launch(clip, particle_values, self.lower, self.upper)
        for plugin in self.plugins:
            self._launch(plugin.make_kernel(), particle_values, plugin.amount)
        return particle_values.numpy()
