# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Paired floating-point analytics on CUDA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

import warp as wp

from .reductions import energy_kernel, sum_kernel


@wp.kernel(module="gpu_analytics.blocks")
def transform_128_kernel(values: wp.array[Any], scale: Any, offset: Any, transformed: wp.array[Any]) -> None:
    """Apply the first analytic transform."""
    index = wp.tid()
    transformed[index] = values[index] * scale + offset


@wp.kernel(module="gpu_analytics.blocks")
def transform_256_kernel(values: wp.array[Any], scale: Any, offset: Any, transformed: wp.array[Any]) -> None:
    """Apply the second analytic transform."""
    index = wp.tid()
    transformed[index] = values[index] * scale + offset


@wp.kernel(module="gpu_analytics.blocks")
def transform_512_kernel(values: wp.array[Any], scale: Any, offset: Any, transformed: wp.array[Any]) -> None:
    """Apply the third analytic transform."""
    index = wp.tid()
    transformed[index] = values[index] * scale + offset


@dataclass(frozen=True)
class AnalyticsResult:
    """Transformed series and reductions for two numeric series.

    Attributes:
        transformed: Arrays ordered by scalar type and transform size.
        totals: Sum for each input series.
        energies: Sum of squares for each input series.
    """

    transformed: tuple[np.ndarray, ...]
    totals: tuple[np.generic, np.generic]
    energies: tuple[np.generic, np.generic]


class AnalyticsBank:
    """Run paired floating-point transforms and reductions.

    Args:
        device: CUDA device used for the analytics arrays.
    """

    def __init__(self, device: str = "cuda:0") -> None:
        self.device = device

    def run(self, values: tuple[np.ndarray, np.ndarray]) -> AnalyticsResult:
        """Return transforms and reductions for float32 and float64 series."""
        values32, values64 = _validate_values(values)
        result32 = self._run_dtype(values32, block_dims=(128, 256, 512))
        result64 = self._run_dtype(values64, block_dims=(128, 256, 512))
        transformed = result32[0] + result64[0]
        return AnalyticsResult(transformed, (result32[1], result64[1]), (result32[2], result64[2]))

    def _run_dtype(
        self, values: np.ndarray, *, block_dims: tuple[int, int, int]
    ) -> tuple[tuple[np.ndarray, ...], np.generic, np.generic]:
        dtype = _warp_dtype(values)
        scalar = values.dtype.type
        source = wp.array(np.ascontiguousarray(values), dtype=dtype, device=self.device)
        transformed = []
        stages = (
            (transform_128_kernel, block_dims[0], scalar(1.25), scalar(0.5)),
            (transform_256_kernel, block_dims[1], scalar(-0.75), scalar(0.125)),
            (transform_512_kernel, block_dims[2], scalar(0.5), scalar(-0.25)),
        )
        for kernel, block_dim, scale, offset in stages:
            result = wp.empty(values.size, dtype=dtype, device=self.device)
            wp.launch(
                kernel,
                dim=values.size,
                inputs=[source, dtype(scale), dtype(offset)],
                outputs=[result],
                device=self.device,
                block_dim=block_dim,
            )
            transformed.append(result)
        total = wp.zeros(1, dtype=dtype, device=self.device)
        energy = wp.zeros(1, dtype=dtype, device=self.device)
        wp.launch(sum_kernel, dim=values.size, inputs=[source], outputs=[total], device=self.device, block_dim=128)
        wp.launch(energy_kernel, dim=values.size, inputs=[source], outputs=[energy], device=self.device, block_dim=256)
        return tuple(item.numpy() for item in transformed), total.numpy()[0], energy.numpy()[0]


def _validate_values(values: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(values, tuple) or len(values) != 2:
        raise TypeError("analytics values must be a pair of floating-point arrays")
    values32, values64 = (np.asarray(item) for item in values)
    for series, dtype in ((values32, np.float32), (values64, np.float64)):
        if series.ndim != 1 or series.size == 0:
            raise ValueError("analytics values must be nonempty one-dimensional arrays")
        if series.dtype != dtype:
            raise TypeError("analytics values must contain float32 then float64 arrays")
    return values32, values64


def _warp_dtype(values: np.ndarray) -> object:
    if values.dtype == np.float32:
        return wp.float32
    if values.dtype == np.float64:
        return wp.float64
    raise TypeError("analytics values must use float32 or float64")
