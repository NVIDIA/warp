# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels for stable two-dimensional finite-difference work."""

from __future__ import annotations

import warp as wp

_DIFFUSION = wp.float32(0.125)


@wp.kernel(module="pde_stepper.preprocess")
def preprocess_kernel(field: wp.array2d[wp.float32], values: wp.array2d[wp.float32]) -> None:
    """Copy the application field into the device-owned stepping buffer."""
    row, column = wp.tid()
    values[row, column] = field[row, column]


@wp.kernel
def exact_step_kernel(values: wp.array2d[wp.float32], advanced: wp.array2d[wp.float32]) -> None:
    """Advance interior cells using IEEE CUDA math."""
    row, column = wp.tid()
    if row > 0 and row < values.shape[0] - 1 and column > 0 and column < values.shape[1] - 1:
        center = values[row, column]
        laplacian = (
            values[row - 1, column]
            + values[row + 1, column]
            + values[row, column - 1]
            + values[row, column + 1]
            - wp.float32(4.0) * center
        )
        advanced[row, column] = center + _DIFFUSION * laplacian


@wp.kernel
def approximate_step_kernel(values: wp.array2d[wp.float32], advanced: wp.array2d[wp.float32]) -> None:
    """Advance interior cells with the approximate stepping mode."""
    row, column = wp.tid()
    if row > 0 and row < values.shape[0] - 1 and column > 0 and column < values.shape[1] - 1:
        center = values[row, column]
        laplacian = (
            values[row - 1, column]
            + values[row + 1, column]
            + values[row, column - 1]
            + values[row, column + 1]
            - wp.float32(4.0) * center
        )
        advanced[row, column] = center + _DIFFUSION * laplacian


@wp.kernel(module="pde_stepper.finalize")
def finalize_kernel(values: wp.array2d[wp.float32], result: wp.array2d[wp.float32]) -> None:
    """Copy the final device buffer into the returned array."""
    row, column = wp.tid()
    result[row, column] = values[row, column]


def configure_step_mode(approximate: bool) -> None:
    """Configure the requested numerical mode."""
    wp.set_module_options({"fast_math": approximate})
