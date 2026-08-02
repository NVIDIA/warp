# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Finite-difference boundary kernels."""

from __future__ import annotations

import warp as wp


@wp.kernel(module="pde_stepper.boundary.zero")
def zero_boundary_kernel(values: wp.array2d[wp.float32]) -> None:
    """Set the one-cell exterior ring to a zero Dirichlet boundary."""
    row, column = wp.tid()
    if row == 0 or row == values.shape[0] - 1 or column == 0 or column == values.shape[1] - 1:
        values[row, column] = wp.float32(0.0)


@wp.kernel(module="pde_stepper.boundary.periodic")
def periodic_boundary_kernel(values: wp.array2d[wp.float32]) -> None:
    """Refresh the one-cell exterior ring from opposite interior cells."""
    row, column = wp.tid()
    if row == 0 or row == values.shape[0] - 1 or column == 0 or column == values.shape[1] - 1:
        source_row = values.shape[0] - 2 if row == 0 else 1 if row == values.shape[0] - 1 else row
        source_column = values.shape[1] - 2 if column == 0 else 1 if column == values.shape[1] - 1 else column
        values[row, column] = values[source_row, source_column]


def boundary_kernel(boundary: str) -> object:
    """Return the kernel for the requested ghost-cell boundary."""
    if boundary == "zero":
        return zero_boundary_kernel
    if boundary == "periodic":
        return periodic_boundary_kernel
    raise ValueError("boundary must be 'zero' or 'periodic'")
