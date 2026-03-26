# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Auxiliary module for testing AOT compilation with mixed generic kernels.

This module contains:
- A generic kernel WITH overloads that should compile successfully
- A generic kernel WITHOUT overloads that should trigger a warning but not prevent compilation
"""

from typing import Any

import warp as wp


@wp.kernel
def scale_with_overloads(x: wp.array(dtype=Any), s: Any):
    """Generic kernel with overloads - should compile successfully."""
    i = wp.tid()
    x[i] = s * x[i]


# Define overloads for scale_with_overloads
scale_f32 = wp.overload(scale_with_overloads, [wp.array(dtype=wp.float32), wp.float32])
scale_f64 = wp.overload(scale_with_overloads, [wp.array(dtype=wp.float64), wp.float64])


@wp.kernel
def multiply_without_overloads(x: wp.array(dtype=Any), y: wp.array(dtype=Any), result: wp.array(dtype=Any)):
    """Generic kernel without overloads - should trigger a warning."""
    i = wp.tid()
    result[i] = x[i] * y[i]
