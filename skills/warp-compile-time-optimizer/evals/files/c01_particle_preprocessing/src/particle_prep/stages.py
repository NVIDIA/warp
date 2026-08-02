# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Warp kernels for particle preprocessing stages."""

from __future__ import annotations

import warp as wp


def make_center_kernel(*, module: object = "unique") -> object:
    """Build a kernel that subtracts a center value from every particle."""

    @wp.kernel(module=module)
    def center_kernel(values: wp.array[float], center: float) -> None:
        index = wp.tid()
        values[index] = values[index] - center

    return center_kernel


def make_scale_kernel(*, module: object = "unique") -> object:
    """Build a kernel that scales every centered particle value."""

    @wp.kernel(module=module)
    def scale_kernel(values: wp.array[float], scale: float) -> None:
        index = wp.tid()
        values[index] = values[index] * scale

    return scale_kernel


def make_clip_kernel(*, module: object = "unique") -> object:
    """Build a kernel that clamps every scaled particle value."""

    @wp.kernel(module=module)
    def clip_kernel(values: wp.array[float], lower: float, upper: float) -> None:
        index = wp.tid()
        values[index] = wp.max(lower, wp.min(values[index], upper))

    return clip_kernel


def configure_stages() -> None:
    """Configure particle stage behavior."""
    wp.set_module_options({"fast_math": True})
