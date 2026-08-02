# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image filtering stages."""

from __future__ import annotations

import warp as wp


@wp.kernel(module="image_pyramid.filters")
def prepare_image_kernel(values: wp.array2d[wp.float32], prepared: wp.array2d[wp.float32]) -> None:
    """Copy one image into a working image."""
    row, column = wp.tid()
    prepared[row, column] = values[row, column]


@wp.kernel(module="image_pyramid.filters")
def level_128_kernel(values: wp.array2d[wp.float32], filtered: wp.array2d[wp.float32]) -> None:
    """Filter and downsample an image level."""
    row, column = wp.tid()
    source_row = row * 2
    source_column = column * 2
    total = wp.float32(0.0)
    for row_offset in range(-1, 2):
        for column_offset in range(-1, 2):
            sample_row = wp.max(0, wp.min(source_row + row_offset, values.shape[0] - 1))
            sample_column = wp.max(0, wp.min(source_column + column_offset, values.shape[1] - 1))
            total += values[sample_row, sample_column]
    filtered[row, column] = total / wp.float32(9.0)


@wp.kernel(module="image_pyramid.filters")
def level_256_kernel(values: wp.array2d[wp.float32], filtered: wp.array2d[wp.float32]) -> None:
    """Filter and downsample an image level."""
    row, column = wp.tid()
    source_row = row * 2
    source_column = column * 2
    total = wp.float32(0.0)
    for row_offset in range(-1, 2):
        for column_offset in range(-1, 2):
            sample_row = wp.max(0, wp.min(source_row + row_offset, values.shape[0] - 1))
            sample_column = wp.max(0, wp.min(source_column + column_offset, values.shape[1] - 1))
            total += values[sample_row, sample_column]
    filtered[row, column] = total / wp.float32(9.0)


@wp.kernel(module="image_pyramid.filters")
def level_512_kernel(values: wp.array2d[wp.float32], filtered: wp.array2d[wp.float32]) -> None:
    """Filter and downsample an image level."""
    row, column = wp.tid()
    source_row = row * 2
    source_column = column * 2
    total = wp.float32(0.0)
    for row_offset in range(-1, 2):
        for column_offset in range(-1, 2):
            sample_row = wp.max(0, wp.min(source_row + row_offset, values.shape[0] - 1))
            sample_column = wp.max(0, wp.min(source_column + column_offset, values.shape[1] - 1))
            total += values[sample_row, sample_column]
    filtered[row, column] = total / wp.float32(9.0)


def configure_filtering() -> None:
    """Configure filtering behavior."""
    wp.set_module_options({"enable_backward": False})
