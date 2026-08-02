# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Geometry stage: point-cloud descriptors."""

from __future__ import annotations

import warp as wp

# This module is always launched at BLOCK_DIM, so declare it as the module
# default too: a preload that does not name a block dimension then builds the
# variant this stage actually uses instead of a second one.
wp.set_module_options({"enable_backward": False, "block_dim": 64})

BLOCK_DIM = 64


@wp.kernel
def centre_points(points: wp.array[wp.vec3], centred: wp.array[wp.vec3], origin: wp.vec3) -> None:
    """Translate points so ``origin`` becomes zero."""
    index = wp.tid()
    centred[index] = points[index] - origin


@wp.kernel
def radial_extent(points: wp.array[wp.vec3], radii: wp.array[wp.float32]) -> None:
    """Distance of each point from the origin."""
    index = wp.tid()
    radii[index] = wp.length(points[index])


@wp.kernel
def local_density(points: wp.array[wp.vec3], density: wp.array[wp.float32], radius: wp.float32) -> None:
    """Count neighbours within ``radius`` of each point."""
    index = wp.tid()
    count = wp.float32(0.0)
    here = points[index]
    for other in range(points.shape[0]):
        if wp.length(points[other] - here) < radius:
            count += wp.float32(1.0)
    density[index] = count / wp.float32(points.shape[0])


@wp.kernel
def orient_normals(points: wp.array[wp.vec3], normals: wp.array[wp.vec3], reference: wp.vec3) -> None:
    """Flip normals to face a reference direction."""
    index = wp.tid()
    direction = wp.normalize(points[index] + wp.vec3(1.0e-6, 1.0e-6, 1.0e-6))
    if wp.dot(direction, reference) < 0.0:
        normals[index] = -direction
    else:
        normals[index] = direction
