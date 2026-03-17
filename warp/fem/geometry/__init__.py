# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deformed geometry support for finite element computations.

This module currently only provides the :class:`DeformedGeometry` class for defining
geometries that are deformed by a displacement field. Most geometry types (grids,
meshes, sparse grids) are available directly in :mod:`warp.fem`.
"""

# isort: skip_file

from warp._src.fem.geometry import DeformedGeometry as DeformedGeometry


# TODO: Remove after cleaning up the public API.

from warp._src.fem import geometry as _geometry


def __getattr__(name):
    from warp._src.utils import get_deprecated_api  # noqa: PLC0415

    return get_deprecated_api(_geometry, "warp.fem", name)
