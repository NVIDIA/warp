# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deformed geometry support for finite element computations.

This module currently only provides the :class:`DeformedGeometry` class for defining
geometries that are deformed by a displacement field. Most geometry types (grids,
meshes, sparse grids) are available directly in :mod:`warp.fem`.
"""

# isort: skip_file

from warp._src.fem.geometry import DeformedGeometry as DeformedGeometry
