# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Geometry processing operations.

This module provides GPU-accelerated geometry operations. It currently covers
isosurface extraction, in two flavors.

Dense-grid backends take a 3-D ``wp.float32`` field sampled at grid nodes and
share the :class:`IsoSurfaceBase` interface, so they can be swapped without
changing calling code: :class:`IsoSurfaceMarchingCubes` produces triangles, and
:class:`IsoSurfaceNets` produces triangles or quads with better-shaped elements.

Sparse extraction skips the dense grid entirely. :func:`sparse_marching_cubes`
takes an implicit function and builds a Lipschitz octree around the level set,
so cost scales with surface area rather than volume. :func:`lipschitz_octree`
and :func:`sparse_marching_cubes_from_cells` expose its two stages separately.

Usage:
    This module must be explicitly imported::

        import warp.geometry
"""

# isort: skip_file

from warp._src.iso_surface import IsoSurfaceBase as IsoSurfaceBase
from warp._src.marching_cubes import IsoSurfaceMarchingCubes as IsoSurfaceMarchingCubes
from warp._src.surface_nets import IsoSurfaceNets as IsoSurfaceNets
from warp._src.sparse_marching_cubes import lipschitz_octree as lipschitz_octree
from warp._src.sparse_marching_cubes import sparse_marching_cubes as sparse_marching_cubes
from warp._src.sparse_marching_cubes import sparse_marching_cubes_from_cells as sparse_marching_cubes_from_cells
