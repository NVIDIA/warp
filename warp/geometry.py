# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Geometry processing utilities.

This module provides functions to process 2D and 3D geometry and their
associated topological data structures (e.g., meshes).

Array-level functions such as :func:`swept_volume_mesh` launch kernels over a whole
mesh or grid. Device functions such as :func:`swept_volume_sdf` evaluate a
single point and may be called from within your own :func:`warp.kernel`
definitions.

Usage:
    This module must be explicitly imported::

        import warp.geometry
"""

# isort: skip_file

from warp._src.geometry import SweptVolumeSignMode as SweptVolumeSignMode
from warp._src.geometry import delaunay_edge_flip as delaunay_edge_flip
from warp._src.geometry import find_triangle_neighbor_edge_index as find_triangle_neighbor_edge_index
from warp._src.geometry import swept_volume_bounds as swept_volume_bounds
from warp._src.geometry import swept_volume_field as swept_volume_field
from warp._src.geometry import swept_volume_mesh as swept_volume_mesh
from warp._src.geometry import swept_volume_sdf as swept_volume_sdf

# Don't expose these quite yet in case we want to change the naming conventions.
# from warp._src.geometry import in_circle as in_circle
# from warp._src.geometry import signed_area as signed_area
from warp._src.geometry import tri_tri_adjacency as tri_tri_adjacency
