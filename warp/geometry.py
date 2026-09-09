# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Geometry processing utilities.

This module provides functions to process 2D and 3D geometry and their
associated topological data structures (e.g., meshes).

Usage:
    This module must be explicitly imported::

        import warp.geometry
"""

# isort: skip_file

from warp._src.geometry import delaunay_edge_flip as delaunay_edge_flip
from warp._src.geometry import find_triangle_neighbor_edge_index as find_triangle_neighbor_edge_index

# Don't expose these quite yet in case we want to change the naming conventions.
# from warp._src.geometry import in_circle as in_circle
# from warp._src.geometry import signed_area as signed_area
from warp._src.geometry import tri_tri_adjacency as tri_tri_adjacency
