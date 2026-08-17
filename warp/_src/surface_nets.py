# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
# The surface nets implementation in this file is ported from the uniform
# (non-adaptive) meshing path of OpenVDB's volume-to-mesh tool
# (openvdb/openvdb/tools/VolumeToMesh.h at v13.0.0, Apache-2.0), adapted from
# sparse-tree traversal to dense grids. See licenses/openvdb-LICENSE.txt.

from __future__ import annotations

from typing import Final, Literal

import warp as wp
from warp._src.iso_surface import IsoSurfaceBase, resolve_domain_bounds, validate_field

# =============================================================================
# Lookup tables ported from OpenVDB (VolumeToMesh.h)
#
# These tables use OpenVDB's cell-corner numbering, which differs from
# IsoSurfaceMarchingCubes.CUBE_CORNER_OFFSETS (OpenVDB cycles the bottom and top faces
# around the y-axis, Warp's marching cubes around the z-axis), so they are
# kept private to this module.
# =============================================================================

# fmt: off

# The (x, y, z) offset from the cell origin for each of the 8 cube corners in
# OpenVDB's numbering. Corner i corresponds to bit i in a cell's sign
# configuration, with bit i set iff corner i is inside (value < threshold).
_SN_CORNER_OFFSETS: Final[tuple[tuple[int, int, int], ...]] = (
    (0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1),  # y=0 face (corners 0-3)
    (0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1),  # y=1 face (corners 4-7)
)

# For each of the 12 edges, in the column order of _SN_EDGE_GROUP_TABLE, the
# pair of corner indices (see _SN_CORNER_OFFSETS) it connects.
_SN_EDGE_TO_CORNERS: Final[tuple[tuple[int, int], ...]] = (
    (0, 1), (1, 2), (3, 2), (0, 3),  # y=0 face edges (columns 1-4)
    (4, 5), (5, 6), (7, 6), (4, 7),  # y=1 face edges (columns 5-8)
    (0, 4), (1, 5), (2, 6), (3, 7),  # vertical edges (columns 9-12)
)

# Ambiguous-face index per cell sign configuration: 0 = unambiguous, otherwise
# the 1-based index (1-6) of the face shared with the neighbor cell that
# disambiguates the configuration. (VolumeToMesh.h sAmbiguousFace)
_SN_AMBIGUOUS_FACE: Final[tuple[int, ...]] = (
    0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 5, 1, 0, 4, 0, 0, 0, 4, 0, 0, 0,
    0, 1, 0, 0, 2, 0, 0, 0, 0, 1, 5, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 2, 2, 0, 5, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 6, 6, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 4, 0, 4, 3, 0, 3, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 1, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    6, 0, 6, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
)

# Edge-group table per cell sign configuration, flattened to 256 rows of 13
# columns (one row per line below): column 0 is the number of vertices the
# cell produces (0-4); columns 1-12 give the 1-based vertex (edge-group) id
# for each of the cell's 12 edges in _SN_EDGE_TO_CORNERS order, or 0 if the
# edge does not cross the isosurface. (VolumeToMesh.h sEdgeGroupTable)
_SN_EDGE_GROUP_TABLE: Final[tuple[int, ...]] = (
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
    1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
    1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0,
    1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
    1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0,
    1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0,
    1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0,
    1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,
    1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1,
    1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1,
    1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1,
    1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1,
    1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1,
    1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
    1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0,
    1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0,
    1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0,
    1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0,
    2, 0, 1, 1, 0, 2, 0, 0, 2, 2, 0, 1, 0,
    1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0,
    1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0,
    1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0,
    1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1,
    1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1,
    2, 2, 1, 1, 2, 1, 0, 0, 1, 2, 1, 0, 1,
    1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1,
    1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1,
    1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1,
    2, 1, 0, 0, 1, 2, 0, 0, 2, 1, 2, 2, 2,
    1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1,
    1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0,
    1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0,
    1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
    1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0,
    1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0,
    2, 2, 2, 1, 1, 1, 1, 0, 0, 1, 2, 1, 0,
    1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0,
    1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0,
    2, 0, 0, 2, 2, 1, 1, 0, 0, 0, 1, 0, 2,
    1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1,
    1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1,
    1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1,
    1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1,
    2, 1, 1, 0, 0, 2, 2, 0, 0, 2, 1, 2, 2,
    1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1,
    1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1,
    1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0,
    1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0,
    1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0,
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0,
    1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0,
    2, 1, 1, 2, 2, 0, 2, 0, 2, 0, 1, 2, 0,
    1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0,
    1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0,
    1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1,
    1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1,
    2, 1, 2, 2, 1, 0, 2, 0, 2, 1, 0, 0, 2,
    1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1,
    2, 0, 2, 0, 2, 0, 1, 0, 1, 2, 2, 1, 1,
    2, 2, 2, 0, 0, 0, 1, 0, 1, 0, 2, 1, 1,
    2, 2, 0, 0, 2, 0, 1, 0, 1, 2, 0, 1, 1,
    1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1,
    1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0,
    2, 1, 0, 0, 1, 0, 2, 2, 0, 1, 0, 2, 0,
    1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0,
    1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0,
    1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0,
    1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0,
    1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0,
    1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1,
    1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1,
    2, 1, 2, 2, 1, 0, 1, 1, 0, 0, 1, 2, 1,
    2, 0, 1, 1, 0, 0, 2, 2, 0, 2, 2, 1, 2,
    1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1,
    1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1,
    1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1,
    1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1,
    1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0,
    1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0,
    2, 1, 1, 0, 0, 2, 2, 1, 1, 1, 2, 1, 0,
    2, 0, 2, 0, 2, 1, 1, 2, 2, 0, 1, 2, 0,
    1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0,
    2, 2, 2, 1, 1, 2, 2, 1, 1, 0, 0, 0, 0,
    2, 2, 0, 2, 0, 1, 1, 2, 2, 2, 1, 0, 0,
    2, 0, 0, 1, 1, 2, 2, 1, 1, 0, 2, 0, 0,
    2, 0, 0, 1, 1, 1, 1, 2, 2, 1, 0, 1, 2,
    2, 2, 0, 2, 0, 2, 2, 1, 1, 0, 0, 2, 1,
    4, 3, 2, 2, 3, 4, 4, 1, 1, 3, 4, 2, 1,
    3, 0, 2, 2, 0, 1, 1, 3, 3, 0, 1, 2, 3,
    2, 0, 2, 0, 2, 2, 2, 1, 1, 2, 0, 0, 1,
    2, 1, 1, 0, 0, 1, 1, 2, 2, 0, 0, 0, 2,
    3, 1, 0, 0, 1, 2, 2, 3, 3, 1, 2, 0, 3,
    2, 0, 0, 0, 0, 1, 1, 2, 2, 0, 1, 0, 2,
    1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0,
    1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0,
    1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0,
    1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0,
    2, 1, 1, 2, 2, 2, 0, 2, 0, 2, 1, 0, 0,
    1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0,
    1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0,
    1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1,
    2, 2, 0, 2, 0, 1, 0, 1, 0, 1, 2, 2, 1,
    2, 2, 1, 1, 2, 2, 0, 2, 0, 0, 0, 1, 2,
    2, 0, 2, 2, 0, 1, 0, 1, 0, 1, 0, 2, 1,
    1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1,
    2, 2, 2, 0, 0, 1, 0, 1, 0, 1, 2, 0, 1,
    1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1,
    1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0,
    1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0,
    1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0,
    1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0,
    1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0,
    2, 2, 2, 1, 1, 0, 0, 1, 1, 0, 2, 0, 0,
    1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0,
    1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0,
    2, 0, 0, 2, 2, 0, 0, 1, 1, 2, 2, 2, 1,
    2, 1, 0, 1, 0, 0, 0, 2, 2, 0, 1, 1, 2,
    3, 2, 1, 1, 2, 0, 0, 3, 3, 2, 0, 1, 3,
    2, 0, 1, 1, 0, 0, 0, 2, 2, 0, 0, 1, 2,
    2, 0, 1, 0, 1, 0, 0, 2, 2, 1, 1, 0, 2,
    2, 1, 1, 0, 0, 0, 0, 2, 2, 0, 1, 0, 2,
    2, 1, 0, 0, 1, 0, 0, 2, 2, 1, 0, 0, 2,
    1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1,
    1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1,
    1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1,
    2, 1, 1, 0, 0, 0, 0, 2, 2, 0, 1, 0, 2,
    1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1,
    1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1,
    2, 1, 1, 2, 2, 0, 0, 1, 1, 1, 0, 1, 2,
    1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1,
    2, 0, 0, 1, 1, 0, 0, 2, 2, 2, 2, 2, 1,
    1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0,
    1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0,
    1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0,
    1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0,
    1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0,
    1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0,
    1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0,
    1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0,
    1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1,
    1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1,
    1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1,
    1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1,
    1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1,
    2, 2, 2, 1, 1, 2, 0, 2, 0, 0, 0, 2, 1,
    2, 1, 0, 1, 0, 2, 0, 2, 0, 1, 2, 2, 1,
    2, 0, 0, 2, 2, 1, 0, 1, 0, 0, 1, 1, 2,
    1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0,
    1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0,
    2, 1, 2, 2, 1, 2, 0, 2, 0, 1, 2, 0, 0,
    1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0,
    1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0,
    1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0,
    2, 2, 0, 0, 2, 1, 0, 1, 0, 2, 1, 1, 0,
    1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0,
    1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1,
    2, 1, 0, 0, 1, 2, 1, 1, 2, 2, 1, 0, 1,
    1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1,
    2, 0, 2, 0, 2, 1, 2, 2, 1, 1, 0, 0, 2,
    2, 0, 1, 1, 0, 1, 2, 2, 1, 0, 1, 2, 1,
    4, 1, 1, 3, 3, 2, 4, 4, 2, 2, 1, 4, 3,
    2, 2, 0, 2, 0, 2, 1, 1, 2, 0, 0, 1, 2,
    3, 0, 0, 1, 1, 2, 3, 3, 2, 2, 0, 3, 1,
    1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0,
    2, 2, 0, 2, 0, 1, 2, 2, 1, 1, 2, 0, 0,
    2, 2, 1, 1, 2, 2, 1, 1, 2, 0, 0, 0, 0,
    2, 0, 1, 1, 0, 2, 1, 1, 2, 2, 0, 0, 0,
    2, 0, 2, 0, 2, 2, 1, 1, 2, 0, 2, 1, 0,
    3, 1, 1, 0, 0, 3, 2, 2, 3, 3, 1, 2, 0,
    2, 1, 0, 0, 1, 1, 2, 2, 1, 0, 0, 2, 0,
    2, 0, 0, 0, 0, 2, 1, 1, 2, 2, 0, 1, 0,
    1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1,
    1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1,
    1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1,
    1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1,
    2, 0, 2, 2, 0, 0, 1, 1, 0, 2, 2, 1, 2,
    3, 1, 1, 2, 2, 0, 3, 3, 0, 0, 1, 3, 2,
    2, 1, 0, 1, 0, 0, 2, 2, 0, 1, 0, 2, 1,
    2, 0, 0, 1, 1, 0, 2, 2, 0, 0, 0, 2, 1,
    1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0,
    1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0,
    2, 2, 1, 1, 2, 0, 1, 1, 0, 2, 0, 0, 0,
    1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0,
    2, 0, 1, 0, 1, 0, 2, 2, 0, 1, 1, 2, 0,
    2, 1, 1, 0, 0, 0, 2, 2, 0, 0, 1, 2, 0,
    2, 1, 0, 0, 1, 0, 2, 2, 0, 1, 0, 2, 0,
    1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0,
    1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1,
    1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1,
    1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1,
    2, 0, 2, 0, 2, 0, 1, 0, 1, 1, 1, 2, 2,
    1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1,
    2, 2, 2, 1, 1, 0, 2, 0, 2, 2, 0, 0, 1,
    1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1,
    2, 0, 0, 2, 2, 0, 1, 0, 1, 1, 1, 0, 2,
    1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0,
    1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0,
    2, 2, 1, 1, 2, 0, 2, 0, 2, 0, 2, 1, 0,
    2, 0, 2, 2, 0, 0, 1, 0, 1, 1, 1, 2, 0,
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0,
    1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0,
    1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0,
    1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0,
    1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1,
    1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1,
    2, 2, 2, 0, 0, 1, 1, 0, 0, 2, 1, 2, 2,
    2, 0, 1, 0, 1, 2, 2, 0, 0, 0, 2, 1, 1,
    1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1,
    2, 1, 1, 2, 2, 1, 1, 0, 0, 0, 0, 0, 2,
    2, 1, 0, 1, 0, 2, 2, 0, 0, 1, 2, 0, 1,
    2, 0, 0, 2, 2, 1, 1, 0, 0, 0, 1, 0, 2,
    1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0,
    1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0,
    3, 1, 2, 2, 1, 3, 3, 0, 0, 1, 3, 2, 0,
    2, 0, 1, 1, 0, 2, 2, 0, 0, 0, 2, 1, 0,
    1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0,
    1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
    2, 2, 0, 0, 2, 1, 1, 0, 0, 2, 1, 0, 0,
    1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0,
    1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1,
    2, 2, 0, 0, 2, 1, 0, 0, 1, 1, 2, 2, 2,
    1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1,
    2, 0, 1, 0, 1, 2, 0, 0, 2, 2, 0, 1, 1,
    1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1,
    3, 1, 1, 3, 3, 2, 0, 0, 2, 2, 1, 0, 3,
    1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1,
    2, 0, 0, 2, 2, 1, 0, 0, 1, 1, 0, 0, 2,
    1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0,
    2, 1, 0, 1, 0, 2, 0, 0, 2, 2, 1, 1, 0,
    2, 1, 2, 2, 1, 1, 0, 0, 1, 0, 0, 2, 0,
    2, 0, 1, 1, 0, 2, 0, 0, 2, 2, 0, 1, 0,
    1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0,
    2, 1, 1, 0, 0, 2, 0, 0, 2, 2, 1, 0, 0,
    1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
    1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1,
    1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1,
    1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1,
    1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1,
    2, 1, 1, 2, 2, 0, 0, 0, 0, 0, 1, 0, 2,
    1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1,
    1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,
    1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0,
    1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0,
    2, 1, 2, 2, 1, 0, 0, 0, 0, 1, 0, 2, 0,
    1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
    1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0,
    1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
    1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
)

# fmt: on

# Cell classification bit-flags (VolumeToMesh.h): bits 0-7 hold the
# (ambiguity-corrected) corner sign configuration, bit 8 is set when corner 0
# is inside, and bits 9-11 mark the cell's three owned lattice edges (from the
# cell origin along +x/+y/+z) as crossing the isosurface. The INSIDE and edge
# bits are derived from the raw (uncorrected) signs.
_SN_INSIDE: Final[int] = 0x100
_SN_XEDGE: Final[int] = 0x200
_SN_YEDGE: Final[int] = 0x400
_SN_ZEDGE: Final[int] = 0x800

# Valid values for the ``topology`` parameter of IsoSurfaceNets.
_SN_TOPOLOGIES: Final[tuple[str, ...]] = ("triangle", "quad")


def _validate_topology(topology: str) -> None:
    """Validate the ``topology`` face-type selector.

    Raises:
        ValueError: If ``topology`` is not one of ``"triangle"`` or ``"quad"``.
    """
    if topology not in _SN_TOPOLOGIES:
        raise ValueError(f"Expected 'topology' to be one of {_SN_TOPOLOGIES}, but got {topology!r}.")


# =============================================================================
# Internal: per-device caches for the lookup tables
# =============================================================================

_sn_ambiguous_face_cache: dict[str, wp.array] = {}
_sn_edge_group_table_cache: dict[str, wp.array] = {}


def _get_sn_ambiguous_face_table(device) -> wp.array:
    """Lazily creates and caches the ambiguous-face table on the target device."""
    device = str(device)
    if device not in _sn_ambiguous_face_cache:
        _sn_ambiguous_face_cache[device] = wp.array(_SN_AMBIGUOUS_FACE, dtype=wp.int32, device=device)
    return _sn_ambiguous_face_cache[device]


def _get_sn_edge_group_table(device) -> wp.array:
    """Lazily creates and caches the edge-group table on the target device."""
    device = str(device)
    if device not in _sn_edge_group_table_cache:
        _sn_edge_group_table_cache[device] = wp.array(_SN_EDGE_GROUP_TABLE, dtype=wp.int32, device=device)
    return _sn_edge_group_table_cache[device]


# =============================================================================
# Kernels
# =============================================================================


@wp.func
def _sn_cell_signs(values: wp.array3d(dtype=wp.float32), i: int, j: int, k: int, threshold: float) -> int:
    """Compute a cell's sign configuration (port of VolumeToMesh.h evalCellSigns()).

    Bit c is set iff corner c (in OpenVDB numbering, see _SN_CORNER_OFFSETS)
    is inside, i.e. its value is below the threshold.
    """
    signs = int(0)
    if values[i, j, k] < threshold:
        signs |= 1
    if values[i + 1, j, k] < threshold:
        signs |= 2
    if values[i + 1, j, k + 1] < threshold:
        signs |= 4
    if values[i, j, k + 1] < threshold:
        signs |= 8
    if values[i, j + 1, k] < threshold:
        signs |= 16
    if values[i + 1, j + 1, k] < threshold:
        signs |= 32
    if values[i + 1, j + 1, k + 1] < threshold:
        signs |= 64
    if values[i, j + 1, k + 1] < threshold:
        signs |= 128
    return signs


# NOTE: differentiating this kernel does nothing, since all of its outputs are
# discrete, but Warp issues warnings if we set enable_backward=False
@wp.kernel
def _sn_classify_cells_kernel(
    values: wp.array3d(dtype=wp.float32),
    threshold: wp.float32,
    ambiguous_face_table: wp.array(dtype=wp.int32),
    edge_group_table: wp.array(dtype=wp.int32),
    cell_flags: wp.array3d(dtype=wp.int32),
    point_counts: wp.array(dtype=wp.int32),
    quad_counts: wp.array(dtype=wp.int32),
):
    """Classify cells and count their vertices and quads.

    Port of VolumeToMesh.h ComputeAuxiliaryData: for each cell intersecting
    the isosurface, stores the packed sign/INSIDE/edge flags (see _SN_INSIDE)
    and the number of vertices the cell produces, and counts one quad per
    owned crossing edge whose four incident cells all exist.
    """
    ci, cj, ck = wp.tid()
    ncell_x = values.shape[0] - 1
    ncell_y = values.shape[1] - 1
    ncell_z = values.shape[2] - 1
    cell = (ci * ncell_y + cj) * ncell_z + ck

    signs = _sn_cell_signs(values, ci, cj, ck, threshold)
    if signs == 0 or signs == 255:
        # cell_flags, point_counts, and quad_counts are zero-initialized
        return

    inside = signs & 1

    flags = int(0)
    if inside != 0:
        flags |= wp.static(_SN_INSIDE)

    # The INSIDE and edge flags above/below are derived from the raw signs;
    # only bits 0-7 store the corrected configuration.
    xedge = inside != ((signs >> 1) & 1)
    yedge = inside != ((signs >> 4) & 1)
    zedge = inside != ((signs >> 3) & 1)
    if xedge:
        flags |= wp.static(_SN_XEDGE)
    if yedge:
        flags |= wp.static(_SN_YEDGE)
    if zedge:
        flags |= wp.static(_SN_ZEDGE)

    # Topological-ambiguity correction (port of VolumeToMesh.h
    # correctCellSigns()): when the configuration is ambiguous across a face
    # and the neighbor across that face is ambiguous across the complementary
    # face, flip the configuration. Out-of-range neighbors never flip.
    face = ambiguous_face_table[signs]
    if face != 0:
        ni = ci
        nj = cj
        nk = ck
        complementary_face = int(0)
        if face == 1:
            nk = ck - 1
            complementary_face = 3
        elif face == 2:
            ni = ci + 1
            complementary_face = 4
        elif face == 3:
            nk = ck + 1
            complementary_face = 1
        elif face == 4:
            ni = ci - 1
            complementary_face = 2
        elif face == 5:
            nj = cj - 1
            complementary_face = 6
        else:
            nj = cj + 1
            complementary_face = 5

        if ni >= 0 and ni < ncell_x and nj >= 0 and nj < ncell_y and nk >= 0 and nk < ncell_z:
            neighbor_signs = _sn_cell_signs(values, ni, nj, nk, threshold)
            if ambiguous_face_table[neighbor_signs] == complementary_face:
                signs = (~signs) & 255

    flags |= signs
    cell_flags[ci, cj, ck] = flags
    point_counts[cell] = edge_group_table[signs * 13]

    # A quad is built around each owned crossing edge; the edge's four
    # incident cells must all be inside the grid.
    quad_count = int(0)
    if xedge and cj > 0 and ck > 0:
        quad_count += 1
    if yedge and ci > 0 and ck > 0:
        quad_count += 1
    if zedge and ci > 0 and cj > 0:
        quad_count += 1
    quad_counts[cell] = quad_count


@wp.kernel
def _sn_compute_points_kernel(
    values: wp.array3d(dtype=wp.float32),
    threshold: wp.float32,
    edge_group_table: wp.array(dtype=wp.int32),
    cell_flags: wp.array3d(dtype=wp.int32),
    point_offsets: wp.array(dtype=wp.int32),
    domain_bounds_lower_corner: wp.vec3,
    grid_pos_delta: wp.vec3,
    verts_pos_out: wp.array(dtype=wp.vec3),
):
    """Compute one vertex per edge group of each intersecting cell.

    Port of VolumeToMesh.h computePoint()/computeCellPoints(): each vertex is
    the average, over the crossing edges of its group, of the linear
    zero-crossing positions in cell-local coordinates. The average is taken in
    index space and transformed to world space afterwards, like OpenVDB.
    """
    ci, cj, ck = wp.tid()
    ncell_y = values.shape[1] - 1
    ncell_z = values.shape[2] - 1
    cell = (ci * ncell_y + cj) * ncell_z + ck

    signs = cell_flags[ci, cj, ck] & 255
    num_points = edge_group_table[signs * 13]
    if num_points == 0:
        return

    # The indexing logic is slightly awkward here because we use an inclusive
    # sum, so we need to check the previous thread's output index, with a
    # special case for the first thread. Doing it this way makes it simpler to
    # fetch the total count in the host function.
    if cell == 0:
        out_ind = int(0)
    else:
        out_ind = point_offsets[cell - 1]

    # Corner values in OpenVDB numbering (see _SN_CORNER_OFFSETS)
    v0 = values[ci, cj, ck]
    v1 = values[ci + 1, cj, ck]
    v2 = values[ci + 1, cj, ck + 1]
    v3 = values[ci, cj, ck + 1]
    v4 = values[ci, cj + 1, ck]
    v5 = values[ci + 1, cj + 1, ck]
    v6 = values[ci + 1, cj + 1, ck + 1]
    v7 = values[ci, cj + 1, ck + 1]

    for group in range(1, num_points + 1):
        avg_x = float(0.0)
        avg_y = float(0.0)
        avg_z = float(0.0)
        samples = int(0)

        if edge_group_table[signs * 13 + 1] == group:  # edge 0-1
            avg_x += (threshold - v0) / (v1 - v0)
            samples += 1

        if edge_group_table[signs * 13 + 2] == group:  # edge 1-2
            avg_x += 1.0
            avg_z += (threshold - v1) / (v2 - v1)
            samples += 1

        if edge_group_table[signs * 13 + 3] == group:  # edge 3-2
            avg_x += (threshold - v3) / (v2 - v3)
            avg_z += 1.0
            samples += 1

        if edge_group_table[signs * 13 + 4] == group:  # edge 0-3
            avg_z += (threshold - v0) / (v3 - v0)
            samples += 1

        if edge_group_table[signs * 13 + 5] == group:  # edge 4-5
            avg_x += (threshold - v4) / (v5 - v4)
            avg_y += 1.0
            samples += 1

        if edge_group_table[signs * 13 + 6] == group:  # edge 5-6
            avg_x += 1.0
            avg_y += 1.0
            avg_z += (threshold - v5) / (v6 - v5)
            samples += 1

        if edge_group_table[signs * 13 + 7] == group:  # edge 7-6
            avg_x += (threshold - v7) / (v6 - v7)
            avg_y += 1.0
            avg_z += 1.0
            samples += 1

        if edge_group_table[signs * 13 + 8] == group:  # edge 4-7
            avg_y += 1.0
            avg_z += (threshold - v4) / (v7 - v4)
            samples += 1

        if edge_group_table[signs * 13 + 9] == group:  # edge 0-4
            avg_y += (threshold - v0) / (v4 - v0)
            samples += 1

        if edge_group_table[signs * 13 + 10] == group:  # edge 1-5
            avg_x += 1.0
            avg_y += (threshold - v1) / (v5 - v1)
            samples += 1

        if edge_group_table[signs * 13 + 11] == group:  # edge 2-6
            avg_x += 1.0
            avg_y += (threshold - v2) / (v6 - v2)
            avg_z += 1.0
            samples += 1

        if edge_group_table[signs * 13 + 12] == group:  # edge 3-7
            avg_y += (threshold - v3) / (v7 - v3)
            avg_z += 1.0
            samples += 1

        if samples > 1:
            weight = 1.0 / float(samples)
            avg_x *= weight
            avg_y *= weight
            avg_z *= weight

        local = wp.vec3(float(ci) + avg_x, float(cj) + avg_y, float(ck) + avg_z)
        verts_pos_out[out_ind + group - 1] = domain_bounds_lower_corner + wp.cw_mul(local, grid_pos_delta)


@wp.func
def _sn_cell_point_index(
    cell_flags: wp.array3d(dtype=wp.int32),
    edge_group_table: wp.array(dtype=wp.int32),
    point_offsets: wp.array(dtype=wp.int32),
    i: int,
    j: int,
    k: int,
    column: int,
) -> int:
    """Index of the vertex on the given edge (table column) of the given cell.

    This is the cell's first vertex index plus the edge's group offset when
    the cell produces more than one vertex (VolumeToMesh.h constructPolygons()).
    """
    ncell_y = cell_flags.shape[1]
    ncell_z = cell_flags.shape[2]
    cell = (i * ncell_y + j) * ncell_z + k

    if cell == 0:
        index = int(0)
    else:
        index = point_offsets[cell - 1]

    signs = cell_flags[i, j, k] & 255
    if edge_group_table[signs * 13] > 1:
        index += edge_group_table[signs * 13 + column] - 1
    return index


@wp.func
def _sn_write_face(
    indices_out: wp.array(dtype=wp.int32),
    quad_topology: bool,
    quad_index: int,
    v0: int,
    v1: int,
    v2: int,
    v3: int,
    reverse: bool,
):
    """Write one quad, or the two triangles it splits into, optionally reversing the winding.

    The triangulation splits the quad along its first diagonal, into the
    ``(0, 1, 2)`` and ``(0, 2, 3)`` corners.
    """
    a = v0
    b = v1
    c = v2
    d = v3
    if reverse:
        a = v3
        b = v2
        c = v1
        d = v0

    if quad_topology:
        indices_out[4 * quad_index + 0] = a
        indices_out[4 * quad_index + 1] = b
        indices_out[4 * quad_index + 2] = c
        indices_out[4 * quad_index + 3] = d
    else:
        indices_out[6 * quad_index + 0] = a
        indices_out[6 * quad_index + 1] = b
        indices_out[6 * quad_index + 2] = c
        indices_out[6 * quad_index + 3] = a
        indices_out[6 * quad_index + 4] = c
        indices_out[6 * quad_index + 5] = d


# NOTE: differentiating this kernel does nothing, since all of its outputs are
# discrete, but Warp issues warnings if we set enable_backward=False
@wp.kernel
def _sn_build_faces_kernel(
    cell_flags: wp.array3d(dtype=wp.int32),
    edge_group_table: wp.array(dtype=wp.int32),
    point_offsets: wp.array(dtype=wp.int32),
    quad_offsets: wp.array(dtype=wp.int32),
    quad_topology: bool,
    indices_out: wp.array(dtype=wp.int32),
):
    """Build one quad around each owned crossing edge of each cell.

    Port of VolumeToMesh.h constructPolygons(): the quad connects the vertices
    of the four cells incident to the edge, and is written out either as-is or
    triangulated, as selected by ``quad_topology``. The winding deliberately
    deviates from OpenVDB's level-set output, which is wound the opposite way
    from IsoSurfaceMarchingCubes; the reversal sense of each axis is flipped so that
    every IsoSurfaceBase backend produces counter-clockwise faces viewed from
    outside for a signed distance field (negative inside).
    """
    ci, cj, ck = wp.tid()
    ncell_y = cell_flags.shape[1]
    ncell_z = cell_flags.shape[2]
    cell = (ci * ncell_y + cj) * ncell_z + ck

    flags = cell_flags[ci, cj, ck]
    if (flags & wp.static(_SN_XEDGE | _SN_YEDGE | _SN_ZEDGE)) == 0:
        return

    inside = (flags & wp.static(_SN_INSIDE)) != 0

    # See the comment about inclusive sums in _sn_compute_points_kernel.
    if cell == 0:
        out_ind = int(0)
    else:
        out_ind = quad_offsets[cell - 1]

    if (flags & wp.static(_SN_XEDGE)) != 0 and cj > 0 and ck > 0:
        q0 = _sn_cell_point_index(cell_flags, edge_group_table, point_offsets, ci, cj, ck, 1)
        q1 = _sn_cell_point_index(cell_flags, edge_group_table, point_offsets, ci, cj - 1, ck, 5)
        q2 = _sn_cell_point_index(cell_flags, edge_group_table, point_offsets, ci, cj - 1, ck - 1, 7)
        q3 = _sn_cell_point_index(cell_flags, edge_group_table, point_offsets, ci, cj, ck - 1, 3)
        _sn_write_face(indices_out, quad_topology, out_ind, q0, q1, q2, q3, not inside)
        out_ind += 1

    if (flags & wp.static(_SN_YEDGE)) != 0 and ci > 0 and ck > 0:
        q0 = _sn_cell_point_index(cell_flags, edge_group_table, point_offsets, ci, cj, ck, 9)
        q1 = _sn_cell_point_index(cell_flags, edge_group_table, point_offsets, ci, cj, ck - 1, 12)
        q2 = _sn_cell_point_index(cell_flags, edge_group_table, point_offsets, ci - 1, cj, ck - 1, 11)
        q3 = _sn_cell_point_index(cell_flags, edge_group_table, point_offsets, ci - 1, cj, ck, 10)
        _sn_write_face(indices_out, quad_topology, out_ind, q0, q1, q2, q3, not inside)
        out_ind += 1

    if (flags & wp.static(_SN_ZEDGE)) != 0 and ci > 0 and cj > 0:
        q0 = _sn_cell_point_index(cell_flags, edge_group_table, point_offsets, ci, cj, ck, 4)
        q1 = _sn_cell_point_index(cell_flags, edge_group_table, point_offsets, ci, cj - 1, ck, 8)
        q2 = _sn_cell_point_index(cell_flags, edge_group_table, point_offsets, ci - 1, cj - 1, ck, 6)
        q3 = _sn_cell_point_index(cell_flags, edge_group_table, point_offsets, ci - 1, cj, ck, 2)
        _sn_write_face(indices_out, quad_topology, out_ind, q0, q1, q2, q3, inside)


# =============================================================================
# Host-side pipeline
# =============================================================================


def surface_nets_extract(
    field: wp.array3d(dtype=wp.float32),
    threshold: float,
    domain_bounds_lower_corner: wp.vec3,
    grid_pos_delta: wp.vec3,
    topology: str,
):
    """Invoke the kernels implementing the three passes of the algorithm.

    The faces are written to a single flat index array holding either the
    algorithm's quads or their triangulation, as selected by ``topology``.
    """
    device = field.device
    ncell_x, ncell_y, ncell_z = field.shape[0] - 1, field.shape[1] - 1, field.shape[2] - 1
    num_cells = ncell_x * ncell_y * ncell_z

    ambiguous_face_table = _get_sn_ambiguous_face_table(device)
    edge_group_table = _get_sn_edge_group_table(device)

    ### First pass: classify the cells and count the vertices and quads
    cell_flags = wp.zeros(shape=(ncell_x, ncell_y, ncell_z), dtype=wp.int32, device=device)
    point_counts = wp.zeros(shape=num_cells, dtype=wp.int32, device=device)
    quad_counts = wp.zeros(shape=num_cells, dtype=wp.int32, device=device)
    wp.launch(
        _sn_classify_cells_kernel,
        dim=(ncell_x, ncell_y, ncell_z),
        inputs=[
            field,
            threshold,
            ambiguous_face_table,
            edge_group_table,
        ],
        outputs=[
            cell_flags,
            point_counts,
            quad_counts,
        ],
        device=device,
    )

    ### Evaluate cumulative sums, to compute the output index for each vertex and quad
    point_offsets = wp.zeros(shape=num_cells, dtype=wp.int32, device=device)
    wp._src.utils.array_scan(point_counts, point_offsets, inclusive=True)
    quad_offsets = wp.zeros(shape=num_cells, dtype=wp.int32, device=device)
    wp._src.utils.array_scan(quad_counts, quad_offsets, inclusive=True)

    # (synchronization point!)
    num_points = int(point_offsets[-1:].numpy()[0])
    num_quads = int(quad_offsets[-1:].numpy()[0])

    ### Second pass: generate the vertices
    verts_pos_out = wp.empty(shape=num_points, dtype=wp.vec3, device=device, requires_grad=field.requires_grad)
    wp.launch(
        _sn_compute_points_kernel,
        dim=(ncell_x, ncell_y, ncell_z),
        inputs=[
            field,
            threshold,
            edge_group_table,
            cell_flags,
            point_offsets,
            domain_bounds_lower_corner,
            grid_pos_delta,
        ],
        outputs=[
            verts_pos_out,
        ],
        device=device,
    )

    ### Third pass: generate the quads, or their triangulation
    quad_topology = topology == "quad"
    indices_per_quad = 4 if quad_topology else 6
    indices_out = wp.empty(shape=indices_per_quad * num_quads, dtype=wp.int32, device=device)
    wp.launch(
        _sn_build_faces_kernel,
        dim=(ncell_x, ncell_y, ncell_z),
        inputs=[
            cell_flags,
            edge_group_table,
            point_offsets,
            quad_offsets,
            quad_topology,
        ],
        outputs=[
            indices_out,
        ],
        device=device,
    )

    return verts_pos_out, indices_out


class IsoSurfaceNets(IsoSurfaceBase):
    """A reusable context for surface nets isosurface extraction.

    This backend is a port of the uniform (non-adaptive) meshing path of
    OpenVDB's volume-to-mesh tool to dense grids. It produces one vertex per
    edge group of each cell crossed by the isosurface (so cells crossed by
    multiple surface sheets produce up to four vertices) and one quad around
    each interior grid edge crossing the isosurface, with the
    topological-ambiguity correction of adjacent cells applied. For fully
    interior isosurfaces the output mesh is closed and 2-manifold; the mesh is
    left open where the isosurface exits the grid domain (unlike
    :class:`warp.geometry.IsoSurfaceMarchingCubes`, which meshes up to the domain boundary).

    This class provides a stateful interface following
    :class:`warp.geometry.IsoSurfaceBase`: initialize it with a specific grid
    configuration and then call the :meth:`~.surface` method multiple times,
    which is efficient for processing fields of the same size. For a simpler,
    stateless operation, use the :meth:`~.extract` class method.

    The ``topology`` parameter selects the type of the faces written to
    :attr:`indices`: ``"triangle"`` (the default) stores the triangulation of
    the quads, with three consecutive indices per triangle, and ``"quad"``
    stores the algorithm's native quads instead, with four consecutive indices
    per quad (matching OpenVDB's output). The faces are generated directly in
    the selected form, without any conversion pass. The triangulation splits
    each quad along its first diagonal, into the ``(0, 1, 2)`` and
    ``(0, 2, 3)`` corners. Faces are wound counter-clockwise viewed from
    outside for a signed distance field (negative inside), matching
    :class:`warp.geometry.IsoSurfaceMarchingCubes` (this deliberately differs from raw OpenVDB
    level-set output, which uses the opposite winding).

    Args:
        nx: Number of grid nodes in the x-direction.
        ny: Number of grid nodes in the y-direction.
        nz: Number of grid nodes in the z-direction.
        domain_bounds_lower_corner: See the documentation in
          :meth:`~.extract`.
        domain_bounds_upper_corner: See the documentation in
          :meth:`~.extract`.
        topology: The type of the faces that :meth:`~.surface` produces:
          ``"triangle"`` or ``"quad"``.

    Attributes:
        nx (int): The number of grid nodes in the x-direction.
        ny (int): The number of grid nodes in the y-direction.
        nz (int): The number of grid nodes in the z-direction.
        domain_bounds_lower_corner (warp.vec3f | tuple | None): The lower bound
          for the mesh coordinate scaling. See the documentation in
          :meth:`~.extract` for more details.
        domain_bounds_upper_corner (warp.vec3f | tuple | None): The upper bound
          for the mesh coordinate scaling. See the documentation in
          :meth:`~.extract` for more details.
        topology (str): The type of the faces that :meth:`~.surface` produces:
          ``"triangle"`` or ``"quad"``.
        verts (warp.array | None): An array of vertex positions of type
          :class:`warp.vec3f` for the output mesh.
          This is populated by calling the :meth:`~.surface` method.
        indices (warp.array | None): An array of face vertex indices of type
          :class:`warp.int32` for the output mesh, with three consecutive
          indices per face when :attr:`topology` is ``"triangle"``, and four
          when it is ``"quad"``.
          This is populated by calling the :meth:`~.surface` method.
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        nz: int,
        *,
        domain_bounds_lower_corner: wp.vec3 | tuple[float, float, float] | None = None,
        domain_bounds_upper_corner: wp.vec3 | tuple[float, float, float] | None = None,
        topology: Literal["triangle", "quad"] = "triangle",
    ):
        _validate_topology(topology)

        super().__init__(
            nx,
            ny,
            nz,
            domain_bounds_lower_corner=domain_bounds_lower_corner,
            domain_bounds_upper_corner=domain_bounds_upper_corner,
        )

        self.topology = topology

    def surface(self, field: wp.array(dtype=float, ndim=3), threshold: float) -> None:
        """Compute a 2D surface mesh of a given isosurface from a 3D scalar field.

        This method is a convenience wrapper that calls the :meth:`~.extract`
        class method and stores the resulting mesh data in the :attr:`verts`
        and :attr:`indices` attributes, with the faces of the type selected by
        :attr:`topology`.

        Args:
          field: A 3D scalar field whose shape must match the grid dimensions
            (nx, ny, nz) of the instance.
          threshold: The field value defining the isosurface to extract.

        Raises:
          ValueError: If the shape of ``field`` does not match the configured
            grid dimensions of the instance.
        """
        self._check_field_shape(field)

        verts, indices = self.extract(
            field=field,
            threshold=wp.float32(threshold),
            domain_bounds_lower_corner=self.domain_bounds_lower_corner,
            domain_bounds_upper_corner=self.domain_bounds_upper_corner,
            topology=self.topology,
        )

        self.verts = verts
        self.indices = indices

    @classmethod
    def extract(
        cls,
        field: wp.array3d(dtype=wp.float32),
        threshold: float = 0.0,
        *,
        domain_bounds_lower_corner: wp.vec3 | tuple[float, float, float] | None = None,
        domain_bounds_upper_corner: wp.vec3 | tuple[float, float, float] | None = None,
        topology: Literal["triangle", "quad"] = "triangle",
    ) -> tuple[wp.array(dtype=wp.vec3), wp.array(dtype=wp.int32)]:
        """Extract a mesh from a 3D scalar field.

        This function generates an isosurface by processing the entire input ``field``.
        The resolution of the output mesh is determined by the shape of the ``field``
        array and may differ along each dimension.

        The coordinates of the mesh can be scaled to a specific bounding box
        using the ``domain_bounds_lower_corner`` and
        ``domain_bounds_upper_corner`` parameters. If a bound is not provided
        (i.e., left as ``None``), it will be assigned a default value that
        aligns the mesh with the integer indices of the input grid.

        Args:
            field: A 3D array representing the scalar values on a regular grid.
            threshold: The field value defining the isosurface to extract.
            domain_bounds_lower_corner: The 3D coordinate that the grid's corner
                at index (0,0,0) maps to. Defaults to ``(0.0, 0.0, 0.0)``
                if ``None``.
            domain_bounds_upper_corner: The 3D coordinate that the grid's corner
                at index (nx-1, ny-1, nz-1) maps to. Defaults to align with the
                grid's maximal indices if ``None``.
            topology: The type of the faces to produce: ``"triangle"`` or
                ``"quad"``.

        Returns:
            A tuple ``(vertices, indices)`` containing the output mesh data.
            The ``indices`` array is a flat list where each group of three
            consecutive integers forms a single triangle, or four a single
            quad, by referencing vertices in the ``vertices`` array.

        Raises:
            ValueError: If ``field`` is not a 3D array or is empty, or if
                ``topology`` is not a valid face type.
            TypeError: If the ``field`` data type is not ``wp.float32``.
        """
        # Do some validation
        validate_field(field)
        _validate_topology(topology)

        # Apply default policies for bounds and compute the grid spacing
        domain_bounds_lower_corner, grid_delta = resolve_domain_bounds(
            field.shape, domain_bounds_lower_corner, domain_bounds_upper_corner
        )

        return surface_nets_extract(field, threshold, domain_bounds_lower_corner, grid_delta, topology)
