# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for FEM geometry conversions.

This module provides functions for converting dense grid topologies to mesh topologies,
including triangular, tetrahedral, quadrilateral, and hexahedral meshes.
"""

# isort: skip_file

from warp._src.fem.utils import grid_to_quads as grid_to_quads
from warp._src.fem.utils import grid_to_tris as grid_to_tris
from warp._src.fem.utils import grid_to_tets as grid_to_tets
from warp._src.fem.utils import grid_to_hexes as grid_to_hexes
