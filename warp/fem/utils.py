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


# TODO: Remove after cleaning up the public API.

from warp._src.fem import linalg as _fem_linalg
from warp._src.fem import utils as _utils


def __getattr__(name):
    from warp._src.utils import get_deprecated_api, warn  # noqa: PLC0415

    # Symbols that have been moved to warp.fem.linalg
    if name in ("array_axpy", "inverse_qr", "symmetric_eigenvalues_qr"):
        warn(
            f"The symbol `warp.fem.utils.{name}` will soon be removed from the public API. Use `warp.fem.linalg.{name}` instead.",
            DeprecationWarning,
        )
        return getattr(_fem_linalg, name)

    return get_deprecated_api(_utils, "warp.fem", name)
