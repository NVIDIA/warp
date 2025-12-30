# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
