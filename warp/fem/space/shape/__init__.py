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

# isort: skip_file

# Keeping those imports non-deprecated for now,
# Will be deprecated after unification of the shape functions API (1.11+)

from warp._src.fem.space.shape.shape_function import ConstantShapeFunction as ConstantShapeFunction
from warp._src.fem.space.shape.cube_shape_function import (
    CubeNedelecFirstKindShapeFunctions as CubeNedelecFirstKindShapeFunctions,
)
from warp._src.fem.space.shape.cube_shape_function import (
    CubeNonConformingPolynomialShapeFunctions as CubeNonConformingPolynomialShapeFunctions,
)
from warp._src.fem.space.shape.cube_shape_function import (
    CubeRaviartThomasShapeFunctions as CubeRaviartThomasShapeFunctions,
)
from warp._src.fem.space.shape.cube_shape_function import CubeSerendipityShapeFunctions as CubeSerendipityShapeFunctions
from warp._src.fem.space.shape.cube_shape_function import CubeShapeFunction as CubeShapeFunction
from warp._src.fem.space.shape.cube_shape_function import (
    CubeTripolynomialShapeFunctions as CubeTripolynomialShapeFunctions,
)
from warp._src.fem.space.shape.square_shape_function import (
    SquareBipolynomialShapeFunctions as SquareBipolynomialShapeFunctions,
)
from warp._src.fem.space.shape.square_shape_function import (
    SquareNedelecFirstKindShapeFunctions as SquareNedelecFirstKindShapeFunctions,
)
from warp._src.fem.space.shape.square_shape_function import (
    SquareNonConformingPolynomialShapeFunctions as SquareNonConformingPolynomialShapeFunctions,
)
from warp._src.fem.space.shape.square_shape_function import (
    SquareRaviartThomasShapeFunctions as SquareRaviartThomasShapeFunctions,
)
from warp._src.fem.space.shape.square_shape_function import (
    SquareSerendipityShapeFunctions as SquareSerendipityShapeFunctions,
)
from warp._src.fem.space.shape.square_shape_function import SquareShapeFunction as SquareShapeFunction
from warp._src.fem.space.shape.tet_shape_function import (
    TetrahedronNedelecFirstKindShapeFunctions as TetrahedronNedelecFirstKindShapeFunctions,
)
from warp._src.fem.space.shape.tet_shape_function import (
    TetrahedronNonConformingPolynomialShapeFunctions as TetrahedronNonConformingPolynomialShapeFunctions,
)
from warp._src.fem.space.shape.tet_shape_function import (
    TetrahedronPolynomialShapeFunctions as TetrahedronPolynomialShapeFunctions,
)
from warp._src.fem.space.shape.tet_shape_function import (
    TetrahedronRaviartThomasShapeFunctions as TetrahedronRaviartThomasShapeFunctions,
)
from warp._src.fem.space.shape.tet_shape_function import TetrahedronShapeFunction as TetrahedronShapeFunction
from warp._src.fem.space.shape.triangle_shape_function import (
    TriangleNedelecFirstKindShapeFunctions as TriangleNedelecFirstKindShapeFunctions,
)
from warp._src.fem.space.shape.triangle_shape_function import (
    TriangleNonConformingPolynomialShapeFunctions as TriangleNonConformingPolynomialShapeFunctions,
)
from warp._src.fem.space.shape.triangle_shape_function import (
    TrianglePolynomialShapeFunctions as TrianglePolynomialShapeFunctions,
)
from warp._src.fem.space.shape.triangle_shape_function import (
    TriangleRaviartThomasShapeFunctions as TriangleRaviartThomasShapeFunctions,
)
from warp._src.fem.space.shape.triangle_shape_function import TriangleShapeFunction as TriangleShapeFunction


# TODO: Remove after cleaning up the public API.

from warp._src.fem.space import shape as _shape


def __getattr__(name):
    from warp._src.utils import get_deprecated_api  # noqa: PLC0415

    return get_deprecated_api(_shape, "warp.fem.space", name)
