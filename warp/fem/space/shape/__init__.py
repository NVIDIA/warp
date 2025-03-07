# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from enum import Enum
from typing import Optional

from warp.fem.geometry import element as _element
from warp.fem.polynomial import Polynomial

from .cube_shape_function import (
    CubeNedelecFirstKindShapeFunctions,
    CubeNonConformingPolynomialShapeFunctions,
    CubeRaviartThomasShapeFunctions,
    CubeSerendipityShapeFunctions,
    CubeShapeFunction,
    CubeTripolynomialShapeFunctions,
)
from .shape_function import ConstantShapeFunction, ShapeFunction
from .square_shape_function import (
    SquareBipolynomialShapeFunctions,
    SquareNedelecFirstKindShapeFunctions,
    SquareNonConformingPolynomialShapeFunctions,
    SquareRaviartThomasShapeFunctions,
    SquareSerendipityShapeFunctions,
    SquareShapeFunction,
)
from .tet_shape_function import (
    TetrahedronNedelecFirstKindShapeFunctions,
    TetrahedronNonConformingPolynomialShapeFunctions,
    TetrahedronPolynomialShapeFunctions,
    TetrahedronRaviartThomasShapeFunctions,
    TetrahedronShapeFunction,
)
from .triangle_shape_function import (
    TriangleNedelecFirstKindShapeFunctions,
    TriangleNonConformingPolynomialShapeFunctions,
    TrianglePolynomialShapeFunctions,
    TriangleRaviartThomasShapeFunctions,
    TriangleShapeFunction,
)


class ElementBasis(Enum):
    """Choice of basis function to equip individual elements"""

    LAGRANGE = "P"
    """Lagrange basis functions :math:`P_k` for simplices, tensor products :math:`Q_k` for squares and cubes"""
    SERENDIPITY = "S"
    """Serendipity elements :math:`S_k`, corresponding to Lagrange nodes with interior points removed (for degree <= 3)"""
    NONCONFORMING_POLYNOMIAL = "dP"
    """Simplex Lagrange basis functions :math:`P_{kd}` embedded into non conforming reference elements (e.g. squares or cubes). Discontinuous only."""
    NEDELEC_FIRST_KIND = "N1"
    """Nédélec (first kind) H(curl) shape functions. Should be used with covariant function space."""
    RAVIART_THOMAS = "RT"
    """Raviart-Thomas H(div) shape functions. Should be used with contravariant function space."""


def get_shape_function(
    element: _element.Element,
    space_dimension: int,
    degree: int,
    element_basis: ElementBasis,
    family: Optional[Polynomial] = None,
):
    """
    Equips a reference element with a shape function basis.

    Args:
        element: the reference element on which to build the shape function
        space_dimension: the dimension of the embedding space
        degree: polynomial degree of the per-element shape functions
        element_basis: type of basis function for the individual elements
        family: Polynomial family used to generate the shape function basis. If not provided, a reasonable basis is chosen.

    Returns:
        the corresponding shape function
    """

    if degree == 0:
        return ConstantShapeFunction(element, space_dimension)

    if family is None:
        family = Polynomial.LOBATTO_GAUSS_LEGENDRE

    if isinstance(element, _element.Square):
        if element_basis == ElementBasis.NEDELEC_FIRST_KIND:
            return SquareNedelecFirstKindShapeFunctions(degree=degree)
        if element_basis == ElementBasis.RAVIART_THOMAS:
            return SquareRaviartThomasShapeFunctions(degree=degree)
        if element_basis == ElementBasis.NONCONFORMING_POLYNOMIAL:
            return SquareNonConformingPolynomialShapeFunctions(degree=degree)
        if element_basis == ElementBasis.SERENDIPITY and degree > 1:
            return SquareSerendipityShapeFunctions(degree=degree, family=family)

        return SquareBipolynomialShapeFunctions(degree=degree, family=family)
    if isinstance(element, _element.Triangle):
        if element_basis == ElementBasis.NEDELEC_FIRST_KIND:
            return TriangleNedelecFirstKindShapeFunctions(degree=degree)
        if element_basis == ElementBasis.RAVIART_THOMAS:
            return TriangleRaviartThomasShapeFunctions(degree=degree)
        if element_basis == ElementBasis.NONCONFORMING_POLYNOMIAL:
            return TriangleNonConformingPolynomialShapeFunctions(degree=degree)
        if element_basis == ElementBasis.SERENDIPITY and degree > 2:
            raise NotImplementedError("Serendipity variant not implemented yet for Triangle elements")

        return TrianglePolynomialShapeFunctions(degree=degree)

    if isinstance(element, _element.Cube):
        if element_basis == ElementBasis.NEDELEC_FIRST_KIND:
            return CubeNedelecFirstKindShapeFunctions(degree=degree)
        if element_basis == ElementBasis.RAVIART_THOMAS:
            return CubeRaviartThomasShapeFunctions(degree=degree)
        if element_basis == ElementBasis.NONCONFORMING_POLYNOMIAL:
            return CubeNonConformingPolynomialShapeFunctions(degree=degree)
        if element_basis == ElementBasis.SERENDIPITY and degree > 1:
            return CubeSerendipityShapeFunctions(degree=degree, family=family)

        return CubeTripolynomialShapeFunctions(degree=degree, family=family)
    if isinstance(element, _element.Tetrahedron):
        if element_basis == ElementBasis.NEDELEC_FIRST_KIND:
            return TetrahedronNedelecFirstKindShapeFunctions(degree=degree)
        if element_basis == ElementBasis.RAVIART_THOMAS:
            return TetrahedronRaviartThomasShapeFunctions(degree=degree)
        if element_basis == ElementBasis.NONCONFORMING_POLYNOMIAL:
            return TetrahedronNonConformingPolynomialShapeFunctions(degree=degree)
        if element_basis == ElementBasis.SERENDIPITY and degree > 2:
            raise NotImplementedError("Serendipity variant not implemented yet for Tet elements")

        return TetrahedronPolynomialShapeFunctions(degree=degree)

    return NotImplementedError("Unrecognized element type")
