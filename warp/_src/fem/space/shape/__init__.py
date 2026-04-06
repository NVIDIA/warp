# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
from enum import Enum

import warp as wp
from warp._src.fem.geometry import Element
from warp._src.fem.polynomial import Polynomial

from .cube_shape_function import (
    CubeBSplineShapeFunctions,
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
    SquareBSplineShapeFunctions,
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
    """Choice of basis function to equip individual elements."""

    LAGRANGE = "P"
    """Lagrange basis functions :math:`P_k` for simplices, tensor products :math:`Q_k` for squares and cubes."""
    SERENDIPITY = "S"
    """Serendipity elements :math:`S_k`, corresponding to Lagrange nodes with interior points removed (for degree <= 3)."""
    NONCONFORMING_POLYNOMIAL = "dP"
    """Simplex Lagrange basis functions :math:`P_{kd}` embedded into non conforming reference elements (e.g. squares or cubes). Discontinuous only."""
    NEDELEC_FIRST_KIND = "N1"
    """Nédélec (first kind) H(curl) shape functions. Should be used with covariant function space."""
    RAVIART_THOMAS = "RT"
    """Raviart-Thomas H(div) shape functions. Should be used with contravariant function space."""
    BSPLINE = "B"
    """B-spline basis functions. Should be used with grid-based geometries only."""


@functools.cache
def make_element_shape_function(
    element: Element,
    degree: int,
    element_basis: ElementBasis | None = None,
    family: Polynomial | None = None,
    scalar_type: type | None = None,
) -> ShapeFunction:
    """Equip a reference element with a shape function basis.

    Args:
        element: the type of reference element on which to build the shape function
        degree: polynomial degree of the per-element shape functions
        element_basis: type of basis function for the individual elements
        family: Polynomial family used to generate the shape function basis. If not provided, a reasonable basis is chosen.

    Returns:
        the corresponding shape function

    Raises:
        NotImplementedError: If the shape function is not implemented for the given element type
    """

    if scalar_type is None:
        scalar_type = wp.float32

    if element_basis is None:
        element_basis = ElementBasis.LAGRANGE
    elif element_basis == ElementBasis.SERENDIPITY and degree == 1:
        # Degree-1 serendipity is always equivalent to Lagrange
        element_basis = ElementBasis.LAGRANGE

    if degree == 0:
        return ConstantShapeFunction(element, scalar_type=scalar_type)

    if family is None:
        family = Polynomial.LOBATTO_GAUSS_LEGENDRE

    if element == Element.SQUARE:
        if element_basis == ElementBasis.NEDELEC_FIRST_KIND:
            return SquareNedelecFirstKindShapeFunctions(degree=degree, scalar_type=scalar_type)
        if element_basis == ElementBasis.RAVIART_THOMAS:
            return SquareRaviartThomasShapeFunctions(degree=degree, scalar_type=scalar_type)
        if element_basis == ElementBasis.NONCONFORMING_POLYNOMIAL:
            return SquareNonConformingPolynomialShapeFunctions(degree=degree, scalar_type=scalar_type)
        if element_basis == ElementBasis.SERENDIPITY and degree > 1:
            return SquareSerendipityShapeFunctions(degree=degree, family=family, scalar_type=scalar_type)
        if element_basis == ElementBasis.BSPLINE:
            return SquareBSplineShapeFunctions(degree=degree, scalar_type=scalar_type)

        return SquareBipolynomialShapeFunctions(degree=degree, family=family, scalar_type=scalar_type)
    if element == Element.TRIANGLE:
        if element_basis == ElementBasis.NEDELEC_FIRST_KIND:
            return TriangleNedelecFirstKindShapeFunctions(degree=degree, scalar_type=scalar_type)
        if element_basis == ElementBasis.RAVIART_THOMAS:
            return TriangleRaviartThomasShapeFunctions(degree=degree, scalar_type=scalar_type)
        if element_basis == ElementBasis.NONCONFORMING_POLYNOMIAL:
            return TriangleNonConformingPolynomialShapeFunctions(degree=degree, scalar_type=scalar_type)
        if element_basis == ElementBasis.SERENDIPITY and degree > 2:
            raise NotImplementedError("Serendipity variant not implemented yet for Triangle elements")

        return TrianglePolynomialShapeFunctions(degree=degree, scalar_type=scalar_type)

    if element == Element.CUBE:
        if element_basis == ElementBasis.NEDELEC_FIRST_KIND:
            return CubeNedelecFirstKindShapeFunctions(degree=degree, scalar_type=scalar_type)
        if element_basis == ElementBasis.RAVIART_THOMAS:
            return CubeRaviartThomasShapeFunctions(degree=degree, scalar_type=scalar_type)
        if element_basis == ElementBasis.NONCONFORMING_POLYNOMIAL:
            return CubeNonConformingPolynomialShapeFunctions(degree=degree, scalar_type=scalar_type)
        if element_basis == ElementBasis.SERENDIPITY and degree > 1:
            return CubeSerendipityShapeFunctions(degree=degree, family=family, scalar_type=scalar_type)
        if element_basis == ElementBasis.BSPLINE:
            return CubeBSplineShapeFunctions(degree=degree, scalar_type=scalar_type)

        return CubeTripolynomialShapeFunctions(degree=degree, family=family, scalar_type=scalar_type)
    if element == Element.TETRAHEDRON:
        if element_basis == ElementBasis.NEDELEC_FIRST_KIND:
            return TetrahedronNedelecFirstKindShapeFunctions(degree=degree, scalar_type=scalar_type)
        if element_basis == ElementBasis.RAVIART_THOMAS:
            return TetrahedronRaviartThomasShapeFunctions(degree=degree, scalar_type=scalar_type)
        if element_basis == ElementBasis.NONCONFORMING_POLYNOMIAL:
            return TetrahedronNonConformingPolynomialShapeFunctions(degree=degree, scalar_type=scalar_type)
        if element_basis == ElementBasis.SERENDIPITY and degree > 2:
            raise NotImplementedError("Serendipity variant not implemented yet for Tet elements")

        return TetrahedronPolynomialShapeFunctions(degree=degree, scalar_type=scalar_type)

    raise NotImplementedError(f"Unrecognized element type {element}")
