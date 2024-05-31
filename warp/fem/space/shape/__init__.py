from enum import Enum
from typing import Optional

from warp.fem.geometry import element as _element
from warp.fem.polynomial import Polynomial

from .cube_shape_function import (
    CubeNonConformingPolynomialShapeFunctions,
    CubeSerendipityShapeFunctions,
    CubeTripolynomialShapeFunctions,
)
from .shape_function import ConstantShapeFunction, ShapeFunction
from .square_shape_function import (
    SquareBipolynomialShapeFunctions,
    SquareNonConformingPolynomialShapeFunctions,
    SquareSerendipityShapeFunctions,
)
from .tet_shape_function import TetrahedronNonConformingPolynomialShapeFunctions, TetrahedronPolynomialShapeFunctions
from .triangle_shape_function import Triangle2DNonConformingPolynomialShapeFunctions, Triangle2DPolynomialShapeFunctions


class ElementBasis(Enum):
    """Choice of basis function to equip individual elements"""

    LAGRANGE = 0
    """Lagrange basis functions :math:`P_k` for simplices, tensor products :math:`Q_k` for squares and cubes"""
    SERENDIPITY = 1
    """Serendipity elements :math:`S_k`, corresponding to Lagrange nodes with interior points removed (for degree <= 3)"""
    NONCONFORMING_POLYNOMIAL = 2
    """Simplex Lagrange basis functions :math:`P_{kd}` embedded into non conforming reference elements (e.g. squares or cubes). Discontinuous only."""


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
        if element_basis == ElementBasis.NONCONFORMING_POLYNOMIAL:
            return SquareNonConformingPolynomialShapeFunctions(degree=degree)
        if element_basis == ElementBasis.SERENDIPITY and degree > 1:
            return SquareSerendipityShapeFunctions(degree=degree, family=family)

        return SquareBipolynomialShapeFunctions(degree=degree, family=family)
    if isinstance(element, _element.Triangle):
        if element_basis == ElementBasis.NONCONFORMING_POLYNOMIAL:
            return Triangle2DNonConformingPolynomialShapeFunctions(degree=degree)
        if element_basis == ElementBasis.SERENDIPITY and degree > 2:
            raise NotImplementedError("Serendipity variant not implemented yet for Triangle elements")

        return Triangle2DPolynomialShapeFunctions(degree=degree)

    if isinstance(element, _element.Cube):
        if element_basis == ElementBasis.NONCONFORMING_POLYNOMIAL:
            return CubeNonConformingPolynomialShapeFunctions(degree=degree)
        if element_basis == ElementBasis.SERENDIPITY and degree > 1:
            return CubeSerendipityShapeFunctions(degree=degree, family=family)

        return CubeTripolynomialShapeFunctions(degree=degree, family=family)
    if isinstance(element, _element.Tetrahedron):
        if element_basis == ElementBasis.NONCONFORMING_POLYNOMIAL:
            return TetrahedronNonConformingPolynomialShapeFunctions(degree=degree)
        if element_basis == ElementBasis.SERENDIPITY and degree > 2:
            raise NotImplementedError("Serendipity variant not implemented yet for Tet elements")

        return TetrahedronPolynomialShapeFunctions(degree=degree)

    return NotImplementedError("Unrecognized element type")
