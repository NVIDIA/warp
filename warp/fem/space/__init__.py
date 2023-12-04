from typing import Optional
from enum import Enum

import warp.fem.domain as _domain
import warp.fem.geometry as _geometry
import warp.fem.polynomial as _polynomial

from .function_space import FunctionSpace
from .topology import SpaceTopology
from .basis_space import BasisSpace, PointBasisSpace
from .collocated_function_space import CollocatedFunctionSpace

from .grid_2d_function_space import (
    GridPiecewiseConstantBasis,
    GridBipolynomialBasisSpace,
    GridDGBipolynomialBasisSpace,
    GridSerendipityBasisSpace,
    GridDGSerendipityBasisSpace,
    GridDGPolynomialBasisSpace,
)
from .grid_3d_function_space import (
    GridTripolynomialBasisSpace,
    GridDGTripolynomialBasisSpace,
    Grid3DPiecewiseConstantBasis,
    Grid3DSerendipityBasisSpace,
    Grid3DDGSerendipityBasisSpace,
    Grid3DDGPolynomialBasisSpace,
)
from .trimesh_2d_function_space import (
    Trimesh2DPiecewiseConstantBasis,
    Trimesh2DPolynomialBasisSpace,
    Trimesh2DDGPolynomialBasisSpace,
    Trimesh2DNonConformingPolynomialBasisSpace,
)
from .tetmesh_function_space import (
    TetmeshPiecewiseConstantBasis,
    TetmeshPolynomialBasisSpace,
    TetmeshDGPolynomialBasisSpace,
    TetmeshNonConformingPolynomialBasisSpace,
)
from .quadmesh_2d_function_space import (
    Quadmesh2DPiecewiseConstantBasis,
    Quadmesh2DBipolynomialBasisSpace,
    Quadmesh2DDGBipolynomialBasisSpace,
    Quadmesh2DSerendipityBasisSpace,
    Quadmesh2DDGSerendipityBasisSpace,
    Quadmesh2DPolynomialBasisSpace,
)
from .hexmesh_function_space import (
    HexmeshPiecewiseConstantBasis,
    HexmeshTripolynomialBasisSpace,
    HexmeshDGTripolynomialBasisSpace,
    HexmeshSerendipityBasisSpace,
    HexmeshDGSerendipityBasisSpace,
    HexmeshPolynomialBasisSpace,
)

from .partition import SpacePartition, make_space_partition
from .restriction import SpaceRestriction


from .dof_mapper import DofMapper, IdentityMapper, SymmetricTensorMapper, SkewSymmetricTensorMapper


def make_space_restriction(
    space: Optional[FunctionSpace] = None,
    space_partition: Optional[SpacePartition] = None,
    domain: Optional[_domain.GeometryDomain] = None,
    space_topology: Optional[SpaceTopology] = None,
    device=None,
    temporary_store: "Optional[warp.fem.cache.TemporaryStore]" = None,
) -> SpaceRestriction:
    """
    Restricts a function space partition to a Domain, i.e. a subset of its elements.

    One of `space_partition`, `space_topology`, or `space` must be provided (and will be considered in that order).

    Args:
        space: (deprecated) if neither `space_partition` nor `space_topology` are provided, the space defining the topology to restrict
        space_partition: the subset of nodes from the space topology to consider
        domain: the domain to restrict the space to, defaults to all cells of the space geometry or partition.
        space_topology: the space topology to be restricted, if `space_partition` is ``None``.
        device: device on which to perform and store computations
        temporary_store: shared pool from which to allocate temporary arrays
    """

    if space_partition is None:
        if space_topology is None:
            assert space is not None
            space_topology = space.topology

        if domain is None:
            domain = _domain.Cells(geometry=space_topology.geometry)

        space_partition = make_space_partition(
            space_topology=space_topology, geometry_partition=domain.geometry_partition
        )
    elif domain is None:
        domain = _domain.Cells(geometry=space_partition.geo_partition)

    return SpaceRestriction(
        space_partition=space_partition, domain=domain, device=device, temporary_store=temporary_store
    )


class ElementBasis(Enum):
    """Choice of basis function to equip individual elements"""

    LAGRANGE = 0
    """Lagrange basis functions :math:`P_k` for simplices, tensor products :math:`Q_k` for squares and cubes"""
    SERENDIPITY = 1
    """Serendipity elements :math:`S_k`, corresponding to Lagrange nodes with interior points removed (for degree <= 3)"""
    NONCONFORMING_POLYNOMIAL = 2
    """Simplex Lagrange basis functions :math:`P_{kd}` embedded into non conforming reference elements (e.g. squares or cubes). Discontinuous only."""


def make_polynomial_basis_space(
    geo: _geometry.Geometry,
    degree: int = 1,
    element_basis: Optional[ElementBasis] = None,
    discontinuous: bool = False,
    family: Optional[_polynomial.Polynomial] = None,
) -> BasisSpace:
    """
    Equips a geometry with a polynomial basis.

    Args:
        geo: the Geometry on which to build the space
        degree: polynomial degree of the per-element shape functions
        discontinuous: if True, use Discontinuous Galerkin shape functions. Discontinuous is implied if degree is 0, i.e, piecewise-constant shape functions.
        element_basis: type of basis function for the individual elements
        family: Polynomial family used to generate the shape function basis. If not provided, a reasonable basis is chosen.

    Returns:
        the constructed basis space
    """

    base_geo = geo.base if isinstance(geo, _geometry.DeformedGeometry) else geo

    if element_basis is None:
        element_basis = ElementBasis.LAGRANGE

    if isinstance(base_geo, _geometry.Grid2D):
        if degree == 0:
            return GridPiecewiseConstantBasis(geo)

        if element_basis == ElementBasis.SERENDIPITY and degree > 1:
            if discontinuous:
                return GridDGSerendipityBasisSpace(geo, degree=degree, family=family)
            else:
                return GridSerendipityBasisSpace(geo, degree=degree, family=family)

        if element_basis == ElementBasis.NONCONFORMING_POLYNOMIAL:
            return GridDGPolynomialBasisSpace(geo, degree=degree)

        if discontinuous:
            return GridDGBipolynomialBasisSpace(geo, degree=degree, family=family)
        else:
            return GridBipolynomialBasisSpace(geo, degree=degree, family=family)

    if isinstance(base_geo, _geometry.Grid3D):
        if degree == 0:
            return Grid3DPiecewiseConstantBasis(geo)

        if element_basis == ElementBasis.SERENDIPITY and degree > 1:
            if discontinuous:
                return Grid3DDGSerendipityBasisSpace(geo, degree=degree, family=family)
            else:
                return Grid3DSerendipityBasisSpace(geo, degree=degree, family=family)

        if element_basis == ElementBasis.NONCONFORMING_POLYNOMIAL:
            return Grid3DDGPolynomialBasisSpace(geo, degree=degree)

        if discontinuous:
            return GridDGTripolynomialBasisSpace(geo, degree=degree, family=family)
        else:
            return GridTripolynomialBasisSpace(geo, degree=degree, family=family)

    if isinstance(base_geo, _geometry.Trimesh2D):
        if degree == 0:
            return Trimesh2DPiecewiseConstantBasis(geo)

        if element_basis == ElementBasis.SERENDIPITY and degree > 2:
            raise NotImplementedError("Serendipity variant not implemented yet")

        if element_basis == ElementBasis.NONCONFORMING_POLYNOMIAL:
            return Trimesh2DNonConformingPolynomialBasisSpace(geo, degree=degree)

        if discontinuous:
            return Trimesh2DDGPolynomialBasisSpace(geo, degree=degree)
        else:
            return Trimesh2DPolynomialBasisSpace(geo, degree=degree)

    if isinstance(base_geo, _geometry.Tetmesh):
        if degree == 0:
            return TetmeshPiecewiseConstantBasis(geo)

        if element_basis == ElementBasis.SERENDIPITY and degree > 2:
            raise NotImplementedError("Serendipity variant not implemented yet")

        if element_basis == ElementBasis.NONCONFORMING_POLYNOMIAL:
            return TetmeshNonConformingPolynomialBasisSpace(geo, degree=degree)

        if discontinuous:
            return TetmeshDGPolynomialBasisSpace(geo, degree=degree)
        else:
            return TetmeshPolynomialBasisSpace(geo, degree=degree)

    if isinstance(base_geo, _geometry.Quadmesh2D):
        if degree == 0:
            return Quadmesh2DPiecewiseConstantBasis(geo)

        if element_basis == ElementBasis.SERENDIPITY and degree > 1:
            if discontinuous:
                return Quadmesh2DDGSerendipityBasisSpace(geo, degree=degree, family=family)
            else:
                return Quadmesh2DSerendipityBasisSpace(geo, degree=degree, family=family)

        if element_basis == ElementBasis.NONCONFORMING_POLYNOMIAL:
            return Quadmesh2DPolynomialBasisSpace(geo, degree=degree)

        if discontinuous:
            return Quadmesh2DDGBipolynomialBasisSpace(geo, degree=degree, family=family)
        else:
            return Quadmesh2DBipolynomialBasisSpace(geo, degree=degree, family=family)

    if isinstance(base_geo, _geometry.Hexmesh):
        if degree == 0:
            return HexmeshPiecewiseConstantBasis(geo)

        if element_basis == ElementBasis.SERENDIPITY and degree > 1:
            if discontinuous:
                return HexmeshDGSerendipityBasisSpace(geo, degree=degree, family=family)
            else:
                return HexmeshSerendipityBasisSpace(geo, degree=degree, family=family)

        if element_basis == ElementBasis.NONCONFORMING_POLYNOMIAL:
            return HexmeshPolynomialBasisSpace(geo, degree=degree)

        if discontinuous:
            return HexmeshDGTripolynomialBasisSpace(geo, degree=degree, family=family)
        else:
            return HexmeshTripolynomialBasisSpace(geo, degree=degree, family=family)

    raise NotImplementedError()


def make_collocated_function_space(
    basis_space: BasisSpace, dtype: type = float, dof_mapper: Optional[DofMapper] = None
) -> CollocatedFunctionSpace:
    """
    Constructs a function space from a basis space and a value type, such that all degrees of freedom of the value type are stored at each of the basis nodes.

    Args:
        geo: the Geometry on which to build the space
        dtype: value type the function space. If ``dof_mapper`` is provided, the value type from the DofMapper will be used instead.
        dof_mapper: mapping from node degrees of freedom to function values, defaults to Identity. Useful for reduced coordinates, e.g. :py:class:`SymmetricTensorMapper` maps 2x2 (resp 3x3) symmetric tensors to 3 (resp 6) degrees of freedom.

    Returns:
        the constructed function space
    """
    return CollocatedFunctionSpace(basis_space, dtype=dtype, dof_mapper=dof_mapper)


def make_polynomial_space(
    geo: _geometry.Geometry,
    dtype: type = float,
    dof_mapper: Optional[DofMapper] = None,
    degree: int = 1,
    element_basis: Optional[ElementBasis] = None,
    discontinuous: bool = False,
    family: Optional[_polynomial.Polynomial] = None,
) -> CollocatedFunctionSpace:
    """
    Equips a geometry with a collocated, polynomial function space.
    Equivalent to successive calls to :func:`make_polynomial_basis_space` and `make_collocated_function_space`.

    Args:
        geo: the Geometry on which to build the space
        dtype: value type the function space. If ``dof_mapper`` is provided, the value type from the DofMapper will be used instead.
        dof_mapper: mapping from node degrees of freedom to function values, defaults to Identity. Useful for reduced coordinates, e.g. :py:class:`SymmetricTensorMapper` maps 2x2 (resp 3x3) symmetric tensors to 3 (resp 6) degrees of freedom.
        degree: polynomial degree of the per-element shape functions
        discontinuous: if True, use Discontinuous Galerkin shape functions. Discontinuous is implied if degree is 0, i.e, piecewise-constant shape functions.
        element_basis: type of basis function for the individual elements
        family: Polynomial family used to generate the shape function basis. If not provided, a reasonable basis is chosen.

    Returns:
        the constructed function space
    """

    basis_space = make_polynomial_basis_space(geo, degree, element_basis, discontinuous, family)
    return CollocatedFunctionSpace(basis_space, dtype=dtype, dof_mapper=dof_mapper)
