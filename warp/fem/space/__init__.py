from typing import Optional

import warp.fem.domain
import warp.fem.geometry
import warp.fem.polynomial

from .function_space import FunctionSpace
from .nodal_function_space import NodalFunctionSpace

from .grid_2d_function_space import (
    GridPiecewiseConstantSpace,
    GridBipolynomialSpace,
    GridDGBipolynomialSpace,
)
from .grid_3d_function_space import (
    GridTripolynomialSpace,
    GridDGTripolynomialSpace,
    Grid3DPiecewiseConstantSpace,
)
from .trimesh_2d_function_space import (
    Trimesh2DPiecewiseConstantSpace,
    Trimesh2DPolynomialSpace,
    Trimesh2DDGPolynomialSpace,
)
from .tetmesh_function_space import TetmeshPiecewiseConstantSpace, TetmeshPolynomialSpace, TetmeshDGPolynomialSpace

from .partition import SpacePartition, make_space_partition
from .restriction import SpaceRestriction


from .dof_mapper import DofMapper, IdentityMapper, SymmetricTensorMapper


def make_space_restriction(
    space: FunctionSpace,
    space_partition: Optional[SpacePartition] = None,
    domain: Optional[warp.fem.domain.GeometryDomain] = None,
    device=None,
    temporary_store: "Optional[warp.fem.cache.TemporaryStore]" = None,
) -> SpaceRestriction:
    """
    Restricts a function space to a Domain, i.e. a subset of its elements.

    Args:
        space: the space to be restricted
        space_partition: if provided, the subset of nodes from ``space`` to consider
        domain: the domain to restrict the space to, defaults to all cells of the space geometry or partition.
        device: device on which to perform and store computations
        temporary_store: shared pool from which to allocate temporary arrays
    """
    if space_partition is None:
        if domain is None:
            domain = warp.fem.domain.Cells(geometry=space.geometry)
        space_partition = make_space_partition(space, domain.geometry_partition)
    elif domain is None:
        domain = warp.fem.domain.Cells(geometry=space_partition.geo_partition)

    return SpaceRestriction(
        space=space, space_partition=space_partition, domain=domain, device=device, temporary_store=temporary_store
    )


def make_polynomial_space(
    geo: warp.fem.geometry.Geometry,
    dtype: type = float,
    dof_mapper: Optional[DofMapper] = None,
    degree: int = 1,
    discontinuous: bool = False,
    family: Optional[warp.fem.polynomial.Polynomial] = None,
) -> FunctionSpace:
    """
    Equip elements of a geometry with a Lagrange polynomial function space

    Args:
        geo: the Geometry on which to build the space
        dtype: value type the function space. If ``dof_mapper`` is provided, the value type from the DofMapper will be used instead.
        dof_mapper: mapping from node degrees of freedom to function values, defaults to Identity. Useful for reduced coordinates, e.g. :py:class:`SymmetricTensorMapper` maps 2x2 (resp 3x3) symmetric tensors to 3 (resp 6) degrees of freedom.
        degree: polynomial degree of the per-element shape functions
        discontinuous: if True, use Discontinuous Galerkin shape functions. Discontinuous is implied if degree is 0, i.e, piecewise-constant shape functions.
        family: Polynomial family used to generate the shape function basis. If not provided, a reasonable basis is chosen.

    Returns:
        the constructed function space
    """

    if isinstance(geo, warp.fem.geometry.Grid2D):
        if degree == 0:
            return GridPiecewiseConstantSpace(geo, dtype=dtype, dof_mapper=dof_mapper)

        if discontinuous:
            return GridDGBipolynomialSpace(geo, dtype=dtype, dof_mapper=dof_mapper, degree=degree, family=family)
        else:
            return GridBipolynomialSpace(geo, dtype=dtype, dof_mapper=dof_mapper, degree=degree, family=family)

    if isinstance(geo, warp.fem.geometry.Grid3D):
        if degree == 0:
            return Grid3DPiecewiseConstantSpace(geo, dtype=dtype, dof_mapper=dof_mapper)

        if discontinuous:
            return GridDGTripolynomialSpace(geo, dtype=dtype, dof_mapper=dof_mapper, degree=degree, family=family)
        else:
            return GridTripolynomialSpace(geo, dtype=dtype, dof_mapper=dof_mapper, degree=degree, family=family)

    if isinstance(geo, warp.fem.geometry.Trimesh2D):
        if degree == 0:
            return Trimesh2DPiecewiseConstantSpace(geo, dtype=dtype, dof_mapper=dof_mapper)

        if discontinuous:
            return Trimesh2DDGPolynomialSpace(geo, dtype=dtype, dof_mapper=dof_mapper, degree=degree)
        else:
            return Trimesh2DPolynomialSpace(geo, dtype=dtype, dof_mapper=dof_mapper, degree=degree)

    if isinstance(geo, warp.fem.geometry.Tetmesh):
        if degree == 0:
            return TetmeshPiecewiseConstantSpace(geo, dtype=dtype, dof_mapper=dof_mapper)

        if discontinuous:
            return TetmeshDGPolynomialSpace(geo, dtype=dtype, dof_mapper=dof_mapper, degree=degree)
        else:
            return TetmeshPolynomialSpace(geo, dtype=dtype, dof_mapper=dof_mapper, degree=degree)

    raise NotImplementedError
