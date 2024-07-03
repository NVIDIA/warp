from .cache import TemporaryStore, borrow_temporary, borrow_temporary_like, set_default_temporary_store
from .dirichlet import normalize_dirichlet_projector, project_linear_system
from .domain import BoundarySides, Cells, FrontierSides, GeometryDomain, Sides, Subdomain
from .field import DiscreteField, FieldLike, make_restriction, make_test, make_trial
from .geometry import (
    ExplicitGeometryPartition,
    Geometry,
    GeometryPartition,
    Grid2D,
    Grid3D,
    Hexmesh,
    LinearGeometryPartition,
    Nanogrid,
    Quadmesh2D,
    Tetmesh,
    Trimesh2D,
)
from .integrate import integrate, interpolate
from .operator import (
    D,
    at_node,
    average,
    curl,
    deformation_gradient,
    degree,
    div,
    div_outer,
    grad,
    grad_average,
    grad_jump,
    grad_outer,
    inner,
    integrand,
    jump,
    lookup,
    measure,
    measure_ratio,
    normal,
    outer,
    position,
)
from .polynomial import Polynomial
from .quadrature import ExplicitQuadrature, NodalQuadrature, PicQuadrature, Quadrature, RegularQuadrature
from .space import (
    BasisSpace,
    DofMapper,
    ElementBasis,
    FunctionSpace,
    PointBasisSpace,
    SkewSymmetricTensorMapper,
    SpacePartition,
    SpaceRestriction,
    SpaceTopology,
    SymmetricTensorMapper,
    make_collocated_function_space,
    make_polynomial_basis_space,
    make_polynomial_space,
    make_space_partition,
    make_space_restriction,
)
from .types import NULL_ELEMENT_INDEX, Coords, Domain, ElementIndex, Field, Sample, make_free_sample
