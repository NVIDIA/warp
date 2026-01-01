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

"""Finite Element Method (FEM) toolkit for solving differential equations.

This module provides tools for solving physical systems described as partial differential
equations (PDEs) using finite-element-based Galerkin methods. It supports diffusion,
convection, fluid flow, and elasticity problems with various FEM formulations and
discretization schemes.

The core workflow involves defining geometries (grids, meshes, NanoVDB volumes), function
spaces with shape functions, integration domains, and using the :func:`integrate` function
with :func:`integrand`-decorated kernels to build linear and bilinear forms for solving
linear systems.

Usage:
    This module must be explicitly imported::

        import warp.fem

See Also:
    :doc:`/domain_modules/fem` for comprehensive documentation and examples.
"""

# isort: skip_file

from warp._src.fem.geometry.adaptive_nanogrid import AdaptiveNanogrid as AdaptiveNanogrid
from warp._src.fem.space.basis_space import BasisSpace as BasisSpace
from warp._src.fem.domain import BoundarySides as BoundarySides
from warp._src.fem.geometry.partition import CellBasedGeometryPartition as CellBasedGeometryPartition
from warp._src.fem.domain import Cells as Cells
from warp._src.fem.types import Coords as Coords
from warp._src.fem.field.field import DiscreteField as DiscreteField
from warp._src.fem.space.dof_mapper import DofMapper as DofMapper
from warp._src.fem.types import Domain as Domain
from warp._src.fem.geometry.element import Element as Element
from warp._src.fem.space.shape import ElementBasis as ElementBasis
from warp._src.fem.types import ElementIndex as ElementIndex
from warp._src.fem.types import ElementKind as ElementKind
from warp._src.fem.geometry.partition import ExplicitGeometryPartition as ExplicitGeometryPartition
from warp._src.fem.quadrature.quadrature import ExplicitQuadrature as ExplicitQuadrature
from warp._src.fem.types import Field as Field
from warp._src.fem.field.field import FieldLike as FieldLike
from warp._src.fem.domain import FrontierSides as FrontierSides
from warp._src.fem.space.function_space import FunctionSpace as FunctionSpace
from warp._src.fem.geometry.geometry import Geometry as Geometry
from warp._src.fem.domain import GeometryDomain as GeometryDomain
from warp._src.fem.field.field import GeometryField as GeometryField
from warp._src.fem.geometry.partition import GeometryPartition as GeometryPartition
from warp._src.fem.geometry.grid_2d import Grid2D as Grid2D
from warp._src.fem.geometry.grid_3d import Grid3D as Grid3D
from warp._src.fem.geometry.hexmesh import Hexmesh as Hexmesh
from warp._src.fem.field.field import ImplicitField as ImplicitField
from warp._src.fem.operator import Integrand as Integrand
from warp._src.fem.geometry.partition import LinearGeometryPartition as LinearGeometryPartition
from warp._src.fem.geometry.nanogrid import Nanogrid as Nanogrid
from warp._src.fem.quadrature.quadrature import NodalQuadrature as NodalQuadrature
from warp._src.fem.types import NodeIndex as NodeIndex
from warp._src.fem.field.field import NonconformingField as NonconformingField
from warp._src.fem.operator import Operator as Operator
from warp._src.fem.quadrature.pic_quadrature import PicQuadrature as PicQuadrature
from warp._src.fem.space.point_basis_space import PointBasisSpace as PointBasisSpace
from warp._src.fem.polynomial import Polynomial as Polynomial
from warp._src.fem.geometry.quadmesh import Quadmesh2D as Quadmesh2D
from warp._src.fem.geometry.quadmesh import Quadmesh3D as Quadmesh3D
from warp._src.fem.quadrature.quadrature import Quadrature as Quadrature
from warp._src.fem.types import QuadraturePointIndex as QuadraturePointIndex
from warp._src.fem.quadrature.quadrature import RegularQuadrature as RegularQuadrature
from warp._src.fem.space.basis_space import ShapeBasisSpace as ShapeBasisSpace
from warp._src.fem.space.shape.shape_function import ShapeFunction as ShapeFunction
from warp._src.fem.domain import Sides as Sides
from warp._src.fem.space.dof_mapper import SkewSymmetricTensorMapper as SkewSymmetricTensorMapper
from warp._src.fem.space.partition import SpacePartition as SpacePartition
from warp._src.fem.space.restriction import SpaceRestriction as SpaceRestriction
from warp._src.fem.space.topology import SpaceTopology as SpaceTopology
from warp._src.fem.domain import Subdomain as Subdomain
from warp._src.fem.space.dof_mapper import SymmetricTensorMapper as SymmetricTensorMapper
from warp._src.fem.cache import Temporary as Temporary
from warp._src.fem.cache import TemporaryStore as TemporaryStore
from warp._src.fem.geometry.tetmesh import Tetmesh as Tetmesh
from warp._src.fem.geometry.trimesh import Trimesh2D as Trimesh2D
from warp._src.fem.geometry.trimesh import Trimesh3D as Trimesh3D
from warp._src.fem.field.field import UniformField as UniformField
from warp._src.fem.adaptivity import adaptive_nanogrid_from_field as adaptive_nanogrid_from_field
from warp._src.fem.adaptivity import adaptive_nanogrid_from_hierarchy as adaptive_nanogrid_from_hierarchy
from warp._src.fem.operator import at_node as at_node
from warp._src.fem.operator import average as average
from warp._src.fem.cache import borrow_temporary as borrow_temporary
from warp._src.fem.cache import borrow_temporary_like as borrow_temporary_like
from warp._src.fem.operator import cells as cells
from warp._src.fem.operator import curl as curl
from warp._src.fem.operator import D as D
from warp._src.fem.operator import deformation_gradient as deformation_gradient
from warp._src.fem.operator import degree as degree
from warp._src.fem.operator import div as div
from warp._src.fem.operator import div_outer as div_outer
from warp._src.fem.operator import element_closest_point as element_closest_point
from warp._src.fem.operator import element_coordinates as element_coordinates
from warp._src.fem.operator import element_index as element_index
from warp._src.fem.operator import element_partition_index as element_partition_index
from warp._src.fem.operator import grad as grad
from warp._src.fem.operator import grad_average as grad_average
from warp._src.fem.operator import grad_jump as grad_jump
from warp._src.fem.operator import grad_outer as grad_outer
from warp._src.fem.operator import inner as inner
from warp._src.fem.operator import integrand as integrand
from warp._src.fem.integrate import integrate as integrate
from warp._src.fem.integrate import interpolate as interpolate
from warp._src.fem.operator import jump as jump
from warp._src.fem.operator import lookup as lookup
from warp._src.fem.space import make_collocated_function_space as make_collocated_function_space
from warp._src.fem.space import make_contravariant_function_space as make_contravariant_function_space
from warp._src.fem.space import make_covariant_function_space as make_covariant_function_space
from warp._src.fem.field import make_discrete_field as make_discrete_field
from warp._src.fem.space import make_element_based_space_topology as make_element_based_space_topology
from warp._src.fem.space.shape import make_element_shape_function as make_element_shape_function
from warp._src.fem.types import make_free_sample as make_free_sample
from warp._src.fem.space import make_polynomial_basis_space as make_polynomial_basis_space
from warp._src.fem.space import make_polynomial_space as make_polynomial_space
from warp._src.fem.field import make_restriction as make_restriction
from warp._src.fem.space.partition import make_space_partition as make_space_partition
from warp._src.fem.space import make_space_restriction as make_space_restriction
from warp._src.fem.field import make_test as make_test
from warp._src.fem.field import make_trial as make_trial
from warp._src.fem.operator import measure as measure
from warp._src.fem.operator import measure_ratio as measure_ratio
from warp._src.fem.operator import node_count as node_count
from warp._src.fem.operator import node_index as node_index
from warp._src.fem.operator import node_inner_weight as node_inner_weight
from warp._src.fem.operator import node_inner_weight_gradient as node_inner_weight_gradient
from warp._src.fem.operator import node_outer_weight as node_outer_weight
from warp._src.fem.operator import node_outer_weight_gradient as node_outer_weight_gradient
from warp._src.fem.operator import node_partition_index as node_partition_index
from warp._src.fem.operator import normal as normal
from warp._src.fem.dirichlet import normalize_dirichlet_projector as normalize_dirichlet_projector
from warp._src.fem.operator import outer as outer
from warp._src.fem.operator import partition_lookup as partition_lookup
from warp._src.fem.operator import position as position
from warp._src.fem.dirichlet import project_linear_system as project_linear_system
from warp._src.fem.dirichlet import project_system_matrix as project_system_matrix
from warp._src.fem.dirichlet import project_system_rhs as project_system_rhs
from warp._src.fem.types import Sample as Sample
from warp._src.fem.cache import set_default_temporary_store as set_default_temporary_store
from warp._src.fem.operator import to_cell_side as to_cell_side
from warp._src.fem.operator import to_inner_cell as to_inner_cell
from warp._src.fem.operator import to_outer_cell as to_outer_cell
from warp._src.fem.types import NULL_ELEMENT_INDEX as NULL_ELEMENT_INDEX
from warp._src.fem.types import NULL_NODE_INDEX as NULL_NODE_INDEX
from warp._src.fem.types import NULL_QP_INDEX as NULL_QP_INDEX
from warp._src.fem.types import OUTSIDE as OUTSIDE

from . import cache as cache
from . import field as field
from . import geometry as geometry
from . import linalg as linalg
from . import polynomial as polynomial
from . import space as space
from . import utils as utils


# TODO: Remove after cleaning up the public API.

from warp._src import fem as _fem


def __getattr__(name):
    from warp._src.utils import get_deprecated_api  # noqa: PLC0415

    return get_deprecated_api(_fem, "warp", name)
