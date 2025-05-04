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

from .adaptivity import adaptive_nanogrid_from_field, adaptive_nanogrid_from_hierarchy
from .cache import TemporaryStore, borrow_temporary, borrow_temporary_like, set_default_temporary_store
from .dirichlet import normalize_dirichlet_projector, project_linear_system
from .domain import BoundarySides, Cells, FrontierSides, GeometryDomain, Sides, Subdomain
from .field import (
    DiscreteField,
    FieldLike,
    ImplicitField,
    NonconformingField,
    UniformField,
    make_discrete_field,
    make_restriction,
    make_test,
    make_trial,
)
from .geometry import (
    AdaptiveNanogrid,
    ExplicitGeometryPartition,
    Geometry,
    GeometryPartition,
    Grid2D,
    Grid3D,
    Hexmesh,
    LinearGeometryPartition,
    Nanogrid,
    Quadmesh2D,
    Quadmesh3D,
    Tetmesh,
    Trimesh2D,
    Trimesh3D,
)
from .integrate import integrate, interpolate
from .operator import (
    D,
    at_node,
    average,
    cells,
    curl,
    deformation_gradient,
    degree,
    div,
    div_outer,
    element_closest_point,
    element_coordinates,
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
    node_count,
    node_index,
    normal,
    outer,
    partition_lookup,
    position,
    to_cell_side,
    to_inner_cell,
    to_outer_cell,
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
    make_contravariant_function_space,
    make_covariant_function_space,
    make_polynomial_basis_space,
    make_polynomial_space,
    make_space_partition,
    make_space_restriction,
)
from .types import (
    NULL_ELEMENT_INDEX,
    NULL_QP_INDEX,
    Coords,
    Domain,
    ElementIndex,
    Field,
    QuadraturePointIndex,
    Sample,
    make_free_sample,
)
