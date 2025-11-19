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

from __future__ import annotations

from typing import Any

from warp._src.fem.geometry import Geometry
from warp._src.fem.space.basis_space import BasisSpace
from warp._src.fem.types import Coords, ElementIndex, ElementKind
from warp._src.types import type_is_matrix, type_is_vector, type_size

from .topology import SpaceTopology

_wp_module_name_ = "warp.fem.space.function_space"


class FunctionSpace:
    """
    Interface class for function spaces, i.e. geometry + interpolation basis

    The value of a function `f` at a position `x` is generally computed as
      ``f(x) = L(x)[sum_i f_i N_i(x)]``
    with:
        - ``f_i`` the value of the ith node's degrees-of-freedom (dof)
        - ``N_i(x)`` the weight associated to the node at `x`
        - ``L(x)`` local linear transformation from node-space to world-space
    """

    dtype: type
    """Value type of the interpolation functions"""

    dof_dtype: type
    """Data type of the degrees of freedom of each node"""

    weight_dtype: type
    """Data type of the shape functions associated to each node"""

    LocalValueMap: type
    """Type of the local map for transforming vector-valued functions from reference to world space"""

    VALUE_DOF_COUNT: int
    """Number of degrees of freedom per value, as a Warp constant"""

    NODE_DOF_COUNT: int
    """Number of degrees of freedom per node, as a Warp constant"""

    ORDER: int
    """Polynomial degree of the function space, used to determine integration order"""

    def __init__(self, basis: BasisSpace):
        self._basis = basis
        self._topology = basis.topology
        self.ElementArg = self.topology.ElementArg

        if self._topology.is_trace:
            self.element_inner_reference_gradient_transform = self.geometry.side_inner_inverse_deformation_gradient
            self.element_outer_reference_gradient_transform = self.geometry.side_outer_inverse_deformation_gradient
        else:
            self.element_inner_reference_gradient_transform = self.geometry.cell_inverse_deformation_gradient
            self.element_outer_reference_gradient_transform = self.geometry.cell_inverse_deformation_gradient

    def node_count(self) -> int:
        """Number of nodes in the interpolation basis"""
        return self.topology.node_count()

    @property
    def topology(self) -> SpaceTopology:
        """Underlying geometry"""
        return self._topology

    @property
    def basis(self) -> BasisSpace:
        """Underlying basis space"""
        return self._basis

    @property
    def geometry(self) -> Geometry:
        """Underlying geometry"""
        return self.topology.geometry

    @property
    def element_kind(self) -> ElementKind:
        """Kind of element the function space is expressed over"""
        return ElementKind.CELL if self.dimension == self.geometry.dimension else ElementKind.SIDE

    @property
    def dimension(self) -> int:
        """Function space embedding dimension"""
        return self.topology.dimension

    @property
    def degree(self) -> int:
        """Maximum polynomial degree of the underlying basis"""
        return self.ORDER

    @property
    def name(self):
        raise NotImplementedError

    def __str__(self):
        return self.name

    def trace(self) -> FunctionSpace:
        """Trace of the function space over lower-dimensional elements of the geometry"""
        raise NotImplementedError

    def make_field(self, space_partition=None):
        """Creates a zero-initialized discrete field over the function space holding values for all degrees of freedom of nodes in a space partition

        Args:
            space_partition: If provided, the subset of nodes to consider

        See also: :func:`make_space_partition`
        """
        raise NotImplementedError

    def gradient_valid(self) -> bool:
        """Whether gradient operator can be computed. Only for scalar and vector fields as higher-order tensors are not supported yet"""
        return not type_is_matrix(self.dtype)

    def divergence_valid(self) -> bool:
        """Whether divergence of this field can be computed. Only for vector and tensor fields with same dimension as embedding geometry"""
        if type_is_vector(self.dtype):
            return type_size(self.dtype) == self.geometry.dimension
        if type_is_matrix(self.dtype):
            return self.dtype._shape_[0] == self.geometry.dimension
        return False

    @staticmethod
    def node_basis_element(dof_coord: int):
        """Basis element for node degrees of freedom.

        Assumes 0 <= dof_coord < NODE_DOF_COUNT
        """
        raise NotImplementedError

    @staticmethod
    def value_basis_element(dof_coord: int):
        """Basis element for the function space values

        Assumes 0 <= dof_coord < VALUE_DOF_COUNT
        """
        raise NotImplementedError

    @staticmethod
    def local_value_map_inner(
        elt_arg: SpaceTopology.ElementArg,
        element_index: ElementIndex,
        coords: Coords,
    ):
        """Builds the local value map transforming from node to world space"""
        raise NotImplementedError

    @staticmethod
    def local_value_map_outer(
        elt_arg: SpaceTopology.ElementArg,
        element_index: ElementIndex,
        coords: Coords,
    ):
        """Builds the local value map transforming vector-valued from node to world space"""
        raise NotImplementedError

    @staticmethod
    def space_value(
        dof_value: FunctionSpace.dof_dtype,
        node_weight: FunctionSpace.weight_dtype,
        local_value_map: FunctionSpace.LocalValueMap,
    ):
        """
        Assembles the world-space value of the function space
        Args:
         - dof_value: node value in the degrees-of-freedom basis
         - node_weight: weight associated to the node, as given per the basis space
         - local_value_map: data encoding local transformation from node space to world space, as given per `local_map_value_(inn|out)er`
        """
        raise NotImplementedError

    @staticmethod
    def space_gradient(
        dof_value: FunctionSpace.dof_dtype,
        node_weight_gradient: Any,
        local_value_map: FunctionSpace.LocalValueMap,
    ):
        """
        Assembles the world-space gradient of the function space
        Args:
         - dof_value: node value in the degrees-of-freedom basis
         - node_weight_gradient: gradient of the weight associated to the node, either w.r.t element or world space
         - local_value_map: data encoding local transformation from node space to world space, as given per `local_map_value_(inn|out)er`
        """
        raise NotImplementedError

    @staticmethod
    def space_divergence(
        dof_value: FunctionSpace.dof_dtype,
        node_weight_gradient: Any,
        local_value_map: FunctionSpace.LocalValueMap,
    ):
        """ "
        Assembles the world-space divergence of the function space
        Args:
         - dof_value: node value in the degrees-of-freedom basis
         - node_weight_gradient: gradient of the weight associated to the node, either w.r.t element or world space
         - local_value_map: data encoding local transformation from node space to world space, as given per `local_map_value_(inn|out)er`
        """
        raise NotImplementedError

    @staticmethod
    def dof_value(
        space_value: FunctionSpace.dtype,
        node_weight: FunctionSpace.weight_dtype,
        local_value_map: FunctionSpace.LocalValueMap,
    ):
        """
        Computes the projection of a world-space value onto the basis of a single degree of freedom
        Args:
            - space_value: world-space value
            - node_weight: weight associated to the node, as given per the basis space
            - local_value_map: data encoding local transformation from node space to world space, as given per `local_map_value_(inn|out)er`
        """
        raise NotImplementedError
