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

from typing import Any

import warp as wp
from warp.fem import cache
from warp.fem.geometry import Geometry
from warp.fem.types import Coords, ElementIndex, ElementKind, Sample, make_free_sample

from .topology import SpaceTopology


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

    SpaceArg: wp.codegen.Struct
    """Structure containing arguments to be passed to device function"""

    LocalValueMap: type
    """Type of the local map for transforming vector-valued functions from reference to world space"""

    VALUE_DOF_COUNT: int
    """Number of degrees of freedom per value, as a Warp constant"""

    NODE_DOF_COUNT: int
    """Number of degrees of freedom per node, as a Warp constant"""

    ORDER: int
    """Polynomial degree of the function space, used to determine integration order"""

    def __init__(self, topology: SpaceTopology):
        self._topology = topology
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

    def space_arg_value(self, device) -> wp.codegen.StructInstance:
        """Value of the arguments to be passed to device functions"""
        raise NotImplementedError

    @property
    def topology(self) -> SpaceTopology:
        """Underlying geometry"""
        return self._topology

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

    def trace(self) -> "FunctionSpace":
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
        return not wp.types.type_is_matrix(self.dtype)

    def divergence_valid(self) -> bool:
        """Whether divergence of this field can be computed. Only for vector and tensor fields with same dimension as embedding geometry"""
        if wp.types.type_is_vector(self.dtype):
            return wp.types.type_length(self.dtype) == self.geometry.dimension
        if wp.types.type_is_matrix(self.dtype):
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
        elt_arg: "SpaceTopology.ElementArg",
        element_index: ElementIndex,
        coords: Coords,
    ):
        """Builds the local value map transforming from node to world space"""
        raise NotImplementedError

    @staticmethod
    def local_value_map_outer(
        elt_arg: "SpaceTopology.ElementArg",
        element_index: ElementIndex,
        coords: Coords,
    ):
        """Builds the local value map transforming vector-valued from node to world space"""
        raise NotImplementedError

    @staticmethod
    def node_coords_in_element(
        elt_arg: "SpaceTopology.ElementArg",
        space_arg: "SpaceArg",  # noqa: F821
        element_index: ElementIndex,
        node_index_in_elt: int,
    ):
        """Coordinates inside element of a given node"""
        raise NotImplementedError

    @staticmethod
    def node_quadrature_weight(
        elt_arg: "SpaceTopology.ElementArg",
        space_arg: "SpaceArg",  # noqa: F821
        element_index: ElementIndex,
        node_index_in_elt: int,
    ):
        """Weight of a given node when used as a quadrature point"""
        raise NotImplementedError

    @staticmethod
    def element_inner_weight(
        elt_arg: "SpaceTopology.ElementArg",
        space_arg: "SpaceArg",  # noqa: F821
        element_index: ElementIndex,
        coords: Coords,
        node_index_in_elt: int,
    ):
        """Inner weight for a node at given coordinates"""
        raise NotImplementedError

    @staticmethod
    def element_inner_weight_gradient(
        elt_arg: "SpaceTopology.ElementArg",
        space_arg: "SpaceArg",  # noqa: F821
        element_index: ElementIndex,
        coords: Coords,
        node_index_in_elt: int,
    ):
        """Inner weight gradient w.r.t. reference space for a node at given coordinates"""
        raise NotImplementedError

    @staticmethod
    def element_outer_weight(
        elt_arg: "SpaceTopology.ElementArg",
        space_arg: "SpaceArg",  # noqa: F821
        element_index: ElementIndex,
        coords: Coords,
        node_index_in_elt: int,
    ):
        """Outer weight for a node at given coordinates"""
        raise NotImplementedError

    @staticmethod
    def element_outer_weight_gradient(
        elt_arg: "SpaceTopology.ElementArg",
        space_arg: "SpaceArg",  # noqa: F821
        element_index: ElementIndex,
        coords: Coords,
        node_index_in_elt: int,
    ):
        """Outer weight gradient w.r.t reference space for a node at given coordinates"""
        raise NotImplementedError

    def space_value(
        dof_value: "FunctionSpace.dof_dtype",
        node_weight: Any,
        local_value_map: "FunctionSpace.LocalValueMap",
    ):
        """
        Assembles the world-space value of the function space
        Args:
         - dof_value: node value in the degrees-of-freedom basis
         - node_weight: weight associated to the node, as given per `element_(inn|out)er_weight`
         - local_value_map: local transformation from node space to world space, as given per `local_map_value_(inn|out)er`
        """
        raise NotADirectoryError

    def space_gradient(
        dof_value: "FunctionSpace.dof_dtype",
        node_weight: Any,
        local_value_map: "FunctionSpace.LocalValueMap",
        grad_transform: Any,
    ):
        """
        Assembles the world-space gradient of the function space
        Args:
         - dof_value: node value in the degrees-of-freedom basis
         - node_weight_gradient: gradient of the weight associated to the node, as given per `element_(inn|out)er_weight_gradient`
         - local_value_map: local transformation from node space to world space, as given per `local_map_value_(inn|out)er`
         - grad_transform: transform mapping the reference space gradient to worls-space gradient (inverse deformation gradient)
        """
        raise NotImplementedError

    def space_divergence(
        dof_value: "FunctionSpace.dof_dtype",
        node_weight: Any,
        local_value_map: "FunctionSpace.LocalValueMap",
        grad_transform: Any,
    ):
        """ "
        Assembles the world-space divergence of the function space
        Args:
         - dof_value: node value in the degrees-of-freedom basis
         - node_weight_gradient: gradient of the weight associated to the node, as given per `element_(inn|out)er_weight_gradient`
         - local_value_map: local transformation from node space to world space, as given per `local_map_value_(inn|out)er`
         - grad_transform: transform mapping the reference space gradient to worls-space gradient (inverse deformation gradient)
        """
        raise NotImplementedError

    @staticmethod
    def node_dof_value(
        elt_arg: "FunctionSpace.ElementArg",
        space_arg: "FunctionSpace.SpaceArg",
        element_index: ElementIndex,
        node_index_in_elt: int,
        space_value: "FunctionSpace.dtype",
    ):
        """Converts space value to node degrees of freedom"""
        raise NotImplementedError

    def _make_side_inner_inverse_deformation_gradient(self):
        @cache.dynamic_func(suffix=self.name)
        def side_inner_inverse_deformation_gradient(args: self.ElementArg, s: Sample):
            cell_index = self.side_inner_cell_index(args, s.element_index)
            cell_coords = self.side_inner_cell_coords(args, s.element_index, s.element_coords)
            cell_arg = self.side_to_cell_arg(args)
            return self.geometry.cell_inverse_deformation_gradient(cell_arg, make_free_sample(cell_index, cell_coords))

        return side_inner_inverse_deformation_gradient

    def _make_side_outer_inverse_deformation_gradient(self):
        @cache.dynamic_func(suffix=self.name)
        def side_outer_inverse_deformation_gradient(args: self.ElementArg, s: Sample):
            cell_index = self.side_outer_cell_index(args, s.element_index)
            cell_coords = self.side_outer_cell_coords(args, s.element_index, s.element_coords)
            cell_arg = self.side_to_cell_arg(args)
            return self.geometry.cell_inverse_deformation_gradient(cell_arg, make_free_sample(cell_index, cell_coords))

        return side_outer_inverse_deformation_gradient
