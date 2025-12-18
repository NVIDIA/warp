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

import weakref
from functools import cached_property
from typing import ClassVar, Optional

import numpy as np

import warp as wp
from warp._src.fem import cache
from warp._src.fem.geometry import Geometry
from warp._src.fem.types import (
    NULL_ELEMENT_INDEX,
    NULL_QP_INDEX,
    Coords,
    ElementIndex,
    QuadraturePointIndex,
    make_free_sample,
)
from warp._src.types import type_repr, types_equal

from .shape import ShapeFunction
from .topology import SpaceTopology

_wp_module_name_ = "warp.fem.space.basis_space"


class BasisSpace:
    """Interface class for defining shape functions over a geometry.

    A basis space is a component of a function space, and is responsible for defining the node positions
    and their weights over individual elements of the geometry.
    The connectivity pattern between elements of geometry is defined by the :class:`SpaceTopology`.
    The actual valuation of the space is defined by the :class:`FunctionSpace`, allowing to reuse a single basis
    space for multiple value types (e.g, scalar, vector, or tensor).

    See also: :func:`make_polynomial_basis_space`, :func:`make_collocated_function_space`
    """

    _dynamic_attribute_constructors: ClassVar = {
        "node_coords_in_element": lambda obj: obj.make_node_coords_in_element(),
        "node_quadrature_weight": lambda obj: obj.make_node_quadrature_weight(),
        "element_inner_weight": lambda obj: obj.make_element_inner_weight(),
        "element_inner_weight_gradient": lambda obj: obj.make_element_inner_weight_gradient(),
        "element_outer_weight": lambda obj: obj.make_element_outer_weight(),
        "element_outer_weight_gradient": lambda obj: obj.make_element_outer_weight_gradient(),
    }

    @wp.struct
    class BasisArg:
        """Argument structure to be passed to device functions"""

        pass

    def __init__(self, topology: SpaceTopology):
        self._topology = topology

        cache.setup_dynamic_attributes(self, cls=__class__)

    @property
    def topology(self) -> SpaceTopology:
        """Underlying topology of the basis space"""
        return self._topology

    @property
    def geometry(self) -> Geometry:
        """Underlying geometry of the basis space"""
        return self._topology.geometry

    @property
    def value(self) -> ShapeFunction.Value:
        """Value type for the underlying shape functions"""
        raise NotImplementedError()

    @cache.cached_arg_value
    def basis_arg_value(self, device) -> "BasisArg":
        """Value for the argument structure to be passed to device functions"""
        arg = self.BasisArg()
        self.fill_basis_arg(arg, device)
        return arg

    def fill_basis_arg(self, arg, device):
        """Fill the arguments to be passed to basis-related device functions"""
        pass

    # Helpers for generating node positions

    def node_positions(self, out: Optional[wp.array] = None) -> wp.array:
        """Returns a temporary array containing the world position for each node"""

        pos_type = cache.cached_vec_type(length=self.geometry.dimension, dtype=float)

        node_coords_in_element = self.make_node_coords_in_element()

        @cache.dynamic_kernel(suffix=self.name, kernel_options={"max_unroll": 4, "enable_backward": False})
        def fill_node_positions(
            geo_cell_arg: self.geometry.CellArg,
            topo_arg: self.topology.TopologyArg,
            basis_arg: self.BasisArg,
            node_positions: wp.array(dtype=pos_type),
        ):
            element_index = wp.tid()

            element_node_count = self.topology.element_node_count(geo_cell_arg, topo_arg, element_index)
            for n in range(element_node_count):
                node_index = self.topology.element_node_index(geo_cell_arg, topo_arg, element_index, n)
                coords = node_coords_in_element(geo_cell_arg, topo_arg, basis_arg, element_index, n)

                sample = make_free_sample(element_index, coords)
                pos = self.geometry.cell_position(geo_cell_arg, sample)

                node_positions[node_index] = pos

        shape = (self.topology.node_count(),)
        if out is None:
            node_positions = wp.empty(
                shape=shape,
                dtype=pos_type,
            )
        else:
            if out.shape != shape or not types_equal(pos_type, out.dtype):
                raise ValueError(
                    f"Out node positions array must have shape {shape} and data type {type_repr(pos_type)}"
                )
            node_positions = out

        wp.launch(
            dim=self.geometry.cell_count(),
            kernel=fill_node_positions,
            inputs=[
                self.geometry.cell_arg_value(device=node_positions.device),
                self.topology.topo_arg_value(device=node_positions.device),
                self.basis_arg_value(device=node_positions.device),
                node_positions,
            ],
        )

        return node_positions

    def make_node_coords_in_element(self):
        raise NotImplementedError()

    def make_node_quadrature_weight(self):
        raise NotImplementedError()

    def make_element_inner_weight(self):
        raise NotImplementedError()

    def make_element_outer_weight(self):
        return self.make_element_inner_weight()

    def make_element_inner_weight_gradient(self):
        raise NotImplementedError()

    def make_element_outer_weight_gradient(self):
        return self.make_element_inner_weight_gradient()

    def make_trace_node_quadrature_weight(self):
        raise NotImplementedError()

    def trace(self) -> "TraceBasisSpace":
        return TraceBasisSpace(self)

    @property
    def weight_type(self):
        if self.value is ShapeFunction.Value.Scalar:
            return float

        return cache.cached_vec_type(length=self.geometry.cell_dimension, dtype=float)

    @property
    def weight_gradient_type(self):
        if self.value is ShapeFunction.Value.Scalar:
            return wp.types.vector(length=self.geometry.cell_dimension, dtype=float)

        return cache.cached_mat_type(
            shape=(self.geometry.cell_dimension, self.geometry.cell_dimension),
            dtype=float,
        )


class ShapeBasisSpace(BasisSpace):
    """Base class for defining shape-function-based basis spaces."""

    def __init__(self, topology: SpaceTopology, shape: ShapeFunction):
        self._shape = shape
        self.ORDER = self._shape.ORDER

        super().__init__(topology)

    @property
    def shape(self) -> ShapeFunction:
        """Shape functions used for defining individual element basis"""
        return self._shape

    @property
    def value(self) -> ShapeFunction.Value:
        return self.shape.value

    @cached_property
    def name(self):
        return f"{self.topology.name}_{self._shape.name}"

    def make_node_coords_in_element(self):
        shape_node_coords_in_element = self._shape.make_node_coords_in_element()

        @cache.dynamic_func(suffix=self.name)
        def node_coords_in_element(
            elt_arg: self.geometry.CellArg,
            topo_arg: self.topology.TopologyArg,
            basis_arg: self.BasisArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            return shape_node_coords_in_element(node_index_in_elt)

        return node_coords_in_element

    def make_node_quadrature_weight(self):
        shape_node_quadrature_weight = self._shape.make_node_quadrature_weight()

        if shape_node_quadrature_weight is None:
            return None

        @cache.dynamic_func(suffix=self.name)
        def node_quadrature_weight(
            elt_arg: self.geometry.CellArg,
            topo_arg: self.topology.TopologyArg,
            basis_arg: self.BasisArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            return shape_node_quadrature_weight(node_index_in_elt)

        return node_quadrature_weight

    def make_element_inner_weight(self):
        shape_element_inner_weight = self._shape.make_element_inner_weight()

        @cache.dynamic_func(suffix=self.name)
        def element_inner_weight(
            elt_arg: self.geometry.CellArg,
            topo_arg: self.topology.TopologyArg,
            basis_arg: self.BasisArg,
            element_index: ElementIndex,
            coords: Coords,
            node_index_in_elt: int,
            qp_index: QuadraturePointIndex,
        ):
            if wp.static(self.value == ShapeFunction.Value.Scalar):
                return shape_element_inner_weight(coords, node_index_in_elt)
            else:
                sign = self.topology.element_node_sign(elt_arg, topo_arg, element_index, node_index_in_elt)
                return sign * shape_element_inner_weight(coords, node_index_in_elt)

        return element_inner_weight

    def make_element_inner_weight_gradient(self):
        shape_element_inner_weight_gradient = self._shape.make_element_inner_weight_gradient()

        @cache.dynamic_func(suffix=self.name)
        def element_inner_weight_gradient(
            elt_arg: self.geometry.CellArg,
            topo_arg: self.topology.TopologyArg,
            basis_arg: self.BasisArg,
            element_index: ElementIndex,
            coords: Coords,
            node_index_in_elt: int,
            qp_index: QuadraturePointIndex,
        ):
            if wp.static(self.value == ShapeFunction.Value.Scalar):
                return shape_element_inner_weight_gradient(coords, node_index_in_elt)
            else:
                sign = self.topology.element_node_sign(elt_arg, topo_arg, element_index, node_index_in_elt)
                return sign * shape_element_inner_weight_gradient(coords, node_index_in_elt)

        return element_inner_weight_gradient

    def make_trace_node_quadrature_weight(self, trace_basis):
        shape_trace_node_quadrature_weight = self._shape.make_trace_node_quadrature_weight()

        if shape_trace_node_quadrature_weight is None:
            return None

        @cache.dynamic_func(suffix=self.name)
        def trace_node_quadrature_weight(
            geo_side_arg: trace_basis.geometry.SideArg,
            topo_arg: self.topology.TopologyArg,
            basis_arg: trace_basis.BasisArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            _neighbor_elem, index_in_neighbor = trace_basis.topology.neighbor_cell_index(
                geo_side_arg, topo_arg, element_index, node_index_in_elt
            )
            return shape_trace_node_quadrature_weight(index_in_neighbor)

        return trace_node_quadrature_weight

    def trace(self) -> "TraceBasisSpace":
        if self.ORDER == 0:
            return PiecewiseConstantBasisSpaceTrace(self)

        return TraceBasisSpace(self)

    @property
    def node_grid(self):
        if not hasattr(self._topology, "node_grid"):
            raise AttributeError(f"{self._topology.name} does not define a node grid")
        return self._topology.node_grid

    @property
    def node_triangulation(self):
        if not hasattr(self._shape, "element_node_triangulation"):
            raise AttributeError(f"Shape function {self._shape.name} does not define a node triangulation")
        return lambda: ShapeBasisSpace._node_triangulation(weakref.proxy(self))

    @property
    def node_tets(self):
        if not hasattr(self._shape, "element_node_tets"):
            raise AttributeError(f"Shape function {self._shape.name} does not define node tets")
        return lambda: ShapeBasisSpace._node_tets(weakref.proxy(self))

    @property
    def node_hexes(self):
        if not hasattr(self._shape, "element_node_hexes"):
            raise AttributeError(f"Shape function {self._shape.name} does not define node hexes")
        return lambda: ShapeBasisSpace._node_hexes(weakref.proxy(self))

    @property
    def vtk_cells(self):
        if not hasattr(self._shape, "element_vtk_cells"):
            raise AttributeError(f"Shape function {self._shape.name} does not define VTK cells")
        return lambda: ShapeBasisSpace._vtk_cells(weakref.proxy(self))

    def _node_triangulation(self):
        element_node_indices = self._topology.element_node_indices().numpy()
        element_triangles = self._shape.element_node_triangulation()

        tri_indices = element_node_indices[:, element_triangles].reshape(-1, 3)
        return tri_indices

    def _node_tets(self):
        element_node_indices = self._topology.element_node_indices().numpy()
        element_tets = self._shape.element_node_tets()

        tet_indices = element_node_indices[:, element_tets].reshape(-1, 4)
        return tet_indices

    def _node_hexes(self):
        element_node_indices = self._topology.element_node_indices().numpy()
        element_hexes = self._shape.element_node_hexes()

        hex_indices = element_node_indices[:, element_hexes].reshape(-1, 8)
        return hex_indices

    def _vtk_cells(self):
        element_node_indices = self._topology.element_node_indices().numpy()
        element_vtk_cells, element_vtk_cell_types = self._shape.element_vtk_cells()

        idx_per_cell = element_vtk_cells.shape[1]
        cell_indices = element_node_indices[:, element_vtk_cells].reshape(-1, idx_per_cell)
        cells = np.hstack((np.full((cell_indices.shape[0], 1), idx_per_cell), cell_indices))

        return cells.flatten(), np.tile(element_vtk_cell_types, element_node_indices.shape[0])


class TraceBasisSpace(BasisSpace):
    """Auto-generated trace space evaluating the cell-defined basis on the geometry sides"""

    def __init__(self, basis: BasisSpace):
        self.ORDER = basis.ORDER
        self._basis = basis
        self.BasisArg = self._basis.BasisArg
        self.basis_arg_value = self._basis.basis_arg_value
        self.fill_basis_arg = self._basis.fill_basis_arg

        super().__init__(basis.topology.trace())

    @property
    def name(self):
        return f"{self._basis.name}_Trace"

    @property
    def value(self) -> ShapeFunction.Value:
        return self._basis.value

    def make_node_coords_in_element(self):
        node_coords_in_cell = self._basis.make_node_coords_in_element()

        @cache.dynamic_func(suffix=self._basis.name)
        def trace_node_coords_in_element(
            geo_side_arg: self.geometry.SideArg,
            topo_arg: self.topology.TopologyArg,
            basis_arg: self.BasisArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            neighbour_elem, index_in_neighbour = self.topology.neighbor_cell_index(
                geo_side_arg, topo_arg, element_index, node_index_in_elt
            )
            geo_cell_arg = self.geometry.side_to_cell_arg(geo_side_arg)
            neighbour_coords = node_coords_in_cell(
                geo_cell_arg,
                topo_arg,
                basis_arg,
                neighbour_elem,
                index_in_neighbour,
            )

            return self.geometry.side_from_cell_coords(geo_side_arg, element_index, neighbour_elem, neighbour_coords)

        return trace_node_coords_in_element

    def make_node_quadrature_weight(self):
        return self._basis.make_trace_node_quadrature_weight(self)

    def make_element_inner_weight(self):
        cell_inner_weight = self._basis.make_element_inner_weight()

        @cache.dynamic_func(suffix=self._basis.name)
        def trace_element_inner_weight(
            geo_side_arg: self.geometry.SideArg,
            topo_arg: self.topology.TopologyArg,
            basis_arg: self.BasisArg,
            element_index: ElementIndex,
            coords: Coords,
            node_index_in_elt: int,
            qp_index: QuadraturePointIndex,
        ):
            cell_index, index_in_cell = self.topology.inner_cell_index(
                geo_side_arg, topo_arg, element_index, node_index_in_elt
            )
            if cell_index == NULL_ELEMENT_INDEX:
                return self.weight_type(0.0)

            cell_coords = self.geometry.side_inner_cell_coords(geo_side_arg, element_index, coords)

            geo_cell_arg = self.geometry.side_to_cell_arg(geo_side_arg)
            return cell_inner_weight(
                geo_cell_arg,
                topo_arg,
                basis_arg,
                cell_index,
                cell_coords,
                index_in_cell,
                NULL_QP_INDEX,
            )

        return trace_element_inner_weight

    def make_element_outer_weight(self):
        cell_outer_weight = self._basis.make_element_outer_weight()

        @cache.dynamic_func(suffix=self._basis.name)
        def trace_element_outer_weight(
            geo_side_arg: self.geometry.SideArg,
            topo_arg: self.topology.TopologyArg,
            basis_arg: self.BasisArg,
            element_index: ElementIndex,
            coords: Coords,
            node_index_in_elt: int,
            qp_index: QuadraturePointIndex,
        ):
            cell_index, index_in_cell = self.topology.outer_cell_index(
                geo_side_arg, topo_arg, element_index, node_index_in_elt
            )
            if cell_index == NULL_ELEMENT_INDEX:
                return self.weight_type(0.0)

            cell_coords = self.geometry.side_outer_cell_coords(geo_side_arg, element_index, coords)

            geo_cell_arg = self.geometry.side_to_cell_arg(geo_side_arg)
            return cell_outer_weight(
                geo_cell_arg,
                topo_arg,
                basis_arg,
                cell_index,
                cell_coords,
                index_in_cell,
                NULL_QP_INDEX,
            )

        return trace_element_outer_weight

    def make_element_inner_weight_gradient(self):
        cell_inner_weight_gradient = self._basis.make_element_inner_weight_gradient()

        @cache.dynamic_func(suffix=self._basis.name)
        def trace_element_inner_weight_gradient(
            geo_side_arg: self.geometry.SideArg,
            topo_arg: self.topology.TopologyArg,
            basis_arg: self.BasisArg,
            element_index: ElementIndex,
            coords: Coords,
            node_index_in_elt: int,
            qp_index: QuadraturePointIndex,
        ):
            cell_index, index_in_cell = self.topology.inner_cell_index(
                geo_side_arg, topo_arg, element_index, node_index_in_elt
            )
            if cell_index == NULL_ELEMENT_INDEX:
                return self.weight_gradient_type(0.0)

            cell_coords = self.geometry.side_inner_cell_coords(geo_side_arg, element_index, coords)
            geo_cell_arg = self.geometry.side_to_cell_arg(geo_side_arg)
            return cell_inner_weight_gradient(
                geo_cell_arg,
                topo_arg,
                basis_arg,
                cell_index,
                cell_coords,
                index_in_cell,
                NULL_QP_INDEX,
            )

        return trace_element_inner_weight_gradient

    def make_element_outer_weight_gradient(self):
        cell_outer_weight_gradient = self._basis.make_element_outer_weight_gradient()

        @cache.dynamic_func(suffix=self._basis.name)
        def trace_element_outer_weight_gradient(
            geo_side_arg: self.geometry.SideArg,
            topo_arg: self.topology.TopologyArg,
            basis_arg: self.BasisArg,
            element_index: ElementIndex,
            coords: Coords,
            node_index_in_elt: int,
            qp_index: QuadraturePointIndex,
        ):
            cell_index, index_in_cell = self.topology.outer_cell_index(
                geo_side_arg, topo_arg, element_index, node_index_in_elt
            )
            if cell_index == NULL_ELEMENT_INDEX:
                return self.weight_gradient_type(0.0)

            cell_coords = self.geometry.side_outer_cell_coords(geo_side_arg, element_index, coords)
            geo_cell_arg = self.geometry.side_to_cell_arg(geo_side_arg)
            return cell_outer_weight_gradient(
                geo_cell_arg,
                topo_arg,
                basis_arg,
                cell_index,
                cell_coords,
                index_in_cell,
                NULL_QP_INDEX,
            )

        return trace_element_outer_weight_gradient

    def __eq__(self, other: "TraceBasisSpace") -> bool:
        return self._basis == other._basis


class PiecewiseConstantBasisSpaceTrace(TraceBasisSpace):
    def make_node_coords_in_element(self):
        # Makes the single node visible to all sides; useful for interpolating on boundaries
        # For higher-order non-conforming elements direct interpolation on boundary is not possible,
        # need to do proper integration then solve with mass matrix

        CENTER_COORDS = Coords(self.geometry.reference_side().prototype.center())

        @cache.dynamic_func(suffix=self._basis.name)
        def trace_node_coords_in_element(
            geo_side_arg: self.geometry.SideArg,
            topo_arg: self.topology.TopologyArg,
            basis_arg: self.BasisArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            return CENTER_COORDS

        return trace_node_coords_in_element
