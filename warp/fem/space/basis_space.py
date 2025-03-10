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

from typing import Optional

import numpy as np

import warp as wp
from warp.fem import cache
from warp.fem.geometry import Geometry
from warp.fem.quadrature import Quadrature
from warp.fem.types import (
    NULL_ELEMENT_INDEX,
    NULL_QP_INDEX,
    Coords,
    ElementIndex,
    QuadraturePointIndex,
    make_free_sample,
)

from .shape import ShapeFunction
from .topology import RegularDiscontinuousSpaceTopology, SpaceTopology


class BasisSpace:
    """Interface class for defining a shape function space over a geometry.

    A basis space makes it easy to define multiple function spaces sharing the same basis (and thus nodes) but with different valuation functions;
    however, it is not a required component of a function space.

    See also: :func:`make_polynomial_basis_space`, :func:`make_collocated_function_space`
    """

    @wp.struct
    class BasisArg:
        """Argument structure to be passed to device functions"""

        pass

    def __init__(self, topology: SpaceTopology):
        self._topology = topology

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

    def basis_arg_value(self, device) -> "BasisArg":
        """Value for the argument structure to be passed to device functions"""
        return BasisSpace.BasisArg()

    # Helpers for generating node positions

    def node_positions(self, out: Optional[wp.array] = None) -> wp.array:
        """Returns a temporary array containing the world position for each node"""

        pos_type = cache.cached_vec_type(length=self.geometry.dimension, dtype=float)

        node_coords_in_element = self.make_node_coords_in_element()

        @cache.dynamic_kernel(suffix=self.name, kernel_options={"max_unroll": 4, "enable_backward": False})
        def fill_node_positions(
            geo_cell_arg: self.geometry.CellArg,
            basis_arg: self.BasisArg,
            topo_arg: self.topology.TopologyArg,
            node_positions: wp.array(dtype=pos_type),
        ):
            element_index = wp.tid()

            element_node_count = self.topology.element_node_count(geo_cell_arg, topo_arg, element_index)
            for n in range(element_node_count):
                node_index = self.topology.element_node_index(geo_cell_arg, topo_arg, element_index, n)
                coords = node_coords_in_element(geo_cell_arg, basis_arg, element_index, n)

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
            if out.shape != shape or not wp.types.types_equal(pos_type, out.dtype):
                raise ValueError(
                    f"Out node positions array must have shape {shape} and data type {wp.types.type_repr(pos_type)}"
                )
            node_positions = out

        wp.launch(
            dim=self.geometry.cell_count(),
            kernel=fill_node_positions,
            inputs=[
                self.geometry.cell_arg_value(device=node_positions.device),
                self.basis_arg_value(device=node_positions.device),
                self.topology.topo_arg_value(device=node_positions.device),
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
            return wp.vec(length=self.geometry.cell_dimension, dtype=float)

        return cache.cached_mat_type(shape=(self.geometry.cell_dimension, self.geometry.cell_dimension), dtype=float)


class ShapeBasisSpace(BasisSpace):
    """Base class for defining shape-function-based basis spaces."""

    def __init__(self, topology: SpaceTopology, shape: ShapeFunction):
        super().__init__(topology)
        self._shape = shape

        if self.value is not ShapeFunction.Value.Scalar:
            self.BasisArg = self.topology.TopologyArg
            self.basis_arg_value = self.topology.topo_arg_value

        self.ORDER = self._shape.ORDER

        if hasattr(shape, "element_node_triangulation"):
            self.node_triangulation = self._node_triangulation
        if hasattr(shape, "element_node_tets"):
            self.node_tets = self._node_tets
        if hasattr(shape, "element_node_hexes"):
            self.node_hexes = self._node_hexes
        if hasattr(shape, "element_vtk_cells"):
            self.vtk_cells = self._vtk_cells
        if hasattr(topology, "node_grid"):
            self.node_grid = topology.node_grid

    @property
    def shape(self) -> ShapeFunction:
        """Shape functions used for defining individual element basis"""
        return self._shape

    @property
    def value(self) -> ShapeFunction.Value:
        return self.shape.value

    @property
    def name(self):
        return f"{self.topology.name}_{self._shape.name}"

    def make_node_coords_in_element(self):
        shape_node_coords_in_element = self._shape.make_node_coords_in_element()

        @cache.dynamic_func(suffix=self.name)
        def node_coords_in_element(
            elt_arg: self.geometry.CellArg,
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
            basis_arg: self.BasisArg,
            element_index: ElementIndex,
            coords: Coords,
            node_index_in_elt: int,
            qp_index: QuadraturePointIndex,
        ):
            if wp.static(self.value == ShapeFunction.Value.Scalar):
                return shape_element_inner_weight(coords, node_index_in_elt)
            else:
                sign = self.topology.element_node_sign(elt_arg, basis_arg, element_index, node_index_in_elt)
                return sign * shape_element_inner_weight(coords, node_index_in_elt)

        return element_inner_weight

    def make_element_inner_weight_gradient(self):
        shape_element_inner_weight_gradient = self._shape.make_element_inner_weight_gradient()

        @cache.dynamic_func(suffix=self.name)
        def element_inner_weight_gradient(
            elt_arg: self.geometry.CellArg,
            basis_arg: self.BasisArg,
            element_index: ElementIndex,
            coords: Coords,
            node_index_in_elt: int,
            qp_index: QuadraturePointIndex,
        ):
            if wp.static(self.value == ShapeFunction.Value.Scalar):
                return shape_element_inner_weight_gradient(coords, node_index_in_elt)
            else:
                sign = self.topology.element_node_sign(elt_arg, basis_arg, element_index, node_index_in_elt)
                return sign * shape_element_inner_weight_gradient(coords, node_index_in_elt)

        return element_inner_weight_gradient

    def make_trace_node_quadrature_weight(self, trace_basis):
        shape_trace_node_quadrature_weight = self._shape.make_trace_node_quadrature_weight()

        if shape_trace_node_quadrature_weight is None:
            return None

        @cache.dynamic_func(suffix=self.name)
        def trace_node_quadrature_weight(
            geo_side_arg: trace_basis.geometry.SideArg,
            basis_arg: trace_basis.BasisArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            neighbour_elem, index_in_neighbour = trace_basis.topology.neighbor_cell_index(
                geo_side_arg, element_index, node_index_in_elt
            )
            return shape_trace_node_quadrature_weight(index_in_neighbour)

        return trace_node_quadrature_weight

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
        super().__init__(basis.topology.trace())

        self.ORDER = basis.ORDER

        self._basis = basis
        self.BasisArg = self._basis.BasisArg
        self.basis_arg_value = self._basis.basis_arg_value

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
            basis_arg: self.BasisArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            neighbour_elem, index_in_neighbour = self.topology.neighbor_cell_index(
                geo_side_arg, element_index, node_index_in_elt
            )
            geo_cell_arg = self.geometry.side_to_cell_arg(geo_side_arg)
            neighbour_coords = node_coords_in_cell(
                geo_cell_arg,
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
            basis_arg: self.BasisArg,
            element_index: ElementIndex,
            coords: Coords,
            node_index_in_elt: int,
            qp_index: QuadraturePointIndex,
        ):
            cell_index, index_in_cell = self.topology.inner_cell_index(geo_side_arg, element_index, node_index_in_elt)
            if cell_index == NULL_ELEMENT_INDEX:
                return self.weight_type(0.0)

            cell_coords = self.geometry.side_inner_cell_coords(geo_side_arg, element_index, coords)

            geo_cell_arg = self.geometry.side_to_cell_arg(geo_side_arg)
            return cell_inner_weight(geo_cell_arg, basis_arg, cell_index, cell_coords, index_in_cell, NULL_QP_INDEX)

        return trace_element_inner_weight

    def make_element_outer_weight(self):
        cell_outer_weight = self._basis.make_element_outer_weight()

        @cache.dynamic_func(suffix=self._basis.name)
        def trace_element_outer_weight(
            geo_side_arg: self.geometry.SideArg,
            basis_arg: self.BasisArg,
            element_index: ElementIndex,
            coords: Coords,
            node_index_in_elt: int,
            qp_index: QuadraturePointIndex,
        ):
            cell_index, index_in_cell = self.topology.outer_cell_index(geo_side_arg, element_index, node_index_in_elt)
            if cell_index == NULL_ELEMENT_INDEX:
                return self.weight_type(0.0)

            cell_coords = self.geometry.side_outer_cell_coords(geo_side_arg, element_index, coords)

            geo_cell_arg = self.geometry.side_to_cell_arg(geo_side_arg)
            return cell_outer_weight(geo_cell_arg, basis_arg, cell_index, cell_coords, index_in_cell, NULL_QP_INDEX)

        return trace_element_outer_weight

    def make_element_inner_weight_gradient(self):
        cell_inner_weight_gradient = self._basis.make_element_inner_weight_gradient()

        @cache.dynamic_func(suffix=self._basis.name)
        def trace_element_inner_weight_gradient(
            geo_side_arg: self.geometry.SideArg,
            basis_arg: self.BasisArg,
            element_index: ElementIndex,
            coords: Coords,
            node_index_in_elt: int,
            qp_index: QuadraturePointIndex,
        ):
            cell_index, index_in_cell = self.topology.inner_cell_index(geo_side_arg, element_index, node_index_in_elt)
            if cell_index == NULL_ELEMENT_INDEX:
                return self.weight_gradient_type(0.0)

            cell_coords = self.geometry.side_inner_cell_coords(geo_side_arg, element_index, coords)
            geo_cell_arg = self.geometry.side_to_cell_arg(geo_side_arg)
            return cell_inner_weight_gradient(
                geo_cell_arg, basis_arg, cell_index, cell_coords, index_in_cell, NULL_QP_INDEX
            )

        return trace_element_inner_weight_gradient

    def make_element_outer_weight_gradient(self):
        cell_outer_weight_gradient = self._basis.make_element_outer_weight_gradient()

        @cache.dynamic_func(suffix=self._basis.name)
        def trace_element_outer_weight_gradient(
            geo_side_arg: self.geometry.SideArg,
            basis_arg: self.BasisArg,
            element_index: ElementIndex,
            coords: Coords,
            node_index_in_elt: int,
            qp_index: QuadraturePointIndex,
        ):
            cell_index, index_in_cell = self.topology.outer_cell_index(geo_side_arg, element_index, node_index_in_elt)
            if cell_index == NULL_ELEMENT_INDEX:
                return self.weight_gradient_type(0.0)

            cell_coords = self.geometry.side_outer_cell_coords(geo_side_arg, element_index, coords)
            geo_cell_arg = self.geometry.side_to_cell_arg(geo_side_arg)
            return cell_outer_weight_gradient(
                geo_cell_arg, basis_arg, cell_index, cell_coords, index_in_cell, NULL_QP_INDEX
            )

        return trace_element_outer_weight_gradient

    def __eq__(self, other: "TraceBasisSpace") -> bool:
        return self._topo == other._topo


class PiecewiseConstantBasisSpace(ShapeBasisSpace):
    class Trace(TraceBasisSpace):
        def make_node_coords_in_element(self):
            # Makes the single node visible to all sides; useful for interpolating on boundaries
            # For higher-order non-conforming elements direct interpolation on boundary is not possible,
            # need to do proper integration then solve with mass matrix

            CENTER_COORDS = Coords(self.geometry.reference_side().center())

            @cache.dynamic_func(suffix=self._basis.name)
            def trace_node_coords_in_element(
                geo_side_arg: self.geometry.SideArg,
                basis_arg: self.BasisArg,
                element_index: ElementIndex,
                node_index_in_elt: int,
            ):
                return CENTER_COORDS

            return trace_node_coords_in_element

    def trace(self):
        return PiecewiseConstantBasisSpace.Trace(self)


def make_discontinuous_basis_space(geometry: Geometry, shape: ShapeFunction):
    topology = RegularDiscontinuousSpaceTopology(geometry, shape.NODES_PER_ELEMENT)

    if shape.NODES_PER_ELEMENT == 1:
        # piecewise-constant space
        return PiecewiseConstantBasisSpace(topology=topology, shape=shape)

    return ShapeBasisSpace(topology=topology, shape=shape)


class UnstructuredPointTopology(SpaceTopology):
    """Topology for unstructured points defined from quadrature formula. See :class:`PointBasisSpace`"""

    def __init__(self, quadrature: Quadrature):
        if quadrature.max_points_per_element() is None:
            raise ValueError("Quadrature must define a maximum number of points per element")

        if quadrature.domain.element_count() != quadrature.domain.geometry_element_count():
            raise ValueError("Point topology may only be defined on quadrature domains than span the whole geometry")

        self._quadrature = quadrature
        self.TopologyArg = quadrature.Arg

        super().__init__(quadrature.domain.geometry, max_nodes_per_element=quadrature.max_points_per_element())

        self.element_node_index = self._make_element_node_index()
        self.element_node_count = self._make_element_node_count()
        self.side_neighbor_node_counts = self._make_side_neighbor_node_counts()

    def node_count(self):
        return self._quadrature.total_point_count()

    @property
    def name(self):
        return f"PointTopology_{self._quadrature}"

    def topo_arg_value(self, device) -> SpaceTopology.TopologyArg:
        """Value of the topology argument structure to be passed to device functions"""
        return self._quadrature.arg_value(device)

    def _make_element_node_index(self):
        @cache.dynamic_func(suffix=self.name)
        def element_node_index(
            elt_arg: self.geometry.CellArg,
            topo_arg: self.TopologyArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            return self._quadrature.point_index(elt_arg, topo_arg, element_index, element_index, node_index_in_elt)

        return element_node_index

    def _make_element_node_count(self):
        @cache.dynamic_func(suffix=self.name)
        def element_node_count(
            elt_arg: self.geometry.CellArg,
            topo_arg: self.TopologyArg,
            element_index: ElementIndex,
        ):
            return self._quadrature.point_count(elt_arg, topo_arg, element_index, element_index)

        return element_node_count

    def _make_side_neighbor_node_counts(self):
        @cache.dynamic_func(suffix=self.name)
        def side_neighbor_node_counts(
            side_arg: self.geometry.SideArg,
            element_index: ElementIndex,
        ):
            return 0, 0

        return side_neighbor_node_counts


class PointBasisSpace(BasisSpace):
    """An unstructured :class:`BasisSpace` that is non-zero at a finite set of points only.

    The node locations and nodal quadrature weights are defined by a :class:`Quadrature` formula.
    """

    def __init__(self, quadrature: Quadrature):
        self._quadrature = quadrature

        topology = UnstructuredPointTopology(quadrature)
        super().__init__(topology)

        self.BasisArg = quadrature.Arg
        self.basis_arg_value = quadrature.arg_value
        self.ORDER = 0

        self.make_element_outer_weight = self.make_element_inner_weight
        self.make_element_outer_weight_gradient = self.make_element_outer_weight_gradient

    @property
    def name(self):
        return f"{self._quadrature.name}_Point"

    @property
    def value(self) -> ShapeFunction.Value:
        return ShapeFunction.Value.Scalar

    def make_node_coords_in_element(self):
        @cache.dynamic_func(suffix=self.name)
        def node_coords_in_element(
            elt_arg: self._quadrature.domain.ElementArg,
            basis_arg: self.BasisArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            return self._quadrature.point_coords(elt_arg, basis_arg, element_index, element_index, node_index_in_elt)

        return node_coords_in_element

    def make_node_quadrature_weight(self):
        @cache.dynamic_func(suffix=self.name)
        def node_quadrature_weight(
            elt_arg: self._quadrature.domain.ElementArg,
            basis_arg: self.BasisArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            return self._quadrature.point_weight(elt_arg, basis_arg, element_index, element_index, node_index_in_elt)

        return node_quadrature_weight

    def make_element_inner_weight(self):
        _DIRAC_INTEGRATION_RADIUS = wp.constant(1.0e-6)

        @cache.dynamic_func(suffix=self.name)
        def element_inner_weight(
            elt_arg: self._quadrature.domain.ElementArg,
            basis_arg: self.BasisArg,
            element_index: ElementIndex,
            coords: Coords,
            node_index_in_elt: int,
            qp_index: QuadraturePointIndex,
        ):
            qp_coord = self._quadrature.point_coords(
                elt_arg, basis_arg, element_index, element_index, node_index_in_elt
            )
            return wp.where(wp.length_sq(coords - qp_coord) < _DIRAC_INTEGRATION_RADIUS, 1.0, 0.0)

        return element_inner_weight

    def make_element_inner_weight_gradient(self):
        gradient_vec = cache.cached_vec_type(length=self.geometry.cell_dimension, dtype=float)

        @cache.dynamic_func(suffix=self.name)
        def element_inner_weight_gradient(
            elt_arg: self._quadrature.domain.ElementArg,
            basis_arg: self.BasisArg,
            element_index: ElementIndex,
            coords: Coords,
            node_index_in_elt: int,
            qp_index: QuadraturePointIndex,
        ):
            return gradient_vec(0.0)

        return element_inner_weight_gradient

    def make_trace_node_quadrature_weight(self, trace_basis):
        @cache.dynamic_func(suffix=self.name)
        def trace_node_quadrature_weight(
            elt_arg: trace_basis.geometry.SideArg,
            basis_arg: trace_basis.BasisArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            return 0.0

        return trace_node_quadrature_weight
