from typing import Optional

import warp as wp

from warp.fem.types import ElementIndex, Coords, make_free_sample
from warp.fem.geometry import Geometry
from warp.fem import cache

from .topology import SpaceTopology
from .shape import ShapeFunction


class BasisSpace:
    """Interface class for defining a scalar-valued basis over a geometry.

    A basis space makes it easy to define multiple function spaces sharing the same basis (and thus nodes) but with different valuation functions;
    however, it is not a required ingredient of a function space.

    See also: :func:`make_polynomial_basis_space`, :func:`make_collocated_function_space`
    """

    @wp.struct
    class BasisArg:
        """Argument structure to be passed to device functions"""

        pass

    def __init__(self, topology: SpaceTopology, shape: ShapeFunction):
        self._topology = topology
        self._shape = shape

        self.NODES_PER_ELEMENT = self._topology.NODES_PER_ELEMENT
        self.ORDER = self._shape.ORDER

        if hasattr(shape, "element_node_triangulation"):
            self.node_triangulation = self._node_triangulation
        if hasattr(shape, "element_node_tets"):
            self.node_tets = self._node_tets
        if hasattr(shape, "element_node_hexes"):
            self.node_hexes = self._node_hexes

    @property
    def topology(self) -> SpaceTopology:
        """Underlying topology of the basis space"""
        return self._topology

    @property
    def geometry(self) -> Geometry:
        """Underlying geometry of the basis space"""
        return self._topology.geometry

    @property
    def shape(self) -> ShapeFunction:
        """Shape functions used for defining individual element basis"""
        return self._shape

    def basis_arg_value(self, device) -> "BasisArg":
        """Value for the argument structure to be passed to device functions"""
        return BasisSpace.BasisArg()

    @property
    def name(self):
        return f"{self.__class__.__qualname__}_{self._shape.name}"

    # Helpers for generating node positions

    def node_positions(self, out: Optional[wp.array] = None) -> wp.array:
        """Returns a temporary array containing the world position for each node"""

        NODES_PER_ELEMENT = self.NODES_PER_ELEMENT

        pos_type = cache.cached_vec_type(length=self.geometry.dimension, dtype=float)

        node_coords_in_element = self.make_node_coords_in_element()

        @cache.dynamic_kernel(suffix=self.name)
        def fill_node_positions(
            geo_cell_arg: self.geometry.CellArg,
            basis_arg: self.BasisArg,
            topo_arg: self.topology.TopologyArg,
            node_positions: wp.array(dtype=pos_type),
        ):
            element_index = wp.tid()

            for n in range(NODES_PER_ELEMENT):
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
                    f"Out node positions array must have shape {shape} and data type {wp.types.type_repr(post_type)}"
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
        ):
            return shape_element_inner_weight(coords, node_index_in_elt)

        return element_inner_weight

    def make_element_outer_weight(self):
        return self.make_element_inner_weight()

    def make_element_inner_weight_gradient(self):
        shape_element_inner_weight_gradient = self._shape.make_element_inner_weight_gradient()

        @cache.dynamic_func(suffix=self.name)
        def element_inner_weight_gradient(
            elt_arg: self.geometry.CellArg,
            basis_arg: self.BasisArg,
            element_index: ElementIndex,
            coords: Coords,
            node_index_in_elt: int,
        ):
            grad = shape_element_inner_weight_gradient(coords, node_index_in_elt)
            return self.geometry.cell_transform_reference_gradient(elt_arg, element_index, coords, grad)

        return element_inner_weight_gradient

    def make_element_outer_weight_gradient(self):
        return self.make_element_inner_weight_gradient()

    def trace(self) -> "TraceBasisSpace":
        return TraceBasisSpace(self)

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


class TraceBasisSpace(BasisSpace):
    """Auto-generated trace space evaluating the cell-defined basis on the geometry sides"""

    def __init__(self, basis: BasisSpace):
        super().__init__(basis.topology.trace(), basis.shape)

        self._basis = basis
        self.BasisArg = self._basis.BasisArg
        self.basis_arg_value = self._basis.basis_arg_value

    @property
    def name(self):
        return f"{self._basis.name}_Trace"

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
        shape_trace_node_quadrature_weight = self._shape.make_trace_node_quadrature_weight()

        @cache.dynamic_func(suffix=self._basis.name)
        def trace_node_quadrature_weight(
            geo_side_arg: self.geometry.SideArg,
            basis_arg: self.BasisArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            neighbour_elem, index_in_neighbour = self.topology.neighbor_cell_index(
                geo_side_arg, element_index, node_index_in_elt
            )
            return shape_trace_node_quadrature_weight(index_in_neighbour)

        return trace_node_quadrature_weight

    def make_element_inner_weight(self):
        cell_inner_weight = self._basis.make_element_inner_weight()

        @cache.dynamic_func(suffix=self._basis.name)
        def trace_element_inner_weight(
            geo_side_arg: self.geometry.SideArg,
            basis_arg: self.BasisArg,
            element_index: ElementIndex,
            coords: Coords,
            node_index_in_elt: int,
        ):
            cell_index, index_in_cell = self.topology.inner_cell_index(geo_side_arg, element_index, node_index_in_elt)
            if index_in_cell < 0:
                return 0.0

            cell_coords = self.geometry.side_inner_cell_coords(geo_side_arg, element_index, coords)

            geo_cell_arg = self.geometry.side_to_cell_arg(geo_side_arg)
            return cell_inner_weight(
                geo_cell_arg,
                basis_arg,
                cell_index,
                cell_coords,
                index_in_cell,
            )

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
        ):
            cell_index, index_in_cell = self.topology.outer_cell_index(geo_side_arg, element_index, node_index_in_elt)
            if index_in_cell < 0:
                return 0.0

            cell_coords = self.geometry.side_outer_cell_coords(geo_side_arg, element_index, coords)

            geo_cell_arg = self.geometry.side_to_cell_arg(geo_side_arg)
            return cell_outer_weight(
                geo_cell_arg,
                basis_arg,
                cell_index,
                cell_coords,
                index_in_cell,
            )

        return trace_element_outer_weight

    def make_element_inner_weight_gradient(self):
        cell_inner_weight_gradient = self._basis.make_element_inner_weight_gradient()
        grad_vec_type = wp.vec(length=self.geometry.dimension, dtype=float)

        @cache.dynamic_func(suffix=self._basis.name)
        def trace_element_inner_weight_gradient(
            geo_side_arg: self.geometry.SideArg,
            basis_arg: self.BasisArg,
            element_index: ElementIndex,
            coords: Coords,
            node_index_in_elt: int,
        ):
            cell_index, index_in_cell = self.topology.inner_cell_index(geo_side_arg, element_index, node_index_in_elt)
            if index_in_cell < 0:
                return grad_vec_type(0.0)

            cell_coords = self.geometry.side_inner_cell_coords(geo_side_arg, element_index, coords)
            geo_cell_arg = self.geometry.side_to_cell_arg(geo_side_arg)
            return cell_inner_weight_gradient(geo_cell_arg, basis_arg, cell_index, cell_coords, index_in_cell)

        return trace_element_inner_weight_gradient

    def make_element_outer_weight_gradient(self):
        cell_outer_weight_gradient = self._basis.make_element_outer_weight_gradient()
        grad_vec_type = wp.vec(length=self.geometry.dimension, dtype=float)

        @cache.dynamic_func(suffix=self._basis.name)
        def trace_element_outer_weight_gradient(
            geo_side_arg: self.geometry.SideArg,
            basis_arg: self.BasisArg,
            element_index: ElementIndex,
            coords: Coords,
            node_index_in_elt: int,
        ):
            cell_index, index_in_cell = self.topology.outer_cell_index(geo_side_arg, element_index, node_index_in_elt)
            if index_in_cell < 0:
                return grad_vec_type(0.0)

            cell_coords = self.geometry.side_outer_cell_coords(geo_side_arg, element_index, coords)

            geo_cell_arg = self.geometry.side_to_cell_arg(geo_side_arg)
            return cell_outer_weight_gradient(geo_cell_arg, basis_arg, cell_index, cell_coords, index_in_cell)

        return trace_element_outer_weight_gradient

    def __eq__(self, other: "TraceBasisSpace") -> bool:
        return self._topo == other._topo
