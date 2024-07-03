import warp as wp
from warp.fem import cache
from warp.fem.geometry import Nanogrid
from warp.fem.geometry.nanogrid import _add_axis_flag
from warp.fem.polynomial import is_closed
from warp.fem.types import ElementIndex

from .shape import (
    CubeSerendipityShapeFunctions,
    CubeTripolynomialShapeFunctions,
    ShapeFunction,
)
from .topology import SpaceTopology, forward_base_topology


@wp.struct
class NanogridTopologyArg:
    vertex_grid: wp.uint64
    face_grid: wp.uint64
    edge_grid: wp.uint64

    vertex_count: int
    edge_count: int
    face_count: int


class NanogridSpaceTopology(SpaceTopology):
    TopologyArg = NanogridTopologyArg

    def __init__(
        self,
        grid: Nanogrid,
        shape: ShapeFunction,
        need_edge_indices: bool = True,
        need_face_indices: bool = True,
    ):
        if not is_closed(shape.family):
            raise ValueError("A closed polynomial family is required to define a continuous function space")

        super().__init__(grid, shape.NODES_PER_ELEMENT)
        self._grid = grid
        self._shape = shape

        self._vertex_grid = grid.vertex_grid.id

        self._edge_grid = grid.edge_grid.id if need_edge_indices else -1
        self._face_grid = grid.face_grid.id if need_face_indices else -1
        self._edge_count = grid.edge_count() if need_edge_indices else 0
        self._face_count = grid.side_count() if need_face_indices else 0

    @cache.cached_arg_value
    def topo_arg_value(self, device):
        arg = NanogridTopologyArg()

        arg.vertex_grid = self._vertex_grid
        arg.face_grid = self._face_grid
        arg.edge_grid = self._edge_grid

        arg.vertex_count = self._grid.vertex_count()
        arg.face_count = self._face_count
        arg.edge_count = self._edge_count
        return arg


@wp.func
def _cell_vertex_coord(cell_ijk: wp.vec3i, n: int):
    return cell_ijk + wp.vec3i((n & 4) >> 2, (n & 2) >> 1, n & 1)


@wp.func
def _cell_edge_coord(cell_ijk: wp.vec3i, axis: int, offset: int):
    e_ijk = cell_ijk
    e_ijk[(axis + 1) % 3] += offset >> 1
    e_ijk[(axis + 2) % 3] += offset & 1
    return _add_axis_flag(e_ijk, axis)


@wp.func
def _cell_face_coord(cell_ijk: wp.vec3i, axis: int, offset: int):
    f_ijk = cell_ijk
    f_ijk[axis] += offset
    return _add_axis_flag(f_ijk, axis)


class NanogridTripolynomialSpaceTopology(NanogridSpaceTopology):
    def __init__(self, grid: Nanogrid, shape: CubeTripolynomialShapeFunctions):
        super().__init__(grid, shape, need_edge_indices=shape.ORDER >= 2, need_face_indices=shape.ORDER >= 2)

        self.element_node_index = self._make_element_node_index()

    def node_count(self) -> int:
        ORDER = self._shape.ORDER
        INTERIOR_NODES_PER_EDGE = max(0, ORDER - 1)
        INTERIOR_NODES_PER_FACE = INTERIOR_NODES_PER_EDGE**2
        INTERIOR_NODES_PER_CELL = INTERIOR_NODES_PER_EDGE**3

        return (
            self._grid.vertex_count()
            + self._edge_count * INTERIOR_NODES_PER_EDGE
            + self._face_count * INTERIOR_NODES_PER_FACE
            + self._grid.cell_count() * INTERIOR_NODES_PER_CELL
        )

    def _make_element_node_index(self):
        ORDER = self._shape.ORDER
        INTERIOR_NODES_PER_EDGE = wp.constant(max(0, ORDER - 1))
        INTERIOR_NODES_PER_FACE = wp.constant(INTERIOR_NODES_PER_EDGE**2)
        INTERIOR_NODES_PER_CELL = wp.constant(INTERIOR_NODES_PER_EDGE**3)

        @cache.dynamic_func(suffix=self.name)
        def element_node_index(
            geo_arg: Nanogrid.CellArg,
            topo_arg: NanogridTopologyArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            node_type, type_instance, type_index = self._shape.node_type_and_type_index(node_index_in_elt)

            ijk = geo_arg.cell_ijk[element_index]

            if node_type == CubeTripolynomialShapeFunctions.VERTEX:
                n_ijk = _cell_vertex_coord(ijk, type_instance)
                return wp.volume_lookup_index(topo_arg.vertex_grid, n_ijk[0], n_ijk[1], n_ijk[2])

            offset = topo_arg.vertex_count

            if node_type == CubeTripolynomialShapeFunctions.EDGE:
                axis = type_instance >> 2
                node_offset = type_instance & 3

                n_ijk = _cell_edge_coord(ijk, axis, node_offset)

                edge_index = wp.volume_lookup_index(topo_arg.edge_grid, n_ijk[0], n_ijk[1], n_ijk[2])
                return offset + INTERIOR_NODES_PER_EDGE * edge_index + type_index

            offset += INTERIOR_NODES_PER_EDGE * topo_arg.edge_count

            if node_type == CubeTripolynomialShapeFunctions.FACE:
                axis = type_instance >> 1
                node_offset = type_instance & 1

                n_ijk = _cell_face_coord(ijk, axis, node_offset)

                face_index = wp.volume_lookup_index(topo_arg.face_grid, n_ijk[0], n_ijk[1], n_ijk[2])
                return offset + INTERIOR_NODES_PER_FACE * face_index + type_index

            offset += INTERIOR_NODES_PER_FACE * topo_arg.face_count

            return offset + INTERIOR_NODES_PER_CELL * element_index + type_index

        return element_node_index


class NanogridSerendipitySpaceTopology(NanogridSpaceTopology):
    def __init__(self, grid: Nanogrid, shape: CubeSerendipityShapeFunctions):
        super().__init__(grid, shape, need_edge_indices=True, need_face_indices=False)

        self.element_node_index = self._make_element_node_index()

    def node_count(self) -> int:
        return self.geometry.vertex_count() + (self._shape.ORDER - 1) * self._edge_count

    def _make_element_node_index(self):
        ORDER = self._shape.ORDER

        @cache.dynamic_func(suffix=self.name)
        def element_node_index(
            cell_arg: Nanogrid.CellArg,
            topo_arg: NanogridSpaceTopology.TopologyArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            node_type, type_index = self._shape.node_type_and_type_index(node_index_in_elt)

            ijk = cell_arg.cell_ijk[element_index]

            if node_type == CubeSerendipityShapeFunctions.VERTEX:
                n_ijk = _cell_vertex_coord(ijk, type_index)
                return wp.volume_lookup_index(topo_arg.vertex_grid, n_ijk[0], n_ijk[1], n_ijk[2])

            type_instance, index_in_edge = CubeSerendipityShapeFunctions._cube_edge_index(node_type, type_index)
            axis = type_instance >> 2
            node_offset = type_instance & 3

            n_ijk = _cell_edge_coord(ijk, axis, node_offset)

            edge_index = wp.volume_lookup_index(topo_arg.edge_grid, n_ijk[0], n_ijk[1], n_ijk[2])
            return topo_arg.vertex_count + (ORDER - 1) * edge_index + index_in_edge

        return element_node_index


def make_nanogrid_space_topology(grid: Nanogrid, shape: ShapeFunction):
    if isinstance(shape, CubeSerendipityShapeFunctions):
        return forward_base_topology(NanogridSerendipitySpaceTopology, grid, shape)

    if isinstance(shape, CubeTripolynomialShapeFunctions):
        return forward_base_topology(NanogridTripolynomialSpaceTopology, grid, shape)

    raise ValueError(f"Unsupported shape function {shape.name}")
