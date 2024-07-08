import numpy as np

import warp as wp
from warp.fem import cache
from warp.fem.geometry import Grid2D
from warp.fem.polynomial import is_closed
from warp.fem.types import ElementIndex

from .shape import (
    ShapeFunction,
    SquareBipolynomialShapeFunctions,
    SquareSerendipityShapeFunctions,
)
from .topology import SpaceTopology, forward_base_topology


class Grid2DSpaceTopology(SpaceTopology):
    def __init__(self, grid: Grid2D, shape: ShapeFunction):
        if not is_closed(shape.family):
            raise ValueError("A closed polynomial family is required to define a continuous function space")

        super().__init__(grid, shape.NODES_PER_ELEMENT)
        self._shape = shape

    @wp.func
    def _vertex_coords(vidx_in_cell: int):
        x = vidx_in_cell // 2
        y = vidx_in_cell - 2 * x
        return wp.vec2i(x, y)

    @wp.func
    def _vertex_index(cell_arg: Grid2D.CellArg, cell_index: ElementIndex, vidx_in_cell: int):
        res = cell_arg.res
        x_stride = res[1] + 1

        corner = Grid2D.get_cell(res, cell_index) + Grid2DSpaceTopology._vertex_coords(vidx_in_cell)
        return Grid2D._from_2d_index(x_stride, corner)


class GridBipolynomialSpaceTopology(Grid2DSpaceTopology):
    def __init__(self, grid: Grid2D, shape: SquareBipolynomialShapeFunctions):
        super().__init__(grid, shape)

        self.element_node_index = self._make_element_node_index()

    def node_count(self) -> int:
        return (self.geometry.res[0] * self._shape.ORDER + 1) * (self.geometry.res[1] * self._shape.ORDER + 1)

    def _make_element_node_index(self):
        ORDER = self._shape.ORDER

        @cache.dynamic_func(suffix=self.name)
        def element_node_index(
            cell_arg: Grid2D.CellArg,
            topo_arg: Grid2DSpaceTopology.TopologyArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            res = cell_arg.res
            cell = Grid2D.get_cell(res, element_index)

            node_i = node_index_in_elt // (ORDER + 1)
            node_j = node_index_in_elt - (ORDER + 1) * node_i

            node_x = ORDER * cell[0] + node_i
            node_y = ORDER * cell[1] + node_j

            node_pitch = (res[1] * ORDER) + 1
            node_index = node_pitch * node_x + node_y

            return node_index

        return element_node_index

    def node_grid(self):
        res = self.geometry.res

        cell_coords = np.array(self._shape.LOBATTO_COORDS)[:-1]

        grid_coords_x = np.repeat(np.arange(0, res[0], dtype=float), len(cell_coords)) + np.tile(
            cell_coords, reps=res[0]
        )
        grid_coords_x = np.append(grid_coords_x, res[0])
        X = grid_coords_x * self.geometry.cell_size[0] + self.geometry.origin[0]

        grid_coords_y = np.repeat(np.arange(0, res[1], dtype=float), len(cell_coords)) + np.tile(
            cell_coords, reps=res[1]
        )
        grid_coords_y = np.append(grid_coords_y, res[1])
        Y = grid_coords_y * self.geometry.cell_size[1] + self.geometry.origin[1]

        return np.meshgrid(X, Y, indexing="ij")


class GridSerendipitySpaceTopology(Grid2DSpaceTopology):
    def __init__(self, grid: Grid2D, shape: SquareSerendipityShapeFunctions):
        super().__init__(grid, shape)

        self.element_node_index = self._make_element_node_index()

    TopologyArg = Grid2D.SideArg

    def topo_arg_value(self, device):
        return self.geometry.side_arg_value(device)

    def node_count(self) -> int:
        return self.geometry.vertex_count() + (self._shape.ORDER - 1) * self.geometry.side_count()

    def _make_element_node_index(self):
        ORDER = self._shape.ORDER

        @cache.dynamic_func(suffix=self.name)
        def element_node_index(
            cell_arg: Grid2D.CellArg,
            topo_arg: Grid2D.SideArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            node_type, type_index = self._shape.node_type_and_type_index(node_index_in_elt)

            if node_type == SquareSerendipityShapeFunctions.VERTEX:
                return Grid2DSpaceTopology._vertex_index(cell_arg, element_index, type_index)

            side_offset, index_in_side = SquareSerendipityShapeFunctions.side_offset_and_index(type_index)
            axis = 1 - (node_type - SquareSerendipityShapeFunctions.EDGE_X)

            cell = Grid2D.get_cell(cell_arg.res, element_index)
            origin = wp.vec2i(cell[Grid2D.ROTATION[axis, 0]] + side_offset, cell[Grid2D.ROTATION[axis, 1]])

            side = Grid2D.Side(axis, origin)
            side_index = Grid2D.side_index(topo_arg, side)

            res = cell_arg.res
            vertex_count = (res[0] + 1) * (res[1] + 1)

            return vertex_count + (ORDER - 1) * side_index + index_in_side

        return element_node_index


def make_grid_2d_space_topology(grid: Grid2D, shape: ShapeFunction):
    if isinstance(shape, SquareSerendipityShapeFunctions):
        return forward_base_topology(GridSerendipitySpaceTopology, grid, shape)

    if isinstance(shape, SquareBipolynomialShapeFunctions):
        return forward_base_topology(GridBipolynomialSpaceTopology, grid, shape)

    raise ValueError(f"Unsupported shape function {shape.name}")
