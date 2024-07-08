import numpy as np

import warp as wp
from warp.fem import cache
from warp.fem.geometry import Grid3D
from warp.fem.polynomial import is_closed
from warp.fem.types import ElementIndex

from .shape import (
    CubeSerendipityShapeFunctions,
    CubeTripolynomialShapeFunctions,
    ShapeFunction,
)
from .topology import SpaceTopology, forward_base_topology


class Grid3DSpaceTopology(SpaceTopology):
    def __init__(self, grid: Grid3D, shape: ShapeFunction):
        if not is_closed(shape.family):
            raise ValueError("A closed polynomial family is required to define a continuous function space")

        super().__init__(grid, shape.NODES_PER_ELEMENT)
        self._shape = shape
        self._grid = grid

    @wp.func
    def _vertex_coords(vidx_in_cell: int):
        x = vidx_in_cell // 4
        y = (vidx_in_cell - 4 * x) // 2
        z = vidx_in_cell - 4 * x - 2 * y
        return wp.vec3i(x, y, z)

    @wp.func
    def _vertex_index(cell_arg: Grid3D.CellArg, cell_index: ElementIndex, vidx_in_cell: int):
        res = cell_arg.res
        strides = wp.vec2i((res[1] + 1) * (res[2] + 1), res[2] + 1)

        corner = Grid3D.get_cell(res, cell_index) + Grid3DSpaceTopology._vertex_coords(vidx_in_cell)
        return Grid3D._from_3d_index(strides, corner)


class GridTripolynomialSpaceTopology(Grid3DSpaceTopology):
    def __init__(self, grid: Grid3D, shape: CubeTripolynomialShapeFunctions):
        super().__init__(grid, shape)

        self.element_node_index = self._make_element_node_index()

    def node_count(self) -> int:
        return (
            (self.geometry.res[0] * self._shape.ORDER + 1)
            * (self.geometry.res[1] * self._shape.ORDER + 1)
            * (self.geometry.res[2] * self._shape.ORDER + 1)
        )

    def _make_element_node_index(self):
        ORDER = self._shape.ORDER

        @cache.dynamic_func(suffix=self.name)
        def element_node_index(
            cell_arg: Grid3D.CellArg,
            topo_arg: Grid3DSpaceTopology.TopologyArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            res = cell_arg.res
            cell = Grid3D.get_cell(res, element_index)

            node_i, node_j, node_k = self._shape._node_ijk(node_index_in_elt)

            node_x = ORDER * cell[0] + node_i
            node_y = ORDER * cell[1] + node_j
            node_z = ORDER * cell[2] + node_k

            node_pitch_y = (res[2] * ORDER) + 1
            node_pitch_x = node_pitch_y * ((res[1] * ORDER) + 1)
            node_index = node_pitch_x * node_x + node_pitch_y * node_y + node_z

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

        grid_coords_z = np.repeat(np.arange(0, res[2], dtype=float), len(cell_coords)) + np.tile(
            cell_coords, reps=res[2]
        )
        grid_coords_z = np.append(grid_coords_z, res[2])
        Z = grid_coords_z * self.geometry.cell_size[2] + self.geometry.origin[2]

        return np.meshgrid(X, Y, Z, indexing="ij")


class Grid3DSerendipitySpaceTopology(Grid3DSpaceTopology):
    def __init__(self, grid: Grid3D, shape: CubeSerendipityShapeFunctions):
        super().__init__(grid, shape)

        self.element_node_index = self._make_element_node_index()

    def node_count(self) -> int:
        return self.geometry.vertex_count() + (self._shape.ORDER - 1) * self.geometry.edge_count()

    def _make_element_node_index(self):
        ORDER = self._shape.ORDER

        @cache.dynamic_func(suffix=self.name)
        def element_node_index(
            cell_arg: Grid3D.CellArg,
            topo_arg: Grid3DSpaceTopology.TopologyArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            res = cell_arg.res
            cell = Grid3D.get_cell(res, element_index)

            node_type, type_index = self._shape.node_type_and_type_index(node_index_in_elt)

            if node_type == CubeSerendipityShapeFunctions.VERTEX:
                return Grid3DSpaceTopology._vertex_index(cell_arg, element_index, type_index)

            axis = CubeSerendipityShapeFunctions._edge_axis(node_type)
            node_all = CubeSerendipityShapeFunctions._edge_coords(type_index)

            res = cell_arg.res

            edge_index = 0
            if axis > 0:
                edge_index += (res[1] + 1) * (res[2] + 1) * res[0]
            if axis > 1:
                edge_index += (res[0] + 1) * (res[2] + 1) * res[1]

            res_loc = Grid3D._world_to_local(axis, res)
            cell_loc = Grid3D._world_to_local(axis, cell)

            edge_index += (res_loc[1] + 1) * (res_loc[2] + 1) * cell_loc[0]
            edge_index += (res_loc[2] + 1) * (cell_loc[1] + node_all[1])
            edge_index += cell_loc[2] + node_all[2]

            vertex_count = (res[0] + 1) * (res[1] + 1) * (res[2] + 1)

            return vertex_count + (ORDER - 1) * edge_index + (node_all[0] - 1)

        return element_node_index


def make_grid_3d_space_topology(grid: Grid3D, shape: ShapeFunction):
    if isinstance(shape, CubeSerendipityShapeFunctions):
        return forward_base_topology(Grid3DSerendipitySpaceTopology, grid, shape)

    if isinstance(shape, CubeTripolynomialShapeFunctions):
        return forward_base_topology(GridTripolynomialSpaceTopology, grid, shape)

    raise ValueError(f"Unsupported shape function {shape.name}")
