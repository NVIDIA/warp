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

import numpy as np

import warp as wp
from warp.fem import cache
from warp.fem.geometry import Grid2D
from warp.fem.polynomial import is_closed
from warp.fem.types import NULL_NODE_INDEX, ElementIndex

from .shape import SquareBipolynomialShapeFunctions, SquareShapeFunction
from .topology import SpaceTopology, forward_base_topology


class Grid2DSpaceTopology(SpaceTopology):
    def __init__(self, grid: Grid2D, shape: SquareShapeFunction):
        self._shape = shape
        super().__init__(grid, shape.NODES_PER_ELEMENT)

        self.element_node_index = self._make_element_node_index()

    TopologyArg = Grid2D.SideArg

    @property
    def name(self):
        return f"{self.geometry.name}_{self._shape.name}"

    def topo_arg_value(self, device):
        return self.geometry.side_arg_value(device)

    def node_count(self) -> int:
        return (
            self.geometry.vertex_count() * self._shape.VERTEX_NODE_COUNT
            + self.geometry.side_count() * self._shape.EDGE_NODE_COUNT
            + self.geometry.cell_count() * self._shape.INTERIOR_NODE_COUNT
        )

    def _make_element_node_index(self):
        VERTEX_NODE_COUNT = self._shape.VERTEX_NODE_COUNT
        EDGE_NODE_COUNT = self._shape.EDGE_NODE_COUNT
        INTERIOR_NODE_COUNT = self._shape.INTERIOR_NODE_COUNT

        @cache.dynamic_func(suffix=self.name)
        def element_node_index(
            cell_arg: Grid2D.CellArg,
            topo_arg: Grid2D.SideArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            node_type, type_instance, type_index = self._shape.node_type_and_type_index(node_index_in_elt)

            if wp.static(VERTEX_NODE_COUNT > 0):
                if node_type == SquareShapeFunction.VERTEX:
                    return (
                        Grid2DSpaceTopology._vertex_index(cell_arg, element_index, type_instance) * VERTEX_NODE_COUNT
                        + type_index
                    )

            res = cell_arg.res
            vertex_count = (res[0] + 1) * (res[1] + 1)
            global_offset = vertex_count

            if wp.static(INTERIOR_NODE_COUNT > 0):
                if node_type == SquareShapeFunction.INTERIOR:
                    return global_offset + element_index * INTERIOR_NODE_COUNT + type_index

                cell_count = res[0] * res[1]
                global_offset += INTERIOR_NODE_COUNT * cell_count

            if wp.static(EDGE_NODE_COUNT > 0):
                axis = 1 - (node_type - SquareShapeFunction.EDGE_X)

                cell = Grid2D.get_cell(cell_arg.res, element_index)
                origin = wp.vec2i(cell[Grid2D.ROTATION[axis, 0]] + type_instance, cell[Grid2D.ROTATION[axis, 1]])

                side = Grid2D.Side(axis, origin)
                side_index = Grid2D.side_index(topo_arg, side)

                vertex_count = (res[0] + 1) * (res[1] + 1)

                return global_offset + EDGE_NODE_COUNT * side_index + type_index

            return NULL_NODE_INDEX  # unreachable

        return element_node_index

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


class GridBipolynomialSpaceTopology(SpaceTopology):
    def __init__(self, grid: Grid2D, shape: SquareBipolynomialShapeFunctions):
        super().__init__(grid, shape.NODES_PER_ELEMENT)
        self._shape = shape
        self.element_node_index = self._make_element_node_index()

    def node_count(self) -> int:
        return (self.geometry.res[0] * self._shape.ORDER + 1) * (self.geometry.res[1] * self._shape.ORDER + 1)

    def _make_element_node_index(self):
        ORDER = self._shape.ORDER

        @cache.dynamic_func(suffix=self.name)
        def element_node_index(
            cell_arg: Grid2D.CellArg,
            topo_arg: self.TopologyArg,
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


def make_grid_2d_space_topology(grid: Grid2D, shape: SquareShapeFunction):
    if isinstance(shape, SquareBipolynomialShapeFunctions) and is_closed(shape.family):
        return forward_base_topology(GridBipolynomialSpaceTopology, grid, shape)

    if isinstance(shape, SquareShapeFunction):
        return forward_base_topology(Grid2DSpaceTopology, grid, shape)

    raise ValueError(f"Unsupported shape function {shape.name}")
