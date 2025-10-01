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
from warp.fem.geometry import Grid3D
from warp.fem.polynomial import is_closed
from warp.fem.types import ElementIndex

from .shape import (
    CubeShapeFunction,
    CubeTripolynomialShapeFunctions,
)
from .topology import SpaceTopology, forward_base_topology


class Grid3DSpaceTopology(SpaceTopology):
    def __init__(self, grid: Grid3D, shape: CubeShapeFunction):
        self._shape = shape
        super().__init__(grid, shape.NODES_PER_ELEMENT)
        self.element_node_index = self._make_element_node_index()

    @property
    def name(self):
        return f"{self.geometry.name}_{self._shape.name}"

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

    def node_count(self) -> int:
        return (
            self.geometry.vertex_count() * self._shape.VERTEX_NODE_COUNT
            + self.geometry.edge_count() * self._shape.EDGE_NODE_COUNT
            + self.geometry.side_count() * self._shape.FACE_NODE_COUNT
            + self.geometry.cell_count() * self._shape.INTERIOR_NODE_COUNT
        )

    def _make_element_node_index(self):
        VERTEX_NODE_COUNT = self._shape.VERTEX_NODE_COUNT
        EDGE_NODE_COUNT = self._shape.EDGE_NODE_COUNT
        FACE_NODE_COUNT = self._shape.FACE_NODE_COUNT
        INTERIOR_NODE_COUNT = self._shape.INTERIOR_NODE_COUNT

        @cache.dynamic_func(suffix=self.name)
        def element_node_index(
            cell_arg: Grid3D.CellArg,
            topo_arg: Grid3DSpaceTopology.TopologyArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            res = cell_arg.res
            cell = Grid3D.get_cell(res, element_index)

            node_type, type_instance, type_index = self._shape.node_type_and_type_index(node_index_in_elt)

            if wp.static(VERTEX_NODE_COUNT > 0):
                if node_type == CubeShapeFunction.VERTEX:
                    return (
                        Grid3DSpaceTopology._vertex_index(cell_arg, element_index, type_instance) * VERTEX_NODE_COUNT
                        + type_index
                    )

            res = cell_arg.res
            vertex_count = (res[0] + 1) * (res[1] + 1) * (res[2] + 1)
            global_offset = vertex_count * VERTEX_NODE_COUNT

            if wp.static(EDGE_NODE_COUNT > 0):
                if node_type == CubeShapeFunction.EDGE:
                    axis = CubeShapeFunction._edge_axis(type_instance)
                    node_all = CubeShapeFunction._edge_coords(type_instance, type_index)

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

                    return global_offset + EDGE_NODE_COUNT * edge_index + type_index

                edge_count = (
                    (res[0] + 1) * (res[1] + 1) * (res[2])
                    + (res[0]) * (res[1] + 1) * (res[2] + 1)
                    + (res[0] + 1) * (res[1]) * (res[2] + 1)
                )
                global_offset += edge_count * EDGE_NODE_COUNT

            if wp.static(FACE_NODE_COUNT > 0):
                if node_type == CubeShapeFunction.FACE:
                    axis = CubeShapeFunction._face_axis(type_instance)
                    face_offset = CubeShapeFunction._face_offset(type_instance)

                    face_index = 0
                    if axis > 0:
                        face_index += (res[0] + 1) * res[1] * res[2]
                    if axis > 1:
                        face_index += (res[1] + 1) * res[2] * res[0]

                    res_loc = Grid3D._world_to_local(axis, res)
                    cell_loc = Grid3D._world_to_local(axis, cell)

                    face_index += res_loc[1] * res_loc[2] * (cell_loc[0] + face_offset)
                    face_index += res_loc[2] * cell_loc[1]
                    face_index += cell_loc[2]

                    return global_offset + FACE_NODE_COUNT * face_index + type_index

                face_count = (
                    (res[0] + 1) * res[1] * res[2] + res[0] * (res[1] + 1) * res[2] + res[0] * res[1] * (res[2] + 1)
                )
                global_offset += face_count * FACE_NODE_COUNT

            # interior
            return global_offset + element_index * INTERIOR_NODE_COUNT + type_index

        return element_node_index


class GridTripolynomialSpaceTopology(SpaceTopology):
    def __init__(self, grid: Grid3D, shape: CubeTripolynomialShapeFunctions):
        super().__init__(grid, shape.NODES_PER_ELEMENT)
        self._shape = shape

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
            topo_arg: self.TopologyArg,
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


def make_grid_3d_space_topology(grid: Grid3D, shape: CubeShapeFunction):
    if isinstance(shape, CubeTripolynomialShapeFunctions) and is_closed(shape.family):
        return forward_base_topology(GridTripolynomialSpaceTopology, grid, shape)

    if isinstance(shape, CubeShapeFunction):
        return forward_base_topology(Grid3DSpaceTopology, grid, shape)

    raise ValueError(f"Unsupported shape function {shape.name}")
