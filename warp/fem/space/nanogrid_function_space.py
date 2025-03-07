# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Union

import warp as wp
from warp.fem import cache
from warp.fem.geometry import AdaptiveNanogrid, Nanogrid
from warp.fem.types import ElementIndex

from .shape import CubeShapeFunction
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
        grid: Union[Nanogrid, AdaptiveNanogrid],
        shape: CubeShapeFunction,
    ):
        self._shape = shape
        super().__init__(grid, shape.NODES_PER_ELEMENT)
        self._grid = grid

        self._vertex_grid = grid.vertex_grid.id

        need_edge_indices = shape.EDGE_NODE_COUNT > 0
        need_face_indices = shape.FACE_NODE_COUNT > 0

        if isinstance(grid, Nanogrid):
            self._edge_grid = grid.edge_grid.id if need_edge_indices else -1
            self._face_grid = grid.face_grid.id if need_face_indices else -1
            self._edge_count = grid.edge_count() if need_edge_indices else 0
            self._face_count = grid.side_count() if need_face_indices else 0
        else:
            self._edge_grid = grid.stacked_edge_grid.id if need_edge_indices else -1
            self._face_grid = grid.stacked_face_grid.id if need_face_indices else -1
            self._edge_count = grid.stacked_edge_count() if need_edge_indices else 0
            self._face_count = grid.stacked_face_count() if need_face_indices else 0

        self.element_node_index = self._make_element_node_index()

    @property
    def name(self):
        return f"{self.geometry.name}_{self._shape.name}"

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

    def _make_element_node_index(self):
        element_node_index_generic = self._make_element_node_index_generic()

        @cache.dynamic_func(suffix=self.name)
        def element_node_index(
            geo_arg: Nanogrid.CellArg,
            topo_arg: NanogridTopologyArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            ijk = geo_arg.cell_ijk[element_index]
            return element_node_index_generic(topo_arg, element_index, node_index_in_elt, ijk, 0)

        if isinstance(self._grid, Nanogrid):
            return element_node_index

        @cache.dynamic_func(suffix=self.name)
        def element_node_index_adaptive(
            geo_arg: AdaptiveNanogrid.CellArg,
            topo_arg: NanogridTopologyArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            ijk = geo_arg.cell_ijk[element_index]
            level = int(geo_arg.cell_level[element_index])
            return element_node_index_generic(topo_arg, element_index, node_index_in_elt, ijk, level)

        return element_node_index_adaptive

    def node_count(self) -> int:
        return (
            self._grid.vertex_count() * self._shape.VERTEX_NODE_COUNT
            + self._edge_count * self._shape.EDGE_NODE_COUNT
            + self._face_count * self._shape.FACE_NODE_COUNT
            + self._grid.cell_count() * self._shape.INTERIOR_NODE_COUNT
        )

    def _make_element_node_index_generic(self):
        VERTEX_NODE_COUNT = self._shape.VERTEX_NODE_COUNT
        EDGE_NODE_COUNT = self._shape.EDGE_NODE_COUNT
        FACE_NODE_COUNT = self._shape.FACE_NODE_COUNT
        INTERIOR_NODE_COUNT = self._shape.INTERIOR_NODE_COUNT

        @cache.dynamic_func(suffix=self.name)
        def element_node_index_generic(
            topo_arg: NanogridTopologyArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
            ijk: wp.vec3i,
            level: int,
        ):
            node_type, type_instance, type_index = self._shape.node_type_and_type_index(node_index_in_elt)

            if wp.static(VERTEX_NODE_COUNT > 0):
                if node_type == CubeShapeFunction.VERTEX:
                    n_ijk = _cell_vertex_coord(ijk, level, type_instance)
                    return (
                        wp.volume_lookup_index(topo_arg.vertex_grid, n_ijk[0], n_ijk[1], n_ijk[2]) * VERTEX_NODE_COUNT
                        + type_index
                    )

            offset = topo_arg.vertex_count * VERTEX_NODE_COUNT

            if wp.static(EDGE_NODE_COUNT > 0):
                if node_type == CubeShapeFunction.EDGE:
                    axis = type_instance >> 2
                    node_offset = type_instance & 3

                    n_ijk = _cell_edge_coord(ijk, level, axis, node_offset)

                    edge_index = wp.volume_lookup_index(topo_arg.edge_grid, n_ijk[0], n_ijk[1], n_ijk[2])
                    return offset + EDGE_NODE_COUNT * edge_index + type_index

                offset += EDGE_NODE_COUNT * topo_arg.edge_count

            if wp.static(FACE_NODE_COUNT > 0):
                if node_type == CubeShapeFunction.FACE:
                    axis = type_instance >> 1
                    node_offset = type_instance & 1

                    n_ijk = _cell_face_coord(ijk, level, axis, node_offset)

                    face_index = wp.volume_lookup_index(topo_arg.face_grid, n_ijk[0], n_ijk[1], n_ijk[2])
                    return offset + FACE_NODE_COUNT * face_index + type_index

                offset += FACE_NODE_COUNT * topo_arg.face_count

            return offset + INTERIOR_NODE_COUNT * element_index + type_index

        return element_node_index_generic


@wp.func
def _cell_vertex_coord(cell_ijk: wp.vec3i, cell_level: int, n: int):
    return cell_ijk + AdaptiveNanogrid.fine_ijk(wp.vec3i((n & 4) >> 2, (n & 2) >> 1, n & 1), cell_level)


@wp.func
def _cell_edge_coord(cell_ijk: wp.vec3i, cell_level: int, axis: int, offset: int):
    e_ijk = AdaptiveNanogrid.coarse_ijk(cell_ijk, cell_level)
    e_ijk[(axis + 1) % 3] += offset >> 1
    e_ijk[(axis + 2) % 3] += offset & 1
    return AdaptiveNanogrid.encode_axis_and_level(e_ijk, axis, cell_level)


@wp.func
def _cell_face_coord(cell_ijk: wp.vec3i, cell_level: int, axis: int, offset: int):
    f_ijk = AdaptiveNanogrid.coarse_ijk(cell_ijk, cell_level)
    f_ijk[axis] += offset
    return AdaptiveNanogrid.encode_axis_and_level(f_ijk, axis, cell_level)


def make_nanogrid_space_topology(grid: Union[Nanogrid, AdaptiveNanogrid], shape: CubeShapeFunction):
    if isinstance(shape, CubeShapeFunction):
        return forward_base_topology(NanogridSpaceTopology, grid, shape)

    raise ValueError(f"Unsupported shape function {shape.name}")
