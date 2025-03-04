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

import warp as wp
from warp.fem import cache
from warp.fem.geometry import Quadmesh2D
from warp.fem.polynomial import is_closed
from warp.fem.types import NULL_NODE_INDEX, ElementIndex

from .shape import SquareShapeFunction
from .topology import SpaceTopology, forward_base_topology


@wp.struct
class Quadmesh2DTopologyArg:
    edge_vertex_indices: wp.array(dtype=wp.vec2i)
    quad_edge_indices: wp.array2d(dtype=int)

    vertex_count: int
    edge_count: int
    cell_count: int


class QuadmeshSpaceTopology(SpaceTopology):
    TopologyArg = Quadmesh2DTopologyArg

    def __init__(self, mesh: Quadmesh2D, shape: SquareShapeFunction):
        if shape.value == SquareShapeFunction.Value.Scalar and not is_closed(shape.family):
            raise ValueError("A closed polynomial family is required to define a continuous function space")

        self._shape = shape
        super().__init__(mesh, shape.NODES_PER_ELEMENT)
        self._mesh = mesh

        self._compute_quad_edge_indices()
        self.element_node_index = self._make_element_node_index()
        self.element_node_sign = self._make_element_node_sign()

    @property
    def name(self):
        return f"{self.geometry.name}_{self._shape.name}"

    @cache.cached_arg_value
    def topo_arg_value(self, device):
        arg = Quadmesh2DTopologyArg()
        arg.quad_edge_indices = self._quad_edge_indices.to(device)
        arg.edge_vertex_indices = self._mesh.edge_vertex_indices.to(device)

        arg.vertex_count = self._mesh.vertex_count()
        arg.edge_count = self._mesh.side_count()
        arg.cell_count = self._mesh.cell_count()
        return arg

    def _compute_quad_edge_indices(self):
        self._quad_edge_indices = wp.empty(
            dtype=int, device=self._mesh.quad_vertex_indices.device, shape=(self._mesh.cell_count(), 4)
        )

        wp.launch(
            kernel=QuadmeshSpaceTopology._compute_quad_edge_indices_kernel,
            dim=self._mesh.edge_quad_indices.shape,
            device=self._mesh.quad_vertex_indices.device,
            inputs=[
                self._mesh.edge_quad_indices,
                self._mesh.edge_vertex_indices,
                self._mesh.quad_vertex_indices,
                self._quad_edge_indices,
            ],
        )

    @wp.func
    def _find_edge_index_in_quad(
        edge_vtx: wp.vec2i,
        quad_vtx: wp.vec4i,
    ):
        for k in range(3):
            if (edge_vtx[0] == quad_vtx[k] and edge_vtx[1] == quad_vtx[k + 1]) or (
                edge_vtx[1] == quad_vtx[k] and edge_vtx[0] == quad_vtx[k + 1]
            ):
                return k
        return 3

    @wp.kernel
    def _compute_quad_edge_indices_kernel(
        edge_quad_indices: wp.array(dtype=wp.vec2i),
        edge_vertex_indices: wp.array(dtype=wp.vec2i),
        quad_vertex_indices: wp.array2d(dtype=int),
        quad_edge_indices: wp.array2d(dtype=int),
    ):
        e = wp.tid()

        edge_vtx = edge_vertex_indices[e]
        edge_quads = edge_quad_indices[e]

        q0 = edge_quads[0]
        q0_vtx = wp.vec4i(
            quad_vertex_indices[q0, 0],
            quad_vertex_indices[q0, 1],
            quad_vertex_indices[q0, 2],
            quad_vertex_indices[q0, 3],
        )
        q0_edge = QuadmeshSpaceTopology._find_edge_index_in_quad(edge_vtx, q0_vtx)
        quad_edge_indices[q0, q0_edge] = e

        q1 = edge_quads[1]
        if q1 != q0:
            t1_vtx = wp.vec4i(
                quad_vertex_indices[q1, 0],
                quad_vertex_indices[q1, 1],
                quad_vertex_indices[q1, 2],
                quad_vertex_indices[q1, 3],
            )
            t1_edge = QuadmeshSpaceTopology._find_edge_index_in_quad(edge_vtx, t1_vtx)
            quad_edge_indices[q1, t1_edge] = e

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

        SHAPE_TO_QUAD_IDX = wp.constant(wp.vec4i([0, 3, 1, 2]))

        @cache.dynamic_func(suffix=self.name)
        def element_node_index(
            cell_arg: self._mesh.CellArg,
            topo_arg: QuadmeshSpaceTopology.TopologyArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            node_type, type_instance, type_index = self._shape.node_type_and_type_index(node_index_in_elt)

            if wp.static(VERTEX_NODE_COUNT > 0):
                if node_type == SquareShapeFunction.VERTEX:
                    return (
                        cell_arg.topology.quad_vertex_indices[element_index, SHAPE_TO_QUAD_IDX[type_instance]]
                        * VERTEX_NODE_COUNT
                        + type_index
                    )

            global_offset = topo_arg.vertex_count * VERTEX_NODE_COUNT

            if wp.static(INTERIOR_NODE_COUNT > 0):
                if node_type == SquareShapeFunction.INTERIOR:
                    return global_offset + element_index * INTERIOR_NODE_COUNT + type_index

                global_offset += INTERIOR_NODE_COUNT * topo_arg.cell_count

            if wp.static(EDGE_NODE_COUNT > 0):
                # EDGE_X, EDGE_Y
                side_start = wp.where(
                    node_type == SquareShapeFunction.EDGE_X,
                    wp.where(type_instance == 0, 0, 2),
                    wp.where(type_instance == 0, 3, 1),
                )

                side_index = topo_arg.quad_edge_indices[element_index, side_start]
                local_vs = cell_arg.topology.quad_vertex_indices[element_index, side_start]
                global_vs = topo_arg.edge_vertex_indices[side_index][0]

                # Flip indexing direction
                flipped = int(side_start >= 2) ^ int(local_vs != global_vs)
                index_in_side = wp.where(flipped, EDGE_NODE_COUNT - 1 - type_index, type_index)

                return global_offset + EDGE_NODE_COUNT * side_index + index_in_side

            return NULL_NODE_INDEX  # should never happen

        return element_node_index

    def _make_element_node_sign(self):
        @cache.dynamic_func(suffix=self.name)
        def element_node_sign(
            cell_arg: self._mesh.CellArg,
            topo_arg: QuadmeshSpaceTopology.TopologyArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            node_type, type_instance, type_index = self._shape.node_type_and_type_index(node_index_in_elt)

            if node_type == SquareShapeFunction.EDGE_X or node_type == SquareShapeFunction.EDGE_Y:
                side_start = wp.where(
                    node_type == SquareShapeFunction.EDGE_X,
                    wp.where(type_instance == 0, 0, 2),
                    wp.where(type_instance == 0, 3, 1),
                )

                side_index = topo_arg.quad_edge_indices[element_index, side_start]
                local_vs = cell_arg.topology.quad_vertex_indices[element_index, side_start]
                global_vs = topo_arg.edge_vertex_indices[side_index][0]

                # Flip indexing direction
                flipped = int(side_start >= 2) ^ int(local_vs != global_vs)
                return wp.where(flipped, -1.0, 1.0)

            return 1.0

        return element_node_sign


def make_quadmesh_space_topology(mesh: Quadmesh2D, shape: SquareShapeFunction):
    if isinstance(shape, SquareShapeFunction):
        return forward_base_topology(QuadmeshSpaceTopology, mesh, shape)

    raise ValueError(f"Unsupported shape function {shape.name}")
