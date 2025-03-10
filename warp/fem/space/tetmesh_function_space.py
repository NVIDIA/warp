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
from warp.fem.geometry import Tetmesh
from warp.fem.types import ElementIndex

from .shape import (
    ShapeFunction,
    TetrahedronPolynomialShapeFunctions,
    TetrahedronShapeFunction,
)
from .topology import SpaceTopology, forward_base_topology


@wp.struct
class TetmeshTopologyArg:
    tet_edge_indices: wp.array2d(dtype=int)
    tet_face_indices: wp.array2d(dtype=int)
    face_vertex_indices: wp.array(dtype=wp.vec3i)
    face_tet_indices: wp.array(dtype=wp.vec2i)

    vertex_count: int
    edge_count: int
    face_count: int


class TetmeshSpaceTopology(SpaceTopology):
    TopologyArg = TetmeshTopologyArg

    def __init__(
        self,
        mesh: Tetmesh,
        shape: TetrahedronShapeFunction,
    ):
        self._shape = shape
        super().__init__(mesh, shape.NODES_PER_ELEMENT)
        self._mesh = mesh

        need_tet_edge_indices = self._shape.EDGE_NODE_COUNT > 0
        need_tet_face_indices = self._shape.FACE_NODE_COUNT > 0

        if need_tet_edge_indices:
            self._tet_edge_indices = self._mesh.tet_edge_indices
            self._edge_count = self._mesh.edge_count()
        else:
            self._tet_edge_indices = wp.empty(shape=(0, 0), dtype=int)
            self._edge_count = 0

        if need_tet_face_indices:
            self._compute_tet_face_indices()
        else:
            self._tet_face_indices = wp.empty(shape=(0, 0), dtype=int)

        self.element_node_index = self._make_element_node_index()
        self.element_node_sign = self._make_element_node_sign()

    @property
    def name(self):
        return f"{self.geometry.name}_{self._shape.name}"

    @cache.cached_arg_value
    def topo_arg_value(self, device):
        arg = TetmeshTopologyArg()
        arg.tet_face_indices = self._tet_face_indices.to(device)
        arg.tet_edge_indices = self._tet_edge_indices.to(device)
        arg.face_vertex_indices = self._mesh.face_vertex_indices.to(device)
        arg.face_tet_indices = self._mesh.face_tet_indices.to(device)

        arg.vertex_count = self._mesh.vertex_count()
        arg.face_count = self._mesh.side_count()
        arg.edge_count = self._edge_count
        return arg

    def _compute_tet_face_indices(self):
        self._tet_face_indices = wp.empty(
            dtype=int, device=self._mesh.tet_vertex_indices.device, shape=(self._mesh.cell_count(), 4)
        )

        wp.launch(
            kernel=TetmeshSpaceTopology._compute_tet_face_indices_kernel,
            dim=self._mesh._face_tet_indices.shape,
            device=self._mesh.tet_vertex_indices.device,
            inputs=[
                self._mesh.face_tet_indices,
                self._mesh.face_vertex_indices,
                self._mesh.tet_vertex_indices,
                self._tet_face_indices,
            ],
        )

    @wp.func
    def _find_face_index_in_tet(
        face_vtx: wp.vec3i,
        tet_vtx: wp.vec4i,
    ):
        for k in range(3):
            tvk = wp.vec3i(tet_vtx[k], tet_vtx[(k + 1) % 4], tet_vtx[(k + 2) % 4])

            # Use fact that face always start with min vertex
            min_t = wp.min(tvk)
            max_t = wp.max(tvk)
            mid_t = tvk[0] + tvk[1] + tvk[2] - min_t - max_t

            if min_t == face_vtx[0] and (
                (face_vtx[2] == max_t and face_vtx[1] == mid_t) or (face_vtx[1] == max_t and face_vtx[2] == mid_t)
            ):
                return k

        return 3

    @wp.kernel
    def _compute_tet_face_indices_kernel(
        face_tet_indices: wp.array(dtype=wp.vec2i),
        face_vertex_indices: wp.array(dtype=wp.vec3i),
        tet_vertex_indices: wp.array2d(dtype=int),
        tet_face_indices: wp.array2d(dtype=int),
    ):
        e = wp.tid()

        face_vtx = face_vertex_indices[e]
        face_tets = face_tet_indices[e]

        t0 = face_tets[0]
        t0_vtx = wp.vec4i(
            tet_vertex_indices[t0, 0], tet_vertex_indices[t0, 1], tet_vertex_indices[t0, 2], tet_vertex_indices[t0, 3]
        )
        t0_face = TetmeshSpaceTopology._find_face_index_in_tet(face_vtx, t0_vtx)
        tet_face_indices[t0, t0_face] = e

        t1 = face_tets[1]
        if t1 != t0:
            t1_vtx = wp.vec4i(
                tet_vertex_indices[t1, 0],
                tet_vertex_indices[t1, 1],
                tet_vertex_indices[t1, 2],
                tet_vertex_indices[t1, 3],
            )
            t1_face = TetmeshSpaceTopology._find_face_index_in_tet(face_vtx, t1_vtx)
            tet_face_indices[t1, t1_face] = e

    def node_count(self) -> int:
        return (
            self._mesh.vertex_count() * self._shape.VERTEX_NODE_COUNT
            + self._mesh.edge_count() * self._shape.EDGE_NODE_COUNT
            + self._mesh.side_count() * self._shape.FACE_NODE_COUNT
            + self._mesh.cell_count() * self._shape.INTERIOR_NODE_COUNT
        )

    def _make_element_node_index(self):
        VERTEX_NODE_COUNT = self._shape.VERTEX_NODE_COUNT
        INTERIOR_NODES_PER_EDGE = self._shape.EDGE_NODE_COUNT
        INTERIOR_NODES_PER_FACE = self._shape.FACE_NODE_COUNT
        INTERIOR_NODES_PER_CELL = self._shape.INTERIOR_NODE_COUNT

        @cache.dynamic_func(suffix=self.name)
        def element_node_index(
            geo_arg: Tetmesh.CellArg,
            topo_arg: TetmeshTopologyArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            node_type, type_index = self._shape.node_type_and_type_index(node_index_in_elt)

            if node_type == TetrahedronPolynomialShapeFunctions.VERTEX:
                vertex = type_index // VERTEX_NODE_COUNT
                vertex_node = type_index - VERTEX_NODE_COUNT * vertex
                return geo_arg.tet_vertex_indices[element_index][vertex] * VERTEX_NODE_COUNT + vertex_node

            global_offset = topo_arg.vertex_count * VERTEX_NODE_COUNT

            if node_type == TetrahedronPolynomialShapeFunctions.EDGE:
                edge = type_index // INTERIOR_NODES_PER_EDGE
                edge_node = type_index - INTERIOR_NODES_PER_EDGE * edge

                global_edge_index = topo_arg.tet_edge_indices[element_index][edge]

                # Test if we need to swap edge direction
                if wp.static(INTERIOR_NODES_PER_EDGE > 1):
                    c1, c2 = TetrahedronShapeFunction.edge_vidx(edge)
                    if geo_arg.tet_vertex_indices[element_index][c1] > geo_arg.tet_vertex_indices[element_index][c2]:
                        edge_node = INTERIOR_NODES_PER_EDGE - 1 - edge_node

                return global_offset + INTERIOR_NODES_PER_EDGE * global_edge_index + edge_node

            global_offset += INTERIOR_NODES_PER_EDGE * topo_arg.edge_count

            if node_type == TetrahedronPolynomialShapeFunctions.FACE:
                face = type_index // INTERIOR_NODES_PER_FACE
                face_node = type_index - INTERIOR_NODES_PER_FACE * face

                global_face_index = topo_arg.tet_face_indices[element_index][face]

                if wp.static(INTERIOR_NODES_PER_FACE == 3):
                    # Hard code for P4 case, 3 nodes per face
                    # Higher orders would require rotating triangle coordinates, this is not supported yet

                    vidx = geo_arg.tet_vertex_indices[element_index][(face + face_node) % 4]
                    fvi = topo_arg.face_vertex_indices[global_face_index]

                    if vidx == fvi[0]:
                        face_node = 0
                    elif vidx == fvi[1]:
                        face_node = 1
                    else:
                        face_node = 2

                return global_offset + INTERIOR_NODES_PER_FACE * global_face_index + face_node

            global_offset += INTERIOR_NODES_PER_FACE * topo_arg.face_count

            return global_offset + INTERIOR_NODES_PER_CELL * element_index + type_index

        return element_node_index

    def _make_element_node_sign(self):
        INTERIOR_NODES_PER_EDGE = self._shape.EDGE_NODE_COUNT
        INTERIOR_NODES_PER_FACE = self._shape.FACE_NODE_COUNT

        @cache.dynamic_func(suffix=self.name)
        def element_node_sign(
            geo_arg: self.geometry.CellArg,
            topo_arg: TetmeshTopologyArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            node_type, type_index = self._shape.node_type_and_type_index(node_index_in_elt)

            if wp.static(INTERIOR_NODES_PER_EDGE > 0):
                if node_type == TetrahedronShapeFunction.EDGE:
                    edge = type_index // INTERIOR_NODES_PER_EDGE
                    c1, c2 = TetrahedronShapeFunction.edge_vidx(edge)

                    return wp.where(
                        geo_arg.tet_vertex_indices[element_index][c1] > geo_arg.tet_vertex_indices[element_index][c2],
                        -1.0,
                        1.0,
                    )

            if wp.static(INTERIOR_NODES_PER_FACE > 0):
                if node_type == TetrahedronShapeFunction.FACE:
                    face = type_index // INTERIOR_NODES_PER_FACE

                    global_face_index = topo_arg.tet_face_indices[element_index][face]
                    inner = topo_arg.face_tet_indices[global_face_index][0]

                    return wp.where(inner == element_index, 1.0, -1.0)

            return 1.0

        return element_node_sign


def make_tetmesh_space_topology(mesh: Tetmesh, shape: ShapeFunction):
    if isinstance(shape, TetrahedronShapeFunction):
        return forward_base_topology(TetmeshSpaceTopology, mesh, shape)

    raise ValueError(f"Unsupported shape function {shape.name}")
