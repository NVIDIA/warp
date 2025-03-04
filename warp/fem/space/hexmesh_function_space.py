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

import warp as wp
from warp.fem import cache
from warp.fem.geometry import Hexmesh
from warp.fem.geometry.hexmesh import (
    EDGE_VERTEX_INDICES,
    FACE_ORIENTATION,
    FACE_TRANSLATION,
)
from warp.fem.types import ElementIndex

from .shape import CubeShapeFunction
from .topology import SpaceTopology, forward_base_topology

_FACE_ORIENTATION_I = wp.constant(wp.mat(shape=(16, 2), dtype=int)(FACE_ORIENTATION))
_FACE_TRANSLATION_I = wp.constant(wp.mat(shape=(4, 2), dtype=int)(FACE_TRANSLATION))

# map from shape function vertex indexing to hexmesh vertex indexing
_CUBE_TO_HEX_VERTEX = wp.constant(wp.vec(length=8, dtype=int)([0, 4, 3, 7, 1, 5, 2, 6]))

# map from shape function edge indexing to hexmesh edge indexing
_CUBE_TO_HEX_EDGE = wp.constant(wp.vec(length=12, dtype=int)([0, 4, 2, 6, 3, 1, 7, 5, 8, 11, 9, 10]))


@wp.struct
class HexmeshTopologyArg:
    hex_edge_indices: wp.array2d(dtype=int)
    hex_face_indices: wp.array2d(dtype=wp.vec2i)

    vertex_count: int
    edge_count: int
    face_count: int


class HexmeshSpaceTopology(SpaceTopology):
    TopologyArg = HexmeshTopologyArg

    def __init__(
        self,
        mesh: Hexmesh,
        shape: CubeShapeFunction,
    ):
        self._shape = shape
        super().__init__(mesh, shape.NODES_PER_ELEMENT)
        self._mesh = mesh

        need_edge_indices = shape.EDGE_NODE_COUNT > 0
        need_face_indices = shape.FACE_NODE_COUNT > 0

        if need_edge_indices:
            self._hex_edge_indices = self._mesh.hex_edge_indices
            self._edge_count = self._mesh.edge_count()
        else:
            self._hex_edge_indices = wp.empty(shape=(0, 0), dtype=int)
            self._edge_count = 0

        if need_face_indices:
            self._compute_hex_face_indices()
        else:
            self._hex_face_indices = wp.empty(shape=(0, 0), dtype=wp.vec2i)

        self._compute_hex_face_indices()

        self.element_node_index = self._make_element_node_index()
        self.element_node_sign = self._make_element_node_sign()

    @property
    def name(self):
        return f"{self.geometry.name}_{self._shape.name}"

    @cache.cached_arg_value
    def topo_arg_value(self, device):
        arg = HexmeshTopologyArg()
        arg.hex_edge_indices = self._hex_edge_indices.to(device)
        arg.hex_face_indices = self._hex_face_indices.to(device)

        arg.vertex_count = self._mesh.vertex_count()
        arg.face_count = self._mesh.side_count()
        arg.edge_count = self._edge_count
        return arg

    def _compute_hex_face_indices(self):
        self._hex_face_indices = wp.empty(
            dtype=wp.vec2i, device=self._mesh.hex_vertex_indices.device, shape=(self._mesh.cell_count(), 6)
        )

        wp.launch(
            kernel=HexmeshSpaceTopology._compute_hex_face_indices_kernel,
            dim=self._mesh.side_count(),
            device=self._mesh.hex_vertex_indices.device,
            inputs=[
                self._mesh.face_hex_indices,
                self._mesh._face_hex_face_orientation,
                self._hex_face_indices,
            ],
        )

    @wp.kernel
    def _compute_hex_face_indices_kernel(
        face_hex_indices: wp.array(dtype=wp.vec2i),
        face_hex_face_ori: wp.array(dtype=wp.vec4i),
        hex_face_indices: wp.array2d(dtype=wp.vec2i),
    ):
        f = wp.tid()

        # face indices from CubeShapeFunction always have positive orientation,
        # while Hexmesh faces are oriented to point "outside"
        # We need to flip orientation for faces at offset 0

        hx0 = face_hex_indices[f][0]
        local_face_0 = face_hex_face_ori[f][0]
        ori_0 = face_hex_face_ori[f][1]

        local_face_offset_0 = CubeShapeFunction._face_offset(local_face_0)
        flip_0 = ori_0 ^ (1 - local_face_offset_0)

        hex_face_indices[hx0, local_face_0] = wp.vec2i(f, flip_0)

        hx1 = face_hex_indices[f][1]
        local_face_1 = face_hex_face_ori[f][2]
        ori_1 = face_hex_face_ori[f][3]

        local_face_offset_1 = CubeShapeFunction._face_offset(local_face_1)
        flip_1 = ori_1 ^ (1 - local_face_offset_1)

        hex_face_indices[hx1, local_face_1] = wp.vec2i(f, flip_1)

    def node_count(self) -> int:
        return (
            self._mesh.vertex_count() * self._shape.VERTEX_NODE_COUNT
            + self._mesh.edge_count() * self._shape.EDGE_NODE_COUNT
            + self._mesh.side_count() * self._shape.FACE_NODE_COUNT
            + self._mesh.cell_count() * self._shape.INTERIOR_NODE_COUNT
        )

    @wp.func
    def _rotate_face_coordinates(ori: int, offset: int, coords: wp.vec2i):
        fv = ori // 2

        rot_i = wp.dot(_FACE_ORIENTATION_I[2 * ori], coords)
        rot_j = wp.dot(_FACE_ORIENTATION_I[2 * ori + 1], coords)

        return wp.vec2i(rot_i, rot_j) + _FACE_TRANSLATION_I[fv]

    def _make_element_node_index(self):
        VERTEX_NODE_COUNT = self._shape.VERTEX_NODE_COUNT
        EDGE_NODE_COUNT = self._shape.EDGE_NODE_COUNT
        FACE_NODE_COUNT = self._shape.FACE_NODE_COUNT
        INTERIOR_NODE_COUNT = self._shape.INTERIOR_NODE_COUNT

        @cache.dynamic_func(suffix=self.name)
        def element_node_index(
            geo_arg: Hexmesh.CellArg,
            topo_arg: HexmeshTopologyArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            node_type, type_instance, type_index = self._shape.node_type_and_type_index(node_index_in_elt)

            if wp.static(VERTEX_NODE_COUNT > 0):
                if node_type == CubeShapeFunction.VERTEX:
                    return (
                        geo_arg.hex_vertex_indices[element_index, _CUBE_TO_HEX_VERTEX[type_instance]]
                        * VERTEX_NODE_COUNT
                        + type_index
                    )

            offset = topo_arg.vertex_count * VERTEX_NODE_COUNT

            if wp.static(EDGE_NODE_COUNT > 0):
                if node_type == CubeShapeFunction.EDGE:
                    hex_edge = _CUBE_TO_HEX_EDGE[type_instance]
                    edge_index = topo_arg.hex_edge_indices[element_index, hex_edge]

                    v0 = geo_arg.hex_vertex_indices[element_index, EDGE_VERTEX_INDICES[hex_edge, 0]]
                    v1 = geo_arg.hex_vertex_indices[element_index, EDGE_VERTEX_INDICES[hex_edge, 1]]

                    if v0 > v1:
                        type_index = EDGE_NODE_COUNT - 1 - type_index

                    return offset + EDGE_NODE_COUNT * edge_index + type_index

                offset += EDGE_NODE_COUNT * topo_arg.edge_count

            if wp.static(FACE_NODE_COUNT > 0):
                if node_type == CubeShapeFunction.FACE:
                    face_index_and_ori = topo_arg.hex_face_indices[element_index, type_instance]
                    face_index = face_index_and_ori[0]
                    face_orientation = face_index_and_ori[1]

                    face_offset = CubeShapeFunction._face_offset(type_instance)

                    if wp.static(FACE_NODE_COUNT > 1):
                        face_coords = self._shape._face_node_ij(type_index)
                        rot_face_coords = HexmeshSpaceTopology._rotate_face_coordinates(
                            face_orientation, face_offset, face_coords
                        )
                        type_index = self._shape._linear_face_node_index(type_index, rot_face_coords)

                    return offset + FACE_NODE_COUNT * face_index + type_index

                offset += FACE_NODE_COUNT * topo_arg.face_count

            return offset + INTERIOR_NODE_COUNT * element_index + type_index

        return element_node_index

    def _make_element_node_sign(self):
        EDGE_NODE_COUNT = self._shape.EDGE_NODE_COUNT
        FACE_NODE_COUNT = self._shape.FACE_NODE_COUNT

        @cache.dynamic_func(suffix=self.name)
        def element_node_sign(
            geo_arg: self.geometry.CellArg,
            topo_arg: HexmeshTopologyArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            node_type, type_instance, type_index = self._shape.node_type_and_type_index(node_index_in_elt)

            if wp.static(EDGE_NODE_COUNT > 0):
                if node_type == CubeShapeFunction.EDGE:
                    hex_edge = _CUBE_TO_HEX_EDGE[type_instance]
                    v0 = geo_arg.hex_vertex_indices[element_index, EDGE_VERTEX_INDICES[hex_edge, 0]]
                    v1 = geo_arg.hex_vertex_indices[element_index, EDGE_VERTEX_INDICES[hex_edge, 1]]
                    return wp.where(v0 > v1, -1.0, 1.0)

            if wp.static(FACE_NODE_COUNT > 0):
                if node_type == CubeShapeFunction.FACE:
                    face_index_and_ori = topo_arg.hex_face_indices[element_index, type_instance]
                    flip = face_index_and_ori[1] & 1
                    return wp.where(flip == 0, 1.0, -1.0)

            return 1.0

        return element_node_sign


def make_hexmesh_space_topology(mesh: Hexmesh, shape: CubeShapeFunction):
    if isinstance(shape, CubeShapeFunction):
        return forward_base_topology(HexmeshSpaceTopology, mesh, shape)

    raise ValueError(f"Unsupported shape function {shape.name}")
