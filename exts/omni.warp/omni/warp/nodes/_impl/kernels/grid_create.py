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

"""Warp kernel creating a grid mesh geometry."""

from typing import Tuple

import warp as wp

#   Helpers
# -----------------------------------------------------------------------------


@wp.func
def _define_face(
    face: int,
    vertex_1: int,
    vertex_2: int,
    vertex_3: int,
    vertex_4: int,
    out_face_vertex_indices: wp.array(dtype=int),
):
    out_face_vertex_indices[face * 4 + 0] = vertex_1
    out_face_vertex_indices[face * 4 + 1] = vertex_2
    out_face_vertex_indices[face * 4 + 2] = vertex_3
    out_face_vertex_indices[face * 4 + 3] = vertex_4


@wp.func
def _set_face_normals(
    face: int,
    normal: wp.vec3,
    out_normals: wp.array(dtype=wp.vec3),
):
    out_normals[face * 4 + 0] = normal
    out_normals[face * 4 + 1] = normal
    out_normals[face * 4 + 2] = normal
    out_normals[face * 4 + 3] = normal


@wp.func
def _set_face_uvs(
    face: int,
    uv_1: wp.vec2,
    uv_2: wp.vec2,
    uv_3: wp.vec2,
    uv_4: wp.vec2,
    out_uvs: wp.array(dtype=wp.vec2),
):
    out_uvs[face * 4 + 0] = uv_1
    out_uvs[face * 4 + 1] = uv_2
    out_uvs[face * 4 + 2] = uv_3
    out_uvs[face * 4 + 3] = uv_4


#   Kernel
# -----------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _kernel(
    half_size: wp.vec2,
    res: wp.vec2i,
    update_topology: int,
    dt_pos: wp.vec2,
    dt_uv: wp.vec2,
    out_points: wp.array(dtype=wp.vec3),
    out_face_vertex_indices: wp.array(dtype=int),
    out_normals: wp.array(dtype=wp.vec3),
    out_uvs: wp.array(dtype=wp.vec2),
):
    """Kernel to create a geometry mesh grid."""
    tid = wp.tid()

    i = int(tid % res[0])
    j = int(tid / res[0])

    if i == 0 and j == 0:
        point = 0
        out_points[point] = wp.vec3(
            half_size[0],
            0.0,
            half_size[1],
        )

    if i == 0:
        point = (j + 1) * (res[0] + 1)
        out_points[point] = wp.vec3(
            half_size[0],
            0.0,
            half_size[1] - dt_pos[1] * float(j + 1),
        )

    if j == 0:
        point = i + 1
        out_points[point] = wp.vec3(
            half_size[0] - dt_pos[0] * float(i + 1),
            0.0,
            half_size[1],
        )

    point = (j + 1) * (res[0] + 1) + i + 1
    out_points[point] = wp.vec3(
        half_size[0] - dt_pos[0] * float(i + 1),
        0.0,
        half_size[1] - dt_pos[1] * float(j + 1),
    )

    if update_topology:
        face = tid

        # Face vertex indices.
        vertex_4 = point
        vertex_3 = vertex_4 - 1
        vertex_1 = vertex_3 - res[0]
        vertex_2 = vertex_1 - 1
        _define_face(face, vertex_1, vertex_2, vertex_3, vertex_4, out_face_vertex_indices)

        # Vertex normals.
        _set_face_normals(face, wp.vec3(0.0, 1.0, 0.0), out_normals)

        # Vertex UVs.
        s_0 = 1.0 - dt_uv[0] * float(i)
        s_1 = 1.0 - dt_uv[0] * float(i + 1)
        t_0 = dt_uv[1] * float(j)
        t_1 = dt_uv[1] * float(j + 1)
        _set_face_uvs(
            face,
            wp.vec2(s_1, t_0),
            wp.vec2(s_0, t_0),
            wp.vec2(s_0, t_1),
            wp.vec2(s_1, t_1),
            out_uvs,
        )


#   Launcher
# -----------------------------------------------------------------------------


def grid_create_launch_kernel(
    out_points: wp.array,
    out_face_vertex_counts: wp.array,
    out_face_vertex_indices: wp.array,
    out_normals: wp.array,
    out_uvs: wp.array,
    size: Tuple[float, float],
    dims: Tuple[int, int],
    update_topology: bool = True,
):
    """Launches the kernel."""
    face_count = dims[0] * dims[1]

    half_size = (
        size[0] * 0.5,
        size[1] * 0.5,
    )
    dt_pos = wp.vec2(
        size[0] / float(dims[0]),
        size[1] / float(dims[1]),
    )
    dt_uv = (
        1.0 / float(dims[0]),
        1.0 / float(dims[1]),
    )

    wp.launch(
        kernel=_kernel,
        dim=face_count,
        inputs=[
            half_size,
            dims,
            update_topology,
            dt_pos,
            dt_uv,
        ],
        outputs=[
            out_points,
            out_face_vertex_indices,
            out_normals,
            out_uvs,
        ],
    )

    out_face_vertex_counts.fill_(4)
