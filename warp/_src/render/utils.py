# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np

import warp as wp


def bourke_color_map(low, high, v):
    c = [1.0, 1.0, 1.0]

    if v < low:
        v = low
    if v > high:
        v = high
    dv = high - low

    if v < (low + 0.25 * dv):
        c[0] = 0.0
        c[1] = 4.0 * (v - low) / dv
    elif v < (low + 0.5 * dv):
        c[0] = 0.0
        c[2] = 1.0 + 4.0 * (low + 0.25 * dv - v) / dv
    elif v < (low + 0.75 * dv):
        c[0] = 4.0 * (v - low - 0.5 * dv) / dv
        c[2] = 0.0
    else:
        c[1] = 1.0 + 4.0 * (low + 0.75 * dv - v) / dv
        c[2] = 0.0

    return c


def tab10_color_map(i):
    # matplotlib "tab10" colors
    colors = [
        [31, 119, 180],
        [255, 127, 14],
        [44, 160, 44],
        [214, 39, 40],
        [148, 103, 189],
        [140, 86, 75],
        [227, 119, 194],
        [127, 127, 127],
        [188, 189, 34],
        [23, 190, 207],
    ]
    num_colors = len(colors)
    return [c / 255.0 for c in colors[i % num_colors]]


# triangulate mesh around given surface with given thickness
@wp.kernel
def solidify_mesh_kernel(
    indices: wp.array(dtype=int, ndim=2),
    vertices: wp.array(dtype=wp.vec3, ndim=1),
    thickness: wp.array(dtype=float, ndim=1),
    # outputs
    out_vertices: wp.array(dtype=wp.vec3, ndim=1),
    out_indices: wp.array(dtype=int, ndim=2),
):
    tid = wp.tid()
    i = indices[tid, 0]
    j = indices[tid, 1]
    k = indices[tid, 2]

    vi = vertices[i]
    vj = vertices[j]
    vk = vertices[k]

    normal = wp.normalize(wp.cross(vj - vi, vk - vi))
    ti = normal * thickness[i]
    tj = normal * thickness[j]
    tk = normal * thickness[k]

    # wedge vertices
    vi0 = vi + ti
    vi1 = vi - ti
    vj0 = vj + tj
    vj1 = vj - tj
    vk0 = vk + tk
    vk1 = vk - tk

    i0 = i * 2
    i1 = i * 2 + 1
    j0 = j * 2
    j1 = j * 2 + 1
    k0 = k * 2
    k1 = k * 2 + 1

    out_vertices[i0] = vi0
    out_vertices[i1] = vi1
    out_vertices[j0] = vj0
    out_vertices[j1] = vj1
    out_vertices[k0] = vk0
    out_vertices[k1] = vk1

    oid = tid * 8
    out_indices[oid + 0, 0] = i0
    out_indices[oid + 0, 1] = j0
    out_indices[oid + 0, 2] = k0
    out_indices[oid + 1, 0] = j0
    out_indices[oid + 1, 1] = k1
    out_indices[oid + 1, 2] = k0
    out_indices[oid + 2, 0] = j0
    out_indices[oid + 2, 1] = j1
    out_indices[oid + 2, 2] = k1
    out_indices[oid + 3, 0] = j0
    out_indices[oid + 3, 1] = i1
    out_indices[oid + 3, 2] = j1
    out_indices[oid + 4, 0] = j0
    out_indices[oid + 4, 1] = i0
    out_indices[oid + 4, 2] = i1
    out_indices[oid + 5, 0] = j1
    out_indices[oid + 5, 1] = i1
    out_indices[oid + 5, 2] = k1
    out_indices[oid + 6, 0] = i1
    out_indices[oid + 6, 1] = i0
    out_indices[oid + 6, 2] = k0
    out_indices[oid + 7, 0] = i1
    out_indices[oid + 7, 1] = k0
    out_indices[oid + 7, 2] = k1


def solidify_mesh(faces: np.ndarray, vertices: np.ndarray, thickness: Union[list, float]):
    """
    Triangulate mesh around given surface with given thickness.
    :param faces: array of face indices (Nx3)
    :param vertices: array of vertex positions (Mx3)
    :param thickness: array of thickness values (Mx1) or single thickness value
    :return: tuple of (faces, vertices)
    """
    faces = np.array(faces).reshape(-1, 3)
    out_faces = wp.zeros((len(faces) * 8, 3), dtype=wp.int32)
    out_vertices = wp.zeros(len(vertices) * 2, dtype=wp.vec3)
    if not isinstance(thickness, np.ndarray) and not isinstance(thickness, list):
        thickness = [thickness] * len(vertices)
    wp.launch(
        solidify_mesh_kernel,
        dim=len(faces),
        inputs=[wp.array(faces, dtype=int), wp.array(vertices, dtype=wp.vec3), wp.array(thickness, dtype=float)],
        outputs=[out_vertices, out_faces],
    )
    faces = out_faces.numpy()
    vertices = out_vertices.numpy()
    return faces, vertices
