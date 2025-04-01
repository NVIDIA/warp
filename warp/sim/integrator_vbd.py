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

import numpy as np

import warp as wp

from ..types import float32, matrix
from .collide import (
    TriMeshCollisionDetector,
    TriMeshCollisionInfo,
    get_edge_colliding_edges,
    get_edge_colliding_edges_count,
    get_triangle_colliding_vertices,
    get_triangle_colliding_vertices_count,
    get_vertex_colliding_triangles,
    get_vertex_colliding_triangles_count,
    triangle_closest_point,
)
from .integrator import Integrator
from .model import PARTICLE_FLAG_ACTIVE, Control, Model, ModelShapeMaterials, State

wp.set_module_options({"enable_backward": False})

VBD_DEBUG_PRINTING_OPTIONS = {
    # "elasticity_force_hessian",
    # "contact_force_hessian",
    # "contact_force_hessian_vt",
    # "contact_force_hessian_ee",
    # "overall_force_hessian",
    # "inertia_force_hessian",
    # "connectivity",
    # "contact_info",
}


class mat66(matrix(shape=(6, 6), dtype=float32)):
    pass


class mat32(matrix(shape=(3, 2), dtype=float32)):
    pass


class mat43(matrix(shape=(4, 3), dtype=float32)):
    pass


@wp.struct
class ForceElementAdjacencyInfo:
    r"""
    - vertex_adjacent_[element]: the flatten adjacency information. Its size is \sum_{i\inV} 2*N_i, where N_i is the
    number of vertex i’s adjacent [element]. For each adjacent element it stores 2 information:
        - the id of the adjacent element
        - the order of the vertex in the element, which is essential to compute the force and hessian for the vertex
    - vertex_adjacent_[element]_offsets: stores where each vertex’ information starts in the  flatten adjacency array.
    Its size is |V|+1 such that the number of vertex i’s adjacent [element] can be computed as
    vertex_adjacent_[element]_offsets[i+1]-vertex_adjacent_[element]_offsets[i].
    """

    v_adj_faces: wp.array(dtype=int)
    v_adj_faces_offsets: wp.array(dtype=int)

    v_adj_edges: wp.array(dtype=int)
    v_adj_edges_offsets: wp.array(dtype=int)

    def to(self, device):
        if device.is_cpu:
            return self
        else:
            adjacency_gpu = ForceElementAdjacencyInfo()
            adjacency_gpu.v_adj_faces = self.v_adj_faces.to(device)
            adjacency_gpu.v_adj_faces_offsets = self.v_adj_faces_offsets.to(device)

            adjacency_gpu.v_adj_edges = self.v_adj_edges.to(device)
            adjacency_gpu.v_adj_edges_offsets = self.v_adj_edges_offsets.to(device)

            return adjacency_gpu


@wp.func
def get_vertex_num_adjacent_edges(adjacency: ForceElementAdjacencyInfo, vertex: wp.int32):
    return (adjacency.v_adj_edges_offsets[vertex + 1] - adjacency.v_adj_edges_offsets[vertex]) >> 1


@wp.func
def get_vertex_adjacent_edge_id_order(adjacency: ForceElementAdjacencyInfo, vertex: wp.int32, edge: wp.int32):
    offset = adjacency.v_adj_edges_offsets[vertex]
    return adjacency.v_adj_edges[offset + edge * 2], adjacency.v_adj_edges[offset + edge * 2 + 1]


@wp.func
def get_vertex_num_adjacent_faces(adjacency: ForceElementAdjacencyInfo, vertex: wp.int32):
    return (adjacency.v_adj_faces_offsets[vertex + 1] - adjacency.v_adj_faces_offsets[vertex]) >> 1


@wp.func
def get_vertex_adjacent_face_id_order(adjacency: ForceElementAdjacencyInfo, vertex: wp.int32, face: wp.int32):
    offset = adjacency.v_adj_faces_offsets[vertex]
    return adjacency.v_adj_faces[offset + face * 2], adjacency.v_adj_faces[offset + face * 2 + 1]


@wp.kernel
def _test_compute_force_element_adjacency(
    adjacency: ForceElementAdjacencyInfo,
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    face_indices: wp.array(dtype=wp.int32, ndim=2),
):
    wp.printf("num vertices: %d\n", adjacency.v_adj_edges_offsets.shape[0] - 1)
    for vertex in range(adjacency.v_adj_edges_offsets.shape[0] - 1):
        num_adj_edges = get_vertex_num_adjacent_edges(adjacency, vertex)
        for i_bd in range(num_adj_edges):
            bd_id, v_order = get_vertex_adjacent_edge_id_order(adjacency, vertex, i_bd)

            if edge_indices[bd_id, v_order] != vertex:
                print("Error!!!")
                wp.printf("vertex: %d | num_adj_edges: %d\n", vertex, num_adj_edges)
                wp.printf("--iBd: %d | ", i_bd)
                wp.printf("edge id: %d | v_order: %d\n", bd_id, v_order)

        num_adj_faces = get_vertex_num_adjacent_faces(adjacency, vertex)

        for i_face in range(num_adj_faces):
            face, v_order = get_vertex_adjacent_face_id_order(
                adjacency,
                vertex,
                i_face,
            )

            if face_indices[face, v_order] != vertex:
                print("Error!!!")
                wp.printf("vertex: %d | num_adj_faces: %d\n", vertex, num_adj_faces)
                wp.printf("--i_face: %d | face id: %d | v_order: %d\n", i_face, face, v_order)
                wp.printf(
                    "--face: %d %d %d\n",
                    face_indices[face, 0],
                    face_indices[face, 1],
                    face_indices[face, 2],
                )


@wp.func
def build_orthonormal_basis(n: wp.vec3):
    """
    Builds an orthonormal basis given a normal vector `n`. Return the two axes that is perpendicular to `n`.

    :param n: A 3D vector (list or array-like) representing the normal vector
    """
    b1 = wp.vec3()
    b2 = wp.vec3()
    if n[2] < 0.0:
        a = 1.0 / (1.0 - n[2])
        b = n[0] * n[1] * a
        b1[0] = 1.0 - n[0] * n[0] * a
        b1[1] = -b
        b1[2] = n[0]

        b2[0] = b
        b2[1] = n[1] * n[1] * a - 1.0
        b2[2] = -n[1]
    else:
        a = 1.0 / (1.0 + n[2])
        b = -n[0] * n[1] * a
        b1[0] = 1.0 - n[0] * n[0] * a
        b1[1] = b
        b1[2] = -n[0]

        b2[0] = b
        b2[1] = 1.0 - n[1] * n[1] * a
        b2[2] = -n[1]

    return b1, b2


@wp.func
def calculate_triangle_deformation_gradient(
    face: int, tri_indices: wp.array(dtype=wp.int32, ndim=2), pos: wp.array(dtype=wp.vec3), tri_pose: wp.mat22
):
    F = mat32()
    v1 = pos[tri_indices[face, 1]] - pos[tri_indices[face, 0]]
    v2 = pos[tri_indices[face, 2]] - pos[tri_indices[face, 0]]

    F[0, 0] = v1[0]
    F[1, 0] = v1[1]
    F[2, 0] = v1[2]
    F[0, 1] = v2[0]
    F[1, 1] = v2[1]
    F[2, 1] = v2[2]

    F = F * tri_pose
    return F


@wp.func
def green_strain(F: mat32):
    return 0.5 * (wp.transpose(F) * F - wp.identity(n=2, dtype=float))


@wp.func
def assemble_membrane_hessian(h: mat66, m1: float, m2: float):
    h_vert = wp.mat33(
        m1 * (h[0, 0] * m1 + h[3, 0] * m2) + m2 * (h[0, 3] * m1 + h[3, 3] * m2),
        m1 * (h[0, 1] * m1 + h[3, 1] * m2) + m2 * (h[0, 4] * m1 + h[3, 4] * m2),
        m1 * (h[0, 2] * m1 + h[3, 2] * m2) + m2 * (h[0, 5] * m1 + h[3, 5] * m2),
        m1 * (h[1, 0] * m1 + h[4, 0] * m2) + m2 * (h[1, 3] * m1 + h[4, 3] * m2),
        m1 * (h[1, 1] * m1 + h[4, 1] * m2) + m2 * (h[1, 4] * m1 + h[4, 4] * m2),
        m1 * (h[1, 2] * m1 + h[4, 2] * m2) + m2 * (h[1, 5] * m1 + h[4, 5] * m2),
        m1 * (h[2, 0] * m1 + h[5, 0] * m2) + m2 * (h[2, 3] * m1 + h[5, 3] * m2),
        m1 * (h[2, 1] * m1 + h[5, 1] * m2) + m2 * (h[2, 4] * m1 + h[5, 4] * m2),
        m1 * (h[2, 2] * m1 + h[5, 2] * m2) + m2 * (h[2, 5] * m1 + h[5, 5] * m2),
    )

    return h_vert


@wp.func
def evaluate_stvk_force_hessian(
    face: int,
    v_order: int,
    pos: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    tri_pose: wp.mat22,
    area: float,
    mu: float,
    lmbd: float,
    damping: float,
):
    D2W_DFDF = mat66()
    F = calculate_triangle_deformation_gradient(face, tri_indices, pos, tri_pose)
    G = green_strain(F)

    S = 2.0 * mu * G + lmbd * (G[0, 0] + G[1, 1]) * wp.identity(n=2, dtype=float)

    F12 = -area * F * S * wp.transpose(tri_pose)

    Dm_inv1_1 = tri_pose[0, 0]
    Dm_inv2_1 = tri_pose[1, 0]
    Dm_inv1_2 = tri_pose[0, 1]
    Dm_inv2_2 = tri_pose[1, 1]

    F1_1 = F[0, 0]
    F2_1 = F[1, 0]
    F3_1 = F[2, 0]
    F1_2 = F[0, 1]
    F2_2 = F[1, 1]
    F3_2 = F[2, 1]

    F1_1_sqr = F1_1 * F1_1
    F2_1_sqr = F2_1 * F2_1
    F3_1_sqr = F3_1 * F3_1
    F1_2_sqr = F1_2 * F1_2
    F2_2_sqr = F2_2 * F2_2
    F3_2_sqr = F3_2 * F3_2

    e_uu = G[0, 0]
    e_vv = G[1, 1]
    e_uv = G[0, 1]
    e_uuvvSum = e_uu + e_vv

    D2W_DFDF[0, 0] = F1_1 * (F1_1 * lmbd + 2.0 * F1_1 * mu) + 2.0 * mu * e_uu + lmbd * (e_uuvvSum) + F1_2_sqr * mu

    D2W_DFDF[1, 0] = F1_1 * (F2_1 * lmbd + 2.0 * F2_1 * mu) + F1_2 * F2_2 * mu
    D2W_DFDF[0, 1] = D2W_DFDF[1, 0]

    D2W_DFDF[2, 0] = F1_1 * (F3_1 * lmbd + 2.0 * F3_1 * mu) + F1_2 * F3_2 * mu
    D2W_DFDF[0, 2] = D2W_DFDF[2, 0]

    D2W_DFDF[3, 0] = 2.0 * mu * e_uv + F1_1 * F1_2 * lmbd + F1_1 * F1_2 * mu
    D2W_DFDF[0, 3] = D2W_DFDF[3, 0]

    D2W_DFDF[4, 0] = F1_1 * F2_2 * lmbd + F1_2 * F2_1 * mu
    D2W_DFDF[0, 4] = D2W_DFDF[4, 0]

    D2W_DFDF[5, 0] = F1_1 * F3_2 * lmbd + F1_2 * F3_1 * mu
    D2W_DFDF[0, 5] = D2W_DFDF[5, 0]

    D2W_DFDF[1, 1] = F2_1 * (F2_1 * lmbd + 2.0 * F2_1 * mu) + 2.0 * mu * e_uu + lmbd * (e_uuvvSum) + F2_2_sqr * mu

    D2W_DFDF[2, 1] = F2_1 * (F3_1 * lmbd + 2.0 * F3_1 * mu) + F2_2 * F3_2 * mu
    D2W_DFDF[1, 2] = D2W_DFDF[2, 1]

    D2W_DFDF[3, 1] = F1_2 * F2_1 * lmbd + F1_1 * F2_2 * mu
    D2W_DFDF[1, 3] = D2W_DFDF[3, 1]

    D2W_DFDF[4, 1] = 2.0 * mu * e_uv + F2_1 * F2_2 * lmbd + F2_1 * F2_2 * mu
    D2W_DFDF[1, 4] = D2W_DFDF[4, 1]

    D2W_DFDF[5, 1] = F2_1 * F3_2 * lmbd + F2_2 * F3_1 * mu
    D2W_DFDF[1, 5] = D2W_DFDF[5, 1]

    D2W_DFDF[2, 2] = F3_1 * (F3_1 * lmbd + 2.0 * F3_1 * mu) + 2.0 * mu * e_uu + lmbd * (e_uuvvSum) + F3_2_sqr * mu

    D2W_DFDF[3, 2] = F1_2 * F3_1 * lmbd + F1_1 * F3_2 * mu
    D2W_DFDF[2, 3] = D2W_DFDF[3, 2]

    D2W_DFDF[4, 2] = F2_2 * F3_1 * lmbd + F2_1 * F3_2 * mu
    D2W_DFDF[2, 4] = D2W_DFDF[4, 2]

    D2W_DFDF[5, 2] = 2.0 * mu * e_uv + F3_1 * F3_2 * lmbd + F3_1 * F3_2 * mu
    D2W_DFDF[2, 5] = D2W_DFDF[5, 2]

    D2W_DFDF[3, 3] = F1_2 * (F1_2 * lmbd + 2.0 * F1_2 * mu) + 2.0 * mu * e_vv + lmbd * (e_uuvvSum) + F1_1_sqr * mu

    D2W_DFDF[4, 3] = F1_2 * (F2_2 * lmbd + 2.0 * F2_2 * mu) + F1_1 * F2_1 * mu
    D2W_DFDF[3, 4] = D2W_DFDF[4, 3]

    D2W_DFDF[5, 3] = F1_2 * (F3_2 * lmbd + 2.0 * F3_2 * mu) + F1_1 * F3_1 * mu
    D2W_DFDF[3, 5] = D2W_DFDF[5, 3]

    D2W_DFDF[4, 4] = F2_2 * (F2_2 * lmbd + 2.0 * F2_2 * mu) + 2.0 * mu * e_vv + lmbd * (e_uuvvSum) + F2_1_sqr * mu

    D2W_DFDF[5, 4] = F2_2 * (F3_2 * lmbd + 2.0 * F3_2 * mu) + F2_1 * F3_1 * mu
    D2W_DFDF[4, 5] = D2W_DFDF[5, 4]

    D2W_DFDF[5, 5] = F3_2 * (F3_2 * lmbd + 2.0 * F3_2 * mu) + 2.0 * mu * e_vv + lmbd * (e_uuvvSum) + F3_1_sqr * mu

    D2W_DFDF = D2W_DFDF * area

    # m1s = wp.vec3(-Dm_inv1_1 - Dm_inv2_1, Dm_inv1_1, Dm_inv2_1)
    # m2s = wp.vec3(-Dm_inv1_2 - Dm_inv2_2, Dm_inv1_2, Dm_inv2_2)
    #
    # m1 = m1s[v_order]
    # m2 = m2s[v_order]

    if v_order == 0:
        m1 = -Dm_inv1_1 - Dm_inv2_1
        m2 = -Dm_inv1_2 - Dm_inv2_2
        f = wp.vec3(-F12[0, 0] - F12[0, 1], -F12[1, 0] - F12[1, 1], -F12[2, 0] - F12[2, 1])
    elif v_order == 1:
        m1 = Dm_inv1_1
        m2 = Dm_inv1_2
        f = wp.vec3(F12[0, 0], F12[1, 0], F12[2, 0])
    else:
        m1 = Dm_inv2_1
        m2 = Dm_inv2_2
        f = wp.vec3(F12[0, 1], F12[1, 1], F12[2, 1])

    h = assemble_membrane_hessian(D2W_DFDF, m1, m2)

    return f, h


@wp.func
def mat_vec_cross_from_3_basis(e1: wp.vec3, e2: wp.vec3, e3: wp.vec3, a: wp.vec3):
    e1_cross_a = wp.cross(e1, a)
    e2_cross_a = wp.cross(e2, a)
    e3_cross_a = wp.cross(e3, a)

    return wp.mat33(
        e1_cross_a[0],
        e2_cross_a[0],
        e3_cross_a[0],
        e1_cross_a[1],
        e2_cross_a[1],
        e3_cross_a[1],
        e1_cross_a[2],
        e2_cross_a[2],
        e3_cross_a[2],
    )


@wp.func
def mat_vec_cross(mat: wp.mat33, a: wp.vec3):
    e1 = wp.vec3(mat[0, 0], mat[1, 0], mat[2, 0])
    e2 = wp.vec3(mat[0, 1], mat[1, 1], mat[2, 1])
    e3 = wp.vec3(mat[0, 2], mat[1, 2], mat[2, 2])

    return mat_vec_cross_from_3_basis(e1, e2, e3, a)


@wp.func
def evaluate_dihedral_angle_based_bending_force_hessian(
    bending_index: int,
    v_order: int,
    pos: wp.array(dtype=wp.vec3),
    pos_prev: wp.array(dtype=wp.vec3),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    edge_rest_angle: wp.array(dtype=float),
    edge_rest_length: wp.array(dtype=float),
    stiffness: float,
    damping: float,
    dt: float,
):
    if edge_indices[bending_index, 0] == -1 or edge_indices[bending_index, 1] == -1:
        return wp.vec3(0.0), wp.mat33(0.0)

    x1 = pos[edge_indices[bending_index, 0]]
    x2 = pos[edge_indices[bending_index, 2]]
    x3 = pos[edge_indices[bending_index, 3]]
    x4 = pos[edge_indices[bending_index, 1]]

    e1 = wp.vec3(1.0, 0.0, 0.0)
    e2 = wp.vec3(0.0, 1.0, 0.0)
    e3 = wp.vec3(0.0, 0.0, 1.0)

    n1 = wp.cross((x2 - x1), (x3 - x1))
    n2 = wp.cross((x3 - x4), (x2 - x4))
    n1_norm = wp.length(n1)
    n2_norm = wp.length(n2)

    # degenerated bending edge
    if n1_norm < 1.0e-6 or n2_norm < 1.0e-6:
        return wp.vec3(0.0), wp.mat33(0.0)

    n1_n = n1 / n1_norm
    n2_n = n2 / n2_norm

    # avoid the infinite gradient of acos at -1 or 1
    cos_theta = wp.dot(n1_n, n2_n)
    if wp.abs(cos_theta) > 0.9999:
        cos_theta = 0.9999 * wp.sign(cos_theta)

    angle_sign = wp.sign(wp.dot(wp.cross(n2, n1), x3 - x2))
    theta = wp.acos(cos_theta) * angle_sign
    rest_angle = edge_rest_angle[bending_index]

    dE_dtheta = stiffness * (theta - rest_angle)

    d_theta_d_cos_theta = angle_sign * (-1.0 / wp.sqrt(1.0 - cos_theta * cos_theta))
    sin_theta = angle_sign * wp.sqrt(1.0 - cos_theta * cos_theta)
    one_over_sin_theta = 1.0 / sin_theta
    d_one_over_sin_theta_d_cos_theta = cos_theta / (sin_theta * sin_theta * sin_theta)

    e_rest_len = edge_rest_length[bending_index]

    if v_order == 0:
        d_cos_theta_dx1 = 1.0 / n1_norm * (-wp.cross(x3 - x1, n2_n) + wp.cross(x2 - x1, n2_n))
        d_one_over_sin_theta_dx1 = d_cos_theta_dx1 * d_one_over_sin_theta_d_cos_theta

        d_theta_dx1 = d_theta_d_cos_theta * d_cos_theta_dx1
        d2_theta_dx1_dx1 = -wp.outer(d_one_over_sin_theta_dx1, d_cos_theta_dx1)

        dE_dx1 = e_rest_len * dE_dtheta * d_theta_d_cos_theta * d_cos_theta_dx1

        d2_E_dx1_dx1 = (
            e_rest_len * stiffness * (wp.outer(d_theta_dx1, d_theta_dx1) + (theta - rest_angle) * d2_theta_dx1_dx1)
        )

        bending_force = -dE_dx1
        bending_hessian = d2_E_dx1_dx1
    elif v_order == 1:
        d_cos_theta_dx4 = 1.0 / n2_norm * (-wp.cross(x2 - x4, n1_n) + wp.cross(x3 - x4, n1_n))
        d_one_over_sin_theta_dx4 = d_cos_theta_dx4 * d_one_over_sin_theta_d_cos_theta

        d_theta_dx4 = d_theta_d_cos_theta * d_cos_theta_dx4
        d2_theta_dx4_dx4 = -wp.outer(d_one_over_sin_theta_dx4, d_cos_theta_dx4)

        dE_dx4 = e_rest_len * dE_dtheta * d_theta_d_cos_theta * d_cos_theta_dx4
        d2_E_dx4_dx4 = (
            e_rest_len * stiffness * (wp.outer(d_theta_dx4, d_theta_dx4) + (theta - rest_angle) * (d2_theta_dx4_dx4))
        )

        bending_force = -dE_dx4
        bending_hessian = d2_E_dx4_dx4
    elif v_order == 2:
        d_cos_theta_dx2 = 1.0 / n1_norm * wp.cross(x3 - x1, n2_n) - 1.0 / n2_norm * wp.cross(x3 - x4, n1_n)
        dn1_dx2 = mat_vec_cross_from_3_basis(e1, e2, e3, x3 - x1)
        dn2_dx2 = -mat_vec_cross_from_3_basis(e1, e2, e3, x3 - x4)
        d_one_over_sin_theta_dx2 = d_cos_theta_dx2 * d_one_over_sin_theta_d_cos_theta
        d2_cos_theta_dx2_dx2 = -mat_vec_cross(dn2_dx2, (x3 - x1)) / (n1_norm * n2_norm) + mat_vec_cross(
            dn1_dx2, x3 - x4
        ) / (n1_norm * n2_norm)

        d_theta_dx2 = d_theta_d_cos_theta * d_cos_theta_dx2
        d2_theta_dx2_dx2 = (
            -wp.outer(d_one_over_sin_theta_dx2, d_cos_theta_dx2) - one_over_sin_theta * d2_cos_theta_dx2_dx2
        )

        dE_dx2 = e_rest_len * dE_dtheta * d_theta_d_cos_theta * d_cos_theta_dx2
        d2_E_dx2_dx2 = (
            e_rest_len * stiffness * (wp.outer(d_theta_dx2, d_theta_dx2) + (theta - rest_angle) * d2_theta_dx2_dx2)
        )

        bending_force = -dE_dx2
        bending_hessian = d2_E_dx2_dx2
    else:
        d_cos_theta_dx3 = -1.0 / n1_norm * wp.cross(x2 - x1, n2_n) + 1.0 / n2_norm * wp.cross(x2 - x4, n1_n)
        dn1_dx3 = -mat_vec_cross_from_3_basis(e1, e2, e3, x2 - x1)
        dn2_dx3 = mat_vec_cross_from_3_basis(e1, e2, e3, x2 - x4)
        d_one_over_sin_theta_dx3 = d_cos_theta_dx3 * d_one_over_sin_theta_d_cos_theta
        d2_cos_theta_dx3_dx3 = mat_vec_cross(dn2_dx3, (x2 - x1)) / (n1_norm * n2_norm) - mat_vec_cross(
            dn1_dx3, x2 - x4
        ) / (n1_norm * n2_norm)

        d_theta_dx3 = d_theta_d_cos_theta * d_cos_theta_dx3
        d2_theta_dx3_dx3 = (
            -wp.outer(d_one_over_sin_theta_dx3, d_cos_theta_dx3) - one_over_sin_theta * d2_cos_theta_dx3_dx3
        )

        dE_dx3 = e_rest_len * dE_dtheta * d_theta_d_cos_theta * d_cos_theta_dx3

        d2_E_dx3_dx3 = (
            e_rest_len * stiffness * (wp.outer(d_theta_dx3, d_theta_dx3) + (theta - rest_angle) * d2_theta_dx3_dx3)
        )

        bending_force = -dE_dx3
        bending_hessian = d2_E_dx3_dx3

    displacement = pos_prev[edge_indices[bending_index, v_order]] - pos[edge_indices[bending_index, v_order]]
    h_d = bending_hessian * (damping / dt)
    f_d = h_d * displacement

    bending_force = bending_force + f_d
    bending_hessian = bending_hessian + h_d

    return bending_force, bending_hessian


@wp.func
def evaluate_ground_contact_force_hessian(
    particle_pos: wp.vec3,
    particle_prev_pos: wp.vec3,
    particle_radius: float,
    ground_normal: wp.vec3,
    ground_level: float,
    soft_contact_ke: float,
    soft_contact_kd: float,
    friction_mu: float,
    friction_epsilon: float,
    dt: float,
):
    penetration_depth = -(wp.dot(ground_normal, particle_pos) + ground_level - particle_radius)

    if penetration_depth > 0:
        ground_contact_force_norm = penetration_depth * soft_contact_ke
        ground_contact_force = ground_normal * ground_contact_force_norm
        ground_contact_hessian = soft_contact_ke * wp.outer(ground_normal, ground_normal)

        dx = particle_pos - particle_prev_pos

        if wp.dot(dx, ground_normal) < 0:
            damping_hessian = (soft_contact_kd / dt) * ground_contact_hessian
            ground_contact_hessian = ground_contact_hessian + damping_hessian
            ground_contact_force = ground_contact_force - damping_hessian * dx

        # friction
        e0, e1 = build_orthonormal_basis(ground_normal)

        T = mat32(e0[0], e1[0], e0[1], e1[1], e0[2], e1[2])

        relative_translation = dx
        u = wp.transpose(T) * relative_translation
        eps_u = friction_epsilon * dt

        friction_force, friction_hessian = compute_friction(friction_mu, ground_contact_force_norm, T, u, eps_u)
        ground_contact_force = ground_contact_force + friction_force
        ground_contact_hessian = ground_contact_hessian + friction_hessian
    else:
        ground_contact_force = wp.vec3(0.0, 0.0, 0.0)
        ground_contact_hessian = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    return ground_contact_force, ground_contact_hessian


@wp.func
def evaluate_body_particle_contact(
    particle_index: int,
    particle_pos: wp.vec3,
    particle_prev_pos: wp.vec3,
    contact_index: int,
    soft_contact_ke: float,
    soft_contact_kd: float,
    friction_mu: float,
    friction_epsilon: float,
    particle_radius: wp.array(dtype=float),
    shape_materials: ModelShapeMaterials,
    shape_body: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    contact_shape: wp.array(dtype=int),
    contact_body_pos: wp.array(dtype=wp.vec3),
    contact_body_vel: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    dt: float,
):
    shape_index = contact_shape[contact_index]
    body_index = shape_body[shape_index]

    X_wb = wp.transform_identity()
    X_com = wp.vec3()
    if body_index >= 0:
        X_wb = body_q[body_index]
        X_com = body_com[body_index]

    # body position in world space
    bx = wp.transform_point(X_wb, contact_body_pos[contact_index])
    r = bx - wp.transform_point(X_wb, X_com)

    n = contact_normal[contact_index]

    penetration_depth = -(wp.dot(n, particle_pos - bx) - particle_radius[particle_index])
    if penetration_depth > 0:
        body_contact_force_norm = penetration_depth * soft_contact_ke
        body_contact_force = n * body_contact_force_norm
        body_contact_hessian = soft_contact_ke * wp.outer(n, n)

        mu = shape_materials.mu[shape_index]

        dx = particle_pos - particle_prev_pos

        if wp.dot(n, dx) < 0:
            damping_hessian = (soft_contact_kd / dt) * body_contact_hessian
            body_contact_hessian = body_contact_hessian + damping_hessian
            body_contact_force = body_contact_force - damping_hessian * dx

        # body velocity
        body_v_s = wp.spatial_vector()
        if body_index >= 0:
            body_v_s = body_qd[body_index]

        body_w = wp.spatial_top(body_v_s)
        body_v = wp.spatial_bottom(body_v_s)

        # compute the body velocity at the particle position
        bv = body_v + wp.cross(body_w, r) + wp.transform_vector(X_wb, contact_body_vel[contact_index])

        relative_translation = dx - bv * dt

        # friction
        e0, e1 = build_orthonormal_basis(n)

        T = mat32(e0[0], e1[0], e0[1], e1[1], e0[2], e1[2])

        u = wp.transpose(T) * relative_translation
        eps_u = friction_epsilon * dt

        friction_force, friction_hessian = compute_friction(mu, body_contact_force_norm, T, u, eps_u)
        body_contact_force = body_contact_force + friction_force
        body_contact_hessian = body_contact_hessian + friction_hessian
    else:
        body_contact_force = wp.vec3(0.0, 0.0, 0.0)
        body_contact_hessian = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    return body_contact_force, body_contact_hessian


@wp.func
def evaluate_self_contact_force_norm(dis: float, collision_radius: float, k: float):
    # Adjust distance and calculate penetration depth

    penetration_depth = collision_radius - dis

    # Initialize outputs
    dEdD = wp.float32(0.0)
    d2E_dDdD = wp.float32(0.0)

    # C2 continuity calculation
    tau = collision_radius * 0.5
    if tau > dis > 1e-5:
        k2 = 0.5 * tau * tau * k
        dEdD = -k2 / dis
        d2E_dDdD = k2 / (dis * dis)
    else:
        dEdD = -k * penetration_depth
        d2E_dDdD = k

    return dEdD, d2E_dDdD


@wp.func
def evaluate_edge_edge_contact(
    v: int,
    v_order: int,
    e1: int,
    e2: int,
    pos: wp.array(dtype=wp.vec3),
    pos_prev: wp.array(dtype=wp.vec3),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    collision_radius: float,
    collision_stiffness: float,
    collision_damping: float,
    friction_coefficient: float,
    friction_epsilon: float,
    dt: float,
    edge_edge_parallel_epsilon: float,
):
    r"""
    Returns the edge-edge contact force and hessian, including the friction force.
    Args:
        v:
        v_order: \in {0, 1, 2, 3}, 0, 1 is vertex 0, 1 of e1, 2,3 is vertex 0, 1 of e2
        e0
        e1
        pos
        edge_indices
        collision_radius
        collision_stiffness
        dt
    """
    e1_v1 = edge_indices[e1, 2]
    e1_v2 = edge_indices[e1, 3]

    e1_v1_pos = pos[e1_v1]
    e1_v2_pos = pos[e1_v2]

    e2_v1 = edge_indices[e2, 2]
    e2_v2 = edge_indices[e2, 3]

    e2_v1_pos = pos[e2_v1]
    e2_v2_pos = pos[e2_v2]

    st = wp.closest_point_edge_edge(e1_v1_pos, e1_v2_pos, e2_v1_pos, e2_v2_pos, edge_edge_parallel_epsilon)
    s = st[0]
    t = st[1]
    e1_vec = e1_v2_pos - e1_v1_pos
    e2_vec = e2_v2_pos - e2_v1_pos
    c1 = e1_v1_pos + e1_vec * s
    c2 = e2_v1_pos + e2_vec * t

    # c1, c2, s, t = closest_point_edge_edge_2(e1_v1_pos, e1_v2_pos, e2_v1_pos, e2_v2_pos)

    diff = c1 - c2
    dis = st[2]
    collision_normal = diff / dis

    if dis < collision_radius:
        bs = wp.vec4(1.0 - s, s, -1.0 + t, -t)
        v_bary = bs[v_order]

        dEdD, d2E_dDdD = evaluate_self_contact_force_norm(dis, collision_radius, collision_stiffness)

        collision_force = -dEdD * v_bary * collision_normal
        collision_hessian = d2E_dDdD * v_bary * v_bary * wp.outer(collision_normal, collision_normal)

        # friction
        c1_prev = pos_prev[e1_v1] + (pos_prev[e1_v2] - pos_prev[e1_v1]) * s
        c2_prev = pos_prev[e2_v1] + (pos_prev[e2_v2] - pos_prev[e2_v1]) * t

        dx = (c1 - c1_prev) - (c2 - c2_prev)
        axis_1, axis_2 = build_orthonormal_basis(collision_normal)

        T = mat32(
            axis_1[0],
            axis_2[0],
            axis_1[1],
            axis_2[1],
            axis_1[2],
            axis_2[2],
        )

        u = wp.transpose(T) * dx
        eps_U = friction_epsilon * dt

        # fmt: off
        if wp.static("contact_force_hessian_ee" in VBD_DEBUG_PRINTING_OPTIONS):
            wp.printf(
                "    collision force:\n    %f %f %f,\n    collision hessian:\n    %f %f %f,\n    %f %f %f,\n    %f %f %f\n",
                collision_force[0], collision_force[1], collision_force[2], collision_hessian[0, 0], collision_hessian[0, 1], collision_hessian[0, 2], collision_hessian[1, 0], collision_hessian[1, 1], collision_hessian[1, 2], collision_hessian[2, 0], collision_hessian[2, 1], collision_hessian[2, 2],
            )
        # fmt: on

        friction_force, friction_hessian = compute_friction(friction_coefficient, -dEdD, T, u, eps_U)
        friction_force = friction_force * v_bary
        friction_hessian = friction_hessian * v_bary * v_bary

        # # fmt: off
        # if wp.static("contact_force_hessian_ee" in VBD_DEBUG_PRINTING_OPTIONS):
        #     wp.printf(
        #         "    friction force:\n    %f %f %f,\n    friction hessian:\n    %f %f %f,\n    %f %f %f,\n    %f %f %f\n",
        #         friction_force[0], friction_force[1], friction_force[2], friction_hessian[0, 0], friction_hessian[0, 1], friction_hessian[0, 2], friction_hessian[1, 0], friction_hessian[1, 1], friction_hessian[1, 2], friction_hessian[2, 0], friction_hessian[2, 1], friction_hessian[2, 2],
        #     )
        # # fmt: on

        if v_order == 0:
            displacement = pos_prev[e1_v1] - e1_v1_pos
        elif v_order == 1:
            displacement = pos_prev[e1_v2] - e1_v2_pos
        elif v_order == 2:
            displacement = pos_prev[e2_v1] - e2_v1_pos
        else:
            displacement = pos_prev[e2_v2] - e2_v2_pos

        collision_normal_normal_sign = wp.vec4(1.0, 1.0, -1.0, -1.0)
        if wp.dot(displacement, collision_normal * collision_normal_normal_sign[v_order]) > 0:
            damping_hessian = (collision_damping / dt) * collision_hessian
            collision_hessian = collision_hessian + damping_hessian
            collision_force = collision_force + damping_hessian * displacement

        collision_force = collision_force + friction_force
        collision_hessian = collision_hessian + friction_hessian
    else:
        collision_force = wp.vec3(0.0, 0.0, 0.0)
        collision_hessian = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    return collision_force, collision_hessian


@wp.func
def evaluate_vertex_triangle_collision_force_hessian(
    v: int,
    v_order: int,
    tri: int,
    pos: wp.array(dtype=wp.vec3),
    pos_prev: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    collision_radius: float,
    collision_stiffness: float,
    collision_damping: float,
    friction_coefficient: float,
    friction_epsilon: float,
    dt: float,
):
    a = pos[tri_indices[tri, 0]]
    b = pos[tri_indices[tri, 1]]
    c = pos[tri_indices[tri, 2]]

    p = pos[v]

    closest_p, bary, feature_type = triangle_closest_point(a, b, c, p)

    diff = p - closest_p
    dis = wp.length(diff)
    collision_normal = diff / dis

    if dis < collision_radius:
        bs = wp.vec4(-bary[0], -bary[1], -bary[2], 1.0)
        v_bary = bs[v_order]

        dEdD, d2E_dDdD = evaluate_self_contact_force_norm(dis, collision_radius, collision_stiffness)

        collision_force = -dEdD * v_bary * collision_normal
        collision_hessian = d2E_dDdD * v_bary * v_bary * wp.outer(collision_normal, collision_normal)

        # friction force
        dx_v = p - pos_prev[v]

        closest_p_prev = (
            bary[0] * pos_prev[tri_indices[tri, 0]]
            + bary[1] * pos_prev[tri_indices[tri, 1]]
            + bary[2] * pos_prev[tri_indices[tri, 2]]
        )

        dx = dx_v - (closest_p - closest_p_prev)

        e0, e1 = build_orthonormal_basis(collision_normal)

        T = mat32(e0[0], e1[0], e0[1], e1[1], e0[2], e1[2])

        u = wp.transpose(T) * dx
        eps_U = friction_epsilon * dt

        friction_force, friction_hessian = compute_friction(friction_coefficient, -dEdD, T, u, eps_U)

        # fmt: off
        if wp.static("contact_force_hessian_vt" in VBD_DEBUG_PRINTING_OPTIONS):
            wp.printf(
                "v: %d dEdD: %f\nnormal force: %f %f %f\nfriction force: %f %f %f\n",
                v,
                dEdD,
                collision_force[0], collision_force[1], collision_force[2], friction_force[0], friction_force[1], friction_force[2],
            )
        # fmt: on

        if v_order == 0:
            displacement = pos_prev[tri_indices[tri, 0]] - a
        elif v_order == 1:
            displacement = pos_prev[tri_indices[tri, 1]] - b
        elif v_order == 2:
            displacement = pos_prev[tri_indices[tri, 2]] - c
        else:
            displacement = pos_prev[v] - p

        collision_normal_normal_sign = wp.vec4(-1.0, -1.0, -1.0, 1.0)
        if wp.dot(displacement, collision_normal * collision_normal_normal_sign[v_order]) > 0:
            damping_hessian = (collision_damping / dt) * collision_hessian
            collision_hessian = collision_hessian + damping_hessian
            collision_force = collision_force + damping_hessian * displacement

        collision_force = collision_force + v_bary * friction_force
        collision_hessian = collision_hessian + v_bary * v_bary * friction_hessian
    else:
        collision_force = wp.vec3(0.0, 0.0, 0.0)
        collision_hessian = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    return collision_force, collision_hessian


@wp.func
def compute_friction(mu: float, normal_contact_force: float, T: mat32, u: wp.vec2, eps_u: float):
    """
    Returns the 1D friction force and hessian.
    Args:
        mu: Friction coefficient.
        normal_contact_force: normal contact force.
        T: Transformation matrix (3x2 matrix).
        u: 2D displacement vector.
    """
    # Friction
    u_norm = wp.length(u)

    if u_norm > 0.0:
        # IPC friction
        if u_norm > eps_u:
            # constant stage
            f1_SF_over_x = 1.0 / u_norm
        else:
            # smooth transition
            f1_SF_over_x = (-u_norm / eps_u + 2.0) / eps_u

        force = -mu * normal_contact_force * T * (f1_SF_over_x * u)

        # Different from IPC, we treat the contact normal as constant
        # this significantly improves the stability
        hessian = mu * normal_contact_force * T * (f1_SF_over_x * wp.identity(2, float)) * wp.transpose(T)
    else:
        force = wp.vec3(0.0, 0.0, 0.0)
        hessian = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    return force, hessian


@wp.kernel
def forward_step(
    dt: float,
    gravity: wp.vec3,
    prev_pos: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    external_force: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.uint32),
    inertia: wp.array(dtype=wp.vec3),
):
    particle = wp.tid()

    prev_pos[particle] = pos[particle]
    if not particle_flags[particle] & PARTICLE_FLAG_ACTIVE:
        inertia[particle] = prev_pos[particle]
        return
    vel_new = vel[particle] + (gravity + external_force[particle] * inv_mass[particle]) * dt
    pos[particle] = pos[particle] + vel_new * dt
    inertia[particle] = pos[particle]


@wp.kernel
def forward_step_penetration_free(
    dt: float,
    gravity: wp.vec3,
    prev_pos: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    inv_mass: wp.array(dtype=float),
    external_force: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.uint32),
    pos_prev_collision_detection: wp.array(dtype=wp.vec3),
    particle_conservative_bounds: wp.array(dtype=float),
    inertia: wp.array(dtype=wp.vec3),
):
    particle_index = wp.tid()

    prev_pos[particle_index] = pos[particle_index]
    if not particle_flags[particle_index] & PARTICLE_FLAG_ACTIVE:
        inertia[particle_index] = prev_pos[particle_index]
        return
    vel_new = vel[particle_index] + (gravity + external_force[particle_index] * inv_mass[particle_index]) * dt
    pos_inertia = pos[particle_index] + vel_new * dt
    inertia[particle_index] = pos_inertia

    pos[particle_index] = apply_conservative_bound_truncation(
        particle_index, pos_inertia, pos_prev_collision_detection, particle_conservative_bounds
    )


@wp.kernel
def compute_particle_conservative_bound(
    # inputs
    conservative_bound_relaxation: float,
    collision_query_radius: float,
    adjacency: ForceElementAdjacencyInfo,
    collision_info: TriMeshCollisionInfo,
    # outputs
    particle_conservative_bounds: wp.array(dtype=float),
):
    particle_index = wp.tid()
    min_dist = wp.min(collision_query_radius, collision_info.vertex_colliding_triangles_min_dist[particle_index])

    # bound from neighbor triangles
    for i_adj_tri in range(
        get_vertex_num_adjacent_faces(
            adjacency,
            particle_index,
        )
    ):
        tri_index, vertex_order = get_vertex_adjacent_face_id_order(
            adjacency,
            particle_index,
            i_adj_tri,
        )
        min_dist = wp.min(min_dist, collision_info.triangle_colliding_vertices_min_dist[tri_index])

    # bound from neighbor edges
    for i_adj_edge in range(
        get_vertex_num_adjacent_edges(
            adjacency,
            particle_index,
        )
    ):
        nei_edge_index, vertex_order_on_edge = get_vertex_adjacent_edge_id_order(
            adjacency,
            particle_index,
            i_adj_edge,
        )
        # vertex is on the edge; otherwise it only effects the bending energy
        if vertex_order_on_edge == 2 or vertex_order_on_edge == 3:
            # collisions of neighbor edges
            min_dist = wp.min(min_dist, collision_info.edge_colliding_edges_min_dist[nei_edge_index])

    particle_conservative_bounds[particle_index] = conservative_bound_relaxation * min_dist


@wp.kernel
def validate_conservative_bound(
    pos: wp.array(dtype=wp.vec3),
    pos_prev_collision_detection: wp.array(dtype=wp.vec3),
    particle_conservative_bounds: wp.array(dtype=float),
):
    v_index = wp.tid()

    displacement = wp.length(pos[v_index] - pos_prev_collision_detection[v_index])

    if displacement > particle_conservative_bounds[v_index] * 1.01 and displacement > 1e-5:
        # wp.expect_eq(displacement <= particle_conservative_bounds[v_index] * 1.01, True)
        wp.printf(
            "Vertex %d has moved by %f exceeded the limit of %f\n",
            v_index,
            displacement,
            particle_conservative_bounds[v_index],
        )


@wp.func
def apply_conservative_bound_truncation(
    v_index: wp.int32,
    pos_new: wp.vec3,
    pos_prev_collision_detection: wp.array(dtype=wp.vec3),
    particle_conservative_bounds: wp.array(dtype=float),
):
    particle_pos_prev_collision_detection = pos_prev_collision_detection[v_index]
    accumulated_displacement = pos_new - particle_pos_prev_collision_detection
    conservative_bound = particle_conservative_bounds[v_index]

    accumulated_displacement_norm = wp.length(accumulated_displacement)
    if accumulated_displacement_norm > conservative_bound and conservative_bound > 1e-5:
        accumulated_displacement_norm_truncated = conservative_bound
        accumulated_displacement = accumulated_displacement * (
            accumulated_displacement_norm_truncated / accumulated_displacement_norm
        )

        return particle_pos_prev_collision_detection + accumulated_displacement
    else:
        return pos_new


@wp.kernel
def VBD_solve_trimesh_no_self_contact(
    dt: float,
    particle_ids_in_color: wp.array(dtype=wp.int32),
    prev_pos: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    pos_new: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    inertia: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.uint32),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    tri_poses: wp.array(dtype=wp.mat22),
    tri_materials: wp.array(dtype=float, ndim=2),
    tri_areas: wp.array(dtype=float),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    edge_rest_angles: wp.array(dtype=float),
    edge_rest_length: wp.array(dtype=float),
    edge_bending_properties: wp.array(dtype=float, ndim=2),
    adjacency: ForceElementAdjacencyInfo,
    # contact info
    #   self contact
    soft_contact_ke: float,
    soft_contact_kd: float,
    friction_mu: float,
    friction_epsilon: float,
    #   body-particle contact
    particle_radius: wp.array(dtype=float),
    body_particle_contact_buffer_pre_alloc: int,
    body_particle_contact_buffer: wp.array(dtype=int),
    body_particle_contact_count: wp.array(dtype=int),
    shape_materials: ModelShapeMaterials,
    shape_body: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    contact_shape: wp.array(dtype=int),
    contact_body_pos: wp.array(dtype=wp.vec3),
    contact_body_vel: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    # ground-particle contact
    has_ground: bool,
    ground: wp.array(dtype=float),
):
    tid = wp.tid()

    particle_index = particle_ids_in_color[tid]

    if not particle_flags[particle_index] & PARTICLE_FLAG_ACTIVE:
        return

    particle_pos = pos[particle_index]
    particle_prev_pos = pos[particle_index]

    dt_sqr_reciprocal = 1.0 / (dt * dt)

    # inertia force and hessian
    f = mass[particle_index] * (inertia[particle_index] - pos[particle_index]) * (dt_sqr_reciprocal)
    h = mass[particle_index] * dt_sqr_reciprocal * wp.identity(n=3, dtype=float)

    # elastic force and hessian
    for i_adj_tri in range(get_vertex_num_adjacent_faces(adjacency, particle_index)):
        tri_id, particle_order = get_vertex_adjacent_face_id_order(adjacency, particle_index, i_adj_tri)

        # fmt: off
        if wp.static("connectivity" in VBD_DEBUG_PRINTING_OPTIONS):
            wp.printf(
                "particle: %d | num_adj_faces: %d | ",
                particle_index,
                get_vertex_num_adjacent_faces(particle_index, adjacency),
            )
            wp.printf("i_face: %d | face id: %d | v_order: %d | ", i_adj_tri, tri_id, particle_order)
            wp.printf(
                "face: %d %d %d\n",
                tri_indices[tri_id, 0],
                tri_indices[tri_id, 1],
                tri_indices[tri_id, 2],
            )
        # fmt: on

        f_tri, h_tri = evaluate_stvk_force_hessian(
            tri_id,
            particle_order,
            pos,
            tri_indices,
            tri_poses[tri_id],
            tri_areas[tri_id],
            tri_materials[tri_id, 0],
            tri_materials[tri_id, 1],
            tri_materials[tri_id, 2],
        )
        # compute damping
        k_d = tri_materials[tri_id, 2]
        h_d = h_tri * (k_d / dt)

        f_d = h_d * (prev_pos[particle_index] - pos[particle_index])

        f = f + f_tri + f_d
        h = h + h_tri + h_d

        # fmt: off
        if wp.static("elasticity_force_hessian" in VBD_DEBUG_PRINTING_OPTIONS):
            wp.printf(
                "particle: %d, i_adj_tri: %d, particle_order: %d, \nforce:\n %f %f %f, \nhessian:, \n%f %f %f, \n%f %f %f, \n%f %f %f\n",
                particle_index,
                i_adj_tri,
                particle_order,
                f[0], f[1], f[2], h[0, 0], h[0, 1], h[0, 2], h[1, 0], h[1, 1], h[1, 2], h[2, 0], h[2, 1], h[2, 2],
            )
        # fmt: on

    for i_adj_edge in range(get_vertex_num_adjacent_edges(adjacency, particle_index)):
        nei_edge_index, vertex_order_on_edge = get_vertex_adjacent_edge_id_order(adjacency, particle_index, i_adj_edge)
        f_edge, h_edge = evaluate_dihedral_angle_based_bending_force_hessian(
            nei_edge_index,
            vertex_order_on_edge,
            pos,
            prev_pos,
            edge_indices,
            edge_rest_angles,
            edge_rest_length,
            edge_bending_properties[nei_edge_index, 0],
            edge_bending_properties[nei_edge_index, 1],
            dt,
        )

        f = f + f_edge
        h = h + h_edge

    # body-particle contact
    particle_contact_count = min(body_particle_contact_count[particle_index], body_particle_contact_buffer_pre_alloc)

    offset = body_particle_contact_buffer_pre_alloc * particle_index
    for contact_counter in range(particle_contact_count):
        # the index to access body-particle data, which is size-variable and only contains active contact
        contact_index = body_particle_contact_buffer[offset + contact_counter]

        body_contact_force, body_contact_hessian = evaluate_body_particle_contact(
            particle_index,
            particle_pos,
            particle_prev_pos,
            contact_index,
            soft_contact_ke,
            soft_contact_kd,
            friction_mu,
            friction_epsilon,
            particle_radius,
            shape_materials,
            shape_body,
            body_q,
            body_qd,
            body_com,
            contact_shape,
            contact_body_pos,
            contact_body_vel,
            contact_normal,
            dt,
        )

        f = f + body_contact_force
        h = h + body_contact_hessian

    if has_ground:
        ground_normal = wp.vec3(ground[0], ground[1], ground[2])
        ground_level = ground[3]
        ground_contact_force, ground_contact_hessian = evaluate_ground_contact_force_hessian(
            particle_pos,
            particle_prev_pos,
            particle_radius[particle_index],
            ground_normal,
            ground_level,
            soft_contact_ke,
            soft_contact_kd,
            friction_mu,
            friction_epsilon,
            dt,
        )

        f = f + ground_contact_force
        h = h + ground_contact_hessian

    if abs(wp.determinant(h)) > 1e-5:
        hInv = wp.inverse(h)
        pos_new[particle_index] = particle_pos + hInv * f


@wp.kernel
def VBD_copy_particle_positions_back(
    particle_ids_in_color: wp.array(dtype=wp.int32),
    pos: wp.array(dtype=wp.vec3),
    pos_new: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    particle = particle_ids_in_color[tid]

    pos[particle] = pos_new[particle]


@wp.kernel
def update_velocity(
    dt: float, prev_pos: wp.array(dtype=wp.vec3), pos: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3)
):
    particle = wp.tid()
    vel[particle] = (pos[particle] - prev_pos[particle]) / dt


@wp.kernel
def convert_body_particle_contact_data_kernel(
    # inputs
    body_particle_contact_buffer_pre_alloc: int,
    soft_contact_particle: wp.array(dtype=int),
    contact_count: wp.array(dtype=int),
    contact_max: int,
    # outputs
    body_particle_contact_buffer: wp.array(dtype=int),
    body_particle_contact_count: wp.array(dtype=int),
):
    contact_index = wp.tid()
    count = min(contact_max, contact_count[0])
    if contact_index >= count:
        return

    particle_index = soft_contact_particle[contact_index]
    offset = particle_index * body_particle_contact_buffer_pre_alloc

    contact_counter = wp.atomic_add(body_particle_contact_count, particle_index, 1)
    if contact_counter < body_particle_contact_buffer_pre_alloc:
        body_particle_contact_buffer[offset + contact_counter] = contact_index


@wp.kernel
def VBD_solve_trimesh_with_self_contact_penetration_free(
    dt: float,
    particle_ids_in_color: wp.array(dtype=wp.int32),
    pos_prev: wp.array(dtype=wp.vec3),
    pos: wp.array(dtype=wp.vec3),
    pos_new: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    mass: wp.array(dtype=float),
    inertia: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.uint32),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    tri_poses: wp.array(dtype=wp.mat22),
    tri_materials: wp.array(dtype=float, ndim=2),
    tri_areas: wp.array(dtype=float),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    edge_rest_angles: wp.array(dtype=float),
    edge_rest_length: wp.array(dtype=float),
    edge_bending_properties: wp.array(dtype=float, ndim=2),
    adjacency: ForceElementAdjacencyInfo,
    # contact info
    #   self contact
    collision_info: TriMeshCollisionInfo,
    collision_radius: float,
    soft_contact_ke: float,
    soft_contact_kd: float,
    friction_mu: float,
    friction_epsilon: float,
    pos_prev_collision_detection: wp.array(dtype=wp.vec3),
    particle_conservative_bounds: wp.array(dtype=float),
    edge_edge_parallel_epsilon: float,
    #   body-particle contact
    particle_radius: wp.array(dtype=float),
    body_particle_contact_buffer_pre_alloc: int,
    body_particle_contact_buffer: wp.array(dtype=int),
    body_particle_contact_count: wp.array(dtype=int),
    shape_materials: ModelShapeMaterials,
    shape_body: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    contact_shape: wp.array(dtype=int),
    contact_body_pos: wp.array(dtype=wp.vec3),
    contact_body_vel: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    # ground-particle contact
    has_ground: bool,
    ground: wp.array(dtype=float),
):
    t_id = wp.tid()

    particle_index = particle_ids_in_color[t_id]
    particle_pos = pos[particle_index]
    particle_prev_pos = pos_prev[particle_index]

    if not particle_flags[particle_index] & PARTICLE_FLAG_ACTIVE:
        return

    dt_sqr_reciprocal = 1.0 / (dt * dt)

    # inertia force and hessian
    f = mass[particle_index] * (inertia[particle_index] - pos[particle_index]) * (dt_sqr_reciprocal)
    h = mass[particle_index] * dt_sqr_reciprocal * wp.identity(n=3, dtype=float)

    # fmt: off
    if wp.static("inertia_force_hessian" in VBD_DEBUG_PRINTING_OPTIONS):
        wp.printf(
            "particle: %d after accumulate inertia\nforce:\n %f %f %f, \nhessian:, \n%f %f %f, \n%f %f %f, \n%f %f %f\n",
            particle_index,
            f[0], f[1], f[2], h[0, 0], h[0, 1], h[0, 2], h[1, 0], h[1, 1], h[1, 2], h[2, 0], h[2, 1], h[2, 2],
        )

    # v-side of the v-f collision force
    if wp.static("contact_info" in VBD_DEBUG_PRINTING_OPTIONS):
        wp.printf("Has %d colliding triangles\n", get_vertex_colliding_triangles_count(collision_info, particle_index))
    for i_v_collision in range(get_vertex_colliding_triangles_count(collision_info, particle_index)):
        colliding_t = get_vertex_colliding_triangles(collision_info, particle_index, i_v_collision)
        if wp.static("contact_info" in VBD_DEBUG_PRINTING_OPTIONS):
            wp.printf(
                "vertex %d is colliding with triangle: %d-(%d, %d, %d)",
                particle_index,
                colliding_t,
                tri_indices[colliding_t, 0],
                tri_indices[colliding_t, 1],
                tri_indices[colliding_t, 2],
            )
        # fmt: on

        collision_force, collision_hessian = evaluate_vertex_triangle_collision_force_hessian(
            particle_index,
            3,
            colliding_t,
            pos,
            pos_prev,
            tri_indices,
            collision_radius,
            soft_contact_ke,
            soft_contact_kd,
            friction_mu,
            friction_epsilon,
            dt,
        )
        f = f + collision_force
        h = h + collision_hessian

        # fmt: off
        if wp.static("contact_force_hessian_vt" in VBD_DEBUG_PRINTING_OPTIONS):
            wp.printf(
                "vertex: %d collision %d:\nforce:\n %f %f %f, \nhessian:, \n%f %f %f, \n%f %f %f, \n%f %f %f\n",
                particle_index,
                i_v_collision,
                collision_force[0], collision_force[1], collision_force[2], collision_hessian[0, 0], collision_hessian[0, 1], collision_hessian[0, 2], collision_hessian[1, 0], collision_hessian[1, 1], collision_hessian[1, 2], collision_hessian[2, 0], collision_hessian[2, 1], collision_hessian[2, 2],
            )
        # fmt: on

    # elastic force and hessian
    for i_adj_tri in range(get_vertex_num_adjacent_faces(adjacency, particle_index)):
        tri_index, vertex_order = get_vertex_adjacent_face_id_order(adjacency, particle_index, i_adj_tri)

        # fmt: off
        if wp.static("connectivity" in VBD_DEBUG_PRINTING_OPTIONS):
            wp.printf(
                "particle: %d | num_adj_faces: %d | ",
                particle_index,
                get_vertex_num_adjacent_faces(particle_index, adjacency),
            )
            wp.printf("i_face: %d | face id: %d | v_order: %d | ", i_adj_tri, tri_index, vertex_order)
            wp.printf(
                "face: %d %d %d\n",
                tri_indices[tri_index, 0],
                tri_indices[tri_index, 1],
                tri_indices[tri_index, 2],
            )
        # fmt: on

        f_tri, h_tri = evaluate_stvk_force_hessian(
            tri_index,
            vertex_order,
            pos,
            tri_indices,
            tri_poses[tri_index],
            tri_areas[tri_index],
            tri_materials[tri_index, 0],
            tri_materials[tri_index, 1],
            tri_materials[tri_index, 2],
        )
        # compute damping
        k_d = tri_materials[tri_index, 2]
        h_d = h_tri * (k_d / dt)

        f_d = h_d * (pos_prev[particle_index] - pos[particle_index])

        f = f + f_tri + f_d
        h = h + h_tri + h_d

        # t-side of vt-collision from the neighbor triangles
        # fmt: off
        if wp.static("contact_info" in VBD_DEBUG_PRINTING_OPTIONS):
            wp.printf(
                "Nei triangle %d has %d colliding vertice\n",
                tri_index,
                get_triangle_colliding_vertices_count(collision_info, tri_index),
            )
        # fmt: on
        for i_t_collision in range(get_triangle_colliding_vertices_count(collision_info, tri_index)):
            colliding_v = get_triangle_colliding_vertices(collision_info, tri_index, i_t_collision)

            collision_force, collision_hessian = evaluate_vertex_triangle_collision_force_hessian(
                colliding_v,
                vertex_order,
                tri_index,
                pos,
                pos_prev,
                tri_indices,
                collision_radius,
                soft_contact_ke,
                soft_contact_kd,
                friction_mu,
                friction_epsilon,
                dt,
            )

            f = f + collision_force
            h = h + collision_hessian

    # edge-edge collision force and hessian
    for i_adj_edge in range(get_vertex_num_adjacent_edges(adjacency, particle_index)):
        nei_edge_index, vertex_order_on_edge = get_vertex_adjacent_edge_id_order(adjacency, particle_index, i_adj_edge)
        # vertex is on the edge; otherwise it only effects the bending energy n
        if edge_bending_properties[nei_edge_index, 0] != 0:
            f_edge, h_edge = evaluate_dihedral_angle_based_bending_force_hessian(
                nei_edge_index, vertex_order_on_edge, pos, pos_prev, edge_indices, edge_rest_angles, edge_rest_length,
                edge_bending_properties[nei_edge_index, 0], edge_bending_properties[nei_edge_index, 1], dt
            )

            f = f + f_edge
            h = h + h_edge

        if vertex_order_on_edge == 2 or vertex_order_on_edge == 3:
            # collisions of neighbor triangles
            if wp.static("contact_info" in VBD_DEBUG_PRINTING_OPTIONS):
                wp.printf(
                    "Nei edge %d has %d colliding edge\n",
                    nei_edge_index,
                    get_edge_colliding_edges_count(collision_info, nei_edge_index),
                )
            for i_e_collision in range(get_edge_colliding_edges_count(collision_info, nei_edge_index)):
                colliding_e = get_edge_colliding_edges(collision_info, nei_edge_index, i_e_collision)

                collision_force, collision_hessian = evaluate_edge_edge_contact(
                    particle_index,
                    vertex_order_on_edge - 2,
                    nei_edge_index,
                    colliding_e,
                    pos,
                    pos_prev,
                    edge_indices,
                    collision_radius,
                    soft_contact_ke,
                    soft_contact_kd,
                    friction_mu,
                    friction_epsilon,
                    dt,
                    edge_edge_parallel_epsilon,
                )
                f = f + collision_force
                h = h + collision_hessian

                # fmt: off
                if wp.static("contact_force_hessian_ee" in VBD_DEBUG_PRINTING_OPTIONS):
                    wp.printf(
                        "vertex: %d edge %d collision %d:\nforce:\n %f %f %f, \nhessian:, \n%f %f %f, \n%f %f %f, \n%f %f %f\n",
                        particle_index,
                        nei_edge_index,
                        i_e_collision,
                        collision_force[0], collision_force[1], collision_force[2], collision_hessian[0, 0], collision_hessian[0, 1], collision_hessian[0, 2], collision_hessian[1, 0], collision_hessian[1, 1], collision_hessian[1, 2], collision_hessian[2, 0], collision_hessian[2, 1], collision_hessian[2, 2],
                    )
                # fmt: on

    # body-particle contact
    particle_contact_count = min(body_particle_contact_count[particle_index], body_particle_contact_buffer_pre_alloc)

    offset = body_particle_contact_buffer_pre_alloc * particle_index
    for contact_counter in range(particle_contact_count):
        # the index to access body-particle data, which is size-variable and only contains active contact
        contact_index = body_particle_contact_buffer[offset + contact_counter]

        body_contact_force, body_contact_hessian = evaluate_body_particle_contact(
            particle_index,
            particle_pos,
            particle_prev_pos,
            contact_index,
            soft_contact_ke,
            soft_contact_kd,
            friction_mu,
            friction_epsilon,
            particle_radius,
            shape_materials,
            shape_body,
            body_q,
            body_qd,
            body_com,
            contact_shape,
            contact_body_pos,
            contact_body_vel,
            contact_normal,
            dt,
        )

        f = f + body_contact_force
        h = h + body_contact_hessian

    if has_ground:
        ground_normal = wp.vec3(ground[0], ground[1], ground[2])
        ground_level = ground[3]
        ground_contact_force, ground_contact_hessian = evaluate_ground_contact_force_hessian(
            particle_pos,
            particle_prev_pos,
            particle_radius[particle_index],
            ground_normal,
            ground_level,
            soft_contact_ke,
            soft_contact_kd,
            friction_mu,
            friction_epsilon,
            dt,
        )

        f = f + ground_contact_force
        h = h + ground_contact_hessian

    # fmt: off
    if wp.static("overall_force_hessian" in VBD_DEBUG_PRINTING_OPTIONS):
        wp.printf(
            "vertex: %d final\noverall force:\n %f %f %f, \noverall hessian:, \n%f %f %f, \n%f %f %f, \n%f %f %f\n",
            particle_index,
            f[0], f[1], f[2], h[0, 0], h[0, 1], h[0, 2], h[1, 0], h[1, 1], h[1, 2], h[2, 0], h[2, 1], h[2, 2],
        )
    # fmt: on

    if abs(wp.determinant(h)) > 1e-5:
        h_inv = wp.inverse(h)
        particle_pos_new = pos[particle_index] + h_inv * f

        pos_new[particle_index] = apply_conservative_bound_truncation(
            particle_index, particle_pos_new, pos_prev_collision_detection, particle_conservative_bounds
        )


class VBDIntegrator(Integrator):
    """An implicit integrator using Vertex Block Descent (VBD) for cloth simulation.

    References:
        - Anka He Chen, Ziheng Liu, Yin Yang, and Cem Yuksel. 2024. Vertex Block Descent. ACM Trans. Graph. 43, 4, Article 116 (July 2024), 16 pages. https://doi.org/10.1145/3658179

    Note that VBDIntegrator's constructor requires a :class:`Model` object as input, so that it can do some precomputation and preallocate the space.
    After construction, you must provide the same :class:`Model` object that you used that was used during construction.
    Currently, you must manually provide particle coloring and assign it to `model.particle_coloring` to make VBD work.

    VBDIntegrator.simulate accepts three arguments: class:`Model`, :class:`State`, and :class:`Control` (optional) objects, this time-integrator
    may be used to advance the simulation state forward in time.

    Example
    -------

    .. code-block:: python

        model.particle_coloring = # load or generate particle coloring
        integrator = wp.VBDIntegrator(model)

        # simulation loop
        for i in range(100):
            state = integrator.simulate(model, state_in, state_out, dt, control)

    """

    def __init__(
        self,
        model: Model,
        iterations=10,
        handle_self_contact=False,
        penetration_free_conservative_bound_relaxation=0.42,
        friction_epsilon=1e-2,
        body_particle_contact_buffer_pre_alloc=4,
        vertex_collision_buffer_pre_alloc=32,
        edge_collision_buffer_pre_alloc=64,
        triangle_collision_buffer_pre_alloc=32,
        edge_edge_parallel_epsilon=1e-5,
    ):
        self.device = model.device
        self.model = model
        self.iterations = iterations

        # add new attributes for VBD solve
        self.particle_q_prev = wp.zeros_like(model.particle_q, device=self.device)
        self.inertia = wp.zeros_like(model.particle_q, device=self.device)

        self.adjacency = self.compute_force_element_adjacency(model).to(self.device)

        # data for body-particle collision
        self.body_particle_contact_buffer_pre_alloc = body_particle_contact_buffer_pre_alloc
        self.body_particle_contact_buffer = wp.zeros(
            (self.body_particle_contact_buffer_pre_alloc * model.particle_count,),
            dtype=wp.int32,
            device=self.device,
        )
        self.body_particle_contact_count = wp.zeros((model.particle_count,), dtype=wp.int32, device=self.device)

        self.handle_self_contact = handle_self_contact

        if handle_self_contact:
            if self.model.soft_contact_margin < self.model.soft_contact_radius:
                raise ValueError(
                    "model.soft_contact_margin is smaller than self.model.soft_contact_radius, this will result in missing contacts and cause instability. \n"
                    "It is advisable to make model.soft_contact_margin 1.5~2 times larger than self.model.soft_contact_radius."
                )

            self.conservative_bound_relaxation = penetration_free_conservative_bound_relaxation
            self.pos_prev_collision_detection = wp.zeros_like(model.particle_q, device=self.device)
            self.particle_conservative_bounds = wp.full((model.particle_count,), dtype=float, device=self.device)

            self.trimesh_collision_detector = TriMeshCollisionDetector(
                self.model,
                vertex_collision_buffer_pre_alloc=vertex_collision_buffer_pre_alloc,
                edge_collision_buffer_pre_alloc=edge_collision_buffer_pre_alloc,
                triangle_collision_buffer_pre_alloc=triangle_collision_buffer_pre_alloc,
                edge_edge_parallel_epsilon=edge_edge_parallel_epsilon,
            )

        self.friction_epsilon = friction_epsilon

        if len(self.model.particle_coloring) == 0:
            raise ValueError(
                "model.particle_coloring is empty! When using the VBDIntegrator you must call ModelBuilder.color() "
                "or ModelBuilder.set_coloring() before calling ModelBuilder.finalize()."
            )

        # tests
        # wp.launch(kernel=_test_compute_force_element_adjacency,
        #           inputs=[self.adjacency, model.edge_indices, model.tri_indices],
        #           dim=1, device=self.device)

    def compute_force_element_adjacency(self, model):
        adjacency = ForceElementAdjacencyInfo()
        edges_array = model.edge_indices.to("cpu")

        if edges_array.size:
            # build vertex-edge adjacency data
            num_vertex_adjacent_edges = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32, device="cpu")

            wp.launch(
                kernel=self.count_num_adjacent_edges,
                inputs=[edges_array, num_vertex_adjacent_edges],
                dim=1,
                device="cpu",
            )

            num_vertex_adjacent_edges = num_vertex_adjacent_edges.numpy()
            vertex_adjacent_edges_offsets = np.empty(shape=(self.model.particle_count + 1,), dtype=wp.int32)
            vertex_adjacent_edges_offsets[1:] = np.cumsum(2 * num_vertex_adjacent_edges)[:]
            vertex_adjacent_edges_offsets[0] = 0
            adjacency.v_adj_edges_offsets = wp.array(vertex_adjacent_edges_offsets, dtype=wp.int32, device="cpu")

            # temporal variables to record how much adjacent edges has been filled to each vertex
            vertex_adjacent_edges_fill_count = wp.zeros(
                shape=(self.model.particle_count,), dtype=wp.int32, device="cpu"
            )

            edge_adjacency_array_size = 2 * num_vertex_adjacent_edges.sum()
            # vertex order: o0: 0, o1: 1, v0: 2, v1: 3,
            adjacency.v_adj_edges = wp.empty(shape=(edge_adjacency_array_size,), dtype=wp.int32, device="cpu")

            wp.launch(
                kernel=self.fill_adjacent_edges,
                inputs=[
                    edges_array,
                    adjacency.v_adj_edges_offsets,
                    vertex_adjacent_edges_fill_count,
                    adjacency.v_adj_edges,
                ],
                dim=1,
                device="cpu",
            )
        else:
            adjacency.v_adj_edges_offsets = wp.empty(shape=(0,), dtype=wp.int32, device="cpu")
            adjacency.v_adj_edges = wp.empty(shape=(0,), dtype=wp.int32, device="cpu")

        # compute adjacent triangles

        # count number of adjacent faces for each vertex
        face_indices = model.tri_indices.to("cpu")
        num_vertex_adjacent_faces = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32, device="cpu")
        wp.launch(
            kernel=self.count_num_adjacent_faces, inputs=[face_indices, num_vertex_adjacent_faces], dim=1, device="cpu"
        )

        # preallocate memory based on counting results
        num_vertex_adjacent_faces = num_vertex_adjacent_faces.numpy()
        vertex_adjacent_faces_offsets = np.empty(shape=(self.model.particle_count + 1,), dtype=wp.int32)
        vertex_adjacent_faces_offsets[1:] = np.cumsum(2 * num_vertex_adjacent_faces)[:]
        vertex_adjacent_faces_offsets[0] = 0
        adjacency.v_adj_faces_offsets = wp.array(vertex_adjacent_faces_offsets, dtype=wp.int32, device="cpu")

        vertex_adjacent_faces_fill_count = wp.zeros(shape=(self.model.particle_count,), dtype=wp.int32, device="cpu")

        face_adjacency_array_size = 2 * num_vertex_adjacent_faces.sum()
        # (face, vertex_order) * num_adj_faces * num_particles
        # vertex order: v0: 0, v1: 1, o0: 2, v2: 3
        adjacency.v_adj_faces = wp.empty(shape=(face_adjacency_array_size,), dtype=wp.int32, device="cpu")

        wp.launch(
            kernel=self.fill_adjacent_faces,
            inputs=[
                face_indices,
                adjacency.v_adj_faces_offsets,
                vertex_adjacent_faces_fill_count,
                adjacency.v_adj_faces,
            ],
            dim=1,
            device="cpu",
        )

        return adjacency

    def simulate(self, model: Model, state_in: State, state_out: State, dt: float, control: Control = None):
        if model is not self.model:
            raise ValueError("model must be the one used to initialize VBDIntegrator")

        if self.handle_self_contact:
            self.simulate_one_step_with_collisions_penetration_free(model, state_in, state_out, dt, control)
        else:
            self.simulate_one_step_no_self_contact(model, state_in, state_out, dt, control)

    def simulate_one_step_no_self_contact(
        self, model: Model, state_in: State, state_out: State, dt: float, control: Control = None
    ):
        self.convert_body_particle_contact_data()

        wp.launch(
            kernel=forward_step,
            inputs=[
                dt,
                model.gravity,
                self.particle_q_prev,
                state_in.particle_q,
                state_in.particle_qd,
                self.model.particle_inv_mass,
                state_in.particle_f,
                self.model.particle_flags,
                self.inertia,
            ],
            dim=self.model.particle_count,
            device=self.device,
        )

        for _iter in range(self.iterations):
            for color_counter in range(len(self.model.particle_coloring)):
                wp.launch(
                    kernel=VBD_solve_trimesh_no_self_contact,
                    inputs=[
                        dt,
                        self.model.particle_coloring[color_counter],
                        self.particle_q_prev,
                        state_in.particle_q,
                        state_out.particle_q,
                        state_in.particle_qd,
                        self.model.particle_mass,
                        self.inertia,
                        self.model.particle_flags,
                        self.model.tri_indices,
                        self.model.tri_poses,
                        self.model.tri_materials,
                        self.model.tri_areas,
                        self.model.edge_indices,
                        self.model.edge_rest_angle,
                        self.model.edge_rest_length,
                        self.model.edge_bending_properties,
                        self.adjacency,
                        self.model.soft_contact_ke,
                        self.model.soft_contact_kd,
                        self.model.soft_contact_mu,
                        self.friction_epsilon,
                        #   body-particle contact
                        self.model.particle_radius,
                        self.body_particle_contact_buffer_pre_alloc,
                        self.body_particle_contact_buffer,
                        self.body_particle_contact_count,
                        self.model.shape_materials,
                        self.model.shape_body,
                        self.model.body_q,
                        self.model.body_qd,
                        self.model.body_com,
                        self.model.soft_contact_shape,
                        self.model.soft_contact_body_pos,
                        self.model.soft_contact_body_vel,
                        self.model.soft_contact_normal,
                        self.model.ground,
                        self.model.ground_plane,
                    ],
                    dim=self.model.particle_coloring[color_counter].size,
                    device=self.device,
                )

                wp.launch(
                    kernel=VBD_copy_particle_positions_back,
                    inputs=[self.model.particle_coloring[color_counter], state_in.particle_q, state_out.particle_q],
                    dim=self.model.particle_coloring[color_counter].size,
                    device=self.device,
                )

        wp.launch(
            kernel=update_velocity,
            inputs=[dt, self.particle_q_prev, state_out.particle_q, state_out.particle_qd],
            dim=self.model.particle_count,
            device=self.device,
        )

    def simulate_one_step_with_collisions_penetration_free(
        self, model: Model, state_in: State, state_out: State, dt: float, control: Control = None
    ):
        self.convert_body_particle_contact_data()
        # collision detection before initialization to compute conservative bounds for initialization
        self.collision_detection_penetration_free(state_in, dt)

        wp.launch(
            kernel=forward_step_penetration_free,
            inputs=[
                dt,
                model.gravity,
                self.particle_q_prev,
                state_in.particle_q,
                state_in.particle_qd,
                self.model.particle_inv_mass,
                state_in.particle_f,
                self.model.particle_flags,
                self.pos_prev_collision_detection,
                self.particle_conservative_bounds,
                self.inertia,
            ],
            dim=self.model.particle_count,
            device=self.device,
        )

        # after initialization, we do another collision detection to update the bounds
        self.collision_detection_penetration_free(state_in, dt)

        for _iter in range(self.iterations):
            for i_color in range(len(self.model.particle_coloring)):
                wp.launch(
                    kernel=VBD_solve_trimesh_with_self_contact_penetration_free,
                    inputs=[
                        dt,
                        self.model.particle_coloring[i_color],
                        self.particle_q_prev,
                        state_in.particle_q,
                        state_out.particle_q,
                        state_in.particle_qd,
                        self.model.particle_mass,
                        self.inertia,
                        self.model.particle_flags,
                        self.model.tri_indices,
                        self.model.tri_poses,
                        self.model.tri_materials,
                        self.model.tri_areas,
                        self.model.edge_indices,
                        self.model.edge_rest_angle,
                        self.model.edge_rest_length,
                        self.model.edge_bending_properties,
                        self.adjacency,
                        #   self-contact
                        self.trimesh_collision_detector.collision_info,
                        self.model.soft_contact_radius,
                        self.model.soft_contact_ke,
                        self.model.soft_contact_kd,
                        self.model.soft_contact_mu,
                        self.friction_epsilon,
                        self.pos_prev_collision_detection,
                        self.particle_conservative_bounds,
                        self.trimesh_collision_detector.edge_edge_parallel_epsilon,
                        #   body-particle contact
                        self.model.particle_radius,
                        self.body_particle_contact_buffer_pre_alloc,
                        self.body_particle_contact_buffer,
                        self.body_particle_contact_count,
                        self.model.shape_materials,
                        self.model.shape_body,
                        self.model.body_q,
                        self.model.body_qd,
                        self.model.body_com,
                        self.model.soft_contact_shape,
                        self.model.soft_contact_body_pos,
                        self.model.soft_contact_body_vel,
                        self.model.soft_contact_normal,
                        self.model.ground,
                        self.model.ground_plane,
                    ],
                    dim=self.model.particle_coloring[i_color].shape[0],
                    device=self.device,
                )

                wp.launch(
                    kernel=VBD_copy_particle_positions_back,
                    inputs=[self.model.particle_coloring[i_color], state_in.particle_q, state_out.particle_q],
                    dim=self.model.particle_coloring[i_color].size,
                    device=self.device,
                )

        wp.launch(
            kernel=update_velocity,
            inputs=[dt, self.particle_q_prev, state_out.particle_q, state_out.particle_qd],
            dim=self.model.particle_count,
            device=self.device,
        )

    def collision_detection_penetration_free(self, current_state, dt):
        self.trimesh_collision_detector.refit(current_state.particle_q)
        self.trimesh_collision_detector.vertex_triangle_collision_detection(self.model.soft_contact_margin)
        self.trimesh_collision_detector.edge_edge_collision_detection(self.model.soft_contact_margin)

        self.pos_prev_collision_detection.assign(current_state.particle_q)
        wp.launch(
            kernel=compute_particle_conservative_bound,
            inputs=[
                self.conservative_bound_relaxation,
                self.model.soft_contact_margin,
                self.adjacency,
                self.trimesh_collision_detector.collision_info,
            ],
            outputs=[
                self.particle_conservative_bounds,
            ],
            dim=self.model.particle_count,
            device=self.device,
        )

    def convert_body_particle_contact_data(self):
        self.body_particle_contact_count.zero_()

        wp.launch(
            kernel=convert_body_particle_contact_data_kernel,
            inputs=[
                self.body_particle_contact_buffer_pre_alloc,
                self.model.soft_contact_particle,
                self.model.soft_contact_count,
                self.model.soft_contact_max,
            ],
            outputs=[self.body_particle_contact_buffer, self.body_particle_contact_count],
            dim=self.model.soft_contact_max,
            device=self.device,
        )

    def rebuild_bvh(self, state: State):
        """This function will rebuild the BVHs used for detecting self-contacts using the input `state`.

        When the simulated object deforms significantly, simply refitting the BVH can lead to deterioration of the BVH's
        quality. In these cases, rebuilding the entire tree is necessary to achieve better querying efficiency.

        Args:
            state (wp.sim.State):  The state whose particle positions (:attr:`State.particle_q`) will be used for rebuilding the BVHs.
        """
        self.trimesh_collision_detector.rebuild(state.particle_q)

    @wp.kernel
    def count_num_adjacent_edges(
        edges_array: wp.array(dtype=wp.int32, ndim=2), num_vertex_adjacent_edges: wp.array(dtype=wp.int32)
    ):
        for edge_id in range(edges_array.shape[0]):
            o0 = edges_array[edge_id, 0]
            o1 = edges_array[edge_id, 1]

            v0 = edges_array[edge_id, 2]
            v1 = edges_array[edge_id, 3]

            num_vertex_adjacent_edges[v0] = num_vertex_adjacent_edges[v0] + 1
            num_vertex_adjacent_edges[v1] = num_vertex_adjacent_edges[v1] + 1

            if o0 != -1:
                num_vertex_adjacent_edges[o0] = num_vertex_adjacent_edges[o0] + 1
            if o1 != -1:
                num_vertex_adjacent_edges[o1] = num_vertex_adjacent_edges[o1] + 1

    @wp.kernel
    def fill_adjacent_edges(
        edges_array: wp.array(dtype=wp.int32, ndim=2),
        vertex_adjacent_edges_offsets: wp.array(dtype=wp.int32),
        vertex_adjacent_edges_fill_count: wp.array(dtype=wp.int32),
        vertex_adjacent_edges: wp.array(dtype=wp.int32),
    ):
        for edge_id in range(edges_array.shape[0]):
            v0 = edges_array[edge_id, 2]
            v1 = edges_array[edge_id, 3]

            fill_count_v0 = vertex_adjacent_edges_fill_count[v0]
            buffer_offset_v0 = vertex_adjacent_edges_offsets[v0]
            vertex_adjacent_edges[buffer_offset_v0 + fill_count_v0 * 2] = edge_id
            vertex_adjacent_edges[buffer_offset_v0 + fill_count_v0 * 2 + 1] = 2
            vertex_adjacent_edges_fill_count[v0] = fill_count_v0 + 1

            fill_count_v1 = vertex_adjacent_edges_fill_count[v1]
            buffer_offset_v1 = vertex_adjacent_edges_offsets[v1]
            vertex_adjacent_edges[buffer_offset_v1 + fill_count_v1 * 2] = edge_id
            vertex_adjacent_edges[buffer_offset_v1 + fill_count_v1 * 2 + 1] = 3
            vertex_adjacent_edges_fill_count[v1] = fill_count_v1 + 1

            o0 = edges_array[edge_id, 0]
            if o0 != -1:
                fill_count_o0 = vertex_adjacent_edges_fill_count[o0]
                buffer_offset_o0 = vertex_adjacent_edges_offsets[o0]
                vertex_adjacent_edges[buffer_offset_o0 + fill_count_o0 * 2] = edge_id
                vertex_adjacent_edges[buffer_offset_o0 + fill_count_o0 * 2 + 1] = 0
                vertex_adjacent_edges_fill_count[o0] = fill_count_o0 + 1

            o1 = edges_array[edge_id, 1]
            if o1 != -1:
                fill_count_o1 = vertex_adjacent_edges_fill_count[o1]
                buffer_offset_o1 = vertex_adjacent_edges_offsets[o1]
                vertex_adjacent_edges[buffer_offset_o1 + fill_count_o1 * 2] = edge_id
                vertex_adjacent_edges[buffer_offset_o1 + fill_count_o1 * 2 + 1] = 1
                vertex_adjacent_edges_fill_count[o1] = fill_count_o1 + 1

    @wp.kernel
    def count_num_adjacent_faces(
        face_indices: wp.array(dtype=wp.int32, ndim=2), num_vertex_adjacent_faces: wp.array(dtype=wp.int32)
    ):
        for face in range(face_indices.shape[0]):
            v0 = face_indices[face, 0]
            v1 = face_indices[face, 1]
            v2 = face_indices[face, 2]

            num_vertex_adjacent_faces[v0] = num_vertex_adjacent_faces[v0] + 1
            num_vertex_adjacent_faces[v1] = num_vertex_adjacent_faces[v1] + 1
            num_vertex_adjacent_faces[v2] = num_vertex_adjacent_faces[v2] + 1

    @wp.kernel
    def fill_adjacent_faces(
        face_indices: wp.array(dtype=wp.int32, ndim=2),
        vertex_adjacent_faces_offsets: wp.array(dtype=wp.int32),
        vertex_adjacent_faces_fill_count: wp.array(dtype=wp.int32),
        vertex_adjacent_faces: wp.array(dtype=wp.int32),
    ):
        for face in range(face_indices.shape[0]):
            v0 = face_indices[face, 0]
            v1 = face_indices[face, 1]
            v2 = face_indices[face, 2]

            fill_count_v0 = vertex_adjacent_faces_fill_count[v0]
            buffer_offset_v0 = vertex_adjacent_faces_offsets[v0]
            vertex_adjacent_faces[buffer_offset_v0 + fill_count_v0 * 2] = face
            vertex_adjacent_faces[buffer_offset_v0 + fill_count_v0 * 2 + 1] = 0
            vertex_adjacent_faces_fill_count[v0] = fill_count_v0 + 1

            fill_count_v1 = vertex_adjacent_faces_fill_count[v1]
            buffer_offset_v1 = vertex_adjacent_faces_offsets[v1]
            vertex_adjacent_faces[buffer_offset_v1 + fill_count_v1 * 2] = face
            vertex_adjacent_faces[buffer_offset_v1 + fill_count_v1 * 2 + 1] = 1
            vertex_adjacent_faces_fill_count[v1] = fill_count_v1 + 1

            fill_count_v2 = vertex_adjacent_faces_fill_count[v2]
            buffer_offset_v2 = vertex_adjacent_faces_offsets[v2]
            vertex_adjacent_faces[buffer_offset_v2 + fill_count_v2 * 2] = face
            vertex_adjacent_faces[buffer_offset_v2 + fill_count_v2 * 2 + 1] = 2
            vertex_adjacent_faces_fill_count[v2] = fill_count_v2 + 1
