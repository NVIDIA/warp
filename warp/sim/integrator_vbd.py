# Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import numpy as np

import warp as wp

from ..types import float32, matrix
from .integrator import Integrator
from .model import PARTICLE_FLAG_ACTIVE, Control, Model, ModelShapeMaterials, State


class mat66(matrix(shape=(6, 6), dtype=float32)):
    pass


class mat32(matrix(shape=(3, 2), dtype=float32)):
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
def get_vertex_num_adjacent_edges(vertex: wp.int32, adjacency: ForceElementAdjacencyInfo):
    return (adjacency.v_adj_edges_offsets[vertex + 1] - adjacency.v_adj_edges_offsets[vertex]) >> 1


@wp.func
def get_vertex_adjacent_edge_id_order(vertex: wp.int32, edge: wp.int32, adjacency: ForceElementAdjacencyInfo):
    offset = adjacency.v_adj_edges_offsets[vertex]
    return adjacency.v_adj_edges[offset + edge * 2], adjacency.v_adj_edges[offset + edge * 2 + 1]


@wp.func
def get_vertex_num_adjacent_faces(vertex: wp.int32, adjacency: ForceElementAdjacencyInfo):
    return (adjacency.v_adj_faces_offsets[vertex + 1] - adjacency.v_adj_faces_offsets[vertex]) >> 1


@wp.func
def get_vertex_adjacent_face_id_order(vertex: wp.int32, face: wp.int32, adjacency: ForceElementAdjacencyInfo):
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
        num_adj_edges = get_vertex_num_adjacent_edges(vertex, adjacency)
        for i_bd in range(num_adj_edges):
            bd_id, v_order = get_vertex_adjacent_edge_id_order(vertex, i_bd, adjacency)

            if edge_indices[bd_id, v_order] != vertex:
                print("Error!!!")
                wp.printf("vertex: %d | num_adj_edges: %d\n", vertex, num_adj_edges)
                wp.printf("--iBd: %d | ", i_bd)
                wp.printf("edge id: %d | v_order: %d\n", bd_id, v_order)

        num_adj_faces = get_vertex_num_adjacent_faces(vertex, adjacency)

        for i_face in range(num_adj_faces):
            face, v_order = get_vertex_adjacent_face_id_order(vertex, i_face, adjacency)

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

    # wp.printf("face: %d, \nF12:\n, %f %f\n, %f %f\n, %f %f\nG:\n%f %f\n, %f %f\n",
    #           face,
    #           F[0, 0], F[0, 1],
    #           F[1, 0], F[1, 1],
    #           F[2, 0], F[2, 1],
    #
    #           G[0, 0], G[0, 1],
    #           G[1, 0], G[1, 1],
    #           )

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
def evaluate_ground_contact_force_hessian(
    vertex_pos: wp.vec3,
    vertex_prev_pos: wp.vec3,
    particle_radius: float,
    ground_normal: wp.vec3,
    ground_level: float,
    soft_contact_ke: float,
    friction_mu: float,
    friction_epsilon: float,
    dt: float,
):
    penetration_depth = -(wp.dot(ground_normal, vertex_pos) + ground_level - particle_radius)

    if penetration_depth > 0:
        ground_contact_force_norm = penetration_depth * soft_contact_ke
        ground_contact_force = ground_normal * ground_contact_force_norm
        ground_contact_hessian = soft_contact_ke * wp.outer(ground_normal, ground_normal)

        dx = vertex_pos - vertex_prev_pos

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

        mu = 0.5 * (friction_mu + shape_materials.mu[shape_index])

        dx = particle_pos - particle_prev_pos

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
def compute_friction(mu: float, normal_contact_force: float, T: mat32, u: wp.vec2, eps_u: float):
    """
    Returns the friction force and hessian.
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
def VBD_solve_trimesh(
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
    adjacency: ForceElementAdjacencyInfo,
    # contact info
    #   self contact
    soft_contact_ke: float,
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
    # wp.printf("vId: %d\n", particle)

    if not particle_flags[particle_index] & PARTICLE_FLAG_ACTIVE:
        return

    particle_pos = pos[particle_index]
    particle_prev_pos = pos[particle_index]

    dt_sqr_reciprocal = 1.0 / (dt * dt)

    # inertia force and hessian
    f = mass[particle_index] * (inertia[particle_index] - pos[particle_index]) * (dt_sqr_reciprocal)
    h = mass[particle_index] * dt_sqr_reciprocal * wp.identity(n=3, dtype=float)

    # elastic force and hessian
    for i_adj_tri in range(get_vertex_num_adjacent_faces(particle_index, adjacency)):
        # wp.printf("particle: %d | num_adj_faces: %d | ", particle, get_particle_num_adjacent_faces(particle, adjacency))
        tri_id, particle_order = get_vertex_adjacent_face_id_order(particle_index, i_adj_tri, adjacency)

        # wp.printf("i_face: %d | face id: %d | v_order: %d | ", i_adj_tri, tri_id, particle_order)
        # wp.printf("face: %d %d %d\n", tri_indices[tri_id, 0], tri_indices[tri_id, 1], tri_indices[tri_id, 2], )

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

        # wp.printf("particle: %d, i_adj_tri: %d, particle_order: %d, \nforce:\n %f %f %f, \nhessian:, \n%f %f %f, \n%f %f %f, \n%f %f %f\n",
        #           particle, i_adj_tri, particle_order,
        #           f[0], f[1], f[2],
        #           h[0, 0], h[0, 1], h[0, 2],
        #           h[1, 0], h[1, 1], h[1, 2],
        #           h[2, 0], h[2, 1], h[2, 2],
        #           )

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
        body_particle_contact_buffer_pre_alloc=4,
        friction_epsilon=1e-2,
    ):
        self.device = model.device
        self.model = model
        self.iterations = iterations

        # add new attributes for VBD solve
        self.particle_q_prev = wp.zeros_like(model.particle_q, device=self.device)
        self.inertia = wp.zeros_like(model.particle_q, device=self.device)

        self.adjacency = self.compute_force_element_adjacency(model).to(self.device)

        self.body_particle_contact_buffer_pre_alloc = body_particle_contact_buffer_pre_alloc
        self.body_particle_contact_buffer = wp.zeros(
            (self.body_particle_contact_buffer_pre_alloc * model.particle_count,),
            dtype=wp.int32,
            device=self.device,
        )
        self.body_particle_contact_count = wp.zeros((model.particle_count,), dtype=wp.int32, device=self.device)
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
                    kernel=VBD_solve_trimesh,
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
                        self.adjacency,
                        self.model.soft_contact_ke,
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

            o0 = edges_array[edge_id, 2]
            if o0 != -1:
                fill_count_o0 = vertex_adjacent_edges_fill_count[o0]
                buffer_offset_o0 = vertex_adjacent_edges_offsets[o0]
                vertex_adjacent_edges[buffer_offset_o0 + fill_count_o0 * 2] = edge_id
                vertex_adjacent_edges[buffer_offset_o0 + fill_count_o0 * 2 + 1] = 0
                vertex_adjacent_edges_fill_count[o0] = fill_count_o0 + 1

            o1 = edges_array[edge_id, 3]
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
