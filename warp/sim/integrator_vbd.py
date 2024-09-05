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
from .model import PARTICLE_FLAG_ACTIVE, Control, Model, State


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
    vertex = wp.tid()

    prev_pos[vertex] = pos[vertex]
    if not particle_flags[vertex] & PARTICLE_FLAG_ACTIVE:
        inertia[vertex] = prev_pos[vertex]
        return
    vel_new = vel[vertex] + (gravity + external_force[vertex] * inv_mass[vertex]) * dt
    pos[vertex] = pos[vertex] + vel_new * dt
    inertia[vertex] = pos[vertex]


@wp.kernel
def VBD_solve_trimesh(
    dt: float,
    vertex_ids_in_color: wp.array(dtype=wp.int32),
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
):
    t_id = wp.tid()

    vertex = vertex_ids_in_color[t_id]
    # wp.printf("vId: %d\n", vertex)

    if not particle_flags[vertex] & PARTICLE_FLAG_ACTIVE:
        return

    dtSqrReciprocal = 1.0 / (dt * dt)

    # inertia force and hessian
    f = mass[vertex] * (inertia[vertex] - pos[vertex]) * (dtSqrReciprocal)
    h = mass[vertex] * dtSqrReciprocal * wp.identity(n=3, dtype=float)

    # elastic force and hessian
    for i_adj_tri in range(get_vertex_num_adjacent_faces(vertex, adjacency)):
        # wp.printf("vertex: %d | num_adj_faces: %d | ", vertex, get_vertex_num_adjacent_faces(vertex, adjacency))
        tri_id, vertex_order = get_vertex_adjacent_face_id_order(vertex, i_adj_tri, adjacency)

        # wp.printf("i_face: %d | face id: %d | v_order: %d | ", i_adj_tri, tri_id, vertex_order)
        # wp.printf("face: %d %d %d\n", tri_indices[tri_id, 0], tri_indices[tri_id, 1], tri_indices[tri_id, 2], )

        f_tri, h_tri = evaluate_stvk_force_hessian(
            tri_id,
            vertex_order,
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

        f_d = h_d * (prev_pos[vertex] - pos[vertex])

        f = f + f_tri + f_d
        h = h + h_tri + h_d

        # wp.printf("vertex: %d, i_adj_tri: %d, vertex_order: %d, \nforce:\n %f %f %f, \nhessian:, \n%f %f %f, \n%f %f %f, \n%f %f %f\n",
        #           vertex, i_adj_tri, vertex_order,
        #           f[0], f[1], f[2],
        #           h[0, 0], h[0, 1], h[0, 2],
        #           h[1, 0], h[1, 1], h[1, 2],
        #           h[2, 0], h[2, 1], h[2, 2],
        #           )

    if abs(wp.determinant(h)) > 1e-5:
        hInv = wp.inverse(h)
        pos_new[vertex] = pos[vertex] + hInv * f


@wp.kernel
def VBD_copy_particle_positions_back(
    vertex_ids_in_color: wp.array(dtype=wp.int32),
    pos: wp.array(dtype=wp.vec3),
    pos_new: wp.array(dtype=wp.vec3),
):
    t_id = wp.tid()
    vertex = vertex_ids_in_color[t_id]

    pos[vertex] = pos_new[vertex]


@wp.kernel
def update_velocity(
    dt: float, prev_pos: wp.array(dtype=wp.vec3), pos: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3)
):
    vertex = wp.tid()
    vel[vertex] = (pos[vertex] - prev_pos[vertex]) / dt


class VBDIntegrator(Integrator):
    def __init__(self, model: Model, iterations=10):
        self.device = model.device
        self.model = model
        self.iterations = iterations

        # add new attributes for VBD solve
        self.particle_q_prev = wp.zeros_like(model.particle_q, device=self.device)
        self.inertia = wp.zeros_like(model.particle_q, device=self.device)

        self.adjacency = self.compute_force_element_adjacency(model).to(self.device)

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
            for i_color in range(len(self.model.coloring)):
                wp.launch(
                    kernel=VBD_solve_trimesh,
                    inputs=[
                        dt,
                        self.model.coloring[i_color],
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
                    ],
                    dim=self.model.coloring[i_color].size,
                    device=self.device,
                )

                wp.launch(
                    kernel=VBD_copy_particle_positions_back,
                    inputs=[self.model.coloring[i_color], state_in.particle_q, state_out.particle_q],
                    dim=self.model.coloring[i_color].size,
                    device=self.device,
                )

        wp.launch(
            kernel=update_velocity,
            inputs=[dt, self.particle_q_prev, state_out.particle_q, state_out.particle_qd],
            dim=self.model.particle_count,
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
