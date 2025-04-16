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

import unittest

import warp as wp
import warp.examples
import warp.sim
from warp.sim.collide import (
    TriMeshCollisionDetector,
    init_triangle_collision_data_kernel,
    triangle_closest_point,
    vertex_adjacent_to_triangle,
)
from warp.sim.integrator_euler import eval_triangles_contact
from warp.tests.unittest_utils import *


@wp.kernel
def vertex_triangle_collision_detection_brute_force(
    query_radius: float,
    bvh_id: wp.uint64,
    pos: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    vertex_colliding_triangles: wp.array(dtype=wp.int32),
    vertex_colliding_triangles_count: wp.array(dtype=wp.int32),
    vertex_colliding_triangles_offsets: wp.array(dtype=wp.int32),
    vertex_colliding_triangles_buffer_size: wp.array(dtype=wp.int32),
    vertex_colliding_triangles_min_dist: wp.array(dtype=float),
    triangle_colliding_vertices: wp.array(dtype=wp.int32),
    triangle_colliding_vertices_count: wp.array(dtype=wp.int32),
    triangle_colliding_vertices_buffer_offsets: wp.array(dtype=wp.int32),
    triangle_colliding_vertices_buffer_sizes: wp.array(dtype=wp.int32),
    triangle_colliding_vertices_min_dist: wp.array(dtype=float),
    resize_flags: wp.array(dtype=wp.int32),
):
    v_index = wp.tid()
    v = pos[v_index]

    vertex_num_collisions = wp.int32(0)
    min_dis_to_tris = query_radius
    for tri_index in range(tri_indices.shape[0]):
        t1 = tri_indices[tri_index, 0]
        t2 = tri_indices[tri_index, 1]
        t3 = tri_indices[tri_index, 2]
        if vertex_adjacent_to_triangle(v_index, t1, t2, t3):
            continue

        u1 = pos[t1]
        u2 = pos[t2]
        u3 = pos[t3]

        closest_p, bary, feature_type = triangle_closest_point(u1, u2, u3, v)

        dis = wp.length(closest_p - v)

        if dis < query_radius:
            vertex_num_collisions = vertex_num_collisions + 1
            min_dis_to_tris = wp.min(dis, min_dis_to_tris)

            wp.atomic_add(triangle_colliding_vertices_count, tri_index, 1)
            wp.atomic_min(triangle_colliding_vertices_min_dist, tri_index, dis)

    vertex_colliding_triangles_count[v_index] = vertex_num_collisions
    vertex_colliding_triangles_min_dist[v_index] = min_dis_to_tris


@wp.kernel
def vertex_triangle_collision_detection_brute_force_no_triangle_buffers(
    query_radius: float,
    bvh_id: wp.uint64,
    pos: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    vertex_colliding_triangles: wp.array(dtype=wp.int32),
    vertex_colliding_triangles_count: wp.array(dtype=wp.int32),
    vertex_colliding_triangles_offsets: wp.array(dtype=wp.int32),
    vertex_colliding_triangles_buffer_size: wp.array(dtype=wp.int32),
    vertex_colliding_triangles_min_dist: wp.array(dtype=float),
    triangle_colliding_vertices_min_dist: wp.array(dtype=float),
    resize_flags: wp.array(dtype=wp.int32),
):
    v_index = wp.tid()
    v = pos[v_index]

    vertex_num_collisions = wp.int32(0)
    min_dis_to_tris = query_radius
    for tri_index in range(tri_indices.shape[0]):
        t1 = tri_indices[tri_index, 0]
        t2 = tri_indices[tri_index, 1]
        t3 = tri_indices[tri_index, 2]
        if vertex_adjacent_to_triangle(v_index, t1, t2, t3):
            continue

        u1 = pos[t1]
        u2 = pos[t2]
        u3 = pos[t3]

        closest_p, bary, feature_type = triangle_closest_point(u1, u2, u3, v)

        dis = wp.length(closest_p - v)

        if dis < query_radius:
            vertex_num_collisions = vertex_num_collisions + 1
            min_dis_to_tris = wp.min(dis, min_dis_to_tris)

            wp.atomic_min(triangle_colliding_vertices_min_dist, tri_index, dis)

    vertex_colliding_triangles_count[v_index] = vertex_num_collisions
    vertex_colliding_triangles_min_dist[v_index] = min_dis_to_tris


@wp.kernel
def validate_vertex_collisions(
    query_radius: float,
    bvh_id: wp.uint64,
    pos: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    vertex_colliding_triangles: wp.array(dtype=wp.int32),
    vertex_colliding_triangles_count: wp.array(dtype=wp.int32),
    vertex_colliding_triangles_offsets: wp.array(dtype=wp.int32),
    vertex_colliding_triangles_buffer_size: wp.array(dtype=wp.int32),
    vertex_colliding_triangles_min_dist: wp.array(dtype=float),
    resize_flags: wp.array(dtype=wp.int32),
):
    v_index = wp.tid()
    v = pos[v_index]

    num_cols = vertex_colliding_triangles_count[v_index]
    offset = vertex_colliding_triangles_offsets[v_index]
    min_dis = vertex_colliding_triangles_min_dist[v_index]
    for col in range(vertex_colliding_triangles_buffer_size[v_index]):
        vertex_index = vertex_colliding_triangles[2 * (offset + col)]
        tri_index = vertex_colliding_triangles[2 * (offset + col) + 1]
        if col < num_cols:
            t1 = tri_indices[tri_index, 0]
            t2 = tri_indices[tri_index, 1]
            t3 = tri_indices[tri_index, 2]
            # wp.expect_eq(vertex_on_triangle(v_index, t1, t2, t3), False)

            u1 = pos[t1]
            u2 = pos[t2]
            u3 = pos[t3]

            closest_p, bary, feature_type = triangle_closest_point(u1, u2, u3, v)
            dis = wp.length(closest_p - v)
            wp.expect_eq(dis < query_radius, True)
            wp.expect_eq(dis >= min_dis, True)
            wp.expect_eq(v_index == vertex_colliding_triangles[2 * (offset + col)], True)

            # wp.printf("vertex %d, offset %d, num cols %d, colliding with triangle: %d, dis: %f\n",
            #           v_index, offset, num_cols, tri_index, dis)
        else:
            wp.expect_eq(vertex_index == -1, True)
            wp.expect_eq(tri_index == -1, True)


@wp.kernel
def validate_triangle_collisions(
    query_radius: float,
    bvh_id: wp.uint64,
    pos: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.int32, ndim=2),
    triangle_colliding_vertices: wp.array(dtype=wp.int32),
    triangle_colliding_vertices_count: wp.array(dtype=wp.int32),
    triangle_colliding_vertices_buffer_offsets: wp.array(dtype=wp.int32),
    triangle_colliding_vertices_buffer_sizes: wp.array(dtype=wp.int32),
    triangle_colliding_vertices_min_dist: wp.array(dtype=float),
    resize_flags: wp.array(dtype=wp.int32),
):
    tri_index = wp.tid()

    t1 = tri_indices[tri_index, 0]
    t2 = tri_indices[tri_index, 1]
    t3 = tri_indices[tri_index, 2]
    # wp.expect_eq(vertex_on_triangle(v_index, t1, t2, t3), False)

    u1 = pos[t1]
    u2 = pos[t2]
    u3 = pos[t3]

    num_cols = triangle_colliding_vertices_count[tri_index]
    offset = triangle_colliding_vertices_buffer_offsets[tri_index]
    min_dis = triangle_colliding_vertices_min_dist[tri_index]
    for col in range(wp.min(num_cols, triangle_colliding_vertices_buffer_sizes[tri_index])):
        v_index = triangle_colliding_vertices[offset + col]
        v = pos[v_index]

        closest_p, bary, feature_type = triangle_closest_point(u1, u2, u3, v)
        dis = wp.length(closest_p - v)
        wp.expect_eq(dis < query_radius, True)
        wp.expect_eq(dis >= min_dis, True)

        # wp.printf("vertex %d, offset %d, num cols %d, colliding with triangle: %d, dis: %f\n",
        #           v_index, offset, num_cols, tri_index, dis)


@wp.kernel
def edge_edge_collision_detection_brute_force(
    query_radius: float,
    bvh_id: wp.uint64,
    pos: wp.array(dtype=wp.vec3),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    edge_colliding_edges_offsets: wp.array(dtype=wp.int32),
    edge_colliding_edges_buffer_sizes: wp.array(dtype=wp.int32),
    edge_edge_parallel_epsilon: float,
    # outputs
    edge_colliding_edges: wp.array(dtype=wp.int32),
    edge_colliding_edges_count: wp.array(dtype=wp.int32),
    edge_colliding_edges_min_dist: wp.array(dtype=float),
    resize_flags: wp.array(dtype=wp.int32),
):
    e_index = wp.tid()

    e0_v0 = edge_indices[e_index, 2]
    e0_v1 = edge_indices[e_index, 3]

    e0_v0_pos = pos[e0_v0]
    e0_v1_pos = pos[e0_v1]

    min_dis_to_edges = query_radius
    edge_num_collisions = wp.int32(0)
    for e1_index in range(edge_indices.shape[0]):
        e1_v0 = edge_indices[e1_index, 2]
        e1_v1 = edge_indices[e1_index, 3]

        if e0_v0 == e1_v0 or e0_v0 == e1_v1 or e0_v1 == e1_v0 or e0_v1 == e1_v1:
            continue

        e1_v0_pos = pos[e1_v0]
        e1_v1_pos = pos[e1_v1]

        st = wp.closest_point_edge_edge(e0_v0_pos, e0_v1_pos, e1_v0_pos, e1_v1_pos, edge_edge_parallel_epsilon)
        s = st[0]
        t = st[1]
        c1 = e0_v0_pos + (e0_v1_pos - e0_v0_pos) * s
        c2 = e1_v0_pos + (e1_v1_pos - e1_v0_pos) * t

        dist = wp.length(c1 - c2)
        if dist < query_radius:
            edge_buffer_offset = edge_colliding_edges_offsets[e_index]
            edge_buffer_size = edge_colliding_edges_offsets[e_index + 1] - edge_buffer_offset

            # record e-e collision to e0, and leave e1; e1 will detect this collision from its own thread
            min_dis_to_edges = wp.min(min_dis_to_edges, dist)
            if edge_num_collisions < edge_buffer_size:
                edge_colliding_edges[edge_buffer_offset + edge_num_collisions] = e1_index
            else:
                resize_flags[1] = 1

            edge_num_collisions = edge_num_collisions + 1

    edge_colliding_edges_count[e_index] = edge_num_collisions
    edge_colliding_edges_min_dist[e_index] = min_dis_to_edges


@wp.kernel
def validate_edge_collisions(
    query_radius: float,
    bvh_id: wp.uint64,
    pos: wp.array(dtype=wp.vec3),
    edge_indices: wp.array(dtype=wp.int32, ndim=2),
    edge_colliding_edges_offsets: wp.array(dtype=wp.int32),
    edge_colliding_edges_buffer_sizes: wp.array(dtype=wp.int32),
    edge_edge_parallel_epsilon: float,
    # outputs
    edge_colliding_edges: wp.array(dtype=wp.int32),
    edge_colliding_edges_count: wp.array(dtype=wp.int32),
    edge_colliding_edges_min_dist: wp.array(dtype=float),
    resize_flags: wp.array(dtype=wp.int32),
):
    e0_index = wp.tid()

    e0_v0 = edge_indices[e0_index, 2]
    e0_v1 = edge_indices[e0_index, 3]

    e0_v0_pos = pos[e0_v0]
    e0_v1_pos = pos[e0_v1]

    num_cols = edge_colliding_edges_count[e0_index]
    offset = edge_colliding_edges_offsets[e0_index]
    min_dist = edge_colliding_edges_min_dist[e0_index]
    for col in range(edge_colliding_edges_buffer_sizes[e0_index]):
        e1_index = edge_colliding_edges[2 * (offset + col) + 1]

        if col < num_cols:
            e1_v0 = edge_indices[e1_index, 2]
            e1_v1 = edge_indices[e1_index, 3]

            if e0_v0 == e1_v0 or e0_v0 == e1_v1 or e0_v1 == e1_v0 or e0_v1 == e1_v1:
                wp.expect_eq(False, True)

            e1_v0_pos = pos[e1_v0]
            e1_v1_pos = pos[e1_v1]

            st = wp.closest_point_edge_edge(e0_v0_pos, e0_v1_pos, e1_v0_pos, e1_v1_pos, edge_edge_parallel_epsilon)
            s = st[0]
            t = st[1]
            c1 = e0_v0_pos + (e0_v1_pos - e0_v0_pos) * s
            c2 = e1_v0_pos + (e1_v1_pos - e1_v0_pos) * t

            dist = wp.length(c2 - c1)

            wp.expect_eq(dist >= min_dist, True)
            wp.expect_eq(e0_index == edge_colliding_edges[2 * (offset + col)], True)
        else:
            wp.expect_eq(e1_index == -1, True)
            wp.expect_eq(edge_colliding_edges[2 * (offset + col)] == -1, True)


def init_model(vs, fs, device, record_triangle_contacting_vertices=True):
    vertices = [wp.vec3(v) for v in vs]

    builder = wp.sim.ModelBuilder()
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0, 200.0, 0.0),
        rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.0),
        scale=1.0,
        vertices=vertices,
        indices=fs,
        vel=wp.vec3(0.0, 0.0, 0.0),
        density=0.02,
        tri_ke=0,
        tri_ka=0,
        tri_kd=0,
    )
    model = builder.finalize(device=device)

    collision_detector = TriMeshCollisionDetector(model=model, record_triangle_contacting_vertices=True)

    return model, collision_detector


def get_data():
    from pxr import Usd, UsdGeom

    usd_stage = Usd.Stage.Open(os.path.join(warp.examples.get_asset_directory(), "bunny.usd"))
    usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/root/bunny"))

    vertices = np.array(usd_geom.GetPointsAttr().Get())
    faces = np.array(usd_geom.GetFaceVertexIndicesAttr().Get())

    return vertices, faces


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
def test_vertex_triangle_collision(test, device):
    vertices, faces = get_data()

    # record triangle contacting vertices
    model, collision_detector = init_model(vertices, faces, device, True)

    rs = [1e-2, 2e-2, 5e-2, 1e-1]

    for query_radius in rs:
        collision_detector.vertex_triangle_collision_detection(query_radius)
        vertex_colliding_triangles_count_1 = collision_detector.vertex_colliding_triangles_count.numpy()
        vertex_min_dis_1 = collision_detector.vertex_colliding_triangles_min_dist.numpy()

        triangle_colliding_vertices_count_1 = collision_detector.triangle_colliding_vertices_count.numpy()
        triangle_min_dis_1 = collision_detector.triangle_colliding_vertices_min_dist.numpy()

        wp.launch(
            kernel=validate_vertex_collisions,
            inputs=[
                query_radius,
                collision_detector.bvh_tris.id,
                collision_detector.model.particle_q,
                collision_detector.model.tri_indices,
                collision_detector.vertex_colliding_triangles,
                collision_detector.vertex_colliding_triangles_count,
                collision_detector.vertex_colliding_triangles_offsets,
                collision_detector.vertex_colliding_triangles_buffer_sizes,
                collision_detector.vertex_colliding_triangles_min_dist,
                collision_detector.resize_flags,
            ],
            dim=model.particle_count,
            device=device,
        )

        wp.launch(
            kernel=validate_triangle_collisions,
            inputs=[
                query_radius,
                collision_detector.bvh_tris.id,
                collision_detector.model.particle_q,
                collision_detector.model.tri_indices,
                collision_detector.triangle_colliding_vertices,
                collision_detector.triangle_colliding_vertices_count,
                collision_detector.triangle_colliding_vertices_offsets,
                collision_detector.triangle_colliding_vertices_buffer_sizes,
                collision_detector.triangle_colliding_vertices_min_dist,
                collision_detector.resize_flags,
            ],
            dim=model.tri_count,
            device=model.device,
        )

        wp.launch(
            kernel=init_triangle_collision_data_kernel,
            inputs=[
                query_radius,
                collision_detector.triangle_colliding_vertices_count,
                collision_detector.triangle_colliding_vertices_min_dist,
                collision_detector.resize_flags,
            ],
            dim=model.tri_count,
            device=model.device,
        )

        wp.launch(
            kernel=vertex_triangle_collision_detection_brute_force,
            inputs=[
                query_radius,
                collision_detector.bvh_tris.id,
                collision_detector.model.particle_q,
                collision_detector.model.tri_indices,
                collision_detector.vertex_colliding_triangles,
                collision_detector.vertex_colliding_triangles_count,
                collision_detector.vertex_colliding_triangles_offsets,
                collision_detector.vertex_colliding_triangles_buffer_sizes,
                collision_detector.vertex_colliding_triangles_min_dist,
                collision_detector.triangle_colliding_vertices,
                collision_detector.triangle_colliding_vertices_count,
                collision_detector.triangle_colliding_vertices_offsets,
                collision_detector.triangle_colliding_vertices_buffer_sizes,
                collision_detector.triangle_colliding_vertices_min_dist,
                collision_detector.resize_flags,
            ],
            dim=model.particle_count,
            device=model.device,
        )

        vertex_colliding_triangles_count_2 = collision_detector.vertex_colliding_triangles_count.numpy()
        vertex_min_dis_2 = collision_detector.vertex_colliding_triangles_min_dist.numpy()

        triangle_colliding_vertices_count_2 = collision_detector.triangle_colliding_vertices_count.numpy()
        triangle_min_dis_2 = collision_detector.triangle_colliding_vertices_min_dist.numpy()

        assert_np_equal(vertex_colliding_triangles_count_2, vertex_colliding_triangles_count_1)
        assert_np_equal(triangle_min_dis_2, triangle_min_dis_1)
        assert_np_equal(triangle_colliding_vertices_count_2, triangle_colliding_vertices_count_1)
        assert_np_equal(vertex_min_dis_2, vertex_min_dis_1)

        # do not record triangle contacting vertices
        model, collision_detector = init_model(vertices, faces, device, False)

        rs = [1e-2, 2e-2, 5e-2, 1e-1]

    for query_radius in rs:
        collision_detector.vertex_triangle_collision_detection(query_radius)
        vertex_colliding_triangles_count_1 = collision_detector.vertex_colliding_triangles_count.numpy()
        vertex_min_dis_1 = collision_detector.vertex_colliding_triangles_min_dist.numpy()

        triangle_min_dis_1 = collision_detector.triangle_colliding_vertices_min_dist.numpy()

        wp.launch(
            kernel=validate_vertex_collisions,
            inputs=[
                query_radius,
                collision_detector.bvh_tris.id,
                collision_detector.model.particle_q,
                collision_detector.model.tri_indices,
                collision_detector.vertex_colliding_triangles,
                collision_detector.vertex_colliding_triangles_count,
                collision_detector.vertex_colliding_triangles_offsets,
                collision_detector.vertex_colliding_triangles_buffer_sizes,
                collision_detector.vertex_colliding_triangles_min_dist,
                collision_detector.resize_flags,
            ],
            dim=model.particle_count,
            device=device,
        )

        wp.launch(
            kernel=vertex_triangle_collision_detection_brute_force_no_triangle_buffers,
            inputs=[
                query_radius,
                collision_detector.bvh_tris.id,
                collision_detector.model.particle_q,
                collision_detector.model.tri_indices,
                collision_detector.vertex_colliding_triangles,
                collision_detector.vertex_colliding_triangles_count,
                collision_detector.vertex_colliding_triangles_offsets,
                collision_detector.vertex_colliding_triangles_buffer_sizes,
                collision_detector.vertex_colliding_triangles_min_dist,
                collision_detector.triangle_colliding_vertices_min_dist,
                collision_detector.resize_flags,
            ],
            dim=model.particle_count,
            device=model.device,
        )

        vertex_colliding_triangles_count_2 = collision_detector.vertex_colliding_triangles_count.numpy()
        vertex_min_dis_2 = collision_detector.vertex_colliding_triangles_min_dist.numpy()
        triangle_min_dis_2 = collision_detector.triangle_colliding_vertices_min_dist.numpy()

        assert_np_equal(vertex_colliding_triangles_count_2, vertex_colliding_triangles_count_1)
        assert_np_equal(triangle_min_dis_2, triangle_min_dis_1)
        assert_np_equal(vertex_min_dis_2, vertex_min_dis_1)


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
def test_edge_edge_collision(test, device):
    vertices, faces = get_data()

    model, collision_detector = init_model(vertices, faces, device)

    rs = [1e-2, 2e-2, 5e-2, 1e-1]
    edge_edge_parallel_epsilon = 1e-5

    for query_radius in rs:
        collision_detector.edge_edge_collision_detection(query_radius)
        edge_colliding_edges_count_1 = collision_detector.edge_colliding_edges_count.numpy()
        edge_min_dist_1 = collision_detector.edge_colliding_edges_min_dist.numpy()

        wp.launch(
            kernel=validate_edge_collisions,
            inputs=[
                query_radius,
                collision_detector.bvh_edges.id,
                collision_detector.model.particle_q,
                collision_detector.model.edge_indices,
                collision_detector.edge_colliding_edges_offsets,
                collision_detector.edge_colliding_edges_buffer_sizes,
                edge_edge_parallel_epsilon,
            ],
            outputs=[
                collision_detector.edge_colliding_edges,
                collision_detector.edge_colliding_edges_count,
                collision_detector.edge_colliding_edges_min_dist,
                collision_detector.resize_flags,
            ],
            dim=model.particle_count,
            device=device,
        )

        wp.launch(
            kernel=edge_edge_collision_detection_brute_force,
            inputs=[
                query_radius,
                collision_detector.bvh_edges.id,
                collision_detector.model.particle_q,
                collision_detector.model.edge_indices,
                collision_detector.edge_colliding_edges_offsets,
                collision_detector.edge_colliding_edges_buffer_sizes,
                edge_edge_parallel_epsilon,
            ],
            outputs=[
                collision_detector.edge_colliding_edges,
                collision_detector.edge_colliding_edges_count,
                collision_detector.edge_colliding_edges_min_dist,
                collision_detector.resize_flags,
            ],
            dim=model.edge_count,
            device=device,
        )

        edge_colliding_edges_count_2 = collision_detector.edge_colliding_edges_count.numpy()
        edge_min_dist_2 = collision_detector.edge_colliding_edges_min_dist.numpy()

        assert_np_equal(edge_colliding_edges_count_2, edge_colliding_edges_count_1)
        assert_np_equal(edge_min_dist_1, edge_min_dist_2)


def test_particle_collision(test, device):
    with wp.ScopedDevice(device):
        contact_radius = 1.23
        builder1 = wp.sim.ModelBuilder()
        builder1.add_cloth_grid(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=100,
            dim_y=100,
            cell_x=0.1,
            cell_y=0.1,
            mass=0.1,
            particle_radius=contact_radius,
        )

    cloth_grid = builder1.finalize()
    cloth_grid_particle_radius = cloth_grid.particle_radius.numpy()
    assert_np_equal(cloth_grid_particle_radius, np.full(cloth_grid_particle_radius.shape, contact_radius), tol=1e-5)

    vertices = [
        [2.0, 0.0, 0.0],
        [2.0, 2.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
    ]
    vertices = [wp.vec3(v) for v in vertices]
    faces = [0, 1, 2, 3, 4, 5]

    builder2 = wp.sim.ModelBuilder()
    builder2.add_cloth_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.0),
        scale=1.0,
        vertices=vertices,
        indices=faces,
        tri_ke=1e4,
        tri_ka=1e4,
        tri_kd=1e-5,
        edge_ke=10,
        edge_kd=0.0,
        vel=wp.vec3(0.0, 0.0, 0.0),
        density=0.1,
        particle_radius=contact_radius,
    )
    cloth_mesh = builder2.finalize()
    cloth_mesh_particle_radius = cloth_mesh.particle_radius.numpy()
    assert_np_equal(cloth_mesh_particle_radius, np.full(cloth_mesh_particle_radius.shape, contact_radius), tol=1e-5)

    state = cloth_mesh.state()
    particle_f = wp.zeros_like(state.particle_q)
    wp.launch(
        kernel=eval_triangles_contact,
        dim=cloth_mesh.tri_count * cloth_mesh.particle_count,
        inputs=[
            cloth_mesh.particle_count,
            state.particle_q,
            state.particle_qd,
            cloth_mesh.tri_indices,
            cloth_mesh.tri_materials,
            cloth_mesh.particle_radius,
        ],
        outputs=[particle_f],
    )
    test.assertTrue((np.linalg.norm(particle_f.numpy(), axis=1) != 0).all())

    builder3 = wp.sim.ModelBuilder()
    builder3.add_cloth_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.0),
        scale=1.0,
        vertices=vertices,
        indices=faces,
        tri_ke=1e4,
        tri_ka=1e4,
        tri_kd=1e-5,
        edge_ke=10,
        edge_kd=0.0,
        vel=wp.vec3(0.0, 0.0, 0.0),
        density=0.1,
        particle_radius=0.5,
    )
    cloth_mesh_2 = builder3.finalize()
    cloth_mesh_2_particle_radius = cloth_mesh_2.particle_radius.numpy()
    assert_np_equal(cloth_mesh_2_particle_radius, np.full(cloth_mesh_2_particle_radius.shape, 0.5), tol=1e-5)

    state_2 = cloth_mesh_2.state()
    particle_f_2 = wp.zeros_like(cloth_mesh_2.particle_q)
    wp.launch(
        kernel=eval_triangles_contact,
        dim=cloth_mesh_2.tri_count * cloth_mesh_2.particle_count,
        inputs=[
            cloth_mesh_2.particle_count,
            state_2.particle_q,
            state_2.particle_qd,
            cloth_mesh_2.tri_indices,
            cloth_mesh_2.tri_materials,
            cloth_mesh_2.particle_radius,
        ],
        outputs=[particle_f_2],
    )
    test.assertTrue((np.linalg.norm(particle_f_2.numpy(), axis=1) == 0).all())


def test_mesh_ground_collision_index(test, device):
    # create a mesh with 1 triangle for testing
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.5, 2.0, 0.0],
        ]
    )
    mesh = wp.sim.Mesh(vertices=vertices, indices=[0, 1, 2])
    builder = wp.sim.ModelBuilder()
    # create body with nonzero mass to ensure it is not static
    # and contact points will be computed
    b = builder.add_body(m=1.0)
    builder.add_shape_mesh(
        body=b,
        mesh=mesh,
        has_shape_collision=False,
    )
    # add another mesh that is not in contact
    b2 = builder.add_body(m=1.0, origin=wp.transform((0.0, 3.0, 0.0), wp.quat_identity()))
    builder.add_shape_mesh(
        body=b2,
        mesh=mesh,
        has_shape_collision=False,
    )
    model = builder.finalize(device=device)
    test.assertEqual(model.rigid_contact_max, 6)
    test.assertEqual(model.shape_contact_pair_count, 0)
    test.assertEqual(model.shape_ground_contact_pair_count, 2)
    model.ground = True
    # ensure all the mesh vertices will be within the contact margin
    model.rigid_contact_margin = 2.0
    state = model.state()
    wp.sim.collide(model, state)
    test.assertEqual(model.rigid_contact_count.numpy()[0], 3)
    tids = model.rigid_contact_tids.list()
    test.assertEqual(sorted(tids), [-1, -1, -1, 0, 1, 2])
    tids = [t for t in tids if t != -1]
    # retrieve the mesh vertices from the contact thread indices
    assert_np_equal(model.rigid_contact_point0.numpy()[:3], vertices[tids])
    assert_np_equal(model.rigid_contact_point1.numpy()[:3, 0], vertices[tids, 0])
    assert_np_equal(model.rigid_contact_point1.numpy()[:3, 1:], np.zeros((3, 2)))
    assert_np_equal(model.rigid_contact_normal.numpy()[:3], np.tile([0.0, 1.0, 0.0], (3, 1)))


devices = get_test_devices(mode="basic")


class TestCollision(unittest.TestCase):
    pass


add_function_test(TestCollision, "test_vertex_triangle_collision", test_vertex_triangle_collision, devices=devices)
add_function_test(TestCollision, "test_edge_edge_collision", test_edge_edge_collision, devices=devices)
add_function_test(TestCollision, "test_particle_collision", test_particle_collision, devices=devices)
add_function_test(TestCollision, "test_mesh_ground_collision_index", test_mesh_ground_collision_index, devices=devices)

if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=True)
