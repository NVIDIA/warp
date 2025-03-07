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
from warp.sim.collide import *
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
    for col in range(wp.min(num_cols, vertex_colliding_triangles_buffer_size[v_index])):
        tri_index = vertex_colliding_triangles[offset + col]

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

        # wp.printf("vertex %d, offset %d, num cols %d, colliding with triangle: %d, dis: %f\n",
        #           v_index, offset, num_cols, tri_index, dis)


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
    for col in range(wp.min(num_cols, edge_colliding_edges_buffer_sizes[e0_index])):
        e1_index = edge_colliding_edges[offset + col]

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


class Example:
    def __init__(self, device, vs, fs):
        self.device = device

        self.input_scale_factor = 1.0
        self.renderer_scale_factor = 0.01
        vertices = [wp.vec3(v) * self.input_scale_factor for v in vs]

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
        self.model = builder.finalize(device=self.device)

        self.collision_detector = TriMeshCollisionDetector(model=self.model)

    def run_vertex_triangle_test(self):
        rs = [1e-2, 2e-2, 5e-2, 1e-1]
        for query_radius in rs:
            self.collision_detector.vertex_triangle_collision_detection(query_radius)
            vertex_colliding_triangles_count_1 = self.collision_detector.vertex_colliding_triangles_count.numpy()
            vertex_min_dis_1 = self.collision_detector.vertex_colliding_triangles_min_dist.numpy()

            triangle_colliding_vertices_count_1 = self.collision_detector.triangle_colliding_vertices_count.numpy()
            triangle_min_dis_1 = self.collision_detector.triangle_colliding_vertices_min_dist.numpy()

            wp.launch(
                kernel=init_triangle_collision_data_kernel,
                inputs=[
                    query_radius,
                    self.collision_detector.triangle_colliding_vertices_count,
                    self.collision_detector.triangle_colliding_vertices_min_dist,
                    self.collision_detector.resize_flags,
                ],
                dim=self.model.tri_count,
                device=self.model.device,
            )

            wp.launch(
                kernel=vertex_triangle_collision_detection_brute_force,
                inputs=[
                    query_radius,
                    self.collision_detector.bvh_tris.id,
                    self.collision_detector.model.particle_q,
                    self.collision_detector.model.tri_indices,
                    self.collision_detector.vertex_colliding_triangles,
                    self.collision_detector.vertex_colliding_triangles_count,
                    self.collision_detector.vertex_colliding_triangles_offsets,
                    self.collision_detector.vertex_colliding_triangles_buffer_sizes,
                    self.collision_detector.vertex_colliding_triangles_min_dist,
                    self.collision_detector.triangle_colliding_vertices,
                    self.collision_detector.triangle_colliding_vertices_count,
                    self.collision_detector.triangle_colliding_vertices_offsets,
                    self.collision_detector.triangle_colliding_vertices_buffer_sizes,
                    self.collision_detector.triangle_colliding_vertices_min_dist,
                    self.collision_detector.resize_flags,
                ],
                dim=self.model.particle_count,
                device=self.model.device,
            )

            vertex_colliding_triangles_count_2 = self.collision_detector.vertex_colliding_triangles_count.numpy()
            vertex_min_dis_2 = self.collision_detector.vertex_colliding_triangles_min_dist.numpy()

            triangle_colliding_vertices_count_2 = self.collision_detector.triangle_colliding_vertices_count.numpy()
            triangle_min_dis_2 = self.collision_detector.triangle_colliding_vertices_min_dist.numpy()

            assert (vertex_colliding_triangles_count_2 == vertex_colliding_triangles_count_1).all()
            assert (triangle_min_dis_2 == triangle_min_dis_1).all()
            assert (triangle_colliding_vertices_count_2 == triangle_colliding_vertices_count_1).all()
            assert (vertex_min_dis_2 == vertex_min_dis_1).all()

            wp.launch(
                kernel=validate_vertex_collisions,
                inputs=[
                    query_radius,
                    self.collision_detector.bvh_tris.id,
                    self.collision_detector.model.particle_q,
                    self.collision_detector.model.tri_indices,
                    self.collision_detector.vertex_colliding_triangles,
                    self.collision_detector.vertex_colliding_triangles_count,
                    self.collision_detector.vertex_colliding_triangles_offsets,
                    self.collision_detector.vertex_colliding_triangles_buffer_sizes,
                    self.collision_detector.vertex_colliding_triangles_min_dist,
                    self.collision_detector.resize_flags,
                ],
                dim=self.model.particle_count,
                device=self.model.device,
            )

            wp.launch(
                kernel=validate_triangle_collisions,
                inputs=[
                    query_radius,
                    self.collision_detector.bvh_tris.id,
                    self.collision_detector.model.particle_q,
                    self.collision_detector.model.tri_indices,
                    self.collision_detector.triangle_colliding_vertices,
                    self.collision_detector.triangle_colliding_vertices_count,
                    self.collision_detector.triangle_colliding_vertices_offsets,
                    self.collision_detector.triangle_colliding_vertices_buffer_sizes,
                    self.collision_detector.triangle_colliding_vertices_min_dist,
                    self.collision_detector.resize_flags,
                ],
                dim=self.model.tri_count,
                device=self.model.device,
            )

    def run_edge_edge_test(self):
        rs = [1e-2, 2e-2, 5e-2, 1e-1]
        edge_edge_parallel_epsilon = 1e-5
        for query_radius in rs:
            self.collision_detector.edge_edge_collision_detection(query_radius)
            edge_colliding_edges_count_1 = self.collision_detector.edge_colliding_edges_count.numpy()
            edge_min_dist_1 = self.collision_detector.edge_colliding_edges_min_dist.numpy()

            print(edge_colliding_edges_count_1)

            wp.launch(
                kernel=edge_edge_collision_detection_brute_force,
                inputs=[
                    query_radius,
                    self.collision_detector.bvh_edges.id,
                    self.collision_detector.model.particle_q,
                    self.collision_detector.model.edge_indices,
                    self.collision_detector.edge_colliding_edges_offsets,
                    self.collision_detector.edge_colliding_edges_buffer_sizes,
                    edge_edge_parallel_epsilon,
                ],
                outputs=[
                    self.collision_detector.edge_colliding_edges,
                    self.collision_detector.edge_colliding_edges_count,
                    self.collision_detector.edge_colliding_edges_min_dist,
                    self.collision_detector.resize_flags,
                ],
                dim=self.model.edge_count,
                device=self.model.device,
            )

            edge_colliding_edges_count_2 = self.collision_detector.edge_colliding_edges_count.numpy()
            edge_min_dist_2 = self.collision_detector.edge_colliding_edges_min_dist.numpy()

            assert (edge_colliding_edges_count_2 == edge_colliding_edges_count_1).all()
            assert (edge_min_dist_1 == edge_min_dist_2).all()

            wp.launch(
                kernel=validate_edge_collisions,
                inputs=[
                    query_radius,
                    self.collision_detector.bvh_edges.id,
                    self.collision_detector.model.particle_q,
                    self.collision_detector.model.edge_indices,
                    self.collision_detector.edge_colliding_edges_offsets,
                    self.collision_detector.edge_colliding_edges_buffer_sizes,
                    edge_edge_parallel_epsilon,
                ],
                outputs=[
                    self.collision_detector.edge_colliding_edges,
                    self.collision_detector.edge_colliding_edges_count,
                    self.collision_detector.edge_colliding_edges_min_dist,
                    self.collision_detector.resize_flags,
                ],
                dim=self.model.particle_count,
                device=self.model.device,
            )

    def set_points_fixed(self, model, fixed_particles):
        if len(fixed_particles):
            flags = model.particle_flags.numpy()
            for fixed_vertex_id in fixed_particles:
                flags[fixed_vertex_id] = wp.uint32(int(flags[fixed_vertex_id]) & ~int(PARTICLE_FLAG_ACTIVE))

            model.particle_flags = wp.array(flags, device=model.device)


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
def test_vertex_triangle_collision(test, device):
    from pxr import Usd, UsdGeom

    def get_data():
        usd_stage = Usd.Stage.Open(os.path.join(warp.examples.get_asset_directory(), "bunny.usd"))
        usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/root/bunny"))

        vertices = np.array(usd_geom.GetPointsAttr().Get())
        faces = np.array(usd_geom.GetFaceVertexIndicesAttr().Get())

        return vertices, faces

    vertices, faces = get_data()

    sim = Example(device, vertices, faces)
    sim.run_vertex_triangle_test()


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
def test_edge_edge_collision(test, device):
    from pxr import Usd, UsdGeom

    def get_data():
        usd_stage = Usd.Stage.Open(os.path.join(warp.examples.get_asset_directory(), "bunny.usd"))
        usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/root/bunny"))

        vertices = np.array(usd_geom.GetPointsAttr().Get())
        faces = np.array(usd_geom.GetFaceVertexIndicesAttr().Get())

        return vertices, faces

    vertices, faces = get_data()

    sim = Example(device, vertices, faces)
    sim.run_edge_edge_test()


devices = get_test_devices()


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


class TestCollision(unittest.TestCase):
    pass


# add_function_test(TestCollision, "test_vertex_triangle_collision", test_vertex_triangle_collision, devices=devices)
# add_function_test(TestCollision, "test_edge_edge_collision", test_vertex_triangle_collision, devices=devices)
add_function_test(TestCollision, "test_particle_collision", test_particle_collision, devices=devices)

if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
