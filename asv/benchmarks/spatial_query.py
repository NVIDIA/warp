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

import importlib.util
import itertools
import os

import numpy as np
from asv_runner.benchmarks.mark import skip_benchmark_if, skip_for_params
from numpy.random import default_rng

import warp as wp
import warp.examples

pxr = importlib.util.find_spec("pxr")
USD_AVAILABLE = pxr is not None

wp.set_module_options({"enable_backward": False})

NUM_QUERY_POINTS = 1000000
NUM_TRIES = 10
NUM_MESHES = 10
seed = 42


@wp.kernel
def sample_mesh_query_no_sign(
    mesh: wp.uint64,
    query_points: wp.array(dtype=wp.vec3),
    query_d_max: float,
    query_closest_points: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    p = query_points[tid]
    query = wp.mesh_query_point_no_sign(mesh, p, query_d_max)

    if query.result:
        face = query.face
        cp = wp.vec3(float(face), query.u, query.v)
        query_closest_points[tid] = cp


class MeshQuery:
    params = [[1, 2, 4, 8], ["bunny", "bear", "rocks"]]
    param_names = ["leaf_size", "asset"]
    number = 20
    timeout = 60

    def setup(self, leaf_size, asset):
        from pxr import Usd, UsdGeom

        wp.init()
        wp.build.clear_kernel_cache()
        self.device = wp.get_device("cuda:0")
        wp.load_module(device=self.device)

        asset_stage = Usd.Stage.Open(os.path.join(wp.examples.get_asset_directory(), f"{asset}.usd"))
        mesh_geom = UsdGeom.Mesh(asset_stage.GetPrimAtPath(f"/root/{asset}"))

        points = np.array(mesh_geom.GetPointsAttr().Get())
        indices = np.array(mesh_geom.GetFaceVertexIndicesAttr().Get())
        bounding_box = np.array([points.min(axis=0), points.max(axis=0)])

        global seed
        rng = np.random.default_rng(seed)
        seed = seed + 1

        query_points_np = rng.uniform(bounding_box[0, :], bounding_box[1, :], size=(NUM_QUERY_POINTS, 3)).astype(
            np.float32
        )
        self.query_points = wp.array(query_points_np, dtype=wp.vec3, device=self.device)

        # create wp mesh
        self.mesh = wp.Mesh(
            points=wp.array(points, dtype=wp.vec3, device=self.device),
            velocities=None,
            indices=wp.array(indices, dtype=int, device=self.device),
            bvh_leaf_size=leaf_size,
        )

        self.query_closest_points = wp.empty_like(self.query_points, device=self.device)

        self.cmd = wp.launch(
            sample_mesh_query_no_sign,
            dim=(NUM_QUERY_POINTS,),
            inputs=[self.mesh.id, self.query_points, 1.0e7, self.query_closest_points],
            device=self.device,
            record_cmd=True,
        )
        # Warmup
        self.cmd.launch()
        wp.synchronize_device(self.device)

    @skip_benchmark_if(USD_AVAILABLE is False)
    def time_mesh_query_closest_point(self, leaf_size, asset):
        self.cmd.launch()
        wp.synchronize_device(self.device)


@wp.struct
class Camera:
    """Basic camera for ray casting"""

    horizontal: float
    vertical: float
    aspect: float
    e: float
    tan: float
    pos: wp.vec3
    rot: wp.quat


@wp.struct
class DirectionalLights:
    """Stores arrays of directional light directions and intensities."""

    dirs: wp.array(dtype=wp.vec3)
    intensities: wp.array(dtype=float)
    num_lights: int


def get_v_t_collision_kernel(is_wp_bvh):
    @wp.kernel
    def vertex_triangle_collision_detection_kernel_broad_only_base(
        query_radius: float,
        geom_id: wp.uint64,
        pos: wp.array(dtype=wp.vec3),
        vertex_colliding_triangles_offsets: wp.array(dtype=wp.int32),
        # outputs
        vertex_colliding_triangles: wp.array(dtype=wp.int32),
        vertex_colliding_triangles_count: wp.array(dtype=wp.int32),
    ):
        v_index = wp.tid()
        v = pos[v_index]
        vertex_buffer_offset = vertex_colliding_triangles_offsets[v_index]
        vertex_buffer_size = vertex_colliding_triangles_offsets[v_index + 1] - vertex_buffer_offset

        lower = wp.vec3(v[0] - query_radius, v[1] - query_radius, v[2] - query_radius)
        upper = wp.vec3(v[0] + query_radius, v[1] + query_radius, v[2] + query_radius)

        tri_index = wp.int32(0)
        vertex_num_collisions = wp.int32(0)

        if wp.static(is_wp_bvh):
            query = wp.bvh_query_aabb(geom_id, lower, upper)
            while wp.bvh_query_next(query, tri_index):
                if vertex_num_collisions < vertex_buffer_size:
                    vertex_colliding_triangles[2 * (vertex_buffer_offset + vertex_num_collisions)] = v_index
                    vertex_colliding_triangles[2 * (vertex_buffer_offset + vertex_num_collisions) + 1] = tri_index

                vertex_num_collisions = vertex_num_collisions + 1
        else:
            query = wp.mesh_query_aabb(geom_id, lower, upper)
            while wp.mesh_query_aabb_next(query, tri_index):
                if vertex_num_collisions < vertex_buffer_size:
                    vertex_colliding_triangles[2 * (vertex_buffer_offset + vertex_num_collisions)] = v_index
                    vertex_colliding_triangles[2 * (vertex_buffer_offset + vertex_num_collisions) + 1] = tri_index

                vertex_num_collisions = vertex_num_collisions + 1

        vertex_colliding_triangles_count[v_index] = vertex_num_collisions

    return vertex_triangle_collision_detection_kernel_broad_only_base


def get_ray_query_kernel(is_wp_bvh):
    @wp.kernel
    def ray_vs_aabb_query_kernel_base(
        geom_id: wp.uint64,
        camera: Camera,
        mesh_pos: wp.array(dtype=wp.vec3),
        mesh_rot: wp.array(dtype=wp.quat),
        rays_width: int,
        rays_height: int,
        rays: wp.array(dtype=wp.vec3),
    ):
        tid = wp.tid()

        x = tid % rays_width
        y = rays_height - 1 - tid // rays_width

        sx = 2.0 * float(x) / float(rays_width) - 1.0
        sy = 2.0 * float(y) / float(rays_height) - 1.0

        # compute view ray in world space
        ro_world = camera.pos
        rd_world = wp.normalize(
            wp.quat_rotate(camera.rot, wp.vec3(sx * camera.tan * camera.aspect, sy * camera.tan, -1.0))
        )

        # compute view ray in mesh space
        inv = wp.transform_inverse(wp.transform(mesh_pos[0], mesh_rot[0]))
        ro = wp.transform_point(inv, ro_world)
        rd = wp.transform_vector(inv, rd_world)

        color = wp.vec3(0.0, 0.0, 0.0)

        index = wp.int(0)

        if wp.static(is_wp_bvh):
            query = wp.bvh_query_ray(
                geom_id,
                ro,
                rd,
            )
            while wp.bvh_query_next(query, index):
                color += wp.vec3(1.0, 1.0, 1.0)
        else:
            query = wp.mesh_query_ray(geom_id, ro, rd, 1.0e6)
            if query.result:
                color += wp.vec3(1.0, 1.0, 1.0)

        rays[tid] = color

    return ray_vs_aabb_query_kernel_base


def _rand_rotation(rand_eng, rotation_sigma_rad: float) -> np.ndarray:
    if rotation_sigma_rad <= 0.0:
        return np.eye(3)
    axis = rand_eng.normal(size=3)
    angle = rand_eng.normal(loc=0.0, scale=rotation_sigma_rad)

    rot_mat_wp = wp.quat_to_matrix(wp.quat_from_axis_angle(wp.vec3(axis), angle))

    return np.array(rot_mat_wp).reshape(3, 3)


@wp.kernel
def compute_tri_aabbs(
    pos: wp.array(dtype=wp.vec3),
    tri_indices: wp.array(dtype=wp.int32, ndim=1),
    lower_bounds: wp.array(dtype=wp.vec3),
    upper_bounds: wp.array(dtype=wp.vec3),
):
    t_id = wp.tid()

    v1 = pos[tri_indices[t_id * 3]]
    v2 = pos[tri_indices[t_id * 3 + 1]]
    v3 = pos[tri_indices[t_id * 3 + 2]]

    lower_bounds[t_id] = wp.min(wp.min(v1, v2), v3)
    upper_bounds[t_id] = wp.max(wp.max(v1, v2), v3)


def replicate_mesh_with_random_perturbation(
    points: np.ndarray,  # (N,3) float
    faces: np.ndarray,  # (M,3) int
    K: int,
    rand_eng,
    *,
    translation_sigma: float = 0.9,  # stddev of xyz translation
    rotation_sigma_rad: float = 0.25,  # stddev of rotation angle (radians)
    scale_sigma: float = 0.05,  # stddev for uniform scale around 1.0
    vertex_noise_sigma: float = 0.0,  # per-vertex jitter (pre-transform)
    include_original: bool = False,  # also keep the unperturbed original
) -> tuple[np.ndarray, np.ndarray]:
    """
    Make K randomized copies of a triangle mesh and merge into one mesh.
    Each copy gets:
      - optional per-vertex Gaussian jitter (vertex_noise_sigma)
      - a random rigid transform (rotation + translation)
      - a random uniform scale (1 + N(0, scale_sigma)), clipped to >0
    Returns (merged_points, merged_faces).
    """
    points = np.asarray(points, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be (N,3).")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must be (M,3).")
    N = points.shape[0]

    copies = []
    faces_all = []

    offset = 0

    if include_original:
        copies.append(points.copy())
        faces_all.append(faces.copy() + offset)
        offset += N

    for _ in range(K):
        pts = points.copy()

        # per-vertex jitter (optional)
        if vertex_noise_sigma > 0.0:
            pts += rand_eng.normal(scale=vertex_noise_sigma, size=pts.shape)

        # random transform
        R = _rand_rotation(rand_eng, rotation_sigma_rad)
        s = 1.0 + rand_eng.normal(loc=0.0, scale=scale_sigma) if scale_sigma > 0 else 1.0
        s = max(s, 1e-8)  # avoid nonpositive/zero scale
        t = rand_eng.normal(loc=0.0, scale=translation_sigma, size=3)

        # apply: row-vectors => pts @ R^T
        pts = (pts @ R.T) * s + t

        copies.append(pts)
        faces_all.append(faces.copy() + offset)
        offset += N

    merged_points = np.vstack(copies) if copies else points.copy()
    merged_faces = np.vstack(faces_all) if faces_all else faces.copy()

    return merged_points, merged_faces


class BvhAABBQuery:
    params = [[0.002, 0.004, 0.008], [1, 2, 4], ["cpu", "cuda"]]
    param_names = ["query_radius", "leaf_size", "device"]

    number = 5
    timeout = 120

    def setup(self, query_radius, leaf_size, device):
        with wp.ScopedDevice(device):
            from pxr import Usd, UsdGeom

            global seed

            rand_eng = default_rng(seed)  # or default_rng() for non-deterministic
            seed = seed + 1

            wp.init()
            wp.build.clear_kernel_cache()
            self.device = wp.get_device(device)
            wp.load_module(device=self.device)

            asset_stage = Usd.Stage.Open(os.path.join(wp.examples.get_asset_directory(), "bunny.usd"))
            mesh_geom = UsdGeom.Mesh(asset_stage.GetPrimAtPath("/root/bunny"))

            points_base = np.array(mesh_geom.GetPointsAttr().Get())
            indices_base = np.array(mesh_geom.GetFaceVertexIndicesAttr().Get())

            if NUM_MESHES != 1:
                points, indices = replicate_mesh_with_random_perturbation(
                    points_base, indices_base.reshape((-1, 3)), NUM_MESHES, rand_eng
                )
                indices = indices.reshape(-1)
            else:
                points = points_base
                indices = indices_base

            bounding_box = np.array([points.min(axis=0), points.max(axis=0)])

            self.points = wp.array(points, dtype=wp.vec3)
            self.indices = wp.array(indices, dtype=int)

            bb_min = bounding_box[0]
            bb_max = bounding_box[1]

            # Option A (no extra cast, if your NumPy supports dtype in Generator.random)
            query_points_np = bb_min + (bb_max - bb_min) * rand_eng.random((NUM_QUERY_POINTS, 3), dtype=np.float32)

            # Option B (always works; uses uniform + cast)

            self.query_points = wp.array(query_points_np, dtype=wp.vec3)

            num_faces = int(indices.shape[0] / 3)
            # create wp Bvh
            self.lowers = wp.zeros(num_faces, dtype=wp.vec3)
            self.uppers = wp.zeros(num_faces, dtype=wp.vec3)

            wp.launch(
                dim=num_faces,
                kernel=compute_tri_aabbs,
                inputs=[self.points, self.indices],
                outputs=[self.lowers, self.uppers],
            )

            self.bvh = wp.Bvh(self.lowers, self.uppers, leaf_size=leaf_size)

            self.mesh = wp.Mesh(self.points, wp.array(indices, dtype=int), leaf_size=leaf_size)

            buffer_size_per_vertex = 32
            self.vertex_colliding_triangles_offsets = wp.array(
                np.arange(0, buffer_size_per_vertex * (NUM_QUERY_POINTS + 1), buffer_size_per_vertex, dtype=int),
                dtype=wp.int32,
            )
            self.vertex_colliding_triangles = wp.zeros(2 * buffer_size_per_vertex * NUM_QUERY_POINTS, dtype=wp.int32)
            self.vertex_colliding_triangles_count = wp.zeros(NUM_QUERY_POINTS, dtype=wp.int32)

            self.bvh_vertex_triangle_collision_detection_kernel = get_v_t_collision_kernel(True)
            self.mesh_vertex_triangle_collision_detection_kernel = get_v_t_collision_kernel(False)

            if self.bvh.device.is_cpu:
                self.cmd_bvh = wp.launch(
                    dim=NUM_QUERY_POINTS,
                    kernel=self.bvh_vertex_triangle_collision_detection_kernel,
                    inputs=[
                        query_radius,
                        self.bvh.id,
                        self.query_points,
                        self.vertex_colliding_triangles_offsets,
                    ],
                    outputs=[self.vertex_colliding_triangles, self.vertex_colliding_triangles_count],
                    record_cmd=True,
                )

                self.cmd_mesh = wp.launch(
                    dim=NUM_QUERY_POINTS,
                    kernel=self.mesh_vertex_triangle_collision_detection_kernel,
                    inputs=[
                        query_radius,
                        self.mesh.id,
                        self.query_points,
                        self.vertex_colliding_triangles_offsets,
                    ],
                    outputs=[self.vertex_colliding_triangles, self.vertex_colliding_triangles_count],
                    record_cmd=True,
                )
            else:
                wp.load_module(device=device)
                with wp.ScopedCapture(force_module_load=False) as capture:
                    for _ in range(NUM_TRIES):
                        wp.launch(
                            dim=NUM_QUERY_POINTS,
                            kernel=self.bvh_vertex_triangle_collision_detection_kernel,
                            inputs=[
                                query_radius,
                                self.bvh.id,
                                self.query_points,
                                self.vertex_colliding_triangles_offsets,
                            ],
                            outputs=[self.vertex_colliding_triangles, self.vertex_colliding_triangles_count],
                        )

                self.cuda_graph_bvh_aabb_vs_aabb = capture.graph

                with wp.ScopedCapture(force_module_load=False) as capture:
                    for _ in range(NUM_TRIES):
                        wp.launch(
                            dim=NUM_QUERY_POINTS,
                            kernel=self.mesh_vertex_triangle_collision_detection_kernel,
                            inputs=[
                                query_radius,
                                self.mesh.id,
                                self.query_points,
                                self.vertex_colliding_triangles_offsets,
                            ],
                            outputs=[self.vertex_colliding_triangles, self.vertex_colliding_triangles_count],
                        )

                self.cuda_graph_mesh_aabb_vs_aabb = capture.graph

                # warm up run
                wp.capture_launch(self.cuda_graph_bvh_aabb_vs_aabb)
                wp.capture_launch(self.cuda_graph_mesh_aabb_vs_aabb)
                wp.synchronize_device(self.device)

    @skip_for_params(
        [
            t
            for t in list(
                itertools.product(
                    [
                        0.002,
                        0.004,
                        0.008,
                    ],
                    [1, 2, 4],
                    ["cpu"],
                )
            )
            if t != (0.002, 1, "cpu")
        ]
    )
    @skip_benchmark_if(USD_AVAILABLE is False)
    def time_bvh_aabb_vs_aabb_query(self, query_radius, leaf_size, device):
        if self.bvh.device.is_cpu:
            self.cmd_bvh.launch()
        else:
            wp.capture_launch(self.cuda_graph_bvh_aabb_vs_aabb)
        wp.synchronize_device(self.device)

    @skip_for_params(
        [
            t
            for t in list(
                itertools.product(
                    [
                        0.002,
                        0.004,
                        0.008,
                    ],
                    [1, 2, 4],
                    ["cpu"],
                )
            )
            if t != (0.002, 1, "cpu")
        ]
    )
    @skip_benchmark_if(USD_AVAILABLE is False)
    def time_mesh_aabb_vs_aabb_query(self, query_radius, leaf_size, device):
        if self.bvh.device.is_cpu:
            self.cmd_mesh.launch()
        else:
            wp.capture_launch(self.cuda_graph_mesh_aabb_vs_aabb)
        wp.synchronize_device(self.device)


class BvhRayQuery:
    params = [
        [
            480,
            1080,
        ],
        [1, 2, 4],
        ["cpu", "cuda"],
    ]
    param_names = ["resolution", "leaf_size", "device"]

    number = 5
    timeout = 120

    def setup(self, resolution, leaf_size, device):
        cam_pos = wp.vec3(0.0, 0.75, 7.0)
        cam_rot = wp.quat(0.0, 0.0, 0.0, 1.0)
        horizontal_aperture = 36.0
        vertical_aperture = 36.0
        aspect = horizontal_aperture / vertical_aperture
        focal_length = 50.0

        from pxr import Usd, UsdGeom

        global seed

        self.device = wp.get_device(device)

        with wp.ScopedDevice(self.device):
            rand_eng = default_rng(seed)  # or default_rng() for non-deterministic
            seed = seed + 1

            wp.init()
            wp.build.clear_kernel_cache()
            wp.load_module(device=self.device)

            asset_stage = Usd.Stage.Open(os.path.join(wp.examples.get_asset_directory(), "bunny.usd"))
            mesh_geom = UsdGeom.Mesh(asset_stage.GetPrimAtPath("/root/bunny"))

            points_base = np.array(mesh_geom.GetPointsAttr().Get())
            indices_base = np.array(mesh_geom.GetFaceVertexIndicesAttr().Get())

            if NUM_MESHES != 1:
                points, indices = replicate_mesh_with_random_perturbation(
                    points_base, indices_base.reshape((-1, 3)), NUM_MESHES, rand_eng
                )
                indices = indices.reshape(-1)
            else:
                points = points_base
                indices = indices_base

            self.points = wp.array(points, dtype=wp.vec3)
            self.indices = wp.array(indices, dtype=int)

            bounding_box = np.array([points.min(axis=0), points.max(axis=0)])

            num_faces = int(indices.shape[0] / 3)
            # create wp Bvh
            self.lowers = wp.zeros(num_faces, dtype=wp.vec3)
            self.uppers = wp.zeros(num_faces, dtype=wp.vec3)

            wp.launch(
                dim=num_faces,
                kernel=compute_tri_aabbs,
                inputs=[self.points, self.indices],
                outputs=[self.lowers, self.uppers],
            )

            self.bvh = wp.Bvh(self.lowers, self.uppers, leaf_size=leaf_size)
            self.mesh = wp.Mesh(self.points, wp.array(indices, dtype=int), leaf_size=leaf_size)

            bb_min = bounding_box[0]
            bb_max = bounding_box[1]

            # Option A (no extra cast, if your NumPy supports dtype in Generator.random)
            query_points_np = bb_min + (bb_max - bb_min) * rand_eng.random((NUM_QUERY_POINTS, 3), dtype=np.float32)

            # Option B (always works; uses uniform + cast)

            self.query_points = wp.array(query_points_np, dtype=wp.vec3)

            self.mesh_pos = wp.zeros(1, dtype=wp.vec3, requires_grad=True)
            self.mesh_rot = wp.array(np.array([0.0, 0.0, 0.0, 1.0]), dtype=wp.quat, requires_grad=True)
            num_faces = int(indices.shape[0] / 3)

            # construct camera
            self.camera = Camera()
            self.camera.horizontal = horizontal_aperture
            self.camera.vertical = vertical_aperture
            self.camera.aspect = aspect
            self.camera.e = focal_length
            self.camera.tan = vertical_aperture / (2.0 * focal_length)
            self.camera.pos = cam_pos
            self.camera.rot = cam_rot

            # construct lights
            self.lights = DirectionalLights()
            self.lights.dirs = wp.array(np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), dtype=wp.vec3, requires_grad=True)
            self.lights.intensities = wp.array(np.array([2.0, 0.2]), dtype=float, requires_grad=True)
            self.lights.num_lights = 2

            # construct rays
            self.rays_width = resolution
            self.rays_height = resolution
            self.num_rays = self.rays_width * self.rays_height
            self.rays = wp.zeros(self.num_rays, dtype=wp.vec3, requires_grad=True)

            self.bvh_ray_vs_aabb_query_kernel = get_ray_query_kernel(True)
            self.mesh_ray_vs_aabb_query_kernel = get_ray_query_kernel(False)

            if self.bvh.device.is_cpu:
                self.cmd_bvh_query = wp.launch(
                    dim=self.num_rays,
                    kernel=self.bvh_ray_vs_aabb_query_kernel,
                    inputs=[
                        self.bvh.id,
                        self.camera,
                        self.mesh_pos,
                        self.mesh_rot,
                        self.rays_width,
                        self.rays_height,
                    ],
                    outputs=[
                        self.rays,
                    ],
                    record_cmd=True,
                )
                self.cmd_mesh_query = wp.launch(
                    dim=self.num_rays,
                    kernel=self.mesh_ray_vs_aabb_query_kernel,
                    inputs=[
                        self.mesh.id,
                        self.camera,
                        self.mesh_pos,
                        self.mesh_rot,
                        self.rays_width,
                        self.rays_height,
                    ],
                    outputs=[
                        self.rays,
                    ],
                    record_cmd=True,
                )

            else:
                wp.load_module(device=device)
                with wp.ScopedCapture(force_module_load=False) as capture:
                    for _ in range(NUM_TRIES):
                        wp.launch(
                            dim=self.num_rays,
                            kernel=self.bvh_ray_vs_aabb_query_kernel,
                            inputs=[
                                self.bvh.id,
                                self.camera,
                                self.mesh_pos,
                                self.mesh_rot,
                                self.rays_width,
                                self.rays_height,
                            ],
                            outputs=[
                                self.rays,
                            ],
                        )

                self.cuda_graph_bvh_ray_vs_aabb = capture.graph

                with wp.ScopedCapture(force_module_load=False) as capture:
                    for _ in range(NUM_TRIES):
                        wp.launch(
                            dim=self.num_rays,
                            kernel=self.mesh_ray_vs_aabb_query_kernel,
                            inputs=[
                                self.mesh.id,
                                self.camera,
                                self.mesh_pos,
                                self.mesh_rot,
                                self.rays_width,
                                self.rays_height,
                            ],
                            outputs=[
                                self.rays,
                            ],
                        )
                self.cuda_graph_mesh_ray_vs_aabb = capture.graph

                # warm up run
                wp.capture_launch(self.cuda_graph_bvh_ray_vs_aabb)
                wp.capture_launch(self.cuda_graph_mesh_ray_vs_aabb)
                wp.synchronize_device()

    @skip_for_params([t for t in itertools.product([480, 1080], [1, 2, 4], ["cpu"]) if t != (1080, 1, "cpu")])
    @skip_benchmark_if(USD_AVAILABLE is False)
    def time_bvh_ray_vs_aabb_query(self, resolution, leaf_size, device):
        if self.bvh.device.is_cpu:
            self.cmd_bvh_query.launch()
        else:
            wp.capture_launch(self.cuda_graph_bvh_ray_vs_aabb)
        wp.synchronize_device()

    @skip_for_params([t for t in itertools.product([480, 1080], [1, 2, 4], ["cpu"]) if t != (1080, 1, "cpu")])
    @skip_benchmark_if(USD_AVAILABLE is False)
    def time_mesh_ray_vs_aabb_query(self, resolution, leaf_size, device):
        if self.bvh.device.is_cpu:
            self.cmd_mesh_query.launch()
        else:
            wp.capture_launch(self.cuda_graph_mesh_ray_vs_aabb)
        wp.synchronize_device()
