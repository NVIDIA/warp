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

import itertools
import os
import unittest

import numpy as np

import warp as wp
import warp.examples
from warp.tests.unittest_utils import *


# triangulate a list of polygon face indices
def triangulate(face_counts, face_indices):
    num_tris = np.sum(np.subtract(face_counts, 2))
    num_tri_vtx = num_tris * 3
    tri_indices = np.zeros(num_tri_vtx, dtype=int)
    ctr = 0
    wedgeIdx = 0

    for nb in face_counts:
        for i in range(nb - 2):
            tri_indices[ctr] = face_indices[wedgeIdx]
            tri_indices[ctr + 1] = face_indices[wedgeIdx + i + 1]
            tri_indices[ctr + 2] = face_indices[wedgeIdx + i + 2]
            ctr += 3
        wedgeIdx += nb

    return tri_indices


@wp.kernel
def mesh_query_ray_loss(
    mesh: wp.uint64,
    query_points: wp.array(dtype=wp.vec3),
    query_dirs: wp.array(dtype=wp.vec3),
    intersection_points: wp.array(dtype=wp.vec3),
    loss: wp.array(dtype=float),
):
    tid = wp.tid()

    p = query_points[tid]
    D = query_dirs[tid]

    max_t = 10012.0
    t = float(0.0)
    bary_u = float(0.0)
    bary_v = float(0.0)
    sign = float(0.0)
    normal = wp.vec3()
    face_index = int(0)

    q = wp.vec3()

    if wp.mesh_query_ray(mesh, p, D, max_t, t, bary_u, bary_v, sign, normal, face_index):
        q = wp.mesh_eval_position(mesh, face_index, bary_u, bary_v)

    intersection_points[tid] = q
    l = q[0]
    loss[tid] = l

    query = wp.mesh_query_ray(mesh, p, D, max_t)
    wp.expect_eq(query.t, t)
    wp.expect_eq(query.u, bary_u)
    wp.expect_eq(query.v, bary_v)
    wp.expect_eq(query.sign, sign)
    wp.expect_eq(query.normal, normal)
    wp.expect_eq(query.face, face_index)


@wp.func
def intersect_ray_triangle(
    p: wp.vec3,
    dir: wp.vec3,
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    max_t: float,
):
    eps = 1.0e-6

    e1 = b - a
    e2 = c - a

    pvec = wp.cross(dir, e2)
    det = wp.dot(e1, pvec)

    if wp.abs(det) < eps:
        return max_t

    inv_det = 1.0 / det

    tvec = p - a
    u = wp.dot(tvec, pvec) * inv_det
    if u < 0.0 or u > 1.0:
        return max_t

    qvec = wp.cross(tvec, e1)
    v = wp.dot(dir, qvec) * inv_det
    if v < 0.0 or u + v > 1.0:
        return max_t

    t = wp.dot(e2, qvec) * inv_det
    if t < eps or t > max_t:
        return max_t

    return t


@wp.func
def raycast_brutal(
    points: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=int),
    p: wp.vec3,
    dir: wp.vec3,
    max_t: float,
):
    t_closest = max_t
    face_closest = int(-1)
    num_faces = int(indices.shape[0] / 3)

    for face in range(num_faces):
        i = indices[face * 3 + 0]
        j = indices[face * 3 + 1]
        k = indices[face * 3 + 2]

        a = points[i]
        b = points[j]
        c = points[k]

        t = intersect_ray_triangle(p, dir, a, b, c, max_t)
        if t < t_closest:
            t_closest = t
            face_closest = face

    return face_closest


@wp.kernel
def mesh_query_ray_count_intersections_brutal(
    points: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=int),
    ray_starts: wp.array(dtype=wp.vec3),
    ray_directions: wp.array(dtype=wp.vec3),
    counts: wp.array(dtype=int),
):
    tid = wp.tid()
    p = ray_starts[tid]
    dir = ray_directions[tid]

    # Count all intersections (similar pattern to mesh_query_ray_brutal)
    intersection_count = int(0)
    num_faces = int(indices.shape[0] / 3)
    max_t = 1.0e10

    for face in range(num_faces):
        i = indices[face * 3 + 0]
        j = indices[face * 3 + 1]
        k = indices[face * 3 + 2]

        a = points[i]
        b = points[j]
        c = points[k]

        t = intersect_ray_triangle(p, dir, a, b, c, max_t)
        if t < max_t:
            intersection_count += 1

    counts[tid] = intersection_count


@wp.kernel
def mesh_query_ray_count_intersections_kernel(
    mesh: wp.uint64,
    ray_starts: wp.array(dtype=wp.vec3),
    ray_directions: wp.array(dtype=wp.vec3),
    counts: wp.array(dtype=int),
):
    tid = wp.tid()
    p = ray_starts[tid]
    dir = ray_directions[tid]

    counts[tid] = wp.mesh_query_ray_count_intersections(mesh, p, dir)


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
def test_mesh_query_ray_grad(test, device):
    from pxr import Usd, UsdGeom  # noqa: PLC0415

    # test tri
    # print("Testing Single Triangle")
    # mesh_points = wp.array(np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]), dtype=wp.vec3, device=device)
    # mesh_indices = wp.array(np.array([0,1,2]), dtype=int, device=device)

    mesh = Usd.Stage.Open(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "torus.usda")))
    mesh_geom = UsdGeom.Mesh(mesh.GetPrimAtPath("/World/Torus"))

    mesh_counts = mesh_geom.GetFaceVertexCountsAttr().Get()
    mesh_indices = mesh_geom.GetFaceVertexIndicesAttr().Get()

    tri_indices = triangulate(mesh_counts, mesh_indices)

    mesh_points = wp.array(np.array(mesh_geom.GetPointsAttr().Get()), dtype=wp.vec3, device=device)
    mesh_indices = wp.array(np.array(tri_indices), dtype=int, device=device)

    if device.is_cpu:
        constructors = ["sah", "median"]
    else:
        constructors = ["sah", "median", "lbvh"]

    leaf_sizes = [1, 2, 4]

    for leaf_size, constructor in itertools.product(leaf_sizes, constructors):
        p = wp.vec3(50.0, 50.0, 0.0)
        D = wp.vec3(0.0, -1.0, 0.0)

        # create mesh
        mesh = wp.Mesh(
            points=mesh_points,
            velocities=None,
            indices=mesh_indices,
            bvh_constructor=constructor,
            bvh_leaf_size=leaf_size,
        )

        tape = wp.Tape()

        # analytic gradients
        with tape:
            query_points = wp.array(p, dtype=wp.vec3, device=device, requires_grad=True)
            query_dirs = wp.array(D, dtype=wp.vec3, device=device, requires_grad=True)
            intersection_points = wp.zeros(n=1, dtype=wp.vec3, device=device)
            loss = wp.zeros(n=1, dtype=float, device=device, requires_grad=True)

            wp.launch(
                kernel=mesh_query_ray_loss,
                dim=1,
                inputs=[mesh.id, query_points, query_dirs, intersection_points, loss],
                device=device,
            )

        tape.backward(loss=loss)
        q = intersection_points.numpy().flatten()
        analytic_p = tape.gradients[query_points].numpy().flatten()
        analytic_D = tape.gradients[query_dirs].numpy().flatten()

        # numeric gradients

        # ray origin
        eps = 1.0e-3
        loss_values_p = []
        numeric_p = np.zeros(3)

        offset_query_points = [
            wp.vec3(p[0] - eps, p[1], p[2]),
            wp.vec3(p[0] + eps, p[1], p[2]),
            wp.vec3(p[0], p[1] - eps, p[2]),
            wp.vec3(p[0], p[1] + eps, p[2]),
            wp.vec3(p[0], p[1], p[2] - eps),
            wp.vec3(p[0], p[1], p[2] + eps),
        ]

        for i in range(6):
            q = offset_query_points[i]

            query_points = wp.array(q, dtype=wp.vec3, device=device)
            query_dirs = wp.array(D, dtype=wp.vec3, device=device)
            intersection_points = wp.zeros(n=1, dtype=wp.vec3, device=device)
            loss = wp.zeros(n=1, dtype=float, device=device)

            wp.launch(
                kernel=mesh_query_ray_loss,
                dim=1,
                inputs=[mesh.id, query_points, query_dirs, intersection_points, loss],
                device=device,
            )

            loss_values_p.append(loss.numpy()[0])

        for i in range(3):
            l_0 = loss_values_p[i * 2]
            l_1 = loss_values_p[i * 2 + 1]
            gradient = (l_1 - l_0) / (2.0 * eps)
            numeric_p[i] = gradient

        # ray dir
        loss_values_D = []
        numeric_D = np.zeros(3)

        offset_query_dirs = [
            wp.vec3(D[0] - eps, D[1], D[2]),
            wp.vec3(D[0] + eps, D[1], D[2]),
            wp.vec3(D[0], D[1] - eps, D[2]),
            wp.vec3(D[0], D[1] + eps, D[2]),
            wp.vec3(D[0], D[1], D[2] - eps),
            wp.vec3(D[0], D[1], D[2] + eps),
        ]

        for i in range(6):
            q = offset_query_dirs[i]

            query_points = wp.array(p, dtype=wp.vec3, device=device)
            query_dirs = wp.array(q, dtype=wp.vec3, device=device)
            intersection_points = wp.zeros(n=1, dtype=wp.vec3, device=device)
            loss = wp.zeros(n=1, dtype=float, device=device)

            wp.launch(
                kernel=mesh_query_ray_loss,
                dim=1,
                inputs=[mesh.id, query_points, query_dirs, intersection_points, loss],
                device=device,
            )

            loss_values_D.append(loss.numpy()[0])

        for i in range(3):
            l_0 = loss_values_D[i * 2]
            l_1 = loss_values_D[i * 2 + 1]
            gradient = (l_1 - l_0) / (2.0 * eps)
            numeric_D[i] = gradient

        error_p = ((analytic_p - numeric_p) * (analytic_p - numeric_p)).sum(axis=0)
        error_D = ((analytic_D - numeric_D) * (analytic_D - numeric_D)).sum(axis=0)

        tolerance = 1.0e-3
        test.assertTrue(error_p < tolerance, f"error is {error_p} which is >= {tolerance}")
        test.assertTrue(error_D < tolerance, f"error is {error_D} which is >= {tolerance}")


@wp.kernel
def mesh_query_ray_with_results(
    mesh: wp.uint64,
    ray_starts: wp.array(dtype=wp.vec3),
    ray_directions: wp.array(dtype=wp.vec3),
    max_t: float,
    faces: wp.array(dtype=int),
    counts: wp.array(dtype=int),
):
    tid = wp.tid()

    p = ray_starts[tid]
    dir = ray_directions[tid]

    t = float(0.0)
    u = float(0.0)
    v = float(0.0)
    sign = float(0.0)
    n = wp.vec3()
    f = int(-1)

    hit = wp.mesh_query_ray(mesh, p, dir, max_t, t, u, v, sign, n, f)

    faces[tid] = f
    counts[tid] = int(hit)


@wp.kernel
def mesh_query_ray_brutal(
    points: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=int),
    ray_starts: wp.array(dtype=wp.vec3),
    ray_directions: wp.array(dtype=wp.vec3),
    max_t: float,
    faces: wp.array(dtype=int),
    counts: wp.array(dtype=int),
):
    tid = wp.tid()

    p = ray_starts[tid]
    dir = ray_directions[tid]

    face_closest = raycast_brutal(points, indices, p, dir, max_t)
    hit = face_closest >= 0

    faces[tid] = face_closest
    counts[tid] = int(hit)


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
def test_mesh_query_ray_count_intersections(test, device):
    """Stress test for mesh_query_ray_count_intersections with various ray configurations"""
    from pxr import Usd, UsdGeom  # noqa: PLC0415

    # Load a complex mesh (torus is good for multiple intersections)
    usd_stage = Usd.Stage.Open(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "torus.usda")))
    usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/World/Torus"))

    mesh_counts = usd_geom.GetFaceVertexCountsAttr().Get()
    mesh_indices = usd_geom.GetFaceVertexIndicesAttr().Get()
    tri_indices = triangulate(mesh_counts, mesh_indices)

    points_np = np.array(usd_geom.GetPointsAttr().Get())
    indices_np = np.array(tri_indices)

    points = wp.array(points_np, dtype=wp.vec3, device=device)
    indices = wp.array(indices_np, dtype=int, device=device)

    if device.is_cpu:
        constructors = ["sah", "median"]
    else:
        constructors = ["sah", "median", "lbvh"]

    leaf_sizes = [1, 2, 4]

    # Compute bounding box for ray generation
    world_min = np.min(points_np, axis=0)
    world_max = np.max(points_np, axis=0)
    world_center = (world_min + world_max) / 2.0
    world_size = world_max - world_min
    max_extent = np.max(world_size)

    # Generate stress test rays
    num_rays = 5000  # Stress test with many rays
    rng = np.random.default_rng(42)

    ray_starts_list = []
    ray_dirs_list = []

    # 1. Rays from above (perpendicular to XY plane) - should get multiple hits through torus
    num_rays_vertical = num_rays // 5
    xy = rng.uniform(0.0, 1.0, size=(num_rays_vertical, 2)) * world_size[:2] + world_min[:2]
    z = np.full((num_rays_vertical, 1), world_max[2] + max_extent * 0.5, dtype=np.float32)
    ray_starts_list.append(np.concatenate((xy, z), axis=1))
    ray_dirs_vertical = np.zeros((num_rays_vertical, 3))
    ray_dirs_vertical[:, 2] = -1.0
    ray_dirs_list.append(ray_dirs_vertical)

    # 2. Rays from multiple sides (should pierce torus from different angles)
    num_rays_sides = num_rays // 5
    for axis in range(3):
        starts = rng.uniform(0.0, 1.0, size=(num_rays_sides, 3)) * world_size + world_min
        # Move rays outside the bounding box along the chosen axis
        starts[:, axis] = world_min[axis] - max_extent * 0.5

        dirs = np.zeros((num_rays_sides, 3))
        dirs[:, axis] = 1.0

        ray_starts_list.append(starts)
        ray_dirs_list.append(dirs)

    # 3. Rays through center with random angles (likely to get multiple hits)
    num_rays_center = num_rays // 5
    angles_theta = rng.uniform(0, 2 * np.pi, size=num_rays_center)
    angles_phi = rng.uniform(0, np.pi, size=num_rays_center)

    # Start rays from outside, pointing toward center
    offset_distance = max_extent * 1.5
    ray_starts_center = np.zeros((num_rays_center, 3), dtype=np.float32)
    ray_starts_center[:, 0] = world_center[0] + offset_distance * np.sin(angles_phi) * np.cos(angles_theta)
    ray_starts_center[:, 1] = world_center[1] + offset_distance * np.sin(angles_phi) * np.sin(angles_theta)
    ray_starts_center[:, 2] = world_center[2] + offset_distance * np.cos(angles_phi)

    ray_dirs_center = world_center - ray_starts_center
    ray_dirs_center = ray_dirs_center / np.linalg.norm(ray_dirs_center, axis=1, keepdims=True)

    ray_starts_list.append(ray_starts_center)
    ray_dirs_list.append(ray_dirs_center)

    # Combine all rays
    ray_starts_np = np.concatenate(ray_starts_list, axis=0).astype(np.float32)
    ray_dirs_np = np.concatenate(ray_dirs_list, axis=0).astype(np.float32)
    total_rays = len(ray_starts_np)

    ray_starts = wp.array(ray_starts_np, dtype=wp.vec3, device=device)
    ray_dirs = wp.array(ray_dirs_np, dtype=wp.vec3, device=device)

    for leaf_size, constructor in itertools.product(leaf_sizes, constructors):
        mesh = wp.Mesh(
            points=points,
            indices=indices,
            bvh_constructor=constructor,
            bvh_leaf_size=leaf_size,
        )

        # Brute force count
        counts_brutal = wp.empty(n=total_rays, dtype=int, device=device)
        wp.launch(
            kernel=mesh_query_ray_count_intersections_brutal,
            dim=total_rays,
            inputs=[points, indices, ray_starts, ray_dirs],
            outputs=[counts_brutal],
            device=device,
        )

        # BVH-accelerated count
        counts_bvh = wp.empty(n=total_rays, dtype=int, device=device)
        wp.launch(
            kernel=mesh_query_ray_count_intersections_kernel,
            dim=total_rays,
            inputs=[mesh.id, ray_starts, ray_dirs],
            outputs=[counts_bvh],
            device=device,
        )

        # Compare results
        counts_brutal_np = counts_brutal.numpy()

        # Verify they match
        assert_array_equal(counts_bvh, counts_brutal)

        # Additional validation: check that we have interesting cases
        # (rays with 0, 1, 2+ intersections)
        unique_counts = np.unique(counts_brutal_np)
        test.assertGreater(
            len(unique_counts),
            2,
            f"Test should have rays with different intersection counts (found: {unique_counts})",
        )
        test.assertTrue(np.any(counts_brutal_np == 0), "Test should have rays with 0 intersections")
        test.assertTrue(
            np.any(counts_brutal_np >= 2), "Test should have rays with 2+ intersections (stress test for torus)"
        )


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
def test_mesh_query_ray_and_groups(test, device):
    from pxr import Usd, UsdGeom  # noqa: PLC0415

    usd_stage = Usd.Stage.Open(os.path.join(wp.examples.get_asset_directory(), "bunny.usd"))
    usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/root/bunny"))

    points_np = np.array(usd_geom.GetPointsAttr().Get())
    indices_np = np.array(usd_geom.GetFaceVertexIndicesAttr().Get())

    points = wp.array(points_np, dtype=wp.vec3, device=device)
    indices = wp.array(indices_np, dtype=int, device=device)
    num_faces = int(indices.shape[0] / 3)

    if device.is_cpu:
        constructors = ["sah", "median"]
    else:
        constructors = ["sah", "median", "lbvh"]

    leaf_sizes = [1, 2, 4]

    world_min = np.min(points_np, axis=0)
    world_max = np.max(points_np, axis=0)
    world_size = world_max - world_min

    num_rays = 10000
    rng = np.random.default_rng(123)

    xy = rng.uniform(0.0, 1.0, size=(num_rays, 2)) * world_size[:2] + world_min[:2]
    z = np.full((num_rays, 1), world_max[2] + 0.1 * world_size[2], dtype=np.float32)

    ray_starts_np = np.concatenate((xy, z), axis=1)
    ray_dirs_np = np.zeros_like(ray_starts_np)
    ray_dirs_np[:, 2] = -1.0

    ray_starts = wp.array(ray_starts_np, dtype=wp.vec3, device=device)
    ray_dirs = wp.array(ray_dirs_np, dtype=wp.vec3, device=device)

    groups_np = np.zeros(num_faces, dtype=np.int32)
    groups_np[num_faces // 2 :] = 1
    groups = wp.array(groups_np, dtype=int, device=device)

    max_t = float(1.0e6)

    for leaf_size, constructor in itertools.product(leaf_sizes, constructors):
        mesh = wp.Mesh(
            points=points,
            indices=indices,
            bvh_constructor=constructor,
            bvh_leaf_size=leaf_size,
        )

        mesh_grouped = wp.Mesh(
            points=points,
            indices=indices,
            groups=groups,
            bvh_constructor=constructor,
            bvh_leaf_size=leaf_size,
        )

        counts_brutal = wp.empty(n=num_rays, dtype=int, device=device)
        faces_brutal = wp.empty(n=num_rays, dtype=int, device=device)

        wp.launch(
            kernel=mesh_query_ray_brutal,
            dim=num_rays,
            inputs=[points, indices, ray_starts, ray_dirs, max_t],
            outputs=[faces_brutal, counts_brutal],
            device=device,
        )

        faces = wp.empty(n=num_rays, dtype=int, device=device)
        counts = wp.empty(n=num_rays, dtype=int, device=device)

        wp.launch(
            kernel=mesh_query_ray_with_results,
            dim=num_rays,
            inputs=[mesh.id, ray_starts, ray_dirs, max_t],
            outputs=[faces, counts],
            device=device,
        )

        assert_array_equal(counts, counts_brutal)
        assert_array_equal(faces, faces_brutal)

        faces_grouped = wp.empty(n=num_rays, dtype=int, device=device)
        counts_grouped = wp.empty(n=num_rays, dtype=int, device=device)

        wp.launch(
            kernel=mesh_query_ray_with_results,
            dim=num_rays,
            inputs=[mesh_grouped.id, ray_starts, ray_dirs, max_t],
            outputs=[faces_grouped, counts_grouped],
            device=device,
        )

        assert_array_equal(counts_grouped, counts_brutal)
        assert_array_equal(faces_grouped, faces_brutal)

        counts_anyhit = wp.empty(n=num_rays, dtype=int, device=device)

        @wp.kernel
        def mesh_query_ray_anyhit_kernel(
            mesh: wp.uint64,
            ray_starts: wp.array(dtype=wp.vec3),
            ray_directions: wp.array(dtype=wp.vec3),
            max_t: float,
            counts: wp.array(dtype=int),
        ):
            tid = wp.tid()
            p = ray_starts[tid]
            dir = ray_directions[tid]
            counts[tid] = int(wp.mesh_query_ray_anyhit(mesh, p, dir, max_t))

        wp.launch(
            kernel=mesh_query_ray_anyhit_kernel,
            dim=num_rays,
            inputs=[mesh.id, ray_starts, ray_dirs, max_t],
            outputs=[counts_anyhit],
            device=device,
        )

        assert_array_equal(counts_anyhit, counts_brutal)

        wp.synchronize_device(device)


@wp.kernel
def raycast_kernel(
    mesh: wp.uint64,
    ray_starts: wp.array(dtype=wp.vec3),
    ray_directions: wp.array(dtype=wp.vec3),
    count: wp.array(dtype=int),
):
    t = float(0.0)  # hit distance along ray
    u = float(0.0)  # hit face barycentric u
    v = float(0.0)  # hit face barycentric v
    sign = float(0.0)  # hit face sign
    n = wp.vec3()  # hit face normal
    f = int(0)  # hit face index
    max_dist = 1e6  # max raycast distance

    # ray cast against the mesh
    tid = wp.tid()

    if wp.mesh_query_ray(mesh, ray_starts[tid], ray_directions[tid], max_dist, t, u, v, sign, n, f):
        wp.atomic_add(count, 0, 1)


# tests rays against a quad of two connected triangles
# with rays exactly falling on the edge, tests that
# there are no leaks


def test_mesh_query_ray_edge(test, device):
    if device.is_cpu:
        constructors = ["sah", "median"]
    else:
        constructors = ["sah", "median", "lbvh"]

    leaf_sizes = [1, 2, 4]

    # Create raycast starts and directions
    xx, yy = np.meshgrid(np.arange(0.1, 0.4, 0.01), np.arange(0.1, 0.4, 0.01))
    xx = xx.flatten().reshape(-1, 1)
    yy = yy.flatten().reshape(-1, 1)
    zz = np.ones_like(xx)

    ray_starts = np.concatenate((xx, yy, zz), axis=1)
    ray_dirs = np.zeros_like(ray_starts)
    ray_dirs[:, 2] = -1.0

    n = len(ray_starts)

    ray_starts = wp.array(ray_starts, shape=(n,), dtype=wp.vec3, device=device)
    ray_dirs = wp.array(ray_dirs, shape=(n,), dtype=wp.vec3, device=device)

    # Create simple square mesh
    vertices = np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.0]], dtype=np.float32)

    triangles = np.array([[1, 0, 2], [1, 2, 3]], dtype=np.int32)

    for leaf_size, constructor in itertools.product(leaf_sizes, constructors):
        mesh = wp.Mesh(
            points=wp.array(vertices, dtype=wp.vec3, device=device),
            indices=wp.array(triangles.flatten(), dtype=int, device=device),
            bvh_constructor=constructor,
            bvh_leaf_size=leaf_size,
        )

        counts = wp.zeros(1, dtype=int, device=device)

        wp.launch(kernel=raycast_kernel, dim=n, inputs=[mesh.id, ray_starts, ray_dirs, counts], device=device)
        wp.synchronize()

        test.assertEqual(counts.numpy()[0], n)


devices = get_test_devices()


class TestMeshQueryRay(unittest.TestCase):
    def test_mesh_query_codegen_adjoints_with_select(self):
        def kernel_fn(
            mesh: wp.uint64,
        ):
            v = wp.vec3(0.0, 0.0, 0.0)
            d = 1e-6

            if True:
                query = wp.mesh_query_ray(mesh, v, v, d)
            else:
                query = wp.mesh_query_ray(mesh, v, v, d)

        wp.Kernel(func=kernel_fn)


add_function_test(TestMeshQueryRay, "test_mesh_query_ray_edge", test_mesh_query_ray_edge, devices=devices)
add_function_test(TestMeshQueryRay, "test_mesh_query_ray_grad", test_mesh_query_ray_grad, devices=devices)
add_function_test(
    TestMeshQueryRay,
    "test_mesh_query_ray_count_intersections",
    test_mesh_query_ray_count_intersections,
    devices=devices,
)
add_function_test(TestMeshQueryRay, "test_mesh_query_ray_and_groups", test_mesh_query_ray_and_groups, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
