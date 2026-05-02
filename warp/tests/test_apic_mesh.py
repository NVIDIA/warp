# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for APIC mesh serialization.

These tests verify that wp.Mesh objects can be correctly serialized and
deserialized through APIC, with handle fixup ensuring mesh queries work
correctly after loading.
"""

import os
import tempfile
import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import add_function_test, get_test_devices


def create_unit_cube_mesh(device):
    """Create a simple unit cube mesh centered at origin."""
    points = np.array(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ],
        dtype=np.float32,
    )

    indices = np.array(
        [
            0,
            1,
            2,
            0,
            2,
            3,  # Front
            4,
            6,
            5,
            4,
            7,
            6,  # Back
            0,
            3,
            7,
            0,
            7,
            4,  # Left
            1,
            5,
            6,
            1,
            6,
            2,  # Right
            0,
            4,
            5,
            0,
            5,
            1,  # Bottom
            3,
            2,
            6,
            3,
            6,
            7,  # Top
        ],
        dtype=np.int32,
    )

    mesh_points = wp.array(points, dtype=wp.vec3, device=device)
    mesh_indices = wp.array(indices, dtype=int, device=device)
    return wp.Mesh(points=mesh_points, indices=mesh_indices)


def create_tetrahedron_mesh(device):
    """Create a simple tetrahedron mesh."""
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.5, 0.5, 1.0]],
        dtype=np.float32,
    )

    indices = np.array(
        [0, 1, 2, 0, 1, 3, 1, 2, 3, 2, 0, 3],
        dtype=np.int32,
    )

    mesh_points = wp.array(points, dtype=wp.vec3, device=device)
    mesh_indices = wp.array(indices, dtype=int, device=device)
    return wp.Mesh(points=mesh_points, indices=mesh_indices)


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@wp.kernel
def mesh_query_point_kernel(
    mesh_id: wp.handle,
    query_points: wp.array(dtype=wp.vec3),
    closest_points: wp.array(dtype=wp.vec3),
    distances: wp.array(dtype=float),
    faces: wp.array(dtype=int),
):
    tid = wp.tid()
    p = query_points[tid]
    face = int(0)
    u = float(0.0)
    v = float(0.0)
    wp.mesh_query_point_no_sign(mesh_id, p, 100.0, face, u, v)
    cp = wp.mesh_eval_position(mesh_id, face, u, v)
    closest_points[tid] = cp
    distances[tid] = wp.length(cp - p)
    faces[tid] = face


@wp.kernel
def mesh_query_ray_kernel(
    mesh_id: wp.handle,
    ray_origins: wp.array(dtype=wp.vec3),
    ray_dirs: wp.array(dtype=wp.vec3),
    hit_distances: wp.array(dtype=float),
    hit_faces: wp.array(dtype=int),
    hit_flags: wp.array(dtype=int),
):
    tid = wp.tid()
    origin = ray_origins[tid]
    direction = ray_dirs[tid]
    t = float(0.0)
    u = float(0.0)
    v = float(0.0)
    sign = float(0.0)
    normal = wp.vec3(0.0, 0.0, 0.0)
    face = int(0)
    hit = wp.mesh_query_ray(mesh_id, origin, direction, 100.0, t, u, v, sign, normal, face)
    if hit:
        hit_distances[tid] = t
        hit_faces[tid] = face
        hit_flags[tid] = 1
    else:
        hit_distances[tid] = -1.0
        hit_faces[tid] = -1
        hit_flags[tid] = 0


@wp.kernel
def mesh_eval_position_kernel(
    mesh_id: wp.handle,
    face_indices: wp.array(dtype=int),
    bary_u: wp.array(dtype=float),
    bary_v: wp.array(dtype=float),
    positions: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    pos = wp.mesh_eval_position(mesh_id, face_indices[tid], bary_u[tid], bary_v[tid])
    positions[tid] = pos


@wp.kernel
def mesh_combined_operations_kernel(
    mesh_id: wp.handle,
    query_points: wp.array(dtype=wp.vec3),
    results: wp.array(dtype=float),
):
    tid = wp.tid()
    p = query_points[tid]
    face = int(0)
    u = float(0.0)
    v = float(0.0)
    wp.mesh_query_point_no_sign(mesh_id, p, 100.0, face, u, v)
    cp = wp.mesh_eval_position(mesh_id, face, u, v)
    dist = wp.length(cp - p)
    ray_dir = wp.normalize(wp.vec3(0.0, 0.0, 0.0) - p)
    t = float(0.0)
    ru = float(0.0)
    rv = float(0.0)
    sign = float(0.0)
    normal = wp.vec3(0.0, 0.0, 0.0)
    ray_face = int(0)
    hit = wp.mesh_query_ray(mesh_id, p, ray_dir, 100.0, t, ru, rv, sign, normal, ray_face)
    if hit:
        results[tid] = dist + t
    else:
        results[tid] = dist


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------


def test_apic_mesh_query_point(test, device):
    """Test mesh point queries through APIC save/load."""
    mesh = create_unit_cube_mesh(device)
    n = 8
    query_pts = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0],
        ],
        dtype=np.float32,
    )
    query_points = wp.array(query_pts, dtype=wp.vec3, device=device)
    closest_points = wp.zeros(n, dtype=wp.vec3, device=device)
    distances = wp.zeros(n, dtype=float, device=device)
    faces = wp.zeros(n, dtype=int, device=device)

    with tempfile.TemporaryDirectory() as tmpdir:
        graph_path = os.path.join(tmpdir, "mesh_query_point")

        wp.load_module(device=device)
        with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
            wp.launch(
                mesh_query_point_kernel,
                dim=n,
                inputs=[mesh.id, query_points, closest_points, distances, faces],
                device=device,
            )

        wp.capture_launch(capture.graph)
        wp.synchronize_device(device)
        ref_closest = closest_points.numpy().copy()
        ref_distances = distances.numpy().copy()

        test.assertAlmostEqual(ref_distances[0], 0.5, places=4)

        closest_points.zero_()
        distances.zero_()
        faces.zero_()

        wp.capture_save(
            capture.graph,
            graph_path,
            inputs={"query_points": query_points},
            outputs={"closest_points": closest_points, "distances": distances, "faces": faces},
        )

        loaded = wp.capture_load(graph_path, device=device)
        wp.capture_launch(loaded)
        wp.synchronize_device(device)

        new_closest = wp.zeros(n, dtype=wp.vec3, device=device)
        new_distances = wp.zeros(n, dtype=float, device=device)
        loaded.get_param("closest_points", new_closest)
        loaded.get_param("distances", new_distances)

        np.testing.assert_array_almost_equal(new_closest.numpy(), ref_closest, decimal=4)
        np.testing.assert_array_almost_equal(new_distances.numpy(), ref_distances, decimal=4)


def test_apic_mesh_query_ray(test, device):
    """Test mesh ray queries through APIC save/load."""
    mesh = create_unit_cube_mesh(device)
    n = 6
    origins = np.array(
        [[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, -2.0]],
        dtype=np.float32,
    )
    dirs = np.array(
        [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    ray_origins = wp.array(origins, dtype=wp.vec3, device=device)
    ray_dirs = wp.array(dirs, dtype=wp.vec3, device=device)
    hit_distances = wp.zeros(n, dtype=float, device=device)
    hit_faces = wp.zeros(n, dtype=int, device=device)
    hit_flags = wp.zeros(n, dtype=int, device=device)

    with tempfile.TemporaryDirectory() as tmpdir:
        graph_path = os.path.join(tmpdir, "mesh_query_ray")

        wp.load_module(device=device)
        with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
            wp.launch(
                mesh_query_ray_kernel,
                dim=n,
                inputs=[mesh.id, ray_origins, ray_dirs, hit_distances, hit_faces, hit_flags],
                device=device,
            )

        wp.capture_launch(capture.graph)
        wp.synchronize_device(device)
        ref_distances = hit_distances.numpy().copy()
        ref_flags = hit_flags.numpy().copy()

        test.assertTrue(np.all(ref_flags == 1))
        np.testing.assert_array_almost_equal(ref_distances, np.full(n, 1.5), decimal=4)

        hit_distances.zero_()
        hit_faces.zero_()
        hit_flags.zero_()

        wp.capture_save(
            capture.graph,
            graph_path,
            inputs={"ray_origins": ray_origins, "ray_dirs": ray_dirs},
            outputs={"hit_distances": hit_distances, "hit_faces": hit_faces, "hit_flags": hit_flags},
        )

        loaded = wp.capture_load(graph_path, device=device)
        wp.capture_launch(loaded)
        wp.synchronize_device(device)

        new_distances = wp.zeros(n, dtype=float, device=device)
        new_flags = wp.zeros(n, dtype=int, device=device)
        loaded.get_param("hit_distances", new_distances)
        loaded.get_param("hit_flags", new_flags)

        np.testing.assert_array_almost_equal(new_distances.numpy(), ref_distances, decimal=4)
        np.testing.assert_array_equal(new_flags.numpy(), ref_flags)


def test_apic_mesh_eval_position(test, device):
    """Test mesh position evaluation through APIC save/load."""
    mesh = create_tetrahedron_mesh(device)
    n = 4
    face_indices = wp.array([0, 0, 1, 2], dtype=int, device=device)
    bary_u = wp.array([1.0, 0.0, 0.5, 0.33], dtype=float, device=device)
    bary_v = wp.array([0.0, 1.0, 0.25, 0.33], dtype=float, device=device)
    positions = wp.zeros(n, dtype=wp.vec3, device=device)

    with tempfile.TemporaryDirectory() as tmpdir:
        graph_path = os.path.join(tmpdir, "mesh_eval_position")

        wp.load_module(device=device)
        with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
            wp.launch(
                mesh_eval_position_kernel,
                dim=n,
                inputs=[mesh.id, face_indices, bary_u, bary_v, positions],
                device=device,
            )

        wp.capture_launch(capture.graph)
        wp.synchronize_device(device)
        ref_positions = positions.numpy().copy()

        positions.zero_()

        wp.capture_save(
            capture.graph,
            graph_path,
            inputs={"face_indices": face_indices, "bary_u": bary_u, "bary_v": bary_v},
            outputs={"positions": positions},
        )

        loaded = wp.capture_load(graph_path, device=device)
        wp.capture_launch(loaded)
        wp.synchronize_device(device)

        new_positions = wp.zeros(n, dtype=wp.vec3, device=device)
        loaded.get_param("positions", new_positions)

        np.testing.assert_array_almost_equal(new_positions.numpy(), ref_positions, decimal=4)


def test_apic_mesh_combined_operations(test, device):
    """Test combined mesh operations through APIC save/load."""
    mesh = create_unit_cube_mesh(device)
    n = 8
    query_pts = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.5],
            [-0.5, -0.5, -0.5],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
        ],
        dtype=np.float32,
    )
    query_points = wp.array(query_pts, dtype=wp.vec3, device=device)
    results = wp.zeros(n, dtype=float, device=device)

    with tempfile.TemporaryDirectory() as tmpdir:
        graph_path = os.path.join(tmpdir, "mesh_combined")

        wp.load_module(device=device)
        with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
            wp.launch(mesh_combined_operations_kernel, dim=n, inputs=[mesh.id, query_points, results], device=device)

        wp.capture_launch(capture.graph)
        wp.synchronize_device(device)
        ref_results = results.numpy().copy()

        results.zero_()

        wp.capture_save(capture.graph, graph_path, inputs={"query_points": query_points}, outputs={"results": results})

        loaded = wp.capture_load(graph_path, device=device)
        wp.capture_launch(loaded)
        wp.synchronize_device(device)

        new_results = wp.zeros(n, dtype=float, device=device)
        loaded.get_param("results", new_results)

        np.testing.assert_array_almost_equal(new_results.numpy(), ref_results, decimal=4)


def test_apic_mesh_handle_in_struct(test, device):
    """Test mesh handle stored in a struct array through APIC save/load."""

    @wp.struct
    class MeshQuery:
        mesh_id: wp.handle
        query_point: wp.vec3
        max_dist: float

    @wp.kernel
    def query_from_struct_kernel(
        queries: wp.array(dtype=MeshQuery),
        distances: wp.array(dtype=float),
    ):
        tid = wp.tid()
        q = queries[tid]
        face = int(0)
        u = float(0.0)
        v = float(0.0)
        wp.mesh_query_point_no_sign(q.mesh_id, q.query_point, q.max_dist, face, u, v)
        cp = wp.mesh_eval_position(q.mesh_id, face, u, v)
        distances[tid] = wp.length(cp - q.query_point)

    mesh = create_unit_cube_mesh(device)
    n = 4
    queries = wp.zeros(n, dtype=MeshQuery, device=device)
    distances = wp.zeros(n, dtype=float, device=device)

    queries_host = queries.numpy()
    query_pts = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
    for i in range(n):
        queries_host[i]["mesh_id"] = mesh.id
        queries_host[i]["query_point"] = query_pts[i]
        queries_host[i]["max_dist"] = 100.0

    wp.copy(queries, wp.array(queries_host, dtype=MeshQuery, device=device))
    wp.synchronize_device(device)

    with tempfile.TemporaryDirectory() as tmpdir:
        graph_path = os.path.join(tmpdir, "mesh_struct")

        wp.load_module(device=device)
        with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
            wp.launch(query_from_struct_kernel, dim=n, inputs=[queries, distances], device=device)

        wp.capture_launch(capture.graph)
        wp.synchronize_device(device)
        ref_distances = distances.numpy().copy()

        test.assertAlmostEqual(ref_distances[0], 0.5, places=3)

        distances.zero_()

        wp.capture_save(capture.graph, graph_path, inputs={"queries": queries}, outputs={"distances": distances})

        loaded = wp.capture_load(graph_path, device=device)
        wp.capture_launch(loaded)
        wp.synchronize_device(device)

        new_distances = wp.zeros(n, dtype=float, device=device)
        loaded.get_param("distances", new_distances)

        np.testing.assert_array_almost_equal(new_distances.numpy(), ref_distances, decimal=4)


def test_apic_multiple_meshes(test, device):
    """Test multiple meshes through APIC save/load."""

    @wp.kernel
    def query_multiple_meshes_kernel(
        mesh1_id: wp.handle,
        mesh2_id: wp.handle,
        query_points: wp.array(dtype=wp.vec3),
        dist1: wp.array(dtype=float),
        dist2: wp.array(dtype=float),
    ):
        tid = wp.tid()
        p = query_points[tid]
        face1 = int(0)
        u1 = float(0.0)
        v1 = float(0.0)
        wp.mesh_query_point_no_sign(mesh1_id, p, 100.0, face1, u1, v1)
        cp1 = wp.mesh_eval_position(mesh1_id, face1, u1, v1)
        dist1[tid] = wp.length(cp1 - p)

        face2 = int(0)
        u2 = float(0.0)
        v2 = float(0.0)
        wp.mesh_query_point_no_sign(mesh2_id, p, 100.0, face2, u2, v2)
        cp2 = wp.mesh_eval_position(mesh2_id, face2, u2, v2)
        dist2[tid] = wp.length(cp2 - p)

    mesh1 = create_unit_cube_mesh(device)
    mesh2 = create_tetrahedron_mesh(device)
    n = 4
    query_pts = np.array(
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    query_points = wp.array(query_pts, dtype=wp.vec3, device=device)
    dist1 = wp.zeros(n, dtype=float, device=device)
    dist2 = wp.zeros(n, dtype=float, device=device)

    with tempfile.TemporaryDirectory() as tmpdir:
        graph_path = os.path.join(tmpdir, "multiple_meshes")

        wp.load_module(device=device)
        with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
            wp.launch(
                query_multiple_meshes_kernel,
                dim=n,
                inputs=[mesh1.id, mesh2.id, query_points, dist1, dist2],
                device=device,
            )

        wp.capture_launch(capture.graph)
        wp.synchronize_device(device)
        ref_dist1 = dist1.numpy().copy()
        ref_dist2 = dist2.numpy().copy()

        dist1.zero_()
        dist2.zero_()

        wp.capture_save(
            capture.graph, graph_path, inputs={"query_points": query_points}, outputs={"dist1": dist1, "dist2": dist2}
        )

        loaded = wp.capture_load(graph_path, device=device)
        wp.capture_launch(loaded)
        wp.synchronize_device(device)

        new_dist1 = wp.zeros(n, dtype=float, device=device)
        new_dist2 = wp.zeros(n, dtype=float, device=device)
        loaded.get_param("dist1", new_dist1)
        loaded.get_param("dist2", new_dist2)

        np.testing.assert_array_almost_equal(new_dist1.numpy(), ref_dist1, decimal=4)
        np.testing.assert_array_almost_equal(new_dist2.numpy(), ref_dist2, decimal=4)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestApicMesh(unittest.TestCase):
    pass


devices = get_test_devices()

add_function_test(TestApicMesh, "test_apic_mesh_query_point", test_apic_mesh_query_point, devices=devices)
add_function_test(TestApicMesh, "test_apic_mesh_query_ray", test_apic_mesh_query_ray, devices=devices)
add_function_test(TestApicMesh, "test_apic_mesh_eval_position", test_apic_mesh_eval_position, devices=devices)
add_function_test(
    TestApicMesh, "test_apic_mesh_combined_operations", test_apic_mesh_combined_operations, devices=devices
)
add_function_test(TestApicMesh, "test_apic_mesh_handle_in_struct", test_apic_mesh_handle_in_struct, devices=devices)
add_function_test(TestApicMesh, "test_apic_multiple_meshes", test_apic_multiple_meshes, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
