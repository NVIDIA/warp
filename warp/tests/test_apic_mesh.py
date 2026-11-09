# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for APIC mesh serialization.

These tests verify that wp.Mesh objects can be correctly serialized and
deserialized through APIC, with handle fixup ensuring mesh queries work
correctly after loading.
"""

import gc
import os
import struct
import tempfile
import textwrap
import unittest
import weakref

import numpy as np

import warp as wp
from warp.tests.unittest_utils import (
    add_function_test,
    get_test_devices_with_cuda_graph_module_load,
    run_python_subprocess,
)

# On-disk record used to construct malformed relocations for loader validation.
_WRP_HANDLE_LOCATION = struct.Struct("<IQQQ")  # region ID, byte offset, byte stride, element count


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
    query_points: wp.array[wp.vec3],
    closest_points: wp.array[wp.vec3],
    distances: wp.array[float],
    faces: wp.array[int],
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
    ray_origins: wp.array[wp.vec3],
    ray_dirs: wp.array[wp.vec3],
    hit_distances: wp.array[float],
    hit_faces: wp.array[int],
    hit_flags: wp.array[int],
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
    face_indices: wp.array[int],
    bary_u: wp.array[float],
    bary_v: wp.array[float],
    positions: wp.array[wp.vec3],
):
    tid = wp.tid()
    pos = wp.mesh_eval_position(mesh_id, face_indices[tid], bary_u[tid], bary_v[tid])
    positions[tid] = pos


@wp.kernel
def mesh_combined_operations_kernel(
    mesh_id: wp.handle,
    query_points: wp.array[wp.vec3],
    results: wp.array[float],
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


@wp.kernel
def mesh_handle_array_query_kernel(
    mesh_ids: wp.array[wp.handle],
    query_points: wp.array[wp.vec3],
    distances: wp.array2d[float],
):
    mesh_index, tid = wp.tid()
    p = query_points[tid]
    query = wp.mesh_query_point_no_sign(mesh_ids[mesh_index], p, 100.0)
    cp = wp.mesh_eval_position(mesh_ids[mesh_index], query.face, query.u, query.v)
    distances[mesh_index, tid] = wp.length(cp - p)


@wp.struct
class MeshHandleField:
    """A mesh reference that can also be nested inside another struct."""

    mesh_id: wp.handle


@wp.struct
class MeshHandleOutput:
    """An output whose mesh reference is nested after an ordinary integer field."""

    tag: wp.uint64
    nested: MeshHandleField


@wp.kernel
def overwrite_mesh_handle_struct_kernel(
    mesh_id: wp.handle, handles: wp.array[MeshHandleOutput], distances: wp.array[float]
):
    tid = wp.tid()
    value = MeshHandleOutput()
    value.nested.mesh_id = mesh_id
    handles[tid] = value
    query = wp.mesh_query_point_no_sign(handles[tid].nested.mesh_id, wp.vec3(0.0), 100.0)
    distances[tid] = wp.length(wp.mesh_eval_position(handles[tid].nested.mesh_id, query.face, query.u, query.v))


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------


_REPLAY_SCRIPT = textwrap.dedent(
    """
    import sys

    import numpy as np

    import warp as wp

    graph_path, device, name, shape, result_path = sys.argv[1:6]
    # No mesh is created in this process: loading must restore the meshes and
    # replace their saved IDs before any kernel can use them.
    graph = wp.capture_load(graph_path, device=device)
    wp.capture_launch(graph)
    wp.synchronize_device(device)
    result = wp.zeros(tuple(int(dim) for dim in shape.split(",")), dtype=float, device=device)
    graph.get_param(name, result)
    np.save(result_path, result.numpy())
    """
)


def replay_in_new_process(test, device, graph_path, name, shape):
    """Replay without the original meshes and return the named float array.

    The original process's meshes are not alive in the new process, so mesh
    handles are only valid there if the meshes were serialized and remapped.
    """
    result_path = graph_path + "_result.npy"
    shape_arg = ",".join(str(dim) for dim in shape)
    result = run_python_subprocess(
        _REPLAY_SCRIPT, graph_path, str(device), name, shape_arg, result_path, hide_gpu=device.is_cpu
    )
    test.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    return np.load(result_path)


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
    """Test mesh handle stored in a struct array through APIC save/load.

    The graph is replayed both in the saving process and in a new process. Only the
    new process catches a mesh that was not serialized, because the saving process
    still holds the original mesh that the stale handle points to.
    """

    # A nested handle after another field exercises both recursion and field offsets.
    @wp.struct
    class MeshQuery:
        query_point: wp.vec3
        tag: wp.uint64
        mesh: MeshHandleField
        other_mesh_id: wp.handle
        max_dist: float

    @wp.kernel
    def query_from_struct_kernel(
        queries: wp.array[MeshQuery],
        distances: wp.array2d[float],
    ):
        tid, mesh_index = wp.tid()
        q = queries[tid]
        mesh_id = q.mesh.mesh_id if mesh_index == 0 else q.other_mesh_id
        face = int(0)
        u = float(0.0)
        v = float(0.0)
        wp.mesh_query_point_no_sign(mesh_id, q.query_point, q.max_dist, face, u, v)
        cp = wp.mesh_eval_position(mesh_id, face, u, v)
        distances[tid, mesh_index] = wp.length(cp - q.query_point)

    mesh = create_unit_cube_mesh(device)
    other_mesh = create_tetrahedron_mesh(device)
    n = 4
    queries = wp.zeros(n, dtype=MeshQuery, device=device)
    distances = wp.zeros((n, 2), dtype=float, device=device)

    queries_host = queries.numpy()
    query_pts = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]
    for i in range(n):
        queries_host[i]["tag"] = mesh.id
        queries_host[i]["mesh"]["mesh_id"] = mesh.id
        queries_host[i]["other_mesh_id"] = other_mesh.id
        queries_host[i]["query_point"] = query_pts[i]
        queries_host[i]["max_dist"] = 100.0

    wp.copy(queries, wp.array(queries_host, dtype=MeshQuery, device=device))
    wp.synchronize_device(device)

    with tempfile.TemporaryDirectory() as tmpdir:
        graph_path = os.path.join(tmpdir, "mesh_struct")

        wp.load_module(device=device)
        with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
            wp.launch(query_from_struct_kernel, dim=distances.shape, inputs=[queries, distances], device=device)

        wp.capture_launch(capture.graph)
        wp.synchronize_device(device)
        ref_distances = distances.numpy().copy()

        np.testing.assert_allclose(ref_distances[0], [0.5, 0.0], atol=1.0e-6)

        distances.zero_()

        wp.capture_save(capture.graph, graph_path, inputs={"queries": queries}, outputs={"distances": distances})

        loaded = wp.capture_load(graph_path, device=device)
        wp.capture_launch(loaded)
        wp.synchronize_device(device)

        new_distances = wp.zeros_like(distances)
        loaded.get_param("distances", new_distances)

        np.testing.assert_array_almost_equal(new_distances.numpy(), ref_distances, decimal=4)

        # A uint64 field stays an ordinary integer even when it equals a mesh ID.
        restored_queries = wp.zeros_like(queries)
        loaded.get_param("queries", restored_queries)
        np.testing.assert_array_equal(restored_queries.numpy()["tag"], np.full(n, mesh.id, dtype=np.uint64))

        new_process_distances = replay_in_new_process(test, device, graph_path, "distances", distances.shape)
        np.testing.assert_array_almost_equal(new_process_distances, ref_distances, decimal=4)


def test_apic_mesh_handle_array(test, device):
    """Saved graphs must restore the correct mesh for each handle-array element."""
    cube_mesh = create_unit_cube_mesh(device)
    tetrahedron_mesh = create_tetrahedron_mesh(device)
    mesh_ids = wp.array([cube_mesh.id, tetrahedron_mesh.id], dtype=wp.handle, device=device)
    query_points = wp.array(
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=wp.vec3, device=device
    )
    distances = wp.zeros((mesh_ids.shape[0], query_points.shape[0]), dtype=float, device=device)

    with tempfile.TemporaryDirectory() as tmpdir:
        graph_path = os.path.join(tmpdir, "mesh_handle_array")

        wp.load_module(device=device)
        with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
            wp.launch(
                mesh_handle_array_query_kernel,
                dim=distances.shape,
                inputs=[mesh_ids, query_points, distances],
                device=device,
            )

        wp.capture_launch(capture.graph)
        ref_distances = distances.numpy().copy()

        # The two meshes give different distances at the origin. These checks
        # catch an ID mapped to the wrong mesh, as well as an ID left unmapped.
        test.assertAlmostEqual(ref_distances[0, 0], 0.5, places=4)
        test.assertAlmostEqual(ref_distances[1, 0], 0.0, places=4)

        distances.zero_()
        wp.capture_save(
            capture.graph, graph_path, inputs={"query_points": query_points}, outputs={"distances": distances}
        )

        # Same-process replay could still use the original meshes and conceal a
        # missing serialization step. The child process has only the saved graph.
        new_process_distances = replay_in_new_process(test, device, graph_path, "distances", distances.shape)
        np.testing.assert_array_almost_equal(new_process_distances, ref_distances, decimal=4)


def test_apic_mesh_overlapping_handle_views(test, device):
    """Overlapping handle views must reload the correct mesh at every position."""
    cube = create_unit_cube_mesh(device)
    tetrahedron = create_tetrahedron_mesh(device)
    handles = wp.array([cube.id, tetrahedron.id] * 4, dtype=wp.handle, device=device)
    points = wp.zeros(1, dtype=wp.vec3, device=device)
    wp.load_module(device=device)
    for name, views, expected in (
        ("contiguous", (handles[:4], handles[2:]), [0.5, 0.0] * 5),
        ("strided", (handles[::2], handles[2:7]), [0.5] * 4 + [0.5, 0.0, 0.5, 0.0, 0.5]),
    ):
        with test.subTest(views=name):
            result = wp.zeros((len(expected), 1), dtype=float, device=device)
            with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
                offset = 0
                for view in views:
                    wp.launch(
                        mesh_handle_array_query_kernel,
                        dim=(view.size, 1),
                        inputs=[view, points, result[offset : offset + view.size]],
                        device=device,
                    )
                    offset += view.size
            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "overlapping_handles")
                wp.capture_save(capture.graph, path, outputs={"result": result})
                replayed = replay_in_new_process(test, device, path, "result", result.shape)
                np.testing.assert_allclose(replayed[:, 0], expected, atol=1.0e-6)


def test_apic_mesh_save_before_first_launch(test, device):
    """Saving must tolerate invalid output handles that the first replay will overwrite."""
    mesh = create_unit_cube_mesh(device)
    stale_mesh = create_tetrahedron_mesh(device)
    stale_id = stale_mesh.id
    del stale_mesh
    bvh = wp.Bvh(
        lowers=wp.array([[0.0, 0.0, 0.0]], dtype=wp.vec3, device=device),
        uppers=wp.array([[1.0, 1.0, 1.0]], dtype=wp.vec3, device=device),
    )

    # None of these values identifies a live mesh: an arbitrary integer, the ID
    # of a deleted mesh, and a valid handle for a different kind of object.
    values = []
    for invalid_mesh_id in (1, stale_id, bvh.id):
        value = MeshHandleOutput()
        value.nested.mesh_id = invalid_mesh_id
        values.append(value)
    handles = wp.array(values, dtype=MeshHandleOutput, device=device)
    distances = wp.zeros(3, dtype=float, device=device)
    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
        wp.launch(overwrite_mesh_handle_struct_kernel, dim=3, inputs=[mesh.id, handles, distances], device=device)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "unlaunched_output")
        # Deliberately save before capture_launch(). The output still contains
        # the invalid IDs, so snapshotting must not dereference them as meshes.
        wp.capture_save(capture.graph, path, outputs={"distances": distances})
        loaded = wp.capture_load(path, device=device)
        wp.capture_launch(loaded)
        loaded.get_param("distances", distances)
        # Replay writes a real mesh ID into every element before querying it.
        # The centre of the cube is 0.5 units from its surface.
        np.testing.assert_allclose(distances.numpy(), [0.5, 0.5, 0.5], atol=1.0e-6)


def test_apic_multiple_meshes(test, device):
    """Test multiple meshes through APIC save/load."""

    @wp.kernel
    def query_multiple_meshes_kernel(
        mesh1_id: wp.handle,
        mesh2_id: wp.handle,
        query_points: wp.array[wp.vec3],
        dist1: wp.array[float],
        dist2: wp.array[float],
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
# CPU capture and loader validation
# ---------------------------------------------------------------------------


class TestApicMesh(unittest.TestCase):
    def test_save_numpy_handle_alias_preserves_integers(self):
        """Only slots exposed as handles may change when overlapping views are restored."""
        mesh = create_unit_cube_mesh("cpu")
        backing = np.zeros((2, 4), dtype=MeshHandleField.numpy_dtype())
        backing["mesh_id"] = mesh.id
        handles = wp.array(backing, dtype=MeshHandleField, copy=False, device="cpu")
        flat = handles.flatten()
        integers = wp.array(backing.view(np.uint64), dtype=wp.uint64, copy=False, device="cpu")

        # All slots contain the same mesh ID, including the ordinary integers.
        # View shapes and registration order must not broaden the remapped area.
        for name, views, indices in (
            ("short_slice", (handles[:1, :1],), [0]),
            ("overlapping_strides", (handles[:, ::2], handles[:1, 1:3]), [0, 1, 2, 4, 6]),
            ("three_overlapping_views", (flat[:3:2], flat[4:6], flat[1:6:2]), [0, 1, 2, 3, 4, 5]),
            ("slice_and_transpose", (handles[:, 1:3], handles.transpose()[:, :1]), [0, 1, 2, 3, 5, 6]),
        ):
            for integer_view_first in (False, True):
                with self.subTest(views=name, integer_view_first=integer_view_first):
                    with wp.ScopedCapture(device="cpu", apic=True) as capture:
                        pass
                    outputs = {str(i): view for i, view in enumerate(views)}
                    if integer_view_first:
                        outputs = {"integers": integers, **outputs}
                    else:
                        outputs["integers"] = integers
                    with tempfile.TemporaryDirectory() as tmpdir:
                        path = os.path.join(tmpdir, "integer_alias")
                        wp.capture_save(capture.graph, path, outputs=outputs)
                        loaded = wp.capture_load(path, device="cpu")
                        result = wp.zeros(integers.shape, dtype=wp.uint64, device="cpu")
                        loaded.get_param("integers", result)
                        restored = result.numpy().reshape(-1)
                        restored_id = int(restored[indices[0]])
                        self.assertNotEqual(restored_id, mesh.id)
                        expected = np.full(backing.size, mesh.id, dtype=np.uint64)
                        expected[indices] = restored_id
                        np.testing.assert_array_equal(restored, expected)

    def test_save_numpy_handle_aliases_with_different_capacities(self):
        """Save the full region shared by independently wrapped NumPy handle arrays."""

        @wp.kernel
        def query_handle_fields(handles: wp.array[MeshHandleField], distances: wp.array[float]):
            tid = wp.tid()
            mesh_id = handles[tid].mesh_id
            query = wp.mesh_query_point_no_sign(mesh_id, wp.vec3(0.0), 100.0)
            distances[tid] = wp.length(wp.mesh_eval_position(mesh_id, query.face, query.u, query.v))

        device = wp.get_device("cpu")
        cube_mesh = create_unit_cube_mesh(device)
        tetrahedron_mesh = create_tetrahedron_mesh(device)
        backing = np.zeros(2, dtype=MeshHandleField.numpy_dtype())
        backing["mesh_id"] = [cube_mesh.id, tetrahedron_mesh.id]
        short = wp.array(backing[:1], dtype=MeshHandleField, copy=False, device=device)
        full = wp.array(backing, dtype=MeshHandleField, copy=False, device=device)
        wp.load_module(device=device)

        for full_array_first in (False, True):
            with self.subTest(full_array_first=full_array_first):
                result = wp.zeros(3, dtype=float, device=device)
                views = ((short, result[:1]), (full, result[1:]))
                if full_array_first:
                    views = views[::-1]
                with wp.ScopedCapture(device=device, apic=True, force_module_load=False) as capture:
                    for handles, distances in views:
                        wp.launch(query_handle_fields, dim=handles.size, inputs=[handles, distances], device=device)

                with tempfile.TemporaryDirectory() as tmpdir:
                    path = os.path.join(tmpdir, "numpy_handle_aliases")
                    wp.capture_save(capture.graph, path, outputs={"result": result})
                    # The second mesh occurs only beyond the short view. A new
                    # process catches missing bytes as well as missing meshes.
                    replayed = replay_in_new_process(self, device, path, "result", (3,))
                    np.testing.assert_allclose(replayed, [0.5, 0.5, 0.0], atol=1.0e-6)

    def test_live_numpy_handle_views(self):
        """Live CPU capture must accept transposed and sliced NumPy-backed handle arrays."""

        @wp.kernel
        def read_handle_fields(handles: wp.array2d[MeshHandleField], result: wp.array2d[wp.uint64]):
            i, j = wp.tid()
            result[i, j] = handles[i, j].mesh_id

        backing = np.zeros((2, 3), dtype=MeshHandleField.numpy_dtype())
        # Distinct values make a wrong stride or missing element visible. This
        # kernel only copies the handle bits, so it needs no actual Mesh objects.
        backing["mesh_id"] = np.arange(1, 7).reshape((2, 3))
        wp.load_module(device="cpu")
        views = (("transposed", backing.T), ("sliced_then_transposed", backing[:, 1:].T))
        for name, view in views:
            with self.subTest(view=name):
                handles = wp.array(view, dtype=MeshHandleField, copy=False, device="cpu")
                result = wp.zeros(view.shape, dtype=wp.uint64, device="cpu")
                # copy=False retains the NumPy strides. Warp can underestimate
                # this view's capacity, but live replay does not snapshot it and
                # must not reject it using serialization-only bounds checks.
                with wp.ScopedCapture(device="cpu", apic=False) as capture:
                    wp.launch(read_handle_fields, dim=view.shape, inputs=[handles, result], device="cpu")

                for _ in range(2):
                    wp.capture_launch(capture.graph)
                    np.testing.assert_array_equal(result.numpy(), view["mesh_id"])

    def test_load_rejects_out_of_bounds_handle_locations(self):
        """Malformed handle runs must fail loading before reading outside a snapshot."""
        from warp.tests.test_apic import _read_apic_section, _replace_apic_section  # noqa: PLC0415

        handles = wp.zeros(2, dtype=wp.handle, device="cpu")
        with wp.ScopedCapture(device="cpu", apic=True) as capture:
            pass
        for name, offset, stride, count in (
            ("offset", 16, 8, 1),
            ("offset_overflow", (1 << 64) - 1, 8, 1),
            ("count", 0, 8, 3),
            ("count_overflow", 0, 8, (1 << 64) - 1),
            ("stride_overflow", 0, (1 << 64) - 1, 2),
            ("zero_stride", 0, 0, 2),
        ):
            with self.subTest(case=name), tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "invalid_handles.wrp")
                wp.capture_save(capture.graph, path, outputs={"handles": handles})
                metadata = _read_apic_section(path, 1)
                record_offset = len(metadata) - _WRP_HANDLE_LOCATION.size
                region_id = _WRP_HANDLE_LOCATION.unpack_from(metadata, record_offset)[0]
                _WRP_HANDLE_LOCATION.pack_into(metadata, record_offset, region_id, offset, stride, count)
                _replace_apic_section(path, 1, metadata)
                with self.assertRaisesRegex(RuntimeError, "handle relocation.*out of bounds"):
                    wp.capture_load(path, device="cpu")

    def test_save_rejects_invalid_mesh_handles(self):
        """Saving a scalar handle must reject unknown or deleted meshes without crashing."""
        mesh = create_unit_cube_mesh("cpu")
        stale_id = mesh.id
        del mesh
        handles = wp.zeros(1, dtype=MeshHandleOutput, device="cpu")
        distances = wp.zeros(1, dtype=float, device="cpu")
        wp.load_module(device="cpu")

        for name, candidate in (("unknown", 1), ("deleted", stale_id)):
            with self.subTest(handle=name), tempfile.TemporaryDirectory() as tmpdir:
                with wp.ScopedCapture(device="cpu", apic=True, force_module_load=False) as capture:
                    wp.launch(
                        overwrite_mesh_handle_struct_kernel,
                        dim=1,
                        inputs=[candidate, handles, distances],
                        device="cpu",
                    )
                # Do not launch: saving must validate the scalar ID before a
                # kernel or serializer can dereference it.
                with self.assertRaisesRegex(RuntimeError, "invalid or expired mesh ID"):
                    wp.capture_save(capture.graph, os.path.join(tmpdir, "invalid_mesh"))


@wp.kernel
def cpu_graph_mesh_hits(mesh: wp.handle, hits: wp.array[int]):
    query = wp.mesh_query_aabb(mesh, wp.vec3(9.5, -0.5, -0.5), wp.vec3(11.5, 1.5, 0.5))
    index = int(0)
    while wp.mesh_query_aabb_next(query, index):
        wp.atomic_add(hits, 0, 1)


@wp.kernel
def cpu_graph_mesh_attributes(mesh: wp.handle, values: wp.array[wp.vec3], offset: int):
    values[offset] = wp.mesh_eval_position(mesh, 0, 1.0, 0.0)
    values[offset + 1] = wp.mesh_eval_velocity(mesh, 0, 1.0, 0.0)


class TestCpuGraphGeometry(unittest.TestCase):
    @staticmethod
    def triangle(x):
        return np.array([(x, 0.0, 0.0), (x + 1.0, 0.0, 0.0), (x, 1.0, 0.0)], dtype=np.float32)

    def make_mesh(self, support_winding_number=False):
        return wp.Mesh(
            points=wp.array(self.triangle(0.0), dtype=wp.vec3, device="cpu"),
            indices=wp.array([0, 1, 2], dtype=int, device="cpu"),
            velocities=wp.zeros(3, dtype=wp.vec3, device="cpu"),
            support_winding_number=support_winding_number,
        )

    def test_refit_matches_eager_and_replays_current_points(self):
        """Refit moved geometry in sequence on every CPU graph launch."""
        wp.load_module(device="cpu")
        for winding in (False, True):
            with self.subTest(support_winding_number=winding):
                mesh = self.make_mesh(winding)
                source = wp.array(self.triangle(10.0), dtype=wp.vec3, device="cpu")
                hits = wp.zeros(1, dtype=int, device="cpu")

                def work(mesh, source, hits):
                    hits.zero_()
                    wp.copy(mesh.points, source)
                    mesh.refit()
                    wp.launch(cpu_graph_mesh_hits, dim=1, inputs=[mesh.id, hits], device="cpu")

                work(mesh, source, hits)
                np.testing.assert_array_equal(hits.numpy(), [1])
                mesh.points.assign(self.triangle(0.0))
                mesh.refit()
                hits.zero_()
                with wp.ScopedCapture(device="cpu") as capture:
                    work(mesh, source, hits)

                # CPU capture records operations without executing them.
                np.testing.assert_array_equal(mesh.points.numpy(), self.triangle(0.0))
                np.testing.assert_array_equal(hits.numpy(), [0])
                for x, expected in ((10.0, 1), (30.0, 0), (10.0, 1)):
                    source.assign(self.triangle(x))
                    wp.capture_launch(capture.graph)
                    np.testing.assert_array_equal(hits.numpy(), [expected])

    def test_in_place_attribute_updates_preserve_refit_order(self):
        """Replay point and velocity copies before each refit and geometry query."""
        mesh = self.make_mesh()
        points_a = wp.array(self.triangle(10.0), dtype=wp.vec3, device="cpu")
        points_b = wp.array(self.triangle(20.0), dtype=wp.vec3, device="cpu")
        velocities_a = wp.full(3, value=wp.vec3(1.0), dtype=wp.vec3, device="cpu")
        velocities_b = wp.full(3, value=wp.vec3(2.0), dtype=wp.vec3, device="cpu")
        values = wp.zeros(4, dtype=wp.vec3, device="cpu")
        hits_a = wp.zeros(1, dtype=int, device="cpu")
        hits_b = wp.zeros(1, dtype=int, device="cpu")
        wp.load_module(device="cpu")
        with wp.ScopedCapture(device="cpu") as capture:
            wp.copy(mesh.points, points_a)
            wp.copy(mesh.velocities, velocities_a)
            mesh.refit()
            wp.launch(cpu_graph_mesh_attributes, dim=1, inputs=[mesh.id, values, 0], device="cpu")
            hits_a.zero_()
            wp.launch(cpu_graph_mesh_hits, dim=1, inputs=[mesh.id, hits_a], device="cpu")
            wp.copy(mesh.points, points_b)
            wp.copy(mesh.velocities, velocities_b)
            mesh.refit()
            wp.launch(cpu_graph_mesh_attributes, dim=1, inputs=[mesh.id, values, 2], device="cpu")
            hits_b.zero_()
            wp.launch(cpu_graph_mesh_hits, dim=1, inputs=[mesh.id, hits_b], device="cpu")

        for x_a, x_b, v_a, v_b, expected_hits in ((10.0, 20.0, 1.0, 2.0, 1), (30.0, 40.0, 3.0, 4.0, 0)):
            points_a.assign(self.triangle(x_a))
            points_b.assign(self.triangle(x_b))
            velocities_a.fill_(wp.vec3(v_a))
            velocities_b.fill_(wp.vec3(v_b))
            wp.capture_launch(capture.graph)
            np.testing.assert_allclose(values.numpy(), [[x_a, 0, 0], [v_a] * 3, [x_b, 0, 0], [v_b] * 3])
            np.testing.assert_array_equal(hits_a.numpy(), [expected_hits])
            np.testing.assert_array_equal(hits_b.numpy(), [0])

    def test_attribute_replacement_rejects_capture_before_mutation(self):
        """Reject pointer replacement without changing Python or native mesh state."""
        wp.load_module(device="cpu")
        for apic in (False, True):
            for attribute in ("points", "velocities"):
                with self.subTest(apic=apic, attribute=attribute):
                    mesh = self.make_mesh()
                    original = getattr(mesh, attribute)
                    replacement = wp.full(3, value=wp.vec3(10.0), dtype=wp.vec3, device="cpu")
                    values = wp.zeros(2, dtype=wp.vec3, device="cpu")
                    # A rejected operation may be caught while capture continues.
                    with wp.ScopedCapture(device="cpu", apic=apic) as capture:
                        with self.assertRaisesRegex(NotImplementedError, "CPU graph capture"):
                            setattr(mesh, attribute, replacement)
                        self.assertIs(getattr(mesh, attribute), original)
                        wp.launch(cpu_graph_mesh_attributes, dim=1, inputs=[mesh.id, values, 0], device="cpu")
                    wp.capture_launch(capture.graph)
                    np.testing.assert_allclose(values.numpy(), [[0, 0, 0], [0, 0, 0]])
                    # Eager replacement must remain supported after capture ends.
                    setattr(mesh, attribute, replacement)
                    wp.launch(cpu_graph_mesh_attributes, dim=1, inputs=[mesh.id, values, 0], device="cpu")
                    expected = [[10, 10, 10], [0, 0, 0]] if attribute == "points" else [[0, 0, 0], [10, 10, 10]]
                    np.testing.assert_allclose(values.numpy(), expected)

    def test_graph_retains_mesh_and_refit_buffers(self):
        """Keep the mesh and its native refit buffers alive until graph destruction."""
        wp.load_module(device="cpu")

        def record():
            mesh = self.make_mesh()
            source = wp.array(self.triangle(10.0), dtype=wp.vec3, device="cpu")
            hits = wp.zeros(1, dtype=int, device="cpu")
            refs = {
                name: weakref.ref(value)
                for name, value in (
                    ("mesh", mesh),
                    ("points", mesh.points),
                    ("velocities", mesh.velocities),
                    ("indices", mesh.indices),
                    ("source", source),
                    ("hits", hits),
                )
            }
            with wp.ScopedCapture(device="cpu") as capture:
                hits.zero_()
                wp.copy(mesh.points, source)
                mesh.refit()
                wp.launch(cpu_graph_mesh_hits, dim=1, inputs=[mesh.id, hits], device="cpu")
            return capture.graph, refs

        graph, refs = record()
        gc.collect()
        for name, ref in refs.items():
            self.assertIsNotNone(ref(), f"Captured {name} was released before graph replay")
        for _ in range(2):
            wp.capture_launch(graph)
            np.testing.assert_array_equal(refs["hits"]().numpy(), [1])
        del graph
        gc.collect()
        for name, ref in refs.items():
            self.assertIsNone(ref(), f"Captured {name} leaked after graph destruction")

    def test_nested_graph_retains_mesh_and_refit_buffers(self):
        """Retain nested refit meshes after child graphs and caller references disappear."""
        wp.load_module(device="cpu")

        def wrap(graph, apic):
            with wp.ScopedCapture(device="cpu", apic=apic) as capture:
                wp.capture_launch(graph)
            return capture.graph

        def record(depth, apic):
            mesh = self.make_mesh()
            refs = [weakref.ref(value) for value in (mesh, mesh.points, mesh.velocities, mesh.indices)]
            with wp.ScopedCapture(device="cpu", apic=apic) as capture:
                mesh.refit()
            graph = capture.graph
            child_ref = weakref.ref(graph)
            del capture
            for _ in range(depth):
                graph = wrap(graph, apic)
            return graph, refs, child_ref

        hits = wp.zeros(1, dtype=int, device="cpu")
        for apic in (False, True):
            for depth in (1, 2):
                with self.subTest(apic=apic, depth=depth):
                    graph, refs, child_ref = record(depth, apic)
                    gc.collect()
                    self.assertIsNone(child_ref(), "The parent should not need to retain the child graph")
                    for ref in refs:
                        self.assertIsNotNone(ref(), "A nested refit lost its mesh or backing arrays")
                    for x, expected in ((10.0, 1), (30.0, 0), (10.0, 1)):
                        refs[1]().assign(self.triangle(x))
                        wp.capture_launch(graph)
                        hits.zero_()
                        wp.launch(cpu_graph_mesh_hits, dim=1, inputs=[refs[0]().id, hits], device="cpu")
                        np.testing.assert_array_equal(hits.numpy(), [expected])
                    if apic:
                        with tempfile.TemporaryDirectory() as tmpdir:
                            with self.assertRaisesRegex(RuntimeError, r"wp\.Mesh\.refit\(\)"):
                                wp.capture_save(graph, os.path.join(tmpdir, "nested_refit"))
                    del graph
                    gc.collect()
                    for ref in refs:
                        self.assertIsNone(ref(), "A nested refit leaked its mesh or backing arrays")

    def test_mesh_refit_rejects_serialization(self):
        """Reject process-local mesh refit records instead of saving dangling IDs."""
        mesh = self.make_mesh()
        with wp.ScopedCapture(device="cpu", apic=True) as capture:
            mesh.refit()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "mesh_update")
            with self.assertRaisesRegex(RuntimeError, r"wp\.Mesh\.refit\(\)"):
                wp.capture_save(capture.graph, path)
            self.assertFalse(os.path.exists(path + ".wrp"))


# Register the replay tests on CPU and CUDA devices.
devices = get_test_devices_with_cuda_graph_module_load()

add_function_test(TestApicMesh, "test_apic_mesh_query_point", test_apic_mesh_query_point, devices=devices)
add_function_test(TestApicMesh, "test_apic_mesh_query_ray", test_apic_mesh_query_ray, devices=devices)
add_function_test(TestApicMesh, "test_apic_mesh_eval_position", test_apic_mesh_eval_position, devices=devices)
add_function_test(
    TestApicMesh, "test_apic_mesh_combined_operations", test_apic_mesh_combined_operations, devices=devices
)
add_function_test(TestApicMesh, "test_apic_mesh_handle_in_struct", test_apic_mesh_handle_in_struct, devices=devices)
add_function_test(TestApicMesh, "test_apic_multiple_meshes", test_apic_multiple_meshes, devices=devices)
add_function_test(TestApicMesh, "test_apic_mesh_handle_array", test_apic_mesh_handle_array, devices=devices)
add_function_test(
    TestApicMesh, "test_apic_mesh_overlapping_handle_views", test_apic_mesh_overlapping_handle_views, devices=devices
)
add_function_test(
    TestApicMesh,
    "test_apic_mesh_save_before_first_launch",
    test_apic_mesh_save_before_first_launch,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
