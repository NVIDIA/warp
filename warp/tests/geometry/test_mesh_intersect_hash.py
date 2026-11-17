# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the mesh intersection example on CPU and available CUDA devices."""

import unittest

import numpy as np

import warp as wp
from warp.examples.geometry.example_mesh_intersect_hash import (
    intersect_models,
    make_box,
    mesh_edges,
    primitive_pair_intersects,
)
from warp.tests.geometry.test_intersect_tri import segment_triangle_reference
from warp.tests.unittest_utils import add_function_test, get_test_devices


@wp.kernel(enable_backward=False)
def intersect_exhaustive(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array2d[int],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array2d[int],
    hits: wp.array2d[int],
):
    i, j = wp.tid()
    hits[i, j] = wp.intersect_tri_tri(
        vertices_a[faces_a[i, 0]],
        vertices_a[faces_a[i, 1]],
        vertices_a[faces_a[i, 2]],
        vertices_b[faces_b[j, 0]],
        vertices_b[faces_b[j, 1]],
        vertices_b[faces_b[j, 2]],
    )


def check_mesh_pair(test, device, model_a, model_b, *, cell_size=0.5, grid_dim=16):
    """Compare the example's hashed result against an independent exhaustive reference."""
    candidates_host, hashed_host = intersect_models(model_a, model_b, device, cell_size=cell_size, grid_dim=grid_dim)
    arrays = []
    for vertices, faces in (model_a, model_b):
        arrays.extend((wp.array(vertices, dtype=wp.vec3, device=device), wp.array(faces, dtype=int, device=device)))
    shape = (len(model_a[1]), len(model_b[1]))
    exhaustive = wp.zeros(shape, dtype=int, device=device)
    wp.launch(intersect_exhaustive, dim=shape, inputs=[*arrays, exhaustive], device=device)
    exhaustive_host = exhaustive.numpy()
    np.testing.assert_array_equal(hashed_host, exhaustive_host)
    test.assertTrue(np.all(candidates_host[exhaustive_host != 0] == 1), "Broad phase missed or duplicated a hit")
    test.assertLessEqual(int(candidates_host.max()), 1, "A hash bucket returned a pair more than once")
    return int(np.count_nonzero(candidates_host)), int(np.count_nonzero(hashed_host))


def test_box_meshes(test, device):
    model_a = make_box()
    cases = (
        ("overlapping", make_box(translation=(0.7, 0.3, 0.2), angle=0.31), True),
        ("separated", make_box(translation=(4.0, 0.0, 0.0)), False),
        ("contained", make_box(scale=0.25, angle=0.31), False),
        # Coarse triangles must be found even when they span multiple cells.
        ("unequal_triangles", make_box(subdivisions=1, translation=(0.7, 0.3, 0.2), angle=0.31), True),
    )
    for name, model_b, expected_hit in cases:
        with test.subTest(case=name):
            candidates, hits = check_mesh_pair(test, device, model_a, model_b)
            total = len(model_a[1]) * len(model_b[1])
            test.assertEqual(hits > 0, expected_hit)
            test.assertLess(candidates, total)
            print(f"{device} {name}: {candidates}/{total} candidate pairs, {hits} intersecting pairs")


def test_hash_collisions(test, device):
    # Negative coordinates and a tiny table force distant cells into the same buckets.
    model_a = make_box(translation=(-10.0, -10.0, -10.0))
    model_b = make_box(subdivisions=1, translation=(-9.3, -9.7, -9.8), angle=0.31)
    candidates, hits = check_mesh_pair(test, device, model_a, model_b, cell_size=0.125, grid_dim=2)
    test.assertGreater(hits, 0)
    test.assertLess(candidates, len(model_a[1]) * len(model_b[1]))


def test_default_models(test, device):
    model_a = make_box(subdivisions=6)
    model_b = make_box(subdivisions=6, translation=(0.7, 0.3, 0.2), angle=0.31)
    candidates, hits = check_mesh_pair(test, device, model_a, model_b)
    test.assertGreater(hits, 0)
    test.assertLess(candidates, len(model_a[1]) * len(model_b[1]))


def test_triangle_contacts(test, device):
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    model_a = (np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32), faces)
    cases = (
        ("crossing", [[0.25, 0.25, -1], [0.25, 0.25, 1], [0.75, 0.25, 0]], 1),
        ("coplanar", [[0.1, 0.1, 0], [0.6, 0.1, 0], [0.1, 0.6, 0]], 1),
        ("touching", [[1, 0, 0], [2, 0, 1], [2, 1, 0]], 1),
        ("separated", [[0, 0, 1], [1, 0, 1], [0, 1, 1]], 0),
    )
    for name, vertices, expected_hits in cases:
        with test.subTest(case=name):
            model_b = (np.array(vertices, dtype=np.float32), faces)
            _, hits = check_mesh_pair(test, device, model_a, model_b, cell_size=0.25)
            test.assertEqual(hits, expected_hits)


def test_segment_models(test, device):
    # Match the example's default scene, plus coarse triangles and hash collisions.
    for subdivisions, shift, cell_size, grid_dim in (
        (6, 0.0, 0.5, 16),
        (1, -10.0, 0.125, 2),
    ):
        with test.subTest(subdivisions=subdivisions):
            model_a = mesh_edges(make_box(subdivisions=6, translation=(shift,) * 3))
            model_b = make_box(
                subdivisions=subdivisions, translation=(shift + 0.7, shift + 0.3, shift + 0.2), angle=0.31
            )
            candidates, hits = intersect_models(model_a, model_b, device, cell_size=cell_size, grid_dim=grid_dim)
            segments = model_a[0][model_a[1]].astype(np.float64)
            triangles = model_b[0][model_b[1]].astype(np.float64)
            # Exhaustive, independent NumPy reference, including pairs rejected
            # by the spatial hash. Neither model shares coplanar primitives.
            points = np.concatenate(
                (
                    np.broadcast_to(segments[:, None], (len(segments), len(triangles), 2, 3)),
                    np.broadcast_to(triangles[None], (len(segments), len(triangles), 3, 3)),
                ),
                axis=2,
            ).reshape(-1, 5, 3)
            reference = segment_triangle_reference(points).reshape(hits.shape)
            np.testing.assert_array_equal(hits, reference)
            test.assertTrue(np.all(candidates[reference] == 1), "Broad phase missed or duplicated a segment hit")
            test.assertLessEqual(int(candidates.max()), 1)
            test.assertGreater(int(reference.sum()), 0)
            test.assertLess(int(candidates.sum()), candidates.size)


def test_segment_contacts(test, device):
    model_b = (np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32), np.array([[0, 1, 2]], dtype=np.int32))
    segments = np.array(
        [
            [[0.25, 0.25, -1], [0.25, 0.25, 1]],
            [[-1, 0.25, 0], [2, 0.25, 0]],
            [[-1, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [1, 0, 0]],
            [[0.25, 0.25, 0], [0.25, 0.25, 0]],
            [[2, 2, 0], [3, 3, 0]],
            [[0.25, 0.25, 1], [0.5, 0.25, 1]],
        ],
        dtype=np.float32,
    )
    model_a = (segments.reshape(-1, 3), np.arange(segments.shape[0] * 2, dtype=np.int32).reshape(-1, 2))
    candidates, hits = intersect_models(model_a, model_b, device, cell_size=0.125, grid_dim=2)
    expected = np.array([[1], [1], [1], [1], [1], [0], [0]])
    np.testing.assert_array_equal(hits, expected)
    test.assertTrue(np.all(candidates[expected != 0] == 1))
    test.assertLessEqual(int(candidates.max()), 1)


@wp.kernel(enable_backward=False)
def intersect_exhaustive_with_tolerance(
    vertices_a: wp.array[wp.vec3],
    faces_a: wp.array2d[int],
    vertices_b: wp.array[wp.vec3],
    faces_b: wp.array2d[int],
    tol: wp.float64,
    hits: wp.array2d[int],
):
    i, j = wp.tid()
    hits[i, j] = int(primitive_pair_intersects(vertices_a, faces_a, vertices_b, faces_b, i, j, tol))


def exhaustive_tolerance_hits(device, model_a, model_b, tolerance):
    arrays = []
    for vertices, faces in (model_a, model_b):
        arrays.extend((wp.array(vertices, dtype=wp.vec3, device=device), wp.array(faces, dtype=int, device=device)))
    shape = (len(model_a[1]), len(model_b[1]))
    reference = wp.empty(shape, dtype=int, device=device)
    wp.launch(
        intersect_exhaustive_with_tolerance,
        dim=shape,
        inputs=[*arrays, wp.float64(tolerance), reference],
        device=device,
    )
    return reference.numpy()


def check_tolerance_pair(test, device, model_a, model_b, tolerance):
    candidates, hashed = intersect_models(model_a, model_b, device, cell_size=0.125, grid_dim=2, tol=tolerance)
    expected = exhaustive_tolerance_hits(device, model_a, model_b, tolerance)
    np.testing.assert_array_equal(hashed, expected)
    test.assertTrue(np.all(candidates[expected != 0] == 1), "Tolerance broad phase missed or duplicated a hit")
    test.assertLessEqual(int(candidates.max()), 1)
    return candidates, hashed


def test_tolerance_candidates(test, device):
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    tiny = np.array([(0, 0, 0), (0.001, 0, 0), (0, 0.001, 0)], dtype=np.float32)
    separated = tiny + np.array([0, 0, 0.1], dtype=np.float32)
    # The default plane tolerance accepts these separated parallel triangles.
    # Fixed sphere padding would discard them before the narrow phase.
    for a, b in ((tiny, separated), (separated, tiny)):
        _, hits = check_tolerance_pair(test, device, (a, faces), (b, faces), 1e-6)
        test.assertEqual(int(hits[0, 0]), 1)
        _, hits = check_tolerance_pair(test, device, (a, faces), (b, faces), 0.0)
        test.assertEqual(int(hits[0, 0]), 0)


def test_near_parallel_padding_limit(test, device):
    # Characterize the documented limitation: simple padding does not enclose
    # all the tangential displacements accepted by positive plane tolerance.
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    a = np.array([(-2, 0, 0), (-1, 0, 0), (-1, 1, 0)], dtype=np.float32)
    b = np.array([(1, 0, 0.75e-6), (2, 0, 1.5e-6), (1, 1, 0.75e-6)], dtype=np.float32)
    for axis in range(3):
        for scale in (0.125, 1, 4):
            aa, bb = np.roll(a, axis, axis=1) * scale, np.roll(b, axis, axis=1) * scale
            model_a, model_b = (aa, faces), (bb, faces)
            tolerance = 1e-6 * scale**3
            candidates, hits = intersect_models(model_a, model_b, device, tol=tolerance)
            exhaustive = exhaustive_tolerance_hits(device, model_a, model_b, tolerance)
            test.assertEqual(int(exhaustive[0, 0]), 1)
            test.assertEqual(int(candidates[0, 0]), 0)
            test.assertEqual(int(hits[0, 0]), 0)
            # With snapping disabled there is no geometric intersection.
            _, hits = check_tolerance_pair(test, device, model_a, model_b, 0.0)
            test.assertEqual(int(hits[0, 0]), 0)


def test_segment_tolerance_candidates(test, device):
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    tiny = np.array([(0, 0, 0), (0.001, 0, 0), (0, 0.001, 0)], dtype=np.float32)
    segments = np.array(
        [
            [(0.0001, 0.0001, 0.1), (0.0002, 0.0001, 0.1)],
            [(0.0001, 0.0001, -0.1), (0.0002, 0.0001, -0.1)],
            [(0.0001, 0.0001, 0.1), (0.0001, 0.0001, 0.1)],
            [(0.1, 0.1, 0.1), (0.2, 0.1, 0.1)],
        ],
        dtype=np.float32,
    )
    edges = np.arange(len(segments) * 2, dtype=np.int32).reshape(-1, 2)
    _, hits = check_tolerance_pair(test, device, (segments.reshape(-1, 3), edges), (tiny, faces), 1e-6)
    np.testing.assert_array_equal(hits[:, 0], [1, 1, 1, 0])
    _, hits = check_tolerance_pair(test, device, (segments.reshape(-1, 3), edges), (tiny, faces), 0.0)
    np.testing.assert_array_equal(hits[:, 0], [0, 0, 0, 0])


def test_tolerance_cutoff_and_random_pairs(test, device):
    # Powers of two map the plane cutoff to a large spatial separation, which
    # the unexpanded hash cannot find. Test both sides of the float32 cutoff.
    size = 2.0**-10
    base = np.array([(0, 0, 0), (size, 0, 0), (0, size, 0)], dtype=np.float32)
    cutoff = np.float32(1e-6)
    heights = (
        np.array(
            [np.nextafter(cutoff, np.float32(0)), cutoff, np.nextafter(cutoff, np.float32(np.inf))], dtype=np.float32
        )
        / size**2
    )
    targets = np.stack([base + np.array([0, 0, float(h)]) for h in heights]).astype(np.float32)
    faces = np.arange(len(targets) * 3, dtype=np.int32).reshape(-1, 3)
    _, hits = check_tolerance_pair(
        test, device, (base, np.array([[0, 1, 2]], dtype=np.int32)), (targets.reshape(-1, 3), faces), 1e-6
    )
    np.testing.assert_array_equal(hits[0], [1, 1, 0])

    rng = np.random.default_rng(719)
    a = rng.uniform(-1, 1, (32, 3, 3)).astype(np.float32)
    b = rng.uniform(-1, 1, (32, 3, 3)).astype(np.float32)
    a[:16] *= 0.001
    b[:16] *= 0.001
    b[:16, :, 2] += 0.1
    faces = np.arange(96, dtype=np.int32).reshape(-1, 3)
    for tolerance in (0.0, 1e-7, 1e-6, 1e-5):
        check_tolerance_pair(test, device, (a.reshape(-1, 3), faces), (b.reshape(-1, 3), faces), tolerance)


class TestMeshIntersectHash(unittest.TestCase):
    def test_unique_mesh_edges(self):
        for subdivisions in (1, 3, 6):
            with self.subTest(subdivisions=subdivisions):
                vertices, edges = mesh_edges(make_box(subdivisions=subdivisions))
                # A triangulated sphere has E=3F/2 and V=E-F+2.
                self.assertEqual(len(edges), 18 * subdivisions**2)
                self.assertEqual(len(vertices), 6 * subdivisions**2 + 2)
                self.assertTrue(np.all(edges[:, 0] < edges[:, 1]))
                self.assertEqual(len(np.unique(edges, axis=0)), len(edges))


devices = get_test_devices()
add_function_test(
    TestMeshIntersectHash, "test_near_parallel_padding_limit", test_near_parallel_padding_limit, devices=devices
)
add_function_test(TestMeshIntersectHash, "test_tolerance_candidates", test_tolerance_candidates, devices=devices)
add_function_test(
    TestMeshIntersectHash, "test_segment_tolerance_candidates", test_segment_tolerance_candidates, devices=devices
)
add_function_test(
    TestMeshIntersectHash,
    "test_tolerance_cutoff_and_random_pairs",
    test_tolerance_cutoff_and_random_pairs,
    devices=devices,
)
add_function_test(TestMeshIntersectHash, "test_box_meshes", test_box_meshes, devices=devices, check_output=False)
add_function_test(TestMeshIntersectHash, "test_hash_collisions", test_hash_collisions, devices=devices)
add_function_test(TestMeshIntersectHash, "test_default_models", test_default_models, devices=devices)
add_function_test(TestMeshIntersectHash, "test_triangle_contacts", test_triangle_contacts, devices=devices)
add_function_test(TestMeshIntersectHash, "test_segment_models", test_segment_models, devices=devices)
add_function_test(TestMeshIntersectHash, "test_segment_contacts", test_segment_contacts, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
