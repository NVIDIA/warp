# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools
import os
import unittest

import numpy as np

import warp as wp
import warp.examples
from warp.tests.unittest_utils import *


@wp.func
def min_vec3(a: wp.vec3, b: wp.vec3):
    return wp.vec3(wp.min(a[0], b[0]), wp.min(a[1], b[1]), wp.min(a[2], b[2]))


@wp.func
def max_vec3(a: wp.vec3, b: wp.vec3):
    return wp.vec3(wp.max(a[0], b[0]), wp.max(a[1], b[1]), wp.max(a[2], b[2]))


@wp.kernel
def compute_bounds(
    indices: wp.array[int],
    positions: wp.array[wp.vec3],
    lowers: wp.array[wp.vec3],
    uppers: wp.array[wp.vec3],
):
    tid = wp.tid()
    i = indices[tid * 3 + 0]
    j = indices[tid * 3 + 1]
    k = indices[tid * 3 + 2]

    x0 = positions[i]  # point zero
    x1 = positions[j]  # point one
    x2 = positions[k]  # point two

    lower = min_vec3(min_vec3(x0, x1), x2)
    upper = max_vec3(max_vec3(x0, x1), x2)

    lowers[tid] = lower
    uppers[tid] = upper


@wp.kernel
def compute_num_contacts(
    lowers: wp.array[wp.vec3], uppers: wp.array[wp.vec3], mesh_id: wp.uint64, counts: wp.array[int]
):
    tid = wp.tid()

    upper = uppers[tid]
    lower = lowers[tid]

    query = wp.mesh_query_aabb(mesh_id, lower, upper)
    count = int(0)

    # index = int(-1)
    # while wp.mesh_query_aabb_next(query, index):

    for _index in query:
        count = count + 1

    counts[tid] = count


def test_compute_bounds(test, device):
    # create two touching triangles.
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, -1, 1]])
    indices = np.array([0, 1, 2, 1, 2, 3])
    m = wp.Mesh(
        points=wp.array(points, dtype=wp.vec3, device=device),
        indices=wp.array(indices, dtype=int, device=device),
    )

    num_tris = int(len(indices) / 3)

    # First compute bounds of each of the triangles.
    lowers = wp.empty(n=num_tris, dtype=wp.vec3, device=device)
    uppers = wp.empty_like(lowers)
    wp.launch(
        kernel=compute_bounds,
        dim=num_tris,
        inputs=[m.indices, m.points],
        outputs=[lowers, uppers],
        device=device,
    )

    lower_view = lowers.numpy()
    upper_view = uppers.numpy()

    # Confirm the bounds of each triangle are correct.
    test.assertTrue(lower_view[0][0] == 0)
    test.assertTrue(lower_view[0][1] == 0)
    test.assertTrue(lower_view[0][2] == 0)

    test.assertTrue(upper_view[0][0] == 1)
    test.assertTrue(upper_view[0][1] == 1)
    test.assertTrue(upper_view[0][2] == 0)

    test.assertTrue(lower_view[1][0] == -1)
    test.assertTrue(lower_view[1][1] == -1)
    test.assertTrue(lower_view[1][2] == 0)

    test.assertTrue(upper_view[1][0] == 1)
    test.assertTrue(upper_view[1][1] == 1)
    test.assertTrue(upper_view[1][2] == 1)


def test_mesh_query_aabb_count_overlap(test, device):
    # create two touching triangles.
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, -1, 1]])
    indices = np.array([0, 1, 2, 1, 2, 3])
    m = wp.Mesh(
        points=wp.array(points, dtype=wp.vec3, device=device),
        indices=wp.array(indices, dtype=int, device=device),
    )

    num_tris = int(len(indices) / 3)

    # Compute AABB of each of the triangles.
    lowers = wp.empty(n=num_tris, dtype=wp.vec3, device=device)
    uppers = wp.empty_like(lowers)
    wp.launch(
        kernel=compute_bounds,
        dim=num_tris,
        inputs=[m.indices, m.points],
        outputs=[lowers, uppers],
        device=device,
    )

    counts = wp.empty(n=num_tris, dtype=int, device=device)

    wp.launch(
        kernel=compute_num_contacts,
        dim=num_tris,
        inputs=[lowers, uppers, m.id],
        outputs=[counts],
        device=device,
    )

    view = counts.numpy()

    # 2 triangles that share a vertex having overlapping AABBs.
    for c in view:
        test.assertTrue(c == 2)


def test_mesh_query_aabb_count_nonoverlap(test, device):
    # create two separate triangles.
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [10, 0, 0], [10, 1, 0], [10, 0, 1]])
    indices = np.array([0, 1, 2, 3, 4, 5])
    m = wp.Mesh(
        points=wp.array(points, dtype=wp.vec3, device=device),
        indices=wp.array(indices, dtype=int, device=device),
    )

    num_tris = int(len(indices) / 3)

    lowers = wp.empty(n=num_tris, dtype=wp.vec3, device=device)
    uppers = wp.empty_like(lowers)
    wp.launch(
        kernel=compute_bounds,
        dim=num_tris,
        inputs=[m.indices, m.points],
        outputs=[lowers, uppers],
        device=device,
    )

    counts = wp.empty(n=num_tris, dtype=int, device=device)

    wp.launch(
        kernel=compute_num_contacts,
        dim=num_tris,
        inputs=[lowers, uppers, m.id],
        outputs=[counts],
        device=device,
    )

    view = counts.numpy()

    # AABB query only returns one triangle at a time, the triangles are not close enough to overlap.
    for c in view:
        test.assertTrue(c == 1)


@wp.kernel
def compute_num_contact_with_checksums(
    lowers: wp.array[wp.vec3],
    uppers: wp.array[wp.vec3],
    mesh_id: wp.uint64,
    counts: wp.array[int],
    check_sums: wp.array[int],
):
    tid = wp.tid()

    upper = uppers[tid]
    lower = lowers[tid]

    query = wp.mesh_query_aabb(mesh_id, lower, upper)
    count = int(0)

    check_sum = int(0)
    for _index in query:
        check_sum = check_sum ^ _index
        count = count + 1

    counts[tid] = count
    check_sums[tid] = check_sum


@wp.func
def intersect_aabb_aabb(a_lower: wp.vec3, a_upper: wp.vec3, b_lower: wp.vec3, b_upper: wp.vec3):
    if (
        a_lower[0] > b_upper[0]
        or a_lower[1] > b_upper[1]
        or a_lower[2] > b_upper[2]
        or a_upper[0] < b_lower[0]
        or a_upper[1] < b_lower[1]
        or a_upper[2] < b_lower[2]
    ):
        return False
    else:
        return True


@wp.kernel
def compute_num_contact_with_checksums_brutal(
    lowers: wp.array[wp.vec3],
    uppers: wp.array[wp.vec3],
    mesh_points: wp.array[wp.vec3],
    mesh_indices: wp.array[int],
    counts: wp.array[int],
    check_sums: wp.array[int],
):
    tid = wp.tid()

    upper = uppers[tid]
    lower = lowers[tid]

    check_sum = int(0)
    count = int(0)
    num_faces = mesh_indices.shape[0] / 3

    for face in range(num_faces):
        i = mesh_indices[face * 3 + 0]
        j = mesh_indices[face * 3 + 1]
        k = mesh_indices[face * 3 + 2]

        x0 = mesh_points[i]  # point zero
        x1 = mesh_points[j]  # point one
        x2 = mesh_points[k]  # point two

        tri_lower = min_vec3(min_vec3(x0, x1), x2)
        tri_upper = max_vec3(max_vec3(x0, x1), x2)

        if intersect_aabb_aabb(lower, upper, tri_lower, tri_upper):
            check_sum = check_sum ^ face
            count = count + 1

    counts[tid] = count
    check_sums[tid] = check_sum


def load_mesh():
    from pxr import Usd, UsdGeom  # noqa: PLC0415

    usd_stage = Usd.Stage.Open(os.path.join(wp.examples.get_asset_directory(), "bunny.usd"))
    usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/root/bunny"))

    vertices = np.array(usd_geom.GetPointsAttr().Get())
    faces = np.array(usd_geom.GetFaceVertexIndicesAttr().Get())

    return vertices, faces


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
def test_mesh_query_aabb_count_overlap_with_checksum(test, device):
    if device.is_cpu:
        constructors = ["sah", "median"]
    else:
        constructors = ["sah", "median", "lbvh"]

    if wp.is_cubql_available():
        constructors.append("cubql")

    leaf_sizes = [1, 2, 4]

    points, indices = load_mesh()
    points_wp = wp.array(points, dtype=wp.vec3, device=device)
    indices_wp = wp.array(indices, dtype=int, device=device)

    for leaf_size, constructor in itertools.product(leaf_sizes, constructors):
        m = wp.Mesh(points=points_wp, indices=indices_wp, bvh_constructor=constructor, bvh_leaf_size=leaf_size)

        num_test_bounds = 10000
        test_bound_relative_size = 0.01

        world_min = np.min(points, axis=0)
        world_max = np.max(points, axis=0)

        world_center = 0.5 * (world_min + world_max)
        world_size = world_max - world_min

        rng = np.random.default_rng(123)

        centers = (
            rng.uniform(-0.5, 0.5, size=num_test_bounds * 3).reshape(num_test_bounds, 3) * world_size + world_center
        )
        diffs = (
            0.5 * test_bound_relative_size * rng.random(num_test_bounds * 3).reshape(num_test_bounds, 3) * world_size
        )

        lowers = wp.array(centers - diffs, dtype=wp.vec3, device=device)
        uppers = wp.array(centers + diffs, dtype=wp.vec3, device=device)

        counts = wp.empty(n=num_test_bounds, dtype=int, device=device)
        checksums = wp.empty(n=num_test_bounds, dtype=int, device=device)

        wp.launch(
            kernel=compute_num_contact_with_checksums,
            dim=num_test_bounds,
            inputs=[lowers, uppers, m.id],
            outputs=[counts, checksums],
            device=device,
        )

        counts_brutal = wp.empty(n=num_test_bounds, dtype=int, device=device)
        checksums_brutal = wp.empty(n=num_test_bounds, dtype=int, device=device)

        wp.launch(
            kernel=compute_num_contact_with_checksums_brutal,
            dim=num_test_bounds,
            inputs=[lowers, uppers, points_wp, indices_wp],
            outputs=[counts_brutal, checksums_brutal],
            device=device,
        )

        assert_array_equal(counts, counts_brutal)
        assert_array_equal(checksums, checksums_brutal)


@wp.kernel
def tile_mesh_query_aabb_kernel(
    mesh_id: wp.uint64,
    lower: wp.vec3,
    upper: wp.vec3,
    faces_intersected: wp.array[int],
):
    query = wp.tile_mesh_query_aabb(mesh_id, lower, upper)

    while wp.tile_query_valid(query):
        result_tile = wp.tile_mesh_query_aabb_next(query)
        result_idx = wp.untile(result_tile)

        # Mark faces as intersected using atomic add (skip -1 which means no result)
        # This ensures we can verify that each face is only reported once
        if result_idx >= 0:
            wp.atomic_add(faces_intersected, result_idx, 1)


@wp.kernel
def mesh_query_aabb_kernel(
    mesh_id: wp.uint64,
    lower: wp.vec3,
    upper: wp.vec3,
    faces_intersected: wp.array[int],
):
    query = wp.mesh_query_aabb(mesh_id, lower, upper)

    index = int(0)
    while wp.mesh_query_aabb_next(query, index):
        wp.atomic_add(faces_intersected, index, 1)


def test_tile_mesh_query_aabb(test, device):
    """Test tile-based mesh AABB query and compare with single-threaded version."""
    # Create a simple mesh (two triangles forming a quad)
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    indices = np.array(
        [
            0,
            1,
            2,  # First triangle
            0,
            2,
            3,  # Second triangle
        ],
        dtype=np.int32,
    )

    points_wp = wp.array(points, dtype=wp.vec3, device=device)
    indices_wp = wp.array(indices, dtype=int, device=device)

    # Cover the cuBQL constructor alongside the default Warp BVH path.
    if device.is_cpu:
        constructors = ["sah", "median"]
    else:
        constructors = ["sah", "median", "lbvh"]

    if wp.is_cubql_available():
        constructors.append("cubql")

    query_lower = wp.vec3(0.2, 0.2, -0.5)
    query_upper = wp.vec3(0.8, 0.8, 0.5)

    for constructor in constructors:
        mesh = wp.Mesh(points=points_wp, indices=indices_wp, bvh_constructor=constructor)

        # Test with single-threaded version (ground truth)
        faces_intersected_single = wp.zeros(shape=(2), dtype=int, device=device)
        wp.launch(
            kernel=mesh_query_aabb_kernel,
            dim=1,
            inputs=[mesh.id, query_lower, query_upper, faces_intersected_single],
            device=device,
        )

        # Test with tile-based version
        block_dim = 64
        faces_intersected_tile = wp.zeros(shape=(2), dtype=int, device=device)
        wp.launch_tiled(
            kernel=tile_mesh_query_aabb_kernel,
            dim=1,
            inputs=[mesh.id, query_lower, query_upper, faces_intersected_tile],
            device=device,
            block_dim=block_dim,
        )

        # Compare results
        single_result = faces_intersected_single.numpy()
        tile_result = faces_intersected_tile.numpy()

        for i in range(2):
            test.assertEqual(
                single_result[i],
                tile_result[i],
                f"[{constructor}] Mismatch at face {i}: single={single_result[i]}, tile={tile_result[i]}",
            )

        # Both triangles should be found exactly once
        test.assertEqual(single_result[0], 1, msg=f"[{constructor}] expected 1 hit on face 0")
        test.assertEqual(single_result[1], 1, msg=f"[{constructor}] expected 1 hit on face 1")

        # Also test tile_query_valid-based loop
        faces_intersected_count = wp.zeros(shape=(2), dtype=int, device=device)
        wp.launch_tiled(
            kernel=tile_mesh_query_aabb_valid_kernel,
            dim=1,
            inputs=[mesh.id, query_lower, query_upper, faces_intersected_count],
            device=device,
            block_dim=block_dim,
        )
        count_result = faces_intersected_count.numpy()
        for i in range(2):
            test.assertEqual(
                single_result[i],
                count_result[i],
                f"[{constructor}] tile_query_valid mismatch at face {i}: "
                f"single={single_result[i]}, count={count_result[i]}",
            )


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
def test_tile_mesh_query_aabb_large(test, device):
    """Test tile-based mesh AABB query with a larger mesh (bunny)."""
    points, indices = load_mesh()

    mesh = wp.Mesh(
        points=wp.array(points, dtype=wp.vec3, device=device), indices=wp.array(indices, dtype=int, device=device)
    )

    num_faces = len(indices) // 3

    # Create a query box that should intersect multiple triangles
    world_min = np.min(points, axis=0)
    world_max = np.max(points, axis=0)
    world_center = 0.5 * (world_min + world_max)
    world_size = world_max - world_min

    query_size = 0.1 * world_size
    query_lower = wp.vec3(
        world_center[0] - query_size[0], world_center[1] - query_size[1], world_center[2] - query_size[2]
    )
    query_upper = wp.vec3(
        world_center[0] + query_size[0], world_center[1] + query_size[1], world_center[2] + query_size[2]
    )

    # Test with single-threaded version (ground truth)
    faces_intersected_single = wp.zeros(shape=(num_faces), dtype=int, device=device)
    wp.launch(
        kernel=mesh_query_aabb_kernel,
        dim=1,
        inputs=[mesh.id, query_lower, query_upper, faces_intersected_single],
        device=device,
    )

    # Test with tile-based version
    block_dim = 64
    faces_intersected_tile = wp.zeros(shape=(num_faces), dtype=int, device=device)
    wp.launch_tiled(
        kernel=tile_mesh_query_aabb_kernel,
        dim=1,
        inputs=[mesh.id, query_lower, query_upper, faces_intersected_tile],
        device=device,
        block_dim=block_dim,
    )

    # Compare results
    single_result = faces_intersected_single.numpy()
    tile_result = faces_intersected_tile.numpy()

    for i in range(num_faces):
        test.assertEqual(
            single_result[i],
            tile_result[i],
            f"Mismatch at face {i}: single={single_result[i]}, tile={tile_result[i]}",
        )


# Tests for new mesh_query_aabb_tiled() API (primary naming convention)
@wp.kernel
def mesh_query_aabb_tiled_kernel(
    mesh_id: wp.uint64,
    lower: wp.vec3,
    upper: wp.vec3,
    faces_intersected: wp.array[int],
):
    query = wp.mesh_query_aabb_tiled(mesh_id, lower, upper)

    while wp.tile_query_valid(query):
        result_tile = wp.mesh_query_aabb_next_tiled(query)
        result_idx = wp.untile(result_tile)

        # Mark faces as intersected using atomic add (skip -1 which means no result)
        if result_idx >= 0:
            wp.atomic_add(faces_intersected, result_idx, 1)


def test_mesh_query_aabb_tiled(test, device):
    """Test mesh_query_aabb_tiled() API (new primary naming convention)."""
    # Create a simple mesh (two triangles forming a quad)
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    indices = np.array(
        [
            0,
            1,
            2,  # First triangle
            0,
            2,
            3,  # Second triangle
        ],
        dtype=np.int32,
    )

    mesh = wp.Mesh(
        points=wp.array(points, dtype=wp.vec3, device=device), indices=wp.array(indices, dtype=int, device=device)
    )

    query_lower = wp.vec3(0.2, 0.2, -0.5)
    query_upper = wp.vec3(0.8, 0.8, 0.5)

    # Test with single-threaded version (ground truth)
    faces_intersected_single = wp.zeros(shape=(2), dtype=int, device=device)
    wp.launch(
        kernel=mesh_query_aabb_kernel,
        dim=1,
        inputs=[mesh.id, query_lower, query_upper, faces_intersected_single],
        device=device,
    )

    # Test with new tiled API
    block_dim = 64
    faces_intersected_tiled = wp.zeros(shape=(2), dtype=int, device=device)
    wp.launch_tiled(
        kernel=mesh_query_aabb_tiled_kernel,
        dim=1,
        inputs=[mesh.id, query_lower, query_upper, faces_intersected_tiled],
        device=device,
        block_dim=block_dim,
    )

    # Compare results
    single_result = faces_intersected_single.numpy()
    tiled_result = faces_intersected_tiled.numpy()

    for i in range(2):
        test.assertEqual(
            single_result[i],
            tiled_result[i],
            f"Mismatch at face {i}: single={single_result[i]}, tiled={tiled_result[i]}",
        )


@wp.kernel
def tile_mesh_query_aabb_valid_kernel(
    mesh_id: wp.uint64,
    lower: wp.vec3,
    upper: wp.vec3,
    faces_intersected: wp.array[int],
):
    query = wp.tile_mesh_query_aabb(mesh_id, lower, upper)

    while wp.tile_query_valid(query):
        result_tile = wp.tile_mesh_query_aabb_next(query)
        result_idx = wp.untile(result_tile)

        if result_idx >= 0:
            wp.atomic_add(faces_intersected, result_idx, 1)


@wp.kernel
def mesh_query_sphere_hits(mesh_id: wp.uint64, center: wp.vec3, radius: float, hits: wp.array[int]):
    query = wp.mesh_query_sphere(mesh_id, center, radius)
    face = int(0)
    while wp.mesh_query_next(query, face):
        hits[face] = 1


@wp.kernel
def mesh_query_sphere_hits_for_loop(mesh_id: wp.uint64, center: wp.vec3, radius: float, hits: wp.array[int]):
    query = wp.mesh_query_sphere(mesh_id, center, radius)
    for face in query:
        hits[face] = 1


@wp.kernel
def mesh_query_sphere_hits_alias(mesh_id: wp.uint64, center: wp.vec3, radius: float, hits: wp.array[int]):
    query = wp.mesh_query_sphere(mesh_id, center, radius)
    face = int(0)
    while wp.mesh_query_aabb_next(query, face):
        hits[face] = 1


@wp.kernel
def mesh_query_aabb_hits(mesh_id: wp.uint64, lower: wp.vec3, upper: wp.vec3, hits: wp.array[int]):
    query = wp.mesh_query_aabb(mesh_id, lower, upper)
    face = int(0)
    while wp.mesh_query_next(query, face):
        hits[face] = 1


@wp.kernel
def mesh_query_runtime_kind_hits(
    mesh_id: wp.uint64,
    use_sphere: bool,
    lower: wp.vec3,
    upper: wp.vec3,
    center: wp.vec3,
    radius: float,
    hits: wp.array[int],
):
    # The kind is selected by a runtime branch, so the merged query decays to the
    # erased parent MeshQuery and iterates via the kind stored at construction.
    if use_sphere:
        query = wp.mesh_query_sphere(mesh_id, center, radius)
    else:
        query = wp.mesh_query_aabb(mesh_id, lower, upper)
    face = int(0)
    while wp.mesh_query_next(query, face):
        hits[face] = 1


@wp.func
def collect_mesh_hits(query: wp.MeshQuery, hits: wp.array[int]):
    # Parameter annotated with the erased parent type: any concrete query kind is
    # accepted, and iteration dispatches on the kind stored at construction.
    face = int(0)
    while wp.mesh_query_next(query, face):
        hits[face] = 1


@wp.func
def collect_mesh_aabb_hits(query: wp.MeshQueryAABB, hits: wp.array[int]):
    # Parameter annotated with the concrete AABB kind (pre-rename convention):
    # keeps the statically-typed AABB iterator.
    face = int(0)
    while wp.mesh_query_next(query, face):
        hits[face] = 1


@wp.kernel
def mesh_query_sphere_via_erased_func(mesh_id: wp.uint64, center: wp.vec3, radius: float, hits: wp.array[int]):
    query = wp.mesh_query_sphere(mesh_id, center, radius)
    collect_mesh_hits(query, hits)


@wp.kernel
def mesh_query_aabb_via_typed_func(mesh_id: wp.uint64, lower: wp.vec3, upper: wp.vec3, hits: wp.array[int]):
    query = wp.mesh_query_aabb(mesh_id, lower, upper)
    collect_mesh_aabb_hits(query, hits)


def test_mesh_query_sphere_alias_next(test, device):
    # mesh_query_aabb_next is documented as an alias for mesh_query_next, so it must
    # also advance sphere queries with the sphere narrow phase.
    rng = np.random.default_rng(123)
    m, tris, _lowers, _uppers = _random_triangle_mesh(device, rng)
    num_tris = tris.shape[0]
    hits_canonical = wp.zeros(num_tris, dtype=int, device=device)
    hits_alias = wp.zeros(num_tris, dtype=int, device=device)
    center = wp.vec3(4.0, 4.0, 4.0)
    wp.launch(mesh_query_sphere_hits, dim=1, inputs=[m.id, center, 1.5, hits_canonical], device=device)
    wp.launch(mesh_query_sphere_hits_alias, dim=1, inputs=[m.id, center, 1.5, hits_alias], device=device)
    expected = hits_canonical.numpy()
    test.assertGreater(expected.sum(), 0)
    np.testing.assert_array_equal(hits_alias.numpy(), expected)


def test_mesh_query_sphere_for_loop(test, device):
    # The for-loop protocol dispatches through a shared iter_cmp() that must select the
    # sphere iterator even when radius == 0 (radius_sq alone cannot distinguish a
    # zero-radius sphere from an AABB query). A zero-radius sphere keeps only triangles
    # that pass exactly through the center, not every triangle whose AABB contains it.
    points = wp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=wp.vec3, device=device)
    indices = wp.array([0, 1, 2], dtype=int, device=device)
    m = wp.Mesh(points=points, indices=indices)

    hits = wp.zeros(1, dtype=int, device=device)

    # (0.25, 0.25, 0) lies on the triangle: expect a hit
    wp.launch(mesh_query_sphere_hits_for_loop, dim=1, inputs=[m.id, wp.vec3(0.25, 0.25, 0.0), 0.0, hits], device=device)
    test.assertEqual(hits.numpy()[0], 1)

    # (0.9, 0.9, 0) is inside the triangle's AABB but not on the triangle: expect no hit
    hits.zero_()
    wp.launch(mesh_query_sphere_hits_for_loop, dim=1, inputs=[m.id, wp.vec3(0.9, 0.9, 0.0), 0.0, hits], device=device)
    test.assertEqual(hits.numpy()[0], 0)

    # nonzero radius through the for-loop must match the while-loop form
    rng = np.random.default_rng(123)
    m2, _tris, _lowers, _uppers = _random_triangle_mesh(device, rng)
    num_tris = _tris.shape[0]
    hits_while = wp.zeros(num_tris, dtype=int, device=device)
    hits_for = wp.zeros(num_tris, dtype=int, device=device)
    center = wp.vec3(4.0, 4.0, 4.0)
    wp.launch(mesh_query_sphere_hits, dim=1, inputs=[m2.id, center, 1.5, hits_while], device=device)
    wp.launch(mesh_query_sphere_hits_for_loop, dim=1, inputs=[m2.id, center, 1.5, hits_for], device=device)
    expected = hits_while.numpy()
    test.assertGreater(expected.sum(), 0)
    np.testing.assert_array_equal(hits_for.numpy(), expected)


def test_mesh_query_runtime_kind(test, device):
    # A query whose kind is chosen by a runtime branch decays to the erased parent
    # MeshQuery type and must match the statically-typed kernels for both kinds.
    rng = np.random.default_rng(123)
    m, tris, _lowers, _uppers = _random_triangle_mesh(device, rng)
    num_tris = tris.shape[0]
    expected = wp.zeros(num_tris, dtype=int, device=device)
    actual = wp.zeros(num_tris, dtype=int, device=device)
    lower = wp.vec3(2.0, 2.0, 2.0)
    upper = wp.vec3(6.0, 6.0, 6.0)
    center = wp.vec3(4.0, 4.0, 4.0)

    wp.launch(mesh_query_sphere_hits, dim=1, inputs=[m.id, center, 1.5, expected], device=device)
    wp.launch(
        mesh_query_runtime_kind_hits, dim=1, inputs=[m.id, True, lower, upper, center, 1.5, actual], device=device
    )
    reference = expected.numpy()
    test.assertGreater(reference.sum(), 0)
    np.testing.assert_array_equal(actual.numpy(), reference)

    expected.zero_()
    actual.zero_()
    wp.launch(mesh_query_aabb_hits, dim=1, inputs=[m.id, lower, upper, expected], device=device)
    wp.launch(
        mesh_query_runtime_kind_hits, dim=1, inputs=[m.id, False, lower, upper, center, 1.5, actual], device=device
    )
    reference = expected.numpy()
    test.assertGreater(reference.sum(), 0)
    np.testing.assert_array_equal(actual.numpy(), reference)

    # A zero-radius sphere through the erased path must keep the sphere narrow phase:
    # an on-triangle point hits, a point merely inside the triangle's AABB does not.
    points = wp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=wp.vec3, device=device)
    indices = wp.array([0, 1, 2], dtype=int, device=device)
    m1 = wp.Mesh(points=points, indices=indices)
    hits = wp.zeros(1, dtype=int, device=device)
    wp.launch(
        mesh_query_runtime_kind_hits,
        dim=1,
        inputs=[m1.id, True, lower, upper, wp.vec3(0.25, 0.25, 0.0), 0.0, hits],
        device=device,
    )
    test.assertEqual(hits.numpy()[0], 1)
    hits.zero_()
    wp.launch(
        mesh_query_runtime_kind_hits,
        dim=1,
        inputs=[m1.id, True, lower, upper, wp.vec3(0.9, 0.9, 0.0), 0.0, hits],
        device=device,
    )
    test.assertEqual(hits.numpy()[0], 0)


def test_mesh_query_erased_func_param(test, device):
    # A sphere query passed to a wp.func annotated with the parent MeshQuery type
    # keeps sphere semantics, and an AABB query passed to a wp.func annotated with
    # the concrete MeshQueryAABB type (pre-rename convention) still resolves.
    rng = np.random.default_rng(123)
    m, tris, _lowers, _uppers = _random_triangle_mesh(device, rng)
    num_tris = tris.shape[0]
    expected = wp.zeros(num_tris, dtype=int, device=device)
    actual = wp.zeros(num_tris, dtype=int, device=device)
    center = wp.vec3(4.0, 4.0, 4.0)

    wp.launch(mesh_query_sphere_hits, dim=1, inputs=[m.id, center, 1.5, expected], device=device)
    wp.launch(mesh_query_sphere_via_erased_func, dim=1, inputs=[m.id, center, 1.5, actual], device=device)
    reference = expected.numpy()
    test.assertGreater(reference.sum(), 0)
    np.testing.assert_array_equal(actual.numpy(), reference)

    expected.zero_()
    actual.zero_()
    lower = wp.vec3(2.0, 2.0, 2.0)
    upper = wp.vec3(6.0, 6.0, 6.0)
    wp.launch(mesh_query_aabb_hits, dim=1, inputs=[m.id, lower, upper, expected], device=device)
    wp.launch(mesh_query_aabb_via_typed_func, dim=1, inputs=[m.id, lower, upper, actual], device=device)
    reference = expected.numpy()
    test.assertGreater(reference.sum(), 0)
    np.testing.assert_array_equal(actual.numpy(), reference)


@wp.kernel
def mesh_bvh_sphere_hits(mesh_id: wp.uint64, center: wp.vec3, radius: float, hits: wp.array[int]):
    bvh = wp.mesh_get_bvh(mesh_id)
    query = wp.bvh_query_sphere(bvh, center, radius)
    bound = int(0)
    while wp.bvh_query_next(query, bound):
        hits[bound] = 1


def _random_triangle_mesh(device, rng, num_tris=2000):
    """A mesh of small, randomly placed triangles (one independent triangle per face)."""
    centers = rng.random((num_tris, 3)).astype(np.float32) * 8.0
    verts = (centers[:, None, :] + (rng.random((num_tris, 3, 3)) - 0.5).astype(np.float32) * 0.4).reshape(-1, 3)
    indices = np.arange(3 * num_tris, dtype=np.int32)
    m = wp.Mesh(
        points=wp.array(verts, dtype=wp.vec3, device=device),
        indices=wp.array(indices, dtype=int, device=device),
    )
    tris = verts.reshape(num_tris, 3, 3)
    lowers = tris.min(axis=1)  # per-triangle AABB == mesh BVH leaf bounds
    uppers = tris.max(axis=1)
    return m, tris, lowers, uppers


def _point_tri_dist2(p, A, B, C):
    """Squared distance from point ``p`` to each triangle (A,B,C), vectorized (Ericson regions)."""
    ab, ac, ap = B - A, C - A, p - A
    d1 = (ab * ap).sum(-1)
    d2 = (ac * ap).sum(-1)
    bp = p - B
    d3 = (ab * bp).sum(-1)
    d4 = (ac * bp).sum(-1)
    cp_ = p - C
    d5 = (ab * cp_).sum(-1)
    d6 = (ac * cp_).sum(-1)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    denom = va + vb + vc
    inv = np.divide(1.0, denom, out=np.zeros_like(denom), where=denom != 0)
    Q = A + (vb * inv)[:, None] * ab + (vc * inv)[:, None] * ac  # interior (face) region
    Q = np.where(((d1 <= 0) & (d2 <= 0))[:, None], A, Q)  # vertex A
    Q = np.where(((d3 >= 0) & (d4 <= d3))[:, None], B, Q)  # vertex B
    Q = np.where(((d6 >= 0) & (d5 <= d6))[:, None], C, Q)  # vertex C
    tAB = np.divide(d1, d1 - d3, out=np.zeros_like(d1), where=(d1 - d3) != 0)
    Q = np.where(((vc <= 0) & (d1 >= 0) & (d3 <= 0))[:, None], A + tAB[:, None] * ab, Q)  # edge AB
    tAC = np.divide(d2, d2 - d6, out=np.zeros_like(d2), where=(d2 - d6) != 0)
    Q = np.where(((vb <= 0) & (d2 >= 0) & (d6 <= 0))[:, None], A + tAC[:, None] * ac, Q)  # edge AC
    dBC = (d4 - d3) + (d5 - d6)
    tBC = np.divide(d4 - d3, dBC, out=np.zeros_like(d4), where=dBC != 0)
    Q = np.where(((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0))[:, None], B + tBC[:, None] * (C - B), Q)  # edge BC
    d = p - Q
    return (d * d).sum(-1)


def test_mesh_query_sphere(test, device):
    # Narrow phase: returned faces are exactly the triangles the sphere actually intersects (closest point
    # on the triangle within radius), not merely AABB overlaps. Use dilate/erode bands to stay robust to
    # float32-vs-float64 rounding right at the boundary.
    rng = np.random.default_rng(123)
    m, tris, _lowers, _uppers = _random_triangle_mesh(device, rng)
    A, B, C = tris[:, 0].astype(np.float64), tris[:, 1].astype(np.float64), tris[:, 2].astype(np.float64)
    num_tris = tris.shape[0]
    hits = wp.zeros(num_tris, dtype=int, device=device)
    eps = 1e-4
    for _ in range(20):
        center = (rng.random(3) * 8.0).astype(np.float32)
        radius = float(rng.random() * 0.8 + 0.1)
        hits.zero_()
        wp.launch(mesh_query_sphere_hits, dim=1, inputs=[m.id, wp.vec3(*center), radius, hits], device=device)
        got = hits.numpy().astype(bool)
        dist = np.sqrt(_point_tri_dist2(center.astype(np.float64), A, B, C))
        np.testing.assert_array_equal(
            got & ~(dist <= radius + eps), False, err_msg="sphere reported a triangle farther than radius"
        )
        np.testing.assert_array_equal(
            (dist <= radius - eps) & ~got, False, err_msg="sphere missed a triangle within radius"
        )


def test_mesh_get_bvh(test, device):
    # mesh_get_bvh exposes the mesh BVH; bvh_query_sphere on it returns the broad (per-triangle AABB) set.
    rng = np.random.default_rng(99)
    m, tris, lowers, uppers = _random_triangle_mesh(device, rng)
    num_tris = tris.shape[0]
    hits = wp.zeros(num_tris, dtype=int, device=device)
    for _ in range(10):
        center = (rng.random(3) * 8.0).astype(np.float32)
        radius = float(rng.random() * 0.8 + 0.1)
        hits.zero_()
        wp.launch(mesh_bvh_sphere_hits, dim=1, inputs=[m.id, wp.vec3(*center), radius, hits], device=device)
        got = hits.numpy().astype(bool)
        cp = np.clip(center, lowers, uppers)
        expected = ((center - cp) ** 2).sum(axis=1) <= radius * radius
        assert_np_equal(got, expected)


devices = get_test_devices()


class TestMeshQueryAABBMethods(unittest.TestCase):
    def test_mesh_query_aabb_codegen_adjoints_with_select(self):
        def kernel_fn(
            mesh: wp.uint64,
        ):
            v = wp.vec3(0.0, 0.0, 0.0)

            if True:
                query = wp.mesh_query_aabb(mesh, v, v)
            else:
                query = wp.mesh_query_aabb(mesh, v, v)

        wp.Kernel(func=kernel_fn)


add_function_test(TestMeshQueryAABBMethods, "test_compute_bounds", test_compute_bounds, devices=devices)
add_function_test(
    TestMeshQueryAABBMethods, "test_mesh_query_aabb_count_overlap", test_mesh_query_aabb_count_overlap, devices=devices
)
add_function_test(
    TestMeshQueryAABBMethods,
    "test_mesh_query_aabb_count_nonoverlap",
    test_mesh_query_aabb_count_nonoverlap,
    devices=devices,
)
add_function_test(
    TestMeshQueryAABBMethods,
    "test_mesh_query_aabb_count_overlap_with_checksum",
    test_mesh_query_aabb_count_overlap_with_checksum,
    devices=devices,
)
add_function_test(
    TestMeshQueryAABBMethods,
    "test_tile_mesh_query_aabb",
    test_tile_mesh_query_aabb,
    devices=devices,
)
add_function_test(
    TestMeshQueryAABBMethods,
    "test_tile_mesh_query_aabb_large",
    test_tile_mesh_query_aabb_large,
    devices=devices,
)
add_function_test(
    TestMeshQueryAABBMethods,
    "test_mesh_query_aabb_tiled",
    test_mesh_query_aabb_tiled,
    devices=devices,
)
add_function_test(TestMeshQueryAABBMethods, "test_mesh_query_sphere", test_mesh_query_sphere, devices=devices)
add_function_test(
    TestMeshQueryAABBMethods,
    "test_mesh_query_sphere_alias_next",
    test_mesh_query_sphere_alias_next,
    devices=devices,
)
add_function_test(
    TestMeshQueryAABBMethods, "test_mesh_query_sphere_for_loop", test_mesh_query_sphere_for_loop, devices=devices
)
add_function_test(
    TestMeshQueryAABBMethods, "test_mesh_query_runtime_kind", test_mesh_query_runtime_kind, devices=devices
)
add_function_test(
    TestMeshQueryAABBMethods, "test_mesh_query_erased_func_param", test_mesh_query_erased_func_param, devices=devices
)
add_function_test(TestMeshQueryAABBMethods, "test_mesh_get_bvh", test_mesh_get_bvh, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
