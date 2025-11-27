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


@wp.func
def min_vec3(a: wp.vec3, b: wp.vec3):
    return wp.vec3(wp.min(a[0], b[0]), wp.min(a[1], b[1]), wp.min(a[2], b[2]))


@wp.func
def max_vec3(a: wp.vec3, b: wp.vec3):
    return wp.vec3(wp.max(a[0], b[0]), wp.max(a[1], b[1]), wp.max(a[2], b[2]))


@wp.kernel
def compute_bounds(
    indices: wp.array(dtype=int),
    positions: wp.array(dtype=wp.vec3),
    lowers: wp.array(dtype=wp.vec3),
    uppers: wp.array(dtype=wp.vec3),
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
    lowers: wp.array(dtype=wp.vec3), uppers: wp.array(dtype=wp.vec3), mesh_id: wp.uint64, counts: wp.array(dtype=int)
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
    lowers: wp.array(dtype=wp.vec3),
    uppers: wp.array(dtype=wp.vec3),
    mesh_id: wp.uint64,
    counts: wp.array(dtype=int),
    check_sums: wp.array(dtype=int),
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
    lowers: wp.array(dtype=wp.vec3),
    uppers: wp.array(dtype=wp.vec3),
    mesh_points: wp.array(dtype=wp.vec3),
    mesh_indices: wp.array(dtype=int),
    counts: wp.array(dtype=int),
    check_sums: wp.array(dtype=int),
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
    faces_intersected: wp.array(dtype=int),
):
    query = wp.tile_mesh_query_aabb(mesh_id, lower, upper)

    # Query returns a tile of indices, one per thread
    result_tile = wp.tile_mesh_query_aabb_next(query)

    # Continue querying while we have results
    while wp.tile_max(result_tile)[0] >= 0:
        # Each thread processes its result from the tile
        result_idx = wp.untile(result_tile)

        # Mark faces as intersected using atomic add (skip -1 which means no result)
        # This ensures we can verify that each face is only reported once
        if result_idx >= 0:
            wp.atomic_add(faces_intersected, result_idx, 1)

        result_tile = wp.tile_mesh_query_aabb_next(query)


@wp.kernel
def mesh_query_aabb_kernel(
    mesh_id: wp.uint64,
    lower: wp.vec3,
    upper: wp.vec3,
    faces_intersected: wp.array(dtype=int),
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
            f"Mismatch at face {i}: single={single_result[i]}, tile={tile_result[i]}",
        )

    # Both triangles should be found exactly once
    test.assertEqual(single_result[0], 1)
    test.assertEqual(single_result[1], 1)


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
    faces_intersected: wp.array(dtype=int),
):
    query = wp.mesh_query_aabb_tiled(mesh_id, lower, upper)

    # Query returns a tile of indices, one per thread
    result_tile = wp.mesh_query_aabb_next_tiled(query)

    # Continue querying while we have results
    while wp.tile_max(result_tile)[0] >= 0:
        # Each thread processes its result from the tile
        result_idx = wp.untile(result_tile)

        # Mark faces as intersected using atomic add (skip -1 which means no result)
        if result_idx >= 0:
            wp.atomic_add(faces_intersected, result_idx, 1)

        result_tile = wp.mesh_query_aabb_next_tiled(query)


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


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
