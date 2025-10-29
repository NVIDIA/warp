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

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


@wp.kernel
def bvh_query_aabb(bvh_id: wp.uint64, lower: wp.vec3, upper: wp.vec3, bounds_intersected: wp.array(dtype=int)):
    query = wp.bvh_query_aabb(bvh_id, lower, upper)
    bounds_nr = int(0)

    while wp.bvh_query_next(query, bounds_nr):
        bounds_intersected[bounds_nr] = 1


@wp.kernel
def bvh_query_ray(bvh_id: wp.uint64, start: wp.vec3, dir: wp.vec3, bounds_intersected: wp.array(dtype=int)):
    query = wp.bvh_query_ray(bvh_id, start, dir)
    bounds_nr = int(0)

    while wp.bvh_query_next(query, bounds_nr):
        bounds_intersected[bounds_nr] = 1


def aabb_overlap(a_lower, a_upper, b_lower, b_upper):
    if (
        a_lower[0] > b_upper[0]
        or a_lower[1] > b_upper[1]
        or a_lower[2] > b_upper[2]
        or a_upper[0] < b_lower[0]
        or a_upper[1] < b_lower[1]
        or a_upper[2] < b_lower[2]
    ):
        return 0
    else:
        return 1


def intersect_ray_aabb(start, rcp_dir, lower, upper):
    l1 = (lower[0] - start[0]) * rcp_dir[0]
    l2 = (upper[0] - start[0]) * rcp_dir[0]
    lmin = min(l1, l2)
    lmax = max(l1, l2)

    l1 = (lower[1] - start[1]) * rcp_dir[1]
    l2 = (upper[1] - start[1]) * rcp_dir[1]
    lmin = max(min(l1, l2), lmin)
    lmax = min(max(l1, l2), lmax)

    l1 = (lower[2] - start[2]) * rcp_dir[2]
    l2 = (upper[2] - start[2]) * rcp_dir[2]
    lmin = max(min(l1, l2), lmin)
    lmax = min(max(l1, l2), lmax)

    if lmax >= 0.0 and lmax >= lmin:
        return 1
    else:
        return 0


def test_bvh(test, type, device, leaf_size):
    rng = np.random.default_rng(123)

    num_bounds = 100
    lowers = rng.random(size=(num_bounds, 3)) * 5.0
    uppers = lowers + rng.random(size=(num_bounds, 3)) * 5.0

    device_lowers = wp.array(lowers, dtype=wp.vec3, device=device)
    device_uppers = wp.array(uppers, dtype=wp.vec3, device=device)

    bvh = wp.Bvh(device_lowers, device_uppers, leaf_size=leaf_size)

    bounds_intersected = wp.zeros(shape=(num_bounds), dtype=int, device=device)

    query_lower = wp.vec3(2.0, 2.0, 2.0)
    query_upper = wp.vec3(8.0, 8.0, 8.0)

    query_start = wp.vec3(0.0, 0.0, 0.0)
    query_dir = wp.normalize(wp.vec3(1.0, 1.0, 1.0))

    for test_case in range(3):
        if type == "AABB":
            wp.launch(
                bvh_query_aabb,
                dim=1,
                inputs=[bvh.id, query_lower, query_upper, bounds_intersected],
                device=device,
            )
        else:
            wp.launch(bvh_query_ray, dim=1, inputs=[bvh.id, query_start, query_dir, bounds_intersected], device=device)

        device_intersected = bounds_intersected.numpy()

        for i in range(num_bounds):
            lower = lowers[i]
            upper = uppers[i]
            if type == "AABB":
                host_intersected = aabb_overlap(lower, upper, query_lower, query_upper)
            else:
                host_intersected = intersect_ray_aabb(query_start, 1.0 / query_dir, lower, upper)

            test.assertEqual(host_intersected, device_intersected[i])

        if test_case == 0 or test_case == 1:
            lowers = rng.random(size=(num_bounds, 3)) * 5.0
            uppers = lowers + rng.random(size=(num_bounds, 3)) * 5.0
            wp.copy(device_lowers, wp.array(lowers, dtype=wp.vec3, device=device))
            wp.copy(device_uppers, wp.array(uppers, dtype=wp.vec3, device=device))
            bounds_intersected.zero_()

            if test_case == 0:
                bvh.refit()
            else:
                bvh.rebuild()


def test_bvh_query_aabb(test, device):
    for leaf_size in [1, 2, 4]:
        test_bvh(test, "AABB", device, leaf_size)


def test_bvh_query_ray(test, device):
    for leaf_size in [1, 2, 4]:
        test_bvh(test, "ray", device, leaf_size)


def test_bvh_ray_query_inside_and_outside_bounds(test, device):
    """Regression test for issue #288: BVH ray queries should detect intersections
    regardless of whether the ray origin is inside or outside the bounding volumes.

    Previously, rays starting outside the bounds would fail to detect intersections.
    """
    # Create a single AABB spanning x=[0.5, 1.0], extending across y and z axes
    lowers = ((0.5, -1.0, -1.0),)
    uppers = ((1.0, 1.0, 1.0),)

    device_lowers = wp.array(lowers, dtype=wp.vec3f, device=device)
    device_uppers = wp.array(uppers, dtype=wp.vec3f, device=device)

    bvh = wp.Bvh(device_lowers, device_uppers)

    bounds_intersected = wp.zeros(shape=1, dtype=int, device=device)

    # Test both ray origins: outside (x=0.0) and inside (x=0.75) the AABB
    for x in (0.0, 0.75):
        query_start = wp.vec3(x, 0.0, 0.0)
        query_dir = wp.vec3(1.0, 0.0, 0.0)  # Ray pointing in +x direction

        wp.launch(bvh_query_ray, dim=1, inputs=[bvh.id, query_start, query_dir, bounds_intersected], device=device)

        device_intersected = bounds_intersected.numpy()
        # Both cases should detect the single intersection
        test.assertEqual(device_intersected.sum(), 1)


def get_random_aabbs(n, center, relative_shift, relative_size, rng):
    centers = rng.uniform(-0.5, 0.5, size=n * 3).reshape(n, 3) * relative_shift + center
    diffs = 0.5 * rng.random(n * 3).reshape(n, 3) * relative_size

    lowers = centers - diffs
    uppers = centers + diffs

    return lowers, uppers


@wp.kernel
def compute_num_contact_with_checksums(
    lowers: wp.array(dtype=wp.vec3),
    uppers: wp.array(dtype=wp.vec3),
    bvh_id: wp.uint64,
    counts: wp.array(dtype=int),
    check_sums: wp.array(dtype=int),
):
    tid = wp.tid()

    upper = uppers[tid]
    lower = lowers[tid]

    query = wp.bvh_query_aabb(bvh_id, lower, upper)
    count = int(0)

    check_sum = int(0)
    index = int(0)
    while wp.bvh_query_next(query, index):
        check_sum = check_sum ^ index
        count += 1

    counts[tid] = count
    check_sums[tid] = check_sum


def test_capture_bvh_rebuild(test, device):
    with wp.ScopedDevice(device):
        rng = np.random.default_rng(123)

        num_item_bounds = 100000
        item_bound_size = 0.01

        relative_shift = 2

        num_test_bounds = 10000
        test_bound_relative_size = 0.05

        center = np.array([0.0, 0.0, 0.0])

        item_lowers_np, item_uppers_np = get_random_aabbs(num_item_bounds, center, relative_shift, item_bound_size, rng)
        item_lowers = wp.array(item_lowers_np, dtype=wp.vec3)
        item_uppers = wp.array(item_uppers_np, dtype=wp.vec3)
        bvh_1 = wp.Bvh(item_lowers, item_uppers)
        item_lowers_2 = wp.zeros_like(item_lowers)
        item_uppers_2 = wp.zeros_like(item_lowers)

        test_lowers_np, test_uppers_np = get_random_aabbs(
            num_test_bounds, center, relative_shift, test_bound_relative_size, rng
        )
        test_lowers = wp.array(test_lowers_np, dtype=wp.vec3)
        test_uppers = wp.array(test_uppers_np, dtype=wp.vec3)

        item_lowers_2_np, item_uppers_2_np = get_random_aabbs(
            num_item_bounds,
            center,
            relative_shift,
            item_bound_size,
            rng,
        )
        item_lowers_2.assign(item_lowers_2_np)
        item_uppers_2.assign(item_uppers_2_np)

        counts_1 = wp.empty(n=num_test_bounds, dtype=int)
        checksums_1 = wp.empty(n=num_test_bounds, dtype=int)
        counts_2 = wp.empty(n=num_test_bounds, dtype=int)
        checksums_2 = wp.empty(n=num_test_bounds, dtype=int)

        wp.load_module(device=device)
        with wp.ScopedCapture(force_module_load=False) as capture:
            wp.copy(item_lowers, item_lowers_2)
            wp.copy(item_uppers, item_uppers_2)
            bvh_1.rebuild()
            wp.launch(
                compute_num_contact_with_checksums,
                dim=num_test_bounds,
                inputs=[test_lowers, test_uppers, bvh_1.id],
                outputs=[counts_1, checksums_1],
            )

        cuda_graph = capture.graph

        for _ in range(10):
            item_lowers_2_np, item_uppers_2_np = get_random_aabbs(
                num_item_bounds,
                center,
                relative_shift,
                item_bound_size,
                rng,
            )
            item_lowers_2.assign(item_lowers_2_np)
            item_uppers_2.assign(item_uppers_2_np)

            wp.capture_launch(cuda_graph)

            bvh_2 = wp.Bvh(item_lowers_2, item_uppers_2)
            wp.launch(
                compute_num_contact_with_checksums,
                dim=num_test_bounds,
                inputs=[test_lowers, test_uppers, bvh_2.id],
                outputs=[counts_2, checksums_2],
                device=device,
            )

            assert_array_equal(counts_1, counts_2)
            assert_array_equal(checksums_1, checksums_2)


@wp.kernel
def tile_bvh_query_aabb_kernel(
    bvh_id: wp.uint64,
    lower: wp.vec3,
    upper: wp.vec3,
    bounds_intersected: wp.array(dtype=int),
):
    query = wp.tile_bvh_query_aabb(bvh_id, lower, upper)

    # Query returns a tile of indices, one per thread
    result_tile = wp.tile_bvh_query_next(query)

    # Continue querying while we have results
    while wp.tile_max(result_tile)[0] >= 0:
        # Each thread processes its result from the tile
        result_idx = wp.untile(result_tile)

        # Mark bounds as intersected using atomic add (skip -1 which means no result)
        # This ensures we can verify that each bound is only reported once
        if result_idx >= 0:
            wp.atomic_add(bounds_intersected, result_idx, 1)

        result_tile = wp.tile_bvh_query_next(query)


@wp.kernel
def tile_bvh_query_ray_kernel(
    bvh_id: wp.uint64,
    start: wp.vec3,
    dir: wp.vec3,
    bounds_intersected: wp.array(dtype=int),
):
    query = wp.tile_bvh_query_ray(bvh_id, start, dir)

    # Query returns a tile of indices, one per thread
    result_tile = wp.tile_bvh_query_next(query)

    # Continue querying while we have results
    while wp.tile_max(result_tile)[0] >= 0:
        # Each thread processes its result from the tile
        result_idx = wp.untile(result_tile)

        # Mark bounds as intersected using atomic add (skip -1 which means no result)
        # This ensures we can verify that each bound is only reported once
        if result_idx >= 0:
            wp.atomic_add(bounds_intersected, result_idx, 1)

        result_tile = wp.tile_bvh_query_next(query)


def test_tile_bvh_query(test, device):
    """Test tile-based BVH query and compare with single-threaded version."""
    rng = np.random.default_rng(456)

    num_bounds = 100
    lowers = rng.random(size=(num_bounds, 3)) * 5.0
    uppers = lowers + rng.random(size=(num_bounds, 3)) * 5.0

    device_lowers = wp.array(lowers, dtype=wp.vec3, device=device)
    device_uppers = wp.array(uppers, dtype=wp.vec3, device=device)

    bvh = wp.Bvh(device_lowers, device_uppers)

    query_lower = wp.vec3(2.0, 2.0, 2.0)
    query_upper = wp.vec3(8.0, 8.0, 8.0)

    # Test with single-threaded version (ground truth)
    bounds_intersected_single = wp.zeros(shape=(num_bounds), dtype=int, device=device)
    wp.launch(
        kernel=bvh_query_aabb,
        dim=1,
        inputs=[bvh.id, query_lower, query_upper, bounds_intersected_single],
        device=device,
    )

    # Test with tile-based version
    block_dim = 64
    bounds_intersected_tile = wp.zeros(shape=(num_bounds), dtype=int, device=device)
    wp.launch_tiled(
        kernel=tile_bvh_query_aabb_kernel,
        dim=1,
        inputs=[bvh.id, query_lower, query_upper, bounds_intersected_tile],
        device=device,
        block_dim=block_dim,
    )

    # Compare results
    single_result = bounds_intersected_single.numpy()
    tile_result = bounds_intersected_tile.numpy()

    for i in range(num_bounds):
        test.assertEqual(
            single_result[i],
            tile_result[i],
            f"Mismatch at bound {i}: single={single_result[i]}, tile={tile_result[i]}",
        )

    # Verify against CPU ground truth
    for i in range(num_bounds):
        lower = lowers[i]
        upper = uppers[i]
        if (
            lower[0] < query_upper[0]
            and upper[0] > query_lower[0]
            and lower[1] < query_upper[1]
            and upper[1] > query_lower[1]
            and lower[2] < query_upper[2]
            and upper[2] > query_lower[2]
        ):
            test.assertEqual(tile_result[i], 1, f"Expected bound {i} to be intersected")
        else:
            test.assertEqual(tile_result[i], 0, f"Expected bound {i} to not be intersected")

    # Verify that no bound was reported more than once
    # (all values should be 0 or 1, never > 1)
    for i in range(num_bounds):
        test.assertIn(
            tile_result[i],
            [0, 1],
            f"Bound {i} was reported {tile_result[i]} times, expected 0 or 1. "
            "This indicates the parallel BVH query reported the same bound multiple times.",
        )


def test_tile_bvh_query_ray(test, device):
    """Test tile-based BVH ray query and compare with single-threaded version."""
    rng = np.random.default_rng(789)

    num_bounds = 100
    lowers = rng.random(size=(num_bounds, 3)) * 5.0
    uppers = lowers + rng.random(size=(num_bounds, 3)) * 5.0

    device_lowers = wp.array(lowers, dtype=wp.vec3, device=device)
    device_uppers = wp.array(uppers, dtype=wp.vec3, device=device)

    bvh = wp.Bvh(device_lowers, device_uppers)

    query_start = wp.vec3(0.0, 0.0, 0.0)
    query_dir = wp.normalize(wp.vec3(1.0, 1.0, 1.0))

    # Test with single-threaded version (ground truth)
    bounds_intersected_single = wp.zeros(shape=(num_bounds), dtype=int, device=device)
    wp.launch(
        kernel=bvh_query_ray,
        dim=1,
        inputs=[bvh.id, query_start, query_dir, bounds_intersected_single],
        device=device,
    )

    # Test with tile-based version
    block_dim = 64
    bounds_intersected_tile = wp.zeros(shape=(num_bounds), dtype=int, device=device)
    wp.launch_tiled(
        kernel=tile_bvh_query_ray_kernel,
        dim=1,
        inputs=[bvh.id, query_start, query_dir, bounds_intersected_tile],
        device=device,
        block_dim=block_dim,
    )

    # Compare results
    single_result = bounds_intersected_single.numpy()
    tile_result = bounds_intersected_tile.numpy()

    for i in range(num_bounds):
        test.assertEqual(
            single_result[i],
            tile_result[i],
            f"Mismatch at bound {i}: single={single_result[i]}, tile={tile_result[i]}",
        )

    # Verify against CPU ground truth
    for i in range(num_bounds):
        lower = lowers[i]
        upper = uppers[i]
        host_intersected = intersect_ray_aabb(query_start, 1.0 / query_dir, lower, upper)
        test.assertEqual(tile_result[i], host_intersected, f"Expected bound {i} intersection to be {host_intersected}")

    # Verify that no bound was reported more than once
    # (all values should be 0 or 1, never > 1)
    for i in range(num_bounds):
        test.assertIn(
            tile_result[i],
            [0, 1],
            f"Bound {i} was reported {tile_result[i]} times, expected 0 or 1. "
            "This indicates the parallel BVH query reported the same bound multiple times.",
        )


# Tests for new bvh_query_*_tiled() API (primary naming convention)
@wp.kernel
def bvh_query_aabb_tiled_kernel(
    bvh_id: wp.uint64,
    lower: wp.vec3,
    upper: wp.vec3,
    bounds_intersected: wp.array(dtype=int),
):
    query = wp.bvh_query_aabb_tiled(bvh_id, lower, upper)

    # Query returns a tile of indices, one per thread
    result_tile = wp.bvh_query_next_tiled(query)

    # Continue querying while we have results
    while wp.tile_max(result_tile)[0] >= 0:
        # Each thread processes its result from the tile
        result_idx = wp.untile(result_tile)

        # Mark bounds as intersected using atomic add (skip -1 which means no result)
        if result_idx >= 0:
            wp.atomic_add(bounds_intersected, result_idx, 1)

        result_tile = wp.bvh_query_next_tiled(query)


@wp.kernel
def bvh_query_ray_tiled_kernel(
    bvh_id: wp.uint64,
    start: wp.vec3,
    dir: wp.vec3,
    bounds_intersected: wp.array(dtype=int),
):
    query = wp.bvh_query_ray_tiled(bvh_id, start, dir)

    # Query returns a tile of indices, one per thread
    result_tile = wp.bvh_query_next_tiled(query)

    # Continue querying while we have results
    while wp.tile_max(result_tile)[0] >= 0:
        # Each thread processes its result from the tile
        result_idx = wp.untile(result_tile)

        # Mark bounds as intersected using atomic add (skip -1 which means no result)
        if result_idx >= 0:
            wp.atomic_add(bounds_intersected, result_idx, 1)

        result_tile = wp.bvh_query_next_tiled(query)


def test_bvh_query_aabb_tiled(test, device):
    """Test bvh_query_aabb_tiled() API (new primary naming convention)."""
    rng = np.random.default_rng(456)

    num_bounds = 100
    lowers = rng.random(size=(num_bounds, 3)) * 5.0
    uppers = lowers + rng.random(size=(num_bounds, 3)) * 5.0

    device_lowers = wp.array(lowers, dtype=wp.vec3, device=device)
    device_uppers = wp.array(uppers, dtype=wp.vec3, device=device)

    bvh = wp.Bvh(device_lowers, device_uppers)

    query_lower = wp.vec3(2.0, 2.0, 2.0)
    query_upper = wp.vec3(8.0, 8.0, 8.0)

    # Test with single-threaded version (ground truth)
    bounds_intersected_single = wp.zeros(shape=(num_bounds), dtype=int, device=device)
    wp.launch(
        kernel=bvh_query_aabb,
        dim=1,
        inputs=[bvh.id, query_lower, query_upper, bounds_intersected_single],
        device=device,
    )

    # Test with new tiled API
    block_dim = 64
    bounds_intersected_tiled = wp.zeros(shape=(num_bounds), dtype=int, device=device)
    wp.launch_tiled(
        kernel=bvh_query_aabb_tiled_kernel,
        dim=1,
        inputs=[bvh.id, query_lower, query_upper, bounds_intersected_tiled],
        device=device,
        block_dim=block_dim,
    )

    # Compare results
    single_result = bounds_intersected_single.numpy()
    tiled_result = bounds_intersected_tiled.numpy()

    for i in range(num_bounds):
        test.assertEqual(
            single_result[i],
            tiled_result[i],
            f"Mismatch at bound {i}: single={single_result[i]}, tiled={tiled_result[i]}",
        )

    # Verify against CPU ground truth
    for i in range(num_bounds):
        lower = lowers[i]
        upper = uppers[i]
        if (
            lower[0] < query_upper[0]
            and upper[0] > query_lower[0]
            and lower[1] < query_upper[1]
            and upper[1] > query_lower[1]
            and lower[2] < query_upper[2]
            and upper[2] > query_lower[2]
        ):
            test.assertEqual(tiled_result[i], 1, f"Expected bound {i} to be intersected")
        else:
            test.assertEqual(tiled_result[i], 0, f"Expected bound {i} to not be intersected")

    # Verify that no bound was reported more than once
    for i in range(num_bounds):
        test.assertIn(
            tiled_result[i],
            [0, 1],
            f"Bound {i} was reported {tiled_result[i]} times, expected 0 or 1. "
            "This indicates the parallel BVH query reported the same bound multiple times.",
        )


def test_bvh_query_ray_tiled(test, device):
    """Test bvh_query_ray_tiled() API (new primary naming convention)."""
    rng = np.random.default_rng(789)

    num_bounds = 100
    lowers = rng.random(size=(num_bounds, 3)) * 5.0
    uppers = lowers + rng.random(size=(num_bounds, 3)) * 5.0

    device_lowers = wp.array(lowers, dtype=wp.vec3, device=device)
    device_uppers = wp.array(uppers, dtype=wp.vec3, device=device)

    bvh = wp.Bvh(device_lowers, device_uppers)

    query_start = wp.vec3(0.0, 0.0, 0.0)
    query_dir = wp.normalize(wp.vec3(1.0, 1.0, 1.0))

    # Test with single-threaded version (ground truth)
    bounds_intersected_single = wp.zeros(shape=(num_bounds), dtype=int, device=device)
    wp.launch(
        kernel=bvh_query_ray,
        dim=1,
        inputs=[bvh.id, query_start, query_dir, bounds_intersected_single],
        device=device,
    )

    # Test with new tiled API
    block_dim = 64
    bounds_intersected_tiled = wp.zeros(shape=(num_bounds), dtype=int, device=device)
    wp.launch_tiled(
        kernel=bvh_query_ray_tiled_kernel,
        dim=1,
        inputs=[bvh.id, query_start, query_dir, bounds_intersected_tiled],
        device=device,
        block_dim=block_dim,
    )

    # Compare results
    single_result = bounds_intersected_single.numpy()
    tiled_result = bounds_intersected_tiled.numpy()

    for i in range(num_bounds):
        test.assertEqual(
            single_result[i],
            tiled_result[i],
            f"Mismatch at bound {i}: single={single_result[i]}, tiled={tiled_result[i]}",
        )

    # Verify against CPU ground truth
    for i in range(num_bounds):
        lower = lowers[i]
        upper = uppers[i]
        host_intersected = intersect_ray_aabb(query_start, 1.0 / query_dir, lower, upper)
        test.assertEqual(tiled_result[i], host_intersected, f"Expected bound {i} intersection to be {host_intersected}")

    # Verify that no bound was reported more than once
    for i in range(num_bounds):
        test.assertIn(
            tiled_result[i],
            [0, 1],
            f"Bound {i} was reported {tiled_result[i]} times, expected 0 or 1. "
            "This indicates the parallel BVH query reported the same bound multiple times.",
        )


devices = get_test_devices()
cuda_devices = get_cuda_test_devices()


class TestBvh(unittest.TestCase):
    def test_bvh_codegen_adjoints_with_select(self):
        def kernel_fn(bvh: wp.uint64):
            v = wp.vec3(0.0, 0.0, 0.0)
            bounds_nr = int(0)

            if True:
                query_1 = wp.bvh_query_aabb(bvh, v, v)
                query_2 = wp.bvh_query_ray(bvh, v, v)

                wp.bvh_query_next(query_1, bounds_nr)
                wp.bvh_query_next(query_2, bounds_nr)
            else:
                query_1 = wp.bvh_query_aabb(bvh, v, v)
                query_2 = wp.bvh_query_ray(bvh, v, v)

                wp.bvh_query_next(query_1, bounds_nr)
                wp.bvh_query_next(query_2, bounds_nr)

        wp.Kernel(func=kernel_fn)

    def test_bvh_new_del(self):
        # test the scenario in which a bvh is created but not initialized before gc
        instance = wp.Bvh.__new__(wp.Bvh)
        instance.__del__()


add_function_test(TestBvh, "test_bvh_aabb", test_bvh_query_aabb, devices=devices)
add_function_test(TestBvh, "test_bvh_ray", test_bvh_query_ray, devices=devices)
add_function_test(
    TestBvh,
    "test_bvh_ray_query_inside_and_outside_bounds",
    test_bvh_ray_query_inside_and_outside_bounds,
    devices=devices,
)
add_function_test(TestBvh, "test_tile_bvh_query_aabb", test_tile_bvh_query, devices=cuda_devices)
add_function_test(TestBvh, "test_tile_bvh_query_ray", test_tile_bvh_query_ray, devices=cuda_devices)

# Tests for new bvh_query_*_tiled() API
add_function_test(TestBvh, "test_bvh_query_aabb_tiled", test_bvh_query_aabb_tiled, devices=cuda_devices)
add_function_test(TestBvh, "test_bvh_query_ray_tiled", test_bvh_query_ray_tiled, devices=cuda_devices)

add_function_test(TestBvh, "test_capture_bvh_rebuild", test_capture_bvh_rebuild, devices=cuda_devices)

if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
