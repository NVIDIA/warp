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


@wp.kernel
def bvh_query_aabb_group(
    bvh_id: wp.uint64,
    lower: wp.vec3,
    upper: wp.vec3,
    group_ids: wp.array(dtype=int),
    bounds_intersected: wp.array(dtype=int),
):
    tid = wp.tid()
    root = wp.bvh_get_group_root(bvh_id, group_ids[tid])
    query = wp.bvh_query_aabb(bvh_id, lower, upper, root)
    bounds_nr = int(0)

    while wp.bvh_query_next(query, bounds_nr):
        bounds_intersected[bounds_nr] = 1


@wp.kernel
def bvh_query_ray_group(
    bvh_id: wp.uint64,
    start: wp.vec3,
    dir: wp.vec3,
    group_ids: wp.array(dtype=int),
    bounds_intersected: wp.array(dtype=int),
):
    tid = wp.tid()
    root = wp.bvh_get_group_root(bvh_id, group_ids[tid])
    query = wp.bvh_query_ray(bvh_id, start, dir, root)
    bounds_nr = int(0)

    while wp.bvh_query_next(query, bounds_nr):
        bounds_intersected[bounds_nr] = 1


@wp.kernel
def get_group_root(bvh_id: wp.uint64, roots: wp.array(dtype=int)):
    tid = wp.tid()
    roots[tid] = wp.bvh_get_group_root(bvh_id, tid)


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


def test_bvh(test, type, device):
    rng = np.random.default_rng(123)

    if device.is_cpu:
        constructors = ["sah", "median"]
    else:
        constructors = ["sah", "median", "lbvh"]

    leaf_sizes = [1, 4]

    for leaf_size, constructor in itertools.product(leaf_sizes, constructors):
        num_bounds = 100
        lowers = rng.random(size=(num_bounds, 3)) * 5.0
        uppers = lowers + rng.random(size=(num_bounds, 3)) * 5.0

        device_lowers = wp.array(lowers, dtype=wp.vec3, device=device)
        device_uppers = wp.array(uppers, dtype=wp.vec3, device=device)

        # create simple group assignment to exercise grouped BVH
        num_groups = 5
        groups_np = rng.integers(0, num_groups - 1, size=num_bounds).astype(np.int32)
        device_groups = wp.array(groups_np, dtype=int, device=device)

        bvh = wp.Bvh(device_lowers, device_uppers, groups=device_groups, constructor=constructor, leaf_size=leaf_size)

        bounds_intersected = wp.zeros(shape=(num_bounds), dtype=int, device=device)
        bounds_intersected_group = wp.zeros(shape=(num_bounds), dtype=int, device=device)

        query_lower = wp.vec3(2.0, 2.0, 2.0)
        query_upper = wp.vec3(8.0, 8.0, 8.0)

        query_start = wp.vec3(0.0, 0.0, 0.0)
        query_dir = wp.normalize(wp.vec3(1.0, 1.0, 1.0))

        for test_case in range(3):
            if type == "AABB":
                wp.launch(
                    kernel=bvh_query_aabb,
                    dim=1,
                    inputs=[bvh.id, query_lower, query_upper, bounds_intersected],
                    device=device,
                )
            else:
                wp.launch(
                    kernel=bvh_query_ray,
                    dim=1,
                    inputs=[bvh.id, query_start, query_dir, bounds_intersected],
                    device=device,
                )

            device_intersected = bounds_intersected.numpy()
            for i in range(num_bounds):
                lower = lowers[i]
                upper = uppers[i]
                if type == "AABB":
                    host_intersected = aabb_overlap(lower, upper, query_lower, query_upper)
                else:
                    host_intersected = intersect_ray_aabb(query_start, 1.0 / query_dir, lower, upper)

                test.assertEqual(host_intersected, device_intersected[i])

            # verify grouped queries restrict to the group's subtree
            unique_groups = np.unique(groups_np)
            if type == "AABB":
                wp.launch(
                    kernel=bvh_query_aabb_group,
                    dim=len(unique_groups),
                    inputs=[
                        bvh.id,
                        query_lower,
                        query_upper,
                        wp.array(unique_groups, dtype=int, device=device),
                        bounds_intersected_group,
                    ],
                    device=device,
                )
            else:
                wp.launch(
                    kernel=bvh_query_ray_group,
                    dim=len(unique_groups),
                    inputs=[
                        bvh.id,
                        query_start,
                        query_dir,
                        wp.array(unique_groups, dtype=int, device=device),
                        bounds_intersected_group,
                    ],
                    device=device,
                )

            device_intersected_group = bounds_intersected_group.numpy()
            for i in range(num_bounds):
                lower = lowers[i]
                upper = uppers[i]
                if type == "AABB":
                    host_intersected = aabb_overlap(lower, upper, query_lower, query_upper)
                else:
                    host_intersected = intersect_ray_aabb(query_start, 1.0 / query_dir, lower, upper)
                test.assertEqual(host_intersected, device_intersected_group[i])

            # verify out of range group id returns -1 root and behaves like full-tree query
            out_of_range_gid = int(unique_groups.max() + 100)
            bounds_intersected_group.zero_()
            if type == "AABB":
                wp.launch(
                    kernel=bvh_query_aabb_group,
                    dim=1,
                    inputs=[
                        bvh.id,
                        query_lower,
                        query_upper,
                        wp.array([out_of_range_gid], dtype=int, device=device),
                        bounds_intersected_group,
                    ],
                    device=device,
                )
            else:
                wp.launch(
                    kernel=bvh_query_ray_group,
                    dim=1,
                    inputs=[
                        bvh.id,
                        query_start,
                        query_dir,
                        wp.array([out_of_range_gid], dtype=int, device=device),
                        bounds_intersected_group,
                    ],
                    device=device,
                )
            device_intersected_missing = bounds_intersected_group.numpy()
            test.assertTrue(np.array_equal(device_intersected_missing, device_intersected))

            if test_case == 0 or test_case == 1:
                lowers = rng.random(size=(num_bounds, 3)) * 5.0
                uppers = lowers + rng.random(size=(num_bounds, 3)) * 5.0
                wp.copy(device_lowers, wp.array(lowers, dtype=wp.vec3, device=device))
                wp.copy(device_uppers, wp.array(uppers, dtype=wp.vec3, device=device))
                bounds_intersected.zero_()
                bounds_intersected_group.zero_()

                if test_case == 0:
                    bvh.refit()
                else:
                    if device.is_cpu:
                        bvh.rebuild(constructor)
                    else:
                        bvh.rebuild()


def test_bvh_query_aabb(test, device):
    test_bvh(test, "AABB", device)


def test_bvh_query_ray(test, device):
    test_bvh(test, "ray", device)


def test_heterogenous_with_sparse_groups(test, device):
    rng = np.random.default_rng(123)

    if device.is_cpu:
        constructors = ["sah", "median"]
    else:
        constructors = ["sah", "median", "lbvh"]

    num_bounds = 100
    num_groups = 10
    lowers = rng.random(size=(num_bounds, 3)) * 5.0
    uppers = lowers + rng.random(size=(num_bounds, 3)) * 5.0

    device_lowers = wp.array(lowers, dtype=wp.vec3, device=device)
    device_uppers = wp.array(uppers, dtype=wp.vec3, device=device)

    groups_np = np.repeat(np.arange(num_groups), num_bounds // num_groups).astype(np.int32)
    # Choose one group to have exactly one primitive
    index = rng.integers(0, num_bounds - 1, size=1).item()
    groups_np[index] = 10
    device_groups = wp.array(groups_np, dtype=int, device=device)

    for constructor in constructors:
        bvh = wp.Bvh(device_lowers, device_uppers, groups=device_groups, constructor=constructor)

        # Test that all group roots are positive
        roots = wp.zeros(shape=num_groups + 1, dtype=int, device=device)
        wp.launch(kernel=get_group_root, dim=num_groups + 1, inputs=[bvh.id, roots], device=device)
        roots_host = roots.numpy()
        test.assertTrue(np.all(roots_host >= 0))


def test_gh_288(test, device):
    num_bounds = 1
    lowers = ((0.5, -1.0, -1.0),) * num_bounds
    uppers = ((1.0, 1.0, 1.0),) * num_bounds

    device_lowers = wp.array(lowers, dtype=wp.vec3f, device=device)
    device_uppers = wp.array(uppers, dtype=wp.vec3f, device=device)

    # single bound with a single group
    groups = wp.array(np.array([0], dtype=np.int32), dtype=int, device=device)

    bvh = wp.Bvh(device_lowers, device_uppers, groups=groups)

    bounds_intersected = wp.zeros(shape=num_bounds, dtype=int, device=device)

    for x in (0.0, 0.75):
        query_start = wp.vec3(x, 0.0, 0.0)
        query_dir = wp.vec3(1.0, 0.0, 0.0)

        wp.launch(
            kernel=bvh_query_ray, dim=1, inputs=[bvh.id, query_start, query_dir, bounds_intersected], device=device
        )

        device_intersected = bounds_intersected.numpy()
        test.assertEqual(device_intersected.sum(), num_bounds)


def get_random_aabbs(
    n,
    center,
    relative_shift,
    relative_size,
    rng,
):
    centers = rng.uniform(-0.5, 0.5, size=n * 3).reshape(n, 3) * relative_shift + center
    diffs = 0.5 * rng.random(n * 3).reshape(n, 3) * relative_size

    lowers = centers - diffs
    uppers = centers + diffs

    return lowers, uppers


@wp.func
def min_vec3(a: wp.vec3, b: wp.vec3):
    return wp.vec3(wp.min(a[0], b[0]), wp.min(a[1], b[1]), wp.min(a[2], b[2]))


@wp.func
def max_vec3(a: wp.vec3, b: wp.vec3):
    return wp.vec3(wp.max(a[0], b[0]), wp.max(a[1], b[1]), wp.max(a[2], b[2]))


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
    bvh_lowers: wp.array(dtype=wp.vec3),
    bvh_uppers: wp.array(dtype=wp.vec3),
    bvh_groups: wp.array(dtype=int),
    test_lowers: wp.array(dtype=wp.vec3),
    test_uppers: wp.array(dtype=wp.vec3),
    test_groups: wp.array(dtype=int),
    counts: wp.array(dtype=int),
    check_sums: wp.array(dtype=int),
):
    tid = wp.tid()

    test_upper = test_uppers[tid]
    test_lower = test_lowers[tid]
    test_group = test_groups[tid]

    check_sum = int(0)
    count = int(0)

    for bvh_box in range(bvh_groups.shape[0]):
        if intersect_aabb_aabb(test_lower, test_upper, bvh_lowers[bvh_box], bvh_uppers[bvh_box]):
            if test_group == bvh_groups[bvh_box]:
                check_sum = check_sum ^ bvh_box
                count = count + 1

    counts[tid] = count
    check_sums[tid] = check_sum


@wp.kernel
def compute_num_contact_with_checksums(
    lowers: wp.array(dtype=wp.vec3),
    uppers: wp.array(dtype=wp.vec3),
    groups: wp.array(dtype=int),
    bvh_id: wp.uint64,
    counts: wp.array(dtype=int),
    check_sums: wp.array(dtype=int),
):
    tid = wp.tid()

    upper = uppers[tid]
    lower = lowers[tid]

    group_root = wp.bvh_get_group_root(bvh_id, groups[tid])
    query = wp.bvh_query_aabb(bvh_id, lower, upper, group_root)
    count = int(0)

    check_sum = int(0)
    index = int(0)
    while wp.bvh_query_next(query, index):
        check_sum = check_sum ^ index
        count += 1

    counts[tid] = count
    check_sums[tid] = check_sum


def test_capture_bvh_rebuild_grouped(test, device):
    with wp.ScopedDevice(device):
        rng = np.random.default_rng(123)

        num_item_bounds = 100000
        num_groups = 100
        item_bound_size = 0.01

        relative_shift = 2

        num_test_bounds = 10000
        test_bound_relative_size = 0.05

        center = np.array([0.0, 0.0, 0.0])

        item_lowers_np, item_uppers_np = get_random_aabbs(num_item_bounds, center, relative_shift, item_bound_size, rng)
        item_lowers = wp.array(item_lowers_np, dtype=wp.vec3)
        item_uppers = wp.array(item_uppers_np, dtype=wp.vec3)
        item_groups = wp.array(np.arange(num_item_bounds, dtype=np.int32) % num_groups, dtype=int)
        bvh_1 = wp.Bvh(item_lowers, item_uppers, groups=item_groups)
        item_lowers_2 = wp.zeros_like(item_lowers)
        item_uppers_2 = wp.zeros_like(item_lowers)

        test_lowers_np, test_uppers_np = get_random_aabbs(
            num_test_bounds, center, relative_shift, test_bound_relative_size, rng
        )
        test_lowers = wp.array(test_lowers_np, dtype=wp.vec3)
        test_uppers = wp.array(test_uppers_np, dtype=wp.vec3)
        test_groups = wp.array(np.arange(num_item_bounds, dtype=np.int32) % num_groups, dtype=int)

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
        counts_brutal = wp.empty(n=num_test_bounds, dtype=int)
        checksums_brutal = wp.empty(n=num_test_bounds, dtype=int)

        wp.load_module(device=device)
        with wp.ScopedCapture(force_module_load=False) as capture:
            wp.copy(item_lowers, item_lowers_2)
            wp.copy(item_uppers, item_uppers_2)
            bvh_1.rebuild()
            wp.launch(
                kernel=compute_num_contact_with_checksums,
                dim=num_test_bounds,
                inputs=[test_lowers, test_uppers, test_groups, bvh_1.id],
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

            bvh_2 = wp.Bvh(item_lowers_2, item_uppers_2, groups=item_groups)
            wp.launch(
                kernel=compute_num_contact_with_checksums,
                dim=num_test_bounds,
                inputs=[test_lowers, test_uppers, test_groups, bvh_2.id],
                outputs=[counts_2, checksums_2],
                device=device,
            )

            assert_array_equal(counts_1, counts_2)
            assert_array_equal(checksums_1, checksums_2)

            wp.launch(
                kernel=compute_num_contact_with_checksums_brutal,
                dim=num_test_bounds,
                inputs=[item_lowers_2, item_uppers_2, item_groups, test_lowers, test_uppers, test_groups],
                outputs=[counts_brutal, checksums_brutal],
                device=device,
            )

            assert_array_equal(counts_1, counts_brutal)
            assert_array_equal(checksums_1, checksums_brutal)


devices = get_test_devices()
cuda_devices = get_cuda_test_devices()


class TestGroupedBvh(unittest.TestCase):
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


add_function_test(TestGroupedBvh, "test_grouped_bvh_aabb", test_bvh_query_aabb, devices=devices)
add_function_test(TestGroupedBvh, "test_grouped_bvh_ray", test_bvh_query_ray, devices=devices)
add_function_test(TestGroupedBvh, "test_grouped_gh_288", test_gh_288, devices=devices)
add_function_test(
    TestGroupedBvh, "test_heterogenous_with_sparse_groups", test_heterogenous_with_sparse_groups, devices=devices
)

add_function_test(
    TestGroupedBvh, "test_grouped_capture_bvh_rebuild", test_capture_bvh_rebuild_grouped, devices=cuda_devices
)

if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
