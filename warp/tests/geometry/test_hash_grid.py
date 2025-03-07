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

num_points = 4096
dim_x = 128
dim_y = 128
dim_z = 128

scale = 150.0

cell_radius = 8.0
query_radius = 8.0

num_runs = 4

print_enabled = False


@wp.kernel
def count_neighbors(grid: wp.uint64, radius: float, points: wp.array(dtype=wp.vec3), counts: wp.array(dtype=int)):
    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)

    # query point
    p = points[i]
    count = int(0)

    # construct query around point p
    for index in wp.hash_grid_query(grid, p, radius):
        # compute distance to point
        d = wp.length(p - points[index])

        if d <= radius:
            count += 1

    counts[i] = count


@wp.kernel
def count_neighbors_reference(
    radius: float, points: wp.array(dtype=wp.vec3), counts: wp.array(dtype=int), num_points: int
):
    tid = wp.tid()

    i = tid % num_points
    j = tid // num_points

    # query point
    p = points[i]
    q = points[j]

    # compute distance to point
    d = wp.length(p - q)

    if d <= radius:
        wp.atomic_add(counts, i, 1)


def particle_grid(dim_x, dim_y, dim_z, lower, radius, jitter):
    rng = np.random.default_rng(123)
    points = np.meshgrid(np.linspace(0, dim_x, dim_x), np.linspace(0, dim_y, dim_y), np.linspace(0, dim_z, dim_z))
    points_t = np.array((points[0], points[1], points[2])).T * radius * 2.0 + np.array(lower)
    points_t = points_t + rng.random(size=points_t.shape) * radius * jitter

    return points_t.reshape((-1, 3))


def test_hashgrid_query(test, device):
    wp.load_module(device=device)

    grid = wp.HashGrid(dim_x, dim_y, dim_z, device)

    for i in range(num_runs):
        if print_enabled:
            print(f"Run: {i + 1}")
            print("---------")

        points = particle_grid(16, 32, 16, (0.0, 0.3, 0.0), cell_radius * 0.25, 0.1)

        points_arr = wp.array(points, dtype=wp.vec3, device=device)
        counts_arr = wp.zeros(len(points), dtype=int, device=device)
        counts_arr_ref = wp.zeros(len(points), dtype=int, device=device)

        profiler = {}

        with wp.ScopedTimer("grid operations", print=print_enabled, dict=profiler, synchronize=True):
            with wp.ScopedTimer("brute", print=print_enabled, dict=profiler, synchronize=True):
                wp.launch(
                    kernel=count_neighbors_reference,
                    dim=len(points) * len(points),
                    inputs=[query_radius, points_arr, counts_arr_ref, len(points)],
                    device=device,
                )
                wp.synchronize_device(device)

            with wp.ScopedTimer("grid build", print=print_enabled, dict=profiler, synchronize=True):
                grid.build(points_arr, cell_radius)

            with wp.ScopedTimer("grid query", print=print_enabled, dict=profiler, synchronize=True):
                wp.launch(
                    kernel=count_neighbors,
                    dim=len(points),
                    inputs=[grid.id, query_radius, points_arr, counts_arr],
                    device=device,
                )

        counts = counts_arr.numpy()
        counts_ref = counts_arr_ref.numpy()

        if print_enabled:
            print(f"Grid min: {np.min(counts)} max: {np.max(counts)} avg: {np.mean(counts)}")
            print(f"Ref min: {np.min(counts_ref)} max: {np.max(counts_ref)} avg: {np.mean(counts_ref)}")

            print(f"Passed: {np.array_equal(counts, counts_ref)}")

        assert_np_equal(counts, counts_ref)


def test_hashgrid_inputs(test, device):
    points = particle_grid(16, 32, 16, (0.0, 0.3, 0.0), cell_radius * 0.25, 0.1)
    points_ref = wp.array(points, dtype=wp.vec3, device=device)
    counts_ref = wp.zeros(len(points), dtype=int, device=device)

    grid = wp.HashGrid(dim_x, dim_y, dim_z, device)
    grid.build(points_ref, cell_radius)

    # get reference counts
    wp.launch(
        kernel=count_neighbors, dim=len(points), inputs=[grid.id, query_radius, points_ref, counts_ref], device=device
    )

    # test with strided 1d input arrays
    for stride in [2, 3]:
        with test.subTest(msg=f"stride_{stride}"):
            points_buffer = wp.zeros(len(points) * stride, dtype=wp.vec3, device=device)
            points_strided = points_buffer[::stride]
            wp.copy(points_strided, points_ref)
            counts_strided = wp.zeros(len(points), dtype=int, device=device)

            grid = wp.HashGrid(dim_x, dim_y, dim_z, device)
            grid.build(points_strided, cell_radius)

            wp.launch(
                kernel=count_neighbors,
                dim=len(points),
                inputs=[grid.id, query_radius, points_ref, counts_strided],
                device=device,
            )

            assert_array_equal(counts_strided, counts_ref)

    # test with multidimensional input arrays
    for ndim in [2, 3, 4]:
        with test.subTest(msg=f"ndim_{ndim}"):
            shape = (len(points) // (2 ** (ndim - 1)), *((ndim - 1) * (2,)))
            points_ndim = wp.zeros(shape, dtype=wp.vec3, device=device)
            wp.copy(points_ndim, points_ref)
            counts_ndim = wp.zeros(len(points), dtype=int, device=device)

            grid = wp.HashGrid(dim_x, dim_y, dim_z, device)
            grid.build(points_ndim, cell_radius)

            wp.launch(
                kernel=count_neighbors,
                dim=len(points),
                inputs=[grid.id, query_radius, points_ref, counts_ndim],
                device=device,
            )

            assert_array_equal(counts_ndim, counts_ref)


devices = get_test_devices()


class TestHashGrid(unittest.TestCase):
    def test_hashgrid_codegen_adjoints_with_select(self):
        def kernel_fn(grid: wp.uint64):
            v = wp.vec3(0.0, 0.0, 0.0)

            if True:
                query = wp.hash_grid_query(grid, v, 0.0)
            else:
                query = wp.hash_grid_query(grid, v, 0.0)

        wp.Kernel(func=kernel_fn)

    def test_hashgrid_new_del(self):
        # test the scenario in which a hashgrid is created but not initialized before gc
        instance = wp.HashGrid.__new__(wp.HashGrid)
        instance.__del__()


add_function_test(TestHashGrid, "test_hashgrid_query", test_hashgrid_query, devices=devices)
add_function_test(TestHashGrid, "test_hashgrid_inputs", test_hashgrid_inputs, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=False)
