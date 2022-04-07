# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np

import warp as wp
from warp.tests.test_base import *

np.random.seed(532)

wp.init()

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
def count_neighbors(grid : wp.uint64,
                    radius: float,
                    points: wp.array(dtype=wp.vec3),
                    counts: wp.array(dtype=int)):

    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)

    # query point    
    p = points[i]
    count = int(0)

    # construct query around point p
    neighbors = wp.hash_grid_query(grid, p, radius)

    for index in neighbors:

        # compute distance to point
        d = wp.length(p - points[index])

        if (d <= radius):
            count += 1

    counts[i] = count

@wp.kernel
def count_neighbors_reference(
                    radius: float,
                    points: wp.array(dtype=wp.vec3),
                    counts: wp.array(dtype=int),
                    num_points: int):

    tid = wp.tid()

    i = tid%num_points
    j = tid//num_points
    
    # query point
    p = points[i]
    q = points[j]

    # compute distance to point
    d = wp.length(p - q)

    if (d <= radius):
        wp.atomic_add(counts, i, 1)


def test_hashgrid_query(test, device):
        
    grid = wp.HashGrid(dim_x, dim_y, dim_z, device)

    for i in range(num_runs):

        if (print_enabled):
            print(f"Run: {i+1}")
            print("---------")

        np.random.seed(532)
        points = np.random.rand(num_points, 3)*scale - np.array((scale, scale, scale))*0.5

        def particle_grid(dim_x, dim_y, dim_z, lower, radius, jitter):
            points = np.meshgrid(np.linspace(0, dim_x, dim_x), np.linspace(0, dim_y, dim_y), np.linspace(0, dim_z, dim_z))
            points_t = np.array((points[0], points[1], points[2])).T*radius*2.0 + np.array(lower)
            points_t = points_t + np.random.rand(*points_t.shape)*radius*jitter
            
            return points_t.reshape((-1, 3))

        points = particle_grid(16, 32, 16, (0.0, 0.3, 0.0), cell_radius*0.25, 0.1)

        points_arr = wp.array(points, dtype=wp.vec3, device=device)
        counts_arr = wp.zeros(len(points), dtype=int, device=device)
        counts_arr_ref = wp.zeros(len(points), dtype=int, device=device)

        with wp.ScopedTimer("brute", active=print_enabled):
            wp.launch(kernel=count_neighbors_reference, dim=len(points)*len(points), inputs=[query_radius, points_arr, counts_arr_ref, len(points)], device=device)
            wp.synchronize()

        with wp.ScopedTimer("grid build", active=print_enabled):
            grid.build(points_arr, cell_radius)
            wp.synchronize()

        with wp.ScopedTimer("grid query", active=print_enabled):
            wp.launch(kernel=count_neighbors, dim=len(points), inputs=[grid.id, query_radius, points_arr, counts_arr], device=device)
            wp.synchronize()

        counts = counts_arr.numpy()
        counts_ref = counts_arr_ref.numpy()

        if (print_enabled):
            print(f"Grid min: {np.min(counts)} max: {np.max(counts)} avg: {np.mean(counts)}")
            print(f"Ref min: {np.min(counts_ref)} max: {np.max(counts_ref)} avg: {np.mean(counts_ref)}")

            print(f"Passed: {np.array_equal(counts, counts_ref)}")


        test.assertTrue(np.array_equal(counts, counts_ref))
        
def register(parent):

    devices = wp.get_devices()

    class TestHashGrid(parent):
        pass

    add_function_test(TestHashGrid, "test_hashgrid_query", test_hashgrid_query, devices=devices)

    return TestHashGrid

if __name__ == '__main__':
    c = register(unittest.TestCase)
    unittest.main(verbosity=2, failfast=False)