import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warp as wp
import numpy as np

import ctypes

np.random.seed(532)

wp.config.mode = "release"
#wp.config.verify_cuda = True

wp.init()

num_points = 32768
dim_x = 128
dim_y = 128
dim_z = 128

scale = 150.0

cell_radius = 8.0
query_radius = 8.0

num_runs = 16

device = "cuda"

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
    query = wp.hash_grid_query(grid, p, radius)
    index = int(0)

    while(wp.hash_grid_query_next(query, index)):

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


grid = wp.HashGrid(dim_x, dim_y, dim_z, device)

for i in range(num_runs):

    print(f"Run: {i+1}")
    print("---------")

    np.random.seed(532)
    points = np.random.rand(num_points, 3)*scale - np.array((scale, scale, scale))*0.5

    points_arr = wp.array(points, dtype=wp.vec3, device=device)
    counts_arr = wp.zeros(len(points), dtype=int, device=device)
    counts_arr_ref = wp.zeros(len(points), dtype=int, device=device)

    with wp.ScopedTimer("brute"):
        wp.launch(kernel=count_neighbors_reference, dim=len(points)*len(points), inputs=[query_radius, points_arr, counts_arr_ref, len(points)], device=device)
        wp.synchronize()

    with wp.ScopedTimer("grid build"):
        grid.build(points_arr, cell_radius)
        wp.synchronize()

    with wp.ScopedTimer("grid query"):
        wp.launch(kernel=count_neighbors, dim=len(points), inputs=[grid.id, query_radius, points_arr, counts_arr], device=device)
        wp.synchronize()
        #wp.launch(kernel=count_neighbors, dim=1, inputs=[grid, query_radius, points_arr, counts_arr], device=device)


    counts = counts_arr.numpy()
    counts_ref = counts_arr_ref.numpy()
    
    print(f"Grid min: {np.min(counts)} max: {np.max(counts)} avg: {np.mean(counts)}")
    print(f"Ref min: {np.min(counts_ref)} max: {np.max(counts_ref)} avg: {np.mean(counts_ref)}")

    print(f"Passed: {np.array_equal(counts, counts_ref)}")


