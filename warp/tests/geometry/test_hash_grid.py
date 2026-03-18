# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from typing import Any

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

dim_x = 128
dim_y = 128
dim_z = 128

cell_radius = 8.0
query_radius = 8.0


# Generic kernel supporting float16, float32, and float64 precision
@wp.kernel
def count_neighbors(grid: wp.uint64, radius: Any, points: wp.array(dtype=Any), counts: wp.array(dtype=int)):
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


# Forward-declare concrete overloads to avoid extra module recompilations
# and to verify JIT compilation with explicit types (tests simple_type_codes)
count_neighbors_f16 = wp.overload(
    count_neighbors, [wp.uint64, wp.float16, wp.array(dtype=wp.vec3h), wp.array(dtype=int)]
)
count_neighbors_f32 = wp.overload(
    count_neighbors, [wp.uint64, wp.float32, wp.array(dtype=wp.vec3f), wp.array(dtype=int)]
)
count_neighbors_f64 = wp.overload(
    count_neighbors, [wp.uint64, wp.float64, wp.array(dtype=wp.vec3d), wp.array(dtype=int)]
)


@wp.func
def periodic_dist_1d(a: float, b: float, period: float) -> float:
    """Minimum image signed displacement in 1D."""
    d = a - b
    half = period * 0.5
    if d > half:
        d -= period
    elif d < -half:
        d += period
    return d


@wp.kernel
def count_neighbors_periodic(
    grid: wp.uint64,
    radius: float,
    period: float,
    points: wp.array(dtype=wp.vec3),
    counts: wp.array(dtype=int),
):
    tid = wp.tid()
    i = wp.hash_grid_point_id(grid, tid)
    p = points[i]
    count = int(0)

    for index in wp.hash_grid_query(grid, p, radius):
        q = points[index]
        dx = periodic_dist_1d(p[0], q[0], period)
        dy = periodic_dist_1d(p[1], q[1], period)
        dz = periodic_dist_1d(p[2], q[2], period)
        d = wp.sqrt(dx * dx + dy * dy + dz * dz)
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
    grid = wp.HashGrid(dim_x, dim_y, dim_z, device)

    points = particle_grid(16, 32, 16, (0.0, 0.3, 0.0), cell_radius * 0.25, 0.1)

    points_arr = wp.array(points, dtype=wp.vec3, device=device)
    counts_arr = wp.zeros(len(points), dtype=int, device=device)
    counts_arr_ref = wp.zeros(len(points), dtype=int, device=device)

    wp.launch(
        kernel=count_neighbors_reference,
        dim=len(points) * len(points),
        inputs=[query_radius, points_arr, counts_arr_ref, len(points)],
        device=device,
    )

    grid.build(points_arr, cell_radius)

    wp.launch(
        kernel=count_neighbors,
        dim=len(points),
        inputs=[wp.uint64(grid.id), query_radius, points_arr, counts_arr],
        device=device,
    )

    assert_np_equal(counts_arr.numpy(), counts_arr_ref.numpy())


def test_hashgrid_inputs(test, device):
    points = particle_grid(16, 32, 16, (0.0, 0.3, 0.0), cell_radius * 0.25, 0.1)
    points_ref = wp.array(points, dtype=wp.vec3, device=device)
    counts_ref = wp.zeros(len(points), dtype=int, device=device)

    grid = wp.HashGrid(dim_x, dim_y, dim_z, device)
    grid.build(points_ref, cell_radius)

    # get reference counts
    wp.launch(
        kernel=count_neighbors,
        dim=len(points),
        inputs=[wp.uint64(grid.id), query_radius, points_ref, counts_ref],
        device=device,
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
                inputs=[wp.uint64(grid.id), query_radius, points_ref, counts_strided],
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
                inputs=[wp.uint64(grid.id), query_radius, points_ref, counts_ndim],
                device=device,
            )

            assert_array_equal(counts_ndim, counts_ref)


def test_hashgrid_multiple_streams(test, device):
    with wp.ScopedDevice(device):
        points = particle_grid(16, 32, 16, (0.0, 0.3, 0.0), cell_radius * 0.25, 0.1)
        points_ref = wp.array(points, dtype=wp.vec3)
        counts_ref = wp.zeros(len(points), dtype=int)

        grid_dim = 64
        grid_ref = wp.HashGrid(grid_dim, grid_dim, grid_dim)
        grid_ref.build(points_ref, cell_radius)

        # get reference counts
        wp.launch(
            kernel=count_neighbors,
            dim=len(points),
            inputs=[wp.uint64(grid_ref.id), query_radius, points_ref, counts_ref],
        )

        # create multiple streams
        num_streams = 10
        streams = [wp.Stream(device=device) for _ in range(num_streams)]
        counts_per_stream = [wp.zeros(len(points), dtype=int) for _ in range(num_streams)]

        # test whether HashGrid and radix sort work with multiple streams without race conditions
        for i in range(num_streams):
            with wp.ScopedStream(streams[i]):
                grid = wp.HashGrid(grid_dim, grid_dim, grid_dim)
                grid.build(points_ref, cell_radius)

                # get counts for this stream
                wp.launch(
                    kernel=count_neighbors,
                    dim=len(points),
                    inputs=[wp.uint64(grid.id), query_radius, points_ref, counts_per_stream[i]],
                )

        # run this loop after all streams are scheduled to ensure asynchronous behaviour above
        for i in range(num_streams):
            assert_array_equal(counts_per_stream[i], counts_ref)


def test_hashgrid_multiprecision(test, device):
    """Test hash grid with all precision types using both generic and concrete-typed kernels."""
    # 4 points in a tight cluster (max distance sqrt(2)*0.1 ≈ 0.14) + 1 isolated point
    points = np.array(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.1, 0.1, 0.0], [10.0, 10.0, 10.0]],
        dtype=np.float64,
    )
    expected_counts = np.array([4, 4, 4, 4, 1])

    precision_types = [
        (wp.float16, wp.vec3h, count_neighbors_f16),
        (wp.float32, wp.vec3, count_neighbors_f32),
        (wp.float64, wp.vec3d, count_neighbors_f64),
    ]

    for scalar_dtype, vec3_dtype, kernel in precision_types:
        with test.subTest(dtype=scalar_dtype.__name__):
            points_arr = wp.array(points, dtype=vec3_dtype, device=device)
            counts_arr = wp.zeros(len(points), dtype=int, device=device)

            grid = wp.HashGrid(128, 128, 128, device, dtype=scalar_dtype)
            grid.build(points_arr, 0.2)

            wp.launch(
                kernel=kernel,
                dim=len(points),
                inputs=[wp.uint64(grid.id), scalar_dtype(0.2), points_arr, counts_arr],
                device=device,
            )

            assert_np_equal(counts_arr.numpy(), expected_counts)


def test_hashgrid_invalid_dtype(test, device):
    """Test that HashGrid rejects invalid dtypes."""
    with test.assertRaises(TypeError):
        wp.HashGrid(32, 32, 32, device, dtype=wp.int32)


def test_hashgrid_build_invalid_radius(test, device):
    """Test that build() rejects zero and negative radii."""
    grid = wp.HashGrid(32, 32, 32, device)
    points = wp.zeros(10, dtype=wp.vec3, device=device)
    with test.assertRaises(ValueError):
        grid.build(points, 0.0)
    with test.assertRaises(ValueError):
        grid.build(points, -1.0)


def test_hashgrid_dtype_validation(test, device):
    """Test that HashGrid enforces dtype consistency for all precision types."""
    dtype_map = {
        wp.float16: wp.vec3h,
        wp.float32: wp.vec3,
        wp.float64: wp.vec3d,
    }

    for grid_dtype, expected_vec3 in dtype_map.items():
        with test.subTest(grid_dtype=grid_dtype.__name__):
            # Use default dtype for float32 to test backward compatibility
            if grid_dtype == wp.float32:
                grid = wp.HashGrid(32, 32, 32, device)
                test.assertEqual(grid.dtype, wp.float32)
            else:
                grid = wp.HashGrid(32, 32, 32, device, dtype=grid_dtype)

            # Should accept matching vec3 type
            points = wp.zeros(10, dtype=expected_vec3, device=device)
            grid.build(points, 1.0)

            # Should reject mismatched vec3 types
            for other_dtype, other_vec3 in dtype_map.items():
                if other_dtype != grid_dtype:
                    with test.assertRaises(TypeError):
                        grid.build(wp.zeros(10, dtype=other_vec3, device=device), 1.0)


def test_hashgrid_edge_cases(test, device):
    """Test hash grid with edge cases: single point, in-range, and out-of-range pairs."""
    grid_dim = 32

    # Test with single point
    with test.subTest(case="single_point"):
        grid = wp.HashGrid(grid_dim, grid_dim, grid_dim, device)
        points = wp.array([[0.5, 0.5, 0.5]], dtype=wp.vec3, device=device)
        counts = wp.zeros(1, dtype=int, device=device)

        grid.build(points, 1.0)

        wp.launch(
            kernel=count_neighbors,
            dim=1,
            inputs=[wp.uint64(grid.id), 1.0, points, counts],
            device=device,
        )

        # Single point should count itself as neighbor (distance 0 <= radius)
        test.assertEqual(counts.numpy()[0], 1)

    # Test with two points - one in range, one out of range
    with test.subTest(case="two_points"):
        grid = wp.HashGrid(grid_dim, grid_dim, grid_dim, device)
        points = wp.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=wp.vec3, device=device)
        counts = wp.zeros(2, dtype=int, device=device)

        grid.build(points, 1.0)

        wp.launch(
            kernel=count_neighbors,
            dim=2,
            inputs=[wp.uint64(grid.id), 0.6, points, counts],  # radius 0.6, distance is 0.5
            device=device,
        )

        # Both points should see each other (distance 0.5 <= 0.6) plus themselves
        counts_np = counts.numpy()
        test.assertEqual(counts_np[0], 2)
        test.assertEqual(counts_np[1], 2)

    # Test with two points out of range
    with test.subTest(case="two_points_out_of_range"):
        grid = wp.HashGrid(grid_dim, grid_dim, grid_dim, device)
        points = wp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=wp.vec3, device=device)
        counts = wp.zeros(2, dtype=int, device=device)

        grid.build(points, 1.0)

        wp.launch(
            kernel=count_neighbors,
            dim=2,
            inputs=[wp.uint64(grid.id), 0.5, points, counts],  # radius 0.5, distance is 2.0
            device=device,
        )

        # Each point should only see itself
        counts_np = counts.numpy()
        test.assertEqual(counts_np[0], 1)
        test.assertEqual(counts_np[1], 1)


def test_hashgrid_negative_wrapping(test, device):
    """Test that hash grid wrapping works correctly with negative coordinates.

    With a small grid, the truncation bug (int() instead of floor()) causes
    points near negative cell boundaries to map to the wrong physical cell.
    Virtual cell -1 and cell 0 map to different physical cells after modulo
    wrapping, so a misplaced point becomes invisible to queries that should
    find it via the wrapped cell.
    """
    grid_dim = 4
    cell_width = 1.0
    period = float(grid_dim) * cell_width
    radius = 0.5

    grid = wp.HashGrid(grid_dim, grid_dim, grid_dim, device)

    # Point A at -0.3: should be virtual cell -1 (physical 3)
    #   Bug: int(-0.3) = 0 -> physical cell 0 (WRONG)
    # Point B at 3.9: virtual cell 3 -> physical cell 3 (correct in both cases)
    # Periodic distance: 0.2 (A wraps to 3.7 in [0, 4), |3.9 - 3.7| = 0.2),
    # well within radius 0.5.
    #
    # With the bug, query from A searches only physical cell 0 and misses B
    # in physical cell 3. Query from B wraps to include physical cell 0 and
    # finds the misplaced A, producing an asymmetric result [1, 2].
    points = wp.array(
        [[-0.3, 0.0, 0.0], [3.9, 0.0, 0.0]],
        dtype=wp.vec3,
        device=device,
    )
    counts = wp.zeros(2, dtype=int, device=device)

    grid.build(points, cell_width)

    wp.launch(
        kernel=count_neighbors_periodic,
        dim=2,
        inputs=[wp.uint64(grid.id), radius, period, points, counts],
        device=device,
    )

    counts_np = counts.numpy()
    test.assertEqual(counts_np[0], 2)  # A finds self + B
    test.assertEqual(counts_np[1], 2)  # B finds self + A


devices = get_test_devices()
cuda_devices = get_cuda_test_devices()


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
add_function_test(TestHashGrid, "test_hashgrid_multiple_streams", test_hashgrid_multiple_streams, devices=cuda_devices)
add_function_test(TestHashGrid, "test_hashgrid_multiprecision", test_hashgrid_multiprecision, devices=devices)
add_function_test(TestHashGrid, "test_hashgrid_invalid_dtype", test_hashgrid_invalid_dtype, devices=devices)
add_function_test(
    TestHashGrid, "test_hashgrid_build_invalid_radius", test_hashgrid_build_invalid_radius, devices=devices
)
add_function_test(TestHashGrid, "test_hashgrid_dtype_validation", test_hashgrid_dtype_validation, devices=devices)
add_function_test(TestHashGrid, "test_hashgrid_edge_cases", test_hashgrid_edge_cases, devices=devices)
add_function_test(TestHashGrid, "test_hashgrid_negative_wrapping", test_hashgrid_negative_wrapping, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=False)
