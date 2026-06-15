# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


# Volume write tests
@wp.kernel
def test_volume_store_f(volume: wp.uint64, points: wp.array[wp.vec3]):
    tid = wp.tid()

    p = points[tid]
    i = int(p[0])
    j = int(p[1])
    k = int(p[2])

    wp.volume_store_f(volume, i, j, k, float(i + 100 * j + 10000 * k))


@wp.kernel
def test_volume_readback_f(volume: wp.uint64, points: wp.array[wp.vec3], values: wp.array[wp.float32]):
    tid = wp.tid()

    p = points[tid]
    i = int(p[0])
    j = int(p[1])
    k = int(p[2])

    values[tid] = wp.volume_lookup_f(volume, i, j, k)


@wp.kernel
def test_get_list_of_tiles(
    volume: wp.uint64,
    points_is: wp.array2d[wp.int32],
    points_ws: wp.array[wp.vec3],
    tiles_is: wp.array2d[wp.int32],
    tiles_ws: wp.array2d[wp.int32],
):
    tid = wp.tid()

    tiles_is[tid, 0] = points_is[tid, 0]
    tiles_is[tid, 1] = points_is[tid, 1]
    tiles_is[tid, 2] = points_is[tid, 2]

    q = wp.volume_world_to_index(volume, points_ws[tid])
    tiles_ws[tid, 0] = int(q[0] / 8.0) * 8
    tiles_ws[tid, 1] = int(q[1] / 8.0) * 8
    tiles_ws[tid, 2] = int(q[2] / 8.0) * 8


@wp.kernel
def test_volume_tile_store_f(volume: wp.uint64, tiles: wp.array2d[wp.int32]):
    tid = wp.tid()

    ti = tiles[tid, 0]
    tj = tiles[tid, 1]
    tk = tiles[tid, 2]

    for r in range(512):
        ii = ti + (r / 64) % 8
        jj = tj + (r / 8) % 8
        kk = tk + r % 8
        wp.volume_store_f(volume, ii, jj, kk, float(100 * ii + 10 * jj + kk))


@wp.kernel
def test_volume_tile_store_ws_f(volume: wp.uint64, tiles: wp.array[wp.vec3]):
    tid = wp.tid()

    q = wp.volume_world_to_index(volume, tiles[tid])
    ti = int(wp.round(q[0]))
    tj = int(wp.round(q[1]))
    tk = int(wp.round(q[2]))

    for r in range(512):
        ii = ti + (r / 64) % 8
        jj = tj + (r / 8) % 8
        kk = tk + r % 8
        wp.volume_store_f(volume, ii, jj, kk, float(100 * ii + 10 * jj + kk))


@wp.kernel
def test_volume_tile_readback_f(volume: wp.uint64, tiles: wp.array2d[wp.int32], values: wp.array[wp.float32]):
    tid = wp.tid()

    ti = tiles[tid, 0]
    tj = tiles[tid, 1]
    tk = tiles[tid, 2]

    for r in range(512):
        ii = ti + (r / 64) % 8
        jj = tj + (r / 8) % 8
        kk = tk + r % 8
        values[tid * 512 + r] = wp.volume_lookup_f(volume, ii, jj, kk)


@wp.kernel
def test_volume_tile_store_v(volume: wp.uint64, tiles: wp.array2d[wp.int32]):
    tid = wp.tid()

    ti = tiles[tid, 0]
    tj = tiles[tid, 1]
    tk = tiles[tid, 2]

    for r in range(512):
        ii = ti + (r / 64) % 8
        jj = tj + (r / 8) % 8
        kk = tk + r % 8
        wp.volume_store_v(volume, ii, jj, kk, wp.vec3(float(ii), float(jj), float(kk)))


@wp.kernel
def test_volume_tile_readback_v(volume: wp.uint64, tiles: wp.array2d[wp.int32], values: wp.array[wp.vec3]):
    tid = wp.tid()

    ti = tiles[tid, 0]
    tj = tiles[tid, 1]
    tk = tiles[tid, 2]

    for r in range(512):
        ii = ti + (r / 64) % 8
        jj = tj + (r / 8) % 8
        kk = tk + r % 8
        values[tid * 512 + r] = wp.volume_lookup_v(volume, ii, jj, kk)


@wp.kernel
def test_volume_tile_store_i(volume: wp.uint64, tiles: wp.array2d[wp.int32]):
    tid = wp.tid()

    ti = tiles[tid, 0]
    tj = tiles[tid, 1]
    tk = tiles[tid, 2]

    for r in range(512):
        ii = ti + (r / 64) % 8
        jj = tj + (r / 8) % 8
        kk = tk + r % 8
        wp.volume_store_i(volume, ii, jj, kk, 100 * ii + 10 * jj + kk)


@wp.kernel
def test_volume_tile_readback_i(volume: wp.uint64, tiles: wp.array2d[wp.int32], values: wp.array[wp.int32]):
    tid = wp.tid()

    ti = tiles[tid, 0]
    tj = tiles[tid, 1]
    tk = tiles[tid, 2]

    for r in range(512):
        ii = ti + (r / 64) % 8
        jj = tj + (r / 8) % 8
        kk = tk + r % 8
        values[tid * 512 + r] = wp.volume_lookup_i(volume, ii, jj, kk)


@wp.kernel
def test_volume_tile_store_v4(volume: wp.uint64, tiles: wp.array2d[wp.int32]):
    tid = wp.tid()

    ti = tiles[tid, 0]
    tj = tiles[tid, 1]
    tk = tiles[tid, 2]

    for r in range(512):
        ii = ti + (r / 64) % 8
        jj = tj + (r / 8) % 8
        kk = tk + r % 8
        wp.volume_store(volume, ii, jj, kk, wp.vec4(float(ii), float(jj), float(kk), float(100 * ii + 10 * jj + kk)))


@wp.kernel
def test_volume_tile_readback_v4(volume: wp.uint64, tiles: wp.array2d[wp.int32], values: wp.array[wp.vec4]):
    tid = wp.tid()

    ti = tiles[tid, 0]
    tj = tiles[tid, 1]
    tk = tiles[tid, 2]

    for r in range(512):
        ii = ti + (r / 64) % 8
        jj = tj + (r / 8) % 8
        kk = tk + r % 8
        values[tid * 512 + r] = wp.volume_lookup(volume, ii, jj, kk, dtype=wp.vec4)


@wp.kernel
def test_volume_tile_store_u32(volume: wp.uint64, tiles: wp.array2d[wp.int32]):
    tid = wp.tid()

    ti = tiles[tid, 0]
    tj = tiles[tid, 1]
    tk = tiles[tid, 2]

    for r in range(512):
        ii = ti + (r / 64) % 8
        jj = tj + (r / 8) % 8
        kk = tk + r % 8
        wp.volume_store(volume, ii, jj, kk, wp.uint32(tid * 512 + r + 17))


@wp.kernel
def test_volume_tile_readback_u32(volume: wp.uint64, tiles: wp.array2d[wp.int32], values: wp.array[wp.uint32]):
    tid = wp.tid()

    ti = tiles[tid, 0]
    tj = tiles[tid, 1]
    tk = tiles[tid, 2]

    for r in range(512):
        ii = ti + (r / 64) % 8
        jj = tj + (r / 8) % 8
        kk = tk + r % 8
        values[tid * 512 + r] = wp.volume_lookup(volume, ii, jj, kk, dtype=wp.uint32)


@wp.kernel
def test_volume_tile_store_i64(volume: wp.uint64, tiles: wp.array2d[wp.int32]):
    tid = wp.tid()

    ti = tiles[tid, 0]
    tj = tiles[tid, 1]
    tk = tiles[tid, 2]

    for r in range(512):
        ii = ti + (r / 64) % 8
        jj = tj + (r / 8) % 8
        kk = tk + r % 8
        wp.volume_store(volume, ii, jj, kk, wp.int64(1000000000000) + wp.int64(tid * 512 + r))


@wp.kernel
def test_volume_tile_readback_i64(volume: wp.uint64, tiles: wp.array2d[wp.int32], values: wp.array[wp.int64]):
    tid = wp.tid()

    ti = tiles[tid, 0]
    tj = tiles[tid, 1]
    tk = tiles[tid, 2]

    for r in range(512):
        ii = ti + (r / 64) % 8
        jj = tj + (r / 8) % 8
        kk = tk + r % 8
        values[tid * 512 + r] = wp.volume_lookup(volume, ii, jj, kk, dtype=wp.int64)


@wp.kernel
def test_volume_tile_store_f64(volume: wp.uint64, tiles: wp.array2d[wp.int32]):
    tid = wp.tid()

    ti = tiles[tid, 0]
    tj = tiles[tid, 1]
    tk = tiles[tid, 2]

    for r in range(512):
        ii = ti + (r / 64) % 8
        jj = tj + (r / 8) % 8
        kk = tk + r % 8
        wp.volume_store(
            volume,
            ii,
            jj,
            kk,
            wp.float64(ii) * wp.float64(0.25) + wp.float64(jj) * wp.float64(0.125) + wp.float64(kk),
        )


@wp.kernel
def test_volume_tile_readback_f64(volume: wp.uint64, tiles: wp.array2d[wp.int32], values: wp.array[wp.float64]):
    tid = wp.tid()

    ti = tiles[tid, 0]
    tj = tiles[tid, 1]
    tk = tiles[tid, 2]

    for r in range(512):
        ii = ti + (r / 64) % 8
        jj = tj + (r / 8) % 8
        kk = tk + r % 8
        values[tid * 512 + r] = wp.volume_lookup(volume, ii, jj, kk, dtype=wp.float64)


@wp.kernel
def test_volume_tile_store_v3d(volume: wp.uint64, tiles: wp.array2d[wp.int32]):
    tid = wp.tid()

    ti = tiles[tid, 0]
    tj = tiles[tid, 1]
    tk = tiles[tid, 2]

    for r in range(512):
        ii = ti + (r / 64) % 8
        jj = tj + (r / 8) % 8
        kk = tk + r % 8
        wp.volume_store(volume, ii, jj, kk, wp.vec3d(wp.float64(ii), wp.float64(jj * 2), wp.float64(kk * 3)))


@wp.kernel
def test_volume_tile_readback_v3d(volume: wp.uint64, tiles: wp.array2d[wp.int32], values: wp.array[wp.vec3d]):
    tid = wp.tid()

    ti = tiles[tid, 0]
    tj = tiles[tid, 1]
    tk = tiles[tid, 2]

    for r in range(512):
        ii = ti + (r / 64) % 8
        jj = tj + (r / 8) % 8
        kk = tk + r % 8
        values[tid * 512 + r] = wp.volume_lookup(volume, ii, jj, kk, dtype=wp.vec3d)


@wp.kernel
def test_volume_readback_index(volume: wp.uint64, points: wp.array2d[wp.int32], values: wp.array[wp.int32]):
    tid = wp.tid()

    values[tid] = wp.volume_lookup_index(volume, points[tid, 0], points[tid, 1], points[tid, 2])


def _sort_rows(values):
    return values[np.lexsort(values.T[::-1])]


def _unique_rows(values):
    return np.unique(_sort_rows(values), axis=0)


def _tile_voxels(tiles):
    voxels = np.empty((tiles.shape[0] * 512, 3), dtype=np.int32)
    for t, (ti, tj, tk) in enumerate(tiles):
        for r in range(512):
            voxels[t * 512 + r] = [ti + (r // 64) % 8, tj + (r // 8) % 8, tk + r % 8]
    return voxels


def _grid_parent_counts(voxels):
    leaf_count = len(_unique_rows((voxels // 8) * 8))
    lower_count = len(_unique_rows((voxels // 128) * 128))
    upper_count = len(_unique_rows((voxels // 4096) * 4096))
    return leaf_count, lower_count, upper_count


def _lookup_indices(volume, points, device):
    points_d = wp.array(points, dtype=wp.int32, device=device)
    values = wp.empty(points.shape[0], dtype=wp.int32, device=device)
    wp.launch(test_volume_readback_index, dim=points.shape[0], inputs=[volume.id, points_d, values], device=device)
    return values.numpy()


def test_volume_allocation(test, device):
    voxel_size = 0.125
    background_value = 123.456
    translation = wp.vec3(-12.3, 4.56, -789)

    axis = np.linspace(-11, 11, 23)
    points_ref = np.array([[x, y, z] for x in axis for y in axis for z in axis])
    values_ref = np.array([x + 100 * y + 10000 * z for x in axis for y in axis for z in axis])
    num_points = len(points_ref)
    bb_max = np.array([11, 11, 11])
    volume_a = wp.Volume.allocate(
        -bb_max,
        bb_max,
        voxel_size=voxel_size,
        bg_value=background_value,
        translation=translation,
        device=device,
    )
    volume_b = wp.Volume.allocate(
        -bb_max * voxel_size + translation,
        bb_max * voxel_size + translation,
        voxel_size=voxel_size,
        bg_value=background_value,
        translation=translation,
        points_in_world_space=True,
        device=device,
    )

    assert wp.types.types_equal(volume_a.dtype, wp.float32)
    assert wp.types.types_equal(volume_b.dtype, wp.float32)

    points = wp.array(points_ref, dtype=wp.vec3, device=device)
    values_a = wp.empty(num_points, dtype=wp.float32, device=device)
    values_b = wp.empty(num_points, dtype=wp.float32, device=device)
    wp.launch(test_volume_store_f, dim=num_points, inputs=[volume_a.id, points], device=device)
    wp.launch(test_volume_store_f, dim=num_points, inputs=[volume_b.id, points], device=device)
    wp.launch(test_volume_readback_f, dim=num_points, inputs=[volume_a.id, points, values_a], device=device)
    wp.launch(test_volume_readback_f, dim=num_points, inputs=[volume_b.id, points, values_b], device=device)

    np.testing.assert_equal(values_a.numpy(), values_ref)
    np.testing.assert_equal(values_b.numpy(), values_ref)


def test_volume_allocate_by_tiles_f(test, device):
    voxel_size = 0.125
    background_value = 123.456
    translation = wp.vec3(-12.3, 4.56, -789)

    num_tiles = 1000
    rng = np.random.default_rng(101215)
    tiles = rng.integers(-512, 512, size=(num_tiles, 3), dtype=np.int32)
    points_is = tiles * 8  # points in index space
    points_ws = points_is * voxel_size + translation  # points in world space

    values_ref = np.empty(num_tiles * 512)
    for t in range(num_tiles):
        ti, tj, tk = points_is[t]
        for i in range(8):
            for j in range(8):
                for k in range(8):
                    values_ref[t * 512 + i * 64 + j * 8 + k] = float(100 * (ti + i) + 10 * (tj + j) + (tk + k))

    points_is_d = wp.array(points_is, dtype=wp.int32, device=device)
    points_ws_d = wp.array(points_ws, dtype=wp.vec3, device=device)
    volume_a = wp.Volume.allocate_by_tiles(points_is_d, voxel_size, background_value, translation, device=device)
    volume_b = wp.Volume.allocate_by_tiles(points_ws_d, voxel_size, background_value, translation, device=device)

    assert wp.types.types_equal(volume_a.dtype, wp.float32)
    assert wp.types.types_equal(volume_b.dtype, wp.float32)

    values_a = wp.empty(num_tiles * 512, dtype=wp.float32, device=device)
    values_b = wp.empty(num_tiles * 512, dtype=wp.float32, device=device)

    wp.launch(test_volume_tile_store_f, dim=num_tiles, inputs=[volume_a.id, points_is_d], device=device)
    wp.launch(test_volume_tile_store_ws_f, dim=num_tiles, inputs=[volume_b.id, points_ws_d], device=device)
    wp.launch(test_volume_tile_readback_f, dim=num_tiles, inputs=[volume_a.id, points_is_d, values_a], device=device)
    wp.launch(test_volume_tile_readback_f, dim=num_tiles, inputs=[volume_b.id, points_is_d, values_b], device=device)

    np.testing.assert_equal(values_a.numpy(), values_ref)
    np.testing.assert_equal(values_b.numpy(), values_ref)


def test_volume_allocate_by_tiles_v(test, device):
    num_tiles = 1000
    rng = np.random.default_rng(101215)
    tiles = rng.integers(-512, 512, size=(num_tiles, 3), dtype=np.int32)
    points_is = tiles * 8

    values_ref = np.empty((len(tiles) * 512, 3))
    for t in range(len(tiles)):
        ti, tj, tk = points_is[t]
        for i in range(8):
            for j in range(8):
                for k in range(8):
                    values_ref[t * 512 + i * 64 + j * 8 + k] = [ti + i, tj + j, tk + k]

    points_d = wp.array(points_is, dtype=wp.int32, device=device)
    volume = wp.Volume.allocate_by_tiles(points_d, 0.1, wp.vec3(1, 2, 3), device=device)

    assert wp.types.types_equal(volume.dtype, wp.vec3)

    values = wp.empty(len(points_d) * 512, dtype=wp.vec3, device=device)

    wp.launch(test_volume_tile_store_v, dim=len(points_d), inputs=[volume.id, points_d], device=device)
    wp.launch(test_volume_tile_readback_v, dim=len(points_d), inputs=[volume.id, points_d, values], device=device)

    values_res = values.numpy()
    np.testing.assert_equal(values_res, values_ref)


def test_volume_allocate_by_tiles_index(test, device):
    num_tiles = 10
    rng = np.random.default_rng(101215)
    tiles = rng.integers(-512, 512, size=(num_tiles, 3), dtype=np.int32)
    points_is = tiles * 8

    points_d = wp.array(points_is, dtype=wp.int32, device=device)
    volume = wp.Volume.allocate_by_tiles(points_d, 0.1, bg_value=None, device=device)

    assert volume.is_index

    vol_tiles = volume.get_tiles().numpy() / 8
    vol_tile_sorted = vol_tiles[np.lexsort(vol_tiles.T[::-1])]
    vol_tile_unique = np.unique(vol_tile_sorted, axis=0)

    tile_sorted = tiles[np.lexsort(tiles.T[::-1])]
    tile_unique = np.unique(tile_sorted, axis=0)

    np.testing.assert_equal(tile_unique, vol_tile_unique)


def test_volume_allocate_by_tiles_additional_types(test, device):
    points_np = np.array(
        [
            [0, 0, 0],
            [8, 0, 0],
            [0, 8, 0],
            [0, 0, 128],
            [0, 0, 128],
            [128, 0, 0],
        ],
        dtype=np.int32,
    )
    tile_count, lower_count, upper_count = _grid_parent_counts(points_np)
    points_d = wp.array(points_np, dtype=wp.int32, device=device)

    def check_type(bg_value, dtype, store_kernel, readback_kernel, assert_values):
        status = wp.zeros(1, dtype=wp.uint32, device=device)
        volume_exact = wp.Volume.allocate_by_tiles(points_d, 1.0, bg_value, device=device)
        volume_rebuildable = wp.Volume.allocate_by_tiles(
            points_d,
            1.0,
            bg_value,
            device=device,
            graph_rebuildable=True,
            max_tiles=tile_count,
            max_lower_nodes=lower_count,
            max_upper_nodes=upper_count,
            status=status,
        )
        wp.synchronize_device(device)

        test.assertEqual(int(status.numpy()[0]), wp.Volume.REBUILD_SUCCESS)
        assert wp.types.types_equal(volume_exact.dtype, dtype)
        assert wp.types.types_equal(volume_rebuildable.dtype, dtype)

        tiles = _sort_rows(volume_exact.get_tiles().numpy())
        tiles_d = wp.array(tiles, dtype=wp.int32, device=device)
        exact_values = wp.empty(tile_count * 512, dtype=dtype, device=device)
        rebuildable_values = wp.empty(tile_count * 512, dtype=dtype, device=device)

        wp.launch(store_kernel, dim=tile_count, inputs=[volume_exact.id, tiles_d], device=device)
        wp.launch(store_kernel, dim=tile_count, inputs=[volume_rebuildable.id, tiles_d], device=device)
        wp.launch(readback_kernel, dim=tile_count, inputs=[volume_exact.id, tiles_d, exact_values], device=device)
        wp.launch(
            readback_kernel, dim=tile_count, inputs=[volume_rebuildable.id, tiles_d, rebuildable_values], device=device
        )
        assert_values(rebuildable_values.numpy(), exact_values.numpy())

    check_type(
        wp.uint32(7),
        wp.uint32,
        test_volume_tile_store_u32,
        test_volume_tile_readback_u32,
        np.testing.assert_array_equal,
    )
    check_type(
        wp.int64(7),
        wp.int64,
        test_volume_tile_store_i64,
        test_volume_tile_readback_i64,
        np.testing.assert_array_equal,
    )
    check_type(
        wp.float64(7.0),
        wp.float64,
        test_volume_tile_store_f64,
        test_volume_tile_readback_f64,
        np.testing.assert_allclose,
    )
    check_type(
        wp.vec3d(1.0, 2.0, 3.0),
        wp.vec3d,
        test_volume_tile_store_v3d,
        test_volume_tile_readback_v3d,
        np.testing.assert_allclose,
    )


def test_volume_allocation_from_voxels(test, device):
    point_count = 387
    rng = np.random.default_rng(101215)

    # Create from world-space points
    points = wp.array(rng.uniform(5.0, 10.0, size=(point_count, 3)), dtype=float, device=device)

    volume = wp.Volume.allocate_by_voxels(
        voxel_points=points, voxel_size=0.25, translation=(0.0, 5.0, 10.0), device=device
    )

    assert volume.is_index

    test.assertNotEqual(volume.id, 0)

    test.assertAlmostEqual(volume.get_voxel_size(), (0.25, 0.25, 0.25))
    voxel_count = volume.get_voxel_count()
    test.assertGreaterEqual(point_count, voxel_count)
    test.assertGreaterEqual(voxel_count, 1)

    voxels = volume.get_voxels()

    # Check that world-to-index transform has been correctly applied
    voxel_low = np.min(voxels.numpy(), axis=0)
    voxel_up = np.max(voxels.numpy(), axis=0)
    np.testing.assert_array_less([19, -1, -21], voxel_low)
    np.testing.assert_array_less(voxel_up, [41, 21, 1])

    # Recreate the volume from ijk coords
    volume_from_ijk = wp.Volume.allocate_by_voxels(
        voxel_points=voxels, voxel_size=0.25, translation=(0.0, 5.0, 10.0), device=device
    )

    assert volume_from_ijk.is_index

    assert volume_from_ijk.get_voxel_count() == voxel_count
    ijk_voxels = volume_from_ijk.get_voxels().numpy()

    voxels = voxels.numpy()
    voxel_sorted = voxels[np.lexsort(voxels.T[::-1])]
    ijk_voxel_sorted = ijk_voxels[np.lexsort(ijk_voxels.T[::-1])]

    np.testing.assert_equal(voxel_sorted, ijk_voxel_sorted)


def test_volume_rebuildable_setup_matches_exact_tiles(test, device):
    points_np = np.array(
        [
            [-1, -2, -3],
            [0, 0, 0],
            [1, 2, 3],
            [0, 8, 7],
            [0, 0, 128],
            [0, 0, 128],
            [128, 0, 0],
        ],
        dtype=np.int32,
    )
    tile_count, lower_count, upper_count = _grid_parent_counts(points_np)
    points_d = wp.array(points_np, dtype=wp.int32, device=device)

    status_f = wp.zeros(1, dtype=wp.uint32, device=device)
    volume_f_exact = wp.Volume.allocate_by_tiles(points_d, 1.0, 123.456, device=device)
    volume_f_rebuildable = wp.Volume.allocate_by_tiles(
        points_d,
        1.0,
        123.456,
        device=device,
        graph_rebuildable=True,
        max_tiles=tile_count,
        max_lower_nodes=lower_count,
        max_upper_nodes=upper_count,
        status=status_f,
    )
    wp.synchronize_device(device)

    test.assertEqual(int(status_f.numpy()[0]), wp.Volume.REBUILD_SUCCESS)

    exact_tiles = _sort_rows(volume_f_exact.get_tiles().numpy())
    rebuildable_tiles_out = wp.full((tile_count, 3), -999, dtype=wp.int32, device=device)
    volume_f_rebuildable.get_tiles(out=rebuildable_tiles_out)
    rebuildable_tiles = _sort_rows(_valid_coord_rows(rebuildable_tiles_out.numpy()))
    np.testing.assert_array_equal(rebuildable_tiles, exact_tiles)

    tiles_d = wp.array(exact_tiles, dtype=wp.int32, device=device)
    exact_values = wp.empty(tile_count * 512, dtype=wp.float32, device=device)
    rebuildable_values = wp.empty(tile_count * 512, dtype=wp.float32, device=device)
    wp.launch(test_volume_tile_store_f, dim=tile_count, inputs=[volume_f_exact.id, tiles_d], device=device)
    wp.launch(test_volume_tile_store_f, dim=tile_count, inputs=[volume_f_rebuildable.id, tiles_d], device=device)
    wp.launch(
        test_volume_tile_readback_f, dim=tile_count, inputs=[volume_f_exact.id, tiles_d, exact_values], device=device
    )
    wp.launch(
        test_volume_tile_readback_f,
        dim=tile_count,
        inputs=[volume_f_rebuildable.id, tiles_d, rebuildable_values],
        device=device,
    )
    np.testing.assert_array_equal(rebuildable_values.numpy(), exact_values.numpy())

    status_v = wp.zeros(1, dtype=wp.uint32, device=device)
    volume_v_exact = wp.Volume.allocate_by_tiles(points_d, 1.0, wp.vec3(1.0, 2.0, 3.0), device=device)
    volume_v_rebuildable = wp.Volume.allocate_by_tiles(
        points_d,
        1.0,
        wp.vec3(1.0, 2.0, 3.0),
        device=device,
        graph_rebuildable=True,
        max_tiles=tile_count,
        max_lower_nodes=lower_count,
        max_upper_nodes=upper_count,
        status=status_v,
    )
    wp.synchronize_device(device)

    test.assertEqual(int(status_v.numpy()[0]), wp.Volume.REBUILD_SUCCESS)

    exact_vec_values = wp.empty(tile_count * 512, dtype=wp.vec3, device=device)
    rebuildable_vec_values = wp.empty(tile_count * 512, dtype=wp.vec3, device=device)
    wp.launch(test_volume_tile_store_v, dim=tile_count, inputs=[volume_v_exact.id, tiles_d], device=device)
    wp.launch(test_volume_tile_store_v, dim=tile_count, inputs=[volume_v_rebuildable.id, tiles_d], device=device)
    wp.launch(
        test_volume_tile_readback_v,
        dim=tile_count,
        inputs=[volume_v_exact.id, tiles_d, exact_vec_values],
        device=device,
    )
    wp.launch(
        test_volume_tile_readback_v,
        dim=tile_count,
        inputs=[volume_v_rebuildable.id, tiles_d, rebuildable_vec_values],
        device=device,
    )
    np.testing.assert_array_equal(rebuildable_vec_values.numpy(), exact_vec_values.numpy())

    status_i = wp.zeros(1, dtype=wp.uint32, device=device)
    volume_i_exact = wp.Volume.allocate_by_tiles(points_d, 1.0, int(7), device=device)
    volume_i_rebuildable = wp.Volume.allocate_by_tiles(
        points_d,
        1.0,
        int(7),
        device=device,
        graph_rebuildable=True,
        max_tiles=tile_count,
        max_lower_nodes=lower_count,
        max_upper_nodes=upper_count,
        status=status_i,
    )
    wp.synchronize_device(device)

    test.assertEqual(int(status_i.numpy()[0]), wp.Volume.REBUILD_SUCCESS)

    exact_int_values = wp.empty(tile_count * 512, dtype=wp.int32, device=device)
    rebuildable_int_values = wp.empty(tile_count * 512, dtype=wp.int32, device=device)
    wp.launch(test_volume_tile_store_i, dim=tile_count, inputs=[volume_i_exact.id, tiles_d], device=device)
    wp.launch(test_volume_tile_store_i, dim=tile_count, inputs=[volume_i_rebuildable.id, tiles_d], device=device)
    wp.launch(
        test_volume_tile_readback_i,
        dim=tile_count,
        inputs=[volume_i_exact.id, tiles_d, exact_int_values],
        device=device,
    )
    wp.launch(
        test_volume_tile_readback_i,
        dim=tile_count,
        inputs=[volume_i_rebuildable.id, tiles_d, rebuildable_int_values],
        device=device,
    )
    np.testing.assert_array_equal(rebuildable_int_values.numpy(), exact_int_values.numpy())

    status_v4 = wp.zeros(1, dtype=wp.uint32, device=device)
    volume_v4_exact = wp.Volume.allocate_by_tiles(points_d, 1.0, wp.vec4(1.0, 2.0, 3.0, 4.0), device=device)
    volume_v4_rebuildable = wp.Volume.allocate_by_tiles(
        points_d,
        1.0,
        wp.vec4(1.0, 2.0, 3.0, 4.0),
        device=device,
        graph_rebuildable=True,
        max_tiles=tile_count,
        max_lower_nodes=lower_count,
        max_upper_nodes=upper_count,
        status=status_v4,
    )
    wp.synchronize_device(device)

    test.assertEqual(int(status_v4.numpy()[0]), wp.Volume.REBUILD_SUCCESS)

    exact_vec4_values = wp.empty(tile_count * 512, dtype=wp.vec4, device=device)
    rebuildable_vec4_values = wp.empty(tile_count * 512, dtype=wp.vec4, device=device)
    wp.launch(test_volume_tile_store_v4, dim=tile_count, inputs=[volume_v4_exact.id, tiles_d], device=device)
    wp.launch(test_volume_tile_store_v4, dim=tile_count, inputs=[volume_v4_rebuildable.id, tiles_d], device=device)
    wp.launch(
        test_volume_tile_readback_v4,
        dim=tile_count,
        inputs=[volume_v4_exact.id, tiles_d, exact_vec4_values],
        device=device,
    )
    wp.launch(
        test_volume_tile_readback_v4,
        dim=tile_count,
        inputs=[volume_v4_rebuildable.id, tiles_d, rebuildable_vec4_values],
        device=device,
    )
    np.testing.assert_array_equal(rebuildable_vec4_values.numpy(), exact_vec4_values.numpy())

    status_index = wp.zeros(1, dtype=wp.uint32, device=device)
    volume_index_exact = wp.Volume.allocate_by_tiles(points_d, 1.0, bg_value=None, device=device)
    volume_index_rebuildable = wp.Volume.allocate_by_tiles(
        points_d,
        1.0,
        bg_value=None,
        device=device,
        graph_rebuildable=True,
        max_tiles=tile_count,
        max_lower_nodes=lower_count,
        max_upper_nodes=upper_count,
        status=status_index,
    )
    wp.synchronize_device(device)

    test.assertEqual(int(status_index.numpy()[0]), wp.Volume.REBUILD_SUCCESS)

    tile_voxels = _tile_voxels(exact_tiles)
    inactive_voxels = np.array([[4096, 4096, 4096], [-4097, 0, 0], [0, 7, 128]], dtype=np.int32)
    query_voxels = np.vstack((tile_voxels, inactive_voxels))
    np.testing.assert_array_equal(
        _lookup_indices(volume_index_rebuildable, query_voxels, device),
        _lookup_indices(volume_index_exact, query_voxels, device),
    )


def test_volume_rebuildable_setup_matches_exact_voxels(test, device):
    points_np = np.array(
        [
            [-1, -2, -3],
            [0, 0, 0],
            [1, 2, 3],
            [0, 8, 7],
            [0, 0, 128],
            [0, 0, 128],
            [128, 0, 0],
            [129, 0, 0],
        ],
        dtype=np.int32,
    )
    active_voxels = _unique_rows(points_np)
    leaf_count, lower_count, upper_count = _grid_parent_counts(active_voxels)
    points_d = wp.array(points_np, dtype=wp.int32, device=device)
    status = wp.zeros(1, dtype=wp.uint32, device=device)

    volume_exact = wp.Volume.allocate_by_voxels(points_d, 1.0, device=device)
    volume_rebuildable = wp.Volume.allocate_by_voxels(
        points_d,
        1.0,
        device=device,
        graph_rebuildable=True,
        max_active_voxels=len(active_voxels),
        max_leaf_nodes=leaf_count,
        max_lower_nodes=lower_count,
        max_upper_nodes=upper_count,
        status=status,
    )
    wp.synchronize_device(device)

    test.assertEqual(int(status.numpy()[0]), wp.Volume.REBUILD_SUCCESS)

    exact_voxels = _sort_rows(volume_exact.get_voxels().numpy())
    rebuildable_voxels_out = wp.full((len(active_voxels), 3), -999, dtype=wp.int32, device=device)
    volume_rebuildable.get_voxels(out=rebuildable_voxels_out)
    rebuildable_voxels = _sort_rows(_valid_coord_rows(rebuildable_voxels_out.numpy()))
    np.testing.assert_array_equal(rebuildable_voxels, exact_voxels)

    inactive_voxels = np.array([[4096, 4096, 4096], [-4097, 0, 0], [0, 7, 128]], dtype=np.int32)
    query_voxels = np.vstack((active_voxels, inactive_voxels))
    np.testing.assert_array_equal(
        _lookup_indices(volume_rebuildable, query_voxels, device),
        _lookup_indices(volume_exact, query_voxels, device),
    )


def _valid_coord_rows(values, sentinel=-999):
    return values[np.any(values != sentinel, axis=1)]


def test_volume_rebuild_by_tiles_capture(test, device):
    points_initial = wp.array([[0, 0, 0]], dtype=wp.int32, device=device)
    points_rebuild = wp.array([[8, 0, 0], [8, 0, 0], [0, 8, 0]], dtype=wp.int32, device=device)
    status = wp.zeros(1, dtype=wp.uint32, device=device)

    volume = wp.Volume.allocate_by_tiles(
        points_initial,
        voxel_size=1.0,
        bg_value=0.0,
        device=device,
        graph_rebuildable=True,
        max_tiles=4,
        max_lower_nodes=4,
        max_upper_nodes=4,
        status=status,
    )

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, force_module_load=False) as capture:
        volume.rebuild_by_tiles(points_rebuild)

    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)

    test.assertEqual(int(status.numpy()[0]), wp.Volume.REBUILD_SUCCESS)

    tiles = wp.full((4, 3), -999, dtype=wp.int32, device=device)
    volume.get_tiles(out=tiles)
    tiles_np = _valid_coord_rows(tiles.numpy())
    tiles_np = tiles_np[np.lexsort(tiles_np.T[::-1])]
    np.testing.assert_array_equal(tiles_np, np.array([[0, 8, 0], [8, 0, 0]], dtype=np.int32))


def test_volume_rebuild_by_tiles_parent_key_capture(test, device):
    points_initial = wp.array([[0, 0, 0]], dtype=wp.int32, device=device)
    points_rebuild = wp.array([[0, 0, 0], [0, 0, 128], [0, 8, 0], [0, 0, 128]], dtype=wp.int32, device=device)
    status = wp.zeros(1, dtype=wp.uint32, device=device)

    volume = wp.Volume.allocate_by_tiles(
        points_initial,
        voxel_size=1.0,
        bg_value=0.0,
        device=device,
        graph_rebuildable=True,
        max_tiles=3,
        max_lower_nodes=2,
        max_upper_nodes=1,
        status=status,
    )

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, force_module_load=False) as capture:
        volume.rebuild_by_tiles(points_rebuild)

    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)

    test.assertEqual(int(status.numpy()[0]), wp.Volume.REBUILD_SUCCESS)


def test_volume_rebuild_by_voxels_capture(test, device):
    points_initial = wp.array([[0, 0, 0]], dtype=wp.int32, device=device)
    points_rebuild = wp.array([[1, 2, 3], [1, 2, 3], [4, 5, 6]], dtype=wp.int32, device=device)
    status = wp.zeros(1, dtype=wp.uint32, device=device)

    volume = wp.Volume.allocate_by_voxels(
        points_initial,
        voxel_size=1.0,
        device=device,
        graph_rebuildable=True,
        max_active_voxels=4,
        max_leaf_nodes=4,
        max_lower_nodes=4,
        max_upper_nodes=4,
        status=status,
    )

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, force_module_load=False) as capture:
        volume.rebuild_by_voxels(points_rebuild)

    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)

    test.assertEqual(int(status.numpy()[0]), wp.Volume.REBUILD_SUCCESS)

    voxels = wp.full((4, 3), -999, dtype=wp.int32, device=device)
    volume.get_voxels(out=voxels)
    voxels_np = _valid_coord_rows(voxels.numpy())
    voxels_np = voxels_np[np.lexsort(voxels_np.T[::-1])]
    np.testing.assert_array_equal(voxels_np, np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32))


def test_volume_rebuild_capacity_status_capture(test, device):
    points_initial = wp.array([[0, 0, 0]], dtype=wp.int32, device=device)
    tile_points_overflow = wp.array([[0, 0, 0], [8, 0, 0]], dtype=wp.int32, device=device)
    voxel_points_overflow = wp.array([[1, 2, 3], [4, 5, 6]], dtype=wp.int32, device=device)
    tile_status = wp.zeros(1, dtype=wp.uint32, device=device)
    voxel_status = wp.zeros(1, dtype=wp.uint32, device=device)

    tile_volume = wp.Volume.allocate_by_tiles(
        points_initial,
        voxel_size=1.0,
        bg_value=0.0,
        device=device,
        graph_rebuildable=True,
        max_tiles=1,
        max_lower_nodes=1,
        max_upper_nodes=1,
        status=tile_status,
    )
    voxel_volume = wp.Volume.allocate_by_voxels(
        points_initial,
        voxel_size=1.0,
        device=device,
        graph_rebuildable=True,
        max_active_voxels=1,
        max_leaf_nodes=1,
        max_lower_nodes=1,
        max_upper_nodes=1,
        status=voxel_status,
    )

    wp.load_module(device=device)
    with wp.ScopedCapture(device=device, force_module_load=False) as capture:
        tile_volume.rebuild_by_tiles(tile_points_overflow)
        voxel_volume.rebuild_by_voxels(voxel_points_overflow)

    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)

    test.assertTrue(int(tile_status.numpy()[0]) & wp.Volume.REBUILD_LEAF_CAPACITY_EXCEEDED)
    test.assertTrue(int(voxel_status.numpy()[0]) & wp.Volume.REBUILD_VOXEL_CAPACITY_EXCEEDED)


devices = get_selected_cuda_test_devices()
capture_devices = get_selected_cuda_test_devices_with_mempool()


class TestVolumeWrite(unittest.TestCase):
    pass


add_function_test(TestVolumeWrite, "test_volume_allocation", test_volume_allocation, devices=devices)
add_function_test(TestVolumeWrite, "test_volume_allocate_by_tiles_f", test_volume_allocate_by_tiles_f, devices=devices)
add_function_test(TestVolumeWrite, "test_volume_allocate_by_tiles_v", test_volume_allocate_by_tiles_v, devices=devices)
add_function_test(
    TestVolumeWrite, "test_volume_allocate_by_tiles_index", test_volume_allocate_by_tiles_index, devices=devices
)
add_function_test(
    TestVolumeWrite,
    "test_volume_allocate_by_tiles_additional_types",
    test_volume_allocate_by_tiles_additional_types,
    devices=devices,
)
add_function_test(
    TestVolumeWrite,
    "test_volume_allocation_from_voxels",
    test_volume_allocation_from_voxels,
    devices=devices,
)
add_function_test(
    TestVolumeWrite,
    "test_volume_rebuildable_setup_matches_exact_tiles",
    test_volume_rebuildable_setup_matches_exact_tiles,
    devices=devices,
)
add_function_test(
    TestVolumeWrite,
    "test_volume_rebuildable_setup_matches_exact_voxels",
    test_volume_rebuildable_setup_matches_exact_voxels,
    devices=devices,
)
add_function_test(
    TestVolumeWrite,
    "test_volume_rebuild_by_tiles_capture",
    test_volume_rebuild_by_tiles_capture,
    devices=capture_devices,
)
add_function_test(
    TestVolumeWrite,
    "test_volume_rebuild_by_tiles_parent_key_capture",
    test_volume_rebuild_by_tiles_parent_key_capture,
    devices=capture_devices,
)
add_function_test(
    TestVolumeWrite,
    "test_volume_rebuild_by_voxels_capture",
    test_volume_rebuild_by_voxels_capture,
    devices=capture_devices,
)
add_function_test(
    TestVolumeWrite,
    "test_volume_rebuild_capacity_status_capture",
    test_volume_rebuild_capacity_status_capture,
    devices=capture_devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
