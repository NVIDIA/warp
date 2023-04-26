# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import warp as wp
from warp.tests.test_base import *

import numpy as np

wp.init()
# wp.config.cache_kernels = False


# Volume write tests
@wp.kernel
def test_volume_store_f(volume: wp.uint64, points: wp.array(dtype=wp.vec3)):
    tid = wp.tid()

    p = points[tid]
    i = int(p[0])
    j = int(p[1])
    k = int(p[2])

    wp.volume_store_f(volume, i, j, k, float(i + 100 * j + 10000 * k))


@wp.kernel
def test_volume_readback_f(volume: wp.uint64, points: wp.array(dtype=wp.vec3), values: wp.array(dtype=wp.float32)):
    tid = wp.tid()

    p = points[tid]
    i = int(p[0])
    j = int(p[1])
    k = int(p[2])

    values[tid] = wp.volume_lookup_f(volume, i, j, k)


@wp.kernel
def test_get_list_of_tiles(
    volume: wp.uint64,
    points_is: wp.array2d(dtype=wp.int32),
    points_ws: wp.array(dtype=wp.vec3),
    tiles_is: wp.array2d(dtype=wp.int32),
    tiles_ws: wp.array2d(dtype=wp.int32),
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
def test_volume_tile_store_f(volume: wp.uint64, tiles: wp.array2d(dtype=wp.int32)):
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
def test_volume_tile_store_ws_f(volume: wp.uint64, tiles: wp.array(dtype=wp.vec3)):
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
def test_volume_tile_readback_f(
    volume: wp.uint64, tiles: wp.array2d(dtype=wp.int32), values: wp.array(dtype=wp.float32)
):
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
def test_volume_tile_store_v(volume: wp.uint64, tiles: wp.array2d(dtype=wp.int32)):
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
def test_volume_tile_readback_v(volume: wp.uint64, tiles: wp.array2d(dtype=wp.int32), values: wp.array(dtype=wp.vec3)):
    tid = wp.tid()

    ti = tiles[tid, 0]
    tj = tiles[tid, 1]
    tk = tiles[tid, 2]

    for r in range(512):
        ii = ti + (r / 64) % 8
        jj = tj + (r / 8) % 8
        kk = tk + r % 8
        values[tid * 512 + r] = wp.volume_lookup_v(volume, ii, jj, kk)


rng = np.random.default_rng(101215)


def register(parent):
    devices = get_test_devices()

    class TestVolumes(parent):
        def test_volume_allocation(self):
            voxel_size = 0.125
            background_value = 123.456
            translation = wp.vec3(-12.3, 4.56, -789)

            axis = np.linspace(-11, 11, 23)
            points_ref = np.array([[x, y, z] for x in axis for y in axis for z in axis])
            values_ref = np.array([x + 100 * y + 10000 * z for x in axis for y in axis for z in axis])
            num_points = len(points_ref)
            bb_max = np.array([11, 11, 11])
            for device in devices:
                if device.is_cpu:
                    continue

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
                points = wp.array(points_ref, dtype=wp.vec3, device=device)
                values_a = wp.empty(num_points, dtype=wp.float32, device=device)
                values_b = wp.empty(num_points, dtype=wp.float32, device=device)
                wp.launch(test_volume_store_f, dim=num_points, inputs=[volume_a.id, points], device=device)
                wp.launch(test_volume_store_f, dim=num_points, inputs=[volume_b.id, points], device=device)
                wp.launch(test_volume_readback_f, dim=num_points, inputs=[volume_a.id, points, values_a], device=device)
                wp.launch(test_volume_readback_f, dim=num_points, inputs=[volume_b.id, points, values_b], device=device)

                np.testing.assert_equal(values_a.numpy(), values_ref)
                np.testing.assert_equal(values_b.numpy(), values_ref)

        def test_volume_allocate_by_tiles_f(self):
            voxel_size = 0.125
            background_value = 123.456
            translation = wp.vec3(-12.3, 4.56, -789)

            num_tiles = 1000
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

            for device in devices:
                if device.is_cpu:
                    continue

                points_is_d = wp.array(points_is, dtype=wp.int32, device=device)
                points_ws_d = wp.array(points_ws, dtype=wp.vec3, device=device)
                volume_a = wp.Volume.allocate_by_tiles(
                    points_is_d, voxel_size, background_value, translation, device=device
                )
                volume_b = wp.Volume.allocate_by_tiles(
                    points_ws_d, voxel_size, background_value, translation, device=device
                )
                values_a = wp.empty(num_tiles * 512, dtype=wp.float32, device=device)
                values_b = wp.empty(num_tiles * 512, dtype=wp.float32, device=device)

                wp.launch(test_volume_tile_store_f, dim=num_tiles, inputs=[volume_a.id, points_is_d], device=device)
                wp.launch(test_volume_tile_store_ws_f, dim=num_tiles, inputs=[volume_b.id, points_ws_d], device=device)
                wp.launch(
                    test_volume_tile_readback_f,
                    dim=num_tiles,
                    inputs=[volume_a.id, points_is_d, values_a],
                    device=device,
                )
                wp.launch(
                    test_volume_tile_readback_f,
                    dim=num_tiles,
                    inputs=[volume_b.id, points_is_d, values_b],
                    device=device,
                )

                np.testing.assert_equal(values_a.numpy(), values_ref)
                np.testing.assert_equal(values_b.numpy(), values_ref)

        def test_volume_allocate_by_tiles_v(self):
            num_tiles = 1000
            tiles = rng.integers(-512, 512, size=(num_tiles, 3), dtype=np.int32)
            points_is = tiles * 8

            values_ref = np.empty((len(tiles) * 512, 3))
            for t in range(len(tiles)):
                ti, tj, tk = points_is[t]
                for i in range(8):
                    for j in range(8):
                        for k in range(8):
                            values_ref[t * 512 + i * 64 + j * 8 + k] = [ti + i, tj + j, tk + k]

            for device in devices:
                if device.is_cpu:
                    continue

                points_d = wp.array(points_is, dtype=wp.int32, device=device)
                volume = wp.Volume.allocate_by_tiles(points_d, 0.1, wp.vec3(1, 2, 3), device=device)
                values = wp.empty(len(points_d) * 512, dtype=wp.vec3, device=device)

                wp.launch(test_volume_tile_store_v, dim=len(points_d), inputs=[volume.id, points_d], device=device)
                wp.launch(
                    test_volume_tile_readback_v, dim=len(points_d), inputs=[volume.id, points_d, values], device=device
                )

                values_res = values.numpy()
                np.testing.assert_equal(values_res, values_ref)

    return TestVolumes


if __name__ == "__main__":
    wp.force_load()
    c = register(unittest.TestCase)
    unittest.main(verbosity=2)
