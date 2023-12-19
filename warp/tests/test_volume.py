# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

wp.init()


# float volume tests
@wp.kernel
def test_volume_lookup_f(volume: wp.uint64, points: wp.array(dtype=wp.vec3)):
    tid = wp.tid()

    p = points[tid]
    expected = p[0] * p[1] * p[2]
    if abs(p[0]) > 10.0 or abs(p[1]) > 10.0 or abs(p[2]) > 10.0:
        expected = 10.0

    i = int(p[0])
    j = int(p[1])
    k = int(p[2])

    expect_eq(wp.volume_lookup_f(volume, i, j, k), expected)


@wp.kernel
def test_volume_sample_closest_f(volume: wp.uint64, points: wp.array(dtype=wp.vec3)):
    tid = wp.tid()

    p = points[tid]
    i = round(p[0])
    j = round(p[1])
    k = round(p[2])
    expected = i * j * k
    if abs(i) > 10.0 or abs(j) > 10.0 or abs(k) > 10.0:
        expected = 10.0

    expect_eq(wp.volume_sample_f(volume, p, wp.Volume.CLOSEST), expected)

    q = wp.volume_index_to_world(volume, p)
    q_inv = wp.volume_world_to_index(volume, q)
    expect_eq(p, q_inv)


@wp.kernel
def test_volume_sample_linear_f(volume: wp.uint64, points: wp.array(dtype=wp.vec3)):
    tid = wp.tid()

    p = points[tid]

    expected = p[0] * p[1] * p[2]
    if abs(p[0]) > 10.0 or abs(p[1]) > 10.0 or abs(p[2]) > 10.0:
        return  # not testing against background values

    expect_near(wp.volume_sample_f(volume, p, wp.Volume.LINEAR), expected, 2.0e-4)


@wp.kernel
def test_volume_sample_grad_linear_f(volume: wp.uint64, points: wp.array(dtype=wp.vec3)):
    tid = wp.tid()

    p = points[tid]

    expected_val = p[0] * p[1] * p[2]
    expected_gx = p[1] * p[2]
    expected_gy = p[0] * p[2]
    expected_gz = p[0] * p[1]

    if abs(p[0]) > 10.0 or abs(p[1]) > 10.0 or abs(p[2]) > 10.0:
        return  # not testing against background values

    grad = wp.vec3(0.0, 0.0, 0.0)
    val = wp.volume_sample_grad_f(volume, p, wp.Volume.LINEAR, grad)

    expect_near(val, expected_val, 2.0e-4)
    expect_near(grad[0], expected_gx, 2.0e-4)
    expect_near(grad[1], expected_gy, 2.0e-4)
    expect_near(grad[2], expected_gz, 2.0e-4)


@wp.kernel
def test_volume_sample_local_f_linear_values(
    volume: wp.uint64, points: wp.array(dtype=wp.vec3), values: wp.array(dtype=wp.float32)
):
    tid = wp.tid()
    p = points[tid]
    values[tid] = wp.volume_sample_f(volume, p, wp.Volume.LINEAR)


@wp.kernel
def test_volume_sample_grad_local_f_linear_values(
    volume: wp.uint64, points: wp.array(dtype=wp.vec3), values: wp.array(dtype=wp.float32), case_num: int
):
    tid = wp.tid()
    p = points[tid]

    grad = wp.vec3(0.0, 0.0, 0.0)
    val = wp.volume_sample_grad_f(volume, p, wp.Volume.LINEAR, grad)

    if case_num == 1:
        val = grad[0]
    elif case_num == 2:
        val = grad[1]
    elif case_num == 3:
        val = grad[2]
    values[tid] = val


@wp.kernel
def test_volume_sample_world_f_linear_values(
    volume: wp.uint64, points: wp.array(dtype=wp.vec3), values: wp.array(dtype=wp.float32)
):
    tid = wp.tid()
    q = points[tid]
    p = wp.volume_world_to_index(volume, q)
    values[tid] = wp.volume_sample_f(volume, p, wp.Volume.LINEAR)


@wp.kernel
def test_volume_sample_grad_world_f_linear_values(
    volume: wp.uint64, points: wp.array(dtype=wp.vec3), values: wp.array(dtype=wp.float32), case_num: int
):
    tid = wp.tid()
    q = points[tid]
    p = wp.volume_world_to_index(volume, q)

    grad = wp.vec3(0.0, 0.0, 0.0)
    val = wp.volume_sample_grad_f(volume, p, wp.Volume.LINEAR, grad)

    if case_num == 1:
        val = grad[0]
    elif case_num == 2:
        val = grad[1]
    elif case_num == 3:
        val = grad[2]

    values[tid] = val


# vec3f volume tests
@wp.kernel
def test_volume_lookup_v(volume: wp.uint64, points: wp.array(dtype=wp.vec3)):
    tid = wp.tid()

    p = points[tid]
    expected = wp.vec3(
        p[0] + 2.0 * p[1] + 3.0 * p[2], 4.0 * p[0] + 5.0 * p[1] + 6.0 * p[2], 7.0 * p[0] + 8.0 * p[1] + 9.0 * p[2]
    )
    if abs(p[0]) > 10.0 or abs(p[1]) > 10.0 or abs(p[2]) > 10.0:
        expected = wp.vec3(10.8, -4.13, 10.26)

    i = int(p[0])
    j = int(p[1])
    k = int(p[2])

    expect_eq(wp.volume_lookup_v(volume, i, j, k), expected)


@wp.kernel
def test_volume_sample_closest_v(volume: wp.uint64, points: wp.array(dtype=wp.vec3)):
    tid = wp.tid()

    p = points[tid]
    i = round(p[0])
    j = round(p[1])
    k = round(p[2])
    expected = wp.vec3(i + 2.0 * j + 3.0 * k, 4.0 * i + 5.0 * j + 6.0 * k, 7.0 * i + 8.0 * j + 9.0 * k)
    if abs(i) > 10.0 or abs(j) > 10.0 or abs(k) > 10.0:
        expected = wp.vec3(10.8, -4.13, 10.26)

    expect_eq(wp.volume_sample_v(volume, p, wp.Volume.CLOSEST), expected)

    q = wp.volume_index_to_world(volume, p)
    q_inv = wp.volume_world_to_index(volume, q)
    expect_eq(p, q_inv)


@wp.kernel
def test_volume_sample_linear_v(volume: wp.uint64, points: wp.array(dtype=wp.vec3)):
    tid = wp.tid()

    p = points[tid]

    expected = wp.vec3(
        p[0] + 2.0 * p[1] + 3.0 * p[2], 4.0 * p[0] + 5.0 * p[1] + 6.0 * p[2], 7.0 * p[0] + 8.0 * p[1] + 9.0 * p[2]
    )
    if abs(p[0]) > 10.0 or abs(p[1]) > 10.0 or abs(p[2]) > 10.0:
        return  # not testing against background values

    expect_near(wp.volume_sample_v(volume, p, wp.Volume.LINEAR), expected, 2.0e-4)


@wp.kernel
def test_volume_sample_local_v_linear_values(
    volume: wp.uint64, points: wp.array(dtype=wp.vec3), values: wp.array(dtype=wp.float32)
):
    tid = wp.tid()
    p = points[tid]
    ones = wp.vec3(1.0, 1.0, 1.0)
    values[tid] = wp.dot(wp.volume_sample_v(volume, p, wp.Volume.LINEAR), ones)


@wp.kernel
def test_volume_sample_world_v_linear_values(
    volume: wp.uint64, points: wp.array(dtype=wp.vec3), values: wp.array(dtype=wp.float32)
):
    tid = wp.tid()
    q = points[tid]
    p = wp.volume_world_to_index(volume, q)
    ones = wp.vec3(1.0, 1.0, 1.0)
    values[tid] = wp.dot(wp.volume_sample_v(volume, p, wp.Volume.LINEAR), ones)


# int32 volume tests
@wp.kernel
def test_volume_lookup_i(volume: wp.uint64, points: wp.array(dtype=wp.vec3)):
    tid = wp.tid()

    p = points[tid]
    i = int(p[0])
    j = int(p[1])
    k = int(p[2])
    expected = i * j * k
    if abs(i) > 10 or abs(j) > 10 or abs(k) > 10:
        expected = 10

    expect_eq(wp.volume_lookup_i(volume, i, j, k), expected)


@wp.kernel
def test_volume_sample_i(volume: wp.uint64, points: wp.array(dtype=wp.vec3)):
    tid = wp.tid()

    p = points[tid]
    i = round(p[0])
    j = round(p[1])
    k = round(p[2])
    expected = int(i * j * k)
    if abs(i) > 10.0 or abs(j) > 10.0 or abs(k) > 10.0:
        expected = 10

    expect_eq(wp.volume_sample_i(volume, p), expected)

    q = wp.volume_index_to_world(volume, p)
    q_inv = wp.volume_world_to_index(volume, q)
    expect_eq(p, q_inv)


# Index/world transformation tests
@wp.kernel
def test_volume_index_to_world(
    volume: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    values: wp.array(dtype=wp.float32),
    grad_values: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    p = points[tid]
    ones = wp.vec3(1.0, 1.0, 1.0)
    values[tid] = wp.dot(wp.volume_index_to_world(volume, p), ones)
    grad_values[tid] = wp.volume_index_to_world_dir(volume, ones)


@wp.kernel
def test_volume_world_to_index(
    volume: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    values: wp.array(dtype=wp.float32),
    grad_values: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    p = points[tid]
    ones = wp.vec3(1.0, 1.0, 1.0)
    values[tid] = wp.dot(wp.volume_world_to_index(volume, p), ones)
    grad_values[tid] = wp.volume_world_to_index_dir(volume, ones)


# Volume write tests
@wp.kernel
def test_volume_store_f(volume: wp.uint64, points: wp.array(dtype=wp.vec3), values: wp.array(dtype=wp.float32)):
    tid = wp.tid()

    p = points[tid]
    i = int(p[0])
    j = int(p[1])
    k = int(p[2])

    # NB: Writing outside the allocated domain overwrites the background value of the Volume
    if abs(i) <= 11 and abs(j) <= 11 and abs(k) <= 11:
        wp.volume_store_f(volume, i, j, k, float(i + 100 * j + 10000 * k))
    values[tid] = wp.volume_lookup_f(volume, i, j, k)


@wp.kernel
def test_volume_store_v(volume: wp.uint64, points: wp.array(dtype=wp.vec3), values: wp.array(dtype=wp.vec3)):
    tid = wp.tid()

    p = points[tid]
    i = int(p[0])
    j = int(p[1])
    k = int(p[2])

    # NB: Writing outside the allocated domain overwrites the background value of the Volume
    if abs(i) <= 11 and abs(j) <= 11 and abs(k) <= 11:
        wp.volume_store_v(volume, i, j, k, p)
    values[tid] = wp.volume_lookup_v(volume, i, j, k)


@wp.kernel
def test_volume_store_i(volume: wp.uint64, points: wp.array(dtype=wp.vec3), values: wp.array(dtype=wp.int32)):
    tid = wp.tid()

    p = points[tid]
    i = int(p[0])
    j = int(p[1])
    k = int(p[2])

    # NB: Writing outside the allocated domain overwrites the background value of the Volume
    if abs(i) <= 11 and abs(j) <= 11 and abs(k) <= 11:
        wp.volume_store_i(volume, i, j, k, i + 100 * j + 10000 * k)
    values[tid] = wp.volume_lookup_i(volume, i, j, k)


devices = get_test_devices()
rng = np.random.default_rng(101215)

# Note about the test grids:
# test_grid and test_int32_grid
#   active region: [-10,10]^3
#   values: v[i,j,k] = i * j * k
#   voxel size: 0.25
#
# test_vec_grid
#   active region: [-10,10]^3
#   values: v[i,j,k] = (i + 2*j + 3*k, 4*i + 5*j + 6*k, 7*i + 8*j + 9*k)
#   voxel size: 0.25
#
# torus
#   index to world transformation:
#      [0.1, 0, 0, 0]
#      [0, 0, 0.1, 0]
#      [0, 0.1, 0, 0]
#      [1, 2, 3, 1]
#   (-90 degrees rotation along X)
#   voxel size: 0.1
volume_paths = {
    "float": os.path.abspath(os.path.join(os.path.dirname(__file__), "assets/test_grid.nvdb")),
    "int32": os.path.abspath(os.path.join(os.path.dirname(__file__), "assets/test_int32_grid.nvdb")),
    "vec3f": os.path.abspath(os.path.join(os.path.dirname(__file__), "assets/test_vec_grid.nvdb")),
    "torus": os.path.abspath(os.path.join(os.path.dirname(__file__), "assets/torus.nvdb")),
    "float_write": os.path.abspath(os.path.join(os.path.dirname(__file__), "assets/test_grid.nvdb")),
}

test_volume_tiles = (
    np.array([[i, j, k] for i in range(-2, 2) for j in range(-2, 2) for k in range(-2, 2)], dtype=np.int32) * 8
)

volumes = {}
for value_type, path in volume_paths.items():
    volumes[value_type] = {}
    volume_data = open(path, "rb").read()
    for device in devices:
        try:
            volume = wp.Volume.load_from_nvdb(volume_data, device)
        except RuntimeError as e:
            raise RuntimeError(f'Failed to load volume from "{path}" to {device} memory:\n{e}')

        volumes[value_type][device.alias] = volume

axis = np.linspace(-1, 1, 3)
point_grid = np.array([[x, y, z] for x in axis for y in axis for z in axis], dtype=np.float32)


def test_volume_sample_linear_f_gradient(test, device):
    points = rng.uniform(-10.0, 10.0, size=(100, 3))
    values = wp.array(np.zeros(1), dtype=wp.float32, device=device, requires_grad=True)
    for test_case in points:
        uvws = wp.array(test_case, dtype=wp.vec3, device=device, requires_grad=True)
        xyzs = wp.array(test_case * 0.25, dtype=wp.vec3, device=device, requires_grad=True)

        tape = wp.Tape()
        with tape:
            wp.launch(
                test_volume_sample_local_f_linear_values,
                dim=1,
                inputs=[volumes["float"][device.alias].id, uvws, values],
                device=device,
            )
        tape.backward(values)

        x, y, z = test_case
        grad_expected = np.array([y * z, x * z, x * y])
        grad_computed = tape.gradients[uvws].numpy()[0]
        np.testing.assert_allclose(grad_computed, grad_expected, rtol=1e-4)

        tape = wp.Tape()
        with tape:
            wp.launch(
                test_volume_sample_world_f_linear_values,
                dim=1,
                inputs=[volumes["float"][device.alias].id, xyzs, values],
                device=device,
            )
        tape.backward(values)

        x, y, z = test_case
        grad_expected = np.array([y * z, x * z, x * y]) / 0.25
        grad_computed = tape.gradients[xyzs].numpy()[0]
        np.testing.assert_allclose(grad_computed, grad_expected, rtol=1e-4)


def test_volume_sample_grad_linear_f_gradient(test, device):
    points = rng.uniform(-10.0, 10.0, size=(100, 3))
    values = wp.array(np.zeros(1), dtype=wp.float32, device=device, requires_grad=True)
    for test_case in points:
        uvws = wp.array(test_case, dtype=wp.vec3, device=device, requires_grad=True)
        xyzs = wp.array(test_case * 0.25, dtype=wp.vec3, device=device, requires_grad=True)

        for case_num in range(4):
            tape = wp.Tape()
            with tape:
                wp.launch(
                    test_volume_sample_grad_local_f_linear_values,
                    dim=1,
                    inputs=[volumes["float"][device.alias].id, uvws, values, case_num],
                    device=device,
                )
            tape.backward(values)

            x, y, z = test_case
            grad_computed = tape.gradients[uvws].numpy()[0]
            if case_num == 0:
                grad_expected = np.array([y * z, x * z, x * y])
            elif case_num == 1:
                grad_expected = np.array([0.0, z, y])
            elif case_num == 2:
                grad_expected = np.array([z, 0.0, x])
            elif case_num == 3:
                grad_expected = np.array([y, x, 0.0])

            np.testing.assert_allclose(grad_computed, grad_expected, rtol=1e-4)
            tape.zero()

        for case_num in range(4):
            tape = wp.Tape()
            with tape:
                wp.launch(
                    test_volume_sample_grad_world_f_linear_values,
                    dim=1,
                    inputs=[volumes["float"][device.alias].id, xyzs, values, case_num],
                    device=device,
                )
            tape.backward(values)

            x, y, z = test_case
            grad_computed = tape.gradients[xyzs].numpy()[0]
            if case_num == 0:
                grad_expected = np.array([y * z, x * z, x * y]) / 0.25
            elif case_num == 1:
                grad_expected = np.array([0.0, z, y]) / 0.25
            elif case_num == 2:
                grad_expected = np.array([z, 0.0, x]) / 0.25
            elif case_num == 3:
                grad_expected = np.array([y, x, 0.0]) / 0.25

            np.testing.assert_allclose(grad_computed, grad_expected, rtol=1e-4)
            tape.zero()


def test_volume_sample_linear_v_gradient(test, device):
    points = rng.uniform(-10.0, 10.0, size=(100, 3))
    values = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
    for test_case in points:
        uvws = wp.array(test_case, dtype=wp.vec3, device=device, requires_grad=True)
        xyzs = wp.array(test_case * 0.25, dtype=wp.vec3, device=device, requires_grad=True)

        tape = wp.Tape()
        with tape:
            wp.launch(
                test_volume_sample_local_v_linear_values,
                dim=1,
                inputs=[volumes["vec3f"][device.alias].id, uvws, values],
                device=device,
            )
        tape.backward(values)

        grad_expected = np.array([6.0, 15.0, 24.0])
        grad_computed = tape.gradients[uvws].numpy()[0]
        np.testing.assert_allclose(grad_computed, grad_expected, rtol=1e-4)

        tape = wp.Tape()
        with tape:
            wp.launch(
                test_volume_sample_world_v_linear_values,
                dim=1,
                inputs=[volumes["vec3f"][device.alias].id, xyzs, values],
                device=device,
            )
        tape.backward(values)

        grad_expected = np.array([6.0, 15.0, 24.0]) / 0.25
        grad_computed = tape.gradients[xyzs].numpy()[0]
        np.testing.assert_allclose(grad_computed, grad_expected, rtol=1e-4)


def test_volume_transform_gradient(test, device):
    values = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
    grad_values = wp.zeros(1, dtype=wp.vec3, device=device)
    test_points = rng.uniform(-10.0, 10.0, size=(10, 3))
    for test_case in test_points:
        points = wp.array(test_case, dtype=wp.vec3, device=device, requires_grad=True)
        tape = wp.Tape()
        with tape:
            wp.launch(
                test_volume_index_to_world,
                dim=1,
                inputs=[volumes["torus"][device.alias].id, points, values, grad_values],
                device=device,
            )
        tape.backward(values)

        grad_computed = tape.gradients[points].numpy()
        grad_expected = grad_values.numpy()
        np.testing.assert_allclose(grad_computed, grad_expected, rtol=1e-4)

        grad_computed = tape.gradients[points].numpy()
        grad_expected = grad_values.numpy()
        np.testing.assert_allclose(grad_computed, grad_expected, rtol=1e-4)


def test_volume_store(test, device):
    values_ref = np.array([x + 100 * y + 10000 * z for x, y, z in point_grid])
    points = wp.array(point_grid, dtype=wp.vec3, device=device)
    values = wp.empty(len(point_grid), dtype=wp.float32, device=device)
    wp.launch(
        test_volume_store_f,
        dim=len(point_grid),
        inputs=[volumes["float_write"][device.alias].id, points, values],
        device=device,
    )

    values_res = values.numpy()
    np.testing.assert_equal(values_res, values_ref)


def test_volume_allocation_f(test, device):
    bg_value = -123.0
    points_np = np.append(point_grid, [[8096, 8096, 8096]], axis=0)
    values_ref = np.append(np.array([x + 100 * y + 10000 * z for x, y, z in point_grid]), bg_value)

    volume = wp.Volume.allocate(min=[-11, -11, -11], max=[11, 11, 11], voxel_size=0.1, bg_value=bg_value, device=device)
    points = wp.array(points_np, dtype=wp.vec3, device=device)
    values = wp.empty(len(points_np), dtype=wp.float32, device=device)
    wp.launch(test_volume_store_f, dim=len(points_np), inputs=[volume.id, points, values], device=device)

    values_res = values.numpy()
    np.testing.assert_equal(values_res, values_ref)


def test_volume_allocation_v(test, device):
    bg_value = (-1, 2.0, -3)
    points_np = np.append(point_grid, [[8096, 8096, 8096]], axis=0)
    values_ref = np.append(point_grid, [bg_value], axis=0)

    volume = wp.Volume.allocate(min=[-11, -11, -11], max=[11, 11, 11], voxel_size=0.1, bg_value=bg_value, device=device)
    points = wp.array(points_np, dtype=wp.vec3, device=device)
    values = wp.empty(len(points_np), dtype=wp.vec3, device=device)
    wp.launch(test_volume_store_v, dim=len(points_np), inputs=[volume.id, points, values], device=device)

    values_res = values.numpy()
    np.testing.assert_equal(values_res, values_ref)


def test_volume_allocation_i(test, device):
    bg_value = -123
    points_np = np.append(point_grid, [[8096, 8096, 8096]], axis=0)
    values_ref = np.append(np.array([x + 100 * y + 10000 * z for x, y, z in point_grid], dtype=np.int32), bg_value)

    volume = wp.Volume.allocate(min=[-11, -11, -11], max=[11, 11, 11], voxel_size=0.1, bg_value=bg_value, device=device)
    points = wp.array(points_np, dtype=wp.vec3, device=device)
    values = wp.empty(len(points_np), dtype=wp.int32, device=device)
    wp.launch(test_volume_store_i, dim=len(points_np), inputs=[volume.id, points, values], device=device)

    values_res = values.numpy()
    np.testing.assert_equal(values_res, values_ref)


def test_volume_introspection(test, device):
    for volume_names in ("float", "vec3f"):
        with test.subTest(volume_names=volume_names):
            volume = volumes[volume_names][device.alias]
            tiles_actual = volume.get_tiles().numpy()
            tiles_sorted = tiles_actual[np.lexsort(tiles_actual.T[::-1])]
            voxel_size = np.array(volume.get_voxel_size())

            np.testing.assert_equal(test_volume_tiles, tiles_sorted)
            np.testing.assert_equal([0.25] * 3, voxel_size)


def test_volume_from_numpy(test, device):
    # Volume.allocate_from_tiles() is only available with CUDA
    mins = np.array([-3.0, -3.0, -3.0])
    voxel_size = 0.2
    maxs = np.array([3.0, 3.0, 3.0])
    nums = np.ceil((maxs - mins) / (voxel_size)).astype(dtype=int)
    center = np.array([0.0, 0.0, 0.0])
    rad = 2.5
    sphere_sdf_np = np.zeros(tuple(nums))
    for x in range(nums[0]):
        for y in range(nums[1]):
            for z in range(nums[2]):
                pos = mins + voxel_size * np.array([x, y, z])
                dis = np.linalg.norm(pos - center)
                sphere_sdf_np[x, y, z] = dis - rad
    sphere_vdb = wp.Volume.load_from_numpy(sphere_sdf_np, mins, voxel_size, rad + 3.0 * voxel_size, device=device)

    test.assertNotEqual(sphere_vdb.id, 0)

    sphere_vdb_array = sphere_vdb.array()
    test.assertEqual(sphere_vdb_array.dtype, wp.uint8)
    test.assertFalse(sphere_vdb_array.owner)


class TestVolume(unittest.TestCase):
    pass


add_function_test(
    TestVolume, "test_volume_sample_linear_f_gradient", test_volume_sample_linear_f_gradient, devices=devices
)
add_function_test(
    TestVolume, "test_volume_sample_grad_linear_f_gradient", test_volume_sample_grad_linear_f_gradient, devices=devices
)
add_function_test(
    TestVolume, "test_volume_sample_linear_v_gradient", test_volume_sample_linear_v_gradient, devices=devices
)
add_function_test(TestVolume, "test_volume_transform_gradient", test_volume_transform_gradient, devices=devices)
add_function_test(TestVolume, "test_volume_store", test_volume_store, devices=devices)
add_function_test(
    TestVolume, "test_volume_allocation_f", test_volume_allocation_f, devices=get_unique_cuda_test_devices()
)
add_function_test(
    TestVolume, "test_volume_allocation_v", test_volume_allocation_v, devices=get_unique_cuda_test_devices()
)
add_function_test(
    TestVolume, "test_volume_allocation_i", test_volume_allocation_i, devices=get_unique_cuda_test_devices()
)
add_function_test(TestVolume, "test_volume_introspection", test_volume_introspection, devices=devices)
add_function_test(TestVolume, "test_volume_from_numpy", test_volume_from_numpy, devices=get_unique_cuda_test_devices())

points = {}
points_jittered = {}
for device in devices:
    points_jittered_np = point_grid + rng.uniform(-0.5, 0.5, size=point_grid.shape)
    points[device.alias] = wp.array(point_grid, dtype=wp.vec3, device=device)
    points_jittered[device.alias] = wp.array(points_jittered_np, dtype=wp.vec3, device=device)

    add_kernel_test(
        TestVolume,
        test_volume_lookup_f,
        dim=len(point_grid),
        inputs=[volumes["float"][device.alias].id, points[device.alias]],
        devices=[device],
    )
    add_kernel_test(
        TestVolume,
        test_volume_sample_closest_f,
        dim=len(point_grid),
        inputs=[volumes["float"][device.alias].id, points_jittered[device.alias]],
        devices=[device.alias],
    )
    add_kernel_test(
        TestVolume,
        test_volume_sample_linear_f,
        dim=len(point_grid),
        inputs=[volumes["float"][device.alias].id, points_jittered[device.alias]],
        devices=[device.alias],
    )
    add_kernel_test(
        TestVolume,
        test_volume_sample_grad_linear_f,
        dim=len(point_grid),
        inputs=[volumes["float"][device.alias].id, points_jittered[device.alias]],
        devices=[device.alias],
    )

    add_kernel_test(
        TestVolume,
        test_volume_lookup_v,
        dim=len(point_grid),
        inputs=[volumes["vec3f"][device.alias].id, points[device.alias]],
        devices=[device.alias],
    )
    add_kernel_test(
        TestVolume,
        test_volume_sample_closest_v,
        dim=len(point_grid),
        inputs=[volumes["vec3f"][device.alias].id, points_jittered[device.alias]],
        devices=[device.alias],
    )
    add_kernel_test(
        TestVolume,
        test_volume_sample_linear_v,
        dim=len(point_grid),
        inputs=[volumes["vec3f"][device.alias].id, points_jittered[device.alias]],
        devices=[device.alias],
    )

    add_kernel_test(
        TestVolume,
        test_volume_lookup_i,
        dim=len(point_grid),
        inputs=[volumes["int32"][device.alias].id, points[device.alias]],
        devices=[device.alias],
    )
    add_kernel_test(
        TestVolume,
        test_volume_sample_i,
        dim=len(point_grid),
        inputs=[volumes["int32"][device.alias].id, points_jittered[device.alias]],
        devices=[device.alias],
    )


if __name__ == "__main__":
    wp.build.clear_kernel_cache()
    unittest.main(verbosity=2)
