# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
import unittest
from typing import Any

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


# float volume tests
@wp.kernel
def test_volume_lookup_f(volume: wp.uint64, points: wp.array[wp.vec3]):
    tid = wp.tid()

    p = points[tid]
    expected = p[0] * p[1] * p[2]
    if abs(p[0]) > 10.0 or abs(p[1]) > 10.0 or abs(p[2]) > 10.0:
        expected = 10.0

    i = int(p[0])
    j = int(p[1])
    k = int(p[2])

    expect_eq(wp.volume_lookup_f(volume, i, j, k), expected)
    expect_eq(wp.volume_lookup(volume, i, j, k, dtype=wp.float32), expected)


@wp.kernel
def test_volume_sample_closest_f(volume: wp.uint64, points: wp.array[wp.vec3]):
    tid = wp.tid()

    p = points[tid]
    i = round(p[0])
    j = round(p[1])
    k = round(p[2])
    expected = i * j * k
    if abs(i) > 10.0 or abs(j) > 10.0 or abs(k) > 10.0:
        expected = 10.0

    expect_eq(wp.volume_sample_f(volume, p, wp.Volume.CLOSEST), expected)
    expect_eq(wp.volume_sample(volume, p, wp.Volume.CLOSEST, dtype=wp.float32), expected)

    q = wp.volume_index_to_world(volume, p)
    q_inv = wp.volume_world_to_index(volume, q)
    expect_eq(p, q_inv)


@wp.kernel
def test_volume_sample_linear_f(volume: wp.uint64, points: wp.array[wp.vec3]):
    tid = wp.tid()

    p = points[tid]

    expected = p[0] * p[1] * p[2]
    if abs(p[0]) > 10.0 or abs(p[1]) > 10.0 or abs(p[2]) > 10.0:
        return  # not testing against background values

    expect_near(wp.volume_sample_f(volume, p, wp.Volume.LINEAR), expected, 2.0e-4)
    expect_near(wp.volume_sample(volume, p, wp.Volume.LINEAR, dtype=wp.float32), expected, 2.0e-4)


@wp.kernel
def test_volume_sample_grad_linear_f(volume: wp.uint64, points: wp.array[wp.vec3]):
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

    val = wp.volume_sample_grad(volume, p, wp.Volume.LINEAR, grad, dtype=wp.float32)

    expect_near(val, expected_val, 2.0e-4)
    expect_near(grad[0], expected_gx, 2.0e-4)
    expect_near(grad[1], expected_gy, 2.0e-4)
    expect_near(grad[2], expected_gz, 2.0e-4)


@wp.kernel
def test_volume_sample_local_f_linear_values(
    volume: wp.uint64, points: wp.array[wp.vec3], values: wp.array[wp.float32]
):
    tid = wp.tid()
    p = points[tid]
    values[tid] = wp.volume_sample_f(volume, p, wp.Volume.LINEAR)


@wp.kernel
def test_volume_sample_grad_local_f_linear_values(
    volume: wp.uint64, points: wp.array[wp.vec3], values: wp.array[wp.float32], case_num: int
):
    tid = wp.tid()
    p = points[tid]

    grad = wp.vec3(0.0, 0.0, 0.0)
    val = wp.volume_sample_grad_f(volume, p, wp.Volume.LINEAR, grad)
    if case_num == 0:
        values[tid] = val
    elif case_num == 1:
        values[tid] = grad[0]
    elif case_num == 2:
        values[tid] = grad[1]
    elif case_num == 3:
        values[tid] = grad[2]


@wp.kernel
def test_volume_sample_world_f_linear_values(
    volume: wp.uint64, points: wp.array[wp.vec3], values: wp.array[wp.float32]
):
    tid = wp.tid()
    q = points[tid]
    p = wp.volume_world_to_index(volume, q)
    values[tid] = wp.volume_sample_f(volume, p, wp.Volume.LINEAR)


@wp.kernel
def test_volume_sample_grad_world_f_linear_values(
    volume: wp.uint64, points: wp.array[wp.vec3], values: wp.array[wp.float32], case_num: int
):
    tid = wp.tid()
    q = points[tid]
    p = wp.volume_world_to_index(volume, q)

    grad = wp.vec3(0.0, 0.0, 0.0)
    val = wp.volume_sample_grad_f(volume, p, wp.Volume.LINEAR, grad)
    if case_num == 0:
        values[tid] = val
    elif case_num == 1:
        values[tid] = grad[0]
    elif case_num == 2:
        values[tid] = grad[1]
    elif case_num == 3:
        values[tid] = grad[2]


# vec3f volume tests
@wp.kernel
def test_volume_lookup_v(volume: wp.uint64, points: wp.array[wp.vec3]):
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
    expect_eq(wp.volume_lookup(volume, i, j, k, dtype=wp.vec3), expected)


@wp.kernel
def test_volume_sample_closest_v(volume: wp.uint64, points: wp.array[wp.vec3]):
    tid = wp.tid()

    p = points[tid]
    i = round(p[0])
    j = round(p[1])
    k = round(p[2])
    expected = wp.vec3(i + 2.0 * j + 3.0 * k, 4.0 * i + 5.0 * j + 6.0 * k, 7.0 * i + 8.0 * j + 9.0 * k)
    if abs(i) > 10.0 or abs(j) > 10.0 or abs(k) > 10.0:
        expected = wp.vec3(10.8, -4.13, 10.26)

    expect_eq(wp.volume_sample_v(volume, p, wp.Volume.CLOSEST), expected)
    expect_eq(wp.volume_sample(volume, p, wp.Volume.CLOSEST, dtype=wp.vec3), expected)

    q = wp.volume_index_to_world(volume, p)
    q_inv = wp.volume_world_to_index(volume, q)
    expect_eq(p, q_inv)


@wp.kernel
def test_volume_sample_linear_v(volume: wp.uint64, points: wp.array[wp.vec3]):
    tid = wp.tid()

    p = points[tid]

    expected = wp.vec3(
        p[0] + 2.0 * p[1] + 3.0 * p[2], 4.0 * p[0] + 5.0 * p[1] + 6.0 * p[2], 7.0 * p[0] + 8.0 * p[1] + 9.0 * p[2]
    )
    if abs(p[0]) > 10.0 or abs(p[1]) > 10.0 or abs(p[2]) > 10.0:
        return  # not testing against background values

    expect_near(wp.volume_sample_v(volume, p, wp.Volume.LINEAR), expected, 2.0e-4)
    expect_near(wp.volume_sample(volume, p, wp.Volume.LINEAR, dtype=wp.vec3), expected, 2.0e-4)


@wp.kernel
def test_volume_sample_grad_linear_v(volume: wp.uint64, points: wp.array[wp.vec3]):
    tid = wp.tid()

    p = points[tid]

    if abs(p[0]) > 10.0 or abs(p[1]) > 10.0 or abs(p[2]) > 10.0:
        return  # not testing against background values

    expected_val = wp.vec3(
        p[0] + 2.0 * p[1] + 3.0 * p[2], 4.0 * p[0] + 5.0 * p[1] + 6.0 * p[2], 7.0 * p[0] + 8.0 * p[1] + 9.0 * p[2]
    )
    expected_grad = wp.mat33(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)

    grad = wp.mat33(0.0)
    val = wp.volume_sample_grad(volume, p, wp.Volume.LINEAR, grad, dtype=wp.vec3)

    expect_near(val, expected_val, 2.0e-4)
    expect_near(grad[0], expected_grad[0], 2.0e-4)
    expect_near(grad[1], expected_grad[1], 2.0e-4)
    expect_near(grad[2], expected_grad[2], 2.0e-4)


@wp.kernel
def test_volume_sample_local_v_linear_values(
    volume: wp.uint64, points: wp.array[wp.vec3], values: wp.array[wp.float32]
):
    tid = wp.tid()
    p = points[tid]
    ones = wp.vec3(1.0, 1.0, 1.0)
    values[tid] = wp.dot(wp.volume_sample_v(volume, p, wp.Volume.LINEAR), ones)


@wp.kernel
def test_volume_sample_world_v_linear_values(
    volume: wp.uint64, points: wp.array[wp.vec3], values: wp.array[wp.float32]
):
    tid = wp.tid()
    q = points[tid]
    p = wp.volume_world_to_index(volume, q)
    ones = wp.vec3(1.0, 1.0, 1.0)
    values[tid] = wp.dot(wp.volume_sample_v(volume, p, wp.Volume.LINEAR), ones)


# int32 volume tests
@wp.kernel
def test_volume_lookup_i(volume: wp.uint64, points: wp.array[wp.vec3]):
    tid = wp.tid()

    p = points[tid]
    i = int(p[0])
    j = int(p[1])
    k = int(p[2])
    expected = i * j * k
    if abs(i) > 10 or abs(j) > 10 or abs(k) > 10:
        expected = 10

    expect_eq(wp.volume_lookup_i(volume, i, j, k), expected)
    expect_eq(wp.volume_lookup(volume, i, j, k, dtype=wp.int32), expected)


@wp.kernel
def test_volume_sample_i(volume: wp.uint64, points: wp.array[wp.vec3]):
    tid = wp.tid()

    p = points[tid]
    i = round(p[0])
    j = round(p[1])
    k = round(p[2])
    expected = int(i * j * k)
    if abs(i) > 10.0 or abs(j) > 10.0 or abs(k) > 10.0:
        expected = 10

    expect_eq(wp.volume_sample_i(volume, p), expected)
    expect_eq(wp.volume_sample(volume, p, wp.Volume.CLOSEST, dtype=wp.int32), expected)

    q = wp.volume_index_to_world(volume, p)
    q_inv = wp.volume_world_to_index(volume, q)
    expect_eq(p, q_inv)


# Index/world transformation tests
@wp.kernel
def test_volume_index_to_world(
    volume: wp.uint64,
    points: wp.array[wp.vec3],
    values: wp.array[wp.float32],
    grad_values: wp.array[wp.vec3],
):
    tid = wp.tid()
    p = points[tid]
    ones = wp.vec3(1.0, 1.0, 1.0)
    values[tid] = wp.dot(wp.volume_index_to_world(volume, p), ones)
    grad_values[tid] = wp.volume_index_to_world_dir(volume, ones)


@wp.kernel
def test_volume_world_to_index(
    volume: wp.uint64,
    points: wp.array[wp.vec3],
    values: wp.array[wp.float32],
    grad_values: wp.array[wp.vec3],
):
    tid = wp.tid()
    p = points[tid]
    ones = wp.vec3(1.0, 1.0, 1.0)
    values[tid] = wp.dot(wp.volume_world_to_index(volume, p), ones)
    grad_values[tid] = wp.volume_world_to_index_dir(volume, ones)


# Volume write tests
@wp.kernel
def test_volume_store_f(volume: wp.uint64, points: wp.array[wp.vec3], values: wp.array[wp.float32]):
    tid = wp.tid()

    p = points[tid]
    i = int(p[0])
    j = int(p[1])
    k = int(p[2])

    wp.volume_store(volume, i, j, k, float(i + 100 * j + 10000 * k))
    values[tid] = wp.volume_lookup_f(volume, i, j, k)


@wp.kernel
def test_volume_store_v(volume: wp.uint64, points: wp.array[wp.vec3], values: wp.array[wp.vec3]):
    tid = wp.tid()

    p = points[tid]
    i = int(p[0])
    j = int(p[1])
    k = int(p[2])

    wp.volume_store(volume, i, j, k, p)
    values[tid] = wp.volume_lookup_v(volume, i, j, k)


@wp.kernel
def test_volume_store_i(volume: wp.uint64, points: wp.array[wp.vec3], values: wp.array[wp.int32]):
    tid = wp.tid()

    p = points[tid]
    i = int(p[0])
    j = int(p[1])
    k = int(p[2])

    wp.volume_store(volume, i, j, k, i + 100 * j + 10000 * k)
    values[tid] = wp.volume_lookup_i(volume, i, j, k)


@wp.kernel
def test_volume_store_v4(volume: wp.uint64, points: wp.array[wp.vec3], values: wp.array[wp.vec4]):
    tid = wp.tid()

    p = points[tid]
    i = int(p[0])
    j = int(p[1])
    k = int(p[2])

    v = wp.vec4(p[0], p[1], p[2], float(i + 100 * j + 10000 * k))

    wp.volume_store(volume, i, j, k, v)

    values[tid] = wp.volume_lookup(volume, i, j, k, dtype=wp.vec4)


@wp.kernel
def test_volume_store_v4d(volume: wp.uint64, points: wp.array[wp.vec3], values: wp.array[wp.vec4d]):
    tid = wp.tid()

    p = points[tid]
    i = int(p[0])
    j = int(p[1])
    k = int(p[2])

    v = wp.vec4d(wp.float64(p[0]), wp.float64(p[1]), wp.float64(p[2]), wp.float64(i + 100 * j + 10000 * k))

    wp.volume_store(volume, i, j, k, v)

    values[tid] = wp.volume_lookup(volume, i, j, k, dtype=wp.vec4d)


devices = get_test_devices()

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
    "float": os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "test_grid.nvdb")),
    "int32": os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "test_int32_grid.nvdb")),
    "vec3f": os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "test_vec_grid.nvdb")),
    "index": os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "test_index_grid.nvdb")),
    "torus": os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "torus.nvdb")),
    "float_write": os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "test_grid.nvdb")),
}

test_volume_tiles = (
    np.array([[i, j, k] for i in range(-2, 2) for j in range(-2, 2) for k in range(-2, 2)], dtype=np.int32) * 8
)

axis = np.linspace(-1, 1, 3)
point_grid = np.array([[x, y, z] for x in axis for y in axis for z in axis], dtype=np.float32)

_volume_cache = {}
_point_cache = {}
_jittered_point_cache = {}


def _get_volume(value_type, device):
    device = wp.get_device(device)
    key = (value_type, device.alias)
    volume = _volume_cache.get(key)
    if volume is not None:
        return volume

    path = volume_paths[value_type]
    with open(path, "rb") as stream:
        volume_data = stream.read()

    try:
        volume = wp.Volume.load_from_nvdb(volume_data, device)
    except RuntimeError as error:
        raise RuntimeError(f'Failed to load volume from "{path}" to {device} memory:\n{error}') from error

    wp.synchronize_device(device)
    _volume_cache[key] = volume
    return volume


def _get_points(device, jittered=False):
    device = wp.get_device(device)
    cache = _jittered_point_cache if jittered else _point_cache
    points = cache.get(device.alias)
    if points is not None:
        return points

    point_data = point_grid
    if jittered:
        rng = np.random.default_rng(101215)
        point_data = point_grid + rng.uniform(-0.5, 0.5, size=point_grid.shape)

    points = wp.array(point_data, dtype=wp.vec3, device=device)
    cache[device.alias] = points
    return points


def _volume_kernel_inputs(value_type, jittered=False):
    def inputs_factory(device):
        return [
            _get_volume(value_type, device).id,
            _get_points(device, jittered=jittered),
        ]

    return inputs_factory


def test_volume_sample_linear_f_gradient(test, device):
    rng = np.random.default_rng(101215)
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
                inputs=[_get_volume("float", device).id, uvws, values],
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
                inputs=[_get_volume("float", device).id, xyzs, values],
                device=device,
            )
        tape.backward(values)

        x, y, z = test_case
        grad_expected = np.array([y * z, x * z, x * y]) / 0.25
        grad_computed = tape.gradients[xyzs].numpy()[0]
        np.testing.assert_allclose(grad_computed, grad_expected, rtol=1e-4)


def test_volume_sample_grad_linear_f_gradient(test, device):
    rng = np.random.default_rng(101215)
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
                    inputs=[_get_volume("float", device).id, uvws, values, case_num],
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
                    inputs=[_get_volume("float", device).id, xyzs, values, case_num],
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
    rng = np.random.default_rng(101215)
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
                inputs=[_get_volume("vec3f", device).id, uvws, values],
                device=device,
            )
        tape.backward(values)

        grad_expected = np.array([12.0, 15.0, 18.0])
        grad_computed = tape.gradients[uvws].numpy()[0]
        np.testing.assert_allclose(grad_computed, grad_expected, rtol=1e-4)

        tape = wp.Tape()
        with tape:
            wp.launch(
                test_volume_sample_world_v_linear_values,
                dim=1,
                inputs=[_get_volume("vec3f", device).id, xyzs, values],
                device=device,
            )
        tape.backward(values)

        grad_expected = np.array([12.0, 15.0, 18.0]) / 0.25
        grad_computed = tape.gradients[xyzs].numpy()[0]
        np.testing.assert_allclose(grad_computed, grad_expected, rtol=1e-4)


def test_volume_transform_gradient(test, device):
    values = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
    grad_values = wp.zeros(1, dtype=wp.vec3, device=device)
    rng = np.random.default_rng(101215)
    test_points = rng.uniform(-10.0, 10.0, size=(10, 3))
    for test_case in test_points:
        points = wp.array(test_case, dtype=wp.vec3, device=device, requires_grad=True)
        tape = wp.Tape()
        with tape:
            wp.launch(
                test_volume_index_to_world,
                dim=1,
                inputs=[_get_volume("torus", device).id, points, values, grad_values],
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
        inputs=[_get_volume("float_write", device).id, points, values],
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


def test_volume_allocation_v4(test, device):
    bg_value = (-1, 2.0, -3, 5)
    points_np = np.append(point_grid, [[8096, 8096, 8096]], axis=0)

    w_ref = np.array([x + 100 * y + 10000 * z for x, y, z in point_grid])[:, np.newaxis]
    values_ref = np.append(np.hstack((point_grid, w_ref)), [bg_value], axis=0)

    volume = wp.Volume.allocate(min=[-11, -11, -11], max=[11, 11, 11], voxel_size=0.1, bg_value=bg_value, device=device)
    points = wp.array(points_np, dtype=wp.vec3, device=device)
    values = wp.empty(len(points_np), dtype=wp.vec4, device=device)
    wp.launch(test_volume_store_v4, dim=len(points_np), inputs=[volume.id, points, values], device=device)

    values_res = values.numpy()
    np.testing.assert_equal(values_res, values_ref)


def test_volume_allocation_v4d(test, device):
    bg_value = wp.vec4d(-1.0, 2.0, -3.0, 5.0)
    points_np = np.append(point_grid, [[8096, 8096, 8096]], axis=0)

    w_ref = np.array([x + 100 * y + 10000 * z for x, y, z in point_grid])[:, np.newaxis]
    values_ref = np.append(np.hstack((point_grid, w_ref)), [bg_value], axis=0)

    volume = wp.Volume.allocate(min=[-11, -11, -11], max=[11, 11, 11], voxel_size=0.1, bg_value=bg_value, device=device)
    test.assertIs(volume.dtype, wp.vec4d)
    test.assertEqual(volume.get_grid_info().type_str, "Vec4d")

    points = wp.array(points_np, dtype=wp.vec3, device=device)
    values = wp.empty(len(points_np), dtype=wp.vec4d, device=device)
    wp.launch(test_volume_store_v4d, dim=len(points_np), inputs=[volume.id, points, values], device=device)

    values_res = values.numpy()
    np.testing.assert_equal(values_res, values_ref)


def test_volume_introspection(test, device):
    for volume_names in ("float", "vec3f"):
        with test.subTest(volume_names=volume_names):
            volume = _get_volume(volume_names, device)
            tiles_actual = volume.get_tiles().numpy()
            tiles_sorted = tiles_actual[np.lexsort(tiles_actual.T[::-1])]
            voxel_size = np.array(volume.get_voxel_size())

            np.testing.assert_equal(test_volume_tiles, tiles_sorted)
            np.testing.assert_equal([0.25] * 3, voxel_size)

            voxel_count = volume.get_voxel_count()
            voxels_actual = volume.get_voxels().numpy()
            assert voxel_count == voxels_actual.shape[0]

            # Voxel coordinates should be unique
            voxels_unique = np.unique(voxels_actual, axis=0)
            assert voxel_count == voxels_unique.shape[0]

            # Get back tiles from voxels, should match get_tiles()
            voxel_tiles = 8 * (voxels_unique // 8)
            voxel_tiles_sorted = voxel_tiles[np.lexsort(voxel_tiles.T[::-1])]
            voxel_tiles_unique = np.unique(voxel_tiles_sorted, axis=0)

            np.testing.assert_equal(voxel_tiles_unique, tiles_sorted)


def test_volume_multiple_grids(test, device):
    volume = _get_volume("index", device)

    volume_2 = volume.load_next_grid()

    test.assertIsNotNone(volume_2)

    test.assertNotEqual(volume.id, volume_2.id)
    test.assertNotEqual(volume.get_voxel_count(), volume_2.get_voxel_count())

    test.assertEqual(volume.get_grid_info().grid_count, volume_2.get_grid_info().grid_count)
    test.assertEqual(volume.get_grid_info().grid_index + 1, volume_2.get_grid_info().grid_index)

    volume_3 = volume_2.load_next_grid()
    test.assertIsNone(volume_3)


def test_volume_feature_array(test, device):
    volume = _get_volume("index", device)

    test.assertEqual(volume.get_feature_array_count(), 1)

    array = volume.feature_array(0, dtype=wp.uint64)
    test.assertEqual(array.device, device)
    test.assertEqual(array.dtype, wp.uint64)

    # fVDB convention, data starts with array ndim + shape
    np.testing.assert_equal(array.numpy()[0:4], [3, volume.get_voxel_count(), 2, 3])


@wp.kernel
def fill_leaf_values_kernel(volume: wp.uint64, ijk: wp.array2d[wp.int32], values: wp.array[Any]):
    tid = wp.tid()

    i = ijk[tid, 0]
    j = ijk[tid, 1]
    k = ijk[tid, 2]

    expect_eq(tid, wp.volume_lookup_index(volume, i, j, k))

    values[tid] = wp.volume_lookup(volume, i, j, k, dtype=values.dtype)


@wp.kernel
def test_volume_sample_index_kernel(
    volume: wp.uint64,
    points: wp.array[wp.vec3],
    values: wp.array[Any],
    background: wp.array[Any],
    sampled_values: wp.array[Any],
):
    tid = wp.tid()
    p = points[tid]

    ref = wp.volume_sample(volume, p, wp.Volume.LINEAR, dtype=values.dtype)
    sampled_values[tid] = wp.volume_sample_index(volume, p, wp.Volume.LINEAR, values, background[0])
    expect_eq(sampled_values[tid], ref)


@wp.kernel
def test_volume_sample_grad_index_kernel(
    volume: wp.uint64,
    points: wp.array[wp.vec3],
    values: wp.array[Any],
    background: wp.array[Any],
    sampled_values: wp.array[Any],
    sampled_grads: wp.array[Any],
):
    tid = wp.tid()
    p = points[tid]

    ref_grad = sampled_grads.dtype()
    ref = wp.volume_sample_grad(volume, p, wp.Volume.LINEAR, ref_grad, dtype=values.dtype)

    grad = type(ref_grad)()
    sampled_values[tid] = wp.volume_sample_grad_index(volume, p, wp.Volume.LINEAR, values, background[0], grad)
    expect_eq(sampled_values[tid], ref)

    expect_eq(grad[0], ref_grad[0])
    expect_eq(grad[1], ref_grad[1])
    expect_eq(grad[2], ref_grad[2])
    sampled_grads[tid] = grad


def test_volume_sample_index(test, device):
    rng = np.random.default_rng(101215)
    points = rng.uniform(-10.0, 10.0, size=(100, 3))
    points[0:10, 0] += 100.0  # ensure some points are over unallocated voxels
    uvws = wp.array(points, dtype=wp.vec3, device=device)

    bg_values = {
        "float": 10.0,
        "vec3f": wp.vec3(10.8, -4.13, 10.26),
    }
    grad_types = {
        "float": wp.vec3,
        "vec3f": wp.mat33,
    }

    for volume_names in ("float", "vec3f"):
        with test.subTest(volume_names=volume_names):
            volume = _get_volume(volume_names, device)

            ijk = volume.get_voxels()

            values = wp.empty(shape=volume.get_voxel_count(), dtype=volume.dtype, device=device, requires_grad=True)

            vid = wp.uint64(volume.id)
            wp.launch(fill_leaf_values_kernel, dim=values.shape, inputs=[vid, ijk, values], device=device)

            sampled_values = wp.empty(shape=points.shape[0], dtype=volume.dtype, device=device, requires_grad=True)
            background = wp.array([bg_values[volume_names]], dtype=volume.dtype, device=device, requires_grad=True)

            tape = wp.Tape()
            with tape:
                wp.launch(
                    test_volume_sample_index_kernel,
                    dim=points.shape[0],
                    inputs=[vid, uvws, values, background, sampled_values],
                    device=device,
                )

            sampled_values.grad.fill_(1.0)
            tape.backward()

            # test adjoint w.r.t voxel and background value arrays
            # we should have sum(sampled_values) = sum(adj_values * values) + (adj_background * background)
            sum_sampled_values = np.sum(sampled_values.numpy(), axis=0)
            sum_values_adj = np.sum(values.numpy() * values.grad.numpy(), axis=0)
            sum_background_adj = background.numpy()[0] * background.grad.numpy()[0]

            np.testing.assert_allclose(sum_sampled_values, sum_values_adj + sum_background_adj, rtol=1.0e-3)

            tape.reset()

            sampled_grads = wp.empty(
                shape=points.shape[0], dtype=grad_types[volume_names], device=device, requires_grad=True
            )

            with tape:
                wp.launch(
                    test_volume_sample_grad_index_kernel,
                    dim=points.shape[0],
                    inputs=[vid, uvws, values, background, sampled_values, sampled_grads],
                    device=device,
                )

            sampled_values.grad.fill_(1.0)
            tape.backward()

            # we should have sum(sampled_values) = sum(adj_values * values) + (adj_background * background)
            sum_sampled_values = np.sum(sampled_values.numpy(), axis=0)
            sum_values_adj = np.sum(values.numpy() * values.grad.numpy(), axis=0)
            sum_background_adj = background.numpy()[0] * background.grad.numpy()[0]
            np.testing.assert_allclose(sum_sampled_values, sum_values_adj + sum_background_adj, rtol=1.0e-3)

            tape.zero()
            sampled_values.grad.fill_(0.0)
            sampled_grads.grad.fill_(1.0)
            tape.backward()

            # we should have sum(sampled_grad, axes=(0, -1)) = sum(adj_values * values) + (adj_background * background)
            sum_sampled_grads = np.sum(np.sum(sampled_grads.numpy(), axis=0), axis=-1)
            sum_values_adj = np.sum(values.numpy() * values.grad.numpy(), axis=0)
            sum_background_adj = background.numpy()[0] * background.grad.numpy()[0]
            np.testing.assert_allclose(sum_sampled_grads, sum_values_adj + sum_background_adj, rtol=1.0e-3)


# vec4 volume gradient tests
#
# The sampled field is v(x) = A x with a 4-by-3 A, so the trilinear interpolant reproduces it
# exactly and the Jacobian is A at every point. Row i of A is d(v_i)/d(uvw).
_VEC4_FIELD_JACOBIAN = np.array(
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype=np.float64
)

# A dense 16-by-16-by-16 voxel block spanning [-8, 7] on each axis.
_vec4_field_tiles = np.array([[i, j, k] for i in (-8, 0) for j in (-8, 0) for k in (-8, 0)], dtype=np.int32)


def _vec4_field_values(ijk):
    """Evaluate the linear vec4 field ``v(x) = A x`` at the given index-space coordinates."""
    return np.asarray(ijk, dtype=np.float64) @ _VEC4_FIELD_JACOBIAN.T


def _vec4_field_points(device):
    """Draw index-space sample points for the vec4 gradient tests.

    Points stay well inside the allocated block so that every trilinear corner is active, and close
    to the origin so that the interpolated magnitudes -- and with them the single-precision
    interpolation error -- stay small.
    """
    rng = np.random.default_rng(101215)
    points = rng.uniform(-2.0, 2.0, size=(64, 3))
    return points, wp.array(points, dtype=wp.vec3, device=device)


def _vec4_field_tolerance(points):
    """Return the absolute tolerance for values and gradients sampled at the given points.

    The trilinear weights are evaluated in single precision whatever the volume's scalar type, so
    the error of both the value and the Jacobian scales with the magnitude of the interpolated
    values rather than with the scalar type's own precision.
    """
    return 32.0 * np.finfo(np.float32).eps * np.abs(_vec4_field_values(points)).max()


def _vec4_grad_type(dtype):
    """Return the Jacobian type that ``volume_sample_grad`` expects for the given vec4 type."""
    return wp.types.matrix(shape=(4, 3), dtype=dtype._wp_scalar_type_)


@wp.kernel
def volume_store_values_kernel(volume: wp.uint64, ijk: wp.array2d[wp.int32], values: wp.array[Any]):
    tid = wp.tid()

    wp.volume_store(volume, ijk[tid, 0], ijk[tid, 1], ijk[tid, 2], values[tid])


@wp.kernel
def volume_sample_grad_v4_kernel(
    volume: wp.uint64,
    points: wp.array[wp.vec3],
    values: wp.array[Any],
    grads: wp.array[Any],
):
    tid = wp.tid()

    grad = grads.dtype()
    values[tid] = wp.volume_sample_grad(volume, points[tid], wp.Volume.LINEAR, grad, dtype=values.dtype)
    grads[tid] = grad


@wp.kernel
def volume_sample_grad_index_v4_kernel(
    volume: wp.uint64,
    points: wp.array[wp.vec3],
    voxel_data: wp.array[Any],
    background: wp.array[Any],
    values: wp.array[Any],
    grads: wp.array[Any],
):
    tid = wp.tid()

    grad = grads.dtype()
    values[tid] = wp.volume_sample_grad_index(volume, points[tid], wp.Volume.LINEAR, voxel_data, background[0], grad)
    grads[tid] = grad


def _allocate_vec4_field_volume(dtype, device):
    """Allocate a vec4 volume of the given type holding the linear field ``v(x) = A x``."""
    tile_points = wp.array(_vec4_field_tiles, dtype=wp.int32, device=device)
    volume = wp.Volume.allocate_by_tiles(tile_points, voxel_size=1.0, bg_value=dtype(0.0), device=device)

    ijk = volume.get_voxels()
    values = wp.array(_vec4_field_values(ijk.numpy()), dtype=dtype, device=device)
    wp.launch(
        volume_store_values_kernel,
        dim=volume.get_voxel_count(),
        inputs=[wp.uint64(volume.id), ijk, values],
        device=device,
    )
    return volume


def test_volume_sample_grad_v4(test, device):
    """Sample vec4 volumes and their 4-by-3 Jacobians."""
    points_np, points = _vec4_field_points(device)
    tolerance = _vec4_field_tolerance(points_np)

    expected_values = _vec4_field_values(points_np)
    expected_grads = np.broadcast_to(_VEC4_FIELD_JACOBIAN, (points_np.shape[0], 4, 3))

    for dtype in (wp.vec4f, wp.vec4d):
        with test.subTest(dtype=wp.types.type_repr(dtype)):
            volume = _allocate_vec4_field_volume(dtype, device)

            values = wp.empty(shape=points_np.shape[0], dtype=dtype, device=device)
            grads = wp.empty(shape=points_np.shape[0], dtype=_vec4_grad_type(dtype), device=device)
            wp.launch(
                volume_sample_grad_v4_kernel,
                dim=points_np.shape[0],
                inputs=[wp.uint64(volume.id), points, values, grads],
                device=device,
            )

            np.testing.assert_allclose(values.numpy(), expected_values, rtol=0.0, atol=tolerance)
            np.testing.assert_allclose(grads.numpy(), expected_grads, rtol=0.0, atol=tolerance)


def test_volume_sample_grad_index_v4(test, device):
    """Sample vec4 voxel data and its 4-by-3 Jacobian through an index volume."""
    points_np, points = _vec4_field_points(device)
    tolerance = _vec4_field_tolerance(points_np)

    expected_values = _vec4_field_values(points_np)
    expected_grads = np.broadcast_to(_VEC4_FIELD_JACOBIAN, (points_np.shape[0], 4, 3))

    tile_points = wp.array(_vec4_field_tiles, dtype=wp.int32, device=device)
    volume = wp.Volume.allocate_by_tiles(tile_points, voxel_size=1.0, bg_value=None, device=device)
    ijk = volume.get_voxels().numpy()

    for dtype in (wp.vec4f, wp.vec4d):
        with test.subTest(dtype=wp.types.type_repr(dtype)):
            voxel_data = wp.array(_vec4_field_values(ijk), dtype=dtype, device=device, requires_grad=True)
            background = wp.array([dtype(-13.0)], dtype=dtype, device=device, requires_grad=True)

            values = wp.empty(shape=points_np.shape[0], dtype=dtype, device=device, requires_grad=True)
            grads = wp.empty(shape=points_np.shape[0], dtype=_vec4_grad_type(dtype), device=device, requires_grad=True)

            tape = wp.Tape()
            with tape:
                wp.launch(
                    volume_sample_grad_index_v4_kernel,
                    dim=points_np.shape[0],
                    inputs=[wp.uint64(volume.id), points, voxel_data, background, values, grads],
                    device=device,
                )

            np.testing.assert_allclose(values.numpy(), expected_values, rtol=0.0, atol=tolerance)
            np.testing.assert_allclose(grads.numpy(), expected_grads, rtol=0.0, atol=tolerance)

            values.grad.fill_(1.0)
            tape.backward()

            # we should have sum(values) = sum(adj_voxel_data * voxel_data) + (adj_background * background)
            sum_values = np.sum(values.numpy(), axis=0)
            sum_voxel_data_adj = np.sum(voxel_data.numpy() * voxel_data.grad.numpy(), axis=0)
            sum_background_adj = background.numpy()[0] * background.grad.numpy()[0]
            np.testing.assert_allclose(sum_values, sum_voxel_data_adj + sum_background_adj, rtol=1.0e-3)

            tape.zero()
            values.grad.fill_(0.0)
            grads.grad.fill_(1.0)
            tape.backward()

            # we should have sum(grads, axes=(0, -1)) = sum(adj_voxel_data * voxel_data) + (adj_background * background)
            sum_grads = np.sum(np.sum(grads.numpy(), axis=0), axis=-1)
            sum_voxel_data_adj = np.sum(voxel_data.numpy() * voxel_data.grad.numpy(), axis=0)
            sum_background_adj = background.numpy()[0] * background.grad.numpy()[0]
            np.testing.assert_allclose(sum_grads, sum_voxel_data_adj + sum_background_adj, rtol=1.0e-3)


def test_volume_sample_grad_index_v4_adjoint(test, device):
    """Differentiate a vec4 sample and its Jacobian with respect to the sample point."""
    rng = np.random.default_rng(101215)

    tile_points = wp.array(_vec4_field_tiles, dtype=wp.int32, device=device)
    volume = wp.Volume.allocate_by_tiles(tile_points, voxel_size=1.0, bg_value=None, device=device)
    voxel_data_np = rng.uniform(-1.0, 1.0, size=(volume.get_voxel_count(), 4))

    # Keep every sample point and its shifted copies inside one trilinear cell: the interpolant is
    # only piecewise smooth, and its Jacobian jumps across integer voxel planes.
    step = 0.1
    points_np = rng.integers(-4, 4, size=(64, 3)) + rng.uniform(0.25, 0.75, size=(64, 3))

    def sample(dtype, points_np, requires_grad):
        points = wp.array(points_np, dtype=wp.vec3, device=device, requires_grad=requires_grad)
        voxel_data = wp.array(voxel_data_np, dtype=dtype, device=device, requires_grad=True)
        background = wp.array([dtype(0.0)], dtype=dtype, device=device, requires_grad=True)
        values = wp.empty(shape=points_np.shape[0], dtype=dtype, device=device, requires_grad=requires_grad)
        grads = wp.empty(
            shape=points_np.shape[0], dtype=_vec4_grad_type(dtype), device=device, requires_grad=requires_grad
        )

        tape = wp.Tape()
        with tape:
            wp.launch(
                volume_sample_grad_index_v4_kernel,
                dim=points_np.shape[0],
                inputs=[wp.uint64(volume.id), points, voxel_data, background, values, grads],
                device=device,
            )
        return tape, points, values, grads

    def central_difference(dtype, reduce):
        """Differentiate the reduction of a forward quantity along each index-space axis."""
        derivative = np.empty((points_np.shape[0], 3))
        for axis in range(3):
            forward = points_np.copy()
            forward[:, axis] += step
            backward = points_np.copy()
            backward[:, axis] -= step
            plus = sample(dtype, forward, False)
            minus = sample(dtype, backward, False)
            derivative[:, axis] = (reduce(plus) - reduce(minus)) / (2.0 * step)
        return derivative

    for dtype in (wp.vec4f, wp.vec4d):
        with test.subTest(dtype=wp.types.type_repr(dtype)):
            tape, points, values, grads = sample(dtype, points_np, True)

            values.grad.fill_(1.0)
            tape.backward()
            adj_value_uvw = points.grad.numpy().copy()

            tape.zero()
            points.grad.zero_()
            values.grad.fill_(0.0)
            grads.grad.fill_(1.0)
            tape.backward()
            adj_grad_uvw = points.grad.numpy().copy()

            # Within a cell the interpolant is linear along each axis, so a central difference is
            # exact and only carries the rounding error of the single-precision weights.
            eps = np.finfo(np.float32).eps
            value_scale = np.abs(np.sum(values.numpy(), axis=-1)).max()
            grad_scale = np.abs(np.sum(grads.numpy(), axis=(-2, -1))).max()

            np.testing.assert_allclose(
                adj_value_uvw,
                central_difference(dtype, lambda s: np.sum(s[2].numpy(), axis=-1)),
                rtol=0.0,
                atol=32.0 * eps * value_scale / step,
            )
            np.testing.assert_allclose(
                adj_grad_uvw,
                central_difference(dtype, lambda s: np.sum(s[3].numpy(), axis=(-2, -1))),
                rtol=0.0,
                atol=32.0 * eps * grad_scale / step,
            )


def test_volume_sample_grad_rejects_transposed_grad(test, device):
    """Reject a Jacobian argument that stores one column, rather than one row, per value component."""
    mat34f = wp.types.matrix(shape=(3, 4), dtype=wp.float32)

    @wp.kernel(module="unique")
    def kernel(volume: wp.uint64, points: wp.array[wp.vec3], values: wp.array[wp.vec4f]):
        grad = mat34f()
        values[0] = wp.volume_sample_grad(volume, points[0], wp.Volume.LINEAR, grad, dtype=wp.vec4f)

    points = wp.zeros(1, dtype=wp.vec3, device=device)
    values = wp.zeros(1, dtype=wp.vec4f, device=device)

    with test.assertRaisesRegex(RuntimeError, r"Incompatible gradient type, expected mat43f, got mat34f"):
        wp.launch(kernel, dim=1, inputs=[wp.uint64(0), points, values], device=device)


def test_volume_sample_index_equivalent_dtype(test, device):
    """Sample voxel data whose type is structurally equal to a supported one but a distinct object."""
    vec4d_alias = wp.types.vector(length=4, dtype=wp.float64)
    mat43d = wp.types.matrix(shape=(4, 3), dtype=wp.float64)
    test.assertIsNot(vec4d_alias, wp.vec4d)

    @wp.kernel(module="unique")
    def kernel(
        volume: wp.uint64,
        voxel_data: wp.array[vec4d_alias],
        background: wp.array[vec4d_alias],
        sampled: wp.array[vec4d_alias],
        with_grad: wp.array[vec4d_alias],
        grads: wp.array[mat43d],
    ):
        uvw = wp.vec3(1.5, 1.5, 1.5)
        sampled[0] = wp.volume_sample_index(volume, uvw, wp.Volume.LINEAR, voxel_data, background[0])

        grad = mat43d()
        with_grad[0] = wp.volume_sample_grad_index(volume, uvw, wp.Volume.LINEAR, voxel_data, background[0], grad)
        grads[0] = grad

    tile_points = wp.array(_vec4_field_tiles, dtype=wp.int32, device=device)
    volume = wp.Volume.allocate_by_tiles(tile_points, voxel_size=1.0, bg_value=None, device=device)

    constant = (1.0, -2.0, 3.0, -4.0)
    voxel_data = wp.array(np.tile(constant, (volume.get_voxel_count(), 1)), dtype=vec4d_alias, device=device)
    background = wp.array([vec4d_alias()], dtype=vec4d_alias, device=device)
    sampled = wp.zeros(1, dtype=vec4d_alias, device=device)
    with_grad = wp.zeros(1, dtype=vec4d_alias, device=device)
    grads = wp.zeros(1, dtype=mat43d, device=device)

    wp.launch(
        kernel,
        dim=1,
        inputs=[wp.uint64(volume.id), voxel_data, background, sampled, with_grad, grads],
        device=device,
    )

    # A constant field interpolates to itself and has a zero Jacobian.
    np.testing.assert_equal(sampled.numpy()[0], np.array(constant))
    np.testing.assert_equal(with_grad.numpy()[0], np.array(constant))
    np.testing.assert_equal(grads.numpy()[0], np.zeros((4, 3)))


def test_volume_allocation_by_tiles_v4d(test, device):
    """Allocate a vec4d volume through the tile builder, which is supported on CPU as well as CUDA."""
    bg_value = wp.vec4d(-1.0, 2.0, -3.0, 5.0)
    tile_points = wp.array(np.array([[0, 0, 0]], dtype=np.int32), dtype=wp.int32, device=device)

    volume = wp.Volume.allocate_by_tiles(tile_points, voxel_size=0.1, bg_value=bg_value, device=device)
    test.assertIs(volume.dtype, wp.vec4d)
    test.assertEqual(volume.get_grid_info().type_str, "Vec4d")

    # One voxel inside the allocated tile, one far outside it so that the background is read back.
    points_np = np.array([[1.0, 2.0, 3.0], [8096.0, 8096.0, 8096.0]])
    values_ref = np.array([[1.0, 2.0, 3.0, 1.0 + 100.0 * 2.0 + 10000.0 * 3.0], list(bg_value)])

    points = wp.array(points_np, dtype=wp.vec3, device=device)
    values = wp.empty(len(points_np), dtype=wp.vec4d, device=device)
    wp.launch(test_volume_store_v4d, dim=len(points_np), inputs=[volume.id, points, values], device=device)

    np.testing.assert_equal(values.numpy(), values_ref)


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
    test.assertIsNone(sphere_vdb_array.deleter)


def test_volume_from_numpy_3d(test, device):
    # Volume.allocate_from_tiles() is only available with CUDA
    mins = np.array([-3.0, -3.0, -3.0])
    voxel_size = 0.2
    maxs = np.array([3.0, 3.0, 3.0])
    nums = np.ceil((maxs - mins) / (voxel_size)).astype(dtype=int)
    centers = np.array([[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    rad = 2.5
    sphere_sdf_np = np.zeros((*tuple(nums), 3))
    for x in range(nums[0]):
        for y in range(nums[1]):
            for z in range(nums[2]):
                for k in range(3):
                    pos = mins + voxel_size * np.array([x, y, z])
                    dis = np.linalg.norm(pos - centers[k])
                    sphere_sdf_np[x, y, z, k] = dis - rad
    sphere_vdb = wp.Volume.load_from_numpy(
        sphere_sdf_np, mins, voxel_size, (rad + 3.0 * voxel_size,) * 3, device=device
    )

    test.assertNotEqual(sphere_vdb.id, 0)

    sphere_vdb_array = sphere_vdb.array()
    test.assertEqual(sphere_vdb_array.dtype, wp.uint8)
    test.assertIsNone(sphere_vdb_array.deleter)


def test_volume_from_numpy_anisotropic(test, device):
    """Verify loading NumPy data with per-axis voxel sizes."""
    mins = np.array([-2.0, -2.0, -2.0])
    voxel_size = (0.2, 0.3, 0.4)
    maxs = np.array([2.0, 2.0, 2.0])
    nums = np.ceil((maxs - mins) / np.array(voxel_size)).astype(dtype=int)
    center = np.array([0.0, 0.0, 0.0])
    rad = 1.5
    sphere_sdf_np = np.zeros(tuple(nums), dtype=np.float32)
    for x in range(nums[0]):
        for y in range(nums[1]):
            for z in range(nums[2]):
                pos = mins + np.array(voxel_size) * np.array([x, y, z])
                dis = np.linalg.norm(pos - center)
                sphere_sdf_np[x, y, z] = dis - rad

    sphere_vdb = wp.Volume.load_from_numpy(sphere_sdf_np, mins, voxel_size, bg_value=0.0, device=device)
    test.assertNotEqual(sphere_vdb.id, 0)

    # Verify the grid transform has the expected diagonal voxel sizes
    info = sphere_vdb.get_grid_info()
    transform = np.array(info.transform_matrix).reshape(3, 3)
    np.testing.assert_allclose(np.diag(transform), list(voxel_size), atol=1e-6)


def test_volume_from_numpy_3d_anisotropic(test, device):
    """Verify loading vector-valued NumPy data with anisotropic voxels."""
    mins = np.array([-1.0, -1.0, -1.0])
    voxel_size = (0.1, 0.2, 0.3)
    maxs = np.array([1.0, 1.0, 1.0])
    nums = np.ceil((maxs - mins) / np.array(voxel_size)).astype(dtype=int)
    data = np.zeros((*tuple(nums), 3), dtype=np.float32)

    volume = wp.Volume.load_from_numpy(data, mins, voxel_size, bg_value=(0.0, 0.0, 0.0), device=device)
    test.assertNotEqual(volume.id, 0)

    info = volume.get_grid_info()
    transform = np.array(info.transform_matrix).reshape(3, 3)
    np.testing.assert_allclose(np.diag(transform), list(voxel_size), atol=1e-6)
    np.testing.assert_allclose(np.array(info.translation), mins, atol=1e-6)


def test_volume_from_numpy_bad_voxel_size(test, device):
    """Reject voxel sizes with the wrong number of elements."""
    data = np.zeros((8, 8, 8), dtype=np.float32)
    with test.assertRaises(ValueError):
        wp.Volume.load_from_numpy(data, (0, 0, 0), voxel_size=(0.1, 0.2), bg_value=0.0, device=device)


def test_volume_from_numpy_numpy_scalar(test, device):
    """Accept NumPy scalar types as voxel sizes."""
    mins = np.array([-2.0, -2.0, -2.0])
    voxel_size = np.float32(0.5)
    shape = (16, 16, 16)
    data = np.zeros(shape, dtype=np.float32)

    volume = wp.Volume.load_from_numpy(data, mins, voxel_size, bg_value=0.0, device=device)
    test.assertNotEqual(volume.id, 0)


def test_volume_bad_voxel_size_values(test, device):
    """Reject zero, negative, and nonfinite voxel sizes."""
    data = np.zeros((8, 8, 8), dtype=np.float32)
    with test.assertRaises(ValueError):
        wp.Volume.load_from_numpy(data, (0, 0, 0), voxel_size=0.0, bg_value=0.0, device=device)
    with test.assertRaises(ValueError):
        wp.Volume.load_from_numpy(data, (0, 0, 0), voxel_size=-0.1, bg_value=0.0, device=device)
    with test.assertRaises(ValueError):
        wp.Volume.load_from_numpy(data, (0, 0, 0), voxel_size=(0.1, -0.2, 0.3), bg_value=0.0, device=device)
    with test.assertRaises(ValueError):
        wp.Volume.load_from_numpy(data, (0, 0, 0), voxel_size=float("inf"), bg_value=0.0, device=device)


def test_volume_bad_voxel_size_type(test, device):
    """Reject nonnumeric and nonsequence voxel sizes."""
    data = np.zeros((8, 8, 8), dtype=np.float32)
    with test.assertRaises(TypeError):
        wp.Volume.load_from_numpy(data, (0, 0, 0), voxel_size=None, bg_value=0.0, device=device)
    with test.assertRaises(TypeError):
        wp.Volume.load_from_numpy(data, (0, 0, 0), voxel_size="0.5", bg_value=0.0, device=device)


def test_volume_allocate_bad_voxel_size(test, device):
    """Reject wrong-length voxel sizes during volume allocation."""
    with test.assertRaises(ValueError):
        wp.Volume.allocate(
            min=[0, 0, 0],
            max=[2.0, 3.0, 4.0],
            voxel_size=(0.2, 0.3),
            bg_value=0.0,
            points_in_world_space=True,
            device=device,
        )


def test_volume_allocate_anisotropic(test, device):
    """Allocate a volume with anisotropic voxel sizes."""
    volume = wp.Volume.allocate(
        min=[0, 0, 0],
        max=[2.0, 3.0, 4.0],
        voxel_size=(0.2, 0.3, 0.4),
        bg_value=0.0,
        translation=(0.0, 0.0, 0.0),
        points_in_world_space=True,
        device=device,
    )
    test.assertNotEqual(volume.id, 0)

    info = volume.get_grid_info()
    transform = np.array(info.transform_matrix).reshape(3, 3)
    np.testing.assert_allclose(np.diag(transform), [0.2, 0.3, 0.4], atol=1e-6)

    # Verify per-axis world-to-index conversion produced the expected tiles
    # 2.0/0.2=10, 3.0/0.3=10, 4.0/0.4=10 voxels per axis → 2 tiles per axis
    tiles = volume.get_tiles().numpy()
    test.assertEqual(tiles.shape[0], 8)
    np.testing.assert_array_equal(np.unique(tiles[:, 0]), [0, 8])
    np.testing.assert_array_equal(np.unique(tiles[:, 1]), [0, 8])
    np.testing.assert_array_equal(np.unique(tiles[:, 2]), [0, 8])


def test_volume_aniso_transform(test, device):
    # XY-rotation + z scale
    transform = [
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 2],
    ]

    points = wp.array([[-1, 1, 4]], dtype=float, device=device)
    volume = wp.Volume.allocate_by_voxels(voxel_points=points, transform=transform, device=device)

    # Check that world points are correctly converted to local space
    voxels = volume.get_voxels().numpy()
    assert_np_equal(voxels, [[1, 1, 2]])

    # Check that we retrieve the correct transform from the grid metadata
    assert_np_equal(volume.get_voxel_size(), [-1, 1, 2])
    assert_np_equal(transform, np.array(volume.get_grid_info().transform_matrix).reshape(3, 3))


def test_volume_write(test, device):
    codecs = ["none", "zip", "blosc"]
    try:
        import blosc  # noqa: F401,PLC0415
    except ImportError:
        codecs.pop()

    for volume_name in ("float", "vec3f", "index"):
        for codec in codecs:
            with test.subTest(volume_name=volume_name, codec=codec):
                volume = _get_volume(volume_name, device)
                fd, file_path = tempfile.mkstemp(suffix=".nvdb")
                os.close(fd)
                try:
                    volume.save_to_nvdb(file_path, codec=codec)
                    with open(file_path, "rb") as f:
                        volume_2 = wp.Volume.load_from_nvdb(f)
                    next_volume = volume
                    while next_volume:
                        np.testing.assert_array_equal(next_volume.array().numpy(), volume_2.array().numpy())
                        next_volume = next_volume.load_next_grid()
                        volume_2 = volume_2.load_next_grid()

                finally:
                    os.remove(file_path)

    with test.subTest(volume_write="unsupported"):
        volume = _get_volume("index", device)
        volume = volume.load_next_grid()

        fd, file_path = tempfile.mkstemp(suffix=".nvdb")
        os.close(fd)

        try:
            with test.assertRaises(RuntimeError):
                volume.save_to_nvdb(file_path, codec=codec)
        finally:
            os.remove(file_path)


class TestVolume(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        try:
            _volume_cache.clear()
            _point_cache.clear()
            _jittered_point_cache.clear()
        finally:
            super().tearDownClass()

    def test_volume_new_del(self):
        """Delete a volume that was allocated without initialization."""
        instance = wp.Volume.__new__(wp.Volume)
        instance.__del__()


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
add_function_test(TestVolume, "test_volume_allocation_f", test_volume_allocation_f, devices=devices)
add_function_test(TestVolume, "test_volume_allocation_v", test_volume_allocation_v, devices=devices)
add_function_test(TestVolume, "test_volume_allocation_i", test_volume_allocation_i, devices=devices)
add_function_test(TestVolume, "test_volume_allocation_v4", test_volume_allocation_v4, devices=devices)
add_function_test(TestVolume, "test_volume_allocation_v4d", test_volume_allocation_v4d, devices=devices)
add_function_test(TestVolume, "test_volume_introspection", test_volume_introspection, devices=devices)
add_function_test(TestVolume, "test_volume_from_numpy", test_volume_from_numpy, devices=devices)
add_function_test(TestVolume, "test_volume_from_numpy_3d", test_volume_from_numpy_3d, devices=devices)
add_function_test(
    TestVolume,
    "test_volume_from_numpy_anisotropic",
    test_volume_from_numpy_anisotropic,
    devices=devices,
)
add_function_test(
    TestVolume,
    "test_volume_from_numpy_3d_anisotropic",
    test_volume_from_numpy_3d_anisotropic,
    devices=devices,
)
add_function_test(
    TestVolume,
    "test_volume_from_numpy_bad_voxel_size",
    test_volume_from_numpy_bad_voxel_size,
    devices=devices,
)
add_function_test(
    TestVolume,
    "test_volume_from_numpy_numpy_scalar",
    test_volume_from_numpy_numpy_scalar,
    devices=devices,
)
add_function_test(
    TestVolume,
    "test_volume_bad_voxel_size_values",
    test_volume_bad_voxel_size_values,
    devices=devices,
)
add_function_test(
    TestVolume,
    "test_volume_bad_voxel_size_type",
    test_volume_bad_voxel_size_type,
    devices=devices,
)
add_function_test(
    TestVolume,
    "test_volume_allocate_bad_voxel_size",
    test_volume_allocate_bad_voxel_size,
    devices=devices,
)
add_function_test(
    TestVolume,
    "test_volume_allocate_anisotropic",
    test_volume_allocate_anisotropic,
    devices=get_selected_cuda_test_devices(),
)
add_function_test(
    TestVolume, "test_volume_aniso_transform", test_volume_aniso_transform, devices=get_selected_cuda_test_devices()
)
add_function_test(TestVolume, "test_volume_multiple_grids", test_volume_multiple_grids, devices=devices)
add_function_test(TestVolume, "test_volume_feature_array", test_volume_feature_array, devices=devices)
add_function_test(TestVolume, "test_volume_sample_index", test_volume_sample_index, devices=devices)
add_function_test(TestVolume, "test_volume_sample_grad_v4", test_volume_sample_grad_v4, devices=devices)
add_function_test(TestVolume, "test_volume_sample_grad_index_v4", test_volume_sample_grad_index_v4, devices=devices)
add_function_test(
    TestVolume, "test_volume_sample_grad_index_v4_adjoint", test_volume_sample_grad_index_v4_adjoint, devices=devices
)
add_function_test(
    TestVolume,
    "test_volume_sample_grad_rejects_transposed_grad",
    test_volume_sample_grad_rejects_transposed_grad,
    devices=devices,
)
add_function_test(
    TestVolume,
    "test_volume_sample_index_equivalent_dtype",
    test_volume_sample_index_equivalent_dtype,
    devices=devices,
)
add_function_test(
    TestVolume, "test_volume_allocation_by_tiles_v4d", test_volume_allocation_by_tiles_v4d, devices=devices
)
add_function_test(TestVolume, "test_volume_write", test_volume_write, devices=[wp.get_device("cpu")])

for device in devices:
    add_kernel_test(
        TestVolume,
        test_volume_lookup_f,
        dim=len(point_grid),
        inputs_factory=_volume_kernel_inputs("float"),
        devices=[device],
    )
    add_kernel_test(
        TestVolume,
        test_volume_sample_closest_f,
        dim=len(point_grid),
        inputs_factory=_volume_kernel_inputs("float", jittered=True),
        devices=[device.alias],
    )
    add_kernel_test(
        TestVolume,
        test_volume_sample_linear_f,
        dim=len(point_grid),
        inputs_factory=_volume_kernel_inputs("float", jittered=True),
        devices=[device.alias],
    )
    add_kernel_test(
        TestVolume,
        test_volume_sample_grad_linear_f,
        dim=len(point_grid),
        inputs_factory=_volume_kernel_inputs("float", jittered=True),
        devices=[device.alias],
    )

    add_kernel_test(
        TestVolume,
        test_volume_lookup_v,
        dim=len(point_grid),
        inputs_factory=_volume_kernel_inputs("vec3f"),
        devices=[device.alias],
    )
    add_kernel_test(
        TestVolume,
        test_volume_sample_closest_v,
        dim=len(point_grid),
        inputs_factory=_volume_kernel_inputs("vec3f", jittered=True),
        devices=[device.alias],
    )
    add_kernel_test(
        TestVolume,
        test_volume_sample_linear_v,
        dim=len(point_grid),
        inputs_factory=_volume_kernel_inputs("vec3f", jittered=True),
        devices=[device.alias],
    )
    add_kernel_test(
        TestVolume,
        test_volume_sample_grad_linear_v,
        dim=len(point_grid),
        inputs_factory=_volume_kernel_inputs("vec3f", jittered=True),
        devices=[device.alias],
    )

    add_kernel_test(
        TestVolume,
        test_volume_lookup_i,
        dim=len(point_grid),
        inputs_factory=_volume_kernel_inputs("int32"),
        devices=[device.alias],
    )
    add_kernel_test(
        TestVolume,
        test_volume_sample_i,
        dim=len(point_grid),
        inputs_factory=_volume_kernel_inputs("int32", jittered=True),
        devices=[device.alias],
    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
