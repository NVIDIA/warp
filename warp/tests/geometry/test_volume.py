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

import os
import tempfile
import unittest
from typing import Any

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


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
    expect_eq(wp.volume_lookup(volume, i, j, k, dtype=wp.float32), expected)


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
    expect_eq(wp.volume_sample(volume, p, wp.Volume.CLOSEST, dtype=wp.float32), expected)

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
    expect_near(wp.volume_sample(volume, p, wp.Volume.LINEAR, dtype=wp.float32), expected, 2.0e-4)


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

    val = wp.volume_sample_grad(volume, p, wp.Volume.LINEAR, grad, dtype=wp.float32)

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
    expect_eq(wp.volume_lookup(volume, i, j, k, dtype=wp.vec3), expected)


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
    expect_eq(wp.volume_sample(volume, p, wp.Volume.CLOSEST, dtype=wp.vec3), expected)

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
    expect_near(wp.volume_sample(volume, p, wp.Volume.LINEAR, dtype=wp.vec3), expected, 2.0e-4)


@wp.kernel
def test_volume_sample_grad_linear_v(volume: wp.uint64, points: wp.array(dtype=wp.vec3)):
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
    expect_eq(wp.volume_lookup(volume, i, j, k, dtype=wp.int32), expected)


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
    expect_eq(wp.volume_sample(volume, p, wp.Volume.CLOSEST, dtype=wp.int32), expected)

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

    wp.volume_store(volume, i, j, k, float(i + 100 * j + 10000 * k))
    values[tid] = wp.volume_lookup_f(volume, i, j, k)


@wp.kernel
def test_volume_store_v(volume: wp.uint64, points: wp.array(dtype=wp.vec3), values: wp.array(dtype=wp.vec3)):
    tid = wp.tid()

    p = points[tid]
    i = int(p[0])
    j = int(p[1])
    k = int(p[2])

    wp.volume_store(volume, i, j, k, p)
    values[tid] = wp.volume_lookup_v(volume, i, j, k)


@wp.kernel
def test_volume_store_i(volume: wp.uint64, points: wp.array(dtype=wp.vec3), values: wp.array(dtype=wp.int32)):
    tid = wp.tid()

    p = points[tid]
    i = int(p[0])
    j = int(p[1])
    k = int(p[2])

    wp.volume_store(volume, i, j, k, i + 100 * j + 10000 * k)
    values[tid] = wp.volume_lookup_i(volume, i, j, k)


@wp.kernel
def test_volume_store_v4(volume: wp.uint64, points: wp.array(dtype=wp.vec3), values: wp.array(dtype=wp.vec4)):
    tid = wp.tid()

    p = points[tid]
    i = int(p[0])
    j = int(p[1])
    k = int(p[2])

    v = wp.vec4(p[0], p[1], p[2], float(i + 100 * j + 10000 * k))

    wp.volume_store(volume, i, j, k, v)

    values[tid] = wp.volume_lookup(volume, i, j, k, dtype=wp.vec4)


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

volumes = {}
for value_type, path in volume_paths.items():
    volumes[value_type] = {}
    volume_data = open(path, "rb").read()
    for device in devices:
        try:
            volume = wp.Volume.load_from_nvdb(volume_data, device)
        except RuntimeError as e:
            raise RuntimeError(f'Failed to load volume from "{path}" to {device} memory:\n{e}') from e

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

        grad_expected = np.array([12.0, 15.0, 18.0])
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

        grad_expected = np.array([12.0, 15.0, 18.0]) / 0.25
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


def test_volume_introspection(test, device):
    for volume_names in ("float", "vec3f"):
        with test.subTest(volume_names=volume_names):
            volume = volumes[volume_names][device.alias]
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
    volume = volumes["index"][device.alias]

    volume_2 = volume.load_next_grid()

    test.assertIsNotNone(volume_2)

    test.assertNotEqual(volume.id, volume_2.id)
    test.assertNotEqual(volume.get_voxel_count(), volume_2.get_voxel_count())

    test.assertEqual(volume.get_grid_info().grid_count, volume_2.get_grid_info().grid_count)
    test.assertEqual(volume.get_grid_info().grid_index + 1, volume_2.get_grid_info().grid_index)

    volume_3 = volume_2.load_next_grid()
    test.assertIsNone(volume_3)


def test_volume_feature_array(test, device):
    volume = volumes["index"][device.alias]

    test.assertEqual(volume.get_feature_array_count(), 1)

    array = volume.feature_array(0, dtype=wp.uint64)
    test.assertEqual(array.device, device)
    test.assertEqual(array.dtype, wp.uint64)

    # fVDB convention, data starts with array ndim + shape
    np.testing.assert_equal(array.numpy()[0:4], [3, volume.get_voxel_count(), 2, 3])


@wp.kernel
def fill_leaf_values_kernel(volume: wp.uint64, ijk: wp.array2d(dtype=wp.int32), values: wp.array(dtype=Any)):
    tid = wp.tid()

    i = ijk[tid, 0]
    j = ijk[tid, 1]
    k = ijk[tid, 2]

    expect_eq(tid, wp.volume_lookup_index(volume, i, j, k))

    values[tid] = wp.volume_lookup(volume, i, j, k, dtype=values.dtype)


@wp.kernel
def test_volume_sample_index_kernel(
    volume: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    values: wp.array(dtype=Any),
    background: wp.array(dtype=Any),
    sampled_values: wp.array(dtype=Any),
):
    tid = wp.tid()
    p = points[tid]

    ref = wp.volume_sample(volume, p, wp.Volume.LINEAR, dtype=values.dtype)
    sampled_values[tid] = wp.volume_sample_index(volume, p, wp.Volume.LINEAR, values, background[0])
    expect_eq(sampled_values[tid], ref)


@wp.kernel
def test_volume_sample_grad_index_kernel(
    volume: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    values: wp.array(dtype=Any),
    background: wp.array(dtype=Any),
    sampled_values: wp.array(dtype=Any),
    sampled_grads: wp.array(dtype=Any),
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
            volume = volumes[volume_names][device.alias]

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
        import blosc  # noqa: F401
    except ImportError:
        codecs.pop()

    for volume_name in ("float", "vec3f", "index"):
        for codec in codecs:
            with test.subTest(volume_name=volume_name, codec=codec):
                volume = volumes[volume_name][device.alias]
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
        volume = volumes["index"][device.alias]
        volume = volume.load_next_grid()

        fd, file_path = tempfile.mkstemp(suffix=".nvdb")
        os.close(fd)

        try:
            with test.assertRaises(RuntimeError):
                volume.save_to_nvdb(file_path, codec=codec)
        finally:
            os.remove(file_path)


class TestVolume(unittest.TestCase):
    def test_volume_new_del(self):
        # test the scenario in which a volume is created but not initialized before gc
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
add_function_test(
    TestVolume, "test_volume_allocation_f", test_volume_allocation_f, devices=get_selected_cuda_test_devices()
)
add_function_test(
    TestVolume, "test_volume_allocation_v", test_volume_allocation_v, devices=get_selected_cuda_test_devices()
)
add_function_test(
    TestVolume, "test_volume_allocation_i", test_volume_allocation_i, devices=get_selected_cuda_test_devices()
)
add_function_test(
    TestVolume, "test_volume_allocation_v4", test_volume_allocation_v4, devices=get_selected_cuda_test_devices()
)
add_function_test(TestVolume, "test_volume_introspection", test_volume_introspection, devices=devices)
add_function_test(
    TestVolume, "test_volume_from_numpy", test_volume_from_numpy, devices=get_selected_cuda_test_devices()
)
add_function_test(
    TestVolume, "test_volume_from_numpy_3d", test_volume_from_numpy_3d, devices=get_selected_cuda_test_devices()
)
add_function_test(
    TestVolume, "test_volume_aniso_transform", test_volume_aniso_transform, devices=get_selected_cuda_test_devices()
)
add_function_test(TestVolume, "test_volume_multiple_grids", test_volume_multiple_grids, devices=devices)
add_function_test(TestVolume, "test_volume_feature_array", test_volume_feature_array, devices=devices)
add_function_test(TestVolume, "test_volume_sample_index", test_volume_sample_index, devices=devices)
add_function_test(TestVolume, "test_volume_write", test_volume_write, devices=[wp.get_device("cpu")])

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
        test_volume_sample_grad_linear_v,
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
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
