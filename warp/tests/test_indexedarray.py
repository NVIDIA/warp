# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Any

import numpy as np

import warp as wp
from warp.tests.test_array import FillStruct
from warp.tests.unittest_utils import *


@wp.kernel
def kernel_1d(a: wp.indexedarray(dtype=float), expected: wp.array(dtype=float)):
    i = wp.tid()

    wp.expect_eq(a[i], expected[i])

    a[i] = 2.0 * a[i]

    wp.atomic_add(a, i, 1.0)

    wp.expect_eq(a[i], 2.0 * expected[i] + 1.0)


def test_indexedarray_1d(test, device):
    values = np.arange(10, dtype=np.float32)
    arr = wp.array(data=values, device=device)

    indices = wp.array([1, 3, 5, 7, 9], dtype=int, device=device)

    iarr = wp.indexedarray1d(arr, [indices])

    test.assertEqual(iarr.dtype, arr.dtype)
    test.assertEqual(iarr.ndim, 1)
    test.assertEqual(iarr.shape, (5,))
    test.assertEqual(iarr.size, 5)

    expected_arr = wp.array(data=[1, 3, 5, 7, 9], dtype=float, device=device)

    wp.launch(kernel_1d, dim=iarr.size, inputs=[iarr, expected_arr], device=device)


@wp.kernel
def kernel_2d(a: wp.indexedarray2d(dtype=float), expected: wp.array2d(dtype=float)):
    i, j = wp.tid()

    # check expected values
    wp.expect_eq(a[i, j], expected[i, j])

    # test wp.view()
    wp.expect_eq(a[i][j], a[i, j])

    a[i, j] = 2.0 * a[i, j]

    wp.atomic_add(a, i, j, 1.0)

    wp.expect_eq(a[i, j], 2.0 * expected[i, j] + 1.0)


def test_indexedarray_2d(test, device):
    values = np.arange(100, dtype=np.float32).reshape((10, 10))
    arr = wp.array(data=values, device=device)

    indices0 = wp.array([1, 3], dtype=int, device=device)
    indices1 = wp.array([2, 4, 8], dtype=int, device=device)

    iarr = wp.indexedarray2d(arr, [indices0, indices1])

    test.assertEqual(iarr.dtype, arr.dtype)
    test.assertEqual(iarr.ndim, 2)
    test.assertEqual(iarr.shape, (2, 3))
    test.assertEqual(iarr.size, 6)

    expected_values = [[12, 14, 18], [32, 34, 38]]
    expected_arr = wp.array(data=expected_values, dtype=float, device=device)

    wp.launch(kernel_2d, dim=iarr.shape, inputs=[iarr, expected_arr], device=device)


@wp.kernel
def kernel_3d(a: wp.indexedarray3d(dtype=float), expected: wp.array3d(dtype=float)):
    i, j, k = wp.tid()

    # check expected values
    wp.expect_eq(a[i, j, k], expected[i, j, k])

    # test wp.view()
    wp.expect_eq(a[i][j][k], a[i, j, k])
    wp.expect_eq(a[i, j][k], a[i, j, k])
    wp.expect_eq(a[i][j, k], a[i, j, k])

    a[i, j, k] = 2.0 * a[i, j, k]

    wp.atomic_add(a, i, j, k, 1.0)

    wp.expect_eq(a[i, j, k], 2.0 * expected[i, j, k] + 1.0)


def test_indexedarray_3d(test, device):
    values = np.arange(1000, dtype=np.float32).reshape((10, 10, 10))
    arr = wp.array(data=values, device=device)

    indices0 = wp.array([1, 3], dtype=int, device=device)
    indices1 = wp.array([2, 4, 8], dtype=int, device=device)
    indices2 = wp.array([0, 5], dtype=int, device=device)

    iarr = wp.indexedarray3d(arr, [indices0, indices1, indices2])

    test.assertEqual(iarr.dtype, arr.dtype)
    test.assertEqual(iarr.ndim, 3)
    test.assertEqual(iarr.shape, (2, 3, 2))
    test.assertEqual(iarr.size, 12)

    expected_values = [
        [[120, 125], [140, 145], [180, 185]],
        [[320, 325], [340, 345], [380, 385]],
    ]
    expected_arr = wp.array(data=expected_values, dtype=float, device=device)

    wp.launch(kernel_3d, dim=iarr.shape, inputs=[iarr, expected_arr], device=device)


@wp.kernel
def kernel_4d(a: wp.indexedarray4d(dtype=float), expected: wp.array4d(dtype=float)):
    i, j, k, l = wp.tid()

    # check expected values
    wp.expect_eq(a[i, j, k, l], expected[i, j, k, l])

    # test wp.view()
    wp.expect_eq(a[i][j][k][l], a[i, j, k, l])
    wp.expect_eq(a[i][j, k, l], a[i, j, k, l])
    wp.expect_eq(a[i, j][k, l], a[i, j, k, l])
    wp.expect_eq(a[i, j, k][l], a[i, j, k, l])

    a[i, j, k, l] = 2.0 * a[i, j, k, l]

    wp.atomic_add(a, i, j, k, l, 1.0)

    wp.expect_eq(a[i, j, k, l], 2.0 * expected[i, j, k, l] + 1.0)


def test_indexedarray_4d(test, device):
    values = np.arange(10000, dtype=np.float32).reshape((10, 10, 10, 10))
    arr = wp.array(data=values, device=device)

    indices0 = wp.array([1, 3], dtype=int, device=device)
    indices1 = wp.array([2, 4, 8], dtype=int, device=device)
    indices2 = wp.array([0, 5], dtype=int, device=device)
    indices3 = wp.array([6, 7, 9], dtype=int, device=device)

    iarr = wp.indexedarray4d(arr, [indices0, indices1, indices2, indices3])

    test.assertEqual(iarr.dtype, arr.dtype)
    test.assertEqual(iarr.ndim, 4)
    test.assertEqual(iarr.shape, (2, 3, 2, 3))
    test.assertEqual(iarr.size, 36)

    expected_values = [
        [
            [[1206, 1207, 1209], [1256, 1257, 1259]],
            [[1406, 1407, 1409], [1456, 1457, 1459]],
            [[1806, 1807, 1809], [1856, 1857, 1859]],
        ],
        [
            [[3206, 3207, 3209], [3256, 3257, 3259]],
            [[3406, 3407, 3409], [3456, 3457, 3459]],
            [[3806, 3807, 3809], [3856, 3857, 3859]],
        ],
    ]
    expected_arr = wp.array(data=expected_values, dtype=float, device=device)

    wp.launch(kernel_4d, dim=iarr.shape, inputs=[iarr, expected_arr], device=device)


def test_indexedarray_mixed(test, device):
    # [[[ 0,  1,  2,  3],
    #   [ 4,  5,  6,  7],
    #   [ 8,  9, 10, 11],
    #   [12, 13, 14, 15]],
    #  [[16, 17, 18, 19],
    #   [20, 21, 22, 23],
    #   [24, 25, 26, 27],
    #   [28, 29, 30, 31]],
    #  [[32, 33, 34, 35],
    #   [36, 37, 38, 39],
    #   [40, 41, 42, 43],
    #   [44, 45, 46, 47],
    #  [[48, 49, 50, 51],
    #   [52, 53, 54, 55],
    #   [56, 57, 58, 59],
    #   [60, 61, 62, 63]]]]
    values = np.arange(64, dtype=np.float32).reshape((4, 4, 4))

    indices = wp.array([0, 3], dtype=int, device=device)

    # -----

    arr = wp.array(data=values, device=device)
    iarr = wp.indexedarray(arr, [indices, None, None])
    test.assertEqual(iarr.dtype, arr.dtype)
    test.assertEqual(iarr.ndim, 3)
    test.assertEqual(iarr.shape, (2, 4, 4))
    test.assertEqual(iarr.size, 32)

    expected_values = [
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
        [[48, 49, 50, 51], [52, 53, 54, 55], [56, 57, 58, 59], [60, 61, 62, 63]],
    ]
    expected_arr = wp.array(data=expected_values, dtype=float, device=device)

    wp.launch(kernel_3d, dim=iarr.shape, inputs=[iarr, expected_arr], device=device)

    # -----

    arr = wp.array(data=values, device=device)
    iarr = wp.indexedarray(arr, [indices, indices, None])
    test.assertEqual(iarr.dtype, arr.dtype)
    test.assertEqual(iarr.ndim, 3)
    test.assertEqual(iarr.shape, (2, 2, 4))
    test.assertEqual(iarr.size, 16)

    expected_values = [[[0, 1, 2, 3], [12, 13, 14, 15]], [[48, 49, 50, 51], [60, 61, 62, 63]]]
    expected_arr = wp.array(data=expected_values, dtype=float, device=device)

    wp.launch(kernel_3d, dim=iarr.shape, inputs=[iarr, expected_arr], device=device)

    # -----

    arr = wp.array(data=values, device=device)
    iarr = wp.indexedarray(arr, [indices, None, indices])
    test.assertEqual(iarr.dtype, arr.dtype)
    test.assertEqual(iarr.ndim, 3)
    test.assertEqual(iarr.shape, (2, 4, 2))
    test.assertEqual(iarr.size, 16)

    expected_values = [[[0, 3], [4, 7], [8, 11], [12, 15]], [[48, 51], [52, 55], [56, 59], [60, 63]]]
    expected_arr = wp.array(data=expected_values, dtype=float, device=device)

    wp.launch(kernel_3d, dim=iarr.shape, inputs=[iarr, expected_arr], device=device)

    # -----

    arr = wp.array(data=values, device=device)
    iarr = wp.indexedarray(arr, [None, indices, indices])
    test.assertEqual(iarr.dtype, arr.dtype)
    test.assertEqual(iarr.ndim, 3)
    test.assertEqual(iarr.shape, (4, 2, 2))
    test.assertEqual(iarr.size, 16)

    expected_values = [[[0, 3], [12, 15]], [[16, 19], [28, 31]], [[32, 35], [44, 47]], [[48, 51], [60, 63]]]
    expected_arr = wp.array(data=expected_values, dtype=float, device=device)

    wp.launch(kernel_3d, dim=iarr.shape, inputs=[iarr, expected_arr], device=device)


vec2i = wp.types.vector(length=2, dtype=wp.int32)
vec3i = wp.types.vector(length=3, dtype=wp.int32)
vec4i = wp.types.vector(length=4, dtype=wp.int32)


@wp.kernel
def shape_kernel_1d(arr: wp.indexedarray1d(dtype=float), expected: int):
    wp.expect_eq(arr.shape[0], expected)


@wp.kernel
def shape_kernel_2d(arr: wp.indexedarray2d(dtype=float), expected: vec2i):
    wp.expect_eq(arr.shape[0], expected[0])
    wp.expect_eq(arr.shape[1], expected[1])

    # 1d slice
    view = arr[0]
    wp.expect_eq(view.shape[0], expected[1])


@wp.kernel
def shape_kernel_3d(arr: wp.indexedarray3d(dtype=float), expected: vec3i):
    wp.expect_eq(arr.shape[0], expected[0])
    wp.expect_eq(arr.shape[1], expected[1])
    wp.expect_eq(arr.shape[2], expected[2])

    # 2d slice
    view2 = arr[0]
    wp.expect_eq(view2.shape[0], expected[1])
    wp.expect_eq(view2.shape[1], expected[2])

    # 1d slice
    view1 = arr[0, 0]
    wp.expect_eq(view1.shape[0], expected[2])


@wp.kernel
def shape_kernel_4d(arr: wp.indexedarray4d(dtype=float), expected: vec4i):
    wp.expect_eq(arr.shape[0], expected[0])
    wp.expect_eq(arr.shape[1], expected[1])
    wp.expect_eq(arr.shape[2], expected[2])
    wp.expect_eq(arr.shape[3], expected[3])

    # 3d slice
    view3 = arr[0]
    wp.expect_eq(view3.shape[0], expected[1])
    wp.expect_eq(view3.shape[1], expected[2])
    wp.expect_eq(view3.shape[2], expected[3])

    # 2d slice
    view2 = arr[0, 0]
    wp.expect_eq(view2.shape[0], expected[2])
    wp.expect_eq(view2.shape[1], expected[3])

    # 1d slice
    view1 = arr[0, 0, 0]
    wp.expect_eq(view1.shape[0], expected[3])


def test_indexedarray_shape(test, device):
    with wp.ScopedDevice(device):
        data1 = wp.zeros(10, dtype=float)
        data2 = wp.zeros((10, 20), dtype=float)
        data3 = wp.zeros((10, 20, 30), dtype=float)
        data4 = wp.zeros((10, 20, 30, 40), dtype=float)

        indices1 = wp.array(data=[2, 7], dtype=int)
        indices2 = wp.array(data=[2, 7, 12, 17], dtype=int)
        indices3 = wp.array(data=[2, 7, 12, 17, 22, 27], dtype=int)
        indices4 = wp.array(data=[2, 7, 12, 17, 22, 27, 32, 37], dtype=int)

        ia1 = wp.indexedarray(data1, [indices1])
        wp.launch(shape_kernel_1d, dim=1, inputs=[ia1, 2])

        ia2_1 = wp.indexedarray(data2, [indices1, None])
        ia2_2 = wp.indexedarray(data2, [None, indices2])
        ia2_3 = wp.indexedarray(data2, [indices1, indices2])
        wp.launch(shape_kernel_2d, dim=1, inputs=[ia2_1, vec2i(2, 20)])
        wp.launch(shape_kernel_2d, dim=1, inputs=[ia2_2, vec2i(10, 4)])
        wp.launch(shape_kernel_2d, dim=1, inputs=[ia2_3, vec2i(2, 4)])

        ia3_1 = wp.indexedarray(data3, [indices1, None, None])
        ia3_2 = wp.indexedarray(data3, [None, indices2, None])
        ia3_3 = wp.indexedarray(data3, [None, None, indices3])
        ia3_4 = wp.indexedarray(data3, [indices1, indices2, None])
        ia3_5 = wp.indexedarray(data3, [indices1, None, indices3])
        ia3_6 = wp.indexedarray(data3, [None, indices2, indices3])
        ia3_7 = wp.indexedarray(data3, [indices1, indices2, indices3])
        wp.launch(shape_kernel_3d, dim=1, inputs=[ia3_1, vec3i(2, 20, 30)])
        wp.launch(shape_kernel_3d, dim=1, inputs=[ia3_2, vec3i(10, 4, 30)])
        wp.launch(shape_kernel_3d, dim=1, inputs=[ia3_3, vec3i(10, 20, 6)])
        wp.launch(shape_kernel_3d, dim=1, inputs=[ia3_4, vec3i(2, 4, 30)])
        wp.launch(shape_kernel_3d, dim=1, inputs=[ia3_5, vec3i(2, 20, 6)])
        wp.launch(shape_kernel_3d, dim=1, inputs=[ia3_6, vec3i(10, 4, 6)])
        wp.launch(shape_kernel_3d, dim=1, inputs=[ia3_7, vec3i(2, 4, 6)])

        ia4_1 = wp.indexedarray(data4, [indices1, None, None, None])
        ia4_2 = wp.indexedarray(data4, [indices1, None, None, indices4])
        ia4_3 = wp.indexedarray(data4, [None, indices2, indices3, None])
        ia4_4 = wp.indexedarray(data4, [indices1, indices2, indices3, indices4])
        wp.launch(shape_kernel_4d, dim=1, inputs=[ia4_1, vec4i(2, 20, 30, 40)])
        wp.launch(shape_kernel_4d, dim=1, inputs=[ia4_2, vec4i(2, 20, 30, 8)])
        wp.launch(shape_kernel_4d, dim=1, inputs=[ia4_3, vec4i(10, 4, 6, 40)])
        wp.launch(shape_kernel_4d, dim=1, inputs=[ia4_4, vec4i(2, 4, 6, 8)])

        wp.synchronize_device(device)


def test_indexedarray_getitem(test, device):
    with wp.ScopedDevice(device):
        data = wp.array(data=np.arange(1000, dtype=np.int32).reshape((10, 10, 10)))

        I = wp.array(data=[0, 1, 2], dtype=int)

        # use constructor
        a1 = wp.indexedarray(data, [None, None, I])
        a2 = wp.indexedarray(data, [None, I])
        a3 = wp.indexedarray(data, [None, I, I])
        a4 = wp.indexedarray(data, [I])
        a5 = wp.indexedarray(data, [I, None, I])
        a6 = wp.indexedarray(data, [I, I])
        a7 = wp.indexedarray(data, [I, I, I])

        # use array.__getitem__()
        b1 = data[:, :, I]
        b2 = data[:, I]
        b3 = data[:, I, I]
        b4 = data[I]
        b5 = data[I, :, I]
        b6 = data[I, I]
        b7 = data[I, I, I]

        test.assertEqual(type(a1), type(b1))
        test.assertEqual(type(a2), type(b2))
        test.assertEqual(type(a3), type(b3))
        test.assertEqual(type(a4), type(b4))
        test.assertEqual(type(a5), type(b5))
        test.assertEqual(type(a6), type(b6))
        test.assertEqual(type(a7), type(b7))

        assert_np_equal(a1.numpy(), b1.numpy())
        assert_np_equal(a2.numpy(), b2.numpy())
        assert_np_equal(a3.numpy(), b3.numpy())
        assert_np_equal(a4.numpy(), b4.numpy())
        assert_np_equal(a5.numpy(), b5.numpy())
        assert_np_equal(a6.numpy(), b6.numpy())
        assert_np_equal(a7.numpy(), b7.numpy())


def test_indexedarray_slicing(test, device):
    with wp.ScopedDevice(device):
        data = wp.array(data=np.arange(1000, dtype=np.int32).reshape((10, 10, 10)))

        # test equivalence of slicing and indexing the same range
        s = slice(0, 3)
        I = wp.array(data=[0, 1, 2], dtype=int)

        a0 = data[s, s, s]
        test.assertEqual(type(a0), wp.array)
        a1 = data[s, s, I]
        test.assertEqual(type(a1), wp.indexedarray)
        a2 = data[s, I, s]
        test.assertEqual(type(a2), wp.indexedarray)
        a3 = data[s, I, I]
        test.assertEqual(type(a3), wp.indexedarray)
        a4 = data[I, s, s]
        test.assertEqual(type(a4), wp.indexedarray)
        a5 = data[I, s, I]
        test.assertEqual(type(a5), wp.indexedarray)
        a6 = data[I, I, s]
        test.assertEqual(type(a6), wp.indexedarray)
        a7 = data[I, I, I]
        test.assertEqual(type(a7), wp.indexedarray)

        expected = a0.numpy()

        assert_np_equal(a1.numpy(), expected)
        assert_np_equal(a2.numpy(), expected)
        assert_np_equal(a3.numpy(), expected)
        assert_np_equal(a4.numpy(), expected)
        assert_np_equal(a5.numpy(), expected)
        assert_np_equal(a6.numpy(), expected)
        assert_np_equal(a7.numpy(), expected)


# generic increment kernels that work with any array (regular or indexed)
@wp.kernel
def inc_1d(a: Any):
    i = wp.tid()
    a[i] = a[i] + 1


@wp.kernel
def inc_2d(a: Any):
    i, j = wp.tid()
    a[i, j] = a[i, j] + 1


@wp.kernel
def inc_3d(a: Any):
    i, j, k = wp.tid()
    a[i, j, k] = a[i, j, k] + 1


@wp.kernel
def inc_4d(a: Any):
    i, j, k, l = wp.tid()
    a[i, j, k, l] = a[i, j, k, l] + 1


# optional overloads to avoid module reloading
wp.overload(inc_1d, [wp.array1d(dtype=int)])
wp.overload(inc_2d, [wp.array2d(dtype=int)])
wp.overload(inc_3d, [wp.array3d(dtype=int)])
wp.overload(inc_4d, [wp.array4d(dtype=int)])

wp.overload(inc_1d, [wp.indexedarray1d(dtype=int)])
wp.overload(inc_2d, [wp.indexedarray2d(dtype=int)])
wp.overload(inc_3d, [wp.indexedarray3d(dtype=int)])
wp.overload(inc_4d, [wp.indexedarray4d(dtype=int)])


def test_indexedarray_generics(test, device):
    with wp.ScopedDevice(device):
        data1 = wp.zeros((5,), dtype=int)
        data2 = wp.zeros((5, 5), dtype=int)
        data3 = wp.zeros((5, 5, 5), dtype=int)
        data4 = wp.zeros((5, 5, 5, 5), dtype=int)

        indices = wp.array(data=[0, 4], dtype=int)

        ia1 = wp.indexedarray(data1, [indices])
        ia2 = wp.indexedarray(data2, [indices, indices])
        ia3 = wp.indexedarray(data3, [indices, indices, indices])
        ia4 = wp.indexedarray(data4, [indices, indices, indices, indices])

        wp.launch(inc_1d, dim=data1.shape, inputs=[data1])
        wp.launch(inc_2d, dim=data2.shape, inputs=[data2])
        wp.launch(inc_3d, dim=data3.shape, inputs=[data3])
        wp.launch(inc_4d, dim=data4.shape, inputs=[data4])

        wp.launch(inc_1d, dim=ia1.shape, inputs=[ia1])
        wp.launch(inc_2d, dim=ia2.shape, inputs=[ia2])
        wp.launch(inc_3d, dim=ia3.shape, inputs=[ia3])
        wp.launch(inc_4d, dim=ia4.shape, inputs=[ia4])

        expected1 = np.ones(5, dtype=np.int32)
        expected1[0] = 2
        expected1[4] = 2

        expected2 = np.ones((5, 5), dtype=np.int32)
        expected2[0, 0] = 2
        expected2[0, 4] = 2
        expected2[4, 0] = 2
        expected2[4, 4] = 2

        expected3 = np.ones((5, 5, 5), dtype=np.int32)
        expected3[0, 0, 0] = 2
        expected3[0, 0, 4] = 2
        expected3[0, 4, 0] = 2
        expected3[0, 4, 4] = 2
        expected3[4, 0, 0] = 2
        expected3[4, 0, 4] = 2
        expected3[4, 4, 0] = 2
        expected3[4, 4, 4] = 2

        expected4 = np.ones((5, 5, 5, 5), dtype=np.int32)
        expected4[0, 0, 0, 0] = 2
        expected4[0, 0, 0, 4] = 2
        expected4[0, 0, 4, 0] = 2
        expected4[0, 0, 4, 4] = 2
        expected4[0, 4, 0, 0] = 2
        expected4[0, 4, 0, 4] = 2
        expected4[0, 4, 4, 0] = 2
        expected4[0, 4, 4, 4] = 2
        expected4[4, 0, 0, 0] = 2
        expected4[4, 0, 0, 4] = 2
        expected4[4, 0, 4, 0] = 2
        expected4[4, 0, 4, 4] = 2
        expected4[4, 4, 0, 0] = 2
        expected4[4, 4, 0, 4] = 2
        expected4[4, 4, 4, 0] = 2
        expected4[4, 4, 4, 4] = 2

        assert_np_equal(data1.numpy(), expected1)
        assert_np_equal(data2.numpy(), expected2)
        assert_np_equal(data3.numpy(), expected3)
        assert_np_equal(data4.numpy(), expected4)

        assert_np_equal(ia1.numpy(), np.full((2,), 2, dtype=np.int32))
        assert_np_equal(ia2.numpy(), np.full((2, 2), 2, dtype=np.int32))
        assert_np_equal(ia3.numpy(), np.full((2, 2, 2), 2, dtype=np.int32))
        assert_np_equal(ia4.numpy(), np.full((2, 2, 2, 2), 2, dtype=np.int32))


def test_indexedarray_empty(test, device):
    # Test whether common operations work with empty (zero-sized) indexed arrays
    # without throwing exceptions.

    def test_empty_ops(ndim, nrows, ncols, wptype, nptype):
        data_shape = (1,) * ndim
        dtype_shape = ()

        if wptype in wp.types.scalar_types:
            # scalar, vector, or matrix
            if ncols > 0:
                if nrows > 0:
                    wptype = wp.types.matrix((nrows, ncols), wptype)
                else:
                    wptype = wp.types.vector(ncols, wptype)
                dtype_shape = wptype._shape_
            fill_value = wptype(42)
        else:
            # struct
            fill_value = wptype()

        # create a data array
        data = wp.empty(data_shape, dtype=wptype, device=device, requires_grad=True)

        # create a zero-sized array of indices
        indices = wp.empty(0, dtype=int, device=device)

        a = data[indices]

        # we expect dim to be zero for the empty indexed array, unchanged otherwise
        expected_shape = (0, *data_shape[1:])

        test.assertEqual(a.size, 0)
        test.assertEqual(a.shape, expected_shape)

        # all of these methods should succeed with zero-sized arrays
        a.zero_()
        a.fill_(fill_value)
        b = a.contiguous()

        b = wp.empty_like(a)
        b = wp.zeros_like(a)
        b = wp.full_like(a, fill_value)
        b = wp.clone(a)

        wp.copy(a, b)
        a.assign(b)

        na = a.numpy()
        test.assertEqual(na.size, 0)
        test.assertEqual(na.shape, (*expected_shape, *dtype_shape))
        test.assertEqual(na.dtype, nptype)

        test.assertEqual(a.list(), [])

    for ndim in range(1, 5):
        # test with scalars, vectors, and matrices
        for nptype, wptype in wp.types.np_dtype_to_warp_type.items():
            # scalars
            test_empty_ops(ndim, 0, 0, wptype, nptype)

            for ncols in [2, 3, 4, 5]:
                # vectors
                test_empty_ops(ndim, 0, ncols, wptype, nptype)
                # square matrices
                test_empty_ops(ndim, ncols, ncols, wptype, nptype)

            # non-square matrices
            test_empty_ops(ndim, 2, 3, wptype, nptype)
            test_empty_ops(ndim, 3, 2, wptype, nptype)
            test_empty_ops(ndim, 3, 4, wptype, nptype)
            test_empty_ops(ndim, 4, 3, wptype, nptype)

        # test with structs
        test_empty_ops(ndim, 0, 0, FillStruct, FillStruct.numpy_dtype())


def test_indexedarray_fill_scalar(test, device):
    dim_x = 4

    for nptype, wptype in wp.types.np_dtype_to_warp_type.items():
        data1 = wp.zeros(dim_x, dtype=wptype, device=device)
        data2 = wp.zeros((dim_x, dim_x), dtype=wptype, device=device)
        data3 = wp.zeros((dim_x, dim_x, dim_x), dtype=wptype, device=device)
        data4 = wp.zeros((dim_x, dim_x, dim_x, dim_x), dtype=wptype, device=device)

        indices = wp.array(np.arange(0, dim_x, 2, dtype=np.int32), device=device)

        a1 = data1[indices]
        a2 = data2[indices]
        a3 = data3[indices]
        a4 = data4[indices]

        assert_np_equal(a1.numpy(), np.zeros(a1.shape, dtype=nptype))
        assert_np_equal(a2.numpy(), np.zeros(a2.shape, dtype=nptype))
        assert_np_equal(a3.numpy(), np.zeros(a3.shape, dtype=nptype))
        assert_np_equal(a4.numpy(), np.zeros(a4.shape, dtype=nptype))

        # fill with int value
        fill_value = 42

        a1.fill_(fill_value)
        a2.fill_(fill_value)
        a3.fill_(fill_value)
        a4.fill_(fill_value)

        assert_np_equal(a1.numpy(), np.full(a1.shape, fill_value, dtype=nptype))
        assert_np_equal(a2.numpy(), np.full(a2.shape, fill_value, dtype=nptype))
        assert_np_equal(a3.numpy(), np.full(a3.shape, fill_value, dtype=nptype))
        assert_np_equal(a4.numpy(), np.full(a4.shape, fill_value, dtype=nptype))

        a1.zero_()
        a2.zero_()
        a3.zero_()
        a4.zero_()

        assert_np_equal(a1.numpy(), np.zeros(a1.shape, dtype=nptype))
        assert_np_equal(a2.numpy(), np.zeros(a2.shape, dtype=nptype))
        assert_np_equal(a3.numpy(), np.zeros(a3.shape, dtype=nptype))
        assert_np_equal(a4.numpy(), np.zeros(a4.shape, dtype=nptype))

        if wptype in wp.types.float_types:
            # fill with float value
            fill_value = 13.37

            a1.fill_(fill_value)
            a2.fill_(fill_value)
            a3.fill_(fill_value)
            a4.fill_(fill_value)

            assert_np_equal(a1.numpy(), np.full(a1.shape, fill_value, dtype=nptype))
            assert_np_equal(a2.numpy(), np.full(a2.shape, fill_value, dtype=nptype))
            assert_np_equal(a3.numpy(), np.full(a3.shape, fill_value, dtype=nptype))
            assert_np_equal(a4.numpy(), np.full(a4.shape, fill_value, dtype=nptype))

        # fill with Warp scalar value
        fill_value = wptype(17)

        a1.fill_(fill_value)
        a2.fill_(fill_value)
        a3.fill_(fill_value)
        a4.fill_(fill_value)

        assert_np_equal(a1.numpy(), np.full(a1.shape, fill_value.value, dtype=nptype))
        assert_np_equal(a2.numpy(), np.full(a2.shape, fill_value.value, dtype=nptype))
        assert_np_equal(a3.numpy(), np.full(a3.shape, fill_value.value, dtype=nptype))
        assert_np_equal(a4.numpy(), np.full(a4.shape, fill_value.value, dtype=nptype))


def test_indexedarray_fill_vector(test, device):
    # test filling a vector array with scalar or vector values (vec_type, list, or numpy array)

    dim_x = 4

    for nptype, wptype in wp.types.np_dtype_to_warp_type.items():
        # vector types
        vector_types = [
            wp.types.vector(2, wptype),
            wp.types.vector(3, wptype),
            wp.types.vector(4, wptype),
            wp.types.vector(5, wptype),
        ]

        for vec_type in vector_types:
            vec_len = vec_type._length_

            data1 = wp.zeros(dim_x, dtype=vec_type, device=device)
            data2 = wp.zeros((dim_x, dim_x), dtype=vec_type, device=device)
            data3 = wp.zeros((dim_x, dim_x, dim_x), dtype=vec_type, device=device)
            data4 = wp.zeros((dim_x, dim_x, dim_x, dim_x), dtype=vec_type, device=device)

            indices = wp.array(np.arange(0, dim_x, 2, dtype=np.int32), device=device)

            a1 = data1[indices]
            a2 = data2[indices]
            a3 = data3[indices]
            a4 = data4[indices]

            assert_np_equal(a1.numpy(), np.zeros((*a1.shape, vec_len), dtype=nptype))
            assert_np_equal(a2.numpy(), np.zeros((*a2.shape, vec_len), dtype=nptype))
            assert_np_equal(a3.numpy(), np.zeros((*a3.shape, vec_len), dtype=nptype))
            assert_np_equal(a4.numpy(), np.zeros((*a4.shape, vec_len), dtype=nptype))

            # fill with int scalar
            fill_value = 42

            a1.fill_(fill_value)
            a2.fill_(fill_value)
            a3.fill_(fill_value)
            a4.fill_(fill_value)

            assert_np_equal(a1.numpy(), np.full((*a1.shape, vec_len), fill_value, dtype=nptype))
            assert_np_equal(a2.numpy(), np.full((*a2.shape, vec_len), fill_value, dtype=nptype))
            assert_np_equal(a3.numpy(), np.full((*a3.shape, vec_len), fill_value, dtype=nptype))
            assert_np_equal(a4.numpy(), np.full((*a4.shape, vec_len), fill_value, dtype=nptype))

            # test zeroing
            a1.zero_()
            a2.zero_()
            a3.zero_()
            a4.zero_()

            assert_np_equal(a1.numpy(), np.zeros((*a1.shape, vec_len), dtype=nptype))
            assert_np_equal(a2.numpy(), np.zeros((*a2.shape, vec_len), dtype=nptype))
            assert_np_equal(a3.numpy(), np.zeros((*a3.shape, vec_len), dtype=nptype))
            assert_np_equal(a4.numpy(), np.zeros((*a4.shape, vec_len), dtype=nptype))

            # vector values can be passed as a list, numpy array, or Warp vector instance
            fill_list = [17, 42, 99, 101, 127][:vec_len]
            fill_arr = np.array(fill_list, dtype=nptype)
            fill_vec = vec_type(fill_list)

            expected1 = np.tile(fill_arr, a1.size).reshape((*a1.shape, vec_len))
            expected2 = np.tile(fill_arr, a2.size).reshape((*a2.shape, vec_len))
            expected3 = np.tile(fill_arr, a3.size).reshape((*a3.shape, vec_len))
            expected4 = np.tile(fill_arr, a4.size).reshape((*a4.shape, vec_len))

            # fill with list of vector length
            a1.fill_(fill_list)
            a2.fill_(fill_list)
            a3.fill_(fill_list)
            a4.fill_(fill_list)

            assert_np_equal(a1.numpy(), expected1)
            assert_np_equal(a2.numpy(), expected2)
            assert_np_equal(a3.numpy(), expected3)
            assert_np_equal(a4.numpy(), expected4)

            # clear
            a1.zero_()
            a2.zero_()
            a3.zero_()
            a4.zero_()

            # fill with numpy array of vector length
            a1.fill_(fill_arr)
            a2.fill_(fill_arr)
            a3.fill_(fill_arr)
            a4.fill_(fill_arr)

            assert_np_equal(a1.numpy(), expected1)
            assert_np_equal(a2.numpy(), expected2)
            assert_np_equal(a3.numpy(), expected3)
            assert_np_equal(a4.numpy(), expected4)

            # clear
            a1.zero_()
            a2.zero_()
            a3.zero_()
            a4.zero_()

            # fill with vec instance
            a1.fill_(fill_vec)
            a2.fill_(fill_vec)
            a3.fill_(fill_vec)
            a4.fill_(fill_vec)

            assert_np_equal(a1.numpy(), expected1)
            assert_np_equal(a2.numpy(), expected2)
            assert_np_equal(a3.numpy(), expected3)
            assert_np_equal(a4.numpy(), expected4)

            if wptype in wp.types.float_types:
                # fill with float scalar
                fill_value = 13.37

                a1.fill_(fill_value)
                a2.fill_(fill_value)
                a3.fill_(fill_value)
                a4.fill_(fill_value)

                assert_np_equal(a1.numpy(), np.full((*a1.shape, vec_len), fill_value, dtype=nptype))
                assert_np_equal(a2.numpy(), np.full((*a2.shape, vec_len), fill_value, dtype=nptype))
                assert_np_equal(a3.numpy(), np.full((*a3.shape, vec_len), fill_value, dtype=nptype))
                assert_np_equal(a4.numpy(), np.full((*a4.shape, vec_len), fill_value, dtype=nptype))

                # fill with float list of vector length
                fill_list = [-2.5, -1.25, 1.25, 2.5, 5.0][:vec_len]

                a1.fill_(fill_list)
                a2.fill_(fill_list)
                a3.fill_(fill_list)
                a4.fill_(fill_list)

                expected1 = np.tile(np.array(fill_list, dtype=nptype), a1.size).reshape((*a1.shape, vec_len))
                expected2 = np.tile(np.array(fill_list, dtype=nptype), a2.size).reshape((*a2.shape, vec_len))
                expected3 = np.tile(np.array(fill_list, dtype=nptype), a3.size).reshape((*a3.shape, vec_len))
                expected4 = np.tile(np.array(fill_list, dtype=nptype), a4.size).reshape((*a4.shape, vec_len))

                assert_np_equal(a1.numpy(), expected1)
                assert_np_equal(a2.numpy(), expected2)
                assert_np_equal(a3.numpy(), expected3)
                assert_np_equal(a4.numpy(), expected4)


def test_indexedarray_fill_matrix(test, device):
    # test filling a matrix array with scalar or matrix values (mat_type, nested list, or 2d numpy array)

    dim_x = 4

    for nptype, wptype in wp.types.np_dtype_to_warp_type.items():
        # matrix types
        matrix_types = [
            # square matrices
            wp.types.matrix((2, 2), wptype),
            wp.types.matrix((3, 3), wptype),
            wp.types.matrix((4, 4), wptype),
            wp.types.matrix((5, 5), wptype),
            # non-square matrices
            wp.types.matrix((2, 3), wptype),
            wp.types.matrix((3, 2), wptype),
            wp.types.matrix((3, 4), wptype),
            wp.types.matrix((4, 3), wptype),
        ]

        for mat_type in matrix_types:
            mat_len = mat_type._length_
            mat_shape = mat_type._shape_

            data1 = wp.zeros(dim_x, dtype=mat_type, device=device)
            data2 = wp.zeros((dim_x, dim_x), dtype=mat_type, device=device)
            data3 = wp.zeros((dim_x, dim_x, dim_x), dtype=mat_type, device=device)
            data4 = wp.zeros((dim_x, dim_x, dim_x, dim_x), dtype=mat_type, device=device)

            indices = wp.array(np.arange(0, dim_x, 2, dtype=np.int32), device=device)

            a1 = data1[indices]
            a2 = data2[indices]
            a3 = data3[indices]
            a4 = data4[indices]

            assert_np_equal(a1.numpy(), np.zeros((*a1.shape, *mat_shape), dtype=nptype))
            assert_np_equal(a2.numpy(), np.zeros((*a2.shape, *mat_shape), dtype=nptype))
            assert_np_equal(a3.numpy(), np.zeros((*a3.shape, *mat_shape), dtype=nptype))
            assert_np_equal(a4.numpy(), np.zeros((*a4.shape, *mat_shape), dtype=nptype))

            # fill with scalar
            fill_value = 42

            a1.fill_(fill_value)
            a2.fill_(fill_value)
            a3.fill_(fill_value)
            a4.fill_(fill_value)

            assert_np_equal(a1.numpy(), np.full((*a1.shape, *mat_shape), fill_value, dtype=nptype))
            assert_np_equal(a2.numpy(), np.full((*a2.shape, *mat_shape), fill_value, dtype=nptype))
            assert_np_equal(a3.numpy(), np.full((*a3.shape, *mat_shape), fill_value, dtype=nptype))
            assert_np_equal(a4.numpy(), np.full((*a4.shape, *mat_shape), fill_value, dtype=nptype))

            # test zeroing
            a1.zero_()
            a2.zero_()
            a3.zero_()
            a4.zero_()

            assert_np_equal(a1.numpy(), np.zeros((*a1.shape, *mat_shape), dtype=nptype))
            assert_np_equal(a2.numpy(), np.zeros((*a2.shape, *mat_shape), dtype=nptype))
            assert_np_equal(a3.numpy(), np.zeros((*a3.shape, *mat_shape), dtype=nptype))
            assert_np_equal(a4.numpy(), np.zeros((*a4.shape, *mat_shape), dtype=nptype))

            # matrix values can be passed as a 1d numpy array, 2d numpy array, flat list, nested list, or Warp matrix instance
            if wptype != wp.bool:
                fill_arr1 = np.arange(mat_len, dtype=nptype)
            else:
                fill_arr1 = np.ones(mat_len, dtype=nptype)
            fill_arr2 = fill_arr1.reshape(mat_shape)
            fill_list1 = list(fill_arr1)
            fill_list2 = [list(row) for row in fill_arr2]
            fill_mat = mat_type(fill_arr1)

            expected1 = np.tile(fill_arr1, a1.size).reshape((*a1.shape, *mat_shape))
            expected2 = np.tile(fill_arr1, a2.size).reshape((*a2.shape, *mat_shape))
            expected3 = np.tile(fill_arr1, a3.size).reshape((*a3.shape, *mat_shape))
            expected4 = np.tile(fill_arr1, a4.size).reshape((*a4.shape, *mat_shape))

            # fill with 1d numpy array
            a1.fill_(fill_arr1)
            a2.fill_(fill_arr1)
            a3.fill_(fill_arr1)
            a4.fill_(fill_arr1)

            assert_np_equal(a1.numpy(), expected1)
            assert_np_equal(a2.numpy(), expected2)
            assert_np_equal(a3.numpy(), expected3)
            assert_np_equal(a4.numpy(), expected4)

            # clear
            a1.zero_()
            a2.zero_()
            a3.zero_()
            a4.zero_()

            # fill with 2d numpy array
            a1.fill_(fill_arr2)
            a2.fill_(fill_arr2)
            a3.fill_(fill_arr2)
            a4.fill_(fill_arr2)

            assert_np_equal(a1.numpy(), expected1)
            assert_np_equal(a2.numpy(), expected2)
            assert_np_equal(a3.numpy(), expected3)
            assert_np_equal(a4.numpy(), expected4)

            # clear
            a1.zero_()
            a2.zero_()
            a3.zero_()
            a4.zero_()

            # fill with flat list
            a1.fill_(fill_list1)
            a2.fill_(fill_list1)
            a3.fill_(fill_list1)
            a4.fill_(fill_list1)

            assert_np_equal(a1.numpy(), expected1)
            assert_np_equal(a2.numpy(), expected2)
            assert_np_equal(a3.numpy(), expected3)
            assert_np_equal(a4.numpy(), expected4)

            # clear
            a1.zero_()
            a2.zero_()
            a3.zero_()
            a4.zero_()

            # fill with nested list
            a1.fill_(fill_list2)
            a2.fill_(fill_list2)
            a3.fill_(fill_list2)
            a4.fill_(fill_list2)

            assert_np_equal(a1.numpy(), expected1)
            assert_np_equal(a2.numpy(), expected2)
            assert_np_equal(a3.numpy(), expected3)
            assert_np_equal(a4.numpy(), expected4)

            # clear
            a1.zero_()
            a2.zero_()
            a3.zero_()
            a4.zero_()

            # fill with mat instance
            a1.fill_(fill_mat)
            a2.fill_(fill_mat)
            a3.fill_(fill_mat)
            a4.fill_(fill_mat)

            assert_np_equal(a1.numpy(), expected1)
            assert_np_equal(a2.numpy(), expected2)
            assert_np_equal(a3.numpy(), expected3)
            assert_np_equal(a4.numpy(), expected4)


def test_indexedarray_fill_struct(test, device):
    dim_x = 8

    nptype = FillStruct.numpy_dtype()

    data1 = wp.zeros(dim_x, dtype=FillStruct, device=device)
    data2 = wp.zeros((dim_x, dim_x), dtype=FillStruct, device=device)
    data3 = wp.zeros((dim_x, dim_x, dim_x), dtype=FillStruct, device=device)
    data4 = wp.zeros((dim_x, dim_x, dim_x, dim_x), dtype=FillStruct, device=device)

    indices = wp.array(np.arange(0, dim_x, 2, dtype=np.int32), device=device)

    a1 = data1[indices]
    a2 = data2[indices]
    a3 = data3[indices]
    a4 = data4[indices]

    assert_np_equal(a1.numpy(), np.zeros(a1.shape, dtype=nptype))
    assert_np_equal(a2.numpy(), np.zeros(a2.shape, dtype=nptype))
    assert_np_equal(a3.numpy(), np.zeros(a3.shape, dtype=nptype))
    assert_np_equal(a4.numpy(), np.zeros(a4.shape, dtype=nptype))

    s = FillStruct()

    # fill with default struct value (should be all zeros)
    a1.fill_(s)
    a2.fill_(s)
    a3.fill_(s)
    a4.fill_(s)

    assert_np_equal(a1.numpy(), np.zeros(a1.shape, dtype=nptype))
    assert_np_equal(a2.numpy(), np.zeros(a2.shape, dtype=nptype))
    assert_np_equal(a3.numpy(), np.zeros(a3.shape, dtype=nptype))
    assert_np_equal(a4.numpy(), np.zeros(a4.shape, dtype=nptype))

    # scalars
    s.i1 = -17
    s.i2 = 42
    s.i4 = 99
    s.i8 = 101
    s.f2 = -1.25
    s.f4 = 13.37
    s.f8 = 0.125
    # vectors
    s.v2 = [21, 22]
    s.v3 = [31, 32, 33]
    s.v4 = [41, 42, 43, 44]
    s.v5 = [51, 52, 53, 54, 55]
    # matrices
    s.m2 = [[61, 62]] * 2
    s.m3 = [[71, 72, 73]] * 3
    s.m4 = [[81, 82, 83, 84]] * 4
    s.m5 = [[91, 92, 93, 94, 95]] * 5
    # arrays
    s.a1 = wp.zeros((2,) * 1, dtype=float, device=device)
    s.a2 = wp.zeros((2,) * 2, dtype=float, device=device)
    s.a3 = wp.zeros((2,) * 3, dtype=float, device=device)
    s.a4 = wp.zeros((2,) * 4, dtype=float, device=device)

    # fill with custom struct value
    a1.fill_(s)
    a2.fill_(s)
    a3.fill_(s)
    a4.fill_(s)

    ns = s.numpy_value()

    expected1 = np.empty(a1.shape, dtype=nptype)
    expected2 = np.empty(a2.shape, dtype=nptype)
    expected3 = np.empty(a3.shape, dtype=nptype)
    expected4 = np.empty(a4.shape, dtype=nptype)

    expected1.fill(ns)
    expected2.fill(ns)
    expected3.fill(ns)
    expected4.fill(ns)

    assert_np_equal(a1.numpy(), expected1)
    assert_np_equal(a2.numpy(), expected2)
    assert_np_equal(a3.numpy(), expected3)
    assert_np_equal(a4.numpy(), expected4)

    # test clearing
    a1.zero_()
    a2.zero_()
    a3.zero_()
    a4.zero_()

    assert_np_equal(a1.numpy(), np.zeros(a1.shape, dtype=nptype))
    assert_np_equal(a2.numpy(), np.zeros(a2.shape, dtype=nptype))
    assert_np_equal(a3.numpy(), np.zeros(a3.shape, dtype=nptype))
    assert_np_equal(a4.numpy(), np.zeros(a4.shape, dtype=nptype))


devices = get_test_devices()


class TestIndexedArray(unittest.TestCase):
    pass


add_function_test(TestIndexedArray, "test_indexedarray_1d", test_indexedarray_1d, devices=devices)
add_function_test(TestIndexedArray, "test_indexedarray_2d", test_indexedarray_2d, devices=devices)
add_function_test(TestIndexedArray, "test_indexedarray_3d", test_indexedarray_3d, devices=devices)
add_function_test(TestIndexedArray, "test_indexedarray_4d", test_indexedarray_4d, devices=devices)
add_function_test(TestIndexedArray, "test_indexedarray_mixed", test_indexedarray_mixed, devices=devices)
add_function_test(TestIndexedArray, "test_indexedarray_shape", test_indexedarray_shape, devices=devices)
add_function_test(TestIndexedArray, "test_indexedarray_getitem", test_indexedarray_getitem, devices=devices)
add_function_test(TestIndexedArray, "test_indexedarray_slicing", test_indexedarray_slicing, devices=devices)
add_function_test(TestIndexedArray, "test_indexedarray_generics", test_indexedarray_generics, devices=devices)
add_function_test(TestIndexedArray, "test_indexedarray_empty", test_indexedarray_empty, devices=devices)
add_function_test(TestIndexedArray, "test_indexedarray_fill_scalar", test_indexedarray_fill_scalar, devices=devices)
add_function_test(TestIndexedArray, "test_indexedarray_fill_vector", test_indexedarray_fill_vector, devices=devices)
add_function_test(TestIndexedArray, "test_indexedarray_fill_matrix", test_indexedarray_fill_matrix, devices=devices)
add_function_test(TestIndexedArray, "test_indexedarray_fill_struct", test_indexedarray_fill_struct, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
