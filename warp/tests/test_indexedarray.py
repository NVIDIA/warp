# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# include parent path
import numpy as np
from typing import Any

import warp as wp
from warp.tests.test_base import *

import unittest

wp.init()


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

    iarr = wp.indexedarray(arr, [indices])

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

    iarr = wp.indexedarray(arr, [indices0, indices1])

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

    iarr = wp.indexedarray(arr, [indices0, indices1, indices2])

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

    iarr = wp.indexedarray(arr, [indices0, indices1, indices2, indices3])

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


def register(parent):
    devices = get_test_devices()

    class TestIndexedArray(parent):
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

    return TestIndexedArray


if __name__ == "__main__":
    c = register(unittest.TestCase)
    unittest.main(verbosity=2)
