# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# include parent path
import numpy as np

import warp as wp
from warp.tests.test_base import *

import unittest

wp.init()


@wp.kernel
def mul_1d(a: wp.array1d(dtype=float), s: float):
    i = wp.tid()
    a[i] = a[i] * s


@wp.kernel
def mul_2d(a: wp.array2d(dtype=float), s: float):
    i, j = wp.tid()
    a[i, j] = a[i, j] * s


@wp.kernel
def mul_3d(a: wp.array3d(dtype=float), s: float):
    i, j, k = wp.tid()
    a[i, j, k] = a[i, j, k] * s


@wp.kernel
def mul_4d(a: wp.array4d(dtype=float), s: float):
    i, j, k, l = wp.tid()
    a[i, j, k, l] = a[i, j, k, l] * s


def test_copy_strided(test, device):
    with wp.ScopedDevice(device):
        np_data1 = np.arange(10, dtype=np.float32)
        np_data2 = np.arange(100, dtype=np.float32).reshape((10, 10))
        np_data3 = np.arange(1000, dtype=np.float32).reshape((10, 10, 10))
        np_data4 = np.arange(10000, dtype=np.float32).reshape((10, 10, 10, 10))

        wp_data1 = wp.array(data=np_data1, copy=True)
        wp_data2 = wp.array(data=np_data2, copy=True)
        wp_data3 = wp.array(data=np_data3, copy=True)
        wp_data4 = wp.array(data=np_data4, copy=True)

        expected1 = np_data1[1::2]
        expected2 = np_data2[1::2, 1::2]
        expected3 = np_data3[1::2, 1::2, 1::2]
        expected4 = np_data4[1::2, 1::2, 1::2, 1::2]

        a1 = wp_data1[1::2]
        a2 = wp_data2[1::2, 1::2]
        a3 = wp_data3[1::2, 1::2, 1::2]
        a4 = wp_data4[1::2, 1::2, 1::2, 1::2]

        assert_np_equal(a1.numpy(), expected1)
        assert_np_equal(a2.numpy(), expected2)
        assert_np_equal(a3.numpy(), expected3)
        assert_np_equal(a4.numpy(), expected4)

        b1 = wp.zeros_like(a1)
        b2 = wp.zeros_like(a2)
        b3 = wp.zeros_like(a3)
        b4 = wp.zeros_like(a4)

        test.assertFalse(a1.is_contiguous)
        test.assertFalse(a2.is_contiguous)
        test.assertFalse(a3.is_contiguous)
        test.assertFalse(a4.is_contiguous)

        test.assertTrue(b1.is_contiguous)
        test.assertTrue(b2.is_contiguous)
        test.assertTrue(b3.is_contiguous)
        test.assertTrue(b4.is_contiguous)

        # copy non-contiguous to contiguous
        wp.copy(b1, a1)
        wp.copy(b2, a2)
        wp.copy(b3, a3)
        wp.copy(b4, a4)

        assert_np_equal(a1.numpy(), b1.numpy())
        assert_np_equal(a2.numpy(), b2.numpy())
        assert_np_equal(a3.numpy(), b3.numpy())
        assert_np_equal(a4.numpy(), b4.numpy())

        s = 2.0

        wp.launch(mul_1d, dim=b1.shape, inputs=[b1, s])
        wp.launch(mul_2d, dim=b2.shape, inputs=[b2, s])
        wp.launch(mul_3d, dim=b3.shape, inputs=[b3, s])
        wp.launch(mul_4d, dim=b4.shape, inputs=[b4, s])

        # copy contiguous to non-contiguous
        wp.copy(a1, b1)
        wp.copy(a2, b2)
        wp.copy(a3, b3)
        wp.copy(a4, b4)

        assert_np_equal(a1.numpy(), b1.numpy())
        assert_np_equal(a2.numpy(), b2.numpy())
        assert_np_equal(a3.numpy(), b3.numpy())
        assert_np_equal(a4.numpy(), b4.numpy())

        assert_np_equal(a1.numpy(), expected1 * s)
        assert_np_equal(a2.numpy(), expected2 * s)
        assert_np_equal(a3.numpy(), expected3 * s)
        assert_np_equal(a4.numpy(), expected4 * s)


def test_copy_indexed(test, device):
    with wp.ScopedDevice(device):
        np_data1 = np.arange(10, dtype=np.float32)
        np_data2 = np.arange(100, dtype=np.float32).reshape((10, 10))
        np_data3 = np.arange(1000, dtype=np.float32).reshape((10, 10, 10))
        np_data4 = np.arange(10000, dtype=np.float32).reshape((10, 10, 10, 10))

        wp_data1 = wp.array(data=np_data1, copy=True)
        wp_data2 = wp.array(data=np_data2, copy=True)
        wp_data3 = wp.array(data=np_data3, copy=True)
        wp_data4 = wp.array(data=np_data4, copy=True)

        np_indices = np.array([1, 5, 8, 9])
        wp_indices = wp.array(data=np_indices, dtype=wp.int32)

        # Note: Indexing using multiple index arrays works differently
        #       in Numpy and Warp, so the syntax is different.

        expected1 = np_data1[np_indices]
        expected2 = np_data2[np_indices][:, np_indices]
        expected3 = np_data3[np_indices][:, np_indices][:, :, np_indices]
        expected4 = np_data4[np_indices][:, np_indices][:, :, np_indices][:, :, :, np_indices]

        a1 = wp_data1[wp_indices]
        a2 = wp_data2[wp_indices, wp_indices]
        a3 = wp_data3[wp_indices, wp_indices, wp_indices]
        a4 = wp_data4[wp_indices, wp_indices, wp_indices, wp_indices]

        assert_np_equal(a1.numpy(), expected1)
        assert_np_equal(a2.numpy(), expected2)
        assert_np_equal(a3.numpy(), expected3)
        assert_np_equal(a4.numpy(), expected4)

        b1 = wp.zeros_like(a1)
        b2 = wp.zeros_like(a2)
        b3 = wp.zeros_like(a3)
        b4 = wp.zeros_like(a4)

        test.assertFalse(a1.is_contiguous)
        test.assertFalse(a2.is_contiguous)
        test.assertFalse(a3.is_contiguous)
        test.assertFalse(a4.is_contiguous)

        test.assertTrue(b1.is_contiguous)
        test.assertTrue(b2.is_contiguous)
        test.assertTrue(b3.is_contiguous)
        test.assertTrue(b4.is_contiguous)

        # copy non-contiguous to contiguous
        wp.copy(b1, a1)
        wp.copy(b2, a2)
        wp.copy(b3, a3)
        wp.copy(b4, a4)

        assert_np_equal(a1.numpy(), b1.numpy())
        assert_np_equal(a2.numpy(), b2.numpy())
        assert_np_equal(a3.numpy(), b3.numpy())
        assert_np_equal(a4.numpy(), b4.numpy())

        s = 2.0

        wp.launch(mul_1d, dim=b1.shape, inputs=[b1, s])
        wp.launch(mul_2d, dim=b2.shape, inputs=[b2, s])
        wp.launch(mul_3d, dim=b3.shape, inputs=[b3, s])
        wp.launch(mul_4d, dim=b4.shape, inputs=[b4, s])

        # copy contiguous to non-contiguous
        wp.copy(a1, b1)
        wp.copy(a2, b2)
        wp.copy(a3, b3)
        wp.copy(a4, b4)

        assert_np_equal(a1.numpy(), b1.numpy())
        assert_np_equal(a2.numpy(), b2.numpy())
        assert_np_equal(a3.numpy(), b3.numpy())
        assert_np_equal(a4.numpy(), b4.numpy())

        assert_np_equal(a1.numpy(), expected1 * s)
        assert_np_equal(a2.numpy(), expected2 * s)
        assert_np_equal(a3.numpy(), expected3 * s)
        assert_np_equal(a4.numpy(), expected4 * s)


def register(parent):
    devices = get_test_devices()

    class TestCopy(parent):
        pass

    add_function_test(TestCopy, "test_copy_strided", test_copy_strided, devices=devices)
    add_function_test(TestCopy, "test_copy_indexed", test_copy_indexed, devices=devices)

    return TestCopy


if __name__ == "__main__":
    c = register(unittest.TestCase)
    unittest.main(verbosity=2)
