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

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


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


def test_copy_adjoint(test, device):
    state_in = wp.from_numpy(
        np.array([1.0, 2.0, 3.0]).astype(np.float32), dtype=wp.float32, requires_grad=True, device=device
    )
    state_out = wp.zeros(state_in.shape, dtype=wp.float32, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        wp.copy(state_out, state_in)

    grads = {state_out: wp.from_numpy(np.array([1.0, 1.0, 1.0]).astype(np.float32), dtype=wp.float32, device=device)}
    tape.backward(grads=grads)

    assert_np_equal(state_in.grad.numpy(), np.array([1.0, 1.0, 1.0]).astype(np.float32))


devices = get_test_devices()


class TestCopy(unittest.TestCase):
    pass


add_function_test(TestCopy, "test_copy_strided", test_copy_strided, devices=devices)
add_function_test(TestCopy, "test_copy_indexed", test_copy_indexed, devices=devices)
add_function_test(TestCopy, "test_copy_adjoint", test_copy_adjoint, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
