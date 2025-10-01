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

import gc
import importlib
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


@wp.kernel
def all_equal_kernel(a: wp.array(dtype=float), value: float, result: wp.array(dtype=int)):
    tid = wp.tid()
    wp.atomic_min(result, 0, int(a[tid] == value))


def assert_all_equal(a: wp.array(dtype=float), value: float):
    result = wp.ones(1, dtype=int, device=a.device)
    wp.launch(all_equal_kernel, dim=a.shape, inputs=[a, value, result], device=a.device)
    assert result.numpy()[0] == 1


def test_copy_strided(test, _, device1, device2):
    np_data1 = np.arange(10, dtype=np.float32)
    np_data2 = np.arange(100, dtype=np.float32).reshape((10, 10))
    np_data3 = np.arange(1000, dtype=np.float32).reshape((10, 10, 10))
    np_data4 = np.arange(10000, dtype=np.float32).reshape((10, 10, 10, 10))

    wp_data1 = wp.array(data=np_data1, copy=True, device=device1)
    wp_data2 = wp.array(data=np_data2, copy=True, device=device1)
    wp_data3 = wp.array(data=np_data3, copy=True, device=device1)
    wp_data4 = wp.array(data=np_data4, copy=True, device=device1)

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

    b1 = wp.zeros_like(a1, device=device2)
    b2 = wp.zeros_like(a2, device=device2)
    b3 = wp.zeros_like(a3, device=device2)
    b4 = wp.zeros_like(a4, device=device2)

    test.assertFalse(a1.is_contiguous)
    test.assertFalse(a2.is_contiguous)
    test.assertFalse(a3.is_contiguous)
    test.assertFalse(a4.is_contiguous)

    test.assertTrue(b1.is_contiguous)
    test.assertTrue(b2.is_contiguous)
    test.assertTrue(b3.is_contiguous)
    test.assertTrue(b4.is_contiguous)

    # copy non-contiguous to contiguous
    wp.synchronize_device(device1)
    wp.copy(b1, a1)
    wp.copy(b2, a2)
    wp.copy(b3, a3)
    wp.copy(b4, a4)

    assert_np_equal(a1.numpy(), b1.numpy())
    assert_np_equal(a2.numpy(), b2.numpy())
    assert_np_equal(a3.numpy(), b3.numpy())
    assert_np_equal(a4.numpy(), b4.numpy())

    s = 2.0

    wp.launch(mul_1d, dim=b1.shape, inputs=[b1, s], device=device2)
    wp.launch(mul_2d, dim=b2.shape, inputs=[b2, s], device=device2)
    wp.launch(mul_3d, dim=b3.shape, inputs=[b3, s], device=device2)
    wp.launch(mul_4d, dim=b4.shape, inputs=[b4, s], device=device2)

    # copy contiguous to non-contiguous
    wp.synchronize_device(device2)
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


def test_copy_indexed(test, _, device1, device2):
    np_data1 = np.arange(10, dtype=np.float32)
    np_data2 = np.arange(100, dtype=np.float32).reshape((10, 10))
    np_data3 = np.arange(1000, dtype=np.float32).reshape((10, 10, 10))
    np_data4 = np.arange(10000, dtype=np.float32).reshape((10, 10, 10, 10))

    wp_data1 = wp.array(data=np_data1, copy=True, device=device1)
    wp_data2 = wp.array(data=np_data2, copy=True, device=device1)
    wp_data3 = wp.array(data=np_data3, copy=True, device=device1)
    wp_data4 = wp.array(data=np_data4, copy=True, device=device1)

    np_indices = np.array([1, 5, 8, 9])
    wp_indices = wp.array(data=np_indices, dtype=wp.int32, device=device1)

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

    b1 = wp.zeros_like(a1, device=device2)
    b2 = wp.zeros_like(a2, device=device2)
    b3 = wp.zeros_like(a3, device=device2)
    b4 = wp.zeros_like(a4, device=device2)

    test.assertFalse(a1.is_contiguous)
    test.assertFalse(a2.is_contiguous)
    test.assertFalse(a3.is_contiguous)
    test.assertFalse(a4.is_contiguous)

    test.assertTrue(b1.is_contiguous)
    test.assertTrue(b2.is_contiguous)
    test.assertTrue(b3.is_contiguous)
    test.assertTrue(b4.is_contiguous)

    # copy non-contiguous to contiguous
    wp.synchronize_device(device1)
    wp.copy(b1, a1)
    wp.copy(b2, a2)
    wp.copy(b3, a3)
    wp.copy(b4, a4)

    assert_np_equal(a1.numpy(), b1.numpy())
    assert_np_equal(a2.numpy(), b2.numpy())
    assert_np_equal(a3.numpy(), b3.numpy())
    assert_np_equal(a4.numpy(), b4.numpy())

    s = 2.0

    wp.launch(mul_1d, dim=b1.shape, inputs=[b1, s], device=device2)
    wp.launch(mul_2d, dim=b2.shape, inputs=[b2, s], device=device2)
    wp.launch(mul_3d, dim=b3.shape, inputs=[b3, s], device=device2)
    wp.launch(mul_4d, dim=b4.shape, inputs=[b4, s], device=device2)

    # copy contiguous to non-contiguous
    wp.synchronize_device(device2)
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


def test_copy_large_stride(test, _, device1, device2):
    # NOTE: This test can use ~8GB of memory. To prevent errors, we skip if there is insufficient memory.

    gc.collect()

    if device1.is_cpu or device2.is_cpu:
        if importlib.util.find_spec("psutil") is None:
            test.skipTest("The 'psutil' package is required to check available memory")

    if device1.free_memory < 12e9 or device2.free_memory < 12e9:
        test.skipTest("Insufficient free memory")

    N = 500_000_000

    a_data = wp.empty((N, 2), dtype=wp.float32, device=device1)
    a = a_data[:, 0]  # array with a large stride (offsets > 32 bits)

    b = wp.empty_like(a, device=device2)

    a.fill_(1)
    b.fill_(2)

    # NOTE: use a memory-efficient check to avoid running out of memory
    assert_all_equal(a, 1)
    assert_all_equal(b, 2)

    # copy to large-strided array
    wp.synchronize_device(device2)
    wp.copy(a, b)
    assert_all_equal(a, 2)

    a.fill_(1)

    # copy from large-strided array
    wp.synchronize_device(device1)
    wp.copy(b, a)
    assert_all_equal(b, 1)


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


for src_device in devices:
    src_name = "cpu" if src_device.is_cpu else f"cuda{src_device.ordinal}"
    for dst_device in devices:
        dst_name = "cpu" if dst_device.is_cpu else f"cuda{dst_device.ordinal}"
        add_function_test(
            TestCopy,
            f"test_copy_strided_{src_name}_{dst_name}",
            test_copy_strided,
            devices=None,
            device1=src_device,
            device2=dst_device,
        )
        add_function_test(
            TestCopy,
            f"test_copy_indexed_{src_name}_{dst_name}",
            test_copy_indexed,
            devices=None,
            device1=src_device,
            device2=dst_device,
        )
        add_function_test(
            TestCopy,
            f"test_copy_large_stride_{src_name}_{dst_name}",
            test_copy_large_stride,
            devices=None,
            device1=src_device,
            device2=dst_device,
        )

add_function_test(TestCopy, "test_copy_adjoint", test_copy_adjoint, devices=devices)

if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
