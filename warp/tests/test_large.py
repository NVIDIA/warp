# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import unittest
from typing import Any

import warp as wp
from warp._src.types import check_array_shape
from warp.tests.unittest_utils import *


@wp.kernel
def conditional_sum(result: wp.array(dtype=wp.uint64)):
    i, _j, _k = wp.tid()

    if i == 0:
        wp.atomic_add(result, 0, wp.uint64(1))


def test_large_launch_large_kernel(test, device):
    """Test tid() on kernel launch of 2**33 threads.

    The function conditional sum will add 1 to result for every thread that has an i index of 0.
    Due to the size of the grid, this test is not run on CPUs
    """
    test_result = wp.zeros(shape=(1,), dtype=wp.uint64, device=device)

    large_dim_length = 2**16
    half_result = large_dim_length * large_dim_length

    wp.launch(kernel=conditional_sum, dim=[2, large_dim_length, large_dim_length], inputs=[test_result], device=device)
    test.assertEqual(test_result.numpy()[0], half_result)


@wp.kernel
def count_elements(result: wp.array(dtype=wp.uint64)):
    wp.atomic_add(result, 0, wp.uint64(1))


def test_large_launch_max_blocks(test, device):
    # Loop over 1000x1x1 elements using a grid of 256 threads
    test_result = wp.zeros(shape=(1,), dtype=wp.uint64, device=device)
    wp.launch(count_elements, (1000,), inputs=[test_result], max_blocks=1, device=device)
    test.assertEqual(test_result.numpy()[0], 1000)

    # Loop over 2x10x10 elements using a grid of 256 threads, using the tid() index to count half the elements
    test_result.zero_()
    wp.launch(
        conditional_sum,
        (
            2,
            50,
            10,
        ),
        inputs=[test_result],
        max_blocks=1,
        device=device,
    )
    test.assertEqual(test_result.numpy()[0], 500)


def test_large_launch_very_large_kernel(test, device):
    """Due to the size of the grid, this test is not run on CPUs"""

    # Dim is chosen to be larger than the maximum CUDA one-dimensional grid size (total threads)
    dim = (2**31 - 1) * 256 + 1
    test_result = wp.zeros(shape=(1,), dtype=wp.uint64, device=device)
    wp.launch(count_elements, (dim,), inputs=[test_result], device=device)
    test.assertEqual(test_result.numpy()[0], dim)


@wp.kernel
def check_array_equal_value_2d(data: wp.array2d(dtype=Any), expect: Any):
    i, j = wp.tid()
    wp.expect_eq(data[i, j], expect)


@wp.kernel
def check_array_equal_value_3d(data: wp.array3d(dtype=Any), expect: Any):
    i, j, k = wp.tid()
    wp.expect_eq(data[i, j, k], expect)


@wp.kernel
def check_array_equal_value_4d(data: wp.array4d(dtype=Any), expect: Any):
    i, j, k, l = wp.tid()
    wp.expect_eq(data[i, j, k, l], expect)


def test_large_arrays(test, device):
    # Exercises zero_/fill_/kernel-launch on arrays with >2**31 elements across
    # 2-D, 3-D, and 4-D shapes. int8 keeps each allocation near 2 GiB.
    ndim_kernels = (
        (2, check_array_equal_value_2d),
        (3, check_array_equal_value_3d),
        (4, check_array_equal_value_4d),
    )
    for ndim, kernel in ndim_kernels:
        dim_x = math.ceil((2**31) ** (1 / ndim))
        shape = (dim_x,) * ndim

        a1 = wp.zeros(shape, dtype=wp.int8, device=device)
        a1.fill_(127)
        wp.launch(kernel, shape, inputs=[a1, wp.int8(127)], device=device)

        a1.zero_()
        wp.launch(kernel, shape, inputs=[a1, wp.int8(0)], device=device)


def test_large_array_excessive_zeros(test, device):
    # Tests the allocation of an array with length exceeding 2**31-1 in a dimension

    with test.assertRaisesRegex(
        ValueError, "Array shapes must not exceed the maximum representable value of a signed 32-bit integer"
    ):
        _ = wp.zeros((2**31), dtype=int, device=device)


devices = get_test_devices()


class TestLarge(unittest.TestCase):
    def test_large_array_excessive_numpy(self):
        # Shape-validation is pure Python; no ndarray or device allocation
        # needed to exercise the 2**31-element boundary check. The user-facing
        # wp.array(ndarray) path reaches this same validator via
        # wp.array.__init__ -> _init_from_data -> _init_new -> check_array_shape;
        # the separate test below covers the integration path.
        with self.assertRaisesRegex(
            ValueError,
            "Array shapes must not exceed the maximum representable value of a signed 32-bit integer",
        ):
            check_array_shape((2**31,))

    def test_large_array_excessive_ndarray(self):
        # Exercise the wp.array(ndarray) -> _init_from_data -> _init_new path
        # at the 2**31 boundary without allocating a real 2**31-element buffer.
        # np.broadcast_to returns a zero-stride view over a single-element
        # source, and np.asarray preserves the view (no copy when dtypes
        # match). _init_new calls check_array_shape before any device
        # allocation, so the ValueError fires for free.
        large_view = np.broadcast_to(np.zeros(1, dtype=np.int8), (2**31,))
        with self.assertRaisesRegex(
            ValueError,
            "Array shapes must not exceed the maximum representable value of a signed 32-bit integer",
        ):
            _ = wp.array(large_view, dtype=wp.int8, device="cpu")


add_function_test(
    TestLarge,
    "test_large_launch_large_kernel",
    test_large_launch_large_kernel,
    devices=get_selected_cuda_test_devices(),
)

add_function_test(TestLarge, "test_large_launch_max_blocks", test_large_launch_max_blocks, devices=devices)
add_function_test(
    TestLarge,
    "test_large_launch_very_large_kernel",
    test_large_launch_very_large_kernel,
    devices=get_selected_cuda_test_devices(),
)

add_function_test(TestLarge, "test_large_arrays", test_large_arrays, devices=devices)
add_function_test(TestLarge, "test_large_array_excessive_zeros", test_large_array_excessive_zeros, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
