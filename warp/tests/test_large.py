# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import math
import unittest

import warp as wp
from warp.tests.unittest_utils import *

wp.init()


@wp.kernel
def conditional_sum(result: wp.array(dtype=wp.uint64)):
    i, j, k = wp.tid()

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


def test_large_arrays_slow(test, device):
    # The goal of this test is to use arrays just large enough to know
    # if there's a flaw in handling arrays with more than 2**31-1 elements
    # Unfortunately, it takes a long time to run so it won't be run automatically
    # without changes to support how frequently a test may be run
    total_elements = 2**31 + 8

    # 1-D to 4-D arrays: test zero_, fill_, then zero_ for scalar data types:
    for total_dims in range(1, 5):
        dim_x = math.ceil(total_elements ** (1 / total_dims))
        shape_tuple = tuple([dim_x] * total_dims)

        for nptype, wptype in wp.types.np_dtype_to_warp_type.items():
            a1 = wp.zeros(shape_tuple, dtype=wptype, device=device)
            assert_np_equal(a1.numpy(), np.zeros_like(a1.numpy()))

            a1.fill_(127)
            assert_np_equal(a1.numpy(), 127 * np.ones_like(a1.numpy()))

            a1.zero_()
            assert_np_equal(a1.numpy(), np.zeros_like(a1.numpy()))


def test_large_arrays_fast(test, device):
    # A truncated version of test_large_arrays_slow meant to catch basic errors
    total_elements = 2**31 + 8

    nptype = np.dtype(np.int8)
    wptype = wp.types.np_dtype_to_warp_type[nptype]

    a1 = wp.zeros((total_elements,), dtype=wptype, device=device)
    assert_np_equal(a1.numpy(), np.zeros_like(a1.numpy()))

    a1.fill_(127)
    assert_np_equal(a1.numpy(), 127 * np.ones_like(a1.numpy()))

    a1.zero_()
    assert_np_equal(a1.numpy(), np.zeros_like(a1.numpy()))


devices = get_test_devices()


class TestLarge(unittest.TestCase):
    pass


add_function_test(
    TestLarge, "test_large_launch_large_kernel", test_large_launch_large_kernel, devices=get_unique_cuda_test_devices()
)

add_function_test(TestLarge, "test_large_launch_max_blocks", test_large_launch_max_blocks, devices=devices)
add_function_test(
    TestLarge,
    "test_large_launch_very_large_kernel",
    test_large_launch_very_large_kernel,
    devices=get_unique_cuda_test_devices(),
)

add_function_test(TestLarge, "test_large_arrays_fast", test_large_arrays_fast, devices=devices)


if __name__ == "__main__":
    wp.build.clear_kernel_cache()
    unittest.main(verbosity=2)
