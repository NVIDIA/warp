# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import unittest

import warp as wp
from warp.tests.unittest_utils import *


@wp.kernel
def test_pow(e: float, result: float):
    tid = wp.tid()

    y = wp.pow(-2.0, e)

    wp.expect_eq(y, result)


def test_fast_math_disabled(test, device):
    # on all systems pow() should handle negative base correctly with fast math off
    wp.set_module_options({"fast_math": False})
    wp.launch(test_pow, dim=1, inputs=[2.0, 4.0], device=device)


@unittest.expectedFailure
def test_fast_math_cuda(test, device):
    # on CUDA with --fast-math enabled taking the pow()
    # of a negative number will result in a NaN

    wp.set_module_options({"fast_math": True})
    try:
        wp.launch(test_pow, dim=1, inputs=[2.0, 4.0], device=device)
    finally:
        # Turn fast math back off
        wp.set_module_options({"fast_math": False})


class TestFastMath(unittest.TestCase):
    def test_fast_math_cpu(self):
        # on all systems pow() should handle negative base correctly
        wp.set_module_options({"fast_math": True})
        try:
            wp.launch(test_pow, dim=1, inputs=[2.0, 4.0], device="cpu")
        finally:
            wp.set_module_options({"fast_math": False})


devices = get_test_devices()

add_function_test(TestFastMath, "test_fast_math_cuda", test_fast_math_cuda, devices=get_cuda_test_devices())
add_function_test(TestFastMath, "test_fast_math_disabled", test_fast_math_disabled, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
