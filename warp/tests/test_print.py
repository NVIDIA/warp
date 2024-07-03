# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import sys
import unittest

import warp as wp
from warp.tests.unittest_utils import *


@wp.kernel
def test_print_kernel():
    wp.print(1.0)
    wp.print("this is a string")
    wp.printf("this is a float %f\n", 457.5)
    wp.printf("this is an int %d\n", 123)


def test_print(test, device):
    wp.load_module(device=device)
    capture = StdOutCapture()
    capture.begin()
    wp.launch(kernel=test_print_kernel, dim=1, inputs=[], device=device)
    wp.synchronize_device(device)
    s = capture.end()

    # We skip the win32 comparison for now since the capture sometimes is an empty string
    if sys.platform != "win32":
        test.assertRegex(
            s,
            rf"1{os.linesep}"
            rf"this is a string{os.linesep}"
            rf"this is a float 457\.500000{os.linesep}"
            rf"this is an int 123",
        )


class TestPrint(unittest.TestCase):
    pass


devices = get_test_devices()
add_function_test(TestPrint, "test_print", test_print, devices=devices, check_output=False)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
