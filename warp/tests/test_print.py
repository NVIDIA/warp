# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import unittest

import warp as wp

from warp.tests.test_base import *


wp.init()


@wp.kernel
def test_print_kernel():
    wp.print(1.0)
    wp.print("this is a string")
    wp.printf("this is a float %f\n", 457.5)
    wp.printf("this is an int %d\n", 123)


def test_print(test, device):
    capture = StdOutCapture()
    capture.begin()
    wp.launch(kernel=test_print_kernel, dim=1, inputs=[], device=device)
    wp.synchronize()
    s = capture.end()

    test.assertRegex(
        s,
        rf"1{os.linesep}"
        rf"this is a string{os.linesep}"
        rf"this is a float 457\.500000{os.linesep}"
        rf"this is an int 123",
    )


def register(parent):
    devices = get_test_devices()
    devices = tuple(x for x in devices if not x.is_cpu)

    class TestPrint(parent):
        pass

    add_function_test(TestPrint, "test_print", test_print, devices=devices, check_output=False)
    return TestPrint


if __name__ == "__main__":
    wp.build.clear_kernel_cache()
    _ = register(unittest.TestCase)
    unittest.main(verbosity=2)
