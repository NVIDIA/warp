# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import unittest

import warp as wp
from warp.tests.unittest_utils import *


def setUpModule():
    wp.config.verify_fp = True  # Enable checking floating-point values to be finite


def tearDownModule():
    wp.config.verify_fp = False


@wp.struct
class TestStruct:
    field: wp.float32


@wp.kernel(enable_backward=False)
def finite_kernel(foos: wp.array[TestStruct]):
    i = wp.tid()
    foos[i].field += wp.float32(1.0)


def test_finite(test, device):
    foos = wp.zeros((10,), dtype=TestStruct, device=device)

    wp.launch(
        kernel=finite_kernel,
        dim=(10,),
        inputs=[foos],
        device=device,
    )
    wp.synchronize()

    expected = TestStruct()
    expected.field = 1.0
    for f in foos.list():
        if f.field != expected.field:
            raise AssertionError(f"Unexpected result, got: {f} expected: {expected}")


@wp.kernel(enable_backward=False)
def slot_atomic_add_overflow_kernel(foos: wp.array[TestStruct]):
    foos[0].field += wp.float32(3.0e38)


@wp.kernel(enable_backward=False)
def slot_atomic_sub_overflow_kernel(foos: wp.array[TestStruct]):
    foos[0].field -= wp.float32(3.0e38)


def test_verify_fp_reports_composite_slot_overflow(test, device):
    """Report non-finite values produced by composite slot atomics."""
    if wp.config.mode == "debug":
        test.skipTest("verify_fp asserts before infinity warning checks in debug mode")

    if sys.platform == "win32":
        test.skipTest("Skipping test on Windows due to unreliable stdout capture")

    for op, kernel, initial in (
        ("add", slot_atomic_add_overflow_kernel, 3.0e38),
        ("sub", slot_atomic_sub_overflow_kernel, -3.0e38),
    ):
        with test.subTest(op=op):
            foo = TestStruct()
            foo.field = initial
            foos = wp.array([foo], dtype=TestStruct, device=device)

            capture = StdOutCapture()
            capture.begin()
            wp.launch(kernel=kernel, dim=1, inputs=[foos], device=device)
            wp.synchronize_device(device)
            output = capture.end()

            test.assertRegex(output, r"inf")


@wp.kernel(enable_backward=False)
def nan_kernel(foos: wp.array[TestStruct]):
    i = wp.tid()
    foos[i].field /= wp.float32(0.0)  # Division by zero produces Not-a-Number (NaN)


def test_nan(test, device):
    if wp.config.mode == "debug":
        test.skipTest("verify_fp asserts before NaN warning checks in debug mode")

    if sys.platform == "win32":
        test.skipTest("Skipping test on Windows due to unreliable stdout capture")

    foos = wp.zeros((10,), dtype=TestStruct, device=device)

    capture = StdOutCapture()
    capture.begin()

    wp.launch(
        kernel=nan_kernel,
        dim=(10,),
        inputs=[foos],
        device=device,
    )
    wp.synchronize()

    output = capture.end()

    # Check that the output contains warnings about "nan" being produced.
    test.assertRegex(output, r"nan")


devices = get_test_devices()


class TestVerifyFP(unittest.TestCase):
    pass


add_function_test(TestVerifyFP, "test_finite", test_finite, devices=devices)
add_function_test(
    TestVerifyFP,
    "test_verify_fp_reports_composite_slot_overflow",
    test_verify_fp_reports_composite_slot_overflow,
    devices=devices,
    check_output=False,
)
add_function_test(TestVerifyFP, "test_nan", test_nan, devices=devices, check_output=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
