# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import unittest

import warp as wp
import warp.tests.aux_test_compile_consts_dummy
from warp.tests.unittest_utils import *

wp.init()

LOCAL_ONE = wp.constant(1)

SQRT3_OVER_3 = wp.constant(0.57735026919)
UNIT_VEC = wp.constant(wp.vec3(SQRT3_OVER_3, SQRT3_OVER_3, SQRT3_OVER_3))
ONE_FP16 = wp.constant(wp.float16(1.0))
TEST_BOOL = wp.constant(True)


class Foobar:
    ONE = wp.constant(1)
    TWO = wp.constant(2)


@wp.kernel
def test_constants_bool():
    if TEST_BOOL:
        expect_eq(1.0, 1.0)
    else:
        expect_eq(1.0, -1.0)


@wp.kernel
def test_constants_int(a: int):
    if Foobar.ONE > 0:
        a = 123 + Foobar.TWO + warp.tests.aux_test_compile_consts_dummy.MINUS_ONE
    else:
        a = 456 + LOCAL_ONE
    expect_eq(a, 124)


@wp.kernel
def test_constants_float(x: float):
    x = SQRT3_OVER_3
    for i in range(3):
        expect_eq(UNIT_VEC[i], x)

    approx_one = wp.dot(UNIT_VEC, UNIT_VEC)
    expect_near(approx_one, 1.0, 1e-6)

    # test casting
    expect_near(wp.float32(ONE_FP16), 1.0, 1e-6)


def test_constant_closure_capture(test, device):
    def make_closure_kernel(cst):
        def closure_kernel_fn(expected: int):
            wp.expect_eq(cst, expected)

        return wp.Kernel(func=closure_kernel_fn)

    one_closure = make_closure_kernel(Foobar.ONE)
    two_closure = make_closure_kernel(Foobar.TWO)

    wp.launch(one_closure, dim=(1), inputs=[1], device=device)
    wp.launch(two_closure, dim=(1), inputs=[2], device=device)


class TestConstants(unittest.TestCase):
    def test_constant_math(self):
        # test doing math with python defined constants in *python* scope
        twopi = wp.pi * 2.0

        import math

        self.assertEqual(twopi, math.pi * 2.0)


a = 0
x = 0.0

devices = get_test_devices()

add_kernel_test(TestConstants, test_constants_bool, dim=1, inputs=[], devices=devices)
add_kernel_test(TestConstants, test_constants_int, dim=1, inputs=[a], devices=devices)
add_kernel_test(TestConstants, test_constants_float, dim=1, inputs=[x], devices=devices)

add_function_test(TestConstants, "test_constant_closure_capture", test_constant_closure_capture, devices=devices)


if __name__ == "__main__":
    wp.build.clear_kernel_cache()
    unittest.main(verbosity=2)
