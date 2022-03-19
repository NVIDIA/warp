# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys

import unittest
from warp.tests.test_base import *
import warp.tests.test_compile_consts_dummy

import warp as wp

wp.init()

LOCAL_ONE = wp.constant(1)

SQRT3_OVER_3 = wp.constant(0.57735026919)
UNIT_VEC = wp.constant(wp.vec3(SQRT3_OVER_3.val, SQRT3_OVER_3.val, SQRT3_OVER_3.val))

class Foobar:
    ONE = wp.constant(1)
    TWO = wp.constant(2)

@wp.kernel
def test_constants_int(a: int):
    if Foobar.ONE > 0:
        a = 123 + Foobar.TWO + warp.tests.test_compile_consts_dummy.MINUS_ONE
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


def register(parent):

    class TestConstants(parent):
        pass

    a = 0
    x = 0.0        

    add_kernel_test(TestConstants, test_constants_int, dim=1, inputs=[a], devices=wp.get_devices())
    add_kernel_test(TestConstants, test_constants_float, dim=1, inputs=[x], devices=wp.get_devices())

    return TestConstants

if __name__ == '__main__':
    c = register(unittest.TestCase)
    unittest.main(verbosity=2)
