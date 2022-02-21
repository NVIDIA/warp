import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import test_base

import warp as wp

import test_compile_consts_dummy

wp.config.cache_kernels = False

wp.init()

LOCAL_ONE = wp.constant(1)

SQRT3_OVER_3 = wp.constant(0.57735026919)
UNIT_VEC = wp.constant(wp.vec3(SQRT3_OVER_3.val, SQRT3_OVER_3.val, SQRT3_OVER_3.val))

class Foobar:
    ONE = wp.constant(1)
    TWO = wp.constant(2)

@wp.kernel
def test_constants0(a: int):
    if Foobar.ONE > 0:
        a = 123 + Foobar.TWO + test_compile_consts_dummy.MINUS_ONE
    else:
        a = 456 + LOCAL_ONE
    expect_eq(a, 124)

@wp.kernel
def test_constants1(x: float):
    x = SQRT3_OVER_3
    for i in range(3):
        expect_eq(UNIT_VEC[i], x)

    approx_one = wp.dot(UNIT_VEC, UNIT_VEC)
    expect_near(approx_one, 1.0, 1e-6)

a = 0
x = 0.0

class TestConstants(test_base.TestBase):
    pass

TestConstants.add_kernel_test(test_constants0, dim=1, inputs=[a], devices=["cpu", "cuda"])
TestConstants.add_kernel_test(test_constants1, dim=1, inputs=[x], devices=["cpu", "cuda"])

if __name__ == '__main__':
    unittest.main(verbosity=2)
