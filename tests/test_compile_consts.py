import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import test_base

import warp as wp

import test_compile_consts_dummy

wp.config.mode = "release"
wp.config.cache_kernels = False

wp.init()


device = "cuda"

LOCAL_ONE = wp.constant(1)

class Foobar:
    ONE = wp.constant(1)
    TWO = wp.constant(2)

@wp.kernel
def test_constants(x: int):
    if Foobar.ONE > 0:
        x = 123 + Foobar.TWO + test_compile_consts_dummy.MINUS_ONE
    else:
        x = 456 + LOCAL_ONE
    expect_eq(x, 124)

x = 0

class TestConstants(test_base.TestBase):
    pass

TestConstants.add_kernel_test(test_constants, dim=1, inputs=[x], devices=["cpu", "cuda"])

if __name__ == '__main__':
    unittest.main(verbosity=2)
