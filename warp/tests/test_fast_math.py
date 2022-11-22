import warp as wp
import numpy as np

import unittest

import warp as wp
from warp.tests.test_base import *


wp.init()

@wp.kernel
def test_pow(x: float, e: float, result: float):

    tid = wp.tid()

    y = wp.pow(-2.0, e)

    wp.expect_eq(y, result)


def test_fast_math(test, device):
        
    # on all systems pow() should handle negative base correctly
    wp.set_module_options({"fast_math": False})
    wp.launch(test_pow, dim=1, inputs=[-2.0, 2.0, 4.0], device=device)

    # on CUDA with --fast-math enabled taking the pow()
    # of a negative number will result in a NaN
    if wp.get_device(device).is_cuda:
        wp.set_module_options({"fast_math": True})
        
        with test.assertRaises(Exception):
            with CheckOutput():
                wp.launch(test_pow, dim=1, inputs=[-2.0, 2.0, 2.0], device=device)


def register(parent):

    class TestFastMath(parent):
        pass
    
    devices = wp.get_devices()

    add_function_test(TestFastMath, "test_fast_math", test_fast_math, devices=devices)

    return TestFastMath


if __name__ == '__main__':
    c = register(unittest.TestCase)
    unittest.main(verbosity=2)


