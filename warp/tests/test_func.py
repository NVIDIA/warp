# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# include parent path
import numpy as np
import math

import warp as wp
from warp.tests.test_base import *

import unittest

wp.init()

@wp.func
def sqr(x: float):
    return x*x

# test nested user function calls
# and explicit return type hints
@wp.func
def cube(x: float) -> float:
    return sqr(x)*x


@wp.func
def custom(x: int):
    return x + 1

@wp.func
def custom(x: float):
    return x + 1.0

@wp.func
def custom(x: wp.vec3):
    return x + wp.vec3(1.0, 0.0, 0.0)


@wp.kernel
def test_overload_func():
    # tests overloading a custom @wp.func


    i = custom(1)
    f = custom(1.0)
    v = custom(wp.vec3(1.0, 0.0, 0.0))

    wp.expect_eq(i, 2)
    wp.expect_eq(f, 2.0)
    wp.expect_eq(v, wp.vec3(2.0, 0.0, 0.0))



def test_func_export(test, device):
    # tests calling native functions from Python
    
    q = wp.quat_identity()
    assert_np_equal(np.array([*q]), np.array([0.0, 0.0, 0.0, 1.0]))

    r = wp.quat_from_axis_angle((1.0, 0.0, 0.0), 2.0)
    assert_np_equal(np.array([*r]), np.array([0.8414709568023682, 0.0, 0.0, 0.5403022170066833]), tol=1.e-3)

    q = wp.quat(1.0, 2.0, 3.0, 4.0)
    q = wp.normalize(q)
    assert_np_equal(np.array([*q]), np.array([0.18257418274879456, 0.3651483654975891, 0.547722578048706, 0.7302967309951782]), tol=1.e-3)

    v2 = wp.vec2(1.0, 2.0)
    v2 = wp.normalize(v2)
    assert_np_equal(np.array([*v2]), np.array([0.4472135901451111, 0.8944271802902222]), tol=1.e-3)

    v3 = wp.vec3(1.0, 2.0, 3.0)
    v3 = wp.normalize(v3)
    assert_np_equal(np.array([*v3]), np.array([0.26726123690605164, 0.5345224738121033, 0.8017836809158325]), tol=1.e-3)

    v4 = wp.vec4(1.0, 2.0, 3.0, 4.0)
    v4 = wp.normalize(v4)
    assert_np_equal(np.array([*v4]), np.array([0.18257418274879456, 0.3651483654975891, 0.547722578048706, 0.7302967309951782]), tol=1.e-3)

    t = wp.transform_identity()
    assert_np_equal(np.array([*t]), np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))

    f = wp.sin(math.pi*0.5)
    test.assertAlmostEqual(f, 1.0, places=3)



def register(parent):

    devices = wp.get_devices()

    class TestFunc(parent):
        pass

    add_kernel_test(TestFunc, kernel=test_overload_func, name="test_overload_func", dim=1, devices=devices)
    add_function_test(TestFunc, func=test_func_export, name="test_func_export", devices=["cpu"])

    return TestFunc

if __name__ == '__main__':
    c = register(unittest.TestCase)
    #unittest.main(verbosity=2)

    wp.force_load()
    
    loader = unittest.defaultTestLoader
    testSuite = loader.loadTestsFromTestCase(c)
    testSuite.debug()


