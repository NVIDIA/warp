# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# include parent path
import os
import sys
import numpy as np
import math
import ctypes

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import test_base

import warp as wp

wp.init()

@wp.kernel
def test_operators_scalar():

    a = 1.0
    b = 2.0

    c = a*b
    d = a+b
    e = a/b
    f = a-b

    expect_eq(c, 2.0)
    expect_eq(d, 3.0)
    expect_eq(e, 0.5)
    expect_eq(f, -1.0)
    


@wp.kernel
def test_operators_vec3():
    
    v = vec3(1.0, 2.0, 3.0)

    r0 = v*3.0
    r1 = 3.0*v

    expect_eq(r0, vec3(3.0, 6.0, 9.0))
    expect_eq(r1, vec3(3.0, 6.0, 9.0))

    col0 = vec3(1.0, 0.0, 0.0)
    col1 = vec3(0.0, 2.0, 0.0)
    col2 = vec3(0.0, 0.0, 3.0)

    m = mat33(col0, col1, col2)

    expect_eq(m*vec3(1.0, 0.0, 0.0), col0)
    expect_eq(m*vec3(0.0, 1.0, 0.0), col1)
    expect_eq(m*vec3(0.0, 0.0, 1.0), col2)

    two = vec3(1.0)*2.0
    expect_eq(two, vec3(2.0, 2.0, 2.0))

@wp.kernel
def test_operators_vec4():
    
    v = vec4(1.0, 2.0, 3.0, 4.0)

    r0 = v*3.0
    r1 = 3.0*v

    expect_eq(r0, vec4(3.0, 6.0, 9.0, 12.0))
    expect_eq(r1, vec4(3.0, 6.0, 9.0, 12.0))

    col0 = vec4(1.0, 0.0, 0.0, 0.0)
    col1 = vec4(0.0, 2.0, 0.0, 0.0)
    col2 = vec4(0.0, 0.0, 3.0, 0.0)
    col3 = vec4(0.0, 0.0, 0.0, 4.0)

    m = mat44(col0, col1, col2, col3)

    expect_eq(m*vec4(1.0, 0.0, 0.0, 0.0), col0)
    expect_eq(m*vec4(0.0, 1.0, 0.0, 0.0), col1)
    expect_eq(m*vec4(0.0, 0.0, 1.0, 0.0), col2)
    expect_eq(m*vec4(0.0, 0.0, 0.0, 1.0), col3)

    two = vec4(1.0)*2.0
    expect_eq(two, vec4(2.0, 2.0, 2.0, 2.0))
       


devices = wp.get_devices()

class TestOperators(test_base.TestBase):
    pass

TestOperators.add_kernel_test(test_operators_scalar, dim=1, devices=devices)
TestOperators.add_kernel_test(test_operators_vec3, dim=1, devices=devices)
TestOperators.add_kernel_test(test_operators_vec4, dim=1, devices=devices)

if __name__ == '__main__':
    unittest.main(verbosity=2)
