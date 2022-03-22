# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import math

import warp as wp
from warp.tests.test_base import *

wp.init()

@wp.kernel
def test_operators_scalar_float():

    a = 1.0
    b = 2.0

    c = a*b
    d = a+b
    e = a/b
    f = a-b
    g = b**8.0
    h = 10.0//3.0

    expect_eq(c, 2.0)
    expect_eq(d, 3.0)
    expect_eq(e, 0.5)
    expect_eq(f, -1.0)
    expect_eq(g, 256.0)
    expect_eq(h, 3.0)
    
@wp.kernel
def test_operators_scalar_int():

    a = 1
    b = 2

    c = a*b
    d = a+b
    e = a/b
    f = a-b
    # g = b**8    # integer pow not implemented
    h = 10//3

    expect_eq(c, 2)
    expect_eq(d, 3)
    expect_eq(e, 0)
    expect_eq(f, -1)
    # expect_eq(g, 256)
    expect_eq(h, 3)

@wp.kernel
def test_operators_vector_index():

    v = wp.vec4(1.0, 2.0, 3.0, 4.0)

    expect_eq(v[0], 1.0)
    expect_eq(v[1], 2.0)
    expect_eq(v[2], 3.0)
    expect_eq(v[3], 4.0)


@wp.kernel
def test_operators_matrix_index():

    m22 = wp.mat22(1.0, 2.0, 3.0, 4.0)

    expect_eq(m22[0,0], 1.0)
    expect_eq(m22[0,1], 2.0)
    expect_eq(m22[1,0], 3.0)
    expect_eq(m22[1,1], 4.0)
    


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
       

def register(parent):

    devices = wp.get_devices()

    class TestOperators(parent):
        pass

    add_kernel_test(TestOperators, test_operators_scalar_float, dim=1, devices=devices)
    add_kernel_test(TestOperators, test_operators_scalar_int, dim=1, devices=devices)
    add_kernel_test(TestOperators, test_operators_matrix_index, dim=1, devices=devices)
    add_kernel_test(TestOperators, test_operators_vector_index, dim=1, devices=devices)
    add_kernel_test(TestOperators, test_operators_vec3, dim=1, devices=devices)
    add_kernel_test(TestOperators, test_operators_vec4, dim=1, devices=devices)

    return TestOperators

if __name__ == '__main__':
    c = register(unittest.TestCase)
    unittest.main(verbosity=2)
