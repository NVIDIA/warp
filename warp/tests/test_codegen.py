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

import unittest

import warp as wp
from warp.tests.test_base import *

#wp.config.mode = "debug"

wp.init()

@wp.kernel
def test_rename():


    a = 0
    b = 1
    
    a = b
    a = 2

    wp.expect_eq(a, 2)
    wp.expect_eq(b, 1)
    

@wp.kernel
def test_inplace():

    a = 1.0
    a += 2.0
    
    wp.expect_eq(a, 3.0)


@wp.kernel
def test_constant(c: float):

    a = 0.0
    a = c + 1.0

    wp.expect_eq(a, 2.0)

@wp.kernel
def test_dynamic_for_rename(n: int):

    f0 = int(0.0)
    f1 = int(1.0)

    for i in range(0, n):
        
        f = f0 + f1
        
        f0 = f1
        f1 = f

    wp.expect_eq(f1, 89)

@wp.kernel
def test_dynamic_for_inplace(n: int):

    a = float(0.0)

    for i in range(0, n):
        a += 1.0

    wp.expect_eq(n, 10)


@wp.kernel
def test_reassign():

    f0 = 1.0
    f1 = f0

    f1 = f1 + 2.0

    wp.expect_eq(f1, 3.0)
    wp.expect_eq(f0, 1.0)

@wp.kernel
def test_dynamic_reassign(n: int):

    f0 = wp.vec3()
    f1 = f0

    for i in range(0, n):
        f1 = f1 - wp.vec3(2.0, 0.0, 0.0)

    wp.expect_eq(f1, wp.vec3(-4.0, 0.0, 0.0))
    wp.expect_eq(f0, wp.vec3())


@wp.kernel
def test_range_static_sum(result: wp.array(dtype=int)):

    a = int(0)
    for i in range(10):
        a = a + 1

    b = int(0)
    for i in range(0, 10):
        b = b + 1

    c = int(0)
    for i in range(0, 20, 2):
        c = c + 1

    result[0] = a
    result[1] = b
    result[2] = c


@wp.kernel
def test_range_dynamic_sum(start: int, end: int, step: int, result: wp.array(dtype=int)):

    a = int(0)
    for i in range(end):
        a = a + 1

    b = int(0)
    for i in range(start, end):
        b = b + 1

    c = int(0)
    for i in range(start, end*step, step):
        c = c + 1

    d = int(0)
    for i in range(end*step, start, -step):
        d = d + 1

    result[0] = a
    result[1] = b
    result[2] = c
    result[3] = d

@wp.kernel
def test_range_dynamic(start: int, end: int, step: int, result: wp.array(dtype=int)):

    output = int(0)
    for i in range(start, end, step):
        result[output] = i
        output += 1

@wp.kernel
def test_range_dynamic_nested(n: int):
    
    sum1 = float(0.0)
    sum2 = float(0.0)
    sum3 = float(0.0)

    for i in range(n):
        sum1 = sum1 + 1.0
        sum3 = sum3 + 1.0

        for j in range(n):
            sum2 = sum2 + 1.0
            sum3 = sum3 + 1.0
        
        sum3 = sum3 + 1.0


    wp.expect_eq(sum1, float(n))
    wp.expect_eq(sum2, float(n*n))
    wp.expect_eq(sum3, float(n*n + 2*n))



@wp.kernel
def test_while(n: int):

    i = int(0)

    while i < n:
        i = i + 1

    wp.expect_eq(i, n)


def register(parent):

    class TestCodeGen(parent):
        pass

    devices = wp.get_devices()

    add_kernel_test(TestCodeGen, name="test_inplace", kernel=test_inplace, dim=1, devices=devices)
    add_kernel_test(TestCodeGen, name="test_rename", kernel=test_rename, dim=1, devices=devices)
    add_kernel_test(TestCodeGen, name="test_constant", kernel=test_constant, inputs=[1.0], dim=1, devices=devices)
    add_kernel_test(TestCodeGen, name="test_dynamic_for_rename", kernel=test_dynamic_for_rename, inputs=[10], dim=1, devices=devices)
    add_kernel_test(TestCodeGen, name="test_dynamic_for_inplace", kernel=test_dynamic_for_inplace, inputs=[10], dim=1, devices=devices)
    add_kernel_test(TestCodeGen, name="test_reassign", kernel=test_reassign, dim=1, devices=devices)
    add_kernel_test(TestCodeGen, name="test_dynamic_reassign", kernel=test_dynamic_reassign, inputs=[2], dim=1, devices=devices)

    add_kernel_test(TestCodeGen, name="test_range_dynamic_forward", kernel=test_range_dynamic, dim=1, inputs=[0, 4, 1], expect=[0, 1, 2, 3], devices=devices)
    add_kernel_test(TestCodeGen, name="test_range_dynamic_reverse", kernel=test_range_dynamic, dim=1, inputs=[4, 0, -1], expect=[4, 3, 2, 1], devices=devices)
    add_kernel_test(TestCodeGen, name="test_range_dynamic_foward_step", kernel=test_range_dynamic, dim=1, inputs=[0, 8, 2], expect=[0, 2, 4, 6], devices=devices)
    add_kernel_test(TestCodeGen, name="test_range_dynamic_reverse_step", kernel=test_range_dynamic, dim=1, inputs=[8, 0, -2], expect=[8, 6, 4, 2], devices=devices)

    add_kernel_test(TestCodeGen, name="test_range_static_sum", kernel=test_range_static_sum, dim=1, expect=[10, 10, 10], devices=devices)
    add_kernel_test(TestCodeGen, name="test_range_dynamic_sum", kernel=test_range_dynamic_sum, dim=1, inputs=[0, 10, 2], expect=[10, 10, 10, 10], devices=devices)
    add_kernel_test(TestCodeGen, name="test_range_dynamic_sum_zero", kernel=test_range_dynamic_sum, dim=1, inputs=[0, 0, 1], expect=[0, 0, 0, 0], devices=devices)

    add_kernel_test(TestCodeGen, name="test_range_dynamic_nested", kernel=test_range_dynamic_nested, dim=1, inputs=[4], devices=devices)

    add_kernel_test(TestCodeGen, name="test_while_zero", kernel=test_while, dim=1, inputs=[0], devices=devices)
    add_kernel_test(TestCodeGen, name="test_while_positive", kernel=test_while, dim=1, inputs=[16], devices=devices)

    return TestCodeGen


if __name__ == '__main__':
    c = register(unittest.TestCase)
    unittest.main(verbosity=2)


