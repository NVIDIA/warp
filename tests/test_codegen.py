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

    f0 = wp.vec3(0.0, 0.0, 0.0)
    f1 = f0

    for i in range(0, n):
        f1 = f1 - wp.vec3(2.0, 0.0, 0.0)

    wp.expect_eq(f1, wp.vec3(-4.0, 0.0, 0.0))
    wp.expect_eq(f0, wp.vec3(0.0, 0.0, 0.0))


@wp.kernel
def test_range_static(result: wp.array(dtype=int)):

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
def test_range_dynamic(start: int, end: int, step: int, result: wp.array(dtype=int)):

    a = int(0)
    for i in range(end):
        a = a + 1

    b = int(0)
    for i in range(start, end):
        b = b + 1

    c = int(0)
    for i in range(start, end*step, step):
        c = c + 1

    result[0] = a
    result[1] = b
    result[2] = c


devices = ["cpu", "cuda"]

class TestCodeGen(test_base.TestBase):   
    pass

TestCodeGen.add_kernel_test(name="test_inplace", kernel=test_inplace, dim=1, devices=devices)
TestCodeGen.add_kernel_test(name="test_rename", kernel=test_rename, dim=1, devices=devices)
TestCodeGen.add_kernel_test(name="test_constant", kernel=test_constant, inputs=[1.0], dim=1, devices=devices)
TestCodeGen.add_kernel_test(name="test_dynamic_for_rename", kernel=test_dynamic_for_rename, inputs=[10], dim=1, devices=devices)
TestCodeGen.add_kernel_test(name="test_dynamic_for_inplace", kernel=test_dynamic_for_inplace, inputs=[10], dim=1, devices=devices)
TestCodeGen.add_kernel_test(name="test_reassign", kernel=test_reassign, dim=1, devices=devices)
TestCodeGen.add_kernel_test(name="test_dynamic_reassign", kernel=test_dynamic_reassign, inputs=[2], dim=1, devices=devices)
TestCodeGen.add_kernel_test(name="test_range_static", kernel=test_range_static, dim=1, expect=[10, 10, 10], devices=devices)
TestCodeGen.add_kernel_test(name="test_range_dynamic", kernel=test_range_dynamic, dim=1, inputs=[0, 10, 2], expect=[10, 10, 10], devices=devices)

if __name__ == '__main__':
    unittest.main(verbosity=2)


