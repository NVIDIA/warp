# include parent path
import os
import sys
import numpy as np
import math
import ctypes

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warp as wp

import unittest
import test_base

wp.init()

@wp.kernel
def test_args(v1: float,
              v2: wp.vec2,
              v3: wp.vec3,
              v4: wp.vec4,
              m22: wp.mat22,
              m33: wp.mat33,
              m44: wp.mat44,
              xform: wp.transform):

    wp.expect_eq(v1, 1.0)
    wp.expect_eq(v2, wp.vec2(1.0, 2.0))
    wp.expect_eq(v3, wp.vec3(1.0, 2.0, 3.0))
    wp.expect_eq(v4, wp.vec4(1.0, 2.0, 3.0, 4.0))
    
    # wp.expect_eq(m22, wp.mat22(1.0, 2.0, 
    #                            3.0, 4.0))
    
    # wp.expect_eq(m33, wp.mat33(1.0, 2.0, 3.0,
    #                            4.0, 5.0, 6.0,
    #                            7.0, 8.0, 9.0))

    # wp.expect_eq(m44, wp.mat44(1.0, 2.0, 3.0, 4.0,
    #                            5.0, 6.0, 7.0, 8.0,
    #                            9.0, 10.0, 11.0, 12.0,
    #                            13.0, 14.0, 15.0, 16.0))

@wp.kernel
def add_vec2(dest: wp.array(dtype=wp.vec2),
             c: wp.vec2):

    tid = wp.tid()
    wp.store(dest, tid, c)


@wp.kernel
def transform_vec2(dest: wp.array(dtype=wp.vec2),
    m: wp.mat22,
    v: wp.vec2):

    tid = wp.tid()

    p = wp.mul(m, v)
    wp.store(dest, tid, p)


@wp.kernel
def add_vec3(dest: wp.array(dtype=wp.vec3),
             c: wp.vec3):

    tid = wp.tid()
    wp.store(dest, tid, c)

@wp.kernel
def transform_vec3(dest: wp.array(dtype=wp.vec3),
    m: wp.mat33,
    v: wp.vec3):

    tid = wp.tid()

    p = wp.mul(m, v)
    wp.store(dest, tid, p)


@wp.kernel
def transform_multiply(xforms: wp.array(dtype=wp.transform),
    a: wp.transform):

    tid = wp.tid()

    xforms[tid] = wp.transform_multiply(xforms[tid], a)
   
def test_args(test, device):

    wp.launch(all_types, dim=1, 
        inputs=[1.0,
                wp.vec2(1.0, 2.0),
                wp.vec3(1.0, 2.0, 3.0),
                wp.vec4(1.0, 2.0, 3.0, 4.0),

                wp.mat22(1.0, 2.0,
                         3.0, 4.0),

                wp.mat33(1.0, 2.0, 3.0,
                         4.0, 5.0, 6.0,
                         7.0, 8.0, 9.0),

                wp.mat44(1.0, 2.0, 3.0, 4.0,
                         5.0, 6.0, 7.0, 8.0,
                         9.0, 10.0, 11.0, 12.0,
                         13.0, 14.0, 15.0, 16.0)], device=device)

def test_vec2_arg(test, device, n):
    
    dest = wp.zeros(n=n, dtype=wp.vec2, device=device)
    c = np.array((1.0, 2.0))

    wp.launch(add_vec2, dim=n, inputs=[dest, c], device=device)

    # ensure type can round-trip from Python->GPU->Python
    test.assertTrue(np.array_equal(dest.numpy(), np.tile(c, (n, 1))))

def test_vec2_transform(test, device, n):

    dest = wp.zeros(n=n, dtype=wp.vec2, device=device)
    c = np.array((1.0, 2.0))
    m = np.array(((3.0, -1.0),
                  (2.5, 4.0)))

    wp.launch(transform_vec2, dim=n, inputs=[dest, m, c], device=device)
    test.assertTrue(np.array_equal(dest.numpy(), np.tile(m@c, (n, 1))))

def test_vec3_arg(test, device, n):
        
    dest = wp.zeros(n=n, dtype=wp.vec3, device=device)
    c = np.array((1.0, 2.0, 3.0))

    wp.launch(add_vec3, dim=n, inputs=[dest, c], device=device)
    test.assertTrue(np.array_equal(dest.numpy(), np.tile(c, (n, 1))))

def test_vec3_transform(test, device, n):

    dest = wp.zeros(n=n, dtype=wp.vec3, device=device)
    c = np.array((1.0, 2.0, 3.0))
    m = np.array(((1.0, 2.0, 3.0),
                  (4.0, 5.0, 6.0),
                  (7.0, 8.0, 9.0)))

    wp.launch(transform_vec3, dim=n, inputs=[dest, m, c], device=device)
    test.assertTrue(np.array_equal(dest.numpy(), np.tile(m@c, (n, 1))))

def test_transform_multiply(test, device, n):

    a = wp.transform((0.0, 1.0, 0.0), wp.quat_identity())

    x = []
    for i in range(10):
        x.append(wp.transform_identity())

    xforms = wp.array(x, dtype=wp.transform, device=device)

    wp.launch(transform_multiply, dim=n, inputs=[xforms, a], device=device)
    print(xforms)


devices = wp.get_devices()

class TestCTypes(test_base.TestBase):
    pass

v = wp.vec3(1.0, 2.0, 3.0)
t = wp.mat22(1.0, 2.0,
                   3.0, 4.0),


inputs = [1.0,
          wp.vec2(1.0, 2.0),
          wp.vec3(1.0, 2.0, 3.0),

          wp.vec4(1.0, 2.0, 3.0, 4.0),
          wp.mat22(1.0, 2.0,
                   3.0, 4.0),
          wp.mat33(1.0, 2.0, 3.0,
                   4.0, 5.0, 6.0,
                   7.0, 8.0, 9.0),
          wp.mat44(1.0, 2.0, 3.0, 4.0,
                   5.0, 6.0, 7.0, 8.0,
                   9.0, 10.0, 11.0, 12.0,
                   13.0, 14.0, 15.0, 16.0)]

TestCTypes.add_kernel_test("test_args", test_args, dim=1, inputs=inputs, devices=devices)
TestCTypes.add_function_test("test_vec2_arg", test_vec2_arg, devices=devices, n=8)
TestCTypes.add_function_test("test_vec2_transform", test_vec2_transform, devices=devices, n=8)
TestCTypes.add_function_test("test_vec3_arg", test_vec3_arg, devices=devices, n=8)
TestCTypes.add_function_test("test_vec3_transform", test_vec3_transform, devices=devices, n=8)
TestCTypes.add_function_test("test_transform_multiply", test_transform_multiply, devices=devices, n=8)



if __name__ == '__main__':
    unittest.main(verbosity=2)