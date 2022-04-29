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

dim_x = wp.constant(2)
dim_y = wp.constant(2)
dim_z = wp.constant(2)
dim_w = wp.constant(2)


@wp.kernel
def kernel1d(a: wp.array(dtype=int, ndim=1)):

    i = wp.tid()

    wp.expect_eq(a[i], i)

@wp.kernel
def kernel2d(a: wp.array(dtype=int, ndim=2)):

    i, j = wp.tid()

    wp.expect_eq(a[i,j], i*dim_y + j)


@wp.kernel
def kernel3d(a: wp.array(dtype=int, ndim=3)):

    i, j, k = wp.tid()

    wp.expect_eq(a[i,j,k], i*dim_y*dim_z + j*dim_z + k)

@wp.kernel
def kernel4d(a: wp.array(dtype=int, ndim=4)):

    i, j, k, l = wp.tid()

    wp.expect_eq(a[i,j,k,l], i*dim_y*dim_z*dim_w + j*dim_z*dim_w + k*dim_w + l)


def test1d(test, device):
    
    a = np.arange(0, dim_x.val).reshape(dim_x.val)

    wp.launch(kernel1d, dim=a.shape, inputs=[wp.array(a, dtype=int, device=device)], device=device)


def test2d(test, device):
    
    a = np.arange(0, dim_x.val*dim_y.val).reshape(dim_x.val,dim_y.val)

    wp.launch(kernel2d, dim=a.shape, inputs=[wp.array(a, dtype=int, device=device)], device=device)


def test3d(test, device):
    
    a = np.arange(0, dim_x.val*dim_y.val*dim_z.val).reshape(dim_x.val,dim_y.val,dim_z.val)

    wp.launch(kernel3d, dim=a.shape, inputs=[wp.array(a, dtype=int, device=device)], device=device)


def test4d(test, device):
    
    a = np.arange(0, dim_x.val*dim_y.val*dim_z.val*dim_w.val).reshape(dim_x.val,dim_y.val,dim_z.val,dim_w.val)

    wp.launch(kernel4d, dim=a.shape, inputs=[wp.array(a, dtype=int, device=device)], device=device)
    

def register(parent):

    devices = wp.get_devices()

    class TestLaunch(parent):
        pass

    add_function_test(TestLaunch, "test_1d_launch", test1d, devices=devices)
    add_function_test(TestLaunch, "test_2d_launch", test2d, devices=devices)
    add_function_test(TestLaunch, "test_3d_launch", test3d, devices=devices)
    add_function_test(TestLaunch, "test_4d_launch", test4d, devices=devices)

    return TestLaunch

if __name__ == '__main__':
    c = register(unittest.TestCase)
    unittest.main(verbosity=2)