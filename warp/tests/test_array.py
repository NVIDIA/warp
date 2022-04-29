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


@wp.kernel
def kernel_1d(a: wp.array(dtype=int, ndim=1)):

    i = wp.tid()

    wp.expect_eq(a[i], wp.tid())

    a[i] = a[i]*2
    wp.atomic_add(a, i, 1)

    wp.expect_eq(a[i], wp.tid()*2 + 1)


def test_1d(test, device):

    dim_x = 4

    a = np.arange(0, dim_x, dtype=np.int32)

    arr = wp.array(a, device=device)

    test.assertEqual(arr.shape, a.shape)
    test.assertEqual(arr.size, a.size)
    test.assertEqual(arr.ndim, a.ndim)

    with CheckOutput(test):
        wp.launch(kernel_1d, dim=arr.size, inputs=[arr], device=device)


@wp.kernel
def kernel_2d(a: wp.array(dtype=int, ndim=2), m: int, n: int):

    i = wp.tid()//n
    j = wp.tid()%n

    wp.expect_eq(a[i,j], wp.tid())
    wp.expect_eq(a[i][j], wp.tid())

    a[i,j] = a[i,j]*2
    wp.atomic_add(a, i, j, 1)

    wp.expect_eq(a[i,j], wp.tid()*2 + 1)




def test_2d(test, device):

    dim_x = 4
    dim_y = 2

    a = np.arange(0, dim_x*dim_y, dtype=np.int32)
    a = a.reshape(dim_x, dim_y)

    arr = wp.array(a, device=device)

    test.assertEqual(arr.shape, a.shape)
    test.assertEqual(arr.size, a.size)
    test.assertEqual(arr.ndim, a.ndim)

    with CheckOutput(test):
        wp.launch(kernel_2d, dim=arr.size, inputs=[arr, dim_x, dim_y], device=device)


@wp.kernel
def kernel_3d(a: wp.array(dtype=int, ndim=3), m: int, n: int, o: int):

    i = wp.tid()//(n*o)
    j = wp.tid()%(n*o)//o
    k = wp.tid()%o

    wp.expect_eq(a[i,j,k], wp.tid())
    wp.expect_eq(a[i][j][k], wp.tid())

    a[i,j,k] = a[i,j,k]*2
    a[i][j][k] = a[i][j][k]*2
    wp.atomic_add(a, i, j, k, 1)

    wp.expect_eq(a[i,j,k], wp.tid()*4 + 1)


def test_3d(test, device):
    
    dim_x = 8
    dim_y = 4
    dim_z = 2

    a = np.arange(0, dim_x*dim_y*dim_z, dtype=np.int32)
    a = a.reshape(dim_x, dim_y, dim_z)

    arr = wp.array(a, device=device)

    test.assertEqual(arr.shape, a.shape)
    test.assertEqual(arr.size, a.size)
    test.assertEqual(arr.ndim, a.ndim)

    with CheckOutput(test):
        wp.launch(kernel_3d, dim=arr.size, inputs=[arr, dim_x, dim_y, dim_z], device=device)


@wp.kernel
def kernel_4d(a: wp.array(dtype=int, ndim=4), m: int, n: int, o: int, p: int):

    i = wp.tid()//(n*o*p)
    j = wp.tid()%(n*o*p)//(o*p)
    k = wp.tid()%(o*p)/p
    l = wp.tid()%p



    wp.expect_eq(a[i,j,k,l], wp.tid())
    wp.expect_eq(a[i][j][k][l], wp.tid())

def test_4d(test, device):
    
    dim_x = 16
    dim_y = 8
    dim_z = 4
    dim_w = 2

    a = np.arange(0, dim_x*dim_y*dim_z*dim_w, dtype=np.int32)
    a = a.reshape(dim_x, dim_y, dim_z, dim_w)

    arr = wp.array(a, device=device)

    test.assertEqual(arr.shape, a.shape)
    test.assertEqual(arr.size, a.size)
    test.assertEqual(arr.ndim, a.ndim)

    with CheckOutput(test):
        wp.launch(kernel_4d, dim=arr.size, inputs=[arr, dim_x, dim_y, dim_z, dim_w], device=device)
        



def register(parent):

    devices = wp.get_devices()

    class TestArray(parent):
        pass

    add_function_test(TestArray, "test_1d_array", test_1d, devices=devices)
    add_function_test(TestArray, "test_2d_array", test_2d, devices=devices)
    add_function_test(TestArray, "test_3d_array", test_3d, devices=devices)
    add_function_test(TestArray, "test_4d_array", test_4d, devices=devices)

    return TestArray

if __name__ == '__main__':
    c = register(unittest.TestCase)
    unittest.main(verbosity=2)