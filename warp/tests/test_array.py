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
        

@wp.kernel
def kernel_4d_transposed(a: wp.array(dtype=int, ndim=4), m: int, n: int, o: int, p: int):

    i = wp.tid()//(n*o*p)
    j = wp.tid()%(n*o*p)//(o*p)
    k = wp.tid()%(o*p)/p
    l = wp.tid()%p

    wp.expect_eq(a[l,k,j,i], wp.tid())
    wp.expect_eq(a[l][k][j][i], wp.tid())


def test_4d_transposed(test, device):
    
    dim_x = 16
    dim_y = 8
    dim_z = 4
    dim_w = 2

    a = np.arange(0, dim_x*dim_y*dim_z*dim_w, dtype=np.int32)
    a = a.reshape(dim_x, dim_y, dim_z, dim_w)
    
    arr = wp.array(a, device=device)

    # Transpose the array manually, as using the wp.array() constructor with arr.T would make it contiguous first
    a_T = a.T
    arr_T = wp.array(
        dtype=arr.dtype, shape=a_T.shape, strides=a_T.__array_interface__["strides"],
        capacity=arr.capacity, ptr=arr.ptr, owner=False, requires_grad=arr.requires_grad, device=device)

    test.assertFalse(arr_T.is_contiguous)
    test.assertEqual(arr_T.shape, a_T.shape)
    test.assertEqual(arr_T.strides, a_T.__array_interface__["strides"])
    test.assertEqual(arr_T.size, a_T.size)
    test.assertEqual(arr_T.ndim, a_T.ndim)

    with CheckOutput(test):
        wp.launch(kernel_4d_transposed, dim=arr_T.size, inputs=[arr_T, dim_x, dim_y, dim_z, dim_w], device=device)


@wp.kernel
def lower_bound_kernel(values: wp.array(dtype=float), arr: wp.array(dtype=float), indices: wp.array(dtype=int)):
    tid = wp.tid()

    indices[tid] = wp.lower_bound(arr, values[tid])


def test_lower_bound(test, device):
    arr = wp.array(np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float), dtype=float, device=device)
    values = wp.array(np.array([-0.1, 0.0, 2.5, 4.0, 5.0, 5.5], dtype=float), dtype=float, device=device)
    indices = wp.zeros(6, dtype=int, device=device)

    wp.launch(kernel=lower_bound_kernel, dim=6, inputs=[values, arr, indices], device=device)

    test.assertTrue((np.array([0, 0, 3, 4, 5, 5]) == indices.numpy()).all())


@wp.kernel
def f1(arr: wp.array(dtype=float)):
    
    wp.expect_eq(arr.shape[0], 10)


@wp.kernel
def f2(arr: wp.array2d(dtype=float)):

    wp.expect_eq(arr.shape[0], 10)
    wp.expect_eq(arr.shape[1], 20)

    slice = arr[0]
    wp.expect_eq(slice.shape[0], 20)

@wp.kernel
def f3(arr: wp.array3d(dtype=float)):

    wp.expect_eq(arr.shape[0], 10)
    wp.expect_eq(arr.shape[1], 20)
    wp.expect_eq(arr.shape[2], 30)

    slice = arr[0,0]
    wp.expect_eq(slice.shape[0], 30)

@wp.kernel
def f4(arr: wp.array4d(dtype=float)):

    wp.expect_eq(arr.shape[0], 10)
    wp.expect_eq(arr.shape[1], 20)
    wp.expect_eq(arr.shape[2], 30)
    wp.expect_eq(arr.shape[3], 40)

    slice = arr[0,0,0]
    wp.expect_eq(slice.shape[0], 40)


def test_shape(test, device):
    
    with CheckOutput(test):
        a1 = wp.zeros(dtype=float, shape=10, device=device)
        wp.launch(f1, dim=1, inputs=[a1], device=device)

        a2 = wp.zeros(dtype=float, shape=(10, 20), device=device)
        wp.launch(f2, dim=1, inputs=[a2], device=device)

        a3 = wp.zeros(dtype=float, shape=(10, 20, 30), device=device)
        wp.launch(f3, dim=1, inputs=[a3], device=device)

        a4 = wp.zeros(dtype=float, shape=(10, 20, 30, 40), device=device)
        wp.launch(f4, dim=1, inputs=[a4], device=device)


def test_fill_zero(test, device):

    dim_x = 4

    # test zeroing:
    for nptype,wptype in wp.types.np_dtype_to_warp_type.items():

        a1 = wp.zeros(dim_x, dtype=wptype, device=device)
        a2 = wp.zeros((dim_x,dim_x), dtype=wptype, device=device)
        a3 = wp.zeros((dim_x,dim_x,dim_x), dtype=wptype, device=device)
        a4 = wp.zeros((dim_x,dim_x,dim_x,dim_x), dtype=wptype, device=device)

        a1.fill_(127)
        a2.fill_(127)
        a3.fill_(127)
        a4.fill_(127)

        a1.zero_()
        a2.zero_()
        a3.zero_()
        a4.zero_()

        assert_np_equal(a1.numpy(), np.zeros_like(a1.numpy()))
        assert_np_equal(a2.numpy(), np.zeros_like(a2.numpy()))
        assert_np_equal(a3.numpy(), np.zeros_like(a3.numpy()))
        assert_np_equal(a4.numpy(), np.zeros_like(a4.numpy()))

        # test some vector types too:
        v1 = wp.zeros(dim_x, dtype=wp.vec(3,wptype), device=device)
        v2 = wp.zeros((dim_x,dim_x), dtype=wp.vec(3,wptype), device=device)
        v3 = wp.zeros((dim_x,dim_x,dim_x), dtype=wp.vec(3,wptype), device=device)
        v4 = wp.zeros((dim_x,dim_x,dim_x,dim_x), dtype=wp.vec(3,wptype), device=device)

        v1.fill_(127)
        v2.fill_(127)
        v3.fill_(127)
        v4.fill_(127)

        v1.zero_()
        v2.zero_()
        v3.zero_()
        v4.zero_()
        
        assert_np_equal(v1.numpy(), np.zeros_like(v1.numpy()))
        assert_np_equal(v2.numpy(), np.zeros_like(v2.numpy()))
        assert_np_equal(v3.numpy(), np.zeros_like(v3.numpy()))
        assert_np_equal(v4.numpy(), np.zeros_like(v4.numpy()))

    # test fill with scalar constant:
    for nptype,wptype in wp.types.np_dtype_to_warp_type.items():
        
        a1 = wp.zeros(dim_x, dtype=wptype, device=device)
        a2 = wp.zeros((dim_x,dim_x), dtype=wptype, device=device)
        a3 = wp.zeros((dim_x,dim_x,dim_x), dtype=wptype, device=device)
        a4 = wp.zeros((dim_x,dim_x,dim_x,dim_x), dtype=wptype, device=device)
        
        a1.fill_(127)
        a2.fill_(127)
        a3.fill_(127)
        a4.fill_(127)
        
        assert_np_equal(a1.numpy(), 127 * np.ones_like(a1.numpy()))
        assert_np_equal(a2.numpy(), 127 * np.ones_like(a2.numpy()))
        assert_np_equal(a3.numpy(), 127 * np.ones_like(a3.numpy()))
        assert_np_equal(a4.numpy(), 127 * np.ones_like(a4.numpy()))
        
        # test some vector types too:
        v1 = wp.zeros(dim_x, dtype=wp.vec(3,wptype), device=device)
        v2 = wp.zeros((dim_x,dim_x), dtype=wp.vec(3,wptype), device=device)
        v3 = wp.zeros((dim_x,dim_x,dim_x), dtype=wp.vec(3,wptype), device=device)
        v4 = wp.zeros((dim_x,dim_x,dim_x,dim_x), dtype=wp.vec(3,wptype), device=device)

        v1.fill_(127)
        v2.fill_(127)
        v3.fill_(127)
        v4.fill_(127)
        
        assert_np_equal(v1.numpy(), 127 * np.ones_like(v1.numpy()))
        assert_np_equal(v2.numpy(), 127 * np.ones_like(v2.numpy()))
        assert_np_equal(v3.numpy(), 127 * np.ones_like(v3.numpy()))
        assert_np_equal(v4.numpy(), 127 * np.ones_like(v4.numpy()))
    
    # test fill with vector constant:
    for nptype,wptype in wp.types.np_dtype_to_warp_type.items():
        
        vectype = wp.vec(3,wptype)

        vecvalue = vectype(1,2,3)

        # test some vector types too:
        v1 = wp.zeros(dim_x, dtype=vectype, device=device)
        v2 = wp.zeros((dim_x,dim_x), dtype=vectype, device=device)
        v3 = wp.zeros((dim_x,dim_x,dim_x), dtype=vectype, device=device)
        v4 = wp.zeros((dim_x,dim_x,dim_x,dim_x), dtype=vectype, device=device)

        v1.fill_( vecvalue )
        v2.fill_( vecvalue )
        v3.fill_( vecvalue )
        v4.fill_( vecvalue )

        e1 = np.tile( np.array([1,2,3],dtype=nptype)[None,:], (dim_x,1) )
        e2 = np.tile( np.array([1,2,3],dtype=nptype)[None,None,:], (dim_x,dim_x,1) )
        e3 = np.tile( np.array([1,2,3],dtype=nptype)[None,None,None,:], (dim_x,dim_x,dim_x,1) )
        e4 = np.tile( np.array([1,2,3],dtype=nptype)[None,None,None,None,:], (dim_x,dim_x,dim_x,dim_x,1) )
        
        assert_np_equal(v1.numpy(), e1)
        assert_np_equal(v2.numpy(), e2)
        assert_np_equal(v3.numpy(), e3)
        assert_np_equal(v4.numpy(), e4)

    # specific tests for floating point values:
    for nptype in [ np.dtype(np.float16), np.dtype(np.float32), np.dtype(np.float64) ]:

        wptype = wp.types.np_dtype_to_warp_type[nptype]
        
        vectype = wp.vec(3,wptype)

        vecvalue = vectype(1.25,2.5,3.75)

        # test some vector types too:
        v1 = wp.zeros(dim_x, dtype=vectype, device=device)
        v2 = wp.zeros((dim_x,dim_x), dtype=vectype, device=device)
        v3 = wp.zeros((dim_x,dim_x,dim_x), dtype=vectype, device=device)
        v4 = wp.zeros((dim_x,dim_x,dim_x,dim_x), dtype=vectype, device=device)

        v1.fill_( vecvalue )
        v2.fill_( vecvalue )
        v3.fill_( vecvalue )
        v4.fill_( vecvalue )

        e1 = np.tile( np.array([1.25,2.5,3.75],dtype=nptype)[None,:], (dim_x,1) )
        e2 = np.tile( np.array([1.25,2.5,3.75],dtype=nptype)[None,None,:], (dim_x,dim_x,1) )
        e3 = np.tile( np.array([1.25,2.5,3.75],dtype=nptype)[None,None,None,:], (dim_x,dim_x,dim_x,1) )
        e4 = np.tile( np.array([1.25,2.5,3.75],dtype=nptype)[None,None,None,None,:], (dim_x,dim_x,dim_x,dim_x,1) )
        
        assert_np_equal(v1.numpy(), e1)
        assert_np_equal(v2.numpy(), e2)
        assert_np_equal(v3.numpy(), e3)
        assert_np_equal(v4.numpy(), e4)


    # test fill small arrays with scalar constant:
    for xdim in [1,2,3,5,6,7]:
        for nptype,wptype in wp.types.np_dtype_to_warp_type.items():
            
            a1 = wp.zeros(xdim, dtype=wptype, device=device)
            a1.fill_(127)
            assert_np_equal(a1.numpy(), 127 * np.ones_like(a1.numpy()))
        

def test_round_trip(test, device):

    dim_x = 4

    for nptype,wptype in wp.types.np_dtype_to_warp_type.items():
        a_np = np.random.randn(dim_x).astype(nptype)
        a = wp.array(a_np,device=device)
        test.assertEqual(a.dtype,wptype)

        assert_np_equal(a.numpy(), a_np)

        v_np = np.random.randn(dim_x,3).astype(nptype)
        v = wp.array(v_np,dtype=wp.vec(3,wptype),device=device)

        assert_np_equal(v.numpy(), v_np)


def register(parent):

    devices = wp.get_devices()

    class TestArray(parent):
        pass

    add_function_test(TestArray, "test_shape", test_shape, devices=devices)
    add_function_test(TestArray, "test_1d_array", test_1d, devices=devices)
    add_function_test(TestArray, "test_2d_array", test_2d, devices=devices)
    add_function_test(TestArray, "test_3d_array", test_3d, devices=devices)
    add_function_test(TestArray, "test_4d_array", test_4d, devices=devices)
    add_function_test(TestArray, "test_4d_array_transposed", test_4d_transposed, devices=devices)
    add_function_test(TestArray, "test_lower_bound", test_lower_bound, devices=devices)
    add_function_test(TestArray, "test_fill_zero", test_fill_zero, devices=devices)
    add_function_test(TestArray, "test_round_trip", test_round_trip, devices=devices)

    return TestArray

if __name__ == '__main__':
    c = register(unittest.TestCase)
    unittest.main(verbosity=2)