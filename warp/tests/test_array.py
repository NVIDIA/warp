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


@wp.kernel
def sum_array(arr: wp.array(dtype=float), loss: wp.array(dtype=float)):
    tid = wp.tid()
    wp.atomic_add(loss, 0, arr[tid])


def test_flatten(test, device):

    np_arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=float)
    arr = wp.array(np_arr, dtype=float, shape=np_arr.shape, device=device, requires_grad=True)
    arr_flat = arr.flatten()
    arr_comp = wp.array(np_arr.flatten(), dtype=float, device=device)
    assert_array_equal(arr_flat, arr_comp)

    loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        wp.launch(
            kernel=sum_array,
            dim=len(arr_flat),
            inputs=[arr_flat, loss],
            device=device
        )

    tape.backward(loss=loss)
    grad = tape.gradients[arr_flat]

    ones = wp.array(np.ones((8,), dtype=float,), dtype=float, device=device)
    assert_array_equal(grad, ones)
    test.assertEqual(loss.numpy()[0], 36)


def test_reshape(test, device):

    np_arr = np.arange(6, dtype=float)
    arr = wp.array(np_arr, dtype=float, device=device, requires_grad=True)
    arr_reshaped = arr.reshape((3, 2))
    arr_comp = wp.array(np_arr.reshape((3, 2)), dtype=float, device=device)
    assert_array_equal(arr_reshaped, arr_comp)
    
    arr_reshaped = arr_reshaped.reshape(6)
    assert_array_equal(arr_reshaped, arr)

    loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        wp.launch(
            kernel=sum_array,
            dim=len(arr_reshaped),
            inputs=[arr_reshaped, loss],
            device=device
        )

    tape.backward(loss=loss)
    grad = tape.gradients[arr_reshaped]

    ones = wp.array(np.ones((6,), dtype=float,), dtype=float, device=device)
    assert_array_equal(grad, ones)
    test.assertEqual(loss.numpy()[0], 15)


@wp.kernel
def compare_stepped_window_a(x: wp.array2d(dtype=float)):

    wp.expect_eq(x[0,0], 1.0)
    wp.expect_eq(x[0,1], 2.0)
    wp.expect_eq(x[1,0], 9.0)
    wp.expect_eq(x[1,1], 10.0)


@wp.kernel
def compare_stepped_window_b(x: wp.array2d(dtype=float)):

    wp.expect_eq(x[0,0], 3.0)
    wp.expect_eq(x[0,1], 4.0)
    wp.expect_eq(x[1,0], 7.0)
    wp.expect_eq(x[1,1], 8.0)
    wp.expect_eq(x[2,0], 11.0)
    wp.expect_eq(x[2,1], 12.0)


def test_slicing(test, device):

    np_arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=float)
    arr = wp.array(np_arr, dtype=float, shape=np_arr.shape, device=device, requires_grad=True)
    
    slice_a = arr[1,:,:]        # test indexing
    slice_b = arr[1:2,:,:]      # test slicing
    slice_c = arr[-1,:,:]       # test negative indexing
    slice_d = arr[-2:-1,:,:]    # test negative slicing
    slice_e = arr[-1:3,:,:]     # test mixed slicing
    slice_e2 = slice_e[0,0,:]   # test 2x slicing
    slice_f = arr[0:3:2,0,:]    # test step

    assert_array_equal(slice_a, wp.array(np_arr[1,:,:], dtype=float, device=device))
    assert_array_equal(slice_b, wp.array(np_arr[1:2,:,:], dtype=float, device=device))
    assert_array_equal(slice_c, wp.array(np_arr[-1,:,:], dtype=float, device=device))
    assert_array_equal(slice_d, wp.array(np_arr[-2:-1,:,:], dtype=float, device=device))
    assert_array_equal(slice_e, wp.array(np_arr[-1:3,:,:], dtype=float, device=device))
    assert_array_equal(slice_e2, wp.array(np_arr[2,0,:], dtype=float, device=device))

    # wp does not support copying from/to non-contiguous arrays
    # stepped windows must read on the device the original array was created on
    wp.launch(kernel=compare_stepped_window_a, dim=1, inputs=[slice_f], device=device)
    
    slice_flat = slice_b.flatten()
    loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        wp.launch(
            kernel=sum_array,
            dim=len(slice_flat),
            inputs=[slice_flat, loss],
            device=device
        )

    tape.backward(loss=loss)
    grad = tape.gradients[slice_flat]

    ones = wp.array(np.ones((4,), dtype=float,), dtype=float, device=device)
    assert_array_equal(grad, ones)
    test.assertEqual(loss.numpy()[0], 26)

    index_a = arr[1]
    index_b = arr[2, 1]
    index_c = arr[1,:]
    index_d = arr[:,1]

    assert_array_equal(index_a, wp.array(np_arr[1], dtype=float, device=device))
    assert_array_equal(index_b, wp.array(np_arr[2, 1], dtype=float, device=device))
    assert_array_equal(index_c, wp.array(np_arr[1, :], dtype=float, device=device))
    wp.launch(kernel=compare_stepped_window_b, dim=1, inputs=[index_d], device=device)

    np_arr = np.zeros(10, dtype=int)
    wp_arr = wp.array(np_arr, dtype=int, device=device)

    assert_array_equal(wp_arr[:5], wp.array(np_arr[:5], dtype=int, device=device))
    assert_array_equal(wp_arr[1:5], wp.array(np_arr[1:5], dtype=int, device=device))
    assert_array_equal(wp_arr[-9:-5:1], wp.array(np_arr[-9:-5:1], dtype=int, device=device))
    assert_array_equal(wp_arr[:5,], wp.array(np_arr[:5], dtype=int, device=device))

def test_view(test, device):

    np_arr_a = np.arange(1, 10, 1, dtype=np.uint32)
    np_arr_b = np.arange(1, 10, 1, dtype=np.float32)
    np_arr_c = np.arange(1, 10, 1, dtype=np.uint16)
    np_arr_d = np.arange(1, 10, 1, dtype=np.float16)
    np_arr_e = np.ones((4, 4), dtype=np.float32)

    wp_arr_a = wp.array(np_arr_a, dtype=wp.uint32, device=device)
    wp_arr_b = wp.array(np_arr_b, dtype=wp.float32, device=device)
    wp_arr_c = wp.array(np_arr_a, dtype=wp.uint16, device=device)
    wp_arr_d = wp.array(np_arr_b, dtype=wp.float16, device=device)
    wp_arr_e = wp.array(np_arr_e, dtype=wp.vec4, device=device)
    wp_arr_f = wp.array(np_arr_e, dtype=wp.quat, device=device)
    
    assert np.array_equal(np_arr_a.view(dtype=np.float32), wp_arr_a.view(dtype=wp.float32).numpy())
    assert np.array_equal(np_arr_b.view(dtype=np.uint32), wp_arr_b.view(dtype=wp.uint32).numpy())
    assert np.array_equal(np_arr_c.view(dtype=np.float16), wp_arr_c.view(dtype=wp.float16).numpy())
    assert np.array_equal(np_arr_d.view(dtype=np.uint16), wp_arr_d.view(dtype=wp.uint16).numpy())
    assert_array_equal(wp_arr_e.view(dtype=wp.quat), wp_arr_f)


@wp.kernel
def compare_2darrays(x: wp.array2d(dtype=float), y: wp.array2d(dtype=float), z: wp.array2d(dtype=int)):
    i,j = wp.tid()

    if x[i,j] == y[i,j]:
        z[i,j] = 1


@wp.kernel
def compare_3darrays(x: wp.array3d(dtype=float), y: wp.array3d(dtype=float), z: wp.array3d(dtype=int)):
    i,j,k = wp.tid()

    if x[i,j,k] == y[i,j,k]:
        z[i,j,k] = 1


def test_transpose(test, device):
    
    # test default transpose in non-square 2d case
    # wp does not support copying from/to non-contiguous arrays so check in kernel
    np_arr = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    arr = wp.array(np_arr, dtype=float, device=device)
    arr_transpose = arr.transpose()
    arr_compare = wp.array(np_arr.transpose(), dtype=float, device=device)
    check = wp.zeros(shape=(2, 3), dtype=int, device=device)

    wp.launch(compare_2darrays, dim=(2, 3), inputs=[arr_transpose, arr_compare, check], device=device)
    assert np.array_equal(check.numpy(), np.ones((2, 3), dtype=int))

    # test transpose in square 3d case
    # wp does not support copying from/to non-contiguous arrays so check in kernel
    np_arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=float)
    arr = wp.array3d(np_arr, dtype=float, shape=np_arr.shape, device=device, requires_grad=True)
    arr_transpose = arr.transpose((0, 2, 1))
    arr_compare = wp.array3d(np_arr.transpose((0, 2, 1)), dtype=float, device=device)
    check = wp.zeros(shape=(3, 2, 2), dtype=int, device=device)

    wp.launch(compare_3darrays, dim=(3, 2, 2), inputs=[arr_transpose, arr_compare, check], device=device)
    assert np.array_equal(check.numpy(), np.ones((3, 2, 2), dtype=int))

    # test transpose in square 3d case without axes supplied
    arr_transpose = arr.transpose()
    arr_compare = wp.array3d(np_arr.transpose(), dtype=float, device=device)
    check = wp.zeros(shape=(2, 2, 3), dtype=int, device=device)

    wp.launch(compare_3darrays, dim=(2, 2, 3), inputs=[arr_transpose, arr_compare, check], device=device)
    assert np.array_equal(check.numpy(), np.ones((2, 2, 3), dtype=int))

    # test transpose in 1d case (should be noop)
    np_arr = np.array([1, 2, 3], dtype=float)
    arr = wp.array(np_arr, dtype=float, device=device)

    assert np.array_equal(np_arr.transpose(), arr.transpose().numpy())


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
        v1 = wp.zeros(dim_x, dtype=wp.types.vector(3,wptype), device=device)
        v2 = wp.zeros((dim_x,dim_x), dtype=wp.types.vector(3,wptype), device=device)
        v3 = wp.zeros((dim_x,dim_x,dim_x), dtype=wp.types.vector(3,wptype), device=device)
        v4 = wp.zeros((dim_x,dim_x,dim_x,dim_x), dtype=wp.types.vector(3,wptype), device=device)

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
        v1 = wp.zeros(dim_x, dtype=wp.types.vector(3,wptype), device=device)
        v2 = wp.zeros((dim_x,dim_x), dtype=wp.types.vector(3,wptype), device=device)
        v3 = wp.zeros((dim_x,dim_x,dim_x), dtype=wp.types.vector(3,wptype), device=device)
        v4 = wp.zeros((dim_x,dim_x,dim_x,dim_x), dtype=wp.types.vector(3,wptype), device=device)

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
        
        vectype = wp.types.vector(3,wptype)

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
        
        vectype = wp.types.vector(3,wptype)

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

        v_np = np.random.randn(dim_x, 3).astype(nptype)
        v = wp.array(v_np,dtype=wp.types.vector(3, wptype),device=device)

        assert_np_equal(v.numpy(), v_np)


def register(parent):

    devices = get_test_devices()

    class TestArray(parent):
        pass

    add_function_test(TestArray, "test_shape", test_shape, devices=devices)
    add_function_test(TestArray, "test_flatten", test_flatten, devices=devices)
    add_function_test(TestArray, "test_reshape", test_reshape, devices=devices)
    add_function_test(TestArray, "test_slicing", test_slicing, devices=devices)
    add_function_test(TestArray, "test_transpose", test_transpose, devices=devices)
    add_function_test(TestArray, "test_view", test_view, devices=devices)

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