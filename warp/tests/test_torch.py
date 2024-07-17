# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


@wp.kernel
def op_kernel(x: wp.array(dtype=float), y: wp.array(dtype=float)):
    tid = wp.tid()
    y[tid] = 0.5 - x[tid] * 2.0


@wp.kernel
def inc(a: wp.array(dtype=float)):
    tid = wp.tid()
    a[tid] = a[tid] + 1.0


@wp.kernel
def inc_vector(a: wp.array(dtype=wp.vec3f)):
    tid = wp.tid()
    a[tid] = a[tid] + wp.vec3f(1.0)


@wp.kernel
def inc_matrix(a: wp.array(dtype=wp.mat22f)):
    tid = wp.tid()
    a[tid] = a[tid] + wp.mat22f(1.0)


@wp.kernel
def arange(start: int, step: int, a: wp.array(dtype=int)):
    tid = wp.tid()
    a[tid] = start + step * tid


# copy elements between non-contiguous 1d arrays of float
@wp.kernel
def copy1d_float_kernel(dst: wp.array(dtype=float), src: wp.array(dtype=float)):
    i = wp.tid()
    dst[i] = src[i]


# copy elements between non-contiguous 2d arrays of float
@wp.kernel
def copy2d_float_kernel(dst: wp.array2d(dtype=float), src: wp.array2d(dtype=float)):
    i, j = wp.tid()
    dst[i, j] = src[i, j]


# copy elements between non-contiguous 3d arrays of float
@wp.kernel
def copy3d_float_kernel(dst: wp.array3d(dtype=float), src: wp.array3d(dtype=float)):
    i, j, k = wp.tid()
    dst[i, j, k] = src[i, j, k]


# copy elements between non-contiguous 2d arrays of vec3
@wp.kernel
def copy2d_vec3_kernel(dst: wp.array2d(dtype=wp.vec3), src: wp.array2d(dtype=wp.vec3)):
    i, j = wp.tid()
    dst[i, j] = src[i, j]


# copy elements between non-contiguous 2d arrays of mat22
@wp.kernel
def copy2d_mat22_kernel(dst: wp.array2d(dtype=wp.mat22), src: wp.array2d(dtype=wp.mat22)):
    i, j = wp.tid()
    dst[i, j] = src[i, j]


def test_dtype_from_torch(test, device):
    import torch

    def test_conversions(torch_type, warp_type):
        test.assertEqual(wp.dtype_from_torch(torch_type), warp_type)

    test_conversions(torch.float16, wp.float16)
    test_conversions(torch.float32, wp.float32)
    test_conversions(torch.float64, wp.float64)
    test_conversions(torch.int8, wp.int8)
    test_conversions(torch.int16, wp.int16)
    test_conversions(torch.int32, wp.int32)
    test_conversions(torch.int64, wp.int64)
    test_conversions(torch.uint8, wp.uint8)
    test_conversions(torch.bool, wp.bool)


def test_dtype_to_torch(test, device):
    import torch

    def test_conversions(warp_type, torch_type):
        test.assertEqual(wp.dtype_to_torch(warp_type), torch_type)

    test_conversions(wp.float16, torch.float16)
    test_conversions(wp.float32, torch.float32)
    test_conversions(wp.float64, torch.float64)
    test_conversions(wp.int8, torch.int8)
    test_conversions(wp.int16, torch.int16)
    test_conversions(wp.int32, torch.int32)
    test_conversions(wp.int64, torch.int64)
    test_conversions(wp.uint8, torch.uint8)
    test_conversions(wp.uint16, torch.int16)
    test_conversions(wp.uint32, torch.int32)
    test_conversions(wp.uint64, torch.int64)
    test_conversions(wp.bool, torch.bool)


def test_device_conversion(test, device):
    torch_device = wp.device_to_torch(device)
    warp_device = wp.device_from_torch(torch_device)
    test.assertEqual(warp_device, device)


def test_torch_zerocopy(test, device):
    import torch

    a = wp.zeros(10, dtype=wp.float32, device=device)
    t = wp.to_torch(a)
    assert a.ptr == t.data_ptr()

    torch_device = wp.device_to_torch(device)

    t = torch.zeros(10, dtype=torch.float32, device=torch_device)
    a = wp.from_torch(t)
    assert a.ptr == t.data_ptr()


def test_from_torch(test, device):
    import torch

    torch_device = wp.device_to_torch(device)

    # automatically determine warp dtype
    def wrap_scalar_tensor_implicit(torch_dtype, expected_warp_dtype):
        t = torch.zeros(10, dtype=torch_dtype, device=torch_device)
        a = wp.from_torch(t)
        assert a.dtype == expected_warp_dtype
        assert a.shape == tuple(t.shape)

    wrap_scalar_tensor_implicit(torch.float64, wp.float64)
    wrap_scalar_tensor_implicit(torch.float32, wp.float32)
    wrap_scalar_tensor_implicit(torch.float16, wp.float16)
    wrap_scalar_tensor_implicit(torch.int64, wp.int64)
    wrap_scalar_tensor_implicit(torch.int32, wp.int32)
    wrap_scalar_tensor_implicit(torch.int16, wp.int16)
    wrap_scalar_tensor_implicit(torch.int8, wp.int8)
    wrap_scalar_tensor_implicit(torch.uint8, wp.uint8)
    wrap_scalar_tensor_implicit(torch.bool, wp.bool)

    # explicitly specify warp dtype
    def wrap_scalar_tensor_explicit(torch_dtype, expected_warp_dtype):
        t = torch.zeros(10, dtype=torch_dtype, device=torch_device)
        a = wp.from_torch(t, expected_warp_dtype)
        assert a.dtype == expected_warp_dtype
        assert a.shape == tuple(t.shape)

    wrap_scalar_tensor_explicit(torch.float64, wp.float64)
    wrap_scalar_tensor_explicit(torch.float32, wp.float32)
    wrap_scalar_tensor_explicit(torch.float16, wp.float16)
    wrap_scalar_tensor_explicit(torch.int64, wp.int64)
    wrap_scalar_tensor_explicit(torch.int64, wp.uint64)
    wrap_scalar_tensor_explicit(torch.int32, wp.int32)
    wrap_scalar_tensor_explicit(torch.int32, wp.uint32)
    wrap_scalar_tensor_explicit(torch.int16, wp.int16)
    wrap_scalar_tensor_explicit(torch.int16, wp.uint16)
    wrap_scalar_tensor_explicit(torch.int8, wp.int8)
    wrap_scalar_tensor_explicit(torch.int8, wp.uint8)
    wrap_scalar_tensor_explicit(torch.uint8, wp.uint8)
    wrap_scalar_tensor_explicit(torch.uint8, wp.int8)
    wrap_scalar_tensor_explicit(torch.bool, wp.uint8)
    wrap_scalar_tensor_explicit(torch.bool, wp.int8)
    wrap_scalar_tensor_explicit(torch.bool, wp.bool)

    def wrap_vec_tensor(n, desired_warp_dtype):
        t = torch.zeros((10, n), dtype=torch.float32, device=torch_device)
        a = wp.from_torch(t, desired_warp_dtype)
        assert a.dtype == desired_warp_dtype
        assert a.shape == (10,)

    wrap_vec_tensor(2, wp.vec2)
    wrap_vec_tensor(3, wp.vec3)
    wrap_vec_tensor(4, wp.vec4)
    wrap_vec_tensor(6, wp.spatial_vector)
    wrap_vec_tensor(7, wp.transform)

    def wrap_mat_tensor(n, m, desired_warp_dtype):
        t = torch.zeros((10, n, m), dtype=torch.float32, device=torch_device)
        a = wp.from_torch(t, desired_warp_dtype)
        assert a.dtype == desired_warp_dtype
        assert a.shape == (10,)

    wrap_mat_tensor(2, 2, wp.mat22)
    wrap_mat_tensor(3, 3, wp.mat33)
    wrap_mat_tensor(4, 4, wp.mat44)
    wrap_mat_tensor(6, 6, wp.spatial_matrix)

    def wrap_vec_tensor_with_grad(n, desired_warp_dtype):
        t = torch.zeros((10, n), dtype=torch.float32, device=torch_device)
        a = wp.from_torch(t, desired_warp_dtype, requires_grad=True)
        assert a.dtype == desired_warp_dtype
        assert a.shape == (10,)

    wrap_vec_tensor_with_grad(2, wp.vec2)
    wrap_vec_tensor_with_grad(3, wp.vec3)
    wrap_vec_tensor_with_grad(4, wp.vec4)
    wrap_vec_tensor_with_grad(6, wp.spatial_vector)
    wrap_vec_tensor_with_grad(7, wp.transform)

    def wrap_mat_tensor_with_grad(n, m, desired_warp_dtype):
        t = torch.zeros((10, n, m), dtype=torch.float32, device=torch_device)
        a = wp.from_torch(t, desired_warp_dtype, requires_grad=True)
        assert a.dtype == desired_warp_dtype
        assert a.shape == (10,)

    wrap_mat_tensor_with_grad(2, 2, wp.mat22)
    wrap_mat_tensor_with_grad(3, 3, wp.mat33)
    wrap_mat_tensor_with_grad(4, 4, wp.mat44)
    wrap_mat_tensor_with_grad(6, 6, wp.spatial_matrix)


def test_array_ctype_from_torch(test, device):
    import torch

    torch_device = wp.device_to_torch(device)

    # automatically determine warp dtype
    def wrap_scalar_tensor_implicit(torch_dtype):
        t = torch.zeros(10, dtype=torch_dtype, device=torch_device)
        a = wp.from_torch(t, return_ctype=True)
        warp_dtype = wp.dtype_from_torch(torch_dtype)
        ctype_size = ctypes.sizeof(warp_dtype._type_)
        assert a.data == t.data_ptr()
        assert a.grad == 0
        assert a.ndim == 1
        assert a.shape[0] == t.shape[0]
        assert a.strides[0] == t.stride()[0] * ctype_size

    wrap_scalar_tensor_implicit(torch.float64)
    wrap_scalar_tensor_implicit(torch.float32)
    wrap_scalar_tensor_implicit(torch.float16)
    wrap_scalar_tensor_implicit(torch.int64)
    wrap_scalar_tensor_implicit(torch.int32)
    wrap_scalar_tensor_implicit(torch.int16)
    wrap_scalar_tensor_implicit(torch.int8)
    wrap_scalar_tensor_implicit(torch.uint8)
    wrap_scalar_tensor_implicit(torch.bool)

    # explicitly specify warp dtype
    def wrap_scalar_tensor_explicit(torch_dtype, warp_dtype):
        t = torch.zeros(10, dtype=torch_dtype, device=torch_device)
        a = wp.from_torch(t, dtype=warp_dtype, return_ctype=True)
        ctype_size = ctypes.sizeof(warp_dtype._type_)
        assert a.data == t.data_ptr()
        assert a.grad == 0
        assert a.ndim == 1
        assert a.shape[0] == t.shape[0]
        assert a.strides[0] == t.stride()[0] * ctype_size

    wrap_scalar_tensor_explicit(torch.float64, wp.float64)
    wrap_scalar_tensor_explicit(torch.float32, wp.float32)
    wrap_scalar_tensor_explicit(torch.float16, wp.float16)
    wrap_scalar_tensor_explicit(torch.int64, wp.int64)
    wrap_scalar_tensor_explicit(torch.int64, wp.uint64)
    wrap_scalar_tensor_explicit(torch.int32, wp.int32)
    wrap_scalar_tensor_explicit(torch.int32, wp.uint32)
    wrap_scalar_tensor_explicit(torch.int16, wp.int16)
    wrap_scalar_tensor_explicit(torch.int16, wp.uint16)
    wrap_scalar_tensor_explicit(torch.int8, wp.int8)
    wrap_scalar_tensor_explicit(torch.int8, wp.uint8)
    wrap_scalar_tensor_explicit(torch.uint8, wp.uint8)
    wrap_scalar_tensor_explicit(torch.uint8, wp.int8)
    wrap_scalar_tensor_explicit(torch.bool, wp.uint8)
    wrap_scalar_tensor_explicit(torch.bool, wp.int8)
    wrap_scalar_tensor_explicit(torch.bool, wp.bool)

    def wrap_vec_tensor(vec_dtype):
        t = torch.zeros((10, vec_dtype._length_), dtype=torch.float32, device=torch_device)
        a = wp.from_torch(t, dtype=vec_dtype, return_ctype=True)
        ctype_size = ctypes.sizeof(vec_dtype._type_)
        assert a.data == t.data_ptr()
        assert a.grad == 0
        assert a.ndim == 1
        assert a.shape[0] == t.shape[0]
        assert a.strides[0] == t.stride()[0] * ctype_size

    wrap_vec_tensor(wp.vec2)
    wrap_vec_tensor(wp.vec3)
    wrap_vec_tensor(wp.vec4)
    wrap_vec_tensor(wp.spatial_vector)
    wrap_vec_tensor(wp.transform)

    def wrap_mat_tensor(mat_dtype):
        t = torch.zeros((10, *mat_dtype._shape_), dtype=torch.float32, device=torch_device)
        a = wp.from_torch(t, dtype=mat_dtype, return_ctype=True)
        ctype_size = ctypes.sizeof(mat_dtype._type_)
        assert a.data == t.data_ptr()
        assert a.grad == 0
        assert a.ndim == 1
        assert a.shape[0] == t.shape[0]
        assert a.strides[0] == t.stride()[0] * ctype_size

    wrap_mat_tensor(wp.mat22)
    wrap_mat_tensor(wp.mat33)
    wrap_mat_tensor(wp.mat44)
    wrap_mat_tensor(wp.spatial_matrix)

    def wrap_vec_tensor_with_existing_grad(vec_dtype):
        t = torch.zeros((10, vec_dtype._length_), dtype=torch.float32, device=torch_device, requires_grad=True)
        t.grad = torch.zeros((10, vec_dtype._length_), dtype=torch.float32, device=torch_device)
        a = wp.from_torch(t, dtype=vec_dtype, return_ctype=True)
        ctype_size = ctypes.sizeof(vec_dtype._type_)
        assert a.data == t.data_ptr()
        assert a.grad == t.grad.data_ptr()
        assert a.ndim == 1
        assert a.shape[0] == t.shape[0]
        assert a.strides[0] == t.stride()[0] * ctype_size

    wrap_vec_tensor_with_existing_grad(wp.vec2)
    wrap_vec_tensor_with_existing_grad(wp.vec3)
    wrap_vec_tensor_with_existing_grad(wp.vec4)
    wrap_vec_tensor_with_existing_grad(wp.spatial_vector)
    wrap_vec_tensor_with_existing_grad(wp.transform)

    def wrap_vec_tensor_with_new_grad(vec_dtype):
        t = torch.zeros((10, vec_dtype._length_), dtype=torch.float32, device=torch_device)
        a = wp.from_torch(t, dtype=vec_dtype, requires_grad=True, return_ctype=True)
        ctype_size = ctypes.sizeof(vec_dtype._type_)
        assert a.data == t.data_ptr()
        assert a.grad == t.grad.data_ptr()
        assert a.ndim == 1
        assert a.shape[0] == t.shape[0]
        assert a.strides[0] == t.stride()[0] * ctype_size

    wrap_vec_tensor_with_new_grad(wp.vec2)
    wrap_vec_tensor_with_new_grad(wp.vec3)
    wrap_vec_tensor_with_new_grad(wp.vec4)
    wrap_vec_tensor_with_new_grad(wp.spatial_vector)
    wrap_vec_tensor_with_new_grad(wp.transform)

    def wrap_vec_tensor_with_torch_grad(vec_dtype):
        t = torch.zeros((10, vec_dtype._length_), dtype=torch.float32, device=torch_device)
        grad = torch.zeros((10, vec_dtype._length_), dtype=torch.float32, device=torch_device)
        a = wp.from_torch(t, dtype=vec_dtype, grad=grad, return_ctype=True)
        ctype_size = ctypes.sizeof(vec_dtype._type_)
        assert a.data == t.data_ptr()
        assert a.grad == grad.data_ptr()
        assert a.ndim == 1
        assert a.shape[0] == t.shape[0]
        assert a.strides[0] == t.stride()[0] * ctype_size

    wrap_vec_tensor_with_torch_grad(wp.vec2)
    wrap_vec_tensor_with_torch_grad(wp.vec3)
    wrap_vec_tensor_with_torch_grad(wp.vec4)
    wrap_vec_tensor_with_torch_grad(wp.spatial_vector)
    wrap_vec_tensor_with_torch_grad(wp.transform)

    def wrap_vec_tensor_with_warp_grad(vec_dtype):
        t = torch.zeros((10, vec_dtype._length_), dtype=torch.float32, device=torch_device)
        grad = wp.zeros(10, dtype=vec_dtype, device=device)
        a = wp.from_torch(t, dtype=vec_dtype, grad=grad, return_ctype=True)
        ctype_size = ctypes.sizeof(vec_dtype._type_)
        assert a.data == t.data_ptr()
        assert a.grad == grad.ptr
        assert a.ndim == 1
        assert a.shape[0] == t.shape[0]
        assert a.strides[0] == t.stride()[0] * ctype_size

    wrap_vec_tensor_with_warp_grad(wp.vec2)
    wrap_vec_tensor_with_warp_grad(wp.vec3)
    wrap_vec_tensor_with_warp_grad(wp.vec4)
    wrap_vec_tensor_with_warp_grad(wp.spatial_vector)
    wrap_vec_tensor_with_warp_grad(wp.transform)


def test_to_torch(test, device):
    import torch

    def wrap_scalar_array(warp_dtype, expected_torch_dtype):
        a = wp.zeros(10, dtype=warp_dtype, device=device)
        t = wp.to_torch(a)
        assert t.dtype == expected_torch_dtype
        assert tuple(t.shape) == a.shape

    wrap_scalar_array(wp.float64, torch.float64)
    wrap_scalar_array(wp.float32, torch.float32)
    wrap_scalar_array(wp.float16, torch.float16)
    wrap_scalar_array(wp.int64, torch.int64)
    wrap_scalar_array(wp.int32, torch.int32)
    wrap_scalar_array(wp.int16, torch.int16)
    wrap_scalar_array(wp.int8, torch.int8)
    wrap_scalar_array(wp.uint8, torch.uint8)
    wrap_scalar_array(wp.bool, torch.bool)

    # not supported by torch
    # wrap_scalar_array(wp.uint64, torch.int64)
    # wrap_scalar_array(wp.uint32, torch.int32)
    # wrap_scalar_array(wp.uint16, torch.int16)

    def wrap_vec_array(n, warp_dtype):
        a = wp.zeros(10, dtype=warp_dtype, device=device)
        t = wp.to_torch(a)
        assert t.dtype == torch.float32
        assert tuple(t.shape) == (10, n)

    wrap_vec_array(2, wp.vec2)
    wrap_vec_array(3, wp.vec3)
    wrap_vec_array(4, wp.vec4)
    wrap_vec_array(6, wp.spatial_vector)
    wrap_vec_array(7, wp.transform)

    def wrap_mat_array(n, m, warp_dtype):
        a = wp.zeros(10, dtype=warp_dtype, device=device)
        t = wp.to_torch(a)
        assert t.dtype == torch.float32
        assert tuple(t.shape) == (10, n, m)

    wrap_mat_array(2, 2, wp.mat22)
    wrap_mat_array(3, 3, wp.mat33)
    wrap_mat_array(4, 4, wp.mat44)
    wrap_mat_array(6, 6, wp.spatial_matrix)


def test_from_torch_slices(test, device):
    import torch

    torch_device = wp.device_to_torch(device)

    # 1D slice, contiguous
    t_base = torch.arange(10, dtype=torch.float32, device=torch_device)
    t = t_base[2:9]
    a = wp.from_torch(t)
    assert a.ptr == t.data_ptr()
    assert a.is_contiguous
    assert a.shape == tuple(t.shape)
    assert_np_equal(a.numpy(), t.cpu().numpy())

    # 1D slice with non-contiguous stride
    t_base = torch.arange(10, dtype=torch.float32, device=torch_device)
    t = t_base[2:9:2]
    a = wp.from_torch(t)
    assert a.ptr == t.data_ptr()
    assert not a.is_contiguous
    assert a.shape == tuple(t.shape)
    # copy contents to contiguous array
    a_contiguous = wp.empty_like(a)
    wp.launch(copy1d_float_kernel, dim=a.shape, inputs=[a_contiguous, a], device=device)
    assert_np_equal(a_contiguous.numpy(), t.cpu().numpy())

    # 2D slices (non-contiguous)
    t_base = torch.arange(24, dtype=torch.float32, device=torch_device).reshape((4, 6))
    t = t_base[1:3, 2:5]
    a = wp.from_torch(t)
    assert a.ptr == t.data_ptr()
    assert not a.is_contiguous
    assert a.shape == tuple(t.shape)
    # copy contents to contiguous array
    a_contiguous = wp.empty_like(a)
    wp.launch(copy2d_float_kernel, dim=a.shape, inputs=[a_contiguous, a], device=device)
    assert_np_equal(a_contiguous.numpy(), t.cpu().numpy())

    # 3D slices (non-contiguous)
    t_base = torch.arange(36, dtype=torch.float32, device=torch_device).reshape((4, 3, 3))
    t = t_base[::2, 0:1, 1:2]
    a = wp.from_torch(t)
    assert a.ptr == t.data_ptr()
    assert not a.is_contiguous
    assert a.shape == tuple(t.shape)
    # copy contents to contiguous array
    a_contiguous = wp.empty_like(a)
    wp.launch(copy3d_float_kernel, dim=a.shape, inputs=[a_contiguous, a], device=device)
    assert_np_equal(a_contiguous.numpy(), t.cpu().numpy())

    # 2D slices of vec3 (inner contiguous, outer non-contiguous)
    t_base = torch.arange(150, dtype=torch.float32, device=torch_device).reshape((10, 5, 3))
    t = t_base[1:7:2, 2:5]
    a = wp.from_torch(t, dtype=wp.vec3)
    assert a.ptr == t.data_ptr()
    assert not a.is_contiguous
    assert a.shape == tuple(t.shape[:-1])
    # copy contents to contiguous array
    a_contiguous = wp.empty_like(a)
    wp.launch(copy2d_vec3_kernel, dim=a.shape, inputs=[a_contiguous, a], device=device)
    assert_np_equal(a_contiguous.numpy(), t.cpu().numpy())

    # 2D slices of mat22 (inner contiguous, outer non-contiguous)
    t_base = torch.arange(200, dtype=torch.float32, device=torch_device).reshape((10, 5, 2, 2))
    t = t_base[1:7:2, 2:5]
    a = wp.from_torch(t, dtype=wp.mat22)
    assert a.ptr == t.data_ptr()
    assert not a.is_contiguous
    assert a.shape == tuple(t.shape[:-2])
    # copy contents to contiguous array
    a_contiguous = wp.empty_like(a)
    wp.launch(copy2d_mat22_kernel, dim=a.shape, inputs=[a_contiguous, a], device=device)
    assert_np_equal(a_contiguous.numpy(), t.cpu().numpy())


def test_from_torch_zero_strides(test, device):
    import torch

    torch_device = wp.device_to_torch(device)

    t_base = torch.arange(9, dtype=torch.float32, device=torch_device).reshape((3, 3))

    # expand outermost dimension
    t = t_base.unsqueeze(0).expand(3, -1, -1)
    a = wp.from_torch(t)
    assert a.ptr == t.data_ptr()
    assert not a.is_contiguous
    assert a.shape == tuple(t.shape)
    a_contiguous = wp.empty_like(a)
    wp.launch(copy3d_float_kernel, dim=a.shape, inputs=[a_contiguous, a], device=device)
    assert_np_equal(a_contiguous.numpy(), t.cpu().numpy())

    # expand middle dimension
    t = t_base.unsqueeze(1).expand(-1, 3, -1)
    a = wp.from_torch(t)
    assert a.ptr == t.data_ptr()
    assert not a.is_contiguous
    assert a.shape == tuple(t.shape)
    a_contiguous = wp.empty_like(a)
    wp.launch(copy3d_float_kernel, dim=a.shape, inputs=[a_contiguous, a], device=device)
    assert_np_equal(a_contiguous.numpy(), t.cpu().numpy())

    # expand innermost dimension
    t = t_base.unsqueeze(2).expand(-1, -1, 3)
    a = wp.from_torch(t)
    assert a.ptr == t.data_ptr()
    assert not a.is_contiguous
    assert a.shape == tuple(t.shape)
    a_contiguous = wp.empty_like(a)
    wp.launch(copy3d_float_kernel, dim=a.shape, inputs=[a_contiguous, a], device=device)
    assert_np_equal(a_contiguous.numpy(), t.cpu().numpy())


def test_torch_mgpu_from_torch(test, device):
    import torch

    n = 32

    t0 = torch.arange(0, n, 1, dtype=torch.int32, device="cuda:0")
    t1 = torch.arange(0, n * 2, 2, dtype=torch.int32, device="cuda:1")

    a0 = wp.from_torch(t0, dtype=wp.int32)
    a1 = wp.from_torch(t1, dtype=wp.int32)

    assert a0.device == "cuda:0"
    assert a1.device == "cuda:1"

    expected0 = np.arange(0, n, 1)
    expected1 = np.arange(0, n * 2, 2)

    assert_np_equal(a0.numpy(), expected0)
    assert_np_equal(a1.numpy(), expected1)


def test_torch_mgpu_to_torch(test, device):
    n = 32

    with wp.ScopedDevice("cuda:0"):
        a0 = wp.empty(n, dtype=wp.int32)
        wp.launch(arange, dim=a0.size, inputs=[0, 1, a0])

    with wp.ScopedDevice("cuda:1"):
        a1 = wp.empty(n, dtype=wp.int32)
        wp.launch(arange, dim=a1.size, inputs=[0, 2, a1])

    t0 = wp.to_torch(a0)
    t1 = wp.to_torch(a1)

    assert str(t0.device) == "cuda:0"
    assert str(t1.device) == "cuda:1"

    expected0 = np.arange(0, n, 1, dtype=np.int32)
    expected1 = np.arange(0, n * 2, 2, dtype=np.int32)

    assert_np_equal(t0.cpu().numpy(), expected0)
    assert_np_equal(t1.cpu().numpy(), expected1)


def test_torch_mgpu_interop(test, device):
    import torch

    n = 1024 * 1024

    with torch.cuda.device(0):
        t0 = torch.arange(n, dtype=torch.float32, device="cuda")
        a0 = wp.from_torch(t0)
        wp.launch(inc, dim=a0.size, inputs=[a0], stream=wp.stream_from_torch())

    with torch.cuda.device(1):
        t1 = torch.arange(n, dtype=torch.float32, device="cuda")
        a1 = wp.from_torch(t1)
        wp.launch(inc, dim=a1.size, inputs=[a1], stream=wp.stream_from_torch())

    assert a0.device == "cuda:0"
    assert a1.device == "cuda:1"

    expected = np.arange(n, dtype=int) + 1

    # ensure the torch tensors were modified by warp
    assert_np_equal(t0.cpu().numpy(), expected)
    assert_np_equal(t1.cpu().numpy(), expected)


def test_torch_autograd(test, device):
    """Test torch autograd with a custom Warp op"""

    import torch

    # custom autograd op
    class TestFunc(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            # allocate output array
            y = torch.empty_like(x)

            ctx.x = x
            ctx.y = y

            wp.launch(kernel=op_kernel, dim=len(x), inputs=[wp.from_torch(x)], outputs=[wp.from_torch(y)])

            return y

        @staticmethod
        def backward(ctx, adj_y):
            # adjoints should be allocated as zero initialized
            adj_x = torch.zeros_like(ctx.x).contiguous()
            adj_y = adj_y.contiguous()

            wp_x = wp.from_torch(ctx.x, grad=adj_x)
            wp_y = wp.from_torch(ctx.y, grad=adj_y)

            wp.launch(
                kernel=op_kernel,
                dim=len(ctx.x),
                # fwd inputs
                inputs=[wp_x],
                outputs=[wp_y],
                # adj inputs (already stored in input/output arrays, passing null pointers)
                adj_inputs=[None],
                adj_outputs=[None],
                adjoint=True,
            )

            return adj_x

    # run autograd on given device
    with wp.ScopedDevice(device):
        torch_device = wp.device_to_torch(device)

        # input data
        x = torch.ones(16, dtype=torch.float32, device=torch_device, requires_grad=True)

        # execute op
        y = TestFunc.apply(x)

        # compute grads
        l = y.sum()
        l.backward()

        passed = (x.grad == -2.0).all()
        assert passed.item()


def test_torch_graph_torch_stream(test, device):
    """Capture Torch graph on Torch stream"""

    wp.load_module(device=device)

    import torch

    torch_device = wp.device_to_torch(device)

    n = 1024 * 1024
    t = torch.zeros(n, dtype=torch.float32, device=torch_device)
    a = wp.from_torch(t)

    g = torch.cuda.CUDAGraph()

    # create a device-specific torch stream to use for capture
    # (otherwise torch.cuda.graph reuses its capture stream, which can be problematic if it's from a different device)
    torch_stream = torch.cuda.Stream(device=torch_device)

    # make warp use the same stream
    warp_stream = wp.stream_from_torch(torch_stream)

    # capture graph
    with wp.ScopedStream(warp_stream), torch.cuda.graph(g, stream=torch_stream):
        wp.capture_begin(force_module_load=False, external=True)
        try:
            t += 1.0
            wp.launch(inc, dim=n, inputs=[a])
            t += 1.0
            wp.launch(inc, dim=n, inputs=[a])
        finally:
            wp.capture_end()

    # replay graph
    num_iters = 10
    for _i in range(num_iters):
        g.replay()

    passed = (t == num_iters * 4.0).all()
    assert passed.item()


def test_torch_graph_warp_stream(test, device):
    """Capture Torch graph on Warp stream"""

    import torch

    torch_device = wp.device_to_torch(device)

    n = 1024 * 1024
    t = torch.zeros(n, dtype=torch.float32, device=torch_device)
    a = wp.from_torch(t)

    g = torch.cuda.CUDAGraph()

    # make torch use the warp stream from the given device
    torch_stream = wp.stream_to_torch(device)

    # capture graph
    with wp.ScopedDevice(device), torch.cuda.graph(g, stream=torch_stream):
        wp.capture_begin(force_module_load=False, external=True)
        try:
            t += 1.0
            wp.launch(inc, dim=n, inputs=[a])
            t += 1.0
            wp.launch(inc, dim=n, inputs=[a])
        finally:
            wp.capture_end()

    # replay graph
    num_iters = 10
    for _i in range(num_iters):
        g.replay()

    passed = (t == num_iters * 4.0).all()
    assert passed.item()


def test_warp_graph_warp_stream(test, device):
    """Capture Warp graph on Warp stream"""

    import torch

    torch_device = wp.device_to_torch(device)

    n = 1024 * 1024
    t = torch.zeros(n, dtype=torch.float32, device=torch_device)
    a = wp.from_torch(t)

    # make torch use the warp stream from the given device
    torch_stream = wp.stream_to_torch(device)

    # capture graph
    with wp.ScopedDevice(device), torch.cuda.stream(torch_stream):
        wp.capture_begin(force_module_load=False)
        try:
            t += 1.0
            wp.launch(inc, dim=n, inputs=[a])
            t += 1.0
            wp.launch(inc, dim=n, inputs=[a])
        finally:
            g = wp.capture_end()

    # replay graph
    num_iters = 10
    for _i in range(num_iters):
        wp.capture_launch(g)

    passed = (t == num_iters * 4.0).all()
    assert passed.item()


def test_warp_graph_torch_stream(test, device):
    """Capture Warp graph on Torch stream"""

    wp.load_module(device=device)

    import torch

    torch_device = wp.device_to_torch(device)

    n = 1024 * 1024
    t = torch.zeros(n, dtype=torch.float32, device=torch_device)
    a = wp.from_torch(t)

    # create a device-specific torch stream to use for capture
    # (the default torch stream is not suitable for graph capture)
    torch_stream = torch.cuda.Stream(device=torch_device)

    # make warp use the same stream
    warp_stream = wp.stream_from_torch(torch_stream)

    # capture graph
    with wp.ScopedStream(warp_stream), torch.cuda.stream(torch_stream):
        wp.capture_begin(force_module_load=False)
        try:
            t += 1.0
            wp.launch(inc, dim=n, inputs=[a])
            t += 1.0
            wp.launch(inc, dim=n, inputs=[a])
        finally:
            g = wp.capture_end()

    # replay graph
    num_iters = 10
    for _i in range(num_iters):
        wp.capture_launch(g)

    passed = (t == num_iters * 4.0).all()
    assert passed.item()


def test_direct(test, device):
    """Pass Torch tensors to Warp kernels directly"""

    import torch

    torch_device = wp.device_to_torch(device)
    n = 12

    s = torch.arange(n, dtype=torch.float32, device=torch_device)
    v = torch.arange(n, dtype=torch.float32, device=torch_device).reshape((n // 3, 3))
    m = torch.arange(n, dtype=torch.float32, device=torch_device).reshape((n // 4, 2, 2))

    wp.launch(inc, dim=n, inputs=[s], device=device)
    wp.launch(inc_vector, dim=n // 3, inputs=[v], device=device)
    wp.launch(inc_matrix, dim=n // 4, inputs=[m], device=device)

    expected = torch.arange(1, n + 1, dtype=torch.float32, device=torch_device)

    assert torch.equal(s, expected)
    assert torch.equal(v.reshape(n), expected)
    assert torch.equal(m.reshape(n), expected)


class TestTorch(unittest.TestCase):
    pass


test_devices = get_test_devices()

try:
    import torch

    # check which Warp devices work with Torch
    # CUDA devices may fail if Torch was not compiled with CUDA support
    torch_compatible_devices = []
    torch_compatible_cuda_devices = []

    for d in test_devices:
        try:
            t = torch.arange(10, device=wp.device_to_torch(d))
            t += 1
            torch_compatible_devices.append(d)
            if d.is_cuda:
                torch_compatible_cuda_devices.append(d)
        except Exception as e:
            print(f"Skipping Torch tests on device '{d}' due to exception: {e}")

    add_function_test(TestTorch, "test_dtype_from_torch", test_dtype_from_torch, devices=None)
    add_function_test(TestTorch, "test_dtype_to_torch", test_dtype_to_torch, devices=None)

    if torch_compatible_devices:
        add_function_test(TestTorch, "test_device_conversion", test_device_conversion, devices=torch_compatible_devices)
        add_function_test(TestTorch, "test_from_torch", test_from_torch, devices=torch_compatible_devices)
        add_function_test(TestTorch, "test_from_torch_slices", test_from_torch_slices, devices=torch_compatible_devices)
        add_function_test(
            TestTorch, "test_array_ctype_from_torch", test_array_ctype_from_torch, devices=torch_compatible_devices
        )
        add_function_test(
            TestTorch,
            "test_from_torch_zero_strides",
            test_from_torch_zero_strides,
            devices=torch_compatible_devices,
        )
        add_function_test(TestTorch, "test_to_torch", test_to_torch, devices=torch_compatible_devices)
        add_function_test(TestTorch, "test_torch_zerocopy", test_torch_zerocopy, devices=torch_compatible_devices)
        add_function_test(TestTorch, "test_torch_autograd", test_torch_autograd, devices=torch_compatible_devices)
        add_function_test(TestTorch, "test_direct", test_direct, devices=torch_compatible_devices)

    if torch_compatible_cuda_devices:
        add_function_test(
            TestTorch,
            "test_torch_graph_torch_stream",
            test_torch_graph_torch_stream,
            devices=torch_compatible_cuda_devices,
        )
        add_function_test(
            TestTorch,
            "test_torch_graph_warp_stream",
            test_torch_graph_warp_stream,
            devices=torch_compatible_cuda_devices,
        )
        add_function_test(
            TestTorch,
            "test_warp_graph_warp_stream",
            test_warp_graph_warp_stream,
            devices=torch_compatible_cuda_devices,
        )
        add_function_test(
            TestTorch,
            "test_warp_graph_torch_stream",
            test_warp_graph_torch_stream,
            devices=torch_compatible_cuda_devices,
        )

    # multi-GPU tests
    if len(torch_compatible_cuda_devices) > 1:
        add_function_test(TestTorch, "test_torch_mgpu_from_torch", test_torch_mgpu_from_torch)
        add_function_test(TestTorch, "test_torch_mgpu_to_torch", test_torch_mgpu_to_torch)
        add_function_test(TestTorch, "test_torch_mgpu_interop", test_torch_mgpu_interop)

except Exception as e:
    print(f"Skipping Torch tests due to exception: {e}")


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
