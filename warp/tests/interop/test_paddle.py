# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

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


def test_dtype_from_paddle(test, device):
    import paddle

    def test_conversions(paddle_type, warp_type):
        test.assertEqual(wp.dtype_from_paddle(paddle_type), warp_type)

    test_conversions(paddle.float16, wp.float16)
    test_conversions(paddle.float32, wp.float32)
    test_conversions(paddle.float64, wp.float64)
    test_conversions(paddle.int8, wp.int8)
    test_conversions(paddle.int16, wp.int16)
    test_conversions(paddle.int32, wp.int32)
    test_conversions(paddle.int64, wp.int64)
    test_conversions(paddle.uint8, wp.uint8)
    test_conversions(paddle.bool, wp.bool)


def test_dtype_to_paddle(test, device):
    import paddle

    def test_conversions(warp_type, paddle_type):
        test.assertEqual(wp.dtype_to_paddle(warp_type), paddle_type)

    test_conversions(wp.float16, paddle.float16)
    test_conversions(wp.float32, paddle.float32)
    test_conversions(wp.float64, paddle.float64)
    test_conversions(wp.int8, paddle.int8)
    test_conversions(wp.int16, paddle.int16)
    test_conversions(wp.int32, paddle.int32)
    test_conversions(wp.int64, paddle.int64)
    test_conversions(wp.uint8, paddle.uint8)
    test_conversions(wp.uint16, paddle.int16)
    test_conversions(wp.uint32, paddle.int32)
    test_conversions(wp.uint64, paddle.int64)
    test_conversions(wp.bool, paddle.bool)


def test_device_conversion(test, device):
    paddle_device = wp.device_to_paddle(device)
    warp_device = wp.device_from_paddle(paddle_device)
    test.assertEqual(warp_device, device)


def test_paddle_zerocopy(test, device):
    import paddle

    a = wp.zeros(10, dtype=wp.float32, device=device)
    t = wp.to_paddle(a)
    assert a.ptr == t.data_ptr()

    paddle_device = wp.device_to_paddle(device)

    t = paddle.zeros([10], dtype=paddle.float32).to(device=paddle_device)
    a = wp.from_paddle(t)
    assert a.ptr == t.data_ptr()


def test_from_paddle(test, device):
    import paddle

    paddle_device = wp.device_to_paddle(device)

    # automatically determine warp dtype
    def wrap_scalar_tensor_implicit(paddle_dtype, expected_warp_dtype):
        t = paddle.zeros([10], dtype=paddle_dtype).to(device=paddle_device)
        a = wp.from_paddle(t)
        assert a.dtype == expected_warp_dtype
        assert a.shape == tuple(t.shape)

    wrap_scalar_tensor_implicit(paddle.float64, wp.float64)
    wrap_scalar_tensor_implicit(paddle.float32, wp.float32)
    wrap_scalar_tensor_implicit(paddle.float16, wp.float16)
    wrap_scalar_tensor_implicit(paddle.int64, wp.int64)
    wrap_scalar_tensor_implicit(paddle.int32, wp.int32)
    wrap_scalar_tensor_implicit(paddle.int16, wp.int16)
    wrap_scalar_tensor_implicit(paddle.int8, wp.int8)
    wrap_scalar_tensor_implicit(paddle.uint8, wp.uint8)
    wrap_scalar_tensor_implicit(paddle.bool, wp.bool)

    # explicitly specify warp dtype
    def wrap_scalar_tensor_explicit(paddle_dtype, expected_warp_dtype):
        t = paddle.zeros([10], dtype=paddle_dtype).to(device=paddle_device)
        a = wp.from_paddle(t, expected_warp_dtype)
        assert a.dtype == expected_warp_dtype
        assert a.shape == tuple(t.shape)

    wrap_scalar_tensor_explicit(paddle.float64, wp.float64)
    wrap_scalar_tensor_explicit(paddle.float32, wp.float32)
    wrap_scalar_tensor_explicit(paddle.float16, wp.float16)
    wrap_scalar_tensor_explicit(paddle.int64, wp.int64)
    wrap_scalar_tensor_explicit(paddle.int64, wp.uint64)
    wrap_scalar_tensor_explicit(paddle.int32, wp.int32)
    wrap_scalar_tensor_explicit(paddle.int32, wp.uint32)
    wrap_scalar_tensor_explicit(paddle.int16, wp.int16)
    wrap_scalar_tensor_explicit(paddle.int16, wp.uint16)
    wrap_scalar_tensor_explicit(paddle.int8, wp.int8)
    wrap_scalar_tensor_explicit(paddle.int8, wp.uint8)
    wrap_scalar_tensor_explicit(paddle.uint8, wp.uint8)
    wrap_scalar_tensor_explicit(paddle.uint8, wp.int8)
    wrap_scalar_tensor_explicit(paddle.bool, wp.uint8)
    wrap_scalar_tensor_explicit(paddle.bool, wp.int8)
    wrap_scalar_tensor_explicit(paddle.bool, wp.bool)

    def wrap_vec_tensor(n, desired_warp_dtype):
        t = paddle.zeros((10, n), dtype=paddle.float32).to(device=paddle_device)
        a = wp.from_paddle(t, desired_warp_dtype)
        assert a.dtype == desired_warp_dtype
        assert a.shape == (10,)

    wrap_vec_tensor(2, wp.vec2)
    wrap_vec_tensor(3, wp.vec3)
    wrap_vec_tensor(4, wp.vec4)
    wrap_vec_tensor(6, wp.spatial_vector)
    wrap_vec_tensor(7, wp.transform)

    def wrap_mat_tensor(n, m, desired_warp_dtype):
        t = paddle.zeros((10, n, m), dtype=paddle.float32).to(device=paddle_device)
        a = wp.from_paddle(t, desired_warp_dtype)
        assert a.dtype == desired_warp_dtype
        assert a.shape == (10,)

    wrap_mat_tensor(2, 2, wp.mat22)
    wrap_mat_tensor(3, 3, wp.mat33)
    wrap_mat_tensor(4, 4, wp.mat44)
    wrap_mat_tensor(6, 6, wp.spatial_matrix)

    def wrap_vec_tensor_with_grad(n, desired_warp_dtype):
        t = paddle.zeros((10, n), dtype=paddle.float32).to(device=paddle_device)
        a = wp.from_paddle(t, desired_warp_dtype)
        a.reuqires_grad = True
        assert a.dtype == desired_warp_dtype
        assert a.shape == (10,)

    wrap_vec_tensor_with_grad(2, wp.vec2)
    wrap_vec_tensor_with_grad(3, wp.vec3)
    wrap_vec_tensor_with_grad(4, wp.vec4)
    wrap_vec_tensor_with_grad(6, wp.spatial_vector)
    wrap_vec_tensor_with_grad(7, wp.transform)

    def wrap_mat_tensor_with_grad(n, m, desired_warp_dtype):
        t = paddle.zeros((10, n, m), dtype=paddle.float32).to(device=paddle_device)
        a = wp.from_paddle(t, desired_warp_dtype, requires_grad=True)
        assert a.dtype == desired_warp_dtype
        assert a.shape == (10,)

    wrap_mat_tensor_with_grad(2, 2, wp.mat22)
    wrap_mat_tensor_with_grad(3, 3, wp.mat33)
    wrap_mat_tensor_with_grad(4, 4, wp.mat44)
    wrap_mat_tensor_with_grad(6, 6, wp.spatial_matrix)


def test_array_ctype_from_paddle(test, device):
    import paddle

    paddle_device = wp.device_to_paddle(device)

    # automatically determine warp dtype
    def wrap_scalar_tensor_implicit(paddle_dtype):
        t = paddle.zeros([10], dtype=paddle_dtype).to(device=paddle_device)
        a = wp.from_paddle(t, return_ctype=True)
        warp_dtype = wp.dtype_from_paddle(paddle_dtype)
        ctype_size = ctypes.sizeof(warp_dtype._type_)
        assert a.data == t.data_ptr()
        assert a.grad == 0
        assert a.ndim == 1
        assert a.shape[0] == t.shape[0]
        assert a.strides[0] == t.strides[0] * ctype_size

    wrap_scalar_tensor_implicit(paddle.float64)
    wrap_scalar_tensor_implicit(paddle.float32)
    wrap_scalar_tensor_implicit(paddle.float16)
    wrap_scalar_tensor_implicit(paddle.int64)
    wrap_scalar_tensor_implicit(paddle.int32)
    wrap_scalar_tensor_implicit(paddle.int16)
    wrap_scalar_tensor_implicit(paddle.int8)
    wrap_scalar_tensor_implicit(paddle.uint8)
    wrap_scalar_tensor_implicit(paddle.bool)

    # explicitly specify warp dtype
    def wrap_scalar_tensor_explicit(paddle_dtype, warp_dtype):
        t = paddle.zeros([10], dtype=paddle_dtype).to(device=paddle_device)
        a = wp.from_paddle(t, dtype=warp_dtype, return_ctype=True)
        ctype_size = ctypes.sizeof(warp_dtype._type_)
        assert a.data == t.data_ptr()
        assert a.grad == 0
        assert a.ndim == 1
        assert a.shape[0] == t.shape[0]
        assert a.strides[0] == t.strides[0] * ctype_size

    wrap_scalar_tensor_explicit(paddle.float64, wp.float64)
    wrap_scalar_tensor_explicit(paddle.float32, wp.float32)
    wrap_scalar_tensor_explicit(paddle.float16, wp.float16)
    wrap_scalar_tensor_explicit(paddle.int64, wp.int64)
    wrap_scalar_tensor_explicit(paddle.int64, wp.uint64)
    wrap_scalar_tensor_explicit(paddle.int32, wp.int32)
    wrap_scalar_tensor_explicit(paddle.int32, wp.uint32)
    wrap_scalar_tensor_explicit(paddle.int16, wp.int16)
    wrap_scalar_tensor_explicit(paddle.int16, wp.uint16)
    wrap_scalar_tensor_explicit(paddle.int8, wp.int8)
    wrap_scalar_tensor_explicit(paddle.int8, wp.uint8)
    wrap_scalar_tensor_explicit(paddle.uint8, wp.uint8)
    wrap_scalar_tensor_explicit(paddle.uint8, wp.int8)
    wrap_scalar_tensor_explicit(paddle.bool, wp.uint8)
    wrap_scalar_tensor_explicit(paddle.bool, wp.int8)
    wrap_scalar_tensor_explicit(paddle.bool, wp.bool)

    def wrap_vec_tensor(vec_dtype):
        t = paddle.zeros((10, vec_dtype._length_), dtype=paddle.float32).to(device=paddle_device)
        a = wp.from_paddle(t, dtype=vec_dtype, return_ctype=True)
        ctype_size = ctypes.sizeof(vec_dtype._type_)
        assert a.data == t.data_ptr()
        assert a.grad == 0
        assert a.ndim == 1
        assert a.shape[0] == t.shape[0]
        assert a.strides[0] == t.strides[0] * ctype_size

    wrap_vec_tensor(wp.vec2)
    wrap_vec_tensor(wp.vec3)
    wrap_vec_tensor(wp.vec4)
    wrap_vec_tensor(wp.spatial_vector)
    wrap_vec_tensor(wp.transform)

    def wrap_mat_tensor(mat_dtype):
        t = paddle.zeros((10, *mat_dtype._shape_), dtype=paddle.float32).to(device=paddle_device)
        a = wp.from_paddle(t, dtype=mat_dtype, return_ctype=True)
        ctype_size = ctypes.sizeof(mat_dtype._type_)
        assert a.data == t.data_ptr()
        assert a.grad == 0
        assert a.ndim == 1
        assert a.shape[0] == t.shape[0]
        assert a.strides[0] == t.strides[0] * ctype_size

    wrap_mat_tensor(wp.mat22)
    wrap_mat_tensor(wp.mat33)
    wrap_mat_tensor(wp.mat44)
    wrap_mat_tensor(wp.spatial_matrix)

    def wrap_vec_tensor_with_existing_grad(vec_dtype):
        t = paddle.zeros((10, vec_dtype._length_), dtype=paddle.float32).to(device=paddle_device)
        t.stop_gradient = False
        t.grad_ = paddle.zeros((10, vec_dtype._length_), dtype=paddle.float32).to(device=paddle_device)
        a = wp.from_paddle(t, dtype=vec_dtype, return_ctype=True)
        ctype_size = ctypes.sizeof(vec_dtype._type_)
        assert a.data == t.data_ptr()
        assert a.grad == t.grad.data_ptr()
        assert a.ndim == 1
        assert a.shape[0] == t.shape[0]
        assert a.strides[0] == t.strides[0] * ctype_size

    wrap_vec_tensor_with_existing_grad(wp.vec2)
    wrap_vec_tensor_with_existing_grad(wp.vec3)
    wrap_vec_tensor_with_existing_grad(wp.vec4)
    wrap_vec_tensor_with_existing_grad(wp.spatial_vector)
    wrap_vec_tensor_with_existing_grad(wp.transform)

    def wrap_vec_tensor_with_new_grad(vec_dtype):
        t = paddle.zeros((10, vec_dtype._length_), dtype=paddle.float32).to(device=paddle_device)
        a = wp.from_paddle(t, dtype=vec_dtype, requires_grad=True, return_ctype=True)
        ctype_size = ctypes.sizeof(vec_dtype._type_)
        assert a.data == t.data_ptr()
        assert a.grad == t.grad.data_ptr()
        assert a.ndim == 1
        assert a.shape[0] == t.shape[0]
        assert a.strides[0] == t.strides[0] * ctype_size

    wrap_vec_tensor_with_new_grad(wp.vec2)
    wrap_vec_tensor_with_new_grad(wp.vec3)
    wrap_vec_tensor_with_new_grad(wp.vec4)
    wrap_vec_tensor_with_new_grad(wp.spatial_vector)
    wrap_vec_tensor_with_new_grad(wp.transform)

    def wrap_vec_tensor_with_paddle_grad(vec_dtype):
        t = paddle.zeros((10, vec_dtype._length_), dtype=paddle.float32).to(device=paddle_device)
        grad = paddle.zeros((10, vec_dtype._length_), dtype=paddle.float32).to(device=paddle_device)
        a = wp.from_paddle(t, dtype=vec_dtype, grad=grad, return_ctype=True)
        ctype_size = ctypes.sizeof(vec_dtype._type_)
        assert a.data == t.data_ptr()
        assert a.grad == grad.data_ptr()
        assert a.ndim == 1
        assert a.shape[0] == t.shape[0]
        assert a.strides[0] == t.strides[0] * ctype_size

    wrap_vec_tensor_with_paddle_grad(wp.vec2)
    wrap_vec_tensor_with_paddle_grad(wp.vec3)
    wrap_vec_tensor_with_paddle_grad(wp.vec4)
    wrap_vec_tensor_with_paddle_grad(wp.spatial_vector)
    wrap_vec_tensor_with_paddle_grad(wp.transform)

    def wrap_vec_tensor_with_warp_grad(vec_dtype):
        t = paddle.zeros((10, vec_dtype._length_), dtype=paddle.float32).to(device=paddle_device)
        grad = wp.zeros(10, dtype=vec_dtype, device=device)
        a = wp.from_paddle(t, dtype=vec_dtype, grad=grad, return_ctype=True)
        ctype_size = ctypes.sizeof(vec_dtype._type_)
        assert a.data == t.data_ptr()
        assert a.grad == grad.ptr
        assert a.ndim == 1
        assert a.shape[0] == t.shape[0]
        assert a.strides[0] == t.strides[0] * ctype_size

    wrap_vec_tensor_with_warp_grad(wp.vec2)
    wrap_vec_tensor_with_warp_grad(wp.vec3)
    wrap_vec_tensor_with_warp_grad(wp.vec4)
    wrap_vec_tensor_with_warp_grad(wp.spatial_vector)
    wrap_vec_tensor_with_warp_grad(wp.transform)


def test_to_paddle(test, device):
    import paddle

    def wrap_scalar_array(warp_dtype, expected_paddle_dtype):
        a = wp.zeros(10, dtype=warp_dtype, device=device)
        t = wp.to_paddle(a)
        assert t.dtype == expected_paddle_dtype
        assert tuple(t.shape) == a.shape

    wrap_scalar_array(wp.float64, paddle.float64)
    wrap_scalar_array(wp.float32, paddle.float32)
    wrap_scalar_array(wp.float16, paddle.float16)
    wrap_scalar_array(wp.int64, paddle.int64)
    wrap_scalar_array(wp.int32, paddle.int32)
    wrap_scalar_array(wp.int16, paddle.int16)
    wrap_scalar_array(wp.int8, paddle.int8)
    wrap_scalar_array(wp.uint8, paddle.uint8)
    wrap_scalar_array(wp.bool, paddle.bool)

    # not supported by paddle
    # wrap_scalar_array(wp.uint64, paddle.int64)
    # wrap_scalar_array(wp.uint32, paddle.int32)
    # wrap_scalar_array(wp.uint16, paddle.int16)

    def wrap_vec_array(n, warp_dtype):
        a = wp.zeros(10, dtype=warp_dtype, device=device)
        t = wp.to_paddle(a)
        assert t.dtype == paddle.float32
        assert tuple(t.shape) == (10, n)

    wrap_vec_array(2, wp.vec2)
    wrap_vec_array(3, wp.vec3)
    wrap_vec_array(4, wp.vec4)
    wrap_vec_array(6, wp.spatial_vector)
    wrap_vec_array(7, wp.transform)

    def wrap_mat_array(n, m, warp_dtype):
        a = wp.zeros(10, dtype=warp_dtype, device=device)
        t = wp.to_paddle(a)
        assert t.dtype == paddle.float32
        assert tuple(t.shape) == (10, n, m)

    wrap_mat_array(2, 2, wp.mat22)
    wrap_mat_array(3, 3, wp.mat33)
    wrap_mat_array(4, 4, wp.mat44)
    wrap_mat_array(6, 6, wp.spatial_matrix)


def test_from_paddle_slices(test, device):
    import paddle

    paddle_device = wp.device_to_paddle(device)

    # 1D slice, contiguous
    t_base = paddle.arange(10, dtype=paddle.float32).to(device=paddle_device)
    t = t_base[2:9]
    a = wp.from_paddle(t)
    assert a.ptr == t.data_ptr()
    assert a.is_contiguous
    assert a.shape == tuple(t.shape)
    assert_np_equal(a.numpy(), t.numpy())

    # 1D slice with non-contiguous stride
    t_base = paddle.arange(10, dtype=paddle.float32).to(device=paddle_device)
    t = t_base[2:9:2]
    a = wp.from_paddle(t)
    assert a.ptr == t.data_ptr()
    assert not a.is_contiguous
    assert a.shape == tuple(t.shape)
    # copy contents to contiguous array
    a_contiguous = wp.empty_like(a)
    wp.launch(copy1d_float_kernel, dim=a.shape, inputs=[a_contiguous, a], device=device)
    assert_np_equal(a_contiguous.numpy(), t.numpy())

    # 2D slices (non-contiguous)
    t_base = paddle.arange(24, dtype=paddle.float32).to(device=paddle_device).reshape((4, 6))
    t = t_base[1:3, 2:5]
    a = wp.from_paddle(t)
    assert a.ptr == t.data_ptr()
    assert not a.is_contiguous
    assert a.shape == tuple(t.shape)
    # copy contents to contiguous array
    a_contiguous = wp.empty_like(a)
    wp.launch(copy2d_float_kernel, dim=a.shape, inputs=[a_contiguous, a], device=device)
    assert_np_equal(a_contiguous.numpy(), t.numpy())

    # 3D slices (non-contiguous)
    t_base = paddle.arange(36, dtype=paddle.float32).to(device=paddle_device).reshape((4, 3, 3))
    t = t_base[::2, 0:1, 1:2]
    a = wp.from_paddle(t)
    assert a.ptr == t.data_ptr()
    assert not a.is_contiguous
    assert a.shape == tuple(t.shape)
    # copy contents to contiguous array
    a_contiguous = wp.empty_like(a)
    wp.launch(copy3d_float_kernel, dim=a.shape, inputs=[a_contiguous, a], device=device)
    assert_np_equal(a_contiguous.numpy(), t.numpy())

    # 2D slices of vec3 (inner contiguous, outer non-contiguous)
    t_base = paddle.arange(150, dtype=paddle.float32).to(device=paddle_device).reshape((10, 5, 3))
    t = t_base[1:7:2, 2:5]
    a = wp.from_paddle(t, dtype=wp.vec3)
    assert a.ptr == t.data_ptr()
    assert not a.is_contiguous
    assert a.shape == tuple(t.shape[:-1])
    # copy contents to contiguous array
    a_contiguous = wp.empty_like(a)
    wp.launch(copy2d_vec3_kernel, dim=a.shape, inputs=[a_contiguous, a], device=device)
    assert_np_equal(a_contiguous.numpy(), t.numpy())

    # 2D slices of mat22 (inner contiguous, outer non-contiguous)
    t_base = paddle.arange(200, dtype=paddle.float32).to(device=paddle_device).reshape((10, 5, 2, 2))
    t = t_base[1:7:2, 2:5]
    a = wp.from_paddle(t, dtype=wp.mat22)
    assert a.ptr == t.data_ptr()
    assert not a.is_contiguous
    assert a.shape == tuple(t.shape[:-2])
    # copy contents to contiguous array
    a_contiguous = wp.empty_like(a)
    wp.launch(copy2d_mat22_kernel, dim=a.shape, inputs=[a_contiguous, a], device=device)
    assert_np_equal(a_contiguous.numpy(), t.numpy())


def test_from_paddle_zero_strides(test, device):
    import paddle

    paddle_device = wp.device_to_paddle(device)

    t_base = paddle.arange(9, dtype=paddle.float32).to(device=paddle_device).reshape((3, 3))

    # expand outermost dimension
    t = t_base.unsqueeze(0).expand([3, -1, -1])
    a = wp.from_paddle(t)
    assert a.ptr == t.data_ptr()
    assert a.is_contiguous
    assert a.shape == tuple(t.shape)
    a_contiguous = wp.empty_like(a)
    wp.launch(copy3d_float_kernel, dim=a.shape, inputs=[a_contiguous, a], device=device)
    assert_np_equal(a_contiguous.numpy(), t.numpy())

    # expand middle dimension
    t = t_base.unsqueeze(1).expand([-1, 3, -1])
    a = wp.from_paddle(t)
    assert a.ptr == t.data_ptr()
    assert a.is_contiguous
    assert a.shape == tuple(t.shape)
    a_contiguous = wp.empty_like(a)
    wp.launch(copy3d_float_kernel, dim=a.shape, inputs=[a_contiguous, a], device=device)
    assert_np_equal(a_contiguous.numpy(), t.numpy())

    # expand innermost dimension
    t = t_base.unsqueeze(2).expand([-1, -1, 3])
    a = wp.from_paddle(t)
    assert a.ptr == t.data_ptr()
    assert a.is_contiguous
    assert a.shape == tuple(t.shape)
    a_contiguous = wp.empty_like(a)
    wp.launch(copy3d_float_kernel, dim=a.shape, inputs=[a_contiguous, a], device=device)
    assert_np_equal(a_contiguous.numpy(), t.numpy())


def test_paddle_autograd(test, device):
    """Test paddle autograd with a custom Warp op"""

    import paddle

    # custom autograd op
    class TestFunc(paddle.autograd.PyLayer):
        @staticmethod
        def forward(ctx, x):
            # ensure Paddle operations complete before running Warp
            wp.synchronize_device()

            # allocate output array
            y = paddle.empty_like(x)

            ctx.x = x
            ctx.y = y

            wp.launch(kernel=op_kernel, dim=len(x), inputs=[wp.from_paddle(x)], outputs=[wp.from_paddle(y)])

            # ensure Warp operations complete before returning data to Paddle
            wp.synchronize_device()

            return y

        @staticmethod
        def backward(ctx, adj_y):
            # ensure Paddle operations complete before running Warp
            wp.synchronize_device()

            # adjoints should be allocated as zero initialized
            adj_x = paddle.zeros_like(ctx.x).contiguous()
            adj_y = adj_y.contiguous()

            wp_x = wp.from_paddle(ctx.x, grad=adj_x)
            wp_y = wp.from_paddle(ctx.y, grad=adj_y)

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

            # ensure Warp operations complete before returning data to Paddle
            wp.synchronize_device()

            return adj_x

    # run autograd on given device
    with wp.ScopedDevice(device):
        paddle_device = wp.device_to_paddle(device)

        # input data
        x = paddle.ones(16, dtype=paddle.float32).to(device=paddle_device)
        x.stop_gradient = False

        # execute op
        y = TestFunc.apply(x)

        # compute grads
        l = y.sum()
        l.backward()

        passed = (x.grad == -2.0).all()
        assert passed.item()


def test_warp_graph_warp_stream(test, device):
    """Capture Warp graph on Warp stream"""

    import paddle

    paddle_device = wp.device_to_paddle(device)

    n = 1024 * 1024
    t = paddle.zeros(n, dtype=paddle.float32).to(device=paddle_device)
    a = wp.from_paddle(t)

    # make paddle use the warp stream from the given device
    paddle_stream = wp.stream_to_paddle(device)

    # capture graph
    with wp.ScopedDevice(device), paddle.device.stream_guard(paddle.device.Stream(paddle_stream)):
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


def test_warp_graph_paddle_stream(test, device):
    """Capture Warp graph on Paddle stream"""

    wp.load_module(device=device)

    import paddle

    paddle_device = wp.device_to_paddle(device)

    n = 1024 * 1024
    t = paddle.zeros(n, dtype=paddle.float32).to(device=paddle_device)
    a = wp.from_paddle(t)

    # create a device-specific paddle stream to use for capture
    # (the default paddle stream is not suitable for graph capture)
    paddle_stream = paddle.device.Stream(device=paddle_device)

    # make warp use the same stream
    warp_stream = wp.stream_from_paddle(paddle_stream)

    # capture graph
    with wp.ScopedStream(warp_stream):
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
    """Pass Paddle tensors to Warp kernels directly"""

    import paddle

    paddle_device = wp.device_to_paddle(device)
    n = 12

    s = paddle.arange(n, dtype=paddle.float32).to(device=paddle_device)
    v = paddle.arange(n, dtype=paddle.float32).to(device=paddle_device).reshape((n // 3, 3))
    m = paddle.arange(n, dtype=paddle.float32).to(device=paddle_device).reshape((n // 4, 2, 2))

    wp.launch(inc, dim=n, inputs=[s], device=device)
    wp.launch(inc_vector, dim=n // 3, inputs=[v], device=device)
    wp.launch(inc_matrix, dim=n // 4, inputs=[m], device=device)

    expected = paddle.arange(1, n + 1, dtype=paddle.float32).to(device=paddle_device)

    assert paddle.equal_all(s, expected).item()
    assert paddle.equal_all(v.reshape([n]), expected).item()
    assert paddle.equal_all(m.reshape([n]), expected).item()


class TestPaddle(unittest.TestCase):
    pass


test_devices = get_test_devices()

try:
    import paddle

    # check which Warp devices work with Paddle
    # CUDA devices may fail if Paddle was not compiled with CUDA support
    paddle_compatible_devices = []
    paddle_compatible_cuda_devices = []

    for d in test_devices:
        try:
            t = paddle.arange(10).to(device=wp.device_to_paddle(d))
            t += 1
            paddle_compatible_devices.append(d)
            if d.is_cuda:
                paddle_compatible_cuda_devices.append(d)
        except Exception as e:
            print(f"Skipping Paddle tests on device '{d}' due to exception: {e}")

    add_function_test(TestPaddle, "test_dtype_from_paddle", test_dtype_from_paddle, devices=None)
    add_function_test(TestPaddle, "test_dtype_to_paddle", test_dtype_to_paddle, devices=None)

    if paddle_compatible_devices:
        add_function_test(
            TestPaddle, "test_device_conversion", test_device_conversion, devices=paddle_compatible_devices
        )
        add_function_test(TestPaddle, "test_from_paddle", test_from_paddle, devices=paddle_compatible_devices)
        add_function_test(
            TestPaddle, "test_from_paddle_slices", test_from_paddle_slices, devices=paddle_compatible_devices
        )
        add_function_test(
            TestPaddle, "test_array_ctype_from_paddle", test_array_ctype_from_paddle, devices=paddle_compatible_devices
        )
        add_function_test(
            TestPaddle,
            "test_from_paddle_zero_strides",
            test_from_paddle_zero_strides,
            devices=paddle_compatible_devices,
        )
        add_function_test(TestPaddle, "test_to_paddle", test_to_paddle, devices=paddle_compatible_devices)
        add_function_test(TestPaddle, "test_paddle_zerocopy", test_paddle_zerocopy, devices=paddle_compatible_devices)
        add_function_test(TestPaddle, "test_paddle_autograd", test_paddle_autograd, devices=paddle_compatible_devices)
        add_function_test(TestPaddle, "test_direct", test_direct, devices=paddle_compatible_devices)

    # NOTE: Graph not supported now
    # if paddle_compatible_cuda_devices:
    #     add_function_test(
    #         TestPaddle,
    #         "test_warp_graph_warp_stream",
    #         test_warp_graph_warp_stream,
    #         devices=paddle_compatible_cuda_devices,
    #     )
    #     add_function_test(
    #         TestPaddle,
    #         "test_warp_graph_paddle_stream",
    #         test_warp_graph_paddle_stream,
    #         devices=paddle_compatible_cuda_devices,
    #     )

    # multi-GPU not supported yet.
    # if len(paddle_compatible_cuda_devices) > 1:
    #     add_function_test(TestPaddle, "test_paddle_mgpu_from_paddle", test_paddle_mgpu_from_paddle)
    #     add_function_test(TestPaddle, "test_paddle_mgpu_to_paddle", test_paddle_mgpu_to_paddle)
    #     add_function_test(TestPaddle, "test_paddle_mgpu_interop", test_paddle_mgpu_interop)

except Exception as e:
    print(f"Skipping Paddle tests due to exception: {e}")


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
