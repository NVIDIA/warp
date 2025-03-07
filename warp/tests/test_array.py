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
from typing import Any

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


@wp.kernel
def kernel_1d(a: wp.array(dtype=int, ndim=1)):
    i = wp.tid()

    wp.expect_eq(a[i], wp.tid())

    a[i] = a[i] * 2
    wp.atomic_add(a, i, 1)

    wp.expect_eq(a[i], wp.tid() * 2 + 1)


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
    i = wp.tid() // n
    j = wp.tid() % n

    wp.expect_eq(a[i, j], wp.tid())
    wp.expect_eq(a[i][j], wp.tid())

    a[i, j] = a[i, j] * 2
    wp.atomic_add(a, i, j, 1)

    wp.expect_eq(a[i, j], wp.tid() * 2 + 1)


def test_2d(test, device):
    dim_x = 4
    dim_y = 2

    a = np.arange(0, dim_x * dim_y, dtype=np.int32)
    a = a.reshape(dim_x, dim_y)

    arr = wp.array(a, device=device)

    test.assertEqual(arr.shape, a.shape)
    test.assertEqual(arr.size, a.size)
    test.assertEqual(arr.ndim, a.ndim)

    with CheckOutput(test):
        wp.launch(kernel_2d, dim=arr.size, inputs=[arr, dim_x, dim_y], device=device)


@wp.kernel
def kernel_3d(a: wp.array(dtype=int, ndim=3), m: int, n: int, o: int):
    i = wp.tid() // (n * o)
    j = wp.tid() % (n * o) // o
    k = wp.tid() % o

    wp.expect_eq(a[i, j, k], wp.tid())
    wp.expect_eq(a[i][j][k], wp.tid())

    a[i, j, k] = a[i, j, k] * 2
    a[i][j][k] = a[i][j][k] * 2
    wp.atomic_add(a, i, j, k, 1)

    wp.expect_eq(a[i, j, k], wp.tid() * 4 + 1)


def test_3d(test, device):
    dim_x = 8
    dim_y = 4
    dim_z = 2

    a = np.arange(0, dim_x * dim_y * dim_z, dtype=np.int32)
    a = a.reshape(dim_x, dim_y, dim_z)

    arr = wp.array(a, device=device)

    test.assertEqual(arr.shape, a.shape)
    test.assertEqual(arr.size, a.size)
    test.assertEqual(arr.ndim, a.ndim)

    with CheckOutput(test):
        wp.launch(kernel_3d, dim=arr.size, inputs=[arr, dim_x, dim_y, dim_z], device=device)


@wp.kernel
def kernel_4d(a: wp.array(dtype=int, ndim=4), m: int, n: int, o: int, p: int):
    i = wp.tid() // (n * o * p)
    j = wp.tid() % (n * o * p) // (o * p)
    k = wp.tid() % (o * p) / p
    l = wp.tid() % p

    wp.expect_eq(a[i, j, k, l], wp.tid())
    wp.expect_eq(a[i][j][k][l], wp.tid())


def test_4d(test, device):
    dim_x = 16
    dim_y = 8
    dim_z = 4
    dim_w = 2

    a = np.arange(0, dim_x * dim_y * dim_z * dim_w, dtype=np.int32)
    a = a.reshape(dim_x, dim_y, dim_z, dim_w)

    arr = wp.array(a, device=device)

    test.assertEqual(arr.shape, a.shape)
    test.assertEqual(arr.size, a.size)
    test.assertEqual(arr.ndim, a.ndim)

    with CheckOutput(test):
        wp.launch(kernel_4d, dim=arr.size, inputs=[arr, dim_x, dim_y, dim_z, dim_w], device=device)


@wp.kernel
def kernel_4d_transposed(a: wp.array(dtype=int, ndim=4), m: int, n: int, o: int, p: int):
    i = wp.tid() // (n * o * p)
    j = wp.tid() % (n * o * p) // (o * p)
    k = wp.tid() % (o * p) / p
    l = wp.tid() % p

    wp.expect_eq(a[l, k, j, i], wp.tid())
    wp.expect_eq(a[l][k][j][i], wp.tid())


def test_4d_transposed(test, device):
    dim_x = 16
    dim_y = 8
    dim_z = 4
    dim_w = 2

    a = np.arange(0, dim_x * dim_y * dim_z * dim_w, dtype=np.int32)
    a = a.reshape(dim_x, dim_y, dim_z, dim_w)

    arr = wp.array(a, device=device)

    # Transpose the array manually, as using the wp.array() constructor with arr.T would make it contiguous first
    a_T = a.T
    arr_T = wp.array(
        dtype=arr.dtype,
        shape=a_T.shape,
        strides=a_T.__array_interface__["strides"],
        capacity=arr.capacity,
        ptr=arr.ptr,
        requires_grad=arr.requires_grad,
        device=device,
    )

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

    slice = arr[0, 0]
    wp.expect_eq(slice.shape[0], 30)


@wp.kernel
def f4(arr: wp.array4d(dtype=float)):
    wp.expect_eq(arr.shape[0], 10)
    wp.expect_eq(arr.shape[1], 20)
    wp.expect_eq(arr.shape[2], 30)
    wp.expect_eq(arr.shape[3], 40)

    slice = arr[0, 0, 0]
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


def test_negative_shape(test, device):
    with test.assertRaisesRegex(ValueError, "Array shapes must be non-negative"):
        _ = wp.zeros(shape=-1, dtype=int, device=device)

    with test.assertRaisesRegex(ValueError, "Array shapes must be non-negative"):
        _ = wp.zeros(shape=-(2**32), dtype=int, device=device)

    with test.assertRaisesRegex(ValueError, "Array shapes must be non-negative"):
        _ = wp.zeros(shape=(10, -1), dtype=int, device=device)


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
        wp.launch(kernel=sum_array, dim=len(arr_flat), inputs=[arr_flat, loss], device=device)

    tape.backward(loss=loss)
    grad = tape.gradients[arr_flat]

    ones = wp.array(
        np.ones(
            (8,),
            dtype=float,
        ),
        dtype=float,
        device=device,
    )
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
        wp.launch(kernel=sum_array, dim=len(arr_reshaped), inputs=[arr_reshaped, loss], device=device)

    tape.backward(loss=loss)
    grad = tape.gradients[arr_reshaped]

    ones = wp.array(
        np.ones(
            (6,),
            dtype=float,
        ),
        dtype=float,
        device=device,
    )
    assert_array_equal(grad, ones)
    test.assertEqual(loss.numpy()[0], 15)

    np_arr = np.arange(6, dtype=float)
    arr = wp.array(np_arr, dtype=float, device=device)
    arr_infer = arr.reshape((-1, 3))
    arr_comp = wp.array(np_arr.reshape((-1, 3)), dtype=float, device=device)
    assert_array_equal(arr_infer, arr_comp)


@wp.kernel
def compare_stepped_window_a(x: wp.array2d(dtype=float)):
    wp.expect_eq(x[0, 0], 1.0)
    wp.expect_eq(x[0, 1], 2.0)
    wp.expect_eq(x[1, 0], 9.0)
    wp.expect_eq(x[1, 1], 10.0)


@wp.kernel
def compare_stepped_window_b(x: wp.array2d(dtype=float)):
    wp.expect_eq(x[0, 0], 3.0)
    wp.expect_eq(x[0, 1], 4.0)
    wp.expect_eq(x[1, 0], 7.0)
    wp.expect_eq(x[1, 1], 8.0)
    wp.expect_eq(x[2, 0], 11.0)
    wp.expect_eq(x[2, 1], 12.0)


def test_slicing(test, device):
    np_arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=float)
    arr = wp.array(np_arr, dtype=float, shape=np_arr.shape, device=device, requires_grad=True)

    slice_a = arr[1, :, :]  # test indexing
    slice_b = arr[1:2, :, :]  # test slicing
    slice_c = arr[-1, :, :]  # test negative indexing
    slice_d = arr[-2:-1, :, :]  # test negative slicing
    slice_e = arr[-1:3, :, :]  # test mixed slicing
    slice_e2 = slice_e[0, 0, :]  # test 2x slicing
    slice_f = arr[0:3:2, 0, :]  # test step

    assert_array_equal(slice_a, wp.array(np_arr[1, :, :], dtype=float, device=device))
    assert_array_equal(slice_b, wp.array(np_arr[1:2, :, :], dtype=float, device=device))
    assert_array_equal(slice_c, wp.array(np_arr[-1, :, :], dtype=float, device=device))
    assert_array_equal(slice_d, wp.array(np_arr[-2:-1, :, :], dtype=float, device=device))
    assert_array_equal(slice_e, wp.array(np_arr[-1:3, :, :], dtype=float, device=device))
    assert_array_equal(slice_e2, wp.array(np_arr[2, 0, :], dtype=float, device=device))

    # wp does not support copying from/to non-contiguous arrays
    # stepped windows must read on the device the original array was created on
    wp.launch(kernel=compare_stepped_window_a, dim=1, inputs=[slice_f], device=device)

    slice_flat = slice_b.flatten()
    loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)
    tape = wp.Tape()
    with tape:
        wp.launch(kernel=sum_array, dim=len(slice_flat), inputs=[slice_flat, loss], device=device)

    tape.backward(loss=loss)
    grad = tape.gradients[slice_flat]

    ones = wp.array(
        np.ones(
            (4,),
            dtype=float,
        ),
        dtype=float,
        device=device,
    )
    assert_array_equal(grad, ones)
    test.assertEqual(loss.numpy()[0], 26)

    index_a = arr[1]
    index_b = arr[2, 1]
    index_c = arr[1, :]
    index_d = arr[:, 1]

    assert_array_equal(index_a, wp.array(np_arr[1], dtype=float, device=device))
    assert_array_equal(index_b, wp.array(np_arr[2, 1], dtype=float, device=device))
    assert_array_equal(index_c, wp.array(np_arr[1, :], dtype=float, device=device))
    wp.launch(kernel=compare_stepped_window_b, dim=1, inputs=[index_d], device=device)

    np_arr = np.zeros(10, dtype=int)
    wp_arr = wp.array(np_arr, dtype=int, device=device)

    assert_array_equal(wp_arr[:5], wp.array(np_arr[:5], dtype=int, device=device))
    assert_array_equal(wp_arr[1:5], wp.array(np_arr[1:5], dtype=int, device=device))
    assert_array_equal(wp_arr[-9:-5:1], wp.array(np_arr[-9:-5:1], dtype=int, device=device))
    assert_array_equal(wp_arr[:5,], wp.array(np_arr[:5], dtype=int, device=device))  # noqa: E231


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

    assert_np_equal(wp_arr_a.view(dtype=wp.float32).numpy(), np_arr_a.view(dtype=np.float32))
    assert_np_equal(wp_arr_b.view(dtype=wp.uint32).numpy(), np_arr_b.view(dtype=np.uint32))
    assert_np_equal(wp_arr_c.view(dtype=wp.float16).numpy(), np_arr_c.view(dtype=np.float16))
    assert_np_equal(wp_arr_d.view(dtype=wp.uint16).numpy(), np_arr_d.view(dtype=np.uint16))
    assert_array_equal(wp_arr_e.view(dtype=wp.quat), wp_arr_f)


def test_clone_adjoint(test, device):
    state_in = wp.from_numpy(
        np.array([1.0, 2.0, 3.0]).astype(np.float32), dtype=wp.float32, requires_grad=True, device=device
    )

    tape = wp.Tape()
    with tape:
        state_out = wp.clone(state_in)

    grads = {state_out: wp.from_numpy(np.array([1.0, 1.0, 1.0]).astype(np.float32), dtype=wp.float32, device=device)}
    tape.backward(grads=grads)

    assert_np_equal(state_in.grad.numpy(), np.array([1.0, 1.0, 1.0]).astype(np.float32))


def test_assign_adjoint(test, device):
    state_in = wp.from_numpy(
        np.array([1.0, 2.0, 3.0]).astype(np.float32), dtype=wp.float32, requires_grad=True, device=device
    )
    state_out = wp.zeros(state_in.shape, dtype=wp.float32, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        state_out.assign(state_in)

    grads = {state_out: wp.from_numpy(np.array([1.0, 1.0, 1.0]).astype(np.float32), dtype=wp.float32, device=device)}
    tape.backward(grads=grads)

    assert_np_equal(state_in.grad.numpy(), np.array([1.0, 1.0, 1.0]).astype(np.float32))


@wp.kernel
def compare_2darrays(x: wp.array2d(dtype=float), y: wp.array2d(dtype=float), z: wp.array2d(dtype=int)):
    i, j = wp.tid()

    if x[i, j] == y[i, j]:
        z[i, j] = 1


@wp.kernel
def compare_3darrays(x: wp.array3d(dtype=float), y: wp.array3d(dtype=float), z: wp.array3d(dtype=int)):
    i, j, k = wp.tid()

    if x[i, j, k] == y[i, j, k]:
        z[i, j, k] = 1


def test_transpose(test, device):
    # test default transpose in non-square 2d case
    # wp does not support copying from/to non-contiguous arrays so check in kernel
    np_arr = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    arr = wp.array(np_arr, dtype=float, device=device)
    arr_transpose = arr.transpose()
    arr_compare = wp.array(np_arr.transpose(), dtype=float, device=device)
    check = wp.zeros(shape=(2, 3), dtype=int, device=device)

    wp.launch(compare_2darrays, dim=(2, 3), inputs=[arr_transpose, arr_compare, check], device=device)
    assert_np_equal(check.numpy(), np.ones((2, 3), dtype=int))

    # test transpose in square 3d case
    # wp does not support copying from/to non-contiguous arrays so check in kernel
    np_arr = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=float)
    arr = wp.array3d(np_arr, dtype=float, shape=np_arr.shape, device=device, requires_grad=True)
    arr_transpose = arr.transpose((0, 2, 1))
    arr_compare = wp.array3d(np_arr.transpose((0, 2, 1)), dtype=float, device=device)
    check = wp.zeros(shape=(3, 2, 2), dtype=int, device=device)

    wp.launch(compare_3darrays, dim=(3, 2, 2), inputs=[arr_transpose, arr_compare, check], device=device)
    assert_np_equal(check.numpy(), np.ones((3, 2, 2), dtype=int))

    # test transpose in square 3d case without axes supplied
    arr_transpose = arr.transpose()
    arr_compare = wp.array3d(np_arr.transpose(), dtype=float, device=device)
    check = wp.zeros(shape=(2, 2, 3), dtype=int, device=device)

    wp.launch(compare_3darrays, dim=(2, 2, 3), inputs=[arr_transpose, arr_compare, check], device=device)
    assert_np_equal(check.numpy(), np.ones((2, 2, 3), dtype=int))

    # test transpose in 1d case (should be noop)
    np_arr = np.array([1, 2, 3], dtype=float)
    arr = wp.array(np_arr, dtype=float, device=device)

    assert_np_equal(arr.transpose().numpy(), np_arr.transpose())


def test_fill_scalar(test, device):
    dim_x = 4

    for nptype, wptype in wp.types.np_dtype_to_warp_type.items():
        a1 = wp.zeros(dim_x, dtype=wptype, device=device)
        a2 = wp.zeros((dim_x, dim_x), dtype=wptype, device=device)
        a3 = wp.zeros((dim_x, dim_x, dim_x), dtype=wptype, device=device)
        a4 = wp.zeros((dim_x, dim_x, dim_x, dim_x), dtype=wptype, device=device)

        assert_np_equal(a1.numpy(), np.zeros(a1.shape, dtype=nptype))
        assert_np_equal(a2.numpy(), np.zeros(a2.shape, dtype=nptype))
        assert_np_equal(a3.numpy(), np.zeros(a3.shape, dtype=nptype))
        assert_np_equal(a4.numpy(), np.zeros(a4.shape, dtype=nptype))

        # fill with int value
        fill_value = 42

        a1.fill_(fill_value)
        a2.fill_(fill_value)
        a3.fill_(fill_value)
        a4.fill_(fill_value)

        assert_np_equal(a1.numpy(), np.full(a1.shape, fill_value, dtype=nptype))
        assert_np_equal(a2.numpy(), np.full(a2.shape, fill_value, dtype=nptype))
        assert_np_equal(a3.numpy(), np.full(a3.shape, fill_value, dtype=nptype))
        assert_np_equal(a4.numpy(), np.full(a4.shape, fill_value, dtype=nptype))

        a1.zero_()
        a2.zero_()
        a3.zero_()
        a4.zero_()

        assert_np_equal(a1.numpy(), np.zeros(a1.shape, dtype=nptype))
        assert_np_equal(a2.numpy(), np.zeros(a2.shape, dtype=nptype))
        assert_np_equal(a3.numpy(), np.zeros(a3.shape, dtype=nptype))
        assert_np_equal(a4.numpy(), np.zeros(a4.shape, dtype=nptype))

        if wptype in wp.types.float_types:
            # fill with float value
            fill_value = 13.37

            a1.fill_(fill_value)
            a2.fill_(fill_value)
            a3.fill_(fill_value)
            a4.fill_(fill_value)

            assert_np_equal(a1.numpy(), np.full(a1.shape, fill_value, dtype=nptype))
            assert_np_equal(a2.numpy(), np.full(a2.shape, fill_value, dtype=nptype))
            assert_np_equal(a3.numpy(), np.full(a3.shape, fill_value, dtype=nptype))
            assert_np_equal(a4.numpy(), np.full(a4.shape, fill_value, dtype=nptype))

        # fill with Warp scalar value
        fill_value = wptype(17)

        a1.fill_(fill_value)
        a2.fill_(fill_value)
        a3.fill_(fill_value)
        a4.fill_(fill_value)

        assert_np_equal(a1.numpy(), np.full(a1.shape, fill_value.value, dtype=nptype))
        assert_np_equal(a2.numpy(), np.full(a2.shape, fill_value.value, dtype=nptype))
        assert_np_equal(a3.numpy(), np.full(a3.shape, fill_value.value, dtype=nptype))
        assert_np_equal(a4.numpy(), np.full(a4.shape, fill_value.value, dtype=nptype))


def test_fill_vector(test, device):
    # test filling a vector array with scalar or vector values (vec_type, list, or numpy array)

    dim_x = 4

    for nptype, wptype in wp.types.np_dtype_to_warp_type.items():
        # vector types
        vector_types = [
            wp.types.vector(2, wptype),
            wp.types.vector(3, wptype),
            wp.types.vector(4, wptype),
            wp.types.vector(5, wptype),
        ]

        for vec_type in vector_types:
            vec_len = vec_type._length_

            a1 = wp.zeros(dim_x, dtype=vec_type, device=device)
            a2 = wp.zeros((dim_x, dim_x), dtype=vec_type, device=device)
            a3 = wp.zeros((dim_x, dim_x, dim_x), dtype=vec_type, device=device)
            a4 = wp.zeros((dim_x, dim_x, dim_x, dim_x), dtype=vec_type, device=device)

            assert_np_equal(a1.numpy(), np.zeros((*a1.shape, vec_len), dtype=nptype))
            assert_np_equal(a2.numpy(), np.zeros((*a2.shape, vec_len), dtype=nptype))
            assert_np_equal(a3.numpy(), np.zeros((*a3.shape, vec_len), dtype=nptype))
            assert_np_equal(a4.numpy(), np.zeros((*a4.shape, vec_len), dtype=nptype))

            # fill with int scalar
            fill_value = 42

            a1.fill_(fill_value)
            a2.fill_(fill_value)
            a3.fill_(fill_value)
            a4.fill_(fill_value)

            assert_np_equal(a1.numpy(), np.full((*a1.shape, vec_len), fill_value, dtype=nptype))
            assert_np_equal(a2.numpy(), np.full((*a2.shape, vec_len), fill_value, dtype=nptype))
            assert_np_equal(a3.numpy(), np.full((*a3.shape, vec_len), fill_value, dtype=nptype))
            assert_np_equal(a4.numpy(), np.full((*a4.shape, vec_len), fill_value, dtype=nptype))

            # test zeroing
            a1.zero_()
            a2.zero_()
            a3.zero_()
            a4.zero_()

            assert_np_equal(a1.numpy(), np.zeros((*a1.shape, vec_len), dtype=nptype))
            assert_np_equal(a2.numpy(), np.zeros((*a2.shape, vec_len), dtype=nptype))
            assert_np_equal(a3.numpy(), np.zeros((*a3.shape, vec_len), dtype=nptype))
            assert_np_equal(a4.numpy(), np.zeros((*a4.shape, vec_len), dtype=nptype))

            # vector values can be passed as a list, numpy array, or Warp vector instance
            fill_list = [17, 42, 99, 101, 127][:vec_len]
            fill_arr = np.array(fill_list, dtype=nptype)
            fill_vec = vec_type(fill_list)

            expected1 = np.tile(fill_arr, a1.size).reshape((*a1.shape, vec_len))
            expected2 = np.tile(fill_arr, a2.size).reshape((*a2.shape, vec_len))
            expected3 = np.tile(fill_arr, a3.size).reshape((*a3.shape, vec_len))
            expected4 = np.tile(fill_arr, a4.size).reshape((*a4.shape, vec_len))

            # fill with list of vector length
            a1.fill_(fill_list)
            a2.fill_(fill_list)
            a3.fill_(fill_list)
            a4.fill_(fill_list)

            assert_np_equal(a1.numpy(), expected1)
            assert_np_equal(a2.numpy(), expected2)
            assert_np_equal(a3.numpy(), expected3)
            assert_np_equal(a4.numpy(), expected4)

            # clear
            a1.zero_()
            a2.zero_()
            a3.zero_()
            a4.zero_()

            # fill with numpy array of vector length
            a1.fill_(fill_arr)
            a2.fill_(fill_arr)
            a3.fill_(fill_arr)
            a4.fill_(fill_arr)

            assert_np_equal(a1.numpy(), expected1)
            assert_np_equal(a2.numpy(), expected2)
            assert_np_equal(a3.numpy(), expected3)
            assert_np_equal(a4.numpy(), expected4)

            # clear
            a1.zero_()
            a2.zero_()
            a3.zero_()
            a4.zero_()

            # fill with vec instance
            a1.fill_(fill_vec)
            a2.fill_(fill_vec)
            a3.fill_(fill_vec)
            a4.fill_(fill_vec)

            assert_np_equal(a1.numpy(), expected1)
            assert_np_equal(a2.numpy(), expected2)
            assert_np_equal(a3.numpy(), expected3)
            assert_np_equal(a4.numpy(), expected4)

            if wptype in wp.types.float_types:
                # fill with float scalar
                fill_value = 13.37

                a1.fill_(fill_value)
                a2.fill_(fill_value)
                a3.fill_(fill_value)
                a4.fill_(fill_value)

                assert_np_equal(a1.numpy(), np.full((*a1.shape, vec_len), fill_value, dtype=nptype))
                assert_np_equal(a2.numpy(), np.full((*a2.shape, vec_len), fill_value, dtype=nptype))
                assert_np_equal(a3.numpy(), np.full((*a3.shape, vec_len), fill_value, dtype=nptype))
                assert_np_equal(a4.numpy(), np.full((*a4.shape, vec_len), fill_value, dtype=nptype))

                # fill with float list of vector length
                fill_list = [-2.5, -1.25, 1.25, 2.5, 5.0][:vec_len]

                a1.fill_(fill_list)
                a2.fill_(fill_list)
                a3.fill_(fill_list)
                a4.fill_(fill_list)

                expected1 = np.tile(np.array(fill_list, dtype=nptype), a1.size).reshape((*a1.shape, vec_len))
                expected2 = np.tile(np.array(fill_list, dtype=nptype), a2.size).reshape((*a2.shape, vec_len))
                expected3 = np.tile(np.array(fill_list, dtype=nptype), a3.size).reshape((*a3.shape, vec_len))
                expected4 = np.tile(np.array(fill_list, dtype=nptype), a4.size).reshape((*a4.shape, vec_len))

                assert_np_equal(a1.numpy(), expected1)
                assert_np_equal(a2.numpy(), expected2)
                assert_np_equal(a3.numpy(), expected3)
                assert_np_equal(a4.numpy(), expected4)


def test_fill_matrix(test, device):
    # test filling a matrix array with scalar or matrix values (mat_type, nested list, or 2d numpy array)

    dim_x = 4

    for nptype, wptype in wp.types.np_dtype_to_warp_type.items():
        # matrix types
        matrix_types = [
            # square matrices
            wp.types.matrix((2, 2), wptype),
            wp.types.matrix((3, 3), wptype),
            wp.types.matrix((4, 4), wptype),
            wp.types.matrix((5, 5), wptype),
            # non-square matrices
            wp.types.matrix((2, 3), wptype),
            wp.types.matrix((3, 2), wptype),
            wp.types.matrix((3, 4), wptype),
            wp.types.matrix((4, 3), wptype),
        ]

        for mat_type in matrix_types:
            mat_len = mat_type._length_
            mat_shape = mat_type._shape_

            a1 = wp.zeros(dim_x, dtype=mat_type, device=device)
            a2 = wp.zeros((dim_x, dim_x), dtype=mat_type, device=device)
            a3 = wp.zeros((dim_x, dim_x, dim_x), dtype=mat_type, device=device)
            a4 = wp.zeros((dim_x, dim_x, dim_x, dim_x), dtype=mat_type, device=device)

            assert_np_equal(a1.numpy(), np.zeros((*a1.shape, *mat_shape), dtype=nptype))
            assert_np_equal(a2.numpy(), np.zeros((*a2.shape, *mat_shape), dtype=nptype))
            assert_np_equal(a3.numpy(), np.zeros((*a3.shape, *mat_shape), dtype=nptype))
            assert_np_equal(a4.numpy(), np.zeros((*a4.shape, *mat_shape), dtype=nptype))

            # fill with scalar
            fill_value = 42

            a1.fill_(fill_value)
            a2.fill_(fill_value)
            a3.fill_(fill_value)
            a4.fill_(fill_value)

            assert_np_equal(a1.numpy(), np.full((*a1.shape, *mat_shape), fill_value, dtype=nptype))
            assert_np_equal(a2.numpy(), np.full((*a2.shape, *mat_shape), fill_value, dtype=nptype))
            assert_np_equal(a3.numpy(), np.full((*a3.shape, *mat_shape), fill_value, dtype=nptype))
            assert_np_equal(a4.numpy(), np.full((*a4.shape, *mat_shape), fill_value, dtype=nptype))

            # test zeroing
            a1.zero_()
            a2.zero_()
            a3.zero_()
            a4.zero_()

            assert_np_equal(a1.numpy(), np.zeros((*a1.shape, *mat_shape), dtype=nptype))
            assert_np_equal(a2.numpy(), np.zeros((*a2.shape, *mat_shape), dtype=nptype))
            assert_np_equal(a3.numpy(), np.zeros((*a3.shape, *mat_shape), dtype=nptype))
            assert_np_equal(a4.numpy(), np.zeros((*a4.shape, *mat_shape), dtype=nptype))

            # matrix values can be passed as a 1d numpy array, 2d numpy array, flat list, nested list, or Warp matrix instance
            if wptype != wp.bool:
                fill_arr1 = np.arange(mat_len, dtype=nptype)
            else:
                fill_arr1 = np.ones(mat_len, dtype=nptype)
            fill_arr2 = fill_arr1.reshape(mat_shape)
            fill_list1 = list(fill_arr1)
            fill_list2 = [list(row) for row in fill_arr2]
            fill_mat = mat_type(fill_arr1)

            expected1 = np.tile(fill_arr1, a1.size).reshape((*a1.shape, *mat_shape))
            expected2 = np.tile(fill_arr1, a2.size).reshape((*a2.shape, *mat_shape))
            expected3 = np.tile(fill_arr1, a3.size).reshape((*a3.shape, *mat_shape))
            expected4 = np.tile(fill_arr1, a4.size).reshape((*a4.shape, *mat_shape))

            # fill with 1d numpy array
            a1.fill_(fill_arr1)
            a2.fill_(fill_arr1)
            a3.fill_(fill_arr1)
            a4.fill_(fill_arr1)

            assert_np_equal(a1.numpy(), expected1)
            assert_np_equal(a2.numpy(), expected2)
            assert_np_equal(a3.numpy(), expected3)
            assert_np_equal(a4.numpy(), expected4)

            # clear
            a1.zero_()
            a2.zero_()
            a3.zero_()
            a4.zero_()

            # fill with 2d numpy array
            a1.fill_(fill_arr2)
            a2.fill_(fill_arr2)
            a3.fill_(fill_arr2)
            a4.fill_(fill_arr2)

            assert_np_equal(a1.numpy(), expected1)
            assert_np_equal(a2.numpy(), expected2)
            assert_np_equal(a3.numpy(), expected3)
            assert_np_equal(a4.numpy(), expected4)

            # clear
            a1.zero_()
            a2.zero_()
            a3.zero_()
            a4.zero_()

            # fill with flat list
            a1.fill_(fill_list1)
            a2.fill_(fill_list1)
            a3.fill_(fill_list1)
            a4.fill_(fill_list1)

            assert_np_equal(a1.numpy(), expected1)
            assert_np_equal(a2.numpy(), expected2)
            assert_np_equal(a3.numpy(), expected3)
            assert_np_equal(a4.numpy(), expected4)

            # clear
            a1.zero_()
            a2.zero_()
            a3.zero_()
            a4.zero_()

            # fill with nested list
            a1.fill_(fill_list2)
            a2.fill_(fill_list2)
            a3.fill_(fill_list2)
            a4.fill_(fill_list2)

            assert_np_equal(a1.numpy(), expected1)
            assert_np_equal(a2.numpy(), expected2)
            assert_np_equal(a3.numpy(), expected3)
            assert_np_equal(a4.numpy(), expected4)

            # clear
            a1.zero_()
            a2.zero_()
            a3.zero_()
            a4.zero_()

            # fill with mat instance
            a1.fill_(fill_mat)
            a2.fill_(fill_mat)
            a3.fill_(fill_mat)
            a4.fill_(fill_mat)

            assert_np_equal(a1.numpy(), expected1)
            assert_np_equal(a2.numpy(), expected2)
            assert_np_equal(a3.numpy(), expected3)
            assert_np_equal(a4.numpy(), expected4)


@wp.struct
class FillStruct:
    # scalar members (make sure to test float16)
    i1: wp.int8
    i2: wp.int16
    i4: wp.int32
    i8: wp.int64
    f2: wp.float16
    f4: wp.float32
    f8: wp.float16
    # vector members (make sure to test vectors of float16)
    v2: wp.types.vector(2, wp.int64)
    v3: wp.types.vector(3, wp.float32)
    v4: wp.types.vector(4, wp.float16)
    v5: wp.types.vector(5, wp.uint8)
    # matrix members (make sure to test matrices of float16)
    m2: wp.types.matrix((2, 2), wp.float64)
    m3: wp.types.matrix((3, 3), wp.int32)
    m4: wp.types.matrix((4, 4), wp.float16)
    m5: wp.types.matrix((5, 5), wp.int8)
    # arrays
    a1: wp.array(dtype=float)
    a2: wp.array2d(dtype=float)
    a3: wp.array3d(dtype=float)
    a4: wp.array4d(dtype=float)


def test_fill_struct(test, device):
    dim_x = 4

    nptype = FillStruct.numpy_dtype()

    a1 = wp.zeros(dim_x, dtype=FillStruct, device=device)
    a2 = wp.zeros((dim_x, dim_x), dtype=FillStruct, device=device)
    a3 = wp.zeros((dim_x, dim_x, dim_x), dtype=FillStruct, device=device)
    a4 = wp.zeros((dim_x, dim_x, dim_x, dim_x), dtype=FillStruct, device=device)

    assert_np_equal(a1.numpy(), np.zeros(a1.shape, dtype=nptype))
    assert_np_equal(a2.numpy(), np.zeros(a2.shape, dtype=nptype))
    assert_np_equal(a3.numpy(), np.zeros(a3.shape, dtype=nptype))
    assert_np_equal(a4.numpy(), np.zeros(a4.shape, dtype=nptype))

    s = FillStruct()

    # fill with default struct value (should be all zeros)
    a1.fill_(s)
    a2.fill_(s)
    a3.fill_(s)
    a4.fill_(s)

    assert_np_equal(a1.numpy(), np.zeros(a1.shape, dtype=nptype))
    assert_np_equal(a2.numpy(), np.zeros(a2.shape, dtype=nptype))
    assert_np_equal(a3.numpy(), np.zeros(a3.shape, dtype=nptype))
    assert_np_equal(a4.numpy(), np.zeros(a4.shape, dtype=nptype))

    # scalars
    s.i1 = -17
    s.i2 = 42
    s.i4 = 99
    s.i8 = 101
    s.f2 = -1.25
    s.f4 = 13.37
    s.f8 = 0.125
    # vectors
    s.v2 = [21, 22]
    s.v3 = [31, 32, 33]
    s.v4 = [41, 42, 43, 44]
    s.v5 = [51, 52, 53, 54, 55]
    # matrices
    s.m2 = [[61, 62]] * 2
    s.m3 = [[71, 72, 73]] * 3
    s.m4 = [[81, 82, 83, 84]] * 4
    s.m5 = [[91, 92, 93, 94, 95]] * 5
    # arrays
    s.a1 = wp.zeros((2,) * 1, dtype=float, device=device)
    s.a2 = wp.zeros((2,) * 2, dtype=float, device=device)
    s.a3 = wp.zeros((2,) * 3, dtype=float, device=device)
    s.a4 = wp.zeros((2,) * 4, dtype=float, device=device)

    # fill with custom struct value
    a1.fill_(s)
    a2.fill_(s)
    a3.fill_(s)
    a4.fill_(s)

    ns = s.numpy_value()

    expected1 = np.empty(a1.shape, dtype=nptype)
    expected2 = np.empty(a2.shape, dtype=nptype)
    expected3 = np.empty(a3.shape, dtype=nptype)
    expected4 = np.empty(a4.shape, dtype=nptype)

    expected1.fill(ns)
    expected2.fill(ns)
    expected3.fill(ns)
    expected4.fill(ns)

    assert_np_equal(a1.numpy(), expected1)
    assert_np_equal(a2.numpy(), expected2)
    assert_np_equal(a3.numpy(), expected3)
    assert_np_equal(a4.numpy(), expected4)

    # test clearing
    a1.zero_()
    a2.zero_()
    a3.zero_()
    a4.zero_()

    assert_np_equal(a1.numpy(), np.zeros(a1.shape, dtype=nptype))
    assert_np_equal(a2.numpy(), np.zeros(a2.shape, dtype=nptype))
    assert_np_equal(a3.numpy(), np.zeros(a3.shape, dtype=nptype))
    assert_np_equal(a4.numpy(), np.zeros(a4.shape, dtype=nptype))


def test_fill_slices(test, device):
    # test fill_ and zero_ for non-contiguous arrays
    # Note: we don't need to test the whole range of dtypes (vectors, matrices, structs) here

    dim_x = 8

    for nptype, wptype in wp.types.np_dtype_to_warp_type.items():
        a1 = wp.zeros(dim_x, dtype=wptype, device=device)
        a2 = wp.zeros((dim_x, dim_x), dtype=wptype, device=device)
        a3 = wp.zeros((dim_x, dim_x, dim_x), dtype=wptype, device=device)
        a4 = wp.zeros((dim_x, dim_x, dim_x, dim_x), dtype=wptype, device=device)

        assert_np_equal(a1.numpy(), np.zeros(a1.shape, dtype=nptype))
        assert_np_equal(a2.numpy(), np.zeros(a2.shape, dtype=nptype))
        assert_np_equal(a3.numpy(), np.zeros(a3.shape, dtype=nptype))
        assert_np_equal(a4.numpy(), np.zeros(a4.shape, dtype=nptype))

        # partititon each array into even and odd slices
        a1a = a1[::2]
        a1b = a1[1::2]
        a2a = a2[::2]
        a2b = a2[1::2]
        a3a = a3[::2]
        a3b = a3[1::2]
        a4a = a4[::2]
        a4b = a4[1::2]

        # fill even slices
        fill_a = 17
        a1a.fill_(fill_a)
        a2a.fill_(fill_a)
        a3a.fill_(fill_a)
        a4a.fill_(fill_a)

        # ensure filled slices are correct
        assert_np_equal(a1a.numpy(), np.full(a1a.shape, fill_a, dtype=nptype))
        assert_np_equal(a2a.numpy(), np.full(a2a.shape, fill_a, dtype=nptype))
        assert_np_equal(a3a.numpy(), np.full(a3a.shape, fill_a, dtype=nptype))
        assert_np_equal(a4a.numpy(), np.full(a4a.shape, fill_a, dtype=nptype))

        # ensure unfilled slices are unaffected
        assert_np_equal(a1b.numpy(), np.zeros(a1b.shape, dtype=nptype))
        assert_np_equal(a2b.numpy(), np.zeros(a2b.shape, dtype=nptype))
        assert_np_equal(a3b.numpy(), np.zeros(a3b.shape, dtype=nptype))
        assert_np_equal(a4b.numpy(), np.zeros(a4b.shape, dtype=nptype))

        # fill odd slices
        fill_b = 42
        a1b.fill_(fill_b)
        a2b.fill_(fill_b)
        a3b.fill_(fill_b)
        a4b.fill_(fill_b)

        # ensure filled slices are correct
        assert_np_equal(a1b.numpy(), np.full(a1b.shape, fill_b, dtype=nptype))
        assert_np_equal(a2b.numpy(), np.full(a2b.shape, fill_b, dtype=nptype))
        assert_np_equal(a3b.numpy(), np.full(a3b.shape, fill_b, dtype=nptype))
        assert_np_equal(a4b.numpy(), np.full(a4b.shape, fill_b, dtype=nptype))

        # ensure unfilled slices are unaffected
        assert_np_equal(a1a.numpy(), np.full(a1a.shape, fill_a, dtype=nptype))
        assert_np_equal(a2a.numpy(), np.full(a2a.shape, fill_a, dtype=nptype))
        assert_np_equal(a3a.numpy(), np.full(a3a.shape, fill_a, dtype=nptype))
        assert_np_equal(a4a.numpy(), np.full(a4a.shape, fill_a, dtype=nptype))

        # clear even slices
        a1a.zero_()
        a2a.zero_()
        a3a.zero_()
        a4a.zero_()

        # ensure cleared slices are correct
        assert_np_equal(a1a.numpy(), np.zeros(a1a.shape, dtype=nptype))
        assert_np_equal(a2a.numpy(), np.zeros(a2a.shape, dtype=nptype))
        assert_np_equal(a3a.numpy(), np.zeros(a3a.shape, dtype=nptype))
        assert_np_equal(a4a.numpy(), np.zeros(a4a.shape, dtype=nptype))

        # ensure uncleared slices are unaffected
        assert_np_equal(a1b.numpy(), np.full(a1b.shape, fill_b, dtype=nptype))
        assert_np_equal(a2b.numpy(), np.full(a2b.shape, fill_b, dtype=nptype))
        assert_np_equal(a3b.numpy(), np.full(a3b.shape, fill_b, dtype=nptype))
        assert_np_equal(a4b.numpy(), np.full(a4b.shape, fill_b, dtype=nptype))

        # re-fill even slices
        a1a.fill_(fill_a)
        a2a.fill_(fill_a)
        a3a.fill_(fill_a)
        a4a.fill_(fill_a)

        # clear odd slices
        a1b.zero_()
        a2b.zero_()
        a3b.zero_()
        a4b.zero_()

        # ensure cleared slices are correct
        assert_np_equal(a1b.numpy(), np.zeros(a1b.shape, dtype=nptype))
        assert_np_equal(a2b.numpy(), np.zeros(a2b.shape, dtype=nptype))
        assert_np_equal(a3b.numpy(), np.zeros(a3b.shape, dtype=nptype))
        assert_np_equal(a4b.numpy(), np.zeros(a4b.shape, dtype=nptype))

        # ensure uncleared slices are unaffected
        assert_np_equal(a1a.numpy(), np.full(a1a.shape, fill_a, dtype=nptype))
        assert_np_equal(a2a.numpy(), np.full(a2a.shape, fill_a, dtype=nptype))
        assert_np_equal(a3a.numpy(), np.full(a3a.shape, fill_a, dtype=nptype))
        assert_np_equal(a4a.numpy(), np.full(a4a.shape, fill_a, dtype=nptype))


def test_full_scalar(test, device):
    dim = 4

    for ndim in range(1, 5):
        shape = (dim,) * ndim

        for nptype, wptype in wp.types.np_dtype_to_warp_type.items():
            # fill with int value and specific dtype
            fill_value = 42
            a = wp.full(shape, fill_value, dtype=wptype, device=device)
            na = a.numpy()

            test.assertEqual(a.shape, shape)
            test.assertEqual(a.dtype, wptype)
            test.assertEqual(na.shape, shape)
            test.assertEqual(na.dtype, nptype)
            assert_np_equal(na, np.full(shape, fill_value, dtype=nptype))

            if wptype in wp.types.float_types:
                # fill with float value and specific dtype
                fill_value = 13.37
                a = wp.full(shape, fill_value, dtype=wptype, device=device)
                na = a.numpy()

                test.assertEqual(a.shape, shape)
                test.assertEqual(a.dtype, wptype)
                test.assertEqual(na.shape, shape)
                test.assertEqual(na.dtype, nptype)
                assert_np_equal(na, np.full(shape, fill_value, dtype=nptype))

        # fill with int value and automatically inferred dtype
        fill_value = 42
        a = wp.full(shape, fill_value, device=device)
        na = a.numpy()

        test.assertEqual(a.shape, shape)
        test.assertEqual(a.dtype, wp.int32)
        test.assertEqual(na.shape, shape)
        test.assertEqual(na.dtype, np.int32)
        assert_np_equal(na, np.full(shape, fill_value, dtype=np.int32))

        # fill with float value and automatically inferred dtype
        fill_value = 13.37
        a = wp.full(shape, fill_value, device=device)
        na = a.numpy()

        test.assertEqual(a.shape, shape)
        test.assertEqual(a.dtype, wp.float32)
        test.assertEqual(na.shape, shape)
        test.assertEqual(na.dtype, np.float32)
        assert_np_equal(na, np.full(shape, fill_value, dtype=np.float32))


def test_full_vector(test, device):
    dim = 4

    for ndim in range(1, 5):
        shape = (dim,) * ndim

        # full from scalar
        for veclen in [2, 3, 4, 5]:
            npshape = (*shape, veclen)

            for nptype, wptype in wp.types.np_dtype_to_warp_type.items():
                vectype = wp.types.vector(veclen, wptype)

                # fill with scalar int value and specific dtype
                fill_value = 42
                a = wp.full(shape, fill_value, dtype=vectype, device=device)
                na = a.numpy()

                test.assertEqual(a.shape, shape)
                test.assertEqual(a.dtype, vectype)
                test.assertEqual(na.shape, npshape)
                test.assertEqual(na.dtype, nptype)
                assert_np_equal(na, np.full(a.size * veclen, fill_value, dtype=nptype).reshape(npshape))

                if wptype in wp.types.float_types:
                    # fill with scalar float value and specific dtype
                    fill_value = 13.37
                    a = wp.full(shape, fill_value, dtype=vectype, device=device)
                    na = a.numpy()

                    test.assertEqual(a.shape, shape)
                    test.assertEqual(a.dtype, vectype)
                    test.assertEqual(na.shape, npshape)
                    test.assertEqual(na.dtype, nptype)
                    assert_np_equal(na, np.full(a.size * veclen, fill_value, dtype=nptype).reshape(npshape))

                # fill with vector value and specific dtype
                fill_vec = vectype(42)
                a = wp.full(shape, fill_vec, dtype=vectype, device=device)
                na = a.numpy()

                test.assertEqual(a.shape, shape)
                test.assertEqual(a.dtype, vectype)
                test.assertEqual(na.shape, npshape)
                test.assertEqual(na.dtype, nptype)
                assert_np_equal(na, np.full(a.size * veclen, 42, dtype=nptype).reshape(npshape))

                # fill with vector value and automatically inferred dtype
                a = wp.full(shape, fill_vec, device=device)
                na = a.numpy()

                test.assertEqual(a.shape, shape)
                test.assertEqual(a.dtype, vectype)
                test.assertEqual(na.shape, npshape)
                test.assertEqual(na.dtype, nptype)
                assert_np_equal(na, np.full(a.size * veclen, 42, dtype=nptype).reshape(npshape))

        fill_lists = [
            [17, 42],
            [17, 42, 99],
            [17, 42, 99, 101],
            [17, 42, 99, 101, 127],
        ]

        # full from list and numpy array
        for fill_list in fill_lists:
            veclen = len(fill_list)
            npshape = (*shape, veclen)

            for nptype, wptype in wp.types.np_dtype_to_warp_type.items():
                vectype = wp.types.vector(veclen, wptype)

                # fill with list and specific dtype
                a = wp.full(shape, fill_list, dtype=vectype, device=device)
                na = a.numpy()

                test.assertEqual(a.shape, shape)
                test.assertEqual(a.dtype, vectype)
                test.assertEqual(na.shape, npshape)
                test.assertEqual(na.dtype, nptype)

                expected = np.tile(np.array(fill_list, dtype=nptype), a.size).reshape(npshape)
                assert_np_equal(na, expected)

                fill_arr = np.array(fill_list, dtype=nptype)

                # fill with numpy array and specific dtype
                a = wp.full(shape, fill_arr, dtype=vectype, device=device)
                na = a.numpy()

                test.assertEqual(a.shape, shape)
                test.assertEqual(a.dtype, vectype)
                test.assertEqual(na.shape, npshape)
                test.assertEqual(na.dtype, nptype)
                assert_np_equal(na, expected)

                # fill with numpy array and automatically infer dtype
                a = wp.full(shape, fill_arr, device=device)
                na = a.numpy()

                test.assertEqual(a.shape, shape)
                test.assertTrue(wp.types.types_equal(a.dtype, vectype))
                test.assertEqual(na.shape, npshape)
                test.assertEqual(na.dtype, nptype)
                assert_np_equal(na, expected)

            # fill with list and automatically infer dtype
            a = wp.full(shape, fill_list, device=device)
            na = a.numpy()

            test.assertEqual(a.shape, shape)

            # check that the inferred dtype is a vector
            # Note that we cannot guarantee the scalar type, because it depends on numpy and may vary by platform
            # (e.g. int64 on Linux and int32 on Windows).
            test.assertEqual(a.dtype._wp_generic_type_str_, "vec_t")
            test.assertEqual(a.dtype._length_, veclen)

            expected = np.tile(np.array(fill_list), a.size).reshape(npshape)
            assert_np_equal(na, expected)


def test_full_matrix(test, device):
    dim = 4

    for ndim in range(1, 5):
        shape = (dim,) * ndim

        for nptype, wptype in wp.types.np_dtype_to_warp_type.items():
            matrix_types = [
                # square matrices
                wp.types.matrix((2, 2), wptype),
                wp.types.matrix((3, 3), wptype),
                wp.types.matrix((4, 4), wptype),
                wp.types.matrix((5, 5), wptype),
                # non-square matrices
                wp.types.matrix((2, 3), wptype),
                wp.types.matrix((3, 2), wptype),
                wp.types.matrix((3, 4), wptype),
                wp.types.matrix((4, 3), wptype),
            ]

            for mattype in matrix_types:
                npshape = (*shape, *mattype._shape_)

                # fill with scalar int value and specific dtype
                fill_value = 42
                a = wp.full(shape, fill_value, dtype=mattype, device=device)
                na = a.numpy()

                test.assertEqual(a.shape, shape)
                test.assertEqual(a.dtype, mattype)
                test.assertEqual(na.shape, npshape)
                test.assertEqual(na.dtype, nptype)
                assert_np_equal(na, np.full(a.size * mattype._length_, fill_value, dtype=nptype).reshape(npshape))

                if wptype in wp.types.float_types:
                    # fill with scalar float value and specific dtype
                    fill_value = 13.37
                    a = wp.full(shape, fill_value, dtype=mattype, device=device)
                    na = a.numpy()

                    test.assertEqual(a.shape, shape)
                    test.assertEqual(a.dtype, mattype)
                    test.assertEqual(na.shape, npshape)
                    test.assertEqual(na.dtype, nptype)
                    assert_np_equal(na, np.full(a.size * mattype._length_, fill_value, dtype=nptype).reshape(npshape))

                # fill with matrix value and specific dtype
                fill_mat = mattype(42)
                a = wp.full(shape, fill_mat, dtype=mattype, device=device)
                na = a.numpy()

                test.assertEqual(a.shape, shape)
                test.assertEqual(a.dtype, mattype)
                test.assertEqual(na.shape, npshape)
                test.assertEqual(na.dtype, nptype)
                assert_np_equal(na, np.full(a.size * mattype._length_, 42, dtype=nptype).reshape(npshape))

                # fill with matrix value and automatically inferred dtype
                fill_mat = mattype(42)
                a = wp.full(shape, fill_mat, device=device)
                na = a.numpy()

                test.assertEqual(a.shape, shape)
                test.assertEqual(a.dtype, mattype)
                test.assertEqual(na.shape, npshape)
                test.assertEqual(na.dtype, nptype)
                assert_np_equal(na, np.full(a.size * mattype._length_, 42, dtype=nptype).reshape(npshape))

                # fill with 1d numpy array and specific dtype
                if wptype != wp.bool:
                    fill_arr1d = np.arange(mattype._length_, dtype=nptype)
                else:
                    fill_arr1d = np.ones(mattype._length_, dtype=nptype)
                a = wp.full(shape, fill_arr1d, dtype=mattype, device=device)
                na = a.numpy()

                test.assertEqual(a.shape, shape)
                test.assertEqual(a.dtype, mattype)
                test.assertEqual(na.shape, npshape)
                test.assertEqual(na.dtype, nptype)

                expected = np.tile(fill_arr1d, a.size).reshape(npshape)
                assert_np_equal(na, expected)

                # fill with 2d numpy array and specific dtype
                fill_arr2d = fill_arr1d.reshape(mattype._shape_)
                a = wp.full(shape, fill_arr2d, dtype=mattype, device=device)
                na = a.numpy()

                test.assertEqual(a.shape, shape)
                test.assertEqual(a.dtype, mattype)
                test.assertEqual(na.shape, npshape)
                test.assertEqual(na.dtype, nptype)
                assert_np_equal(na, expected)

                # fill with 2d numpy array and automatically infer dtype
                a = wp.full(shape, fill_arr2d, device=device)
                na = a.numpy()

                test.assertEqual(a.shape, shape)
                test.assertTrue(wp.types.types_equal(a.dtype, mattype))
                test.assertEqual(na.shape, npshape)
                test.assertEqual(na.dtype, nptype)
                assert_np_equal(na, expected)

                # fill with flat list and specific dtype
                fill_list1d = list(fill_arr1d)
                a = wp.full(shape, fill_list1d, dtype=mattype, device=device)
                na = a.numpy()

                test.assertEqual(a.shape, shape)
                test.assertEqual(a.dtype, mattype)
                test.assertEqual(na.shape, npshape)
                test.assertEqual(na.dtype, nptype)
                assert_np_equal(na, expected)

                # fill with nested list and specific dtype
                fill_list2d = [list(row) for row in fill_arr2d]
                a = wp.full(shape, fill_list2d, dtype=mattype, device=device)
                na = a.numpy()

                test.assertEqual(a.shape, shape)
                test.assertEqual(a.dtype, mattype)
                test.assertEqual(na.shape, npshape)
                test.assertEqual(na.dtype, nptype)
                assert_np_equal(na, expected)

        mat_lists = [
            # square matrices
            [[1, 2], [3, 4]],
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
            # non-square matrices
            [[1, 2, 3, 4], [5, 6, 7, 8]],
            [[1, 2], [3, 4], [5, 6], [7, 8]],
        ]

        # fill with nested lists and automatically infer dtype
        for fill_list in mat_lists:
            num_rows = len(fill_list)
            num_cols = len(fill_list[0])
            npshape = (*shape, num_rows, num_cols)

            a = wp.full(shape, fill_list, device=device)
            na = a.numpy()

            test.assertEqual(a.shape, shape)

            # check that the inferred dtype is a correctly shaped matrix
            # Note that we cannot guarantee the scalar type, because it depends on numpy and may vary by platform
            # (e.g. int64 on Linux and int32 on Windows).
            test.assertEqual(a.dtype._wp_generic_type_str_, "mat_t")
            test.assertEqual(a.dtype._shape_, (num_rows, num_cols))

            expected = np.tile(np.array(fill_list).flatten(), a.size).reshape(npshape)
            assert_np_equal(na, expected)


def test_full_struct(test, device):
    dim = 4

    for ndim in range(1, 5):
        shape = (dim,) * ndim

        s = FillStruct()

        # fill with default struct (should be zeros)
        a = wp.full(shape, s, dtype=FillStruct, device=device)
        na = a.numpy()

        test.assertEqual(a.shape, shape)
        test.assertEqual(a.dtype, FillStruct)
        test.assertEqual(na.shape, shape)
        test.assertEqual(na.dtype, FillStruct.numpy_dtype())
        assert_np_equal(na, np.zeros(a.shape, dtype=FillStruct.numpy_dtype()))

        # scalars
        s.i1 = -17
        s.i2 = 42
        s.i4 = 99
        s.i8 = 101
        s.f2 = -1.25
        s.f4 = 13.37
        s.f8 = 0.125
        # vectors
        s.v2 = [21, 22]
        s.v3 = [31, 32, 33]
        s.v4 = [41, 42, 43, 44]
        s.v5 = [51, 52, 53, 54, 55]
        # matrices
        s.m2 = [[61, 62]] * 2
        s.m3 = [[71, 72, 73]] * 3
        s.m4 = [[81, 82, 83, 84]] * 4
        s.m5 = [[91, 92, 93, 94, 95]] * 5
        # arrays
        s.a1 = wp.zeros((2,) * 1, dtype=float, device=device)
        s.a2 = wp.zeros((2,) * 2, dtype=float, device=device)
        s.a3 = wp.zeros((2,) * 3, dtype=float, device=device)
        s.a4 = wp.zeros((2,) * 4, dtype=float, device=device)

        # fill with initialized struct and explicit dtype
        a = wp.full(shape, s, dtype=FillStruct, device=device)
        na = a.numpy()

        test.assertEqual(a.shape, shape)
        test.assertEqual(a.dtype, FillStruct)
        test.assertEqual(na.shape, shape)
        test.assertEqual(na.dtype, FillStruct.numpy_dtype())

        expected = np.empty(shape, dtype=FillStruct.numpy_dtype())
        expected.fill(s.numpy_value())
        assert_np_equal(na, expected)

        # fill with initialized struct and automatically inferred dtype
        a = wp.full(shape, s, device=device)
        na = a.numpy()

        test.assertEqual(a.shape, shape)
        test.assertEqual(a.dtype, FillStruct)
        test.assertEqual(na.shape, shape)
        test.assertEqual(na.dtype, FillStruct.numpy_dtype())
        assert_np_equal(na, expected)


def test_ones_scalar(test, device):
    dim = 4

    for ndim in range(1, 5):
        shape = (dim,) * ndim

        for nptype, wptype in wp.types.np_dtype_to_warp_type.items():
            a = wp.ones(shape, dtype=wptype, device=device)
            na = a.numpy()

            test.assertEqual(a.shape, shape)
            test.assertEqual(a.dtype, wptype)
            test.assertEqual(na.shape, shape)
            test.assertEqual(na.dtype, nptype)
            assert_np_equal(na, np.ones(shape, dtype=nptype))


def test_ones_vector(test, device):
    dim = 4

    for ndim in range(1, 5):
        shape = (dim,) * ndim

        for veclen in [2, 3, 4, 5]:
            npshape = (*shape, veclen)

            for nptype, wptype in wp.types.np_dtype_to_warp_type.items():
                vectype = wp.types.vector(veclen, wptype)

                a = wp.ones(shape, dtype=vectype, device=device)
                na = a.numpy()

                test.assertEqual(a.shape, shape)
                test.assertEqual(a.dtype, vectype)
                test.assertEqual(na.shape, npshape)
                test.assertEqual(na.dtype, nptype)
                assert_np_equal(na, np.ones(npshape, dtype=nptype))


def test_ones_matrix(test, device):
    dim = 4

    for ndim in range(1, 5):
        shape = (dim,) * ndim

        for nptype, wptype in wp.types.np_dtype_to_warp_type.items():
            matrix_types = [
                # square matrices
                wp.types.matrix((2, 2), wptype),
                wp.types.matrix((3, 3), wptype),
                wp.types.matrix((4, 4), wptype),
                wp.types.matrix((5, 5), wptype),
                # non-square matrices
                wp.types.matrix((2, 3), wptype),
                wp.types.matrix((3, 2), wptype),
                wp.types.matrix((3, 4), wptype),
                wp.types.matrix((4, 3), wptype),
            ]

            for mattype in matrix_types:
                npshape = (*shape, *mattype._shape_)

                a = wp.ones(shape, dtype=mattype, device=device)
                na = a.numpy()

                test.assertEqual(a.shape, shape)
                test.assertEqual(a.dtype, mattype)
                test.assertEqual(na.shape, npshape)
                test.assertEqual(na.dtype, nptype)
                assert_np_equal(na, np.ones(npshape, dtype=nptype))


def test_ones_like_scalar(test, device):
    dim = 4

    for ndim in range(1, 5):
        shape = (dim,) * ndim

        for nptype, wptype in wp.types.np_dtype_to_warp_type.items():
            # source array
            a = wp.zeros(shape, dtype=wptype, device=device)
            na = a.numpy()
            test.assertEqual(a.shape, shape)
            test.assertEqual(a.dtype, wptype)
            test.assertEqual(na.shape, shape)
            test.assertEqual(na.dtype, nptype)
            assert_np_equal(na, np.zeros(shape, dtype=nptype))

            # ones array
            b = wp.ones_like(a)
            nb = b.numpy()
            test.assertEqual(b.shape, shape)
            test.assertEqual(b.dtype, wptype)
            test.assertEqual(nb.shape, shape)
            test.assertEqual(nb.dtype, nptype)
            assert_np_equal(nb, np.ones(shape, dtype=nptype))


def test_ones_like_vector(test, device):
    dim = 4

    for ndim in range(1, 5):
        shape = (dim,) * ndim

        for veclen in [2, 3, 4, 5]:
            npshape = (*shape, veclen)

            for nptype, wptype in wp.types.np_dtype_to_warp_type.items():
                vectype = wp.types.vector(veclen, wptype)

                # source array
                a = wp.zeros(shape, dtype=vectype, device=device)
                na = a.numpy()
                test.assertEqual(a.shape, shape)
                test.assertEqual(a.dtype, vectype)
                test.assertEqual(na.shape, npshape)
                test.assertEqual(na.dtype, nptype)
                assert_np_equal(na, np.zeros(npshape, dtype=nptype))

                # ones array
                b = wp.ones_like(a)
                nb = b.numpy()
                test.assertEqual(b.shape, shape)
                test.assertEqual(b.dtype, vectype)
                test.assertEqual(nb.shape, npshape)
                test.assertEqual(nb.dtype, nptype)
                assert_np_equal(nb, np.ones(npshape, dtype=nptype))


def test_ones_like_matrix(test, device):
    dim = 4

    for ndim in range(1, 5):
        shape = (dim,) * ndim

        for nptype, wptype in wp.types.np_dtype_to_warp_type.items():
            matrix_types = [
                # square matrices
                wp.types.matrix((2, 2), wptype),
                wp.types.matrix((3, 3), wptype),
                wp.types.matrix((4, 4), wptype),
                wp.types.matrix((5, 5), wptype),
                # non-square matrices
                wp.types.matrix((2, 3), wptype),
                wp.types.matrix((3, 2), wptype),
                wp.types.matrix((3, 4), wptype),
                wp.types.matrix((4, 3), wptype),
            ]

            for mattype in matrix_types:
                npshape = (*shape, *mattype._shape_)

                # source array
                a = wp.zeros(shape, dtype=mattype, device=device)
                na = a.numpy()
                test.assertEqual(a.shape, shape)
                test.assertEqual(a.dtype, mattype)
                test.assertEqual(na.shape, npshape)
                test.assertEqual(na.dtype, nptype)
                assert_np_equal(na, np.zeros(npshape, dtype=nptype))

                # ones array
                b = wp.ones_like(a)
                nb = b.numpy()
                test.assertEqual(b.shape, shape)
                test.assertEqual(b.dtype, mattype)
                test.assertEqual(nb.shape, npshape)
                test.assertEqual(nb.dtype, nptype)
                assert_np_equal(nb, np.ones(npshape, dtype=nptype))


def test_round_trip(test, device):
    rng = np.random.default_rng(123)
    dim_x = 4

    for nptype, wptype in wp.types.np_dtype_to_warp_type.items():
        a_np = rng.standard_normal(size=dim_x).astype(nptype)
        a = wp.array(a_np, device=device)
        test.assertEqual(a.dtype, wptype)

        assert_np_equal(a.numpy(), a_np)

        v_np = rng.standard_normal(size=(dim_x, 3)).astype(nptype)
        v = wp.array(v_np, dtype=wp.types.vector(3, wptype), device=device)

        assert_np_equal(v.numpy(), v_np)


def test_empty_array(test, device):
    # Test whether common operations work with empty (zero-sized) arrays
    # without throwing exceptions.

    def test_empty_ops(ndim, nrows, ncols, wptype, nptype):
        shape = (0,) * ndim
        dtype_shape = ()

        if wptype in wp.types.scalar_types:
            # scalar, vector, or matrix
            if ncols > 0:
                if nrows > 0:
                    wptype = wp.types.matrix((nrows, ncols), wptype)
                else:
                    wptype = wp.types.vector(ncols, wptype)
                dtype_shape = wptype._shape_
            fill_value = wptype(42)
        else:
            # struct
            fill_value = wptype()

        # create a zero-sized array
        a = wp.empty(shape, dtype=wptype, device=device, requires_grad=True)

        test.assertEqual(a.ptr, None)
        test.assertEqual(a.size, 0)
        test.assertEqual(a.shape, shape)
        test.assertEqual(a.grad.ptr, None)
        test.assertEqual(a.grad.size, 0)
        test.assertEqual(a.grad.shape, shape)

        # all of these methods should succeed with zero-sized arrays
        a.zero_()
        a.fill_(fill_value)
        b = a.flatten()
        b = a.reshape((0,))
        b = a.transpose()
        b = a.contiguous()

        b = wp.empty_like(a)
        b = wp.zeros_like(a)
        b = wp.full_like(a, fill_value)
        b = wp.clone(a)

        wp.copy(a, b)
        a.assign(b)

        na = a.numpy()
        test.assertEqual(na.size, 0)
        test.assertEqual(na.shape, (*shape, *dtype_shape))
        test.assertEqual(na.dtype, nptype)

        test.assertEqual(a.list(), [])

    for ndim in range(1, 5):
        # test with scalars, vectors, and matrices
        for nptype, wptype in wp.types.np_dtype_to_warp_type.items():
            # scalars
            test_empty_ops(ndim, 0, 0, wptype, nptype)

            for ncols in [2, 3, 4, 5]:
                # vectors
                test_empty_ops(ndim, 0, ncols, wptype, nptype)
                # square matrices
                test_empty_ops(ndim, ncols, ncols, wptype, nptype)

            # non-square matrices
            test_empty_ops(ndim, 2, 3, wptype, nptype)
            test_empty_ops(ndim, 3, 2, wptype, nptype)
            test_empty_ops(ndim, 3, 4, wptype, nptype)
            test_empty_ops(ndim, 4, 3, wptype, nptype)

        # test with structs
        test_empty_ops(ndim, 0, 0, FillStruct, FillStruct.numpy_dtype())


def test_empty_from_numpy(test, device):
    # Test whether wrapping an empty (zero-sized) numpy array works correctly

    def test_empty_from_data(ndim, nrows, ncols, wptype, nptype):
        shape = (0,) * ndim
        dtype_shape = ()

        if ncols > 0:
            if nrows > 0:
                wptype = wp.types.matrix((nrows, ncols), wptype)
            else:
                wptype = wp.types.vector(ncols, wptype)
            dtype_shape = wptype._shape_

        npshape = (*shape, *dtype_shape)

        na = np.empty(npshape, dtype=nptype)
        a = wp.array(na, dtype=wptype, device=device)
        test.assertEqual(a.size, 0)
        test.assertEqual(a.shape, shape)

    for ndim in range(1, 5):
        # test with scalars, vectors, and matrices
        for nptype, wptype in wp.types.np_dtype_to_warp_type.items():
            # scalars
            test_empty_from_data(ndim, 0, 0, wptype, nptype)

            for ncols in [2, 3, 4, 5]:
                # vectors
                test_empty_from_data(ndim, 0, ncols, wptype, nptype)
                # square matrices
                test_empty_from_data(ndim, ncols, ncols, wptype, nptype)

            # non-square matrices
            test_empty_from_data(ndim, 2, 3, wptype, nptype)
            test_empty_from_data(ndim, 3, 2, wptype, nptype)
            test_empty_from_data(ndim, 3, 4, wptype, nptype)
            test_empty_from_data(ndim, 4, 3, wptype, nptype)


def test_empty_from_list(test, device):
    # Test whether creating an array from an empty Python list works correctly

    def test_empty_from_data(nrows, ncols, wptype):
        if ncols > 0:
            if nrows > 0:
                wptype = wp.types.matrix((nrows, ncols), wptype)
            else:
                wptype = wp.types.vector(ncols, wptype)

        a = wp.array([], dtype=wptype, device=device)
        test.assertEqual(a.size, 0)
        test.assertEqual(a.shape, (0,))

    # test with scalars, vectors, and matrices
    for wptype in wp.types.scalar_types:
        # scalars
        test_empty_from_data(0, 0, wptype)

        for ncols in [2, 3, 4, 5]:
            # vectors
            test_empty_from_data(0, ncols, wptype)
            # square matrices
            test_empty_from_data(ncols, ncols, wptype)

        # non-square matrices
        test_empty_from_data(2, 3, wptype)
        test_empty_from_data(3, 2, wptype)
        test_empty_from_data(3, 4, wptype)
        test_empty_from_data(4, 3, wptype)


def test_to_list_scalar(test, device):
    dim = 3
    fill_value = 42

    for ndim in range(1, 5):
        shape = (dim,) * ndim

        for wptype in wp.types.scalar_types:
            a = wp.full(shape, fill_value, dtype=wptype, device=device)
            l = a.list()

            test.assertEqual(len(l), a.size)
            test.assertTrue(all(x == fill_value for x in l))


def test_to_list_vector(test, device):
    dim = 3

    for ndim in range(1, 5):
        shape = (dim,) * ndim

        for veclen in [2, 3, 4, 5]:
            for wptype in wp.types.scalar_types:
                vectype = wp.types.vector(veclen, wptype)
                fill_value = vectype(42)

                a = wp.full(shape, fill_value, dtype=vectype, device=device)
                l = a.list()

                test.assertEqual(len(l), a.size)
                test.assertTrue(all(x == fill_value for x in l))


def test_to_list_matrix(test, device):
    dim = 3

    for ndim in range(1, 5):
        shape = (dim,) * ndim

        for wptype in wp.types.scalar_types:
            matrix_types = [
                # square matrices
                wp.types.matrix((2, 2), wptype),
                wp.types.matrix((3, 3), wptype),
                wp.types.matrix((4, 4), wptype),
                wp.types.matrix((5, 5), wptype),
                # non-square matrices
                wp.types.matrix((2, 3), wptype),
                wp.types.matrix((3, 2), wptype),
                wp.types.matrix((3, 4), wptype),
                wp.types.matrix((4, 3), wptype),
            ]

            for mattype in matrix_types:
                fill_value = mattype(42)

                a = wp.full(shape, fill_value, dtype=mattype, device=device)
                l = a.list()

                test.assertEqual(len(l), a.size)
                test.assertTrue(all(x == fill_value for x in l))


def test_to_list_struct(test, device):
    @wp.struct
    class Inner:
        h: wp.float16
        v: wp.vec3

    @wp.struct
    class ListStruct:
        i: int
        f: float
        h: wp.float16
        vi: wp.vec2i
        vf: wp.vec3f
        vh: wp.vec4h
        mi: wp.types.matrix((2, 2), int)
        mf: wp.types.matrix((3, 3), float)
        mh: wp.types.matrix((4, 4), wp.float16)
        inner: Inner
        a1: wp.array(dtype=int)
        a2: wp.array2d(dtype=float)
        a3: wp.array3d(dtype=wp.float16)
        bool: wp.bool

    dim = 3

    s = ListStruct()
    s.i = 42
    s.f = 2.5
    s.h = -1.25
    s.vi = wp.vec2i(1, 2)
    s.vf = wp.vec3f(0.1, 0.2, 0.3)
    s.vh = wp.vec4h(1.0, 2.0, 3.0, 4.0)
    s.mi = [[1, 2], [3, 4]]
    s.mf = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    s.mh = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    s.inner = Inner()
    s.inner.h = 1.5
    s.inner.v = [1, 2, 3]
    s.a1 = wp.empty(1, dtype=int, device=device)
    s.a2 = wp.empty((1, 1), dtype=float, device=device)
    s.a3 = wp.empty((1, 1, 1), dtype=wp.float16, device=device)
    s.bool = True

    for ndim in range(1, 5):
        shape = (dim,) * ndim

        a = wp.full(shape, s, dtype=ListStruct, device=device)
        l = a.list()

        for i in range(a.size):
            test.assertEqual(l[i].i, s.i)
            test.assertEqual(l[i].f, s.f)
            test.assertEqual(l[i].h, s.h)
            test.assertEqual(l[i].vi, s.vi)
            test.assertEqual(l[i].vf, s.vf)
            test.assertEqual(l[i].vh, s.vh)
            test.assertEqual(l[i].mi, s.mi)
            test.assertEqual(l[i].mf, s.mf)
            test.assertEqual(l[i].mh, s.mh)
            test.assertEqual(l[i].bool, s.bool)
            test.assertEqual(l[i].inner.h, s.inner.h)
            test.assertEqual(l[i].inner.v, s.inner.v)
            test.assertEqual(l[i].a1.dtype, s.a1.dtype)
            test.assertEqual(l[i].a1.ndim, s.a1.ndim)
            test.assertEqual(l[i].a2.dtype, s.a2.dtype)
            test.assertEqual(l[i].a2.ndim, s.a2.ndim)
            test.assertEqual(l[i].a3.dtype, s.a3.dtype)
            test.assertEqual(l[i].a3.ndim, s.a3.ndim)


@wp.kernel
def kernel_array_to_bool(array_null: wp.array(dtype=float), array_valid: wp.array(dtype=float)):
    if not array_null:
        # always succeed
        wp.expect_eq(0, 0)
    else:
        # force failure
        wp.expect_eq(1, 2)

    if array_valid:
        # always succeed
        wp.expect_eq(0, 0)
    else:
        # force failure
        wp.expect_eq(1, 2)


def test_array_to_bool(test, device):
    arr = wp.zeros(8, dtype=float, device=device)

    wp.launch(kernel_array_to_bool, dim=1, inputs=[None, arr], device=device)


@wp.struct
class InputStruct:
    param1: int
    param2: float
    param3: wp.vec3
    param4: wp.array(dtype=float)


@wp.struct
class OutputStruct:
    param1: int
    param2: float
    param3: wp.vec3


@wp.kernel
def struct_array_kernel(inputs: wp.array(dtype=InputStruct), outputs: wp.array(dtype=OutputStruct)):
    tid = wp.tid()

    wp.expect_eq(inputs[tid].param1, tid)
    wp.expect_eq(inputs[tid].param2, float(tid * tid))

    wp.expect_eq(inputs[tid].param3[0], 1.0)
    wp.expect_eq(inputs[tid].param3[1], 2.0)
    wp.expect_eq(inputs[tid].param3[2], 3.0)

    wp.expect_eq(inputs[tid].param4[0], 1.0)
    wp.expect_eq(inputs[tid].param4[1], 2.0)
    wp.expect_eq(inputs[tid].param4[2], 3.0)

    o = OutputStruct()
    o.param1 = inputs[tid].param1
    o.param2 = inputs[tid].param2
    o.param3 = inputs[tid].param3

    outputs[tid] = o


def test_array_of_structs(test, device):
    num_items = 10

    l = []
    for i in range(num_items):
        s = InputStruct()
        s.param1 = i
        s.param2 = float(i * i)
        s.param3 = wp.vec3(1.0, 2.0, 3.0)
        s.param4 = wp.array([1.0, 2.0, 3.0], dtype=float, device=device)

        l.append(s)

    # initialize array from list of structs
    inputs = wp.array(l, dtype=InputStruct, device=device)
    outputs = wp.zeros(num_items, dtype=OutputStruct, device=device)

    # pass to our compute kernel
    wp.launch(struct_array_kernel, dim=num_items, inputs=[inputs, outputs], device=device)

    out_numpy = outputs.numpy()
    out_list = outputs.list()
    out_cptr = outputs.to("cpu").cptr()

    for i in range(num_items):
        test.assertEqual(out_numpy[i][0], l[i].param1)
        test.assertEqual(out_numpy[i][1], l[i].param2)
        assert_np_equal(out_numpy[i][2], np.array(l[i].param3))

        # test named slices of numpy structured array
        test.assertEqual(out_numpy["param1"][i], l[i].param1)
        test.assertEqual(out_numpy["param2"][i], l[i].param2)
        assert_np_equal(out_numpy["param3"][i], np.array(l[i].param3))

        test.assertEqual(out_list[i].param1, l[i].param1)
        test.assertEqual(out_list[i].param2, l[i].param2)
        test.assertEqual(out_list[i].param3, l[i].param3)

        test.assertEqual(out_cptr[i].param1, l[i].param1)
        test.assertEqual(out_cptr[i].param2, l[i].param2)
        test.assertEqual(out_cptr[i].param3, l[i].param3)


@wp.struct
class GradStruct:
    param1: int
    param2: float
    param3: wp.vec3


@wp.kernel
def test_array_of_structs_grad_kernel(inputs: wp.array(dtype=GradStruct), loss: wp.array(dtype=float)):
    tid = wp.tid()

    wp.atomic_add(loss, 0, inputs[tid].param2 * 2.0)


def test_array_of_structs_grad(test, device):
    num_items = 10

    l = []
    for i in range(num_items):
        g = GradStruct()
        g.param2 = float(i)

        l.append(g)

    a = wp.array(l, dtype=GradStruct, device=device, requires_grad=True)
    loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)

    with wp.Tape() as tape:
        wp.launch(test_array_of_structs_grad_kernel, dim=num_items, inputs=[a, loss], device=device)

    tape.backward(loss)

    grads = a.grad.numpy()
    assert_np_equal(grads["param2"], np.full(num_items, 2.0, dtype=np.float32))


@wp.struct
class NumpyStruct:
    x: int
    v: wp.vec3


def test_array_of_structs_from_numpy(test, device):
    num_items = 10

    na = np.zeros(num_items, dtype=NumpyStruct.numpy_dtype())
    na["x"] = 17
    na["v"] = (1, 2, 3)

    a = wp.array(data=na, dtype=NumpyStruct, device=device)

    assert_np_equal(a.numpy(), na)


def test_array_of_structs_roundtrip(test, device):
    num_items = 10

    value = NumpyStruct()
    value.x = 17
    value.v = wp.vec3(1.0, 2.0, 3.0)

    # create Warp structured array
    a = wp.full(num_items, value, device=device)

    # convert to NumPy structured array
    na = a.numpy()

    expected = np.zeros(num_items, dtype=NumpyStruct.numpy_dtype())
    expected["x"] = value.x
    expected["v"] = value.v

    assert_np_equal(na, expected)

    # modify a field
    na["x"] = 42

    # convert back to Warp array
    a = wp.from_numpy(na, NumpyStruct, device=device)

    expected["x"] = 42

    assert_np_equal(a.numpy(), expected)


def test_array_from_numpy(test, device):
    arr = np.array((1.0, 2.0, 3.0), dtype=float)

    result = wp.from_numpy(arr, device=device)
    expected = wp.array((1.0, 2.0, 3.0), dtype=wp.float32, shape=(3,))
    assert_np_equal(result.numpy(), expected.numpy())

    result = wp.from_numpy(arr, dtype=wp.vec3, device=device)
    expected = wp.array(((1.0, 2.0, 3.0),), dtype=wp.vec3, shape=(1,))
    assert_np_equal(result.numpy(), expected.numpy())

    # --------------------------------------------------------------------------

    arr = np.array(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)), dtype=float)

    result = wp.from_numpy(arr, device=device)
    expected = wp.array(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)), dtype=wp.vec3, shape=(2,))
    assert_np_equal(result.numpy(), expected.numpy())

    result = wp.from_numpy(arr, dtype=wp.float32, device=device)
    expected = wp.array(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)), dtype=wp.float32, shape=(2, 3))
    assert_np_equal(result.numpy(), expected.numpy())

    result = wp.from_numpy(arr, dtype=wp.float32, shape=(6,), device=device)
    expected = wp.array((1.0, 2.0, 3.0, 4.0, 5.0, 6.0), dtype=wp.float32, shape=(6,))
    assert_np_equal(result.numpy(), expected.numpy())

    # --------------------------------------------------------------------------

    arr = np.array(
        (
            (
                (1.0, 2.0, 3.0, 4.0),
                (2.0, 3.0, 4.0, 5.0),
                (3.0, 4.0, 5.0, 6.0),
                (4.0, 5.0, 6.0, 7.0),
            ),
            (
                (2.0, 3.0, 4.0, 5.0),
                (3.0, 4.0, 5.0, 6.0),
                (4.0, 5.0, 6.0, 7.0),
                (5.0, 6.0, 7.0, 8.0),
            ),
        ),
        dtype=float,
    )

    result = wp.from_numpy(arr, device=device)
    expected = wp.array(
        (
            (
                (1.0, 2.0, 3.0, 4.0),
                (2.0, 3.0, 4.0, 5.0),
                (3.0, 4.0, 5.0, 6.0),
                (4.0, 5.0, 6.0, 7.0),
            ),
            (
                (2.0, 3.0, 4.0, 5.0),
                (3.0, 4.0, 5.0, 6.0),
                (4.0, 5.0, 6.0, 7.0),
                (5.0, 6.0, 7.0, 8.0),
            ),
        ),
        dtype=wp.mat44,
        shape=(2,),
    )
    assert_np_equal(result.numpy(), expected.numpy())

    result = wp.from_numpy(arr, dtype=wp.float32, device=device)
    expected = wp.array(
        (
            (
                (1.0, 2.0, 3.0, 4.0),
                (2.0, 3.0, 4.0, 5.0),
                (3.0, 4.0, 5.0, 6.0),
                (4.0, 5.0, 6.0, 7.0),
            ),
            (
                (2.0, 3.0, 4.0, 5.0),
                (3.0, 4.0, 5.0, 6.0),
                (4.0, 5.0, 6.0, 7.0),
                (5.0, 6.0, 7.0, 8.0),
            ),
        ),
        dtype=wp.float32,
        shape=(2, 4, 4),
    )
    assert_np_equal(result.numpy(), expected.numpy())

    result = wp.from_numpy(arr, dtype=wp.vec4, device=device).reshape((8,))  # Reshape from (2, 4)
    expected = wp.array(
        (
            (1.0, 2.0, 3.0, 4.0),
            (2.0, 3.0, 4.0, 5.0),
            (3.0, 4.0, 5.0, 6.0),
            (4.0, 5.0, 6.0, 7.0),
            (2.0, 3.0, 4.0, 5.0),
            (3.0, 4.0, 5.0, 6.0),
            (4.0, 5.0, 6.0, 7.0),
            (5.0, 6.0, 7.0, 8.0),
        ),
        dtype=wp.vec4,
        shape=(8,),
    )
    assert_np_equal(result.numpy(), expected.numpy())

    result = wp.from_numpy(arr, dtype=wp.float32, shape=(32,), device=device)
    expected = wp.array(
        (
            1.0,
            2.0,
            3.0,
            4.0,
            2.0,
            3.0,
            4.0,
            5.0,
            3.0,
            4.0,
            5.0,
            6.0,
            4.0,
            5.0,
            6.0,
            7.0,
            2.0,
            3.0,
            4.0,
            5.0,
            3.0,
            4.0,
            5.0,
            6.0,
            4.0,
            5.0,
            6.0,
            7.0,
            5.0,
            6.0,
            7.0,
            8.0,
        ),
        dtype=wp.float32,
        shape=(32,),
    )
    assert_np_equal(result.numpy(), expected.numpy())


def test_array_aliasing_from_numpy(test, device):
    device = wp.get_device(device)
    assert device.is_cpu

    a_np = np.ones(8, dtype=np.int32)
    a_wp = wp.array(a_np, dtype=int, copy=False, device=device)
    test.assertIs(a_wp._ref, a_np)  # check that some ref is kept to original array
    test.assertEqual(a_wp.ptr, a_np.ctypes.data)

    a_np_2 = a_wp.numpy()
    test.assertTrue((a_np_2 == 1).all())

    # updating source array should update aliased array
    a_np.fill(2)
    test.assertTrue((a_np_2 == 2).all())

    # trying to alias from a different type should do a copy
    # do it twice to check that the copy buffer is not being reused for different arrays

    b_np = np.ones(8, dtype=np.int64)
    c_np = np.zeros(8, dtype=np.int64)
    b_wp = wp.array(b_np, dtype=int, copy=False, device=device)
    c_wp = wp.array(c_np, dtype=int, copy=False, device=device)

    test.assertNotEqual(b_wp.ptr, b_np.ctypes.data)
    test.assertNotEqual(b_wp.ptr, c_wp.ptr)

    b_np_2 = b_wp.numpy()
    c_np_2 = c_wp.numpy()
    test.assertTrue((b_np_2 == 1).all())
    test.assertTrue((c_np_2 == 0).all())


def test_array_from_cai(test, device):
    import torch

    @wp.kernel
    def first_row_plus_one(x: wp.array2d(dtype=float)):
        i, j = wp.tid()
        if i == 0:
            x[i, j] += 1.0

    # start with torch tensor
    arr = torch.zeros((3, 3))
    torch_device = wp.device_to_torch(device)
    arr_torch = arr.to(torch_device)

    # wrap as warp array via __cuda_array_interface__
    arr_warp = wp.array(arr_torch, device=device)

    wp.launch(kernel=first_row_plus_one, dim=(3, 3), inputs=[arr_warp], device=device)

    # re-wrap as torch array
    arr_torch = wp.to_torch(arr_warp)

    # transpose
    arr_torch = torch.as_strided(arr_torch, size=(3, 3), stride=(arr_torch.stride(1), arr_torch.stride(0)))

    # re-wrap as warp array with new strides
    arr_warp = wp.array(arr_torch, device=device)

    wp.launch(kernel=first_row_plus_one, dim=(3, 3), inputs=[arr_warp], device=device)

    assert_np_equal(arr_warp.numpy(), np.array([[2, 1, 1], [1, 0, 0], [1, 0, 0]]))


@wp.kernel
def inplace_add_1d(x: wp.array(dtype=float), y: wp.array(dtype=float)):
    i = wp.tid()
    x[i] += y[i]


@wp.kernel
def inplace_add_2d(x: wp.array2d(dtype=float), y: wp.array2d(dtype=float)):
    i, j = wp.tid()
    x[i, j] += y[i, j]


@wp.kernel
def inplace_add_3d(x: wp.array3d(dtype=float), y: wp.array3d(dtype=float)):
    i, j, k = wp.tid()
    x[i, j, k] += y[i, j, k]


@wp.kernel
def inplace_add_4d(x: wp.array4d(dtype=float), y: wp.array4d(dtype=float)):
    i, j, k, l = wp.tid()
    x[i, j, k, l] += y[i, j, k, l]


@wp.kernel
def inplace_sub_1d(x: wp.array(dtype=float), y: wp.array(dtype=float)):
    i = wp.tid()
    x[i] -= y[i]


@wp.kernel
def inplace_sub_2d(x: wp.array2d(dtype=float), y: wp.array2d(dtype=float)):
    i, j = wp.tid()
    x[i, j] -= y[i, j]


@wp.kernel
def inplace_sub_3d(x: wp.array3d(dtype=float), y: wp.array3d(dtype=float)):
    i, j, k = wp.tid()
    x[i, j, k] -= y[i, j, k]


@wp.kernel
def inplace_sub_4d(x: wp.array4d(dtype=float), y: wp.array4d(dtype=float)):
    i, j, k, l = wp.tid()
    x[i, j, k, l] -= y[i, j, k, l]


@wp.kernel
def inplace_add_vecs(x: wp.array(dtype=wp.vec3), y: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    x[i] += y[i]


@wp.kernel
def inplace_add_mats(x: wp.array(dtype=wp.mat33), y: wp.array(dtype=wp.mat33)):
    i = wp.tid()
    x[i] += y[i]


@wp.kernel
def inplace_add_rhs(x: wp.array(dtype=float), y: wp.array(dtype=float), z: wp.array(dtype=float)):
    i = wp.tid()
    a = y[i]
    a += x[i]
    wp.atomic_add(z, 0, a)


vec9 = wp.vec(length=9, dtype=float)


@wp.kernel
def inplace_add_custom_vec(x: wp.array(dtype=vec9), y: wp.array(dtype=vec9)):
    i = wp.tid()
    x[i] += y[i]
    x[i] += y[i]


def test_array_inplace_diff_ops(test, device):
    N = 3
    x1 = wp.ones(N, dtype=float, requires_grad=True, device=device)
    x2 = wp.ones((N, N), dtype=float, requires_grad=True, device=device)
    x3 = wp.ones((N, N, N), dtype=float, requires_grad=True, device=device)
    x4 = wp.ones((N, N, N, N), dtype=float, requires_grad=True, device=device)

    y1 = wp.clone(x1, requires_grad=True, device=device)
    y2 = wp.clone(x2, requires_grad=True, device=device)
    y3 = wp.clone(x3, requires_grad=True, device=device)
    y4 = wp.clone(x4, requires_grad=True, device=device)

    v1 = wp.ones(1, dtype=wp.vec3, requires_grad=True, device=device)
    v2 = wp.clone(v1, requires_grad=True, device=device)

    m1 = wp.ones(1, dtype=wp.mat33, requires_grad=True, device=device)
    m2 = wp.clone(m1, requires_grad=True, device=device)

    x = wp.ones(1, dtype=float, requires_grad=True, device=device)
    y = wp.clone(x, requires_grad=True, device=device)
    z = wp.zeros(1, dtype=float, requires_grad=True, device=device)

    np_ones_1d = np.ones(N, dtype=float)
    np_ones_2d = np.ones((N, N), dtype=float)
    np_ones_3d = np.ones((N, N, N), dtype=float)
    np_ones_4d = np.ones((N, N, N, N), dtype=float)

    np_twos_1d = np.full(N, 2.0, dtype=float)
    np_twos_2d = np.full((N, N), 2.0, dtype=float)
    np_twos_3d = np.full((N, N, N), 2.0, dtype=float)
    np_twos_4d = np.full((N, N, N, N), 2.0, dtype=float)

    tape = wp.Tape()
    with tape:
        wp.launch(inplace_add_1d, N, inputs=[x1, y1], device=device)
        wp.launch(inplace_add_2d, (N, N), inputs=[x2, y2], device=device)
        wp.launch(inplace_add_3d, (N, N, N), inputs=[x3, y3], device=device)
        wp.launch(inplace_add_4d, (N, N, N, N), inputs=[x4, y4], device=device)

    tape.backward(grads={x1: wp.ones_like(x1), x2: wp.ones_like(x2), x3: wp.ones_like(x3), x4: wp.ones_like(x4)})

    assert_np_equal(x1.grad.numpy(), np_ones_1d)
    assert_np_equal(x2.grad.numpy(), np_ones_2d)
    assert_np_equal(x3.grad.numpy(), np_ones_3d)
    assert_np_equal(x4.grad.numpy(), np_ones_4d)

    assert_np_equal(y1.grad.numpy(), np_ones_1d)
    assert_np_equal(y2.grad.numpy(), np_ones_2d)
    assert_np_equal(y3.grad.numpy(), np_ones_3d)
    assert_np_equal(y4.grad.numpy(), np_ones_4d)

    assert_np_equal(x1.numpy(), np_twos_1d)
    assert_np_equal(x2.numpy(), np_twos_2d)
    assert_np_equal(x3.numpy(), np_twos_3d)
    assert_np_equal(x4.numpy(), np_twos_4d)

    x1.grad.zero_()
    x2.grad.zero_()
    x3.grad.zero_()
    x4.grad.zero_()
    tape.reset()

    with tape:
        wp.launch(inplace_sub_1d, N, inputs=[x1, y1], device=device)
        wp.launch(inplace_sub_2d, (N, N), inputs=[x2, y2], device=device)
        wp.launch(inplace_sub_3d, (N, N, N), inputs=[x3, y3], device=device)
        wp.launch(inplace_sub_4d, (N, N, N, N), inputs=[x4, y4], device=device)

    tape.backward(grads={x1: wp.ones_like(x1), x2: wp.ones_like(x2), x3: wp.ones_like(x3), x4: wp.ones_like(x4)})

    assert_np_equal(x1.grad.numpy(), np_ones_1d)
    assert_np_equal(x2.grad.numpy(), np_ones_2d)
    assert_np_equal(x3.grad.numpy(), np_ones_3d)
    assert_np_equal(x4.grad.numpy(), np_ones_4d)

    assert_np_equal(y1.grad.numpy(), -np_ones_1d)
    assert_np_equal(y2.grad.numpy(), -np_ones_2d)
    assert_np_equal(y3.grad.numpy(), -np_ones_3d)
    assert_np_equal(y4.grad.numpy(), -np_ones_4d)

    assert_np_equal(x1.numpy(), np_ones_1d)
    assert_np_equal(x2.numpy(), np_ones_2d)
    assert_np_equal(x3.numpy(), np_ones_3d)
    assert_np_equal(x4.numpy(), np_ones_4d)

    x1.grad.zero_()
    x2.grad.zero_()
    x3.grad.zero_()
    x4.grad.zero_()
    tape.reset()

    with tape:
        wp.launch(inplace_add_vecs, 1, inputs=[v1, v2], device=device)
        wp.launch(inplace_add_mats, 1, inputs=[m1, m2], device=device)
        wp.launch(inplace_add_rhs, 1, inputs=[x, y, z], device=device)

    tape.backward(loss=z, grads={v1: wp.ones_like(v1, requires_grad=False), m1: wp.ones_like(m1, requires_grad=False)})

    assert_np_equal(v1.numpy(), np.full(shape=(1, 3), fill_value=2.0, dtype=float))
    assert_np_equal(v1.grad.numpy(), np.ones(shape=(1, 3), dtype=float))
    assert_np_equal(v2.grad.numpy(), np.ones(shape=(1, 3), dtype=float))

    assert_np_equal(m1.numpy(), np.full(shape=(1, 3, 3), fill_value=2.0, dtype=float))
    assert_np_equal(m1.grad.numpy(), np.ones(shape=(1, 3, 3), dtype=float))
    assert_np_equal(m2.grad.numpy(), np.ones(shape=(1, 3, 3), dtype=float))

    assert_np_equal(x.grad.numpy(), np.ones(1, dtype=float))
    assert_np_equal(y.grad.numpy(), np.ones(1, dtype=float))
    tape.reset()

    x = wp.zeros(1, dtype=vec9, requires_grad=True, device=device)
    y = wp.ones(1, dtype=vec9, requires_grad=True, device=device)

    with tape:
        wp.launch(inplace_add_custom_vec, 1, inputs=[x, y], device=device)

    tape.backward(grads={x: wp.ones_like(x)})

    assert_np_equal(x.numpy(), np.full((1, 9), 2.0, dtype=float))
    assert_np_equal(y.grad.numpy(), np.full((1, 9), 2.0, dtype=float))


@wp.kernel
def inplace_mul_1d(x: wp.array(dtype=float), y: wp.array(dtype=float)):
    i = wp.tid()
    x[i] *= y[i]


@wp.kernel
def inplace_div_1d(x: wp.array(dtype=float), y: wp.array(dtype=float)):
    i = wp.tid()
    x[i] /= y[i]


@wp.kernel
def inplace_add_non_atomic_types(x: wp.array(dtype=Any), y: wp.array(dtype=Any)):
    i = wp.tid()
    x[i] += y[i]


uint16vec3 = wp.vec(length=3, dtype=wp.uint16)


def test_array_inplace_non_diff_ops(test, device):
    N = 3
    x1 = wp.full(N, value=10.0, dtype=float, device=device)
    y1 = wp.full(N, value=5.0, dtype=float, device=device)

    wp.launch(inplace_mul_1d, N, inputs=[x1, y1], device=device)
    assert_np_equal(x1.numpy(), np.full(N, fill_value=50.0, dtype=float))

    x1.fill_(10.0)
    y1.fill_(5.0)
    wp.launch(inplace_div_1d, N, inputs=[x1, y1], device=device)
    assert_np_equal(x1.numpy(), np.full(N, fill_value=2.0, dtype=float))

    for dtype in wp.types.non_atomic_types + (wp.vec2b, wp.vec2ub, wp.vec2s, wp.vec2us, uint16vec3):
        x = wp.full(N, value=0, dtype=dtype, device=device)
        y = wp.full(N, value=1, dtype=dtype, device=device)

        wp.launch(inplace_add_non_atomic_types, N, inputs=[x, y], device=device)
        assert_np_equal(x.numpy(), y.numpy())


@wp.kernel
def inc_scalar(a: wp.array(dtype=float)):
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


def test_direct_from_numpy(test, device):
    """Pass NumPy arrays to Warp kernels directly"""

    n = 12

    s = np.arange(n, dtype=np.float32)
    v = np.arange(n, dtype=np.float32).reshape((n // 3, 3))
    m = np.arange(n, dtype=np.float32).reshape((n // 4, 2, 2))

    wp.launch(inc_scalar, dim=n, inputs=[s], device=device)
    wp.launch(inc_vector, dim=n // 3, inputs=[v], device=device)
    wp.launch(inc_matrix, dim=n // 4, inputs=[m], device=device)

    expected = np.arange(1, n + 1, dtype=np.float32)

    assert_np_equal(s, expected)
    assert_np_equal(v.reshape(n), expected)
    assert_np_equal(m.reshape(n), expected)


@wp.kernel
def kernel_array_from_ptr(
    ptr: wp.uint64,
):
    arr = wp.array(ptr=ptr, shape=(2, 3), dtype=wp.float32)
    arr[0, 0] = 1.0
    arr[0, 1] = 2.0
    arr[0, 2] = 3.0


def test_kernel_array_from_ptr(test, device):
    arr = wp.zeros(shape=(2, 3), dtype=wp.float32, device=device)
    wp.launch(kernel_array_from_ptr, dim=(1,), inputs=(arr.ptr,), device=device)
    assert_np_equal(arr.numpy(), np.array(((1.0, 2.0, 3.0), (0.0, 0.0, 0.0))))


def test_array_from_int32_domain(test, device):
    wp.zeros(np.array([1504, 1080, 520], dtype=np.int32), dtype=wp.float32, device=device)


def test_array_from_int64_domain(test, device):
    wp.zeros(np.array([1504, 1080, 520], dtype=np.int64), dtype=wp.float32, device=device)


def test_numpy_array_interface(test, device):
    # We should be able to convert between NumPy and Warp arrays using __array_interface__ on CPU.
    # This tests all scalar types supported by both.

    n = 10

    scalar_types = wp.types.scalar_types

    for dtype in scalar_types:
        # test round trip
        a1 = wp.zeros(n, dtype=dtype, device="cpu")
        na = np.array(a1)
        a2 = wp.array(na, device="cpu")

        assert a1.dtype == a2.dtype
        assert a1.shape == a2.shape
        assert a1.strides == a2.strides


@wp.kernel
def kernel_indexing_types(
    arr_1d: wp.array(dtype=wp.int32, ndim=1),
    arr_2d: wp.array(dtype=wp.int32, ndim=2),
    arr_3d: wp.array(dtype=wp.int32, ndim=3),
    arr_4d: wp.array(dtype=wp.int32, ndim=4),
):
    x = arr_1d[wp.uint8(0)]
    y = arr_1d[wp.int16(1)]
    z = arr_1d[wp.uint32(2)]
    w = arr_1d[wp.int64(3)]

    x = arr_2d[wp.uint8(0), wp.uint8(0)]
    y = arr_2d[wp.int16(1), wp.int16(1)]
    z = arr_2d[wp.uint32(2), wp.uint32(2)]
    w = arr_2d[wp.int64(3), wp.int64(3)]

    x = arr_3d[wp.uint8(0), wp.uint8(0), wp.uint8(0)]
    y = arr_3d[wp.int16(1), wp.int16(1), wp.int16(1)]
    z = arr_3d[wp.uint32(2), wp.uint32(2), wp.uint32(2)]
    w = arr_3d[wp.int64(3), wp.int64(3), wp.int64(3)]

    x = arr_4d[wp.uint8(0), wp.uint8(0), wp.uint8(0), wp.uint8(0)]
    y = arr_4d[wp.int16(1), wp.int16(1), wp.int16(1), wp.int16(1)]
    z = arr_4d[wp.uint32(2), wp.uint32(2), wp.uint32(2), wp.uint32(2)]
    w = arr_4d[wp.int64(3), wp.int64(3), wp.int64(3), wp.int64(3)]

    arr_1d[wp.uint8(0)] = 123
    arr_1d[wp.int16(1)] = 123
    arr_1d[wp.uint32(2)] = 123
    arr_1d[wp.int64(3)] = 123

    arr_2d[wp.uint8(0), wp.uint8(0)] = 123
    arr_2d[wp.int16(1), wp.int16(1)] = 123
    arr_2d[wp.uint32(2), wp.uint32(2)] = 123
    arr_2d[wp.int64(3), wp.int64(3)] = 123

    arr_3d[wp.uint8(0), wp.uint8(0), wp.uint8(0)] = 123
    arr_3d[wp.int16(1), wp.int16(1), wp.int16(1)] = 123
    arr_3d[wp.uint32(2), wp.uint32(2), wp.uint32(2)] = 123
    arr_3d[wp.int64(3), wp.int64(3), wp.int64(3)] = 123

    arr_4d[wp.uint8(0), wp.uint8(0), wp.uint8(0), wp.uint8(0)] = 123
    arr_4d[wp.int16(1), wp.int16(1), wp.int16(1), wp.int16(1)] = 123
    arr_4d[wp.uint32(2), wp.uint32(2), wp.uint32(2), wp.uint32(2)] = 123
    arr_4d[wp.int64(3), wp.int64(3), wp.int64(3), wp.int64(3)] = 123

    wp.atomic_add(arr_1d, wp.uint8(0), 123)
    wp.atomic_sub(arr_1d, wp.int16(1), 123)
    wp.atomic_min(arr_1d, wp.uint32(2), 123)
    wp.atomic_max(arr_1d, wp.int64(3), 123)

    wp.atomic_add(arr_2d, wp.uint8(0), wp.uint8(0), 123)
    wp.atomic_sub(arr_2d, wp.int16(1), wp.int16(1), 123)
    wp.atomic_min(arr_2d, wp.uint32(2), wp.uint32(2), 123)
    wp.atomic_max(arr_2d, wp.int64(3), wp.int64(3), 123)

    wp.atomic_add(arr_3d, wp.uint8(0), wp.uint8(0), wp.uint8(0), 123)
    wp.atomic_sub(arr_3d, wp.int16(1), wp.int16(1), wp.int16(1), 123)
    wp.atomic_min(arr_3d, wp.uint32(2), wp.uint32(2), wp.uint32(2), 123)
    wp.atomic_max(arr_3d, wp.int64(3), wp.int64(3), wp.int64(3), 123)

    wp.atomic_add(arr_4d, wp.uint8(0), wp.uint8(0), wp.uint8(0), wp.uint8(0), 123)
    wp.atomic_sub(arr_4d, wp.int16(1), wp.int16(1), wp.int16(1), wp.int16(1), 123)
    wp.atomic_min(arr_4d, wp.uint32(2), wp.uint32(2), wp.uint32(2), wp.uint32(2), 123)
    wp.atomic_max(arr_4d, wp.int64(3), wp.int64(3), wp.int64(3), wp.int64(3), 123)


def test_indexing_types(test, device):
    arr_1d = wp.zeros(shape=(4,), dtype=wp.int32, device=device)
    arr_2d = wp.zeros(shape=(4, 4), dtype=wp.int32, device=device)
    arr_3d = wp.zeros(shape=(4, 4, 4), dtype=wp.int32, device=device)
    arr_4d = wp.zeros(shape=(4, 4, 4, 4), dtype=wp.int32, device=device)
    wp.launch(
        kernel=kernel_indexing_types,
        dim=1,
        inputs=(arr_1d, arr_2d, arr_3d, arr_4d),
        device=device,
    )


def test_alloc_strides(test, device):
    def test_transposed(shape, dtype):
        # allocate without specifying strides
        a1 = wp.zeros(shape, dtype=dtype)

        # allocate with contiguous strides
        strides = wp.types.strides_from_shape(shape, dtype)
        a2 = wp.zeros(shape, dtype=dtype, strides=strides)

        # allocate with transposed (reversed) shape/strides
        rshape = shape[::-1]
        rstrides = strides[::-1]
        a3 = wp.zeros(rshape, dtype=dtype, strides=rstrides)

        # ensure that correct capacity was allocated
        assert a2.capacity == a1.capacity
        assert a3.capacity == a1.capacity

    with wp.ScopedDevice(device):
        shapes = [(5, 5), (5, 3), (3, 5), (2, 3, 4), (4, 2, 3), (3, 2, 4)]
        for shape in shapes:
            with test.subTest(msg=f"shape={shape}"):
                test_transposed(shape, wp.int8)
                test_transposed(shape, wp.float32)
                test_transposed(shape, wp.vec3)


def test_casting(test, device):
    idxs = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
    idxs = wp.array(idxs, device=device).reshape((-1, 3))
    idxs = wp.array(idxs, shape=idxs.shape[0], dtype=wp.vec3i, device=device)
    assert idxs.dtype is wp.vec3i
    assert idxs.shape == (4,)
    assert idxs.strides == (12,)


@wp.kernel
def array_len_kernel(
    a1: wp.array(dtype=int),
    a2: wp.array(dtype=float, ndim=3),
    out: wp.array(dtype=int),
):
    length = len(a1)
    wp.expect_eq(len(a1), 123)
    out[0] = len(a1)

    length = len(a2)
    wp.expect_eq(len(a2), 2)
    out[1] = len(a2)


def test_array_len(test, device):
    a1 = wp.zeros(123, dtype=int, device=device)
    a2 = wp.zeros((2, 3, 4), dtype=float, device=device)
    out = wp.empty(2, dtype=int, device=device)
    wp.launch(
        array_len_kernel,
        dim=(1,),
        inputs=(
            a1,
            a2,
        ),
        outputs=(out,),
        device=device,
    )

    test.assertEqual(out.numpy()[0], 123)
    test.assertEqual(out.numpy()[1], 2)


def test_cuda_interface_conversion(test, device):
    class MyArrayInterface:
        def __init__(self, data):
            self.data = np.array(data)
            self.__array_interface__ = self.data.__array_interface__
            self.__cuda_array_interface__ = self.data.__array_interface__
            self.__len__ = self.data.__len__

    array = MyArrayInterface((1, 2, 3))
    wp_array = wp.array(array, dtype=wp.int8, device=device)
    assert wp_array.ptr != 0

    array = MyArrayInterface((1, 2, 3))
    wp_array = wp.array(array, dtype=wp.float32, device=device)
    assert wp_array.ptr != 0

    array = MyArrayInterface((1, 2, 3))
    wp_array = wp.array(array, dtype=wp.vec3, device=device)
    assert wp_array.ptr != 0

    array = MyArrayInterface((1, 2, 3, 4))
    wp_array = wp.array(array, dtype=wp.mat22, device=device)
    assert wp_array.ptr != 0


devices = get_test_devices()


class TestArray(unittest.TestCase):
    def test_array_new_del(self):
        # test the scenario in which an array instance is created but not initialized before gc
        instance = wp.array.__new__(wp.array)
        instance.__del__()


add_function_test(TestArray, "test_shape", test_shape, devices=devices)
add_function_test(TestArray, "test_negative_shape", test_negative_shape, devices=devices)
add_function_test(TestArray, "test_flatten", test_flatten, devices=devices)
add_function_test(TestArray, "test_reshape", test_reshape, devices=devices)
add_function_test(TestArray, "test_slicing", test_slicing, devices=devices)
add_function_test(TestArray, "test_transpose", test_transpose, devices=devices)
add_function_test(TestArray, "test_view", test_view, devices=devices)
add_function_test(TestArray, "test_clone_adjoint", test_clone_adjoint, devices=devices)
add_function_test(TestArray, "test_assign_adjoint", test_assign_adjoint, devices=devices)

add_function_test(TestArray, "test_1d_array", test_1d, devices=devices)
add_function_test(TestArray, "test_2d_array", test_2d, devices=devices)
add_function_test(TestArray, "test_3d_array", test_3d, devices=devices)
add_function_test(TestArray, "test_4d_array", test_4d, devices=devices)
add_function_test(TestArray, "test_4d_array_transposed", test_4d_transposed, devices=devices)

add_function_test(TestArray, "test_fill_scalar", test_fill_scalar, devices=devices)
add_function_test(TestArray, "test_fill_vector", test_fill_vector, devices=devices)
add_function_test(TestArray, "test_fill_matrix", test_fill_matrix, devices=devices)
add_function_test(TestArray, "test_fill_struct", test_fill_struct, devices=devices)
add_function_test(TestArray, "test_fill_slices", test_fill_slices, devices=devices)
add_function_test(TestArray, "test_full_scalar", test_full_scalar, devices=devices)
add_function_test(TestArray, "test_full_vector", test_full_vector, devices=devices)
add_function_test(TestArray, "test_full_matrix", test_full_matrix, devices=devices)
add_function_test(TestArray, "test_full_struct", test_full_struct, devices=devices)
add_function_test(TestArray, "test_ones_scalar", test_ones_scalar, devices=devices)
add_function_test(TestArray, "test_ones_vector", test_ones_vector, devices=devices)
add_function_test(TestArray, "test_ones_matrix", test_ones_matrix, devices=devices)
add_function_test(TestArray, "test_ones_like_scalar", test_ones_like_scalar, devices=devices)
add_function_test(TestArray, "test_ones_like_vector", test_ones_like_vector, devices=devices)
add_function_test(TestArray, "test_ones_like_matrix", test_ones_like_matrix, devices=devices)
add_function_test(TestArray, "test_empty_array", test_empty_array, devices=devices)
add_function_test(TestArray, "test_empty_from_numpy", test_empty_from_numpy, devices=devices)
add_function_test(TestArray, "test_empty_from_list", test_empty_from_list, devices=devices)
add_function_test(TestArray, "test_to_list_scalar", test_to_list_scalar, devices=devices)
add_function_test(TestArray, "test_to_list_vector", test_to_list_vector, devices=devices)
add_function_test(TestArray, "test_to_list_matrix", test_to_list_matrix, devices=devices)
add_function_test(TestArray, "test_to_list_struct", test_to_list_struct, devices=devices)

add_function_test(TestArray, "test_lower_bound", test_lower_bound, devices=devices)
add_function_test(TestArray, "test_round_trip", test_round_trip, devices=devices)
add_function_test(TestArray, "test_array_to_bool", test_array_to_bool, devices=devices)
add_function_test(TestArray, "test_array_of_structs", test_array_of_structs, devices=devices)
add_function_test(TestArray, "test_array_of_structs_grad", test_array_of_structs_grad, devices=devices)
add_function_test(TestArray, "test_array_of_structs_from_numpy", test_array_of_structs_from_numpy, devices=devices)
add_function_test(TestArray, "test_array_of_structs_roundtrip", test_array_of_structs_roundtrip, devices=devices)
add_function_test(TestArray, "test_array_from_numpy", test_array_from_numpy, devices=devices)
add_function_test(TestArray, "test_array_aliasing_from_numpy", test_array_aliasing_from_numpy, devices=["cpu"])
add_function_test(TestArray, "test_numpy_array_interface", test_numpy_array_interface, devices=["cpu"])

add_function_test(TestArray, "test_array_inplace_diff_ops", test_array_inplace_diff_ops, devices=devices)
add_function_test(TestArray, "test_array_inplace_non_diff_ops", test_array_inplace_non_diff_ops, devices=devices)
add_function_test(TestArray, "test_direct_from_numpy", test_direct_from_numpy, devices=["cpu"])
add_function_test(TestArray, "test_kernel_array_from_ptr", test_kernel_array_from_ptr, devices=devices)

add_function_test(TestArray, "test_array_from_int32_domain", test_array_from_int32_domain, devices=devices)
add_function_test(TestArray, "test_array_from_int64_domain", test_array_from_int64_domain, devices=devices)
add_function_test(TestArray, "test_indexing_types", test_indexing_types, devices=devices)

add_function_test(TestArray, "test_alloc_strides", test_alloc_strides, devices=devices)
add_function_test(TestArray, "test_casting", test_casting, devices=devices)
add_function_test(TestArray, "test_array_len", test_array_len, devices=devices)
add_function_test(TestArray, "test_cuda_interface_conversion", test_cuda_interface_conversion, devices=devices)

try:
    import torch

    # check which Warp devices work with Torch
    # CUDA devices may fail if Torch was not compiled with CUDA support
    torch_compatible_devices = []
    torch_compatible_cuda_devices = []

    for d in devices:
        try:
            t = torch.arange(10, device=wp.device_to_torch(d))
            t += 1
            torch_compatible_devices.append(d)
            if d.is_cuda:
                torch_compatible_cuda_devices.append(d)
        except Exception as e:
            print(f"Skipping Array tests that use Torch on device '{d}' due to exception: {e}")

    add_function_test(TestArray, "test_array_from_cai", test_array_from_cai, devices=torch_compatible_cuda_devices)

except Exception as e:
    print(f"Skipping Array tests that use Torch due to exception: {e}")


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
