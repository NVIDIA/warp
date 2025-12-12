# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import warp as wp
from warp.tests.unittest_utils import *

devices = get_test_devices()


@wp.kernel
def test_zeros():
    arr = wp.zeros(shape=(2, 3), dtype=int)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            arr[i][j] = i * arr.shape[1] + j

    wp.expect_eq(arr[0][0], 0)
    wp.expect_eq(arr[0][1], 1)
    wp.expect_eq(arr[0][2], 2)
    wp.expect_eq(arr[1][0], 3)
    wp.expect_eq(arr[1][1], 4)
    wp.expect_eq(arr[1][2], 5)

    r0 = arr[0]
    wp.expect_eq(r0.shape[0], 3)
    wp.expect_eq(r0.shape[1], 0)
    wp.expect_eq(r0[0], 0)
    wp.expect_eq(r0[1], 1)
    wp.expect_eq(r0[2], 2)

    r1 = arr[1]
    wp.expect_eq(r1.shape[0], 3)
    wp.expect_eq(r1.shape[1], 0)
    wp.expect_eq(r1[0], 3)
    wp.expect_eq(r1[1], 4)
    wp.expect_eq(r1[2], 5)

    s = arr[:2, 1:]
    wp.expect_eq(s.shape[0], 2)
    wp.expect_eq(s.shape[1], 2)
    wp.expect_eq(s.shape[2], 0)
    wp.expect_eq(s[0, 0], 1)
    wp.expect_eq(s[0, 1], 2)
    wp.expect_eq(s[1, 0], 4)
    wp.expect_eq(s[1, 1], 5)


@wp.func
def test_func_arg_func(arr: wp.array(ndim=2, dtype=int)):
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            arr[i][j] = i * arr.shape[1] + j


@wp.kernel
def test_func_arg():
    arr = wp.zeros(shape=(2, 3), dtype=int)
    test_func_arg_func(arr)

    wp.expect_eq(arr[0][0], 0)
    wp.expect_eq(arr[0][1], 1)
    wp.expect_eq(arr[0][2], 2)
    wp.expect_eq(arr[1][0], 3)
    wp.expect_eq(arr[1][1], 4)
    wp.expect_eq(arr[1][2], 5)


@wp.func
def test_func_return_func():
    arr = wp.zeros(shape=2, dtype=int)
    for i in range(arr.shape[0]):
        arr[i] = i

    return arr


@wp.kernel
def test_func_return():
    arr = test_func_return_func()

    wp.expect_eq(arr[0], 0)
    wp.expect_eq(arr[1], 1)


@wp.func
def test_func_return_annotation_func() -> wp.fixedarray(shape=(2, 3), dtype=int):
    arr = wp.zeros(shape=(2, 3), dtype=int)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            arr[i][j] = i * arr.shape[1] + j

    return arr


@wp.kernel
def test_func_return_annotation():
    arr = test_func_return_annotation_func()

    wp.expect_eq(arr[0][0], 0)
    wp.expect_eq(arr[0][1], 1)
    wp.expect_eq(arr[0][2], 2)
    wp.expect_eq(arr[1][0], 3)
    wp.expect_eq(arr[1][1], 4)
    wp.expect_eq(arr[1][2], 5)


def test_error_invalid_func_return_annotation(test, device):
    @wp.func
    def func() -> wp.array(ndim=2, dtype=int):
        arr = wp.zeros(shape=(2, 3), dtype=int)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                arr[i][j] = i * arr.shape[1] + j

        return arr

    @wp.kernel
    def kernel():
        arr = func()

    with test.assertRaisesRegex(
        wp.WarpCodegenError,
        r"The function `func` returns a fixed-size array whereas it has its return type annotated as `Array\[int32\]`.$",
    ):
        wp.launch(kernel, 1, device=device)


def test_error_runtime_shape(test, device):
    @wp.kernel
    def kernel():
        tid = wp.tid()
        wp.zeros(shape=(tid,), dtype=int)

    with test.assertRaisesRegex(
        RuntimeError,
        r"the `shape` argument must be specified as a constant when zero-initializing an array$",
    ):
        wp.launch(kernel, 1, device=device)


@wp.kernel
def test_capture_if_kernel():
    arr = wp.zeros(shape=(2, 3), dtype=int)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            arr[i][j] = i * arr.shape[1] + j

    wp.expect_eq(arr[0][0], 0)
    wp.expect_eq(arr[0][1], 1)
    wp.expect_eq(arr[0][2], 2)
    wp.expect_eq(arr[1][0], 3)
    wp.expect_eq(arr[1][1], 4)
    wp.expect_eq(arr[1][2], 5)


def test_capture_if(test, device):
    if (
        not wp.get_device(device).is_cuda
        or wp._src.context.runtime.toolkit_version < (12, 4)
        or wp._src.context.runtime.driver_version < (12, 4)
    ):
        return

    def foo():
        wp.launch(test_capture_if_kernel, dim=512, block_dim=128, device=device)

    cond = wp.ones(1, dtype=wp.int32, device=device)
    with wp.ScopedCapture(device=device) as capture:
        wp.capture_if(condition=cond, on_true=foo)

    wp.capture_launch(capture.graph)


@wp.struct
class test_func_struct_MyStruct:
    offset: int
    dist: float


@wp.func
def test_func_struct_func():
    arr = wp.zeros(shape=(2, 3), dtype=test_func_struct_MyStruct)
    count = float(arr.shape[0] * arr.shape[1] - 1)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            arr[i][j].offset = i * arr.shape[1] + j
            arr[i][j].dist = float(arr[i][j].offset) / count

    return arr


@wp.kernel
def test_func_struct():
    arr = test_func_struct_func()

    wp.expect_eq(arr[0][0].offset, 0)
    wp.expect_near(arr[0][0].dist, 0.0)
    wp.expect_eq(arr[0][1].offset, 1)
    wp.expect_near(arr[0][1].dist, 0.2)
    wp.expect_eq(arr[0][2].offset, 2)
    wp.expect_near(arr[0][2].dist, 0.4)
    wp.expect_eq(arr[1][0].offset, 3)
    wp.expect_near(arr[1][0].dist, 0.6)
    wp.expect_eq(arr[1][1].offset, 4)
    wp.expect_near(arr[1][1].dist, 0.8)
    wp.expect_eq(arr[1][2].offset, 5)
    wp.expect_near(arr[1][2].dist, 1.0)


@wp.kernel
def test_fixedarray_ptr_reinterpret(output: wp.array(dtype=wp.vec2)):
    # Allocate a fixedarray inside the kernel
    m_b = wp.zeros(shape=(24,), dtype=wp.float32)

    # Initialize some values
    for i in range(24):
        m_b[i] = float(i)

    # Use .ptr to create a reinterpreted view as vec2
    m_b_2d = wp.array(ptr=m_b.ptr, shape=(12,), dtype=wp.vec2)

    # Read from the reinterpreted array
    for i in range(12):
        output[i] = m_b_2d[i]


def test_fixedarray_ptr(test, device):
    output = wp.zeros(shape=(12,), dtype=wp.vec2, device=device)

    wp.launch(test_fixedarray_ptr_reinterpret, dim=1, inputs=[], outputs=[output], device=device)

    expected = np.array([(i * 2, i * 2 + 1) for i in range(12)], dtype=np.float32)
    assert_np_equal(output.numpy(), expected)


# Test fixedarray polymorphism with array templates (regression test for array type matching)
# This test ensures that fixedarray can match array template parameters even when the template
# has a concrete dtype (not Any), which is essential since fixedarray inherits from array.
@wp.func
def array_template_func_concrete(arr: wp.array(ndim=1, dtype=wp.vec2f)):
    """Function with concrete array template - fixedarray should match"""
    return arr[0]


@wp.func
def array_template_func_generic(arr: wp.array(ndim=1, dtype=Any)):
    """Function with generic array template - fixedarray should match"""
    return arr[0]


@wp.kernel
def test_fixedarray_array_polymorphism():
    """Test that fixedarray matches array templates with both concrete and generic dtypes"""
    # Test with concrete dtype (vec2f)
    fixed_arr_vec2 = wp.zeros(shape=(3,), dtype=wp.vec2f)
    fixed_arr_vec2[0] = wp.vec2f(1.0, 2.0)
    result_vec2 = array_template_func_concrete(fixed_arr_vec2)
    wp.expect_eq(result_vec2, wp.vec2f(1.0, 2.0))

    # Test with generic dtype (Any)
    fixed_arr_int = wp.zeros(shape=(5,), dtype=int)
    fixed_arr_int[0] = 42
    result_int = array_template_func_generic(fixed_arr_int)
    wp.expect_eq(result_int, 42)

    # Test with float32
    fixed_arr_float = wp.zeros(shape=(4,), dtype=float)
    fixed_arr_float[0] = 3.14
    result_float = array_template_func_generic(fixed_arr_float)
    wp.expect_eq(result_float, 3.14)

    # Test with vec3f
    fixed_arr_vec3 = wp.zeros(shape=(2,), dtype=wp.vec3f)
    fixed_arr_vec3[0] = wp.vec3f(1.0, 2.0, 3.0)
    result_vec3 = array_template_func_generic(fixed_arr_vec3)
    wp.expect_eq(result_vec3, wp.vec3f(1.0, 2.0, 3.0))


class TestFixedArray(unittest.TestCase):
    pass


add_kernel_test(TestFixedArray, kernel=test_zeros, name="test_zeros", dim=1, devices=devices)
add_kernel_test(TestFixedArray, kernel=test_func_arg, name="test_func_arg", dim=1, devices=devices)
add_kernel_test(TestFixedArray, kernel=test_func_return, name="test_func_return", dim=1, devices=devices)
add_kernel_test(
    TestFixedArray, kernel=test_func_return_annotation, name="test_func_return_annotation", dim=1, devices=devices
)
add_function_test(
    TestFixedArray,
    "test_error_invalid_func_return_annotation",
    test_error_invalid_func_return_annotation,
    devices=devices,
)
add_function_test(TestFixedArray, "test_error_runtime_shape", test_error_runtime_shape, devices=devices)
add_function_test(TestFixedArray, "test_capture_if", test_capture_if, devices=devices)
add_kernel_test(TestFixedArray, kernel=test_func_struct, name="test_func_struct", dim=1, devices=devices)
add_function_test(TestFixedArray, "test_fixedarray_ptr", test_fixedarray_ptr, devices=devices)
add_kernel_test(
    TestFixedArray,
    kernel=test_fixedarray_array_polymorphism,
    name="test_fixedarray_array_polymorphism",
    dim=1,
    devices=devices,
)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
