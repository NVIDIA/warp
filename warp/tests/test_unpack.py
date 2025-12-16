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

import warp as wp
from warp.tests.unittest_utils import *


@wp.kernel
def test_unpack_array_slice_kernel(arr: wp.array(dtype=int)):
    v1 = wp.vec3i(*arr[:3])
    wp.expect_eq(v1, wp.vec3i(1, 2, 3))

    v2 = wp.vec2i(*arr[0:4:2])
    wp.expect_eq(v2, wp.vec2i(1, 3))

    v3 = wp.vec3i(*arr[2:5])
    wp.expect_eq(v3, wp.vec3i(3, 4, 5))

    x = wp.max(*arr[:2])
    wp.expect_eq(x, 2)


def test_unpack_array_slice(test, device):
    arr = wp.array((1, 2, 3, 4, 5, 6, 7, 8, 9), dtype=int, device=device)
    wp.launch(test_unpack_array_slice_kernel, dim=1, inputs=(arr,), device=device)
    wp.synchronize_device(device)


@wp.kernel
def test_unpack_vec_kernel():
    v = wp.vec2(5.0, 10.0)
    # Unpack vec2 into max function (2 args)
    result = wp.max(*v)
    wp.expect_eq(result, 10.0)

    # Unpack vec2 into vec2 constructor
    v2 = wp.vec2(*v)
    wp.expect_eq(v2, v)


@wp.kernel
def test_unpack_vec_with_extra_args_kernel():
    # Unpack vec3 and add a 4th element to make vec4
    v3 = wp.vec3(1.0, 2.0, 3.0)
    v4 = wp.vec4(*v3, 4.0)
    wp.expect_eq(v4, wp.vec4(1.0, 2.0, 3.0, 4.0))

    # Unpack vec2 and add elements to make vec4
    v2 = wp.vec2(2.0, 3.0)
    v4b = wp.vec4(1.0, *v2, 4.0)
    wp.expect_eq(v4b, wp.vec4(1.0, 2.0, 3.0, 4.0))

    # Unpack vec3i and add integer element
    v3i = wp.vec3i(2, 3, 4)
    v4i = wp.vec4i(1, *v3i)
    wp.expect_eq(v4i, wp.vec4i(1, 2, 3, 4))


@wp.kernel
def test_unpack_multiple_vecs_kernel():
    v2a = wp.vec2(1.0, 2.0)
    v2b = wp.vec2(3.0, 4.0)
    v4 = wp.vec4(*v2a, *v2b)
    wp.expect_eq(v4, wp.vec4(1.0, 2.0, 3.0, 4.0))


@wp.kernel
def test_unpack_vector_slice_kernel():
    v = wp.vec4(1.0, 2.0, 3.0, 4.0)

    # Slice middle elements
    v2 = wp.vec2(*v[1:3])
    wp.expect_eq(v2, wp.vec2(2.0, 3.0))

    # Slice from start
    v3 = wp.vec3(*v[:3])
    wp.expect_eq(v3, wp.vec3(1.0, 2.0, 3.0))

    # Slice to end
    v3b = wp.vec3(*v[1:])
    wp.expect_eq(v3b, wp.vec3(2.0, 3.0, 4.0))

    # Combine slice with extra args
    v3i = wp.vec3i(1, 2, 3)
    v4i = wp.vec4i(0, *v3i[1:3], 4)
    wp.expect_eq(v4i, wp.vec4i(0, 2, 3, 4))


@wp.kernel
def test_unpack_vector_slice_negative_kernel():
    v = wp.vec4(1.0, 2.0, 3.0, 4.0)

    # Negative upper bound
    v2 = wp.vec2(*v[:-2])
    wp.expect_eq(v2, wp.vec2(1.0, 2.0))

    # Negative lower bound
    v2b = wp.vec2(*v[-2:])
    wp.expect_eq(v2b, wp.vec2(3.0, 4.0))


@wp.kernel
def test_unpack_kernel():
    m = wp.mat22(1.0, 2.0, 3.0, 4.0)
    m2 = wp.matrix_from_rows(*m)
    wp.expect_eq(m2, m)


mat23 = wp.types.matrix(shape=(2, 3), dtype=float)


@wp.kernel
def test_unpack_matrix_slice_kernel():
    m = wp.mat33(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)

    # Get first two rows
    m2 = wp.matrix_from_rows(*m[:2])
    expected = mat23(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    wp.expect_eq(m2, expected)

    # Get last two rows
    m3 = wp.matrix_from_rows(*m[1:])
    expected3 = mat23(4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
    wp.expect_eq(m3, expected3)

    # Negative indexing
    m4 = wp.matrix_from_rows(*m[-2:])
    wp.expect_eq(m4, expected3)


@wp.kernel
def test_unpack_quat_to_vec4_kernel():
    q = wp.quat(1.0, 2.0, 3.0, 4.0)
    v = wp.vec4(*q)
    wp.expect_eq(v, wp.vec4(1.0, 2.0, 3.0, 4.0))


@wp.kernel
def test_unpack_quat_to_quat_kernel():
    q = wp.quat(1.0, 2.0, 3.0, 4.0)
    q2 = wp.quat(*q)
    wp.expect_eq(q2, q)


@wp.kernel
def test_unpack_quat_slice_kernel():
    q = wp.quat(1.0, 2.0, 3.0, 4.0)

    # Get xyz components
    v3 = wp.vec3(*q[:3])
    wp.expect_eq(v3, wp.vec3(1.0, 2.0, 3.0))

    # Get last two components (z, w)
    v2 = wp.vec2(*q[2:])
    wp.expect_eq(v2, wp.vec2(3.0, 4.0))

    # Negative indexing
    v2b = wp.vec2(*q[-2:])
    wp.expect_eq(v2b, wp.vec2(3.0, 4.0))


def test_unpack_error_non_constant_bounds(test, device):
    @wp.kernel
    def kernel(arr: wp.array(dtype=int), n: int):
        # n is not compile-time constant
        v = wp.vec3i(*arr[:n])

    with test.assertRaisesRegex(wp.WarpCodegenValueError, "Slice component must be a compile-time constant"):
        wp.launch(kernel, dim=1, inputs=[wp.zeros(10, dtype=int, device=device), 3], device=device)


def test_unpack_error_missing_stop_bound(test, device):
    @wp.kernel
    def kernel(arr: wp.array(dtype=int)):
        v = wp.vec3i(*arr[0:])  # Missing upper bound

    with test.assertRaisesRegex(wp.WarpCodegenValueError, "requires explicit upper bound"):
        wp.launch(kernel, dim=1, inputs=[wp.zeros(10, dtype=int, device=device)], device=device)


def test_unpack_error_unsupported_type(test, device):
    @wp.kernel
    def kernel():
        x = 42

        # int is not unpackable
        v = wp.vec3i(*x)

    with test.assertRaisesRegex(wp.WarpCodegenTypeError, "Starred expressions are only supported for"):
        wp.launch(kernel, dim=1, inputs=[], device=device)


def test_unpack_error_negative_index(test, device):
    @wp.kernel
    def kernel(arr: wp.array(dtype=int)):
        v = wp.vec3i(*arr[:-3])

    with test.assertRaisesRegex(wp.WarpCodegenValueError, "cannot be negative for arrays"):
        wp.launch(kernel, dim=1, inputs=[wp.zeros(10, dtype=int, device=device)], device=device)


def test_unpack_error_negative_step(test, device):
    @wp.kernel
    def kernel(arr: wp.array(dtype=int)):
        v = wp.vec3i(*arr[5:2:-1])

    with test.assertRaisesRegex(wp.WarpCodegenValueError, "step cannot be negative"):
        wp.launch(kernel, dim=1, inputs=[wp.zeros(10, dtype=int, device=device)], device=device)


devices = get_test_devices()


class TestUnpack(unittest.TestCase):
    pass


add_function_test(TestUnpack, "test_unpack_array_slice", test_unpack_array_slice, devices=devices)
add_kernel_test(TestUnpack, test_unpack_vec_kernel, dim=1, devices=devices)
add_kernel_test(TestUnpack, test_unpack_vec_with_extra_args_kernel, dim=1, devices=devices)
add_kernel_test(TestUnpack, test_unpack_multiple_vecs_kernel, dim=1, devices=devices)
add_kernel_test(TestUnpack, test_unpack_vector_slice_kernel, dim=1, devices=devices)
add_kernel_test(TestUnpack, test_unpack_vector_slice_negative_kernel, dim=1, devices=devices)
add_kernel_test(TestUnpack, test_unpack_kernel, dim=1, devices=devices)
add_kernel_test(TestUnpack, test_unpack_matrix_slice_kernel, dim=1, devices=devices)
add_kernel_test(TestUnpack, test_unpack_quat_to_vec4_kernel, dim=1, devices=devices)
add_kernel_test(TestUnpack, test_unpack_quat_to_quat_kernel, dim=1, devices=devices)
add_kernel_test(TestUnpack, test_unpack_quat_slice_kernel, dim=1, devices=devices)
add_function_test(
    TestUnpack, "test_unpack_error_non_constant_bounds", test_unpack_error_non_constant_bounds, devices=devices
)
add_function_test(
    TestUnpack, "test_unpack_error_missing_stop_bound", test_unpack_error_missing_stop_bound, devices=devices
)
add_function_test(TestUnpack, "test_unpack_error_unsupported_type", test_unpack_error_unsupported_type, devices=devices)
add_function_test(TestUnpack, "test_unpack_error_negative_index", test_unpack_error_negative_index, devices=devices)
add_function_test(TestUnpack, "test_unpack_error_negative_step", test_unpack_error_negative_step, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
