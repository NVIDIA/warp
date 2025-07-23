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

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

np_float_types = [np.float16, np.float32, np.float64]

kernel_cache = {}


def getkernel(func, suffix=""):
    key = func.__name__ + "_" + suffix
    if key not in kernel_cache:
        kernel_cache[key] = wp.Kernel(func=func, key=key)
    return kernel_cache[key]


def test_anon_constructor_error_length_mismatch(test, device):
    @wp.kernel
    def kernel():
        wp.vector(wp.vector(length=2, dtype=float), length=3, dtype=float)

    with test.assertRaisesRegex(
        RuntimeError,
        r"incompatible vector of length 3 given when copy constructing a vector of length 2$",
    ):
        wp.launch(kernel, dim=1, inputs=[], device=device)


def test_anon_constructor_error_numeric_arg_missing(test, device):
    @wp.kernel
    def kernel():
        wp.vector(1.0, 2.0, length=12345)

    with test.assertRaisesRegex(
        RuntimeError,
        r"incompatible number of values given \(2\) when constructing a vector of length 12345$",
    ):
        wp.launch(kernel, dim=1, inputs=[], device=device)


def test_anon_constructor_error_length_arg_missing(test, device):
    @wp.kernel
    def kernel():
        wp.vector()

    with test.assertRaisesRegex(
        RuntimeError,
        r"the `length` argument must be specified when zero-initializing a vector$",
    ):
        wp.launch(kernel, dim=1, inputs=[], device=device)


def test_anon_constructor_error_numeric_args_mismatch(test, device):
    @wp.kernel
    def kernel():
        wp.vector(1.0, 2)

    with test.assertRaisesRegex(
        RuntimeError,
        r"all values given when constructing a vector must have the same type$",
    ):
        wp.launch(kernel, dim=1, inputs=[], device=device)


def test_tpl_constructor_error_incompatible_sizes(test, device):
    @wp.kernel
    def kernel():
        wp.vec3(wp.vec2(1.0, 2.0))

    with test.assertRaisesRegex(
        RuntimeError, "incompatible vector of length 3 given when copy constructing a vector of length 2"
    ):
        wp.launch(kernel, dim=1, inputs=[], device=device)


def test_tpl_constructor_error_numeric_args_mismatch(test, device):
    @wp.kernel
    def kernel():
        wp.vec2(1.0, 2)

    with test.assertRaisesRegex(
        RuntimeError,
        r"all values given when constructing a vector must have the same type$",
    ):
        wp.launch(kernel, dim=1, inputs=[], device=device)


def test_casting_constructors(test, device, dtype, register_kernels=False):
    np_type = np.dtype(dtype)
    wp_type = wp.types.np_dtype_to_warp_type[np_type]
    vec3 = wp.types.vector(length=3, dtype=wp_type)

    np16 = np.dtype(np.float16)
    wp16 = wp.types.np_dtype_to_warp_type[np16]

    np32 = np.dtype(np.float32)
    wp32 = wp.types.np_dtype_to_warp_type[np32]

    np64 = np.dtype(np.float64)
    wp64 = wp.types.np_dtype_to_warp_type[np64]

    def cast_float16(a: wp.array(dtype=wp_type, ndim=2), b: wp.array(dtype=wp16, ndim=2)):
        tid = wp.tid()

        v1 = vec3(a[tid, 0], a[tid, 1], a[tid, 2])
        v2 = wp.vector(v1, dtype=wp16)

        b[tid, 0] = v2[0]
        b[tid, 1] = v2[1]
        b[tid, 2] = v2[2]

    def cast_float32(a: wp.array(dtype=wp_type, ndim=2), b: wp.array(dtype=wp32, ndim=2)):
        tid = wp.tid()

        v1 = vec3(a[tid, 0], a[tid, 1], a[tid, 2])
        v2 = wp.vector(v1, dtype=wp32)

        b[tid, 0] = v2[0]
        b[tid, 1] = v2[1]
        b[tid, 2] = v2[2]

    def cast_float64(a: wp.array(dtype=wp_type, ndim=2), b: wp.array(dtype=wp64, ndim=2)):
        tid = wp.tid()

        v1 = vec3(a[tid, 0], a[tid, 1], a[tid, 2])
        v2 = wp.vector(v1, dtype=wp64)

        b[tid, 0] = v2[0]
        b[tid, 1] = v2[1]
        b[tid, 2] = v2[2]

    kernel_16 = getkernel(cast_float16, suffix=dtype.__name__)
    kernel_32 = getkernel(cast_float32, suffix=dtype.__name__)
    kernel_64 = getkernel(cast_float64, suffix=dtype.__name__)

    if register_kernels:
        return

    # check casting to float 16
    a = wp.array(np.ones((1, 3), dtype=np_type), dtype=wp_type, requires_grad=True, device=device)
    b = wp.array(np.zeros((1, 3), dtype=np16), dtype=wp16, requires_grad=True, device=device)
    b_result = np.ones((1, 3), dtype=np16)
    b_grad = wp.array(np.ones((1, 3), dtype=np16), dtype=wp16, device=device)
    a_grad = wp.array(np.ones((1, 3), dtype=np_type), dtype=wp_type, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(kernel=kernel_16, dim=1, inputs=[a, b], device=device)

    tape.backward(grads={b: b_grad})
    out = tape.gradients[a].numpy()

    assert_np_equal(b.numpy(), b_result)
    assert_np_equal(out, a_grad.numpy())

    # check casting to float 32
    a = wp.array(np.ones((1, 3), dtype=np_type), dtype=wp_type, requires_grad=True, device=device)
    b = wp.array(np.zeros((1, 3), dtype=np32), dtype=wp32, requires_grad=True, device=device)
    b_result = np.ones((1, 3), dtype=np32)
    b_grad = wp.array(np.ones((1, 3), dtype=np32), dtype=wp32, device=device)
    a_grad = wp.array(np.ones((1, 3), dtype=np_type), dtype=wp_type, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(kernel=kernel_32, dim=1, inputs=[a, b], device=device)

    tape.backward(grads={b: b_grad})
    out = tape.gradients[a].numpy()

    assert_np_equal(b.numpy(), b_result)
    assert_np_equal(out, a_grad.numpy())

    # check casting to float 64
    a = wp.array(np.ones((1, 3), dtype=np_type), dtype=wp_type, requires_grad=True, device=device)
    b = wp.array(np.zeros((1, 3), dtype=np64), dtype=wp64, requires_grad=True, device=device)
    b_result = np.ones((1, 3), dtype=np64)
    b_grad = wp.array(np.ones((1, 3), dtype=np64), dtype=wp64, device=device)
    a_grad = wp.array(np.ones((1, 3), dtype=np_type), dtype=wp_type, device=device)

    tape = wp.Tape()
    with tape:
        wp.launch(kernel=kernel_64, dim=1, inputs=[a, b], device=device)

    tape.backward(grads={b: b_grad})
    out = tape.gradients[a].numpy()

    assert_np_equal(b.numpy(), b_result)
    assert_np_equal(out, a_grad.numpy())


@wp.kernel
def test_vector_constructors_value_func():
    a = wp.vec2()
    b = wp.vector(a, dtype=wp.float16)
    c = wp.vector(a)
    d = wp.vector(a, length=2)
    e = wp.vector(1.0, 2.0, 3.0, dtype=float)


# Test matrix constructors using explicit type (float16)
# note that these tests are specifically not using generics / closure
# args to create kernels dynamically (like the rest of this file)
# as those use different code paths to resolve arg types which
# has lead to regressions.
@wp.kernel
def test_vector_constructors_explicit_precision():
    # construction for custom matrix types
    ones = wp.vector(wp.float16(1.0), length=2)
    zeros = wp.vector(length=2, dtype=wp.float16)
    custom = wp.vector(wp.float16(0.0), wp.float16(1.0))

    for i in range(2):
        wp.expect_eq(ones[i], wp.float16(1.0))
        wp.expect_eq(zeros[i], wp.float16(0.0))
        wp.expect_eq(custom[i], wp.float16(i))


# Same as above but with a default (float/int) type
# which tests some different code paths that
# need to ensure types are correctly canonicalized
# during codegen
@wp.kernel
def test_vector_constructors_default_precision():
    # construction for custom matrix types
    ones = wp.vector(1.0, length=2)
    zeros = wp.vector(length=2, dtype=float)
    custom = wp.vector(0.0, 1.0)

    for i in range(2):
        wp.expect_eq(ones[i], 1.0)
        wp.expect_eq(zeros[i], 0.0)
        wp.expect_eq(custom[i], float(i))


CONSTANT_LENGTH = wp.constant(10)


# tests that we can use global constants in length keyword argument
# for vector constructor
@wp.kernel
def test_vector_constructors_constant_length():
    v = wp.vector(length=(CONSTANT_LENGTH), dtype=float)

    for i in range(CONSTANT_LENGTH):
        v[i] = float(i)


devices = get_test_devices()


class TestVecConstructors(unittest.TestCase):
    pass


add_function_test(
    TestVecConstructors,
    "test_anon_constructor_error_length_mismatch",
    test_anon_constructor_error_length_mismatch,
    devices=devices,
)
add_function_test(
    TestVecConstructors,
    "test_anon_constructor_error_numeric_arg_missing",
    test_anon_constructor_error_numeric_arg_missing,
    devices=devices,
)
add_function_test(
    TestVecConstructors,
    "test_anon_constructor_error_length_arg_missing",
    test_anon_constructor_error_length_arg_missing,
    devices=devices,
)
add_function_test(
    TestVecConstructors,
    "test_anon_constructor_error_numeric_args_mismatch",
    test_anon_constructor_error_numeric_args_mismatch,
    devices=devices,
)
add_function_test(
    TestVecConstructors,
    "test_tpl_constructor_error_incompatible_sizes",
    test_tpl_constructor_error_incompatible_sizes,
    devices=devices,
)
add_function_test(
    TestVecConstructors,
    "test_tpl_constructor_error_numeric_args_mismatch",
    test_tpl_constructor_error_numeric_args_mismatch,
    devices=devices,
)
add_kernel_test(TestVecConstructors, test_vector_constructors_value_func, dim=1, devices=devices)
add_kernel_test(TestVecConstructors, test_vector_constructors_explicit_precision, dim=1, devices=devices)
add_kernel_test(TestVecConstructors, test_vector_constructors_default_precision, dim=1, devices=devices)
add_kernel_test(TestVecConstructors, test_vector_constructors_constant_length, dim=1, devices=devices)

for dtype in np_float_types:
    add_function_test_register_kernel(
        TestVecConstructors,
        f"test_casting_constructors_{dtype.__name__}",
        test_casting_constructors,
        devices=devices,
        dtype=dtype,
    )

if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=True)
