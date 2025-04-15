# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import unittest
from typing import Any

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


# basic kernel with one input and output
@wp.kernel
def triple_kernel(input: wp.array(dtype=float), output: wp.array(dtype=float)):
    tid = wp.tid()
    output[tid] = 3.0 * input[tid]


# generic kernel with one scalar input and output
@wp.kernel
def triple_kernel_scalar(input: wp.array(dtype=Any), output: wp.array(dtype=Any)):
    tid = wp.tid()
    output[tid] = input.dtype(3) * input[tid]


# generic kernel with one vector/matrix input and output
@wp.kernel
def triple_kernel_vecmat(input: wp.array(dtype=Any), output: wp.array(dtype=Any)):
    tid = wp.tid()
    output[tid] = input.dtype.dtype(3) * input[tid]


# kernel with multiple inputs and outputs
@wp.kernel
def multiarg_kernel(
    # inputs
    a: wp.array(dtype=float),
    b: wp.array(dtype=float),
    c: wp.array(dtype=float),
    # outputs
    ab: wp.array(dtype=float),
    bc: wp.array(dtype=float),
):
    tid = wp.tid()
    ab[tid] = a[tid] + b[tid]
    bc[tid] = b[tid] + c[tid]


# various types for testing
scalar_types = wp.types.scalar_types
vector_types = []
matrix_types = []
for dim in [2, 3, 4]:
    for T in scalar_types:
        vector_types.append(wp.vec(dim, T))
        matrix_types.append(wp.mat((dim, dim), T))

# explicitly overload generic kernels to avoid module reloading during tests
for T in scalar_types:
    wp.overload(triple_kernel_scalar, [wp.array(dtype=T), wp.array(dtype=T)])
for T in [*vector_types, *matrix_types]:
    wp.overload(triple_kernel_vecmat, [wp.array(dtype=T), wp.array(dtype=T)])


def _jax_version():
    try:
        import jax

        return jax.__version_info__
    except ImportError:
        return (0, 0, 0)


def test_dtype_from_jax(test, device):
    import jax.numpy as jp

    def test_conversions(jax_type, warp_type):
        test.assertEqual(wp.dtype_from_jax(jax_type), warp_type)
        test.assertEqual(wp.dtype_from_jax(jp.dtype(jax_type)), warp_type)

    test_conversions(jp.float16, wp.float16)
    test_conversions(jp.float32, wp.float32)
    test_conversions(jp.float64, wp.float64)
    test_conversions(jp.int8, wp.int8)
    test_conversions(jp.int16, wp.int16)
    test_conversions(jp.int32, wp.int32)
    test_conversions(jp.int64, wp.int64)
    test_conversions(jp.uint8, wp.uint8)
    test_conversions(jp.uint16, wp.uint16)
    test_conversions(jp.uint32, wp.uint32)
    test_conversions(jp.uint64, wp.uint64)
    test_conversions(jp.bool_, wp.bool)


def test_dtype_to_jax(test, device):
    import jax.numpy as jp

    def test_conversions(warp_type, jax_type):
        test.assertEqual(wp.dtype_to_jax(warp_type), jax_type)

    test_conversions(wp.float16, jp.float16)
    test_conversions(wp.float32, jp.float32)
    test_conversions(wp.float64, jp.float64)
    test_conversions(wp.int8, jp.int8)
    test_conversions(wp.int16, jp.int16)
    test_conversions(wp.int32, jp.int32)
    test_conversions(wp.int64, jp.int64)
    test_conversions(wp.uint8, jp.uint8)
    test_conversions(wp.uint16, jp.uint16)
    test_conversions(wp.uint32, jp.uint32)
    test_conversions(wp.uint64, jp.uint64)
    test_conversions(wp.bool, jp.bool_)


def test_device_conversion(test, device):
    jax_device = wp.device_to_jax(device)
    warp_device = wp.device_from_jax(jax_device)
    test.assertEqual(warp_device, device)


@unittest.skipUnless(_jax_version() >= (0, 4, 25), "Jax version too old")
def test_jax_kernel_basic(test, device):
    import jax.numpy as jp

    from warp.jax_experimental import jax_kernel

    n = 64

    jax_triple = jax_kernel(triple_kernel)

    @jax.jit
    def f():
        x = jp.arange(n, dtype=jp.float32)
        return jax_triple(x)

    # run on the given device
    with jax.default_device(wp.device_to_jax(device)):
        y = f()

    result = np.asarray(y).reshape((n,))
    expected = 3 * np.arange(n, dtype=np.float32)

    assert_np_equal(result, expected)


@unittest.skipUnless(_jax_version() >= (0, 4, 25), "Jax version too old")
def test_jax_kernel_scalar(test, device):
    import jax.numpy as jp

    from warp.jax_experimental import jax_kernel

    n = 64

    for T in scalar_types:
        jp_dtype = wp.dtype_to_jax(T)
        np_dtype = wp.dtype_to_numpy(T)

        with test.subTest(msg=T.__name__):
            # get the concrete overload
            kernel_instance = triple_kernel_scalar.add_overload([wp.array(dtype=T), wp.array(dtype=T)])

            jax_triple = jax_kernel(kernel_instance)

            @jax.jit
            def f(jax_triple=jax_triple, jp_dtype=jp_dtype):
                x = jp.arange(n, dtype=jp_dtype)
                return jax_triple(x)

            # run on the given device
            with jax.default_device(wp.device_to_jax(device)):
                y = f()

            result = np.asarray(y).reshape((n,))
            expected = 3 * np.arange(n, dtype=np_dtype)

            assert_np_equal(result, expected)


@unittest.skipUnless(_jax_version() >= (0, 4, 25), "Jax version too old")
def test_jax_kernel_vecmat(test, device):
    import jax.numpy as jp

    from warp.jax_experimental import jax_kernel

    for T in [*vector_types, *matrix_types]:
        jp_dtype = wp.dtype_to_jax(T._wp_scalar_type_)
        np_dtype = wp.dtype_to_numpy(T._wp_scalar_type_)

        n = 64 // T._length_
        scalar_shape = (n, *T._shape_)
        scalar_len = n * T._length_

        with test.subTest(msg=T.__name__):
            # get the concrete overload
            kernel_instance = triple_kernel_vecmat.add_overload([wp.array(dtype=T), wp.array(dtype=T)])

            jax_triple = jax_kernel(kernel_instance)

            @jax.jit
            def f(jax_triple=jax_triple, jp_dtype=jp_dtype, scalar_len=scalar_len, scalar_shape=scalar_shape):
                x = jp.arange(scalar_len, dtype=jp_dtype).reshape(scalar_shape)
                return jax_triple(x)

            # run on the given device
            with jax.default_device(wp.device_to_jax(device)):
                y = f()

            result = np.asarray(y).reshape(scalar_shape)
            expected = 3 * np.arange(scalar_len, dtype=np_dtype).reshape(scalar_shape)

            assert_np_equal(result, expected)


@unittest.skipUnless(_jax_version() >= (0, 4, 25), "Jax version too old")
def test_jax_kernel_multiarg(test, device):
    import jax.numpy as jp

    from warp.jax_experimental import jax_kernel

    n = 64

    jax_multiarg = jax_kernel(multiarg_kernel)

    @jax.jit
    def f():
        a = jp.full(n, 1, dtype=jp.float32)
        b = jp.full(n, 2, dtype=jp.float32)
        c = jp.full(n, 3, dtype=jp.float32)
        return jax_multiarg(a, b, c)

    # run on the given device
    with jax.default_device(wp.device_to_jax(device)):
        x, y = f()

    result_x, result_y = np.asarray(x), np.asarray(y)
    expected_x = np.full(n, 3, dtype=np.float32)
    expected_y = np.full(n, 5, dtype=np.float32)

    assert_np_equal(result_x, expected_x)
    assert_np_equal(result_y, expected_y)


@unittest.skipUnless(_jax_version() >= (0, 4, 25), "Jax version too old")
def test_jax_kernel_launch_dims(test, device):
    import jax.numpy as jp

    from warp.jax_experimental import jax_kernel

    n = 64
    m = 32

    # Test with 1D launch dims
    @wp.kernel
    def add_one_kernel(x: wp.array(dtype=float), y: wp.array(dtype=float)):
        tid = wp.tid()
        y[tid] = x[tid] + 1.0

    jax_add_one = jax_kernel(
        add_one_kernel, launch_dims=(n - 2,)
    )  # Intentionally not the same as the first dimension of the input

    @jax.jit
    def f_1d():
        x = jp.arange(n, dtype=jp.float32)
        return jax_add_one(x)

    # Test with 2D launch dims
    @wp.kernel
    def add_one_2d_kernel(x: wp.array2d(dtype=float), y: wp.array2d(dtype=float)):
        i, j = wp.tid()
        y[i, j] = x[i, j] + 1.0

    jax_add_one_2d = jax_kernel(
        add_one_2d_kernel, launch_dims=(n - 2, m - 2)
    )  # Intentionally not the same as the first dimension of the input

    @jax.jit
    def f_2d():
        x = jp.zeros((n, m), dtype=jp.float32) + 3.0
        return jax_add_one_2d(x)

    # run on the given device
    with jax.default_device(wp.device_to_jax(device)):
        y_1d = f_1d()
        y_2d = f_2d()

    result_1d = np.asarray(y_1d).reshape((n - 2,))
    expected_1d = np.arange(n - 2, dtype=np.float32) + 1.0

    result_2d = np.asarray(y_2d).reshape((n - 2, m - 2))
    expected_2d = np.full((n - 2, m - 2), 4.0, dtype=np.float32)

    assert_np_equal(result_1d, expected_1d)
    assert_np_equal(result_2d, expected_2d)


class TestJax(unittest.TestCase):
    pass


# try adding Jax tests if Jax is installed correctly
try:
    # prevent Jax from gobbling up GPU memory
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    import jax

    # NOTE: we must enable 64-bit types in Jax to test the full gamut of types
    jax.config.update("jax_enable_x64", True)

    # check which Warp devices work with Jax
    # CUDA devices may fail if Jax cannot find a CUDA Toolkit
    test_devices = get_test_devices()
    jax_compatible_devices = []
    jax_compatible_cuda_devices = []
    for d in test_devices:
        try:
            with jax.default_device(wp.device_to_jax(d)):
                j = jax.numpy.arange(10, dtype=jax.numpy.float32)
                j += 1
            jax_compatible_devices.append(d)
            if d.is_cuda:
                jax_compatible_cuda_devices.append(d)
        except Exception as e:
            print(f"Skipping Jax DLPack tests on device '{d}' due to exception: {e}")

    add_function_test(TestJax, "test_dtype_from_jax", test_dtype_from_jax, devices=None)
    add_function_test(TestJax, "test_dtype_to_jax", test_dtype_to_jax, devices=None)

    if jax_compatible_devices:
        add_function_test(TestJax, "test_device_conversion", test_device_conversion, devices=jax_compatible_devices)

    if jax_compatible_cuda_devices:
        add_function_test(TestJax, "test_jax_kernel_basic", test_jax_kernel_basic, devices=jax_compatible_cuda_devices)
        add_function_test(
            TestJax, "test_jax_kernel_scalar", test_jax_kernel_scalar, devices=jax_compatible_cuda_devices
        )
        add_function_test(
            TestJax, "test_jax_kernel_vecmat", test_jax_kernel_vecmat, devices=jax_compatible_cuda_devices
        )
        add_function_test(
            TestJax, "test_jax_kernel_multiarg", test_jax_kernel_multiarg, devices=jax_compatible_cuda_devices
        )

        add_function_test(
            TestJax, "test_jax_kernel_launch_dims", test_jax_kernel_launch_dims, devices=jax_compatible_cuda_devices
        )

except Exception as e:
    print(f"Skipping Jax tests due to exception: {e}")


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
