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

# ruff: noqa: PLC0415

import os
import unittest
from functools import partial
from typing import Any

import numpy as np

import warp as wp
from warp._src.jax import get_jax_device
from warp.tests.unittest_utils import *

# default array size for tests
ARRAY_SIZE = 1024 * 1024


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


@wp.kernel
def inc_1d_kernel(x: wp.array(dtype=float), y: wp.array(dtype=float)):
    tid = wp.tid()
    y[tid] = x[tid] + 1.0


@wp.kernel
def inc_2d_kernel(x: wp.array2d(dtype=float), y: wp.array2d(dtype=float)):
    i, j = wp.tid()
    y[i, j] = x[i, j] + 1.0


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
scalar_types = wp._src.types.scalar_types
vector_types = []
matrix_types = []
for dim in [2, 3, 4]:
    for T in scalar_types:
        vector_types.append(wp.types.vector(dim, T))
        matrix_types.append(wp.types.matrix((dim, dim), T))

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


def test_jax_kernel_basic(test, device, use_ffi=False):
    import jax.numpy as jp

    if use_ffi:
        from warp.jax_experimental.ffi import jax_kernel

        jax_triple = jax_kernel(triple_kernel)
    else:
        from warp.jax_experimental.custom_call import jax_kernel

        jax_triple = jax_kernel(triple_kernel, quiet=True)  # suppress deprecation warnings

    n = ARRAY_SIZE

    @jax.jit
    def f():
        x = jp.arange(n, dtype=jp.float32)
        return jax_triple(x)

    # run on the given device
    with jax.default_device(wp.device_to_jax(device)):
        y = f()

    jax.block_until_ready(y)

    result = np.asarray(y).reshape((n,))
    expected = 3 * np.arange(n, dtype=np.float32)

    assert_np_equal(result, expected)


def test_jax_kernel_scalar(test, device, use_ffi=False):
    import jax.numpy as jp

    if use_ffi:
        from warp.jax_experimental.ffi import jax_kernel

        kwargs = {}
    else:
        from warp.jax_experimental.custom_call import jax_kernel

        kwargs = {"quiet": True}

    # use a smallish size to ensure arange * 3 doesn't overflow
    n = 64

    for T in scalar_types:
        jp_dtype = wp.dtype_to_jax(T)
        np_dtype = wp.dtype_to_numpy(T)

        with test.subTest(msg=T.__name__):
            # get the concrete overload
            kernel_instance = triple_kernel_scalar.add_overload([wp.array(dtype=T), wp.array(dtype=T)])

            jax_triple = jax_kernel(kernel_instance, **kwargs)

            @jax.jit
            def f(jax_triple=jax_triple, jp_dtype=jp_dtype):
                x = jp.arange(n, dtype=jp_dtype)
                return jax_triple(x)

            # run on the given device
            with jax.default_device(wp.device_to_jax(device)):
                y = f()

            jax.block_until_ready(y)

            result = np.asarray(y).reshape((n,))
            expected = 3 * np.arange(n, dtype=np_dtype)

            assert_np_equal(result, expected)


def test_jax_kernel_vecmat(test, device, use_ffi=False):
    import jax.numpy as jp

    if use_ffi:
        from warp.jax_experimental.ffi import jax_kernel

        kwargs = {}
    else:
        from warp.jax_experimental.custom_call import jax_kernel

        kwargs = {"quiet": True}

    for T in [*vector_types, *matrix_types]:
        jp_dtype = wp.dtype_to_jax(T._wp_scalar_type_)
        np_dtype = wp.dtype_to_numpy(T._wp_scalar_type_)

        # use a smallish size to ensure arange * 3 doesn't overflow
        n = 64 // T._length_
        scalar_shape = (n, *T._shape_)
        scalar_len = n * T._length_

        with test.subTest(msg=T.__name__):
            # get the concrete overload
            kernel_instance = triple_kernel_vecmat.add_overload([wp.array(dtype=T), wp.array(dtype=T)])

            jax_triple = jax_kernel(kernel_instance, **kwargs)

            @jax.jit
            def f(jax_triple=jax_triple, jp_dtype=jp_dtype, scalar_len=scalar_len, scalar_shape=scalar_shape):
                x = jp.arange(scalar_len, dtype=jp_dtype).reshape(scalar_shape)
                return jax_triple(x)

            # run on the given device
            with jax.default_device(wp.device_to_jax(device)):
                y = f()

            jax.block_until_ready(y)

            result = np.asarray(y).reshape(scalar_shape)
            expected = 3 * np.arange(scalar_len, dtype=np_dtype).reshape(scalar_shape)

            assert_np_equal(result, expected)


def test_jax_kernel_multiarg(test, device, use_ffi=False):
    import jax.numpy as jp

    if use_ffi:
        from warp.jax_experimental.ffi import jax_kernel

        jax_multiarg = jax_kernel(multiarg_kernel, num_outputs=2)
    else:
        from warp.jax_experimental.custom_call import jax_kernel

        jax_multiarg = jax_kernel(multiarg_kernel, quiet=True)

    n = ARRAY_SIZE

    @jax.jit
    def f():
        a = jp.full(n, 1, dtype=jp.float32)
        b = jp.full(n, 2, dtype=jp.float32)
        c = jp.full(n, 3, dtype=jp.float32)
        return jax_multiarg(a, b, c)

    # run on the given device
    with jax.default_device(wp.device_to_jax(device)):
        x, y = f()

    jax.block_until_ready([x, y])

    result_x, result_y = np.asarray(x), np.asarray(y)
    expected_x = np.full(n, 3, dtype=np.float32)
    expected_y = np.full(n, 5, dtype=np.float32)

    assert_np_equal(result_x, expected_x)
    assert_np_equal(result_y, expected_y)


def test_jax_kernel_launch_dims(test, device, use_ffi=False):
    import jax.numpy as jp

    if use_ffi:
        from warp.jax_experimental.ffi import jax_kernel

        kwargs = {}
    else:
        from warp.jax_experimental.custom_call import jax_kernel

        kwargs = {"quiet": True}

    n = 64
    m = 32

    # Test with 1D launch dims
    jax_inc_1d = jax_kernel(
        inc_1d_kernel, launch_dims=(n - 2,), **kwargs
    )  # Intentionally not the same as the first dimension of the input

    @jax.jit
    def f_1d():
        x = jp.arange(n, dtype=jp.float32)
        return jax_inc_1d(x)

    # Test with 2D launch dims
    jax_inc_2d = jax_kernel(
        inc_2d_kernel, launch_dims=(n - 2, m - 2), **kwargs
    )  # Intentionally not the same as the first dimension of the input

    @jax.jit
    def f_2d():
        x = jp.zeros((n, m), dtype=jp.float32) + 3.0
        return jax_inc_2d(x)

    # run on the given device
    with jax.default_device(wp.device_to_jax(device)):
        y_1d = f_1d()
        y_2d = f_2d()

    jax.block_until_ready([y_1d, y_2d])

    result_1d = np.asarray(y_1d).reshape((n - 2,))
    expected_1d = np.arange(n - 2, dtype=np.float32) + 1.0

    result_2d = np.asarray(y_2d).reshape((n - 2, m - 2))
    expected_2d = np.full((n - 2, m - 2), 4.0, dtype=np.float32)

    assert_np_equal(result_1d, expected_1d)
    assert_np_equal(result_2d, expected_2d)


# =========================================================================================================
# JAX FFI
# =========================================================================================================


@wp.kernel
def add_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float), output: wp.array(dtype=float)):
    tid = wp.tid()
    output[tid] = a[tid] + b[tid]


@wp.kernel
def axpy_kernel(x: wp.array(dtype=float), y: wp.array(dtype=float), alpha: float, out: wp.array(dtype=float)):
    tid = wp.tid()
    out[tid] = alpha * x[tid] + y[tid]


@wp.kernel
def sincos_kernel(angle: wp.array(dtype=float), sin_out: wp.array(dtype=float), cos_out: wp.array(dtype=float)):
    tid = wp.tid()
    sin_out[tid] = wp.sin(angle[tid])
    cos_out[tid] = wp.cos(angle[tid])


@wp.kernel
def diagonal_kernel(output: wp.array(dtype=wp.mat33)):
    tid = wp.tid()
    d = float(tid + 1)
    output[tid] = wp.mat33(d, 0.0, 0.0, 0.0, d * 2.0, 0.0, 0.0, 0.0, d * 3.0)


@wp.kernel
def scale_kernel(a: wp.array(dtype=float), s: float, output: wp.array(dtype=float)):
    tid = wp.tid()
    output[tid] = a[tid] * s


@wp.kernel
def scale_vec_kernel(a: wp.array(dtype=wp.vec2), s: float, output: wp.array(dtype=wp.vec2)):
    tid = wp.tid()
    output[tid] = a[tid] * s


@wp.kernel
def accum_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float)):
    tid = wp.tid()
    b[tid] += a[tid]


@wp.kernel
def matmul_kernel(
    a: wp.array2d(dtype=float),  # NxK
    b: wp.array2d(dtype=float),  # KxM
    c: wp.array2d(dtype=float),  # NxM
):
    # launch dims should be (N, M)
    i, j = wp.tid()
    N = a.shape[0]
    K = a.shape[1]
    M = b.shape[1]
    if i < N and j < M:
        s = wp.float32(0)
        for k in range(K):
            s += a[i, k] * b[k, j]
        c[i, j] = s


@wp.kernel
def in_out_kernel(
    a: wp.array(dtype=float),  # input only
    b: wp.array(dtype=float),  # input and output
    c: wp.array(dtype=float),  # output only
):
    tid = wp.tid()
    b[tid] += a[tid]
    c[tid] = 2.0 * a[tid]


@wp.kernel
def multi_out_kernel(
    a: wp.array(dtype=float), b: wp.array(dtype=float), s: float, c: wp.array(dtype=float), d: wp.array(dtype=float)
):
    tid = wp.tid()
    c[tid] = a[tid] + b[tid]
    d[tid] = s * a[tid]


@wp.kernel
def multi_out_kernel_v2(
    a: wp.array(dtype=float), b: wp.array(dtype=float), s: float, c: wp.array(dtype=float), d: wp.array(dtype=float)
):
    tid = wp.tid()
    c[tid] = a[tid] * a[tid]
    d[tid] = a[tid] * b[tid] * s


@wp.kernel
def multi_out_kernel_v3(
    a: wp.array(dtype=float), b: wp.array(dtype=float), s: float, c: wp.array(dtype=float), d: wp.array(dtype=float)
):
    tid = wp.tid()
    c[tid] = a[tid] ** 2.0
    d[tid] = a[tid] * b[tid] * s


@wp.kernel
def scale_sum_square_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float), s: float, c: wp.array(dtype=float)):
    tid = wp.tid()
    c[tid] = (a[tid] * s + b[tid]) ** 2.0


# The Python function to call.
# Note the argument annotations, just like Warp kernels.
def scale_func(
    # inputs
    a: wp.array(dtype=float),
    b: wp.array(dtype=wp.vec2),
    s: float,
    # outputs
    c: wp.array(dtype=float),
    d: wp.array(dtype=wp.vec2),
):
    wp.launch(scale_kernel, dim=a.shape, inputs=[a, s], outputs=[c])
    wp.launch(scale_vec_kernel, dim=b.shape, inputs=[b, s], outputs=[d])


def in_out_func(
    a: wp.array(dtype=float),  # input only
    b: wp.array(dtype=float),  # input and output
    c: wp.array(dtype=float),  # output only
):
    wp.launch(scale_kernel, dim=a.size, inputs=[a, 2.0], outputs=[c])
    wp.launch(accum_kernel, dim=a.size, inputs=[a, b])  # modifies `b`


def double_func(
    # inputs
    a: wp.array(dtype=float),
    # outputs
    b: wp.array(dtype=float),
):
    wp.launch(scale_kernel, dim=a.shape, inputs=[a, 2.0], outputs=[b])


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_jax_kernel_add(test, device):
    # two inputs and one output
    import jax.numpy as jp

    from warp.jax_experimental.ffi import jax_kernel

    jax_add = jax_kernel(add_kernel)

    @jax.jit
    def f():
        n = ARRAY_SIZE
        a = jp.arange(n, dtype=jp.float32)
        b = jp.ones(n, dtype=jp.float32)
        return jax_add(a, b)

    with jax.default_device(wp.device_to_jax(device)):
        (y,) = f()

    jax.block_until_ready(y)

    result = np.asarray(y)
    expected = np.arange(1, ARRAY_SIZE + 1, dtype=np.float32)

    assert_np_equal(result, expected)


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_jax_kernel_sincos(test, device):
    # one input and two outputs
    import jax.numpy as jp

    from warp.jax_experimental.ffi import jax_kernel

    jax_sincos = jax_kernel(sincos_kernel, num_outputs=2)

    n = ARRAY_SIZE

    @jax.jit
    def f():
        a = jp.linspace(0, 2 * jp.pi, n, dtype=jp.float32)
        return jax_sincos(a)

    with jax.default_device(wp.device_to_jax(device)):
        s, c = f()

    jax.block_until_ready([s, c])

    result_s = np.asarray(s)
    result_c = np.asarray(c)

    a = np.linspace(0, 2 * np.pi, n, dtype=np.float32)
    expected_s = np.sin(a)
    expected_c = np.cos(a)

    assert_np_equal(result_s, expected_s, tol=1e-4)
    assert_np_equal(result_c, expected_c, tol=1e-4)


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_jax_kernel_diagonal(test, device):
    # no inputs and one output
    from warp.jax_experimental.ffi import jax_kernel

    jax_diagonal = jax_kernel(diagonal_kernel)

    @jax.jit
    def f():
        # launch dimensions determine output size
        return jax_diagonal(launch_dims=4)

    with jax.default_device(wp.device_to_jax(device)):
        (d,) = f()

    jax.block_until_ready(d)

    result = np.asarray(d)
    expected = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
            [[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 6.0]],
            [[3.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 9.0]],
            [[4.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 12.0]],
        ],
        dtype=np.float32,
    )

    assert_np_equal(result, expected)


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_jax_kernel_in_out(test, device):
    # in-out args
    import jax.numpy as jp

    from warp.jax_experimental.ffi import jax_kernel

    jax_func = jax_kernel(in_out_kernel, num_outputs=2, in_out_argnames=["b"])

    f = jax.jit(jax_func)

    with jax.default_device(wp.device_to_jax(device)):
        a = jp.ones(ARRAY_SIZE, dtype=jp.float32)
        b = jp.arange(ARRAY_SIZE, dtype=jp.float32)
        b, c = f(a, b)

    jax.block_until_ready([b, c])

    assert_np_equal(b, np.arange(1, ARRAY_SIZE + 1, dtype=np.float32))
    assert_np_equal(c, np.full(ARRAY_SIZE, 2, dtype=np.float32))


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_jax_kernel_scale_vec_constant(test, device):
    # multiply vectors by scalar (constant)
    import jax.numpy as jp

    from warp.jax_experimental.ffi import jax_kernel

    jax_scale_vec = jax_kernel(scale_vec_kernel)

    @jax.jit
    def f():
        a = jp.arange(ARRAY_SIZE, dtype=jp.float32).reshape((ARRAY_SIZE // 2, 2))  # array of vec2
        s = 2.0
        return jax_scale_vec(a, s)

    with jax.default_device(wp.device_to_jax(device)):
        (b,) = f()

    jax.block_until_ready(b)

    expected = 2 * np.arange(ARRAY_SIZE, dtype=np.float32).reshape((ARRAY_SIZE // 2, 2))

    assert_np_equal(b, expected)


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_jax_kernel_scale_vec_static(test, device):
    # multiply vectors by scalar (static arg)
    import jax.numpy as jp

    from warp.jax_experimental.ffi import jax_kernel

    jax_scale_vec = jax_kernel(scale_vec_kernel)

    # NOTE: scalar arguments must be static compile-time constants
    @partial(jax.jit, static_argnames=["s"])
    def f(a, s):
        return jax_scale_vec(a, s)

    a = jp.arange(ARRAY_SIZE, dtype=jp.float32).reshape((ARRAY_SIZE // 2, 2))  # array of vec2
    s = 3.0

    with jax.default_device(wp.device_to_jax(device)):
        (b,) = f(a, s)

    jax.block_until_ready(b)

    expected = 3 * np.arange(ARRAY_SIZE, dtype=np.float32).reshape((ARRAY_SIZE // 2, 2))

    assert_np_equal(b, expected)


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_jax_kernel_launch_dims_default(test, device):
    # specify default launch dims
    import jax.numpy as jp

    from warp.jax_experimental.ffi import jax_kernel

    N, M, K = 3, 4, 2

    jax_matmul = jax_kernel(matmul_kernel, launch_dims=(N, M))

    @jax.jit
    def f():
        a = jp.full((N, K), 2, dtype=jp.float32)
        b = jp.full((K, M), 3, dtype=jp.float32)

        # use default launch dims
        return jax_matmul(a, b)

    with jax.default_device(wp.device_to_jax(device)):
        (result,) = f()

    jax.block_until_ready(result)

    expected = np.full((3, 4), 12, dtype=np.float32)

    test.assertEqual(result.shape, expected.shape)
    assert_np_equal(result, expected)


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_jax_kernel_launch_dims_custom(test, device):
    # specify custom launch dims per call
    import jax.numpy as jp

    from warp.jax_experimental.ffi import jax_kernel

    jax_matmul = jax_kernel(matmul_kernel)

    @jax.jit
    def f():
        N1, M1, K1 = 3, 4, 2
        a1 = jp.full((N1, K1), 2, dtype=jp.float32)
        b1 = jp.full((K1, M1), 3, dtype=jp.float32)

        # use custom launch dims
        result1 = jax_matmul(a1, b1, launch_dims=(N1, M1))

        N2, M2, K2 = 4, 3, 2
        a2 = jp.full((N2, K2), 2, dtype=jp.float32)
        b2 = jp.full((K2, M2), 3, dtype=jp.float32)

        # use different custom launch dims
        result2 = jax_matmul(a2, b2, launch_dims=(N2, M2))

        return result1[0], result2[0]

    with jax.default_device(wp.device_to_jax(device)):
        result1, result2 = f()

    jax.block_until_ready([result1, result2])

    expected1 = np.full((3, 4), 12, dtype=np.float32)
    expected2 = np.full((4, 3), 12, dtype=np.float32)

    test.assertEqual(result1.shape, expected1.shape)
    test.assertEqual(result2.shape, expected2.shape)
    assert_np_equal(result1, expected1)
    assert_np_equal(result2, expected2)


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_jax_callable_scale_constant(test, device):
    # scale two arrays using a constant
    import jax.numpy as jp

    from warp.jax_experimental.ffi import jax_callable

    jax_func = jax_callable(scale_func, num_outputs=2)

    @jax.jit
    def f():
        # inputs
        a = jp.arange(ARRAY_SIZE, dtype=jp.float32)
        b = jp.arange(ARRAY_SIZE, dtype=jp.float32).reshape((ARRAY_SIZE // 2, 2))  # wp.vec2
        s = 2.0

        # output shapes
        output_dims = {"c": a.shape, "d": b.shape}

        c, d = jax_func(a, b, s, output_dims=output_dims)

        return c, d

    with jax.default_device(wp.device_to_jax(device)):
        result1, result2 = f()

    jax.block_until_ready([result1, result2])

    expected1 = 2 * np.arange(ARRAY_SIZE, dtype=np.float32)
    expected2 = 2 * np.arange(ARRAY_SIZE, dtype=np.float32).reshape((ARRAY_SIZE // 2, 2))

    assert_np_equal(result1, expected1)
    assert_np_equal(result2, expected2)


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_jax_callable_scale_static(test, device):
    # scale two arrays using a static arg
    import jax.numpy as jp

    from warp.jax_experimental.ffi import jax_callable

    jax_func = jax_callable(scale_func, num_outputs=2)

    # NOTE: scalar arguments must be static compile-time constants
    @partial(jax.jit, static_argnames=["s"])
    def f(a, b, s):
        # output shapes
        output_dims = {"c": a.shape, "d": b.shape}

        c, d = jax_func(a, b, s, output_dims=output_dims)

        return c, d

    with jax.default_device(wp.device_to_jax(device)):
        # inputs
        a = jp.arange(ARRAY_SIZE, dtype=jp.float32)
        b = jp.arange(ARRAY_SIZE, dtype=jp.float32).reshape((ARRAY_SIZE // 2, 2))  # wp.vec2
        s = 3.0
        result1, result2 = f(a, b, s)

    jax.block_until_ready([result1, result2])

    expected1 = 3 * np.arange(ARRAY_SIZE, dtype=np.float32)
    expected2 = 3 * np.arange(ARRAY_SIZE, dtype=np.float32).reshape((ARRAY_SIZE // 2, 2))

    assert_np_equal(result1, expected1)
    assert_np_equal(result2, expected2)


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_jax_callable_in_out(test, device):
    # in-out arguments
    import jax.numpy as jp

    from warp.jax_experimental.ffi import jax_callable

    jax_func = jax_callable(in_out_func, num_outputs=2, in_out_argnames=["b"])

    f = jax.jit(jax_func)

    with jax.default_device(wp.device_to_jax(device)):
        a = jp.ones(ARRAY_SIZE, dtype=jp.float32)
        b = jp.arange(ARRAY_SIZE, dtype=jp.float32)
        b, c = f(a, b)

    jax.block_until_ready([b, c])

    assert_np_equal(b, np.arange(1, ARRAY_SIZE + 1, dtype=np.float32))
    assert_np_equal(c, np.full(ARRAY_SIZE, 2, dtype=np.float32))


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_jax_callable_graph_cache(test, device):
    # test graph caching limits
    import jax
    import jax.numpy as jp

    from warp.jax_experimental.ffi import (
        GraphMode,
        clear_jax_callable_graph_cache,
        get_jax_callable_default_graph_cache_max,
        jax_callable,
        set_jax_callable_default_graph_cache_max,
    )

    # --- test with default cache settings ---

    jax_double = jax_callable(double_func, graph_mode=GraphMode.WARP)
    f = jax.jit(jax_double)
    arrays = []

    test.assertEqual(jax_double.graph_cache_max, get_jax_callable_default_graph_cache_max())

    with jax.default_device(wp.device_to_jax(device)):
        for i in range(10):
            n = 10 + i
            a = jp.arange(n, dtype=jp.float32)
            (b,) = f(a)

            assert_np_equal(b, 2 * np.arange(n, dtype=np.float32))

            # ensure graph cache is always growing
            test.assertEqual(jax_double.graph_cache_size, i + 1)

            # keep JAX array alive to prevent the memory from being reused, thus forcing a new graph capture each time
            arrays.append(a)

    # --- test clearing one callable's cache ---

    clear_jax_callable_graph_cache(jax_double)

    test.assertEqual(jax_double.graph_cache_size, 0)

    # --- test with a custom cache limit ---

    graph_cache_max = 5
    jax_double = jax_callable(double_func, graph_mode=GraphMode.WARP, graph_cache_max=graph_cache_max)
    f = jax.jit(jax_double)
    arrays = []

    test.assertEqual(jax_double.graph_cache_max, graph_cache_max)

    with jax.default_device(wp.device_to_jax(device)):
        for i in range(10):
            n = 10 + i
            a = jp.arange(n, dtype=jp.float32)
            (b,) = f(a)

            assert_np_equal(b, 2 * np.arange(n, dtype=np.float32))

            # ensure graph cache size is capped
            test.assertEqual(jax_double.graph_cache_size, min(i + 1, graph_cache_max))

            # keep JAX array alive to prevent the memory from being reused, thus forcing a new graph capture
            arrays.append(a)

    # --- test clearing all callables' caches ---

    clear_jax_callable_graph_cache()

    with wp._src.jax_experimental.ffi._FFI_REGISTRY_LOCK:
        for c in wp._src.jax_experimental.ffi._FFI_CALLABLE_REGISTRY.values():
            test.assertEqual(c.graph_cache_size, 0)

    # --- test with a custom default cache limit ---

    saved_max = get_jax_callable_default_graph_cache_max()
    try:
        set_jax_callable_default_graph_cache_max(5)
        jax_double = jax_callable(double_func, graph_mode=GraphMode.WARP)
        f = jax.jit(jax_double)
        arrays = []

        test.assertEqual(jax_double.graph_cache_max, get_jax_callable_default_graph_cache_max())

        with jax.default_device(wp.device_to_jax(device)):
            for i in range(10):
                n = 10 + i
                a = jp.arange(n, dtype=jp.float32)
                (b,) = f(a)

                assert_np_equal(b, 2 * np.arange(n, dtype=np.float32))

                # ensure graph cache size is capped
                test.assertEqual(
                    jax_double.graph_cache_size,
                    min(i + 1, get_jax_callable_default_graph_cache_max()),
                )

                # keep JAX array alive to prevent the memory from being reused, thus forcing a new graph capture
                arrays.append(a)

        clear_jax_callable_graph_cache()

    finally:
        set_jax_callable_default_graph_cache_max(saved_max)


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
@unittest.skip(
    "Flaky: race condition in multi-device JAX pmap with FFI - second device output occasionally returns zeros"
)
def test_ffi_jax_callable_pmap_mul(test, device):
    import jax
    import jax.numpy as jp

    from warp.jax_experimental.ffi import jax_callable

    j = jax_callable(double_func, num_outputs=1)

    ndev = jax.local_device_count()
    per_device = max(ARRAY_SIZE // ndev, 64)
    x = jp.arange(ndev * per_device, dtype=jp.float32).reshape((ndev, per_device))

    def per_device_func(v):
        (y,) = j(v)
        return y

    y = jax.pmap(per_device_func)(x)

    jax.block_until_ready(y)

    assert_np_equal(np.asarray(y), 2 * np.asarray(x))


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
@unittest.skip(
    "Flaky: race condition in multi-device JAX pmap with FFI - second device output occasionally returns zeros"
)
def test_ffi_jax_callable_pmap_multi_output(test, device):
    import jax
    import jax.numpy as jp

    from warp.jax_experimental.ffi import jax_callable

    def multi_out_py(
        a: wp.array(dtype=float),
        b: wp.array(dtype=float),
        s: float,
        c: wp.array(dtype=float),
        d: wp.array(dtype=float),
    ):
        wp.launch(multi_out_kernel, dim=a.shape, inputs=[a, b, s], outputs=[c, d])

    j = jax_callable(multi_out_py, num_outputs=2)

    ndev = jax.local_device_count()
    per_device = max(ARRAY_SIZE // ndev, 64)
    a = jp.arange(ndev * per_device, dtype=jp.float32).reshape((ndev, per_device))
    b = jp.ones((ndev, per_device), dtype=jp.float32)
    s = 3.0

    def per_device_func(aa, bb):
        c, d = j(aa, bb, s)
        return c + d  # simple combine to exercise both outputs

    out = jax.pmap(per_device_func)(a, b)

    jax.block_until_ready(out)

    a_np = np.arange(ndev * per_device, dtype=np.float32).reshape((ndev, per_device))
    b_np = np.ones((ndev, per_device), dtype=np.float32)
    ref = (a_np + b_np) + s * a_np
    assert_np_equal(np.asarray(out), ref)


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
@unittest.skip(
    "Flaky: race condition in multi-device JAX pmap with FFI - second device output occasionally returns zeros"
)
def test_ffi_jax_callable_pmap_multi_stage(test, device):
    import jax
    import jax.numpy as jp

    from warp.jax_experimental.ffi import jax_callable

    def multi_stage_py(
        a: wp.array(dtype=float),
        b: wp.array(dtype=float),
        alpha: float,
        tmp: wp.array(dtype=float),
        out: wp.array(dtype=float),
    ):
        wp.launch(add_kernel, dim=a.shape, inputs=[a, b], outputs=[tmp])
        wp.launch(axpy_kernel, dim=a.shape, inputs=[tmp, b, alpha], outputs=[out])

    j = jax_callable(multi_stage_py, num_outputs=2)

    ndev = jax.local_device_count()
    per_device = max(ARRAY_SIZE // ndev, 64)
    a = jp.arange(ndev * per_device, dtype=jp.float32).reshape((ndev, per_device))
    b = jp.ones((ndev, per_device), dtype=jp.float32)
    alpha = 2.5

    def per_device_func(aa, bb):
        tmp, out = j(aa, bb, alpha)
        return tmp + out

    combined = jax.pmap(per_device_func)(a, b)

    jax.block_until_ready(combined)

    a_np = np.arange(ndev * per_device, dtype=np.float32).reshape((ndev, per_device))
    b_np = np.ones((ndev, per_device), dtype=np.float32)
    tmp_ref = a_np + b_np
    out_ref = alpha * (a_np + b_np) + b_np
    ref = tmp_ref + out_ref
    assert_np_equal(np.asarray(combined), ref)


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_callback(test, device):
    # in-out arguments
    import jax.numpy as jp

    from warp.jax_experimental.ffi import register_ffi_callback

    # the Python function to call
    def warp_func(inputs, outputs, attrs, ctx):
        # input arrays
        a = inputs[0]
        b = inputs[1]

        # scalar attributes
        s = attrs["scale"]

        # output arrays
        c = outputs[0]
        d = outputs[1]

        device = wp.device_from_jax(get_jax_device())
        stream = wp.Stream(device, cuda_stream=ctx.stream)

        with wp.ScopedStream(stream):
            # launch with arrays of scalars
            wp.launch(scale_kernel, dim=a.shape, inputs=[a, s], outputs=[c])

            # launch with arrays of vec2
            # NOTE: the input shapes are from JAX arrays, we need to strip the inner dimension for vec2 arrays
            wp.launch(scale_vec_kernel, dim=b.shape[0], inputs=[b, s], outputs=[d])

    # register callback
    register_ffi_callback("warp_func", warp_func)

    n = ARRAY_SIZE

    with jax.default_device(wp.device_to_jax(device)):
        # inputs
        a = jp.arange(n, dtype=jp.float32)
        b = jp.arange(n, dtype=jp.float32).reshape((n // 2, 2))  # array of wp.vec2
        s = 2.0

        # set up call
        out_types = [
            jax.ShapeDtypeStruct(a.shape, jp.float32),
            jax.ShapeDtypeStruct(b.shape, jp.float32),  # array of wp.vec2
        ]
        call = jax.ffi.ffi_call("warp_func", out_types)

        # call it
        c, d = call(a, b, scale=s)

    jax.block_until_ready([c, d])

    assert_np_equal(c, 2 * np.arange(ARRAY_SIZE, dtype=np.float32))
    assert_np_equal(d, 2 * np.arange(ARRAY_SIZE, dtype=np.float32).reshape((ARRAY_SIZE // 2, 2)))


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_jax_kernel_autodiff_simple(test, device):
    import jax
    import jax.numpy as jp

    from warp.jax_experimental.ffi import jax_kernel

    jax_func = jax_kernel(
        scale_sum_square_kernel,
        num_outputs=1,
        enable_backward=True,
    )

    from functools import partial

    @partial(jax.jit, static_argnames=["s"])
    def loss(a, b, s):
        out = jax_func(a, b, s)[0]
        return jp.sum(out)

    n = ARRAY_SIZE
    a = jp.arange(n, dtype=jp.float32)
    b = jp.ones(n, dtype=jp.float32)
    s = 2.0

    with jax.default_device(wp.device_to_jax(device)):
        da, db = jax.grad(loss, argnums=(0, 1))(a, b, s)

    jax.block_until_ready([da, db])

    # reference gradients
    # d/da sum((a*s + b)^2) = sum(2*(a*s + b) * s)
    # d/db sum((a*s + b)^2) = sum(2*(a*s + b))
    a_np = np.arange(n, dtype=np.float32)
    b_np = np.ones(n, dtype=np.float32)
    ref_da = 2.0 * (a_np * s + b_np) * s
    ref_db = 2.0 * (a_np * s + b_np)

    assert_np_equal(np.asarray(da), ref_da)
    assert_np_equal(np.asarray(db), ref_db)


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_jax_kernel_autodiff_jit_of_grad_simple(test, device):
    if device.ordinal > 0:
        test.skipTest("Flaky on device ordinal > 0: JAX FFI jit(grad()) returns zeros")

    import jax
    import jax.numpy as jp

    from warp.jax_experimental.ffi import jax_kernel

    jax_func = jax_kernel(scale_sum_square_kernel, num_outputs=1, enable_backward=True)

    def loss(a, b, s):
        out = jax_func(a, b, s)[0]
        return jp.sum(out)

    grad_fn = jax.grad(loss, argnums=(0, 1))

    # more typical: jit(grad(...)) with static scalar
    jitted_grad = jax.jit(lambda a, b, s: grad_fn(a, b, s), static_argnames=("s",))

    n = ARRAY_SIZE
    a = jp.arange(n, dtype=jp.float32)
    b = jp.ones(n, dtype=jp.float32)
    s = 2.0

    with jax.default_device(wp.device_to_jax(device)):
        da, db = jitted_grad(a, b, s)

    jax.block_until_ready([da, db])

    a_np = np.arange(n, dtype=np.float32)
    b_np = np.ones(n, dtype=np.float32)
    ref_da = 2.0 * (a_np * s + b_np) * s
    ref_db = 2.0 * (a_np * s + b_np)

    assert_np_equal(np.asarray(da), ref_da)
    assert_np_equal(np.asarray(db), ref_db)


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_jax_kernel_autodiff_multi_output(test, device):
    if device.ordinal > 0:
        test.skipTest("Flaky on device ordinal > 0: JAX FFI jit(grad()) returns zeros")

    import jax
    import jax.numpy as jp

    from warp.jax_experimental.ffi import jax_kernel

    jax_func = jax_kernel(multi_out_kernel_v3, num_outputs=2, enable_backward=True)

    def caller(fn, a, b, s):
        c, d = fn(a, b, s)
        return jp.sum(c + d)

    @jax.jit
    def grads(a, b, s):
        # mark s as static in the inner call via partial to avoid hashing
        def _inner(a, b, s):
            return caller(jax_func, a, b, s)

        return jax.grad(lambda a, b: _inner(a, b, 2.0), argnums=(0, 1))(a, b)

    n = ARRAY_SIZE
    a = jp.arange(n, dtype=jp.float32)
    b = jp.ones(n, dtype=jp.float32)
    s = 2.0

    with jax.default_device(wp.device_to_jax(device)):
        da, db = grads(a, b, s)

    jax.block_until_ready([da, db])

    a_np = np.arange(n, dtype=np.float32)
    b_np = np.ones(n, dtype=np.float32)
    # d/da sum(c+d) = 2*a + b*s
    ref_da = 2.0 * a_np + b_np * s
    # d/db sum(c+d) = a*s
    ref_db = a_np * s

    assert_np_equal(np.asarray(da), ref_da)
    assert_np_equal(np.asarray(db), ref_db)


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_jax_kernel_autodiff_jit_of_grad_multi_output(test, device):
    if device.ordinal > 0:
        test.skipTest("Flaky on device ordinal > 0: JAX FFI jit(grad()) returns zeros")

    import jax
    import jax.numpy as jp

    from warp.jax_experimental.ffi import jax_kernel

    jax_func = jax_kernel(multi_out_kernel_v3, num_outputs=2, enable_backward=True)

    def loss(a, b, s):
        c, d = jax_func(a, b, s)
        return jp.sum(c + d)

    grad_fn = jax.grad(loss, argnums=(0, 1))
    jitted_grad = jax.jit(lambda a, b, s: grad_fn(a, b, s), static_argnames=("s",))

    n = ARRAY_SIZE
    a = jp.arange(n, dtype=jp.float32)
    b = jp.ones(n, dtype=jp.float32)
    s = 2.0

    with jax.default_device(wp.device_to_jax(device)):
        da, db = jitted_grad(a, b, s)

    jax.block_until_ready([da, db])

    a_np = np.arange(n, dtype=np.float32)
    b_np = np.ones(n, dtype=np.float32)
    ref_da = 2.0 * a_np + b_np * s
    ref_db = a_np * s

    assert_np_equal(np.asarray(da), ref_da)
    assert_np_equal(np.asarray(db), ref_db)


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_jax_kernel_autodiff_2d(test, device):
    import jax
    import jax.numpy as jp

    from warp.jax_experimental.ffi import jax_kernel

    jax_func = jax_kernel(inc_2d_kernel, num_outputs=1, enable_backward=True)

    @jax.jit
    def loss(a):
        out = jax_func(a)[0]
        return jp.sum(out)

    n, m = 8, 6
    a = jp.arange(n * m, dtype=jp.float32).reshape((n, m))

    with jax.default_device(wp.device_to_jax(device)):
        (da,) = jax.grad(loss, argnums=(0,))(a)

    jax.block_until_ready(da)

    ref = np.ones((n, m), dtype=np.float32)
    assert_np_equal(np.asarray(da), ref)


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_jax_kernel_autodiff_vec2(test, device):
    import jax
    import jax.numpy as jp

    from warp.jax_experimental.ffi import jax_kernel

    jax_func = jax_kernel(scale_vec_kernel, num_outputs=1, enable_backward=True)

    from functools import partial

    @partial(jax.jit, static_argnames=("s",))
    def loss(a, s):
        out = jax_func(a, s)[0]
        return jp.sum(out)

    n = ARRAY_SIZE
    a = jp.arange(n, dtype=jp.float32).reshape((n // 2, 2))
    s = 3.0

    with jax.default_device(wp.device_to_jax(device)):
        (da,) = jax.grad(loss, argnums=(0,))(a, s)

    jax.block_until_ready(da)

    # d/da sum(a*s) = s
    ref = np.full_like(np.asarray(a), s)
    assert_np_equal(np.asarray(da), ref)


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_jax_kernel_autodiff_mat22(test, device):
    import jax
    import jax.numpy as jp

    from warp.jax_experimental.ffi import jax_kernel

    @wp.kernel
    def scale_mat_kernel(a: wp.array(dtype=wp.mat22), s: float, out: wp.array(dtype=wp.mat22)):
        tid = wp.tid()
        out[tid] = a[tid] * s

    jax_func = jax_kernel(scale_mat_kernel, num_outputs=1, enable_backward=True)

    from functools import partial

    @partial(jax.jit, static_argnames=("s",))
    def loss(a, s):
        out = jax_func(a, s)[0]
        return jp.sum(out)

    n = 12  # must be divisible by 4 for 2x2 matrices
    a = jp.arange(n, dtype=jp.float32).reshape((n // 4, 2, 2))
    s = 2.5

    with jax.default_device(wp.device_to_jax(device)):
        (da,) = jax.grad(loss, argnums=(0,))(a, s)

    jax.block_until_ready(da)

    ref = np.full((n // 4, 2, 2), s, dtype=np.float32)
    assert_np_equal(np.asarray(da), ref)


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_jax_kernel_autodiff_static_required(test, device):
    import jax
    import jax.numpy as jp

    from warp.jax_experimental.ffi import jax_kernel

    # Require explicit static_argnames for scalar s
    jax_func = jax_kernel(scale_sum_square_kernel, num_outputs=1, enable_backward=True)

    def loss(a, b, s):
        out = jax_func(a, b, s)[0]
        return jp.sum(out)

    n = ARRAY_SIZE
    a = jp.arange(n, dtype=jp.float32)
    b = jp.ones(n, dtype=jp.float32)
    s = 1.5

    with jax.default_device(wp.device_to_jax(device)):
        da, db = jax.grad(loss, argnums=(0, 1))(a, b, s)

    jax.block_until_ready([da, db])

    a_np = np.arange(n, dtype=np.float32)
    b_np = np.ones(n, dtype=np.float32)
    ref_da = 2.0 * (a_np * s + b_np) * s
    ref_db = 2.0 * (a_np * s + b_np)

    assert_np_equal(np.asarray(da), ref_da)
    assert_np_equal(np.asarray(db), ref_db)


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_jax_kernel_autodiff_pmap_triple(test, device):
    import jax
    import jax.numpy as jp

    from warp.jax_experimental.ffi import jax_kernel

    jax_mul = jax_kernel(triple_kernel, num_outputs=1, enable_backward=True)

    ndev = jax.local_device_count()
    per_device = ARRAY_SIZE // ndev
    x = jp.arange(ndev * per_device, dtype=jp.float32).reshape((ndev, per_device))

    def per_device_loss(x):
        y = jax_mul(x)[0]
        return jp.sum(y)

    grads = jax.pmap(jax.grad(per_device_loss))(x)

    jax.block_until_ready(grads)

    assert_np_equal(np.asarray(grads), np.full((ndev, per_device), 3.0, dtype=np.float32))


@unittest.skipUnless(_jax_version() >= (0, 5, 0), "Jax version too old")
def test_ffi_jax_kernel_autodiff_pmap_multi_output(test, device):
    import jax
    import jax.numpy as jp

    from warp.jax_experimental.ffi import jax_kernel

    jax_mo = jax_kernel(multi_out_kernel_v2, num_outputs=2, enable_backward=True)

    ndev = jax.local_device_count()
    per_device = ARRAY_SIZE // ndev
    a = jp.arange(ndev * per_device, dtype=jp.float32).reshape((ndev, per_device))
    b = jp.arange(ndev * per_device, dtype=jp.float32).reshape((ndev, per_device))
    s = 2.0

    def per_dev_loss(aa, bb):
        c, d = jax_mo(aa, bb, s)
        return jp.sum(c + d)

    da, db = jax.pmap(jax.grad(per_dev_loss, argnums=(0, 1)))(a, b)

    jax.block_until_ready([da, db])

    a_np = np.arange(ndev * per_device, dtype=np.float32).reshape((ndev, per_device))
    b_np = np.arange(ndev * per_device, dtype=np.float32).reshape((ndev, per_device))
    ref_da = 2.0 * a_np + b_np * s
    ref_db = a_np * s
    assert_np_equal(np.asarray(da), ref_da)
    assert_np_equal(np.asarray(db), ref_db)


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
        # tests for both custom_call and ffi variants of jax_kernel(), selected by installed JAX version
        if jax.__version_info__ < (0, 4, 25):
            # no interop supported
            ffi_opts = []
        elif jax.__version_info__ < (0, 5, 0):
            # only custom_call supported
            ffi_opts = [False]
        elif jax.__version_info__ < (0, 8, 0):
            # both custom_call and ffi supported
            ffi_opts = [False, True]
        else:
            # only ffi supported
            ffi_opts = [True]

        for use_ffi in ffi_opts:
            suffix = "ffi" if use_ffi else "cc"
            add_function_test(
                TestJax,
                f"test_jax_kernel_basic_{suffix}",
                test_jax_kernel_basic,
                devices=jax_compatible_cuda_devices,
                use_ffi=use_ffi,
            )
            add_function_test(
                TestJax,
                f"test_jax_kernel_scalar_{suffix}",
                test_jax_kernel_scalar,
                devices=jax_compatible_cuda_devices,
                use_ffi=use_ffi,
            )
            add_function_test(
                TestJax,
                f"test_jax_kernel_vecmat_{suffix}",
                test_jax_kernel_vecmat,
                devices=jax_compatible_cuda_devices,
                use_ffi=use_ffi,
            )
            add_function_test(
                TestJax,
                f"test_jax_kernel_multiarg_{suffix}",
                test_jax_kernel_multiarg,
                devices=jax_compatible_cuda_devices,
                use_ffi=use_ffi,
            )
            add_function_test(
                TestJax,
                f"test_jax_kernel_launch_dims_{suffix}",
                test_jax_kernel_launch_dims,
                devices=jax_compatible_cuda_devices,
                use_ffi=use_ffi,
            )

        # ffi.jax_kernel() tests
        add_function_test(
            TestJax, "test_ffi_jax_kernel_add", test_ffi_jax_kernel_add, devices=jax_compatible_cuda_devices
        )
        add_function_test(
            TestJax, "test_ffi_jax_kernel_sincos", test_ffi_jax_kernel_sincos, devices=jax_compatible_cuda_devices
        )
        add_function_test(
            TestJax, "test_ffi_jax_kernel_diagonal", test_ffi_jax_kernel_diagonal, devices=jax_compatible_cuda_devices
        )
        add_function_test(
            TestJax, "test_ffi_jax_kernel_in_out", test_ffi_jax_kernel_in_out, devices=jax_compatible_cuda_devices
        )
        add_function_test(
            TestJax,
            "test_ffi_jax_kernel_scale_vec_constant",
            test_ffi_jax_kernel_scale_vec_constant,
            devices=jax_compatible_cuda_devices,
        )
        add_function_test(
            TestJax,
            "test_ffi_jax_kernel_scale_vec_static",
            test_ffi_jax_kernel_scale_vec_static,
            devices=jax_compatible_cuda_devices,
        )
        add_function_test(
            TestJax,
            "test_ffi_jax_kernel_launch_dims_default",
            test_ffi_jax_kernel_launch_dims_default,
            devices=jax_compatible_cuda_devices,
        )
        add_function_test(
            TestJax,
            "test_ffi_jax_kernel_launch_dims_custom",
            test_ffi_jax_kernel_launch_dims_custom,
            devices=jax_compatible_cuda_devices,
        )

        # ffi.jax_callable() tests
        add_function_test(
            TestJax,
            "test_ffi_jax_callable_scale_constant",
            test_ffi_jax_callable_scale_constant,
            devices=jax_compatible_cuda_devices,
        )
        add_function_test(
            TestJax,
            "test_ffi_jax_callable_scale_static",
            test_ffi_jax_callable_scale_static,
            devices=jax_compatible_cuda_devices,
        )
        add_function_test(
            TestJax, "test_ffi_jax_callable_in_out", test_ffi_jax_callable_in_out, devices=jax_compatible_cuda_devices
        )
        add_function_test(
            TestJax,
            "test_ffi_jax_callable_graph_cache",
            test_ffi_jax_callable_graph_cache,
            devices=jax_compatible_cuda_devices,
        )

        # pmap tests
        add_function_test(
            TestJax,
            "test_ffi_jax_callable_pmap_multi_output",
            test_ffi_jax_callable_pmap_multi_output,
            devices=None,
        )
        add_function_test(
            TestJax,
            "test_ffi_jax_callable_pmap_mul",
            test_ffi_jax_callable_pmap_mul,
            devices=None,
        )
        add_function_test(
            TestJax,
            "test_ffi_jax_callable_pmap_multi_stage",
            test_ffi_jax_callable_pmap_multi_stage,
            devices=None,
        )

        # ffi callback tests
        add_function_test(TestJax, "test_ffi_callback", test_ffi_callback, devices=jax_compatible_cuda_devices)

        # autodiff tests
        add_function_test(
            TestJax,
            "test_ffi_jax_kernel_autodiff_simple",
            test_ffi_jax_kernel_autodiff_simple,
            devices=jax_compatible_cuda_devices,
        )
        add_function_test(
            TestJax,
            "test_ffi_jax_kernel_autodiff_jit_of_grad_simple",
            test_ffi_jax_kernel_autodiff_jit_of_grad_simple,
            devices=jax_compatible_cuda_devices,
        )
        add_function_test(
            TestJax,
            "test_ffi_jax_kernel_autodiff_multi_output",
            test_ffi_jax_kernel_autodiff_multi_output,
            devices=jax_compatible_cuda_devices,
        )
        add_function_test(
            TestJax,
            "test_ffi_jax_kernel_autodiff_jit_of_grad_multi_output",
            test_ffi_jax_kernel_autodiff_jit_of_grad_multi_output,
            devices=jax_compatible_cuda_devices,
        )
        add_function_test(
            TestJax,
            "test_ffi_jax_kernel_autodiff_2d",
            test_ffi_jax_kernel_autodiff_2d,
            devices=jax_compatible_cuda_devices,
        )
        add_function_test(
            TestJax,
            "test_ffi_jax_kernel_autodiff_vec2",
            test_ffi_jax_kernel_autodiff_vec2,
            devices=jax_compatible_cuda_devices,
        )
        add_function_test(
            TestJax,
            "test_ffi_jax_kernel_autodiff_mat22",
            test_ffi_jax_kernel_autodiff_mat22,
            devices=jax_compatible_cuda_devices,
        )
        add_function_test(
            TestJax,
            "test_ffi_jax_kernel_autodiff_static_required",
            test_ffi_jax_kernel_autodiff_static_required,
            devices=jax_compatible_cuda_devices,
        )

        # autodiff with pmap tests
        add_function_test(
            TestJax,
            "test_ffi_jax_kernel_autodiff_pmap_triple",
            test_ffi_jax_kernel_autodiff_pmap_triple,
            devices=None,
        )
        add_function_test(
            TestJax,
            "test_ffi_jax_kernel_autodiff_pmap_multi_output",
            test_ffi_jax_kernel_autodiff_pmap_multi_output,
            devices=None,
        )

except Exception as e:
    print(f"Skipping Jax tests due to exception: {e}")


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
