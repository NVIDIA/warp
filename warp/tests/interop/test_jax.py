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


@unittest.skipUnless(_jax_version() >= (0, 4, 31), "Jax version too old for FFI custom_vjp")
def test_jax_ad_kernel_simple(test, device):
    import jax
    import jax.numpy as jp

    from warp.jax_experimental.ffi import jax_ad_kernel

    @wp.kernel
    def scale_sum_square_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float), s: float, c: wp.array(dtype=float)):
        tid = wp.tid()
        c[tid] = (a[tid] * s + b[tid]) ** 2.0

    jax_func = jax_ad_kernel(scale_sum_square_kernel, num_outputs=1, static_argnames=("s",), vmap_method="sequential")

    from functools import partial

    @partial(jax.jit, static_argnames=["s"])
    def loss(a, b, s):
        out = jax_func(a, b, s)[0]
        return jp.sum(out)

    n = 16
    a = jp.arange(n, dtype=jp.float32)
    b = jp.ones(n, dtype=jp.float32)
    s = 2.0

    with jax.default_device(wp.device_to_jax(device)):
        da, db = jax.grad(loss, argnums=(0, 1))(a, b, s)

    # reference gradients
    # d/da sum((a*s + b)^2) = sum(2*(a*s + b) * s)
    # d/db sum((a*s + b)^2) = sum(2*(a*s + b))
    a_np = np.arange(n, dtype=np.float32)
    b_np = np.ones(n, dtype=np.float32)
    ref_da = 2.0 * (a_np * s + b_np) * s
    ref_db = 2.0 * (a_np * s + b_np)

    assert_np_equal(np.asarray(da), ref_da)
    assert_np_equal(np.asarray(db), ref_db)


@unittest.skipUnless(_jax_version() >= (0, 4, 31), "Jax version too old for FFI custom_vjp")
def test_jax_ad_kernel_jit_of_grad_simple(test, device):
    import jax
    import jax.numpy as jp

    from warp.jax_experimental.ffi import jax_ad_kernel

    @wp.kernel
    def scale_sum_square_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float), s: float, c: wp.array(dtype=float)):
        tid = wp.tid()
        c[tid] = (a[tid] * s + b[tid]) ** 2.0

    jax_func = jax_ad_kernel(scale_sum_square_kernel, num_outputs=1, static_argnames=("s",))

    def loss(a, b, s):
        out = jax_func(a, b, s)[0]
        return jp.sum(out)

    grad_fn = jax.grad(loss, argnums=(0, 1))

    # more typical: jit(grad(...)) with static scalar
    jitted_grad = jax.jit(lambda a, b, s: grad_fn(a, b, s), static_argnames=("s",))

    n = 16
    a = jp.arange(n, dtype=jp.float32)
    b = jp.ones(n, dtype=jp.float32)
    s = 2.0

    with jax.default_device(wp.device_to_jax(device)):
        da, db = jitted_grad(a, b, s)

    a_np = np.arange(n, dtype=np.float32)
    b_np = np.ones(n, dtype=np.float32)
    ref_da = 2.0 * (a_np * s + b_np) * s
    ref_db = 2.0 * (a_np * s + b_np)

    assert_np_equal(np.asarray(da), ref_da)
    assert_np_equal(np.asarray(db), ref_db)


@unittest.skipUnless(_jax_version() >= (0, 4, 31), "Jax version too old for FFI custom_vjp")
def test_jax_ad_kernel_jit_of_grad_multi_output(test, device):
    import jax
    import jax.numpy as jp

    from warp.jax_experimental.ffi import jax_ad_kernel

    @wp.kernel
    def multi_output_kernel(
        a: wp.array(dtype=float), b: wp.array(dtype=float), s: float, c: wp.array(dtype=float), d: wp.array(dtype=float)
    ):
        tid = wp.tid()
        c[tid] = a[tid] ** 2.0
        d[tid] = a[tid] * b[tid] * s

    jax_func = jax_ad_kernel(multi_output_kernel, num_outputs=2, static_argnames=("s",))

    def loss(a, b, s):
        c, d = jax_func(a, b, s)
        return jp.sum(c + d)

    grad_fn = jax.grad(loss, argnums=(0, 1))
    jitted_grad = jax.jit(lambda a, b, s: grad_fn(a, b, s), static_argnames=("s",))

    n = 16
    a = jp.arange(n, dtype=jp.float32)
    b = jp.ones(n, dtype=jp.float32)
    s = 2.0

    with jax.default_device(wp.device_to_jax(device)):
        da, db = jitted_grad(a, b, s)

    a_np = np.arange(n, dtype=np.float32)
    b_np = np.ones(n, dtype=np.float32)
    ref_da = 2.0 * a_np + b_np * s
    ref_db = a_np * s

    assert_np_equal(np.asarray(da), ref_da)
    assert_np_equal(np.asarray(db), ref_db)


@unittest.skipUnless(_jax_version() >= (0, 4, 31), "Jax version too old for FFI custom_vjp")
def test_jax_ad_kernel_multi_output(test, device):
    import jax
    import jax.numpy as jp

    from warp.jax_experimental.ffi import jax_ad_kernel

    @wp.kernel
    def multi_output_kernel(
        a: wp.array(dtype=float), b: wp.array(dtype=float), s: float, c: wp.array(dtype=float), d: wp.array(dtype=float)
    ):
        tid = wp.tid()
        c[tid] = a[tid] ** 2.0
        d[tid] = a[tid] * b[tid] * s

    jax_func = jax_ad_kernel(multi_output_kernel, num_outputs=2, static_argnames=("s",))

    def caller(fn, a, b, s):
        c, d = fn(a, b, s)
        return jp.sum(c + d)

    @jax.jit
    def grads(a, b, s):
        # mark s as static in the inner call via partial to avoid hashing
        def _inner(a, b, s):
            return caller(jax_func, a, b, s)

        return jax.grad(lambda a, b: _inner(a, b, 2.0), argnums=(0, 1))(a, b)

    n = 16
    a = jp.arange(n, dtype=jp.float32)
    b = jp.ones(n, dtype=jp.float32)
    s = 2.0

    with jax.default_device(wp.device_to_jax(device)):
        da, db = grads(a, b, s)

    a_np = np.arange(n, dtype=np.float32)
    b_np = np.ones(n, dtype=np.float32)
    # d/da sum(c+d) = 2*a + b*s
    ref_da = 2.0 * a_np + b_np * s
    # d/db sum(c+d) = a*s
    ref_db = a_np * s

    assert_np_equal(np.asarray(da), ref_da)
    assert_np_equal(np.asarray(db), ref_db)


@unittest.skipUnless(_jax_version() >= (0, 4, 31), "Jax version too old for FFI custom_vjp")
def test_jax_ad_kernel_vec2(test, device):
    import jax
    import jax.numpy as jp

    from warp.jax_experimental.ffi import jax_ad_kernel

    @wp.kernel
    def scale_vec_kernel(a: wp.array(dtype=wp.vec2), s: float, out: wp.array(dtype=wp.vec2)):
        tid = wp.tid()
        out[tid] = a[tid] * s

    jax_func = jax_ad_kernel(scale_vec_kernel, num_outputs=1, static_argnames=("s",))

    from functools import partial

    @partial(jax.jit, static_argnames=("s",))
    def loss(a, s):
        out = jax_func(a, s)[0]
        return jp.sum(out)

    n = 10
    a = jp.arange(n, dtype=jp.float32).reshape((n // 2, 2))
    s = 3.0

    with jax.default_device(wp.device_to_jax(device)):
        (da,) = jax.grad(loss, argnums=(0,))(a, s)

    # d/da sum(a*s) = s
    ref = np.full_like(np.asarray(a), s)
    assert_np_equal(np.asarray(da), ref)


@unittest.skipUnless(_jax_version() >= (0, 4, 31), "Jax version too old for FFI custom_vjp")
def test_jax_ad_kernel_2d(test, device):
    import jax
    import jax.numpy as jp

    from warp.jax_experimental.ffi import jax_ad_kernel

    @wp.kernel
    def add_one_2d(a: wp.array2d(dtype=float), out: wp.array2d(dtype=float)):
        i, j = wp.tid()
        out[i, j] = a[i, j] + 1.0

    jax_func = jax_ad_kernel(add_one_2d, num_outputs=1)

    @jax.jit
    def loss(a):
        out = jax_func(a)[0]
        return jp.sum(out)

    n, m = 8, 6
    a = jp.arange(n * m, dtype=jp.float32).reshape((n, m))

    with jax.default_device(wp.device_to_jax(device)):
        (da,) = jax.grad(loss, argnums=(0,))(a)

    ref = np.ones((n, m), dtype=np.float32)
    assert_np_equal(np.asarray(da), ref)


@unittest.skipUnless(_jax_version() >= (0, 4, 31), "Jax version too old for FFI custom_vjp")
def test_jax_ad_kernel_mat22(test, device):
    import jax
    import jax.numpy as jp

    from warp.jax_experimental.ffi import jax_ad_kernel

    @wp.kernel
    def scale_mat_kernel(a: wp.array(dtype=wp.mat22), s: float, out: wp.array(dtype=wp.mat22)):
        tid = wp.tid()
        out[tid] = a[tid] * s

    jax_func = jax_ad_kernel(scale_mat_kernel, num_outputs=1, static_argnames=("s",))

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

    ref = np.full((n // 4, 2, 2), s, dtype=np.float32)
    assert_np_equal(np.asarray(da), ref)


@unittest.skipUnless(_jax_version() >= (0, 4, 31), "Jax version too old for FFI custom_vjp")
def test_jax_ad_kernel_rgb_hw_not_c(test, device):
    import jax
    import jax.numpy as jp

    from warp.jax_experimental.ffi import jax_ad_kernel

    @wp.kernel
    def brighten_rgb(a: wp.array2d(dtype=wp.vec3), out: wp.array2d(dtype=wp.vec3)):
        i, j = wp.tid()
        out[i, j] = a[i, j] * 2.0

    jax_fn = jax_ad_kernel(brighten_rgb, num_outputs=1)

    H, W = 7, 5

    @jax.jit
    def f(x):
        return jax_fn(x)[0]

    @jax.jit
    def gradf(x):
        return jax.grad(lambda v: jp.sum(jax_fn(v)[0]))(x)

    x = jp.arange(H * W * 3, dtype=jp.float32).reshape((H, W, 3))

    with jax.default_device(wp.device_to_jax(device)):
        y = f(x)
        g = gradf(x)

    expected_y = 2.0 * np.arange(H * W * 3, dtype=np.float32).reshape((H, W, 3))
    expected_g = np.full((H, W, 3), 2.0, dtype=np.float32)

    assert_np_equal(np.asarray(y), expected_y)
    assert_np_equal(np.asarray(g), expected_g)


@unittest.skipUnless(_jax_version() >= (0, 4, 31), "Jax version too old for FFI custom_vjp")
def test_jax_ad_kernel_vmap_simple(test, device):
    import jax
    import jax.numpy as jp

    from warp.jax_experimental.ffi import jax_ad_kernel

    @wp.kernel
    def scale_sum_square_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float), s: float, c: wp.array(dtype=float)):
        tid = wp.tid()
        c[tid] = (a[tid] * s + b[tid]) ** 2.0

    jax_func = jax_ad_kernel(scale_sum_square_kernel, num_outputs=1, static_argnames=("s",))

    # per-sample loss; close over static scalar s to avoid vmap over statics
    def per_sample_loss(a, b):
        out = jax_func(a, b, 2.0)[0]
        return jp.sum(out)

    B, N = 4, 12
    a = jp.arange(B * N, dtype=jp.float32).reshape((B, N))
    b = jp.ones((B, N), dtype=jp.float32)

    with jax.default_device(wp.device_to_jax(device)):
        da, db = jax.vmap(jax.grad(per_sample_loss, argnums=(0, 1)), in_axes=(0, 0))(a, b)

    a_np = np.arange(B * N, dtype=np.float32).reshape((B, N))
    b_np = np.ones((B, N), dtype=np.float32)
    s = 2.0
    ref_da = 2.0 * (a_np * s + b_np) * s
    ref_db = 2.0 * (a_np * s + b_np)

    assert_np_equal(np.asarray(da), ref_da)
    assert_np_equal(np.asarray(db), ref_db)


@unittest.skipUnless(_jax_version() >= (0, 4, 31), "Jax version too old for FFI custom_vjp")
def test_jax_ad_kernel_vmap_vec2(test, device):
    import jax
    import jax.numpy as jp

    from warp.jax_experimental.ffi import jax_ad_kernel

    @wp.kernel
    def scale_vec_kernel(a: wp.array(dtype=wp.vec2), s: float, out: wp.array(dtype=wp.vec2)):
        tid = wp.tid()
        out[tid] = a[tid] * s

    jax_func = jax_ad_kernel(scale_vec_kernel, num_outputs=1, static_argnames=("s",), vmap_method="sequential")

    def per_sample_loss(a):
        out = jax_func(a, 2.0)[0]
        return jp.sum(out)

    B = 3
    n = 8
    a = jp.arange(B * n, dtype=jp.float32).reshape((B, n // 2, 2))

    with jax.default_device(wp.device_to_jax(device)):
        (da,) = jax.vmap(jax.grad(per_sample_loss, argnums=(0,)), in_axes=(0,))(a)

    ref = np.full((B, n // 2, 2), 2.0, dtype=np.float32)
    assert_np_equal(np.asarray(da), ref)


@unittest.skipUnless(_jax_version() >= (0, 4, 31), "Jax version too old for FFI custom_vjp")
def test_jax_ad_kernel_vmap_multi_output(test, device):
    import jax
    import jax.numpy as jp

    from warp.jax_experimental.ffi import jax_ad_kernel

    @wp.kernel
    def multi_output_kernel(
        a: wp.array(dtype=float), b: wp.array(dtype=float), s: float, c: wp.array(dtype=float), d: wp.array(dtype=float)
    ):
        tid = wp.tid()
        c[tid] = a[tid] ** 2.0
        d[tid] = a[tid] * b[tid] * s

    jax_func = jax_ad_kernel(multi_output_kernel, num_outputs=2, static_argnames=("s",), vmap_method="sequential")

    def per_sample_loss(a, b):
        c, d = jax_func(a, b, 2.0)
        return jp.sum(c + d)

    B, n = 2, 12
    a = jp.arange(B * n, dtype=jp.float32).reshape((B, n))
    b = jp.ones((B, n), dtype=jp.float32)

    with jax.default_device(wp.device_to_jax(device)):
        da, db = jax.vmap(jax.grad(per_sample_loss, argnums=(0, 1)), in_axes=(0, 0))(a, b)

    a_np = np.arange(B * n, dtype=np.float32).reshape((B, n))
    b_np = np.ones((B, n), dtype=np.float32)
    s = 2.0
    ref_da = 2.0 * a_np + b_np * s
    ref_db = a_np * s

    assert_np_equal(np.asarray(da), ref_da)
    assert_np_equal(np.asarray(db), ref_db)


@unittest.skipUnless(_jax_version() >= (0, 4, 31), "Jax version too old for FFI custom_vjp")
def test_jax_ad_kernel_vmap_broadcast_all_simple(test, device):
    import jax
    import jax.numpy as jp

    from warp.jax_experimental.ffi import jax_ad_kernel

    @wp.kernel
    def scale_sum_square_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float), s: float, c: wp.array(dtype=float)):
        tid = wp.tid()
        c[tid] = (a[tid] * s + b[tid]) ** 2.0

    jax_func = jax_ad_kernel(
        scale_sum_square_kernel, num_outputs=1, static_argnames=("s",), vmap_method="broadcast_all"
    )

    def per_sample_loss(a, b):
        out = jax_func(a, b, 2.0)[0]
        return jp.sum(out)

    B, N = 3, 10
    a = jp.arange(B * N, dtype=jp.float32).reshape((B, N))
    b = jp.ones((B, N), dtype=jp.float32)

    with jax.default_device(wp.device_to_jax(device)):
        da, db = jax.vmap(jax.grad(per_sample_loss, argnums=(0, 1)), in_axes=(0, 0))(a, b)

    a_np = np.arange(B * N, dtype=np.float32).reshape((B, N))
    b_np = np.ones((B, N), dtype=np.float32)
    s = 2.0
    ref_da = 2.0 * (a_np * s + b_np) * s
    ref_db = 2.0 * (a_np * s + b_np)

    assert_np_equal(np.asarray(da), ref_da)
    assert_np_equal(np.asarray(db), ref_db)


@unittest.skipUnless(_jax_version() >= (0, 4, 31), "Jax version too old for FFI custom_vjp")
def test_jax_ad_kernel_vmap_expand_dims_simple(test, device):
    import jax
    import jax.numpy as jp

    from warp.jax_experimental.ffi import jax_ad_kernel

    @wp.kernel
    def scale_sum_square_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float), s: float, c: wp.array(dtype=float)):
        tid = wp.tid()
        c[tid] = (a[tid] * s + b[tid]) ** 2.0

    jax_func = jax_ad_kernel(
        scale_sum_square_kernel, num_outputs=1, static_argnames=("s",), vmap_method="broadcast_all"
    )

    def per_sample_loss(a, b):
        out = jax_func(a, b, 2.0)[0]
        return jp.sum(out)

    B, N = 2, 9
    a = jp.arange(B * N, dtype=jp.float32).reshape((B, N))
    b = jp.ones((B, N), dtype=jp.float32)

    with jax.default_device(wp.device_to_jax(device)):
        da, db = jax.vmap(jax.grad(per_sample_loss, argnums=(0, 1)), in_axes=(0, 0))(a, b)

    a_np = np.arange(B * N, dtype=np.float32).reshape((B, N))
    b_np = np.ones((B, N), dtype=np.float32)
    s = 2.0
    ref_da = 2.0 * (a_np * s + b_np) * s
    ref_db = 2.0 * (a_np * s + b_np)

    assert_np_equal(np.asarray(da), ref_da)
    assert_np_equal(np.asarray(db), ref_db)


@unittest.skipUnless(_jax_version() >= (0, 4, 31), "Jax version too old for FFI custom_vjp")
def test_jax_ad_kernel_vmap_broadcast_mismatched_inputs(test, device):
    import jax
    import jax.numpy as jp

    from warp.jax_experimental.ffi import jax_ad_kernel

    @wp.kernel
    def scale_sum_square_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float), s: float, c: wp.array(dtype=float)):
        tid = wp.tid()
        c[tid] = (a[tid] * s + b[tid]) ** 2.0

    jax_func = jax_ad_kernel(
        scale_sum_square_kernel, num_outputs=1, static_argnames=("s",), vmap_method="broadcast_all"
    )

    def per_sample_loss(a, b):
        out = jax_func(a, b, 1.5)[0]
        return jp.sum(out)

    B, N = 3, 8
    a = jp.arange(B * N, dtype=jp.float32).reshape((B, N))
    b = jp.ones((N,), dtype=jp.float32)

    with jax.default_device(wp.device_to_jax(device)):
        da, db = jax.vmap(jax.grad(per_sample_loss, argnums=(0, 1)), in_axes=(0, None))(a, b)

    a_np = np.arange(B * N, dtype=np.float32).reshape((B, N))
    b_np = np.ones((N,), dtype=np.float32)
    s = 1.5
    ref_da = 2.0 * (a_np * s + b_np) * s
    ref_db = 2.0 * (a_np * s + b_np)

    assert_np_equal(np.asarray(da), ref_da)
    assert_np_equal(np.asarray(db), ref_db)


@unittest.skipUnless(_jax_version() >= (0, 4, 31), "Jax version too old for FFI custom_vjp")
def test_jax_ad_kernel_vmap_double_batch(test, device):
    import jax
    import jax.numpy as jp

    from warp.jax_experimental.ffi import jax_ad_kernel

    @wp.kernel
    def scale_sum_square_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float), s: float, c: wp.array(dtype=float)):
        tid = wp.tid()
        c[tid] = (a[tid] * s + b[tid]) ** 2.0

    jax_func = jax_ad_kernel(
        scale_sum_square_kernel, num_outputs=1, static_argnames=("s",), vmap_method="broadcast_all"
    )

    def per_elem_loss(a, b):
        out = jax_func(a, b, 2.0)[0]
        return jp.sum(out)

    B, M, N = 2, 3, 6
    a = jp.arange(B * M * N, dtype=jp.float32).reshape((B, M, N))
    b = jp.ones((B, M, N), dtype=jp.float32)

    with jax.default_device(wp.device_to_jax(device)):
        grad_fun = jax.vmap(jax.vmap(jax.grad(per_elem_loss, argnums=(0, 1))))
        da, db = grad_fun(a, b)

    a_np = np.arange(B * M * N, dtype=np.float32).reshape((B, M, N))
    b_np = np.ones((B, M, N), dtype=np.float32)
    s = 2.0
    ref_da = 2.0 * (a_np * s + b_np) * s
    ref_db = 2.0 * (a_np * s + b_np)

    assert_np_equal(np.asarray(da), ref_da)
    assert_np_equal(np.asarray(db), ref_db)


@unittest.skipUnless(_jax_version() >= (0, 4, 31), "Jax version too old for FFI custom_vjp")
def test_jax_ad_kernel_vmap_2d_broadcast_all(test, device):
    import jax
    import jax.numpy as jp

    from warp.jax_experimental.ffi import jax_ad_kernel

    @wp.kernel
    def scale_2d(a: wp.array2d(dtype=float), s: float, out: wp.array2d(dtype=float)):
        i, j = wp.tid()
        out[i, j] = a[i, j] * s

    jax_func = jax_ad_kernel(scale_2d, num_outputs=1, static_argnames=("s",), vmap_method="broadcast_all")

    def per_sample_loss(a):
        out = jax_func(a, 3.0)[0]
        return jp.sum(out)

    B, H, W = 2, 4, 5
    a = jp.arange(B * H * W, dtype=jp.float32).reshape((B, H, W))

    with jax.default_device(wp.device_to_jax(device)):
        (da,) = jax.vmap(jax.grad(per_sample_loss, argnums=(0,)), in_axes=(0,))(a)

    ref = np.full((B, H, W), 3.0, dtype=np.float32)
    assert_np_equal(np.asarray(da), ref)


@unittest.skipUnless(_jax_version() >= (0, 4, 31), "Jax version too old for FFI custom_vjp")
def test_jax_ad_kernel_vmap_2d_expand_dims(test, device):
    import jax
    import jax.numpy as jp

    from warp.jax_experimental.ffi import jax_ad_kernel

    @wp.kernel
    def add_one_2d(a: wp.array2d(dtype=float), out: wp.array2d(dtype=float)):
        i, j = wp.tid()
        out[i, j] = a[i, j] + 1.0

    jax_func = jax_ad_kernel(add_one_2d, num_outputs=1, vmap_method="broadcast_all")

    def per_sample_loss(a):
        out = jax_func(a)[0]
        return jp.sum(out)

    B, H, W = 2, 3, 4
    a = jp.arange(B * H * W, dtype=jp.float32).reshape((B, H, W))

    with jax.default_device(wp.device_to_jax(device)):
        (da,) = jax.vmap(jax.grad(per_sample_loss, argnums=(0,)), in_axes=(0,))(a)

    ref = np.ones((B, H, W), dtype=np.float32)
    assert_np_equal(np.asarray(da), ref)


@unittest.skipUnless(_jax_version() >= (0, 4, 31), "Jax version too old for FFI custom_vjp")
def test_jax_ad_kernel_launch_dim_and_output_dims(test, device):
    import jax
    import jax.numpy as jp

    from warp.jax_experimental.ffi import jax_ad_kernel

    n = 32

    @wp.kernel
    def add_one_masked(x: wp.array(dtype=float), mask: wp.array(dtype=bool), out: wp.array(dtype=float)):
        tid = wp.tid()
        if mask[tid]:
            out[tid] = x[tid] + 1.0
        else:
            out[tid] = x[tid]

    jax_fn = jax_ad_kernel(add_one_masked, num_outputs=1, launch_dim_arg_index=1, output_dims=(n,))

    x = jp.arange(n, dtype=jp.float32)
    mask = jp.arange(n) % 2 == 0

    with jax.default_device(wp.device_to_jax(device)):
        (y,) = jax.jit(lambda a, m: jax_fn(a, m))(x, mask)

        def loss_fn(a, m):
            y = jax_fn(a, m)[0]
            return jp.sum(y)

        grads = jax.jit(jax.grad(loss_fn))(x, mask)

    y_np = np.asarray(y)
    ref_y = np.where(np.asarray(mask), np.asarray(x) + 1.0, np.asarray(x))
    test.assertTrue(np.allclose(y_np, ref_y, rtol=1e-6, atol=1e-6))

    grads_np = np.asarray(grads)
    ref_grads = np.ones_like(np.asarray(x), dtype=np.float32)
    test.assertTrue(np.allclose(grads_np, ref_grads, rtol=1e-6, atol=1e-6))


@unittest.skipUnless(_jax_version() >= (0, 4, 31), "Jax version too old for FFI custom_vjp")
def test_jax_ad_kernel_auto_static_argnames(test, device):
    import jax
    import jax.numpy as jp

    from warp.jax_experimental.ffi import jax_ad_kernel

    @wp.kernel
    def scale_sum_square_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float), s: float, c: wp.array(dtype=float)):
        tid = wp.tid()
        c[tid] = (a[tid] * s + b[tid]) ** 2.0

    # Omit static_argnames to exercise auto-detection of scalar statics
    jax_func = jax_ad_kernel(scale_sum_square_kernel, num_outputs=1)

    def loss(a, b, s):
        out = jax_func(a, b, s)[0]
        return jp.sum(out)

    n = 20
    a = jp.arange(n, dtype=jp.float32)
    b = jp.ones(n, dtype=jp.float32)
    s = 1.5

    with jax.default_device(wp.device_to_jax(device)):
        da, db = jax.grad(loss, argnums=(0, 1))(a, b, s)

    a_np = np.arange(n, dtype=np.float32)
    b_np = np.ones(n, dtype=np.float32)
    ref_da = 2.0 * (a_np * s + b_np) * s
    ref_db = 2.0 * (a_np * s + b_np)

    assert_np_equal(np.asarray(da), ref_da)
    assert_np_equal(np.asarray(db), ref_db)


@unittest.skipUnless(_jax_version() >= (0, 4, 31), "Jax version too old for shard_map test")
def test_jax_ad_kernel_shard_map_mul2(test, device):
    import jax
    import jax.numpy as jp
    from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as P

    from warp.jax_experimental.ffi import jax_ad_kernel

    if jax.local_device_count() < 2:
        test.skipTest("requires >= 2 local devices")

    @wp.kernel
    def mul2(a: wp.array(dtype=float), out: wp.array(dtype=float)):
        tid = wp.tid()
        out[tid] = 2.0 * a[tid]

    jax_mul = jax_ad_kernel(mul2, num_outputs=1)

    devices = jax.local_devices()
    mesh = jax.sharding.Mesh(np.array(devices), "x")

    n = jax.local_device_count() * 5
    sharding = jax.sharding.NamedSharding(mesh, P("x"))
    a = jp.arange(n, dtype=jp.float32)
    shape = (n,)
    arrays = [jax.device_put(a[idx], d) for d, idx in sharding.addressable_devices_indices_map(shape).items()]
    x = jax.make_array_from_single_device_arrays(shape, sharding, arrays)

    def loss(x):
        y = shard_map(lambda v: jax_mul(v)[0], mesh=mesh, in_specs=(P("x"),), out_specs=P("x"), check_rep=False)(x)
        return jp.sum(y)

    g = jax.grad(loss)(x)
    test.assertTrue(np.allclose(np.asarray(g), np.full(n, 2.0, dtype=np.float32), rtol=1e-5, atol=1e-6))


@unittest.skipUnless(_jax_version() >= (0, 4, 31), "Jax version too old for pmap test")
def test_jax_ad_kernel_pmap_mul2(test, device):
    import jax
    import jax.numpy as jp

    from warp.jax_experimental.ffi import jax_ad_kernel

    if jax.local_device_count() < 2:
        test.skipTest("requires >= 2 local devices")

    @wp.kernel
    def mul2(a: wp.array(dtype=float), out: wp.array(dtype=float)):
        tid = wp.tid()
        out[tid] = 2.0 * a[tid]

    jax_mul = jax_ad_kernel(mul2, num_outputs=1)

    per_device = 6
    ndev = jax.local_device_count()
    x = jp.arange(ndev * per_device, dtype=jp.float32).reshape((ndev, per_device))

    def per_device_loss(x):
        y = jax_mul(x)[0]
        return jp.sum(y)

    grads = jax.pmap(jax.grad(per_device_loss))(x)
    test.assertTrue(
        np.allclose(np.asarray(grads), np.full((ndev, per_device), 2.0, dtype=np.float32), rtol=1e-5, atol=1e-6)
    )


@unittest.skipUnless(_jax_version() >= (0, 4, 31), "Jax version too old for shard_map multi-output test")
def test_jax_ad_kernel_shard_map_multi_output(test, device):
    import jax
    import jax.numpy as jp
    from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as P

    from warp.jax_experimental.ffi import jax_ad_kernel

    if jax.local_device_count() < 2:
        test.skipTest("requires >= 2 local devices")

    @wp.kernel
    def multi_output(
        a: wp.array(dtype=float), b: wp.array(dtype=float), s: float, c: wp.array(dtype=float), d: wp.array(dtype=float)
    ):
        tid = wp.tid()
        c[tid] = a[tid] * a[tid]
        d[tid] = a[tid] * b[tid] * s

    jax_mo = jax_ad_kernel(multi_output, num_outputs=2, static_argnames=("s",))

    devices = jax.local_devices()
    mesh = jax.sharding.Mesh(np.array(devices), "x")

    n = jax.local_device_count() * 5
    sharding = jax.sharding.NamedSharding(mesh, P("x"))
    a = jp.arange(n, dtype=jp.float32)
    b = jp.arange(n, dtype=jp.float32)
    shape = (n,)
    arrays_a = [jax.device_put(a[idx], d) for d, idx in sharding.addressable_devices_indices_map(shape).items()]
    arrays_b = [jax.device_put(b[idx], d) for d, idx in sharding.addressable_devices_indices_map(shape).items()]
    ashard = jax.make_array_from_single_device_arrays(shape, sharding, arrays_a)
    bshard = jax.make_array_from_single_device_arrays(shape, sharding, arrays_b)

    s = 2.0

    def body(aa, bb):
        c, d = jax_mo(aa, bb, s)
        return c + d

    def loss(aa, bb):
        y = shard_map(body, mesh=mesh, in_specs=(P("x"), P("x")), out_specs=P("x"), check_rep=False)(aa, bb)
        return jp.sum(y)

    da, db = jax.grad(loss, argnums=(0, 1))(ashard, bshard)
    a_np = np.arange(n, dtype=np.float32)
    b_np = np.arange(n, dtype=np.float32)
    ref_da = 2.0 * a_np + b_np * s
    ref_db = a_np * s
    assert_np_equal(np.asarray(da), ref_da)
    assert_np_equal(np.asarray(db), ref_db)


@unittest.skipUnless(_jax_version() >= (0, 4, 31), "Jax version too old for pmap multi-output test")
def test_jax_ad_kernel_pmap_multi_output(test, device):
    import jax
    import jax.numpy as jp

    from warp.jax_experimental.ffi import jax_ad_kernel

    if jax.local_device_count() < 2:
        test.skipTest("requires >= 2 local devices")

    @wp.kernel
    def multi_output(
        a: wp.array(dtype=float), b: wp.array(dtype=float), s: float, c: wp.array(dtype=float), d: wp.array(dtype=float)
    ):
        tid = wp.tid()
        c[tid] = a[tid] * a[tid]
        d[tid] = a[tid] * b[tid] * s

    jax_mo = jax_ad_kernel(multi_output, num_outputs=2, static_argnames=("s",))

    per_device = 5
    ndev = jax.local_device_count()
    a = jp.arange(ndev * per_device, dtype=jp.float32).reshape((ndev, per_device))
    b = jp.arange(ndev * per_device, dtype=jp.float32).reshape((ndev, per_device))
    s = 2.0

    def per_dev_loss(aa, bb):
        c, d = jax_mo(aa, bb, s)
        return jp.sum(c + d)

    da, db = jax.pmap(jax.grad(per_dev_loss, argnums=(0, 1)))(a, b)

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

        add_function_test(
            TestJax, "test_jax_ad_kernel_simple", test_jax_ad_kernel_simple, devices=jax_compatible_cuda_devices
        )

        add_function_test(
            TestJax,
            "test_jax_ad_kernel_multi_output",
            test_jax_ad_kernel_multi_output,
            devices=jax_compatible_cuda_devices,
        )
        add_function_test(
            TestJax, "test_jax_ad_kernel_vec2", test_jax_ad_kernel_vec2, devices=jax_compatible_cuda_devices
        )
        add_function_test(TestJax, "test_jax_ad_kernel_2d", test_jax_ad_kernel_2d, devices=jax_compatible_cuda_devices)
        add_function_test(
            TestJax, "test_jax_ad_kernel_mat22", test_jax_ad_kernel_mat22, devices=jax_compatible_cuda_devices
        )
        add_function_test(
            TestJax,
            "test_jax_ad_kernel_rgb_hw_not_c",
            test_jax_ad_kernel_rgb_hw_not_c,
            devices=jax_compatible_cuda_devices,
        )
        add_function_test(
            TestJax,
            "test_jax_ad_kernel_vmap_simple",
            test_jax_ad_kernel_vmap_simple,
            devices=jax_compatible_cuda_devices,
        )
        add_function_test(
            TestJax,
            "test_jax_ad_kernel_auto_static_argnames",
            test_jax_ad_kernel_auto_static_argnames,
            devices=jax_compatible_cuda_devices,
        )
        add_function_test(
            TestJax, "test_jax_ad_kernel_vmap_vec2", test_jax_ad_kernel_vmap_vec2, devices=jax_compatible_cuda_devices
        )
        add_function_test(
            TestJax,
            "test_jax_ad_kernel_vmap_multi_output",
            test_jax_ad_kernel_vmap_multi_output,
            devices=jax_compatible_cuda_devices,
        )
        add_function_test(
            TestJax,
            "test_jax_ad_kernel_vmap_broadcast_all_simple",
            test_jax_ad_kernel_vmap_broadcast_all_simple,
            devices=jax_compatible_cuda_devices,
        )
        add_function_test(
            TestJax,
            "test_jax_ad_kernel_vmap_expand_dims_simple",
            test_jax_ad_kernel_vmap_expand_dims_simple,
            devices=jax_compatible_cuda_devices,
        )
        add_function_test(
            TestJax,
            "test_jax_ad_kernel_vmap_broadcast_mismatched_inputs",
            test_jax_ad_kernel_vmap_broadcast_mismatched_inputs,
            devices=jax_compatible_cuda_devices,
        )
        add_function_test(
            TestJax,
            "test_jax_ad_kernel_vmap_double_batch",
            test_jax_ad_kernel_vmap_double_batch,
            devices=jax_compatible_cuda_devices,
        )
        add_function_test(
            TestJax,
            "test_jax_ad_kernel_vmap_2d_broadcast_all",
            test_jax_ad_kernel_vmap_2d_broadcast_all,
            devices=jax_compatible_cuda_devices,
        )
        add_function_test(
            TestJax,
            "test_jax_ad_kernel_vmap_2d_expand_dims",
            test_jax_ad_kernel_vmap_2d_expand_dims,
            devices=jax_compatible_cuda_devices,
        )

        add_function_test(
            TestJax,
            "test_jax_ad_kernel_jit_of_grad_simple",
            test_jax_ad_kernel_jit_of_grad_simple,
            devices=jax_compatible_cuda_devices,
        )
        add_function_test(
            TestJax,
            "test_jax_ad_kernel_jit_of_grad_multi_output",
            test_jax_ad_kernel_jit_of_grad_multi_output,
            devices=jax_compatible_cuda_devices,
        )

        add_function_test(
            TestJax,
            "test_jax_ad_kernel_pmap_mul2",
            test_jax_ad_kernel_pmap_mul2,
            devices=jax_compatible_cuda_devices,
        )
        add_function_test(
            TestJax,
            "test_jax_ad_kernel_shard_map_mul2",
            test_jax_ad_kernel_shard_map_mul2,
            devices=jax_compatible_cuda_devices,
        )

        add_function_test(
            TestJax,
            "test_jax_ad_kernel_pmap_multi_output",
            test_jax_ad_kernel_pmap_multi_output,
            devices=jax_compatible_cuda_devices,
        )

        add_function_test(
            TestJax,
            "test_jax_ad_kernel_pmap_multi_output",
            test_jax_ad_kernel_pmap_multi_output,
            devices=jax_compatible_cuda_devices,
        )
        add_function_test(
            TestJax,
            "test_jax_ad_kernel_launch_dim_and_output_dims",
            test_jax_ad_kernel_launch_dim_and_output_dims,
            devices=jax_compatible_cuda_devices,
        )


except Exception as e:
    print(f"Skipping Jax tests due to exception: {e}")


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
