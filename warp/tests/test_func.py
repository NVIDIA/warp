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


@wp.func
def sqr(x: float):
    return x * x


# test nested user function calls
# and explicit return type hints
@wp.func
def cube(x: float) -> float:
    return sqr(x) * x


@wp.func
def custom(x: int):
    return x + 1


@wp.func
def custom(x: float):
    return x + 1.0


@wp.func
def custom(x: wp.vec3):
    return x + wp.vec3(1.0, 0.0, 0.0)


@wp.func
def noreturn(x: wp.vec3):
    x = x + wp.vec3(0.0, 1.0, 0.0)

    wp.expect_eq(x, wp.vec3(1.0, 1.0, 0.0))


@wp.kernel
def test_overload_func():
    # tests overloading a custom @wp.func

    i = custom(1)
    f = custom(1.0)
    v = custom(wp.vec3(1.0, 0.0, 0.0))

    wp.expect_eq(i, 2)
    wp.expect_eq(f, 2.0)
    wp.expect_eq(v, wp.vec3(2.0, 0.0, 0.0))

    noreturn(wp.vec3(1.0, 0.0, 0.0))


@wp.func
def foo(x: int):
    # This shouldn't be picked up.
    return x * 2


@wp.func
def foo(x: int):
    return x * 3


@wp.kernel
def test_override_func():
    i = foo(1)
    wp.expect_eq(i, 3)


def test_native_func_export(test, device):
    # tests calling native functions from Python

    q = wp.quat(0.0, 0.0, 0.0, 1.0)
    assert_np_equal(np.array([*q]), np.array([0.0, 0.0, 0.0, 1.0]))

    r = wp.quat_from_axis_angle((1.0, 0.0, 0.0), 2.0)
    assert_np_equal(np.array([*r]), np.array([0.8414709568023682, 0.0, 0.0, 0.5403022170066833]), tol=1.0e-3)

    q = wp.quat(1.0, 2.0, 3.0, 4.0)
    q = wp.normalize(q) * 2.0
    assert_np_equal(
        np.array([*q]),
        np.array([0.18257418274879456, 0.3651483654975891, 0.547722578048706, 0.7302967309951782]) * 2.0,
        tol=1.0e-3,
    )

    v2 = wp.vec2(1.0, 2.0)
    v2 = wp.normalize(v2) * 2.0
    assert_np_equal(np.array([*v2]), np.array([0.4472135901451111, 0.8944271802902222]) * 2.0, tol=1.0e-3)

    v3 = wp.vec3(1.0, 2.0, 3.0)
    v3 = wp.normalize(v3) * 2.0
    assert_np_equal(
        np.array([*v3]), np.array([0.26726123690605164, 0.5345224738121033, 0.8017836809158325]) * 2.0, tol=1.0e-3
    )

    v4 = wp.vec4(1.0, 2.0, 3.0, 4.0)
    v4 = wp.normalize(v4) * 2.0
    assert_np_equal(
        np.array([*v4]),
        np.array([0.18257418274879456, 0.3651483654975891, 0.547722578048706, 0.7302967309951782]) * 2.0,
        tol=1.0e-3,
    )

    m22 = wp.mat22(1.0, 2.0, 3.0, 4.0)
    m22 = m22 + m22

    test.assertEqual(m22[1, 1], 8.0)
    test.assertEqual(str(m22), "[[2.0, 4.0],\n [6.0, 8.0]]")

    t = wp.transform(
        wp.vec3(0.0, 0.0, 0.0),
        wp.quat(0.0, 0.0, 0.0, 1.0),
    )
    assert_np_equal(np.array([*t]), np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))

    f = wp.sin(math.pi * 0.5)
    test.assertAlmostEqual(f, 1.0, places=3)


def test_user_func_export(test, device):
    # tests calling overloaded user-defined functions from Python
    i = custom(1)
    f = custom(1.0)
    v = custom(wp.vec3(1.0, 0.0, 0.0))

    test.assertEqual(i, 2)
    test.assertEqual(f, 2.0)
    assert_np_equal(np.array([*v]), np.array([2.0, 0.0, 0.0]))


def test_func_closure_capture(test, device):
    def make_closure_kernel(func):
        def closure_kernel_fn(data: wp.array(dtype=float), expected: float):
            f = func(data[wp.tid()])
            wp.expect_eq(f, expected)

        key = f"test_func_closure_capture_{func.key}"
        return wp.Kernel(func=closure_kernel_fn, key=key, module=wp.get_module(closure_kernel_fn.__module__))

    sqr_closure = make_closure_kernel(sqr)
    cube_closure = make_closure_kernel(cube)

    data = wp.array([2.0], dtype=float, device=device)
    expected_sqr = 4.0
    expected_cube = 8.0

    wp.launch(sqr_closure, dim=data.shape, inputs=[data, expected_sqr], device=device)
    wp.launch(cube_closure, dim=data.shape, inputs=[data, expected_cube], device=device)


@wp.func
def test_func(param1: wp.int32, param2: wp.int32, param3: wp.int32) -> wp.float32:
    return 1.0


@wp.kernel
def test_return_kernel(test_data: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    test_data[tid] = wp.lerp(test_func(0, 1, 2), test_func(0, 1, 2), 0.5)


def test_return_func(test, device):
    test_data = wp.zeros(100, dtype=wp.float32, device=device)
    wp.launch(kernel=test_return_kernel, dim=test_data.size, inputs=[test_data], device=device)


@wp.func
def multi_valued_func(a: wp.float32, b: wp.float32):
    return a + b, a - b, a * b, a / b


def test_multi_valued_func(test, device):
    @wp.kernel
    def test_multi_valued_kernel(test_data1: wp.array(dtype=wp.float32), test_data2: wp.array(dtype=wp.float32)):
        tid = wp.tid()
        d1, d2 = test_data1[tid], test_data2[tid]
        a, b, c, d = multi_valued_func(d1, d2)
        wp.expect_eq(a, d1 + d2)
        wp.expect_eq(b, d1 - d2)
        wp.expect_eq(c, d1 * d2)
        wp.expect_eq(d, d1 / d2)

    test_data1 = wp.array(np.arange(100), dtype=wp.float32, device=device)
    test_data2 = wp.array(np.arange(100, 0, -1), dtype=wp.float32, device=device)
    wp.launch(kernel=test_multi_valued_kernel, dim=test_data1.size, inputs=[test_data1, test_data2], device=device)


@wp.kernel
def test_func_defaults():
    # test default as expected
    wp.expect_near(1.0, 1.0 + 1.0e-6)

    # test that changing tolerance still works
    wp.expect_near(1.0, 1.1, 0.5)


def register(parent):
    devices = get_test_devices()

    class TestFunc(parent):
        pass

    add_kernel_test(TestFunc, kernel=test_overload_func, name="test_overload_func", dim=1, devices=devices)
    add_function_test(TestFunc, func=test_return_func, name="test_return_func", devices=devices)
    add_kernel_test(TestFunc, kernel=test_override_func, name="test_override_func", dim=1, devices=devices)
    add_function_test(TestFunc, func=test_native_func_export, name="test_native_func_export", devices=["cpu"])
    add_function_test(TestFunc, func=test_user_func_export, name="test_user_func_export", devices=["cpu"])
    add_function_test(TestFunc, func=test_func_closure_capture, name="test_func_closure_capture", devices=devices)
    add_function_test(TestFunc, func=test_multi_valued_func, name="test_multi_valued_func", devices=devices)
    add_kernel_test(TestFunc, kernel=test_func_defaults, name="test_func_defaults", dim=1, devices=devices)

    return TestFunc


if __name__ == "__main__":
    c = register(unittest.TestCase)
    wp.force_load()

    unittest.main(verbosity=2)
