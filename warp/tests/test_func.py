# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import math
import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


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


def test_func_closure_capture(test, device):
    def make_closure_kernel(func):
        def closure_kernel_fn(data: wp.array(dtype=float), expected: float):
            f = func(data[wp.tid()])
            wp.expect_eq(f, expected)

        return wp.Kernel(func=closure_kernel_fn)

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


@wp.func
def sign(x: float):
    return 123.0


@wp.kernel
def test_builtin_shadowing():
    wp.expect_eq(sign(1.23), 123.0)


devices = get_test_devices()


class TestFunc(unittest.TestCase):
    def test_user_func_export(self):
        # tests calling overloaded user-defined functions from Python
        i = custom(1)
        f = custom(1.0)
        v = custom(wp.vec3(1.0, 0.0, 0.0))

        self.assertEqual(i, 2)
        self.assertEqual(f, 2.0)
        assert_np_equal(np.array([*v]), np.array([2.0, 0.0, 0.0]))

    def test_native_func_export(self):
        # tests calling native functions from Python

        q = wp.quat(0.0, 0.0, 0.0, 1.0)
        assert_np_equal(np.array([*q]), np.array([0.0, 0.0, 0.0, 1.0]))

        r = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 2.0)
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

        v = wp.vec2(0.0)
        v += wp.vec2(1.0, 1.0)
        assert v == wp.vec2(1.0, 1.0)
        v -= wp.vec2(1.0, 1.0)
        assert v == wp.vec2(0.0, 0.0)
        v = wp.vec2(2.0, 2.0) - wp.vec2(1.0, 1.0)
        assert v == wp.vec2(1.0, 1.0)
        v *= 2.0
        assert v == wp.vec2(2.0, 2.0)
        v = v * 2.0
        assert v == wp.vec2(4.0, 4.0)
        v = v / 2.0
        assert v == wp.vec2(2.0, 2.0)
        v /= 2.0
        assert v == wp.vec2(1.0, 1.0)
        v = -v
        assert v == wp.vec2(-1.0, -1.0)
        v = +v
        assert v == wp.vec2(-1.0, -1.0)

        m22 = wp.mat22(1.0, 2.0, 3.0, 4.0)
        m22 = m22 + m22

        self.assertEqual(m22[1, 1], 8.0)
        self.assertEqual(str(m22), "[[2.0, 4.0],\n [6.0, 8.0]]")

        t = wp.transform(
            wp.vec3(1.0, 2.0, 3.0),
            wp.quat(4.0, 5.0, 6.0, 7.0),
        )
        self.assertSequenceEqual(t, (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0))
        self.assertSequenceEqual(
            t * wp.transform(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0), (396.0, 432.0, 720.0, 56.0, 70.0, 84.0, -28.0)
        )
        self.assertSequenceEqual(
            t * wp.transform((1.0, 2.0, 3.0), (4.0, 5.0, 6.0, 7.0)), (396.0, 432.0, 720.0, 56.0, 70.0, 84.0, -28.0)
        )

        t = wp.transform()
        self.assertSequenceEqual(t, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))

        t = wp.transform(p=(1.0, 2.0, 3.0), q=(4.0, 5.0, 6.0, 7.0))
        self.assertSequenceEqual(t, (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0))

        t = wp.transform(q=(4.0, 5.0, 6.0, 7.0), p=(1.0, 2.0, 3.0))
        self.assertSequenceEqual(t, (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0))

        t = wp.transform((1.0, 2.0, 3.0), q=(4.0, 5.0, 6.0, 7.0))
        self.assertSequenceEqual(t, (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0))

        t = wp.transform(p=(1.0, 2.0, 3.0))
        self.assertSequenceEqual(t, (1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0))

        t = wp.transform(q=(4.0, 5.0, 6.0, 7.0))
        self.assertSequenceEqual(t, (0.0, 0.0, 0.0, 4.0, 5.0, 6.0, 7.0))

        t = wp.transform((1.0, 2.0, 3.0), (4.0, 5.0, 6.0, 7.0))
        self.assertSequenceEqual(t, (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0))

        t = wp.transform(p=wp.vec3(1.0, 2.0, 3.0), q=wp.quat(4.0, 5.0, 6.0, 7.0))
        self.assertSequenceEqual(t, (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0))

        t = wp.transform(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
        self.assertSequenceEqual(t, (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0))

        t = wp.transform(wp.transform(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0))
        self.assertSequenceEqual(t, (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0))

        t = wp.transform(*wp.transform(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0))
        self.assertSequenceEqual(t, (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0))

        transformf = wp.types.transformation(dtype=float)

        t = wp.transformf((1.0, 2.0, 3.0), (4.0, 5.0, 6.0, 7.0))
        self.assertSequenceEqual(
            t + transformf((2.0, 3.0, 4.0), (5.0, 6.0, 7.0, 8.0)),
            (3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0),
        )
        self.assertSequenceEqual(
            t - transformf((2.0, 3.0, 4.0), (5.0, 6.0, 7.0, 8.0)),
            (-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0),
        )

        f = wp.sin(math.pi * 0.5)
        self.assertAlmostEqual(f, 1.0, places=3)

        m = wp.mat22(0.0, 0.0, 0.0, 0.0)
        m += wp.mat22(1.0, 1.0, 1.0, 1.0)
        assert m == wp.mat22(1.0, 1.0, 1.0, 1.0)
        m -= wp.mat22(1.0, 1.0, 1.0, 1.0)
        assert m == wp.mat22(0.0, 0.0, 0.0, 0.0)
        m = wp.mat22(2.0, 2.0, 2.0, 2.0) - wp.mat22(1.0, 1.0, 1.0, 1.0)
        assert m == wp.mat22(1.0, 1.0, 1.0, 1.0)
        m *= 2.0
        assert m == wp.mat22(2.0, 2.0, 2.0, 2.0)
        m = m * 2.0
        assert m == wp.mat22(4.0, 4.0, 4.0, 4.0)
        m = m / 2.0
        assert m == wp.mat22(2.0, 2.0, 2.0, 2.0)
        m /= 2.0
        assert m == wp.mat22(1.0, 1.0, 1.0, 1.0)
        m = -m
        assert m == wp.mat22(-1.0, -1.0, -1.0, -1.0)
        m = +m
        assert m == wp.mat22(-1.0, -1.0, -1.0, -1.0)
        m = m * m
        assert m == wp.mat22(2.0, 2.0, 2.0, 2.0)

    def test_native_function_error_resolution(self):
        a = wp.mat22f(1.0, 2.0, 3.0, 4.0)
        b = wp.mat22d(1.0, 2.0, 3.0, 4.0)
        with self.assertRaisesRegex(
            RuntimeError,
            r"^Couldn't find a function 'mul' compatible with " r"the arguments 'mat22f, mat22d'$",
        ):
            a * b


add_kernel_test(TestFunc, kernel=test_overload_func, name="test_overload_func", dim=1, devices=devices)
add_function_test(TestFunc, func=test_return_func, name="test_return_func", devices=devices)
add_kernel_test(TestFunc, kernel=test_override_func, name="test_override_func", dim=1, devices=devices)
add_function_test(TestFunc, func=test_func_closure_capture, name="test_func_closure_capture", devices=devices)
add_function_test(TestFunc, func=test_multi_valued_func, name="test_multi_valued_func", devices=devices)
add_kernel_test(TestFunc, kernel=test_func_defaults, name="test_func_defaults", dim=1, devices=devices)
add_kernel_test(TestFunc, kernel=test_builtin_shadowing, name="test_builtin_shadowing", dim=1, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
