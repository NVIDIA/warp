# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

from warp._src.context import Function, get_builtin_call_desc
from warp.tests.unittest_utils import *


def make_mul_builtin(input_types, value_type, defaults=None, export_func=None):
    """Create an isolated built-in for testing Python-scope overload resolution.

    The real ``wp.mul`` overloads all use the same ``(a, b)`` calling form
    without defaults, so they cannot exercise overload-specific parameter
    names, default values, or compile-time-only arguments. This helper
    constructs synthetic Python signatures backed by the existing native
    multiply exports without modifying the global ``wp.mul`` overload group.
    """
    return Function(
        func=None,
        key="mul",
        namespace="wp::",
        input_types=input_types,
        value_func=lambda arg_types, arg_values: value_type,
        export_func=export_func,
        export=True,
        defaults=defaults,
    )


def nps(dtype, value):
    """Create a NumPy scalar value based on the given data type."""
    # Workaround to avoid deprecation warning messages for integer overflows.
    return np.array((value,)).astype(dtype)[0]


def test_int_arg_support(test, device, dtype):
    np_type = wp.dtype_to_numpy(dtype)
    value = -1234567890123456789
    expected = wp.invert(dtype(value))

    test.assertEqual(wp.invert(nps(np_type, value)), expected)


def test_float_arg_support(test, device, dtype):
    np_type = wp.dtype_to_numpy(dtype)
    value = 1.23
    expected = wp.sin(dtype(value))

    if dtype is wp.bfloat16:
        # NumPy has no native bfloat16; dtype_to_numpy returns uint16 which
        # cannot be interpreted as a float by Warp builtins, so skip the
        # NumPy scalar round-trip test.
        return

    test.assertEqual(wp.sin(nps(np_type, value)), expected)


def test_int_int_args_support(test, device, dtype):
    np_type = wp.dtype_to_numpy(dtype)
    value = -1234567890
    expected = wp.mul(dtype(value), dtype(value))

    test.assertEqual(wp.mul(dtype(value), dtype(value)), expected)
    test.assertEqual(wp.mul(dtype(value), nps(np_type, value)), expected)

    test.assertEqual(wp.mul(nps(np_type, value), dtype(value)), expected)
    test.assertEqual(wp.mul(nps(np_type, value), nps(np_type, value)), expected)

    if dtype is wp.int32:
        test.assertEqual(wp.mul(dtype(value), value), expected)
        test.assertEqual(wp.mul(nps(np_type, value), value), expected)
        test.assertEqual(wp.mul(value, value), expected)

        test.assertEqual(wp.mul(value, dtype(value)), expected)
        test.assertEqual(wp.mul(value, nps(np_type, value)), expected)
    else:
        with test.assertRaisesRegex(
            RuntimeError,
            rf"Couldn't find a function 'mul' compatible with the arguments '{dtype.__name__}, int'$",
        ):
            wp.mul(dtype(value), value)

        with test.assertRaisesRegex(
            RuntimeError,
            rf"Couldn't find a function 'mul' compatible with the arguments '{np_type.__name__}, int'$",
        ):
            wp.mul(nps(np_type, value), value)

        with test.assertRaisesRegex(
            RuntimeError,
            rf"Couldn't find a function 'mul' compatible with the arguments 'int, {dtype.__name__}'$",
        ):
            wp.mul(value, dtype(value))

        with test.assertRaisesRegex(
            RuntimeError,
            rf"Couldn't find a function 'mul' compatible with the arguments 'int, {np_type.__name__}'$",
        ):
            wp.mul(value, nps(np_type, value))


class TestBuiltinsResolution(unittest.TestCase):
    def test_builtin_fallback_does_not_retry_primary_shape(self):
        """Evaluate primary-shape overloads only once before falling back."""
        overload_group = make_mul_builtin({"a": wp.float32, "b": wp.float32}, wp.float32)
        overload_group.add_overload(make_mul_builtin({"a": wp.int32, "b": wp.int32}, wp.int32))

        get_builtin_call_desc.cache_clear()
        self.addCleanup(get_builtin_call_desc.cache_clear)

        with self.assertRaisesRegex(
            RuntimeError,
            r"Couldn't find a function 'mul' compatible with the arguments 'mat22f, mat22f'$",
        ):
            overload_group.get_builtin(wp.mat22f(), wp.mat22f())

        cache_info = get_builtin_call_desc.cache_info()
        self.assertEqual(cache_info.hits, 0)
        self.assertEqual(cache_info.misses, 2)

    def test_unavailable_builtin_overloads_are_rejected(self):
        """Reject unavailable overloads before looking up native symbols."""
        cases = (
            ("missing native export", wp.mat22f(), "mat22f"),
            ("incompatible argument", wp.mat22f, "PyCArrayType"),
        )

        for name, value, type_name in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(
                    RuntimeError,
                    rf"^Couldn't find a function 'index' compatible with the arguments '{type_name}, int, int'$",
                ):
                    wp.index(value, 0, 1)

    def test_builtin_with_compile_time_only_default(self):
        """Call Python-scope built-ins with compile-time-only default parameters."""
        cases = (
            ("quat_identity()", wp.quat_identity, {}, (0.0, 0.0, 0.0, 1.0)),
            ("quat_identity(dtype=None)", wp.quat_identity, {"dtype": None}, (0.0, 0.0, 0.0, 1.0)),
            (
                "transform_identity()",
                wp.transform_identity,
                {},
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            ),
            (
                "transform_identity(dtype=None)",
                wp.transform_identity,
                {"dtype": None},
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            ),
        )

        for name, builtin, kwargs, expected in cases:
            with self.subTest(name=name):
                result = builtin(**kwargs)
                np.testing.assert_allclose(result, expected)

        multiply = make_mul_builtin(
            {"value": wp.vec3f, "factor": wp.float32, "dtype": wp.float32},
            wp.vec3f,
            export_func=lambda input_types: {k: v for k, v in input_types.items() if k != "dtype"},
            defaults={"dtype": None},
        )

        for dtype_kwargs in ({}, {"dtype": None}):
            with self.subTest(name="runtime arguments", dtype_kwargs=dtype_kwargs):
                result = multiply(wp.vec3f(1.0, 1.0, 1.0), 4.0, **dtype_kwargs)
                np.testing.assert_allclose(result, (4.0, 4.0, 4.0))

    def test_builtin_overloads_with_different_default_values(self):
        """Apply default values from the selected Python-scope built-in overload."""
        overload_group = make_mul_builtin(
            {"value": wp.vec2f, "factor": wp.float32},
            wp.vec2f,
            defaults={"factor": 2.0},
        )
        overload_group.add_overload(
            make_mul_builtin(
                {"vector": wp.vec3f, "scale": wp.float32},
                wp.vec3f,
                defaults={"scale": 3.0},
            )
        )

        cases = (
            ("primary default", {"value": wp.vec2f(1.0, 1.0)}, (2.0, 2.0)),
            ("overload default", {"vector": wp.vec3f(1.0, 1.0, 1.0)}, (3.0, 3.0, 3.0)),
            ("explicit override", {"vector": wp.vec3f(1.0, 1.0, 1.0), "scale": 4.0}, (4.0, 4.0, 4.0)),
        )

        for name, kwargs, expected in cases:
            with self.subTest(name=name):
                result = overload_group(**kwargs)
                np.testing.assert_allclose(result, expected)

        with self.assertRaisesRegex(
            RuntimeError,
            r"Couldn't find a function 'mul' compatible with the arguments 'vec3f'$",
        ):
            overload_group(value=wp.vec3f(1.0, 1.0, 1.0))

    def test_builtin_overload_required_parameter(self):
        """Reject calls missing a parameter required by the selected Python-scope overload."""
        overload_group = make_mul_builtin(
            {"value": wp.vec2f, "factor": wp.float32},
            wp.vec2f,
            defaults={"factor": 2.0},
        )
        overload_group.add_overload(make_mul_builtin({"vector": wp.vec3f, "scale": wp.float32}, wp.vec3f))

        result = overload_group(vector=wp.vec3f(1.0, 1.0, 1.0), scale=3.0)
        np.testing.assert_allclose(result, (3.0, 3.0, 3.0))

        with self.assertRaisesRegex(
            RuntimeError,
            r"Couldn't find a function 'mul' compatible with the arguments 'vec3f'$",
        ):
            overload_group(value=wp.vec3f(1.0, 1.0, 1.0))

    def test_builtin_unary_and_binary_overloads(self):
        """Resolve Python-scope built-ins with unary and binary overloads."""
        value = wp.vec3f(3.0, 1.0, 2.0)

        self.assertEqual(wp.min(value), 1.0)
        self.assertEqual(wp.max(a=value), 3.0)

        with self.assertRaisesRegex(
            RuntimeError,
            r"Couldn't find a function 'min' compatible with the arguments 'vec3f, vec3d'$",
        ):
            wp.min(value, wp.vec3d(3.0, 1.0, 2.0))

    def test_builtin_overload_with_different_parameter_names(self):
        """Verify Python-scope built-in overloads accept shared and distinct parameter names."""
        # fmt: off
        matrix = wp.mat44f(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        )
        # fmt: on
        transform = wp.transformf()
        point = wp.vec3f(1.0, 2.0, 3.0)

        transform_result = wp.transform_point(point=point, xform=transform)
        matrix_result = wp.transform_point(point=point, mat=matrix)

        np.testing.assert_allclose(transform_result, point)
        np.testing.assert_allclose(matrix_result, point)

        # Reject keywords from incompatible overloads.
        with self.assertRaisesRegex(
            RuntimeError,
            r"Couldn't find a function 'transform_point' compatible with the arguments 'vec3f, mat44f'$",
        ):
            wp.transform_point(point=point, xform=matrix)

        with self.assertRaisesRegex(
            RuntimeError,
            r"Couldn't find a function 'transform_point' compatible with the arguments 'vec3f, transformf'$",
        ):
            wp.transform_point(point=point, mat=transform)

    def test_builtin_overload_with_defaults(self):
        """Verify Python-scope built-in overloads apply their own default values."""
        state = wp.rand_init(42)
        position = wp.vec3f(0.25, 0.5, 0.75)

        result = wp.curlnoise(xyz=position, state=state)
        expected = wp.curlnoise(state, position, wp.uint32(1), 2.0, 0.5)

        np.testing.assert_allclose(result, expected)

    def test_builtin_rejects_keywords_from_other_overload(self):
        """Reject keyword arguments belonging to another overload."""
        state = wp.rand_init(42)
        with self.assertRaisesRegex(
            RuntimeError,
            r"Couldn't find a function 'curlnoise' compatible with the arguments 'uint32, vec3f'$",
        ):
            wp.curlnoise(state=state, xy=wp.vec3f(0.5, 0.5, 0.5))

    def test_int_arg_overflow(self):
        value = -1234567890123456789

        self.assertEqual(wp.invert(wp.int8(value)), 20)
        self.assertEqual(wp.invert(wp.int16(value)), -32492)
        self.assertEqual(wp.invert(wp.int32(value)), 2112454932)
        self.assertEqual(wp.invert(wp.int64(value)), 1234567890123456788)

        self.assertEqual(wp.invert(wp.uint8(value)), 20)
        self.assertEqual(wp.invert(wp.uint16(value)), 33044)
        self.assertEqual(wp.invert(wp.uint32(value)), 2112454932)
        self.assertEqual(wp.invert(wp.uint64(value)), 1234567890123456788)

        self.assertEqual(wp.invert(value), wp.invert(wp.int32(value)))

    def test_float_arg_precision(self):
        value = 1.23
        expected = 0.94248880193169748409

        result = wp.sin(wp.float64(value))
        self.assertIsInstance(result, wp.float64)
        self.assertAlmostEqual(float(result), expected, places=12)

        result = wp.sin(wp.float32(value))
        self.assertIsInstance(result, float)
        self.assertNotAlmostEqual(result, expected, places=12)
        self.assertAlmostEqual(result, expected, places=5)

        result = wp.sin(wp.float16(value))
        self.assertIsInstance(result, wp.float16)
        self.assertNotAlmostEqual(float(result), expected, places=5)
        self.assertAlmostEqual(float(result), expected, places=1)

        self.assertEqual(wp.sin(value), wp.sin(wp.float32(value)))

    def test_legacy_scalar_return_types(self):
        old_setting = wp.config.legacy_scalar_return_types

        try:
            wp.config.legacy_scalar_return_types = True

            self.assertIsInstance(wp.sin(wp.float16(1.23)), float)
            self.assertIsInstance(wp.sin(wp.float64(1.23)), float)
            self.assertIsInstance(wp.invert(wp.int16(-123)), int)
        finally:
            wp.config.legacy_scalar_return_types = old_setting

    def test_int_int_args_overflow(self):
        value = -1234567890

        self.assertEqual(wp.mul(wp.int8(value), wp.int8(value)), 68)
        self.assertEqual(wp.mul(wp.int16(value), wp.int16(value)), -3004)
        self.assertEqual(wp.mul(wp.int32(value), wp.int32(value)), 304084036)
        self.assertEqual(wp.mul(wp.int64(value), wp.int64(value)), 1524157875019052100)

        self.assertEqual(wp.mul(wp.uint8(value), wp.uint8(value)), 68)
        self.assertEqual(wp.mul(wp.uint16(value), wp.uint16(value)), 62532)
        self.assertEqual(wp.mul(wp.uint32(value), wp.uint32(value)), 304084036)
        self.assertEqual(wp.mul(wp.uint64(value), wp.uint64(value)), 1524157875019052100)

        self.assertEqual(wp.mul(value, value), wp.mul(wp.int32(value), wp.int32(value)))

    def test_mat22_arg_precision(self):
        values = (1.23, 2.34, 3.45, 4.56)
        expected = 5.78999999999999914735

        result = wp.trace(wp.mat22d(*values))
        self.assertIsInstance(result, wp.float64)
        self.assertAlmostEqual(float(result), expected, places=12)

        result = wp.trace(wp.mat22f(*values))
        self.assertIsInstance(result, float)
        self.assertNotAlmostEqual(result, expected, places=12)
        self.assertAlmostEqual(result, expected, places=5)

        result = wp.trace(wp.mat22h(*values))
        self.assertIsInstance(result, wp.float16)
        self.assertNotAlmostEqual(float(result), expected, places=5)
        self.assertAlmostEqual(float(result), expected, places=1)

    def test_mat33_arg_precision(self):
        values = (1.23, 2.34, 3.45, 4.56, 5.67, 6.78, 7.89, 8.90, 9.01)
        expected = 15.91000000000000014211

        result = wp.trace(wp.mat33d(*values))
        self.assertIsInstance(result, wp.float64)
        self.assertAlmostEqual(float(result), expected, places=12)

        result = wp.trace(wp.mat33f(*values))
        self.assertIsInstance(result, float)
        self.assertNotAlmostEqual(result, expected, places=12)
        self.assertAlmostEqual(result, expected, places=5)

        result = wp.trace(wp.mat33h(*values))
        self.assertIsInstance(result, wp.float16)
        self.assertNotAlmostEqual(float(result), expected, places=5)
        self.assertAlmostEqual(float(result), expected, places=1)

    def test_mat44_arg_precision(self):
        values = (1.23, 2.34, 3.45, 4.56, 5.67, 6.78, 7.89, 8.90, 9.01, 10.12, 11.23, 12.34, 13.45, 14.56, 15.67, 16.78)
        expected = 36.02000000000000312639

        result = wp.trace(wp.mat44d(*values))
        self.assertIsInstance(result, wp.float64)
        self.assertAlmostEqual(float(result), expected, places=12)

        result = wp.trace(wp.mat44f(*values))
        self.assertIsInstance(result, float)
        self.assertNotAlmostEqual(result, expected, places=12)
        self.assertAlmostEqual(result, expected, places=5)

        result = wp.trace(wp.mat44h(*values))
        self.assertIsInstance(result, wp.float16)
        self.assertNotAlmostEqual(float(result), expected, places=5)
        self.assertAlmostEqual(float(result), expected, places=1)

    def test_mat22_mat22_args_precision(self):
        a_values = (0.12, 1.23, 0.12, 1.23)
        b_values = (1.23, 0.12, 1.23, 0.12)
        expected = 0.59039999999999992486

        result = wp.ddot(wp.mat22d(*a_values), wp.mat22d(*b_values))
        self.assertIsInstance(result, wp.float64)
        self.assertAlmostEqual(float(result), expected, places=12)

        result = wp.ddot(wp.mat22f(*a_values), wp.mat22f(*b_values))
        self.assertIsInstance(result, float)
        self.assertNotAlmostEqual(result, expected, places=12)
        self.assertAlmostEqual(result, expected, places=5)

        result = wp.ddot(wp.mat22h(*a_values), wp.mat22h(*b_values))
        self.assertIsInstance(result, wp.float16)
        self.assertNotAlmostEqual(float(result), expected, places=5)
        self.assertAlmostEqual(float(result), expected, places=1)

    def test_mat33_mat33_args_precision(self):
        a_values = (0.12, 1.23, 2.34, 0.12, 1.23, 2.34, 0.12, 1.23, 2.34)
        b_values = (2.34, 1.23, 0.12, 2.34, 1.23, 0.12, 2.34, 1.23, 0.12)
        expected = 6.22350000000000047606

        result = wp.ddot(wp.mat33d(*a_values), wp.mat33d(*b_values))
        self.assertIsInstance(result, wp.float64)
        self.assertAlmostEqual(float(result), expected, places=12)

        result = wp.ddot(wp.mat33f(*a_values), wp.mat33f(*b_values))
        self.assertIsInstance(result, float)
        self.assertNotAlmostEqual(result, expected, places=12)
        self.assertAlmostEqual(result, expected, places=5)

        result = wp.ddot(wp.mat33h(*a_values), wp.mat33h(*b_values))
        self.assertIsInstance(result, wp.float16)
        self.assertNotAlmostEqual(float(result), expected, places=5)
        self.assertAlmostEqual(float(result), expected, places=1)

    def test_mat44_mat44_args(self):
        a_values = (0.12, 1.23, 2.34, 3.45, 0.12, 1.23, 2.34, 3.45, 0.12, 1.23, 2.34, 3.45, 0.12, 1.23, 2.34, 3.45)
        b_values = (3.45, 2.34, 1.23, 0.12, 3.45, 2.34, 1.23, 0.12, 3.45, 2.34, 1.23, 0.12, 3.45, 2.34, 1.23, 0.12)
        expected = 26.33760000000000189857

        result = wp.ddot(wp.mat44d(*a_values), wp.mat44d(*b_values))
        self.assertIsInstance(result, wp.float64)
        self.assertAlmostEqual(float(result), expected, places=12)

        result = wp.ddot(wp.mat44f(*a_values), wp.mat44f(*b_values))
        self.assertIsInstance(result, float)
        self.assertNotAlmostEqual(result, expected, places=12)
        self.assertAlmostEqual(result, expected, places=5)

        result = wp.ddot(wp.mat44h(*a_values), wp.mat44h(*b_values))
        self.assertIsInstance(result, wp.float16)
        self.assertNotAlmostEqual(float(result), expected, places=5)
        self.assertAlmostEqual(float(result), expected, places=1)

    def test_mat22_float_args_precision(self):
        a_values = (1.23, 2.34, 3.45, 4.56)
        b_value = 0.12
        expected_00 = 0.14759999999999998122
        expected_01 = 0.28079999999999999405
        expected_10 = 0.41399999999999997913
        expected_11 = 0.54719999999999990870

        result = wp.mul(wp.mat22d(*a_values), wp.float64(b_value))
        self.assertAlmostEqual(float(result[0][0]), expected_00, places=12)
        self.assertAlmostEqual(float(result[0][1]), expected_01, places=12)
        self.assertAlmostEqual(float(result[1][0]), expected_10, places=12)
        self.assertAlmostEqual(float(result[1][1]), expected_11, places=12)

        result = wp.mul(wp.mat22f(*a_values), wp.float32(b_value))
        self.assertNotAlmostEqual(result[0][0], expected_00, places=12)
        self.assertNotAlmostEqual(result[0][1], expected_01, places=12)
        self.assertNotAlmostEqual(result[1][0], expected_10, places=12)
        self.assertNotAlmostEqual(result[1][1], expected_11, places=12)
        self.assertAlmostEqual(result[0][0], expected_00, places=5)
        self.assertAlmostEqual(result[0][1], expected_01, places=5)
        self.assertAlmostEqual(result[1][0], expected_10, places=5)
        self.assertAlmostEqual(result[1][1], expected_11, places=5)

        result = wp.mul(wp.mat22h(*a_values), wp.float16(b_value))
        self.assertNotAlmostEqual(float(result[0][0]), expected_00, places=5)
        self.assertNotAlmostEqual(float(result[0][1]), expected_01, places=5)
        self.assertNotAlmostEqual(float(result[1][0]), expected_10, places=5)
        self.assertNotAlmostEqual(float(result[1][1]), expected_11, places=5)
        self.assertAlmostEqual(float(result[0][0]), expected_00, places=1)
        self.assertAlmostEqual(float(result[0][1]), expected_01, places=1)
        self.assertAlmostEqual(float(result[1][0]), expected_10, places=1)
        self.assertAlmostEqual(float(result[1][1]), expected_11, places=1)

    def test_mat33_float_args_precision(self):
        a_values = (1.23, 2.34, 3.45, 4.56, 5.67, 6.78, 7.89, 8.90, 9.01)
        b_value = 0.12
        expected_00 = 0.14759999999999998122
        expected_01 = 0.28079999999999999405
        expected_02 = 0.41399999999999997913
        expected_10 = 0.54719999999999990870
        expected_11 = 0.68040000000000000480
        expected_12 = 0.81359999999999998987
        expected_20 = 0.94679999999999997495
        expected_21 = 1.06800000000000006040
        expected_22 = 1.08119999999999993889

        result = wp.mul(wp.mat33d(*a_values), wp.float64(b_value))
        self.assertAlmostEqual(float(result[0][0]), expected_00, places=12)
        self.assertAlmostEqual(float(result[0][1]), expected_01, places=12)
        self.assertAlmostEqual(float(result[0][2]), expected_02, places=12)
        self.assertAlmostEqual(float(result[1][0]), expected_10, places=12)
        self.assertAlmostEqual(float(result[1][1]), expected_11, places=12)
        self.assertAlmostEqual(float(result[1][2]), expected_12, places=12)
        self.assertAlmostEqual(float(result[2][0]), expected_20, places=12)
        self.assertAlmostEqual(float(result[2][1]), expected_21, places=12)
        self.assertAlmostEqual(float(result[2][2]), expected_22, places=12)

        result = wp.mul(wp.mat33f(*a_values), wp.float32(b_value))
        self.assertNotAlmostEqual(result[0][0], expected_00, places=12)
        self.assertNotAlmostEqual(result[0][1], expected_01, places=12)
        self.assertNotAlmostEqual(result[0][2], expected_02, places=12)
        self.assertNotAlmostEqual(result[1][0], expected_10, places=12)
        self.assertNotAlmostEqual(result[1][1], expected_11, places=12)
        self.assertNotAlmostEqual(result[1][2], expected_12, places=12)
        self.assertNotAlmostEqual(result[2][0], expected_20, places=12)
        self.assertNotAlmostEqual(result[2][1], expected_21, places=12)
        self.assertNotAlmostEqual(result[2][2], expected_22, places=12)
        self.assertAlmostEqual(result[0][0], expected_00, places=5)
        self.assertAlmostEqual(result[0][1], expected_01, places=5)
        self.assertAlmostEqual(result[0][2], expected_02, places=5)
        self.assertAlmostEqual(result[1][0], expected_10, places=5)
        self.assertAlmostEqual(result[1][1], expected_11, places=5)
        self.assertAlmostEqual(result[1][2], expected_12, places=5)
        self.assertAlmostEqual(result[2][0], expected_20, places=5)
        self.assertAlmostEqual(result[2][1], expected_21, places=5)
        self.assertAlmostEqual(result[2][2], expected_22, places=5)

        result = wp.mul(wp.mat33h(*a_values), wp.float16(b_value))
        self.assertNotAlmostEqual(float(result[0][0]), expected_00, places=5)
        self.assertNotAlmostEqual(float(result[0][1]), expected_01, places=5)
        self.assertNotAlmostEqual(float(result[0][2]), expected_02, places=5)
        self.assertNotAlmostEqual(float(result[1][0]), expected_10, places=5)
        self.assertNotAlmostEqual(float(result[1][1]), expected_11, places=5)
        self.assertNotAlmostEqual(float(result[1][2]), expected_12, places=5)
        self.assertNotAlmostEqual(float(result[2][0]), expected_20, places=5)
        self.assertNotAlmostEqual(float(result[2][1]), expected_21, places=5)
        self.assertNotAlmostEqual(float(result[2][2]), expected_22, places=5)
        self.assertAlmostEqual(float(result[0][0]), expected_00, places=1)
        self.assertAlmostEqual(float(result[0][1]), expected_01, places=1)
        self.assertAlmostEqual(float(result[0][2]), expected_02, places=1)
        self.assertAlmostEqual(float(result[1][0]), expected_10, places=1)
        self.assertAlmostEqual(float(result[1][1]), expected_11, places=1)
        self.assertAlmostEqual(float(result[1][2]), expected_12, places=1)
        self.assertAlmostEqual(float(result[2][0]), expected_20, places=1)
        self.assertAlmostEqual(float(result[2][1]), expected_21, places=1)
        self.assertAlmostEqual(float(result[2][2]), expected_22, places=1)

    def test_mat44_float_args_precision(self):
        a_values = (
            1.23,
            2.34,
            3.45,
            4.56,
            5.67,
            6.78,
            7.89,
            8.90,
            9.01,
            10.12,
            11.23,
            12.34,
            13.45,
            14.56,
            15.67,
            16.78,
        )
        b_value = 0.12
        expected_00 = 0.14759999999999998122
        expected_01 = 0.28079999999999999405
        expected_02 = 0.41399999999999997913
        expected_03 = 0.54719999999999990870
        expected_10 = 0.68040000000000000480
        expected_11 = 0.81359999999999998987
        expected_12 = 0.94679999999999997495
        expected_13 = 1.06800000000000006040
        expected_20 = 1.08119999999999993889
        expected_21 = 1.21439999999999992397
        expected_22 = 1.34759999999999990905
        expected_23 = 1.48079999999999989413
        expected_30 = 1.61399999999999987921
        expected_31 = 1.74720000000000008633
        expected_32 = 1.88039999999999984936
        expected_33 = 2.01360000000000027853

        result = wp.mul(wp.mat44d(*a_values), wp.float64(b_value))
        self.assertAlmostEqual(float(result[0][0]), expected_00, places=12)
        self.assertAlmostEqual(float(result[0][1]), expected_01, places=12)
        self.assertAlmostEqual(float(result[0][2]), expected_02, places=12)
        self.assertAlmostEqual(float(result[0][3]), expected_03, places=12)
        self.assertAlmostEqual(float(result[1][0]), expected_10, places=12)
        self.assertAlmostEqual(float(result[1][1]), expected_11, places=12)
        self.assertAlmostEqual(float(result[1][2]), expected_12, places=12)
        self.assertAlmostEqual(float(result[1][3]), expected_13, places=12)
        self.assertAlmostEqual(float(result[2][0]), expected_20, places=12)
        self.assertAlmostEqual(float(result[2][1]), expected_21, places=12)
        self.assertAlmostEqual(float(result[2][2]), expected_22, places=12)
        self.assertAlmostEqual(float(result[2][3]), expected_23, places=12)
        self.assertAlmostEqual(float(result[3][0]), expected_30, places=12)
        self.assertAlmostEqual(float(result[3][1]), expected_31, places=12)
        self.assertAlmostEqual(float(result[3][2]), expected_32, places=12)
        self.assertAlmostEqual(float(result[3][3]), expected_33, places=12)

        result = wp.mul(wp.mat44f(*a_values), wp.float32(b_value))
        self.assertNotAlmostEqual(result[0][0], expected_00, places=12)
        self.assertNotAlmostEqual(result[0][1], expected_01, places=12)
        self.assertNotAlmostEqual(result[0][2], expected_02, places=12)
        self.assertNotAlmostEqual(result[0][3], expected_03, places=12)
        self.assertNotAlmostEqual(result[1][0], expected_10, places=12)
        self.assertNotAlmostEqual(result[1][1], expected_11, places=12)
        self.assertNotAlmostEqual(result[1][2], expected_12, places=12)
        self.assertNotAlmostEqual(result[1][3], expected_13, places=12)
        self.assertNotAlmostEqual(result[2][0], expected_20, places=12)
        self.assertNotAlmostEqual(result[2][1], expected_21, places=12)
        self.assertNotAlmostEqual(result[2][2], expected_22, places=12)
        self.assertNotAlmostEqual(result[2][3], expected_23, places=12)
        self.assertNotAlmostEqual(result[3][0], expected_30, places=12)
        self.assertNotAlmostEqual(result[3][1], expected_31, places=12)
        self.assertNotAlmostEqual(result[3][2], expected_32, places=12)
        self.assertNotAlmostEqual(result[3][3], expected_33, places=12)
        self.assertAlmostEqual(result[0][0], expected_00, places=5)
        self.assertAlmostEqual(result[0][1], expected_01, places=5)
        self.assertAlmostEqual(result[0][2], expected_02, places=5)
        self.assertAlmostEqual(result[0][3], expected_03, places=5)
        self.assertAlmostEqual(result[1][0], expected_10, places=5)
        self.assertAlmostEqual(result[1][1], expected_11, places=5)
        self.assertAlmostEqual(result[1][2], expected_12, places=5)
        self.assertAlmostEqual(result[1][3], expected_13, places=5)
        self.assertAlmostEqual(result[2][0], expected_20, places=5)
        self.assertAlmostEqual(result[2][1], expected_21, places=5)
        self.assertAlmostEqual(result[2][2], expected_22, places=5)
        self.assertAlmostEqual(result[2][3], expected_23, places=5)
        self.assertAlmostEqual(result[3][0], expected_30, places=5)
        self.assertAlmostEqual(result[3][1], expected_31, places=5)
        self.assertAlmostEqual(result[3][2], expected_32, places=5)
        self.assertAlmostEqual(result[3][3], expected_33, places=5)

        result = wp.mul(wp.mat44h(*a_values), wp.float16(b_value))
        self.assertNotAlmostEqual(float(result[0][0]), expected_00, places=5)
        self.assertNotAlmostEqual(float(result[0][1]), expected_01, places=5)
        self.assertNotAlmostEqual(float(result[0][2]), expected_02, places=5)
        self.assertNotAlmostEqual(float(result[0][3]), expected_03, places=5)
        self.assertNotAlmostEqual(float(result[1][0]), expected_10, places=5)
        self.assertNotAlmostEqual(float(result[1][1]), expected_11, places=5)
        self.assertNotAlmostEqual(float(result[1][2]), expected_12, places=5)
        self.assertNotAlmostEqual(float(result[1][3]), expected_13, places=5)
        self.assertNotAlmostEqual(float(result[2][0]), expected_20, places=5)
        self.assertNotAlmostEqual(float(result[2][1]), expected_21, places=5)
        self.assertNotAlmostEqual(float(result[2][2]), expected_22, places=5)
        self.assertNotAlmostEqual(float(result[2][3]), expected_23, places=5)
        self.assertNotAlmostEqual(float(result[3][0]), expected_30, places=5)
        self.assertNotAlmostEqual(float(result[3][1]), expected_31, places=5)
        self.assertNotAlmostEqual(float(result[3][2]), expected_32, places=5)
        self.assertNotAlmostEqual(float(result[3][3]), expected_33, places=5)
        self.assertAlmostEqual(float(result[0][0]), expected_00, places=1)
        self.assertAlmostEqual(float(result[0][1]), expected_01, places=1)
        self.assertAlmostEqual(float(result[0][2]), expected_02, places=1)
        self.assertAlmostEqual(float(result[0][3]), expected_03, places=1)
        self.assertAlmostEqual(float(result[1][0]), expected_10, places=1)
        self.assertAlmostEqual(float(result[1][1]), expected_11, places=1)
        self.assertAlmostEqual(float(result[1][2]), expected_12, places=1)
        self.assertAlmostEqual(float(result[1][3]), expected_13, places=1)
        self.assertAlmostEqual(float(result[2][0]), expected_20, places=1)
        self.assertAlmostEqual(float(result[2][1]), expected_21, places=1)
        self.assertAlmostEqual(float(result[2][2]), expected_22, places=1)
        self.assertAlmostEqual(float(result[2][3]), expected_23, places=1)
        self.assertAlmostEqual(float(result[3][0]), expected_30, places=1)
        self.assertAlmostEqual(float(result[3][1]), expected_31, places=1)
        self.assertAlmostEqual(float(result[3][2]), expected_32, places=1)
        self.assertAlmostEqual(float(result[3][3]), expected_33, places=1)

    def test_vec2_arg_precision(self):
        values = (1.23, 2.34)
        expected = 2.64357712200722438922

        result = wp.length(wp.vec2d(*values))
        self.assertIsInstance(result, wp.float64)
        self.assertAlmostEqual(float(result), expected, places=12)

        result = wp.length(wp.vec2f(*values))
        self.assertIsInstance(result, float)
        self.assertNotAlmostEqual(result, expected, places=12)
        self.assertAlmostEqual(result, expected, places=5)

        result = wp.length(wp.vec2h(*values))
        self.assertIsInstance(result, wp.float16)
        self.assertNotAlmostEqual(float(result), expected, places=5)
        self.assertAlmostEqual(float(result), expected, places=1)

    def test_vec2_arg_overflow(self):
        values = (-1234567890, -1234567890)

        self.assertEqual(wp.length_sq(wp.vec2b(*values)), -120)
        self.assertEqual(wp.length_sq(wp.vec2s(*values)), -6008)
        self.assertEqual(wp.length_sq(wp.vec2i(*values)), 608168072)
        self.assertEqual(wp.length_sq(wp.vec2l(*values)), 3048315750038104200)

        self.assertEqual(wp.length_sq(wp.vec2ub(*values)), 136)
        self.assertEqual(wp.length_sq(wp.vec2us(*values)), 59528)
        self.assertEqual(wp.length_sq(wp.vec2ui(*values)), 608168072)
        self.assertEqual(wp.length_sq(wp.vec2ul(*values)), 3048315750038104200)

    def test_vec3_arg_precision(self):
        values = (1.23, 2.34, 3.45)
        expected = 4.34637780226247727455

        result = wp.length(wp.vec3d(*values))
        self.assertIsInstance(result, wp.float64)
        self.assertAlmostEqual(float(result), expected, places=12)

        result = wp.length(wp.vec3f(*values))
        self.assertIsInstance(result, float)
        self.assertNotAlmostEqual(result, expected, places=12)
        self.assertAlmostEqual(result, expected, places=5)

        result = wp.length(wp.vec3h(*values))
        self.assertIsInstance(result, wp.float16)
        self.assertNotAlmostEqual(float(result), expected, places=5)
        self.assertAlmostEqual(float(result), expected, places=1)

    def test_vec3_arg_overflow(self):
        values = (-1234567890, -1234567890, -1234567890)

        self.assertEqual(wp.length_sq(wp.vec3b(*values)), -52)
        self.assertEqual(wp.length_sq(wp.vec3s(*values)), -9012)
        self.assertEqual(wp.length_sq(wp.vec3i(*values)), 912252108)
        self.assertEqual(wp.length_sq(wp.vec3l(*values)), 4572473625057156300)

        self.assertEqual(wp.length_sq(wp.vec3ub(*values)), 204)
        self.assertEqual(wp.length_sq(wp.vec3us(*values)), 56524)
        self.assertEqual(wp.length_sq(wp.vec3ui(*values)), 912252108)
        self.assertEqual(wp.length_sq(wp.vec3ul(*values)), 4572473625057156300)

    def test_vec4_arg_precision(self):
        values = (1.23, 2.34, 3.45, 4.56)
        expected = 6.29957141399317777086

        result = wp.length(wp.vec4d(*values))
        self.assertIsInstance(result, wp.float64)
        self.assertAlmostEqual(float(result), expected, places=12)

        result = wp.length(wp.vec4f(*values))
        self.assertIsInstance(result, float)
        self.assertNotAlmostEqual(result, expected, places=12)
        self.assertAlmostEqual(result, expected, places=5)

        result = wp.length(wp.vec4h(*values))
        self.assertIsInstance(result, wp.float16)
        self.assertNotAlmostEqual(float(result), expected, places=5)
        self.assertAlmostEqual(float(result), expected, places=1)

    def test_vec4_arg_overflow(self):
        values = (-1234567890, -1234567890, -1234567890, -1234567890)

        self.assertEqual(wp.length_sq(wp.vec4b(*values)), 16)
        self.assertEqual(wp.length_sq(wp.vec4s(*values)), -12016)
        self.assertEqual(wp.length_sq(wp.vec4i(*values)), 1216336144)
        self.assertEqual(wp.length_sq(wp.vec4l(*values)), 6096631500076208400)

        self.assertEqual(wp.length_sq(wp.vec4ub(*values)), 16)
        self.assertEqual(wp.length_sq(wp.vec4us(*values)), 53520)
        self.assertEqual(wp.length_sq(wp.vec4ui(*values)), 1216336144)
        self.assertEqual(wp.length_sq(wp.vec4ul(*values)), 6096631500076208400)

    def test_vec2_vec2_args_precision(self):
        a_values = (1.23, 2.34)
        b_values = (3.45, 4.56)
        expected = 14.91389999999999815827

        result = wp.dot(wp.vec2d(*a_values), wp.vec2d(*b_values))
        self.assertIsInstance(result, wp.float64)
        self.assertAlmostEqual(float(result), expected, places=12)

        result = wp.dot(wp.vec2f(*a_values), wp.vec2f(*b_values))
        self.assertIsInstance(result, float)
        self.assertNotAlmostEqual(result, expected, places=12)
        self.assertAlmostEqual(result, expected, places=5)

        result = wp.dot(wp.vec2h(*a_values), wp.vec2h(*b_values))
        self.assertIsInstance(result, wp.float16)
        self.assertNotAlmostEqual(float(result), expected, places=5)
        self.assertAlmostEqual(float(result), expected, places=1)

    def test_vec2_vec2_args_overflow(self):
        values = (-1234567890, -1234567890)

        self.assertEqual(wp.dot(wp.vec2b(*values), wp.vec2b(*values)), -120)
        self.assertEqual(wp.dot(wp.vec2s(*values), wp.vec2s(*values)), -6008)
        self.assertEqual(wp.dot(wp.vec2i(*values), wp.vec2i(*values)), 608168072)
        self.assertEqual(wp.dot(wp.vec2l(*values), wp.vec2l(*values)), 3048315750038104200)

        self.assertEqual(wp.dot(wp.vec2ub(*values), wp.vec2ub(*values)), 136)
        self.assertEqual(wp.dot(wp.vec2us(*values), wp.vec2us(*values)), 59528)
        self.assertEqual(wp.dot(wp.vec2ui(*values), wp.vec2ui(*values)), 608168072)
        self.assertEqual(wp.dot(wp.vec2ul(*values), wp.vec2ul(*values)), 3048315750038104200)

    def test_vec3_vec3_args_precision(self):
        a_values = (1.23, 2.34, 3.45)
        b_values = (4.56, 5.67, 6.78)
        expected = 42.26760000000000161435

        result = wp.dot(wp.vec3d(*a_values), wp.vec3d(*b_values))
        self.assertIsInstance(result, wp.float64)
        self.assertAlmostEqual(float(result), expected, places=12)

        result = wp.dot(wp.vec3f(*a_values), wp.vec3f(*b_values))
        self.assertIsInstance(result, float)
        self.assertNotAlmostEqual(result, expected, places=12)
        self.assertAlmostEqual(result, expected, places=5)

        result = wp.dot(wp.vec3h(*a_values), wp.vec3h(*b_values))
        self.assertIsInstance(result, wp.float16)
        self.assertNotAlmostEqual(float(result), expected, places=5)
        self.assertAlmostEqual(float(result), expected, places=1)

    def test_vec3_vec3_args_overflow(self):
        values = (-1234567890, -1234567890, -1234567890)

        self.assertEqual(wp.dot(wp.vec3b(*values), wp.vec3b(*values)), -52)
        self.assertEqual(wp.dot(wp.vec3s(*values), wp.vec3s(*values)), -9012)
        self.assertEqual(wp.dot(wp.vec3i(*values), wp.vec3i(*values)), 912252108)
        self.assertEqual(wp.dot(wp.vec3l(*values), wp.vec3l(*values)), 4572473625057156300)

        self.assertEqual(wp.dot(wp.vec3ub(*values), wp.vec3ub(*values)), 204)
        self.assertEqual(wp.dot(wp.vec3us(*values), wp.vec3us(*values)), 56524)
        self.assertEqual(wp.dot(wp.vec3ui(*values), wp.vec3ui(*values)), 912252108)
        self.assertEqual(wp.dot(wp.vec3ul(*values), wp.vec3ul(*values)), 4572473625057156300)

    def test_vec4_vec4_args_precision(self):
        a_values = (1.23, 2.34, 3.45, 4.56)
        b_values = (5.67, 6.78, 7.89, 8.90)
        expected = 90.64379999999999881766

        result = wp.dot(wp.vec4d(*a_values), wp.vec4d(*b_values))
        self.assertIsInstance(result, wp.float64)
        self.assertAlmostEqual(float(result), expected, places=12)

        result = wp.dot(wp.vec4f(*a_values), wp.vec4f(*b_values))
        self.assertIsInstance(result, float)
        self.assertNotAlmostEqual(result, expected, places=12)
        self.assertAlmostEqual(result, expected, places=5)

        result = wp.dot(wp.vec4h(*a_values), wp.vec4h(*b_values))
        self.assertIsInstance(result, wp.float16)
        self.assertNotAlmostEqual(float(result), expected, places=5)
        self.assertAlmostEqual(float(result), expected, places=1)

    def test_vec4_vec4_args_overflow(self):
        values = (-1234567890, -1234567890, -1234567890, -1234567890)

        self.assertEqual(wp.dot(wp.vec4b(*values), wp.vec4b(*values)), 16)
        self.assertEqual(wp.dot(wp.vec4s(*values), wp.vec4s(*values)), -12016)
        self.assertEqual(wp.dot(wp.vec4i(*values), wp.vec4i(*values)), 1216336144)
        self.assertEqual(wp.dot(wp.vec4l(*values), wp.vec4l(*values)), 6096631500076208400)

        self.assertEqual(wp.dot(wp.vec4ub(*values), wp.vec4ub(*values)), 16)
        self.assertEqual(wp.dot(wp.vec4us(*values), wp.vec4us(*values)), 53520)
        self.assertEqual(wp.dot(wp.vec4ui(*values), wp.vec4ui(*values)), 1216336144)
        self.assertEqual(wp.dot(wp.vec4ul(*values), wp.vec4ul(*values)), 6096631500076208400)

    def test_vec2_float_args_precision(self):
        a_values = (1.23, 2.34)
        b_value = 3.45
        expected_x = 4.24350000000000004974
        expected_y = 8.07300000000000039790

        result = wp.mul(wp.vec2d(*a_values), wp.float64(b_value))
        self.assertAlmostEqual(float(result[0]), expected_x, places=12)
        self.assertAlmostEqual(float(result[1]), expected_y, places=12)

        result = wp.mul(wp.vec2f(*a_values), wp.float32(b_value))
        self.assertNotAlmostEqual(result[0], expected_x, places=12)
        self.assertNotAlmostEqual(result[1], expected_y, places=12)
        self.assertAlmostEqual(result[0], expected_x, places=5)
        self.assertAlmostEqual(result[1], expected_y, places=5)

        result = wp.mul(wp.vec2h(*a_values), wp.float16(b_value))
        self.assertNotAlmostEqual(float(result[0]), expected_x, places=5)
        self.assertNotAlmostEqual(float(result[1]), expected_y, places=5)
        self.assertAlmostEqual(float(result[0]), expected_x, places=1)
        self.assertAlmostEqual(float(result[1]), expected_y, places=1)

    def test_vec3_float_args_precision(self):
        a_values = (1.23, 2.34, 3.45)
        b_value = 4.56
        expected_x = 5.60879999999999956373
        expected_y = 10.67039999999999899671
        expected_z = 15.73199999999999931788

        result = wp.mul(wp.vec3d(*a_values), wp.float64(b_value))
        self.assertAlmostEqual(float(result[0]), expected_x, places=12)
        self.assertAlmostEqual(float(result[1]), expected_y, places=12)
        self.assertAlmostEqual(float(result[2]), expected_z, places=12)

        result = wp.mul(wp.vec3f(*a_values), wp.float32(b_value))
        self.assertNotAlmostEqual(result[0], expected_x, places=12)
        self.assertNotAlmostEqual(result[1], expected_y, places=12)
        self.assertNotAlmostEqual(result[2], expected_z, places=12)
        self.assertAlmostEqual(result[0], expected_x, places=5)
        self.assertAlmostEqual(result[1], expected_y, places=5)
        self.assertAlmostEqual(result[2], expected_z, places=5)

        result = wp.mul(wp.vec3h(*a_values), wp.float16(b_value))
        self.assertNotAlmostEqual(float(result[0]), expected_x, places=5)
        self.assertNotAlmostEqual(float(result[1]), expected_y, places=5)
        self.assertNotAlmostEqual(float(result[2]), expected_z, places=5)
        self.assertAlmostEqual(float(result[0]), expected_x, places=1)
        self.assertAlmostEqual(float(result[1]), expected_y, places=1)
        self.assertAlmostEqual(float(result[2]), expected_z, places=1)

    def test_vec4_float_args_precision(self):
        a_values = (1.23, 2.34, 3.45, 4.56)
        b_value = 5.67
        expected_x = 6.97409999999999996589
        expected_y = 13.26779999999999937188
        expected_z = 19.56150000000000233058
        expected_w = 25.85519999999999640750

        result = wp.mul(wp.vec4d(*a_values), wp.float64(b_value))
        self.assertAlmostEqual(float(result[0]), expected_x, places=12)
        self.assertAlmostEqual(float(result[1]), expected_y, places=12)
        self.assertAlmostEqual(float(result[2]), expected_z, places=12)
        self.assertAlmostEqual(float(result[3]), expected_w, places=12)

        result = wp.mul(wp.vec4f(*a_values), wp.float32(b_value))
        self.assertNotAlmostEqual(result[0], expected_x, places=12)
        self.assertNotAlmostEqual(result[1], expected_y, places=12)
        self.assertNotAlmostEqual(result[2], expected_z, places=12)
        self.assertNotAlmostEqual(result[3], expected_w, places=12)
        self.assertAlmostEqual(result[0], expected_x, places=5)
        self.assertAlmostEqual(result[1], expected_y, places=5)
        self.assertAlmostEqual(result[2], expected_z, places=5)
        self.assertAlmostEqual(result[3], expected_w, places=5)

        result = wp.mul(wp.vec4h(*a_values), wp.float16(b_value))
        self.assertNotAlmostEqual(float(result[0]), expected_x, places=5)
        self.assertNotAlmostEqual(float(result[1]), expected_y, places=5)
        self.assertNotAlmostEqual(float(result[2]), expected_z, places=5)
        self.assertNotAlmostEqual(float(result[3]), expected_w, places=5)
        self.assertAlmostEqual(float(result[0]), expected_x, places=1)
        self.assertAlmostEqual(float(result[1]), expected_y, places=1)
        self.assertAlmostEqual(float(result[2]), expected_z, places=1)
        self.assertAlmostEqual(float(result[3]), expected_w, places=1)


for dtype in wp._src.types.int_types:
    add_function_test(
        TestBuiltinsResolution,
        f"test_int_arg_support_{dtype.__name__}",
        test_int_arg_support,
        dtype=dtype,
    )
    add_function_test(
        TestBuiltinsResolution,
        f"test_int_int_args_support_{dtype.__name__}",
        test_int_int_args_support,
        dtype=dtype,
    )

for dtype in wp._src.types.float_types:
    add_function_test(
        TestBuiltinsResolution,
        f"test_float_arg_support_{dtype.__name__}",
        test_float_arg_support,
        dtype=dtype,
    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
