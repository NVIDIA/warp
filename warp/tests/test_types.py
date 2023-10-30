# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import contextlib
import io
import inspect
import unittest

from warp.tests.test_base import *


wp.init()


def test_constant(test, device):
    const = wp.constant(123)
    test.assertEqual(const, 123)

    const = wp.constant(1.25)
    test.assertEqual(const, 1.25)

    const = wp.constant(True)
    test.assertEqual(const, True)

    const = wp.constant(wp.float16(1.25))
    test.assertEqual(const.value, 1.25)

    const = wp.constant(wp.int16(123))
    test.assertEqual(const.value, 123)

    const = wp.constant(wp.vec3i(1, 2, 3))
    test.assertEqual(const, wp.vec3i(1, 2, 3))


def test_constant_error_invalid_type(test, device):
    with test.assertRaisesRegex(
        RuntimeError,
        r"Invalid constant type: <class 'tuple'>$",
    ):
        wp.constant((1, 2, 3))


def test_vector(test, device):
    for dtype in (int, float, wp.float16):
        vec_cls = wp.vec(3, dtype)

        v = vec_cls(1, 2, 3)
        test.assertEqual(v[0], 1)
        test.assertSequenceEqual(v[0:2], (1, 2))
        test.assertSequenceEqual(v, (1, 2, 3))

        v[0] = -1
        test.assertEqual(v[0], -1)
        test.assertSequenceEqual(v[0:2], (-1, 2))
        test.assertSequenceEqual(v, (-1, 2, 3))

        v[1:3] = (-2, -3)
        test.assertEqual(v[0], -1)
        test.assertSequenceEqual(v[0:2], (-1, -2))
        test.assertSequenceEqual(v, (-1, -2, -3))

        v += vec_cls(1, 1, 1)
        test.assertSequenceEqual(v, (0, -1, -2))


def register(parent):
    devices = get_test_devices()

    class TestUtils(parent):
        pass

    add_function_test(TestUtils, "test_constant", test_constant)
    add_function_test(TestUtils, "test_constant_error_invalid_type", test_constant_error_invalid_type)
    add_function_test(TestUtils, "test_vector", test_vector)
    return TestUtils


if __name__ == "__main__":
    _ = register(unittest.TestCase)
    unittest.main(verbosity=2)
