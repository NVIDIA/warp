# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import contextlib
import unittest

import numpy as np

from warp.tests.unittest_utils import *


def nps(dtype, value):
    """Creates a NumPy scalar value based on the given data type."""
    # Workaround to avoid deprecation warning messages for integer overflows.
    return np.array((value,)).astype(dtype)[0]


def npv(dtype, values):
    """Creates a vector of NumPy scalar values based on the given data type."""
    return tuple(nps(dtype, x) for x in values)


def npm(dtype, dim, values):
    """Creates a matrix of NumPy scalar values based on the given data type."""
    return tuple(npv(dtype, values[i * dim : (i + 1) * dim]) for i in range(dim))


def wpv(dtype, values):
    """Creates a vector of Warp scalar values based on the given data type."""
    return tuple(dtype(x) for x in values)


def wpm(dtype, dim, values):
    """Creates a matrix of Warp scalar values based on the given data type."""
    return tuple(wpv(dtype, values[i * dim : (i + 1) * dim]) for i in range(dim))


def test_int_arg_support(test, device, dtype):
    np_type = wp.types.warp_type_to_np_dtype[dtype]
    value = -1234567890123456789
    expected = wp.invert(dtype(value))

    test.assertEqual(wp.invert(nps(np_type, value)), expected)


def test_float_arg_support(test, device, dtype):
    np_type = wp.types.warp_type_to_np_dtype[dtype]
    value = 1.23
    expected = wp.sin(dtype(value))

    test.assertEqual(wp.sin(nps(np_type, value)), expected)


def test_int_int_args_support(test, device, dtype):
    np_type = wp.types.warp_type_to_np_dtype[dtype]
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
            rf"Couldn't find a function 'mul' compatible with " rf"the arguments '{dtype.__name__}, int'$",
        ):
            wp.mul(dtype(value), value)

        with test.assertRaisesRegex(
            RuntimeError,
            rf"Couldn't find a function 'mul' compatible with " rf"the arguments '{np_type.__name__}, int'$",
        ):
            wp.mul(nps(np_type, value), value)

        with test.assertRaisesRegex(
            RuntimeError,
            rf"Couldn't find a function 'mul' compatible with " rf"the arguments 'int, {dtype.__name__}'$",
        ):
            wp.mul(value, dtype(value))

        with test.assertRaisesRegex(
            RuntimeError,
            rf"Couldn't find a function 'mul' compatible with " rf"the arguments 'int, {np_type.__name__}'$",
        ):
            wp.mul(value, nps(np_type, value))


def test_mat_arg_support(test, device, dtype):
    np_type = wp.types.warp_type_to_np_dtype[dtype]
    mat_cls = wp.types.matrix((3, 3), dtype)
    values = (1.23, 2.34, 3.45, 4.56, 5.67, 6.78, 7.89, 8.90, 9.01)
    expected = wp.trace(mat_cls(*values))

    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        test.assertEqual(wp.trace(wpv(dtype, values)), expected)
        test.assertEqual(wp.trace(wpm(dtype, 3, values)), expected)
        test.assertEqual(wp.trace(npv(np_type, values)), expected)
        test.assertEqual(wp.trace(npm(np_type, 3, values)), expected)
        test.assertEqual(wp.trace(np.array(npv(np_type, values))), expected)


def test_mat_mat_args_support(test, device, dtype):
    np_type = wp.types.warp_type_to_np_dtype[dtype]
    mat_cls = wp.types.matrix((3, 3), dtype)
    a_values = (0.12, 1.23, 2.34, 0.12, 1.23, 2.34, 0.12, 1.23, 2.34)
    b_values = (2.34, 1.23, 0.12, 2.34, 1.23, 0.12, 2.34, 1.23, 0.12)
    expected = wp.ddot(mat_cls(*a_values), mat_cls(*b_values))

    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        test.assertEqual(wp.ddot(mat_cls(*a_values), mat_cls(*b_values)), expected)
        test.assertEqual(wp.ddot(mat_cls(*a_values), wpv(dtype, b_values)), expected)
        test.assertEqual(wp.ddot(mat_cls(*a_values), wpm(dtype, 3, b_values)), expected)
        test.assertEqual(wp.ddot(mat_cls(*a_values), npv(np_type, b_values)), expected)
        test.assertEqual(wp.ddot(mat_cls(*a_values), npm(np_type, 3, b_values)), expected)
        test.assertEqual(wp.ddot(mat_cls(*a_values), np.array(npv(np_type, b_values))), expected)

        test.assertEqual(wp.ddot(wpv(dtype, a_values), mat_cls(*b_values)), expected)
        test.assertEqual(wp.ddot(wpv(dtype, a_values), wpv(dtype, b_values)), expected)
        test.assertEqual(wp.ddot(wpv(dtype, a_values), wpm(dtype, 3, b_values)), expected)
        test.assertEqual(wp.ddot(wpv(dtype, a_values), npv(np_type, b_values)), expected)
        test.assertEqual(wp.ddot(wpv(dtype, a_values), npm(np_type, 3, b_values)), expected)
        test.assertEqual(wp.ddot(wpv(dtype, a_values), np.array(npv(np_type, b_values))), expected)

        test.assertEqual(wp.ddot(wpm(dtype, 3, a_values), mat_cls(*b_values)), expected)
        test.assertEqual(wp.ddot(wpm(dtype, 3, a_values), wpv(dtype, b_values)), expected)
        test.assertEqual(wp.ddot(wpm(dtype, 3, a_values), wpm(dtype, 3, b_values)), expected)
        test.assertEqual(wp.ddot(wpm(dtype, 3, a_values), npv(np_type, b_values)), expected)
        test.assertEqual(wp.ddot(wpm(dtype, 3, a_values), npm(np_type, 3, b_values)), expected)
        test.assertEqual(wp.ddot(wpm(dtype, 3, a_values), np.array(npv(np_type, b_values))), expected)

        test.assertEqual(wp.ddot(npv(np_type, a_values), mat_cls(*b_values)), expected)
        test.assertEqual(wp.ddot(npv(np_type, a_values), wpv(dtype, b_values)), expected)
        test.assertEqual(wp.ddot(npv(np_type, a_values), wpm(dtype, 3, b_values)), expected)
        test.assertEqual(wp.ddot(npv(np_type, a_values), npv(np_type, b_values)), expected)
        test.assertEqual(wp.ddot(npv(np_type, a_values), npm(np_type, 3, b_values)), expected)
        test.assertEqual(wp.ddot(npv(np_type, a_values), np.array(npv(np_type, b_values))), expected)

        test.assertEqual(wp.ddot(npm(np_type, 3, a_values), mat_cls(*b_values)), expected)
        test.assertEqual(wp.ddot(npm(np_type, 3, a_values), wpv(dtype, b_values)), expected)
        test.assertEqual(wp.ddot(npm(np_type, 3, a_values), wpm(dtype, 3, b_values)), expected)
        test.assertEqual(wp.ddot(npm(np_type, 3, a_values), npv(np_type, b_values)), expected)
        test.assertEqual(wp.ddot(npm(np_type, 3, a_values), npm(np_type, 3, b_values)), expected)
        test.assertEqual(wp.ddot(npm(np_type, 3, a_values), np.array(npv(np_type, b_values))), expected)

        test.assertEqual(wp.ddot(np.array(npv(np_type, a_values)), mat_cls(*b_values)), expected)
        test.assertEqual(wp.ddot(np.array(npv(np_type, a_values)), wpv(dtype, b_values)), expected)
        test.assertEqual(wp.ddot(np.array(npv(np_type, a_values)), wpm(dtype, 3, b_values)), expected)
        test.assertEqual(wp.ddot(np.array(npv(np_type, a_values)), npv(np_type, b_values)), expected)
        test.assertEqual(wp.ddot(np.array(npv(np_type, a_values)), npm(np_type, 3, b_values)), expected)
        test.assertEqual(wp.ddot(np.array(npv(np_type, a_values)), np.array(npv(np_type, b_values))), expected)

        if dtype is wp.float32:
            test.assertEqual(wp.ddot(mat_cls(*a_values), b_values), expected)
            test.assertEqual(wp.ddot(wpv(dtype, a_values), b_values), expected)
            test.assertEqual(wp.ddot(wpm(dtype, 3, a_values), b_values), expected)
            test.assertEqual(wp.ddot(npv(np_type, a_values), b_values), expected)
            test.assertEqual(wp.ddot(npm(np_type, 3, a_values), b_values), expected)
            test.assertEqual(wp.ddot(a_values, b_values), expected)
            test.assertEqual(wp.ddot(np.array(npv(np_type, a_values)), b_values), expected)

            test.assertEqual(wp.ddot(a_values, mat_cls(*b_values)), expected)
            test.assertEqual(wp.ddot(a_values, wpv(dtype, b_values)), expected)
            test.assertEqual(wp.ddot(a_values, wpm(dtype, 3, b_values)), expected)
            test.assertEqual(wp.ddot(a_values, npv(np_type, b_values)), expected)
            test.assertEqual(wp.ddot(a_values, npm(np_type, 3, b_values)), expected)
            test.assertEqual(wp.ddot(a_values, np.array(npv(np_type, b_values))), expected)
        else:
            with test.assertRaisesRegex(
                RuntimeError,
                r"Couldn't find a function 'ddot' compatible with the arguments 'mat_t, tuple'$",
            ):
                wp.ddot(mat_cls(*a_values), b_values)

            with test.assertRaisesRegex(
                RuntimeError,
                r"Couldn't find a function 'ddot' compatible with the arguments 'tuple, tuple'$",
            ):
                wp.ddot(wpv(dtype, a_values), b_values)

            with test.assertRaisesRegex(
                RuntimeError,
                r"Couldn't find a function 'ddot' compatible with the arguments 'tuple, tuple'$",
            ):
                wp.ddot(wpm(dtype, 3, a_values), b_values)

            with test.assertRaisesRegex(
                RuntimeError,
                r"Couldn't find a function 'ddot' compatible with the arguments 'tuple, tuple'$",
            ):
                wp.ddot(npv(np_type, a_values), b_values)

            with test.assertRaisesRegex(
                RuntimeError,
                r"Couldn't find a function 'ddot' compatible with the arguments 'tuple, tuple'$",
            ):
                wp.ddot(npm(np_type, 3, a_values), b_values)

            with test.assertRaisesRegex(
                RuntimeError,
                r"Couldn't find a function 'ddot' compatible with the arguments 'ndarray, tuple'$",
            ):
                wp.ddot(np.array(npv(np_type, a_values)), b_values)

            with test.assertRaisesRegex(
                RuntimeError,
                r"Couldn't find a function 'ddot' compatible with the arguments 'tuple, mat_t'$",
            ):
                wp.ddot(a_values, mat_cls(*b_values))

            with test.assertRaisesRegex(
                RuntimeError,
                r"Couldn't find a function 'ddot' compatible with the arguments 'tuple, tuple'$",
            ):
                wp.ddot(a_values, wpv(dtype, b_values))

            with test.assertRaisesRegex(
                RuntimeError,
                r"Couldn't find a function 'ddot' compatible with the arguments 'tuple, tuple'$",
            ):
                wp.ddot(a_values, wpm(dtype, 3, b_values))

            with test.assertRaisesRegex(
                RuntimeError,
                r"Couldn't find a function 'ddot' compatible with the arguments 'tuple, tuple'$",
            ):
                wp.ddot(a_values, npv(np_type, b_values))

            with test.assertRaisesRegex(
                RuntimeError,
                r"Couldn't find a function 'ddot' compatible with the arguments 'tuple, tuple'$",
            ):
                wp.ddot(a_values, npm(np_type, 3, b_values))

            with test.assertRaisesRegex(
                RuntimeError,
                r"Couldn't find a function 'ddot' compatible with the arguments 'tuple, ndarray'$",
            ):
                wp.ddot(a_values, np.array(npv(np_type, b_values)))


def test_mat_float_args_support(test, device, dtype):
    np_type = wp.types.warp_type_to_np_dtype[dtype]
    mat_cls = wp.types.matrix((3, 3), dtype)
    a_values = (1.23, 2.34, 3.45, 4.56, 5.67, 6.78, 7.89, 8.90, 9.01)
    b_value = 0.12
    expected = wp.mul(mat_cls(*a_values), dtype(b_value))

    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        test.assertEqual(wp.mul(mat_cls(*a_values), dtype(b_value)), expected)
        test.assertEqual(wp.mul(mat_cls(*a_values), nps(np_type, b_value)), expected)

        test.assertEqual(wp.mul(wpv(dtype, a_values), dtype(b_value)), expected)
        test.assertEqual(wp.mul(wpv(dtype, a_values), nps(np_type, b_value)), expected)

        test.assertEqual(wp.mul(wpm(dtype, 3, a_values), dtype(b_value)), expected)
        test.assertEqual(wp.mul(wpm(dtype, 3, a_values), nps(np_type, b_value)), expected)

        test.assertEqual(wp.mul(npv(np_type, a_values), dtype(b_value)), expected)
        test.assertEqual(wp.mul(npv(np_type, a_values), nps(np_type, b_value)), expected)

        test.assertEqual(wp.mul(npm(np_type, 3, a_values), dtype(b_value)), expected)
        test.assertEqual(wp.mul(npm(np_type, 3, a_values), nps(np_type, b_value)), expected)

        test.assertEqual(wp.mul(np.array(npv(np_type, a_values)), dtype(b_value)), expected)
        test.assertEqual(wp.mul(np.array(npv(np_type, a_values)), nps(np_type, b_value)), expected)

        if dtype is wp.float32:
            test.assertEqual(wp.mul(mat_cls(*a_values), b_value), expected)
            test.assertEqual(wp.mul(wpv(dtype, a_values), b_value), expected)
            test.assertEqual(wp.mul(wpm(dtype, 3, a_values), b_value), expected)
            test.assertEqual(wp.mul(npv(np_type, a_values), b_value), expected)
            test.assertEqual(wp.mul(npm(np_type, 3, a_values), b_value), expected)
            test.assertEqual(wp.mul(a_values, b_value), expected)
            test.assertEqual(wp.mul(np.array(npv(np_type, a_values)), b_value), expected)

            test.assertEqual(wp.mul(a_values, dtype(b_value)), expected)
            test.assertEqual(wp.mul(a_values, nps(np_type, b_value)), expected)
        else:
            with test.assertRaisesRegex(
                RuntimeError,
                r"Couldn't find a function 'mul' compatible with the arguments 'mat_t, float'$",
            ):
                wp.mul(mat_cls(*a_values), b_value)

            with test.assertRaisesRegex(
                RuntimeError,
                r"Couldn't find a function 'mul' compatible with the arguments 'tuple, float'$",
            ):
                wp.mul(wpv(dtype, a_values), b_value)

            with test.assertRaisesRegex(
                RuntimeError,
                r"Couldn't find a function 'mul' compatible with the arguments 'tuple, float'$",
            ):
                wp.mul(wpm(dtype, 3, a_values), b_value)

            with test.assertRaisesRegex(
                RuntimeError,
                r"Couldn't find a function 'mul' compatible with the arguments 'tuple, float'$",
            ):
                wp.mul(npv(np_type, a_values), b_value)

            with test.assertRaisesRegex(
                RuntimeError,
                r"Couldn't find a function 'mul' compatible with the arguments 'tuple, float'$",
            ):
                wp.mul(npm(np_type, 3, a_values), b_value)

            with test.assertRaisesRegex(
                RuntimeError,
                r"Couldn't find a function 'mul' compatible with the arguments 'ndarray, float'$",
            ):
                wp.mul(np.array(npv(np_type, a_values)), b_value)

            with test.assertRaisesRegex(
                RuntimeError,
                rf"Couldn't find a function 'mul' compatible with " rf"the arguments 'tuple, {dtype.__name__}'$",
            ):
                wp.mul(a_values, dtype(b_value))

            with test.assertRaisesRegex(
                RuntimeError,
                rf"Couldn't find a function 'mul' compatible with " rf"the arguments 'tuple, {np_type.__name__}'$",
            ):
                wp.mul(a_values, nps(np_type, b_value))


def test_vec_arg_support(test, device, dtype):
    np_type = wp.types.warp_type_to_np_dtype[dtype]
    vec_cls = wp.types.vector(3, dtype)
    values = (1.23, 2.34, 3.45)
    expected = wp.length(vec_cls(*values))

    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        test.assertAlmostEqual(wp.length(wpv(dtype, values)), expected)
        test.assertAlmostEqual(wp.length(npv(np_type, values)), expected)
        test.assertAlmostEqual(wp.length(np.array(npv(np_type, values))), expected)


def test_vec_vec_args_support(test, device, dtype):
    np_type = wp.types.warp_type_to_np_dtype[dtype]
    vec_cls = wp.types.vector(3, dtype)
    a_values = (1.23, 2.34, 3.45)
    b_values = (4.56, 5.67, 6.78)
    expected = wp.dot(vec_cls(*a_values), vec_cls(*b_values))

    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        test.assertEqual(wp.dot(vec_cls(*a_values), vec_cls(*b_values)), expected)
        test.assertEqual(wp.dot(vec_cls(*a_values), wpv(dtype, b_values)), expected)
        test.assertEqual(wp.dot(vec_cls(*a_values), npv(np_type, b_values)), expected)
        test.assertEqual(wp.dot(vec_cls(*a_values), np.array(npv(np_type, b_values))), expected)

        test.assertEqual(wp.dot(wpv(dtype, a_values), vec_cls(*b_values)), expected)
        test.assertEqual(wp.dot(wpv(dtype, a_values), wpv(dtype, b_values)), expected)
        test.assertEqual(wp.dot(wpv(dtype, a_values), npv(np_type, b_values)), expected)
        test.assertEqual(wp.dot(wpv(dtype, a_values), np.array(npv(np_type, b_values))), expected)

        test.assertEqual(wp.dot(npv(np_type, a_values), vec_cls(*b_values)), expected)
        test.assertEqual(wp.dot(npv(np_type, a_values), wpv(dtype, b_values)), expected)
        test.assertEqual(wp.dot(npv(np_type, a_values), npv(np_type, b_values)), expected)
        test.assertEqual(wp.dot(npv(np_type, a_values), np.array(npv(np_type, b_values))), expected)

        test.assertEqual(wp.dot(np.array(npv(np_type, a_values)), vec_cls(*b_values)), expected)
        test.assertEqual(wp.dot(np.array(npv(np_type, a_values)), wpv(dtype, b_values)), expected)
        test.assertEqual(wp.dot(np.array(npv(np_type, a_values)), npv(np_type, b_values)), expected)
        test.assertEqual(wp.dot(np.array(npv(np_type, a_values)), np.array(npv(np_type, b_values))), expected)

        if dtype is wp.float32:
            test.assertEqual(wp.dot(vec_cls(*a_values), b_values), expected)
            test.assertEqual(wp.dot(wpv(dtype, a_values), b_values), expected)
            test.assertEqual(wp.dot(npv(np_type, a_values), b_values), expected)
            test.assertEqual(wp.dot(a_values, b_values), expected)
            test.assertEqual(wp.dot(np.array(npv(np_type, a_values)), b_values), expected)

            test.assertEqual(wp.dot(a_values, vec_cls(*b_values)), expected)
            test.assertEqual(wp.dot(a_values, wpv(dtype, b_values)), expected)
            test.assertEqual(wp.dot(a_values, npv(np_type, b_values)), expected)
            test.assertEqual(wp.dot(a_values, np.array(npv(np_type, b_values))), expected)
        else:
            with test.assertRaisesRegex(
                RuntimeError,
                r"Couldn't find a function 'dot' compatible with the arguments 'vec_t, tuple'$",
            ):
                wp.dot(vec_cls(*a_values), b_values)

            with test.assertRaisesRegex(
                RuntimeError,
                r"Couldn't find a function 'dot' compatible with the arguments 'tuple, tuple'$",
            ):
                wp.dot(wpv(dtype, a_values), b_values)

            with test.assertRaisesRegex(
                RuntimeError,
                r"Couldn't find a function 'dot' compatible with the arguments 'tuple, tuple'$",
            ):
                wp.dot(npv(np_type, a_values), b_values)

            with test.assertRaisesRegex(
                RuntimeError,
                r"Couldn't find a function 'dot' compatible with the arguments 'ndarray, tuple'$",
            ):
                wp.dot(np.array(npv(np_type, a_values)), b_values)

            with test.assertRaisesRegex(
                RuntimeError,
                r"Couldn't find a function 'dot' compatible with the arguments 'tuple, vec_t'$",
            ):
                wp.dot(a_values, vec_cls(*b_values))

            with test.assertRaisesRegex(
                RuntimeError,
                r"Couldn't find a function 'dot' compatible with the arguments 'tuple, tuple'$",
            ):
                wp.dot(a_values, wpv(dtype, b_values))

            with test.assertRaisesRegex(
                RuntimeError,
                r"Couldn't find a function 'dot' compatible with the arguments 'tuple, tuple'$",
            ):
                wp.dot(a_values, npv(np_type, b_values))

            with test.assertRaisesRegex(
                RuntimeError,
                r"Couldn't find a function 'dot' compatible with the arguments 'tuple, ndarray'$",
            ):
                wp.dot(a_values, np.array(npv(np_type, b_values)))


def test_vec_float_args_support(test, device, dtype):
    np_type = wp.types.warp_type_to_np_dtype[dtype]
    vec_cls = wp.types.vector(3, dtype)
    a_values = (1.23, 2.34, 3.45)
    b_value = 4.56
    expected = wp.mul(vec_cls(*a_values), dtype(b_value))

    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        test.assertEqual(wp.mul(vec_cls(*a_values), dtype(b_value)), expected)
        test.assertEqual(wp.mul(vec_cls(*a_values), nps(np_type, b_value)), expected)

        test.assertEqual(wp.mul(wpv(dtype, a_values), dtype(b_value)), expected)
        test.assertEqual(wp.mul(wpv(dtype, a_values), nps(np_type, b_value)), expected)

        test.assertEqual(wp.mul(npv(np_type, a_values), dtype(b_value)), expected)
        test.assertEqual(wp.mul(npv(np_type, a_values), nps(np_type, b_value)), expected)

        test.assertEqual(wp.mul(np.array(npv(np_type, a_values)), dtype(b_value)), expected)
        test.assertEqual(wp.mul(np.array(npv(np_type, a_values)), nps(np_type, b_value)), expected)

        if dtype is wp.float32:
            test.assertEqual(wp.mul(vec_cls(*a_values), b_value), expected)
            test.assertEqual(wp.mul(wpv(dtype, a_values), b_value), expected)
            test.assertEqual(wp.mul(npv(np_type, a_values), b_value), expected)
            test.assertEqual(wp.mul(a_values, b_value), expected)
            test.assertEqual(wp.mul(np.array(npv(np_type, a_values)), b_value), expected)

            test.assertEqual(wp.mul(a_values, dtype(b_value)), expected)
            test.assertEqual(wp.mul(a_values, nps(np_type, b_value)), expected)
        else:
            with test.assertRaisesRegex(
                RuntimeError,
                r"Couldn't find a function 'mul' compatible with the arguments 'vec_t, float'$",
            ):
                wp.mul(vec_cls(*a_values), b_value)

            with test.assertRaisesRegex(
                RuntimeError,
                r"Couldn't find a function 'mul' compatible with the arguments 'tuple, float'$",
            ):
                wp.mul(wpv(dtype, a_values), b_value)

            with test.assertRaisesRegex(
                RuntimeError,
                r"Couldn't find a function 'mul' compatible with the arguments 'tuple, float'$",
            ):
                wp.mul(npv(np_type, a_values), b_value)

            with test.assertRaisesRegex(
                RuntimeError,
                r"Couldn't find a function 'mul' compatible with the arguments 'ndarray, float'$",
            ):
                wp.mul(np.array(npv(np_type, a_values)), b_value)

            with test.assertRaisesRegex(
                RuntimeError,
                rf"Couldn't find a function 'mul' compatible with " rf"the arguments 'tuple, {dtype.__name__}'$",
            ):
                wp.mul(a_values, dtype(b_value))

            with test.assertRaisesRegex(
                RuntimeError,
                rf"Couldn't find a function 'mul' compatible with " rf"the arguments 'tuple, {np_type.__name__}'$",
            ):
                wp.mul(a_values, nps(np_type, b_value))


class TestBuiltinsResolution(unittest.TestCase):
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
        self.assertAlmostEqual(result, expected, places=12)

        result = wp.sin(wp.float32(value))
        self.assertNotAlmostEqual(result, expected, places=12)
        self.assertAlmostEqual(result, expected, places=5)

        result = wp.sin(wp.float16(value))
        self.assertNotAlmostEqual(result, expected, places=5)
        self.assertAlmostEqual(result, expected, places=1)

        self.assertEqual(wp.sin(value), wp.sin(wp.float32(value)))

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
        values_2d = (values[0:2], values[2:4])
        expected = 5.78999999999999914735

        result = wp.trace(wp.mat22d(*values))
        self.assertAlmostEqual(result, expected, places=12)

        result = wp.trace(wp.mat22f(*values))
        self.assertNotAlmostEqual(result, expected, places=12)
        self.assertAlmostEqual(result, expected, places=5)

        result = wp.trace(wp.mat22h(*values))
        self.assertNotAlmostEqual(result, expected, places=5)
        self.assertAlmostEqual(result, expected, places=1)

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            self.assertEqual(wp.trace(values), wp.trace(wp.mat22f(*values)))
            self.assertEqual(wp.trace(values_2d), wp.trace(wp.mat22f(*values)))

    def test_mat33_arg_precision(self):
        values = (1.23, 2.34, 3.45, 4.56, 5.67, 6.78, 7.89, 8.90, 9.01)
        values_2d = (values[0:3], values[3:6], values[6:9])
        expected = 15.91000000000000014211

        result = wp.trace(wp.mat33d(*values))
        self.assertAlmostEqual(result, expected, places=12)

        result = wp.trace(wp.mat33f(*values))
        self.assertNotAlmostEqual(result, expected, places=12)
        self.assertAlmostEqual(result, expected, places=5)

        result = wp.trace(wp.mat33h(*values))
        self.assertNotAlmostEqual(result, expected, places=5)
        self.assertAlmostEqual(result, expected, places=1)

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            self.assertEqual(wp.trace(values), wp.trace(wp.mat33f(*values)))
            self.assertEqual(wp.trace(values_2d), wp.trace(wp.mat33f(*values)))

    def test_mat44_arg_precision(self):
        values = (1.23, 2.34, 3.45, 4.56, 5.67, 6.78, 7.89, 8.90, 9.01, 10.12, 11.23, 12.34, 13.45, 14.56, 15.67, 16.78)
        values_2d = (values[0:4], values[4:8], values[8:12], values[12:16])
        expected = 36.02000000000000312639

        result = wp.trace(wp.mat44d(*values))
        self.assertAlmostEqual(result, expected, places=12)

        result = wp.trace(wp.mat44f(*values))
        self.assertNotAlmostEqual(result, expected, places=12)
        self.assertAlmostEqual(result, expected, places=5)

        result = wp.trace(wp.mat44h(*values))
        self.assertNotAlmostEqual(result, expected, places=5)
        self.assertAlmostEqual(result, expected, places=1)

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            self.assertEqual(wp.trace(values), wp.trace(wp.mat44f(*values)))
            self.assertEqual(wp.trace(values_2d), wp.trace(wp.mat44f(*values)))

    def test_mat22_mat22_args_precision(self):
        a_values = (0.12, 1.23, 0.12, 1.23)
        a_values_2d = (a_values[0:2], a_values[2:4])
        b_values = (1.23, 0.12, 1.23, 0.12)
        b_values_2d = (b_values[0:2], b_values[2:4])
        expected = 0.59039999999999992486

        result = wp.ddot(wp.mat22d(*a_values), wp.mat22d(*b_values))
        self.assertAlmostEqual(result, expected, places=12)

        result = wp.ddot(wp.mat22f(*a_values), wp.mat22f(*b_values))
        self.assertNotAlmostEqual(result, expected, places=12)
        self.assertAlmostEqual(result, expected, places=5)

        result = wp.ddot(wp.mat22h(*a_values), wp.mat22h(*b_values))
        self.assertNotAlmostEqual(result, expected, places=5)
        self.assertAlmostEqual(result, expected, places=1)

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            self.assertEqual(wp.ddot(a_values, b_values), wp.ddot(wp.mat22f(*a_values), wp.mat22f(*b_values)))
            self.assertEqual(wp.ddot(a_values_2d, b_values_2d), wp.ddot(wp.mat22f(*a_values), wp.mat22f(*b_values)))
            self.assertEqual(wp.ddot(a_values, b_values_2d), wp.ddot(wp.mat22f(*a_values), wp.mat22f(*b_values)))
            self.assertEqual(wp.ddot(a_values_2d, b_values), wp.ddot(wp.mat22f(*a_values), wp.mat22f(*b_values)))

    def test_mat33_mat33_args_precision(self):
        a_values = (0.12, 1.23, 2.34, 0.12, 1.23, 2.34, 0.12, 1.23, 2.34)
        a_values_2d = (a_values[0:3], a_values[3:6], a_values[6:9])
        b_values = (2.34, 1.23, 0.12, 2.34, 1.23, 0.12, 2.34, 1.23, 0.12)
        b_values_2d = (b_values[0:3], b_values[3:6], b_values[6:9])
        expected = 6.22350000000000047606

        result = wp.ddot(wp.mat33d(*a_values), wp.mat33d(*b_values))
        self.assertAlmostEqual(result, expected, places=12)

        result = wp.ddot(wp.mat33f(*a_values), wp.mat33f(*b_values))
        self.assertNotAlmostEqual(result, expected, places=12)
        self.assertAlmostEqual(result, expected, places=5)

        result = wp.ddot(wp.mat33h(*a_values), wp.mat33h(*b_values))
        self.assertNotAlmostEqual(result, expected, places=5)
        self.assertAlmostEqual(result, expected, places=1)

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            self.assertEqual(wp.ddot(a_values, b_values), wp.ddot(wp.mat33f(*a_values), wp.mat33f(*b_values)))
            self.assertEqual(wp.ddot(a_values_2d, b_values_2d), wp.ddot(wp.mat33f(*a_values), wp.mat33f(*b_values)))
            self.assertEqual(wp.ddot(a_values, b_values_2d), wp.ddot(wp.mat33f(*a_values), wp.mat33f(*b_values)))
            self.assertEqual(wp.ddot(a_values_2d, b_values), wp.ddot(wp.mat33f(*a_values), wp.mat33f(*b_values)))

    def test_mat44_mat44_args(self):
        a_values = (0.12, 1.23, 2.34, 3.45, 0.12, 1.23, 2.34, 3.45, 0.12, 1.23, 2.34, 3.45, 0.12, 1.23, 2.34, 3.45)
        a_values_2d = (a_values[0:4], a_values[4:8], a_values[8:12], a_values[12:16])
        b_values = (3.45, 2.34, 1.23, 0.12, 3.45, 2.34, 1.23, 0.12, 3.45, 2.34, 1.23, 0.12, 3.45, 2.34, 1.23, 0.12)
        b_values_2d = (b_values[0:4], b_values[4:8], b_values[8:12], b_values[12:16])
        expected = 26.33760000000000189857

        result = wp.ddot(wp.mat44d(*a_values), wp.mat44d(*b_values))
        self.assertAlmostEqual(result, expected, places=12)

        result = wp.ddot(wp.mat44f(*a_values), wp.mat44f(*b_values))
        self.assertNotAlmostEqual(result, expected, places=12)
        self.assertAlmostEqual(result, expected, places=5)

        result = wp.ddot(wp.mat44h(*a_values), wp.mat44h(*b_values))
        self.assertNotAlmostEqual(result, expected, places=5)
        self.assertAlmostEqual(result, expected, places=1)

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            self.assertEqual(wp.ddot(a_values, b_values), wp.ddot(wp.mat44f(*a_values), wp.mat44f(*b_values)))
            self.assertEqual(wp.ddot(a_values_2d, b_values_2d), wp.ddot(wp.mat44f(*a_values), wp.mat44f(*b_values)))
            self.assertEqual(wp.ddot(a_values, b_values_2d), wp.ddot(wp.mat44f(*a_values), wp.mat44f(*b_values)))
            self.assertEqual(wp.ddot(a_values_2d, b_values), wp.ddot(wp.mat44f(*a_values), wp.mat44f(*b_values)))

    def test_mat22_float_args_precision(self):
        a_values = (1.23, 2.34, 3.45, 4.56)
        a_values_2d = (a_values[0:2], a_values[2:4])
        b_value = 0.12
        expected_00 = 0.14759999999999998122
        expected_01 = 0.28079999999999999405
        expected_10 = 0.41399999999999997913
        expected_11 = 0.54719999999999990870

        result = wp.mul(wp.mat22d(*a_values), wp.float64(b_value))
        self.assertAlmostEqual(result[0][0], expected_00, places=12)
        self.assertAlmostEqual(result[0][1], expected_01, places=12)
        self.assertAlmostEqual(result[1][0], expected_10, places=12)
        self.assertAlmostEqual(result[1][1], expected_11, places=12)

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
        self.assertNotAlmostEqual(result[0][0], expected_00, places=5)
        self.assertNotAlmostEqual(result[0][1], expected_01, places=5)
        self.assertNotAlmostEqual(result[1][0], expected_10, places=5)
        self.assertNotAlmostEqual(result[1][1], expected_11, places=5)
        self.assertAlmostEqual(result[0][0], expected_00, places=1)
        self.assertAlmostEqual(result[0][1], expected_01, places=1)
        self.assertAlmostEqual(result[1][0], expected_10, places=1)
        self.assertAlmostEqual(result[1][1], expected_11, places=1)

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            # Multiplying a 1-D tuple of length 4 is ambiguous because it could match
            # either the `vec4f` or `mat22f` overload. As a result, only the 2-D variant
            # of the tuple is expected to resolve correctly.
            self.assertEqual(wp.mul(a_values_2d, b_value), wp.mul(wp.mat22f(*a_values), wp.float32(b_value)))

    def test_mat33_float_args_precision(self):
        a_values = (1.23, 2.34, 3.45, 4.56, 5.67, 6.78, 7.89, 8.90, 9.01)
        a_values_2d = (a_values[0:3], a_values[3:6], a_values[6:9])
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
        self.assertAlmostEqual(result[0][0], expected_00, places=12)
        self.assertAlmostEqual(result[0][1], expected_01, places=12)
        self.assertAlmostEqual(result[0][2], expected_02, places=12)
        self.assertAlmostEqual(result[1][0], expected_10, places=12)
        self.assertAlmostEqual(result[1][1], expected_11, places=12)
        self.assertAlmostEqual(result[1][2], expected_12, places=12)
        self.assertAlmostEqual(result[2][0], expected_20, places=12)
        self.assertAlmostEqual(result[2][1], expected_21, places=12)
        self.assertAlmostEqual(result[2][2], expected_22, places=12)

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
        self.assertNotAlmostEqual(result[0][0], expected_00, places=5)
        self.assertNotAlmostEqual(result[0][1], expected_01, places=5)
        self.assertNotAlmostEqual(result[0][2], expected_02, places=5)
        self.assertNotAlmostEqual(result[1][0], expected_10, places=5)
        self.assertNotAlmostEqual(result[1][1], expected_11, places=5)
        self.assertNotAlmostEqual(result[1][2], expected_12, places=5)
        self.assertNotAlmostEqual(result[2][0], expected_20, places=5)
        self.assertNotAlmostEqual(result[2][1], expected_21, places=5)
        self.assertNotAlmostEqual(result[2][2], expected_22, places=5)
        self.assertAlmostEqual(result[0][0], expected_00, places=1)
        self.assertAlmostEqual(result[0][1], expected_01, places=1)
        self.assertAlmostEqual(result[0][2], expected_02, places=1)
        self.assertAlmostEqual(result[1][0], expected_10, places=1)
        self.assertAlmostEqual(result[1][1], expected_11, places=1)
        self.assertAlmostEqual(result[1][2], expected_12, places=1)
        self.assertAlmostEqual(result[2][0], expected_20, places=1)
        self.assertAlmostEqual(result[2][1], expected_21, places=1)
        self.assertAlmostEqual(result[2][2], expected_22, places=1)

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            self.assertEqual(wp.mul(a_values, b_value), wp.mul(wp.mat33f(*a_values), wp.float32(b_value)))
            self.assertEqual(wp.mul(a_values_2d, b_value), wp.mul(wp.mat33f(*a_values), wp.float32(b_value)))

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
        a_values_2d = (a_values[0:4], a_values[4:8], a_values[8:12], a_values[12:16])
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
        self.assertAlmostEqual(result[0][0], expected_00, places=12)
        self.assertAlmostEqual(result[0][1], expected_01, places=12)
        self.assertAlmostEqual(result[0][2], expected_02, places=12)
        self.assertAlmostEqual(result[0][3], expected_03, places=12)
        self.assertAlmostEqual(result[1][0], expected_10, places=12)
        self.assertAlmostEqual(result[1][1], expected_11, places=12)
        self.assertAlmostEqual(result[1][2], expected_12, places=12)
        self.assertAlmostEqual(result[1][3], expected_13, places=12)
        self.assertAlmostEqual(result[2][0], expected_20, places=12)
        self.assertAlmostEqual(result[2][1], expected_21, places=12)
        self.assertAlmostEqual(result[2][2], expected_22, places=12)
        self.assertAlmostEqual(result[2][3], expected_23, places=12)
        self.assertAlmostEqual(result[3][0], expected_30, places=12)
        self.assertAlmostEqual(result[3][1], expected_31, places=12)
        self.assertAlmostEqual(result[3][2], expected_32, places=12)
        self.assertAlmostEqual(result[3][3], expected_33, places=12)

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
        self.assertNotAlmostEqual(result[0][0], expected_00, places=5)
        self.assertNotAlmostEqual(result[0][1], expected_01, places=5)
        self.assertNotAlmostEqual(result[0][2], expected_02, places=5)
        self.assertNotAlmostEqual(result[0][3], expected_03, places=5)
        self.assertNotAlmostEqual(result[1][0], expected_10, places=5)
        self.assertNotAlmostEqual(result[1][1], expected_11, places=5)
        self.assertNotAlmostEqual(result[1][2], expected_12, places=5)
        self.assertNotAlmostEqual(result[1][3], expected_13, places=5)
        self.assertNotAlmostEqual(result[2][0], expected_20, places=5)
        self.assertNotAlmostEqual(result[2][1], expected_21, places=5)
        self.assertNotAlmostEqual(result[2][2], expected_22, places=5)
        self.assertNotAlmostEqual(result[2][3], expected_23, places=5)
        self.assertNotAlmostEqual(result[3][0], expected_30, places=5)
        self.assertNotAlmostEqual(result[3][1], expected_31, places=5)
        self.assertNotAlmostEqual(result[3][2], expected_32, places=5)
        self.assertNotAlmostEqual(result[3][3], expected_33, places=5)
        self.assertAlmostEqual(result[0][0], expected_00, places=1)
        self.assertAlmostEqual(result[0][1], expected_01, places=1)
        self.assertAlmostEqual(result[0][2], expected_02, places=1)
        self.assertAlmostEqual(result[0][3], expected_03, places=1)
        self.assertAlmostEqual(result[1][0], expected_10, places=1)
        self.assertAlmostEqual(result[1][1], expected_11, places=1)
        self.assertAlmostEqual(result[1][2], expected_12, places=1)
        self.assertAlmostEqual(result[1][3], expected_13, places=1)
        self.assertAlmostEqual(result[2][0], expected_20, places=1)
        self.assertAlmostEqual(result[2][1], expected_21, places=1)
        self.assertAlmostEqual(result[2][2], expected_22, places=1)
        self.assertAlmostEqual(result[2][3], expected_23, places=1)
        self.assertAlmostEqual(result[3][0], expected_30, places=1)
        self.assertAlmostEqual(result[3][1], expected_31, places=1)
        self.assertAlmostEqual(result[3][2], expected_32, places=1)
        self.assertAlmostEqual(result[3][3], expected_33, places=1)

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            self.assertEqual(wp.mul(a_values, b_value), wp.mul(wp.mat44f(*a_values), wp.float32(b_value)))
            self.assertEqual(wp.mul(a_values_2d, b_value), wp.mul(wp.mat44f(*a_values), wp.float32(b_value)))

    def test_vec2_arg_precision(self):
        values = (1.23, 2.34)
        expected = 2.64357712200722438922

        result = wp.length(wp.vec2d(*values))
        self.assertAlmostEqual(result, expected, places=12)

        result = wp.length(wp.vec2f(*values))
        self.assertNotAlmostEqual(result, expected, places=12)
        self.assertAlmostEqual(result, expected, places=5)

        result = wp.length(wp.vec2h(*values))
        self.assertNotAlmostEqual(result, expected, places=5)
        self.assertAlmostEqual(result, expected, places=1)

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            self.assertEqual(wp.length(values), wp.length(wp.vec2f(*values)))

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

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            self.assertEqual(wp.length_sq(values), wp.length_sq(wp.vec2i(*values)))

    def test_vec3_arg_precision(self):
        values = (1.23, 2.34, 3.45)
        expected = 4.34637780226247727455

        result = wp.length(wp.vec3d(*values))
        self.assertAlmostEqual(result, expected, places=12)

        result = wp.length(wp.vec3f(*values))
        self.assertNotAlmostEqual(result, expected, places=12)
        self.assertAlmostEqual(result, expected, places=5)

        result = wp.length(wp.vec3h(*values))
        self.assertNotAlmostEqual(result, expected, places=5)
        self.assertAlmostEqual(result, expected, places=1)

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            self.assertEqual(wp.length(values), wp.length(wp.vec3f(*values)))

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

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            self.assertEqual(wp.length_sq(values), wp.length_sq(wp.vec3i(*values)))

    def test_vec4_arg_precision(self):
        values = (1.23, 2.34, 3.45, 4.56)
        expected = 6.29957141399317777086

        result = wp.length(wp.vec4d(*values))
        self.assertAlmostEqual(result, expected, places=12)

        result = wp.length(wp.vec4f(*values))
        self.assertNotAlmostEqual(result, expected, places=12)
        self.assertAlmostEqual(result, expected, places=5)

        result = wp.length(wp.vec4h(*values))
        self.assertNotAlmostEqual(result, expected, places=5)
        self.assertAlmostEqual(result, expected, places=1)

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            self.assertEqual(wp.length(values), wp.length(wp.vec4f(*values)))

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

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            self.assertEqual(wp.length_sq(values), wp.length_sq(wp.vec4i(*values)))

    def test_vec2_vec2_args_precision(self):
        a_values = (1.23, 2.34)
        b_values = (3.45, 4.56)
        expected = 14.91389999999999815827

        result = wp.dot(wp.vec2d(*a_values), wp.vec2d(*b_values))
        self.assertAlmostEqual(result, expected, places=12)

        result = wp.dot(wp.vec2f(*a_values), wp.vec2f(*b_values))
        self.assertNotAlmostEqual(result, expected, places=12)
        self.assertAlmostEqual(result, expected, places=5)

        result = wp.dot(wp.vec2h(*a_values), wp.vec2h(*b_values))
        self.assertNotAlmostEqual(result, expected, places=5)
        self.assertAlmostEqual(result, expected, places=1)

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            self.assertEqual(wp.dot(a_values, b_values), wp.dot(wp.vec2f(*a_values), wp.vec2f(*b_values)))

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

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            self.assertEqual(wp.dot(values, values), wp.dot(wp.vec2i(*values), wp.vec2i(*values)))

    def test_vec3_vec3_args_precision(self):
        a_values = (1.23, 2.34, 3.45)
        b_values = (4.56, 5.67, 6.78)
        expected = 42.26760000000000161435

        result = wp.dot(wp.vec3d(*a_values), wp.vec3d(*b_values))
        self.assertAlmostEqual(result, expected, places=12)

        result = wp.dot(wp.vec3f(*a_values), wp.vec3f(*b_values))
        self.assertNotAlmostEqual(result, expected, places=12)
        self.assertAlmostEqual(result, expected, places=5)

        result = wp.dot(wp.vec3h(*a_values), wp.vec3h(*b_values))
        self.assertNotAlmostEqual(result, expected, places=5)
        self.assertAlmostEqual(result, expected, places=1)

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            self.assertEqual(wp.dot(a_values, b_values), wp.dot(wp.vec3f(*a_values), wp.vec3f(*b_values)))

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

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            self.assertEqual(wp.dot(values, values), wp.dot(wp.vec3i(*values), wp.vec3i(*values)))

    def test_vec4_vec4_args_precision(self):
        a_values = (1.23, 2.34, 3.45, 4.56)
        b_values = (5.67, 6.78, 7.89, 8.90)
        expected = 90.64379999999999881766

        result = wp.dot(wp.vec4d(*a_values), wp.vec4d(*b_values))
        self.assertAlmostEqual(result, expected, places=12)

        result = wp.dot(wp.vec4f(*a_values), wp.vec4f(*b_values))
        self.assertNotAlmostEqual(result, expected, places=12)
        self.assertAlmostEqual(result, expected, places=5)

        result = wp.dot(wp.vec4h(*a_values), wp.vec4h(*b_values))
        self.assertNotAlmostEqual(result, expected, places=5)
        self.assertAlmostEqual(result, expected, places=1)

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            self.assertEqual(wp.dot(a_values, b_values), wp.dot(wp.vec4f(*a_values), wp.vec4f(*b_values)))

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

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            self.assertEqual(wp.dot(values, values), wp.dot(wp.vec4i(*values), wp.vec4i(*values)))

    def test_vec2_float_args_precision(self):
        a_values = (1.23, 2.34)
        b_value = 3.45
        expected_x = 4.24350000000000004974
        expected_y = 8.07300000000000039790

        result = wp.mul(wp.vec2d(*a_values), wp.float64(b_value))
        self.assertAlmostEqual(result[0], expected_x, places=12)
        self.assertAlmostEqual(result[1], expected_y, places=12)

        result = wp.mul(wp.vec2f(*a_values), wp.float32(b_value))
        self.assertNotAlmostEqual(result[0], expected_x, places=12)
        self.assertNotAlmostEqual(result[1], expected_y, places=12)
        self.assertAlmostEqual(result[0], expected_x, places=5)
        self.assertAlmostEqual(result[1], expected_y, places=5)

        result = wp.mul(wp.vec2h(*a_values), wp.float16(b_value))
        self.assertNotAlmostEqual(result[0], expected_x, places=5)
        self.assertNotAlmostEqual(result[1], expected_y, places=5)
        self.assertAlmostEqual(result[0], expected_x, places=1)
        self.assertAlmostEqual(result[1], expected_y, places=1)

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            self.assertEqual(wp.mul(a_values, b_value), wp.mul(wp.vec2f(*a_values), wp.float32(b_value)))

    def test_vec3_float_args_precision(self):
        a_values = (1.23, 2.34, 3.45)
        b_value = 4.56
        expected_x = 5.60879999999999956373
        expected_y = 10.67039999999999899671
        expected_z = 15.73199999999999931788

        result = wp.mul(wp.vec3d(*a_values), wp.float64(b_value))
        self.assertAlmostEqual(result[0], expected_x, places=12)
        self.assertAlmostEqual(result[1], expected_y, places=12)
        self.assertAlmostEqual(result[2], expected_z, places=12)

        result = wp.mul(wp.vec3f(*a_values), wp.float32(b_value))
        self.assertNotAlmostEqual(result[0], expected_x, places=12)
        self.assertNotAlmostEqual(result[1], expected_y, places=12)
        self.assertNotAlmostEqual(result[2], expected_z, places=12)
        self.assertAlmostEqual(result[0], expected_x, places=5)
        self.assertAlmostEqual(result[1], expected_y, places=5)
        self.assertAlmostEqual(result[2], expected_z, places=5)

        result = wp.mul(wp.vec3h(*a_values), wp.float16(b_value))
        self.assertNotAlmostEqual(result[0], expected_x, places=5)
        self.assertNotAlmostEqual(result[1], expected_y, places=5)
        self.assertNotAlmostEqual(result[2], expected_z, places=5)
        self.assertAlmostEqual(result[0], expected_x, places=1)
        self.assertAlmostEqual(result[1], expected_y, places=1)
        self.assertAlmostEqual(result[2], expected_z, places=1)

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            self.assertEqual(wp.mul(a_values, b_value), wp.mul(wp.vec3f(*a_values), wp.float32(b_value)))

    def test_vec4_float_args_precision(self):
        a_values = (1.23, 2.34, 3.45, 4.56)
        b_value = 5.67
        expected_x = 6.97409999999999996589
        expected_y = 13.26779999999999937188
        expected_z = 19.56150000000000233058
        expected_w = 25.85519999999999640750

        result = wp.mul(wp.vec4d(*a_values), wp.float64(b_value))
        self.assertAlmostEqual(result[0], expected_x, places=12)
        self.assertAlmostEqual(result[1], expected_y, places=12)
        self.assertAlmostEqual(result[2], expected_z, places=12)
        self.assertAlmostEqual(result[3], expected_w, places=12)

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
        self.assertNotAlmostEqual(result[0], expected_x, places=5)
        self.assertNotAlmostEqual(result[1], expected_y, places=5)
        self.assertNotAlmostEqual(result[2], expected_z, places=5)
        self.assertNotAlmostEqual(result[3], expected_w, places=5)
        self.assertAlmostEqual(result[0], expected_x, places=1)
        self.assertAlmostEqual(result[1], expected_y, places=1)
        self.assertAlmostEqual(result[2], expected_z, places=1)
        self.assertAlmostEqual(result[3], expected_w, places=1)

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            self.assertEqual(wp.mul(a_values, b_value), wp.mul(wp.vec4f(*a_values), wp.float32(b_value)))


for dtype in wp.types.int_types:
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

for dtype in wp.types.float_types:
    add_function_test(
        TestBuiltinsResolution,
        f"test_float_arg_support_{dtype.__name__}",
        test_float_arg_support,
        dtype=dtype,
    )
    add_function_test(
        TestBuiltinsResolution,
        f"test_mat_arg_support_{dtype.__name__}",
        test_mat_arg_support,
        dtype=dtype,
    )
    add_function_test(
        TestBuiltinsResolution,
        f"test_mat_mat_args_support_{dtype.__name__}",
        test_mat_mat_args_support,
        dtype=dtype,
    )
    add_function_test(
        TestBuiltinsResolution,
        f"test_mat_float_args_support_{dtype.__name__}",
        test_mat_float_args_support,
        dtype=dtype,
    )
    add_function_test(
        TestBuiltinsResolution,
        f"test_vec_arg_support_{dtype.__name__}",
        test_vec_arg_support,
        dtype=dtype,
    )
    add_function_test(
        TestBuiltinsResolution,
        f"test_vec_vec_args_support_{dtype.__name__}",
        test_vec_vec_args_support,
        dtype=dtype,
    )
    add_function_test(
        TestBuiltinsResolution,
        f"test_vec_float_args_support_{dtype.__name__}",
        test_vec_float_args_support,
        dtype=dtype,
    )


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
