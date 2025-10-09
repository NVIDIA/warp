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

import unittest

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *

np_signed_int_types = [
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.byte,
]

np_unsigned_int_types = [
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.ubyte,
]

np_int_types = np_signed_int_types + np_unsigned_int_types

np_float_types = [np.float16, np.float32, np.float64]

np_scalar_types = np_int_types + np_float_types


def test_py_arithmetic_ops(test, device, dtype):
    wptype = wp.dtype_from_numpy(np.dtype(dtype))

    def make_scalar(value):
        if wptype in wp._src.types.int_types:
            # Cast to the correct integer type to simulate wrapping.
            return wptype._type_(value).value

        return value

    a = wptype(1)
    test.assertAlmostEqual(+a, make_scalar(1))
    test.assertAlmostEqual(-a, make_scalar(-1))
    test.assertAlmostEqual(a + wptype(5), make_scalar(6))
    test.assertAlmostEqual(a - wptype(5), make_scalar(-4))
    test.assertAlmostEqual(a % wptype(2), make_scalar(1))

    a = wptype(2)
    test.assertAlmostEqual(a * wptype(2), make_scalar(4))
    test.assertAlmostEqual(wptype(2) * a, make_scalar(4))
    test.assertAlmostEqual(a / wptype(2), make_scalar(1))
    test.assertAlmostEqual(wptype(24) / a, make_scalar(12))
    test.assertAlmostEqual(a % wptype(2), make_scalar(0))


def test_py_math_ops(test, device, dtype):
    wptype = wp.dtype_from_numpy(np.dtype(dtype))

    a = wptype(1)
    test.assertAlmostEqual(wp.abs(a), 1)

    if dtype in np_float_types:
        test.assertAlmostEqual(wp.sin(a), 0.84147098480789650488, places=3)
        test.assertAlmostEqual(wp.radians(a), 0.01745329251994329577, places=5)


def test_py_array_ops(test, device, dtype):
    wptype = wp.dtype_from_numpy(np.dtype(dtype))

    tol = {
        np.float16: 5.0e-3,
        np.float32: 1.0e-6,
        np.float64: 1.0e-8,
    }.get(dtype, 0)

    def make_array(*args):
        if wptype in wp._src.types.int_types:
            # Cast to the correct integer type to simulate wrapping.
            return np.array(tuple(wptype._type_(x).value for x in args), dtype=dtype)

        return np.array(args, dtype=dtype)

    arr = wp.array((1, 2, 3), dtype=wptype)
    s = wptype(5)
    assert_np_equal((arr + s).numpy(), make_array(6, 7, 8), tol=tol)
    assert_np_equal((s + arr).numpy(), make_array(6, 7, 8), tol=tol)
    assert_np_equal((arr - s).numpy(), make_array(-4, -3, -2), tol=tol)
    assert_np_equal((s - arr).numpy(), make_array(4, 3, 2), tol=tol)
    assert_np_equal((arr * s).numpy(), make_array(5, 10, 15), tol=tol)
    assert_np_equal((s * arr).numpy(), make_array(5, 10, 15), tol=tol)
    assert_np_equal((arr % s).numpy(), make_array(1, 2, 3), tol=tol)
    assert_np_equal((s % arr).numpy(), make_array(0, 1, 2), tol=tol)

    if dtype in np_float_types:
        assert_np_equal((arr / s).numpy(), make_array(0.2, 0.4, 0.6), tol=tol)
        assert_np_equal((s / arr).numpy(), make_array(5.0, 2.5, 1.666666667), tol=tol)


devices = get_test_devices()


class TestScalarOps(unittest.TestCase):
    pass


for dtype in np_scalar_types:
    add_function_test(
        TestScalarOps, f"test_py_arithmetic_ops_{dtype.__name__}", test_py_arithmetic_ops, devices=None, dtype=dtype
    )
    add_function_test(TestScalarOps, f"test_py_math_ops_{dtype.__name__}", test_py_math_ops, devices=None, dtype=dtype)
    add_function_test(
        TestScalarOps, f"test_py_array_ops_{dtype.__name__}", test_py_array_ops, devices=None, dtype=dtype
    )


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=True)
