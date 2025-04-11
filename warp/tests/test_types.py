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

import unittest

from warp.tests.unittest_utils import *


def test_integers(test, device, dtype):
    value = dtype(0)
    test.assertIsInstance(bool(value), bool)
    test.assertIsInstance(int(value), int)
    test.assertIsInstance(float(value), float)
    test.assertEqual(bool(value), False)
    test.assertEqual(int(value), 0)
    test.assertEqual(float(value), 0.0)
    try:
        ctypes.c_bool(value)
        ctypes.c_int(value)
        ctypes.c_float(value)
    except Exception:
        test.fail()

    value = dtype(123)
    test.assertIsInstance(bool(value), bool)
    test.assertIsInstance(int(value), int)
    test.assertIsInstance(float(value), float)
    test.assertEqual(bool(value), True)
    test.assertEqual(int(value), 123)
    test.assertEqual(float(value), 123.0)
    try:
        ctypes.c_bool(value)
        ctypes.c_int(value)
        ctypes.c_float(value)
    except Exception:
        test.fail()


def test_floats(test, device, dtype):
    value = dtype(0.0)
    test.assertIsInstance(bool(value), bool)
    test.assertIsInstance(int(value), int)
    test.assertIsInstance(float(value), float)
    test.assertEqual(bool(value), False)
    test.assertEqual(int(value), 0)
    test.assertEqual(float(value), 0.0)
    try:
        ctypes.c_bool(value)
        ctypes.c_float(value)
    except Exception:
        test.fail()

    value = dtype(1.25)
    test.assertIsInstance(bool(value), bool)
    test.assertIsInstance(int(value), int)
    test.assertIsInstance(float(value), float)
    test.assertEqual(bool(value), True)
    test.assertEqual(int(value), 1)
    test.assertEqual(float(value), 1.25)
    try:
        ctypes.c_bool(value)
        ctypes.c_float(value)
    except Exception:
        test.fail()


def test_vector(test, device, dtype):
    def make_scalar(x):
        # Cast to the correct integer type to simulate wrapping.
        if dtype in wp.types.int_types:
            return dtype._type_(x).value

        return x

    def make_vec(*args):
        if dtype in wp.types.int_types:
            # Cast to the correct integer type to simulate wrapping.
            return tuple(dtype._type_(x).value for x in args)

        return args

    vec3_cls = wp.vec(3, dtype)
    vec4_cls = wp.vec(4, dtype)

    v = vec4_cls(1, 2, 3, 4)
    test.assertEqual(v[0], make_scalar(1))
    test.assertEqual(v.x, make_scalar(1))
    test.assertEqual(v.y, make_scalar(2))
    test.assertEqual(v.z, make_scalar(3))
    test.assertEqual(v.w, make_scalar(4))
    test.assertSequenceEqual(v[0:2], make_vec(1, 2))
    test.assertSequenceEqual(v, make_vec(1, 2, 3, 4))

    v[0] = -1
    test.assertEqual(v[0], make_scalar(-1))
    test.assertEqual(v.x, make_scalar(-1))
    test.assertEqual(v.y, make_scalar(2))
    test.assertEqual(v.z, make_scalar(3))
    test.assertEqual(v.w, make_scalar(4))
    test.assertSequenceEqual(v[0:2], make_vec(-1, 2))
    test.assertSequenceEqual(v, make_vec(-1, 2, 3, 4))

    v[1:3] = (-2, -3)
    test.assertEqual(v[0], make_scalar(-1))
    test.assertEqual(v.x, make_scalar(-1))
    test.assertEqual(v.y, make_scalar(-2))
    test.assertEqual(v.z, make_scalar(-3))
    test.assertEqual(v.w, make_scalar(4))
    test.assertSequenceEqual(v[0:2], make_vec(-1, -2))
    test.assertSequenceEqual(v, make_vec(-1, -2, -3, 4))

    v.x = 1
    test.assertEqual(v[0], make_scalar(1))
    test.assertEqual(v.x, make_scalar(1))
    test.assertEqual(v.y, make_scalar(-2))
    test.assertEqual(v.z, make_scalar(-3))
    test.assertEqual(v.w, make_scalar(4))
    test.assertSequenceEqual(v[0:2], make_vec(1, -2))
    test.assertSequenceEqual(v, make_vec(1, -2, -3, 4))

    v = vec3_cls(2, 4, 6)
    test.assertSequenceEqual(+v, make_vec(2, 4, 6))
    test.assertSequenceEqual(-v, make_vec(-2, -4, -6))
    test.assertSequenceEqual(v + vec3_cls(1, 1, 1), make_vec(3, 5, 7))
    test.assertSequenceEqual(v - vec3_cls(1, 1, 1), make_vec(1, 3, 5))
    test.assertSequenceEqual(v * dtype(2), make_vec(4, 8, 12))
    test.assertSequenceEqual(dtype(2) * v, make_vec(4, 8, 12))
    test.assertSequenceEqual(v / dtype(2), make_vec(1, 2, 3))
    test.assertSequenceEqual(dtype(12) / v, make_vec(6, 3, 2))

    test.assertTrue(v != vec3_cls(1, 2, 3))
    test.assertEqual(str(v), "[{}]".format(", ".join(str(x) for x in v)))

    # Check added purely for coverage reasons but is this really a desired
    # behaviour? Not allowing to define new attributes using systems like
    # `__slots__` could help improving memory usage.
    v.foo = 123
    test.assertEqual(v.foo, 123)


devices = [x for x in get_test_devices() if x.is_cpu]


class TestTypes(unittest.TestCase):
    def test_bool(self):
        value = wp.bool(False)
        self.assertIsInstance(bool(value), bool)
        self.assertIsInstance(int(value), int)
        self.assertIsInstance(float(value), float)
        self.assertEqual(bool(value), False)
        self.assertEqual(int(value), 0)
        self.assertEqual(float(value), 0.0)
        try:
            ctypes.c_bool(value)
        except Exception:
            self.fail()

        value = wp.bool(True)
        self.assertIsInstance(bool(value), bool)
        self.assertIsInstance(int(value), int)
        self.assertIsInstance(float(value), float)
        self.assertEqual(bool(value), True)
        self.assertEqual(int(value), 1)
        self.assertEqual(float(value), 1.0)
        try:
            ctypes.c_bool(value)
        except Exception:
            self.fail()

        value = wp.bool(0.0)
        self.assertIsInstance(bool(value), bool)
        self.assertIsInstance(int(value), int)
        self.assertIsInstance(float(value), float)
        self.assertEqual(bool(value), False)
        self.assertEqual(int(value), 0)
        self.assertEqual(float(value), 0.0)
        try:
            ctypes.c_bool(value)
        except Exception:
            self.fail()

        value = wp.bool(123)
        self.assertIsInstance(bool(value), bool)
        self.assertIsInstance(int(value), int)
        self.assertIsInstance(float(value), float)
        self.assertEqual(bool(value), True)
        self.assertEqual(int(value), 1)
        self.assertEqual(float(value), 1.0)
        try:
            ctypes.c_bool(value)
        except Exception:
            self.fail()

    def test_constant(self):
        const = wp.constant(123)
        self.assertEqual(const, 123)

        const = wp.constant(1.25)
        self.assertEqual(const, 1.25)

        const = wp.constant(True)
        self.assertEqual(const, True)

        const = wp.constant(wp.float16(1.25))
        self.assertEqual(const.value, 1.25)

        const = wp.constant(wp.int16(123))
        self.assertEqual(const.value, 123)

        const = wp.constant(wp.vec3i(1, 2, 3))
        self.assertEqual(const, wp.vec3i(1, 2, 3))

    def test_constant_error_invalid_type(self):
        with self.assertRaisesRegex(TypeError, r"Invalid constant type: <class 'tuple'>$"):
            wp.constant((1, 2, 3))

    def test_vector_assign(self):
        v = wp.vec3s()
        v[0] = 1
        v[1] = wp.int8(2)
        v[2] = np.int8(3)
        self.assertEqual(v, (1, 2, 3))

        v = wp.vec3h()
        v[0] = 1.0
        v[1] = wp.float16(2.0)
        v[2] = np.float16(3.0)
        self.assertEqual(v, (1.0, 2.0, 3.0))

    def test_vector_error_invalid_arg_count(self):
        with self.assertRaisesRegex(
            ValueError, r"Invalid number of arguments in vector constructor, expected 3 elements, got 2$"
        ):
            wp.vec3(1, 2)

    def test_vector_error_invalid_ptr(self):
        with self.assertRaisesRegex(RuntimeError, r"NULL pointer exception"):
            wp.vec3.from_ptr(0)

    def test_vector_error_invalid_get_item_key(self):
        v = wp.vec3(1, 2, 3)

        with self.assertRaisesRegex(KeyError, r"Invalid key None, expected int or slice"):
            v[None]

    def test_vector_error_invalid_set_item_key(self):
        v = wp.vec3(1, 2, 3)
        with self.assertRaisesRegex(KeyError, r"Invalid key None, expected int or slice"):
            v[None] = 0

    def test_vector_error_invalid_set_item_value(self):
        v1 = wp.vec3i(1, 2, 3)
        v2 = wp.vec3h(1, 2, 3)

        with self.assertRaisesRegex(TypeError, r"Expected to assign a `int32` value but got `str` instead"):
            v1[0] = "123.0"

        with self.assertRaisesRegex(
            TypeError, r"Expected to assign a slice from a sequence of values but got `int` instead"
        ):
            v1[:] = 123

        with self.assertRaisesRegex(
            TypeError, r"Expected to assign a slice from a sequence of `int32` values but got `vec3i` instead"
        ):
            v1[:1] = (v1,)

        with self.assertRaisesRegex(ValueError, r"Can only assign sequence of same size"):
            v1[:1] = (1, 2)

        with self.assertRaisesRegex(
            TypeError, r"Expected to assign a slice from a sequence of `float16` values but got `vec3h` instead"
        ):
            v2[:1] = (v2,)

    def test_matrix(self):
        for dtype in (*wp.types.float_types, float):

            def make_scalar(x, dtype=dtype):
                # Cast to the correct integer type to simulate wrapping.
                if dtype in wp.types.int_types:
                    return dtype._type_(x).value

                return x

            def make_vec(*args, dtype=dtype):
                if dtype in wp.types.int_types:
                    # Cast to the correct integer type to simulate wrapping.
                    return tuple(dtype._type_(x).value for x in args)

                return args

            def make_mat(*args, dtype=dtype):
                if dtype in wp.types.int_types:
                    # Cast to the correct integer type to simulate wrapping.
                    return tuple(tuple(dtype._type_(x).value for x in row) for row in args)

                return args

            mat22_cls = wp.mat((2, 2), dtype)
            mat33_cls = wp.mat((3, 3), dtype)
            vec2_cls = wp.vec(2, dtype)

            m = mat33_cls(((1, 2, 3), (4, 5, 6), (7, 8, 9)))
            self.assertEqual(m[0][0], make_scalar(1))
            self.assertEqual(m[0][1], make_scalar(2))
            self.assertEqual(m[0][2], make_scalar(3))
            self.assertEqual(m[1][0], make_scalar(4))
            self.assertEqual(m[1][1], make_scalar(5))
            self.assertEqual(m[1][2], make_scalar(6))
            self.assertEqual(m[2][0], make_scalar(7))
            self.assertEqual(m[2][1], make_scalar(8))
            self.assertEqual(m[2][2], make_scalar(9))
            self.assertEqual(m[0, 0], make_scalar(1))
            self.assertEqual(m[0, 1], make_scalar(2))
            self.assertEqual(m[0, 2], make_scalar(3))
            self.assertEqual(m[1, 0], make_scalar(4))
            self.assertEqual(m[1, 1], make_scalar(5))
            self.assertEqual(m[1, 2], make_scalar(6))
            self.assertEqual(m[2, 0], make_scalar(7))
            self.assertEqual(m[2, 1], make_scalar(8))
            self.assertEqual(m[2, 2], make_scalar(9))
            self.assertSequenceEqual(m[0], make_vec(1, 2, 3))
            self.assertSequenceEqual(m[1], make_vec(4, 5, 6))
            self.assertSequenceEqual(m[2], make_vec(7, 8, 9))
            self.assertSequenceEqual(m[0][1:3], make_vec(2, 3))
            self.assertSequenceEqual(m[1][0:2], make_vec(4, 5))
            self.assertSequenceEqual(m[2][0:3], make_vec(7, 8, 9))
            # self.assertSequenceEqual(m[0, 1:3], make_vec(2, 3))
            # self.assertSequenceEqual(m[1, 0:2], make_vec(4, 5))
            # self.assertSequenceEqual(m[2, 0:3], make_vec(7, 8, 9))
            self.assertSequenceEqual(m, make_mat((1, 2, 3), (4, 5, 6), (7, 8, 9)))

            m[1, 0] = -4
            self.assertEqual(m[0][0], make_scalar(1))
            self.assertEqual(m[0][1], make_scalar(2))
            self.assertEqual(m[0][2], make_scalar(3))
            self.assertEqual(m[1][0], make_scalar(-4))
            self.assertEqual(m[1][1], make_scalar(5))
            self.assertEqual(m[1][2], make_scalar(6))
            self.assertEqual(m[2][0], make_scalar(7))
            self.assertEqual(m[2][1], make_scalar(8))
            self.assertEqual(m[2][2], make_scalar(9))
            self.assertEqual(m[0, 0], make_scalar(1))
            self.assertEqual(m[0, 1], make_scalar(2))
            self.assertEqual(m[0, 2], make_scalar(3))
            self.assertEqual(m[1, 0], make_scalar(-4))
            self.assertEqual(m[1, 1], make_scalar(5))
            self.assertEqual(m[1, 2], make_scalar(6))
            self.assertEqual(m[2, 0], make_scalar(7))
            self.assertEqual(m[2, 1], make_scalar(8))
            self.assertEqual(m[2, 2], make_scalar(9))
            self.assertSequenceEqual(m[0], make_vec(1, 2, 3))
            self.assertSequenceEqual(m[1], make_vec(-4, 5, 6))
            self.assertSequenceEqual(m[2], make_vec(7, 8, 9))
            self.assertSequenceEqual(m[0][1:3], make_vec(2, 3))
            self.assertSequenceEqual(m[1][0:2], make_vec(-4, 5))
            self.assertSequenceEqual(m[2][0:3], make_vec(7, 8, 9))
            # self.assertSequenceEqual(m[0, 1:3], make_vec(2, 3))
            # self.assertSequenceEqual(m[1, 0:2], make_vec(-4, 5))
            # self.assertSequenceEqual(m[2, 0:3], make_vec(7, 8, 9))
            self.assertSequenceEqual(m, make_mat((1, 2, 3), (-4, 5, 6), (7, 8, 9)))

            m[2] = (-7, 8, -9)
            self.assertEqual(m[0][0], make_scalar(1))
            self.assertEqual(m[0][1], make_scalar(2))
            self.assertEqual(m[0][2], make_scalar(3))
            self.assertEqual(m[1][0], make_scalar(-4))
            self.assertEqual(m[1][1], make_scalar(5))
            self.assertEqual(m[1][2], make_scalar(6))
            self.assertEqual(m[2][0], make_scalar(-7))
            self.assertEqual(m[2][1], make_scalar(8))
            self.assertEqual(m[2][2], make_scalar(-9))
            self.assertEqual(m[0, 0], make_scalar(1))
            self.assertEqual(m[0, 1], make_scalar(2))
            self.assertEqual(m[0, 2], make_scalar(3))
            self.assertEqual(m[1, 0], make_scalar(-4))
            self.assertEqual(m[1, 1], make_scalar(5))
            self.assertEqual(m[1, 2], make_scalar(6))
            self.assertEqual(m[2, 0], make_scalar(-7))
            self.assertEqual(m[2, 1], make_scalar(8))
            self.assertEqual(m[2, 2], make_scalar(-9))
            self.assertSequenceEqual(m[0], make_vec(1, 2, 3))
            self.assertSequenceEqual(m[1], make_vec(-4, 5, 6))
            self.assertSequenceEqual(m[2], make_vec(-7, 8, -9))
            self.assertSequenceEqual(m[0][1:3], make_vec(2, 3))
            self.assertSequenceEqual(m[1][0:2], make_vec(-4, 5))
            self.assertSequenceEqual(m[2][0:3], make_vec(-7, 8, -9))
            # self.assertSequenceEqual(m[0, 1:3], make_vec(2, 3))
            # self.assertSequenceEqual(m[1, 0:2], make_vec(-4, 5))
            # self.assertSequenceEqual(m[2, 0:3], make_vec(-7, 8, -9))
            self.assertSequenceEqual(m, make_mat((1, 2, 3), (-4, 5, 6), (-7, 8, -9)))

            m = mat22_cls(2, 4, 6, 8)
            self.assertSequenceEqual(+m, make_mat((2, 4), (6, 8)))
            self.assertSequenceEqual(-m, make_mat((-2, -4), (-6, -8)))
            self.assertSequenceEqual(m + mat22_cls(1, 1, 1, 1), make_mat((3, 5), (7, 9)))
            self.assertSequenceEqual(m - mat22_cls(1, 1, 1, 1), make_mat((1, 3), (5, 7)))
            self.assertSequenceEqual(m * dtype(2), make_mat((4, 8), (12, 16)))
            self.assertSequenceEqual(dtype(2) * m, make_mat((4, 8), (12, 16)))
            self.assertSequenceEqual(m / dtype(2), make_mat((1, 2), (3, 4)))
            self.assertSequenceEqual(dtype(24) / m, make_mat((12, 6), (4, 3)))

            self.assertSequenceEqual(m * vec2_cls(1, 2), make_vec(10, 22))
            self.assertSequenceEqual(m @ vec2_cls(1, 2), make_vec(10, 22))
            self.assertSequenceEqual(vec2_cls(1, 2) * m, make_vec(14, 20))
            self.assertSequenceEqual(vec2_cls(1, 2) @ m, make_vec(14, 20))

            self.assertTrue(m != mat22_cls(1, 2, 3, 4))
            self.assertEqual(
                str(m),
                "[{}]".format(",\n ".join("[{}]".format(", ".join(str(y) for y in m[x])) for x in range(m._shape_[0]))),
            )

            # Check added purely for coverage reasons but is this really a desired
            # behaviour? Not allowing to define new attributes using systems like
            # `__slots__` could help improving memory usage.
            m.foo = 123
            self.assertEqual(m.foo, 123)

    def test_matrix_error_invalid_arg_count(self):
        with self.assertRaisesRegex(
            ValueError, r"Invalid number of arguments in matrix constructor, expected 4 elements, got 3$"
        ):
            wp.mat22(1, 2, 3)

    def test_matrix_error_invalid_row_count(self):
        with self.assertRaisesRegex(
            TypeError, r"Invalid argument in matrix constructor, expected row of length 2, got \(1, 2, 3\)$"
        ):
            wp.mat22((1, 2, 3), (3, 4, 5))

    def test_matrix_error_invalid_ptr(self):
        with self.assertRaisesRegex(RuntimeError, r"NULL pointer exception"):
            wp.mat22.from_ptr(0)

    def test_matrix_error_invalid_set_row_index(self):
        m = wp.mat22(1, 2, 3, 4)
        with self.assertRaisesRegex(IndexError, r"Invalid row index$"):
            m.set_row(2, (0, 0))

    def test_matrix_error_invalid_get_item_key(self):
        m = wp.mat22(1, 2, 3, 4)

        with self.assertRaisesRegex(KeyError, r"Invalid key None, expected int or pair of ints"):
            m[None]

    def test_matrix_error_invalid_get_item_key_length(self):
        m = wp.mat22(1, 2, 3, 4)

        with self.assertRaisesRegex(KeyError, r"Invalid key, expected one or two indices, got 3"):
            m[0, 1, 2]

    def test_matrix_error_invalid_set_item_key(self):
        m = wp.mat22(1, 2, 3, 4)
        with self.assertRaisesRegex(KeyError, r"Invalid key None, expected int or pair of ints"):
            m[None] = 0

    def test_matrix_error_invalid_set_item_key_length(self):
        m = wp.mat22(1, 2, 3, 4)

        with self.assertRaisesRegex(KeyError, r"Invalid key, expected one or two indices, got 3"):
            m[0, 1, 2] = (0, 0)

    def test_matrix_error_invalid_set_item_value(self):
        m = wp.mat22h(1, 2, 3, 4)

        with self.assertRaisesRegex(TypeError, r"Expected to assign a `float16` value but got `str` instead"):
            m[0, 0] = "123.0"

        with self.assertRaisesRegex(TypeError, r"Expected to assign a `float16` value but got `str` instead"):
            m[0][0] = "123.0"

        with self.assertRaisesRegex(
            TypeError, r"Expected to assign a slice from a sequence of values but got `int` instead"
        ):
            m[0] = 123

        with self.assertRaisesRegex(
            TypeError, r"Expected to assign a slice from a sequence of `float16` values but got `mat22h` instead"
        ):
            m[0] = (m,)

        with self.assertRaisesRegex(
            KeyError, r"Slices are not supported when indexing matrices using the `m\[start:end\]` notation"
        ):
            m[:] = 123

        with self.assertRaisesRegex(
            KeyError, r"Slices are not supported when indexing matrices using the `m\[i, j\]` notation"
        ):
            m[0, :1] = (123,)

        with self.assertRaisesRegex(ValueError, r"Can only assign sequence of same size"):
            m[0][:1] = (1, 2)

    def test_dtype_from_numpy(self):
        import numpy as np

        def test_conversions(np_type, warp_type):
            self.assertEqual(wp.dtype_from_numpy(np_type), warp_type)
            self.assertEqual(wp.dtype_from_numpy(np.dtype(np_type)), warp_type)

        test_conversions(np.float16, wp.float16)
        test_conversions(np.float32, wp.float32)
        test_conversions(np.float64, wp.float64)
        test_conversions(np.int8, wp.int8)
        test_conversions(np.int16, wp.int16)
        test_conversions(np.int32, wp.int32)
        test_conversions(np.int64, wp.int64)
        test_conversions(np.uint8, wp.uint8)
        test_conversions(np.uint16, wp.uint16)
        test_conversions(np.uint32, wp.uint32)
        test_conversions(np.uint64, wp.uint64)
        test_conversions(np.bool_, wp.bool)
        test_conversions(np.byte, wp.int8)
        test_conversions(np.ubyte, wp.uint8)

    def test_dtype_to_numpy(self):
        import numpy as np

        def test_conversions(warp_type, np_type):
            self.assertEqual(wp.dtype_to_numpy(warp_type), np_type)

        test_conversions(wp.float16, np.float16)
        test_conversions(wp.float32, np.float32)
        test_conversions(wp.float64, np.float64)
        test_conversions(wp.int8, np.int8)
        test_conversions(wp.int16, np.int16)
        test_conversions(wp.int32, np.int32)
        test_conversions(wp.int64, np.int64)
        test_conversions(wp.uint8, np.uint8)
        test_conversions(wp.uint16, np.uint16)
        test_conversions(wp.uint32, np.uint32)
        test_conversions(wp.uint64, np.uint64)
        test_conversions(wp.bool, np.bool_)


for dtype in wp.types.int_types:
    add_function_test(TestTypes, f"test_integers_{dtype.__name__}", test_integers, devices=devices, dtype=dtype)

for dtype in wp.types.float_types:
    add_function_test(TestTypes, f"test_floats_{dtype.__name__}", test_floats, devices=devices, dtype=dtype)

for dtype in (*wp.types.scalar_types, int, float):
    add_function_test(TestTypes, f"test_vector_{dtype.__name__}", test_vector, devices=devices, dtype=dtype)

if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
