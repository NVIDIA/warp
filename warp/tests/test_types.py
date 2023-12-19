# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import unittest

from warp.tests.unittest_utils import *

wp.init()


devices = get_test_devices()


class TestTypes(unittest.TestCase):
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
        with self.assertRaisesRegex(
            RuntimeError,
            r"Invalid constant type: <class 'tuple'>$",
        ):
            wp.constant((1, 2, 3))

    def test_vector(self):
        for dtype in tuple(wp.types.scalar_types) + (int, float):

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
            self.assertEqual(v[0], make_scalar(1))
            self.assertEqual(v.x, make_scalar(1))
            self.assertEqual(v.y, make_scalar(2))
            self.assertEqual(v.z, make_scalar(3))
            self.assertEqual(v.w, make_scalar(4))
            self.assertSequenceEqual(v[0:2], make_vec(1, 2))
            self.assertSequenceEqual(v, make_vec(1, 2, 3, 4))

            v[0] = -1
            self.assertEqual(v[0], make_scalar(-1))
            self.assertEqual(v.x, make_scalar(-1))
            self.assertEqual(v.y, make_scalar(2))
            self.assertEqual(v.z, make_scalar(3))
            self.assertEqual(v.w, make_scalar(4))
            self.assertSequenceEqual(v[0:2], make_vec(-1, 2))
            self.assertSequenceEqual(v, make_vec(-1, 2, 3, 4))

            v[1:3] = (-2, -3)
            self.assertEqual(v[0], make_scalar(-1))
            self.assertEqual(v.x, make_scalar(-1))
            self.assertEqual(v.y, make_scalar(-2))
            self.assertEqual(v.z, make_scalar(-3))
            self.assertEqual(v.w, make_scalar(4))
            self.assertSequenceEqual(v[0:2], make_vec(-1, -2))
            self.assertSequenceEqual(v, make_vec(-1, -2, -3, 4))

            v.x = 1
            self.assertEqual(v[0], make_scalar(1))
            self.assertEqual(v.x, make_scalar(1))
            self.assertEqual(v.y, make_scalar(-2))
            self.assertEqual(v.z, make_scalar(-3))
            self.assertEqual(v.w, make_scalar(4))
            self.assertSequenceEqual(v[0:2], make_vec(1, -2))
            self.assertSequenceEqual(v, make_vec(1, -2, -3, 4))

            v = vec3_cls(2, 4, 6)
            self.assertSequenceEqual(+v, make_vec(2, 4, 6))
            self.assertSequenceEqual(-v, make_vec(-2, -4, -6))
            self.assertSequenceEqual(v + vec3_cls(1, 1, 1), make_vec(3, 5, 7))
            self.assertSequenceEqual(v - vec3_cls(1, 1, 1), make_vec(1, 3, 5))
            self.assertSequenceEqual(v * dtype(2), make_vec(4, 8, 12))
            self.assertSequenceEqual(dtype(2) * v, make_vec(4, 8, 12))
            self.assertSequenceEqual(v / dtype(2), make_vec(1, 2, 3))
            self.assertSequenceEqual(dtype(12) / v, make_vec(6, 3, 2))

            self.assertTrue(v != vec3_cls(1, 2, 3))
            self.assertEqual(str(v), "[{}]".format(", ".join(str(x) for x in v)))

            # Check added purely for coverage reasons but is this really a desired
            # behaviour? Not allowing to define new attributes using systems like
            # `__slots__` could help improving memory usage.
            v.foo = 123
            self.assertEqual(v.foo, 123)

    def test_vector_error_invalid_arg_count(self):
        with self.assertRaisesRegex(
            ValueError,
            r"Invalid number of arguments in vector constructor, expected 3 elements, got 2$",
        ):
            wp.vec3(1, 2)

    def test_vector_error_invalid_ptr(self):
        with self.assertRaisesRegex(
            RuntimeError,
            r"NULL pointer exception",
        ):
            wp.vec3.from_ptr(0)

    def test_vector_error_invalid_get_item_key(self):
        v = wp.vec3(1, 2, 3)

        with self.assertRaisesRegex(
            KeyError,
            r"Invalid key None, expected int or slice",
        ):
            v[None]

    def test_vector_error_invalid_set_item_key(self):
        v = wp.vec3(1, 2, 3)
        with self.assertRaisesRegex(
            KeyError,
            r"Invalid key None, expected int or slice",
        ):
            v[None] = 0

    def test_matrix(self):
        for dtype in tuple(wp.types.float_types) + (float,):

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

            def make_mat(*args):
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
            ValueError,
            r"Invalid number of arguments in matrix constructor, expected 4 elements, got 3$",
        ):
            wp.mat22(1, 2, 3)

    def test_matrix_error_invalid_row_count(self):
        with self.assertRaisesRegex(
            TypeError,
            r"Invalid argument in matrix constructor, expected row of length 2, got \(1, 2, 3\)$",
        ):
            wp.mat22((1, 2, 3), (3, 4, 5))

    def test_matrix_error_invalid_ptr(self):
        with self.assertRaisesRegex(
            RuntimeError,
            r"NULL pointer exception",
        ):
            wp.mat22.from_ptr(0)

    def test_matrix_error_invalid_set_row_index(self):
        m = wp.mat22(1, 2, 3, 4)
        with self.assertRaisesRegex(
            IndexError,
            r"Invalid row index$",
        ):
            m.set_row(2, (0, 0))

    def test_matrix_error_invalid_get_item_key(self):
        m = wp.mat22(1, 2, 3, 4)

        with self.assertRaisesRegex(
            KeyError,
            r"Invalid key None, expected int or pair of ints",
        ):
            m[None]

    def test_matrix_error_invalid_get_item_key_length(self):
        m = wp.mat22(1, 2, 3, 4)

        with self.assertRaisesRegex(
            KeyError,
            r"Invalid key, expected one or two indices, got 3",
        ):
            m[0, 1, 2]

    def test_matrix_error_invalid_set_item_key(self):
        m = wp.mat22(1, 2, 3, 4)
        with self.assertRaisesRegex(
            KeyError,
            r"Invalid key None, expected int or pair of ints",
        ):
            m[None] = 0

    def test_matrix_error_invalid_set_item_key_length(self):
        m = wp.mat22(1, 2, 3, 4)

        with self.assertRaisesRegex(
            KeyError,
            r"Invalid key, expected one or two indices, got 3",
        ):
            m[0, 1, 2] = (0, 0)


if __name__ == "__main__":
    wp.build.clear_kernel_cache()
    unittest.main(verbosity=2)
