# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

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


def test_vector_error_invalid_arg_count(test, device):
    with test.assertRaisesRegex(
        ValueError,
        r"Invalid number of arguments in vector constructor, expected 3 elements, got 2$",
    ):
        wp.vec3(1, 2)


def test_vector_error_invalid_ptr(test, device):
    with test.assertRaisesRegex(
        RuntimeError,
        r"NULL pointer exception",
    ):
        wp.vec3.from_ptr(0)


def test_vector_error_invalid_get_item_key(test, device):
    v = wp.vec3(1, 2, 3)

    with test.assertRaisesRegex(
        KeyError,
        r"Invalid key None, expected int or slice",
    ):
        v[None]


def test_vector_error_invalid_set_item_key(test, device):
    v = wp.vec3(1, 2, 3)
    with test.assertRaisesRegex(
        KeyError,
        r"Invalid key None, expected int or slice",
    ):
        v[None] = 0


def test_matrix(test, device, dtype):
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
    test.assertEqual(m[0][0], make_scalar(1))
    test.assertEqual(m[0][1], make_scalar(2))
    test.assertEqual(m[0][2], make_scalar(3))
    test.assertEqual(m[1][0], make_scalar(4))
    test.assertEqual(m[1][1], make_scalar(5))
    test.assertEqual(m[1][2], make_scalar(6))
    test.assertEqual(m[2][0], make_scalar(7))
    test.assertEqual(m[2][1], make_scalar(8))
    test.assertEqual(m[2][2], make_scalar(9))
    test.assertEqual(m[0, 0], make_scalar(1))
    test.assertEqual(m[0, 1], make_scalar(2))
    test.assertEqual(m[0, 2], make_scalar(3))
    test.assertEqual(m[1, 0], make_scalar(4))
    test.assertEqual(m[1, 1], make_scalar(5))
    test.assertEqual(m[1, 2], make_scalar(6))
    test.assertEqual(m[2, 0], make_scalar(7))
    test.assertEqual(m[2, 1], make_scalar(8))
    test.assertEqual(m[2, 2], make_scalar(9))
    test.assertSequenceEqual(m[0], make_vec(1, 2, 3))
    test.assertSequenceEqual(m[1], make_vec(4, 5, 6))
    test.assertSequenceEqual(m[2], make_vec(7, 8, 9))
    test.assertSequenceEqual(m[0][1:3], make_vec(2, 3))
    test.assertSequenceEqual(m[1][0:2], make_vec(4, 5))
    test.assertSequenceEqual(m[2][0:3], make_vec(7, 8, 9))
    # test.assertSequenceEqual(m[0, 1:3], make_vec(2, 3))
    # test.assertSequenceEqual(m[1, 0:2], make_vec(4, 5))
    # test.assertSequenceEqual(m[2, 0:3], make_vec(7, 8, 9))
    test.assertSequenceEqual(m, make_mat((1, 2, 3), (4, 5, 6), (7, 8, 9)))

    m[1, 0] = -4
    test.assertEqual(m[0][0], make_scalar(1))
    test.assertEqual(m[0][1], make_scalar(2))
    test.assertEqual(m[0][2], make_scalar(3))
    test.assertEqual(m[1][0], make_scalar(-4))
    test.assertEqual(m[1][1], make_scalar(5))
    test.assertEqual(m[1][2], make_scalar(6))
    test.assertEqual(m[2][0], make_scalar(7))
    test.assertEqual(m[2][1], make_scalar(8))
    test.assertEqual(m[2][2], make_scalar(9))
    test.assertEqual(m[0, 0], make_scalar(1))
    test.assertEqual(m[0, 1], make_scalar(2))
    test.assertEqual(m[0, 2], make_scalar(3))
    test.assertEqual(m[1, 0], make_scalar(-4))
    test.assertEqual(m[1, 1], make_scalar(5))
    test.assertEqual(m[1, 2], make_scalar(6))
    test.assertEqual(m[2, 0], make_scalar(7))
    test.assertEqual(m[2, 1], make_scalar(8))
    test.assertEqual(m[2, 2], make_scalar(9))
    test.assertSequenceEqual(m[0], make_vec(1, 2, 3))
    test.assertSequenceEqual(m[1], make_vec(-4, 5, 6))
    test.assertSequenceEqual(m[2], make_vec(7, 8, 9))
    test.assertSequenceEqual(m[0][1:3], make_vec(2, 3))
    test.assertSequenceEqual(m[1][0:2], make_vec(-4, 5))
    test.assertSequenceEqual(m[2][0:3], make_vec(7, 8, 9))
    # test.assertSequenceEqual(m[0, 1:3], make_vec(2, 3))
    # test.assertSequenceEqual(m[1, 0:2], make_vec(-4, 5))
    # test.assertSequenceEqual(m[2, 0:3], make_vec(7, 8, 9))
    test.assertSequenceEqual(m, make_mat((1, 2, 3), (-4, 5, 6), (7, 8, 9)))

    m[2] = (-7, 8, -9)
    test.assertEqual(m[0][0], make_scalar(1))
    test.assertEqual(m[0][1], make_scalar(2))
    test.assertEqual(m[0][2], make_scalar(3))
    test.assertEqual(m[1][0], make_scalar(-4))
    test.assertEqual(m[1][1], make_scalar(5))
    test.assertEqual(m[1][2], make_scalar(6))
    test.assertEqual(m[2][0], make_scalar(-7))
    test.assertEqual(m[2][1], make_scalar(8))
    test.assertEqual(m[2][2], make_scalar(-9))
    test.assertEqual(m[0, 0], make_scalar(1))
    test.assertEqual(m[0, 1], make_scalar(2))
    test.assertEqual(m[0, 2], make_scalar(3))
    test.assertEqual(m[1, 0], make_scalar(-4))
    test.assertEqual(m[1, 1], make_scalar(5))
    test.assertEqual(m[1, 2], make_scalar(6))
    test.assertEqual(m[2, 0], make_scalar(-7))
    test.assertEqual(m[2, 1], make_scalar(8))
    test.assertEqual(m[2, 2], make_scalar(-9))
    test.assertSequenceEqual(m[0], make_vec(1, 2, 3))
    test.assertSequenceEqual(m[1], make_vec(-4, 5, 6))
    test.assertSequenceEqual(m[2], make_vec(-7, 8, -9))
    test.assertSequenceEqual(m[0][1:3], make_vec(2, 3))
    test.assertSequenceEqual(m[1][0:2], make_vec(-4, 5))
    test.assertSequenceEqual(m[2][0:3], make_vec(-7, 8, -9))
    # test.assertSequenceEqual(m[0, 1:3], make_vec(2, 3))
    # test.assertSequenceEqual(m[1, 0:2], make_vec(-4, 5))
    # test.assertSequenceEqual(m[2, 0:3], make_vec(-7, 8, -9))
    test.assertSequenceEqual(m, make_mat((1, 2, 3), (-4, 5, 6), (-7, 8, -9)))

    m = mat22_cls(2, 4, 6, 8)
    test.assertSequenceEqual(+m, make_mat((2, 4), (6, 8)))
    test.assertSequenceEqual(-m, make_mat((-2, -4), (-6, -8)))
    test.assertSequenceEqual(m + mat22_cls(1, 1, 1, 1), make_mat((3, 5), (7, 9)))
    test.assertSequenceEqual(m - mat22_cls(1, 1, 1, 1), make_mat((1, 3), (5, 7)))
    test.assertSequenceEqual(m * dtype(2), make_mat((4, 8), (12, 16)))
    test.assertSequenceEqual(dtype(2) * m, make_mat((4, 8), (12, 16)))
    test.assertSequenceEqual(m / dtype(2), make_mat((1, 2), (3, 4)))
    test.assertSequenceEqual(dtype(24) / m, make_mat((12, 6), (4, 3)))

    test.assertSequenceEqual(m * vec2_cls(1, 2), make_vec(10, 22))
    test.assertSequenceEqual(m @ vec2_cls(1, 2), make_vec(10, 22))
    test.assertSequenceEqual(vec2_cls(1, 2) * m, make_vec(14, 20))
    test.assertSequenceEqual(vec2_cls(1, 2) @ m, make_vec(14, 20))

    test.assertTrue(m != mat22_cls(1, 2, 3, 4))
    test.assertEqual(
        str(m), "[{}]".format(",\n ".join("[{}]".format(", ".join(str(y) for y in m[x])) for x in range(m._shape_[0])))
    )

    # Check added purely for coverage reasons but is this really a desired
    # behaviour? Not allowing to define new attributes using systems like
    # `__slots__` could help improving memory usage.
    m.foo = 123
    test.assertEqual(m.foo, 123)


def test_matrix_error_invalid_arg_count(test, device):
    with test.assertRaisesRegex(
        ValueError,
        r"Invalid number of arguments in matrix constructor, expected 4 elements, got 3$",
    ):
        wp.mat22(1, 2, 3)


def test_matrix_error_invalid_row_count(test, device):
    with test.assertRaisesRegex(
        TypeError,
        r"Invalid argument in matrix constructor, expected row of length 2, got \(1, 2, 3\)$",
    ):
        wp.mat22((1, 2, 3), (3, 4, 5))


def test_matrix_error_invalid_ptr(test, device):
    with test.assertRaisesRegex(
        RuntimeError,
        r"NULL pointer exception",
    ):
        wp.mat22.from_ptr(0)


def test_matrix_error_invalid_set_row_index(test, device):
    m = wp.mat22(1, 2, 3, 4)
    with test.assertRaisesRegex(
        IndexError,
        r"Invalid row index$",
    ):
        m.set_row(2, (0, 0))


def test_matrix_error_invalid_get_item_key(test, device):
    m = wp.mat22(1, 2, 3, 4)

    with test.assertRaisesRegex(
        KeyError,
        r"Invalid key None, expected int or pair of ints",
    ):
        m[None]


def test_matrix_error_invalid_get_item_key_length(test, device):
    m = wp.mat22(1, 2, 3, 4)

    with test.assertRaisesRegex(
        KeyError,
        r"Invalid key, expected one or two indices, got 3",
    ):
        m[0, 1, 2]


def test_matrix_error_invalid_set_item_key(test, device):
    m = wp.mat22(1, 2, 3, 4)
    with test.assertRaisesRegex(
        KeyError,
        r"Invalid key None, expected int or pair of ints",
    ):
        m[None] = 0


def test_matrix_error_invalid_set_item_key_length(test, device):
    m = wp.mat22(1, 2, 3, 4)

    with test.assertRaisesRegex(
        KeyError,
        r"Invalid key, expected one or two indices, got 3",
    ):
        m[0, 1, 2] = (0, 0)


def register(parent):
    devices = get_test_devices()

    class TestTypes(parent):
        pass

    add_function_test(TestTypes, "test_constant", test_constant)
    add_function_test(TestTypes, "test_constant_error_invalid_type", test_constant_error_invalid_type)

    for dtype in tuple(wp.types.scalar_types) + (int, float):
        add_function_test(TestTypes, f"test_vector_{dtype.__name__}", test_vector, devices=None, dtype=dtype)

    add_function_test(TestTypes, "test_vector_error_invalid_arg_count", test_vector_error_invalid_arg_count)
    add_function_test(TestTypes, "test_vector_error_invalid_ptr", test_vector_error_invalid_ptr)
    add_function_test(TestTypes, "test_vector_error_invalid_get_item_key", test_vector_error_invalid_get_item_key)
    add_function_test(TestTypes, "test_vector_error_invalid_set_item_key", test_vector_error_invalid_set_item_key)

    for dtype in tuple(wp.types.float_types) + (float,):
        add_function_test(TestTypes, f"test_matrix_{dtype.__name__}", test_matrix, devices=None, dtype=dtype)

    add_function_test(TestTypes, "test_matrix_error_invalid_arg_count", test_matrix_error_invalid_arg_count)
    add_function_test(TestTypes, "test_matrix_error_invalid_row_count", test_matrix_error_invalid_row_count)
    add_function_test(TestTypes, "test_matrix_error_invalid_ptr", test_matrix_error_invalid_ptr)
    add_function_test(TestTypes, "test_matrix_error_invalid_set_row_index", test_matrix_error_invalid_set_row_index)
    add_function_test(TestTypes, "test_matrix_error_invalid_get_item_key", test_matrix_error_invalid_get_item_key)
    add_function_test(TestTypes, "test_matrix_error_invalid_get_item_key_length", test_matrix_error_invalid_get_item_key_length)
    add_function_test(TestTypes, "test_matrix_error_invalid_set_item_key", test_matrix_error_invalid_set_item_key)
    add_function_test(TestTypes, "test_matrix_error_invalid_set_item_key_length", test_matrix_error_invalid_set_item_key_length)

    return TestTypes


if __name__ == "__main__":
    wp.build.clear_kernel_cache()
    _ = register(unittest.TestCase)
    unittest.main(verbosity=2)
