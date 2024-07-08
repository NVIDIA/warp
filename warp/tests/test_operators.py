# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import unittest

import warp as wp
from warp.tests.unittest_utils import *


@wp.kernel
def test_operators_scalar_float():
    a = 1.0
    b = 2.0

    c = a * b
    d = a + b
    e = a / b
    f = a - b
    g = b**8.0
    h = 10.0 // 3.0

    expect_eq(c, 2.0)
    expect_eq(d, 3.0)
    expect_eq(e, 0.5)
    expect_eq(f, -1.0)
    expect_eq(g, 256.0)
    expect_eq(h, 3.0)


@wp.kernel
def test_operators_scalar_int():
    a = 1
    b = 2

    c = a * b
    d = a + b
    e = a / b
    f = a - b
    # g = b**8    # integer pow not implemented
    h = 10 // 3
    i = 10 % 3
    j = 2 << 3
    k = 16 >> 1

    expect_eq(c, 2)
    expect_eq(d, 3)
    expect_eq(e, 0)
    expect_eq(f, -1)
    # expect_eq(g, 256)
    expect_eq(h, 3)
    expect_eq(i, 1)
    expect_eq(j, 16)
    expect_eq(k, 8)

    f0 = wp.uint32(1 << 0)
    f1 = wp.uint32(1 << 3)
    expect_eq(f0 | f1, f0 + f1)
    expect_eq(f0 & f1, wp.uint32(0))

    l = wp.uint8(0)
    for n in range(8):
        l |= wp.uint8(1 << n)
    expect_eq(l, ~wp.uint8(0))


@wp.kernel
def test_operators_vector_index():
    v = wp.vec4(1.0, 2.0, 3.0, 4.0)

    expect_eq(v[0], 1.0)
    expect_eq(v[1], 2.0)
    expect_eq(v[2], 3.0)
    expect_eq(v[3], 4.0)


@wp.kernel
def test_operators_matrix_index():
    m22 = wp.mat22(1.0, 2.0, 3.0, 4.0)

    expect_eq(m22[0, 0], 1.0)
    expect_eq(m22[0, 1], 2.0)
    expect_eq(m22[1, 0], 3.0)
    expect_eq(m22[1, 1], 4.0)


@wp.kernel
def test_operators_vec3():
    v = vec3(1.0, 2.0, 3.0)

    r0 = v * 3.0
    r1 = 3.0 * v

    expect_eq(r0, vec3(3.0, 6.0, 9.0))
    expect_eq(r1, vec3(3.0, 6.0, 9.0))

    col0 = vec3(1.0, 0.0, 0.0)
    col1 = vec3(0.0, 2.0, 0.0)
    col2 = vec3(0.0, 0.0, 3.0)

    m = mat33(col0, col1, col2)

    expect_eq(m * vec3(1.0, 0.0, 0.0), col0)
    expect_eq(m * vec3(0.0, 1.0, 0.0), col1)
    expect_eq(m * vec3(0.0, 0.0, 1.0), col2)

    two = vec3(1.0) * 2.0
    expect_eq(two, vec3(2.0, 2.0, 2.0))


@wp.kernel
def test_operators_vec4():
    v = vec4(1.0, 2.0, 3.0, 4.0)

    r0 = v * 3.0
    r1 = 3.0 * v

    expect_eq(r0, vec4(3.0, 6.0, 9.0, 12.0))
    expect_eq(r1, vec4(3.0, 6.0, 9.0, 12.0))

    col0 = vec4(1.0, 0.0, 0.0, 0.0)
    col1 = vec4(0.0, 2.0, 0.0, 0.0)
    col2 = vec4(0.0, 0.0, 3.0, 0.0)
    col3 = vec4(0.0, 0.0, 0.0, 4.0)

    m = mat44(col0, col1, col2, col3)

    expect_eq(m * vec4(1.0, 0.0, 0.0, 0.0), col0)
    expect_eq(m * vec4(0.0, 1.0, 0.0, 0.0), col1)
    expect_eq(m * vec4(0.0, 0.0, 1.0, 0.0), col2)
    expect_eq(m * vec4(0.0, 0.0, 0.0, 1.0), col3)

    two = vec4(1.0) * 2.0
    expect_eq(two, vec4(2.0, 2.0, 2.0, 2.0))


@wp.kernel
def test_operators_mat22():
    m = mat22(1.0, 2.0, 3.0, 4.0)
    r = mat22(3.0, 6.0, 9.0, 12.0)

    r0 = m * 3.0
    r1 = 3.0 * m

    expect_eq(r0, r)
    expect_eq(r1, r)

    expect_eq(r0[0, 0], 3.0)
    expect_eq(r0[0, 1], 6.0)
    expect_eq(r0[1, 0], 9.0)
    expect_eq(r0[1, 1], 12.0)

    expect_eq(r0[0], wp.vec2(3.0, 6.0))
    expect_eq(r0[1], wp.vec2(9.0, 12.0))


@wp.kernel
def test_operators_mat33():
    m = mat33(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)

    r = mat33(3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0)

    r0 = m * 3.0
    r1 = 3.0 * m

    expect_eq(r0, r)
    expect_eq(r1, r)

    expect_eq(r0[0, 0], 3.0)
    expect_eq(r0[0, 1], 6.0)
    expect_eq(r0[0, 2], 9.0)

    expect_eq(r0[1, 0], 12.0)
    expect_eq(r0[1, 1], 15.0)
    expect_eq(r0[1, 2], 18.0)

    expect_eq(r0[2, 0], 21.0)
    expect_eq(r0[2, 1], 24.0)
    expect_eq(r0[2, 2], 27.0)

    expect_eq(r0[0], wp.vec3(3.0, 6.0, 9.0))
    expect_eq(r0[1], wp.vec3(12.0, 15.0, 18.0))
    expect_eq(r0[2], wp.vec3(21.0, 24.0, 27.0))


@wp.kernel
def test_operators_mat44():
    m = mat44(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0)

    r = mat44(3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0, 33.0, 36.0, 39.0, 42.0, 45.0, 48.0)

    r0 = m * 3.0
    r1 = 3.0 * m

    expect_eq(r0, r)
    expect_eq(r1, r)

    expect_eq(r0[0, 0], 3.0)
    expect_eq(r0[0, 1], 6.0)
    expect_eq(r0[0, 2], 9.0)
    expect_eq(r0[0, 3], 12.0)

    expect_eq(r0[1, 0], 15.0)
    expect_eq(r0[1, 1], 18.0)
    expect_eq(r0[1, 2], 21.0)
    expect_eq(r0[1, 3], 24.0)

    expect_eq(r0[2, 0], 27.0)
    expect_eq(r0[2, 1], 30.0)
    expect_eq(r0[2, 2], 33.0)
    expect_eq(r0[2, 3], 36.0)

    expect_eq(r0[3, 0], 39.0)
    expect_eq(r0[3, 1], 42.0)
    expect_eq(r0[3, 2], 45.0)
    expect_eq(r0[3, 3], 48.0)

    expect_eq(r0[0], wp.vec4(3.0, 6.0, 9.0, 12.0))
    expect_eq(r0[1], wp.vec4(15.0, 18.0, 21.0, 24.0))
    expect_eq(r0[2], wp.vec4(27.0, 30.0, 33.0, 36.0))
    expect_eq(r0[3], wp.vec4(39.0, 42.0, 45.0, 48.0))


devices = get_test_devices()


class TestOperators(unittest.TestCase):
    pass


add_kernel_test(TestOperators, test_operators_scalar_float, dim=1, devices=devices)
add_kernel_test(TestOperators, test_operators_scalar_int, dim=1, devices=devices)
add_kernel_test(TestOperators, test_operators_matrix_index, dim=1, devices=devices)
add_kernel_test(TestOperators, test_operators_vector_index, dim=1, devices=devices)
add_kernel_test(TestOperators, test_operators_vec3, dim=1, devices=devices)
add_kernel_test(TestOperators, test_operators_vec4, dim=1, devices=devices)

add_kernel_test(TestOperators, test_operators_mat22, dim=1, devices=devices)
add_kernel_test(TestOperators, test_operators_mat33, dim=1, devices=devices)
add_kernel_test(TestOperators, test_operators_mat44, dim=1, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
