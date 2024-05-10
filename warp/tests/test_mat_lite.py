# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import unittest

import warp as wp
from warp.tests.unittest_utils import *

mat32d = wp.mat(shape=(3, 2), dtype=wp.float64)


@wp.kernel
def test_matrix_constructor_value_func():
    a = wp.mat22()
    b = wp.matrix(a, shape=(2, 2))
    c = mat32d()
    d = mat32d(c, shape=(3, 2))
    e = mat32d(wp.float64(1.0), wp.float64(2.0), wp.float64(1.0), wp.float64(2.0), wp.float64(1.0), wp.float64(2.0))
    f = mat32d(
        wp.vec3d(wp.float64(1.0), wp.float64(2.0), wp.float64(3.0)),
        wp.vec3d(wp.float64(1.0), wp.float64(2.0), wp.float64(3.0)),
    )
    g = wp.matrix(1.0, shape=(3, 2))


# Test matrix constructors using explicit type (float16)
# note that these tests are specifically not using generics / closure
# args to create kernels dynamically (like the rest of this file)
# as those use different code paths to resolve arg types which
# has lead to regressions.
@wp.kernel
def test_constructors_explicit_precision():
    # construction for custom matrix types
    eye = wp.identity(dtype=wp.float16, n=2)
    zeros = wp.matrix(shape=(2, 2), dtype=wp.float16)
    custom = wp.matrix(wp.float16(0.0), wp.float16(1.0), wp.float16(2.0), wp.float16(3.0), shape=(2, 2))

    for i in range(2):
        for j in range(2):
            if i == j:
                wp.expect_eq(eye[i, j], wp.float16(1.0))
            else:
                wp.expect_eq(eye[i, j], wp.float16(0.0))

            wp.expect_eq(zeros[i, j], wp.float16(0.0))
            wp.expect_eq(custom[i, j], wp.float16(i) * wp.float16(2.0) + wp.float16(j))


# Same as above but with a default (float/int) type
# which tests some different code paths that
# need to ensure types are correctly canonicalized
# during codegen
@wp.kernel
def test_constructors_default_precision():
    # construction for default (float) matrix types
    eye = wp.identity(dtype=float, n=2)
    zeros = wp.matrix(shape=(2, 2), dtype=float)
    custom = wp.matrix(0.0, 1.0, 2.0, 3.0, shape=(2, 2))

    for i in range(2):
        for j in range(2):
            if i == j:
                wp.expect_eq(eye[i, j], 1.0)
            else:
                wp.expect_eq(eye[i, j], 0.0)

            wp.expect_eq(zeros[i, j], 0.0)
            wp.expect_eq(custom[i, j], float(i) * 2.0 + float(j))


@wp.kernel
def test_matrix_mutation(expected: wp.types.matrix(shape=(10, 3), dtype=float)):
    m = wp.matrix(shape=(10, 3), dtype=float)

    # test direct element indexing
    m[0, 0] = 1.0
    m[0, 1] = 2.0
    m[0, 2] = 3.0

    # The nested indexing (matrix->vector->scalar) below does not
    # currently modify m because m[0] returns row vector by
    # value rather than reference, this is different from NumPy
    # which always returns by ref. Not clear how we can support
    # this as well as auto-diff.

    # m[0][1] = 2.0
    # m[0][2] = 3.0

    # test setting rows
    for i in range(1, 10):
        m[i] = m[i - 1] + wp.vec3(1.0, 2.0, 3.0)

    wp.expect_eq(m, expected)


devices = get_test_devices()


class TestMatLite(unittest.TestCase):
    pass


add_kernel_test(TestMatLite, test_matrix_constructor_value_func, dim=1, devices=devices)
add_kernel_test(TestMatLite, test_constructors_explicit_precision, dim=1, devices=devices)
add_kernel_test(TestMatLite, test_constructors_default_precision, dim=1, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=True)
