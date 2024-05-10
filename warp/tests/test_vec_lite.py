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
def test_vector_constructor_value_func():
    a = wp.vec2()
    b = wp.vector(a, dtype=wp.float16)
    c = wp.vector(a)
    d = wp.vector(a, length=2)
    e = wp.vector(1.0, length=3)


# Test matrix constructors using explicit type (float16)
# note that these tests are specifically not using generics / closure
# args to create kernels dynamically (like the rest of this file)
# as those use different code paths to resolve arg types which
# has lead to regressions.
@wp.kernel
def test_constructors_explicit_precision():
    # construction for custom vector types
    ones = wp.vector(wp.float16(1.0), length=2)
    zeros = wp.vector(length=2, dtype=wp.float16)
    custom = wp.vector(wp.float16(0.0), wp.float16(1.0))

    for i in range(2):
        wp.expect_eq(ones[i], wp.float16(1.0))
        wp.expect_eq(zeros[i], wp.float16(0.0))
        wp.expect_eq(custom[i], wp.float16(i))


# Same as above but with a default (float/int) type
# which tests some different code paths that
# need to ensure types are correctly canonicalized
# during codegen
@wp.kernel
def test_constructors_default_precision():
    # construction for custom vector types
    ones = wp.vector(1.0, length=2)
    zeros = wp.vector(length=2, dtype=float)
    custom = wp.vector(0.0, 1.0)

    for i in range(2):
        wp.expect_eq(ones[i], 1.0)
        wp.expect_eq(zeros[i], 0.0)
        wp.expect_eq(custom[i], float(i))


devices = get_test_devices()


class TestVecLite(unittest.TestCase):
    pass


add_kernel_test(TestVecLite, test_vector_constructor_value_func, dim=1, devices=devices)
add_kernel_test(TestVecLite, test_constructors_explicit_precision, dim=1, devices=devices)
add_kernel_test(TestVecLite, test_constructors_default_precision, dim=1, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=True)
