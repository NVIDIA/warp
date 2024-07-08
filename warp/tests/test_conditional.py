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
def test_conditional_if_else():
    a = 0.5
    b = 2.0

    if a > b:
        c = 1.0
    else:
        c = -1.0

    wp.expect_eq(c, -1.0)


@wp.kernel
def test_conditional_if_else_nested():
    a = 1.0
    b = 2.0

    if a > b:
        c = 3.0
        d = 4.0

        if c > d:
            e = 1.0
        else:
            e = -1.0

    else:
        c = 6.0
        d = 7.0

        if c > d:
            e = 2.0
        else:
            e = -2.0

    wp.expect_eq(e, -2.0)


@wp.kernel
def test_boolean_and():
    a = 1.0
    b = 2.0
    c = 1.0

    if a > 0.0 and b > 0.0:
        c = -1.0

    wp.expect_eq(c, -1.0)


@wp.kernel
def test_boolean_or():
    a = 1.0
    b = 2.0
    c = 1.0

    if a > 0.0 and b > 0.0:
        c = -1.0

    wp.expect_eq(c, -1.0)


@wp.kernel
def test_boolean_compound():
    a = 1.0
    b = 2.0
    c = 3.0

    d = 1.0

    if a > 0.0 and b > 0.0 or c > a:
        d = -1.0

    wp.expect_eq(d, -1.0)


@wp.kernel
def test_boolean_literal():
    t = True
    f = False

    r = 1.0

    if t == (not f):
        r = -1.0

    wp.expect_eq(r, -1.0)


@wp.kernel
def test_int_logical_not():
    x = 0
    if not 123:
        x = 123

    wp.expect_eq(x, 0)


@wp.kernel
def test_int_conditional_assign_overload():
    if 123:
        x = 123

    if 234:
        x = 234

    wp.expect_eq(x, 234)


@wp.kernel
def test_bool_param_conditional(foo: bool):
    if foo:
        x = 123

    wp.expect_eq(x, 123)


@wp.kernel
def test_conditional_chain_basic():
    x = -1

    if 0 < x < 1:
        success = False
    else:
        success = True
    wp.expect_eq(success, True)


@wp.kernel
def test_conditional_chain_empty_range():
    x = -1
    y = 4

    if -2 <= x <= 10 <= y:
        success = False
    else:
        success = True
    wp.expect_eq(success, True)


@wp.kernel
def test_conditional_chain_faker():
    x = -1

    # Not actually a chained inequality
    if (-2 < x) < (1 > 0):
        success = False
    else:
        success = True
    wp.expect_eq(success, True)


@wp.kernel
def test_conditional_chain_and():
    x = -1

    if (-2 < x < 0) and (-1 <= x <= -1):
        success = True
    else:
        success = False
    wp.expect_eq(success, True)


@wp.kernel
def test_conditional_chain_eqs():
    x = wp.int32(10)
    y = 10
    z = -10

    if x == y != z:
        success = True
    else:
        success = False
    wp.expect_eq(success, True)


@wp.kernel
def test_conditional_chain_mixed():
    x = 0

    if x < 10 == 1:
        success = False
    else:
        success = True
    wp.expect_eq(success, True)


def test_conditional_unequal_types(test: unittest.TestCase, device):
    # The bad kernel must be in a separate module, otherwise the current module would fail to load
    from warp.tests.aux_test_conditional_unequal_types_kernels import (
        unequal_types_kernel,
    )

    with test.assertRaises(TypeError):
        wp.launch(unequal_types_kernel, dim=(1,), inputs=[], device=device)

    # remove all references to the bad module so that subsequent calls to wp.force_load()
    # won't try to load it unless we explicitly re-import it again
    del wp.context.user_modules["warp.tests.aux_test_conditional_unequal_types_kernels"]
    del sys.modules["warp.tests.aux_test_conditional_unequal_types_kernels"]


devices = get_test_devices()


class TestConditional(unittest.TestCase):
    pass


add_kernel_test(TestConditional, kernel=test_conditional_if_else, dim=1, devices=devices)
add_kernel_test(TestConditional, kernel=test_conditional_if_else_nested, dim=1, devices=devices)
add_kernel_test(TestConditional, kernel=test_boolean_and, dim=1, devices=devices)
add_kernel_test(TestConditional, kernel=test_boolean_or, dim=1, devices=devices)
add_kernel_test(TestConditional, kernel=test_boolean_compound, dim=1, devices=devices)
add_kernel_test(TestConditional, kernel=test_boolean_literal, dim=1, devices=devices)
add_kernel_test(TestConditional, kernel=test_int_logical_not, dim=1, devices=devices)
add_kernel_test(TestConditional, kernel=test_int_conditional_assign_overload, dim=1, devices=devices)
add_kernel_test(TestConditional, kernel=test_bool_param_conditional, dim=1, inputs=[True], devices=devices)
add_kernel_test(TestConditional, kernel=test_conditional_chain_basic, dim=1, devices=devices)
add_kernel_test(TestConditional, kernel=test_conditional_chain_empty_range, dim=1, devices=devices)
add_kernel_test(TestConditional, kernel=test_conditional_chain_faker, dim=1, devices=devices)
add_kernel_test(TestConditional, kernel=test_conditional_chain_and, dim=1, devices=devices)
add_kernel_test(TestConditional, kernel=test_conditional_chain_eqs, dim=1, devices=devices)
add_kernel_test(TestConditional, kernel=test_conditional_chain_mixed, dim=1, devices=devices)
add_function_test(TestConditional, "test_conditional_unequal_types", test_conditional_unequal_types, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
