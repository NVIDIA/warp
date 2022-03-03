# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# include parent path
import os
import sys
import numpy as np
import math
import ctypes

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warp as wp

import unittest
import test_base

wp.init()


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

        if (c > d):
            e = 1.0
        else:
            e = -1.0

    else:

        c = 6.0
        d = 7.0

        if (c > d):
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



devices = wp.get_devices()

class TestConditional(test_base.TestBase):
    pass

TestConditional.add_kernel_test(kernel=test_conditional_if_else, dim=1, devices=devices)
TestConditional.add_kernel_test(kernel=test_conditional_if_else_nested, dim=1, devices=devices)
TestConditional.add_kernel_test(kernel=test_boolean_and, dim=1, devices=devices)
TestConditional.add_kernel_test(kernel=test_boolean_or, dim=1, devices=devices)
TestConditional.add_kernel_test(kernel=test_boolean_compound, dim=1, devices=devices)
TestConditional.add_kernel_test(kernel=test_boolean_literal, dim=1, devices=devices)
TestConditional.add_kernel_test(kernel=test_boolean_and, dim=1, devices=devices)

# wp.launch( 
#     kernel=test_conditional_if_else,
#     dim=1,
#     inputs=[],
#     device=device)

# wp.launch(
#     kernel=test_conditional_if_else_nested,
#     dim=1,
#     inputs=[],
#     device=device)

# wp.launch(
#     kernel=test_boolean_and,
#     dim=1,
#     inputs=[],
#     device=device)

# wp.launch(
#     kernel=test_boolean_or,
#     dim=1,
#     inputs=[],
#     device=device)

# wp.launch(
#     kernel=test_boolean_compound,
#     dim=1,
#     inputs=[],
#     device=device)    

# wp.launch(
#     kernel=test_boolean_literal,
#     dim=1,
#     inputs=[],
#     device=device)


# wp.synchronize()


# print("passed")


if __name__ == '__main__':
    unittest.main(verbosity=2)
