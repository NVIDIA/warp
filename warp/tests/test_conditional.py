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

import warp as wp
from warp.tests.test_base import *

import unittest

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


def register(parent):

    devices = wp.get_devices()

    class TestConditional(parent):
        pass

    add_kernel_test(TestConditional, kernel=test_conditional_if_else, dim=1, devices=devices)
    add_kernel_test(TestConditional, kernel=test_conditional_if_else_nested, dim=1, devices=devices)
    add_kernel_test(TestConditional, kernel=test_boolean_and, dim=1, devices=devices)
    add_kernel_test(TestConditional, kernel=test_boolean_or, dim=1, devices=devices)
    add_kernel_test(TestConditional, kernel=test_boolean_compound, dim=1, devices=devices)
    add_kernel_test(TestConditional, kernel=test_boolean_literal, dim=1, devices=devices)
    add_kernel_test(TestConditional, kernel=test_boolean_and, dim=1, devices=devices)

    return TestConditional


if __name__ == '__main__':
    c = register(unittest.TestCase)
    unittest.main(verbosity=2)
