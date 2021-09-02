# include parent path
import os
import sys
import numpy as np
import math
import ctypes

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import warp as wp
np.random.seed(42)

wp.init()


@wp.kernel
def test_conditional_if_else(dt: float):

    a = 0.5
    b = 2.0

    if a > b:
        c = 1.0
    else:
        c = -1.0

    wp.expect_eq(c, -1.0)
    

@wp.kernel
def test_conditional_if_else_nested(dt: float):

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
def test_boolean_and(dt: float):

    a = 1.0
    b = 2.0
    c = 1.0

    if a > 0.0 and b > 0.0:
        c = -1.0

    wp.expect_eq(c, -1.0)

@wp.kernel
def test_boolean_or(dt: float):

    a = 1.0
    b = 2.0
    c = 1.0

    if a > 0.0 and b > 0.0:
        c = -1.0

    wp.expect_eq(c, -1.0)
    

@wp.kernel
def test_boolean_compound(dt: float):

    a = 1.0
    b = 2.0
    c = 3.0
    
    d = 1.0

    if a > 0.0 and b > 0.0 or c > a:
        d = -1.0

    wp.expect_eq(d, -1.0)

@wp.kernel
def test_boolean_literal(dt: float):

    t = True
    f = False
    
    r = 1.0

    if t == (not f):
        r = -1.0

    wp.expect_eq(r, -1.0)



device = "cpu"

wp.launch(
    kernel=test_conditional_if_else,
    dim=1,
    inputs=[],
    device=device)

wp.launch(
    kernel=test_conditional_if_else_nested,
    dim=1,
    inputs=[],
    device=device)

wp.launch(
    kernel=test_boolean_and,
    dim=1,
    inputs=[],
    device=device)

wp.launch(
    kernel=test_boolean_or,
    dim=1,
    inputs=[],
    device=device)

wp.launch(
    kernel=test_boolean_compound,
    dim=1,
    inputs=[],
    device=device)    

wp.launch(
    kernel=test_boolean_literal,
    dim=1,
    inputs=[],
    device=device)


wp.synchronize()




