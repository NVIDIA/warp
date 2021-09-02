# include parent path
import os
import sys
import numpy as np
import math
import ctypes

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import warp as wp

wp.init()
wp.config.verify_cuda = True

@wp.kernel
def test_rename(n: int):

    a = 0
    b = 1

    a = b
    a = 2

    wp.expect_eq(a, 2)
    wp.expect_eq(b, 1)

@wp.kernel
def test_inplace(n: int):

    a = 1.0
    
    a += 2.0
    
    wp.expect_eq(a, 3.0)

@wp.kernel
def test_constant(c: float):

    a = 0.0
    a = c + 1.0

    wp.expect_eq(a, 2.0)

@wp.kernel
def test_dynamic_for_rename(n: int):

    f0 = int(0.0)
    f1 = int(1.0)

    for i in range(0, n):
        
        f = f0 + f1
        
        f0 = f1
        f1 = f

    wp.expect_eq(f1, 89)

@wp.kernel
def test_dynamic_for_inplace(n: int):

    a = float(0.0)

    for i in range(0, n):
        a += 1.0

    wp.expect_eq(n, 10)


@wp.kernel
def test_reassign(n: int):

    f0 = 1.0
    f1 = f0

    f1 = f1 + 2.0

    wp.expect_eq(f1, 3.0)
    wp.expect_eq(f0, 1.0)

@wp.kernel
def test_dynamic_reassign(n: int):

    f0 = wp.vec3(0.0, 0.0, 0.0)
    f1 = f0

    for i in range(0, n):
        f1 = f1 - wp.vec3(2.0, 0.0, 0.0)

    wp.expect_eq(f1, wp.vec3(-4.0, 0.0, 0.0))
    wp.expect_eq(f0, wp.vec3(0.0, 0.0, 0.0))


device = "cpu"

print("test_inplace")
wp.launch(test_inplace, dim=1, inputs=[], device=device)

print("test_rename")
wp.launch(test_rename, dim=1, inputs=[], device=device)

print("test_constant")
wp.launch(test_constant, dim=1, inputs=[1.0], device=device)

print("test_dynamic_for_rename")
wp.launch(test_dynamic_for_rename, dim=1, inputs=[10], device=device)

print("test_dynamic_for_inplace")
wp.launch(test_dynamic_for_inplace, dim=1, inputs=[10], device=device)

print("test_reassign")
wp.launch(test_reassign, dim=1, inputs=[], device=device)

print("test_dynamic_reassign")
wp.launch(test_dynamic_reassign, dim=1, inputs=[2], device=device)


print("passed")

# print("transform_vec3")
# wp.launch(transform_vec3, dim=n, inputs=[dest, m, c], device=device)
# print(dest)



