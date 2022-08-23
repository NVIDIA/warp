# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import warp as wp

import math

import warp as wp
from warp.tests.test_base import *

import unittest
import importlib 
import os

# dummy module used for testing reload
import warp.tests.test_square as test_square
    

wp.init()

def test_redefine(test, device):
        

    #--------------------------------------------
    # first pass

    @wp.kernel
    def basic(x: wp.array(dtype=float)):
        
        tid = wp.tid()

        x[tid] = float(tid)*1.0

    n = 32

    x = wp.zeros(n, dtype=float, device=device)

    wp.launch(
        kernel=basic, 
        dim=n, 
        inputs=[x], 
        device=device)

    #--------------------------------------------
    # redefine kernel, should trigger a recompile

    @wp.kernel
    def basic(x: wp.array(dtype=float)):
        
        tid = wp.tid()

        x[tid] = float(tid)*2.0
        
    y = wp.zeros(n, dtype=float, device=device)

    wp.launch(
        kernel=basic, 
        dim=n, 
        inputs=[y], 
        device=device)


    assert_np_equal(np.arange(0, n, 1), x.numpy())
    assert_np_equal(np.arange(0, n, 1)*2.0, y.numpy())

    
square_two = """import warp as wp

wp.init()

@wp.func
def sqr(x: float):
    return x*x

@wp.kernel
def kern(expect: float):
    wp.expect_eq(sqr(2.0), expect)


def run(expect, device):
    wp.launch(kern, dim=1, inputs=[expect], device=device)
  
"""

square_four = """import warp as wp

wp.init()

@wp.func
def multiply(x: float):
    return x*x

@wp.kernel
def kern(expect: float):
    wp.expect_eq(multiply(4.0), expect)

def run(expect, device):
    wp.launch(kern, dim=1, inputs=[expect], device=device)
  
"""

def test_reload(test, device):

    # write out the module python and import it
    f = open(os.path.abspath(os.path.join(os.path.dirname(__file__), "test_square.py")), "w")
    f.writelines(square_two)
    f.flush()
    f.close()
    
    importlib.reload(test_square)
    test_square.run(expect=4.0, device=device)    # 2*2=4

    f = open(os.path.abspath(os.path.join(os.path.dirname(__file__), "test_square.py")), "w")
    f.writelines(square_four)
    f.flush()
    f.close()

    # reload module, this should trigger all of the funcs / kernels to be updated
    importlib.reload(test_square)
    test_square.run(expect=16.0, device=device)   # 4*4 = 16



def register(parent):

    devices = wp.get_devices()

    class TestReload(parent):
        pass
    
    add_function_test(TestReload, "test_redefine", test_redefine, devices=devices)
    add_function_test(TestReload, "test_reload", test_reload, devices=devices)
    
    return TestReload

if __name__ == '__main__':
    c = register(unittest.TestCase)
    unittest.main(verbosity=2, failfast=False)
