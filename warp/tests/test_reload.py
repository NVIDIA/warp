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


wp.init()

def test_reload(test, device):
        

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

    


def register(parent):

    devices = wp.get_devices()

    class TestReload(parent):
        pass
    
    add_function_test(TestReload, "test_reload", test_reload, devices=devices)
    
    return TestReload

if __name__ == '__main__':
    c = register(unittest.TestCase)
    unittest.main(verbosity=2, failfast=False)
