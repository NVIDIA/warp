# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# include parent path
import numpy as np
import math

import warp as wp
from warp.tests.test_base import *

import unittest

wp.init()

#from test_func import sqr
import warp.tests.test_func as test_func

@wp.kernel
def test_import_func():

    # test a cross-module function reference is resolved correctly
    x = test_func.sqr(2.0)
    y = test_func.cube(2.0)

    wp.expect_eq(x, 4.0)
    wp.expect_eq(y, 8.0)



def register(parent):

    devices = wp.get_devices()

    class TestImport(parent):
        pass

    add_kernel_test(TestImport, kernel=test_import_func, name="test_import_func", dim=1, devices=devices)

    return TestImport

if __name__ == '__main__':
    c = register(unittest.TestCase)
    #unittest.main(verbosity=2)

    wp.force_load()
    
    loader = unittest.defaultTestLoader
    testSuite = loader.loadTestsFromTestCase(c)
    testSuite.debug()