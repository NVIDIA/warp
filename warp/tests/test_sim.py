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

wp.init()


def test_A(test, device):
    pass


def register(parent):

    devices = wp.get_devices()

    class TestSim(parent):
        pass

    # USD import failures should not count as a test failure
    try:
        from pxr import Usd, UsdGeom
        have_usd = True
    except:
        have_usd = False

    if have_usd:
        add_function_test(TestSim, "test_mesh_query_point", test_A, devices=devices)

    return TestSim

if __name__ == '__main__':
    c = register(unittest.TestCase)

    unittest.main(verbosity=2)
