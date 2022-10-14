# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import importlib
import os
import tempfile
import unittest

import warp as wp
from warp.tests.test_base import *

CODE = """# -*- coding: utf-8 -*-

import warp as wp

@wp.kernel
def increment(x: wp.array(dtype=int)):
    x[0] = x[0] + 1
"""

wp.init()

def test_transient_module(test, device):
    file, file_path = tempfile.mkstemp(suffix=".py", text=True)
    os.close(file)

    try:
        with open(file_path, "w") as f:
            f.write(CODE)

        spec = importlib.util.spec_from_file_location("", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        os.remove(file_path)

    x = wp.array(123, dtype=int)
    wp.launch(module.increment, dim=1, inputs=[x])
    assert_np_equal(x.numpy(), np.array([124]))

def register(parent):
    devices = wp.get_devices()

    class TestTransientModule(parent):
        pass

    add_function_test(TestTransientModule, "test_transient_module", test_transient_module, devices=devices)
    return TestTransientModule

if __name__ == "__main__":
    _ = register(unittest.TestCase)
    unittest.main(verbosity=2)
