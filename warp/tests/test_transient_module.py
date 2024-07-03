# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import tempfile
import unittest
from importlib import util

import warp as wp
from warp.tests.unittest_utils import *

CODE = """# -*- coding: utf-8 -*-

import warp as wp

@wp.struct
class Data:
    x: wp.array(dtype=int)

@wp.func
def increment(x: int):
    # This shouldn't be picked up.
    return x + 123

@wp.func
def increment(x: int):
    return x + 1

@wp.kernel
def compute(data: Data):
    data.x[0] = increment(data.x[0])
"""


def load_code_as_module(code, name):
    file, file_path = tempfile.mkstemp(suffix=".py")

    try:
        with os.fdopen(file, "w") as f:
            f.write(code)

        spec = util.spec_from_file_location(name, file_path)
        module = util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        os.remove(file_path)

    return module


def test_transient_module(test, device):
    module = load_code_as_module(CODE, "")
    # Loading it a second time shouldn't be an issue.
    module = load_code_as_module(CODE, "")

    assert len(module.compute.module.structs) == 1
    assert len(module.compute.module.functions) == 1

    data = module.Data()
    data.x = wp.array([123], dtype=int, device=device)

    wp.set_module_options({"foo": "bar"}, module=module)
    assert wp.get_module_options(module=module).get("foo") == "bar"
    assert module.compute.module.options.get("foo") == "bar"

    wp.launch(module.compute, dim=1, inputs=[data], device=device)
    assert_np_equal(data.x.numpy(), np.array([124]))


devices = get_test_devices()


class TestTransientModule(unittest.TestCase):
    pass


add_function_test(TestTransientModule, "test_transient_module", test_transient_module, devices=devices)

if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
