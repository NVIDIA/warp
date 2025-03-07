# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
