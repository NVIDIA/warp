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

# TODO: add more tests for kernels and generics

import os
import tempfile
import unittest
from importlib import util

import warp as wp
from warp.tests.unittest_utils import *

FUNC_OVERLOAD_1 = """# -*- coding: utf-8 -*-
import warp as wp

@wp.func
def fn():
    wp.print(17)

@wp.func
def fn(value: int):
    wp.print(value)

@wp.kernel
def k():
    print(fn())
    print(fn(99))
"""

# should be same hash as FUNC_OVERLOAD_1
FUNC_OVERLOAD_2 = """# -*- coding: utf-8 -*-
import warp as wp

@wp.func
def fn():
    wp.print(17)

@wp.func
def fn(value: int):
    wp.print(value)

@wp.kernel
def k():
    print(fn())
    print(fn(99))
"""

# should be different hash than FUNC_OVERLOAD_1 (first overload is different)
FUNC_OVERLOAD_3 = """# -*- coding: utf-8 -*-
import warp as wp

@wp.func
def fn():
    wp.print(42)

@wp.func
def fn(value: int):
    wp.print(value)

@wp.kernel
def k():
    print(fn())
    print(fn(99))
"""

# should be different hash than FUNC_OVERLOAD_1 (second overload is different)
FUNC_OVERLOAD_4 = """# -*- coding: utf-8 -*-
import warp as wp

@wp.func
def fn():
    wp.print(17)

@wp.func
def fn(value: int):
    wp.print(value + 1)

@wp.kernel
def k():
    print(fn())
    print(fn(99))
"""

FUNC_GENERIC_1 = """# -*- coding: utf-8 -*-
import warp as wp

from typing import Any

@wp.func
def generic_fn(x: Any):
    return x * x

@wp.func
def generic_fn(x: Any, y: Any):
    return x * y

@wp.kernel
def k():
    print(generic_fn(17))
    print(generic_fn(17, 42))
"""

# should be same hash as FUNC_GENERIC_1
FUNC_GENERIC_2 = """# -*- coding: utf-8 -*-
import warp as wp

from typing import Any

@wp.func
def generic_fn(x: Any):
    return x * x

@wp.func
def generic_fn(x: Any, y: Any):
    return x * y

@wp.kernel
def k():
    print(generic_fn(17))
    print(generic_fn(17, 42))
"""

# should be different hash than FUNC_GENERIC_1 (first overload is different)
FUNC_GENERIC_3 = """# -*- coding: utf-8 -*-
import warp as wp

from typing import Any

@wp.func
def generic_fn(x: Any):
    return x + x

@wp.func
def generic_fn(x: Any, y: Any):
    return x * y

@wp.kernel
def k():
    print(generic_fn(17))
    print(generic_fn(17, 42))
"""

# should be different hash than FUNC_GENERIC_1 (second overload is different)
FUNC_GENERIC_4 = """# -*- coding: utf-8 -*-
import warp as wp

from typing import Any

@wp.func
def generic_fn(x: Any):
    return x * x

@wp.func
def generic_fn(x: Any, y: Any):
    return x + y

@wp.kernel
def k():
    print(generic_fn(17))
    print(generic_fn(17, 42))
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

    return wp.get_module(module.__name__)


def test_function_overload_hashing(test, device):
    m1 = load_code_as_module(FUNC_OVERLOAD_1, "func_overload_1")
    m2 = load_code_as_module(FUNC_OVERLOAD_2, "func_overload_2")
    m3 = load_code_as_module(FUNC_OVERLOAD_3, "func_overload_3")
    m4 = load_code_as_module(FUNC_OVERLOAD_4, "func_overload_4")

    hash1 = m1.hash_module()
    hash2 = m2.hash_module()
    hash3 = m3.hash_module()
    hash4 = m4.hash_module()

    test.assertEqual(hash2, hash1)
    test.assertNotEqual(hash3, hash1)
    test.assertNotEqual(hash4, hash1)


def test_function_generic_overload_hashing(test, device):
    m1 = load_code_as_module(FUNC_GENERIC_1, "func_generic_1")
    m2 = load_code_as_module(FUNC_GENERIC_2, "func_generic_2")
    m3 = load_code_as_module(FUNC_GENERIC_3, "func_generic_3")
    m4 = load_code_as_module(FUNC_GENERIC_4, "func_generic_4")

    hash1 = m1.hash_module()
    hash2 = m2.hash_module()
    hash3 = m3.hash_module()
    hash4 = m4.hash_module()

    test.assertEqual(hash2, hash1)
    test.assertNotEqual(hash3, hash1)
    test.assertNotEqual(hash4, hash1)


SIMPLE_MODULE = """# -*- coding: utf-8 -*-
import warp as wp

@wp.kernel
def k():
    pass
"""


def test_module_load(test, device):
    """Ensure that loading a module does not change its hash"""
    m = load_code_as_module(SIMPLE_MODULE, "simple_module")

    hash1 = m.hash_module()
    m.load(device)
    hash2 = m.hash_module()

    test.assertEqual(hash1, hash2)


class TestModuleHashing(unittest.TestCase):
    pass


devices = get_test_devices()

add_function_test(TestModuleHashing, "test_function_overload_hashing", test_function_overload_hashing)
add_function_test(TestModuleHashing, "test_function_generic_overload_hashing", test_function_generic_overload_hashing)
add_function_test(TestModuleHashing, "test_module_load", test_module_load, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
