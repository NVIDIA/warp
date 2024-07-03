# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

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


class TestModuleHashing(unittest.TestCase):
    pass


add_function_test(TestModuleHashing, "test_function_overload_hashing", test_function_overload_hashing)
add_function_test(TestModuleHashing, "test_function_generic_overload_hashing", test_function_generic_overload_hashing)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
