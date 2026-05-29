# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import unittest

import warp as wp
import warp.tests.test_func as test_func
import warp.tests.unittest_utils as unittest_utils
from warp.tests.unittest_utils import *


@wp.kernel
def test_import_func():
    # test a cross-module function reference is resolved correctly
    x = test_func.sqr(2.0)
    y = test_func.cube(2.0)

    wp.expect_eq(x, 4.0)
    wp.expect_eq(y, 8.0)


devices = get_test_devices()


class TestImport(unittest.TestCase):
    def test_normalize_direct_test_sys_path(self):
        """Direct test execution should not shadow top-level packages."""
        tests_root = os.path.dirname(os.path.abspath(unittest_utils.__file__))
        repo_root = os.path.dirname(os.path.dirname(tests_root))
        original_sys_path = sys.path[:]
        try:
            sys.path[:] = [tests_root, *[p for p in sys.path if os.path.abspath(p or os.getcwd()) != tests_root]]
            unittest_utils._normalize_direct_test_sys_path()

            self.assertNotEqual(os.path.abspath(sys.path[0] or os.getcwd()), tests_root)
            self.assertIn(repo_root, sys.path)
            self.assertNotIn(tests_root, [os.path.abspath(p or os.getcwd()) for p in sys.path])
        finally:
            sys.path[:] = original_sys_path


add_kernel_test(TestImport, kernel=test_import_func, name="test_import_func", dim=1, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
