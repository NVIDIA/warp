# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import warp as wp
import warp.tests.test_func as test_func
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
    pass


add_kernel_test(TestImport, kernel=test_import_func, name="test_import_func", dim=1, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
