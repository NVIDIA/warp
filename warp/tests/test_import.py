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
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
