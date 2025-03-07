# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from warp.tests.unittest_utils import *

devices = get_test_devices()


class TestModuleLite(unittest.TestCase):
    def test_module_lite_load(self):
        # Load current module
        wp.load_module()

        # Load named module
        wp.load_module(wp.config)

        # Load named module (string)
        wp.load_module(wp.config, recursive=True)

    def test_module_lite_options(self):
        wp.set_module_options({"max_unroll": 8})
        module_options = wp.get_module_options()
        self.assertIsInstance(module_options, dict)
        self.assertEqual(module_options["max_unroll"], 8)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
