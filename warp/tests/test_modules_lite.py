# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import warp as wp
import warp.utils
from warp.tests.unittest_utils import *

devices = get_test_devices()


# This kernel is needed to ensure this test module is registered as a Warp module.
# wp.load_module() requires the module to contain at least one Warp kernel, function, or struct.
@wp.kernel
def print_values():
    i = wp.tid()
    wp.print(i)


class TestModuleLite(unittest.TestCase):
    def test_module_lite_load(self):
        # Load current module
        wp.load_module()

        # Load named module
        wp.load_module(warp.utils)

        # Load named module (string)
        wp.load_module("warp.utils", recursive=True)

    def test_module_lite_options(self):
        wp.set_module_options({"max_unroll": 8})
        module_options = wp.get_module_options()
        self.assertIsInstance(module_options, dict)
        self.assertEqual(module_options["max_unroll"], 8)

    def test_module_lite_load_nonexistent(self):
        # Test that loading a non-existent module raises RuntimeError
        with self.assertRaises(RuntimeError) as context:
            wp.load_module("nonexistent_module_that_does_not_exist")

        self.assertIn("does not contain any Warp kernels, functions, or structs", str(context.exception))
        self.assertIn("nonexistent_module_that_does_not_exist", str(context.exception))

    def test_module_lite_load_no_warp_content(self):
        # Test that loading a module without Warp content raises RuntimeError
        # Use a standard library module that definitely has no Warp kernels
        with self.assertRaises(RuntimeError) as context:
            wp.load_module(unittest)

        self.assertIn("does not contain any Warp kernels, functions, or structs", str(context.exception))
        self.assertIn("unittest", str(context.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
