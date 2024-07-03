# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

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
