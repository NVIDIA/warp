# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import unittest

import warp as wp
from warp.tests.test_base import *

wp.init()


def test_module_lite_load(test, device):
    # Load current module
    wp.load_module()

    # Load named module
    wp.load_module(wp.config)

    # Load named module (string)
    wp.load_module(wp.config, recursive=True)


def test_module_lite_options(test, device):
    wp.set_module_options({"max_unroll": 8})
    module_options = wp.get_module_options()
    test.assertIsInstance(module_options, dict)
    test.assertEqual(module_options["max_unroll"], 8)


def register(parent):
    devices = get_test_devices()

    class TestModuleLite(parent):
        pass

    add_function_test(TestModuleLite, "test_module_lite_load", test_module_lite_load, devices=devices)
    add_function_test(TestModuleLite, "test_module_lite_get_options", test_module_lite_options, devices=devices)

    return TestModuleLite


if __name__ == "__main__":
    _ = register(unittest.TestCase)
    unittest.main(verbosity=2)
