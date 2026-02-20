# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from unittest.mock import patch

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


@wp.kernel
def add_one(a: wp.array(dtype=float), b: wp.array(dtype=float)):
    i = wp.tid()
    b[i] = a[i] + 1.0


def test_suffix_resolution_none(test, device):
    wp.config.cuda_arch_suffix = None
    test.assertEqual(device._get_cuda_arch_suffix(device.arch), "")


def test_suffix_invalid_value(test, device):
    for invalid in ("b", "x", "af", "A", "F", ""):
        wp.config.cuda_arch_suffix = invalid
        with test.assertRaises(RuntimeError, msg=f"Should reject suffix {invalid!r}"):
            device._get_cuda_arch_suffix(device.arch)


def test_suffix_a_rejects_arch_below_90(test, device):
    wp.config.cuda_arch_suffix = "a"
    with patch.object(device, "arch", 89):
        with test.assertRaises(RuntimeError):
            device._get_cuda_arch_suffix(89)


def test_suffix_f_rejects_arch_below_100(test, device):
    wp.config.cuda_arch_suffix = "f"
    with patch.object(device, "arch", 90):
        with test.assertRaises(RuntimeError):
            device._get_cuda_arch_suffix(90)


def test_suffix_f_rejects_toolkit_below_12_9(test, device):
    wp.config.cuda_arch_suffix = "f"
    with patch.object(device, "arch", 100), patch.object(device.runtime, "toolkit_version", (12, 8)):
        with test.assertRaises(RuntimeError):
            device._get_cuda_arch_suffix(100)


def test_suffix_a_requires_exact_arch(test, device):
    wp.config.cuda_arch_suffix = "a"
    with patch.object(device, "arch", 90):
        # Exact match must succeed
        test.assertEqual(device._get_cuda_arch_suffix(90), "a")
        # Lower output arch must fail
        with test.assertRaises(RuntimeError):
            device._get_cuda_arch_suffix(80)
        # Higher output arch must fail
        with test.assertRaises(RuntimeError):
            device._get_cuda_arch_suffix(100)


def test_suffix_f_requires_same_family(test, device):
    wp.config.cuda_arch_suffix = "f"
    with patch.object(device, "arch", 100), patch.object(device.runtime, "toolkit_version", (12, 9)):
        # Same family must succeed
        test.assertEqual(device._get_cuda_arch_suffix(100), "f")
        test.assertEqual(device._get_cuda_arch_suffix(103), "f")
        # Different family must fail
        with test.assertRaises(RuntimeError):
            device._get_cuda_arch_suffix(90)


def test_output_name_includes_suffix(test, device):
    if device.arch < 90:
        test.skipTest("Device arch < 90")

    module = add_one.module

    wp.config.cuda_arch_suffix = None
    name_no_suffix = module._get_compile_output_name(device)

    wp.config.cuda_arch_suffix = "a"
    name_with_suffix = module._get_compile_output_name(device)

    test.assertNotEqual(name_no_suffix, name_with_suffix)
    test.assertIn("a", name_with_suffix.split(".sm")[1])


def test_compile_kernel_with_suffix_a(test, device):
    if device.arch < 90:
        test.skipTest("Device arch < 90")

    wp.config.cuda_arch_suffix = "a"

    n = 10
    a = wp.zeros(n, dtype=float, device=device)
    b = wp.zeros(n, dtype=float, device=device)
    wp.launch(add_one, dim=n, inputs=[a, b], device=device)

    np.testing.assert_allclose(b.numpy(), np.ones(n))


def test_compile_kernel_with_suffix_f(test, device):
    if device.arch < 100:
        test.skipTest("Device arch < 100")
    tk = device.runtime.toolkit_version
    if tk is not None and tk < (12, 9):
        test.skipTest("CUDA toolkit < 12.9")

    wp.config.cuda_arch_suffix = "f"

    n = 10
    a = wp.zeros(n, dtype=float, device=device)
    b = wp.zeros(n, dtype=float, device=device)
    wp.launch(add_one, dim=n, inputs=[a, b], device=device)

    np.testing.assert_allclose(b.numpy(), np.ones(n))


devices = get_selected_cuda_test_devices("basic")


class TestCudaArchSuffix(unittest.TestCase):
    def setUp(self):
        self.saved_suffix = wp.config.cuda_arch_suffix

    def tearDown(self):
        wp.config.cuda_arch_suffix = self.saved_suffix


add_function_test(TestCudaArchSuffix, "test_suffix_resolution_none", test_suffix_resolution_none, devices=devices)
add_function_test(TestCudaArchSuffix, "test_suffix_invalid_value", test_suffix_invalid_value, devices=devices)
add_function_test(
    TestCudaArchSuffix, "test_suffix_a_rejects_arch_below_90", test_suffix_a_rejects_arch_below_90, devices=devices
)
add_function_test(
    TestCudaArchSuffix, "test_suffix_f_rejects_arch_below_100", test_suffix_f_rejects_arch_below_100, devices=devices
)
add_function_test(
    TestCudaArchSuffix,
    "test_suffix_f_rejects_toolkit_below_12_9",
    test_suffix_f_rejects_toolkit_below_12_9,
    devices=devices,
)
add_function_test(
    TestCudaArchSuffix, "test_suffix_a_requires_exact_arch", test_suffix_a_requires_exact_arch, devices=devices
)
add_function_test(
    TestCudaArchSuffix, "test_suffix_f_requires_same_family", test_suffix_f_requires_same_family, devices=devices
)
add_function_test(
    TestCudaArchSuffix, "test_output_name_includes_suffix", test_output_name_includes_suffix, devices=devices
)
add_function_test(
    TestCudaArchSuffix, "test_compile_kernel_with_suffix_a", test_compile_kernel_with_suffix_a, devices=devices
)
add_function_test(
    TestCudaArchSuffix, "test_compile_kernel_with_suffix_f", test_compile_kernel_with_suffix_f, devices=devices
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
