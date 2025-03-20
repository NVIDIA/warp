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


def test_devices_get_cuda_device_functions(test, device):
    test.assertTrue(device.is_cuda)
    test.assertTrue(wp.is_device_available(device))

    device_ordinal = device.ordinal
    current_device = wp.get_cuda_device(device_ordinal)
    test.assertEqual(current_device, device)
    current_device = wp.get_cuda_device()  # No-ordinal version
    test.assertTrue(wp.is_device_available(current_device))

    if device == current_device:
        test.assertEqual(device, "cuda")
    else:
        test.assertNotEqual(device, "cuda")

    preferred_device = wp.get_preferred_device()
    test.assertTrue(wp.is_device_available(preferred_device))


def test_devices_map_cuda_device(test, device):
    with wp.ScopedDevice(device):
        saved_alias = device.alias
        # Map alias twice to check code path
        wp.map_cuda_device("new_alias")
        wp.map_cuda_device("new_alias")
        wp.context.runtime.rename_device(device, saved_alias)


def test_devices_verify_cuda_device(test, device):
    verify_cuda_saved = wp.config.verify_cuda

    wp.config.verify_cuda = True

    wp.context.runtime.verify_cuda_device(device)

    wp.config.verify_cuda = verify_cuda_saved


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
def test_devices_can_access_self(test, device):
    test.assertTrue(device.can_access(device))

    for warp_device in wp.get_devices():
        device_str = str(warp_device)

        if (device.is_cpu and warp_device.is_cuda) or (device.is_cuda and warp_device.is_cpu):
            test.assertFalse(device.can_access(warp_device))
            test.assertNotEqual(device, warp_device)
            test.assertNotEqual(device, device_str)


def test_devices_sm_count(test, device):
    if device.is_cuda:
        test.assertTrue(device.sm_count > 0)
    else:
        test.assertEqual(device.sm_count, 0)


devices = get_test_devices()


class TestDevices(unittest.TestCase):
    def test_devices_unmap_imaginary_device(self):
        with self.assertRaises(RuntimeError):
            wp.unmap_cuda_device("imaginary_device:0")


add_function_test(
    TestDevices,
    "test_devices_get_cuda_device_functions",
    test_devices_get_cuda_device_functions,
    devices=get_selected_cuda_test_devices(),
)
add_function_test(
    TestDevices, "test_devices_map_cuda_device", test_devices_map_cuda_device, devices=get_selected_cuda_test_devices()
)
add_function_test(TestDevices, "test_devices_verify_cuda_device", test_devices_verify_cuda_device, devices=devices)
add_function_test(TestDevices, "test_devices_can_access_self", test_devices_can_access_self, devices=devices)
add_function_test(TestDevices, "test_devices_sm_count", test_devices_sm_count, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
