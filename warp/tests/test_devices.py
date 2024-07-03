# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

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


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
