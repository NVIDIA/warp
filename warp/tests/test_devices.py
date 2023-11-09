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


def test_devices_get_device_functions(test, device):
    # save default device
    saved_device = wp.get_device()

    test.assertTrue(saved_device.is_cuda)
    test.assertTrue(wp.is_device_available(device))

    device_ordinal = device.ordinal
    current_device = wp.get_cuda_device(device_ordinal)
    test.assertEqual(current_device, device)

    current_device = wp.get_cuda_device()  # No-ordinal version
    test.assertTrue(wp.is_device_available(current_device))

    preferred_device = wp.get_preferred_device()
    test.assertTrue(wp.is_device_available(preferred_device))

    # restore default device
    wp.set_device(saved_device)


def test_devices_map_device(test, device):
    with wp.ScopedDevice(device):
        saved_alias = device.alias
        # Map alias twice to check code path
        wp.map_cuda_device("new_alias")
        wp.map_cuda_device("new_alias")
        wp.context.runtime.rename_device(device, saved_alias)


def test_devices_unmap_imaginary_device(test, device):
    with test.assertRaises(RuntimeError):
        wp.unmap_cuda_device("imaginary_device:0")


def test_devices_verify_cuda_device(test, device):
    verify_cuda_saved = wp.config.verify_cuda

    wp.config.verify_cuda = True

    wp.context.runtime.verify_cuda_device(device)

    wp.config.verify_cuda = verify_cuda_saved


def test_devices_can_access_self(test, device):
    test.assertTrue(device.can_access(device))

    # Also test CPU access
    cpu_device = wp.get_device("cpu")
    if device != cpu_device:
        test.assertFalse(device.can_access(cpu_device))

    test.assertNotEqual(cpu_device, "cuda")


def register(parent):
    devices = get_test_devices()

    class TestDevices(parent):
        pass

    add_function_test(
        TestDevices,
        "test_devices_get_device_functions",
        test_devices_get_device_functions,
        devices=wp.get_cuda_devices(),
    )
    add_function_test(TestDevices, "test_devices_map_device", test_devices_map_device, devices=wp.get_cuda_devices())
    add_function_test(
        TestDevices, "test_devices_unmap_imaginary_device", test_devices_unmap_imaginary_device, devices=devices
    )
    add_function_test(TestDevices, "test_devices_verify_cuda_device", test_devices_verify_cuda_device, devices=devices)

    if wp.is_cuda_available():
        add_function_test(TestDevices, "test_devices_can_access_self", test_devices_can_access_self, devices=devices)

    return TestDevices


if __name__ == "__main__":
    wp.build.clear_kernel_cache()
    _ = register(unittest.TestCase)
    unittest.main(verbosity=2)
