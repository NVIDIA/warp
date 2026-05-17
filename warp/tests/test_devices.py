# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import types
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
        wp._src.context.runtime.rename_device(device, saved_alias)


def test_devices_verify_cuda_device(test, device):
    verify_cuda_saved = wp.config.verify_cuda

    wp.config.verify_cuda = True

    wp._src.context.runtime.verify_cuda_device(device)

    wp.config.verify_cuda = verify_cuda_saved


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
def test_devices_can_access_self(test, device):
    test.assertTrue(device.can_access(device))

    for warp_device in wp.get_devices():
        device_str = str(warp_device)

        if device.is_cuda and warp_device.is_cpu:
            test.assertEqual(device.can_access(warp_device), device.is_cpu_memory_access_from_gpu_supported)
            test.assertNotEqual(device, warp_device)
            test.assertNotEqual(device, device_str)
        elif device.is_cpu and warp_device.is_cuda:
            test.assertFalse(device.can_access(warp_device))
            test.assertNotEqual(device, warp_device)
            test.assertNotEqual(device, device_str)
        elif device.is_cuda and warp_device.is_cuda and device != warp_device:
            if warp_device.is_mempool_enabled:
                test.assertEqual(device.can_access(warp_device), wp.is_mempool_access_enabled(warp_device, device))
            else:
                test.assertEqual(device.can_access(warp_device), wp.is_peer_access_enabled(warp_device, device))


def test_devices_sm_count(test, device):
    if device.is_cuda:
        test.assertTrue(device.sm_count > 0)
    else:
        test.assertEqual(device.sm_count, 0)


def test_devices_max_shared_memory_per_block(test, device):
    if device.is_cuda:
        test.assertTrue(device.max_shared_memory_per_block > 0)
    else:
        test.assertEqual(device.max_shared_memory_per_block, 0)


devices = get_test_devices()


class TestDevices(unittest.TestCase):
    def test_devices_unmap_imaginary_device(self):
        with self.assertRaises(RuntimeError):
            wp.unmap_cuda_device("imaginary_device:0")

    def test_devices_unmap_cuda_device_clears_access_caches(self):
        runtime = wp._src.context.runtime
        alias = "test_unmap_cuda_device:0"
        target_context = object()
        peer_context = object()
        unrelated_target_context = object()
        unrelated_peer_context = object()
        target_ordinal = 1_000_001
        peer_ordinal = 1_000_002
        unrelated_target_ordinal = 1_000_003
        unrelated_peer_ordinal = 1_000_004

        fake_device = types.SimpleNamespace(
            alias=alias,
            is_cuda=True,
            context=target_context,
            ordinal=target_ordinal,
        )
        stale_peer_keys = (
            (target_context, peer_context),
            (peer_context, target_context),
        )
        unrelated_peer_key = (unrelated_target_context, unrelated_peer_context)
        stale_mempool_keys = (
            (target_ordinal, peer_ordinal),
            (peer_ordinal, target_ordinal),
        )
        unrelated_mempool_key = (unrelated_target_ordinal, unrelated_peer_ordinal)

        runtime.device_map[alias] = fake_device
        runtime.context_map[target_context] = fake_device
        runtime.cuda_devices.append(fake_device)
        for key in stale_peer_keys:
            runtime.cuda_peer_access_enabled[key] = True
        runtime.cuda_peer_access_enabled[unrelated_peer_key] = True
        for key in stale_mempool_keys:
            runtime.cuda_mempool_access_enabled[key] = True
        runtime.cuda_mempool_access_enabled[unrelated_mempool_key] = True

        try:
            runtime.unmap_cuda_device(alias)

            for key in stale_peer_keys:
                self.assertNotIn(key, runtime.cuda_peer_access_enabled)
            self.assertIn(unrelated_peer_key, runtime.cuda_peer_access_enabled)

            for key in stale_mempool_keys:
                self.assertNotIn(key, runtime.cuda_mempool_access_enabled)
            self.assertIn(unrelated_mempool_key, runtime.cuda_mempool_access_enabled)
        finally:
            runtime.device_map.pop(alias, None)
            runtime.context_map.pop(target_context, None)
            if fake_device in runtime.cuda_devices:
                runtime.cuda_devices.remove(fake_device)
            for key in (*stale_peer_keys, unrelated_peer_key):
                runtime.cuda_peer_access_enabled.pop(key, None)
            for key in (*stale_mempool_keys, unrelated_mempool_key):
                runtime.cuda_mempool_access_enabled.pop(key, None)

    def test_devices_get_cuda_supported_archs(self):
        archs = wp.get_cuda_supported_archs()
        self.assertIsInstance(archs, list)

        if wp.is_cuda_available():
            # With CUDA devices present, NVRTC must report architectures
            self.assertTrue(len(archs) > 0, "No CUDA supported architectures found")

        # Validate the list contents (may be non-empty even without
        # CUDA devices when NVRTC is available without a driver)
        for arch in archs:
            self.assertIsInstance(arch, int, f"Architecture value {arch} should be an integer")
            self.assertGreaterEqual(arch, 50, f"Architecture {arch} should be >= 50 (e.g., sm_50)")
            self.assertLessEqual(arch, 150, f"Architecture {arch} seems unreasonably high")

        # Check the list is sorted with no duplicates
        self.assertEqual(archs, sorted(archs), "Architecture list should be sorted")
        self.assertEqual(len(archs), len(set(archs)), "Architecture list should not contain duplicates")


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
add_function_test(
    TestDevices,
    "test_devices_max_shared_memory_per_block",
    test_devices_max_shared_memory_per_block,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
