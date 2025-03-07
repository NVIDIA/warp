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


def get_device_pair_with_mempool_access_support():
    devices = wp.get_cuda_devices()
    for target_device in devices:
        for peer_device in devices:
            if target_device != peer_device:
                if wp.is_mempool_access_supported(target_device, peer_device):
                    return (target_device, peer_device)
    return None


def get_device_pair_without_mempool_access_support():
    devices = wp.get_cuda_devices()
    for target_device in devices:
        for peer_device in devices:
            if target_device != peer_device:
                if not wp.is_mempool_access_supported(target_device, peer_device):
                    return (target_device, peer_device)
    return None


def test_mempool_release_threshold(test, device):
    device = wp.get_device(device)

    assert device.is_mempool_supported

    test.assertEqual(wp.is_mempool_supported(device), device.is_mempool_supported)

    was_enabled = wp.is_mempool_enabled(device)

    # toggle
    wp.set_mempool_enabled(device, not was_enabled)
    test.assertEqual(wp.is_mempool_enabled(device), not was_enabled)

    # restore
    wp.set_mempool_enabled(device, was_enabled)
    test.assertEqual(wp.is_mempool_enabled(device), was_enabled)

    saved_threshold = wp.get_mempool_release_threshold(device)

    # set new absolute threshold
    wp.set_mempool_release_threshold(device, 42000)
    test.assertEqual(wp.get_mempool_release_threshold(device), 42000)

    # set new fractional threshold
    wp.set_mempool_release_threshold(device, 0.5)
    test.assertEqual(wp.get_mempool_release_threshold(device), int(0.5 * device.total_memory))

    # restore threshold
    wp.set_mempool_release_threshold(device, saved_threshold)
    test.assertEqual(wp.get_mempool_release_threshold(device), saved_threshold)


def test_mempool_usage_queries(test, device):
    """Check API to query mempool memory usage."""

    device = wp.get_device(device)
    pre_alloc_mempool_usage_curr = wp.get_mempool_used_mem_current(device)
    pre_alloc_mempool_usage_high = wp.get_mempool_used_mem_high(device)

    # Allocate a 1 MiB array
    test_data = wp.empty(262144, dtype=wp.float32, device=device)
    wp.synchronize_device(device)

    # Query memory usage again
    post_alloc_mempool_usage_curr = wp.get_mempool_used_mem_current(device)
    post_alloc_mempool_usage_high = wp.get_mempool_used_mem_high(device)

    test.assertEqual(
        post_alloc_mempool_usage_curr, pre_alloc_mempool_usage_curr + 1048576, "Memory usage did not increase by 1 MiB"
    )
    test.assertGreaterEqual(post_alloc_mempool_usage_high, 1048576, "High-water mark is not at least 1 MiB")

    # Free the allocation
    del test_data
    wp.synchronize_device(device)

    # Query memory usage
    post_free_mempool_usage_curr = wp.get_mempool_used_mem_current(device)
    post_free_mempool_usage_high = wp.get_mempool_used_mem_high(device)

    test.assertEqual(
        post_free_mempool_usage_curr,
        pre_alloc_mempool_usage_curr,
        "Test didn't end with the same amount of used memory as the test started with.",
    )
    test.assertEqual(
        post_free_mempool_usage_high, post_alloc_mempool_usage_high, "High-water mark should not change after free"
    )


def test_mempool_exceptions(test, device):
    device = wp.get_device(device)

    assert not device.is_mempool_supported

    if device.is_cuda:
        expected_error = RuntimeError
    else:
        expected_error = ValueError

    with test.assertRaises(expected_error):
        wp.get_mempool_release_threshold(device)

    with test.assertRaises(expected_error):
        wp.set_mempool_release_threshold(device, 42000)


def test_mempool_access_self(test, device):
    device = wp.get_device(device)

    assert device.is_mempool_supported

    # setting mempool access to self is a no-op
    wp.set_mempool_access_enabled(device, device, True)
    wp.set_mempool_access_enabled(device, device, False)

    # should always be enabled
    enabled = wp.is_mempool_access_enabled(device, device)
    test.assertTrue(enabled)


@unittest.skipUnless(get_device_pair_with_mempool_access_support(), "Requires devices with mempool access support")
def test_mempool_access(test, _):
    target_device, peer_device = get_device_pair_with_mempool_access_support()

    was_enabled = wp.is_mempool_access_enabled(target_device, peer_device)

    if was_enabled:
        # try disabling
        wp.set_mempool_access_enabled(target_device, peer_device, False)
        is_enabled = wp.is_mempool_access_enabled(target_device, peer_device)
        test.assertFalse(is_enabled)

        # try re-enabling
        wp.set_mempool_access_enabled(target_device, peer_device, True)
        is_enabled = wp.is_mempool_access_enabled(target_device, peer_device)
        test.assertTrue(is_enabled)
    else:
        # try enabling
        wp.set_mempool_access_enabled(target_device, peer_device, True)
        is_enabled = wp.is_mempool_access_enabled(target_device, peer_device)
        test.assertTrue(is_enabled)

        # try re-disabling
        wp.set_mempool_access_enabled(target_device, peer_device, False)
        is_enabled = wp.is_mempool_access_enabled(target_device, peer_device)
        test.assertFalse(is_enabled)


@unittest.skipUnless(
    get_device_pair_without_mempool_access_support(), "Requires devices without mempool access support"
)
def test_mempool_access_exceptions_unsupported(test, _):
    # get a CUDA device pair without mempool access support
    target_device, peer_device = get_device_pair_without_mempool_access_support()

    # querying is ok, but must return False
    test.assertFalse(wp.is_mempool_access_enabled(target_device, peer_device))

    # enabling should raise RuntimeError
    with test.assertRaises(RuntimeError):
        wp.set_mempool_access_enabled(target_device, peer_device, True)

    # disabling should not raise an error
    wp.set_mempool_access_enabled(target_device, peer_device, False)


@unittest.skipUnless(wp.is_cpu_available() and wp.is_cuda_available(), "Requires both CUDA and CPU devices")
def test_mempool_access_exceptions_cpu(test, _):
    # querying is ok, but must return False
    test.assertFalse(wp.is_mempool_access_enabled("cuda:0", "cpu"))
    test.assertFalse(wp.is_mempool_access_enabled("cpu", "cuda:0"))

    # enabling should raise ValueError
    with test.assertRaises(ValueError):
        wp.set_mempool_access_enabled("cpu", "cuda:0", True)
    with test.assertRaises(ValueError):
        wp.set_mempool_access_enabled("cuda:0", "cpu", True)

    # disabling should not raise an error
    wp.set_mempool_access_enabled("cpu", "cuda:0", False)
    wp.set_mempool_access_enabled("cuda:0", "cpu", False)


class TestMempool(unittest.TestCase):
    pass


devices_with_mempools = [d for d in get_test_devices() if d.is_mempool_supported]
devices_without_mempools = [d for d in get_test_devices() if not d.is_mempool_supported]

# test devices with mempool support
add_function_test(
    TestMempool, "test_mempool_release_threshold", test_mempool_release_threshold, devices=devices_with_mempools
)
add_function_test(TestMempool, "test_mempool_usage_queries", test_mempool_usage_queries, devices=devices_with_mempools)
add_function_test(TestMempool, "test_mempool_access_self", test_mempool_access_self, devices=devices_with_mempools)

# test devices without mempool support
add_function_test(TestMempool, "test_mempool_exceptions", test_mempool_exceptions, devices=devices_without_mempools)

# mempool access tests
add_function_test(TestMempool, "test_mempool_access", test_mempool_access)

# mempool access exceptions
add_function_test(TestMempool, "test_mempool_access_exceptions_unsupported", test_mempool_access_exceptions_unsupported)
add_function_test(TestMempool, "test_mempool_access_exceptions_cpu", test_mempool_access_exceptions_cpu)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
