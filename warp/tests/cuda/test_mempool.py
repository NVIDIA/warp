# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import unittest

import warp as wp
from warp.tests.unittest_utils import *


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
    gc.collect()
    wp.synchronize_device(device)

    pre_alloc_mempool_usage_curr = wp.get_mempool_used_mem_current(device)
    pre_alloc_mempool_usage_high = wp.get_mempool_used_mem_high(device)
    test.assertIsInstance(pre_alloc_mempool_usage_curr, int, "before allocation: current usage should be an int")
    test.assertIsInstance(pre_alloc_mempool_usage_high, int, "before allocation: high-water usage should be an int")
    test.assertGreaterEqual(pre_alloc_mempool_usage_curr, 0, "before allocation: current usage should not be negative")
    test.assertGreaterEqual(
        pre_alloc_mempool_usage_high,
        pre_alloc_mempool_usage_curr,
        "before allocation: high-water usage should cover current usage",
    )

    # Allocate a 1 MiB array
    test_data = wp.empty(262144, dtype=wp.float32, device=device)
    wp.synchronize_device(device)

    # Query memory usage again
    post_alloc_mempool_usage_curr = wp.get_mempool_used_mem_current(device)
    post_alloc_mempool_usage_high = wp.get_mempool_used_mem_high(device)
    test.assertIsInstance(post_alloc_mempool_usage_curr, int, "after allocation: current usage should be an int")
    test.assertIsInstance(post_alloc_mempool_usage_high, int, "after allocation: high-water usage should be an int")
    test.assertGreaterEqual(post_alloc_mempool_usage_curr, 0, "after allocation: current usage should not be negative")
    test.assertGreaterEqual(
        post_alloc_mempool_usage_high,
        post_alloc_mempool_usage_curr,
        "after allocation: high-water usage should cover current usage",
    )
    test.assertGreaterEqual(
        post_alloc_mempool_usage_curr,
        pre_alloc_mempool_usage_curr,
        "Current usage should not decrease while the test allocation is alive.",
    )
    test.assertGreaterEqual(
        post_alloc_mempool_usage_high,
        pre_alloc_mempool_usage_high,
        "High-water mark should not decrease after allocation.",
    )

    # Free the allocation
    del test_data
    gc.collect()
    wp.synchronize_device(device)

    # Query memory usage
    post_free_mempool_usage_curr = wp.get_mempool_used_mem_current(device)
    post_free_mempool_usage_high = wp.get_mempool_used_mem_high(device)
    test.assertIsInstance(post_free_mempool_usage_curr, int, "after free: current usage should be an int")
    test.assertIsInstance(post_free_mempool_usage_high, int, "after free: high-water usage should be an int")
    test.assertGreaterEqual(post_free_mempool_usage_curr, 0, "after free: current usage should not be negative")
    test.assertGreaterEqual(
        post_free_mempool_usage_high,
        post_free_mempool_usage_curr,
        "after free: high-water usage should cover current usage",
    )
    test.assertGreaterEqual(
        post_free_mempool_usage_high,
        post_alloc_mempool_usage_high,
        "High-water mark should not decrease after free.",
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


@unittest.skipUnless(get_cuda_device_pair_with_mempool_access_support(), "Requires devices with mempool access support")
def test_mempool_access(test, _):
    target_device, peer_device = get_cuda_device_pair_with_mempool_access_support()

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


@unittest.skipUnless(wp.is_cpu_available(), "Requires a CPU device")
def test_mempool_cpu_unsupported(test, _):
    """CPU does not expose a CUDA-style memory pool: support/enabled are ``False`` and the pool
    query/toggle APIs raise ``ValueError`` (the public mempool API is CUDA-only)."""
    device = wp.get_device("cpu")

    test.assertFalse(wp.is_mempool_supported(device))
    test.assertFalse(device.is_mempool_supported)
    test.assertFalse(wp.is_mempool_enabled(device))

    with test.assertRaises(ValueError):
        wp.set_mempool_enabled(device, True)
    with test.assertRaises(ValueError):
        wp.set_mempool_release_threshold(device, 42000)
    with test.assertRaises(ValueError):
        wp.get_mempool_release_threshold(device)
    with test.assertRaises(ValueError):
        wp.get_mempool_used_mem_current(device)
    with test.assertRaises(ValueError):
        wp.get_mempool_used_mem_high(device)


@unittest.skipUnless(wp.is_cpu_available(), "Requires a CPU device")
def test_graph_capture_allocation_capability(test, _):
    """The internal graph-capture allocation capability is the gate the capture/allocation
    paths use instead of the mempool flag. It is always ``True`` for CPU (host allocation +
    APIC region retention) and, for CUDA, mirrors the device's memory-pool support / enabled
    state."""
    from warp._src.context import (  # noqa: PLC0415
        _is_graph_capture_allocation_enabled,
        _is_graph_capture_allocation_supported,
    )

    cpu = wp.get_device("cpu")
    test.assertTrue(_is_graph_capture_allocation_supported(cpu))
    test.assertTrue(_is_graph_capture_allocation_enabled(cpu))

    for device in wp.get_cuda_devices():
        test.assertEqual(_is_graph_capture_allocation_supported(device), device.is_mempool_supported)
        test.assertEqual(_is_graph_capture_allocation_enabled(device), device.is_mempool_enabled)


class TestMempool(unittest.TestCase):
    pass


# CUDA-only mempool semantics (threshold/usage/self-access). CPU has no memory pool, so it is
# excluded here and covered by test_mempool_exceptions / test_mempool_cpu_unsupported instead.
cuda_devices_with_mempools = get_cuda_test_devices_with_mempool()
devices_without_mempools = [d for d in get_test_devices() if not d.is_mempool_supported]

# test devices with mempool support
add_function_test(
    TestMempool, "test_mempool_release_threshold", test_mempool_release_threshold, devices=cuda_devices_with_mempools
)
add_function_test(
    TestMempool, "test_mempool_usage_queries", test_mempool_usage_queries, devices=cuda_devices_with_mempools
)
add_function_test(TestMempool, "test_mempool_access_self", test_mempool_access_self, devices=cuda_devices_with_mempools)

# CPU has no CUDA-style mempool; it uses a separate graph-capture allocation capability.
add_function_test(TestMempool, "test_mempool_cpu_unsupported", test_mempool_cpu_unsupported)
add_function_test(TestMempool, "test_graph_capture_allocation_capability", test_graph_capture_allocation_capability)

# test devices without mempool support
add_function_test(TestMempool, "test_mempool_exceptions", test_mempool_exceptions, devices=devices_without_mempools)

# mempool access tests
add_function_test(TestMempool, "test_mempool_access", test_mempool_access)

# mempool access exceptions
add_function_test(TestMempool, "test_mempool_access_exceptions_unsupported", test_mempool_access_exceptions_unsupported)
add_function_test(TestMempool, "test_mempool_access_exceptions_cpu", test_mempool_access_exceptions_cpu)


if __name__ == "__main__":
    unittest.main(verbosity=2)
