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


def get_device_pair_with_peer_access_support():
    devices = wp.get_cuda_devices()
    for target_device in devices:
        for peer_device in devices:
            if target_device != peer_device:
                if wp.is_peer_access_supported(target_device, peer_device):
                    return (target_device, peer_device)
    return None


def get_device_pair_without_peer_access_support():
    devices = wp.get_cuda_devices()
    for target_device in devices:
        for peer_device in devices:
            if target_device != peer_device:
                if not wp.is_peer_access_supported(target_device, peer_device):
                    return (target_device, peer_device)
    return None


def test_peer_access_self(test, device):
    device = wp.get_device(device)

    assert device.is_cuda

    # device can access self
    can_access = wp.is_peer_access_supported(device, device)
    test.assertTrue(can_access)

    # setting peer access to self is a no-op
    wp.set_peer_access_enabled(device, device, True)
    wp.set_peer_access_enabled(device, device, False)

    # should always be enabled
    enabled = wp.is_peer_access_enabled(device, device)
    test.assertTrue(enabled)


@unittest.skipUnless(get_device_pair_with_peer_access_support(), "Requires devices with peer access support")
def test_peer_access(test, _):
    target_device, peer_device = get_device_pair_with_peer_access_support()

    was_enabled = wp.is_peer_access_enabled(target_device, peer_device)

    if was_enabled:
        # try disabling
        wp.set_peer_access_enabled(target_device, peer_device, False)
        is_enabled = wp.is_peer_access_enabled(target_device, peer_device)
        test.assertFalse(is_enabled)

        # try re-enabling
        wp.set_peer_access_enabled(target_device, peer_device, True)
        is_enabled = wp.is_peer_access_enabled(target_device, peer_device)
        test.assertTrue(is_enabled)
    else:
        # try enabling
        wp.set_peer_access_enabled(target_device, peer_device, True)
        is_enabled = wp.is_peer_access_enabled(target_device, peer_device)
        test.assertTrue(is_enabled)

        # try re-disabling
        wp.set_peer_access_enabled(target_device, peer_device, False)
        is_enabled = wp.is_peer_access_enabled(target_device, peer_device)
        test.assertFalse(is_enabled)


@unittest.skipUnless(get_device_pair_without_peer_access_support(), "Requires devices without peer access support")
def test_peer_access_exceptions_unsupported(test, _):
    # get a CUDA device pair without peer access support
    target_device, peer_device = get_device_pair_without_peer_access_support()

    # querying is ok, but must return False
    test.assertFalse(wp.is_peer_access_enabled(target_device, peer_device))

    # enabling should raise RuntimeError
    with test.assertRaises(RuntimeError):
        wp.set_peer_access_enabled(target_device, peer_device, True)

    # disabling should not raise an error
    wp.set_peer_access_enabled(target_device, peer_device, False)


@unittest.skipUnless(wp.is_cpu_available() and wp.is_cuda_available(), "Requires both CUDA and CPU devices")
def test_peer_access_exceptions_cpu(test, _):
    # querying is ok, but must return False
    test.assertFalse(wp.is_peer_access_enabled("cuda:0", "cpu"))
    test.assertFalse(wp.is_peer_access_enabled("cpu", "cuda:0"))

    # enabling should raise ValueError
    with test.assertRaises(ValueError):
        wp.set_peer_access_enabled("cpu", "cuda:0", True)
    with test.assertRaises(ValueError):
        wp.set_peer_access_enabled("cuda:0", "cpu", True)

    # disabling should not raise an error
    wp.set_peer_access_enabled("cpu", "cuda:0", False)
    wp.set_peer_access_enabled("cuda:0", "cpu", False)


class TestPeer(unittest.TestCase):
    pass


cuda_test_devices = get_cuda_test_devices()

add_function_test(TestPeer, "test_peer_access_self", test_peer_access_self, devices=cuda_test_devices)

# peer access tests
add_function_test(TestPeer, "test_peer_access", test_peer_access)

# peer access exceptions
add_function_test(TestPeer, "test_peer_access_exceptions_unsupported", test_peer_access_exceptions_unsupported)
add_function_test(TestPeer, "test_peer_access_exceptions_cpu", test_peer_access_exceptions_cpu)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
