# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#   Array Initialization
# ------------------------------------------------------------------------------


def test_array_from_data(test, device):
    wp.array((1.0, 2.0, 3.0), dtype=float)


class TestImplicitInitArrayFromData(unittest.TestCase):
    pass


add_function_test(
    TestImplicitInitArrayFromData,
    "test_array_from_data",
    test_array_from_data,
    check_output=False,
)


def test_array_from_ptr(test, device):
    wp.array(ptr=0, shape=(123,), dtype=float)


class TestImplicitInitArrayFromPtr(unittest.TestCase):
    pass


add_function_test(
    TestImplicitInitArrayFromPtr,
    "test_array_from_ptr",
    test_array_from_ptr,
    check_output=False,
)


#   Builtin Call
# ------------------------------------------------------------------------------


def test_builtin_call(test, device):
    wp.sin(1.23)


class TestImplicitInitBuiltinCall(unittest.TestCase):
    pass


add_function_test(
    TestImplicitInitBuiltinCall,
    "test_builtin_call",
    test_builtin_call,
    check_output=False,
)


#   Devices
# ------------------------------------------------------------------------------


def test_get_cuda_device_count(test, device):
    wp.get_cuda_device_count()


class TestImplicitInitGetCudaDeviceCount(unittest.TestCase):
    pass


add_function_test(
    TestImplicitInitGetCudaDeviceCount,
    "test_get_cuda_device_count",
    test_get_cuda_device_count,
    check_output=False,
)


def test_get_cuda_devices(test, device):
    wp.get_cuda_devices()


class TestImplicitInitGetCudaDevices(unittest.TestCase):
    pass


add_function_test(
    TestImplicitInitGetCudaDevices,
    "test_get_cuda_devices",
    test_get_cuda_devices,
    check_output=False,
)


def test_get_device(test, device):
    wp.get_device("cpu")


class TestImplicitInitGetDevice(unittest.TestCase):
    pass


add_function_test(
    TestImplicitInitGetDevice,
    "test_get_device",
    test_get_device,
    check_output=False,
)


def test_get_devices(test, device):
    wp.get_devices()


class TestImplicitInitGetDevices(unittest.TestCase):
    pass


add_function_test(
    TestImplicitInitGetDevices,
    "test_get_devices",
    test_get_devices,
    check_output=False,
)


def test_get_preferred_device(test, device):
    wp.get_preferred_device()


class TestImplicitInitGetPreferredDevice(unittest.TestCase):
    pass


add_function_test(
    TestImplicitInitGetPreferredDevice,
    "test_get_preferred_device",
    test_get_preferred_device,
    check_output=False,
)


def test_is_cpu_available(test, device):
    wp.is_cpu_available()


class TestImplicitInitIsCpuAvailable(unittest.TestCase):
    pass


add_function_test(
    TestImplicitInitIsCpuAvailable,
    "test_is_cpu_available",
    test_is_cpu_available,
    check_output=False,
)


def test_is_cuda_available(test, device):
    wp.is_cuda_available()


class TestImplicitInitIsCudaAvailable(unittest.TestCase):
    pass


add_function_test(
    TestImplicitInitIsCudaAvailable,
    "test_is_cuda_available",
    test_is_cuda_available,
    check_output=False,
)


def test_is_device_available(test, device):
    wp.is_device_available("cpu")


class TestImplicitInitIsDeviceAvailable(unittest.TestCase):
    pass


add_function_test(
    TestImplicitInitIsDeviceAvailable,
    "test_is_device_available",
    test_is_device_available,
    check_output=False,
)


def test_set_device(test, device):
    wp.set_device("cpu")


class TestImplicitInitSetDevice(unittest.TestCase):
    pass


add_function_test(
    TestImplicitInitSetDevice,
    "test_set_device",
    test_set_device,
    check_output=False,
)


#   Launch
# ------------------------------------------------------------------------------


@wp.kernel
def launch_kernel():
    pass


def test_launch(test, device):
    wp.launch(launch_kernel, dim=1)


class TestImplicitInitLaunch(unittest.TestCase):
    pass


add_function_test(
    TestImplicitInitLaunch,
    "test_launch",
    test_launch,
    check_output=False,
)


#   Mempool
# ------------------------------------------------------------------------------


def test_is_mempool_enabled(test, device):
    wp.is_mempool_enabled("cpu")


class TestImplicitInitIsMempoolEnabled(unittest.TestCase):
    pass


add_function_test(
    TestImplicitInitIsMempoolEnabled,
    "test_is_mempool_enabled",
    test_is_mempool_enabled,
    check_output=False,
)


def test_is_mempool_supported(test, device):
    wp.is_mempool_supported("cpu")


class TestImplicitInitIsMempoolSupported(unittest.TestCase):
    pass


add_function_test(
    TestImplicitInitIsMempoolSupported,
    "test_is_mempool_supported",
    test_is_mempool_supported,
    check_output=False,
)


#   Mempool Access
# ------------------------------------------------------------------------------


def test_is_mempool_access_enabled(test, device):
    wp.is_mempool_access_enabled("cpu", "cpu")


class TestImplicitInitIsMempoolAccessEnabled(unittest.TestCase):
    pass


add_function_test(
    TestImplicitInitIsMempoolAccessEnabled,
    "test_is_mempool_access_enabled",
    test_is_mempool_access_enabled,
    check_output=False,
)


def test_is_mempool_access_supported(test, device):
    wp.is_mempool_access_supported("cpu", "cpu")


class TestImplicitInitIsMempoolAccessSupported(unittest.TestCase):
    pass


add_function_test(
    TestImplicitInitIsMempoolAccessSupported,
    "test_is_mempool_access_supported",
    test_is_mempool_access_supported,
    check_output=False,
)


#   Peer Access
# ------------------------------------------------------------------------------


def test_is_peer_access_enabled(test, device):
    wp.is_peer_access_enabled("cpu", "cpu")


class TestImplicitInitIsPeerAccessEnabled(unittest.TestCase):
    pass


add_function_test(
    TestImplicitInitIsPeerAccessEnabled,
    "test_is_peer_access_enabled",
    test_is_peer_access_enabled,
    check_output=False,
)


def test_is_peer_access_supported(test, device):
    wp.is_peer_access_supported("cpu", "cpu")


class TestImplicitInitIsPeerAccessSupported(unittest.TestCase):
    pass


add_function_test(
    TestImplicitInitIsPeerAccessSupported,
    "test_is_peer_access_supported",
    test_is_peer_access_supported,
    check_output=False,
)


#   Structs
# ------------------------------------------------------------------------------


def test_struct_member_init(test, device):
    @wp.struct
    class S:
        # fp16 requires conversion functions from warp.so
        x: wp.float16
        v: wp.vec3h

    s = S()
    s.x = 42.0
    s.v = wp.vec3h(1.0, 2.0, 3.0)


class TestImplicitInitStructMemberInit(unittest.TestCase):
    pass


add_function_test(
    TestImplicitInitStructMemberInit,
    "test_struct_member_init",
    test_struct_member_init,
    check_output=False,
)


#   Tape
# ------------------------------------------------------------------------------


def test_tape(test, device):
    with wp.Tape():
        pass


class TestImplicitInitTape(unittest.TestCase):
    pass


add_function_test(
    TestImplicitInitTape,
    "test_tape",
    test_tape,
    check_output=False,
)


if __name__ == "__main__":
    # Do not clear the kernel cache or call anything that would initialize Warp
    # since these tests are specifically aiming to catch issues where Warp isn't
    # correctly initialized upon calling certain public APIs.
    unittest.main(verbosity=2, failfast=True)
