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


def setUpModule():
    wp.config.verify_fp = True  # Enable checking floating-point values to be finite


def tearDownModule():
    wp.config.verify_fp = False


@wp.struct
class TestStruct:
    field: wp.float32


@wp.kernel
def finite_kernel(foos: wp.array(dtype=TestStruct)):
    i = wp.tid()
    foos[i].field += wp.float32(1.0)


def test_finite(test, device):
    foos = wp.zeros((10,), dtype=TestStruct, device=device)

    wp.launch(
        kernel=finite_kernel,
        dim=(10,),
        inputs=[foos],
        device=device,
    )
    wp.synchronize()

    expected = TestStruct()
    expected.field = 1.0
    for f in foos.list():
        if f.field != expected.field:
            raise AssertionError(f"Unexpected result, got: {f} expected: {expected}")


@wp.kernel
def nan_kernel(foos: wp.array(dtype=TestStruct)):
    i = wp.tid()
    foos[i].field /= wp.float32(0.0)  # Division by zero produces Not-a-Number (NaN)


def test_nan(test, device):
    foos = wp.zeros((10,), dtype=TestStruct, device=device)

    capture = StdOutCapture()
    capture.begin()

    wp.launch(
        kernel=nan_kernel,
        dim=(10,),
        inputs=[foos],
        device=device,
    )
    wp.synchronize()

    output = capture.end()

    # Check that the output contains warnings about "nan" being produced.
    # Older Windows C runtimes have a bug where stdout sometimes does not get properly flushed.
    if output != "" or sys.platform != "win32":
        test.assertRegex(output, r"nan")


devices = get_test_devices()


class TestVerifyFP(unittest.TestCase):
    pass


add_function_test(TestVerifyFP, "test_finite", test_finite, devices=devices)
add_function_test(TestVerifyFP, "test_nan", test_nan, devices=devices, check_output=False)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
