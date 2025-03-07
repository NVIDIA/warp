# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


@wp.kernel
def test_pow(e: float, expected: float):
    tid = wp.tid()

    y = wp.pow(-2.0, e)

    # Since equality comparisons with NaN's are false, we have to do something manually
    if wp.isnan(expected):
        if not wp.isnan(y):
            print("Error, comparison failed")
            wp.printf("    Expected: %f\n", expected)
            wp.printf("    Actual: %f\n", y)
    else:
        wp.expect_eq(y, expected)


def test_fast_math_disabled(test, device):
    # on all systems pow() should handle negative base correctly with fast math off
    wp.set_module_options({"fast_math": False})
    wp.launch(test_pow, dim=1, inputs=[2.0, 4.0], device=device)


def test_fast_math_cuda(test, device):
    # on CUDA with --fast-math enabled taking the pow()
    # of a negative number will result in a NaN

    wp.set_module_options({"fast_math": True})
    try:
        wp.launch(test_pow, dim=1, inputs=[2.0, wp.NAN], device=device)
    finally:
        # Turn fast math back off
        wp.set_module_options({"fast_math": False})


class TestFastMath(unittest.TestCase):
    def test_fast_math_cpu(self):
        # on all systems pow() should handle negative base correctly
        wp.set_module_options({"fast_math": True})
        try:
            wp.launch(test_pow, dim=1, inputs=[2.0, 4.0], device="cpu")
        finally:
            wp.set_module_options({"fast_math": False})


devices = get_test_devices()

add_function_test(TestFastMath, "test_fast_math_cuda", test_fast_math_cuda, devices=get_cuda_test_devices())
add_function_test(TestFastMath, "test_fast_math_disabled", test_fast_math_disabled, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
