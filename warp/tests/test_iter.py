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


@wp.kernel
def reversed_kernel(
    start: wp.int32,
    end: wp.int32,
    step: wp.int32,
    out_count: wp.array(dtype=wp.int32),
    out_values: wp.array(dtype=wp.int32),
):
    count = wp.int32(0)
    for i in reversed(range(start, end, step)):
        out_values[count] = i
        count += 1

    out_count[0] = count


def test_reversed(test, device):
    count = wp.empty(1, dtype=wp.int32)
    values = wp.empty(32, dtype=wp.int32)

    start, end, step = (-2, 8, 3)
    wp.launch(
        reversed_kernel,
        dim=1,
        inputs=(start, end, step),
        outputs=(count, values),
    )
    expected = tuple(reversed(range(start, end, step)))
    assert count.numpy()[0] == len(expected)
    assert_np_equal(values.numpy()[: len(expected)], expected)

    start, end, step = (9, -3, -2)
    wp.launch(
        reversed_kernel,
        dim=1,
        inputs=(start, end, step),
        outputs=(count, values),
    )
    expected = tuple(reversed(range(start, end, step)))
    assert count.numpy()[0] == len(expected)
    assert_np_equal(values.numpy()[: len(expected)], expected)


devices = get_test_devices()


class TestIter(unittest.TestCase):
    pass


add_function_test(TestIter, "test_reversed", test_reversed, devices=devices)

if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
