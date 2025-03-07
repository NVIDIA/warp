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

import numpy as np

import warp as wp
from warp.tests.unittest_utils import *


def test_pinned(test: unittest.TestCase, device):
    assert wp.get_device(device).is_cuda, "Test device must be a CUDA device"

    n = 1024 * 1024

    ones = np.ones(n, dtype=np.float32)

    # pageable host arrays for synchronous transfers
    a_pageable1 = wp.array(ones, dtype=float, device="cpu")
    a_pageable2 = wp.zeros_like(a_pageable1)

    test.assertFalse(a_pageable1.pinned)
    test.assertFalse(a_pageable2.pinned)

    # pinned host arrays for asynchronous transfers
    a_pinned1 = wp.array(ones, dtype=float, device="cpu", pinned=True)
    a_pinned2 = wp.zeros_like(a_pinned1)

    test.assertTrue(a_pinned1.pinned)
    test.assertTrue(a_pinned2.pinned)

    # device array
    a_device = wp.zeros(n, dtype=float, device=device)

    test.assertFalse(a_device.pinned)

    wp.synchronize_device(device)

    with wp.ScopedTimer("Synchronous copy", print=False) as pageable_timer:
        wp.copy(a_device, a_pageable1)
        wp.copy(a_pageable2, a_device)

    wp.synchronize_device(device)

    with wp.ScopedTimer("Asynchronous copy", print=False) as pinned_timer:
        wp.copy(a_device, a_pinned1)
        wp.copy(a_pinned2, a_device)

    wp.synchronize_device(device)

    # ensure correct results
    assert_np_equal(a_pageable2.numpy(), ones)
    assert_np_equal(a_pinned2.numpy(), ones)

    # ensure that launching asynchronous transfers took less CPU time
    test.assertTrue(pinned_timer.elapsed < pageable_timer.elapsed, "Pinned transfers did not take less CPU time")


devices = get_selected_cuda_test_devices()


class TestPinned(unittest.TestCase):
    pass


add_function_test(TestPinned, "test_pinned", test_pinned, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
