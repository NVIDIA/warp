# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

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
