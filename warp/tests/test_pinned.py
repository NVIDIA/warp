# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import warp as wp
from warp.tests.test_base import *

import unittest

wp.init()

def test_pinned(test, device):

    assert wp.get_device(device).is_cuda, "Test device must be a CUDA device"

    n = 1024 * 1024

    ones = np.ones(n, dtype=np.float32)

    # pageable host arrays for synchronous transfers
    a_pageable1 = wp.array(ones, dtype=float, device="cpu")
    a_pageable2 = wp.zeros_like(a_pageable1)

    assert a_pageable1.pinned == False
    assert a_pageable2.pinned == False

    # pinned host arrays for asynchronous transfers
    a_pinned1 = wp.array(ones, dtype=float, device="cpu", pinned=True)
    a_pinned2 = wp.zeros_like(a_pinned1)

    assert a_pinned1.pinned == True
    assert a_pinned2.pinned == True

    # device array
    a_device = wp.zeros(n, dtype=float, device=device)

    assert a_device.pinned == False

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
    assert pinned_timer.elapsed < pageable_timer.elapsed, "Pinned transfers did not take less CPU time"


def register(parent):

    cuda_devices = wp.get_cuda_devices()

    class TestPinned(parent):
        pass

    if cuda_devices:
        add_function_test(TestPinned, "test_pinned", test_pinned, devices=cuda_devices)

    return TestPinned


if __name__ == '__main__':
    c = register(unittest.TestCase)
    unittest.main(verbosity=2)
