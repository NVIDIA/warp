# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import warp as wp

import math

import warp as wp
from warp.tests.test_base import *

import unittest


wp.init()


@wp.kernel
def inc(a: wp.array(dtype=float)):

    tid = wp.tid()
    a[tid] = a[tid] + 1.0


@wp.kernel
def arange(start: int, step: int, a: wp.array(dtype=int)):

    tid = wp.tid()
    a[tid] = start + step * tid


def test_multigpu_set_device(test, device):

    assert len(wp.get_cuda_devices()) > 1, "At least two CUDA devices are required"

    n = 32

    wp.set_device("cuda:0")
    a0 = wp.empty(n, dtype=int)
    wp.launch(arange, dim=a0.size, inputs=[0, 1, a0])

    wp.set_device("cuda:1")
    a1 = wp.empty(n, dtype=int)
    wp.launch(arange, dim=a1.size, inputs=[0, 1, a1])

    wp.synchronize()

    assert a0.device == "cuda:0"
    assert a1.device == "cuda:1"

    expected = np.arange(n, dtype=int)

    assert_np_equal(a0.numpy(), expected)
    assert_np_equal(a1.numpy(), expected)


def test_multigpu_scoped_device(test, device):

    assert len(wp.get_cuda_devices()) > 1, "At least two CUDA devices are required"

    n = 32

    with wp.ScopedDevice("cuda:0"):
        a0 = wp.empty(n, dtype=int)
        wp.launch(arange, dim=a0.size, inputs=[0, 1, a0])

    with wp.ScopedDevice("cuda:1"):
        a1 = wp.empty(n, dtype=int)
        wp.launch(arange, dim=a1.size, inputs=[0, 1, a1])

    wp.synchronize()

    assert a0.device == "cuda:0"
    assert a1.device == "cuda:1"

    expected = np.arange(n, dtype=int)

    assert_np_equal(a0.numpy(), expected)
    assert_np_equal(a1.numpy(), expected)


def test_multigpu_nesting(test, device):
    
    assert len(wp.get_cuda_devices()) > 1, "At least two CUDA devices are required"

    initial_device = wp.get_device()
    initial_cuda_device = wp.get_cuda_device()

    with wp.ScopedDevice("cuda:1"):
        assert wp.get_device() == "cuda:1"
        assert wp.get_cuda_device() == "cuda:1"

        with wp.ScopedDevice("cuda:0"):
            assert wp.get_device() == "cuda:0"
            assert wp.get_cuda_device() == "cuda:0"

            with wp.ScopedDevice("cpu"):
                assert wp.get_device() == "cpu"
                assert wp.get_cuda_device() == "cuda:0"

                wp.set_device("cuda:1")

                assert wp.get_device() == "cuda:1"
                assert wp.get_cuda_device() == "cuda:1"

            assert wp.get_device() == "cuda:0"
            assert wp.get_cuda_device() == "cuda:0"

        assert wp.get_device() == "cuda:1"
        assert wp.get_cuda_device() == "cuda:1"

    assert wp.get_device() == initial_device
    assert wp.get_cuda_device() == initial_cuda_device


def test_multigpu_pingpong(test, device):
    
    assert len(wp.get_cuda_devices()) > 1, "At least two CUDA devices are required"

    n = 1024 * 1024
    
    a0 = wp.zeros(n, dtype=float, device="cuda:0")
    a1 = wp.zeros(n, dtype=float, device="cuda:1")

    stream0 = wp.get_stream("cuda:0")
    stream1 = wp.get_stream("cuda:1")

    iters = 10

    for i in range(iters):

        wp.launch(inc, dim=a0.size, inputs=[a0], device=a0.device)
        wp.copy(a1, a0, stream=stream0)

        stream1.wait_stream(stream0)

        wp.launch(inc, dim=a1.size, inputs=[a1], device=a1.device)
        wp.copy(a0, a1, stream=stream1)

        stream0.wait_stream(stream1)

    expected = np.full(n, iters * 2, dtype=np.float32)

    assert_np_equal(a0.numpy(), expected)
    assert_np_equal(a1.numpy(), expected)


def register(parent):

    class TestMultigpu(parent):
        pass

    if wp.get_cuda_device_count() > 1:

        add_function_test(TestMultigpu, "test_multigpu_set_device", test_multigpu_set_device)
        add_function_test(TestMultigpu, "test_multigpu_scoped_device", test_multigpu_scoped_device)
        add_function_test(TestMultigpu, "test_multigpu_nesting", test_multigpu_nesting)
        add_function_test(TestMultigpu, "test_multigpu_pingpong", test_multigpu_pingpong)
    
    return TestMultigpu

if __name__ == '__main__':
    c = register(unittest.TestCase)
    unittest.main(verbosity=2, failfast=False)
