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

    n = 32
    
    a0 = wp.zeros(n, dtype=float, device="cuda:0")
    a1 = wp.zeros(n, dtype=float, device="cuda:1")

    iters = 10

    for i in range(iters):

        wp.launch(inc, dim=a0.size, inputs=[a0], device=a0.device)
        wp.synchronize_device(a0.device)
        wp.copy(a1, a0)

        wp.launch(inc, dim=a1.size, inputs=[a1], device=a1.device)
        wp.synchronize_device(a1.device)
        wp.copy(a0, a1)

    wp.synchronize()

    expected = np.full(n, iters * 2, dtype=np.float32)

    assert_np_equal(a0.numpy(), expected)
    assert_np_equal(a1.numpy(), expected)


def test_multigpu_from_torch(test, device):

    import torch

    n = 32

    t0 = torch.arange(0, n, 1, dtype=torch.int32, device="cuda:0")
    t1 = torch.arange(0, n*2, 2, dtype=torch.int32, device="cuda:1")

    a0 = wp.from_torch(t0, dtype=wp.int32)
    a1 = wp.from_torch(t1, dtype=wp.int32)

    wp.synchronize()

    assert a0.device == "cuda:0"
    assert a1.device == "cuda:1"

    expected0 = np.arange(0, n, 1)
    expected1 = np.arange(0, n*2, 2)

    assert_np_equal(a0.numpy(), expected0)
    assert_np_equal(a1.numpy(), expected1)


def test_multigpu_to_torch(test, device):

    n = 32

    with wp.ScopedDevice("cuda:0"):
        a0 = wp.empty(n, dtype=wp.int32)
        wp.launch(arange, dim=a0.size, inputs=[0, 1, a0])

    with wp.ScopedDevice("cuda:1"):
        a1 = wp.empty(n, dtype=wp.int32)
        wp.launch(arange, dim=a1.size, inputs=[0, 2, a1])

    wp.synchronize()

    t0 = wp.to_torch(a0)
    t1 = wp.to_torch(a1)

    assert str(t0.device) == "cuda:0"
    assert str(t1.device) == "cuda:1"

    expected0 = np.arange(0, n, 1, dtype=np.int32)
    expected1 = np.arange(0, n*2, 2, dtype=np.int32)

    assert_np_equal(t0.cpu().numpy(), expected0)
    assert_np_equal(t1.cpu().numpy(), expected1)


def test_multigpu_torch_interop(test, device):

    import torch
    
    n = 32

    with torch.cuda.device(0):
        t0 = torch.arange(n, dtype=torch.float32, device="cuda")
        torch.cuda.synchronize()
        a0 = wp.from_torch(t0)
        wp.launch(inc, dim=a0.size, inputs=[a0], device="cuda")

    with torch.cuda.device(1):
        t1 = torch.arange(n, dtype=torch.float32, device="cuda")
        torch.cuda.synchronize()
        a1 = wp.from_torch(t1)
        wp.launch(inc, dim=a1.size, inputs=[a1], device="cuda")

    wp.synchronize()

    assert a0.device == "cuda:0"
    assert a1.device == "cuda:1"

    expected = np.arange(n, dtype=int) + 1

    # ensure the torch tensors were modified by warp
    assert_np_equal(t0.cpu().numpy(), expected)
    assert_np_equal(t1.cpu().numpy(), expected)


def register(parent):

    class TestMultigpu(parent):

        @classmethod
        def setUpClass(cls):

            cls.saved_device = wp.get_device()

            # if there's only one GPU, emulate a second one using a custom CUDA context
            cuda_devices = wp.get_cuda_devices()
            if len(cuda_devices) == 1:
                cls.emulated_device_alias = "cuda:1"
                cls.emulated_device_context = wp.context.runtime.core.cuda_context_create(0)
                wp.map_cuda_device(cls.emulated_device_alias, cls.emulated_device_context)
            else:
                cls.emulated_device_alias = None

        @classmethod
        def tearDownClass(cls):

            wp.set_device(cls.saved_device)

            if cls.emulated_device_alias is not None:

                # Note: Destroying a custom CUDA context is tricky business.
                # We need to ensure that all resources associated with the context get garbage collected first.
                # (Creating/destroying contexts is not publicly exposed, but it's useful for testing.)

                wp.unmap_cuda_device(cls.emulated_device_alias)

                import gc
                gc.collect()

                wp.context.runtime.core.cuda_context_destroy(cls.emulated_device_context)


    if wp.is_cuda_available():

        add_function_test(TestMultigpu, "test_multigpu_set_device", test_multigpu_set_device)
        add_function_test(TestMultigpu, "test_multigpu_scoped_device", test_multigpu_scoped_device)
        add_function_test(TestMultigpu, "test_multigpu_nesting", test_multigpu_nesting)
        add_function_test(TestMultigpu, "test_multigpu_pingpong", test_multigpu_pingpong)

        # if there are at least two physical CUDA devices and torch is installed
        cuda_devices = wp.get_cuda_devices()
        if len(cuda_devices) > 1 and cuda_devices[0].is_primary and cuda_devices[1].is_primary:
            try:
                import torch
                add_function_test(TestMultigpu, "test_multigpu_from_torch", test_multigpu_from_torch)
                add_function_test(TestMultigpu, "test_multigpu_to_torch", test_multigpu_to_torch)
                add_function_test(TestMultigpu, "test_multigpu_torch_interop", test_multigpu_torch_interop)
            except ImportError:
                pass
    
    return TestMultigpu

if __name__ == '__main__':
    c = register(unittest.TestCase)
    unittest.main(verbosity=2, failfast=False)
