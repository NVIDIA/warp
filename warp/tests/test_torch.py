# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# include parent path
import numpy as np
import unittest

import warp as wp
from warp.tests.test_base import *

wp.init()


@wp.kernel
def op_kernel(x: wp.array(dtype=float), y: wp.array(dtype=float)):

    tid = wp.tid()
    y[tid] = 0.5 - x[tid]*2.0


@wp.kernel
def inc(a: wp.array(dtype=float)):

    tid = wp.tid()
    a[tid] = a[tid] + 1.0


@wp.kernel
def arange(start: int, step: int, a: wp.array(dtype=int)):

    tid = wp.tid()
    a[tid] = start + step * tid


def test_torch_zerocopy(test, device):

    import torch

    a = wp.zeros(10, dtype=wp.float32, device=device)
    t = wp.to_torch(a)
    assert(a.ptr == t.data_ptr())

    torch_device = wp.device_to_torch(device)

    t = torch.zeros(10, dtype=torch.float32, device=torch_device)
    a = wp.from_torch(t)
    assert(a.ptr == t.data_ptr())


def test_torch_mgpu_from_torch(test, device):

    import torch

    n = 32

    t0 = torch.arange(0, n, 1, dtype=torch.int32, device="cuda:0")
    t1 = torch.arange(0, n*2, 2, dtype=torch.int32, device="cuda:1")

    a0 = wp.from_torch(t0, dtype=wp.int32)
    a1 = wp.from_torch(t1, dtype=wp.int32)

    assert a0.device == "cuda:0"
    assert a1.device == "cuda:1"

    expected0 = np.arange(0, n, 1)
    expected1 = np.arange(0, n*2, 2)

    assert_np_equal(a0.numpy(), expected0)
    assert_np_equal(a1.numpy(), expected1)


def test_torch_mgpu_to_torch(test, device):

    n = 32

    with wp.ScopedDevice("cuda:0"):
        a0 = wp.empty(n, dtype=wp.int32)
        wp.launch(arange, dim=a0.size, inputs=[0, 1, a0])

    with wp.ScopedDevice("cuda:1"):
        a1 = wp.empty(n, dtype=wp.int32)
        wp.launch(arange, dim=a1.size, inputs=[0, 2, a1])

    t0 = wp.to_torch(a0)
    t1 = wp.to_torch(a1)

    assert str(t0.device) == "cuda:0"
    assert str(t1.device) == "cuda:1"

    expected0 = np.arange(0, n, 1, dtype=np.int32)
    expected1 = np.arange(0, n*2, 2, dtype=np.int32)

    assert_np_equal(t0.cpu().numpy(), expected0)
    assert_np_equal(t1.cpu().numpy(), expected1)


def test_torch_mgpu_interop(test, device):

    import torch
    
    n = 1024 * 1024

    with torch.cuda.device(0):
        t0 = torch.arange(n, dtype=torch.float32, device="cuda")
        a0 = wp.from_torch(t0)
        wp.launch(inc, dim=a0.size, inputs=[a0], stream=wp.stream_from_torch())

    with torch.cuda.device(1):
        t1 = torch.arange(n, dtype=torch.float32, device="cuda")
        a1 = wp.from_torch(t1)
        wp.launch(inc, dim=a1.size, inputs=[a1], stream=wp.stream_from_torch())

    assert a0.device == "cuda:0"
    assert a1.device == "cuda:1"

    expected = np.arange(n, dtype=int) + 1

    # ensure the torch tensors were modified by warp
    assert_np_equal(t0.cpu().numpy(), expected)
    assert_np_equal(t1.cpu().numpy(), expected)


def test_torch_autograd(test, device):
    """Test torch autograd with a custom Warp op"""

    import torch

    # custom autograd op
    class TestFunc(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x):

            # allocate output array
            y = torch.empty_like(x)

            ctx.x = x
            ctx.y = y

            wp.launch(
                kernel=op_kernel, 
                dim=len(x), 
                inputs=[wp.from_torch(x)], 
                outputs=[wp.from_torch(y)])

            return y

        @staticmethod
        def backward(ctx, adj_y):
            
            # adjoints should be allocated as zero initialized
            adj_x = torch.zeros_like(ctx.x).contiguous()
            adj_y = adj_y.contiguous()

            wp.launch(
                kernel=op_kernel, 
                dim=len(ctx.x), 

                # fwd inputs
                inputs=[wp.from_torch(ctx.x)],
                outputs=[None], 

                # adj inputs
                adj_inputs=[wp.from_torch(adj_x)],
                adj_outputs=[wp.from_torch(adj_y)],
                adjoint=True)

            return adj_x

    # run autograd on given device
    with wp.ScopedDevice(device):

        torch_device = wp.device_to_torch(device)

        # input data
        x = torch.ones(16, dtype=torch.float32, device=torch_device, requires_grad=True)

        # execute op
        y = TestFunc.apply(x)

        # compute grads
        l = y.sum()
        l.backward()

        passed = (x.grad == -2.0).all()
        assert(passed.item())


def test_torch_graph_torch_stream(test, device):
    """Capture Torch graph on Torch stream"""

    import torch

    torch_device = wp.device_to_torch(device)

    n = 1024 * 1024
    t = torch.zeros(n, dtype=torch.float32, device=torch_device)
    a = wp.from_torch(t)

    g = torch.cuda.CUDAGraph()

    # create a device-specific torch stream to use for capture
    # (otherwise torch.cuda.graph reuses its capture stream, which can be problematic if it's from a different device)
    torch_stream = torch.cuda.Stream(device=torch_device)

    # make warp use the same stream
    warp_stream = wp.stream_from_torch(torch_stream)

    # capture graph
    with wp.ScopedStream(warp_stream), torch.cuda.graph(g, stream=torch_stream):
        t += 1.0
        wp.launch(inc, dim=n, inputs=[a])
        t += 1.0
        wp.launch(inc, dim=n, inputs=[a])

    # replay graph
    num_iters = 10
    for i in range(num_iters):
        g.replay()

    passed = (t == num_iters * 4.0).all()
    assert(passed.item())


def test_torch_graph_warp_stream(test, device):
    """Capture Torch graph on Warp stream"""

    import torch

    torch_device = wp.device_to_torch(device)

    n = 1024 * 1024
    t = torch.zeros(n, dtype=torch.float32, device=torch_device)
    a = wp.from_torch(t)

    g = torch.cuda.CUDAGraph()

    # make torch use the warp stream from the given device
    torch_stream = wp.stream_to_torch(device)

    # capture graph
    with wp.ScopedDevice(device), torch.cuda.graph(g, stream=torch_stream):
        t += 1.0
        wp.launch(inc, dim=n, inputs=[a])
        t += 1.0
        wp.launch(inc, dim=n, inputs=[a])

    # replay graph
    num_iters = 10
    for i in range(num_iters):
        g.replay()

    passed = (t == num_iters * 4.0).all()
    assert(passed.item())


def test_warp_graph_warp_stream(test, device):
    """Capture Warp graph on Warp stream"""

    import torch

    torch_device = wp.device_to_torch(device)

    n = 1024 * 1024
    t = torch.zeros(n, dtype=torch.float32, device=torch_device)
    a = wp.from_torch(t)

    # make torch use the warp stream from the given device
    torch_stream = wp.stream_to_torch(device)

    # capture graph
    with wp.ScopedDevice(device), torch.cuda.stream(torch_stream):
        wp.capture_begin()
        t += 1.0
        wp.launch(inc, dim=n, inputs=[a])
        t += 1.0
        wp.launch(inc, dim=n, inputs=[a])
        g = wp.capture_end()

    # replay graph
    num_iters = 10
    for i in range(num_iters):
        wp.capture_launch(g)

    passed = (t == num_iters * 4.0).all()
    assert(passed.item())


def test_warp_graph_torch_stream(test, device):
    """Capture Warp graph on Torch stream"""

    import torch

    torch_device = wp.device_to_torch(device)

    n = 1024 * 1024
    t = torch.zeros(n, dtype=torch.float32, device=torch_device)
    a = wp.from_torch(t)

    # create a device-specific torch stream to use for capture
    # (the default torch stream is not suitable for graph capture)
    torch_stream = torch.cuda.Stream(device=torch_device)

    # make warp use the same stream
    warp_stream = wp.stream_from_torch(torch_stream)

    # capture graph
    with wp.ScopedStream(warp_stream), torch.cuda.stream(torch_stream):
        wp.capture_begin()
        t += 1.0
        wp.launch(inc, dim=n, inputs=[a])
        t += 1.0
        wp.launch(inc, dim=n, inputs=[a])
        g = wp.capture_end()

    # replay graph
    num_iters = 10
    for i in range(num_iters):
        wp.capture_launch(g)

    passed = (t == num_iters * 4.0).all()
    assert(passed.item())


def register(parent):

    class TestTorch(parent):
        pass

    try:
        import torch

        devices = wp.get_devices()
        add_function_test(TestTorch, "test_torch_zerocopy", test_torch_zerocopy, devices=devices)
        add_function_test(TestTorch, "test_torch_autograd", test_torch_autograd, devices=devices)

        cuda_devices = wp.get_cuda_devices()
        add_function_test(TestTorch, "test_torch_graph_torch_stream", test_torch_graph_torch_stream, devices=cuda_devices)
        add_function_test(TestTorch, "test_torch_graph_warp_stream", test_torch_graph_warp_stream, devices=cuda_devices)
        add_function_test(TestTorch, "test_warp_graph_warp_stream", test_warp_graph_warp_stream, devices=cuda_devices)
        add_function_test(TestTorch, "test_warp_graph_torch_stream", test_warp_graph_torch_stream, devices=cuda_devices)

        if len(cuda_devices) > 1:
            add_function_test(TestTorch, "test_torch_mgpu_from_torch", test_torch_mgpu_from_torch)
            add_function_test(TestTorch, "test_torch_mgpu_to_torch", test_torch_mgpu_to_torch)
            add_function_test(TestTorch, "test_torch_mgpu_interop", test_torch_mgpu_interop)

    except ImportError:
        pass

    return TestTorch


if __name__ == '__main__':
    c = register(unittest.TestCase)
    unittest.main(verbosity=2)
