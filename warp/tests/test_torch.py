# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# include parent path
import numpy as np
import math

import torch

import warp as wp
import warp.torch

device = "cuda"

wp.init()


@wp.kernel
def test_kernel(
    x : wp.array(dtype=float),
    y : wp.array(dtype=float)):

    tid = wp.tid()

    y[tid] = 0.5 - x[tid]*2.0


# define PyTorch autograd op to wrap simulate func
class TestFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):

        # allocate output array
        y = torch.empty_like(x)

        ctx.x = x
        ctx.y = y

        wp.launch(
            kernel=test_kernel, 
            dim=len(x), 
            inputs=[wp.torch.from_torch(x)], 
            outputs=[wp.torch.from_torch(y)], 
            device=device)

        return y

    @staticmethod
    def backward(ctx, adj_y):
        
        # adjoints should be allocated as zero initialized
        adj_x = torch.zeros_like(ctx.x).contiguous()
        adj_y = adj_y.contiguous()

        wp.launch(
            kernel=test_kernel, 
            dim=len(ctx.x), 

            # fwd inputs
            inputs=[wp.torch.from_torch(ctx.x)],
            outputs=[None], 

            # adj inputs
            adj_inputs=[wp.torch.from_torch(adj_x)],
            adj_outputs=[wp.torch.from_torch(adj_y)],

            device=device,
            adjoint=True)

        return adj_x


a = wp.array(np.ones(10), dtype=wp.float32, device="cuda")
print(a)
print(a.ptr)

t = wp.torch.to_torch(a)
print(t)
print(t.data_ptr())

assert(a.ptr == t.data_ptr())

# input data
x = torch.ones(16, dtype=torch.float32, device=device, requires_grad=True).contiguous()

# execute op
y = TestFunc.apply(x)

# compute grads
l = y.sum()
l.backward()

print(f"y = {y}")
print(f"x.grad = {x.grad}")

passed = (x.grad == -2.0).all()
print(f"Test passed: {passed.item()}")