# include parent path
import os
import sys
import numpy as np
import math
import ctypes

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pxr import Usd, UsdGeom, Gf, Sdf

import torch

import warp as wp
from warp.torch import torch_to_wp

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
            inputs=[torch_to_wp(x)], 
            outputs=[torch_to_wp(y)], 
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
            inputs=[torch_to_wp(ctx.x)],
            outputs=[None], 

            # adj inputs
            adj_inputs=[torch_to_wp(adj_x)],
            adj_outputs=[torch_to_wp(adj_y)],

            device=device,
            adjoint=True)

        return adj_x



# input data
x = torch.ones(16, dtype=torch.float32, device=device, requires_grad=True).contiguous()

# execute op
y = TestFunc.apply(x)

# compute grads
l = y.sum()
l.backward()

print(f"y = {y}")
print(f"x.grad = {x.grad}")
