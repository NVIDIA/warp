# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Torch <-> Warp bridge: a custom autograd Function that evaluates our Warp
smoothing kernel inside a PyTorch training loop.

Buffers are allocated once and reused across steps to avoid per-iteration
allocation overhead (the kernel runs thousands of times per epoch in the
real pipeline).
"""

import torch

import warp as wp

N = 512


@wp.kernel
def smooth_response(x: wp.array(dtype=float), y: wp.array(dtype=float)):
    i = wp.tid()
    xi = x[i]
    y[i] = xi / (1.0 + xi * xi) + 0.1 * xi


# persistent buffers, allocated once
_x_wp = wp.zeros(N, dtype=float, requires_grad=True)
_y_wp = wp.zeros(N, dtype=float, requires_grad=True)


class WarpSmoothResponse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        _x_wp.assign(wp.from_torch(x.detach().contiguous()))
        tape = wp.Tape()
        with tape:
            wp.launch(smooth_response, dim=N, inputs=[_x_wp], outputs=[_y_wp])
        ctx.tape = tape
        return wp.to_torch(_y_wp).clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        _y_wp.grad.assign(wp.from_torch(grad_output.contiguous()))
        ctx.tape.backward()
        return wp.to_torch(_x_wp.grad).clone()


def warp_smooth_response(x: torch.Tensor) -> torch.Tensor:
    return WarpSmoothResponse.apply(x)
