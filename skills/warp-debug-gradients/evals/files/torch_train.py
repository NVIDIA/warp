# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fit input signals so the Warp response layer matches target responses.

Plain PyTorch training loop; the Warp kernel is wrapped in a custom
autograd.Function (see warp_layer.py).
"""

import torch
from warp_layer import N, warp_smooth_response


def main():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target = torch.rand(N, device=device) * 0.3

    x = torch.zeros(N, device=device, requires_grad=True)
    opt = torch.optim.SGD([x], lr=60.0)

    for it in range(60):
        opt.zero_grad()
        y = warp_smooth_response(x)
        loss = ((y - target) ** 2).mean()
        loss.backward()
        opt.step()
        if it % 5 == 0 or it == 59:
            gn = float(x.grad.norm())
            print(f"iter {it:3d}  loss={loss.item():.6e}  |x.grad|={gn:.4e}")


if __name__ == "__main__":
    main()
