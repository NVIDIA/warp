# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fit sensor offsets: minimize 0.5 * sum_i (x_i - t_i)^2 by gradient descent.

The Hessian is the identity, so plain gradient descent is provably stable for
any lr < 2. We run lr = 1.3 (lower is too slow for the production sweep).
"""

import numpy as np
from optlib import FastTrainStep

import warp as wp

N = 64


@wp.kernel
def loss_kernel(x: wp.array(dtype=float), t: wp.array(dtype=float), loss: wp.array(dtype=float)):
    i = wp.tid()
    d = x[i] - t[i]
    wp.atomic_add(loss, 0, 0.5 * d * d)


def main():
    rng = np.random.default_rng(3)
    t = wp.array(rng.normal(0.0, 1.0, N).astype(np.float32), dtype=float)
    x = wp.array(np.zeros(N, dtype=np.float32), dtype=float, requires_grad=True)
    loss = wp.zeros(1, dtype=float, requires_grad=True)

    opt = FastTrainStep()
    lr = 1.3
    for it in range(40):
        loss.zero_()
        tape = wp.Tape()
        with tape:
            wp.launch(loss_kernel, dim=N, inputs=[x, t], outputs=[loss])
        new = opt.step(tape, loss, x, lr)
        x.assign(new.astype(np.float32))
        if it % 4 == 0 or it == 39:
            print(f"iter {it:3d}  loss={loss.numpy()[0]:.6e}")


if __name__ == "__main__":
    main()
