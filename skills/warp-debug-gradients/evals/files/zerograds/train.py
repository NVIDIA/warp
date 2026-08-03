# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Joint gradient-descent fit of params_a and params_b against a synthetic target.

import numpy as np
from pipeline import allocate_buffers, forward, make_problem

import warp as wp

N = 64
ITERS = 20
LR = 0.05  # try lowering this if the loss oscillates


def main():
    wp.init()

    params_a, params_b, coords, target = make_problem(N, seed=7)
    buffers = allocate_buffers(N)

    for it in range(ITERS):
        tape = wp.Tape()
        with tape:
            # Loss is a plain sum of squared residuals; no 1/N scaling here,
            # so the gradient magnitude grows with N (LR is tuned for N=64).
            loss = forward(params_a, params_b, coords, target, buffers)

        tape.backward(loss)

        grad_a = params_a.grad.numpy()
        grad_b = params_b.grad.numpy()

        print(
            f"iter {it:3d}  loss = {loss.numpy()[0]:.6f}  "
            f"||grad_a|| = {np.linalg.norm(grad_a):.6f}  "
            f"||grad_b|| = {np.linalg.norm(grad_b):.6f}"
        )

        params_a.assign(params_a.numpy() - LR * grad_a)
        params_b.assign(params_b.numpy() - LR * grad_b)

        tape.zero()

    print("final params_a[:4] =", params_a.numpy()[:4])
    print("final params_b[:4] =", params_b.numpy()[:4])


if __name__ == "__main__":
    main()
