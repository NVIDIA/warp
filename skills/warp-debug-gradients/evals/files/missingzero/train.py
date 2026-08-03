# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Gradient-descent calibration of per-channel gains and offsets.

import numpy as np
from model import forward, make_problem

import warp as wp

N = 64
ITERS = 60
LR = 0.05  # tuned for N=64; try lowering this if the loss oscillates


def main():
    wp.init()

    w, b, x, y, loss = make_problem(N, seed=7)

    for it in range(ITERS):
        tape = wp.Tape()
        with tape:
            forward(w, b, x, y, loss)

        tape.backward(loss)

        grad_w = w.grad.numpy()
        grad_b = b.grad.numpy()

        print(
            f"iter {it:3d}  loss = {loss.numpy()[0]:12.6f}  "
            f"||grad_w|| = {np.linalg.norm(grad_w):12.6f}  "
            f"||grad_b|| = {np.linalg.norm(grad_b):12.6f}"
        )

        w.assign(w.numpy() - LR * grad_w)
        b.assign(b.numpy() - LR * grad_b)

    print("final loss   =", loss.numpy()[0])
    print("final w[:4]  =", w.numpy()[:4])
    print("final b[:4]  =", b.numpy()[:4])


if __name__ == "__main__":
    main()
