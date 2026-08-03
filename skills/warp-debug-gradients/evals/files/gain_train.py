# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-pixel gain calibration: minimize the MEAN squared error between
corrected gains and reference exposures, plain gradient descent.

The mean objective L = (1/N) * sum_i (g_i - r_i)^2 has per-coordinate
curvature 2/N, so any learning rate below ~N/2 should be stable. I use
lr = 800 (N = 4096, so the stability bound is ~2048).
"""

import numpy as np

import warp as wp

N = 4096


@wp.kernel
def per_pixel_sqerr(
    gains: wp.array(dtype=float),
    ref: wp.array(dtype=float),
    terms: wp.array(dtype=float),
):
    i = wp.tid()
    d = gains[i] - ref[i]
    terms[i] = d * d


@wp.kernel
def mean_loss(terms: wp.array(dtype=float), loss: wp.array(dtype=float)):
    i = wp.tid()
    wp.atomic_add(loss, 0, terms[i] / float(N))


def main():
    rng = np.random.default_rng(0)
    gains = wp.array(rng.normal(1.0, 0.2, N).astype(np.float32), dtype=float, requires_grad=True)
    ref = wp.array(rng.normal(1.0, 0.05, N).astype(np.float32), dtype=float)
    terms = wp.zeros(N, dtype=float, requires_grad=True)
    loss = wp.zeros(1, dtype=float)

    lr = 800.0
    for it in range(50):
        tape = wp.Tape()
        with tape:
            wp.launch(per_pixel_sqerr, dim=N, inputs=[gains, ref], outputs=[terms])

        # Seed the loss adjoint and backprop to the gains.
        terms.grad.fill_(1.0)
        tape.backward()

        g = gains.grad.numpy().copy()
        new_gains = gains.numpy() - lr * g
        tape.zero()
        gains = wp.array(new_gains, dtype=float, requires_grad=True)

        loss.zero_()
        wp.launch(per_pixel_sqerr, dim=N, inputs=[gains, ref], outputs=[terms])
        wp.launch(mean_loss, dim=N, inputs=[terms], outputs=[loss])
        if it % 5 == 0 or it == 49:
            print(f"iter {it:3d}  mean loss = {loss.numpy()[0]:.6e}")


if __name__ == "__main__":
    main()
