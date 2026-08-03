# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Calibrate contact drive parameters by minimizing the total contact energy.

Gradients w.r.t. ``theta`` are computed with wp.Tape and used for plain
gradient descent.
"""

import model
import numpy as np

import warp as wp

NUM_CONTACTS = 32
TRAIN_ITERS = 60
LEARNING_RATE = 0.05
MU = 1.0


def make_problem(rng: np.random.Generator):
    theta0 = rng.uniform(0.2, 0.8, NUM_CONTACTS).astype(np.float32)
    gain = rng.uniform(0.5, 1.5, NUM_CONTACTS).astype(np.float32)
    stiffness = rng.uniform(0.8, 1.2, NUM_CONTACTS).astype(np.float32)
    force = rng.uniform(1.5, 2.5, NUM_CONTACTS).astype(np.float32)
    return theta0, gain, stiffness, force


def compute_loss(theta: wp.array, gain: wp.array, stiffness: wp.array, force: wp.array):
    """Total contact energy for the current parameters (scalar loss array)."""
    n = theta.shape[0]
    disp = wp.zeros(n, dtype=float, requires_grad=True)
    wp.launch(model.apply_gain, dim=n, inputs=[theta, gain], outputs=[disp])

    loss = wp.zeros(1, dtype=float, requires_grad=True)
    wp.launch(model.contact_energy, dim=n, inputs=[disp, stiffness, force, MU], outputs=[loss])
    return loss


def main():
    rng = np.random.default_rng(7)
    theta_np, gain_np, stiffness_np, force_np = make_problem(rng)

    gain = wp.array(gain_np, dtype=float)
    stiffness = wp.array(stiffness_np, dtype=float)
    force = wp.array(force_np, dtype=float)

    for it in range(TRAIN_ITERS):
        theta = wp.array(theta_np, dtype=float, requires_grad=True)

        tape = wp.Tape()
        with tape:
            loss = compute_loss(theta, gain, stiffness, force)
        tape.backward(loss)

        grad = theta.grad.numpy()
        if it % 10 == 0 or it == TRAIN_ITERS - 1:
            print(f"iter {it:02d}  loss {loss.numpy()[0]:10.6f}  grad_norm {np.linalg.norm(grad):10.6f}")

        theta_np = theta_np - LEARNING_RATE * grad

    print(f"final mean displacement: {np.mean(theta_np * gain_np):.6f}")


if __name__ == "__main__":
    main()
