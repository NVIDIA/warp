# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Calibrate per-channel absorption coefficients against reference readings.

The reference detector readings were produced by a known excitation passing
through a material with unknown per-channel absorption. We recover the
absorption coefficients by gradient descent on a least-squares residual,
with gradients computed by wp.Tape.
"""

import model
import numpy as np

import warp as wp

NUM_CHANNELS = 64
TRAIN_ITERS = 40
LEARNING_RATE = 0.4


def make_problem(rng: np.random.Generator):
    x = np.linspace(0.0, 1.0, NUM_CHANNELS, dtype=np.float32)

    # positive excitation with some structure across channels
    source_np = (1.0 + 0.5 * np.sin(3.0 * np.pi * x)).astype(np.float32)

    # true absorption coefficients we are trying to recover
    mu_true = rng.uniform(0.4, 2.0, size=NUM_CHANNELS).astype(np.float32)

    total_thickness = model.NUM_LAYERS * model.LAYER_THICKNESS
    target_np = source_np * np.exp(-mu_true * total_thickness)

    return source_np, target_np, mu_true


def main():
    rng = np.random.default_rng(7)
    source_np, target_np, mu_true = make_problem(rng)

    target = wp.array(target_np, dtype=float)
    mu_np = np.full(NUM_CHANNELS, 1.0, dtype=np.float32)  # initial guess

    for it in range(TRAIN_ITERS):
        mu = wp.array(mu_np, dtype=float, requires_grad=True)
        signal = wp.array(source_np, dtype=float, requires_grad=True)

        tape = wp.Tape()
        with tape:
            loss = model.run_stack(mu, signal, target)
        tape.backward(loss)

        grad = mu.grad.numpy()
        mu_np = mu_np - LEARNING_RATE * grad

        mu_err = np.linalg.norm(mu_np - mu_true) / np.linalg.norm(mu_true)
        print(
            f"iter {it:02d}  loss {loss.numpy()[0]:12.8f}  |grad| {np.linalg.norm(grad):10.6f}  mu_rel_err {mu_err:8.5f}"
        )


if __name__ == "__main__":
    main()
