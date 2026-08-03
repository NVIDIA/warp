# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tune per-particle gains so that active responses match a target level.

The forward model computes a response per particle, compacts the particles
whose response exceeds the activation threshold into an active list, and
scores the active values against a target level. Gradients w.r.t. the gains
are computed with wp.Tape and used for gradient descent.
"""

import numpy as np
import sim

import warp as wp

NUM_PARTICLES = 24
TRAIN_ITERS = 12

TARGET_LEVEL = 2.4
PENALTY_BETA = 0.05
LEARNING_RATE = 0.05


def make_initial(rng: np.random.Generator):
    positions = np.linspace(0.2, 1.8, NUM_PARTICLES).astype(np.float32)
    gains = (1.0 + 0.3 * rng.standard_normal(NUM_PARTICLES)).astype(np.float32)
    return gains, positions


def main():
    rng = np.random.default_rng(7)
    gains_np, positions_np = make_initial(rng)

    positions = wp.array(positions_np, dtype=float)

    for it in range(TRAIN_ITERS):
        gains = wp.array(gains_np, dtype=float, requires_grad=True)

        tape = wp.Tape()
        with tape:
            loss = sim.run_episode(gains, positions, TARGET_LEVEL, PENALTY_BETA)
        tape.backward(loss)

        grad = gains.grad.numpy()
        grad_norm = np.linalg.norm(grad)
        print(f"iter {it:02d}  loss {loss.numpy()[0]:10.6f}  grad_norm {grad_norm:10.4f}")

        gains_np = gains_np - LEARNING_RATE * grad


if __name__ == "__main__":
    main()
