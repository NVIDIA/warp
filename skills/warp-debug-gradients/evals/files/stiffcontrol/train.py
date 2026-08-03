# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import sim

import warp as wp

learning_rate = 1.0e5
num_iterations = 25
seed = 42


def train(lr, iters, num_steps=sim.NUM_STEPS, verbose=True):
    x_init = wp.array(sim.rest_positions(), dtype=float, requires_grad=True)
    v0 = wp.zeros(sim.NUM_PARTICLES, dtype=float, requires_grad=True)
    target = wp.array(sim.make_target(num_steps, seed=seed), dtype=float)
    loss = wp.zeros(1, dtype=float, requires_grad=True)

    states_x, states_v = sim.allocate_states(num_steps)

    history = []
    for it in range(iters):
        loss.zero_()

        tape = wp.Tape()
        with tape:
            sim.rollout(x_init, v0, states_x, states_v, num_steps, target, loss)

        tape.backward(loss)

        grad = v0.grad.numpy().copy()
        loss_val = float(loss.numpy()[0])
        grad_norm = float(np.linalg.norm(grad))
        history.append((loss_val, grad_norm))

        if verbose:
            print(f"iter {it:3d}  loss {loss_val:.6e}  grad_norm {grad_norm:.6e}")

        tape.zero()
        tape.reset()

        v0.assign(v0.numpy() - lr * grad)

    return history


if __name__ == "__main__":
    wp.init()
    train(learning_rate, num_iterations)
