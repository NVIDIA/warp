# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import sim

import warp as wp

learning_rate = 0.8
num_iterations = 20
seed = 7


def train(lr, iters, num_steps=sim.NUM_STEPS, verbose=True):
    rest = wp.array(sim.rest_positions(), dtype=wp.vec2)
    anchors = wp.array(sim.anchor_positions(), dtype=wp.vec2)
    q0 = wp.array(sim.rest_positions(), dtype=wp.vec2, requires_grad=True)
    control = wp.zeros(sim.NUM_PARTICLES, dtype=wp.vec2, requires_grad=True)
    target = wp.array(sim.make_target(num_steps, seed=seed), dtype=wp.vec2)
    loss = wp.zeros(1, dtype=float, requires_grad=True)
    speeds = wp.zeros(sim.NUM_PARTICLES, dtype=float)

    states_q, states_qd = sim.allocate_states(num_steps)

    history = []
    for it in range(iters):
        loss.zero_()

        tape = wp.Tape()
        with tape:
            sim.rollout(q0, control, anchors, states_q, states_qd, num_steps, target, loss)

            # end-of-episode housekeeping: log final speeds, restore rest pose
            wp.launch(sim.speed_metrics, dim=sim.NUM_PARTICLES, inputs=[states_qd[num_steps - 1]], outputs=[speeds])
            sim.reset_states(rest, states_q, states_qd)

        tape.backward(loss)

        grad = control.grad.numpy().copy()
        loss_val = float(loss.numpy()[0])
        mean_speed = float(speeds.numpy().mean())
        history.append(loss_val)

        if verbose:
            print(
                f"iter {it:3d}  loss {loss_val:.6e}  grad_norm {np.linalg.norm(grad):.6e}  mean_speed {mean_speed:.4f}"
            )

        tape.zero()
        tape.reset()

        control.assign(control.numpy() - lr * grad)

    return history


if __name__ == "__main__":
    wp.init()
    train(learning_rate, num_iterations)
