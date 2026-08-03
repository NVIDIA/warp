# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train the actuator-collar control forces that steer the hanging cable to a
target shape.

Each training iteration simulates one full episode from the rest pose,
recorded on a single tape, and takes a gradient-descent step on the control
forces from the loss on the final cable shape.
"""

import numpy as np
import sim

import warp as wp

NUM_BEADS = 16
NUM_CTRL = 4  # actuator collars at beads 3, 7, 11, 15
NUM_STEPS = 96
SIM_DT = 0.015
TRAIN_ITERS = 400

# hyperparameters (hand-tuned)
SPRING_KE = 40.0  # 80.0 held the shape better but needed dt=0.008
SPRING_KD = 0.2  # 0.5 overdamped the swing and the cable never reached the target side
DRAG = 0.05
REST_LENGTH = 0.25
MASS = 0.3
GUST_AMP = 3.0  # crosswind strength; 5.0 pinned the cable against the shear layer
GUST_K = 10.0  # vertical wavenumber of the shear profile
RELAX_BETA = 1.0  # 0.5 let the chain ratchet apart at this dt
LEARNING_RATE = 1.0  # loss is an unnormalized sum over beads; 0.3 was stable but crawled


@wp.kernel
def sgd_step(grad: wp.array[wp.vec3], lr: float, params: wp.array[wp.vec3]):
    i = wp.tid()
    params[i] = params[i] - lr * grad[i]


def main():
    rng = np.random.default_rng(42)
    wp.init()

    states = [sim.alloc_state(NUM_BEADS) for _ in range(NUM_STEPS + 1)]

    q0 = sim.rest_positions(NUM_BEADS, REST_LENGTH)
    target = wp.array(sim.target_positions(NUM_BEADS, REST_LENGTH), dtype=wp.vec3)

    ctrl = wp.array(
        0.01 * rng.standard_normal((NUM_CTRL, 3)).astype(np.float32),
        dtype=wp.vec3,
        requires_grad=True,
    )
    loss = wp.zeros(1, dtype=float, requires_grad=True)

    for it in range(TRAIN_ITERS):
        states[0].q.assign(q0)
        states[0].qd.zero_()
        loss.zero_()

        tape = wp.Tape()
        with tape:
            for i in range(NUM_STEPS):
                sim.step(
                    states[i],
                    states[i + 1],
                    ctrl,
                    NUM_BEADS,
                    SPRING_KE,
                    SPRING_KD,
                    DRAG,
                    REST_LENGTH,
                    MASS,
                    GUST_AMP,
                    GUST_K,
                    RELAX_BETA,
                    SIM_DT,
                )
            wp.launch(sim.shape_loss, dim=NUM_BEADS, inputs=[states[NUM_STEPS], target, loss])

        tape.backward(loss=loss)

        wp.launch(sgd_step, dim=NUM_CTRL, inputs=[ctrl.grad, LEARNING_RATE, ctrl])

        if it % 10 == 0 or it == TRAIN_ITERS - 1:
            print(f"iter {it:3d}  loss = {loss.numpy()[0]:.6f}")

        tape.zero()

    print(f"final loss: {loss.numpy()[0]:.6f}")


if __name__ == "__main__":
    main()
