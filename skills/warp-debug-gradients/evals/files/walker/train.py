# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train per-particle muscle activations for the walker chain.

Each training iteration simulates several frames from the rest pose. Every
frame is recorded on its own tape (truncated backprop through time, one
frame per window) and gradients are accumulated over the frames before a
single SGD step on the activations.
"""

import numpy as np
import sim

import warp as wp

NUM_PARTICLES = 32
NUM_SUBSTEPS = 8
SIM_DT = 0.02
FRAME_DT = NUM_SUBSTEPS * SIM_DT
FRAMES_PER_ITER = 4
TRAIN_ITERS = 20

LIFT = np.array([0.0, 0.25, 0.0], dtype=np.float32)  # desired offset above the rest pose

# hyperparameters (hand-tuned)
SPRING_KE = 150.0  # 400.0 tracked better but the chain jittered
SPRING_KD = 1.5  # bumped from 1.0 to calm the mid-chain oscillation
DRAG = 0.4
LEARNING_RATE = 8.0  # loss is an unnormalized sum over particles, so this runs large
W_VEL = 0.05  # velocity penalty; 0.5 made the gait sluggish


@wp.kernel
def accumulate_grad(src: wp.array[float], acc: wp.array[float]):
    i = wp.tid()
    acc[i] = acc[i] + src[i]


@wp.kernel
def sgd_step(grad: wp.array[float], lr: float, params: wp.array[float]):
    i = wp.tid()
    params[i] = params[i] - lr * grad[i]


def build_model(num_particles, device):
    """Construct the chain model with the current hyperparameters."""
    return sim.Model(
        num_particles,
        spacing=0.1,
        spring_ke=SPRING_KE,
        spring_kd=SPRING_KD,
        drag=DRAG,
        act_gain=30.0,
        target_radius=0.4,
        target_freq=10.0,
        device=device,
    )


def simulate_frame(model, states, activations, t0):
    """Run one frame of substeps: states[0] -> states[-1]."""
    for s in range(NUM_SUBSTEPS):
        sim.substep(model, states[s], states[s + 1], activations, t0 + s * SIM_DT, SIM_DT)


def main():
    rng = np.random.default_rng(42)
    device = wp.get_preferred_device()

    model = build_model(NUM_PARTICLES, device)
    states = [sim.State(NUM_PARTICLES, device=device) for _ in range(NUM_SUBSTEPS + 1)]

    q_init = wp.array(model.initial_q, dtype=wp.vec3, device=device)
    qd_init = wp.zeros(NUM_PARTICLES, dtype=wp.vec3, device=device)
    goal_targets = wp.array(model.initial_q + LIFT, dtype=wp.vec3, device=device, requires_grad=True)

    activations = wp.array(
        rng.normal(scale=0.1, size=NUM_PARTICLES).astype(np.float32),
        dtype=float,
        device=device,
        requires_grad=True,
    )
    grad_accum = wp.zeros(NUM_PARTICLES, dtype=float, device=device)
    loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)

    for it in range(TRAIN_ITERS):
        # reset the chain to the rest pose at the start of each episode
        wp.copy(states[0].q, q_init)
        wp.copy(states[0].qd, qd_init)
        grad_accum.zero_()

        t = 0.0
        loss_total = 0.0
        for _frame in range(FRAMES_PER_ITER):
            loss.zero_()
            tape = wp.Tape()
            with tape:
                simulate_frame(model, states, activations, t)
                wp.launch(
                    sim.frame_loss,
                    dim=NUM_PARTICLES,
                    inputs=[states[-1].q, states[-1].qd, goal_targets, 1.0, W_VEL],
                    outputs=[loss],
                    device=device,
                )
            tape.backward(loss=loss)

            wp.launch(
                accumulate_grad, dim=NUM_PARTICLES, inputs=[activations.grad], outputs=[grad_accum], device=device
            )
            loss_total += float(loss.numpy()[0])
            tape.zero()

            # carry the end-of-frame state into the next frame
            states[0] = states[-1]
            t += FRAME_DT

        wp.launch(
            sgd_step,
            dim=NUM_PARTICLES,
            inputs=[grad_accum, LEARNING_RATE / FRAMES_PER_ITER],
            outputs=[activations],
            device=device,
        )

        grad_norm = float(np.linalg.norm(grad_accum.numpy()))
        print(f"iter {it:3d}  loss {loss_total:12.6f}  |grad| {grad_norm:.4e}")


if __name__ == "__main__":
    main()
