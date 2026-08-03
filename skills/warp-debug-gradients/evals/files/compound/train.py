# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Jointly fit spring stiffness and drag so the ensemble contracts onto a target cloud.

Each training iteration simulates one full episode from the seeded initial
conditions under a single tape, then takes one gradient step on both scalar
parameters.
"""

import objective
import physics
import state as state_mod

import warp as wp

NUM_PARTICLES = 64
NUM_STEPS = 24
DT = 0.05
SEED = 7
SPREAD = 1.0  # initial positions fill [-SPREAD, SPREAD]^3
SPEED = 1.5  # initial velocity scale

# hyperparameters (hand-tuned)
V_MAX = 0.8  # soft speed limit; 3.0 let the fast tail of the ensemble ring for many steps
STIFFENING = 0.6  # cubic term of the anchor spring; 0.0 reduces to a linear spring
INV_MASS = 1.0
W_POS = 1.0
W_VEL = 0.25  # velocity penalty; the ensemble should come to rest on the target cloud
TARGET_SHRINK = 0.4  # target cloud = shrunk copy of the initial positions

INIT_STIFFNESS = 1.2
INIT_DRAG = 0.3
LR_STIFFNESS = 3e-3
LR_DRAG = 5e-2  # the drag gradient runs an order of magnitude smaller than stiffness
TRAIN_ITERS = 40


def run_episode(init, stiffness, drag, targets, loss, device=None):
    """Simulate one episode from the initial conditions and accumulate the loss."""
    # reset to the initial conditions
    s = state_mod.make_state(wp.clone(init.pos), wp.clone(init.vel))
    for _ in range(NUM_STEPS):
        buffers = state_mod.StepBuffers(NUM_PARTICLES, device=device)
        s = physics.step(s, stiffness, drag, buffers, STIFFENING, V_MAX, INV_MASS, DT, device=device)
    wp.launch(
        objective.episode_loss,
        dim=NUM_PARTICLES,
        inputs=[s, targets, W_POS, W_VEL],
        outputs=[loss],
        device=device,
    )
    return s


def main():
    device = wp.get_preferred_device()

    init = state_mod.alloc_initial_state(NUM_PARTICLES, SEED, spread=SPREAD, speed=SPEED, device=device)
    targets = wp.array(TARGET_SHRINK * init.pos.numpy(), dtype=wp.vec3, device=device, requires_grad=True)

    k_val = INIT_STIFFNESS
    c_val = INIT_DRAG
    for it in range(TRAIN_ITERS):
        stiffness = state_mod.alloc_param(k_val, device=device)
        drag = state_mod.alloc_param(c_val, device=device)
        loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)

        tape = wp.Tape()
        with tape:
            final = run_episode(init, stiffness, drag, targets, loss, device=device)
        tape.backward(loss=loss)

        grad_k = float(stiffness.grad.numpy()[0])
        grad_c = float(drag.grad.numpy()[0])
        k_val = max(k_val - LR_STIFFNESS * grad_k, 0.0)  # keep the coefficients physical
        c_val = max(c_val - LR_DRAG * grad_c, 0.0)

        print(
            f"iter {it:3d}  loss {float(loss.numpy()[0]):10.5f}  "
            f"k {k_val:.4f} (|grad| {abs(grad_k):.4e})  "
            f"c {c_val:.4f} (|grad| {abs(grad_c):.4e})  "
            f"KE {objective.mean_kinetic_energy(final):.4f}"
        )


if __name__ == "__main__":
    main()
