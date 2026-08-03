# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optimize initial throw velocities so each projectile tracks recorded waypoints.

The forward model simulates NUM_STEPS explicit steps with quadratic drag, a
near-ground aerodynamic cushion, and a soft ground layer. The loss compares
position snapshots along the trajectory against recorded (noisy) waypoints.
Gradients w.r.t. the initial velocities are computed with wp.Tape and used for
gradient descent.
"""

import model
import numpy as np

import warp as wp

NUM_PARTICLES = 8
NUM_STEPS = 60
SNAPSHOT_EVERY = 15
DT = 0.02
TRAIN_ITERS = 80
LEARNING_RATE = 0.05

DRAG_COEFF = 0.4


@wp.kernel
def l2_loss(
    x: wp.array[wp.vec3],
    target: wp.array[wp.vec3],
    loss: wp.array[float],
):
    i = wp.tid()
    d = x[i] - target[i]
    wp.atomic_add(loss, 0, wp.dot(d, d))


def run_episode(x0: wp.array, v0: wp.array, drag: wp.array, targets: list):
    """Simulate from (x0, v0) and return the scalar loss array."""
    snapshots = model.rollout(x0, v0, drag, NUM_STEPS, DT, SNAPSHOT_EVERY)
    loss = wp.zeros(1, dtype=float, requires_grad=True)
    for snap, target in zip(snapshots, targets, strict=True):
        wp.launch(l2_loss, dim=x0.shape[0], inputs=[snap, target], outputs=[loss])
    return loss


def make_problem(rng: np.random.Generator):
    """Initial positions, drag coefficients, a reference throw, and waypoints."""
    x0_np = np.zeros((NUM_PARTICLES, 3), dtype=np.float32)
    x0_np[:, 0] = np.linspace(-1.0, 1.0, NUM_PARTICLES)
    x0_np[:, 1] = 0.6

    drag_np = np.full(NUM_PARTICLES, DRAG_COEFF, dtype=np.float32)

    # reference throw: settles onto the ground layer within the horizon
    v_true_np = rng.normal(0.0, 1.0, size=(NUM_PARTICLES, 3)).astype(np.float32)
    v_true_np[:, 1] -= 3.0

    x0 = wp.array(x0_np, dtype=wp.vec3, requires_grad=True)
    v_true = wp.array(v_true_np, dtype=wp.vec3)
    drag = wp.array(drag_np, dtype=float)
    snapshots = model.rollout(x0, v_true, drag, NUM_STEPS, DT, SNAPSHOT_EVERY)

    # recorded waypoints: reference snapshots with observation noise
    targets_np = [s.numpy() + rng.normal(0.0, 0.05, size=(NUM_PARTICLES, 3)).astype(np.float32) for s in snapshots]

    # initial guess: perturbed reference throw
    v0_np = v_true_np + rng.normal(0.0, 0.5, size=v_true_np.shape).astype(np.float32)

    return x0_np, v0_np, drag_np, targets_np


def main():
    rng = np.random.default_rng(7)
    x0_np, v0_np, drag_np, targets_np = make_problem(rng)

    drag = wp.array(drag_np, dtype=float)
    targets = [wp.array(t, dtype=wp.vec3) for t in targets_np]

    for it in range(TRAIN_ITERS):
        x0 = wp.array(x0_np, dtype=wp.vec3, requires_grad=True)
        v0 = wp.array(v0_np, dtype=wp.vec3, requires_grad=True)

        tape = wp.Tape()
        with tape:
            loss = run_episode(x0, v0, drag, targets)
        tape.backward(loss)

        grad = v0.grad.numpy()
        print(f"iter {it:02d}  loss {loss.numpy()[0]:10.6f}  grad_norm {np.linalg.norm(grad):10.4f}")

        v0_np = v0_np - LEARNING_RATE * grad


if __name__ == "__main__":
    main()
