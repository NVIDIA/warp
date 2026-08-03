# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optimize the initial configuration of a hanging chain to match a target shape.

The forward model runs several simulation steps, each consisting of a few
relaxation sweeps followed by a workspace projection. Gradients w.r.t. the
initial positions are computed with wp.Tape and used for a simple
gradient-descent update.
"""

import numpy as np
import solver

import warp as wp

NUM_PARTICLES = 32
NUM_SWEEPS = 3
NUM_STEPS = 6
TRAIN_ITERS = 40

REST_LENGTH = 1.0
STIFFNESS = 0.35
GRAVITY = 0.03
FLOOR_HEIGHT = -0.05
FLOOR_SMOOTHING = 0.015
LEARNING_RATE = 0.15


@wp.kernel
def project_workspace(
    q: wp.array[wp.vec3],
    floor_height: float,
    smoothing: float,
    q_out: wp.array[wp.vec3],
):
    i = wp.tid()
    p = q[i]

    # keep particles inside the workspace: reflect any floor penetration back
    # above the floor plane (smoothed so there is no kink at the plane itself)
    d = p[1] - floor_height
    y = floor_height + wp.sqrt(d * d + smoothing * smoothing)

    q_out[i] = wp.vec3(p[0], y, p[2])


@wp.kernel
def l2_loss(
    q: wp.array[wp.vec3],
    target: wp.array[wp.vec3],
    loss: wp.array[float],
):
    i = wp.tid()
    d = q[i] - target[i]
    wp.atomic_add(loss, 0, wp.dot(d, d))


def simulate(q0: wp.array) -> wp.array:
    """Simulate NUM_STEPS steps from ``q0`` and return the final positions."""
    n = q0.shape[0]
    state_q = wp.clone(q0)

    for _step in range(NUM_STEPS):
        state_q = solver.relax(state_q, NUM_SWEEPS, REST_LENGTH, STIFFNESS, GRAVITY)

        # keep particles inside the workspace
        projected = wp.empty_like(state_q)
        wp.launch(project_workspace, dim=n, inputs=[state_q, FLOOR_HEIGHT, FLOOR_SMOOTHING], outputs=[projected])
        wp.copy(state_q, projected)

    return state_q


def run_episode(q0: wp.array, target: wp.array):
    """Simulate an episode from ``q0`` and return the scalar loss array."""
    state_q = simulate(q0)

    loss = wp.zeros(1, dtype=float, requires_grad=True)
    wp.launch(l2_loss, dim=q0.shape[0], inputs=[state_q, target], outputs=[loss])
    return loss


def make_initial_and_target(rng: np.random.Generator):
    x = (np.arange(NUM_PARTICLES, dtype=np.float32) - 0.5 * (NUM_PARTICLES - 1)) * REST_LENGTH

    base = np.zeros((NUM_PARTICLES, 3), dtype=np.float32)
    base[:, 0] = x

    q0_np = base + rng.normal(0.0, 0.05, size=base.shape).astype(np.float32)
    q0_np[0] = base[0]
    q0_np[-1] = base[-1]

    # a shallow valley that dips toward the workspace floor at mid-chain
    target_np = base.copy()
    target_np[:, 1] = -0.04 * np.cos(0.5 * np.pi * x / x[-1]).astype(np.float32)

    return q0_np, target_np


def main():
    rng = np.random.default_rng(7)
    q0_np, target_np = make_initial_and_target(rng)

    target = wp.array(target_np, dtype=wp.vec3)

    for it in range(TRAIN_ITERS):
        q0 = wp.array(q0_np, dtype=wp.vec3, requires_grad=True)

        tape = wp.Tape()
        with tape:
            loss = run_episode(q0, target)
        tape.backward(loss)

        grad = q0.grad.numpy()
        grad_norm = np.linalg.norm(grad)
        print(f"iter {it:02d}  loss {loss.numpy()[0]:10.6f}  grad_norm {grad_norm:12.4f}")

        q0_np = q0_np - LEARNING_RATE * grad


if __name__ == "__main__":
    main()
