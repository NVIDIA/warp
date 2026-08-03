# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Episode loss and diagnostic metrics."""

from state import ParticleState

import warp as wp


@wp.kernel
def episode_loss(
    sim_state: ParticleState,
    targets: wp.array[wp.vec3],
    w_pos: float,
    w_vel: float,
    loss: wp.array[float],
):
    i = wp.tid()
    d = sim_state.pos[i] - targets[i]
    v = sim_state.vel[i]
    wp.atomic_add(loss, 0, w_pos * wp.dot(d, d) + w_vel * wp.dot(v, v))


def mean_kinetic_energy(sim_state):
    """Diagnostic only (not taped)."""
    v = sim_state.vel.numpy()
    return float(0.5 * (v * v).sum(axis=1).mean())
