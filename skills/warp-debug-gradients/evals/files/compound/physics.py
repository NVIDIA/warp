# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Force evaluation and integration kernels for the particle ensemble."""

from state import ParticleState, make_state

import warp as wp


@wp.kernel
def spring_forces(
    sim_state: ParticleState,
    stiffness: wp.array[float],
    stiffening: float,
    f: wp.array[wp.vec3],
):
    i = wp.tid()
    p = sim_state.pos[i]
    # anchor spring toward the origin with cubic stiffening
    f[i] = -stiffness[0] * (1.0 + stiffening * wp.dot(p, p)) * p


@wp.kernel
def drag_forces(
    sim_state: ParticleState,
    drag: wp.array[float],
    f: wp.array[wp.vec3],
):
    i = wp.tid()
    v = sim_state.vel[i]
    # quadratic aerodynamic drag
    f[i] = -drag[0] * wp.length(v) * v


@wp.kernel
def limit_speed(sim_state: ParticleState, v_max: float):
    i = wp.tid()
    v = sim_state.vel[i]
    # smooth speed limit: |v| asymptotes to v_max, identity for slow particles
    sim_state.vel[i] = v / wp.sqrt(1.0 + wp.dot(v, v) / (v_max * v_max))


@wp.kernel
def integrate(
    sim_state: ParticleState,
    f_spring: wp.array[wp.vec3],
    f_drag: wp.array[wp.vec3],
    inv_mass: float,
    dt: float,
    pos_out: wp.array[wp.vec3],
    vel_out: wp.array[wp.vec3],
):
    i = wp.tid()
    v = sim_state.vel[i] + (f_spring[i] + f_drag[i]) * inv_mass * dt
    pos_out[i] = sim_state.pos[i] + v * dt
    vel_out[i] = v


def step(sim_state, stiffness, drag, buffers, stiffening, v_max, inv_mass, dt, device=None):
    """Advance the ensemble one step; returns the new state."""
    n = sim_state.pos.shape[0]
    wp.launch(
        spring_forces,
        dim=n,
        inputs=[sim_state, stiffness, stiffening],
        outputs=[buffers.f_spring],
        device=device,
    )
    wp.launch(drag_forces, dim=n, inputs=[sim_state, drag], outputs=[buffers.f_drag], device=device)
    # apply the speed limit before integration; the force passes above see the raw velocity
    wp.launch(limit_speed, dim=n, inputs=[sim_state, v_max], device=device)
    wp.launch(
        integrate,
        dim=n,
        inputs=[sim_state, buffers.f_spring, buffers.f_drag, inv_mass, dt],
        outputs=[buffers.pos_next, buffers.vel_next],
        device=device,
    )
    return make_state(buffers.pos_next, buffers.vel_next)
