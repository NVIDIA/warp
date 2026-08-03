# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Projectile ensemble with quadratic aerodynamic drag and a soft ground layer.

Each particle carries its own drag coefficient plus a near-ground aerodynamic
cushion; while a particle is below the ground plane, a spring-like push-back
joins the vertical force.
"""

import warp as wp

GRAVITY = 9.81
FLOOR_Y = 0.0
CONTACT_STIFFNESS = 60.0
CUSHION_LIFT = 4.0
CUSHION_HEIGHT = 0.25


@wp.kernel
def step(
    x: wp.array[wp.vec3],
    v: wp.array[wp.vec3],
    drag: wp.array[float],
    dt: float,
    x_out: wp.array[wp.vec3],
    v_out: wp.array[wp.vec3],
):
    i = wp.tid()
    p = x[i]
    u = v[i]
    c = drag[i]
    speed = wp.length(u)

    # aerodynamic cushion: lift strengthens as the particle nears the ground
    lift = CUSHION_LIFT * wp.exp(-p[1] / CUSHION_HEIGHT)

    f = wp.vec3()
    f[0] = -c * speed * u[0]
    f[1] = lift - GRAVITY - c * speed * u[1]
    f[2] = -c * speed * u[2]

    # soft ground: below the floor a push-back spring joins the vertical force
    if p[1] < FLOOR_Y:
        f[1] = CONTACT_STIFFNESS * (FLOOR_Y - p[1]) + lift - GRAVITY - c * speed * u[1]

    u_new = u + dt * f
    v_out[i] = u_new
    x_out[i] = p + dt * u_new


def rollout(x0: wp.array, v0: wp.array, drag: wp.array, num_steps: int, dt: float, snapshot_every: int):
    """Integrate ``num_steps`` explicit steps, returning a list of position
    snapshots taken every ``snapshot_every`` steps (when ``num_steps`` is a
    multiple of ``snapshot_every``, as at every call site in this fixture,
    the last snapshot is the final state).

    Every step writes into fresh arrays so the tape sees distinct buffers.
    """
    snapshots = []
    x, v = x0, v0
    for s in range(num_steps):
        x_out = wp.empty_like(x)
        v_out = wp.empty_like(v)
        wp.launch(step, dim=x.shape[0], inputs=[x, v, drag, dt], outputs=[x_out, v_out])
        x, v = x_out, v_out
        if (s + 1) % snapshot_every == 0:
            snapshots.append(x)
    return snapshots
