# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Differentiable mass-spring "walker" simulation.

A chain of particles connected by damped springs. Each particle carries a
muscle that pulls it toward a target point orbiting a fixed anchor; the
per-particle activation of that muscle is the trainable control signal.
Ported from an earlier PyTorch prototype.
"""

import numpy as np

import warp as wp


@wp.func
def spring_force(
    qi: wp.vec3,
    vi: wp.vec3,
    qj: wp.vec3,
    vj: wp.vec3,
    ke: float,
    kd: float,
    rest: float,
) -> wp.vec3:
    """Damped spring force on particle i from its neighbor j."""
    d = qj - qi
    length = wp.length(d)
    direction = wp.normalize(d)
    stretch = length - rest
    stretch_rate = wp.dot(direction, vj - vi)
    return direction * (ke * stretch + kd * stretch_rate)


@wp.kernel
def eval_forces(
    q: wp.array[wp.vec3],
    qd: wp.array[wp.vec3],
    activations: wp.array[float],
    anchors: wp.array[wp.vec3],
    spring_ke: float,
    spring_kd: float,
    drag: float,
    act_gain: float,
    rest_length: float,
    target_radius: float,
    target_freq: float,
    sim_time: float,
    f: wp.array[wp.vec3],
):
    i = wp.tid()
    n = q.shape[0]

    qi = q[i]
    vi = qd[i]

    fi = wp.vec3(0.0, 0.0, 0.0)

    # structural springs to chain neighbors
    if i > 0:
        fi += spring_force(qi, vi, q[i - 1], qd[i - 1], spring_ke, spring_kd, rest_length)
    if i < n - 1:
        fi += spring_force(qi, vi, q[i + 1], qd[i + 1], spring_ke, spring_kd, rest_length)

    # quadratic aerodynamic drag
    fi -= drag * wp.length(vi) * vi

    # muscle: pull toward a target orbiting the particle's anchor, with a
    # per-particle gait phase offset and a trainable activation
    phase = target_freq * sim_time + 0.7 * float(i)
    target = anchors[i] + wp.vec3(target_radius * wp.sin(phase), target_radius * wp.cos(phase), 0.0)
    fi += act_gain * activations[i] * (target - qi)

    f[i] = fi


@wp.kernel
def integrate(
    q: wp.array[wp.vec3],
    qd: wp.array[wp.vec3],
    f: wp.array[wp.vec3],
    inv_mass: float,
    gravity: float,
    dt: float,
    q_out: wp.array[wp.vec3],
    qd_out: wp.array[wp.vec3],
):
    i = wp.tid()
    v = qd[i] + (f[i] * inv_mass + wp.vec3(0.0, gravity, 0.0)) * dt
    q_out[i] = q[i] + v * dt
    qd_out[i] = v


@wp.kernel
def frame_loss(
    q: wp.array[wp.vec3],
    qd: wp.array[wp.vec3],
    targets: wp.array[wp.vec3],
    w_pos: float,
    w_vel: float,
    loss: wp.array[float],
):
    i = wp.tid()
    dq = q[i] - targets[i]
    wp.atomic_add(loss, 0, w_pos * wp.dot(dq, dq) + w_vel * wp.dot(qd[i], qd[i]))


class Model:
    """Static description of the walker chain.

    Args:
        num_particles: Number of particles in the chain.
        spacing: Rest distance between neighboring particles.
        spring_ke: Spring stiffness.
        spring_kd: Spring damping.
        drag: Quadratic drag coefficient.
        act_gain: Muscle strength.
        target_radius: Radius of the orbiting muscle targets.
        target_freq: Angular frequency of the target orbit (rad/s).
        mass: Particle mass.
        gravity: Vertical gravitational acceleration (signed).
        device: Warp device for allocations.
    """

    def __init__(
        self,
        num_particles,
        spacing=0.1,
        spring_ke=150.0,
        spring_kd=1.0,
        drag=0.5,
        act_gain=20.0,
        target_radius=0.05,
        target_freq=6.0,
        mass=1.0,
        gravity=-9.8,
        device=None,
    ):
        self.num_particles = num_particles
        self.spacing = spacing
        self.spring_ke = spring_ke
        self.spring_kd = spring_kd
        self.drag = drag
        self.act_gain = act_gain
        self.target_radius = target_radius
        self.target_freq = target_freq
        self.inv_mass = 1.0 / mass
        self.gravity = gravity
        self.device = device

        pts = np.zeros((num_particles, 3), dtype=np.float32)
        pts[:, 0] = spacing * np.arange(num_particles)
        pts[:, 1] = 1.0
        self.anchors = wp.array(pts, dtype=wp.vec3, device=device, requires_grad=True)
        self.initial_q = pts


class State:
    """Per-substep simulation state: positions, velocities, and forces."""

    def __init__(self, num_particles, device=None, requires_grad=True):
        self.q = wp.zeros(num_particles, dtype=wp.vec3, device=device, requires_grad=requires_grad)
        self.qd = wp.zeros(num_particles, dtype=wp.vec3, device=device, requires_grad=requires_grad)
        self.f = wp.zeros(num_particles, dtype=wp.vec3, device=device, requires_grad=requires_grad)


def substep(model, state_in, state_out, activations, sim_time, dt):
    """Advance the simulation one substep from ``state_in`` to ``state_out``."""
    n = model.num_particles
    wp.launch(
        eval_forces,
        dim=n,
        inputs=[
            state_in.q,
            state_in.qd,
            activations,
            model.anchors,
            model.spring_ke,
            model.spring_kd,
            model.drag,
            model.act_gain,
            model.spacing,
            model.target_radius,
            model.target_freq,
            sim_time,
        ],
        outputs=[state_in.f],
        device=model.device,
    )
    wp.launch(
        integrate,
        dim=n,
        inputs=[state_in.q, state_in.qd, state_in.f, model.inv_mass, model.gravity, dt],
        outputs=[state_out.q, state_out.qd],
        device=model.device,
    )
