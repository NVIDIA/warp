# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Differentiable hanging-cable simulation.

A cable is modeled as a chain of beads connected by stiffening springs. Bead 0
is clamped to the ceiling; the rest hang under gravity in a layered crosswind
(a wind-shear profile that alternates direction with height). Actuator
collars are attached at every fourth bead; each applies a constant control
force (the trainable parameter) used to steer the cable toward a target
shape.

All per-bead state lives in a single ``CableState`` struct so kernels take one
argument instead of five arrays. Each simulation step runs:

  1. ``eval_forces``      - springs, gravity, drag, control
  2. ``relax_stretch``    - stabilization pass that relieves link overstretch
  3. ``apply_relaxation`` - write the relaxed positions back into the state
  4. ``integrate``        - semi-implicit Euler into the next state
"""

import numpy as np

import warp as wp


@wp.struct
class CableState:
    q: wp.array[wp.vec3]  # bead positions
    qd: wp.array[wp.vec3]  # bead velocities
    f: wp.array[wp.vec3]  # accumulated forces
    q_relaxed: wp.array[wp.vec3]  # output of the stretch-relaxation pass


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
    """Damped spring with cubic stiffening, acting on bead i from neighbor j."""
    d = qj - qi
    length = wp.length(d)
    direction = wp.normalize(d)
    stretch = length - rest
    stretch_rate = wp.dot(direction, vj - vi)
    return direction * (ke * stretch + ke * stretch * stretch * stretch + kd * stretch_rate)


@wp.kernel
def eval_forces(
    state: CableState,
    ctrl: wp.array[wp.vec3],
    spring_ke: float,
    spring_kd: float,
    drag: float,
    rest_length: float,
    mass: float,
    gust_amp: float,
    gust_k: float,
):
    i = wp.tid()
    n = state.q.shape[0]

    qi = state.q[i]
    vi = state.qd[i]

    fi = wp.vec3(0.0, -9.81 * mass, 0.0)

    # layered crosswind: horizontal push that alternates direction with height
    fi += wp.vec3(gust_amp * wp.sin(gust_k * qi[1]), 0.0, 0.0)

    if i > 0:
        fi += spring_force(qi, vi, state.q[i - 1], state.qd[i - 1], spring_ke, spring_kd, rest_length)
    if i < n - 1:
        fi += spring_force(qi, vi, state.q[i + 1], state.qd[i + 1], spring_ke, spring_kd, rest_length)

    # quadratic air drag
    fi -= drag * wp.length(vi) * vi

    # actuator collars sit on every fourth bead (3, 7, 11, ...)
    if i % 4 == 3:
        fi += ctrl[i / 4]

    state.f[i] = fi


@wp.kernel
def relax_stretch(state: CableState, beta: float, rest_length: float):
    """Stabilization pass: pull each bead back toward its parent so the link
    stretch is relieved by a factor ``beta``. Keeps the stiff chain from
    ratcheting apart at this time step size."""
    i = wp.tid()

    if i == 0:
        state.q_relaxed[i] = state.q[i]
    else:
        d = state.q[i] - state.q[i - 1]
        length = wp.length(d)
        direction = d / length
        state.q_relaxed[i] = state.q[i] - beta * (length - rest_length) * direction


@wp.kernel
def apply_relaxation(state: CableState):
    """Write the relaxed positions back so the integrator sees the stabilized configuration."""
    i = wp.tid()
    state.q[i] = state.q_relaxed[i]


@wp.kernel
def integrate(state_in: CableState, state_out: CableState, dt: float, inv_mass: float):
    i = wp.tid()

    vi = state_in.qd[i] + dt * inv_mass * state_in.f[i]
    if i == 0:
        vi = wp.vec3(0.0, 0.0, 0.0)  # bead 0 is clamped

    state_out.qd[i] = vi
    state_out.q[i] = state_in.q[i] + dt * vi


@wp.kernel
def shape_loss(state: CableState, target: wp.array[wp.vec3], loss: wp.array[float]):
    i = wp.tid()
    d = state.q[i] - target[i]
    wp.atomic_add(loss, 0, wp.length_sq(d))


def alloc_state(num_beads: int, requires_grad: bool = True, device=None) -> CableState:
    state = CableState()
    state.q = wp.zeros(num_beads, dtype=wp.vec3, requires_grad=requires_grad, device=device)
    state.qd = wp.zeros(num_beads, dtype=wp.vec3, requires_grad=requires_grad, device=device)
    state.f = wp.zeros(num_beads, dtype=wp.vec3, requires_grad=requires_grad, device=device)
    state.q_relaxed = wp.zeros(num_beads, dtype=wp.vec3, requires_grad=requires_grad, device=device)
    return state


def rest_positions(num_beads: int, rest_length: float) -> np.ndarray:
    """Initial pose: the cable sticks out horizontally from the clamp."""
    q = np.zeros((num_beads, 3), dtype=np.float32)
    q[:, 0] = np.arange(num_beads) * rest_length
    return q


def target_positions(num_beads: int, rest_length: float) -> np.ndarray:
    """Target pose: a quarter-circle droop from the clamp."""
    span = (num_beads - 1) * rest_length
    t = np.linspace(0.0, 0.5 * np.pi, num_beads)
    q = np.zeros((num_beads, 3), dtype=np.float32)
    q[:, 0] = span * np.sin(t)
    q[:, 1] = span * (np.cos(t) - 1.0)
    return q.astype(np.float32)


def step(
    state_in: CableState,
    state_out: CableState,
    ctrl: wp.array,
    num_beads: int,
    spring_ke: float,
    spring_kd: float,
    drag: float,
    rest_length: float,
    mass: float,
    gust_amp: float,
    gust_k: float,
    relax_beta: float,
    dt: float,
):
    """Advance the cable by one time step."""
    wp.launch(
        eval_forces,
        dim=num_beads,
        inputs=[state_in, ctrl, spring_ke, spring_kd, drag, rest_length, mass, gust_amp, gust_k],
    )
    wp.launch(relax_stretch, dim=num_beads, inputs=[state_in, relax_beta, rest_length])
    wp.launch(apply_relaxation, dim=num_beads, inputs=[state_in])
    wp.launch(integrate, dim=num_beads, inputs=[state_in, state_out, dt, 1.0 / mass])
