# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""State containers and allocation helpers for the particle ensemble."""

import numpy as np

import warp as wp


@wp.struct
class ParticleState:
    pos: wp.array[wp.vec3]
    vel: wp.array[wp.vec3]


def make_state(pos, vel):
    """Wrap position/velocity arrays in a ``ParticleState``."""
    s = ParticleState()
    s.pos = pos
    s.vel = vel
    return s


def alloc_initial_state(num_particles, seed, spread, speed, device=None):
    """Seeded initial conditions: positions in a cube, Gaussian velocities."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-spread, spread, size=(num_particles, 3)).astype(np.float32)
    vel = rng.normal(scale=speed, size=(num_particles, 3)).astype(np.float32)
    return make_state(
        wp.array(pos, dtype=wp.vec3, device=device, requires_grad=True),
        wp.array(vel, dtype=wp.vec3, device=device, requires_grad=True),
    )


class StepBuffers:
    """Scratch arrays for one simulation step.

    A fresh set is allocated every step so each taped launch writes to its
    own buffers.
    """

    def __init__(self, num_particles, device=None):
        self.f_spring = wp.zeros(num_particles, dtype=wp.vec3, device=device)
        self.f_drag = wp.zeros(num_particles, dtype=wp.vec3, device=device, requires_grad=True)
        self.pos_next = wp.zeros(num_particles, dtype=wp.vec3, device=device, requires_grad=True)
        self.vel_next = wp.zeros(num_particles, dtype=wp.vec3, device=device, requires_grad=True)


def alloc_param(value, device=None):
    """Allocate a trainable scalar parameter."""
    return wp.array([value], dtype=float, device=device, requires_grad=True)
