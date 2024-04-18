# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import jax.lax
import jax.numpy as jnp
import numpy as np


@jax.jit
def eval_springs(x, v, indices, rest, ke, kd):
    i = indices[:, 0]
    j = indices[:, 1]

    xi = x[i]
    xj = x[j]

    vi = v[i]
    vj = v[j]

    xij = xi - xj
    vij = vi - vj

    l = jnp.linalg.norm(xij, axis=1)
    l_inv = 1.0 / l

    # normalized spring direction
    dir = (xij.T * l_inv).T

    c = l - rest
    dcdt = jnp.sum(dir * vij, axis=1)

    # damping based on relative velocity.
    fs = dir.T * (ke * c + kd * dcdt)

    f = jnp.zeros_like(v)

    # f = jax.ops.index_add(f, i, -fs.T, indices_are_sorted=False, unique_indices=False)
    # f = jax.ops.index_add(f, j, fs.T, indices_are_sorted=False, unique_indices=False)

    f.at[i].add(-fs.T)
    f.at[j].add(fs.T)

    return f


@jax.jit
def integrate_particles(x, v, f, w, dt):
    g = jnp.array((0.0, 0.0 - 9.8, 0.0))
    s = w > 0.0

    a_ext = g * s[:, None]

    # simple semi-implicit Euler. v1 = v0 + a dt, x1 = x0 + v1 dt
    v += ((f.T * w).T + a_ext) * dt
    x += v * dt

    return (x, v)


class JxIntegrator:
    def __init__(self, cloth):
        self.cloth = cloth

        self.positions = jnp.array(self.cloth.positions)
        self.velocities = jnp.array(self.cloth.velocities)
        self.inv_mass = jnp.array(self.cloth.inv_masses)

        print(self.positions.device_buffer.device())

        self.spring_indices = jnp.array(self.cloth.spring_indices)
        self.spring_lengths = jnp.array(self.cloth.spring_lengths)
        self.spring_stiffness = jnp.array(self.cloth.spring_stiffness)
        self.spring_damping = jnp.array(self.cloth.spring_damping)

    def simulate(self, dt, substeps):
        sim_dt = dt / substeps

        for _s in range(substeps):
            f = eval_springs(
                self.positions,
                self.velocities,
                self.spring_indices.reshape((self.cloth.num_springs, 2)),
                self.spring_lengths,
                self.spring_stiffness,
                self.spring_damping,
            )

            # integrate
            (self.positions, self.velocities) = integrate_particles(
                self.positions, self.velocities, f, self.inv_mass, sim_dt
            )

        return np.array(self.positions)
