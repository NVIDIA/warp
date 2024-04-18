# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import cupy as cp
import cupyx as cpx


def eval_springs(x, v, indices, rest, ke, kd, f):
    i = indices[:, 0]
    j = indices[:, 1]

    xi = x[i]
    xj = x[j]

    vi = v[i]
    vj = v[j]

    xij = xi - xj
    vij = vi - vj

    l = cp.linalg.norm(xij, axis=1)
    l_inv = 1.0 / l

    # normalized spring direction
    dir = (xij.T * l_inv).T

    c = l - rest
    dcdt = cp.sum(dir * vij, axis=1)

    # damping based on relative velocity.
    fs = dir.T * (ke * c + kd * dcdt)

    cpx.scatter_add(f, i, -fs.T)
    cpx.scatter_add(f, j, fs.T)


def integrate_particles(x, v, f, w, dt):
    g = cp.array((0.0, 0.0 - 9.8, 0.0))
    s = w > 0.0

    a_ext = g * s[:, None]

    # simple semi-implicit Euler. v1 = v0 + a dt, x1 = x0 + v1 dt
    v += ((f.T * w).T + a_ext) * dt
    x += v * dt

    # clear forces
    f *= 0.0


class CpIntegrator:
    def __init__(self, cloth):
        self.cloth = cloth

        self.positions = cp.array(self.cloth.positions)
        self.velocities = cp.array(self.cloth.velocities)
        self.inv_mass = cp.array(self.cloth.inv_masses)

        self.spring_indices = cp.array(self.cloth.spring_indices)
        self.spring_lengths = cp.array(self.cloth.spring_lengths)
        self.spring_stiffness = cp.array(self.cloth.spring_stiffness)
        self.spring_damping = cp.array(self.cloth.spring_damping)

        self.forces = cp.zeros((self.cloth.num_particles, 3), dtype=cp.float32)

    def simulate(self, dt, substeps):
        sim_dt = dt / substeps

        for _s in range(substeps):
            eval_springs(
                self.positions,
                self.velocities,
                self.spring_indices.reshape((self.cloth.num_springs, 2)),
                self.spring_lengths,
                self.spring_stiffness,
                self.spring_damping,
                self.forces,
            )

            # integrate
            integrate_particles(self.positions, self.velocities, self.forces, self.inv_mass, sim_dt)

        # return np.array(self.positions)
        return self.positions.get()
