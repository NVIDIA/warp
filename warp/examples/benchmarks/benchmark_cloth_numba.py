# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import cupy as cp
import numpy as np
from numba import cuda, float32

# Notes:
#
# Current implementation requires some familarity of writing custom cuda kernels
# May be improved with cuda ufuncs and/or writing custom numba type extensions.


@cuda.jit(device=True)
def norm(x):
    s = float32(0.0)
    for i in range(3):
        s += x[i] * x[i]
    return math.sqrt(s)


@cuda.jit(device=True)
def dot(x, y):
    s = float32(0.0)
    for i in range(3):
        s += x[i] * y[i]
    return s


@cuda.jit
def eval_springs_cuda(
    num_springs,  # (1,)
    xs,  # position              (N, 3)
    vs,  # velocities            (N, 3)
    indices,  # spring indices        (S, 2)
    rests,  # spring rest length    (S,)
    kes,  # stiffness             (S,)
    kds,  # damping               (S,)
    fs,
):  # forces                (N, 3)
    tidx = cuda.grid(1)

    if tidx < num_springs:
        i, j = indices[tidx][0], indices[tidx][1]
        xi, xj = xs[i], xs[j]
        vi, vj = vs[i], vs[j]
        rest, ke, kd = rests[tidx], kes[tidx], kds[tidx]

        xij = cuda.local.array(3, dtype=cp.float32)
        vij = cuda.local.array(3, dtype=cp.float32)
        for k in range(3):
            xij[k] = xi[k] - xj[k]
        for k in range(3):
            vij[k] = vi[k] - vj[k]

        l = norm(xij)

        l_inv = float32(1.0) / l

        # normalized spring direction
        xij_unit = cuda.local.array(3, dtype=cp.float32)
        for k in range(3):
            xij_unit[k] = xij[k] * l_inv
        c = l - rest
        dcdt = dot(xij_unit, vij)

        # mass-spring-damper model
        fac = ke * c + kd * dcdt
        df = cuda.local.array(3, dtype=cp.float32)
        for k in range(3):
            df[k] = xij_unit[k] * fac

        for k in range(3):
            cuda.atomic.add(fs[i], k, -df[k])
            cuda.atomic.add(fs[j], k, df[k])


# Support const array with cp array?
g = np.array([0.0, 0.0 - 9.8, 0.0], dtype=np.float32)
z = np.array([0.0, 0.0, 0.0], dtype=np.float32)


@cuda.jit
def integrate_particles_cuda(
    xs,  # position  (N, 3)
    vs,  # velocity  (N, 3)
    fs,  # force     (N, 3)
    ws,  # inverse of mass (N,)
    dt,
):  # dt        (1,)
    i = cuda.grid(1)

    if i < xs.shape[0]:
        w = ws[i]
        a = cuda.const.array_like(g) if w > 0.0 else cuda.const.array_like(z)

        for j in range(3):
            # vs[i] += ((f * w) + a) * dt (ideally)
            vs[i][j] = vs[i][j] + ((fs[i][j] * w) + a[j]) * dt
            xs[i][j] = xs[i][j] + vs[i][j] * dt

        fs[i] = 0.0


class NbIntegrator:
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

        self.num_particles = self.positions.shape[0]
        self.integrate_tpb = 4
        self.integrate_nb = self.num_particles // self.integrate_tpb + 1

        self.spring_tpb = 4
        self.spring_nb = self.cloth.num_springs // self.spring_tpb + 1

    def simulate(self, dt, substeps):
        sim_dt = dt / substeps

        for _s in range(substeps):
            eval_springs_cuda[self.spring_nb, self.spring_tpb](
                self.cloth.num_springs,
                self.positions,
                self.velocities,
                self.spring_indices.reshape((self.cloth.num_springs, 2)),
                self.spring_lengths,
                self.spring_stiffness,
                self.spring_damping,
                self.forces,
            )

            # integrate
            integrate_particles_cuda[self.integrate_nb, self.integrate_tpb](
                self.positions, self.velocities, self.forces, self.inv_mass, sim_dt
            )

        return self.positions.get()
