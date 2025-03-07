# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import torch


def eval_springs(x, v, indices, rest, ke, kd, f):
    i = indices[:, 0]
    j = indices[:, 1]

    xi = x[i]
    xj = x[j]

    vi = v[i]
    vj = v[j]

    xij = xi - xj
    vij = vi - vj

    l = torch.linalg.norm(xij, axis=1)
    l_inv = 1.0 / l

    # normalized spring direction
    dir = (xij.T * l_inv).T

    c = l - rest
    dcdt = torch.sum(dir * vij, axis=1)

    # damping based on relative velocity.
    fs = dir.T * (ke * c + kd * dcdt)

    f.index_add_(dim=0, index=i, source=-fs.T)
    f.index_add_(dim=0, index=j, source=fs.T)


def integrate_particles(x, v, f, g, w, dt):
    s = w > 0.0

    a_ext = g * s[:, None]

    # simple semi-implicit Euler. v1 = v0 + a dt, x1 = x0 + v1 dt
    v += ((f.T * w).T + a_ext) * dt
    x += v * dt

    # clear forces
    f *= 0.0


class TrIntegrator:
    def __init__(self, cloth, device):
        self.cloth = cloth

        self.positions = torch.tensor(self.cloth.positions, device=device)
        self.velocities = torch.tensor(self.cloth.velocities, device=device)
        self.inv_mass = torch.tensor(self.cloth.inv_masses, device=device)

        self.spring_indices = torch.tensor(self.cloth.spring_indices, device=device, dtype=torch.long)
        self.spring_lengths = torch.tensor(self.cloth.spring_lengths, device=device)
        self.spring_stiffness = torch.tensor(self.cloth.spring_stiffness, device=device)
        self.spring_damping = torch.tensor(self.cloth.spring_damping, device=device)

        self.forces = torch.zeros((self.cloth.num_particles, 3), dtype=torch.float32, device=device)
        self.gravity = torch.tensor((0.0, 0.0 - 9.8, 0.0), dtype=torch.float32, device=device)

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
            integrate_particles(self.positions, self.velocities, self.forces, self.gravity, self.inv_mass, sim_dt)

        return self.positions.cpu().numpy()
