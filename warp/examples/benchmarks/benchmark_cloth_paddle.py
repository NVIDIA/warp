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

import paddle


def eval_springs(x, v, indices, rest, ke, kd, f):
    i = indices[:, 0]
    j = indices[:, 1]

    xi = x[i]
    xj = x[j]

    vi = v[i]
    vj = v[j]

    xij = xi - xj
    vij = vi - vj

    l = paddle.linalg.norm(xij, axis=1)
    l_inv = 1.0 / l

    # normalized spring direction
    dir = (xij.T * l_inv).T

    c = l - rest
    dcdt = paddle.sum(dir * vij, axis=1)

    # damping based on relative velocity.
    fs = dir.T * (ke * c + kd * dcdt)

    f.index_add_(axis=0, index=i, value=-fs.T)
    f.index_add_(axis=0, index=j, value=fs.T)


def integrate_particles(x, v, f, g, w, dt):
    s = w > 0.0

    a_ext = g * s[:, None].astype(g.dtype)

    # simple semi-implicit Euler. v1 = v0 + a dt, x1 = x0 + v1 dt
    v += ((f.T * w).T + a_ext) * dt
    x += v * dt

    # clear forces
    f *= 0.0


class TrIntegrator:
    def __init__(self, cloth, device):
        self.cloth = cloth

        self.positions = paddle.to_tensor(self.cloth.positions, place=device)
        self.velocities = paddle.to_tensor(self.cloth.velocities, place=device)
        self.inv_mass = paddle.to_tensor(self.cloth.inv_masses, place=device)

        self.spring_indices = paddle.to_tensor(self.cloth.spring_indices, dtype=paddle.int64, place=device)
        self.spring_lengths = paddle.to_tensor(self.cloth.spring_lengths, place=device)
        self.spring_stiffness = paddle.to_tensor(self.cloth.spring_stiffness, place=device)
        self.spring_damping = paddle.to_tensor(self.cloth.spring_damping, place=device)

        self.forces = paddle.zeros((self.cloth.num_particles, 3), dtype=paddle.float32).to(device=device)
        self.gravity = paddle.to_tensor((0.0, 0.0 - 9.8, 0.0), dtype=paddle.float32, place=device)

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
