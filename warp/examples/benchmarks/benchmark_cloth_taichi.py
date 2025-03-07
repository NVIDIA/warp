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

import numpy as np
import taichi as ti


@ti.func
def step(x):
    ret = 0.0
    if x < 0:
        ret = 1
    return ret


@ti.data_oriented
class TiIntegrator:
    @ti.kernel
    def eval_springs(self):
        for tid in range(self.cloth.num_springs):
            i = self.spring_indices[2 * tid]
            j = self.spring_indices[2 * tid + 1]

            ke = self.spring_stiffness[tid]
            kd = self.spring_damping[tid]
            rest = self.spring_lengths[tid]

            xi = self.positions[i]
            xj = self.positions[j]

            vi = self.velocities[i]
            vj = self.velocities[j]

            xij = xi - xj
            vij = vi - vj

            l = xij.norm()
            dir = xij.normalized()

            c = l - rest
            dcdt = dir.dot(vij)

            fs = dir * (ke * c + kd * dcdt)

            self.forces[i] -= fs
            self.forces[j] += fs

    @ti.kernel
    def integrate_particles(self, dt: ti.f32):
        for tid in range(self.cloth.num_particles):
            x0 = self.positions[tid]
            v0 = self.velocities[tid]
            f0 = self.forces[tid]
            w = self.inv_mass[tid]

            g = ti.Vector([0.0, 0.0, 0.0])

            if w > 0.0:
                g = ti.Vector([0.0, -9.81, 0.0])

            v1 = v0 + (f0 * w + g) * dt
            x1 = x0 + v1 * dt

            self.positions[tid] = x1
            self.velocities[tid] = v1
            self.forces[tid] = ti.Vector([0.0, 0.0, 0.0])

    def __init__(self, cloth, device):
        if device == "cpu":
            ti.init(arch=ti.cpu)
        elif device == "cuda":
            ti.init(arch=ti.gpu)
        else:
            raise RuntimeError("Unsupported Taichi device")

        self.cloth = cloth

        self.positions = ti.Vector.field(3, dtype=ti.f32, shape=self.cloth.num_particles)
        self.velocities = ti.Vector.field(3, dtype=ti.f32, shape=self.cloth.num_particles)
        self.inv_mass = ti.field(ti.f32, shape=self.cloth.num_particles)

        self.spring_indices = ti.field(ti.i32, shape=self.cloth.num_springs * 2)
        self.spring_lengths = ti.field(ti.f32, shape=self.cloth.num_springs)
        self.spring_stiffness = ti.field(ti.f32, shape=self.cloth.num_springs)
        self.spring_damping = ti.field(ti.f32, shape=self.cloth.num_springs)

        self.forces = ti.Vector.field(3, dtype=ti.f32, shape=self.cloth.num_particles)

        # upload data
        self.positions.from_numpy(cloth.positions)
        self.velocities.from_numpy(cloth.velocities)
        self.inv_mass.from_numpy(cloth.inv_masses)
        self.forces.from_numpy(np.zeros_like(self.cloth.velocities))

        self.spring_indices.from_numpy(cloth.spring_indices)
        self.spring_lengths.from_numpy(cloth.spring_lengths)
        self.spring_stiffness.from_numpy(cloth.spring_stiffness)
        self.spring_damping.from_numpy(cloth.spring_damping)

    def simulate(self, dt, substeps):
        sim_dt = dt / substeps

        for _s in range(substeps):
            self.eval_springs()

            self.integrate_particles(sim_dt)

        return self.positions.to_numpy()
