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

import warp as wp

wp.clear_kernel_cache()


@wp.kernel
def eval_springs(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    spring_indices: wp.array(dtype=int),
    spring_rest_lengths: wp.array(dtype=float),
    spring_stiffness: wp.array(dtype=float),
    spring_damping: wp.array(dtype=float),
    f: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    i = spring_indices[tid * 2 + 0]
    j = spring_indices[tid * 2 + 1]

    ke = spring_stiffness[tid]
    kd = spring_damping[tid]
    rest = spring_rest_lengths[tid]

    xi = x[i]
    xj = x[j]

    vi = v[i]
    vj = v[j]

    xij = xi - xj
    vij = vi - vj

    l = wp.length(xij)
    l_inv = 1.0 / l

    # normalized spring direction
    dir = xij * l_inv

    c = l - rest
    dcdt = wp.dot(dir, vij)

    # damping based on relative velocity.
    fs = dir * (ke * c + kd * dcdt)

    wp.atomic_sub(f, i, fs)
    wp.atomic_add(f, j, fs)


@wp.kernel
def integrate_particles(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    f: wp.array(dtype=wp.vec3),
    w: wp.array(dtype=float),
    dt: float,
):
    tid = wp.tid()

    x0 = x[tid]
    v0 = v[tid]
    f0 = f[tid]
    inv_mass = w[tid]

    g = wp.vec3()

    # treat particles with inv_mass == 0 as kinematic
    if inv_mass > 0.0:
        g = wp.vec3(0.0, 0.0 - 9.81, 0.0)

    # simple semi-implicit Euler. v1 = v0 + a dt, x1 = x0 + v1 dt
    v1 = v0 + (f0 * inv_mass + g) * dt
    x1 = x0 + v1 * dt

    x[tid] = x1
    v[tid] = v1

    # clear forces
    f[tid] = wp.vec3()


class WpIntegrator:
    def __init__(self, cloth, device):
        self.device = wp.get_device(device)

        with wp.ScopedDevice(self.device):
            self.positions = wp.from_numpy(cloth.positions, dtype=wp.vec3)
            self.positions_host = wp.from_numpy(cloth.positions, dtype=wp.vec3, device="cpu")
            self.invmass = wp.from_numpy(cloth.inv_masses, dtype=float)

            self.velocities = wp.zeros(cloth.num_particles, dtype=wp.vec3)
            self.forces = wp.zeros(cloth.num_particles, dtype=wp.vec3)

            self.spring_indices = wp.from_numpy(cloth.spring_indices, dtype=int)
            self.spring_lengths = wp.from_numpy(cloth.spring_lengths, dtype=float)
            self.spring_stiffness = wp.from_numpy(cloth.spring_stiffness, dtype=float)
            self.spring_damping = wp.from_numpy(cloth.spring_damping, dtype=float)

        self.cloth = cloth

    def simulate(self, dt, substeps):
        sim_dt = dt / substeps

        for _s in range(substeps):
            wp.launch(
                kernel=eval_springs,
                dim=self.cloth.num_springs,
                inputs=[
                    self.positions,
                    self.velocities,
                    self.spring_indices,
                    self.spring_lengths,
                    self.spring_stiffness,
                    self.spring_damping,
                    self.forces,
                ],
                outputs=[],
                device=self.device,
            )

            # integrate
            wp.launch(
                kernel=integrate_particles,
                dim=self.cloth.num_particles,
                inputs=[self.positions, self.velocities, self.forces, self.invmass, sim_dt],
                outputs=[],
                device=self.device,
            )

        # copy data back to host
        if self.device.is_cuda:
            wp.copy(self.positions_host, self.positions)
            wp.synchronize()

            return self.positions_host.numpy()

        else:
            return self.positions.numpy()
