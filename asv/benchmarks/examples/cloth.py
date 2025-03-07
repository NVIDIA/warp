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

import numpy as np

import warp as wp


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


class Cloth:
    number = 400
    params = [32, 128]
    param_names = ["res"]

    def setup(self, res):
        wp.init()
        wp.build.clear_kernel_cache()
        self.device = wp.get_device("cuda:0")
        wp.load_module(device=self.device)

        lower = (0.0, 0.0, 0.0)
        dx = res
        dy = res
        radius = 0.1
        stretch_stiffness = 1000.0
        bend_stiffness = 1000.0
        shear_stiffness = 1000.0
        mass = 0.1
        fix_corners = True

        self.positions = []
        self.velocities = []
        self.inv_masses = []

        self.spring_indices = []
        self.spring_lengths = []
        self.spring_stiffness = []
        self.spring_damping = []

        def grid(x, y, stride):
            return y * stride + x

        def create_spring(i, j, stiffness, damp=10.0):
            length = np.linalg.norm(np.array(self.positions[i]) - np.array(self.positions[j]))

            self.spring_indices.append(i)
            self.spring_indices.append(j)
            self.spring_lengths.append(length)
            self.spring_stiffness.append(stiffness)
            self.spring_damping.append(damp)

        for y in range(dy):
            for x in range(dx):
                p = np.array(lower) + radius * np.array((float(x), float(0.0), float(y)))

                self.positions.append(p)
                self.velocities.append(np.zeros(3))

                if fix_corners and y == 0 and (x == 0 or x == dx - 1):
                    w = 0.0
                else:
                    w = 1.0 / mass

                self.inv_masses.append(w)

        # horizontal springs
        for y in range(dy):
            for x in range(dx):
                index0 = y * dx + x

                if x > 0:
                    index1 = y * dx + x - 1
                    create_spring(index0, index1, stretch_stiffness)

                if x > 1 and bend_stiffness > 0.0:
                    index2 = y * dx + x - 2
                    create_spring(index0, index2, bend_stiffness)

                if y > 0 and x < dx - 1 and shear_stiffness > 0.0:
                    indexDiag = (y - 1) * dx + x + 1
                    create_spring(index0, indexDiag, shear_stiffness)

                if y > 0 and x > 0 and shear_stiffness > 0.0:
                    indexDiag = (y - 1) * dx + x - 1
                    create_spring(index0, indexDiag, shear_stiffness)

        # vertical
        for x in range(dx):
            for y in range(dy):
                index0 = y * dx + x

                if y > 0:
                    index1 = (y - 1) * dx + x
                    create_spring(index0, index1, stretch_stiffness)

                if y > 1 and bend_stiffness > 0.0:
                    index2 = (y - 2) * dx + x
                    create_spring(index0, index2, bend_stiffness)

        # harden to np arrays
        self.positions = np.array(self.positions, dtype=np.float32)
        self.velocities = np.array(self.velocities, dtype=np.float32)
        self.inv_masses = np.array(self.inv_masses, dtype=np.float32)
        self.spring_indices = np.array(self.spring_indices, dtype=np.int32)
        self.spring_lengths = np.array(self.spring_lengths, dtype=np.float32)
        self.spring_stiffness = np.array(self.spring_stiffness, dtype=np.float32)
        self.spring_damping = np.array(self.spring_damping, dtype=np.float32)

        self.num_particles = len(self.positions)
        self.num_springs = len(self.spring_lengths)

        self.positions_wp = wp.from_numpy(self.positions, dtype=wp.vec3, device=self.device)
        self.velocities_wp = wp.zeros(self.num_particles, dtype=wp.vec3, device=self.device)
        self.invmass_wp = wp.from_numpy(self.inv_masses, dtype=float, device=self.device)
        self.spring_indices_wp = wp.from_numpy(self.spring_indices, dtype=int, device=self.device)
        self.spring_lengths_wp = wp.from_numpy(self.spring_lengths, dtype=float, device=self.device)
        self.spring_stiffness_wp = wp.from_numpy(self.spring_stiffness, dtype=float, device=self.device)
        self.spring_damping_wp = wp.from_numpy(self.spring_damping, dtype=float, device=self.device)
        self.forces_wp = wp.zeros(self.num_particles, dtype=wp.vec3, device=self.device)

        sim_fps = 60.0
        sim_substeps = 16
        sim_dt = (1.0 / sim_fps) / sim_substeps

        with wp.ScopedCapture() as capture:
            for _s in range(sim_substeps):
                wp.launch(
                    kernel=eval_springs,
                    dim=self.num_springs,
                    inputs=[
                        self.positions_wp,
                        self.velocities_wp,
                        self.spring_indices_wp,
                        self.spring_lengths_wp,
                        self.spring_stiffness_wp,
                        self.spring_damping_wp,
                        self.forces_wp,
                    ],
                    device=self.device,
                )

                # integrate
                wp.launch(
                    kernel=integrate_particles,
                    dim=self.num_particles,
                    inputs=[self.positions_wp, self.velocities_wp, self.forces_wp, self.invmass_wp, sim_dt],
                    device=self.device,
                )

        self.graph = capture.graph

        for _warmup in range(5):
            wp.capture_launch(self.graph)

        wp.synchronize_device(self.device)

    def time_simulate(self, res):
        wp.capture_launch(self.graph)
        wp.synchronize_device(self.device)
