# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

DT = wp.constant(0.01)
SOFTENING_SQ = wp.constant(0.1**2)  # Softening factor for numerical stability
TILE_SIZE = wp.constant(64)
PARTICLE_MASS = wp.constant(1.0)


@wp.func
def body_body_interaction(p0: wp.vec3, pi: wp.vec3):
    """Return the acceleration of the particle at position `p0` due to the
    particle at position `pi`."""
    r = pi - p0

    dist_sq = wp.length_sq(r) + SOFTENING_SQ

    inv_dist = 1.0 / wp.sqrt(dist_sq)
    inv_dist_cubed = inv_dist * inv_dist * inv_dist

    acc = PARTICLE_MASS * inv_dist_cubed * r

    return acc


@wp.kernel
def integrate_bodies_tiled(
    old_position: wp.array(dtype=wp.vec3),
    velocity: wp.array(dtype=wp.vec3),
    new_position: wp.array(dtype=wp.vec3),
    num_bodies: int,
):
    i = wp.tid()

    p0 = old_position[i]

    accel = wp.vec3(0.0, 0.0, 0.0)

    for k in range(num_bodies / TILE_SIZE):
        k_tile = wp.tile_load(old_position, shape=TILE_SIZE, offset=k * TILE_SIZE)
        for idx in range(TILE_SIZE):
            pi = k_tile[idx]
            accel += body_body_interaction(p0, pi)

    # Advance the velocity one timestep (in-place)
    velocity[i] = velocity[i] + accel * DT

    # Advance the positions (using a second array)
    new_position[i] = old_position[i] + DT * velocity[i]


@wp.kernel
def integrate_bodies_simt(
    old_position: wp.array(dtype=wp.vec3),
    velocity: wp.array(dtype=wp.vec3),
    new_position: wp.array(dtype=wp.vec3),
    num_bodies: int,
):
    i = wp.tid()

    p0 = old_position[i]

    accel = wp.vec3(0.0, 0.0, 0.0)

    for idx in range(num_bodies):
        pi = old_position[idx]
        accel += body_body_interaction(p0, pi)

    # Advance the velocity one timestep (in-place)
    velocity[i] = velocity[i] + accel * DT

    # Advance the positions (using a second array)
    new_position[i] = old_position[i] + DT * velocity[i]


class TileNBody:
    number = 10  # Number of measurements to make between a single setup and teardown

    def setup(self):
        wp.init()
        wp.build.clear_kernel_cache()
        wp.set_module_options({"fast_math": True, "enable_backward": False})
        self.device = wp.get_device("cuda:0")
        wp.load_module(device=self.device)

        self.num_bodies = 65536

        rng = np.random.default_rng(42)

        # Sample the surface of a sphere
        r = 10.0 * (self.num_bodies / 1024) ** (1 / 2)  # Scale factor to maintain a constant density
        phi = np.arccos(1.0 - 2.0 * rng.uniform(size=self.num_bodies))
        theta = rng.uniform(low=0.0, high=2.0 * np.pi, size=self.num_bodies)
        x = r * np.cos(theta) * np.sin(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(phi)

        init_pos_np = np.stack((x, y, z), axis=1)

        self.pos_array_0 = wp.array(init_pos_np, dtype=wp.vec3)
        self.pos_array_1 = wp.empty_like(self.pos_array_0)
        self.vel_array = wp.zeros(self.num_bodies, dtype=wp.vec3)

        self.tile_cmd = wp.launch(
            integrate_bodies_tiled,
            dim=self.num_bodies,
            inputs=[self.pos_array_0, self.vel_array, self.pos_array_1, self.num_bodies],
            block_dim=TILE_SIZE,
            device=self.device,
            record_cmd=True,
        )

        self.simt_cmd = wp.launch(
            integrate_bodies_simt,
            dim=self.num_bodies,
            inputs=[self.pos_array_0, self.vel_array, self.pos_array_1, self.num_bodies],
            device=self.device,
            record_cmd=True,
        )

    def time_tile(self):
        self.tile_cmd.launch()
        wp.synchronize_device(self.device)

    def time_simt(self):
        self.simt_cmd.launch()
        wp.synchronize_device(self.device)
