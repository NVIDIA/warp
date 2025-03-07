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

###########################################################################
# Example NanoVDB
#
# Shows how to implement a particle simulation with collision against
# a NanoVDB signed-distance field. In this example the NanoVDB field
# is created offline in Houdini. The particle kernel uses the Warp
# wp.volume_sample_f() method to compute the SDF and normal at a point.
#
###########################################################################

import os

import numpy as np

import warp as wp
import warp.examples
import warp.render


@wp.func
def volume_grad(volume: wp.uint64, p: wp.vec3):
    eps = 1.0
    q = wp.volume_world_to_index(volume, p)

    # compute gradient of the SDF using finite differences
    dx = wp.volume_sample_f(volume, q + wp.vec3(eps, 0.0, 0.0), wp.Volume.LINEAR) - wp.volume_sample_f(
        volume, q - wp.vec3(eps, 0.0, 0.0), wp.Volume.LINEAR
    )
    dy = wp.volume_sample_f(volume, q + wp.vec3(0.0, eps, 0.0), wp.Volume.LINEAR) - wp.volume_sample_f(
        volume, q - wp.vec3(0.0, eps, 0.0), wp.Volume.LINEAR
    )
    dz = wp.volume_sample_f(volume, q + wp.vec3(0.0, 0.0, eps), wp.Volume.LINEAR) - wp.volume_sample_f(
        volume, q - wp.vec3(0.0, 0.0, eps), wp.Volume.LINEAR
    )

    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.kernel
def simulate(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    volume: wp.uint64,
    margin: float,
    dt: float,
):
    tid = wp.tid()

    x = positions[tid]
    v = velocities[tid]

    v = v + wp.vec3(0.0, -9.8, 0.0) * dt - v * 0.1 * dt
    xpred = x + v * dt
    xpred_local = wp.volume_world_to_index(volume, xpred)

    # d = wp.volume_sample_f(volume, xpred_local, wp.Volume.LINEAR)
    n = wp.vec3()
    d = wp.volume_sample_grad_f(volume, xpred_local, wp.Volume.LINEAR, n)

    if d < margin:
        # n = volume_grad(volume, xpred)
        n = wp.normalize(n)
        err = d - margin

        # mesh collision
        xpred = xpred - n * err

    # ground collision
    if xpred[1] < 0.0:
        xpred = wp.vec3(xpred[0], 0.0, xpred[2])

    # pbd update
    v = (xpred - x) * (1.0 / dt)
    x = xpred

    positions[tid] = x
    velocities[tid] = v


class Example:
    def __init__(self, stage_path="example_nvdb.usd"):
        rng = np.random.default_rng(42)
        self.num_particles = 10000

        fps = 60
        frame_dt = 1.0 / fps
        self.sim_substeps = 3
        self.sim_dt = frame_dt / self.sim_substeps

        self.sim_time = 0.0
        self.sim_timers = {}

        self.sim_margin = 0.15

        init_pos = 10.0 * (rng.random((self.num_particles, 3)) * 2.0 - 1.0) + np.array((0.0, 30.0, 0.0))
        init_vel = rng.random((self.num_particles, 3))

        self.positions = wp.from_numpy(init_pos.astype(np.float32), dtype=wp.vec3)
        self.velocities = wp.from_numpy(init_vel.astype(np.float32), dtype=wp.vec3)

        # load collision volume
        with open(os.path.join(warp.examples.get_asset_directory(), "rocks.nvdb"), "rb") as file:
            # create Volume object
            self.volume = wp.Volume.load_from_nvdb(file)

        # renderer
        self.renderer = None
        if stage_path:
            self.renderer = wp.render.UsdRenderer(stage_path)
            self.renderer.render_ground(size=100.0)

    def step(self):
        with wp.ScopedTimer("step", dict=self.sim_timers):
            for _ in range(self.sim_substeps):
                wp.launch(
                    kernel=simulate,
                    dim=self.num_particles,
                    inputs=[self.positions, self.velocities, self.volume.id, self.sim_margin, self.sim_dt],
                )
                self.sim_time += self.sim_dt

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            self.renderer.begin_frame(self.sim_time)

            self.renderer.render_ref(
                name="collision",
                path=os.path.join(warp.examples.get_asset_directory(), "rocks.usd"),
                pos=wp.vec3(0.0, 0.0, 0.0),
                rot=wp.quat(0.0, 0.0, 0.0, 1.0),
                scale=wp.vec3(1.0, 1.0, 1.0),
                color=(0.35, 0.55, 0.9),
            )
            self.renderer.render_points(
                name="points", points=self.positions.numpy(), radius=self.sim_margin, colors=(0.8, 0.3, 0.2)
            )

            self.renderer.end_frame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="example_nvdb.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_frames", type=int, default=1000, help="Total number of frames.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path)

        for _ in range(args.num_frames):
            example.step()
            example.render()

        if example.renderer:
            example.renderer.save()
