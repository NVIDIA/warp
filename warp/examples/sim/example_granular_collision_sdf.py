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
# Example Sim Granular Collision SDF
#
# Shows how to set up a particle-based granular material model using the
# wp.sim.ModelBuilder(). This version shows how to create collision geometry
# objects from SDFs.
#
# Note: requires a CUDA-capable device
###########################################################################

import os

import numpy as np

import warp as wp
import warp.examples
import warp.sim
import warp.sim.render


class Example:
    def __init__(self, stage_path="example_granular_collision_sdf.usd"):
        fps = 60
        self.frame_dt = 1.0 / fps

        self.sim_substeps = 64
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.radius = 0.1

        builder = wp.sim.ModelBuilder()
        builder.default_particle_radius = self.radius

        builder.add_particle_grid(
            dim_x=16,
            dim_y=32,
            dim_z=16,
            cell_x=self.radius * 2.0,
            cell_y=self.radius * 2.0,
            cell_z=self.radius * 2.0,
            pos=wp.vec3(0.0, 20.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(2.0, 0.0, 0.0),
            mass=0.1,
            jitter=self.radius * 0.1,
        )
        with open(os.path.join(warp.examples.get_asset_directory(), "rocks.nvdb"), "rb") as rock_file:
            rock_vdb = wp.Volume.load_from_nvdb(rock_file.read())

        rock_sdf = wp.sim.SDF(rock_vdb)

        builder.add_shape_sdf(
            ke=1.0e4,
            kd=1000.0,
            kf=1000.0,
            mu=0.5,
            sdf=rock_sdf,
            body=-1,
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat(0.0, 0.0, 0.0, 1.0),
            scale=wp.vec3(1.0, 1.0, 1.0),
        )

        mins = np.array([-3.0, -3.0, -3.0])
        voxel_size = 0.2
        maxs = np.array([3.0, 3.0, 3.0])
        nums = np.ceil((maxs - mins) / (voxel_size)).astype(dtype=int)
        center = np.array([0.0, 0.0, 0.0])
        rad = 2.5
        sphere_sdf_np = np.zeros(tuple(nums))
        for x in range(nums[0]):
            for y in range(nums[1]):
                for z in range(nums[2]):
                    pos = mins + voxel_size * np.array([x, y, z])
                    dis = np.linalg.norm(pos - center)
                    sphere_sdf_np[x, y, z] = dis - rad

        sphere_vdb = wp.Volume.load_from_numpy(sphere_sdf_np, mins, voxel_size, rad + 3.0 * voxel_size)
        sphere_sdf = wp.sim.SDF(sphere_vdb)

        self.sphere_pos = wp.vec3(3.0, 15.0, 0.0)
        self.sphere_scale = 1.0
        self.sphere_radius = rad
        builder.add_shape_sdf(
            ke=1.0e4,
            kd=1000.0,
            kf=1000.0,
            mu=0.5,
            sdf=sphere_sdf,
            body=-1,
            pos=self.sphere_pos,
            scale=wp.vec3(self.sphere_scale, self.sphere_scale, self.sphere_scale),
        )

        self.model = builder.finalize()
        self.model.particle_kf = 25.0

        self.model.soft_contact_kd = 100.0
        self.model.soft_contact_kf *= 2.0

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.integrator = wp.sim.SemiImplicitIntegrator()

        if stage_path:
            self.renderer = wp.sim.render.SimRenderer(self.model, stage_path, scaling=20.0)
        else:
            self.renderer = None

        self.use_cuda_graph = wp.get_device().is_cuda
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            wp.sim.collide(self.model, self.state_0)
            self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)

            # swap states
            (self.state_0, self.state_1) = (self.state_1, self.state_0)

    def step(self):
        with wp.ScopedTimer("step"):
            self.model.particle_grid.build(self.state_0.particle_q, self.radius * 2.0)
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()

        self.sim_time += self.frame_dt

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

            self.renderer.render_sphere(
                name="sphere",
                pos=self.sphere_pos,
                radius=self.sphere_scale * self.sphere_radius,
                rot=wp.quat(0.0, 0.0, 0.0, 1.0),
            )

            self.renderer.render(self.state_0)
            self.renderer.end_frame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="example_granular_collision_sdf.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_frames", type=int, default=400, help="Total number of frames.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path)

        for _ in range(args.num_frames):
            example.step()
            example.render()

        if example.renderer:
            example.renderer.save()
