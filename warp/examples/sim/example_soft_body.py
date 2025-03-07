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
# Example Sim Neo-Hookean
#
# Shows a simulation of an Neo-Hookean FEM beam being twisted through a
# 180 degree rotation.
#
###########################################################################
import math

import warp as wp
import warp.sim
import warp.sim.render


@wp.kernel
def twist_points(
    rest: wp.array(dtype=wp.vec3), points: wp.array(dtype=wp.vec3), mass: wp.array(dtype=float), xform: wp.transform
):
    tid = wp.tid()

    r = rest[tid]
    p = points[tid]
    m = mass[tid]

    # twist the top layer of particles in the beam
    if m == 0 and p[1] != 0.0:
        points[tid] = wp.transform_point(xform, r)


@wp.kernel
def compute_volume(points: wp.array(dtype=wp.vec3), indices: wp.array2d(dtype=int), volume: wp.array(dtype=float)):
    tid = wp.tid()

    i = indices[tid, 0]
    j = indices[tid, 1]
    k = indices[tid, 2]
    l = indices[tid, 3]

    x0 = points[i]
    x1 = points[j]
    x2 = points[k]
    x3 = points[l]

    x10 = x1 - x0
    x20 = x2 - x0
    x30 = x3 - x0

    v = wp.dot(x10, wp.cross(x20, x30)) / 6.0

    wp.atomic_add(volume, 0, v)


class Example:
    def __init__(self, stage_path="example_soft_body.usd", num_frames=300):
        self.sim_substeps = 64
        self.num_frames = num_frames
        fps = 60
        sim_duration = self.num_frames / fps
        self.frame_dt = 1.0 / fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.lift_speed = 2.5 / sim_duration * 2.0  # from Smith et al.
        self.rot_speed = math.pi / sim_duration

        builder = wp.sim.ModelBuilder()

        cell_dim = 15
        cell_size = 2.0 / cell_dim

        center = cell_size * cell_dim * 0.5

        builder.add_soft_grid(
            pos=wp.vec3(-center, 0.0, -center),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            dim_x=cell_dim,
            dim_y=cell_dim,
            dim_z=cell_dim,
            cell_x=cell_size,
            cell_y=cell_size,
            cell_z=cell_size,
            density=100.0,
            fix_bottom=True,
            fix_top=True,
            k_mu=1000.0,
            k_lambda=5000.0,
            k_damp=0.0,
        )

        self.model = builder.finalize()
        self.model.ground = False
        self.model.gravity[1] = 0.0

        self.integrator = wp.sim.SemiImplicitIntegrator()

        self.rest = self.model.state()
        self.rest_vol = (cell_size * cell_dim) ** 3

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.volume = wp.zeros(1, dtype=wp.float32)

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
            self.state_1.clear_forces()

            self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)

            # swap states
            (self.state_0, self.state_1) = (self.state_1, self.state_0)

    def step(self):
        with wp.ScopedTimer("step"):
            xform = wp.transform(
                (0.0, self.lift_speed * self.sim_time, 0.0),
                wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), self.rot_speed * self.sim_time),
            )
            wp.launch(
                kernel=twist_points,
                dim=len(self.state_0.particle_q),
                inputs=[self.rest.particle_q, self.state_0.particle_q, self.model.particle_mass, xform],
            )
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()
            self.volume.zero_()
            wp.launch(
                kernel=compute_volume,
                dim=self.model.tet_count,
                inputs=[self.state_0.particle_q, self.model.tet_indices, self.volume],
            )
        self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render(self.state_0)
            self.renderer.end_frame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="example_soft_body.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_frames", type=int, default=300, help="Total number of frames.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, num_frames=args.num_frames)

        for _ in range(args.num_frames):
            example.step()
            example.render()

        if example.renderer:
            example.renderer.save()
