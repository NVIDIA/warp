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
# Example Sim Rigid Gyroscopic
#
# Demonstrates the Dzhanibekov effect where rigid bodies will tumble in
# free space due to unstable axes of rotation.
#
###########################################################################

import warp as wp
import warp.sim
import warp.sim.render


class Example:
    def __init__(self, stage_path="example_rigid_gyroscopic.usd"):
        fps = 120
        self.sim_dt = 1.0 / fps
        self.sim_time = 0.0

        self.scale = 0.5

        builder = wp.sim.ModelBuilder()

        b = builder.add_body()

        # axis shape
        builder.add_shape_box(
            pos=wp.vec3(0.3 * self.scale, 0.0, 0.0),
            hx=0.25 * self.scale,
            hy=0.1 * self.scale,
            hz=0.1 * self.scale,
            density=100.0,
            body=b,
        )

        # tip shape
        builder.add_shape_box(
            pos=wp.vec3(0.0, 0.0, 0.0),
            hx=0.05 * self.scale,
            hy=0.2 * self.scale,
            hz=1.0 * self.scale,
            density=100.0,
            body=b,
        )

        # initial spin
        builder.body_qd[0] = (25.0, 0.01, 0.01, 0.0, 0.0, 0.0)

        builder.gravity = 0.0
        self.model = builder.finalize()
        self.model.ground = False

        self.integrator = wp.sim.SemiImplicitIntegrator()
        self.state = self.model.state()

        if stage_path:
            self.renderer = wp.sim.render.SimRenderer(self.model, stage_path, scaling=100.0)
        else:
            self.renderer = None

    def step(self):
        with wp.ScopedTimer("step"):
            self.state.clear_forces()
            self.state = self.integrator.simulate(self.model, self.state, self.state, self.sim_dt)
            self.sim_time += self.sim_dt

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render(self.state)
            self.renderer.end_frame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="example_rigid_gyroscopic.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_frames", type=int, default=2000, help="Total number of frames.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path)

        for _ in range(args.num_frames):
            example.step()
            example.render()

        if example.renderer:
            example.renderer.save()
