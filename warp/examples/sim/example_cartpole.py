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
# Example Sim Cartpole
#
# Shows how to set up a simulation of a rigid-body cartpole articulation
# from a URDF using the wp.sim.ModelBuilder().
# Note this example does not include a trained policy.
#
###########################################################################

import math
import os

import numpy as np

import warp as wp
import warp.examples
import warp.sim
import warp.sim.render


class Example:
    def __init__(self, stage_path="example_cartpole.usd", num_envs=8):
        self.num_envs = num_envs

        articulation_builder = wp.sim.ModelBuilder()

        wp.sim.parse_urdf(
            os.path.join(warp.examples.get_asset_directory(), "cartpole.urdf"),
            articulation_builder,
            xform=wp.transform(wp.vec3(), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi * 0.5)),
            floating=False,
            density=100,
            armature=0.1,
            stiffness=0.0,
            damping=0.0,
            limit_ke=1.0e4,
            limit_kd=1.0e1,
            enable_self_collisions=False,
        )

        builder = wp.sim.ModelBuilder()

        self.sim_time = 0.0
        fps = 60
        self.frame_dt = 1.0 / fps

        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        for i in range(self.num_envs):
            builder.add_builder(
                articulation_builder, xform=wp.transform(np.array((i * 2.0, 4.0, 0.0)), wp.quat_identity())
            )

            # joint initial positions
            builder.joint_q[-3:] = [0.0, 0.3, 0.0]

        # finalize model
        self.model = builder.finalize()
        self.model.ground = False

        self.model.joint_attach_ke = 1600.0
        self.model.joint_attach_kd = 20.0

        self.integrator = wp.sim.SemiImplicitIntegrator()

        self.renderer = None
        if stage_path:
            self.renderer = wp.sim.render.SimRenderer(path=stage_path, model=self.model, scaling=15.0)

        self.state = self.model.state()

        wp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state)

        self.use_cuda_graph = wp.get_device().is_cuda
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state.clear_forces()
            self.state = self.integrator.simulate(self.model, self.state, self.state, self.sim_dt)

    def step(self):
        with wp.ScopedTimer("step"):
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
            self.renderer.render(self.state)
            self.renderer.end_frame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="example_cartpole.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_frames", type=int, default=1200, help="Total number of frames.")
    parser.add_argument("--num_envs", type=int, default=8, help="Total number of simulated environments.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, num_envs=args.num_envs)

        for _ in range(args.num_frames):
            example.step()
            example.render()

        if example.renderer:
            example.renderer.save()
