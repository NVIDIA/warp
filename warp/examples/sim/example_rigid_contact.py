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
# Example Sim Rigid Contact
#
# Shows how to set up free rigid bodies with different shape types falling
# and colliding against each other and the ground using wp.sim.ModelBuilder().
#
###########################################################################

import math
import os

import numpy as np
from pxr import Usd, UsdGeom

import warp as wp
import warp.examples
import warp.sim
import warp.sim.render


class Example:
    def __init__(self, stage_path="example_rigid_contact.usd"):
        builder = wp.sim.ModelBuilder()

        self.sim_time = 0.0
        fps = 60
        self.frame_dt = 1.0 / fps

        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.num_bodies = 8
        self.scale = 0.8
        self.ke = 1.0e5
        self.kd = 250.0
        self.kf = 500.0

        # boxes
        for i in range(self.num_bodies):
            b = builder.add_body(origin=wp.transform((i, 1.0, 0.0), wp.quat_identity()))

            builder.add_shape_box(
                pos=wp.vec3(0.0, 0.0, 0.0),
                hx=0.5 * self.scale,
                hy=0.2 * self.scale,
                hz=0.2 * self.scale,
                body=i,
                ke=self.ke,
                kd=self.kd,
                kf=self.kf,
            )

        # spheres
        for i in range(self.num_bodies):
            b = builder.add_body(origin=wp.transform((i, 1.0, 2.0), wp.quat_identity()))

            builder.add_shape_sphere(
                pos=wp.vec3(0.0, 0.0, 0.0), radius=0.25 * self.scale, body=b, ke=self.ke, kd=self.kd, kf=self.kf
            )

        # capsules
        for i in range(self.num_bodies):
            b = builder.add_body(origin=wp.transform((i, 1.0, 6.0), wp.quat_identity()))

            builder.add_shape_capsule(
                pos=wp.vec3(0.0, 0.0, 0.0),
                radius=0.25 * self.scale,
                half_height=self.scale * 0.5,
                up_axis=0,
                body=b,
                ke=self.ke,
                kd=self.kd,
                kf=self.kf,
            )

        # initial spin
        for i in range(len(builder.body_qd)):
            builder.body_qd[i] = (0.0, 2.0, 10.0, 0.0, 0.0, 0.0)

        # meshes
        bunny = self.load_mesh(os.path.join(warp.examples.get_asset_directory(), "bunny.usd"), "/root/bunny")
        for i in range(self.num_bodies):
            b = builder.add_body(
                origin=wp.transform(
                    (i * 0.5 * self.scale, 1.0 + i * 1.7 * self.scale, 4.0 + i * 0.5 * self.scale),
                    wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.pi * 0.1 * i),
                )
            )

            builder.add_shape_mesh(
                body=b,
                mesh=bunny,
                pos=wp.vec3(0.0, 0.0, 0.0),
                scale=wp.vec3(self.scale, self.scale, self.scale),
                ke=self.ke,
                kd=self.kd,
                kf=self.kf,
                density=1e3,
            )

        # finalize model
        self.model = builder.finalize()
        self.model.ground = True

        self.integrator = wp.sim.SemiImplicitIntegrator()

        if stage_path:
            self.renderer = wp.sim.render.SimRenderer(self.model, stage_path, scaling=0.5)
        else:
            self.renderer = None

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        wp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state_0)

        self.use_cuda_graph = wp.get_device().is_cuda
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def load_mesh(self, filename, path):
        asset_stage = Usd.Stage.Open(filename)
        mesh_geom = UsdGeom.Mesh(asset_stage.GetPrimAtPath(path))

        points = np.array(mesh_geom.GetPointsAttr().Get())
        indices = np.array(mesh_geom.GetFaceVertexIndicesAttr().Get()).flatten()

        return wp.sim.Mesh(points, indices)

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            wp.sim.collide(self.model, self.state_0)
            self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        with wp.ScopedTimer("step", active=True):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render", active=True):
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
        default="example_rigid_contact.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_frames", type=int, default=300, help="Total number of frames.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path)

        for _ in range(args.num_frames):
            example.step()
            example.render()

        if example.renderer:
            example.renderer.save()
