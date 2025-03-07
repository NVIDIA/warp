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
# Example Sim Cloth
#
# Shows a simulation of an FEM cloth model colliding against a static
# rigid body mesh using the wp.sim.ModelBuilder().
#
###########################################################################

import math
import os
from enum import Enum

import numpy as np
from pxr import Usd, UsdGeom

import warp as wp
import warp.examples
import warp.sim
import warp.sim.render


class IntegratorType(Enum):
    EULER = "euler"
    XPBD = "xpbd"
    VBD = "vbd"

    def __str__(self):
        return self.value


class Example:
    def __init__(
        self, stage_path="example_cloth.usd", integrator: IntegratorType = IntegratorType.EULER, height=32, width=64
    ):
        self.integrator_type = integrator

        self.sim_height = height
        self.sim_width = width

        fps = 60
        self.sim_substeps = 32
        self.frame_dt = 1.0 / fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.profiler = {}

        builder = wp.sim.ModelBuilder()

        if self.integrator_type == IntegratorType.EULER:
            builder.add_cloth_grid(
                pos=wp.vec3(0.0, 4.0, 0.0),
                rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi * 0.5),
                vel=wp.vec3(0.0, 0.0, 0.0),
                dim_x=self.sim_width,
                dim_y=self.sim_height,
                cell_x=0.1,
                cell_y=0.1,
                mass=0.1,
                fix_left=True,
                tri_ke=1.0e3,
                tri_ka=1.0e3,
                tri_kd=1.0e1,
            )
        elif self.integrator_type == IntegratorType.XPBD:
            builder.add_cloth_grid(
                pos=wp.vec3(0.0, 4.0, 0.0),
                rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi * 0.5),
                vel=wp.vec3(0.0, 0.0, 0.0),
                dim_x=self.sim_width,
                dim_y=self.sim_height,
                cell_x=0.1,
                cell_y=0.1,
                mass=0.1,
                fix_left=True,
                edge_ke=1.0e2,
                add_springs=True,
                spring_ke=1.0e3,
                spring_kd=0.0,
            )
        else:
            # VBD
            builder.add_cloth_grid(
                pos=wp.vec3(0.0, 4.0, 0.0),
                rot=wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi * 0.5),
                vel=wp.vec3(0.0, 0.0, 0.0),
                dim_x=self.sim_width,
                dim_y=self.sim_height,
                cell_x=0.1,
                cell_y=0.1,
                mass=0.1,
                fix_left=True,
                tri_ke=1e4,
                tri_ka=1e4,
                tri_kd=1e-5,
                edge_ke=100,
            )

        usd_stage = Usd.Stage.Open(os.path.join(warp.examples.get_asset_directory(), "bunny.usd"))
        usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/root/bunny"))

        mesh_points = np.array(usd_geom.GetPointsAttr().Get())
        mesh_indices = np.array(usd_geom.GetFaceVertexIndicesAttr().Get())

        mesh = wp.sim.Mesh(mesh_points, mesh_indices)

        builder.add_shape_mesh(
            body=-1,
            mesh=mesh,
            pos=wp.vec3(1.0, 0.0, 1.0),
            rot=wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.pi * 0.5),
            scale=wp.vec3(2.0, 2.0, 2.0),
            ke=1.0e2,
            kd=1.0e2,
            kf=1.0e1,
        )

        if self.integrator_type == IntegratorType.VBD:
            builder.color()

        self.model = builder.finalize()
        self.model.ground = True
        self.model.soft_contact_ke = 1.0e4
        self.model.soft_contact_kd = 1.0e2

        if self.integrator_type == IntegratorType.EULER:
            self.integrator = wp.sim.SemiImplicitIntegrator()
        elif self.integrator_type == IntegratorType.XPBD:
            self.integrator = wp.sim.XPBDIntegrator(iterations=1)
        else:
            self.integrator = wp.sim.VBDIntegrator(self.model, iterations=1)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        if stage_path:
            self.renderer = wp.sim.render.SimRenderer(self.model, stage_path, scaling=40.0)
        else:
            self.renderer = None

        self.use_cuda_graph = wp.get_device().is_cuda
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        wp.sim.collide(self.model, self.state_0)

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)

            # swap states
            (self.state_0, self.state_1) = (self.state_1, self.state_0)

    def step(self):
        with wp.ScopedTimer("step", dict=self.profiler):
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
            self.renderer.render(self.state_0)
            self.renderer.end_frame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="example_cloth.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_frames", type=int, default=300, help="Total number of frames.")
    parser.add_argument(
        "--integrator",
        help="Type of integrator",
        type=IntegratorType,
        choices=list(IntegratorType),
        default=IntegratorType.EULER,
    )
    parser.add_argument("--width", type=int, default=64, help="Cloth resolution in x.")
    parser.add_argument("--height", type=int, default=32, help="Cloth resolution in y.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, integrator=args.integrator, height=args.height, width=args.width)

        for _i in range(args.num_frames):
            example.step()
            example.render()

        frame_times = example.profiler["step"]
        print("\nAverage frame sim time: {:.2f} ms".format(sum(frame_times) / len(frame_times)))

        if example.renderer:
            example.renderer.save()
