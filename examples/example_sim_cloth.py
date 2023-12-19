# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Cloth
#
# Shows a simulation of an FEM cloth model colliding against a static
# rigid body mesh using the wp.sim.ModelBuilder().
#
###########################################################################

import argparse
import math
import os
from enum import Enum

import numpy as np
from pxr import Usd, UsdGeom

import warp as wp
import warp.sim
import warp.sim.render

wp.init()


class IntegratorType(Enum):
    EULER = "euler"
    XPBD = "xpbd"

    def __str__(self):
        return self.value


class Example:
    def __init__(self, stage, integrator=IntegratorType.EULER):
        self.device = wp.get_device()

        self.integrator_type = integrator

        self.sim_width = 64
        self.sim_height = 32

        self.sim_fps = 60.0
        self.sim_substeps = 32
        self.sim_duration = 5.0
        self.sim_frames = int(self.sim_duration * self.sim_fps)
        self.frame_dt = 1.0 / self.sim_fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.sim_use_graph = wp.get_device().is_cuda
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
        else:
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

        usd_stage = Usd.Stage.Open(os.path.join(os.path.dirname(__file__), "assets/bunny.usd"))
        usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/bunny/bunny"))

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

        self.model = builder.finalize()
        self.model.ground = True
        self.model.soft_contact_ke = 1.0e4
        self.model.soft_contact_kd = 1.0e2

        if self.integrator_type == IntegratorType.EULER:
            self.integrator = wp.sim.SemiImplicitIntegrator()
        else:
            self.integrator = wp.sim.XPBDIntegrator(iterations=1)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.renderer = wp.sim.render.SimRenderer(self.model, stage, scaling=40.0)
        self.graph = None

        if self.sim_use_graph:
            # create update graph
            wp.capture_begin(self.device)
            try:
                self.update()
            finally:
                self.graph = wp.capture_end(self.device)

    def update(self):
        with wp.ScopedTimer("simulate", dict=self.profiler):
            if self.sim_use_graph and self.graph:
                wp.capture_launch(self.graph)
                self.sim_time += self.frame_dt
            else:
                wp.sim.collide(self.model, self.state_0)

                for _ in range(self.sim_substeps):
                    self.state_0.clear_forces()

                    self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)

                    if not wp.get_device().is_capturing:
                        self.sim_time += self.sim_dt

                    # swap states
                    (self.state_0, self.state_1) = (self.state_1, self.state_0)

    def render(self, is_live=False):
        with wp.ScopedTimer("render", active=True):
            time = 0.0 if is_live else self.sim_time

            self.renderer.begin_frame(time)
            self.renderer.render(self.state_0)
            self.renderer.end_frame()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--integrator",
        help="Type of integrator",
        type=IntegratorType,
        choices=list(IntegratorType),
        default=IntegratorType.EULER,
    )

    args = parser.parse_args()

    stage_path = os.path.join(os.path.dirname(__file__), "outputs/example_sim_cloth.usd")

    example = Example(stage_path, integrator=args.integrator)

    for i in range(example.sim_frames):
        example.update()
        example.render()

    frame_times = example.profiler["simulate"]
    print("\nAverage frame sim time: {:.2f} ms".format(sum(frame_times) / len(frame_times)))

    example.renderer.save()
