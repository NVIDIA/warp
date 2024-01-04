# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

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
import warp.sim
import warp.sim.render

wp.init()


class Example:
    def __init__(self, stage):
        self.device = wp.get_device()
        builder = wp.sim.ModelBuilder()

        self.sim_time = 0.0
        self.frame_dt = 1.0 / 60.0

        episode_duration = 20.0  # seconds
        self.episode_frames = int(episode_duration / self.frame_dt)

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
        bunny = self.load_mesh(os.path.join(os.path.dirname(__file__), "assets/bunny.usd"), "/bunny/bunny")
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

        self.renderer = wp.sim.render.SimRenderer(self.model, stage, scaling=0.5)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        wp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state_0)

        self.use_graph = wp.get_device().is_cuda
        self.graph = None

        if self.use_graph:
            # create update graph
            wp.capture_begin(self.device)
            try:
                self.update()
            finally:
                self.graph = wp.capture_end(self.device)

    def load_mesh(self, filename, path):
        asset_stage = Usd.Stage.Open(filename)
        mesh_geom = UsdGeom.Mesh(asset_stage.GetPrimAtPath(path))

        points = np.array(mesh_geom.GetPointsAttr().Get())
        indices = np.array(mesh_geom.GetFaceVertexIndicesAttr().Get()).flatten()

        return wp.sim.Mesh(points, indices)

    def update(self):
        with wp.ScopedTimer("simulate", active=True):
            if not self.use_graph or self.graph is None:
                for _ in range(self.sim_substeps):
                    self.state_0.clear_forces()
                    wp.sim.collide(self.model, self.state_0)
                    self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)
                    self.state_0, self.state_1 = self.state_1, self.state_0
            else:
                wp.capture_launch(self.graph)

            if not wp.get_device().is_capturing:
                self.sim_time += self.frame_dt

    def render(self, is_live=False):
        with wp.ScopedTimer("render", active=True):
            time = 0.0 if is_live else self.sim_time

            self.renderer.begin_frame(time)
            self.renderer.render(self.state_0)
            self.renderer.end_frame()


if __name__ == "__main__":
    stage = os.path.join(os.path.dirname(__file__), "outputs/example_sim_rigid_contact.usd")
    example = Example(stage)
    example.run()

    profiler = {}

    with wp.ScopedTimer("simulate", detailed=False, print=False, active=True, dict=profiler):
        for _ in range(example.episode_frames):
            example.update()
            example.render()

        wp.synchronize()

    example.renderer.save()
