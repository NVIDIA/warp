# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Rigid Gyroscopic
#
# Demonstrates the Dzhanibekov effect where rigid bodies will tumble in
# free space due to unstable axes of rotation.
#
###########################################################################

import os

import warp as wp
import warp.sim
import warp.sim.render

wp.init()


class Example:
    def __init__(self, stage):
        self.sim_steps = 2000
        self.sim_dt = 1.0 / 120.0
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
            pos=wp.vec3(0.0, 0.0, 0.0), hx=0.05 * self.scale, hy=0.2 * self.scale, hz=1.0 * self.scale, density=100.0, body=b
        )

        # initial spin
        builder.body_qd[0] = (25.0, 0.01, 0.01, 0.0, 0.0, 0.0)

        builder.gravity = 0.0
        self.model = builder.finalize()
        self.model.ground = False

        self.integrator = wp.sim.SemiImplicitIntegrator()
        self.state = self.model.state()

        self.renderer = wp.sim.render.SimRenderer(self.model, stage, scaling=100.0)

    def update(self):
        with wp.ScopedTimer("simulate", active=True):
            self.state.clear_forces()
            self.state = self.integrator.simulate(self.model, self.state, self.state, self.sim_dt)
            self.sim_time += self.sim_dt

    def render(self, is_live=False):
        with wp.ScopedTimer("render", active=True):
            time = 0.0 if is_live else self.sim_time

            self.renderer.begin_frame(time)
            self.renderer.render(self.state)
            self.renderer.end_frame()


if __name__ == "__main__":
    stage_path = os.path.join(os.path.dirname(__file__), "outputs/example_sim_rigid_gyroscopic.usd")

    example = Example(stage_path)

    for i in range(example.sim_steps):
        example.update()
        example.render()

    example.renderer.save()
