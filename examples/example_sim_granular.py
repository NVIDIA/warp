# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Granular
#
# Shows how to set up a particle-based granular material model using the
# wp.sim.ModelBuilder().
#
###########################################################################

import os

import warp as wp
import warp.sim
import warp.sim.render

wp.init()


class Example:
    def __init__(self, stage):
        self.frame_dt = 1.0 / 60
        self.frame_count = 400

        self.sim_substeps = 64
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_steps = self.frame_count * self.sim_substeps
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
            pos=wp.vec3(0.0, 1.0, 0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(5.0, 0.0, 0.0),
            mass=0.1,
            jitter=self.radius * 0.1,
        )

        self.model = builder.finalize()
        self.model.particle_kf = 25.0

        self.model.soft_contact_kd = 100.0
        self.model.soft_contact_kf *= 2.0

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.integrator = wp.sim.SemiImplicitIntegrator()

        self.renderer = wp.sim.render.SimRenderer(self.model, stage, scaling=20.0)

    def update(self):
        with wp.ScopedTimer("simulate", active=True):
            self.model.particle_grid.build(self.state_0.particle_q, self.radius * 2.0)

            for _ in range(self.sim_substeps):
                self.state_0.clear_forces()

                self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)
                self.sim_time += self.frame_dt

                # swap states
                (self.state_0, self.state_1) = (self.state_1, self.state_0)

    def render(self, is_live=False):
        with wp.ScopedTimer("render", active=True):
            time = 0.0 if is_live else self.sim_time

            self.renderer.begin_frame(time)
            self.renderer.render(self.state_0)
            self.renderer.end_frame()


if __name__ == "__main__":
    stage_path = os.path.join(os.path.dirname(__file__), "outputs/example_sim_granular.usd")

    example = Example(stage_path)

    for _ in range(example.frame_count):
        example.update()
        example.render()

    example.renderer.save()
