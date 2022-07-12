# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Particle Chain
#
# Shows how to set up a simple chain of particles connected by springs
# using wp.sim.ModelBuilder().
#
###########################################################################

import os
import math

import numpy as np

import warp as wp
import warp.sim
import warp.sim.render

wp.init()


class Example:

    def __init__(self, stage):

        self.sim_width = 64
        self.sim_height = 32

        self.sim_fps = 60.0
        self.sim_substeps = 32
        self.sim_duration = 5.0
        self.sim_frames = int(self.sim_duration*self.sim_fps)
        self.sim_dt = (1.0/self.sim_fps)/self.sim_substeps
        self.sim_time = 0.0

        builder = wp.sim.ModelBuilder()

        # anchor
        builder.add_particle((0.0, 1.0, 0.0), (0.0, 0.0, 0.0), 0.0)

        # chain
        for i in range(1, 10):
            builder.add_particle((i, 1.0, 0.0), (0.0, 0., 0.0), 1.0)
            builder.add_spring(i - 1, i, 1.e+6, 0.0, 0)

        self.model = builder.finalize()
        self.model.ground = False

        self.integrator = wp.sim.SemiImplicitIntegrator()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.renderer = wp.sim.render.SimRenderer(self.model, stage)

    def update(self):

        with wp.ScopedTimer("simulate"):

            for s in range(self.sim_substeps):

                self.state_0.clear_forces()
                self.state_1.clear_forces()

                self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)
                self.sim_time += self.sim_dt

                # swap states
                (self.state_0, self.state_1) = (self.state_1, self.state_0)

    def render(self, is_live=False):

        with wp.ScopedTimer("render"):
            time = 0.0 if is_live else self.sim_time

            self.renderer.begin_frame(time)
            self.renderer.render(self.state_0)
            self.renderer.end_frame()


if __name__ == '__main__':
    stage_path = os.path.join(os.path.dirname(__file__), "outputs/example_sim_particle_chain.usd")

    example = Example(stage_path)

    for i in range(example.sim_frames):
        example.update()
        example.render()

    example.renderer.save()