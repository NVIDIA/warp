# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Rigid Force
#
# Shows how to apply an external force (torque) to a rigid body causing
# it to roll.
#
###########################################################################

import os

import warp as wp
import warp.sim
import warp.sim.render

wp.init()


class Example:
    def __init__(self, stage):
        self.sim_fps = 60.0
        self.sim_substeps = 5
        self.sim_duration = 5.0
        self.sim_frames = int(self.sim_duration * self.sim_fps)
        self.sim_dt = (1.0 / self.sim_fps) / self.sim_substeps
        self.sim_time = 0.0

        builder = wp.sim.ModelBuilder()

        b = builder.add_body(origin=wp.transform((0.0, 2.0, 0.0), wp.quat_identity()))
        builder.add_shape_box(body=b, hx=0.5, hy=0.5, hz=0.5, density=1000.0)

        self.model = builder.finalize()
        self.model.ground = True

        self.integrator = wp.sim.XPBDIntegrator()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.renderer = wp.sim.render.SimRenderer(self.model, stage, scaling=20.0)

    def update(self):
        with wp.ScopedTimer("simulate"):
            for _ in range(self.sim_substeps):
                wp.sim.collide(self.model, self.state_0)

                self.state_0.clear_forces()
                self.state_1.clear_forces()

                self.state_0.body_f.assign(
                    [
                        [0.0, 0.0, -3000.0, 0.0, 0.0, 0.0],
                    ]
                )

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


if __name__ == "__main__":
    stage_path = os.path.join(os.path.dirname(__file__), "outputs/example_sim_rigid_force.usd")

    example = Example(stage_path)

    for i in range(example.sim_frames):
        example.update()
        example.render()

    example.renderer.save()
