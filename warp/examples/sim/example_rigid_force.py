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
import argparse

import warp as wp
import warp.sim
import warp.sim.render

wp.init()


class Example:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opengl", action="store_true")

    def __init__(self, stage=None, args=None, **kwargs):
        if args is None:
            # Read args from kwargs, add default arg values from parser
            args = argparse.Namespace(**kwargs)
            args = Example.parser.parse_args(args=[], namespace=args)
        self._args = args

        self.sim_fps = 60.0
        self.sim_substeps = 5
        self.sim_duration = 5.0
        self.frame_dt = 1.0 / self.sim_fps
        self.sim_frames = int(self.sim_duration * self.sim_fps)
        self.sim_dt = (1.0 / self.sim_fps) / self.sim_substeps
        self.sim_time = 0.0

        builder = wp.sim.ModelBuilder()

        b = builder.add_body(origin=wp.transform((0.0, 10.0, 0.0), wp.quat_identity()))
        builder.add_shape_box(body=b, hx=1.0, hy=1.0, hz=1.0, density=100.0)

        self.model = builder.finalize()
        self.model.ground = True

        self.integrator = wp.sim.XPBDIntegrator()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.renderer = None
        if args.opengl:
            self.renderer = wp.sim.render.SimRendererOpenGL(self.model, stage)
        else:
            if stage:
                self.renderer = wp.sim.render.SimRenderer(self.model, stage)

        self.use_graph = wp.get_device().is_cuda
        if self.use_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            wp.sim.collide(self.model, self.state_0)

            self.state_0.clear_forces()
            self.state_1.clear_forces()

            self.state_0.body_f.assign(
                [
                    [0.0, 0.0, -7000.0, 0.0, 0.0, 0.0],
                ]
            )

            self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)

            # swap states
            (self.state_0, self.state_1) = (self.state_1, self.state_0)

    def step(self):
        with wp.ScopedTimer("step"):
            if self.use_graph:
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
    stage_path = os.path.join(wp.examples.get_output_directory(), "example_rigid_force.usd")

    args = Example.parser.parse_args()

    example = Example(stage_path, args=args)

    if args.opengl:
        while example.renderer.is_running():
            example.step()
            example.render()
    else:
        for i in range(example.sim_frames):
            example.step()
            example.render()

    if example.renderer:
        example.renderer.save()

    example.renderer = None
