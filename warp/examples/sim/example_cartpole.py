# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

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
import warp.sim
import warp.sim.render

wp.init()


class Example:
    def __init__(self, stage=None, num_envs=1, print_timers=True):
        builder = wp.sim.ModelBuilder()

        self.num_envs = num_envs

        articulation_builder = wp.sim.ModelBuilder()

        wp.sim.parse_urdf(
            os.path.join(os.path.dirname(__file__), "../assets/cartpole.urdf"),
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
        self.frame_dt = 1.0 / 60.0

        episode_duration = 20.0  # seconds
        self.episode_frames = int(episode_duration / self.frame_dt)

        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        for i in range(num_envs):
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
        if stage:
            self.renderer = wp.sim.render.SimRenderer(path=stage, model=self.model, scaling=15.0)

        self.print_timers = print_timers

        self.state = self.model.state()

        wp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state)

        self.use_graph = wp.get_device().is_cuda
        if self.use_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state.clear_forces()
            self.state = self.integrator.simulate(self.model, self.state, self.state, self.sim_dt)

    def step(self):
        with wp.ScopedTimer("step", active=True, print=self.print_timers):
            if self.use_graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render", active=True, print=self.print_timers):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render(self.state)
            self.renderer.end_frame()


if __name__ == "__main__":
    stage = os.path.join(os.path.dirname(__file__), "example_cartpole.usd")

    example = Example(stage, num_envs=10)

    for _ in range(example.episode_frames):
        example.step()
        example.render()

    if example.renderer:
        example.renderer.save()
