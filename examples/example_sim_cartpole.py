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
    def __init__(self, stage=None, num_envs=1, enable_rendering=True, print_timers=True):
        self.device = wp.get_device()

        builder = wp.sim.ModelBuilder()

        self.num_envs = num_envs

        articulation_builder = wp.sim.ModelBuilder()

        wp.sim.parse_urdf(
            os.path.join(os.path.dirname(__file__), "assets/cartpole.urdf"),
            articulation_builder,
            xform=wp.transform(wp.vec3(), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi * 0.5)),
            floating=False,
            density=100,
            armature=0.1,
            stiffness=0.0,
            damping=0.0,
            shape_ke=1.0e4,
            shape_kd=1.0e2,
            shape_kf=1.0e2,
            shape_mu=1.0,
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

            builder.joint_target[:3] = [0.0, 0.0, 0.0]

        # finalize model
        self.model = builder.finalize()
        self.model.ground = False

        self.model.joint_attach_ke = 1600.0
        self.model.joint_attach_kd = 20.0

        self.integrator = wp.sim.SemiImplicitIntegrator()

        self.enable_rendering = enable_rendering
        self.renderer = None
        if self.enable_rendering:
            self.renderer = wp.sim.render.SimRenderer(path=stage, model=self.model, scaling=15.0)

        self.print_timers = print_timers

        self.state = self.model.state()

        wp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state)

        self.use_graph = wp.get_device().is_cuda
        self.graph = None

        if self.use_graph:
            # create update graph
            wp.capture_begin(self.device)
            try:
                self.update()
            finally:
                self.graph = wp.capture_end(self.device)

    def update(self):
        with wp.ScopedTimer("simulate", active=True, print=self.print_timers):
            if not self.use_graph or self.graph is None:
                for _ in range(self.sim_substeps):
                    self.state.clear_forces()
                    self.state = self.integrator.simulate(self.model, self.state, self.state, self.sim_dt)
            else:
                wp.capture_launch(self.graph)

            if not wp.get_device().is_capturing:
                self.sim_time += self.frame_dt

    def render(self, is_live=False):
        if self.enable_rendering:
            with wp.ScopedTimer("render", active=True, print=self.print_timers):
                time = 0.0 if is_live else self.sim_time

                self.renderer.begin_frame(time)
                self.renderer.render(self.state)
                self.renderer.end_frame()

    def run(self):
        profiler = {}

        with wp.ScopedTimer("simulate", detailed=False, print=False, active=True, dict=profiler):
            for _ in range(self.episode_frames):
                self.update()
                self.render()

            wp.synchronize()

        if self.enable_rendering:
            self.renderer.save()

        avg_time = np.array(profiler["simulate"]).mean() / self.episode_frames
        avg_steps_second = 1000.0 * float(self.num_envs) / avg_time

        print(f"envs: {self.num_envs} steps/second {avg_steps_second} avg_time {avg_time}")

        return 1000.0 * float(self.num_envs) / avg_time


if __name__ == "__main__":
    profile = False

    if profile:
        env_count = 2
        env_times = []
        env_size = []

        for i in range(15):
            example = Example(num_envs=env_count, enable_rendering=False, print_timers=False)
            steps_per_second = example.run()

            env_size.append(env_count)
            env_times.append(steps_per_second)

            env_count *= 2

        # dump times
        for i in range(len(env_times)):
            print(f"envs: {env_size[i]} steps/second: {env_times[i]}")

        # plot
        import matplotlib.pyplot as plt

        plt.figure(1)
        plt.plot(env_size, env_times)
        plt.xscale("log")
        plt.xlabel("Number of Envs")
        plt.yscale("log")
        plt.ylabel("Steps/Second")
        plt.show()

    else:
        stage = os.path.join(os.path.dirname(__file__), "outputs/example_sim_cartpole.usd")
        example = Example(stage, num_envs=10)
        example.run()
