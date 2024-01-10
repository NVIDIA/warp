# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Quadruped
#
# Shows how to set up a simulation of a rigid-body quadruped articulation
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


# Taken from env/environment.py
def compute_env_offsets(num_envs, env_offset=(5.0, 0.0, 5.0), up_axis="Y"):
    # compute positional offsets per environment
    env_offset = np.array(env_offset)
    nonzeros = np.nonzero(env_offset)[0]
    num_dim = nonzeros.shape[0]
    if num_dim > 0:
        side_length = int(np.ceil(num_envs ** (1.0 / num_dim)))
        env_offsets = []
    else:
        env_offsets = np.zeros((num_envs, 3))
    if num_dim == 1:
        for i in range(num_envs):
            env_offsets.append(i * env_offset)
    elif num_dim == 2:
        for i in range(num_envs):
            d0 = i // side_length
            d1 = i % side_length
            offset = np.zeros(3)
            offset[nonzeros[0]] = d0 * env_offset[nonzeros[0]]
            offset[nonzeros[1]] = d1 * env_offset[nonzeros[1]]
            env_offsets.append(offset)
    elif num_dim == 3:
        for i in range(num_envs):
            d0 = i // (side_length * side_length)
            d1 = (i // side_length) % side_length
            d2 = i % side_length
            offset = np.zeros(3)
            offset[0] = d0 * env_offset[0]
            offset[1] = d1 * env_offset[1]
            offset[2] = d2 * env_offset[2]
            env_offsets.append(offset)
    env_offsets = np.array(env_offsets)
    min_offsets = np.min(env_offsets, axis=0)
    correction = min_offsets + (np.max(env_offsets, axis=0) - min_offsets) / 2.0
    if isinstance(up_axis, str):
        up_axis = "XYZ".index(up_axis.upper())
    correction[up_axis] = 0.0  # ensure the envs are not shifted below the ground plane
    env_offsets -= correction
    return env_offsets


class Example:
    def __init__(self, stage=None, num_envs=1, enable_rendering=True, print_timers=True):
        self.device = wp.get_device()
        self.num_envs = num_envs
        articulation_builder = wp.sim.ModelBuilder()
        wp.sim.parse_urdf(
            os.path.join(os.path.dirname(__file__), "assets/quadruped.urdf"),
            articulation_builder,
            xform=wp.transform([0.0, 0.7, 0.0], wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi * 0.5)),
            floating=True,
            density=1000,
            armature=0.01,
            stiffness=120,
            damping=1,
            shape_ke=1.0e4,
            shape_kd=1.0e2,
            shape_kf=1.0e2,
            shape_mu=0.0,
            limit_ke=1.0e4,
            limit_kd=1.0e1,
        )

        builder = wp.sim.ModelBuilder()

        self.sim_time = 0.0
        self.frame_dt = 1.0 / 100.0

        episode_duration = 5.0  # seconds
        self.episode_frames = int(episode_duration / self.frame_dt)

        self.sim_substeps = 5
        self.sim_dt = self.frame_dt / self.sim_substeps

        offsets = compute_env_offsets(num_envs)
        for i in range(num_envs):
            builder.add_builder(articulation_builder, xform=wp.transform(offsets[i], wp.quat_identity()))

            builder.joint_q[-12:] = [0.2, 0.4, -0.6, -0.2, -0.4, 0.6, -0.2, 0.4, -0.6, 0.2, -0.4, 0.6]

            builder.joint_target[-12:] = [0.2, 0.4, -0.6, -0.2, -0.4, 0.6, -0.2, 0.4, -0.6, 0.2, -0.4, 0.6]

        np.set_printoptions(suppress=True)
        # finalize model
        self.model = builder.finalize()
        self.model.ground = True

        self.model.joint_attach_ke = 16000.0
        self.model.joint_attach_kd = 200.0

        self.integrator = wp.sim.XPBDIntegrator()

        self.enable_rendering = enable_rendering
        self.renderer = None
        if self.enable_rendering:
            self.renderer = wp.sim.render.SimRenderer(self.model, stage)

        self.print_timers = print_timers

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

    def update(self):
        with wp.ScopedTimer("simulate", active=True, print=self.print_timers):
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
        if self.enable_rendering:
            with wp.ScopedTimer("render", active=True, print=self.print_timers):
                time = 0.0 if is_live else self.sim_time

                self.renderer.begin_frame(time)
                self.renderer.render(self.state_0)
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
        stage = os.path.join(os.path.dirname(__file__), "outputs/example_sim_quadruped.usd")
        example = Example(stage, num_envs=25, enable_rendering=True)
        example.run()
