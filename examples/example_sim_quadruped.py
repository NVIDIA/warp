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
import numpy as np
import os

import warp as wp
import warp.sim
import warp.sim.render
from env.environment import compute_env_offsets

wp.init()

class Example:

    frame_dt = 1.0/100.0

    episode_duration = 5.0      # seconds
    episode_frames = int(episode_duration/frame_dt)

    sim_substeps = 5
    sim_dt = frame_dt / sim_substeps
    sim_steps = int(episode_duration / sim_dt)
   
    sim_time = 0.0
    render_time = 0.0

    def __init__(self, stage=None, render=True, num_envs=1):

        self.enable_rendering = render

        self.num_envs = num_envs
        articulation_builder = wp.sim.ModelBuilder()
        wp.sim.parse_urdf(os.path.join(os.path.dirname(__file__), "assets/quadruped.urdf"), 
            articulation_builder,
            xform=wp.transform([0.0, 0.7, 0.0], wp.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi*0.5)),
            floating=True,
            density=1000,
            armature=0.01,
            stiffness=120,
            damping=1,
            shape_ke=1.e+4,
            shape_kd=1.e+2,
            shape_kf=1.e+2,
            shape_mu=0.0,
            limit_ke=1.e+4,
            limit_kd=1.e+1)

        builder = wp.sim.ModelBuilder()
        offsets = compute_env_offsets(num_envs)
        for i in range(num_envs):
            builder.add_builder(
                articulation_builder,
                xform=wp.transform(offsets[i], wp.quat_identity()))

            builder.joint_q[-12:] = [
                0.2, 0.4, -0.6,
                -0.2, -0.4, 0.6,
                -0.2, 0.4, -0.6,
                0.2, -0.4, 0.6]

            builder.joint_target[-12:] = [
                0.2, 0.4, -0.6,
                -0.2, -0.4, 0.6,
                -0.2, 0.4, -0.6,
                0.2, -0.4, 0.6]

        np.set_printoptions(suppress=True)
        # finalize model
        self.model = builder.finalize()
        self.model.ground = True

        self.model.joint_attach_ke = 16000.0
        self.model.joint_attach_kd = 200.0

        self.integrator = wp.sim.XPBDIntegrator()

        #-----------------------
        # set up Usd renderer
        self.renderer = None
        if (render):
            self.renderer = wp.sim.render.SimRenderer(self.model, stage)


    def update(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            wp.sim.collide(self.model, self.state_0)
            self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
    
    def render(self, is_live=False):
        time = 0.0 if is_live else self.sim_time

        self.renderer.begin_frame(time)
        self.renderer.render(self.state_0)
        self.renderer.end_frame()

    def run(self, render=True):

        #---------------
        # run simulation

        self.sim_time = 0.0
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        wp.sim.eval_fk(
            self.model,
            self.model.joint_q,
            self.model.joint_qd,
            None,
            self.state_0)

        profiler = {}

        # create update graph
        wp.capture_begin()

        # simulate
        self.update()
                
        graph = wp.capture_end()


        # simulate
        with wp.ScopedTimer("simulate", detailed=False, print=False, active=True, dict=profiler):

            for f in range(0, self.episode_frames):
                
                with wp.ScopedTimer("simulate", active=True):
                    wp.capture_launch(graph)
                self.sim_time += self.frame_dt

                if (self.enable_rendering):

                    with wp.ScopedTimer("render", active=True):
                        self.render()
                    self.renderer.save()

            wp.synchronize()

 
        avg_time = np.array(profiler["simulate"]).mean()/self.episode_frames
        avg_steps_second = 1000.0*float(self.num_envs)/avg_time

        print(f"envs: {self.num_envs} steps/second {avg_steps_second} avg_time {avg_time}")

        return 1000.0*float(self.num_envs)/avg_time

profile = False

if profile:

    env_count = 2
    env_times = []
    env_size = []

    for i in range(15):

        robot = Example(render=False, num_envs=env_count)
        steps_per_second = robot.run()

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
    plt.xscale('log')
    plt.xlabel("Number of Envs")
    plt.yscale('log')
    plt.ylabel("Steps/Second")
    plt.show()

else:

    stage = os.path.join(os.path.dirname(__file__), "outputs/example_sim_quadruped.usd")
    robot = Example(stage, render=True, num_envs=25)
    robot.run()
