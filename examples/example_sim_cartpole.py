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

import os
import math

import numpy as np

import warp as wp
import warp.sim
import warp.sim.render

wp.init()

class Example:

    frame_dt = 1.0/60.0

    episode_duration = 20.0      # seconds
    episode_frames = int(episode_duration/frame_dt)

    sim_substeps = 10
    sim_dt = frame_dt / sim_substeps
    sim_steps = int(episode_duration / sim_dt)
   
    sim_time = 0.0

    def __init__(self, stage=None, render=True, num_envs=1):

        builder = wp.sim.ModelBuilder()

        self.enable_rendering = render

        self.num_envs = num_envs

        articulation_builder = wp.sim.ModelBuilder()

        wp.sim.parse_urdf(
            os.path.join(os.path.dirname(__file__), "assets/cartpole.urdf"),
            articulation_builder,
            xform=wp.transform(np.zeros(3), wp.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi*0.5)),
            floating=False, 
            density=0,
            armature=0.1,
            stiffness=0.0,
            damping=0.0,
            shape_ke=1.e+4,
            shape_kd=1.e+2,
            shape_kf=1.e+2,
            shape_mu=1.0,
            limit_ke=1.e+4,
            limit_kd=1.e+1,
            enable_self_collisions=False)

        builder = wp.sim.ModelBuilder()

        for i in range(num_envs):
            builder.add_builder(
                articulation_builder,
                xform=wp.transform(np.array((i * 2.0, 4.0, 0.0)), wp.quat_identity())
            )

            # joint initial positions
            builder.joint_q[-3:] = [0.0, 0.3, 0.0]

            builder.joint_target[:3] = [0.0, 0.0, 0.0]

        # finalize model
        self.model = builder.finalize()
        self.model.ground = False

        self.model.joint_attach_ke = 1600.0
        self.model.joint_attach_kd = 20.0

        # ensure the module has been compiled before capturing a CUDA graph
        wp.load_module(warp.sim, recursive=True, device=self.model.device)

        self.integrator = wp.sim.SemiImplicitIntegrator()

        #-----------------------
        # set up Usd renderer
        self.renderer = None
        if (render):
            self.renderer = wp.sim.render.SimRenderer(self.model, stage, scaling=15.0)

    def update(self):
        for _ in range(self.sim_substeps):
            self.state.clear_forces()
            self.state = self.integrator.simulate(self.model, self.state, self.state, self.sim_dt)
    
    def render(self, is_live=False):
        time = 0.0 if is_live else self.sim_time

        self.renderer.begin_frame(time)
        self.renderer.render(self.state)
        self.renderer.end_frame()

    def run(self, render=True):

        #---------------
        # run simulation

        self.sim_time = 0.0
        self.state = self.model.state()

        wp.sim.eval_fk(
            self.model,
            self.model.joint_q,
            self.model.joint_qd,
            None,
            self.state)

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

    stage = os.path.join(os.path.dirname(__file__), "outputs/example_sim_cartpole.usd")
    robot = Example(stage, render=True, num_envs=10)
    robot.run()
