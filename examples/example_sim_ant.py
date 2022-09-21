# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Ant
#
# Shows how to set up a simulation of a rigid-body Ant articulation based on
# the OpenAI gym environment using the wp.sim.ModelBuilder() and MCJF
# importer. Note this example does not include a trained policy.
#
###########################################################################

import os
import math

import numpy as np

import warp as wp
import warp.sim
import warp.sim.render

wp.init()

class Robot:

    frame_dt = 1.0/60.0

    episode_duration = 5.0      # seconds
    episode_frames = int(episode_duration/frame_dt)

    sim_substeps = 10
    sim_dt = frame_dt / sim_substeps
    sim_steps = int(episode_duration / sim_dt)
   
    sim_time = 0.0
    render_time = 0.0

    def __init__(self, render=True, num_envs=1, device=None):

        builder = wp.sim.ModelBuilder()

        self.render = render

        self.num_envs = num_envs

        builder = wp.sim.ModelBuilder()

        articulation_builder = wp.sim.ModelBuilder()

        wp.sim.parse_mjcf(os.path.join(os.path.dirname(__file__), "assets/nv_ant.xml"), articulation_builder,
            stiffness=0.0,
            damping=1.0,
            armature=0.1,
            contact_ke=1.e+4,
            contact_kd=1.e+2,
            contact_kf=1.e+4,
            contact_mu=1.0,
            limit_ke=1.e+4,
            limit_kd=1.e+1)


        for i in range(num_envs):

            builder.add_rigid_articulation(
                articulation_builder,
            )

            coord_count = 15
            dof_count = 14
            
            coord_start = i*coord_count
            dof_start = i*dof_count

            # set joint targets to rest pose in mjcf

            # # base
            builder.joint_q[coord_start:coord_start+3] = [i*2.0, 0.70, 0.0]
            builder.joint_q[coord_start+3:coord_start+7] = wp.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi*0.5)

            # joints
            builder.joint_q[coord_start+7:coord_start+coord_count] = [0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0]


        # finalize model
        self.model = builder.finalize(device)
        self.model.ground = True
        self.model.joint_attach_ke *= 32.0
        self.model.joint_attach_kd *= 4.0

        self.integrator = wp.sim.SemiImplicitIntegrator()

        #-----------------------
        # set up Usd renderer
        if (self.render):
            self.renderer = wp.sim.render.SimRenderer(self.model, os.path.join(os.path.dirname(__file__), "outputs/example_sim_ant.usd"))


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

        if (self.model.ground):
            self.model.collide(self.state)

        profiler = {}

        # create update graph
        wp.capture_begin()

        # simulate
        for i in range(0, self.sim_substeps):
            self.state.clear_forces()
            self.state = self.integrator.simulate(self.model, self.state, self.state, self.sim_dt)
            self.sim_time += self.sim_dt
                
        graph = wp.capture_end()


        # simulate
        with wp.ScopedTimer("simulate", detailed=False, print=False, active=True, dict=profiler):

            for f in range(0, self.episode_frames):
                
                wp.capture_launch(graph)
                self.sim_time += self.frame_dt

                if (self.render):

                    with wp.ScopedTimer("render", False):

                        if (self.render):
                            self.render_time += self.frame_dt
                            
                            self.renderer.begin_frame(self.render_time)
                            self.renderer.render(self.state)
                            self.renderer.end_frame()

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

        robot = Robot(render=False, num_envs=env_count)
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

    robot = Robot(render=True, num_envs=64)
    robot.run()
