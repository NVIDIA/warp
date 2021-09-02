import math

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import warp as wp

import tests.test_sim_util as util
from tests.render_sim import SimRenderer

import matplotlib.pyplot as plt

wp.init()

class Robot:

    frame_dt = 1.0/60.0

    episode_duration = 2.0      # seconds
    episode_frames = int(episode_duration/frame_dt)

    sim_substeps = 10
    sim_dt = frame_dt / sim_substeps
    sim_steps = int(episode_duration / sim_dt)
   
    sim_time = 0.0
    render_time = 0.0

    name = "ant"

    def __init__(self, render=True, num_envs=1, device='cpu'):

        builder = wp.sim.ModelBuilder()

        self.device = device
        self.render = render

        self.num_envs = num_envs

        for i in range(num_envs):

            util.parse_mjcf("./tests/assets/" + self.name + ".xml", builder,
                stiffness=0.0,
                damping=1.0,
                armature=0.1,
                contact_ke=1.e+4,
                contact_kd=1.e+2,
                contact_kf=1.e+2,
                contact_mu=0.75,
                limit_ke=1.e+3,
                limit_kd=1.e+1)

        # finalize model
        self.model = builder.finalize(device)
        self.model.ground = True
        self.model.joint_attach_ke *= 16.0
        self.model.joint_attach_kd *= 4.0

        self.integrator = wp.sim.SemiImplicitIntegrator()

        #-----------------------
        # set up Usd renderer
        if (self.render):
            self.renderer = SimRenderer(self.model, "./tests/outputs/" + self.name + ".usd")


    def run(self, render=True):

        #---------------
        # run simulation

        self.sim_time = 0.0
        self.state = self.model.state()

        ant_coord_count = 15
        ant_dof_count = 14

        joint_q = np.zeros((self.num_envs, ant_coord_count), dtype=np.float32)
        joint_qd = np.zeros((self.num_envs, ant_dof_count), dtype=np.float32)

        for i in range(self.num_envs):

            # set joint targets to rest pose in mjcf
            if (self.name == "ant"):
                joint_q[i, 0:3] = [i*2.0, 0.70, 0.0]
                joint_q[i, 3:7] = wp.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi*0.5)

                joint_q[i, 7:] = [0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0]
                #joint_q[7:] = [0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0] 

        wp.sim.eval_fk(
            self.model,
            wp.array(joint_q, dtype=float, device=self.device), 
            wp.array(joint_qd, dtype=float, device=self.device), 
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
                
                # for i in range(0, self.sim_substeps):
                #     self.state.clear_forces()
                #     self.state = self.integrator.simulate(self.model, self.state, self.state, self.sim_dt)
                #     self.sim_time += self.sim_dt

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

        robot = Robot(render=False, device='cuda', num_envs=env_count)
        steps_per_second = robot.run()

        env_size.append(env_count)
        env_times.append(steps_per_second)
        
        env_count *= 2

    # dump times
    for i in range(len(env_times)):
        print(f"envs: {env_size[i]} steps/second: {env_times[i]}")

    # plot
    plt.figure(1)
    plt.plot(env_size, env_times)
    plt.xscale('log')
    plt.xlabel("Number of Envs")
    plt.yscale('log')
    plt.ylabel("Steps/Second")
    plt.show()

else:

    robot = Robot(render=True, device='cuda', num_envs=64)
    robot.run()
