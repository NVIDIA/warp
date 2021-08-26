import math

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import warp as wp

import tests.test_sim_util as util
from tests.render_sim import SimRenderer

wp.init()

class Robot:

    frame_dt = 1.0/60.0

    episode_duration = 2.0      # seconds
    episode_frames = int(episode_duration/frame_dt)

    sim_substeps = 16
    sim_dt = frame_dt / sim_substeps
    sim_steps = int(episode_duration / sim_dt)
   
    sim_time = 0.0
    render_time = 0.0

    name = "ant"

    def __init__(self, render=True, device='cpu'):

        builder = wp.sim.ModelBuilder()

        self.device = device
        self.render = render   

        util.parse_mjcf("./tests/assets/" + self.name + ".xml", builder,
            stiffness=0.0,
            damping=1.0,
            armature=0.05,
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

        joint_q = np.zeros(self.model.joint_coord_count, dtype=np.float32)
        joint_qd = np.zeros(self.model.joint_dof_count, dtype=np.float32)

        # set joint targets to rest pose in mjcf
        if (self.name == "ant"):
            joint_q[0:3] = [0.0, 0.70, 0.0]
            joint_q[3:7] = wp.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi*0.5)

            joint_q[7:] = [0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0]
            #joint_q[7:] = [0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0] 

        if (self.name == "humanoid"):
            joint_q[0:3] = [0.0, 1.70, 0.0]
            joint_q[3:7] = wp.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi*0.5)

        wp.sim.eval_fk(
            self.model,
            wp.array(joint_q, dtype=float, device=self.device), 
            wp.array(joint_qd, dtype=float, device=self.device), 
            self.state)

        if (self.model.ground):
            self.model.collide(self.state)

        for f in range(0, self.episode_frames):
                
            # simulate
            with wp.ScopedTimer("simulate", detailed=False, active=True):

                for i in range(0, self.sim_substeps):
                    self.state.clear_forces()
                    self.state = self.integrator.simulate(self.model, self.state, self.state, self.sim_dt)
                    self.sim_time += self.sim_dt
 
 
             # render
            with wp.ScopedTimer("render", False):

                if (self.render):
                    self.render_time += self.frame_dt
                    
                    self.renderer.begin_frame(self.render_time)
                    self.renderer.render(self.state)
                    self.renderer.end_frame()

        self.renderer.save()

 
robot = Robot(render=True, device='cuda')
robot.run()