# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Sim Rigid Kinematics
#
# Tests rigid body forward and backwards kinematics through the 
# wp.sim.eval_ik() and wp.sim.eval_fk() methods.
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

    episode_duration = 2.0      # seconds
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

        for i in range(num_envs):

            wp.sim.parse_mjcf(os.path.join(os.path.dirname(__file__), "assets/nv_ant.xml"), builder,
                stiffness=0.0,
                damping=1.0,
                armature=0.1,
                contact_ke=1.e+4,
                contact_kd=1.e+2,
                contact_kf=1.e+2,
                contact_mu=0.75,
                limit_ke=1.e+3,
                limit_kd=1.e+1)

            coord_count = 15
            dof_count = 14
            
            coord_start = i*coord_count
            dof_start = i*dof_count

            # base
            builder.joint_q[coord_start:coord_start+3] = [i*2.0, 0.70, 0.0]
            builder.joint_q[coord_start+3:coord_start+7] = wp.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi*0.5)

            # joints
            builder.joint_q[coord_start+7:coord_start+coord_count] = [0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0]
            builder.joint_qd[dof_start+6:dof_start+dof_count] = [1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0]


        # finalize model
        self.model = builder.finalize(device)
        self.model.ground = True
        self.model.joint_attach_ke *= 16.0
        self.model.joint_attach_kd *= 4.0

        self.integrator = wp.sim.SemiImplicitIntegrator()

        #-----------------------
        # set up Usd renderer
        if (self.render):
            self.renderer = wp.sim.render.SimRenderer(self.model, os.path.join(os.path.dirname(__file__), "outputs/example_sim_rigid_kinematics.usd"))


    def run(self, render=True):

        #---------------
        # run simulation

        self.sim_time = 0.0
        self.state = self.model.state()

        # save a copy of joint values
        q_fk = self.model.joint_q.numpy()
        qd_fk = self.model.joint_qd.numpy()

        wp.sim.eval_fk(
            self.model,
            self.model.joint_q,
            self.model.joint_qd,
            None,
            self.state)

        q_ik = wp.zeros_like(self.model.joint_q)
        qd_ik = wp.zeros_like(self.model.joint_qd)

        wp.sim.eval_ik(
            self.model,
            self.state,
            q_ik,
            qd_ik)

        q_err = q_fk - q_ik.numpy()
        qd_err = qd_fk - qd_ik.numpy()

        print(q_err)
        print(qd_err)

        assert(np.abs(q_err).max() < 1.e-6)
        assert(np.abs(qd_err).max() < 1.e-6)

        

robot = Robot(render=False, num_envs=1)
robot.run()
