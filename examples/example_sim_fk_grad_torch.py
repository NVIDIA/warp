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
# wp.sim.eval_ik() and wp.sim.eval_fk() methods. Shows how to connect
# gradients from Warp to PyTorch, through custom autograd nodes.
#
###########################################################################

import os
import math

import numpy as np

import torch

import warp as wp
import warp.sim
import warp.sim.render


wp.init()

class ForwardKinematics(torch.autograd.Function):


    @staticmethod
    def forward(ctx, joint_q, joint_qd, model):

        # ensure Torch operations complete before running Warp
        wp.synchronize_device()

        ctx.tape = wp.Tape()
        ctx.model = model        
        ctx.joint_q = wp.from_torch(joint_q)
        ctx.joint_qd = wp.from_torch(joint_qd)

        # allocate output
        ctx.state = model.state()

        with ctx.tape:
            
            wp.sim.eval_fk(
                model,
                ctx.joint_q,
                ctx.joint_qd,
                None,
                ctx.state)

        # ensure Warp operations complete before returning data to Torch
        wp.synchronize_device()

        return (wp.to_torch(ctx.state.body_q),
                wp.to_torch(ctx.state.body_qd))



    @staticmethod
    def backward(ctx, adj_body_q, adj_body_qd):

        # ensure Torch operations complete before running Warp
        wp.synchronize_device()

        # map incoming Torch grads to our output variables
        ctx.state.body_q.grad = wp.from_torch(adj_body_q, dtype=wp.transform)
        ctx.state.body_qd.grad = wp.from_torch(adj_body_qd, dtype=wp.spatial_vector)

        ctx.tape.backward()

        # ensure Warp operations complete before returning data to Torch
        wp.synchronize_device()

        # return adjoint w.r.t. inputs
        return (wp.to_torch(ctx.tape.gradients[ctx.joint_q]), 
                wp.to_torch(ctx.tape.gradients[ctx.joint_qd]),
                None)

class Robot:

    def __init__(self, render=True, num_envs=1, device=None):

        builder = wp.sim.ModelBuilder()

        builder.add_articulation()

        chain_length = 4
        chain_width = 1.0

        for i in range(chain_length):

            if i == 0:
                parent = -1
                parent_joint_xform = wp.transform([0.0, 0.0, 0.0], wp.quat_identity())           
            else:
                parent = builder.joint_count-1
                parent_joint_xform = wp.transform([chain_width, 0.0, 0.0], wp.quat_identity())

            joint_type = wp.sim.JOINT_REVOLUTE
            joint_axis=(0.0, 0.0, 1.0)
            joint_limit_lower=-np.deg2rad(60.0)
            joint_limit_upper=np.deg2rad(60.0)

            # create body
            b = builder.add_body(
                    parent=parent,
                    origin=wp.transform([i, 0.0, 0.0], wp.quat_identity()),
                    joint_xform=parent_joint_xform,
                    joint_axis=joint_axis,
                    joint_type=joint_type,
                    joint_limit_lower=joint_limit_lower,
                    joint_limit_upper=joint_limit_upper,
                    joint_target_ke=0.0,
                    joint_target_kd=0.0,
                    joint_limit_ke=30.0,
                    joint_limit_kd=30.0,
                    joint_armature=0.1)

            if i == chain_length-1:

                # create end effector
                s = builder.add_shape_sphere( 
                        pos=(0.0, 0.0, 0.0),
                        radius=0.1,
                        density=10.0,
                        body=b)

            else:
                # create shape
                s = builder.add_shape_box( 
                        pos=(chain_width*0.5, 0.0, 0.0),
                        hx=chain_width*0.5,
                        hy=0.1,
                        hz=0.1,
                        density=10.0,
                        body=b)

        # finalize model
        self.model = builder.finalize(device)
        self.model.ground = False

        self.torch_device = wp.device_to_torch(self.model.device)

        #-----------------------
        # set up Usd renderer
        self.renderer = wp.sim.render.SimRenderer(self.model, os.path.join(os.path.dirname(__file__), "outputs/example_sim_fk_grad.usd"))


    def run(self, render=True):

        render_time = 0.0
        train_iters = 1024
        train_rate = 0.01

        target = torch.from_numpy(np.array((2.0, 1.0, 0.0))).to(self.torch_device)

        # optimization variable
        joint_q = torch.zeros(len(self.model.joint_q), requires_grad=True, device=self.torch_device)
        joint_qd = torch.zeros(len(self.model.joint_qd), requires_grad=True, device=self.torch_device)

        for i in range(train_iters):

            (body_q, body_qd) = ForwardKinematics.apply(joint_q, joint_qd, self.model)

            l = torch.norm(body_q[self.model.body_count-1][0:3] - target)**2.0
            l.backward()

            print(l)
            print(joint_q.grad)

            with torch.no_grad():
                joint_q -= joint_q.grad*train_rate
                joint_q.grad.zero_()

            # render
            s = self.model.state()
            s.body_q = wp.from_torch(body_q, dtype=wp.transform)
            s.body_qd = wp.from_torch(body_qd, dtype=wp.spatial_vector)

            self.renderer.begin_frame(render_time)
            self.renderer.render(s)
            self.renderer.render_sphere(name="target", pos=target, rot=wp.quat_identity(), radius=0.1)
            self.renderer.end_frame()

            render_time += 1.0/60.0

        self.renderer.save()
        

robot = Robot(render=True, device="cuda", num_envs=1)
robot.run()
