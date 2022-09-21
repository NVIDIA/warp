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

TARGET = wp.constant(wp.vec3(2.0, 1.0, 0.0))

@wp.kernel
def compute_loss(body_q: wp.array(dtype=wp.transform),
                 body_index: int,
                 loss: wp.array(dtype=float)):


    x = wp.transform_get_translation(body_q[body_index])

    delta = x - TARGET
    loss[0] = wp.dot(delta, delta)

@wp.kernel
def step_kernel(x: wp.array(dtype=float),
                grad: wp.array(dtype=float),
                alpha: float):

    tid = wp.tid()

    # gradient descent step
    x[tid] = x[tid] - grad[tid]*alpha


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


        self.device = wp.get_device(device)

        # finalize model
        self.model = builder.finalize(self.device)
        self.model.ground = False

        self.state = self.model.state()


        #-----------------------
        # set up Usd renderer
        self.renderer = wp.sim.render.SimRenderer(self.model, os.path.join(os.path.dirname(__file__), "outputs/example_sim_fk_grad.usd"))


    def run(self, render=True):

        render_time = 0.0
        train_iters = 1024
        train_rate = 0.01

        # optimization variables
        self.loss = wp.zeros(1, dtype=float, device=self.device)

        self.model.joint_q.requires_grad = True
        self.state.body_q.requires_grad = True
        self.loss.requires_grad = True

        for i in range(train_iters):

            tape = wp.Tape()
            with tape:
                
                wp.sim.eval_fk(
                    self.model,
                    self.model.joint_q,
                    self.model.joint_qd,
                    None,
                    self.state)

                wp.launch(compute_loss, dim=1, inputs=[self.state.body_q, len(self.state.body_q)-1, self.loss], device=self.device)

            tape.backward(loss=self.loss)

            print(self.loss)
            print(tape.gradients[self.model.joint_q])
            
            # gradient descent
            wp.launch(step_kernel, dim=len(self.model.joint_q), inputs=[self.model.joint_q, tape.gradients[self.model.joint_q], train_rate], device=self.device)

            # zero gradients
            tape.zero()

            # render
            self.renderer.begin_frame(render_time)
            self.renderer.render(self.state)
            self.renderer.render_sphere(name="target", pos=TARGET.val, rot=wp.quat_identity(), radius=0.1)
            self.renderer.end_frame()


            render_time += 1.0/60.0

        self.renderer.save()
        

robot = Robot(render=True, device=wp.get_preferred_device(), num_envs=1)
robot.run()
