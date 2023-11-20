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
            wp.sim.eval_fk(model, ctx.joint_q, ctx.joint_qd, None, ctx.state)

        # ensure Warp operations complete before returning data to Torch
        wp.synchronize_device()

        return (wp.to_torch(ctx.state.body_q), wp.to_torch(ctx.state.body_qd))

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
        return (wp.to_torch(ctx.tape.gradients[ctx.joint_q]), wp.to_torch(ctx.tape.gradients[ctx.joint_qd]), None)


class Example:
    def __init__(self, stage, device=None, verbose=False):
        self.verbose = verbose

        self.frame_dt = 1.0 / 60.0

        self.render_time = 0.0

        builder = wp.sim.ModelBuilder()

        builder.add_articulation()

        chain_length = 4
        chain_width = 1.0

        for i in range(chain_length):
            if i == 0:
                parent = -1
                parent_joint_xform = wp.transform([0.0, 0.0, 0.0], wp.quat_identity())
            else:
                parent = builder.joint_count - 1
                parent_joint_xform = wp.transform([chain_width, 0.0, 0.0], wp.quat_identity())

            # create body
            b = builder.add_body(origin=wp.transform([i, 0.0, 0.0], wp.quat_identity()), armature=0.1)

            builder.add_joint_revolute(
                parent=parent,
                child=b,
                axis=wp.vec3(0.0, 0.0, 1.0),
                parent_xform=parent_joint_xform,
                child_xform=wp.transform_identity(),
                limit_lower=-np.deg2rad(60.0),
                limit_upper=np.deg2rad(60.0),
                target_ke=0.0,
                target_kd=0.0,
                limit_ke=30.0,
                limit_kd=30.0,
            )

            if i == chain_length - 1:
                # create end effector
                builder.add_shape_sphere(pos=wp.vec3(0.0, 0.0, 0.0), radius=0.1, density=10.0, body=b)

            else:
                # create shape
                builder.add_shape_box(
                    pos=wp.vec3(chain_width * 0.5, 0.0, 0.0), hx=chain_width * 0.5, hy=0.1, hz=0.1, density=10.0, body=b
                )

        # finalize model
        self.model = builder.finalize(device)
        self.model.ground = False

        self.torch_device = wp.device_to_torch(self.model.device)

        self.renderer = wp.sim.render.SimRenderer(self.model, stage, scaling=50.0)

        self.target = torch.from_numpy(np.array((2.0, 1.0, 0.0))).to(self.torch_device)

        self.body_q = None
        self.body_qd = None

        # optimization variable
        self.joint_q = torch.zeros(len(self.model.joint_q), requires_grad=True, device=self.torch_device)
        self.joint_qd = torch.zeros(len(self.model.joint_qd), requires_grad=True, device=self.torch_device)

        self.train_rate = 0.01

    def update(self):
        (self.body_q, self.body_qd) = ForwardKinematics.apply(self.joint_q, self.joint_qd, self.model)

        l = torch.norm(self.body_q[self.model.body_count - 1][0:3] - self.target) ** 2.0
        l.backward()

        if self.verbose:
            print(l)
            print(self.joint_q.grad)

        with torch.no_grad():
            self.joint_q -= self.joint_q.grad * self.train_rate
            self.joint_q.grad.zero_()

    def render(self):
        s = self.model.state()
        s.body_q = wp.from_torch(self.body_q, dtype=wp.transform, requires_grad=False)
        s.body_qd = wp.from_torch(self.body_qd, dtype=wp.spatial_vector, requires_grad=False)

        self.renderer.begin_frame(self.render_time)
        self.renderer.render(s)
        self.renderer.render_sphere(name="target", pos=self.target, rot=wp.quat_identity(), radius=0.1)
        self.renderer.end_frame()
        self.render_time += self.frame_dt


if __name__ == "__main__":
    stage_path = os.path.join(os.path.dirname(__file__), "outputs/example_sim_fk_grad.usd")
    example = Example(stage_path, device=wp.get_preferred_device(), verbose=True)

    train_iters = 512

    for _ in range(train_iters):
        example.update()
        example.render()

    example.renderer.save()
