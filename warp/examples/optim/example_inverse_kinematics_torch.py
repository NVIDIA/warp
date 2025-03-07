# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example Sim Rigid Kinematics
#
# Tests rigid body forward and backwards kinematics through the
# wp.sim.eval_ik() and wp.sim.eval_fk() methods. Shows how to connect
# gradients from Warp to PyTorch, through custom autograd nodes.
#
###########################################################################

import numpy as np
import torch

import warp as wp
import warp.sim
import warp.sim.render


class ForwardKinematics(torch.autograd.Function):
    @staticmethod
    def forward(ctx, joint_q, joint_qd, model):
        ctx.tape = wp.Tape()
        ctx.model = model
        ctx.joint_q = wp.from_torch(joint_q)
        ctx.joint_qd = wp.from_torch(joint_qd)

        # allocate output
        ctx.state = model.state()

        with ctx.tape:
            wp.sim.eval_fk(model, ctx.joint_q, ctx.joint_qd, None, ctx.state)

        return (wp.to_torch(ctx.state.body_q), wp.to_torch(ctx.state.body_qd))

    @staticmethod
    def backward(ctx, adj_body_q, adj_body_qd):
        # map incoming Torch grads to our output variables
        ctx.state.body_q.grad = wp.from_torch(adj_body_q, dtype=wp.transform)
        ctx.state.body_qd.grad = wp.from_torch(adj_body_qd, dtype=wp.spatial_vector)

        ctx.tape.backward()

        # return adjoint w.r.t. inputs
        return (wp.to_torch(ctx.tape.gradients[ctx.joint_q]), wp.to_torch(ctx.tape.gradients[ctx.joint_qd]), None)


class Example:
    def __init__(self, stage_path="example_inverse_kinematics_torch.usd", verbose=False):
        self.verbose = verbose

        fps = 60
        self.frame_dt = 1.0 / fps
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
        self.model = builder.finalize()
        self.model.ground = False

        self.torch_device = wp.device_to_torch(wp.get_device())

        if stage_path:
            self.renderer = wp.sim.render.SimRenderer(self.model, stage_path, scaling=50.0)
        else:
            self.renderer = None

        self.target = torch.from_numpy(np.array((2.0, 1.0, 0.0))).to(self.torch_device)

        self.body_q = None
        self.body_qd = None

        # optimization variable
        self.joint_q = torch.zeros(len(self.model.joint_q), requires_grad=True, device=self.torch_device)
        self.joint_qd = torch.zeros(len(self.model.joint_qd), requires_grad=True, device=self.torch_device)

        self.train_rate = 0.01

    def forward(self):
        (self.body_q, self.body_qd) = ForwardKinematics.apply(self.joint_q, self.joint_qd, self.model)
        self.loss = torch.norm(self.body_q[self.model.body_count - 1][0:3] - self.target) ** 2.0

    def step(self):
        with wp.ScopedTimer("step"):
            self.forward()
            self.loss.backward()

            if self.verbose:
                print(f"loss: {self.loss}")
                print(f"loss: {self.joint_q.grad}")

            with torch.no_grad():
                self.joint_q -= self.joint_q.grad * self.train_rate
                self.joint_q.grad.zero_()

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            s = self.model.state()
            s.body_q = wp.from_torch(self.body_q, dtype=wp.transform, requires_grad=False)
            s.body_qd = wp.from_torch(self.body_qd, dtype=wp.spatial_vector, requires_grad=False)

            self.renderer.begin_frame(self.render_time)
            self.renderer.render(s)
            self.renderer.render_sphere(
                name="target", pos=self.target, rot=wp.quat_identity(), radius=0.1, color=(1.0, 0.0, 0.0)
            )
            self.renderer.end_frame()
            self.render_time += self.frame_dt


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="example_inverse_kinematics_torch.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--train_iters", type=int, default=512, help="Total number of training iterations.")
    parser.add_argument("--verbose", action="store_true", help="Print out additional status messages during execution.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, verbose=args.verbose)

        for _ in range(args.train_iters):
            example.step()
            example.render()

        if example.renderer:
            example.renderer.save()
