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
# wp.sim.eval_ik() and wp.sim.eval_fk() methods.
#
###########################################################################

import numpy as np

import warp as wp
import warp.sim
import warp.sim.render

TARGET = wp.constant(wp.vec3(2.0, 1.0, 0.0))


@wp.kernel
def compute_loss(body_q: wp.array(dtype=wp.transform), body_index: int, loss: wp.array(dtype=float)):
    x = wp.transform_get_translation(body_q[body_index])

    delta = x - TARGET
    loss[0] = wp.dot(delta, delta)


@wp.kernel
def step_kernel(x: wp.array(dtype=float), grad: wp.array(dtype=float), alpha: float):
    tid = wp.tid()

    # gradient descent step
    x[tid] = x[tid] - grad[tid] * alpha


class Example:
    def __init__(self, stage_path="example_inverse_kinematics.usd", verbose=False):
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
                axis=(0.0, 0.0, 1.0),
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

        self.state = self.model.state()

        if stage_path:
            self.renderer = wp.sim.render.SimRenderer(self.model, stage_path, scaling=50.0)
        else:
            self.renderer = None

        # optimization variables
        self.loss = wp.zeros(1, dtype=float)

        self.model.joint_q.requires_grad = True
        self.state.body_q.requires_grad = True
        self.loss.requires_grad = True

        self.train_rate = 0.01

    def forward(self):
        wp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state)

        wp.launch(compute_loss, dim=1, inputs=[self.state.body_q, len(self.state.body_q) - 1, self.loss])

    def step(self):
        with wp.ScopedTimer("step"):
            tape = wp.Tape()
            with tape:
                self.forward()
            tape.backward(loss=self.loss)

            if self.verbose:
                print(f"loss: {self.loss}")
                print(f"joint_grad: {tape.gradients[self.model.joint_q]}")

            # gradient descent
            wp.launch(
                step_kernel,
                dim=len(self.model.joint_q),
                inputs=[self.model.joint_q, tape.gradients[self.model.joint_q], self.train_rate],
            )

            # zero gradients
            tape.zero()

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            self.renderer.begin_frame(self.render_time)
            self.renderer.render(self.state)
            self.renderer.render_sphere(
                name="target", pos=TARGET, rot=wp.quat_identity(), radius=0.1, color=(1.0, 0.0, 0.0)
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
        default="example_inverse_kinematics.usd",
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
