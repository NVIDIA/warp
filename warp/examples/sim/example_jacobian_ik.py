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
# Example Jacobian
#
# Demonstrates how to compute the Jacobian of a multi-valued function.
# Here, we use the simulation of a cartpole to differentiate
# through the kinematics function. We instantiate multiple copies of the
# cartpole and compute the Jacobian of the state of each cartpole in parallel
# in order to perform inverse kinematics via Jacobian transpose.
#
###########################################################################

import math
import os

import numpy as np

import warp as wp
import warp.examples
import warp.sim
import warp.sim.render


@wp.kernel
def compute_endeffector_position(
    body_q: wp.array(dtype=wp.transform),
    num_links: int,
    ee_link_index: int,
    ee_link_offset: wp.vec3,
    ee_pos: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    ee_pos[tid] = wp.transform_point(body_q[tid * num_links + ee_link_index], ee_link_offset)


class Example:
    def __init__(self, stage_path="example_jacobian_ik.usd", num_envs=10):
        rng = np.random.default_rng(42)

        self.num_envs = num_envs

        fps = 60
        self.frame_dt = 1.0 / fps

        self.render_time = 0.0

        # step size to use for the IK updates
        self.step_size = 0.1

        articulation_builder = wp.sim.ModelBuilder()

        wp.sim.parse_urdf(
            os.path.join(warp.examples.get_asset_directory(), "cartpole.urdf"),
            articulation_builder,
            xform=wp.transform_identity(),
            floating=False,
        )

        builder = wp.sim.ModelBuilder()

        self.num_links = len(articulation_builder.joint_type)
        # use the last link as the end-effector
        self.ee_link_index = self.num_links - 1
        self.ee_link_offset = wp.vec3(0.0, 0.0, 1.0)

        self.dof = len(articulation_builder.joint_q)

        self.target_origin = []
        for i in range(self.num_envs):
            builder.add_builder(
                articulation_builder,
                xform=wp.transform(
                    wp.vec3(i * 2.0, 4.0, 0.0), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi * 0.5)
                ),
            )
            self.target_origin.append((i * 2.0, 4.0, 0.0))
            # joint initial positions
            builder.joint_q[-3:] = rng.uniform(-0.5, 0.5, size=3)
        self.target_origin = np.array(self.target_origin)

        # finalize model
        self.model = builder.finalize()
        self.model.ground = True

        self.model.joint_q.requires_grad = True
        self.model.body_q.requires_grad = True

        self.model.joint_attach_ke = 1600.0
        self.model.joint_attach_kd = 20.0

        self.integrator = wp.sim.SemiImplicitIntegrator()

        if stage_path:
            self.renderer = wp.sim.render.SimRenderer(self.model, stage_path)
        else:
            self.renderer = None

        self.ee_pos = wp.zeros(self.num_envs, dtype=wp.vec3, requires_grad=True)

        self.state = self.model.state(requires_grad=True)

        self.targets = self.target_origin.copy()

        self.profiler = {}

    def compute_ee_position(self):
        # computes the end-effector position from the current joint angles
        wp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state)
        wp.launch(
            compute_endeffector_position,
            dim=self.num_envs,
            inputs=[self.state.body_q, self.num_links, self.ee_link_index, self.ee_link_offset],
            outputs=[self.ee_pos],
        )
        return self.ee_pos

    def compute_jacobian(self):
        # our function has 3 outputs (EE position), so we need a 3xN jacobian per environment
        jacobians = np.empty((self.num_envs, 3, self.dof), dtype=np.float32)
        tape = wp.Tape()
        with tape:
            self.compute_ee_position()
        for output_index in range(3):
            # select which row of the Jacobian we want to compute
            select_index = np.zeros(3)
            select_index[output_index] = 1.0
            e = wp.array(np.tile(select_index, self.num_envs), dtype=wp.vec3)
            tape.backward(grads={self.ee_pos: e})
            q_grad_i = tape.gradients[self.model.joint_q]
            jacobians[:, output_index, :] = q_grad_i.numpy().reshape(self.num_envs, self.dof)
            tape.zero()
        return jacobians

    def compute_fd_jacobian(self, eps=1e-4):
        jacobians = np.zeros((self.num_envs, 3, self.dof), dtype=np.float32)
        q0 = self.model.joint_q.numpy().copy()
        for e in range(self.num_envs):
            for i in range(self.dof):
                q = q0.copy()
                q[e * self.dof + i] += eps
                self.model.joint_q.assign(q)
                self.compute_ee_position()
                f_plus = self.ee_pos.numpy()[e].copy()
                q[e * self.dof + i] -= 2 * eps
                self.model.joint_q.assign(q)
                self.compute_ee_position()
                f_minus = self.ee_pos.numpy()[e].copy()
                jacobians[e, :, i] = (f_plus - f_minus) / (2 * eps)
        return jacobians

    def step(self):
        with wp.ScopedTimer("jacobian", print=False, active=True, dict=self.profiler):
            # compute jacobian
            jacobians = self.compute_jacobian()

        # compute error
        self.ee_pos_np = self.compute_ee_position().numpy()
        error = self.targets - self.ee_pos_np
        self.error = error.reshape(self.num_envs, 3, 1)

        # compute Jacobian transpose update
        delta_q = np.matmul(jacobians.transpose(0, 2, 1), self.error)

        self.model.joint_q = wp.array(
            self.model.joint_q.numpy() + self.step_size * delta_q.flatten(),
            dtype=wp.float32,
            requires_grad=True,
        )

    def render(self):
        if self.renderer is None:
            return

        self.renderer.begin_frame(self.render_time)
        self.renderer.render(self.state)
        self.renderer.render_points("targets", self.targets, radius=0.05)
        self.renderer.render_points("ee_pos", self.ee_pos_np, radius=0.05)
        self.renderer.end_frame()
        self.render_time += self.frame_dt


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="example_jacobian_ik.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--train_iters", type=int, default=50, help="Total number of training iterations.")
    parser.add_argument("--num_envs", type=int, default=10, help="Total number of simulated environments.")
    parser.add_argument(
        "--num_rollouts",
        type=int,
        default=5,
        help="Total number of rollouts. In each rollout, a new set of target points is resampled for all environments.",
    )

    args = parser.parse_known_args()[0]

    rng = np.random.default_rng(42)

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, num_envs=args.num_envs)

        print("autodiff:")
        print(example.compute_jacobian())
        print("finite diff:")
        print(example.compute_fd_jacobian())

        for _ in range(args.num_rollouts):
            # select new random target points for all envs
            example.targets = example.target_origin.copy()
            example.targets[:, 1:] += rng.uniform(-0.5, 0.5, size=(example.num_envs, 2))

            for iter in range(args.train_iters):
                example.step()
                example.render()
                print("iter:", iter, "error:", example.error.mean())

        if example.renderer:
            example.renderer.save()

        avg_time = np.array(example.profiler["jacobian"]).mean()
        avg_steps_second = 1000.0 * float(example.num_envs) / avg_time

        print(f"envs: {example.num_envs} steps/second {avg_steps_second} avg_time {avg_time}")
