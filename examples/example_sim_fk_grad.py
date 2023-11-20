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

import numpy as np

import warp as wp
import warp.sim
import warp.sim.render

wp.init()

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

        self.device = wp.get_device(device)

        # finalize model
        self.model = builder.finalize(self.device)
        self.model.ground = False

        self.state = self.model.state()

        self.renderer = wp.sim.render.SimRenderer(self.model, stage, scaling=50.0)

        # optimization variables
        self.loss = wp.zeros(1, dtype=float, device=self.device)

        self.model.joint_q.requires_grad = True
        self.state.body_q.requires_grad = True
        self.loss.requires_grad = True

        self.train_rate = 0.01

    def update(self):
        tape = wp.Tape()

        with tape:
            wp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state)

            wp.launch(
                compute_loss,
                dim=1,
                inputs=[self.state.body_q, len(self.state.body_q) - 1, self.loss],
                device=self.device,
            )

        tape.backward(loss=self.loss)

        if self.verbose:
            print(self.loss)
            print(tape.gradients[self.model.joint_q])

        # gradient descent
        wp.launch(
            step_kernel,
            dim=len(self.model.joint_q),
            inputs=[self.model.joint_q, tape.gradients[self.model.joint_q], self.train_rate],
            device=self.device,
        )

        # zero gradients
        tape.zero()

    def render(self):
        self.renderer.begin_frame(self.render_time)
        self.renderer.render(self.state)
        self.renderer.render_sphere(name="target", pos=TARGET, rot=wp.quat_identity(), radius=0.1)
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
