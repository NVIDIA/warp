# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

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
import warp.sim
import warp.sim.render

wp.init()


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
    def __init__(self, stage=None, enable_rendering=True, num_envs=1, device=None):
        builder = wp.sim.ModelBuilder()

        self.num_envs = num_envs
        self.device = device

        self.frame_dt = 1.0 / 60.0

        self.render_time = 0.0

        # step size to use for the IK updates
        self.step_size = 0.1

        articulation_builder = wp.sim.ModelBuilder()

        wp.sim.parse_urdf(
            os.path.join(os.path.dirname(__file__), "assets/cartpole.urdf"),
            articulation_builder,
            xform=wp.transform_identity(),
            floating=False,
            density=0,
            armature=0.1,
            stiffness=0.0,
            damping=0.0,
            shape_ke=1.0e4,
            shape_kd=1.0e2,
            shape_kf=1.0e2,
            shape_mu=1.0,
            limit_ke=1.0e4,
            limit_kd=1.0e1,
        )

        builder = wp.sim.ModelBuilder()

        self.num_links = len(articulation_builder.joint_type)
        # use the last link as the end-effector
        self.ee_link_index = self.num_links - 1
        self.ee_link_offset = wp.vec3(0.0, 0.0, 1.0)

        self.dof = len(articulation_builder.joint_q)

        self.target_origin = []
        for i in range(num_envs):
            builder.add_builder(
                articulation_builder,
                xform=wp.transform(
                    wp.vec3(i * 2.0, 4.0, 0.0), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -math.pi * 0.5)
                ),
            )
            self.target_origin.append((i * 2.0, 4.0, 0.0))
            # joint initial positions
            builder.joint_q[-3:] = np.random.uniform(-0.5, 0.5, size=3)
        self.target_origin = np.array(self.target_origin)

        # finalize model
        self.model = builder.finalize(device)
        self.model.ground = True

        self.model.joint_q.requires_grad = True
        self.model.body_q.requires_grad = True

        self.model.joint_attach_ke = 1600.0
        self.model.joint_attach_kd = 20.0

        self.integrator = wp.sim.SemiImplicitIntegrator()

        self.enable_rendering = enable_rendering
        self.renderer = None
        if enable_rendering:
            self.renderer = wp.sim.render.SimRenderer(self.model, stage)

        self.ee_pos = wp.zeros(self.num_envs, dtype=wp.vec3, device=device, requires_grad=True)

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
            device=self.device,
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
            e = wp.array(np.tile(select_index, self.num_envs), dtype=wp.vec3, device=self.device)
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

    def update(self):
        with wp.ScopedTimer("jacobian", print=False, active=True, dict=self.profiler):
            # compute jacobian
            jacobians = self.compute_jacobian()
            # jacobians = self.compute_fd_jacobian()

        # compute error
        self.ee_pos_np = self.compute_ee_position().numpy()
        error = self.targets - self.ee_pos_np
        self.error = error.reshape(self.num_envs, 3, 1)

        # compute Jacobian transpose update
        delta_q = np.matmul(jacobians.transpose(0, 2, 1), self.error)

        self.model.joint_q = wp.array(
            self.model.joint_q.numpy() + self.step_size * delta_q.flatten(),
            dtype=wp.float32,
            device=self.device,
            requires_grad=True,
        )

    def render(self):
        if self.enable_rendering:
            self.renderer.begin_frame(self.render_time)
            self.renderer.render(self.state)
            self.renderer.render_points("targets", self.targets, radius=0.05)
            self.renderer.render_points("ee_pos", self.ee_pos_np, radius=0.05)
            self.renderer.end_frame()
            self.render_time += self.frame_dt

    def run(self):
        if self.enable_rendering:
            print("autodiff:")
            print(self.compute_jacobian())
            print("finite diff:")
            print(self.compute_fd_jacobian())

        for _ in range(5):
            # select new random target points
            self.targets = self.target_origin.copy()
            self.targets[:, 1:] += np.random.uniform(-0.5, 0.5, size=(self.num_envs, 2))

            for iter in range(50):
                self.update()
                self.render()
                print("iter:", iter, "error:", self.error.mean())

        if self.enable_rendering:
            self.renderer.save()

        avg_time = np.array(self.profiler["jacobian"]).mean()
        avg_steps_second = 1000.0 * float(self.num_envs) / avg_time

        print(f"envs: {self.num_envs} steps/second {avg_steps_second} avg_time {avg_time}")

        return 1000.0 * float(self.num_envs) / avg_time


if __name__ == "__main__":
    profile = False

    if profile:
        env_count = 2
        env_times = []
        env_size = []

        for i in range(12):
            example = Example(enable_rendering=False, num_envs=env_count)
            steps_per_second = example.run()

            env_size.append(env_count)
            env_times.append(steps_per_second)

            env_count *= 2

        # dump times
        for i in range(len(env_times)):
            print(f"envs: {env_size[i]} steps/second: {env_times[i]}")

        # plot
        import matplotlib.pyplot as plt

        plt.figure(1)
        plt.plot(env_size, env_times)
        plt.xscale("log")
        plt.xlabel("Number of Envs")
        plt.yscale("log")
        plt.ylabel("Steps/Second")
        plt.show()

    else:
        stage_path = os.path.join(os.path.dirname(__file__), "outputs/example_jacobian_ik.usd")
        example = Example(stage_path, enable_rendering=True, num_envs=10)
        example.run()
