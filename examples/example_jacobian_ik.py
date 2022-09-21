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

import os
import math

import numpy as np

import warp as wp
import warp.sim
import warp.sim.render

wp.init()


class Robot:

    frame_dt = 1.0/60.0

    episode_duration = 20.0      # seconds
    episode_frames = int(episode_duration/frame_dt)

    sim_substeps = 10
    sim_dt = frame_dt / sim_substeps
    sim_steps = int(episode_duration / sim_dt)

    sim_time = 0.0
    render_time = 0.0

    step_size = 0.1

    def __init__(self, render=True, num_envs=1, profile=False, device=None):

        builder = wp.sim.ModelBuilder()

        self.render = render

        self.num_envs = num_envs
        self.profile = profile
        self.device = device

        articulation_builder = wp.sim.ModelBuilder()

        wp.sim.parse_urdf(os.path.join(os.path.dirname(__file__), "assets/cartpole.urdf"), articulation_builder,
                          xform=wp.transform(np.array((0.0, 0.0, 0.0)), wp.quat_from_axis_angle(
                              (1.0, 0.0, 0.0), -math.pi*0.5)),
                          floating=False,
                          density=0,
                          armature=0.1,
                          stiffness=0.0,
                          damping=0.0,
                          shape_ke=1.e+4,
                          shape_kd=1.e+2,
                          shape_kf=1.e+2,
                          shape_mu=1.0,
                          limit_ke=1.e+4,
                          limit_kd=1.e+1)

        builder = wp.sim.ModelBuilder()

        self.num_links = len(articulation_builder.joint_type)
        # use the last link as the end-effector
        self.ee_link_index = self.num_links - 1
        self.ee_link_offset = wp.vec3(0.0, 0.0, 1.0)

        self.dof = len(articulation_builder.joint_q)

        for i in range(num_envs):
            builder.add_rigid_articulation(
                articulation_builder,
                xform=wp.transform(np.array(
                    (i * 2.0, 4.0, 0.0)), wp.quat_from_axis_angle((1.0, 0.0, 0.0), -math.pi*0.5))
            )

            # joint initial positions
            builder.joint_q[-3:] = np.random.uniform(-0.5, 0.5, size=3)

        # finalize model
        self.model = builder.finalize(device)
        self.model.ground = True

        self.model.joint_q.requires_grad = True
        self.model.body_q.requires_grad = True

        self.model.joint_attach_ke = 1600.0
        self.model.joint_attach_kd = 20.0

        self.integrator = wp.sim.SemiImplicitIntegrator()

        # -----------------------
        # set up Usd renderer
        if (self.render):
            self.renderer = wp.sim.render.SimRenderer(self.model, os.path.join(
                os.path.dirname(__file__), "outputs/example_jacobian_ik.usd"))

        self.ee_pos = wp.zeros(self.num_envs, dtype=wp.vec3,
                               device=device, requires_grad=True)

    @wp.kernel
    def compute_endeffector_position(
            body_q: wp.array(dtype=wp.transform),
            num_links: int,
            ee_link_index: int,
            ee_link_offset: wp.vec3,
            ee_pos: wp.array(dtype=wp.vec3)):
        tid = wp.tid()
        ee_pos[tid] = wp.transform_point(
            body_q[tid*num_links + ee_link_index], ee_link_offset)

    def compute_ee_position(self):
        # computes the end-effector position from the current joint angles
        wp.sim.eval_fk(
            self.model,
            self.model.joint_q,
            self.model.joint_qd,
            None,
            self.state)
        wp.launch(
            self.compute_endeffector_position,
            dim=self.num_envs,
            inputs=[
                self.state.body_q,
                self.num_links,
                self.ee_link_index,
                self.ee_link_offset
            ],
            outputs=[
                self.ee_pos
            ],
            device=self.device)
        return self.ee_pos

    @wp.kernel
    def compress_function_output(
            ee_pos: wp.array(dtype=wp.vec3),
            output_index: int,
            output: wp.array(dtype=wp.float32)):
        tid = wp.tid()
        wp.atomic_add(output, 0, ee_pos[tid][output_index])

    def compute_function(self, output_index: int, output: wp.array):
        self.compute_ee_position()
        # output_index selects which row of the Jacobian to evaluate in the backward pass
        wp.launch(
            self.compress_function_output,
            dim=self.num_envs,
            inputs=[
                self.ee_pos,
                output_index
            ],
            outputs=[
                output
            ],
            device=self.device)

    def compute_jacobian(self):
        output_1d = wp.zeros(1, dtype=wp.float32,
                             device=self.device, requires_grad=True)
        # our function has 3 outputs (EE position), so we need a 3xN jacobian per environment
        jacobians = np.empty((self.num_envs, 3, self.dof), dtype=np.float32)
        # jacobians = wp.zeros((self.num_envs, 3, self.dof), dtype=wp.float32, device=self.device)
        for output_index in range(3):
            output_1d.zero_()
            tape = wp.Tape()
            with wp.ScopedTimer("Forward", active=self.profile):
                with tape:
                    self.compute_function(output_index, output_1d)
            with wp.ScopedTimer("Backward", active=self.profile):
                tape.backward(output_1d)
            q_grad_i = tape.gradients[self.model.joint_q]
            jacobians[:, output_index, :] = q_grad_i.numpy().reshape(
                self.num_envs, self.dof)
            tape.zero()
        return jacobians

    def compute_fd_jacobian(self, eps=1e-4):
        jacobians = np.zeros((self.num_envs, 3, self.dof), dtype=np.float32)
        q0 = self.model.joint_q.numpy().copy()
        for e in range(self.num_envs):
            for i in range(self.dof):
                q = q0.copy()
                q[e*self.dof + i] += eps
                self.model.joint_q.assign(q)
                self.compute_ee_position()
                f_plus = self.ee_pos.numpy()[e].copy()
                q[e*self.dof + i] -= 2*eps
                self.model.joint_q.assign(q)
                self.compute_ee_position()
                f_minus = self.ee_pos.numpy()[e].copy()
                jacobians[e, :, i] = (f_plus - f_minus) / (2*eps)
        return jacobians

    def run(self, render=True):

        # ---------------
        # run simulation

        self.sim_time = 0.0
        self.state = self.model.state(requires_grad=True)

        if render:
            print("autodiff:")
            print(self.compute_jacobian())
            print("finite diff:")
            print(self.compute_fd_jacobian())

        for _ in range(5):
            # select new random target points
            targets = []
            for i in range(self.num_envs):
                target_pos = np.array([i * 2.0, 4.0, 0.0])
                target_pos[1] += np.random.uniform(-1.5, 1.5)
                target_pos[2] += np.random.uniform(-1.5, 1.5)
                targets.append(target_pos)
            targets = np.array(targets)

            for iter in range(50):
                # compute jacobian
                jacobians = self.compute_jacobian()
                # jacobians = self.compute_fd_jacobian()

                # compute error
                ee_pos = self.compute_ee_position().numpy()
                error = targets - ee_pos
                error = error.reshape(self.num_envs, 3, 1)

                # compute Jacobian transpose update
                delta_q = np.matmul(jacobians.transpose(0, 2, 1), error)

                self.model.joint_q = wp.array(self.model.joint_q.numpy(
                ) + self.step_size * delta_q.flatten(), dtype=wp.float32, device=self.device, requires_grad=True)

                # render
                if (render):
                    self.render_time += self.frame_dt

                    self.renderer.begin_frame(self.render_time)
                    self.renderer.render(self.state)
                    self.renderer.render_points(
                        "targets", targets, radius=0.05)
                    self.renderer.render_points("ee_pos", ee_pos, radius=0.05)
                    self.renderer.end_frame()

                    self.renderer.save()

                    print("iter:", iter, "error:", error.mean())

        return 0


profile = False

if profile:

    env_count = 2
    env_times = []
    env_size = []

    for i in range(15):

        robot = Robot(render=False, num_envs=env_count)
        steps_per_second = robot.run(render=False)

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
    plt.xscale('log')
    plt.xlabel("Number of Envs")
    plt.yscale('log')
    plt.ylabel("Steps/Second")
    plt.show()

else:

    robot = Robot(render=True, num_envs=10)
    robot.run()
