# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Trajectory Optimization
#
# Shows how to optimize torque trajectories for a simple planar environment
# using Warp's provided Adam optimizer.
#
###########################################################################


import os

import numpy as np

import warp as wp
import warp.sim
import warp.sim.render
from warp.optim import Adam

wp.init()


@wp.kernel
def loss_l2(
    states: wp.array2d(dtype=wp.float32), targets: wp.array2d(dtype=wp.float32), loss: wp.array(dtype=wp.float32)
):
    i, j = wp.tid()
    diff = states[i, j] - targets[i, j]
    l = diff * diff
    wp.atomic_add(loss, 0, l)


@wp.kernel
def apply_torque(torques: wp.array(dtype=wp.float32), start_index: int, body_f: wp.array(dtype=wp.spatial_vector)):
    fx = torques[start_index + 0]
    fz = torques[start_index + 1]
    body_f[0] = wp.spatial_vector(0.0, 0.0, 0.0, fx, 0.0, fz)


@wp.kernel
def save_state(body_q: wp.array(dtype=wp.transform), write_index: int, states: wp.array2d(dtype=wp.float32)):
    pos = wp.transform_get_translation(body_q[0])
    states[write_index, 0] = pos[0]
    states[write_index, 1] = pos[2]


class Example:
    def __init__(self, stage, device=None, verbose=False):
        self.verbose = verbose
        self.frame_dt = 1.0 / 60.0
        self.episode_frames = 100

        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.render_time = 0.0

        self.iter = 0

        builder = wp.sim.ModelBuilder()

        self.device = device

        # add planar joints
        builder = wp.sim.ModelBuilder(gravity=0.0)
        builder.add_articulation()
        b = builder.add_body(origin=wp.transform())
        builder.add_shape_box(pos=wp.vec3(0.0, 0.0, 0.0), hx=0.5, hy=0.5, hz=0.5, density=100.0, body=b)

        # compute reference trajectory
        rad = np.linspace(0.0, np.pi * 2, self.episode_frames)
        self.ref_traj = np.stack([np.cos(rad), np.sin(rad)], axis=1)

        # set initial joint configuration to first reference state
        builder.body_q[0] = wp.transform(p=[self.ref_traj[0][0], 0.0, self.ref_traj[0][1]])

        self.ref_traj = wp.array(self.ref_traj, dtype=wp.float32, device=self.device, requires_grad=True)
        self.last_traj = wp.empty_like(self.ref_traj)

        # finalize model
        self.model = builder.finalize(device, requires_grad=True)

        self.builder = builder
        self.model.ground = False

        self.dof_q = self.model.joint_coord_count
        self.dof_qd = self.model.joint_dof_count
        self.num_bodies = self.model.body_count

        self.action_dim = 2
        self.state_dim = 2

        assert self.ref_traj.shape == (self.episode_frames, self.state_dim)

        self.integrator = wp.sim.SemiImplicitIntegrator()

        # initial guess
        self.actions = wp.array(
            np.zeros(self.episode_frames * self.action_dim) * 100.0,
            dtype=wp.float32,
            device=self.device,
            requires_grad=True,
        )

        self.optimizer = Adam([self.actions], lr=1e2)
        self.loss = wp.zeros(1, dtype=wp.float32, device=self.device, requires_grad=True)

        self.renderer = wp.sim.render.SimRenderer(self.model, stage, scaling=100.0)

        # allocate sim states for trajectory
        self.states = []
        for _ in range(self.episode_frames + 1):
            self.states.append(self.model.state(requires_grad=True))

    def compute_loss(self):
        """
        Advances the system dynamics given the rigid-body state in maximal coordinates and generalized joint torques
        [body_q, body_qd, tau].
        """

        self.last_traj.zero_()

        for i in range(self.episode_frames):
            state = self.states[i]

            for _ in range(self.sim_substeps):
                next_state = self.model.state(requires_grad=True)

                wp.sim.collide(self.model, state)

                # apply generalized torques to rigid body here, instead of planar joints
                wp.launch(
                    apply_torque,
                    1,
                    inputs=[self.actions, i * self.action_dim],
                    outputs=[state.body_f],
                    device=self.device,
                )

                state = self.integrator.simulate(self.model, state, next_state, self.sim_dt, requires_grad=True)

            self.states[i + 1] = state

            # save state
            wp.launch(
                save_state,
                dim=1,
                inputs=[self.states[i + 1].body_q, i],
                outputs=[self.last_traj],
                device=self.device,
            )

        # compute loss
        wp.launch(
            loss_l2,
            dim=self.last_traj.shape,
            inputs=[self.last_traj, self.ref_traj],
            outputs=[self.loss],
            device=self.device,
        )

    def update(self):
        """Runs a single optimizer iteration"""
        self.loss.zero_()
        tape = wp.Tape()
        with tape:
            self.compute_loss()

        if self.verbose and (self.iter + 1) % 10 == 0:
            print(f"Iter {self.iter+1} Loss: {self.loss.numpy()[0]:.3f}")

        tape.backward(loss=self.loss)

        # print("action grad", self.actions.grad.numpy())
        assert not np.isnan(self.actions.grad.numpy()).any(), "NaN in gradient"

        self.optimizer.step([self.actions.grad])
        tape.zero()
        self.iter = self.iter + 1

    def render(self):
        for i in range(self.episode_frames):
            self.renderer.begin_frame(self.render_time)
            self.renderer.render(self.states[i + 1])
            self.renderer.end_frame()
            self.render_time += self.frame_dt


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    stage_path = os.path.join(os.path.dirname(__file__), "outputs/example_sim_trajopt.usd")
    example = Example(stage_path, device=wp.get_preferred_device(), verbose=True)

    # Optimize
    num_iter = 250

    for i in range(num_iter):
        example.update()

        # Render every 25 iters
        if i % 25 == 0:
            example.render()

    example.renderer.save()

    np_states = example.last_traj.numpy()
    np_ref = example.ref_traj.numpy()
    plt.plot(np_ref[:, 0], np_ref[:, 1], label="Reference Trajectory")
    plt.plot(np_states[:, 0], np_states[:, 1], label="Optimized Trajectory")
    plt.grid()
    plt.legend()
    plt.axis("equal")
    plt.show()
#
