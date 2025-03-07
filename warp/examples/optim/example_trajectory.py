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
# Example Trajectory Optimization
#
# Shows how to optimize torque trajectories for a simple planar environment
# using Warp's provided Adam optimizer.
#
###########################################################################

import numpy as np

import warp as wp
import warp.sim
import warp.sim.render
from warp.optim import Adam


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
    def __init__(self, stage_path="example_trajectory.usd", verbose=False, num_frames=100):
        self.verbose = verbose

        fps = 60
        self.frame_dt = 1.0 / fps
        self.num_frames = num_frames

        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.render_time = 0.0

        self.iter = 0

        # add planar joints
        builder = wp.sim.ModelBuilder(gravity=0.0)
        builder.add_articulation()
        b = builder.add_body(origin=wp.transform())
        builder.add_shape_sphere(pos=wp.vec3(0.0, 0.0, 0.0), radius=0.1, density=100.0, body=b)

        # compute reference trajectory
        rad = np.linspace(0.0, np.pi * 2, self.num_frames)
        self.ref_traj = np.stack([np.cos(rad), np.sin(rad)], axis=1)

        # set initial joint configuration to first reference state
        builder.body_q[0] = wp.transform(p=[self.ref_traj[0][0], 0.0, self.ref_traj[0][1]])

        self.ref_traj = wp.array(self.ref_traj, dtype=wp.float32, requires_grad=True)
        self.last_traj = wp.empty_like(self.ref_traj)

        # finalize model
        self.model = builder.finalize(requires_grad=True)

        self.builder = builder
        self.model.ground = False

        self.dof_q = self.model.joint_coord_count
        self.dof_qd = self.model.joint_dof_count
        self.num_bodies = self.model.body_count

        self.action_dim = 2
        self.state_dim = 2

        assert self.ref_traj.shape == (self.num_frames, self.state_dim)

        self.integrator = wp.sim.SemiImplicitIntegrator()

        # initial guess
        self.actions = wp.array(
            np.zeros(self.num_frames * self.action_dim) * 100.0, dtype=wp.float32, requires_grad=True
        )

        self.optimizer = Adam([self.actions], lr=1e2)
        self.loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)

        if stage_path:
            self.renderer = wp.sim.render.SimRenderer(self.model, stage_path, scaling=100.0)
        else:
            self.renderer = None

        # allocate sim states for trajectory
        self.states = []
        for _ in range(self.num_frames + 1):
            self.states.append(self.model.state())

    def forward(self):
        """
        Advances the system dynamics given the rigid-body state in maximal coordinates and generalized joint torques
        [body_q, body_qd, tau].
        """

        self.last_traj.zero_()

        for i in range(self.num_frames):
            state = self.states[i]

            for _ in range(self.sim_substeps):
                next_state = self.model.state(requires_grad=True)

                wp.sim.collide(self.model, state)

                # apply generalized torques to rigid body here, instead of planar joints
                wp.launch(apply_torque, 1, inputs=[self.actions, i * self.action_dim], outputs=[state.body_f])

                state = self.integrator.simulate(self.model, state, next_state, self.sim_dt)

            self.states[i + 1] = state

            # save state
            wp.launch(save_state, dim=1, inputs=[self.states[i + 1].body_q, i], outputs=[self.last_traj])

        # compute loss
        wp.launch(loss_l2, dim=self.last_traj.shape, inputs=[self.last_traj, self.ref_traj], outputs=[self.loss])

    def step(self):
        """Runs a single optimizer iteration"""

        with wp.ScopedTimer("step"):
            self.loss.zero_()
            tape = wp.Tape()
            with tape:
                self.forward()
            tape.backward(loss=self.loss)

            if self.verbose and (self.iter + 1) % 10 == 0:
                print(f"Iter {self.iter + 1} Loss: {self.loss.numpy()[0]:.3f}")

            assert not np.isnan(self.actions.grad.numpy()).any(), "NaN in gradient"

            self.optimizer.step([self.actions.grad])
            tape.zero()
            self.iter = self.iter + 1

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            for i in range(self.num_frames):
                self.renderer.begin_frame(self.render_time)
                self.renderer.render(self.states[i + 1])
                self.renderer.end_frame()
                self.render_time += self.frame_dt


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="example_trajectory.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_frames", type=int, default=100, help="Total number of frames per training iteration.")
    parser.add_argument("--train_iters", type=int, default=250, help="Total number of training iterations.")
    parser.add_argument("--verbose", action="store_true", help="Print out additional status messages during execution.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode, suppressing the opening of any graphical windows.",
    )

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, verbose=args.verbose, num_frames=args.num_frames)

        for i in range(args.train_iters):
            example.step()

            if i % 25 == 0:
                example.render()

        if example.renderer:
            example.renderer.save()

        np_states = example.last_traj.numpy()
        np_ref = example.ref_traj.numpy()

        if not args.headless:
            import matplotlib.pyplot as plt

            plt.plot(np_ref[:, 0], np_ref[:, 1], label="Reference Trajectory")
            plt.plot(np_states[:, 0], np_states[:, 1], label="Optimized Trajectory")
            plt.grid()
            plt.legend()
            plt.axis("equal")
            plt.show()
