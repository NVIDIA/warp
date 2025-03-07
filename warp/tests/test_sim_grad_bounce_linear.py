# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np

import warp as wp
import warp.optim
import warp.sim
import warp.sim.render
from warp.tests.unittest_utils import *


@wp.kernel
def update_trajectory_kernel(
    trajectory: wp.array(dtype=wp.vec3),
    q: wp.array(dtype=wp.transform),
    time_step: wp.int32,
    q_idx: wp.int32,
):
    trajectory[time_step] = wp.transform_get_translation(q[q_idx])


@wp.kernel
def trajectory_loss_kernel(
    trajectory: wp.array(dtype=wp.vec3f),
    target_trajectory: wp.array(dtype=wp.vec3f),
    loss: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    diff = trajectory[tid] - target_trajectory[tid]
    distance_loss = wp.dot(diff, diff)
    wp.atomic_add(loss, 0, distance_loss)


class BallBounceLinearTest:
    def __init__(self, gravity=True, rendering=False):
        # Ball bouncing scenario inspired by https://github.com/NVIDIA/warp/issues/349
        self.fps = 30
        self.num_frames = 60
        self.sim_substeps = 20  # XXX need to use enough substeps to achieve smooth gradients
        self.frame_dt = 1.0 / self.fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_duration = self.num_frames * self.frame_dt
        self.sim_steps = int(self.sim_duration // self.sim_dt)

        self.target_force_linear = 100.0

        if gravity:
            builder = wp.sim.ModelBuilder(up_vector=wp.vec3(0, 0, 1))
        else:
            builder = wp.sim.ModelBuilder(gravity=0.0, up_vector=wp.vec3(0, 0, 1))

        b = builder.add_body(origin=wp.transform((0.5, 0.0, 1.0), wp.quat_identity()), name="ball")
        builder.add_shape_sphere(
            body=b, radius=0.1, density=100.0, ke=2000.0, kd=10.0, kf=200.0, mu=0.2, thickness=0.01
        )
        builder.set_ground_plane(ke=10, kd=10, kf=0.0, mu=0.2)
        self.model = builder.finalize(requires_grad=True)

        self.time = np.linspace(0, self.sim_duration, self.sim_steps)

        self.integrator = wp.sim.SemiImplicitIntegrator()
        if rendering:
            self.renderer = wp.sim.render.SimRendererOpenGL(self.model, "ball_bounce_linear")
        else:
            self.renderer = None

        self.loss = wp.array([0], dtype=wp.float32, requires_grad=True)
        self.states = [self.model.state() for _ in range(self.sim_steps + 1)]
        self.target_states = [self.model.state() for _ in range(self.sim_steps + 1)]

        self.target_force = wp.array([0.0, 0.0, 0.0, 0.0, self.target_force_linear, 0.0], dtype=wp.spatial_vectorf)

        self.trajectory = wp.empty(len(self.time), dtype=wp.vec3, requires_grad=True)
        self.target_trajectory = wp.empty(len(self.time), dtype=wp.vec3)

    def _reset(self):
        self.loss = wp.array([0], dtype=wp.float32, requires_grad=True)

    def generate_target_trajectory(self):
        for i in range(self.sim_steps):
            curr_state = self.target_states[i]
            next_state = self.target_states[i + 1]
            curr_state.clear_forces()
            if i == 0:
                wp.copy(curr_state.body_f, self.target_force, dest_offset=0, src_offset=0, count=1)
            wp.sim.collide(self.model, curr_state)
            self.integrator.simulate(self.model, curr_state, next_state, self.sim_dt)
            wp.launch(kernel=update_trajectory_kernel, dim=1, inputs=[self.target_trajectory, curr_state.body_q, i, 0])

    def forward(self, force: wp.array):
        for i in range(self.sim_steps):
            curr_state = self.states[i]
            next_state = self.states[i + 1]
            curr_state.clear_forces()
            if i == 0:
                wp.copy(curr_state.body_f, force, dest_offset=0, src_offset=0, count=1)
            wp.sim.collide(self.model, curr_state)
            self.integrator.simulate(self.model, curr_state, next_state, self.sim_dt)
            wp.launch(kernel=update_trajectory_kernel, dim=1, inputs=[self.trajectory, curr_state.body_q, i, 0])

            if self.renderer:
                self.renderer.begin_frame(self.time[i])
                self.renderer.render(curr_state)
                self.renderer.end_frame()

    def step(self, force: wp.array):
        self.tape = wp.Tape()
        self._reset()
        with self.tape:
            self.forward(force)
            wp.launch(
                kernel=trajectory_loss_kernel,
                dim=len(self.trajectory),
                inputs=[self.trajectory, self.target_trajectory, self.loss],
            )
        self.tape.backward(self.loss)
        force_grad = force.grad.numpy()[0, 4]
        self.tape.zero()

        return self.loss.numpy()[0], force_grad

    def evaluate(self, num_samples, plot_results=False):
        forces = np.linspace(0, self.target_force_linear * 2, num_samples)
        losses = np.zeros_like(forces)
        grads = np.zeros_like(forces)

        for i, fx in enumerate(forces):
            force = wp.array([[0.0, 0.0, 0.0, 0.0, fx, 0.0]], dtype=wp.spatial_vectorf, requires_grad=True)
            losses[i], grads[i] = self.step(force)
            if plot_results:
                print(f"Iteration {i + 1}/{num_samples}")
                print(f"Force: {fx:.2f}, Loss: {losses[i]:.6f}, Grad: {grads[i]:.6f}")

            assert np.isfinite(losses[i])
            assert np.isfinite(grads[i])
            if i > 0:
                assert grads[i] >= grads[i - 1]

        if plot_results:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Plot the loss curve
            ax1.plot(forces, losses, label="Loss")
            ax1.set_xlabel("Force")
            ax1.set_ylabel("Loss")
            ax1.set_title("Loss vs Force")
            ax1.legend()

            # Make sure the grads are not too large
            grads = np.clip(grads, -1e4, 1e4)

            # Plot the gradient curve
            ax2.plot(forces, grads, label="Gradient", color="orange")
            ax2.set_xlabel("Force")
            ax2.set_ylabel("Gradient")
            ax2.set_title("Gradient vs Force")
            ax2.legend()

            plt.suptitle("Loss and Gradient vs Force")
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()

        return losses, grads


def test_sim_grad_bounce_linear(test, device):
    with wp.ScopedDevice(device):
        model = BallBounceLinearTest()
        model.generate_target_trajectory()

        num_samples = 20
        losses, grads = model.evaluate(num_samples=num_samples)
        # gradients must approximate linear behavior with zero crossing in the middle
        test.assertTrue(np.abs(grads[1:] - grads[:-1]).max() < 1.1)
        test.assertTrue(np.all(grads[: num_samples // 2] <= 0.0))
        test.assertTrue(np.all(grads[num_samples // 2 :] >= 0.0))
        # losses must follow a parabolic behavior
        test.assertTrue(np.allclose(losses[: num_samples // 2], losses[num_samples // 2 :][::-1], atol=1.0))
        diffs = losses[1:] - losses[:-1]
        test.assertTrue(np.all(diffs[: num_samples // 2 - 1] <= 0.0))
        test.assertTrue(np.all(diffs[num_samples // 2 - 1 :] >= 0.0))
        # second derivative must be constant positive
        diffs2 = diffs[1:] - diffs[:-1]
        test.assertTrue(np.allclose(diffs2, diffs2[0], atol=1e-2))
        test.assertTrue(np.all(diffs2 >= 0.0))


class TestSimGradBounceLinear(unittest.TestCase):
    pass


devices = get_test_devices("basic")
add_function_test(TestSimGradBounceLinear, "test_sim_grad_bounce_linear", test_sim_grad_bounce_linear, devices=devices)

if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=True)
