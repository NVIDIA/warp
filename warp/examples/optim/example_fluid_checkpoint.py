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

###########################################################################
# Example Fluid Checkpoint
#
# Shows how to implement a differentiable 2D stable-fluids solver and
# optimize the initial velocity field to form the NVIDIA logo at the end
# of the simulation. Gradient checkpointing to reduce memory usage
# is manually implemented.
#
# References:
# https://github.com/HIPS/autograd/blob/master/examples/fluidsim/fluidsim.py
#
###########################################################################

import math
import os

import numpy as np

import warp as wp
import warp.examples
import warp.optim

try:
    from PIL import Image
except ImportError as err:
    raise ImportError("This example requires the Pillow package. Please install it with 'pip install Pillow'.") from err


N_GRID = wp.constant(512)
DH = 1.0 / N_GRID  # Grid spacing
FLUID_COLUMN_WIDTH = N_GRID / 10.0


@wp.func
def cyclic_index(idx: wp.int32):
    """Helper function to index with periodic boundary conditions."""
    ret_idx = idx % N_GRID
    if ret_idx < 0:
        ret_idx += N_GRID
    return ret_idx


@wp.kernel
def fill_initial_density(density: wp.array2d(dtype=wp.float32)):
    """Initialize the density array with three bands of fluid."""
    i, j = wp.tid()

    y_pos = wp.float32(i)

    if FLUID_COLUMN_WIDTH <= y_pos < 2.0 * FLUID_COLUMN_WIDTH:
        density[i, j] = 1.0
    elif 4.5 * FLUID_COLUMN_WIDTH <= y_pos < 5.5 * FLUID_COLUMN_WIDTH:
        density[i, j] = 1.0
    elif 8.0 * FLUID_COLUMN_WIDTH <= y_pos < 9.0 * FLUID_COLUMN_WIDTH:
        density[i, j] = 1.0
    else:
        density[i, j] = 0.0


@wp.kernel
def advect(
    dt: float,
    vx: wp.array2d(dtype=float),
    vy: wp.array2d(dtype=float),
    f0: wp.array2d(dtype=float),
    f1: wp.array2d(dtype=float),
):
    """Move field f0 according to vx and vy velocities using an implicit Euler integrator."""

    i, j = wp.tid()

    center_xs = wp.float32(i) - vx[i, j] * dt
    center_ys = wp.float32(j) - vy[i, j] * dt

    # Compute indices of source cells.
    left_idx = wp.int32(wp.floor(center_xs))
    bot_idx = wp.int32(wp.floor(center_ys))

    s1 = center_xs - wp.float32(left_idx)  # Relative weight of right cell
    s0 = 1.0 - s1
    t1 = center_ys - wp.float32(bot_idx)  # Relative weight of top cell
    t0 = 1.0 - t1

    i0 = cyclic_index(left_idx)
    i1 = cyclic_index(left_idx + 1)
    j0 = cyclic_index(bot_idx)
    j1 = cyclic_index(bot_idx + 1)

    # Perform bilinear interpolation of the four cells bounding the back-in-time position
    f1[i, j] = s0 * (t0 * f0[i0, j0] + t1 * f0[i0, j1]) + s1 * (t0 * f0[i1, j0] + t1 * f0[i1, j1])


@wp.kernel
def divergence(wx: wp.array2d(dtype=float), wy: wp.array2d(dtype=float), div: wp.array2d(dtype=float)):
    """Compute div(w)."""

    i, j = wp.tid()

    div[i, j] = (
        0.5
        * (
            wx[cyclic_index(i + 1), j]
            - wx[cyclic_index(i - 1), j]
            + wy[i, cyclic_index(j + 1)]
            - wy[i, cyclic_index(j - 1)]
        )
        / DH
    )


@wp.kernel
def jacobi_iter(div: wp.array2d(dtype=float), p0: wp.array2d(dtype=float), p1: wp.array2d(dtype=float)):
    """Calculate a single Jacobi iteration for solving the pressure Poisson equation."""

    i, j = wp.tid()

    p1[i, j] = 0.25 * (
        -DH * DH * div[i, j]
        + p0[cyclic_index(i - 1), j]
        + p0[cyclic_index(i + 1), j]
        + p0[i, cyclic_index(j - 1)]
        + p0[i, cyclic_index(j + 1)]
    )


@wp.kernel
def update_velocities(
    p: wp.array2d(dtype=float),
    wx: wp.array2d(dtype=float),
    wy: wp.array2d(dtype=float),
    vx: wp.array2d(dtype=float),
    vy: wp.array2d(dtype=float),
):
    """Given p and (wx, wy), compute an 'incompressible' velocity field (vx, vy)."""

    i, j = wp.tid()

    vx[i, j] = wx[i, j] - 0.5 * (p[cyclic_index(i + 1), j] - p[cyclic_index(i - 1), j]) / DH
    vy[i, j] = wy[i, j] - 0.5 * (p[i, cyclic_index(j + 1)] - p[i, cyclic_index(j - 1)]) / DH


@wp.kernel
def compute_loss(
    actual_state: wp.array2d(dtype=float), target_state: wp.array2d(dtype=float), loss: wp.array(dtype=float)
):
    i, j = wp.tid()

    loss_value = (
        (actual_state[i, j] - target_state[i, j])
        * (actual_state[i, j] - target_state[i, j])
        / wp.float32(N_GRID * N_GRID)
    )

    wp.atomic_add(loss, 0, loss_value)


class Example:
    def __init__(self, sim_steps=1000):
        self.pressure_arrays = []
        self.wx_arrays = []
        self.wy_arrays = []
        self.vx_arrays = []
        self.vy_arrays = []
        self.density_arrays = []
        self.div_arrays = []

        # Memory usage is minimized when the segment size is approx. sqrt(sim_steps)
        self.segment_size = math.ceil(math.sqrt(sim_steps))

        # TODO: For now, let's just round up sim_steps so each segment is the same size
        self.num_segments = math.ceil(sim_steps / self.segment_size)
        self.sim_steps = self.segment_size * self.num_segments

        self.pressure_iterations = 50
        self.dt = 1.0

        # Store enough arrays to step through a segment without overwriting arrays
        # NOTE: Need an extra array to store the final time-advanced velocities and densities
        for _step in range(self.segment_size + 1):
            self.vx_arrays.append(wp.zeros((N_GRID, N_GRID), dtype=float, requires_grad=True))
            self.vy_arrays.append(wp.zeros((N_GRID, N_GRID), dtype=float, requires_grad=True))
            self.density_arrays.append(wp.zeros((N_GRID, N_GRID), dtype=float, requires_grad=True))

        for _step in range(self.segment_size):
            self.wx_arrays.append(wp.zeros((N_GRID, N_GRID), dtype=float, requires_grad=True))
            self.wy_arrays.append(wp.zeros((N_GRID, N_GRID), dtype=float, requires_grad=True))
            self.div_arrays.append(wp.zeros((N_GRID, N_GRID), dtype=float, requires_grad=True))

            for _iter in range(self.pressure_iterations):
                self.pressure_arrays.append(wp.zeros((N_GRID, N_GRID), dtype=float, requires_grad=True))

        # Allocate one more pressure array for the final time step
        self.pressure_arrays.append(wp.zeros((N_GRID, N_GRID), dtype=float, requires_grad=True))

        # Allocate memory to save the fluid state at the start of each segment
        self.segment_start_vx_arrays = []
        self.segment_start_vy_arrays = []
        self.segment_start_density_arrays = []
        self.segment_start_pressure_arrays = []

        for _segment_index in range(self.num_segments):
            self.segment_start_vx_arrays.append(wp.zeros((N_GRID, N_GRID), dtype=float))
            self.segment_start_vy_arrays.append(wp.zeros((N_GRID, N_GRID), dtype=float))
            self.segment_start_density_arrays.append(wp.zeros((N_GRID, N_GRID), dtype=float))
            self.segment_start_pressure_arrays.append(wp.zeros((N_GRID, N_GRID), dtype=float))

        # To restore previously computed gradients before calling tape.backward()
        self.vx_array_grad_saved = wp.zeros((N_GRID, N_GRID), dtype=float)
        self.vy_array_grad_saved = wp.zeros((N_GRID, N_GRID), dtype=float)
        self.density_array_grad_saved = wp.zeros((N_GRID, N_GRID), dtype=float)
        self.pressure_array_grad_saved = wp.zeros((N_GRID, N_GRID), dtype=float)

        wp.launch(fill_initial_density, (N_GRID, N_GRID), inputs=[self.density_arrays[0]])

        target_base = Image.open(os.path.join(warp.examples.get_asset_directory(), "nvidia_logo.png"))
        target_resized = target_base.resize((N_GRID, N_GRID))

        target_np = np.array(target_resized)[:, :, 0] / 255.0
        self.target_wp = wp.array(target_np, dtype=float)

        self.loss = wp.zeros((1,), dtype=float, requires_grad=True)

        self.train_rate = 0.01
        self.optimizer = warp.optim.Adam([self.vx_arrays[0].flatten(), self.vy_arrays[0].flatten()], lr=self.train_rate)

        # Capture forward/backward passes and tape.zero()
        self.use_cuda_graph = wp.get_device().is_cuda
        self.forward_graph = None
        self.backward_graph = None
        self.zero_tape_graph = None

        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.forward()
            self.forward_graph = capture.graph

            with wp.ScopedCapture() as capture:
                self.backward()
            self.backward_graph = capture.graph

            # tape.zero() launches many memsets, which can be a significant overhead for smaller problems
            with wp.ScopedCapture() as capture:
                self.tape.zero()
            self.zero_tape_graph = capture.graph

    def step(self, step_index) -> None:
        """Perform a single time step from t=step_index-1 to t=step_index.

        1. Self-advection of velocity components (store output in wx_arrays and wy_arrays)
        2. Incompressibility constraint (store output in vx_arrays and vy_arrays)
        3. Advection of density using velocities (vx_arrays, vy_arrays)
        """

        wp.launch(
            advect,
            (N_GRID, N_GRID),
            inputs=[
                self.dt,
                self.vx_arrays[step_index - 1],
                self.vy_arrays[step_index - 1],
                self.vx_arrays[step_index - 1],
            ],
            outputs=[self.wx_arrays[step_index - 1]],
        )
        wp.launch(
            advect,
            (N_GRID, N_GRID),
            inputs=[
                self.dt,
                self.vx_arrays[step_index - 1],
                self.vy_arrays[step_index - 1],
                self.vy_arrays[step_index - 1],
            ],
            outputs=[self.wy_arrays[step_index - 1]],
        )

        # Pressure projection using a few Jacobi iterations
        wp.launch(
            divergence,
            (N_GRID, N_GRID),
            inputs=[self.wx_arrays[step_index - 1], self.wy_arrays[step_index - 1]],
            outputs=[self.div_arrays[step_index - 1]],
        )

        # NOTE: Uses previous step's final pressure as the initial guess
        for k in range(self.pressure_iterations):
            input_index = self.pressure_iterations * (step_index - 1) + k
            output_index = input_index + 1

            wp.launch(
                jacobi_iter,
                (N_GRID, N_GRID),
                inputs=[self.div_arrays[step_index - 1], self.pressure_arrays[input_index]],
                outputs=[self.pressure_arrays[output_index]],
            )

        # NOTE: output_index should be self.pressure_iterations*step_index at this point
        wp.launch(
            update_velocities,
            (N_GRID, N_GRID),
            inputs=[self.pressure_arrays[output_index], self.wx_arrays[step_index - 1], self.wy_arrays[step_index - 1]],
            outputs=[self.vx_arrays[step_index], self.vy_arrays[step_index]],
        )

        wp.launch(
            advect,
            (N_GRID, N_GRID),
            inputs=[
                self.dt,
                self.vx_arrays[step_index],
                self.vy_arrays[step_index],
                self.density_arrays[step_index - 1],
            ],
            outputs=[self.density_arrays[step_index]],
        )

    def forward(self) -> None:
        """Advance the simulation forward in segments, storing the fluid state at the start of each segment.

        The loss function is also evaluated at the end of the function.
        """
        self.loss.zero_()

        for segment_index in range(self.num_segments):
            # Save start-of-segment values
            wp.copy(self.segment_start_vx_arrays[segment_index], self.vx_arrays[0])
            wp.copy(self.segment_start_vy_arrays[segment_index], self.vy_arrays[0])
            wp.copy(self.segment_start_density_arrays[segment_index], self.density_arrays[0])
            wp.copy(self.segment_start_pressure_arrays[segment_index], self.pressure_arrays[0])

            for t in range(1, self.segment_size + 1):
                # sim_t = (segment_index - 1) * self.segment_size + t
                self.step(t)

            # Set the initial conditions for the next segment
            if segment_index < self.num_segments - 1:
                wp.copy(self.vx_arrays[0], self.vx_arrays[-1])
                wp.copy(self.vy_arrays[0], self.vy_arrays[-1])
                wp.copy(self.density_arrays[0], self.density_arrays[-1])
                wp.copy(self.pressure_arrays[0], self.pressure_arrays[-1])

        wp.launch(
            compute_loss,
            (N_GRID, N_GRID),
            inputs=[self.density_arrays[self.segment_size], self.target_wp],
            outputs=[self.loss],
        )

    def backward(self) -> None:
        """Compute the adjoints using a checkpointing approach.

        Starting from the final segment, the forward pass for the segment is
        repeated, this time recording the kernel launches onto a tape. Any
        previously computed adjoints are restored prior to evaluating the
        backward pass for the segment. This process is repeated until the
        adjoints of the initial state have been calculated.
        """

        for segment_index in range(self.num_segments - 1, -1, -1):
            # Restore state at the start of the segment
            wp.copy(self.vx_arrays[0], self.segment_start_vx_arrays[segment_index])
            wp.copy(self.vy_arrays[0], self.segment_start_vy_arrays[segment_index])
            wp.copy(self.density_arrays[0], self.segment_start_density_arrays[segment_index])
            wp.copy(self.pressure_arrays[0], self.segment_start_pressure_arrays[segment_index])

            # Record operations on tape
            with wp.Tape() as self.tape:
                for t in range(1, self.segment_size + 1):
                    self.step(t)

            if segment_index == self.num_segments - 1:
                self.loss.grad.fill_(1.0)

                wp.launch(
                    compute_loss,
                    (N_GRID, N_GRID),
                    inputs=[self.density_arrays[self.segment_size], self.target_wp],
                    outputs=[self.loss],
                    adj_inputs=[self.density_arrays[self.segment_size].grad, None],
                    adj_outputs=[self.loss.grad],
                    adjoint=True,
                )
            else:
                # Fill in previously computed gradients from the last segment
                wp.copy(self.vx_arrays[-1].grad, self.vx_array_grad_saved)
                wp.copy(self.vy_arrays[-1].grad, self.vy_array_grad_saved)
                wp.copy(self.density_arrays[-1].grad, self.density_array_grad_saved)
                wp.copy(self.pressure_arrays[-1].grad, self.pressure_array_grad_saved)

            self.tape.backward()

            if segment_index > 0:
                # Save the gradients to variables and zero-out the gradients for the next segment
                wp.copy(self.vx_array_grad_saved, self.vx_arrays[0].grad)
                wp.copy(self.vy_array_grad_saved, self.vy_arrays[0].grad)
                wp.copy(self.density_array_grad_saved, self.density_arrays[0].grad)
                wp.copy(self.pressure_array_grad_saved, self.pressure_arrays[0].grad)

                self.tape.zero()

        # Done with backward pass, we're interested in self.vx_arrays[0].grad and self.vy_arrays[0].grad


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--num_frames", type=int, default=1000, help="Number of frames to simulate before computing loss."
    )
    parser.add_argument("--train_iters", type=int, default=50, help="Total number of training iterations.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode, suppressing the opening of any graphical windows.",
    )

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(sim_steps=args.num_frames)

        wp.synchronize_device()

        if (device := wp.get_device()).is_cuda:
            print(f"Current memory usage: {wp.get_mempool_used_mem_current(device) / (1024 * 1024 * 1024):.4f} GiB")

        # Main training loop
        for train_iter in range(args.train_iters):
            if example.forward_graph:
                wp.capture_launch(example.forward_graph)
            else:
                example.forward()

            if example.backward_graph:
                wp.capture_launch(example.backward_graph)
            else:
                example.backward()

            example.optimizer.step([example.vx_arrays[0].grad.flatten(), example.vy_arrays[0].grad.flatten()])

            # Clear grad arrays for next iteration
            if example.zero_tape_graph:
                wp.capture_launch(example.zero_tape_graph)
            else:
                example.tape.zero()

            print(f"Iteration {train_iter:05d} loss: {example.loss.numpy()[0]:.6f}")

        if not args.headless:
            import matplotlib
            import matplotlib.pyplot as plt

            if matplotlib.rcParams["figure.raise_window"]:
                matplotlib.rcParams["figure.raise_window"] = False

            fig, ax = plt.subplots()
            image = ax.imshow(example.density_arrays[-1].numpy(), cmap="viridis", origin="lower", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title("Fluid Density")

            # Run the final simulation to the stop time
            for _ in range(args.num_frames):
                example.step(1)
                # Swap pointers
                (example.vx_arrays[0], example.vx_arrays[1]) = (example.vx_arrays[1], example.vx_arrays[0])
                (example.vy_arrays[0], example.vy_arrays[1]) = (example.vy_arrays[1], example.vy_arrays[0])
                (example.density_arrays[0], example.density_arrays[1]) = (
                    example.density_arrays[1],
                    example.density_arrays[0],
                )
                (example.pressure_arrays[0], example.pressure_arrays[example.pressure_iterations]) = (
                    example.pressure_arrays[example.pressure_iterations],
                    example.pressure_arrays[0],
                )

                image.set_data(example.density_arrays[0].numpy())
                plt.pause(0.001)

            plt.show()
