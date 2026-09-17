# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A checkpointed fluid solver with a custom pressure-solve backward pass.

This variant uses the simulation and checkpointing scheme from
``example_fluid_checkpoint.py``. For each pressure solve, the
:class:`warp.Tape` records one ``JacobiSolver`` callback instead of every
Jacobi iteration.

Each damped iteration has the form

    p_next = J p + c div

Here, ``J = (1 - omega) I + omega A``, ``A`` is the four-neighbor average,
and ``c = -omega * DH**2 / 4``, where ``omega = JACOBI_RELAXATION`` and
``DH`` is the grid spacing. Because ``J`` and ``c`` are fixed, the backward
pass computes the gradients of ``p`` and ``div`` directly from the gradient of
``p_next``.

The solver stores one pressure field at each simulation-step boundary within a
checkpoint segment and reuses two scratch grids for the intermediate Jacobi
iterates and their gradients. Its per-segment pressure storage is
``O(segment_size)`` regardless of the number of Jacobi iterations. The forward
and backward passes still execute every configured iteration.

The callback differentiates the finite iteration sequence, including its warm
start, exactly up to roundoff.

Usage:
    python example_fluid_checkpoint_custom_backward.py --headless

Run ``python example_fluid_checkpoint_custom_backward.py --help`` for
available options.
"""

import os
import sys

import numpy as np

import warp as wp
import warp.examples
import warp.optim

try:
    from PIL import Image
except ImportError as err:
    raise ImportError("This example requires the Pillow package. Please install it with 'pip install Pillow'.") from err

try:
    import matplotlib
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


N_GRID = wp.constant(512)
DH = 1.0 / N_GRID  # Grid spacing
# Damping makes the periodic checkerboard mode decay.
JACOBI_RELAXATION = wp.constant(2.0 / 3.0)
FLUID_COLUMN_WIDTH = N_GRID / 10.0


def estimate_segment_size(sim_steps: int) -> int:
    """Estimate a checkpoint segment size from the number of stored grids.

    For segment size ``S``, velocity and density values and gradients
    contribute ``6 * S`` grids, advection and projection intermediates
    contribute another ``6 * S``, and timestep-boundary pressure values and
    gradients contribute ``2 * S``. The solver reuses two scratch grids, so
    their storage does not depend on ``S``. Saving four start-state grids per
    segment adds ``4 * ceil(sim_steps / S)``. The estimate is

        ``14 * S + 4 * ceil(sim_steps / S)``.

    The function checks every valid segment size and returns the one with the
    smallest estimate. The count covers simulation arrays only; it excludes
    runtime and allocator overhead. Other checkpoint schedules may use less
    memory.
    """

    def estimate_storage(segment_size: int) -> int:
        num_segments = (sim_steps + segment_size - 1) // segment_size
        return 14 * segment_size + 4 * num_segments

    # When storage ties, prefer fewer segments to reduce checkpoint transfers
    # and the number of Tape objects.
    return min(range(1, sim_steps + 1), key=lambda size: (estimate_storage(size), -size))


@wp.func
def cyclic_index(idx: int):
    """Helper function to index with periodic boundary conditions."""
    ret_idx = idx % N_GRID
    if ret_idx < 0:
        ret_idx += N_GRID
    return ret_idx


@wp.kernel
def fill_initial_density(density: wp.array2d[float]):
    """Initialize the density array with three bands of fluid."""
    i, j = wp.tid()

    y_pos = float(i)

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
    vx: wp.array2d[float],
    vy: wp.array2d[float],
    f0: wp.array2d[float],
    f1: wp.array2d[float],
):
    """Move field f0 according to vx and vy velocities using an implicit Euler integrator."""

    i, j = wp.tid()

    center_xs = float(i) - vx[i, j] * dt
    center_ys = float(j) - vy[i, j] * dt

    # Compute indices of source cells.
    left_idx = int(wp.floor(center_xs))
    bot_idx = int(wp.floor(center_ys))

    s1 = center_xs - float(left_idx)  # Relative weight of right cell
    s0 = 1.0 - s1
    t1 = center_ys - float(bot_idx)  # Relative weight of top cell
    t0 = 1.0 - t1

    i0 = cyclic_index(left_idx)
    i1 = cyclic_index(left_idx + 1)
    j0 = cyclic_index(bot_idx)
    j1 = cyclic_index(bot_idx + 1)

    # Perform bilinear interpolation of the four cells bounding the back-in-time position
    f1[i, j] = s0 * (t0 * f0[i0, j0] + t1 * f0[i0, j1]) + s1 * (t0 * f0[i1, j0] + t1 * f0[i1, j1])


@wp.kernel
def divergence(wx: wp.array2d[float], wy: wp.array2d[float], div: wp.array2d[float]):
    """Compute backward-difference divergence, paired with the forward pressure gradient."""

    i, j = wp.tid()

    div[i, j] = (wx[i, j] - wx[cyclic_index(i - 1), j] + wy[i, j] - wy[i, cyclic_index(j - 1)]) / DH


@wp.kernel(enable_backward=False)
def jacobi_iter(div: wp.array2d[float], p0: wp.array2d[float], p1: wp.array2d[float]):
    """Calculate a single damped Jacobi iteration for the pressure Poisson equation."""

    i, j = wp.tid()

    p1[i, j] = (1.0 - JACOBI_RELAXATION) * p0[i, j] + 0.25 * JACOBI_RELAXATION * (
        -DH * DH * div[i, j]
        + p0[cyclic_index(i - 1), j]
        + p0[cyclic_index(i + 1), j]
        + p0[i, cyclic_index(j - 1)]
        + p0[i, cyclic_index(j + 1)]
    )


@wp.kernel(enable_backward=False)
def jacobi_iter_adjoint(div_grad: wp.array2d[float], p1_grad: wp.array2d[float], p0_grad: wp.array2d[float]):
    """Apply the adjoint of one damped Jacobi iteration.

    Because ``div`` enters the forward iteration pointwise with coefficient
    ``-0.25 * JACOBI_RELAXATION * DH**2``, its adjoint adds that coefficient
    times ``p1_grad`` to ``div_grad``.

    Each ``p0`` cell affects ``p1`` at the same cell and at its four neighbors.
    Periodic boundaries make these neighbor relationships symmetric, so the
    chain rule applies the same weighted stencil to ``p1_grad`` to compute
    ``p0_grad``. Each thread gathers its five contributions without atomic
    operations.
    """
    i, j = wp.tid()

    div_grad[i, j] += -0.25 * JACOBI_RELAXATION * DH * DH * p1_grad[i, j]
    p0_grad[i, j] = (1.0 - JACOBI_RELAXATION) * p1_grad[i, j] + 0.25 * JACOBI_RELAXATION * (
        p1_grad[cyclic_index(i - 1), j]
        + p1_grad[cyclic_index(i + 1), j]
        + p1_grad[i, cyclic_index(j - 1)]
        + p1_grad[i, cyclic_index(j + 1)]
    )


@wp.kernel(enable_backward=False)
def accumulate_pressure_grad(
    pressure_grad_increment: wp.array2d[float],
    pressure_grad: wp.array2d[float],
):
    i, j = wp.tid()
    pressure_grad[i, j] += pressure_grad_increment[i, j]


# Record one Tape callback for the whole solve and reuse its scratch grids.
class JacobiSolver:
    """Solve for pressure with two reusable scratch grids and a custom Tape adjoint.

    The solver allocates its scratch grids on the current device and reuses them
    for each solve and backward callback on the current stream. The inputs and
    output must be separate ``N_GRID`` by ``N_GRID`` grids on the same device.

    Args:
        iterations: Positive number of Jacobi iterations per pressure solve.
    """

    def __init__(self, iterations: int):
        if iterations < 1:
            raise ValueError("The number of Jacobi iterations must be positive.")
        self.iterations = iterations
        self.scratch0 = wp.empty((N_GRID, N_GRID), dtype=float)
        self.scratch1 = wp.empty((N_GRID, N_GRID), dtype=float)

    def solve(
        self, div: wp.array2d[float], p0: wp.array2d[float], p1: wp.array2d[float], tape: wp.Tape | None = None
    ) -> None:
        """Advance ``p0`` to ``p1``, optionally recording an adjoint on ``tape``.

        Pass the enclosing Tape explicitly when recording a differentiable
        simulation. The callback uses only the input and output gradients;
        all intermediate pressures can be overwritten.
        """
        iterations = self.iterations
        # record_tape=False skips the Tape's array access checks, so mark the
        # custom operation's inputs and output here.
        if tape is not None and wp.config.verify_autograd_array_access:
            div.mark_read()
            p0.mark_read()
            p1.mark_write()

        pressure = p0
        scratch0, scratch1 = self.scratch0, self.scratch1
        for k in range(iterations):
            next_pressure = p1 if k == iterations - 1 else scratch0
            wp.launch(
                jacobi_iter,
                (N_GRID, N_GRID),
                inputs=[div, pressure],
                outputs=[next_pressure],
                device=p0.device,
                record_tape=False,
            )
            pressure = next_pressure
            scratch0, scratch1 = scratch1, scratch0

        if tape is not None:

            def pressure_solve_backward():
                # Copy the seed so the output gradient stays available to other
                # users. The scratch contents from earlier solves are irrelevant.
                g0, g1 = self.scratch0, self.scratch1
                wp.copy(g0, p1.grad)
                for _ in range(iterations):
                    wp.launch(
                        jacobi_iter_adjoint,
                        (N_GRID, N_GRID),
                        inputs=[div.grad, g0],
                        outputs=[g1],
                        device=p0.device,
                        record_tape=False,
                    )
                    g0, g1 = g1, g0

                # The initial guess is the previous step's final pressure.
                # Accumulate its gradient to preserve that path through time.
                wp.launch(
                    accumulate_pressure_grad,
                    (N_GRID, N_GRID),
                    inputs=[g0],
                    outputs=[p0.grad],
                    device=p0.device,
                    record_tape=False,
                )

            # Register every differentiable boundary array so tape.zero() also
            # clears gradients that are only accessed by this callback.
            tape.record_func(pressure_solve_backward, arrays=[div, p0, p1])


@wp.kernel
def update_velocities(
    p: wp.array2d[float],
    wx: wp.array2d[float],
    wy: wp.array2d[float],
    vx: wp.array2d[float],
    vy: wp.array2d[float],
):
    """Subtract the forward pressure gradient, paired with backward divergence."""

    i, j = wp.tid()

    vx[i, j] = wx[i, j] - (p[cyclic_index(i + 1), j] - p[i, j]) / DH
    vy[i, j] = wy[i, j] - (p[i, cyclic_index(j + 1)] - p[i, j]) / DH


@wp.kernel
def compute_loss(actual_state: wp.array2d[float], target_state: wp.array2d[float], loss: wp.array[float]):
    i, j = wp.tid()

    loss_value = (
        (actual_state[i, j] - target_state[i, j]) * (actual_state[i, j] - target_state[i, j]) / float(N_GRID * N_GRID)
    )

    wp.atomic_add(loss, 0, loss_value)


class Example:
    def __init__(self, sim_steps=1000, pressure_iterations=50, segment_size=None):
        if sim_steps < 1:
            raise ValueError("The number of simulation steps must be positive.")
        if pressure_iterations < 1:
            raise ValueError("The number of Jacobi iterations must be positive.")
        if segment_size is not None and not 1 <= segment_size <= sim_steps:
            raise ValueError("The segment size must be between one and the number of simulation steps.")

        self.pressure_arrays = []
        self.wx_arrays = []
        self.wy_arrays = []
        self.vx_arrays = []
        self.vy_arrays = []
        self.density_arrays = []
        self.div_arrays = []

        # To keep the example compact, the pressure solve always runs the configured
        # number of iterations instead of testing for convergence.
        self.pressure_solver = JacobiSolver(pressure_iterations)

        if segment_size is None:
            segment_size = estimate_segment_size(sim_steps)
        self.segment_size = segment_size
        self.segment_lengths = [min(segment_size, sim_steps - start) for start in range(0, sim_steps, segment_size)]
        self.num_segments = len(self.segment_lengths)
        self.sim_steps = sim_steps
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

            # Keep only timestep boundary pressures and their gradients. The
            # solver shares its two scratch grids across the entire segment.
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

    def step(self, step_index, tape: wp.Tape | None = None) -> None:
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

        # Compute the pressure projection with the configured number of Jacobi iterations.
        wp.launch(
            divergence,
            (N_GRID, N_GRID),
            inputs=[self.wx_arrays[step_index - 1], self.wy_arrays[step_index - 1]],
            outputs=[self.div_arrays[step_index - 1]],
        )

        # Use the previous step's final pressure as the initial guess and record
        # one custom backward operation for the entire solve.
        self.pressure_solver.solve(
            self.div_arrays[step_index - 1],
            self.pressure_arrays[step_index - 1],
            self.pressure_arrays[step_index],
            tape,
        )

        wp.launch(
            update_velocities,
            (N_GRID, N_GRID),
            inputs=[self.pressure_arrays[step_index], self.wx_arrays[step_index - 1], self.wy_arrays[step_index - 1]],
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
        """Advance the simulation in segments and compute the loss.

        The method saves the fluid state at the start of each segment and evaluates
        the loss after the final segment.
        """
        self.loss.zero_()

        for segment_index, segment_steps in enumerate(self.segment_lengths):
            # Save start-of-segment values
            wp.copy(self.segment_start_vx_arrays[segment_index], self.vx_arrays[0])
            wp.copy(self.segment_start_vy_arrays[segment_index], self.vy_arrays[0])
            wp.copy(self.segment_start_density_arrays[segment_index], self.density_arrays[0])
            wp.copy(self.segment_start_pressure_arrays[segment_index], self.pressure_arrays[0])

            for t in range(1, segment_steps + 1):
                self.step(t)

            # Set the initial conditions for the next segment
            if segment_index < self.num_segments - 1:
                wp.copy(self.vx_arrays[0], self.vx_arrays[segment_steps])
                wp.copy(self.vy_arrays[0], self.vy_arrays[segment_steps])
                wp.copy(self.density_arrays[0], self.density_arrays[segment_steps])
                wp.copy(self.pressure_arrays[0], self.pressure_arrays[segment_steps])

        final_step_index = self.segment_lengths[-1]
        wp.launch(
            compute_loss,
            (N_GRID, N_GRID),
            inputs=[self.density_arrays[final_step_index], self.target_wp],
            outputs=[self.loss],
        )

    def backward(self) -> None:
        """Compute adjoints by replaying checkpointed segments in reverse order.

        For each segment, restore its saved starting state and record the forward pass
        on a Tape. Run the Tape backward, then carry the start-state adjoints into the
        preceding segment. Continue through the first segment to compute the initial
        state's adjoints.
        """

        for segment_index in range(self.num_segments - 1, -1, -1):
            segment_steps = self.segment_lengths[segment_index]

            # Restore state at the start of the segment
            wp.copy(self.vx_arrays[0], self.segment_start_vx_arrays[segment_index])
            wp.copy(self.vy_arrays[0], self.segment_start_vy_arrays[segment_index])
            wp.copy(self.density_arrays[0], self.segment_start_density_arrays[segment_index])
            wp.copy(self.pressure_arrays[0], self.segment_start_pressure_arrays[segment_index])

            # Record operations on tape
            with wp.Tape() as self.tape:
                for t in range(1, segment_steps + 1):
                    self.step(t, self.tape)

            if segment_index == self.num_segments - 1:
                self.loss.grad.fill_(1.0)

                wp.launch(
                    compute_loss,
                    (N_GRID, N_GRID),
                    inputs=[self.density_arrays[segment_steps], self.target_wp],
                    outputs=[self.loss],
                    adj_inputs=[self.density_arrays[segment_steps].grad, None],
                    adj_outputs=[self.loss.grad],
                    adjoint=True,
                )
            else:
                # Fill in previously computed gradients from the last segment
                wp.copy(self.vx_arrays[segment_steps].grad, self.vx_array_grad_saved)
                wp.copy(self.vy_arrays[segment_steps].grad, self.vy_array_grad_saved)
                wp.copy(self.density_arrays[segment_steps].grad, self.density_array_grad_saved)
                wp.copy(self.pressure_arrays[segment_steps].grad, self.pressure_array_grad_saved)

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

    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--num-frames", type=int, default=1000, help="Number of frames to simulate before computing loss."
    )
    parser.add_argument("--train-iters", type=int, default=50, help="Total number of training iterations.")
    parser.add_argument(
        "--pressure-iterations",
        type=int,
        default=50,
        help="Fixed number of damped Jacobi iterations per pressure solve.",
    )
    parser.add_argument(
        "--segment-size",
        type=int,
        default=None,
        help="Maximum steps per checkpoint segment. If omitted, estimate the size from the number of stored grids.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode, suppressing the opening of any graphical windows.",
    )

    args = parser.parse_known_args()[0]
    if args.num_frames < 1 or args.pressure_iterations < 1:
        parser.error("--num-frames and --pressure-iterations must be positive")
    if args.segment_size is not None and not 1 <= args.segment_size <= args.num_frames:
        parser.error("--segment-size must be between one and --num-frames")

    # Check visualization availability early (before training) so user can cancel if needed
    can_visualize = False
    if not args.headless:
        if not MATPLOTLIB_AVAILABLE:
            print(
                "Warning: matplotlib not found. Skipping visualization. "
                "Install matplotlib to enable visualization: pip install matplotlib",
                file=sys.stderr,
            )
        # matplotlib is available, check if backend supports interactive display
        elif matplotlib.get_backend().lower() == "agg":
            print(
                "Warning: No interactive matplotlib backend available. Skipping visualization. "
                "Install python3-tk (Linux) or PySide6 to enable visualization.",
                file=sys.stderr,
            )
        else:
            can_visualize = True

    with wp.ScopedDevice(args.device):
        example = Example(
            sim_steps=args.num_frames,
            pressure_iterations=args.pressure_iterations,
            segment_size=args.segment_size,
        )

        print(
            f"Checkpoint schedule: {example.sim_steps} steps in {example.num_segments} segments "
            f"of at most {example.segment_size} steps."
        )
        device = wp.get_device()

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

            # CUDA graph executables are created lazily on their first launch,
            # so report memory after the first complete optimization iteration.
            report_memory = train_iter == 0 and device.is_cuda and device.is_mempool_enabled
            if report_memory:
                wp.synchronize_device()

            print(f"Iteration {train_iter:05d} loss: {example.loss.numpy()[0]:.6f}")

            if report_memory:
                print(
                    "CUDA mempool after first optimization iteration:\n"
                    f"  Current usage: {wp.get_mempool_used_mem_current(device) / 2**20:.1f} MiB\n"
                    f"  Peak usage: {wp.get_mempool_used_mem_high(device) / 2**20:.1f} MiB"
                )

        # Visualization
        if can_visualize:
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
                (example.pressure_arrays[0], example.pressure_arrays[1]) = (
                    example.pressure_arrays[1],
                    example.pressure_arrays[0],
                )

                image.set_data(example.density_arrays[0].numpy())
                plt.pause(0.001)

            plt.show()
