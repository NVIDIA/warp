# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# Example Particle Repulsion
#
# Shows how to implement a gradient-based particle simulation.
# A Coulomb potential (with a hard cutoff) and relative velocity damping
# define pair-wise particle interactions. This simulation uses periodic
# boundary conditions to avoid boundary artifacts and instead capture
# local dynamics in a bulk material. The result is a particle sim that
# transforms ordered initial states into a regular lattice. In this case
# the initial particle configuration is the NVIDIA logo, and over time,
# after a phase transition, we see the particles arranged in an
# increasingly uniform lattice.
#
# Forces are computed from inter-particle potentials using wp.grad().
# Neighbors are found using the wp.HashGrid class, and
# wp.hash_grid_query(), wp.hash_grid_query_next() kernel methods.
#
###########################################################################

import os
import sys

import numpy as np

import warp as wp
import warp.examples

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


N_GRID = 128  # Hash grid size
A = 1.0  # Potential strength constant
EPS = 1e-6  # Small constant to avoid division by zero
DAMPING = 0.5  # Inter-particle damping coefficient
RADIUS = 3.0  # Search radius for neighbor query


@wp.func
def periodic_wrap(val: float, size: float) -> float:
    """Wrap a value to [0, size) range.

    Args:
        val: The value to wrap
        size: The size of the periodic domain

    Returns:
        The wrapped value
    """
    result = wp.mod(val, size)
    if result < 0.0:
        result += size
    return result


@wp.func
def periodic_distance(d: float, size: float) -> float:
    """Compute minimum image distance for periodic boundaries.

    Args:
        d: The Euclidean distance between the two particles
        size: The size of the periodic domain

    Returns:
        The minimum image Euclidean distance between the two particles
    """
    half_size = size * 0.5
    if d > half_size:
        d -= size
    elif d < -half_size:
        d += size
    return d


@wp.func
def compute_potential(d: wp.vec3):
    """Compute the potential energy between two particles.

    Args:
        d: The Euclidean distance between the two particles

    Returns:
        The potential energy between the two particles
    """
    return A / (wp.length(d) + EPS)


@wp.kernel(enable_backward=False)
def compute_forces(
    grid: wp.uint64,
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    f: wp.array(dtype=wp.vec3),
):
    """Compute the forces between particles in the hash grid.

    Args:
        grid: The hash grid.
        x: The positions of the particles.
        v: The velocities of the particles.
        f: The forces on the particles.
    """
    tid = wp.tid()

    # Order threads by cell
    i = wp.hash_grid_point_id(grid, tid)

    x_i = x[i]
    v_i = v[i]

    # Hash grid query is periodic
    neighbors = wp.hash_grid_query(grid, x_i, RADIUS)

    f_i = wp.vec3()

    for index in neighbors:
        if index != i:
            x_j = x[index]
            v_j = v[index]

            # Compute periodic distance (minimum image convention)
            dx = periodic_distance(x_i[0] - x_j[0], float(N_GRID))
            dy = periodic_distance(x_i[1] - x_j[1], float(N_GRID))
            d = wp.vec3(dx, dy, 0.0)

            # Compute repulsive force from potential
            f_i -= wp.grad(compute_potential)(d)

            # Damping force based on relative velocity
            v_rel = v_i - v_j
            f_i -= DAMPING * v_rel

    f[i] = f_i


@wp.kernel(enable_backward=False)
def integrate(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    f: wp.array(dtype=wp.vec3),
    dt: float,
):
    """Integrate the positions and velocities of the particles.

    Args:
        x: The positions of the particles
        v: The velocities of the particles
        f: The forces on the particles
        dt: The time step
    """
    tid = wp.tid()

    v_new = v[tid] + f[tid] * dt
    x_new = x[tid] + v_new * dt

    # Wrap positions for toroidal boundaries
    x_new = wp.vec3(periodic_wrap(x_new[0], float(N_GRID)), periodic_wrap(x_new[1], float(N_GRID)), 0.0)

    v[tid] = v_new
    x[tid] = x_new


class Example:
    def __init__(self):
        fps = 60
        self.frame_dt = 1.0 / fps
        self.sim_substeps = 32
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        particles = self.load_logo_particles(N_GRID)
        self.positions = wp.array(particles, dtype=wp.vec3)
        self.velocities = wp.zeros_like(self.positions)
        self.forces = wp.zeros_like(self.positions)

        # Hash grid data structure is 3D, but we only need 2D for this simulation
        self.grid = wp.HashGrid(N_GRID, N_GRID, 1)
        self.grid_cell_size = RADIUS

        self.use_cuda_graph = wp.get_device().is_cuda
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            wp.launch(
                compute_forces, len(self.positions), inputs=[self.grid.id, self.positions, self.velocities, self.forces]
            )
            wp.launch(
                integrate, len(self.positions), inputs=[self.positions, self.velocities, self.forces, self.sim_dt]
            )

    def step(self):
        self.grid.build(self.positions, self.grid_cell_size)

        if self.use_cuda_graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def load_logo_particles(self, grid_size: int, threshold: float = 0.5) -> np.ndarray:
        """Load the NVIDIA logo and create a particle array from black pixels.

        Args:
            grid_size: The size to resize the logo to (grid_size x grid_size).
            threshold: Pixels with normalized values below this threshold are considered black.
                    Default is 0.5 (values range from 0.0 to 1.0).

        Returns:
            np.ndarray: Array of shape (N, 3) where N is the number of black pixels.
                        Each row contains (x, y, z) grid coordinates with z=0.
        """
        # Load the NVIDIA logo
        logo_path = os.path.join(warp.examples.get_asset_directory(), "nvidia_logo.png")
        logo_image = Image.open(logo_path)

        # Resize to match the grid
        logo_resized = logo_image.resize((grid_size, grid_size))

        # Convert to numpy array and normalize to [0, 1]
        # Take first channel (grayscale or R channel)
        logo_np = np.array(logo_resized)[:, :, 0] / 255.0

        # Find coordinates of black pixels (values below threshold)
        black_pixel_coords = np.argwhere(logo_np < threshold)

        # Add z=0 column to make (N, 3) array
        num_particles = black_pixel_coords.shape[0]
        particles = np.zeros((num_particles, 3), dtype=np.float32)
        particles[:, 0:2] = black_pixel_coords.astype(np.float32)

        return particles


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--num-frames", type=int, default=4000, help="Number of frames to simulate.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode, suppressing the opening of any graphical windows.",
    )

    args = parser.parse_known_args()[0]

    # Check visualization availability early so user can cancel if needed
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
        example = Example()

        # Visualization setup
        if can_visualize:
            if matplotlib.rcParams["figure.raise_window"]:
                matplotlib.rcParams["figure.raise_window"] = False

            fig, ax = plt.subplots()
            fig.set_facecolor("black")
            ax.set_facecolor("black")
            pos = example.positions.numpy()
            scatter = ax.scatter(pos[:, 1], pos[:, 0], s=1, c="green")
            ax.set_xlim(0, N_GRID)
            ax.set_ylim(0, N_GRID)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])

        # Main simulation loop
        for _ in range(args.num_frames):
            example.step()

            if can_visualize:
                pos = example.positions.numpy()
                scatter.set_offsets(np.column_stack([pos[:, 1], pos[:, 0]]))
                plt.pause(example.frame_dt)

        # Visualization
        if can_visualize:
            plt.show()
