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
# Example N-Body
#
# Shows how to simulate an N-Body gravitational problem using an all-pairs
# approach with Warp tile primitives.
#
# References:
#   L. Nyland, M. Harris, and J. Prins. "Fast N-Body Simulation with
#   CUDA" in GPU Gems 3. H. Nguyen, Addison-Wesley Professional, 2007.
#   https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-31-fast-n-body-simulation-cuda
#
###########################################################################

import argparse

import numpy as np

import warp as wp

wp.init()

DT = wp.constant(0.016)
SOFTENING_SQ = wp.constant(0.1**2)  # Softening factor for numerical stability
TILE_SIZE = wp.constant(64)
PARTICLE_MASS = wp.constant(1.0)


@wp.func
def body_body_interaction(p0: wp.vec3, pi: wp.vec3):
    """Return the acceleration of the particle at position `p0` due to the
    particle at position `pi`."""
    r = pi - p0

    dist_sq = wp.length_sq(r) + SOFTENING_SQ

    inv_dist = 1.0 / wp.sqrt(dist_sq)
    inv_dist_cubed = inv_dist * inv_dist * inv_dist

    acc = PARTICLE_MASS * inv_dist_cubed * r

    return acc


@wp.kernel
def integrate_bodies_tiled(
    old_position: wp.array(dtype=wp.vec3),
    velocity: wp.array(dtype=wp.vec3),
    new_position: wp.array(dtype=wp.vec3),
    num_bodies: int,
):
    i = wp.tid()

    p0 = old_position[i]

    accel = wp.vec3(0.0, 0.0, 0.0)

    for k in range(num_bodies / TILE_SIZE):
        k_tile = wp.tile_load(old_position, shape=TILE_SIZE, offset=k * TILE_SIZE)
        for idx in range(TILE_SIZE):
            pi = k_tile[idx]
            accel += body_body_interaction(p0, pi)

    # Advance the velocity one timestep (in-place)
    velocity[i] = velocity[i] + accel * DT

    # Advance the positions (using a second array)
    new_position[i] = old_position[i] + DT * velocity[i]


class Example:
    def __init__(self, headless=False, num_bodies=16384):
        self.num_bodies = num_bodies

        rng = np.random.default_rng(42)

        # Sample the surface of a sphere
        phi = np.arccos(1.0 - 2.0 * rng.uniform(low=0.0, high=1.0, size=self.num_bodies))
        theta = rng.uniform(low=0.0, high=2.0 * np.pi, size=self.num_bodies)
        x = np.cos(theta) * np.sin(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(phi)
        init_pos_np = np.stack((x, y, z), axis=1)

        scale = (num_bodies / 1024) ** (1 / 2)  # Scale factor to maintain a constant density
        inner = 0.9625 * scale
        outer = 1.54 * scale
        radii = inner + (outer - inner) * rng.uniform(size=(self.num_bodies, 1))
        init_pos_np = init_pos_np * radii

        axis = np.array([0.0, 0.0, 1.0])
        v_scale = scale * 3.08
        init_vel_np = v_scale * np.cross(init_pos_np, axis)

        self.graph_scale = np.max(radii) * 5.0
        self.pos_array_0 = wp.array(init_pos_np, dtype=wp.vec3)
        self.pos_array_1 = wp.empty_like(self.pos_array_0)
        self.vel_array = wp.array(init_vel_np, dtype=wp.vec3)

        if headless:
            self.scatter_plot = None
        else:
            self.scatter_plot = self.create_plot()

    def create_plot(self):
        import matplotlib.pyplot as plt

        # Create a figure and a 3D axis for the plot
        self.fig = plt.figure()
        ax = self.fig.add_subplot(111, projection="3d")

        # Scatter plot of initial positions
        point_size = 0.05 * self.graph_scale
        init_pos_np = self.pos_array_0.numpy()
        scatter_plot = ax.scatter(
            init_pos_np[:, 0], init_pos_np[:, 1], init_pos_np[:, 2], s=point_size, c="#76b900", alpha=0.5
        )

        # Set axis limits
        ax.set_xlim(-self.graph_scale, self.graph_scale)
        ax.set_ylim(-self.graph_scale, self.graph_scale)
        ax.set_zlim(-self.graph_scale, self.graph_scale)

        return scatter_plot

    def step(self):
        wp.launch(
            integrate_bodies_tiled,
            dim=self.num_bodies,
            inputs=[self.pos_array_0, self.vel_array, self.pos_array_1, self.num_bodies],
            block_dim=TILE_SIZE,
        )

        # Swap arrays
        (self.pos_array_0, self.pos_array_1) = (self.pos_array_1, self.pos_array_0)

    def render(self):
        positions_cpu = self.pos_array_0.numpy()

        # Update scatter plot positions
        self.scatter_plot._offsets3d = (
            positions_cpu[:, 0],
            positions_cpu[:, 1],
            positions_cpu[:, 2],
        )

    # Function to update the scatter plot
    def step_and_render(self, frame):
        self.step()
        self.render()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--num_frames", type=int, default=1000, help="Total number of frames.")
    parser.add_argument("-N", help="Number of bodies. Should be a multiple of 64.", type=int, default=16384)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode, suppressing the opening of any graphical windows.",
    )

    args = parser.parse_known_args()[0]

    if args.device == "cpu":
        print("This example only runs on CUDA devices.")
        exit()

    with wp.ScopedDevice(args.device):
        example = Example(headless=args.headless, num_bodies=args.N)

        if not args.headless:
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation

            # Create the animation
            ani = FuncAnimation(example.fig, example.step_and_render, frames=args.num_frames, interval=50, repeat=False)

            # Display the animation
            plt.show()

        else:
            for _ in range(args.num_frames):
                example.step()
