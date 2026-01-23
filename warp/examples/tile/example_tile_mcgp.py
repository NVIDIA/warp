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

#############################################################################
# Example Tile Monte Carlo Geometry Processing
#
# Shows how to use the built-in wp.Mesh data structure and
# wp.mesh_eval_position() function to implement a tile-based, grid-free
# Laplace solver using a Monte Carlo approach.
#
# References:
#   Rohan Sawhney and Keenan Crane. Monte Carlo Geometry Processing:
#   A Grid-Free Approach to PDE-Based Methods on Volumetric Domains.
#   ACM Trans. Graph., Vol. 38, No. 4, Article 1. Published July 2020
#
##############################################################################

import numpy as np

import warp as wp

TILE_SIZE = 128  # number of samples for random walk
STEPS = 8
SEED = 42
TOL = 0.001

wp.config.enable_backward = False


@wp.func
def update_radius(mesh_id: wp.uint64, p: wp.vec3):
    """new radius is the distance to the closest point on the mesh"""
    radius = 0.0
    query = wp.mesh_query_point(mesh_id, p, 1e6)

    if query.result:
        q = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
        radius = -wp.length(q - p) * query.sign

    return radius


@wp.func
def get_boundary_value(p: wp.vec3):
    """Spherical harmonic Y_3^3 boundary condition.

    Y_3^3(θ, φ) ∝ sin³(θ) · cos(3φ)

    This has 6-fold azimuthal symmetry and vanishes at the poles.
    Shifted to [0, 1] range for visualization.

    Analytic interior solution: u(r, θ, φ) = r³ · sin³(θ) · cos(3φ)
    """
    phi = wp.atan2(p[1], p[0])  # azimuthal angle
    theta = wp.acos(wp.clamp(p[2], -1.0, 1.0))  # polar angle

    sin_theta = wp.sin(theta)
    Y_3_3 = sin_theta * sin_theta * sin_theta * wp.cos(3.0 * phi)

    # Shift from [-1, 1] to [0, 1] for visualization
    return 0.5 + 0.5 * Y_3_3


@wp.func
def walk(rand_offset: int, mesh_id: wp.uint64, seed: int, p: wp.vec3):
    """random walk on spheres"""
    rng = wp.rand_init(seed, rand_offset)
    for _ in range(0, STEPS):
        r = update_radius(mesh_id, p)
        if r < 0.0:  # outside the mesh
            return 0.0
        elif r < TOL:  # within the epsilon boundary
            return get_boundary_value(p)
        pr = wp.sample_unit_sphere_surface(rng) * r
        p += pr
    return get_boundary_value(p)  # closest choice if epsilon boundary not reached


@wp.kernel
def sphere_walk(
    mesh_id: wp.uint64,
    seed: int,
    delta_z: float,
    samples: wp.array2d(dtype=wp.vec3),
    solutions: wp.array2d(dtype=float),
):
    i, j = wp.tid()

    sample_origin = samples[i, j] + wp.vec3(0.0, 0.0, delta_z)

    # every random sample gets a unique offset for rng
    rand_offsets = wp.tile_arange(TILE_SIZE, dtype=int)
    rand_offsets += wp.tile_full(TILE_SIZE, value=(i * TILE_SIZE + j) * TILE_SIZE, dtype=int)

    # mcgp
    walk_results = wp.tile_map(walk, rand_offsets, mesh_id, seed, sample_origin)

    # solution is an average of all walks originating from this position
    walk_sum = wp.tile_sum(walk_results)
    result = walk_sum * (1.0 / wp.float32(TILE_SIZE))

    wp.tile_store(solutions[i], result, offset=(j,))


class Example:
    def __init__(self, seed, height=256, slices=100, sphere_resolution=64):
        self.seed = seed

        # Generate high-resolution sphere mesh
        points, indices = self.create_sphere_mesh(
            radius=1.0,
            lat_segments=sphere_resolution,
            lon_segments=sphere_resolution * 2,
        )
        print(f"Generated sphere mesh: {len(points)} vertices, {len(indices) // 3} triangles")

        self.mesh = wp.Mesh(
            points=wp.array(points, dtype=wp.vec3),
            indices=wp.array(indices, dtype=int),
        )

        # z-slice scanning grid
        self.height = height
        self.width = self.height

        x = np.linspace(-1.0, 1.0, self.width)
        y = np.linspace(-1.0, 1.0, self.height)
        xv, yv = np.meshgrid(x, y, indexing="ij")
        zv = np.zeros_like(xv)
        grid = np.stack((xv, yv, zv), axis=-1)

        self.grid = wp.array(grid, dtype=wp.vec3)
        self.pixels = wp.zeros((self.height, self.width), dtype=float)

        self.slices = slices
        self.z = np.linspace(-1.0, 1.0, self.slices)

        # storage for animation
        self.images = np.zeros((self.slices, self.height, self.width))
        self.analytic_images = np.zeros((self.slices, self.height, self.width))

    def create_sphere_mesh(self, radius=1.0, lat_segments=64, lon_segments=128):
        """Generate a triangulated UV sphere mesh.

        Args:
            radius: Sphere radius
            lat_segments: Number of latitude divisions (pole to pole)
            lon_segments: Number of longitude divisions (around equator)

        Returns:
            vertices: (N, 3) float32 array of vertex positions
            indices: (M,) int32 array of triangle indices
        """
        vertices = []
        indices = []

        # Generate vertices
        for i in range(lat_segments + 1):
            theta = np.pi * i / lat_segments  # 0 to pi (north pole to south pole)
            for j in range(lon_segments):
                phi = 2.0 * np.pi * j / lon_segments  # 0 to 2pi
                x = radius * np.sin(theta) * np.cos(phi)
                y = radius * np.sin(theta) * np.sin(phi)
                z = radius * np.cos(theta)
                vertices.append([x, y, z])

        # Generate triangle indices
        for i in range(lat_segments):
            for j in range(lon_segments):
                next_j = (j + 1) % lon_segments
                v0 = i * lon_segments + j
                v1 = i * lon_segments + next_j
                v2 = (i + 1) * lon_segments + j
                v3 = (i + 1) * lon_segments + next_j

                # Two triangles per quad
                indices.extend([v0, v2, v1])
                indices.extend([v1, v2, v3])

        return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.int32)

    def compute_analytic_slice(self, slice_idx):
        """Compute the analytic solution for Y_3^3 boundary condition at a z-slice.

        Analytic interior solution: u(r, θ, φ) = r³ · sin³(θ) · cos(3φ)
        Simplifies to: u(x, y, z) = r_xy³ · cos(3φ) where r_xy = sqrt(x² + y²)
        """
        z = self.z[slice_idx]
        x = np.linspace(-1.0, 1.0, self.width)
        y = np.linspace(-1.0, 1.0, self.height)
        xv, yv = np.meshgrid(x, y, indexing="ij")

        # Compute spherical coordinates
        r_xy = np.sqrt(xv**2 + yv**2)
        r = np.sqrt(xv**2 + yv**2 + z**2)
        phi = np.arctan2(yv, xv)

        # Analytic solution: r³ · sin³(θ) · cos(3φ) = r_xy³ · cos(3φ)
        # (since sin(θ) = r_xy/r, so r³·sin³(θ) = r_xy³)
        Y_3_3 = (r_xy**3) * np.cos(3.0 * phi)

        # Shift to [0, 1] for visualization (matching boundary condition)
        analytic = 0.5 + 0.5 * Y_3_3

        # Mask points outside the unit sphere
        analytic[r > 1.0] = 0.0

        self.analytic_images[slice_idx] = analytic

    def render(self, slice_idx, compute_analytic=False):
        wp.launch_tiled(
            sphere_walk,
            dim=[self.height, self.width],
            inputs=[self.mesh.id, self.seed, self.z[slice_idx], self.grid],
            outputs=[self.pixels],
            block_dim=TILE_SIZE,
        )

        self.seed += 1

        self.images[slice_idx] = self.pixels.numpy()

        if compute_analytic:
            self.compute_analytic_slice(slice_idx)

        print(f"slice: {slice_idx}")

    def get_animation(self, compare=True):
        if compare:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            fig.patch.set_facecolor("black")
            ax1.set_facecolor("black")
            ax2.set_facecolor("black")
            ax1.axis("off")
            ax2.axis("off")
            ax1.set_title("Monte Carlo", color="white", fontsize=12)
            ax2.set_title("Analytic", color="white", fontsize=12)
            plt.subplots_adjust(top=0.9, bottom=0.02, right=0.98, left=0.02, hspace=0, wspace=0.05)
        else:
            fig, ax1 = plt.subplots()
            fig.patch.set_facecolor("black")
            ax1.set_facecolor("black")
            ax1.axis("off")
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)

        # Compute vmax from data for consistent colormap scaling
        vmax = max(np.max(self.images), 1.0)
        if compare:
            vmax = max(vmax, np.max(self.analytic_images))

        slices = []
        for i in range(self.slices):
            mcgp_frame = ax1.imshow(
                np.flip(self.images[i, :, :].T),
                animated=True,
                cmap="inferno",
                vmin=0,
                vmax=vmax,
            )
            if compare:
                analytic_frame = ax2.imshow(
                    np.flip(self.analytic_images[i, :, :].T),
                    animated=True,
                    cmap="inferno",
                    vmin=0,
                    vmax=vmax,
                )
                slices.append([mcgp_frame, analytic_frame])
            else:
                slices.append([mcgp_frame])

        ani = animation.ArtistAnimation(fig, slices, interval=60, blit=True, repeat_delay=1000)
        return ani


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--height", type=int, default=256, help="Height of rendered image in pixels.")
    parser.add_argument("--slices", type=int, default=100, help="Number of planar z-slices to scan.")
    parser.add_argument("--sphere-resolution", type=int, default=64, help="Sphere mesh resolution (lat segments).")
    parser.add_argument(
        "--compare",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable side-by-side comparison with analytic solution.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode, suppressing the opening of any graphical windows.",
    )

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(SEED, height=args.height, slices=args.slices, sphere_resolution=args.sphere_resolution)

        for i in range(args.slices):
            example.render(i, compute_analytic=args.compare)

        if not args.headless:
            import matplotlib.animation as animation
            import matplotlib.pyplot as plt

            print("Creating the animation")
            anim = example.get_animation(compare=args.compare)
            anim_filename = "example_tile_mcgp_animation.gif"
            anim.save(anim_filename, dpi=60, writer=animation.PillowWriter(fps=5))
            print(f"Saved the animation at `{anim_filename}`")
