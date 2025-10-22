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

import os

import numpy as np
from pxr import Usd, UsdGeom

import warp as wp
import warp.examples

TILE_SIZE = wp.constant(64)  # number of samples for random walk
STEPS = wp.constant(8)
SEED = wp.constant(42)
TOL = wp.constant(0.001)

wp.config.enable_backward = False


@wp.func
def update_radius(p: wp.vec3):
    """new radius is the distance to the closest point on the mesh"""
    radius = 0.0
    query = wp.mesh_query_point(MESH_ID, p, 1e6)

    if query.result:
        q = wp.mesh_eval_position(MESH_ID, query.face, query.u, query.v)
        radius = -wp.length(q - p) * query.sign

    return radius


@wp.func
def get_boundary_value(p: wp.vec3):
    """analytic boundary condition on the surface of the mesh"""
    return wp.abs(wp.sin(5.0 * wp.length(p)))


@wp.func
def walk(p: wp.vec3, rand_offset: int):
    """random walk on spheres"""
    rng = wp.rand_init(SEED, rand_offset)
    for _ in range(0, STEPS):
        r = update_radius(p)
        if r < 0.0:  # outside the mesh
            return 0.0
        elif r < TOL:  # within the epsilon boundary
            return get_boundary_value(p)
        pr = wp.sample_unit_sphere_surface(rng) * r
        p += pr
    return get_boundary_value(p)  # closest choice if epsilon boundary not reached


@wp.kernel
def sphere_walk(delta_z: float, samples: wp.array2d(dtype=wp.vec3), solutions: wp.array2d(dtype=float)):
    i, j = wp.tid()

    sample_origin = samples[i, j] + wp.vec3(0.0, 0.0, delta_z)

    rand_samples = wp.tile_full(TILE_SIZE, value=sample_origin, dtype=wp.vec3)

    # every random sample gets a unique offset for rng
    rand_offsets = wp.tile_arange(TILE_SIZE, dtype=int)
    rand_offsets += wp.tile_full(TILE_SIZE, value=(i * TILE_SIZE + j) * TILE_SIZE, dtype=int)

    # mcgp
    walk_results = wp.tile_map(walk, rand_samples, rand_offsets)

    # solution is an average of all walks originating from this position
    walk_sum = wp.tile_sum(walk_results)
    result = walk_sum * (1.0 / wp.float32(TILE_SIZE))

    wp.tile_store(solutions[i], result, offset=(j,))


class Example:
    def __init__(self, height=256, slices=60):
        usd_stage = Usd.Stage.Open(os.path.join(warp.examples.get_asset_directory(), "bunny.usd"))
        usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/root/bunny"))

        self.mesh = wp.Mesh(
            points=wp.array(usd_geom.GetPointsAttr().Get(), dtype=wp.vec3),
            indices=wp.array(usd_geom.GetFaceVertexIndicesAttr().Get(), dtype=int),
            bvh_leaf_size=1,
        )

        # z-slice scanning grid
        self.height = height
        self.width = self.height

        x = np.linspace(-1.15, 0.85, self.width)
        y = np.linspace(-0.2, 1.8, self.height)
        xv, yv = np.meshgrid(x, y, indexing="ij")
        zv = np.zeros_like(xv)
        grid = np.stack((xv, yv, zv), axis=-1)

        self.grid = wp.array(grid, dtype=wp.vec3)
        self.pixels = wp.zeros((self.height, self.width), dtype=float)

        self.slices = slices
        self.z = np.linspace(-0.6, 0.8, self.slices)

        # storage for animation
        self.images = np.zeros((self.slices, self.height, self.width))

    def render(self, slice):
        wp.launch_tiled(
            sphere_walk,
            dim=[self.height, self.width],
            inputs=[self.z[slice], self.grid],
            outputs=[self.pixels],
            block_dim=TILE_SIZE,
        )

        self.images[slice] = self.pixels.numpy()
        print(f"slice: {slice}")

    def get_animation(self):
        fig, ax = plt.subplots()
        plt.axis("off")
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)

        slices = []
        for i in range(self.slices):
            slice = ax.imshow(np.flip(self.images[i, :, :].T), animated=True)
            slices.append([slice])

        ani = animation.ArtistAnimation(fig, slices, interval=60, blit=True, repeat_delay=1000)
        return ani


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--height", type=int, default=256, help="Height of rendered image in pixels.")
    parser.add_argument("--slices", type=int, default=60, help="Number of planar z-slices to scan.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode, suppressing the opening of any graphical windows.",
    )

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(height=args.height, slices=args.slices)

        # todo: permit runtime constants to be passed to tile map functions
        MESH_ID = wp.constant(wp.uint64(example.mesh.id))

        for i in range(args.slices):
            example.render(i)

        if not args.headless:
            import matplotlib.animation as animation
            import matplotlib.pyplot as plt

            print("Creating the animation")
            anim = example.get_animation()
            anim_filename = "example_tile_mcgp_animation.gif"
            anim.save(anim_filename, dpi=300, writer=animation.PillowWriter(fps=5))
            print(f"Saved the animation at `{anim_filename}`")
