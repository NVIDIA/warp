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
# Example Wave
#
# Shows how to implement a simple 2D wave-equation solver with collision
# against a moving sphere.
#
###########################################################################

import math

import warp as wp
import warp.render


@wp.func
def sample(f: wp.array(dtype=float), x: int, y: int, width: int, height: int):
    # clamp texture coords
    x = wp.clamp(x, 0, width - 1)
    y = wp.clamp(y, 0, height - 1)

    s = f[y * width + x]
    return s


@wp.func
def laplacian(f: wp.array(dtype=float), x: int, y: int, width: int, height: int):
    ddx = sample(f, x + 1, y, width, height) - 2.0 * sample(f, x, y, width, height) + sample(f, x - 1, y, width, height)
    ddy = sample(f, x, y + 1, width, height) - 2.0 * sample(f, x, y, width, height) + sample(f, x, y - 1, width, height)

    return ddx + ddy


@wp.kernel
def wave_displace(
    hcurrent: wp.array(dtype=float),
    hprevious: wp.array(dtype=float),
    width: int,
    height: int,
    center_x: float,
    center_y: float,
    r: float,
    mag: float,
    t: float,
):
    tid = wp.tid()

    x = tid % width
    y = tid // width

    dx = float(x) - center_x
    dy = float(y) - center_y

    dist_sq = float(dx * dx + dy * dy)

    if dist_sq < r * r:
        h = mag * wp.sin(t)

        hcurrent[tid] = h
        hprevious[tid] = h


@wp.kernel
def wave_solve(
    hprevious: wp.array(dtype=float),
    hcurrent: wp.array(dtype=float),
    width: int,
    height: int,
    inv_cell: float,
    k_speed: float,
    k_damp: float,
    dt: float,
):
    tid = wp.tid()

    x = tid % width
    y = tid // width

    l = laplacian(hcurrent, x, y, width, height) * inv_cell * inv_cell

    # integrate
    h1 = hcurrent[tid]
    h0 = hprevious[tid]

    h = 2.0 * h1 - h0 + dt * dt * (k_speed * l - k_damp * (h1 - h0))

    # buffers get swapped each iteration
    hprevious[tid] = h


# simple kernel to apply height deltas to a vertex array
@wp.kernel
def grid_update(heights: wp.array(dtype=float), vertices: wp.array(dtype=wp.vec3)):
    tid = wp.tid()

    h = heights[tid]
    v = vertices[tid]

    v_new = wp.vec3(v[0], h, v[2])

    vertices[tid] = v_new


class Example:
    def __init__(self, stage_path="example_wave.usd", verbose=False):
        self.sim_width = 128
        self.sim_height = 128

        fps = 60
        self.sim_substeps = 16
        self.sim_dt = (1.0 / fps) / self.sim_substeps
        self.sim_time = 0.0

        # wave constants
        self.k_speed = 1.0
        self.k_damp = 0.0

        # grid constants
        self.grid_size = 0.1
        self.grid_displace = 0.5

        self.verbose = verbose

        vertices = []
        self.indices = []

        def grid_index(x, y, stride):
            return y * stride + x

        for z in range(self.sim_height):
            for x in range(self.sim_width):
                pos = (
                    float(x) * self.grid_size,
                    0.0,
                    float(z) * self.grid_size,
                )

                # directly modifies verts_host memory since this is a numpy alias of the same buffer
                vertices.append(pos)

                if x > 0 and z > 0:
                    self.indices.append(grid_index(x - 1, z - 1, self.sim_width))
                    self.indices.append(grid_index(x, z, self.sim_width))
                    self.indices.append(grid_index(x, z - 1, self.sim_width))

                    self.indices.append(grid_index(x - 1, z - 1, self.sim_width))
                    self.indices.append(grid_index(x - 1, z, self.sim_width))
                    self.indices.append(grid_index(x, z, self.sim_width))

        # simulation grids
        self.sim_grid0 = wp.zeros(self.sim_width * self.sim_height, dtype=float)
        self.sim_grid1 = wp.zeros(self.sim_width * self.sim_height, dtype=float)
        self.sim_verts = wp.array(vertices, dtype=wp.vec3)

        # create surface displacement around a point
        self.cx = self.sim_width / 2 + math.sin(self.sim_time) * self.sim_width / 3
        self.cy = self.sim_height / 2 + math.cos(self.sim_time) * self.sim_height / 3

        if stage_path:
            self.renderer = wp.render.UsdRenderer(stage_path)
        else:
            self.renderer = None

    def step(self):
        with wp.ScopedTimer("step"):
            for _s in range(self.sim_substeps):
                # create surface displacement around a point
                self.cx = self.sim_width / 2 + math.sin(self.sim_time) * self.sim_width / 3
                self.cy = self.sim_height / 2 + math.cos(self.sim_time) * self.sim_height / 3

                wp.launch(
                    kernel=wave_displace,
                    dim=self.sim_width * self.sim_height,
                    inputs=[
                        self.sim_grid0,
                        self.sim_grid1,
                        self.sim_width,
                        self.sim_height,
                        self.cx,
                        self.cy,
                        10.0,
                        self.grid_displace,
                        -math.pi * 0.5,
                    ],
                )

                # integrate wave equation
                wp.launch(
                    kernel=wave_solve,
                    dim=self.sim_width * self.sim_height,
                    inputs=[
                        self.sim_grid0,
                        self.sim_grid1,
                        self.sim_width,
                        self.sim_height,
                        1.0 / self.grid_size,
                        self.k_speed,
                        self.k_damp,
                        self.sim_dt,
                    ],
                )

                # swap grids
                (self.sim_grid0, self.sim_grid1) = (self.sim_grid1, self.sim_grid0)

                self.sim_time += self.sim_dt

        with wp.ScopedTimer("mesh", self.verbose):
            # update grid vertices from heights
            wp.launch(kernel=grid_update, dim=self.sim_width * self.sim_height, inputs=[self.sim_grid0, self.sim_verts])

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            vertices = self.sim_verts.numpy()

            self.renderer.begin_frame(self.sim_time)
            self.renderer.render_mesh("surface", vertices, self.indices, colors=(0.35, 0.55, 0.9))
            self.renderer.render_sphere(
                "sphere",
                (self.cx * self.grid_size, 0.0, self.cy * self.grid_size),
                (0.0, 0.0, 0.0, 1.0),
                10.0 * self.grid_size,
                color=(1.0, 1.0, 1.0),
            )
            self.renderer.end_frame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="example_wave.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_frames", type=int, default=300, help="Total number of frames.")
    parser.add_argument("--verbose", action="store_true", help="Print out additional status messages during execution.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, verbose=args.verbose)

        for _ in range(args.num_frames):
            example.step()
            example.render()

        if example.renderer:
            example.renderer.save()
