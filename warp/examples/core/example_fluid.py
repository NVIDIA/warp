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
# Example Fluid
#
# Shows how to implement a simple 2D Stable Fluids solver using
# multidimensional arrays and launches.
#
###########################################################################

import math

import warp as wp
import warp.render

grid_width = wp.constant(256)
grid_height = wp.constant(128)


@wp.func
def lookup_float(f: wp.array2d(dtype=float), x: int, y: int):
    x = wp.clamp(x, 0, grid_width - 1)
    y = wp.clamp(y, 0, grid_height - 1)

    return f[x, y]


@wp.func
def sample_float(f: wp.array2d(dtype=float), x: float, y: float):
    lx = int(wp.floor(x))
    ly = int(wp.floor(y))

    tx = x - float(lx)
    ty = y - float(ly)

    s0 = wp.lerp(lookup_float(f, lx, ly), lookup_float(f, lx + 1, ly), tx)
    s1 = wp.lerp(lookup_float(f, lx, ly + 1), lookup_float(f, lx + 1, ly + 1), tx)

    s = wp.lerp(s0, s1, ty)
    return s


@wp.func
def lookup_vel(f: wp.array2d(dtype=wp.vec2), x: int, y: int):
    if x < 0 or x >= grid_width:
        return wp.vec2()
    if y < 0 or y >= grid_height:
        return wp.vec2()

    return f[x, y]


@wp.func
def sample_vel(f: wp.array2d(dtype=wp.vec2), x: float, y: float):
    lx = int(wp.floor(x))
    ly = int(wp.floor(y))

    tx = x - float(lx)
    ty = y - float(ly)

    s0 = wp.lerp(lookup_vel(f, lx, ly), lookup_vel(f, lx + 1, ly), tx)
    s1 = wp.lerp(lookup_vel(f, lx, ly + 1), lookup_vel(f, lx + 1, ly + 1), tx)

    s = wp.lerp(s0, s1, ty)
    return s


@wp.kernel
def advect(
    u0: wp.array2d(dtype=wp.vec2),
    u1: wp.array2d(dtype=wp.vec2),
    rho0: wp.array2d(dtype=float),
    rho1: wp.array2d(dtype=float),
    dt: float,
):
    i, j = wp.tid()

    u = u0[i, j]

    # trace backward
    p = wp.vec2(float(i), float(j))
    p = p - u * dt

    # advect
    u1[i, j] = sample_vel(u0, p[0], p[1])
    rho1[i, j] = sample_float(rho0, p[0], p[1])


@wp.kernel
def divergence(u: wp.array2d(dtype=wp.vec2), div: wp.array2d(dtype=float)):
    i, j = wp.tid()

    if i == grid_width - 1:
        return
    if j == grid_height - 1:
        return

    dx = (u[i + 1, j][0] - u[i, j][0]) * 0.5
    dy = (u[i, j + 1][1] - u[i, j][1]) * 0.5

    div[i, j] = dx + dy


@wp.kernel
def pressure_solve(p0: wp.array2d(dtype=float), p1: wp.array2d(dtype=float), div: wp.array2d(dtype=float)):
    i, j = wp.tid()

    s1 = lookup_float(p0, i - 1, j)
    s2 = lookup_float(p0, i + 1, j)
    s3 = lookup_float(p0, i, j - 1)
    s4 = lookup_float(p0, i, j + 1)

    # Jacobi update
    err = s1 + s2 + s3 + s4 - div[i, j]

    p1[i, j] = err * 0.25


@wp.kernel
def pressure_apply(p: wp.array2d(dtype=float), u: wp.array2d(dtype=wp.vec2)):
    i, j = wp.tid()

    if i == 0 or i == grid_width - 1:
        return
    if j == 0 or j == grid_height - 1:
        return

    # pressure gradient
    f_p = wp.vec2(p[i + 1, j] - p[i - 1, j], p[i, j + 1] - p[i, j - 1]) * 0.5

    u[i, j] = u[i, j] - f_p


@wp.kernel
def integrate(u: wp.array2d(dtype=wp.vec2), rho: wp.array2d(dtype=float), dt: float):
    i, j = wp.tid()

    # gravity
    f_g = wp.vec2(-90.8, 0.0) * rho[i, j]

    # integrate
    u[i, j] = u[i, j] + dt * f_g

    # fade
    rho[i, j] = rho[i, j] * (1.0 - 0.1 * dt)


@wp.kernel
def init(rho: wp.array2d(dtype=float), u: wp.array2d(dtype=wp.vec2), radius: int, dir: wp.vec2):
    i, j = wp.tid()

    d = wp.length(wp.vec2(float(i - grid_width / 2), float(j - grid_height / 2)))

    if d < radius:
        rho[i, j] = 1.0
        u[i, j] = dir


class Example:
    def __init__(self):
        fps = 60
        self.frame_dt = 1.0 / fps
        self.sim_substeps = 2
        self.iterations = 100  # Number of pressure iterations
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        shape = (grid_width, grid_height)

        self.u0 = wp.zeros(shape, dtype=wp.vec2)
        self.u1 = wp.zeros(shape, dtype=wp.vec2)

        self.rho0 = wp.zeros(shape, dtype=float)
        self.rho1 = wp.zeros(shape, dtype=float)

        self.p0 = wp.zeros(shape, dtype=float)
        self.p1 = wp.zeros(shape, dtype=float)
        self.div = wp.zeros(shape, dtype=float)

        # capture pressure solve as a CUDA graph
        self.use_cuda_graph = wp.get_device().is_cuda
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.pressure_iterations()
            self.graph = capture.graph

    def step(self):
        with wp.ScopedTimer("step"):
            for _ in range(self.sim_substeps):
                shape = (grid_width, grid_height)
                dt = self.sim_dt

                speed = 400.0
                angle = math.sin(self.sim_time * 4.0) * 1.5
                vel = wp.vec2(math.cos(angle) * speed, math.sin(angle) * speed)

                # update emitters
                wp.launch(init, dim=shape, inputs=[self.rho0, self.u0, 5, vel])

                # force integrate
                wp.launch(integrate, dim=shape, inputs=[self.u0, self.rho0, dt])
                wp.launch(divergence, dim=shape, inputs=[self.u0, self.div])

                # pressure solve
                self.p0.zero_()
                self.p1.zero_()

                if self.use_cuda_graph:
                    wp.capture_launch(self.graph)
                else:
                    self.pressure_iterations()

                # velocity update
                wp.launch(pressure_apply, dim=shape, inputs=[self.p0, self.u0])

                # semi-Lagrangian advection
                wp.launch(advect, dim=shape, inputs=[self.u0, self.u1, self.rho0, self.rho1, dt])

                # swap buffers
                (self.u0, self.u1) = (self.u1, self.u0)
                (self.rho0, self.rho1) = (self.rho1, self.rho0)

                self.sim_time += dt

    def pressure_iterations(self):
        for _ in range(self.iterations):
            wp.launch(pressure_solve, dim=self.p0.shape, inputs=[self.p0, self.p1, self.div])

            # swap pressure fields
            (self.p0, self.p1) = (self.p1, self.p0)

    def step_and_render_frame(self, frame_num=None, img=None):
        self.step()

        with wp.ScopedTimer("render"):
            if img:
                img.set_array(self.rho0.numpy())

        return (img,)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--num_frames", type=int, default=100000, help="Total number of frames.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode, suppressing the opening of any graphical windows.",
    )

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example()

        if args.headless:
            for _ in range(args.num_frames):
                example.step()
        else:
            import matplotlib
            import matplotlib.animation as anim
            import matplotlib.pyplot as plt

            fig = plt.figure()

            img = plt.imshow(
                example.rho0.numpy(),
                origin="lower",
                animated=True,
                interpolation="antialiased",
            )
            img.set_norm(matplotlib.colors.Normalize(0.0, 1.0))
            seq = anim.FuncAnimation(
                fig,
                example.step_and_render_frame,
                fargs=(img,),
                frames=args.num_frames,
                blit=True,
                interval=8,
                repeat=False,
            )

            plt.show()
