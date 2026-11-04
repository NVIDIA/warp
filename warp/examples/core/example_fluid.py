# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
def lookup_float(f: wp.array2d[float], x: int, y: int):
    x = wp.clamp(x, 0, grid_width - 1)
    y = wp.clamp(y, 0, grid_height - 1)

    return f[x, y]


@wp.func
def sample_float(f: wp.array2d[float], x: float, y: float):
    lx = int(wp.floor(x))
    ly = int(wp.floor(y))

    tx = x - float(lx)
    ty = y - float(ly)

    s0 = wp.lerp(lookup_float(f, lx, ly), lookup_float(f, lx + 1, ly), tx)
    s1 = wp.lerp(lookup_float(f, lx, ly + 1), lookup_float(f, lx + 1, ly + 1), tx)

    s = wp.lerp(s0, s1, ty)
    return s


@wp.func
def lookup_vel(f: wp.array2d[wp.vec2], x: int, y: int):
    if x < 0 or x >= grid_width:
        return wp.vec2()
    if y < 0 or y >= grid_height:
        return wp.vec2()

    return f[x, y]


@wp.func
def sample_vel(f: wp.array2d[wp.vec2], x: float, y: float):
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
    u0: wp.array2d[wp.vec2],
    u1: wp.array2d[wp.vec2],
    rho0: wp.array2d[float],
    rho1: wp.array2d[float],
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
def divergence(u: wp.array2d[wp.vec2], div: wp.array2d[float]):
    i, j = wp.tid()

    if i == grid_width - 1:
        return
    if j == grid_height - 1:
        return

    dx = (u[i + 1, j][0] - u[i, j][0]) * 0.5
    dy = (u[i, j + 1][1] - u[i, j][1]) * 0.5

    div[i, j] = dx + dy


@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
    return blockIdx.x * blockDim.x + threadIdx.x;
#else
    return fallback;
#endif
"""
)
def pressure_linear_index(fallback: int) -> int:
    ...

@wp.func_native(
    """
#if defined(__CUDA_ARCH__)
    return __ldg(reinterpret_cast<const float*>(address) + index);
#else
    return reinterpret_cast<const float*>(address)[index];
#endif
"""
)
def pressure_read(address: wp.uint64, index: int) -> float:
    ...

@wp.func
def pressure_readonly_cell(
    p0: wp.array2d(dtype=wp.float32),
    p1: wp.array2d(dtype=wp.float32),
    div: wp.array2d(dtype=wp.float32),
    fallback: int,
    width: int,
    height: int,
):
    center = pressure_linear_index(fallback)
    column = center % height
    before_row = wp.where(center >= height, center - height, center)
    after_row = wp.where(center < (width - 1) * height, center + height, center)
    before_column = wp.where(column > 0, center - 1, center)
    after_column = wp.where(column < height - 1, center + 1, center)
    left = pressure_read(p0.ptr, before_row)
    right = pressure_read(p0.ptr, after_row)
    below = pressure_read(p0.ptr, before_column)
    above = pressure_read(p0.ptr, after_column)
    divergence = pressure_read(div.ptr, center)
    destination = wp.array(ptr=p1.ptr, shape=(width * height,), dtype=wp.float32)
    destination[center] = (((left + right) + below) + above - divergence) * 0.25

@wp.kernel
def pressure_solve_256_128(
    p0: wp.array2d(dtype=wp.float32),
    p1: wp.array2d(dtype=wp.float32),
    div: wp.array2d(dtype=wp.float32),
):
    row, column = wp.tid()
    pressure_readonly_cell(p0, p1, div, row * 128 + column, 256, 128)

@wp.kernel
def pressure_solve_257_129(
    p0: wp.array2d(dtype=wp.float32),
    p1: wp.array2d(dtype=wp.float32),
    div: wp.array2d(dtype=wp.float32),
):
    row, column = wp.tid()
    pressure_readonly_cell(p0, p1, div, row * 129 + column, 257, 129)

@wp.kernel
def pressure_solve_512_256(
    p0: wp.array2d(dtype=wp.float32),
    p1: wp.array2d(dtype=wp.float32),
    div: wp.array2d(dtype=wp.float32),
):
    row, column = wp.tid()
    pressure_readonly_cell(p0, p1, div, row * 256 + column, 512, 256)

# Static binding to the exact returned native kernel; the original caller is unchanged.
pressure_solve = pressure_solve_256_128


@wp.kernel
def pressure_apply(p: wp.array2d[float], u: wp.array2d[wp.vec2]):
    i, j = wp.tid()

    if i == 0 or i == grid_width - 1:
        return
    if j == 0 or j == grid_height - 1:
        return

    # pressure gradient
    f_p = wp.vec2(p[i + 1, j] - p[i - 1, j], p[i, j + 1] - p[i, j - 1]) * 0.5

    u[i, j] = u[i, j] - f_p


@wp.kernel
def integrate(u: wp.array2d[wp.vec2], rho: wp.array2d[float], dt: float):
    i, j = wp.tid()

    # gravity
    f_g = wp.vec2(-90.8, 0.0) * rho[i, j]

    # integrate
    u[i, j] = u[i, j] + dt * f_g

    # fade
    rho[i, j] = rho[i, j] * (1.0 - 0.1 * dt)


@wp.kernel
def init(rho: wp.array2d[float], u: wp.array2d[wp.vec2], radius: int, dir: wp.vec2):
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

        # capture pressure solve as a graph
        self.use_graph_capture = True
        if self.use_graph_capture:
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

                if self.use_graph_capture:
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
    parser.add_argument("--num-frames", type=int, default=100000, help="Total number of frames.")
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
