# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Smooth Particle Hydrodynamics
#
# Shows how to implement a SPH
# Neighbors are found using the wp.HashGrid class, and
# wp.hash_grid_query(), wp.hash_grid_query_next() kernel methods.
#
# Reference Publication
# Müller, Matthias, David Charypar, and Markus H. Gross.
# "Particle-based fluid simulation for interactive applications."
# Symposium on Computer animation. Vol. 2. 2003.
#
###########################################################################

import numpy as np

import warp as wp
import warp.render

import os

wp.init()


@wp.func
def square(x: float):
    return x * x


@wp.func
def cube(x: float):
    return x * x * x


@wp.func
def fifth(x: float):
    return x * x * x * x * x


@wp.func
def density_kernel(xyz: wp.vec3, smoothing_length: float):
    # calculate distance
    distance = wp.dot(xyz, xyz)

    return wp.max(cube(square(smoothing_length) - distance), 0.0)


@wp.func
def diff_pressure_kernel(
    xyz: wp.vec3, pressure: float, neighbor_pressure: float, neighbor_rho: float, smoothing_length: float
):
    # calculate distance
    distance = wp.sqrt(wp.dot(xyz, xyz))

    if distance < smoothing_length:
        # calculate terms of kernel
        term_1 = -xyz / distance
        term_2 = (neighbor_pressure + pressure) / (2.0 * neighbor_rho)
        term_3 = square(smoothing_length - distance)
        return term_1 * term_2 * term_3
    else:
        return wp.vec3()


@wp.func
def diff_viscous_kernel(xyz: wp.vec3, v: wp.vec3, neighbor_v: wp.vec3, neighbor_rho: float, smoothing_length: float):
    # calculate distance
    distance = wp.sqrt(wp.dot(xyz, xyz))

    # calculate terms of kernel
    if distance < smoothing_length:
        term_1 = (neighbor_v - v) / neighbor_rho
        term_2 = smoothing_length - distance
        return term_1 * term_2
    else:
        return wp.vec3()


@wp.kernel
def compute_density(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_rho: wp.array(dtype=float),
    density_normalization: float,
    smoothing_length: float,
):
    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)

    # get local particle variables
    x = particle_x[i]

    # store density
    rho = float(0.0)

    # particle contact
    neighbors = wp.hash_grid_query(grid, x, smoothing_length)

    # loop through neighbors to compute density
    for index in neighbors:
        # compute distance
        distance = x - particle_x[index]

        # compute kernel derivative
        rho += density_kernel(distance, smoothing_length)

    # add external potential
    particle_rho[i] = density_normalization * rho


@wp.kernel
def get_acceleration(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    particle_rho: wp.array(dtype=float),
    particle_a: wp.array(dtype=wp.vec3),
    isotropic_exp: float,
    base_density: float,
    gravity: float,
    pressure_normalization: float,
    viscous_normalization: float,
    smoothing_length: float,
):
    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)

    # get local particle variables
    x = particle_x[i]
    v = particle_v[i]
    rho = particle_rho[i]
    pressure = isotropic_exp * (rho - base_density)

    # store forces
    pressure_force = wp.vec3()
    viscous_force = wp.vec3()

    # particle contact
    neighbors = wp.hash_grid_query(grid, x, smoothing_length)

    # loop through neighbors to compute acceleration
    for index in neighbors:
        if index != i:
            # get neighbor velocity
            neighbor_v = particle_v[index]

            # get neighbor density and pressures
            neighbor_rho = particle_rho[index]
            neighbor_pressure = isotropic_exp * (neighbor_rho - base_density)

            # compute relative position
            relative_position = particle_x[index] - x

            # calculate pressure force
            pressure_force += diff_pressure_kernel(
                relative_position, pressure, neighbor_pressure, neighbor_rho, smoothing_length
            )

            # compute kernel derivative
            viscous_force += diff_viscous_kernel(relative_position, v, neighbor_v, neighbor_rho, smoothing_length)

    # sum all forces
    force = pressure_normalization * pressure_force + viscous_normalization * viscous_force

    # add external potential
    particle_a[i] = force / rho + wp.vec3(0.0, gravity, 0.0)


@wp.kernel
def apply_bounds(
    particle_x: wp.array(dtype=wp.vec3),
    particle_v: wp.array(dtype=wp.vec3),
    damping_coef: float,
    width: float,
    height: float,
    length: float,
):
    tid = wp.tid()

    # get pos and velocity
    x = particle_x[tid]
    v = particle_v[tid]

    # clamp x left
    if x[0] < 0.0:
        x = wp.vec3(0.0, x[1], x[2])
        v = wp.vec3(v[0] * damping_coef, v[1], v[2])

    # clamp x right
    if x[0] > width:
        x = wp.vec3(width, x[1], x[2])
        v = wp.vec3(v[0] * damping_coef, v[1], v[2])

    # clamp y bot
    if x[1] < 0.0:
        x = wp.vec3(x[0], 0.0, x[2])
        v = wp.vec3(v[0], v[1] * damping_coef, v[2])

    # clamp z left
    if x[2] < 0.0:
        x = wp.vec3(x[0], x[1], 0.0)
        v = wp.vec3(v[0], v[1], v[2] * damping_coef)

    # clamp z right
    if x[2] > length:
        x = wp.vec3(x[0], x[1], length)
        v = wp.vec3(v[0], v[1], v[2] * damping_coef)

    # apply clamps
    particle_x[tid] = x
    particle_v[tid] = v


@wp.kernel
def kick(particle_v: wp.array(dtype=wp.vec3), particle_a: wp.array(dtype=wp.vec3), dt: float):
    tid = wp.tid()
    v = particle_v[tid]
    particle_v[tid] = v + particle_a[tid] * dt


@wp.kernel
def drift(particle_x: wp.array(dtype=wp.vec3), particle_v: wp.array(dtype=wp.vec3), dt: float):
    tid = wp.tid()
    x = particle_x[tid]
    particle_x[tid] = x + particle_v[tid] * dt


@wp.kernel
def initialize_particles(
    particle_x: wp.array(dtype=wp.vec3), smoothing_length: float, width: float, height: float, length: float
):
    tid = wp.tid()

    # grid size
    nr_x = wp.int32(width / 4.0 / smoothing_length)
    nr_y = wp.int32(height / smoothing_length)
    nr_z = wp.int32(length / 4.0 / smoothing_length)

    # calculate particle position
    z = wp.float(tid % nr_z)
    y = wp.float((tid // nr_z) % nr_y)
    x = wp.float((tid // (nr_z * nr_y)) % nr_x)
    pos = smoothing_length * wp.vec3(x, y, z)

    # add small jitter
    state = wp.rand_init(123, tid)
    pos = pos + 0.001 * smoothing_length * wp.vec3(wp.randn(state), wp.randn(state), wp.randn(state))

    # set position
    particle_x[tid] = pos


class Example:
    def __init__(self, stage):
        # render params
        self.frame_dt = 1.0 / 60.0
        self.frame_count = 600
        self.renderer = wp.render.UsdRenderer(stage)
        self.sim_time = 0.0

        # simulation params
        self.smoothing_length = 0.8  # NOTE change this to adjust number of particles
        self.width = 80.0  # x
        self.height = 80.0  # y
        self.length = 80.0  # z
        self.isotropic_exp = 20
        self.base_density = 1.0
        self.particle_mass = 0.01 * self.smoothing_length**3  # reduce according to smoothing length
        self.dt = 0.01 * self.smoothing_length  # decrease sim dt by smoothing length
        self.dynamic_visc = 0.025
        self.damping_coef = -0.95
        self.gravity = -0.1
        self.n = int(
            self.height * (self.width / 4.0) * (self.height / 4.0) / (self.smoothing_length**3)
        )  # number particles (small box in corner)
        self.sim_step_to_frame_ratio = int(32 / self.smoothing_length)

        # constants
        self.density_normalization = (315.0 * self.particle_mass) / (
            64.0 * np.pi * self.smoothing_length**9
        )  # integrate density kernel
        self.pressure_normalization = -(45.0 * self.particle_mass) / (np.pi * self.smoothing_length**6)
        self.viscous_normalization = (45.0 * self.dynamic_visc * self.particle_mass) / (
            np.pi * self.smoothing_length**6
        )

        # allocate arrays
        self.x = wp.empty(tuple([self.n]), dtype=wp.vec3)
        self.v = wp.zeros(tuple([self.n]), dtype=wp.vec3)
        self.rho = wp.zeros(tuple([self.n]), dtype=float)
        self.a = wp.zeros(tuple([self.n]), dtype=wp.vec3)

        # set random positions
        wp.launch(
            kernel=initialize_particles,
            dim=self.n,
            inputs=[self.x, self.smoothing_length, self.width, self.height, self.length],
        )  # initialize in small area

        # create hash array
        grid_size = int(self.height / (4.0 * self.smoothing_length))
        self.grid = wp.HashGrid(grid_size, grid_size, grid_size)

    def update(self):
        with wp.ScopedTimer("simulate", active=True):
            for s in range(self.sim_step_to_frame_ratio):
                with wp.ScopedTimer("grid build", active=False):
                    # build grid
                    self.grid.build(self.x, self.smoothing_length)

                with wp.ScopedTimer("forces", active=False):
                    # compute density of points
                    wp.launch(
                        kernel=compute_density,
                        dim=self.n,
                        inputs=[self.grid.id, self.x, self.rho, self.density_normalization, self.smoothing_length],
                    )

                    # get new acceleration
                    wp.launch(
                        kernel=get_acceleration,
                        dim=self.n,
                        inputs=[
                            self.grid.id,
                            self.x,
                            self.v,
                            self.rho,
                            self.a,
                            self.isotropic_exp,
                            self.base_density,
                            self.gravity,
                            self.pressure_normalization,
                            self.viscous_normalization,
                            self.smoothing_length,
                        ],
                    )

                    # apply bounds
                    wp.launch(
                        kernel=apply_bounds,
                        dim=self.n,
                        inputs=[self.x, self.v, self.damping_coef, self.width, self.height, self.length],
                    )

                    # kick
                    wp.launch(kernel=kick, dim=self.n, inputs=[self.v, self.a, self.dt])

                    # drift
                    wp.launch(kernel=drift, dim=self.n, inputs=[self.x, self.v, self.dt])

    def render(self, is_live=False):
        with wp.ScopedTimer("render", active=True):
            time = 0.0 if is_live else self.sim_time

            self.renderer.begin_frame(time)
            self.renderer.render_points(points=self.x.numpy(), radius=self.smoothing_length, name="points")
            self.renderer.end_frame()

        self.sim_time += self.frame_dt


if __name__ == "__main__":
    stage_path = os.path.join(os.path.dirname(__file__), "outputs/example_sph.usd")

    example = Example(stage_path)

    for i in range(example.frame_count):
        example.render()
        example.update()

    example.renderer.save()
