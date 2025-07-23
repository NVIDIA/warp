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
# Example APIC Fluid Simulation
#
# Shows how to implement a minimalist APIC fluid simulation using a NanoVDB
# grid and the PicQuadrature class.
###########################################################################

from dataclasses import dataclass
from typing import Any

import numpy as np

import warp as wp
import warp.examples.fem.utils as fem_example_utils
import warp.fem as fem
import warp.render
from warp.fem import Domain, Field, Sample, at_node, div, grad, integrand
from warp.sparse import BsrMatrix, bsr_mm, bsr_mv, bsr_transposed


@wp.func
def collision_sdf(x: wp.vec3):
    # Arbitrary sdf representing collision geometry
    # Here an inverted half-ball of radius 10
    x[1] = wp.min(x[1], 0.0)
    return 10.0 - wp.length(x), -wp.normalize(x)


@integrand
def integrate_fraction(s: Sample, phi: Field):
    return phi(s)


@integrand
def integrate_velocity(
    s: Sample,
    domain: Domain,
    u: Field,
    velocities: wp.array(dtype=wp.vec3),
    velocity_gradients: wp.array(dtype=wp.mat33),
    dt: float,
    gravity: wp.vec3,
):
    """Transfer particle velocities to grid"""
    node_offset = domain(at_node(u, s)) - domain(s)
    vel_apic = velocities[s.qp_index] + velocity_gradients[s.qp_index] * node_offset

    vel_adv = vel_apic + dt * gravity

    # if inside collider, remove normal velocity
    sdf, sdf_gradient = collision_sdf(domain(s))
    if sdf <= 0:
        v_n = wp.dot(vel_adv, sdf_gradient)
        vel_adv -= wp.max(v_n, 0.0) * sdf_gradient

    return wp.dot(u(s), vel_adv)


@integrand
def update_particles(
    s: Sample,
    domain: Domain,
    grid_vel: Field,
    dt: float,
    pos: wp.array(dtype=wp.vec3),
    pos_prev: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    vel_grad: wp.array(dtype=wp.mat33),
):
    """Read particle velocity from grid and advect positions"""
    p_vel = grid_vel(s)
    vel_grad[s.qp_index] = grad(grid_vel, s)

    pos_adv = pos_prev[s.qp_index] + dt * p_vel

    pos[s.qp_index] = pos_adv
    vel[s.qp_index] = p_vel


@integrand
def velocity_boundary_projector_form(s: Sample, domain: Domain, u: Field, v: Field):
    """Projector for velocity-Dirichlet boundary conditions"""

    x = domain(s)
    sdf, sdf_normal = collision_sdf(x)

    if sdf > 0.0:
        # Neuman
        return 0.0

    # Free-slip on boundary
    return wp.dot(u(s), sdf_normal) * wp.dot(v(s), sdf_normal)


@integrand
def divergence_form(s: Sample, domain: Domain, u: Field, psi: Field):
    # Divergence bilinear form
    return div(u, s) * psi(s)


@wp.kernel
def invert_volume_kernel(values: wp.array(dtype=float)):
    i = wp.tid()
    m = values[i]
    values[i] = wp.where(m == 0.0, 0.0, 1.0 / m)


@wp.kernel
def scalar_vector_multiply(
    alpha: wp.array(dtype=float),
    x: wp.array(dtype=wp.vec3),
    y: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    y[i] = alpha[i] * x[i]


@wp.kernel
def scale_transposed_divergence_mat(
    tr_divergence_mat_offsets: wp.array(dtype=int),
    tr_divergence_mat_values: wp.array(dtype=Any),
    inv_fraction_int: wp.array(dtype=float),
):
    # In-place scaling of gradient operator rows with inverse mass

    u_i = wp.tid()
    block_beg = tr_divergence_mat_offsets[u_i]
    block_end = tr_divergence_mat_offsets[u_i + 1]

    for b in range(block_beg, block_end):
        tr_divergence_mat_values[b] = tr_divergence_mat_values[b] * inv_fraction_int[u_i]


def solve_incompressibility(
    divergence_mat: BsrMatrix, dirichlet_projector: BsrMatrix, inv_volume, pressure, velocity, quiet: bool = False
):
    """Solve for divergence-free velocity delta:

    delta_velocity = inv_volume * transpose(divergence_mat) * pressure
    divergence_mat * (velocity + delta_velocity) = 0
    dirichlet_projector * delta_velocity = 0
    """

    # Constraint-free divergence -- computed *before* projection of divergence_mat
    rhs = wp.empty_like(pressure)
    bsr_mv(A=divergence_mat, x=velocity, y=rhs, alpha=-1.0)

    # Project matrix to enforce boundary conditions
    # divergence_matrix -= divergence_matrix * vel_projector
    bsr_mm(alpha=-1.0, x=divergence_mat, y=dirichlet_projector, z=divergence_mat, beta=1.0)

    # Build transposed gradient matrix, scale with inverse fraction
    transposed_divergence_mat = bsr_transposed(divergence_mat)
    wp.launch(
        kernel=scale_transposed_divergence_mat,
        dim=inv_volume.shape[0],
        inputs=[
            transposed_divergence_mat.offsets,
            transposed_divergence_mat.values,
            inv_volume,
        ],
    )

    # For simplicity, assemble Schur complement and solve with CG
    schur = bsr_mm(divergence_mat, transposed_divergence_mat)

    fem_example_utils.bsr_cg(schur, b=rhs, x=pressure, quiet=quiet, tol=1.0e-6, method="cr", max_iters=1000)

    # Apply pressure to velocity
    bsr_mv(A=transposed_divergence_mat, x=pressure, y=velocity, alpha=1.0, beta=1.0)


class Example:
    @dataclass
    class State:
        particle_q: wp.array(dtype=wp.vec3)
        particle_qd: wp.array(dtype=wp.vec3)
        particle_qd_grad: wp.array(dtype=wp.mat33)

    def __init__(self, quiet=False, stage_path="example_apic_fluid.usd", voxel_size=1.0, opengl=False):
        self.gravity = wp.vec3(0.0, -10.0, 0.0)

        fps = 60
        self.sim_substeps = 1
        self.frame_dt = 1.0 / fps
        self.current_frame = 0
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.voxel_size = voxel_size

        self._quiet = quiet

        # particle emission
        PARTICLES_PER_CELL_DIM = 2
        self.radius = float(np.max(voxel_size) / (2 * PARTICLES_PER_CELL_DIM))

        particle_grid_lo = np.full(3, -5)
        particle_grid_hi = np.full(3, 5)
        particle_grid_res = (
            np.array((particle_grid_hi - particle_grid_lo) / voxel_size, dtype=int) * PARTICLES_PER_CELL_DIM
        )

        self.particle_volumes, particle_q = self._spawn_particles(
            particle_grid_res, particle_grid_lo, particle_grid_hi, packing_fraction=1.0
        )
        particle_qd = wp.zeros_like(particle_q)

        particle_count = particle_q.shape[0]
        if not self._quiet:
            print("Particle count:", particle_count)

        # Allocate states
        self.state_0 = self.State(
            wp.clone(particle_q),
            wp.clone(particle_qd),
            particle_qd_grad=wp.zeros(shape=(particle_count), dtype=wp.mat33),
        )
        self.state_1 = self.State(
            wp.clone(particle_q),
            wp.clone(particle_qd),
            particle_qd_grad=wp.zeros(shape=(particle_count), dtype=wp.mat33),
        )

        # Storage for temporary variables
        self.temporary_store = fem.TemporaryStore()

        # initialze renderers
        self.opengl_renderer = None
        self.usd_renderer = None

        try:
            if opengl:
                self.opengl_renderer = warp.render.OpenGLRenderer(
                    screen_width=1024,
                    screen_height=1024,
                )
        except Exception as err:
            wp.utils.warn(f"Could not initialize OpenGL renderer: {err}.")

        try:
            if stage_path:
                self.usd_renderer = warp.render.UsdRenderer(stage_path)
        except Exception as err:
            print(f"Could not initialize Usd renderer '{stage_path}': {err}.")

    def step(self):
        fem.set_default_temporary_store(self.temporary_store)

        self.current_frame = self.current_frame + 1

        with wp.ScopedTimer(f"simulate frame {self.current_frame}", synchronize=True):
            for _s in range(self.sim_substeps):
                # Allocate the voxels and create the warp.fem geometry
                volume = wp.Volume.allocate_by_voxels(
                    voxel_points=self.state_0.particle_q,
                    voxel_size=self.voxel_size,
                )
                grid = fem.Nanogrid(volume)

                # Define function spaces: linear (Q1) for velocity and volume fraction,
                # piecewise-constant for pressure
                linear_basis_space = fem.make_polynomial_basis_space(grid, degree=1)
                velocity_space = fem.make_collocated_function_space(linear_basis_space, dtype=wp.vec3)
                fraction_space = fem.make_collocated_function_space(linear_basis_space, dtype=float)
                strain_space = fem.make_polynomial_space(
                    grid,
                    dtype=float,
                    degree=0,
                    discontinuous=True,
                )

                pressure_field = strain_space.make_field()
                velocity_field = velocity_space.make_field()

                # Define test and trial functions and integrating linear and bilinear forms
                domain = fem.Cells(grid)
                velocity_test = fem.make_test(velocity_space, domain=domain)
                velocity_trial = fem.make_trial(velocity_space, domain=domain)
                fraction_test = fem.make_test(fraction_space, domain=domain)
                strain_test = fem.make_test(strain_space, domain=domain)

                # Build projector for Dirichlet boundary conditions
                vel_projector = fem.integrate(
                    velocity_boundary_projector_form,
                    fields={"u": velocity_trial, "v": velocity_test},
                    assembly="nodal",
                    output_dtype=float,
                )
                fem.normalize_dirichlet_projector(vel_projector)

                # Bin particles to grid cells
                pic = fem.PicQuadrature(
                    domain=domain, positions=self.state_0.particle_q, measures=self.particle_volumes
                )

                # Compute inverse particle volume for each grid node
                inv_volume = fem.integrate(
                    integrate_fraction,
                    quadrature=pic,
                    fields={"phi": fraction_test},
                    output_dtype=float,
                )
                wp.launch(kernel=invert_volume_kernel, dim=inv_volume.shape, inputs=[inv_volume])

                # Velocity right-hand side
                velocity_int = fem.integrate(
                    integrate_velocity,
                    quadrature=pic,
                    fields={"u": velocity_test},
                    values={
                        "velocities": self.state_0.particle_qd,
                        "velocity_gradients": self.state_0.particle_qd_grad,
                        "dt": self.sim_dt,
                        "gravity": self.gravity,
                    },
                    output_dtype=wp.vec3,
                )

                # Compute constraint-free velocity
                wp.launch(
                    kernel=scalar_vector_multiply,
                    dim=inv_volume.shape[0],
                    inputs=[inv_volume, velocity_int, velocity_field.dof_values],
                )

                # Apply velocity boundary conditions:
                # velocity -= vel_projector * velocity
                bsr_mv(
                    A=vel_projector,
                    x=velocity_field.dof_values,
                    y=velocity_field.dof_values,
                    alpha=-1.0,
                    beta=1.0,
                )

                # Assemble divergence operator matrix
                divergence_matrix = fem.integrate(
                    divergence_form,
                    quadrature=pic,
                    fields={"u": velocity_trial, "psi": strain_test},
                    output_dtype=float,
                )

                # Solve unilateral incompressibility
                solve_incompressibility(
                    divergence_matrix,
                    vel_projector,
                    inv_volume,
                    pressure_field.dof_values,
                    velocity_field.dof_values,
                    quiet=self._quiet,
                )

                # (A)PIC advection
                fem.interpolate(
                    update_particles,
                    quadrature=pic,
                    values={
                        "pos": self.state_1.particle_q,
                        "pos_prev": self.state_0.particle_q,
                        "vel": self.state_1.particle_qd,
                        "vel_grad": self.state_1.particle_qd_grad,
                        "dt": self.sim_dt,
                    },
                    fields={"grid_vel": velocity_field},
                )

                # swap states
                (self.state_0, self.state_1) = (self.state_1, self.state_0)

        fem.set_default_temporary_store(None)

    @staticmethod
    def _spawn_particles(res, bounds_lo, bounds_hi, packing_fraction):
        Nx = res[0]
        Ny = res[1]
        Nz = res[2]

        px = np.linspace(bounds_lo[0], bounds_hi[0], Nx + 1)
        py = np.linspace(bounds_lo[1], bounds_hi[1], Ny + 1)
        pz = np.linspace(bounds_lo[2], bounds_hi[2], Nz + 1)

        points = np.stack(np.meshgrid(px, py, pz)).reshape(3, -1).T

        cell_size = (bounds_hi - bounds_lo) / res
        cell_volume = np.prod(cell_size)

        radius = np.max(cell_size) * 0.5
        volume = np.prod(cell_volume) * packing_fraction

        rng = np.random.default_rng()
        points += 2.0 * radius * (rng.random(points.shape) - 0.5)

        volumes = wp.full(points.shape[0], volume, dtype=float)
        points = wp.array(np.ascontiguousarray(points), dtype=wp.vec3)
        return volumes, points

    def render(self, is_live=False):
        if self.usd_renderer is None and self.opengl_renderer is None:
            return

        with wp.ScopedTimer("render", synchronize=True):
            time = self.current_frame * self.frame_dt

            if self.usd_renderer is not None:
                self.usd_renderer.begin_frame(time)
                self.usd_renderer.render_points(
                    "particles",
                    self.state_0.particle_q.numpy(),
                    radius=self.radius,
                )
                self.usd_renderer.end_frame()
            if self.opengl_renderer is not None:
                self.opengl_renderer.begin_frame(time)
                self.opengl_renderer.render_points(
                    "particles",
                    self.state_0.particle_q,
                    radius=self.radius,
                )
                self.opengl_renderer.end_frame()


if __name__ == "__main__":
    import argparse

    wp.set_module_options({"enable_backward": False})

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="example_apic_fluid.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_frames", type=int, default=250, help="Total number of frames.")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--opengl", action="store_true")
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.25,
    )

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(quiet=args.quiet, stage_path=args.stage_path, voxel_size=args.voxel_size, opengl=args.opengl)

        for _ in range(args.num_frames):
            example.step()
            example.render()

        if example.usd_renderer is not None:
            example.usd_renderer.save()
