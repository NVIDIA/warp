# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example APIC Fluid Simulation
#
# Shows how to implement a apic fluid simulation.
###########################################################################

import numpy as np

import warp as wp
import warp.fem as fem
import warp.sim.render
from warp.fem import Domain, Field, Sample, at_node, div, grad, integrand, lookup, normal
from warp.sim import Model, State
from warp.sparse import BsrMatrix, bsr_copy, bsr_mm, bsr_mv, bsr_transposed, bsr_zeros

try:
    from .bsr_utils import bsr_cg
except ImportError:
    from bsr_utils import bsr_cg

wp.init()


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
    vel[s.qp_index] = grid_vel(s)
    vel_grad[s.qp_index] = grad(grid_vel, s)

    pos_adv = pos_prev[s.qp_index] + dt * vel[s.qp_index]

    # Project onto domain
    pos_proj = domain(lookup(domain, pos_adv))
    pos[s.qp_index] = pos_proj


@integrand
def velocity_boundary_projector_form(s: Sample, domain: Domain, u: Field, v: Field):
    """Projector for velocity-Dirichlet boundary conditions"""

    n = normal(domain, s)
    if n[1] > 0.0:
        # Neuman  on top
        return 0.0

    # Free-slip on other sides
    return wp.dot(u(s), n) * wp.dot(v(s), n)


@integrand
def divergence_form(s: Sample, u: Field, psi: Field):
    return div(u, s) * psi(s)


@wp.kernel
def invert_volume_kernel(values: wp.array(dtype=float)):
    i = wp.tid()
    m = values[i]
    if m <= 1.0e-8:
        values[i] = 0.0
    else:
        values[i] = 1.0 / m


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
    tr_divergence_mat_values: wp.array(dtype=wp.mat(shape=(3, 1), dtype=float)),
    inv_fraction_int: wp.array(dtype=float),
):
    u_i = wp.tid()
    block_beg = tr_divergence_mat_offsets[u_i]
    block_end = tr_divergence_mat_offsets[u_i + 1]

    for b in range(block_beg, block_end):
        tr_divergence_mat_values[b] = tr_divergence_mat_values[b] * inv_fraction_int[u_i]


def solve_incompressibility(divergence_mat: BsrMatrix, inv_volume, pressure, velocity, quiet: bool = False):
    """Solve for divergence-free velocity delta:

    delta_velocity = inv_volume * transpose(divergence_mat) * pressure
    divergence_mat * (velocity + delta_velocity) = 0
    """

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

    # For simplicity, assemble schur complement and solve with CG
    schur = bsr_mm(divergence_mat, transposed_divergence_mat)

    rhs = wp.zeros_like(pressure)
    bsr_mv(A=divergence_mat, x=velocity, y=rhs, alpha=-1.0, beta=0.0)
    bsr_cg(schur, b=rhs, x=pressure, quiet=quiet)

    # Apply pressure to velocity
    bsr_mv(A=transposed_divergence_mat, x=pressure, y=velocity, alpha=1.0, beta=1.0)


class Example:
    def __init__(self, quiet=False, stage_path="example_apic_fluid.usd", res=(32, 64, 16)):
        fps = 60
        self.frame_dt = 1.0 / fps
        self.current_frame = 0

        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps

        self._quiet = quiet

        # grid dimensions and particle emission
        grid_res = np.array(res, dtype=int)
        particle_fill_frac = np.array([0.5, 0.5, 1.0])
        grid_lo = wp.vec3(0.0)
        grid_hi = wp.vec3(50, 100, 25)

        grid_cell_size = np.array(grid_hi - grid_lo) / grid_res
        grid_cell_volume = np.prod(grid_cell_size)

        PARTICLES_PER_CELL_DIM = 3
        self.radius = float(np.max(grid_cell_size) / (2 * PARTICLES_PER_CELL_DIM))

        particle_grid_res = np.array(particle_fill_frac * grid_res * PARTICLES_PER_CELL_DIM, dtype=int)
        particle_grid_offset = wp.vec3(self.radius, self.radius, self.radius)

        np.random.seed(0)
        builder = wp.sim.ModelBuilder()
        builder.add_particle_grid(
            dim_x=particle_grid_res[0],
            dim_y=particle_grid_res[1],
            dim_z=particle_grid_res[2],
            cell_x=self.radius * 2.0,
            cell_y=self.radius * 2.0,
            cell_z=self.radius * 2.0,
            pos=wp.vec3(0.0, 0.0, 0.0) + particle_grid_offset,
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0, 0.0, 0.0),
            mass=grid_cell_volume / PARTICLES_PER_CELL_DIM**3,
            jitter=self.radius * 1.0,
            radius_mean=self.radius,
        )

        self.grid = fem.Grid3D(wp.vec3i(grid_res), grid_lo, grid_hi)

        # Function spaces
        self.velocity_space = fem.make_polynomial_space(self.grid, dtype=wp.vec3, degree=1)
        self.fraction_space = fem.make_polynomial_space(self.grid, dtype=float, degree=1)
        self.strain_space = fem.make_polynomial_space(
            self.grid,
            dtype=float,
            degree=0,
        )

        self.pressure_field = self.strain_space.make_field()
        self.velocity_field = self.velocity_space.make_field()

        # Test and trial functions
        self.domain = fem.Cells(self.grid)
        self.velocity_test = fem.make_test(self.velocity_space, domain=self.domain)
        self.velocity_trial = fem.make_trial(self.velocity_space, domain=self.domain)
        self.fraction_test = fem.make_test(self.fraction_space, domain=self.domain)
        self.strain_test = fem.make_test(self.strain_space, domain=self.domain)
        self.strain_trial = fem.make_trial(self.strain_space, domain=self.domain)

        # Enforcing the Dirichlet boundary condition the hard way;
        # build projector for velocity left- and right-hand-sides
        boundary = fem.BoundarySides(self.grid)
        u_bd_test = fem.make_test(space=self.velocity_space, domain=boundary)
        u_bd_trial = fem.make_trial(space=self.velocity_space, domain=boundary)
        u_bd_projector = fem.integrate(
            velocity_boundary_projector_form, fields={"u": u_bd_trial, "v": u_bd_test}, nodal=True, output_dtype=float
        )

        fem.normalize_dirichlet_projector(u_bd_projector)
        self.vel_bd_projector = u_bd_projector

        # Storage for temporary variables
        self.temporary_store = fem.TemporaryStore()

        self._divergence_matrix = bsr_zeros(
            self.strain_space.node_count(),
            self.velocity_space.node_count(),
            block_type=wp.mat(shape=(1, 3), dtype=float),
        )

        # Warp.sim model
        self.model: Model = builder.finalize()

        if not self._quiet:
            print("Particle count:", self.model.particle_count)

        self.state_0: State = self.model.state()
        self.state_0.particle_qd_grad = wp.zeros(shape=(self.model.particle_count), dtype=wp.mat33)

        self.state_1: State = self.model.state()
        self.state_1.particle_qd_grad = wp.zeros(shape=(self.model.particle_count), dtype=wp.mat33)

        try:
            if stage_path:
                self.renderer = warp.sim.render.SimRenderer(self.model, stage_path, scaling=20.0)
            else:
                self.renderer = None
        except Exception as err:
            print(f"Could not initialize SimRenderer for stage '{stage_path}': {err}.")

    def step(self):
        fem.set_default_temporary_store(self.temporary_store)

        self.current_frame = self.current_frame + 1
        with wp.ScopedTimer(f"simulate frame {self.current_frame}", active=True):
            for _s in range(self.sim_substeps):
                # Bin particles to grid cells
                pic = fem.PicQuadrature(
                    domain=fem.Cells(self.grid), positions=self.state_0.particle_q, measures=self.model.particle_mass
                )

                # Borrow some temporary arrays for storing integration results
                inv_volume_temporary = fem.borrow_temporary(
                    self.temporary_store, shape=(self.fraction_space.node_count()), dtype=float
                )
                velocity_int_temporary = fem.borrow_temporary(
                    self.temporary_store, shape=(self.velocity_space.node_count()), dtype=wp.vec3
                )
                inv_volume = inv_volume_temporary.array
                velocity_int = velocity_int_temporary.array

                # Inverse volume fraction
                fem.integrate(
                    integrate_fraction,
                    quadrature=pic,
                    fields={"phi": self.fraction_test},
                    accumulate_dtype=float,
                    output=inv_volume,
                )
                wp.launch(kernel=invert_volume_kernel, dim=inv_volume.shape, inputs=[inv_volume])

                # Velocity right-hand side
                fem.integrate(
                    integrate_velocity,
                    quadrature=pic,
                    fields={"u": self.velocity_test},
                    values={
                        "velocities": self.state_0.particle_qd,
                        "velocity_gradients": self.state_0.particle_qd_grad,
                        "dt": self.sim_dt,
                        "gravity": self.model.gravity,
                    },
                    accumulate_dtype=float,
                    output=velocity_int,
                )

                # Compute constraint-free velocity
                wp.launch(
                    kernel=scalar_vector_multiply,
                    dim=inv_volume.shape[0],
                    inputs=[inv_volume, velocity_int, self.velocity_field.dof_values],
                )

                # Apply velocity boundary conditions:
                # velocity -= vel_bd_projector * velocity
                wp.copy(src=self.velocity_field.dof_values, dest=velocity_int)
                bsr_mv(A=self.vel_bd_projector, x=velocity_int, y=self.velocity_field.dof_values, alpha=-1.0, beta=1.0)

                # Divergence matrix
                fem.integrate(
                    divergence_form,
                    quadrature=pic,
                    fields={"u": self.velocity_trial, "psi": self.strain_test},
                    accumulate_dtype=float,
                    output=self._divergence_matrix,
                )

                # Project matrix to enforce boundary conditions
                divergence_mat_tmp = bsr_copy(self._divergence_matrix)
                bsr_mm(alpha=-1.0, x=divergence_mat_tmp, y=self.vel_bd_projector, z=self._divergence_matrix, beta=1.0)

                # Solve unilateral incompressibility
                solve_incompressibility(
                    self._divergence_matrix,
                    inv_volume,
                    self.pressure_field.dof_values,
                    self.velocity_field.dof_values,
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
                    fields={"grid_vel": self.velocity_field},
                )

                # swap states
                (self.state_0, self.state_1) = (self.state_1, self.state_0)

        fem.set_default_temporary_store(None)

    def render(self, is_live=False):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render", active=True):
            time = self.current_frame * self.frame_dt

            self.renderer.begin_frame(time)
            self.renderer.render(self.state_0)
            self.renderer.end_frame()


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
    parser.add_argument("--num_frames", type=int, default=1000, help="Total number of frames.")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--res",
        type=lambda s: [int(item) for item in s.split(",")],
        default="32,64,16",
        help="Delimited list specifying resolution in x, y, and z.",
    )

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(quiet=args.quiet, stage_path=args.stage_path, res=args.res)

        for _ in range(args.num_frames):
            example.step()
            example.render()

        if example.renderer:
            example.renderer.save()
