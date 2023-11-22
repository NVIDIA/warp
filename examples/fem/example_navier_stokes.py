"""
This example solves a 2D Navier-Stokes flow problem

  Du/dt -nu D(u) + grad p = 0
  Div u = 0

with (hard) velocity-Dirichlet boundary conditions
and using semi-Lagrangian advection
"""

import argparse

import warp as wp
import warp.fem as fem

import numpy as np

from warp.fem.utils import array_axpy

from warp.sparse import bsr_mm, bsr_mv, bsr_copy

try:
    from .bsr_utils import bsr_to_scipy
    from .plot_utils import Plot
    from .mesh_utils import gen_trimesh
except ImportError:
    from bsr_utils import bsr_to_scipy
    from plot_utils import Plot
    from mesh_utils import gen_trimesh

# need to solve a saddle-point system, use scopy for simplicity
from scipy.sparse import bmat
from scipy.sparse.linalg import factorized

import matplotlib.pyplot as plt
import matplotlib.animation as animation


@fem.integrand
def u_boundary_value(s: fem.Sample, domain: fem.Domain, v: fem.Field, top_vel: float):
    # Horizontal velocity on top of domain, zero elsewhere
    if domain(s)[1] == 1.0:
        return wp.dot(wp.vec2f(top_vel, 0.0), v(s))

    return wp.dot(wp.vec2f(0.0, 0.0), v(s))


@fem.integrand
def mass_form(
    s: fem.Sample,
    u: fem.Field,
    v: fem.Field,
):
    return wp.dot(u(s), v(s))


@fem.integrand
def inertia_form(s: fem.Sample, u: fem.Field, v: fem.Field, dt: float):
    return mass_form(s, u, v) / dt


@fem.integrand
def viscosity_form(s: fem.Sample, u: fem.Field, v: fem.Field, nu: float):
    return 2.0 * nu * wp.ddot(fem.D(u, s), fem.D(v, s))


@fem.integrand
def viscosity_and_inertia_form(s: fem.Sample, u: fem.Field, v: fem.Field, dt: float, nu: float):
    return inertia_form(s, u, v, dt) + viscosity_form(s, u, v, nu)


@fem.integrand
def transported_inertia_form(s: fem.Sample, domain: fem.Domain, u: fem.Field, v: fem.Field, dt: float):
    pos = domain(s)
    vel = u(s)

    conv_pos = pos - 0.5 * vel * dt
    conv_s = fem.lookup(domain, conv_pos, s)
    conv_vel = u(conv_s)

    conv_pos = conv_pos - 0.5 * conv_vel * dt
    conv_vel = u(fem.lookup(domain, conv_pos, conv_s))

    return wp.dot(conv_vel, v(s)) / dt


@fem.integrand
def div_form(
    s: fem.Sample,
    u: fem.Field,
    q: fem.Field,
):
    return -q(s) * fem.div(u, s)


class Example:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=25)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--num_frames", type=int, default=1000)
    parser.add_argument("--top_velocity", type=float, default=1.0)
    parser.add_argument("--Re", type=float, default=1000.0)
    parser.add_argument("--tri_mesh", action="store_true", help="Use a triangular mesh")

    def __init__(self, stage=None, quiet=False, args=None, **kwargs):
        if args is None:
            # Read args from kwargs, add default arg values from parser
            args = argparse.Namespace(**kwargs)
            args = Example.parser.parse_args(args=[], namespace=args)
        self._args = args
        self._quiet = quiet

        res = args.resolution
        self.sim_dt = 1.0 / args.resolution
        self.current_frame = 0

        viscosity = args.top_velocity / args.Re

        if args.tri_mesh:
            positions, tri_vidx = gen_trimesh(res=wp.vec2i(res))
            geo = fem.Trimesh2D(tri_vertex_indices=tri_vidx, positions=positions)
        else:
            geo = fem.Grid2D(res=wp.vec2i(res))

        domain = fem.Cells(geometry=geo)
        boundary = fem.BoundarySides(geo)

        # Functions spaces: Q(d)-Q(d-1)
        u_degree = args.degree
        u_space = fem.make_polynomial_space(geo, degree=u_degree, dtype=wp.vec2)
        p_space = fem.make_polynomial_space(geo, degree=u_degree - 1)

        # Viscosity and inertia
        u_test = fem.make_test(space=u_space, domain=domain)
        u_trial = fem.make_trial(space=u_space, domain=domain)

        u_matrix = fem.integrate(
            viscosity_and_inertia_form,
            fields={"u": u_trial, "v": u_test},
            values={"nu": viscosity, "dt": self.sim_dt},
        )

        # Pressure-velocity coupling
        p_test = fem.make_test(space=p_space, domain=domain)
        div_matrix = fem.integrate(div_form, fields={"u": u_trial, "q": p_test})

        # Enforcing the Dirichlet boundary condition the hard way;
        # build projector for velocity left- and right-hand-sides
        u_bd_test = fem.make_test(space=u_space, domain=boundary)
        u_bd_trial = fem.make_trial(space=u_space, domain=boundary)
        u_bd_projector = fem.integrate(mass_form, fields={"u": u_bd_trial, "v": u_bd_test}, nodal=True)
        u_bd_value = fem.integrate(
            u_boundary_value,
            fields={"v": u_bd_test},
            values={"top_vel": args.top_velocity},
            nodal=True,
            output_dtype=wp.vec2d,
        )

        fem.normalize_dirichlet_projector(u_bd_projector, u_bd_value)

        u_bd_rhs = wp.zeros_like(u_bd_value)
        fem.project_linear_system(u_matrix, u_bd_rhs, u_bd_projector, u_bd_value, normalize_projector=False)

        # div_bd_rhs = div_matrix * u_bd_rhs
        div_bd_rhs = wp.zeros(shape=(div_matrix.nrow,), dtype=div_matrix.scalar_type)
        bsr_mv(div_matrix, u_bd_rhs, y=div_bd_rhs)

        # div_matrix = div_matrix - div_matrix * bd_projector
        bsr_mm(x=bsr_copy(div_matrix), y=u_bd_projector, z=div_matrix, alpha=-1.0, beta=1.0)

        # Assemble saddle system with Scipy
        div_matrix = bsr_to_scipy(div_matrix)
        u_matrix = bsr_to_scipy(u_matrix)
        div_bd_rhs = div_bd_rhs.numpy()

        ones = np.ones(shape=(p_space.node_count(), 1), dtype=float)
        saddle_system = bmat(
            [
                [u_matrix, div_matrix.transpose(), None],
                [div_matrix, None, ones],
                [None, ones.transpose(), None],
            ],
        )

        with wp.ScopedTimer("LU factorization"):
            self._solve_saddle = factorized(saddle_system)

        # Save data for computing time steps rhs
        self._u_bd_projector = u_bd_projector
        self._u_bd_rhs = u_bd_rhs
        self._u_test = u_test
        self._div_bd_rhs = div_bd_rhs

        # Velocitiy field

        self._u_field = u_space.make_field()

        self.renderer = Plot(stage)
        self.renderer.add_surface_vector("velocity", self._u_field)

    def update(self):
        self.current_frame += 1

        u_rhs = fem.integrate(
            transported_inertia_form,
            fields={"u": self._u_field, "v": self._u_test},
            values={"dt": self.sim_dt},
            output_dtype=wp.vec2d,
        )

        # Apply boundary conditions
        # u_rhs = (I - P) * u_rhs + u_bd_rhs
        bsr_mv(self._u_bd_projector, x=u_rhs, y=u_rhs, alpha=-1.0, beta=1.0)
        array_axpy(x=self._u_bd_rhs, y=u_rhs, alpha=1.0, beta=1.0)

        # Assemble scipy saddle system rhs
        u_dof_count = self._u_bd_projector.shape[0]
        p_dof_count = self._div_bd_rhs.shape[0]
        tot_dof_count = u_dof_count + p_dof_count + 1

        u_slice = slice(0, u_dof_count)
        p_slice = slice(u_dof_count, tot_dof_count - 1)

        saddle_rhs = np.zeros(tot_dof_count)
        saddle_rhs[u_slice] = u_rhs.numpy().flatten()
        saddle_rhs[p_slice] = self._div_bd_rhs

        x = self._solve_saddle(saddle_rhs)

        # Extract result
        self._u_field.dof_values = x[u_slice].reshape((-1, 2))

    def render(self):
        self.renderer.begin_frame(time = self.current_frame * self.sim_dt)
        self.renderer.add_surface_vector("velocity", self._u_field)
        self.renderer.end_frame()


if __name__ == "__main__":
    wp.init()
    wp.set_module_options({"enable_backward": False})

    args = Example.parser.parse_args()

    example = Example(args=args)
    for k in range(args.num_frames):
        print(f"Frame {k}:")
        example.update()
        example.render()

    example.renderer.add_surface_vector("velocity_final", example._u_field)
    example.renderer.plot(streamlines=set(["velocity_final"]))
