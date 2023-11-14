"""
This example solves a 2D Stokes flow problem

  -nu D(u) + grad p = 0
  Div u = 0

with (soft) velocity-Dirichlet boundary conditions
"""

import argparse

import warp as wp

import numpy as np

import warp.fem as fem

try:
    from .plot_utils import Plot
    from .bsr_utils import bsr_to_scipy
    from .mesh_utils import gen_trimesh, gen_quadmesh
except ImportError:
    from plot_utils import Plot
    from bsr_utils import bsr_to_scipy
    from mesh_utils import gen_trimesh, gen_quadmesh

# Need to solve a saddle-point system, use scipy for simplicity
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve


@fem.integrand
def constant_form(val: wp.vec2):
    return val


@fem.integrand
def viscosity_form(s: fem.Sample, u: fem.Field, v: fem.Field, nu: float):
    return nu * wp.ddot(fem.D(u, s), fem.D(v, s))


@fem.integrand
def top_mass_form(
    s: fem.Sample,
    domain: fem.Domain,
    u: fem.Field,
    v: fem.Field,
):
    # non zero on top boundary of domain only
    nor = fem.normal(domain, s)
    return wp.dot(u(s), v(s)) * wp.max(0.0, nor[1])


@fem.integrand
def mass_form(
    s: fem.Sample,
    u: fem.Field,
    v: fem.Field,
):
    return wp.dot(u(s), v(s))


@fem.integrand
def div_form(
    s: fem.Sample,
    u: fem.Field,
    q: fem.Field,
):
    return q(s) * fem.div(u, s)


class Example:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=50)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--top_velocity", type=float, default=1.0)
    parser.add_argument("--viscosity", type=float, default=1.0)
    parser.add_argument("--boundary_strength", type=float, default=100.0)
    parser.add_argument("--mesh", choices=("grid", "tri", "quad"), default="grid", help="Mesh type")
    parser.add_argument(
        "--nonconforming_pressures", action="store_true", help="For grid, use non-conforming pressure (Q_d/P_{d-1})"
    )

    def __init__(self, stage=None, quiet=False, args=None, **kwargs):
        if args is None:
            # Read args from kwargs, add default arg values from parser
            args = argparse.Namespace(**kwargs)
            args = Example.parser.parse_args(args=[], namespace=args)
        self._args = args
        self._quiet = quiet

        # Grid or triangle mesh geometry
        if args.mesh == "tri":
            positions, tri_vidx = gen_trimesh(res=wp.vec2i(args.resolution))
            geo = fem.Trimesh2D(tri_vertex_indices=tri_vidx, positions=positions)
        elif args.mesh == "quad":
            positions, quad_vidx = gen_quadmesh(res=wp.vec2i(args.resolution))
            geo = fem.Quadmesh2D(quad_vertex_indices=quad_vidx, positions=positions)
        else:
            geo = fem.Grid2D(res=wp.vec2i(args.resolution))

        # Function spaces -- Q_d for vel, P_{d-1} for pressure
        u_space = fem.make_polynomial_space(geo, degree=args.degree, dtype=wp.vec2)
        if args.mesh != "tri" and args.nonconforming_pressures:
            p_space = fem.make_polynomial_space(
                geo, degree=args.degree - 1, element_basis=fem.ElementBasis.NONCONFORMING_POLYNOMIAL
            )
        else:
            p_space = fem.make_polynomial_space(geo, degree=args.degree - 1)

        # Vector and scalar fields
        self._u_field = u_space.make_field()
        self._p_field = p_space.make_field()

        # Interpolate initial condition on boundary (for example purposes)
        self._bd_field = u_space.make_field()
        f_boundary = fem.make_restriction(self._bd_field, domain=fem.BoundarySides(geo))
        top_velocity = wp.vec2(args.top_velocity, 0.0)
        fem.interpolate(constant_form, dest=f_boundary, values={"val": top_velocity})

        self.renderer = Plot(stage)

    def update(self):
        args = self._args
        u_space = self._u_field.space
        p_space = self._p_field.space
        geo = u_space.geometry

        domain = fem.Cells(geometry=geo)
        boundary = fem.BoundarySides(geo)

        # Viscosity
        u_test = fem.make_test(space=u_space, domain=domain)
        u_trial = fem.make_trial(space=u_space, domain=domain)

        u_visc_matrix = fem.integrate(
            viscosity_form,
            fields={"u": u_trial, "v": u_test},
            values={"nu": args.viscosity},
        )

        # Weak velocity boundary conditions
        u_bd_test = fem.make_test(space=u_space, domain=boundary)
        u_bd_trial = fem.make_trial(space=u_space, domain=boundary)
        u_rhs = fem.integrate(top_mass_form, fields={"u": self._bd_field.trace(), "v": u_bd_test})
        u_bd_matrix = fem.integrate(mass_form, fields={"u": u_bd_trial, "v": u_bd_test})

        # Pressure-velocity coupling
        p_test = fem.make_test(space=p_space, domain=domain)
        div_matrix = fem.integrate(div_form, fields={"u": u_trial, "q": p_test})

        # Solve with scipy
        # Assemble saddle-point system with velocity, pressure, and zero-average-pressure constraint
        u_rhs = u_rhs.numpy() * args.boundary_strength
        u_matrix = bsr_to_scipy(u_visc_matrix) + args.boundary_strength * bsr_to_scipy(u_bd_matrix)

        div_matrix = bsr_to_scipy(div_matrix)

        ones = np.ones(shape=(p_space.node_count(), 1), dtype=float)
        saddle_system = bmat(
            [
                [u_matrix, div_matrix.transpose(), None],
                [div_matrix, None, ones],
                [None, ones.transpose(), None],
            ],
            format="csr",
        )

        saddle_rhs = np.zeros(saddle_system.shape[0])
        u_slice = slice(0, 2 * u_space.node_count())
        p_slice = slice(2 * u_space.node_count(), 2 * u_space.node_count() + p_space.node_count())
        saddle_rhs[u_slice] = u_rhs.flatten()

        x = spsolve(saddle_system, saddle_rhs)

        # Extract result

        self._u_field.dof_values = x[u_slice].reshape((-1, 2))
        self._p_field.dof_values = x[p_slice]

    def render(self):
        self.renderer.add_surface("pressure", self._p_field)
        self.renderer.add_surface_vector("velocity", self._u_field)


if __name__ == "__main__":
    wp.init()
    wp.set_module_options({"enable_backward": False})

    args = Example.parser.parse_args()

    example = Example(args=args)
    example.update()
    example.render()

    example.renderer.plot()
