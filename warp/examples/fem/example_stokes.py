# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Stokes
#
# This example solves a 2D Stokes flow problem
#
# -nu D(u) + grad p = 0
# Div u = 0
#
# with (soft) velocity-Dirichlet boundary conditions
###########################################################################

import warp as wp
import warp.examples.fem.utils as fem_example_utils
import warp.fem as fem
import warp.sparse as sparse
from warp.fem.utils import array_axpy


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
    def __init__(
        self,
        quiet=False,
        mesh="grid",
        degree=2,
        resolution=50,
        viscosity=1.0,
        top_velocity=1.0,
        boundary_strength=100.0,
        nonconforming_pressures=False,
    ):
        self._quiet = quiet

        self.viscosity = viscosity
        self.boundary_strength = boundary_strength

        # Grid or triangle mesh geometry
        if mesh == "tri":
            positions, tri_vidx = fem_example_utils.gen_trimesh(res=wp.vec2i(resolution))
            geo = fem.Trimesh2D(tri_vertex_indices=tri_vidx, positions=positions)
        elif mesh == "quad":
            positions, quad_vidx = fem_example_utils.gen_quadmesh(res=wp.vec2i(resolution))
            geo = fem.Quadmesh2D(quad_vertex_indices=quad_vidx, positions=positions)
        else:
            geo = fem.Grid2D(res=wp.vec2i(resolution))

        # Function spaces -- Q_d for vel, P_{d-1} for pressure
        u_space = fem.make_polynomial_space(geo, degree=degree, dtype=wp.vec2)
        if mesh != "tri" and nonconforming_pressures:
            p_space = fem.make_polynomial_space(
                geo, degree=degree - 1, element_basis=fem.ElementBasis.NONCONFORMING_POLYNOMIAL
            )
        else:
            p_space = fem.make_polynomial_space(geo, degree=degree - 1)

        # Vector and scalar fields
        self._u_field = u_space.make_field()
        self._p_field = p_space.make_field()

        # Interpolate initial condition on boundary (for example purposes)
        self._bd_field = u_space.make_field()
        f_boundary = fem.make_restriction(self._bd_field, domain=fem.BoundarySides(geo))
        top_velocity = wp.vec2(top_velocity, 0.0)
        fem.interpolate(constant_form, dest=f_boundary, values={"val": top_velocity})

        self.renderer = fem_example_utils.Plot()

    def step(self):
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
            values={"nu": self.viscosity},
        )

        # Weak velocity boundary conditions
        u_bd_test = fem.make_test(space=u_space, domain=boundary)
        u_bd_trial = fem.make_trial(space=u_space, domain=boundary)
        u_rhs = fem.integrate(
            top_mass_form, fields={"u": self._bd_field.trace(), "v": u_bd_test}, output_dtype=wp.vec2d
        )
        u_bd_matrix = fem.integrate(mass_form, fields={"u": u_bd_trial, "v": u_bd_test})

        # Pressure-velocity coupling
        p_test = fem.make_test(space=p_space, domain=domain)
        div_matrix = fem.integrate(div_form, fields={"u": u_trial, "q": p_test})

        # Define and solve the saddle-point system
        u_matrix = u_visc_matrix
        sparse.bsr_axpy(x=u_bd_matrix, y=u_matrix, alpha=self.boundary_strength, beta=1.0)
        array_axpy(x=u_rhs, y=u_rhs, alpha=0.0, beta=self.boundary_strength)

        p_rhs = wp.zeros(p_space.node_count(), dtype=wp.float64)
        x_u = wp.zeros_like(u_rhs)
        x_p = wp.zeros_like(p_rhs)

        fem_example_utils.bsr_solve_saddle(
            fem_example_utils.SaddleSystem(A=u_matrix, B=div_matrix),
            x_u=x_u,
            x_p=x_p,
            b_u=u_rhs,
            b_p=p_rhs,
            quiet=self._quiet,
        )

        wp.utils.array_cast(in_array=x_u, out_array=self._u_field.dof_values)
        wp.utils.array_cast(in_array=x_p, out_array=self._p_field.dof_values)

    def render(self):
        self.renderer.add_surface("pressure", self._p_field)
        self.renderer.add_surface_vector("velocity", self._u_field)


if __name__ == "__main__":
    import argparse

    wp.set_module_options({"enable_backward": False})

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--resolution", type=int, default=50, help="Grid resolution.")
    parser.add_argument("--degree", type=int, default=2, help="Polynomial degree of shape functions.")
    parser.add_argument(
        "--top_velocity",
        type=float,
        default=1.0,
        help="Horizontal velocity initial condition at the top of the domain.",
    )
    parser.add_argument("--viscosity", type=float, default=1.0, help="Fluid viscosity parameter.")
    parser.add_argument("--boundary_strength", type=float, default=100.0, help="Soft boundary condition strength.")
    parser.add_argument("--mesh", choices=("grid", "tri", "quad"), default="grid", help="Mesh type.")
    parser.add_argument(
        "--nonconforming_pressures", action="store_true", help="For grid, use non-conforming pressure (Q_d/P_{d-1})."
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode, suppressing the opening of any graphical windows.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppresses the printing out of iteration residuals.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(
            quiet=args.quiet,
            mesh=args.mesh,
            degree=args.degree,
            resolution=args.resolution,
            viscosity=args.viscosity,
            top_velocity=args.top_velocity,
            boundary_strength=args.boundary_strength,
            nonconforming_pressures=args.nonconforming_pressures,
        )
        example.step()
        example.render()

        if not args.headless:
            example.renderer.plot()
