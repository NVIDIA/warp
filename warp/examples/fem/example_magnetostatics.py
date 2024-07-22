# Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Magnetostatics
#
# This example demonstrates solving an in-plane magnetostatics problem
# using a curl-curl formulation
#
# 1/mu Curl B + j = 0
# Div. B = 0
#
# solved over field A such that B = Curl A, A = (0, 0, a_z),
# and a_z = 0 on the domain boundary
#
# This example also illustrates using an ImplictField to warp a square mesh
# to a circular domain
###########################################################################

import numpy as np

import warp as wp
import warp.examples.fem.utils as fem_example_utils
import warp.fem as fem

# Vacuum and copper magnetic permeabilities
MU_0 = wp.constant(np.pi * 4.0e-7)
MU_C = wp.constant(1.25e-6)


@wp.func
def square_to_disk(x: wp.vec2):
    # mapping from unit square to unit disk
    return wp.normalize(x) * wp.max(wp.abs(x))


@wp.func
def square_to_disk_grad(x: wp.vec2):
    # gradient of mapping from unit square to unit disk
    if x == wp.vec2(0.0):
        return wp.mat22(0.0)

    d = wp.normalize(x)
    d_grad = (wp.identity(n=2, dtype=float) - wp.outer(d, d)) / wp.length(x)

    ax = wp.abs(x)
    xinf = wp.max(ax)

    xinf_grad = wp.select(ax[0] > ax[1], wp.vec2(0.0, wp.sign(x[1])), wp.vec(wp.sign(x[0]), 0.0))

    return d_grad * xinf + wp.outer(d, xinf_grad)


@wp.func
def permeability_field(pos: wp.vec2, coil_height: float, coil_internal_radius: float, coil_external_radius: float):
    # space-varying permeability

    x = wp.abs(pos[0])
    y = wp.abs(pos[1])
    return wp.select(
        y < coil_height and x > coil_internal_radius and x < coil_external_radius,
        MU_0,
        MU_C,
    )


@wp.func
def current_field(
    pos: wp.vec2, coil_height: float, coil_internal_radius: float, coil_external_radius: float, current: float
):
    # space-varying current direction along z axis (0, +1 or -1)
    x = wp.abs(pos[0])
    y = wp.abs(pos[1])
    return (
        wp.select(
            y < coil_height and x > coil_internal_radius and x < coil_external_radius,
            0.0,
            wp.sign(pos[0]),
        )
        * current
    )


@fem.integrand
def curl_z(u: fem.Field, s: fem.Sample):
    # projection of curl((0, 0, u)) over z axis
    du = fem.grad(u, s)
    return wp.vec2(du[1], -du[0])


@fem.integrand
def curl_curl_form(s: fem.Sample, domain: fem.Domain, u: fem.Field, v: fem.Field, mu: fem.Field):
    return wp.dot(curl_z(u, s), curl_z(v, s)) / mu(s)


@fem.integrand
def mass_form(s: fem.Sample, domain: fem.Domain, v: fem.Field, u: fem.Field):
    return u(s) * v(s)


class Example:
    def __init__(self, quiet=False, degree=2, resolution=32, domain_radius=2.0, current=1.0e6):
        # We mesh the unit disk by first meshing the unit square, then building a deformed geometry
        # from an implicit mapping field
        square_geo = fem.Grid2D(
            bounds_lo=wp.vec2(-domain_radius, -domain_radius),
            bounds_hi=wp.vec2(domain_radius, domain_radius),
            res=wp.vec2i(resolution, resolution),
        )

        def_field = fem.ImplicitField(domain=fem.Cells(square_geo), func=square_to_disk, grad_func=square_to_disk_grad)
        disk_geo = def_field.make_deformed_geometry(relative=False)

        coil_config = {"coil_height": 1.0, "coil_internal_radius": 0.1, "coil_external_radius": 0.3}

        domain = fem.Cells(disk_geo)
        self._permeability_field = fem.ImplicitField(domain, func=permeability_field, values=coil_config)
        self._current_field = fem.ImplicitField(domain, func=current_field, values=dict(current=current, **coil_config))

        z_space = fem.make_polynomial_space(disk_geo, degree=degree, element_basis=fem.ElementBasis.LAGRANGE)
        xy_space = fem.make_polynomial_space(
            disk_geo, degree=degree, element_basis=fem.ElementBasis.LAGRANGE, dtype=wp.vec2
        )

        self.A_field = z_space.make_field()
        self.B_field = xy_space.make_field()

        self.renderer = fem_example_utils.Plot()

    def step(self):
        z_space = self.A_field.space
        disk_geo = z_space.geometry

        u = fem.make_trial(space=z_space)
        v = fem.make_test(space=z_space)
        lhs = fem.integrate(curl_curl_form, fields={"u": u, "v": v, "mu": self._permeability_field})
        rhs = fem.integrate(mass_form, fields={"v": v, "u": self._current_field})

        # Dirichlet BC
        boundary = fem.BoundarySides(disk_geo)
        u_bd = fem.make_trial(space=z_space, domain=boundary)
        v_bd = fem.make_test(space=z_space, domain=boundary)
        dirichlet_bd_proj = fem.integrate(mass_form, fields={"u": u_bd, "v": v_bd}, nodal=True)
        fem.project_linear_system(lhs, rhs, dirichlet_bd_proj)

        x = wp.zeros_like(rhs)
        fem_example_utils.bsr_cg(lhs, b=rhs, x=x, tol=1.0e-8, quiet=False)

        # make sure result is exactly zero outisde of circle
        wp.sparse.bsr_mv(dirichlet_bd_proj, x=x, y=x, alpha=-1.0, beta=1.0)
        wp.utils.array_cast(in_array=x, out_array=self.A_field.dof_values)

        # compute B as curl(A)
        fem.interpolate(curl_z, dest=self.B_field, fields={"u": self.A_field})

    def render(self):
        self.renderer.add_field("A", self.A_field)
        self.renderer.add_field("B", self.B_field)


if __name__ == "__main__":
    import argparse

    wp.set_module_options({"enable_backward": False})

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--resolution", type=int, default=32, help="Grid resolution.")
    parser.add_argument("--degree", type=int, default=2, help="Polynomial degree of shape functions.")
    parser.add_argument("--radius", type=float, default=2.0, help="Radius of simulation domain.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode, suppressing the opening of any graphical windows.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppresses the printing out of iteration residuals.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(quiet=args.quiet, degree=args.degree, resolution=args.resolution, domain_radius=args.radius)
        example.step()
        example.render()

        if not args.headless:
            example.renderer.plot({"A": {"contours": {"levels": 30}}, "B": {"streamlines": {"density": 1.0}}})
