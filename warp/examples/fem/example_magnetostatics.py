# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# Example Magnetostatics
#
# This example demonstrates solving a 3d magnetostatics problem
# (a copper coil with radial current around a cylindrical iron core)
# using a curl-curl formulation and H(curl)-conforming function space
#
# 1/mu Curl B + j = 0
# Div. B = 0
#
# solved over field A such that B = Curl A,
# and Direchlet homogeneous essential boundary conditions
#
# This example also illustrates using an ImplictField to warp a grid mesh
# to a cylindrical domain
###########################################################################

import numpy as np

import warp as wp
import warp.examples.fem.utils as fem_example_utils
import warp.fem as fem

# Physics constants
MU_0 = wp.constant(np.pi * 4.0e-7)  # Vacuum magnetic permeability
MU_c = wp.constant(1.25e-6)  # Copper magnetic permeability
MU_i = wp.constant(6.0e-3)  # Iron magnetic permeability


@wp.func
def cube_to_cylinder(x: wp.vec3):
    # mapping from unit square to unit disk
    pos_xz = wp.vec3(x[0], 0.0, x[2])
    return wp.max(wp.abs(pos_xz)) * wp.normalize(pos_xz) + wp.vec3(0.0, x[1], 0.0)


@wp.func
def cube_to_cylinder_grad(x: wp.vec3):
    # gradient of mapping from unit square to unit disk
    pos_xz = wp.vec3(x[0], 0.0, x[2])
    if pos_xz == wp.vec3(0.0):
        grad = wp.mat33(0.0)
    else:
        dir_xz = wp.normalize(pos_xz)
        dir_grad = (wp.identity(n=3, dtype=float) - wp.outer(dir_xz, dir_xz)) / wp.length(pos_xz)

        abs_xz = wp.abs(pos_xz)
        xinf_grad = wp.where(
            abs_xz[0] > abs_xz[2], wp.vec(wp.sign(pos_xz[0]), 0.0, 0.0), wp.vec3(0.0, 0.0, wp.sign(pos_xz[2]))
        )
        grad = dir_grad * wp.max(abs_xz) + wp.outer(dir_xz, xinf_grad)

    grad[1, 1] = 1.0
    return grad


@wp.func
def permeability_field(
    pos: wp.vec3,
    core_radius: float,
    core_height: float,
    coil_internal_radius: float,
    coil_external_radius: float,
    coil_height: float,
):
    x = wp.abs(pos[0])
    y = wp.abs(pos[1])
    z = wp.abs(pos[2])

    r = wp.sqrt(x * x + z * z)

    if r <= core_radius:
        return wp.where(y < core_height, MU_i, MU_0)

    if r >= coil_internal_radius and r <= coil_external_radius:
        return wp.where(y < coil_height, MU_c, MU_0)

    return MU_0


@wp.func
def current_field(
    pos: wp.vec3,
    current: float,
    coil_internal_radius: float,
    coil_external_radius: float,
    coil_height: float,
):
    x = pos[0]
    y = wp.abs(pos[1])
    z = pos[2]

    r = wp.sqrt(x * x + z * z)

    return wp.where(
        y < coil_height and r >= coil_internal_radius and r <= coil_external_radius,
        wp.vec3(z, 0.0, -x) * current / r,
        wp.vec3(0.0),
    )


@fem.integrand
def curl_curl_form(s: fem.Sample, domain: fem.Domain, u: fem.Field, v: fem.Field, mu: fem.Field):
    return wp.dot(fem.curl(u, s), fem.curl(v, s)) / mu(s)


@fem.integrand
def mass_form(s: fem.Sample, domain: fem.Domain, v: fem.Field, u: fem.Field):
    return wp.dot(u(s), v(s))


@fem.integrand
def curl_expr(s: fem.Sample, u: fem.Field):
    return fem.curl(u, s)


class Example:
    def __init__(self, quiet=False, mesh: str = "grid", resolution=32, domain_radius=2.0, current=1.0e6):
        # We mesh the unit disk by first meshing the unit square, then building a deformed geometry
        # from an implicit mapping field

        if mesh == "hex":
            positions, hex_vidx = fem_example_utils.gen_hexmesh(
                bounds_lo=wp.vec3(-domain_radius, -domain_radius, -domain_radius),
                bounds_hi=wp.vec3(domain_radius, domain_radius, domain_radius),
                res=wp.vec3i(resolution, resolution, resolution),
            )
            cube_geo = fem.Hexmesh(hex_vertex_indices=hex_vidx, positions=positions)
        elif mesh == "tet":
            positions, tet_vidx = fem_example_utils.gen_tetmesh(
                bounds_lo=wp.vec3(-domain_radius, -domain_radius, -domain_radius),
                bounds_hi=wp.vec3(domain_radius, domain_radius, domain_radius),
                res=wp.vec3i(resolution, resolution, resolution),
            )
            cube_geo = fem.Tetmesh(tet_vertex_indices=tet_vidx, positions=positions)
        elif mesh == "nano":
            vol = fem_example_utils.gen_volume(
                bounds_lo=wp.vec3(-domain_radius, -domain_radius, -domain_radius),
                bounds_hi=wp.vec3(domain_radius, domain_radius, domain_radius),
                res=wp.vec3i(resolution, resolution, resolution),
            )
            cube_geo = fem.Nanogrid(grid=vol)
        else:
            cube_geo = fem.Grid3D(
                bounds_lo=wp.vec3(-domain_radius, -domain_radius, -domain_radius),
                bounds_hi=wp.vec3(domain_radius, domain_radius, domain_radius),
                res=wp.vec3i(resolution, resolution, resolution),
            )

        def_field = fem.ImplicitField(
            domain=fem.Cells(cube_geo), func=cube_to_cylinder, grad_func=cube_to_cylinder_grad
        )
        sim_geo = def_field.make_deformed_geometry(relative=False)

        coil_config = {"coil_height": 0.25, "coil_internal_radius": 0.3, "coil_external_radius": 0.4}
        core_config = {"core_height": 1.0, "core_radius": 0.2}

        domain = fem.Cells(sim_geo)
        self._permeability_field = fem.ImplicitField(
            domain, func=permeability_field, values=dict(**coil_config, **core_config)
        )
        self._current_field = fem.ImplicitField(domain, func=current_field, values=dict(current=current, **coil_config))

        A_space = fem.make_polynomial_space(
            sim_geo, degree=1, element_basis=fem.ElementBasis.NEDELEC_FIRST_KIND, dtype=wp.vec3
        )
        self.A_field = A_space.make_field()

        B_space = fem.make_polynomial_space(sim_geo, degree=1, element_basis=fem.ElementBasis.LAGRANGE, dtype=wp.vec3)
        self.B_field = B_space.make_field()

        self.renderer = fem_example_utils.Plot()

    def step(self):
        A_space = self.A_field.space
        sim_geo = A_space.geometry

        u = fem.make_trial(space=A_space)
        v = fem.make_test(space=A_space)
        lhs = fem.integrate(curl_curl_form, fields={"u": u, "v": v, "mu": self._permeability_field}, output_dtype=float)
        rhs = fem.integrate(mass_form, fields={"v": v, "u": self._current_field}, output_dtype=float)

        # Dirichlet BC
        boundary = fem.BoundarySides(sim_geo)
        u_bd = fem.make_trial(space=A_space, domain=boundary)
        v_bd = fem.make_test(space=A_space, domain=boundary)
        dirichlet_bd_proj = fem.integrate(mass_form, fields={"u": u_bd, "v": v_bd}, nodal=True, output_dtype=float)
        fem.project_linear_system(lhs, rhs, dirichlet_bd_proj)

        # solve using Conjugate Residual (numerically rhs may not be in image of lhs)
        fem_example_utils.bsr_cg(lhs, b=rhs, x=self.A_field.dof_values, method="cr", max_iters=250, quiet=False)

        # compute B as curl(A)
        fem.interpolate(curl_expr, dest=self.B_field, fields={"u": self.A_field})

    def render(self):
        self.renderer.add_field("B", self.B_field)


if __name__ == "__main__":
    import argparse

    wp.set_module_options({"enable_backward": False})

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--resolution", type=int, default=32, help="Grid resolution.")
    parser.add_argument("--mesh", type=str, default="grid", choices=["tet", "hex", "grid", "nano"], help="Mesh type.")
    parser.add_argument("--radius", type=float, default=2.0, help="Radius of simulation domain.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode, suppressing the opening of any graphical windows.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppresses the printing out of iteration residuals.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(quiet=args.quiet, mesh=args.mesh, resolution=args.resolution, domain_radius=args.radius)
        example.step()
        example.render()

        if not args.headless:
            example.renderer.plot({"B": {"streamlines": {"density": 1.0}}})
