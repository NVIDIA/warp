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
# Example Deformed Geometry
#
# This example solves a 2d diffusion problem on a deformed (curved) mesh:
#
# nu Div u = 1
#
# with Dirichlet boundary conditions on vertical edges and
# homogeneous Neumann on horizontal edges.
###########################################################################

import warp as wp
import warp.examples.fem.utils as fem_example_utils
import warp.fem as fem
from warp.examples.fem.example_diffusion import diffusion_form, linear_form


@fem.integrand
def deformation_field_expr(
    s: fem.Sample,
    domain: fem.Domain,
):
    """
    Deformation field mapping the unit square to a circular band
    """
    x = domain(s)

    r = x[1] + 0.5
    t = 0.5 * 3.1416 * x[0]

    return r * wp.vec2(wp.sin(t), wp.cos(t))


@fem.integrand
def boundary_projector_form(
    s: fem.Sample,
    domain: fem.Domain,
    u: fem.Field,
    v: fem.Field,
):
    """
    Bilinear boundary condition projector form, non-zero on radial edges
    """
    nor = fem.normal(domain, s)
    active = wp.where(nor[0] < -0.9999 or nor[1] < -0.9999, 1.0, 0.0)
    return active * u(s) * v(s)


class Example:
    def __init__(
        self,
        quiet=False,
        degree=2,
        resolution=50,
        mesh="grid",
        serendipity=False,
        viscosity=2.0,
    ):
        self._quiet = quiet

        self._viscosity = viscosity

        # Grid or triangle mesh geometry
        if mesh == "tri":
            positions, tri_vidx = fem_example_utils.gen_trimesh(res=wp.vec2i(resolution))
            base_geo = fem.Trimesh2D(tri_vertex_indices=tri_vidx, positions=positions)
        elif mesh == "quad":
            positions, quad_vidx = fem_example_utils.gen_quadmesh(res=wp.vec2i(resolution))
            base_geo = fem.Quadmesh2D(quad_vertex_indices=quad_vidx, positions=positions)
        else:
            base_geo = fem.Grid2D(res=wp.vec2i(resolution))

        # Construct deformation field on base geometry
        deformation_space = fem.make_polynomial_space(base_geo, degree=degree, dtype=wp.vec2)
        deformation_field = deformation_space.make_field()
        fem.interpolate(deformation_field_expr, dest=deformation_field)

        self._geo = deformation_field.make_deformed_geometry(relative=False)

        # Scalar function space on deformed geometry
        element_basis = fem.ElementBasis.SERENDIPITY if serendipity else None
        self._scalar_space = fem.make_polynomial_space(self._geo, degree=degree, element_basis=element_basis)

        # Scalar field over our function space
        self._scalar_field = self._scalar_space.make_field()

        self.renderer = fem_example_utils.Plot()

    def step(self):
        geo = self._geo

        domain = fem.Cells(geometry=geo)

        # Right-hand-side (forcing term)
        test = fem.make_test(space=self._scalar_space, domain=domain)
        rhs = fem.integrate(linear_form, fields={"v": test})

        # Diffusion form
        trial = fem.make_trial(space=self._scalar_space, domain=domain)
        matrix = fem.integrate(diffusion_form, fields={"u": trial, "v": test}, values={"nu": self._viscosity})

        # Weakly-imposed boundary conditions on all sides
        boundary = fem.BoundarySides(geo)
        bd_test = fem.make_test(space=self._scalar_space, domain=boundary)
        bd_trial = fem.make_trial(space=self._scalar_space, domain=boundary)

        bd_matrix = fem.integrate(boundary_projector_form, fields={"u": bd_trial, "v": bd_test}, nodal=True)

        fem.project_linear_system(matrix, rhs, bd_matrix)

        # Solve linear system using Conjugate Gradient
        x = wp.zeros_like(rhs)
        fem_example_utils.bsr_cg(matrix, b=rhs, x=x, quiet=self._quiet, tol=1.0e-6)

        # Assign system result to our discrete field
        self._scalar_field.dof_values = x

    def render(self):
        self.renderer.add_field("solution", self._scalar_field)


if __name__ == "__main__":
    import argparse

    wp.set_module_options({"enable_backward": False})

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--resolution", type=int, default=50, help="Grid resolution.")
    parser.add_argument("--degree", type=int, default=2, help="Polynomial degree of shape functions.")
    parser.add_argument("--serendipity", action="store_true", default=False, help="Use Serendipity basis functions.")
    parser.add_argument("--viscosity", type=float, default=2.0, help="Fluid viscosity parameter.")
    parser.add_argument("--mesh", choices=("grid", "tri", "quad"), default="tri", help="Mesh type")
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
            degree=args.degree,
            resolution=args.resolution,
            mesh=args.mesh,
            serendipity=args.serendipity,
            viscosity=args.viscosity,
        )

        example.step()
        example.render()

        if not args.headless:
            example.renderer.plot()
