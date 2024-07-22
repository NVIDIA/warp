# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Diffusion 3D
#
# This example solves a 3d diffusion problem:
#
# nu Div u = 1
#
# with homogeneous Neumann conditions on horizontal sides
# and homogeneous Dirichlet boundary conditions other sides.
###########################################################################

import warp as wp
import warp.examples.fem.utils as fem_example_utils
import warp.fem as fem
from warp.examples.fem.example_diffusion import diffusion_form, linear_form
from warp.sparse import bsr_axpy


@fem.integrand
def vert_boundary_projector_form(
    s: fem.Sample,
    domain: fem.Domain,
    u: fem.Field,
    v: fem.Field,
):
    # Non-zero mass on vertical sides only
    nor = fem.normal(domain, s)
    w = wp.abs(nor[0])
    return w * u(s) * v(s)


class Example:
    def __init__(
        self,
        quiet=False,
        degree=2,
        resolution=10,
        mesh="grid",
        serendipity=False,
        viscosity=2.0,
        boundary_compliance=0.0,
    ):
        self._quiet = quiet

        self._viscosity = viscosity
        self._boundary_compliance = boundary_compliance

        res = wp.vec3i(resolution, max(1, resolution // 2), resolution * 2)

        if mesh == "tet":
            pos, tet_vtx_indices = fem_example_utils.gen_tetmesh(
                res=res,
                bounds_lo=wp.vec3(0.0, 0.0, 0.0),
                bounds_hi=wp.vec3(1.0, 0.5, 2.0),
            )
            self._geo = fem.Tetmesh(tet_vtx_indices, pos)
        elif mesh == "hex":
            pos, hex_vtx_indices = fem_example_utils.gen_hexmesh(
                res=res,
                bounds_lo=wp.vec3(0.0, 0.0, 0.0),
                bounds_hi=wp.vec3(1.0, 0.5, 2.0),
            )
            self._geo = fem.Hexmesh(hex_vtx_indices, pos)
        elif mesh == "nano":
            volume = wp.Volume.allocate(min=[0, 0, 0], max=[1.0, 0.5, 2.0], voxel_size=1.0 / res[0], bg_value=None)
            self._geo = fem.Nanogrid(volume)
        else:
            self._geo = fem.Grid3D(
                res=res,
                bounds_lo=wp.vec3(0.0, 0.0, 0.0),
                bounds_hi=wp.vec3(1.0, 0.5, 2.0),
            )

        # Domain and function spaces
        element_basis = fem.ElementBasis.SERENDIPITY if serendipity else None
        self._scalar_space = fem.make_polynomial_space(self._geo, degree=degree, element_basis=element_basis)

        # Scalar field over our function space
        self._scalar_field: fem.DiscreteField = self._scalar_space.make_field()

        self.renderer = fem_example_utils.Plot()

    def step(self):
        geo = self._geo

        domain = fem.Cells(geometry=geo)

        # Right-hand-side
        test = fem.make_test(space=self._scalar_space, domain=domain)
        rhs = fem.integrate(linear_form, fields={"v": test})

        # Weakly-imposed boundary conditions on Y sides
        with wp.ScopedTimer("Integrate"):
            boundary = fem.BoundarySides(geo)

            bd_test = fem.make_test(space=self._scalar_space, domain=boundary)
            bd_trial = fem.make_trial(space=self._scalar_space, domain=boundary)
            bd_matrix = fem.integrate(vert_boundary_projector_form, fields={"u": bd_trial, "v": bd_test}, nodal=True)

            # Diffusion form
            trial = fem.make_trial(space=self._scalar_space, domain=domain)
            matrix = fem.integrate(diffusion_form, fields={"u": trial, "v": test}, values={"nu": self._viscosity})

        if self._boundary_compliance == 0.0:
            # Hard BC: project linear system
            bd_rhs = wp.zeros_like(rhs)
            fem.project_linear_system(matrix, rhs, bd_matrix, bd_rhs)
        else:
            # Weak BC: add together diffusion and boundary condition matrices
            boundary_strength = 1.0 / self._boundary_compliance
            bsr_axpy(x=bd_matrix, y=matrix, alpha=boundary_strength, beta=1)

        with wp.ScopedTimer("CG solve"):
            x = wp.zeros_like(rhs)
            fem_example_utils.bsr_cg(matrix, b=rhs, x=x, quiet=self._quiet)
            self._scalar_field.dof_values = x

    def render(self):
        self.renderer.add_field("solution", self._scalar_field)


if __name__ == "__main__":
    import argparse

    wp.set_module_options({"enable_backward": False})

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--resolution", type=int, default=10, help="Grid resolution.")
    parser.add_argument("--degree", type=int, default=2, help="Polynomial degree of shape functions.")
    parser.add_argument("--serendipity", action="store_true", default=False, help="Use Serendipity basis functions.")
    parser.add_argument("--viscosity", type=float, default=2.0, help="Fluid viscosity parameter.")
    parser.add_argument(
        "--boundary_compliance", type=float, default=0.0, help="Dirichlet boundary condition compliance."
    )
    parser.add_argument("--mesh", choices=("grid", "tet", "hex", "nano"), default="grid", help="Mesh type.")
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
            boundary_compliance=args.boundary_compliance,
        )

        example.step()
        example.render()

        if not args.headless:
            example.renderer.plot()
