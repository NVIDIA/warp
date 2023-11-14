"""
This example solves a 3d diffusion problem:
 nu Div u = 1
with homogeneous Neumann conditions on horizontal sides and homogeneous Dirichlet boundary conditions other sides.
"""

import argparse

import warp as wp
import warp.fem as fem

from warp.sparse import bsr_axpy

# Import example utilities
# Make sure that works both when imported as module and run as standalone file
try:
    from .example_diffusion import diffusion_form, linear_form
    from .bsr_utils import bsr_cg
    from .mesh_utils import gen_tetmesh
    from .plot_utils import Plot
except ImportError:
    from example_diffusion import diffusion_form, linear_form
    from bsr_utils import bsr_cg
    from mesh_utils import gen_tetmesh, gen_hexmesh
    from plot_utils import Plot


@fem.integrand
def vert_boundary_projector_form(
    s: fem.Sample,
    domain: fem.Domain,
    u: fem.Field,
    v: fem.Field,
):
    # Non-zero mass on vertical sides only
    w = 1.0 - wp.abs(fem.normal(domain, s)[1])
    return w * u(s) * v(s)


class Example:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=10)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--serendipity", action="store_true", default=False)
    parser.add_argument("--viscosity", type=float, default=2.0)
    parser.add_argument("--boundary_compliance", type=float, default=0, help="Dirichlet boundary condition compliance")
    parser.add_argument("--mesh", choices=("grid", "tet", "hex"), default="grid", help="Mesh type")

    def __init__(self, stage=None, quiet=False, args=None, **kwargs):
        if args is None:
            # Read args from kwargs, add default arg values from parser
            args = argparse.Namespace(**kwargs)
            args = Example.parser.parse_args(args=[], namespace=args)
        self._args = args
        self._quiet = quiet

        res = wp.vec3i(args.resolution, args.resolution // 2, args.resolution * 2)

        if args.mesh == "tet":
            pos, tet_vtx_indices = gen_tetmesh(
                res=res,
                bounds_lo=wp.vec3(0.0, 0.0, 0.0),
                bounds_hi=wp.vec3(1.0, 0.5, 2.0),
            )
            self._geo = fem.Tetmesh(tet_vtx_indices, pos)
        elif args.mesh == "hex":
            pos, hex_vtx_indices = gen_hexmesh(
                res=res,
                bounds_lo=wp.vec3(0.0, 0.0, 0.0),
                bounds_hi=wp.vec3(1.0, 0.5, 2.0),
            )
            self._geo = fem.Hexmesh(hex_vtx_indices, pos)
        else:
            self._geo = fem.Grid3D(
                res=res,
                bounds_lo=wp.vec3(0.0, 0.0, 0.0),
                bounds_hi=wp.vec3(1.0, 0.5, 2.0),
            )

        # Domain and function spaces
        element_basis = fem.ElementBasis.SERENDIPITY if args.serendipity else None
        self._scalar_space = fem.make_polynomial_space(self._geo, degree=args.degree, element_basis=element_basis)

        # Scalar field over our function space
        self._scalar_field: fem.DiscreteField = self._scalar_space.make_field()

        self.renderer = Plot(stage)

    def update(self):
        args = self._args
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
            matrix = fem.integrate(diffusion_form, fields={"u": trial, "v": test}, values={"nu": args.viscosity})

        if args.boundary_compliance == 0.0:
            # Hard BC: project linear system
            bd_rhs = wp.zeros_like(rhs)
            fem.project_linear_system(matrix, rhs, bd_matrix, bd_rhs)
        else:
            # Weak BC: add toegether diffusion and boundary condition matrices
            boundary_strength = 1.0 / args.boundary_compliance
            bsr_axpy(x=bd_matrix, y=matrix, alpha=boundary_strength, beta=1)

        with wp.ScopedTimer("CG solve"):
            x = wp.zeros_like(rhs)
            bsr_cg(matrix, b=rhs, x=x, quiet=self._quiet)
            self._scalar_field.dof_values = x

    def render(self):
        self.renderer.add_volume("solution", self._scalar_field)


if __name__ == "__main__":
    wp.init()
    wp.set_module_options({"enable_backward": False})

    args = Example.parser.parse_args()

    example = Example(args=args)
    example.update()
    example.render()

    example.renderer.plot()
