"""
This example solves a 2d diffusion problem:
 nu Div u = 1
with Dirichlet boundary conditions on vertical edges and homogeneous Neumann on horizontal edges.
"""

import argparse

import warp as wp
import warp.fem as fem

from warp.sparse import bsr_axpy
from warp.fem.utils import array_axpy


# Import example utilities
# Make sure that works both when imported as module and run as standalone file
try:
    from .example_diffusion import linear_form, diffusion_form
    from .bsr_utils import bsr_cg
    from .mesh_utils import gen_trimesh, gen_quadmesh
    from .plot_utils import Plot
except ImportError:
    from example_diffusion import linear_form, diffusion_form 
    from bsr_utils import bsr_cg
    from mesh_utils import gen_trimesh, gen_quadmesh
    from plot_utils import Plot

@fem.integrand
def deformation_field_expr(
    s: fem.Sample,
    domain: fem.Domain,
):
    """
    Deformation field mapping the unique square to a circular band
    """
    x = domain(s)

    r = x[1] + 0.5
    t = 0.5 * 3.1416 * x[0]

    return r * wp.vec2(wp.sin(t), wp.cos(t)) - x


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
    active = wp.select(nor[0] < -0.9999 or nor[1] < -0.9999, 0.0, 1.0)
    return active * u(s) * v(s)


class Example:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=50)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--serendipity", action="store_true", default=False)
    parser.add_argument("--viscosity", type=float, default=2.0)
    parser.add_argument("--mesh", choices=("grid", "tri", "quad"), default="grid", help="Mesh type")

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
            base_geo = fem.Trimesh2D(tri_vertex_indices=tri_vidx, positions=positions)
        elif args.mesh == "quad":
            positions, quad_vidx = gen_quadmesh(res=wp.vec2i(args.resolution))
            base_geo = fem.Quadmesh2D(quad_vertex_indices=quad_vidx, positions=positions)
        else:
            base_geo = fem.Grid2D(res=wp.vec2i(args.resolution))

        # Construct deformation field on base geometry
        deformation_space = fem.make_polynomial_space(base_geo, degree=args.degree, dtype=wp.vec2)
        deformation_field = deformation_space.make_field()
        fem.interpolate(deformation_field_expr, dest=deformation_field)

        self._geo = deformation_field.make_deformed_geometry()

        # Scalar function space on deformed geometry
        element_basis = fem.ElementBasis.SERENDIPITY if args.serendipity else None
        self._scalar_space = fem.make_polynomial_space(self._geo, degree=args.degree, element_basis=element_basis)

        # Scalar field over our function space
        self._scalar_field = self._scalar_space.make_field()

        self.renderer = Plot(stage)

    def update(self):
        args = self._args
        geo = self._geo

        domain = fem.Cells(geometry=geo)

        # Right-hand-side (forcing term)
        test = fem.make_test(space=self._scalar_space, domain=domain)
        rhs = fem.integrate(linear_form, fields={"v": test})

        # Diffusion form
        trial = fem.make_trial(space=self._scalar_space, domain=domain)
        matrix = fem.integrate(diffusion_form, fields={"u": trial, "v": test}, values={"nu": args.viscosity})

        # Weakly-imposed boundary conditions on all sides
        boundary = fem.BoundarySides(geo)
        bd_test = fem.make_test(space=self._scalar_space, domain=boundary)
        bd_trial = fem.make_trial(space=self._scalar_space, domain=boundary)

        bd_matrix = fem.integrate(boundary_projector_form, fields={"u": bd_trial, "v": bd_test}, nodal=True)

        fem.project_linear_system(matrix, rhs, bd_matrix)

        # Solve linear system using Conjugate Gradient
        x = wp.zeros_like(rhs)
        bsr_cg(matrix, b=rhs, x=x, quiet=self._quiet, tol=1.0e-6)

        # Assign system result to our discrete field
        self._scalar_field.dof_values = x

    def render(self):
        self.renderer.add_surface("solution", self._scalar_field)


if __name__ == "__main__":
    wp.init()
    wp.set_module_options({"enable_backward": False})

    args = Example.parser.parse_args()

    example = Example(args=args)
    example.update()
    example.render()

    example.renderer.plot()
