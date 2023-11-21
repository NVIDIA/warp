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
    from .bsr_utils import bsr_cg
    from .mesh_utils import gen_trimesh, gen_quadmesh
    from .plot_utils import Plot
except ImportError:
    from bsr_utils import bsr_cg
    from mesh_utils import gen_trimesh, gen_quadmesh
    from plot_utils import Plot

wp.set_module_options({"enable_backward": False})


@fem.integrand
def linear_form(
    s: fem.Sample,
    v: fem.Field,
):
    """Linear form with constant slope 1 -- forcing term of our problem"""
    return v(s)


@fem.integrand
def diffusion_form(s: fem.Sample, u: fem.Field, v: fem.Field, nu: float):
    """Diffusion bilinear form with constant coefficient ``nu``"""
    return nu * wp.dot(
        fem.grad(u, s),
        fem.grad(v, s),
    )


@fem.integrand
def y_boundary_value_form(s: fem.Sample, domain: fem.Domain, v: fem.Field, val: float):
    """Linear form with coefficient val on vertical edges, zero elsewhere"""
    nor = fem.normal(domain, s)
    return val * v(s) * wp.abs(nor[0])


@fem.integrand
def y_boundary_projector_form(
    s: fem.Sample,
    domain: fem.Domain,
    u: fem.Field,
    v: fem.Field,
):
    """
    Bilinear boundary condition projector form, non-zero on vertical edges only.
    """
    # Reuse the above linear form implementation by evaluating one of the participating field and passing it as a normal scalar argument.
    return y_boundary_value_form(s, domain, v, u(s))


class Example:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=50)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--serendipity", action="store_true", default=False)
    parser.add_argument("--viscosity", type=float, default=2.0)
    parser.add_argument("--boundary_value", type=float, default=5.0)
    parser.add_argument("--boundary_compliance", type=float, default=0, help="Dirichlet boundary condition compliance")
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
            self._geo = fem.Trimesh2D(tri_vertex_indices=tri_vidx, positions=positions)
        elif args.mesh == "quad":
            positions, quad_vidx = gen_quadmesh(res=wp.vec2i(args.resolution))
            self._geo = fem.Quadmesh2D(quad_vertex_indices=quad_vidx, positions=positions)
        else:
            self._geo = fem.Grid2D(res=wp.vec2i(args.resolution))

        # Scalar function space
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

        # Boundary conditions on Y sides
        # Use nodal integration so that boundary conditions are specified on each node independently
        boundary = fem.BoundarySides(geo)
        bd_test = fem.make_test(space=self._scalar_space, domain=boundary)
        bd_trial = fem.make_trial(space=self._scalar_space, domain=boundary)

        bd_matrix = fem.integrate(y_boundary_projector_form, fields={"u": bd_trial, "v": bd_test}, nodal=True)
        bd_rhs = fem.integrate(
            y_boundary_value_form, fields={"v": bd_test}, values={"val": args.boundary_value}, nodal=True
        )

        # Assemble linear system
        if args.boundary_compliance == 0.0:
            # Hard BC: project linear system
            fem.project_linear_system(matrix, rhs, bd_matrix, bd_rhs)
        else:
            # Weak BC: add toegether diffusion and boundary condition matrices
            boundary_strength = 1.0 / args.boundary_compliance
            bsr_axpy(x=bd_matrix, y=matrix, alpha=boundary_strength, beta=1)
            array_axpy(x=bd_rhs, y=rhs, alpha=boundary_strength, beta=1)

        # Solve linear system using Conjugate Gradient
        x = wp.zeros_like(rhs)
        bsr_cg(matrix, b=rhs, x=x, quiet=self._quiet)

        # Assign system result to our discrete field
        self._scalar_field.dof_values = x

    def render(self):
        self.renderer.add_surface("solution", self._scalar_field)


if __name__ == "__main__":
    wp.init()

    args = Example.parser.parse_args()

    example = Example(args=args)
    example.update()
    example.render()

    example.renderer.plot()
