"""
This example solves a 2d diffusion problem:
 nu Div u = 1
with Dirichlet boundary conditions on vertical edges and homogeneous Neumann on horizontal edges.
"""

import argparse

import warp as wp

from warp.sparse import bsr_axpy

from warp.fem import Sample, Field, Domain
from warp.fem import Grid2D, Trimesh2D
from warp.fem import make_polynomial_space, ElementBasis
from warp.fem import make_test, make_trial
from warp.fem import Cells, BoundarySides
from warp.fem import integrate
from warp.fem import grad, normal, integrand
from warp.fem import project_linear_system

from warp.fem.utils import array_axpy

from plot_utils import plot_surface
from bsr_utils import bsr_cg
from mesh_utils import gen_trimesh

import matplotlib.pyplot as plt

@integrand
def linear_form(
    s: Sample,
    v: Field,
):
    """Linear form with constant slope 1 -- forcing term of our problem"""
    return v(s)


@integrand
def diffusion_form(s: Sample, u: Field, v: Field, nu: float):
    """Diffusion bilinear form with constant coefficient ``nu``"""
    return nu * wp.dot(
        grad(u, s),
        grad(v, s),
    )


@integrand
def y_boundary_value_form(s: Sample, domain: Domain, v: Field, val: float):
    """Linear form with coefficient val on vertical edges, zero elsewhere"""
    nor = normal(domain, s)
    return val * v(s) * wp.abs(nor[0])


@integrand
def y_boundary_projector_form(
    s: Sample,
    domain: Domain,
    u: Field,
    v: Field,
):
    """
    Bilinear boundary condition projector form, non-zero on vertical edges only.
    """
    # Reuse the above linear form implementation by evaluating one of the participating field and passing it as a normal scalar argument.
    return y_boundary_value_form(s, domain, v, u(s))


if __name__ == "__main__":

    wp.init()
    wp.set_module_options({"enable_backward": False})

    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=50)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--serendipity", action="store_true", default=False)
    parser.add_argument("--viscosity", type=float, default=2.0)
    parser.add_argument("--boundary_value", type=float, default=5.0)
    parser.add_argument("--boundary_compliance", type=float, default=0, help="Dirichlet boundary condition compliance")
    parser.add_argument("--tri_mesh", action="store_true", help="Use a triangular mesh")
    args = parser.parse_args()

    # Grid or triangle mesh geometry
    if args.tri_mesh:
        positions, tri_vidx = gen_trimesh(res=wp.vec2i(args.resolution))
        geo = Trimesh2D(tri_vertex_indices=tri_vidx, positions=positions)
    else:
        geo = Grid2D(res=wp.vec2i(args.resolution))

    # Domain and function spaces
    domain = Cells(geometry=geo)
    element_basis = ElementBasis.SERENDIPITY if args.serendipity else None
    scalar_space = make_polynomial_space(geo, degree=args.degree, element_basis=element_basis)

    # Right-hand-side (forcing term)
    test = make_test(space=scalar_space, domain=domain)
    rhs = integrate(linear_form, fields={"v": test})


    # Diffusion form
    trial = make_trial(space=scalar_space, domain=domain)
    matrix = integrate(diffusion_form, fields={"u": trial, "v": test}, values={"nu": args.viscosity})

    # Boundary conditions on Y sides
    # Use nodal integration so that boundary conditions are specified on each node independently
    boundary = BoundarySides(geo)
    bd_test = make_test(space=scalar_space, domain=boundary)
    bd_trial = make_trial(space=scalar_space, domain=boundary)

    bd_matrix = integrate(y_boundary_projector_form, fields={"u": bd_trial, "v": bd_test}, nodal=True)
    bd_rhs = integrate(y_boundary_value_form, fields={"v": bd_test}, values={"val": args.boundary_value}, nodal=True)

    # Assemble linear system
    if args.boundary_compliance == 0.0:
        # Hard BC: project linear system
        project_linear_system(matrix, rhs, bd_matrix, bd_rhs)
    else:
        # Weak BC: add toegether diffusion and boundary condition matrices
        boundary_strength = 1.0 / args.boundary_compliance
        bsr_axpy(x=bd_matrix, y=matrix, alpha=boundary_strength, beta=1)
        array_axpy(x=bd_rhs, y=rhs, alpha=boundary_strength, beta=1)

    # Solve linear system using Conjugate Gradient
    x = wp.zeros_like(rhs)
    bsr_cg(matrix, b=rhs, x=x)

    # Assign system result to a discrete field,
    scalar_field = scalar_space.make_field()
    scalar_field.dof_values = x

    # Visualize it with matplotlib
    plot_surface(scalar_field)
    plt.show()
