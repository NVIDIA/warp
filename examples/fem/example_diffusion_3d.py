"""
This example solves a 3d diffusion problem:
 nu Div u = 1
with homogeneous Neumann conditions on horizontal sides and homogeneous Dirichlet boundary conditions other sides.
"""

import argparse

import warp as wp
from warp.sparse import bsr_axpy

from warp.fem import Field, Sample, Domain
from warp.fem import Grid3D, Tetmesh
from warp.fem import make_polynomial_space
from warp.fem import make_test, make_trial
from warp.fem import Cells, BoundarySides
from warp.fem import integrate
from warp.fem import normal, integrand
from warp.fem import project_linear_system

from plot_utils import plot_3d_scatter
from bsr_utils import bsr_cg
from mesh_utils import gen_tetmesh

from example_diffusion import diffusion_form, linear_form

import matplotlib.pyplot as plt


@integrand
def vert_boundary_projector_form(
    s: Sample,
    domain: Domain,
    u: Field,
    v: Field,
):
    # Non-zero mass on vertical sides only
    w = 1.0 - wp.abs(normal(domain, s)[1])
    return w * u(s) * v(s)


if __name__ == "__main__":
    wp.init()
    wp.set_module_options({"enable_backward": False})

    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=10)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--viscosity", type=float, default=2.0)
    parser.add_argument("--boundary_compliance", type=float, default=0, help="Dirichlet boundary condition compliance")
    parser.add_argument("--tet_mesh", action="store_true", help="Use a tetrahedral mesh")
    args = parser.parse_args()

    res = wp.vec3i(args.resolution, args.resolution // 2, args.resolution * 2)

    if args.tet_mesh:
        pos, tet_vtx_indices = gen_tetmesh(
            res=res,
            bounds_lo=wp.vec3(0.0, 0.0, 0.0),
            bounds_hi=wp.vec3(1.0, 0.5, 2.0),
        )
        geo = Tetmesh(tet_vtx_indices, pos)
    else:
        geo = Grid3D(
            res=res,
            bounds_lo=wp.vec3(0.0, 0.0, 0.0),
            bounds_hi=wp.vec3(1.0, 0.5, 2.0),
        )

    # Domain and function spaces
    domain = Cells(geometry=geo)
    scalar_space = make_polynomial_space(geo, degree=args.degree)

    # Right-hand-side
    test = make_test(space=scalar_space, domain=domain)
    rhs = integrate(linear_form, fields={"v": test})

    # Weakly-imposed boundary conditions on Y sides
    with wp.ScopedTimer("Integrate"):
        boundary = BoundarySides(geo)

        bd_test = make_test(space=scalar_space, domain=boundary)
        bd_trial = make_trial(space=scalar_space, domain=boundary)
        bd_matrix = integrate(vert_boundary_projector_form, fields={"u": bd_trial, "v": bd_test}, nodal=True)

        # Diffusion form
        trial = make_trial(space=scalar_space, domain=domain)
        matrix = integrate(diffusion_form, fields={"u": trial, "v": test}, values={"nu": args.viscosity})

    if args.boundary_compliance == 0.0:
        # Hard BC: project linear system
        bd_rhs = wp.zeros_like(rhs)
        project_linear_system(matrix, rhs, bd_matrix, bd_rhs)
    else:
        # Weak BC: add toegether diffusion and boundary condition matrices
        boundary_strength = 1.0 / args.boundary_compliance
        bsr_axpy(x=bd_matrix, y=matrix, alpha=100.0, beta=1)

    with wp.ScopedTimer("CG solve"):
        x = wp.zeros_like(rhs)
        bsr_cg(matrix, b=rhs, x=x)

    scalar_field = scalar_space.make_field()
    scalar_field.dof_values = x
    plot_3d_scatter(scalar_field)

    plt.show()
