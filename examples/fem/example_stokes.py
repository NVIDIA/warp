"""
This example solves a 2D Stokes flow problem

  -nu D(u) + grad p = 0
  Div u = 0

with (soft) velocity-Dirichlet boundary conditions
"""

import argparse

import warp as wp

import numpy as np

from warp.fem import Field, Domain, Sample
from warp.fem import Grid2D, Trimesh2D
from warp.fem import make_test, make_trial, make_restriction
from warp.fem import make_polynomial_space, ElementBasis
from warp.fem import Cells, BoundarySides
from warp.fem import integrate, interpolate
from warp.fem import normal, integrand, D, div

from plot_utils import plot_velocities, plot_surface
from bsr_utils import bsr_to_scipy
from mesh_utils import gen_trimesh

from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve

import matplotlib.pyplot as plt


@integrand
def constant_form(val: wp.vec2):
    return val


@integrand
def viscosity_form(s: Sample, u: Field, v: Field, nu: float):
    return nu * wp.ddot(D(u, s), D(v, s))


@integrand
def top_mass_form(
    s: Sample,
    domain: Domain,
    u: Field,
    v: Field,
):
    # non zero on top boundary of domain only
    nor = normal(domain, s)
    return wp.dot(u(s), v(s)) * wp.max(0.0, nor[1])


@integrand
def mass_form(
    s: Sample,
    u: Field,
    v: Field,
):
    return wp.dot(u(s), v(s))


@integrand
def div_form(
    s: Sample,
    u: Field,
    q: Field,
):
    return q(s) * div(u, s)


if __name__ == "__main__":
    wp.init()
    wp.set_module_options({"enable_backward": False})


    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=50)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--top_velocity", type=float, default=1.0)
    parser.add_argument("--viscosity", type=float, default=1.0)
    parser.add_argument("--boundary_strength", type=float, default=100.0)
    parser.add_argument("--tri_mesh", action="store_true", help="Use a triangular mesh")
    parser.add_argument("--nonconforming_pressures", action="store_true", help="For grid, use non-conforming pressure (Q_d/P_{d-1})")
    args = parser.parse_args()
    
    top_velocity = wp.vec2(args.top_velocity, 0.0)

    if args.tri_mesh:
        positions, tri_vidx = gen_trimesh(res=wp.vec2i(args.resolution))
        geo = Trimesh2D(tri_vertex_indices=tri_vidx, positions=positions)
    else:
        geo = Grid2D(res=wp.vec2i(args.resolution))
    
    domain = Cells(geometry=geo)
    boundary = BoundarySides(geo)

    # Function spaces -- Q_d for vel, P_{d-1} for pressure
    u_space = make_polynomial_space(geo, degree=args.degree, dtype=wp.vec2)
    if not args.tri_mesh and args.nonconforming_pressures:
        p_space = make_polynomial_space(geo, degree=args.degree-1, element_basis=ElementBasis.NONCONFORMING_POLYNOMIAL)
    else:
        p_space = make_polynomial_space(geo, degree=args.degree-1)

    # Interpolate initial condition on boundary (mostly for testing)
    f = u_space.make_field()
    f_boundary = make_restriction(f, domain=boundary)
    interpolate(constant_form, dest=f_boundary, values={"val": top_velocity})

    # Viscosity
    u_test = make_test(space=u_space, domain=domain)
    u_trial = make_trial(space=u_space, domain=domain)

    u_visc_matrix = integrate(
        viscosity_form,
        fields={"u": u_trial, "v": u_test},
        values={"nu": args.viscosity},
    )

    # Weak velocity boundary conditions
    u_bd_test = make_test(space=u_space, domain=boundary)
    u_bd_trial = make_trial(space=u_space, domain=boundary)
    u_rhs = integrate(top_mass_form, fields={"u": f.trace(), "v": u_bd_test})
    u_bd_matrix = integrate(mass_form, fields={"u": u_bd_trial, "v": u_bd_test})

    # Pressure-velocity coupling
    p_test = make_test(space=p_space, domain=domain)
    div_matrix = integrate(div_form, fields={"u": u_trial, "q": p_test})

    # Solve with scipy
    # Assemble saddle-point system with velocity, pressure, and zero-average-pressure constraint
    u_rhs = u_rhs.numpy() * args.boundary_strength
    u_matrix = bsr_to_scipy(u_visc_matrix) + args.boundary_strength * bsr_to_scipy(u_bd_matrix)

    div_matrix = bsr_to_scipy(div_matrix)

    ones = np.ones(shape=(p_space.node_count(), 1), dtype=float)
    saddle_system = bmat(
        [
            [u_matrix, div_matrix.transpose(), None],
            [div_matrix, None, ones],
            [None, ones.transpose(), None],
        ],
        format="csr",
    )

    saddle_rhs = np.zeros(saddle_system.shape[0])
    u_slice = slice(0, 2 * u_space.node_count())
    p_slice = slice(
        2 * u_space.node_count(), 2 * u_space.node_count() + p_space.node_count()
    )
    saddle_rhs[u_slice] = u_rhs.flatten()

    x = spsolve(saddle_system, saddle_rhs)

    # Extract result
    u_field = u_space.make_field()
    p_field = p_space.make_field()

    u_field.dof_values = x[u_slice].reshape((-1, 2))
    p_field.dof_values = x[p_slice]

    plot_surface(p_field)
    plot_velocities(u_field)

    plt.show()
