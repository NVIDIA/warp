""" This example illustrates using Mixed FEM to solve a 2D linear elasticity problem
 Div[ E: D(u) ] = 0
with Dirichlet boundary conditions on horizontal sides, and E the elasticity rank-4 tensor
"""
import argparse

import warp as wp

import numpy as np

from warp.fem import Sample, Field, Domain
from warp.fem import Grid2D, Trimesh2D
from warp.fem import make_test, make_trial
from warp.fem import make_polynomial_space, SymmetricTensorMapper
from warp.fem import Cells, BoundarySides
from warp.fem import integrate
from warp.fem import normal, integrand, D
from warp.fem import project_linear_system

from warp.sparse import bsr_transposed, bsr_mm

from plot_utils import plot_velocities
from bsr_utils import bsr_cg, invert_diagonal_bsr_mass_matrix
from mesh_utils import gen_trimesh

import matplotlib.pyplot as plt


@wp.func
def compute_stress(tau: wp.mat22, E: wp.mat33):
    """Strain to stress computation"""
    tau_sym = wp.vec3(tau[0, 0], tau[1, 1], tau[0, 1] + tau[1, 0])
    sig_sym = E * tau_sym
    return wp.mat22(sig_sym[0], 0.5 * sig_sym[2], 0.5 * sig_sym[2], sig_sym[1])


@integrand
def symmetric_grad_form(
    s: Sample,
    u: Field,
    tau: Field,
):
    """D(u) : tau"""
    return wp.ddot(tau(s), D(u, s))


@integrand
def stress_form(s: Sample, u: Field, tau: Field, E: wp.mat33):
    """(E : D(u)) : tau"""
    return wp.ddot(tau(s), compute_stress(D(u, s), E))


@integrand
def horizontal_boundary_projector_form(
    s: Sample,
    domain: Domain,
    u: Field,
    v: Field,
):
    # non zero on horizontal boundary of domain only
    nor = normal(domain, s)
    return wp.dot(u(s), v(s)) * wp.abs(nor[1])


@integrand
def horizontal_displacement_form(
    s: Sample,
    domain: Domain,
    v: Field,
    displacement: float,
):
    # opposed to normal on horizontal boundary of domain only
    nor = normal(domain, s)
    return -wp.abs(nor[1]) * displacement * wp.dot(nor, v(s))


@integrand
def tensor_mass_form(
    s: Sample,
    sig: Field,
    tau: Field,
):
    return wp.ddot(tau(s), sig(s))


if __name__ == "__main__":
    wp.init()
    wp.set_module_options({"enable_backward": False})

    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=25)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--displacement", type=float, default=0.1)
    parser.add_argument("--young_modulus", type=float, default=1.0)
    parser.add_argument("--poisson_ratio", type=float, default=0.5)
    parser.add_argument("--tri_mesh", action="store_true", help="Use a triangular mesh")
    args = parser.parse_args()

    if args.tri_mesh:
        positions, tri_vidx = gen_trimesh(res=wp.vec2i(args.resolution))
        geo = Trimesh2D(tri_vertex_indices=tri_vidx, positions=positions)
    else:
        geo = Grid2D(res=wp.vec2i(args.resolution))

    boundary = BoundarySides(geo)

    # Strain-stress matrix
    young = args.young_modulus
    poisson = args.poisson_ratio
    elasticity_mat = wp.mat33(
        young
        / (1.0 - poisson * poisson)
        * np.array(
            [[1.0, poisson, 0.0], [poisson, 1.0, 0.0], [0.0, 0.0, (2.0 * (1.0 + poisson)) * (1.0 - poisson * poisson)]]
        )
    )

    domain = Cells(geometry=geo)

    # Function spaces -- Q_k for displacement, Q_{k-1}d for stress
    u_space = make_polynomial_space(geo, degree=args.degree, dtype=wp.vec2)
    # Store stress degrees of freedom as symmetric tensors (3 dof) rather than full 2x2 matrices
    tau_space = make_polynomial_space(
        geo, degree=args.degree - 1, discontinuous=True, dof_mapper=SymmetricTensorMapper(wp.mat22)
    )

    # Displacement boundary conditions
    u_bd_test = make_test(space=u_space, domain=boundary)
    u_bd_trial = make_trial(space=u_space, domain=boundary)
    u_bd_rhs = integrate(
        horizontal_displacement_form,
        fields={"v": u_bd_test},
        values={"displacement": args.displacement},
        nodal=True,
        output_dtype=wp.vec2d,
    )
    u_bd_matrix = integrate(horizontal_boundary_projector_form, fields={"u": u_bd_trial, "v": u_bd_test}, nodal=True)

    # Stress/velocity coupling
    u_trial = make_trial(space=u_space, domain=domain)
    tau_test = make_test(space=tau_space, domain=domain)
    tau_trial = make_trial(space=tau_space, domain=domain)

    sym_grad_matrix = integrate(symmetric_grad_form, fields={"u": u_trial, "tau": tau_test})
    stress_matrix = integrate(stress_form, fields={"u": u_trial, "tau": tau_test}, values={"E": elasticity_mat})

    # Compute inverse of the (block-diagonal) tau mass matrix
    tau_inv_mass_matrix = integrate(tensor_mass_form, fields={"sig": tau_trial, "tau": tau_test}, nodal=True)
    invert_diagonal_bsr_mass_matrix(tau_inv_mass_matrix)

    # Assemble system matrix
    u_matrix = bsr_mm(bsr_transposed(sym_grad_matrix), bsr_mm(tau_inv_mass_matrix, stress_matrix))

    # Enforce boundary conditions
    u_rhs = wp.zeros_like(u_bd_rhs)
    project_linear_system(u_matrix, u_rhs, u_bd_matrix, u_bd_rhs)

    x = wp.zeros_like(u_rhs)
    bsr_cg(u_matrix, b=u_rhs, x=x, tol=1.0e-16)

    # Extract result
    u_field = u_space.make_field()
    u_field.dof_values = x  # .reshape((-1, 2))

    plot_velocities(u_field)

    plt.show()
