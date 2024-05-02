# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Mixed Elasticity
#
# This example illustrates using Mixed FEM to solve a
# 2D linear elasticity problem:
#
# Div[ E: D(u) ] = 0
#
# with Dirichlet boundary conditions on horizontal sides,
# and E the elasticity rank-4 tensor
###########################################################################

import numpy as np

import warp as wp
import warp.fem as fem
from warp.sparse import bsr_mm, bsr_transposed

try:
    from .bsr_utils import bsr_cg, invert_diagonal_bsr_mass_matrix
    from .mesh_utils import gen_quadmesh, gen_trimesh
    from .plot_utils import Plot
except ImportError:
    from bsr_utils import bsr_cg, invert_diagonal_bsr_mass_matrix
    from mesh_utils import gen_quadmesh, gen_trimesh
    from plot_utils import Plot

wp.init()


@wp.func
def compute_stress(tau: wp.mat22, E: wp.mat33):
    """Strain to stress computation"""
    tau_sym = wp.vec3(tau[0, 0], tau[1, 1], tau[0, 1] + tau[1, 0])
    sig_sym = E * tau_sym
    return wp.mat22(sig_sym[0], 0.5 * sig_sym[2], 0.5 * sig_sym[2], sig_sym[1])


@fem.integrand
def symmetric_grad_form(
    s: fem.Sample,
    u: fem.Field,
    tau: fem.Field,
):
    """D(u) : tau"""
    return wp.ddot(tau(s), fem.D(u, s))


@fem.integrand
def stress_form(s: fem.Sample, u: fem.Field, tau: fem.Field, E: wp.mat33):
    """(E : D(u)) : tau"""
    return wp.ddot(tau(s), compute_stress(fem.D(u, s), E))


@fem.integrand
def horizontal_boundary_projector_form(
    s: fem.Sample,
    domain: fem.Domain,
    u: fem.Field,
    v: fem.Field,
):
    # non zero on horizontal boundary of domain only
    nor = fem.normal(domain, s)
    return wp.dot(u(s), v(s)) * wp.abs(nor[1])


@fem.integrand
def horizontal_displacement_form(
    s: fem.Sample,
    domain: fem.Domain,
    v: fem.Field,
    displacement: float,
):
    # opposed to normal on horizontal boundary of domain only
    nor = fem.normal(domain, s)
    return -wp.abs(nor[1]) * displacement * wp.dot(nor, v(s))


@fem.integrand
def tensor_mass_form(
    s: fem.Sample,
    sig: fem.Field,
    tau: fem.Field,
):
    return wp.ddot(tau(s), sig(s))


class Example:
    def __init__(
        self,
        quiet=False,
        degree=2,
        resolution=25,
        mesh="grid",
        displacement=0.1,
        young_modulus=1.0,
        poisson_ratio=0.5,
        nonconforming_stresses=False,
    ):
        self._quiet = quiet

        self._displacement = displacement

        # Grid or triangle mesh geometry
        if mesh == "tri":
            positions, tri_vidx = gen_trimesh(res=wp.vec2i(resolution))
            self._geo = fem.Trimesh2D(tri_vertex_indices=tri_vidx, positions=positions)
        elif mesh == "quad":
            positions, quad_vidx = gen_quadmesh(res=wp.vec2i(resolution))
            self._geo = fem.Quadmesh2D(quad_vertex_indices=quad_vidx, positions=positions)
        else:
            self._geo = fem.Grid2D(res=wp.vec2i(resolution))

        # Strain-stress matrix
        young = young_modulus
        poisson = poisson_ratio
        self._elasticity_mat = wp.mat33(
            young
            / (1.0 - poisson * poisson)
            * np.array(
                [
                    [1.0, poisson, 0.0],
                    [poisson, 1.0, 0.0],
                    [0.0, 0.0, (2.0 * (1.0 + poisson)) * (1.0 - poisson * poisson)],
                ]
            )
        )

        # Function spaces -- S_k for displacement, Q_{k-1}d for stress
        self._u_space = fem.make_polynomial_space(
            self._geo, degree=degree, dtype=wp.vec2, element_basis=fem.ElementBasis.SERENDIPITY
        )

        # Store stress degrees of freedom as symmetric tensors (3 dof) rather than full 2x2 matrices
        tau_basis = fem.ElementBasis.NONCONFORMING_POLYNOMIAL if nonconforming_stresses else fem.ElementBasis.LAGRANGE
        self._tau_space = fem.make_polynomial_space(
            self._geo,
            degree=degree - 1,
            discontinuous=True,
            element_basis=tau_basis,
            dof_mapper=fem.SymmetricTensorMapper(wp.mat22),
        )

        self._u_field = self._u_space.make_field()

        self.renderer = Plot()

    def step(self):
        boundary = fem.BoundarySides(self._geo)
        domain = fem.Cells(geometry=self._geo)

        # Displacement boundary conditions
        u_bd_test = fem.make_test(space=self._u_space, domain=boundary)
        u_bd_trial = fem.make_trial(space=self._u_space, domain=boundary)
        u_bd_rhs = fem.integrate(
            horizontal_displacement_form,
            fields={"v": u_bd_test},
            values={"displacement": self._displacement},
            nodal=True,
            output_dtype=wp.vec2d,
        )
        u_bd_matrix = fem.integrate(
            horizontal_boundary_projector_form, fields={"u": u_bd_trial, "v": u_bd_test}, nodal=True
        )

        # Stress/velocity coupling
        u_trial = fem.make_trial(space=self._u_space, domain=domain)
        tau_test = fem.make_test(space=self._tau_space, domain=domain)
        tau_trial = fem.make_trial(space=self._tau_space, domain=domain)

        sym_grad_matrix = fem.integrate(symmetric_grad_form, fields={"u": u_trial, "tau": tau_test})
        stress_matrix = fem.integrate(
            stress_form, fields={"u": u_trial, "tau": tau_test}, values={"E": self._elasticity_mat}
        )

        # Compute inverse of the (block-diagonal) tau mass matrix
        tau_inv_mass_matrix = fem.integrate(tensor_mass_form, fields={"sig": tau_trial, "tau": tau_test}, nodal=True)
        invert_diagonal_bsr_mass_matrix(tau_inv_mass_matrix)

        # Assemble system matrix
        u_matrix = bsr_mm(bsr_transposed(sym_grad_matrix), bsr_mm(tau_inv_mass_matrix, stress_matrix))

        # Enforce boundary conditions
        u_rhs = wp.zeros_like(u_bd_rhs)
        fem.project_linear_system(u_matrix, u_rhs, u_bd_matrix, u_bd_rhs)

        x = wp.zeros_like(u_rhs)
        bsr_cg(u_matrix, b=u_rhs, x=x, tol=1.0e-16, quiet=self._quiet)

        # Extract result
        self._u_field.dof_values = x

    def render(self):
        self.renderer.add_surface_vector("solution", self._u_field)


if __name__ == "__main__":
    import argparse

    wp.set_module_options({"enable_backward": False})

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--resolution", type=int, default=25, help="Grid resolution.")
    parser.add_argument("--degree", type=int, default=2, help="Polynomial degree of shape functions.")
    parser.add_argument("--displacement", type=float, default=0.1)
    parser.add_argument("--young_modulus", type=float, default=1.0)
    parser.add_argument("--poisson_ratio", type=float, default=0.5)
    parser.add_argument("--mesh", choices=("grid", "tri", "quad"), default="grid", help="Mesh type")
    parser.add_argument(
        "--nonconforming_stresses", action="store_true", help="For grid, use non-conforming stresses (Q_d/P_d)"
    )
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
            displacement=args.displacement,
            young_modulus=args.young_modulus,
            poisson_ratio=args.poisson_ratio,
            nonconforming_stresses=args.nonconforming_stresses,
        )
        example.step()
        example.render()

        if not args.headless:
            example.renderer.plot()
