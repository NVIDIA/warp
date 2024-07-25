# Copyright (c) 2024 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Nonconforming Contact
#
# This example demonstrates using nonconforming fields (warp.fem.NonconformingField)
# to solve a no-slip contact problem between two elastic bodies discretized separately.
#
# Div[ E: D(u) ] = g  over each body
# u_top = u_bottom    along the contact surface (top body bottom boundary)
# u_bottom = 0        along the bottom boundary of the bottom body
#
# with E the rank-4 elasticity tensor
#
# Below we use a simple staggered scheme for solving bodies iteratively,
# but more robust methods could be considered (e.g. Augmented Lagrangian)
###########################################################################

import numpy as np

import warp as wp
import warp.examples.fem.utils as fem_example_utils
import warp.fem as fem


@wp.func
def compute_stress(tau: wp.mat22, E: wp.mat33):
    """Strain to stress computation (using Voigt notation to drop tensor order)"""
    tau_sym = wp.vec3(tau[0, 0], tau[1, 1], tau[0, 1] + tau[1, 0])
    sig_sym = E * tau_sym
    return wp.mat22(sig_sym[0], 0.5 * sig_sym[2], 0.5 * sig_sym[2], sig_sym[1])


@fem.integrand
def stress_form(s: fem.Sample, u: fem.Field, tau: fem.Field, E: wp.mat33):
    """Stress inside body:  (E : D(u)) : tau"""
    return wp.ddot(tau(s), compute_stress(fem.D(u, s), E))


@fem.integrand
def boundary_stress_form(
    s: fem.Sample,
    domain: fem.Domain,
    u: fem.Field,
    tau: fem.Field,
):
    """Stress on boundary: u' tau n"""
    return wp.dot(tau(s) * fem.normal(domain, s), u(s))


@fem.integrand
def symmetric_grad_form(
    s: fem.Sample,
    u: fem.Field,
    tau: fem.Field,
):
    """Symmetric part of gradient of displacement: D(u) : tau"""
    return wp.ddot(tau(s), fem.D(u, s))


@fem.integrand
def gravity_form(
    s: fem.Sample,
    v: fem.Field,
    gravity: float,
):
    return -gravity * v(s)[1]


@fem.integrand
def bottom_boundary_projector_form(
    s: fem.Sample,
    domain: fem.Domain,
    u: fem.Field,
    v: fem.Field,
):
    # non zero on bottom boundary only
    nor = fem.normal(domain, s)
    return wp.dot(u(s), v(s)) * wp.max(0.0, -nor[1])


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
        degree=2,
        resolution=16,
        young_modulus=1.0,
        poisson_ratio=0.5,
        nonconforming_stresses=False,
    ):
        self._geo1 = fem.Grid2D(bounds_hi=wp.vec2(1.0, 0.5), res=wp.vec2i(resolution))
        self._geo2 = fem.Grid2D(bounds_lo=(0.33, 0.5), bounds_hi=(0.67, 0.5 + 0.33), res=wp.vec2i(resolution))

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

        # Displacement spaces and fields -- S_k
        self._u1_space = fem.make_polynomial_space(
            self._geo1, degree=degree, dtype=wp.vec2, element_basis=fem.ElementBasis.SERENDIPITY
        )
        self._u2_space = fem.make_polynomial_space(
            self._geo2, degree=degree, dtype=wp.vec2, element_basis=fem.ElementBasis.SERENDIPITY
        )
        self._u1_field = self._u1_space.make_field()
        self._u2_field = self._u2_space.make_field()

        # Stress spaces and fields -- Q_{k-1}d
        # Store stress degrees of freedom as symmetric tensors (3 dof) rather than full 2x2 matrices
        self._tau1_space = fem.make_polynomial_space(
            self._geo1,
            degree=degree - 1,
            discontinuous=True,
            element_basis=fem.ElementBasis.LAGRANGE,
            dof_mapper=fem.SymmetricTensorMapper(wp.mat22),
        )
        self._tau2_space = fem.make_polynomial_space(
            self._geo2,
            degree=degree - 1,
            discontinuous=True,
            element_basis=fem.ElementBasis.LAGRANGE,
            dof_mapper=fem.SymmetricTensorMapper(wp.mat22),
        )

        self._sig1_field = self._tau1_space.make_field()
        self._sig2_field = self._tau2_space.make_field()
        self._sig2_field_new = self._tau2_space.make_field()

        self.renderer = fem_example_utils.Plot()

    def step(self):
        # Solve for the two bodies separately
        # Body (top) is 25x more dense and 5x stiffer than top body
        # Body 1 affects body 2 through bottom displacement dirichlet BC
        # Body 2 affects body 1 through applied strain on top
        self.solve_solid(self._u1_field, self._sig1_field, self._u2_field, self._sig2_field, gravity=1.0, stiffness=1.0)
        self.solve_solid(
            self._u2_field, self._sig2_field_new, self._u1_field, self._sig1_field, gravity=25.0, stiffness=5.0
        )

        # Damped update of coupling stress (for stability)
        alpha = 0.1
        fem.utils.array_axpy(
            x=self._sig2_field_new.dof_values, y=self._sig2_field.dof_values, alpha=alpha, beta=1.0 - alpha
        )

    def solve_solid(
        self,
        u_field,
        stress_field,
        other_u_field,
        other_stress_field,
        gravity: float,
        stiffness: float,
    ):
        u_space = u_field.space
        stress_space = stress_field.space
        geo = u_field.space.geometry

        domain = fem.Cells(geometry=geo)
        boundary = fem.BoundarySides(geometry=geo)

        u_test = fem.make_test(space=u_space, domain=domain)
        u_trial = fem.make_trial(space=u_space, domain=domain)
        tau_test = fem.make_test(space=stress_space, domain=domain)
        tau_trial = fem.make_trial(space=stress_space, domain=domain)

        u_bd_test = fem.make_test(space=u_space, domain=boundary)
        u_bd_trial = fem.make_trial(space=u_space, domain=boundary)

        # Assemble stiffness matrix
        # (Note: this is constant per body, this could be precomputed)
        sym_grad_matrix = fem.integrate(symmetric_grad_form, fields={"u": u_trial, "tau": tau_test})

        tau_inv_mass_matrix = fem.integrate(tensor_mass_form, fields={"sig": tau_trial, "tau": tau_test}, nodal=True)
        fem_example_utils.invert_diagonal_bsr_matrix(tau_inv_mass_matrix)

        stress_matrix = tau_inv_mass_matrix @ fem.integrate(
            stress_form, fields={"u": u_trial, "tau": tau_test}, values={"E": self._elasticity_mat * stiffness}
        )
        stiffness_matrix = sym_grad_matrix.transpose() @ stress_matrix

        # Right-hand-side
        u_rhs = fem.integrate(gravity_form, fields={"v": u_test}, values={"gravity": gravity}, output_dtype=wp.vec2d)

        # Add boundary stress from other solid field
        other_stress_field = fem.field.field.NonconformingField(boundary, other_stress_field)
        fem.utils.array_axpy(
            y=u_rhs,
            x=fem.integrate(
                boundary_stress_form, fields={"u": u_bd_test, "tau": other_stress_field}, output_dtype=wp.vec2d
            ),
        )

        # Enforce boundary conditions
        u_bd_matrix = fem.integrate(
            bottom_boundary_projector_form, fields={"u": u_bd_trial, "v": u_bd_test}, nodal=True
        )

        # read displacement from other body set create bottom boundary Dirichlet BC
        other_u_field = fem.field.field.NonconformingField(boundary, other_u_field)
        u_bd_rhs = fem.integrate(
            bottom_boundary_projector_form, fields={"u": other_u_field, "v": u_bd_test}, nodal=True
        )

        fem.project_linear_system(stiffness_matrix, u_rhs, u_bd_matrix, u_bd_rhs)

        # solve
        x = wp.zeros_like(u_rhs)
        wp.utils.array_cast(in_array=u_field.dof_values, out_array=x)
        fem_example_utils.bsr_cg(stiffness_matrix, b=u_rhs, x=x, tol=1.0e-6, quiet=True)

        # Extract result
        stress = stress_matrix @ x
        wp.utils.array_cast(in_array=x, out_array=u_field.dof_values)
        wp.utils.array_cast(in_array=stress, out_array=stress_field.dof_values)

    def render(self):
        self.renderer.add_field("u1", self._u1_field)
        self.renderer.add_field("u2", self._u2_field)
        self.renderer.add_field("sig1", self._sig1_field)
        self.renderer.add_field("sig2", self._sig2_field)


if __name__ == "__main__":
    import argparse

    wp.set_module_options({"enable_backward": False})

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--resolution", type=int, default=32, help="Grid resolution.")
    parser.add_argument("--degree", type=int, default=2, help="Polynomial degree of shape functions.")
    parser.add_argument("--young_modulus", type=float, default=10.0)
    parser.add_argument("--poisson_ratio", type=float, default=0.9)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode, suppressing the opening of any graphical windows.",
    )

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(
            degree=args.degree,
            resolution=args.resolution,
            young_modulus=args.young_modulus,
            poisson_ratio=args.poisson_ratio,
        )

        for i in range(args.num_steps):
            print("Step", i)
            example.step()
            example.render()

        if not args.headless:
            example.renderer.plot(
                {
                    "rows": 2,
                    "u1": {"displacement": {}},
                    "u2": {"displacement": {}},
                },
            )
