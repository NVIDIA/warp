# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example Mixed Elasticity
#
# This example illustrates using Mixed FEM to solve a nonlinear static elasticity equilibrium problem:
#
# Div[ d/dF Psi(F(u)) ] = 0
#
# with Dirichlet boundary conditions on vertical sides and Psi an elastic potential function of the deformation gradient.
# Here we choose Psi Neo-Hookean, as per Sec 3.2 of "Stable Neo-Hookean Flesh Simulation" (Smith et al. 2018),
# Psi(F) = mu ||F||^2 + lambda (det J - 1 - mu/lambda)^2
#
# which we write as a sequence of Newton iterations:
# int {sigma : grad v}  = 0   for all displacement test functions v
# int {sigma : tau} = int{dPsi/dF : tau} + int{grad du : d2 Psi/dF2  : tau} for all stress test functions tau
###########################################################################

import numpy as np

import warp as wp
import warp.examples.fem.utils as fem_example_utils
import warp.fem as fem


@fem.integrand
def displacement_gradient_form(
    s: fem.Sample,
    u: fem.Field,
    tau: fem.Field,
):
    """grad(u) : tau"""
    return wp.ddot(tau(s), fem.grad(u, s))


@wp.func
def nh_parameters_from_lame(lame: wp.vec2):
    """Parameters such that for small strains model behaves according to Hooke's law"""
    mu_nh = lame[1]
    lambda_nh = lame[0] + lame[1]

    return mu_nh, lambda_nh


@fem.integrand
def nh_stress_form(s: fem.Sample, tau: fem.Field, u_cur: fem.Field, lame: wp.vec2):
    """d Psi/dF : tau"""

    # Deformation gradient
    F = wp.identity(n=2, dtype=float) + fem.grad(u_cur, s)

    # Area term and its derivative w.r.t F
    J = wp.determinant(F)
    dJ_dF = wp.mat22(F[1, 1], -F[1, 0], -F[0, 1], F[0, 0])

    mu_nh, lambda_nh = nh_parameters_from_lame(lame)
    nh_stress = mu_nh * F + (lambda_nh * (J - 1.0) - mu_nh) * dJ_dF

    return wp.ddot(tau(s), nh_stress)


@fem.integrand
def nh_stress_delta_form(s: fem.Sample, tau: fem.Field, u: fem.Field, u_cur: fem.Field, lame: wp.vec2):
    """grad(u) : d2 Psi/dF2 : tau"""

    tau_s = tau(s)
    sigma_s = fem.grad(u, s)

    F = wp.identity(n=2, dtype=float) + fem.grad(u_cur, s)
    dJ_dF = wp.mat22(F[1, 1], -F[1, 0], -F[0, 1], F[0, 0])

    # Gauss--Newton approximation; ignore d2J/dF2 term
    mu_nh, lambda_nh = nh_parameters_from_lame(lame)
    return mu_nh * wp.ddot(tau_s, sigma_s) + lambda_nh * wp.ddot(dJ_dF, tau_s) * wp.ddot(dJ_dF, sigma_s)


@fem.integrand
def vertical_boundary_projector_form(
    s: fem.Sample,
    domain: fem.Domain,
    u: fem.Field,
    v: fem.Field,
):
    # non zero on vertical boundary of domain only
    nor = fem.normal(domain, s)
    return wp.dot(u(s), v(s)) * wp.abs(nor[0])


@fem.integrand
def vertical_displacement_form(
    s: fem.Sample,
    domain: fem.Domain,
    v: fem.Field,
    displacement: float,
):
    # opposed to normal on vertical boundary of domain only
    nor = fem.normal(domain, s)
    return -wp.abs(nor[0]) * displacement * wp.dot(nor, v(s))


@fem.integrand
def tensor_mass_form(
    s: fem.Sample,
    sig: fem.Field,
    tau: fem.Field,
):
    return wp.ddot(tau(s), sig(s))


@fem.integrand
def area_form(s: fem.Sample, u_cur: fem.Field):
    F = wp.identity(n=2, dtype=float) + fem.grad(u_cur, s)
    return wp.determinant(F)


class Example:
    def __init__(
        self,
        quiet=False,
        degree=2,
        resolution=25,
        mesh="grid",
        displacement=0.1,
        poisson_ratio=0.5,
        nonconforming_stresses=False,
    ):
        self._quiet = quiet

        self._displacement = displacement

        # Grid or mesh geometry
        if mesh == "tri":
            positions, tri_vidx = fem_example_utils.gen_trimesh(res=wp.vec2i(resolution))
            self._geo = fem.Trimesh2D(tri_vertex_indices=tri_vidx, positions=positions)
        elif mesh == "quad":
            positions, quad_vidx = fem_example_utils.gen_quadmesh(res=wp.vec2i(resolution))
            self._geo = fem.Quadmesh2D(quad_vertex_indices=quad_vidx, positions=positions)
        else:
            self._geo = fem.Grid2D(res=wp.vec2i(resolution))

        # Lame coefficients from Young modulus and Poisson ratio
        self._lame = wp.vec2(1.0 / (1.0 + poisson_ratio) * np.array([poisson_ratio / (1.0 - poisson_ratio), 0.5]))

        # Function spaces -- S_k for displacement, Q_k or P_{k-1}d for stress
        self._u_space = fem.make_polynomial_space(
            self._geo, degree=degree, dtype=wp.vec2, element_basis=fem.ElementBasis.SERENDIPITY
        )

        if isinstance(self._geo.reference_cell(), fem.geometry.element.Triangle):
            # triangle elements
            tau_basis = fem.ElementBasis.NONCONFORMING_POLYNOMIAL
            tau_degree = degree - 1
        else:
            # square elements
            tau_basis = fem.ElementBasis.LAGRANGE
            tau_degree = degree

        self._tau_space = fem.make_polynomial_space(
            self._geo,
            degree=tau_degree,
            discontinuous=True,
            element_basis=tau_basis,
            family=fem.Polynomial.GAUSS_LEGENDRE,
            dtype=wp.mat22,
        )

        self._u_field = self._u_space.make_field()

        self.renderer = fem_example_utils.Plot()

    def step(self):
        boundary = fem.BoundarySides(self._geo)
        domain = fem.Cells(geometry=self._geo)

        # Displacement boundary conditions
        u_bd_test = fem.make_test(space=self._u_space, domain=boundary)
        u_bd_trial = fem.make_trial(space=self._u_space, domain=boundary)
        u_bd_rhs = fem.integrate(
            vertical_displacement_form,
            fields={"v": u_bd_test},
            values={"displacement": self._displacement},
            assembly="nodal",
            output_dtype=wp.vec2d,
        )
        u_bd_matrix = fem.integrate(
            vertical_boundary_projector_form, fields={"u": u_bd_trial, "v": u_bd_test}, assembly="nodal"
        )

        # Stress/velocity coupling
        u_trial = fem.make_trial(space=self._u_space, domain=domain)
        tau_test = fem.make_test(space=self._tau_space, domain=domain)
        tau_trial = fem.make_trial(space=self._tau_space, domain=domain)

        gradient_matrix = fem.integrate(displacement_gradient_form, fields={"u": u_trial, "tau": tau_test}).transpose()

        # Compute inverse of the (block-diagonal) tau mass matrix
        tau_inv_mass_matrix = fem.integrate(
            tensor_mass_form, fields={"sig": tau_trial, "tau": tau_test}, assembly="nodal"
        )
        fem_example_utils.invert_diagonal_bsr_matrix(tau_inv_mass_matrix)

        # Newton iterations (without line-search for simplicity)
        for newton_iteration in range(5):
            stress_matrix = fem.integrate(
                nh_stress_delta_form,
                fields={"u_cur": self._u_field, "u": u_trial, "tau": tau_test},
                values={"lame": self._lame},
            )

            stress_rhs = fem.integrate(
                nh_stress_form,
                fields={"u_cur": self._u_field, "tau": tau_test},
                values={"lame": self._lame},
                output_dtype=wp.vec(length=stress_matrix.block_shape[0], dtype=wp.float64),
            )

            # Assemble system matrix
            u_matrix = gradient_matrix @ tau_inv_mass_matrix @ stress_matrix

            # Enforce boundary conditions (apply displacement only at first iteration)
            u_rhs = -gradient_matrix @ (tau_inv_mass_matrix @ stress_rhs)
            fem.project_linear_system(u_matrix, u_rhs, u_bd_matrix, u_bd_rhs if newton_iteration == 0 else None)

            x = wp.zeros_like(u_rhs)
            fem_example_utils.bsr_cg(u_matrix, b=u_rhs, x=x, quiet=self._quiet)

            # Extract result -- cast to float32 and accumulate to displacement field
            delta_u = wp.empty_like(self._u_field.dof_values)
            wp.utils.array_cast(in_array=x, out_array=delta_u)
            fem.utils.array_axpy(x=delta_u, y=self._u_field.dof_values)

        # Evaluate area conservation, should converge to 1.0 as Poisson ratio approaches 1.0
        final_area = fem.integrate(
            area_form, quadrature=fem.RegularQuadrature(domain, order=4), fields={"u_cur": self._u_field}
        )
        print(f"Area gain: {final_area}  (using Poisson ratio={self._lame[0] / (self._lame[0] + 2.0 * self._lame[1])})")

    def render(self):
        self.renderer.add_field("solution", self._u_field)


if __name__ == "__main__":
    import argparse

    wp.set_module_options({"enable_backward": False})

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--resolution", type=int, default=25, help="Grid resolution.")
    parser.add_argument("--degree", type=int, default=2, help="Polynomial degree of shape functions.")
    parser.add_argument("--displacement", type=float, default=-0.5)
    parser.add_argument("--poisson_ratio", type=float, default=0.99)
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
            poisson_ratio=args.poisson_ratio,
            nonconforming_stresses=args.nonconforming_stresses,
        )
        example.step()
        example.render()

        if not args.headless:
            example.renderer.plot(options={"solution": {"displacement": {}}})
