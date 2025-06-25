# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# Example: Shape Optimization of a 2D Elastic Cantilever Beam
#
# This example demonstrates shape optimization of a 2D elastic cantilever beam using
# finite element analysis and gradient-based optimization.
#
# Problem Setup:
# - The computational domain is a 2D beam (rectangular mesh), fixed on the left edge
#   (Dirichlet boundary condition: zero displacement).
# - A constant external load is applied to the right edge of the beam, causing it to deform.
# - The beam is discretized using finite elements, and the displacement field is solved
#   for the current geometry at each optimization step using a linear elasticity formulation.
#
# Shape Optimization Strategy:
# - The goal is to optimize the shape of the beam to minimize the total squared norm of the
#   stress field (e.g., compliance or strain energy) over the domain.
# - The positions of the left and right boundary vertices are fixed throughout the optimization
#   to maintain the beam's support and loading conditions.
# - A volume constraint is enforced to preserve the total material volume, preventing trivial
#   solutions that simply shrink the structure.
# - An ad-hoc "quality" term is included in the loss function to penalize degenerate or inverted elements,
#   helping to maintain mesh quality (as remeshing would be out of scope for this example).
# - The optimization is performed by computing the gradient of the objective with respect to
#   the nodal positions (shape derivatives) using the adjoint method or automatic differentiation.
# - At each iteration, the nodal positions are updated in the direction of decreasing objective,
#   subject to the volume constraint and boundary conditions.
###########################################################################

import numpy as np

import warp as wp
import warp.examples.fem.utils as fem_example_utils
import warp.fem as fem
from warp.optim import Adam


@fem.integrand(kernel_options={"max_unroll": 1})
def boundary_projector_form(
    s: fem.Sample,
    domain: fem.Domain,
    u: fem.Field,
    v: fem.Field,
):
    return wp.dot(u(s), v(s))


@fem.integrand
def classify_boundary_sides(
    s: fem.Sample,
    domain: fem.Domain,
    left: wp.array(dtype=int),
    right: wp.array(dtype=int),
):
    nor = fem.normal(domain, s)

    if nor[0] < -0.5:
        left[s.qp_index] = 1
    elif nor[0] > 0.5:
        right[s.qp_index] = 1


@wp.func
def hooke_stress(strain: wp.mat22, lame: wp.vec2):
    """Hookean elasticity"""
    return 2.0 * lame[1] * strain + lame[0] * wp.trace(strain) * wp.identity(n=2, dtype=float)


@fem.integrand
def stress_field(s: fem.Sample, u: fem.Field, lame: wp.vec2):
    return hooke_stress(fem.D(u, s), lame)


@fem.integrand
def hooke_elasticity_form(s: fem.Sample, u: fem.Field, v: fem.Field, lame: wp.vec2):
    return wp.ddot(fem.D(v, s), stress_field(s, u, lame))


@fem.integrand
def applied_load_form(s: fem.Sample, domain: fem.Domain, v: fem.Field, load: wp.vec2):
    return wp.dot(v(s), load)


@fem.integrand
def loss_form(
    s: fem.Sample, domain: fem.Domain, u: fem.Field, lame: wp.vec2, quality_threshold: float, quality_weight: float
):
    stress = stress_field(s, u, lame)
    stress_norm_sq = wp.ddot(stress, stress)

    # As we're not remeshing, add a "quality" term
    # to avoid degenerate and inverted elements

    F = fem.deformation_gradient(domain, s)
    U, S, V = wp.svd2(F)

    quality = wp.min(S) / wp.max(S) / quality_threshold
    quality_pen = -wp.log(wp.max(quality, 0.0001)) * wp.min(0.0, quality - 1.0) * wp.min(0.0, quality - 1.0)

    return stress_norm_sq + quality_pen * quality_weight


@fem.integrand
def volume_form():
    return 1.0


@wp.kernel
def add_volume_loss(
    loss: wp.array(dtype=wp.float32), vol: wp.array(dtype=wp.float32), target_vol: wp.float32, weight: wp.float32
):
    loss[0] += weight * (vol[0] - target_vol) * (vol[0] - target_vol)


class Example:
    def __init__(
        self,
        quiet=False,
        degree=2,
        resolution=25,
        mesh="tri",
        poisson_ratio=0.5,
        load=(0.0, -1),
        lr=1.0e-3,
    ):
        self._quiet = quiet

        # Lame coefficients from Young modulus and Poisson ratio
        self._lame = wp.vec2(1.0 / (1.0 + poisson_ratio) * np.array([poisson_ratio / (1.0 - poisson_ratio), 0.5]))
        self._load = load

        # procedural rectangular domain definition
        bounds_lo = wp.vec2(0.0, 0.8)
        bounds_hi = wp.vec2(1.0, 1.0)
        self._initial_volume = (bounds_hi - bounds_lo)[0] * (bounds_hi - bounds_lo)[1]

        if mesh == "tri":
            # triangle mesh, optimize vertices directly
            positions, tri_vidx = fem_example_utils.gen_trimesh(
                res=wp.vec2i(resolution, resolution // 5), bounds_lo=bounds_lo, bounds_hi=bounds_hi
            )
            self._geo = fem.Trimesh2D(tri_vertex_indices=tri_vidx, positions=positions)
            self._start_geo = fem.Trimesh2D(tri_vertex_indices=tri_vidx, positions=wp.clone(positions))
            self._vertex_positions = positions
        elif mesh == "quad":
            # quad mesh, optimize vertices directly
            positions, quad_vidx = fem_example_utils.gen_quadmesh(
                res=wp.vec2i(resolution, resolution // 5), bounds_lo=bounds_lo, bounds_hi=bounds_hi
            )
            self._geo = fem.Quadmesh2D(quad_vertex_indices=quad_vidx, positions=positions)
            self._start_geo = fem.Quadmesh2D(quad_vertex_indices=quad_vidx, positions=wp.clone(positions))
            self._vertex_positions = positions
        else:
            # grid, optimize nodes of deformation field
            self._start_geo = fem.Grid2D(
                wp.vec2i(resolution, resolution // 5), bounds_lo=bounds_lo, bounds_hi=bounds_hi
            )
            vertex_displacement_space = fem.make_polynomial_space(self._start_geo, degree=1, dtype=wp.vec2)
            vertex_position_field = fem.make_discrete_field(space=vertex_displacement_space)
            vertex_position_field.dof_values = vertex_displacement_space.node_positions()
            self._geo = vertex_position_field.make_deformed_geometry(relative=False)
            self._vertex_positions = vertex_position_field.dof_values

        # make sure positions are differentiable
        self._vertex_positions.requires_grad = True

        # Store initial node positions (for rendering)
        self._u_space = fem.make_polynomial_space(self._geo, degree=degree, dtype=wp.vec2)
        self._start_node_positions = self._u_space.node_positions()

        # displacement field, make sure gradient is stored
        self._u_field = fem.make_discrete_field(space=self._u_space)
        self._u_field.dof_values.requires_grad = True

        # Trial and test functions
        self._u_test = fem.make_test(space=self._u_space)
        self._u_trial = fem.make_trial(space=self._u_space)

        # Identify left and right sides for boundary conditions
        boundary = fem.BoundarySides(self._geo)

        left_mask = wp.zeros(shape=boundary.element_count(), dtype=int)
        right_mask = wp.zeros(shape=boundary.element_count(), dtype=int)

        fem.interpolate(
            classify_boundary_sides,
            quadrature=fem.RegularQuadrature(boundary, order=0),
            values={"left": left_mask, "right": right_mask},
        )

        self._left = fem.Subdomain(boundary, element_mask=left_mask)
        self._right = fem.Subdomain(boundary, element_mask=right_mask)

        # Build projectors for the left-side homogeneous Dirichlet condition
        u_left_bd_test = fem.make_test(space=self._u_space, domain=self._left)
        u_left_bd_trial = fem.make_trial(space=self._u_space, domain=self._left)
        u_left_bd_matrix = fem.integrate(
            boundary_projector_form,
            fields={"u": u_left_bd_trial, "v": u_left_bd_test},
            assembly="nodal",
            output_dtype=float,
        )
        fem.normalize_dirichlet_projector(u_left_bd_matrix)
        self._bd_projector = u_left_bd_matrix

        # Fixed vertices (that the shape optimization should not move)
        # Build projectors for the left and right subdomains and add them together
        vertex_space = fem.make_polynomial_space(self._geo, degree=1, dtype=wp.vec2)
        u_left_vertex_bd_test = fem.make_test(space=vertex_space, domain=self._left)
        u_left_vertex_bd_trial = fem.make_trial(space=vertex_space, domain=self._left)
        u_right_vertex_bd_test = fem.make_test(space=vertex_space, domain=self._right)
        u_right_vertex_bd_trial = fem.make_trial(space=vertex_space, domain=self._right)
        u_fixed_vertex_matrix = fem.integrate(
            boundary_projector_form,
            fields={"u": u_left_vertex_bd_trial, "v": u_left_vertex_bd_test},
            assembly="nodal",
            output_dtype=float,
        ) + fem.integrate(
            boundary_projector_form,
            fields={"u": u_right_vertex_bd_trial, "v": u_right_vertex_bd_test},
            assembly="nodal",
            output_dtype=float,
        )
        fem.normalize_dirichlet_projector(u_fixed_vertex_matrix)
        self._fixed_vertex_projector = u_fixed_vertex_matrix

        self._u_right_test = fem.make_test(space=self._u_space, domain=self._right)

        # initialize renderer
        self.renderer = fem_example_utils.Plot()

        # Initialize Adam optimizer
        # Current implementation assumes scalar arrays, so cast our vec2 arrays to scalars
        self._vertex_positions_scalar = wp.array(self._vertex_positions, dtype=wp.float32).flatten()
        self._vertex_positions_scalar.grad = wp.array(self._vertex_positions.grad, dtype=wp.float32).flatten()
        self.optimizer = Adam([self._vertex_positions_scalar], lr=lr)

    def step(self):
        # Forward step, record adjoint tape for forces
        u = self._u_field.dof_values
        u.zero_()

        u_rhs = wp.empty(self._u_space.node_count(), dtype=wp.vec2f, requires_grad=True)

        tape = wp.Tape()

        with tape:
            fem.integrate(
                applied_load_form,
                fields={"v": self._u_right_test},
                values={"load": self._load},
                output=u_rhs,
            )
            # the elastic force will be zero at the first iteration,
            # but including it on the tape is necessary to compute the gradient of the force equilibrium
            # using the implicit function theorem
            # Note that this will be evaluated in the backward pass using the updated values for "_u_field"
            fem.integrate(
                hooke_elasticity_form,
                fields={"u": self._u_field, "v": self._u_test},
                values={"lame": -self._lame},
                output=u_rhs,
                add=True,
            )

        u_matrix = fem.integrate(
            hooke_elasticity_form,
            fields={"u": self._u_trial, "v": self._u_test},
            values={"lame": self._lame},
            output_dtype=float,
        )
        fem.project_linear_system(u_matrix, u_rhs, self._bd_projector, normalize_projector=False)

        fem_example_utils.bsr_cg(u_matrix, b=u_rhs, x=u, quiet=self._quiet, tol=1e-6, max_iters=1000)

        # Record adjoint of linear solve
        # (For nonlinear elasticity, this should use the final hessian, as per implicit function theorem)
        def solve_linear_system():
            fem_example_utils.bsr_cg(u_matrix, b=u.grad, x=u_rhs.grad, quiet=self._quiet, tol=1e-6, max_iters=1000)
            u_rhs.grad -= self._bd_projector @ u_rhs.grad
            self._u_field.dof_values.grad.zero_()

        tape.record_func(solve_linear_system, arrays=(u_rhs, u))

        # Evaluate residual
        # Integral of squared difference between simulated position and target positions
        loss = wp.empty(shape=1, dtype=wp.float32, requires_grad=True)
        vol = wp.empty(shape=1, dtype=wp.float32, requires_grad=True)

        with tape:
            fem.integrate(
                loss_form,
                fields={"u": self._u_field},
                values={"lame": self._lame, "quality_threshold": 0.2, "quality_weight": 20.0},
                domain=self._u_test.domain,
                output=loss,
            )
            print(f"Loss: {loss}")

            # Add penalization term enforcing constant volume
            fem.integrate(
                volume_form,
                domain=self._u_test.domain,
                output=vol,
            )

            vol_loss_weight = 100.0
            wp.launch(
                add_volume_loss,
                dim=1,
                inputs=(loss, vol, self._initial_volume, vol_loss_weight),
            )

        # perform backward step
        tape.backward(loss=loss)

        # enforce fixed vertices
        self._vertex_positions.grad -= self._fixed_vertex_projector @ self._vertex_positions.grad

        # update positions and reset tape
        self.optimizer.step([self._vertex_positions_scalar.grad])
        tape.zero()

    def render(self):
        # Render using fields defined on start geometry
        # (renderer assumes geometry remains fixed for timesampled fields)
        u_space = fem.make_polynomial_space(self._start_geo, degree=self._u_space.degree, dtype=wp.vec2)
        u_field = fem.make_discrete_field(space=u_space)
        rest_field = fem.make_discrete_field(space=u_space)

        geo_displacement = self._u_space.node_positions() - self._start_node_positions
        u_field.dof_values = self._u_field.dof_values + geo_displacement
        rest_field.dof_values = geo_displacement

        self.renderer.add_field("displacement", u_field)
        self.renderer.add_field("rest", rest_field)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--resolution", type=int, default=10, help="Grid resolution.")
    parser.add_argument("--degree", type=int, default=1, help="Polynomial degree of shape functions.")
    parser.add_argument("--mesh", choices=("tri", "quad", "grid"), default="tri", help="Mesh type")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode, suppressing the opening of any graphical windows.",
    )
    parser.add_argument("--lr", type=float, default=1.0e-3, help="Learning rate.")
    parser.add_argument("--num_iters", type=int, default=250, help="Number of iterations.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(
            quiet=True,
            degree=args.degree,
            resolution=args.resolution,
            mesh=args.mesh,
            poisson_ratio=0.95,
            load=wp.vec2(0.0, -0.1),
            lr=args.lr,
        )

        for _k in range(args.num_iters):
            example.step()
            example.render()

        if not args.headless:
            example.renderer.plot(options={"displacement": {"displacement": {}}, "rest": {"displacement": {}}})
