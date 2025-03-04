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
# Example Burgers
#
# This example simulates an inviscid non-conservative Burgers PDE using
# Discontinuous Galerkin with minmod slope limiter
#
# d u /dt + (u . grad) u = 0
#
###########################################################################

import warp as wp
import warp.examples.fem.utils as fem_example_utils
import warp.fem as fem


@fem.integrand
def vel_mass_form(
    s: fem.Sample,
    u: fem.Field,
    v: fem.Field,
):
    return wp.dot(v(s), u(s))


@fem.integrand
def upwind_transport_form(s: fem.Sample, domain: fem.Domain, u: fem.Field, v: fem.Field, w: fem.Field):
    # Upwinding transport with discontinuous convection velocity,
    # using jump(w v) = jump(w) avg(v) + avg(w) jump(w)

    nor = fem.normal(domain, s)
    w_avg_n = wp.dot(fem.average(w, s), nor)
    w_jump_n = wp.dot(fem.jump(w, s), nor)

    x = domain(s)
    v_avg = fem.average(v, s)
    if x[0] <= 0.0 or x[0] >= 1.0:  # out
        # if x[0] >= 1.0:  # out
        v_jump = v(s)
    else:
        v_jump = fem.jump(v, s)

    u_avg = fem.average(u, s)
    u_jump = fem.jump(u, s)

    return wp.dot(u_avg, v_jump * w_avg_n + v_avg * w_jump_n) + 0.5 * wp.dot(v_jump, u_jump) * (
        wp.abs(w_avg_n) + 0.5 * wp.abs(w_jump_n)
    )


@fem.integrand
def cell_transport_form(s: fem.Sample, domain: fem.Domain, u: fem.Field, v: fem.Field, w: fem.Field):
    # ((w . grad) u) . v = v^T (grad u) w =  grad(u) : (v w^T)
    # with integration by parts
    # u . Div (w v^T) = u^T grad(v) w  + u^T v div (w)

    return -wp.dot(fem.div(w, s) * v(s) + fem.grad(v, s) * w(s), u(s))


@fem.integrand
def initial_condition(s: fem.Sample, domain: fem.Domain):
    x = domain(s)[0] * 2.0
    wave = wp.sin(x * wp.pi)
    return wp.vec2(wp.where(x <= 1.0, wave, 0.0), 0.0)


@fem.integrand
def velocity_norm(s: fem.Sample, u: fem.Field):
    return wp.length(u(s))


@wp.func
def minmod(a: float, b: float):
    sa = wp.sign(a)
    sb = wp.sign(b)
    return wp.where(sa == sb, sa * wp.min(wp.abs(a), wp.abs(b)), 0.0)


@fem.integrand
def slope_limiter(domain: fem.Domain, s: fem.Sample, u: fem.Field, dx: wp.vec2):
    # Minmod slope limiter against P0 discretization (evaluation at cell centers)
    # Assumes regular grid topology

    center_coords = fem.Coords(0.5, 0.5, 0.0)
    cell_center = fem.make_free_sample(s.element_index, center_coords)
    center_pos = domain(cell_center)

    u_center = u(cell_center)

    delta_coords = s.element_coords - center_coords

    neighbour_xp = fem.lookup(domain, center_pos + wp.vec2(dx[0], 0.0))
    neighbour_yp = fem.lookup(domain, center_pos + wp.vec2(0.0, dx[1]))
    neighbour_xm = fem.lookup(domain, center_pos - wp.vec2(dx[0], 0.0))
    neighbour_ym = fem.lookup(domain, center_pos - wp.vec2(0.0, dx[1]))

    u_nxp = u(neighbour_xp)
    u_nyp = u(neighbour_yp)
    u_nxm = u(neighbour_xm)
    u_nym = u(neighbour_ym)

    gx = minmod(u_nxp[0] - u_center[0], u_center[0] - u_nxm[0]) * delta_coords[0]
    gy = minmod(u_nyp[1] - u_center[1], u_center[1] - u_nym[1]) * delta_coords[1]

    delta_u = u(s) - u_center
    return u_center + wp.vec2(minmod(gx, delta_u[0]), minmod(gy, delta_u[1]))


class Example:
    def __init__(self, quiet=False, resolution=50, degree=1):
        self._quiet = quiet

        res = resolution
        self.sim_dt = 1.0 / res
        self.current_frame = 0

        geo = fem.Grid2D(res=wp.vec2i(resolution))

        domain = fem.Cells(geometry=geo)
        sides = fem.Sides(geo)

        basis_space = fem.make_polynomial_basis_space(geo, degree=degree, discontinuous=True)
        vector_space = fem.make_collocated_function_space(basis_space, dtype=wp.vec2)
        scalar_space = fem.make_collocated_function_space(basis_space, dtype=float)

        # Test function for ou vector space
        self._test = fem.make_test(space=vector_space, domain=domain)
        # Test function for integration on sides
        self._side_test = fem.make_test(space=vector_space, domain=sides)

        # Inertia matrix
        # For simplicity, use nodal integration so that inertia matrix is diagonal
        trial = fem.make_trial(space=vector_space, domain=domain)
        matrix_inertia = fem.integrate(
            vel_mass_form, fields={"u": trial, "v": self._test}, output_dtype=wp.float32, nodal=True
        )
        self._inv_mass_matrix = wp.sparse.bsr_copy(matrix_inertia)
        fem_example_utils.invert_diagonal_bsr_matrix(self._inv_mass_matrix)

        # Initial condition
        self.velocity_field = vector_space.make_field()
        fem.interpolate(initial_condition, dest=self.velocity_field)

        # Velocity norm field -- for visualization purposes
        self.velocity_norm_field = scalar_space.make_field()
        fem.interpolate(velocity_norm, dest=self.velocity_norm_field, fields={"u": self.velocity_field})

        self.renderer = fem_example_utils.Plot()
        self.renderer.add_field("u_norm", self.velocity_norm_field)

    def _velocity_delta(self, trial_velocity):
        # Integration on sides
        rhs = fem.integrate(
            upwind_transport_form,
            fields={"u": trial_velocity.trace(), "v": self._side_test, "w": trial_velocity.trace()},
            output_dtype=wp.vec2,
        )

        if self.velocity_field.space.degree > 0:
            # Integration on cells (if not piecewise-constant)
            fem.utils.array_axpy(
                x=fem.integrate(
                    cell_transport_form,
                    fields={"u": trial_velocity, "v": self._test, "w": trial_velocity},
                    output_dtype=wp.vec2,
                    quadrature=fem.RegularQuadrature(
                        order=3, domain=self._test.domain, family=fem.Polynomial.LOBATTO_GAUSS_LEGENDRE
                    ),
                ),
                y=rhs,
                alpha=1.0,
                beta=1.0,
            )
        return self._inv_mass_matrix @ rhs

    def step(self):
        self.current_frame += 1

        # Third-order Strong Stability Preserving Runge-Kutta (SSPRK3)

        k1 = self._velocity_delta(self.velocity_field)

        # tmp = v0 - dt * k1
        tmp = self.velocity_field.space.make_field()
        fem.utils.array_axpy(y=tmp.dof_values, x=self.velocity_field.dof_values, alpha=1.0, beta=0.0)
        fem.utils.array_axpy(y=tmp.dof_values, x=k1, alpha=-self.sim_dt, beta=1.0)
        k2 = self._velocity_delta(tmp)

        # tmp = v0 - dt * (0.25 * k1 + 0.25 * k2)
        fem.utils.array_axpy(y=tmp.dof_values, x=k1, alpha=0.75 * self.sim_dt, beta=1.0)
        fem.utils.array_axpy(y=tmp.dof_values, x=k2, alpha=-0.25 * self.sim_dt, beta=1.0)
        k3 = self._velocity_delta(tmp)

        # v = v0 - dt * (1/6 * k1 + 1/6 * k2 + 2/3 * k3)
        fem.utils.array_axpy(y=self.velocity_field.dof_values, x=k1, alpha=-1.0 / 6.0 * self.sim_dt, beta=1.0)
        fem.utils.array_axpy(y=self.velocity_field.dof_values, x=k2, alpha=-1.0 / 6.0 * self.sim_dt, beta=1.0)
        fem.utils.array_axpy(y=self.velocity_field.dof_values, x=k3, alpha=-2.0 / 3.0 * self.sim_dt, beta=1.0)

        # Apply slope limiter
        if self.velocity_field.space.degree > 0:
            res = self.velocity_field.space.geometry.res
            dx = wp.vec2(1.0 / res[0], 1.0 / res[1])
            fem.interpolate(slope_limiter, dest=tmp, fields={"u": self.velocity_field}, values={"dx": dx})
            wp.copy(src=tmp.dof_values, dest=self.velocity_field.dof_values)

        # Update velocity norm (for visualization)
        fem.interpolate(velocity_norm, dest=self.velocity_norm_field, fields={"u": self.velocity_field})

    def render(self):
        self.renderer.begin_frame(time=self.current_frame * self.sim_dt)
        self.renderer.add_field("u_norm", self.velocity_norm_field)
        self.renderer.end_frame()


if __name__ == "__main__":
    import argparse

    wp.set_module_options({"enable_backward": False})

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--resolution", type=int, default=50, help="Grid resolution.")
    parser.add_argument("--num_frames", type=int, default=250, help="Total number of frames.")
    parser.add_argument("--degree", choices=(0, 1), type=int, default=1, help="Discretization order.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode, suppressing the opening of any graphical windows.",
    )
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(
            quiet=args.quiet,
            resolution=args.resolution,
            degree=args.degree,
        )

        for k in range(args.num_frames):
            print(f"Frame {k}:")
            example.step()
            example.render()

        if not args.headless:
            example.renderer.plot()
