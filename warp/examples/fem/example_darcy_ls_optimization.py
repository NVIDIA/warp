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
# Example: Shape Optimization of 2D Darcy Flow using Level Set Method
#
# This example demonstrates the use of a level set-based shape optimization technique
# to maximize the total Darcy flow (inflow) through a 2D square domain. The domain
# contains a material region whose shape is implicitly represented by a level set function.
#
# Physical Setup:
# - The computational domain is a unit square, discretized using either a structured grid
#   or a triangular mesh.
# - The material region within the domain is defined by the zero level set of a scalar field.
# - The permeability of the domain is a smooth function of the level set, with high permeability
#   inside the material region and low permeability outside.
# - Boundary conditions are set such that one side of the domain has a prescribed inflow pressure
#   and the opposite side has a prescribed outflow (Dirichlet) pressure, driving Darcy flow
#   through the material region.
#
# Optimization Goal:
# - The objective is to optimize the shape of the material region (by evolving the level set)
#   to maximize the total inflow (Darcy flux) across the inflow boundary, subject to a constraint
#   on the total volume of the material region.
#
# Numerical Approach:
# - The pressure field is solved using the finite element method (FEM) for the current material
#   configuration.
# - The level set function is updated using the adjoint method: the gradient of the objective
#   with respect to the level set is computed via automatic differentiation, and the level set
#   is advected in the direction of increasing inflow.
# - The optimization is performed iteratively, with each iteration consisting of a forward
#   pressure solve, loss evaluation, backward (adjoint) computation, and level set update.
# - The code supports both continuous and discontinuous Galerkin formulations for the level set
#   advection step.
#
# Visualization:
# - The script provides visualization of the velocity field and the evolving material region.
#
###########################################################################


import warp as wp
import warp.examples.fem.utils as fem_example_utils
import warp.fem as fem


@fem.integrand
def classify_boundary_sides(
    s: fem.Sample,
    domain: fem.Domain,
    dirichlet: wp.array(dtype=int),
    inflow: wp.array(dtype=int),
):
    """Assign boundary sides to inflow or Dirichlet subdomains based on normal direction"""
    nor = fem.normal(domain, s)

    if nor[0] < -0.5:
        inflow[s.qp_index] = 1
        dirichlet[s.qp_index] = 1
    elif nor[0] > 0.5:
        dirichlet[s.qp_index] = 1


@wp.func
def initial_level_set(x: wp.vec2, radius: float):
    """Initial level set function for the material region -- three circles"""

    return (
        wp.min(
            wp.vec3(
                wp.length(x - wp.vec2(0.667, 0.5)),
                wp.length(x - wp.vec2(0.333, 0.333)),
                wp.length(x - wp.vec2(0.333, 0.667)),
            )
        )
        - radius
    )


@fem.integrand
def identity_form(
    s: fem.Sample,
    p: fem.Field,
    q: fem.Field,
):
    return p(s) * q(s)


@fem.integrand
def material_fraction(s: fem.Sample, level_set: fem.Field, smoothing: float):
    """Sigmoid approximation of the level set interior"""
    return 1.0 / (1.0 + wp.exp(-level_set(s) / smoothing))


@fem.integrand
def permeability(s: fem.Sample, level_set: fem.Field, smoothing: float):
    """Define permeability as strictly proportional to material fraction (arbitrary choice)"""
    return material_fraction(s, level_set, smoothing)


@fem.integrand
def velocity_field(s: fem.Sample, level_set: fem.Field, p: fem.Field, smoothing: float):
    """Velocity field based on permeability and pressure gradient according to Darcy's law"""
    return -permeability(s, level_set, smoothing) * fem.grad(p, s)


@fem.integrand
def diffusion_form(s: fem.Sample, level_set: fem.Field, p: fem.Field, q: fem.Field, smoothing: float, scale: float):
    """Inhomogeneous diffusion form"""
    return scale * wp.dot(velocity_field(s, level_set, p, smoothing), fem.grad(q, s))


@fem.integrand
def inflow_velocity(s: fem.Sample, domain: fem.Domain, level_set: fem.Field, p: fem.Field, smoothing: float):
    return wp.dot(velocity_field(s, level_set, p, smoothing), fem.normal(domain, s))


@fem.integrand
def volume_form(s: fem.Sample, level_set: fem.Field, smoothing: float):
    return material_fraction(s, level_set, smoothing)


@wp.kernel
def combine_losses(
    loss: wp.array(dtype=wp.float32),
    vol: wp.array(dtype=wp.float32),
    target_vol: wp.float32,
    vol_weight: wp.float32,
):
    loss[0] += vol_weight * (vol[0] - target_vol) * (vol[0] - target_vol)


@fem.integrand
def advected_level_set_semi_lagrangian(
    s: fem.Sample, domain: fem.Domain, level_set: fem.Field, velocity: fem.Field, dt: float
):
    x_prev = domain(s) - velocity(s) * dt
    s_prev = fem.lookup(domain, x_prev, guess=s)
    return level_set(s_prev)


# Discontinuous Galerkin variant of level set advection


@fem.integrand
def level_set_transport_form(s: fem.Sample, level_set: fem.Field, psi: fem.Field, velocity: fem.Field, dt: float):
    return dt * wp.dot(fem.grad(level_set, s), velocity(s)) * psi(s)


@fem.integrand
def level_set_transport_form_upwind(
    s: fem.Sample, domain: fem.Domain, level_set: fem.Field, psi: fem.Field, velocity: fem.Field, dt: float
):
    vel = dt * velocity(s)
    vel_n = wp.dot(vel, fem.normal(domain, s))
    return fem.jump(level_set, s) * (-fem.average(psi, s) * vel_n + 0.5 * fem.jump(psi, s) * wp.abs(vel_n))


@fem.integrand
def advected_level_set_upwind(
    s: fem.Sample, domain: fem.Domain, level_set: fem.Field, transport_integrals: wp.array(dtype=float)
):
    return level_set(s) - transport_integrals[s.qp_index] / (fem.measure(domain, s) * s.qp_weight)


class Example:
    def __init__(
        self, quiet=False, degree=2, resolution=25, mesh: str = "grid", dt: float = 1.0, discontinuous: bool = False
    ):
        self._quiet = quiet

        self._smoothing = 0.5 / resolution  # smoothing for level set interface approximation as sigmoid
        self._dt = dt  # level set advection time step (~gradient step size)
        self._discontinuous = discontinuous

        if mesh == "tri":
            positions, tri_vidx = fem_example_utils.gen_trimesh(res=wp.vec2i(resolution, resolution))
            self._geo = fem.Trimesh2D(tri_vertex_indices=tri_vidx, positions=positions, build_bvh=True)
        else:
            self._geo = fem.Grid2D(res=wp.vec2i(resolution, resolution))

        # Pressure, level set, and level set velocity spaces
        self._p_space = fem.make_polynomial_space(self._geo, degree=degree, dtype=float)
        self._ls_space = fem.make_polynomial_space(self._geo, degree=degree, dtype=float, discontinuous=discontinuous)
        self._v_space = fem.make_polynomial_space(self._geo, degree=degree, dtype=wp.vec2)

        # pressure field
        self._p_field = fem.make_discrete_field(space=self._p_space)
        self._p_field.dof_values.requires_grad = True

        # level set field
        self._level_set_field = fem.make_discrete_field(space=self._ls_space)
        self._level_set_field.dof_values.requires_grad = True

        # level set advection velocity field
        self._level_set_velocity_field = fem.make_discrete_field(space=self._v_space)
        self._level_set_velocity_field.dof_values.requires_grad = True

        fem.interpolate(
            fem.ImplicitField(fem.Cells(self._geo), initial_level_set, values={"radius": 0.125}),
            dest=self._level_set_field,
        )

        # recording initial volume, we want to preserve that when doing optimization
        self._target_vol = fem.integrate(
            volume_form,
            domain=fem.Cells(self._geo),
            fields={"level_set": self._level_set_field},
            values={"smoothing": self._smoothing},
        )

        # Trial and test functions for pressure solve
        self._p_test = fem.make_test(space=self._p_space)
        self._p_trial = fem.make_trial(space=self._p_space)

        # For discontinuous level set advection
        if self._discontinuous:
            self._ls_test = fem.make_test(space=self._ls_space)
            self._ls_sides_test = fem.make_test(space=self._ls_space, domain=fem.Sides(self._geo))

        # Identify inflow and outflow sides
        boundary = fem.BoundarySides(self._geo)

        inflow_mask = wp.zeros(shape=boundary.element_count(), dtype=int)
        dirichlet_mask = wp.zeros(shape=boundary.element_count(), dtype=int)

        fem.interpolate(
            classify_boundary_sides,
            quadrature=fem.RegularQuadrature(boundary, order=0),
            values={"inflow": inflow_mask, "dirichlet": dirichlet_mask},
        )

        self._inflow = fem.Subdomain(boundary, element_mask=inflow_mask)
        self._dirichlet = fem.Subdomain(boundary, element_mask=dirichlet_mask)

        # Build projector for the inflow and outflow homogeneous Dirichlet condition
        p_dirichlet_bd_test = fem.make_test(space=self._p_space, domain=self._dirichlet)
        p_dirichlet_bd_trial = fem.make_trial(space=self._p_space, domain=self._dirichlet)
        p_dirichlet_bd_matrix = fem.integrate(
            identity_form,
            fields={"p": p_dirichlet_bd_trial, "q": p_dirichlet_bd_test},
            assembly="nodal",
            output_dtype=float,
        )

        # Inflow prescribed pressure
        p_inflow_bd_test = fem.make_test(space=self._p_space, domain=self._inflow)
        p_inflow_bd_value = fem.integrate(
            identity_form,
            fields={"p": fem.UniformField(self._inflow, 1.0), "q": p_inflow_bd_test},
            assembly="nodal",
            output_dtype=float,
        )
        fem.normalize_dirichlet_projector(p_dirichlet_bd_matrix, p_inflow_bd_value)

        self._bd_projector = p_dirichlet_bd_matrix
        self._bd_prescribed_value = p_inflow_bd_value

        self.renderer = fem_example_utils.Plot()

    def step(self):
        p = self._p_field.dof_values
        p.zero_()
        v = self._level_set_velocity_field.dof_values
        v.zero_()

        # Advected level set field, used in adjoint computations
        advected_level_set = fem.make_discrete_field(space=self._ls_space)
        advected_level_set.dof_values.assign(self._level_set_field.dof_values)
        advected_level_set.dof_values.requires_grad = True
        advected_level_set_restriction = fem.make_restriction(advected_level_set, domain=self._p_test.domain)

        # Forward step, record adjoint tape for forces
        p_rhs = wp.empty(self._p_space.node_count(), dtype=wp.float32, requires_grad=True)

        tape = wp.Tape()
        with tape:
            # Dummy advection step, so backward pass can compute adjoint w.r.t advection velocity
            self.advect_level_set(
                level_set_in=self._level_set_field,
                level_set_out=advected_level_set_restriction,
                velocity=self._level_set_velocity_field,
                dt=1.0,
            )

            # Left-hand-side of implicit solve (zero if p=0, but required for adjoint computation through implicit function theorem)
            fem.integrate(
                diffusion_form,
                fields={
                    "level_set": advected_level_set,
                    "p": self._p_field,
                    "q": self._p_test,
                },
                values={"smoothing": self._smoothing, "scale": -1.0},
                output=p_rhs,
            )

        # Diffusion matrix (inhomogeneous Poisson)
        p_matrix = fem.integrate(
            diffusion_form,
            fields={
                "level_set": advected_level_set,
                "p": self._p_trial,
                "q": self._p_test,
            },
            values={"smoothing": self._smoothing, "scale": 1.0},
            output_dtype=float,
        )

        # Project to enforce Dirichlet boundary conditions then solve linear system
        fem.project_linear_system(
            p_matrix, p_rhs, self._bd_projector, self._bd_prescribed_value, normalize_projector=False
        )

        fem_example_utils.bsr_cg(p_matrix, b=p_rhs, x=p, quiet=self._quiet, tol=1e-6, max_iters=1000)

        # Record adjoint of linear solve
        def solve_linear_system():
            fem_example_utils.bsr_cg(p_matrix, b=p.grad, x=p_rhs.grad, quiet=self._quiet, tol=1e-6, max_iters=1000)
            p_rhs.grad -= self._bd_projector @ p_rhs.grad

        tape.record_func(solve_linear_system, arrays=(p_rhs, p))

        # Evaluate losses
        loss = wp.empty(shape=1, dtype=wp.float32, requires_grad=True)
        vol = wp.empty(shape=1, dtype=wp.float32, requires_grad=True)

        with tape:
            # Main objective: inflow flux
            fem.integrate(
                inflow_velocity,
                fields={"level_set": advected_level_set.trace(), "p": self._p_field.trace()},
                values={"smoothing": self._smoothing},
                domain=self._inflow,
                output=loss,
            )

            # Add penalization term enforcing constant volume
            fem.integrate(
                volume_form,
                fields={"level_set": advected_level_set},
                values={"smoothing": self._smoothing},
                domain=self._p_test.domain,
                output=vol,
            )

            print("Total inflow", loss, "Volume", vol)

            vol_loss_weight = 1000.0
            wp.launch(
                combine_losses,
                dim=1,
                inputs=(loss, vol, self._target_vol, vol_loss_weight),
            )

        # perform backward step
        tape.backward(loss=loss)

        # Advect level set with velocity field adjoint
        v.assign(v.grad)
        self.advect_level_set(
            level_set_in=advected_level_set,
            level_set_out=self._level_set_field,
            velocity=self._level_set_velocity_field,
            dt=-self._dt,
        )

        # Zero-out gradients used in tape
        tape.zero()

    def advect_level_set(self, level_set_in: fem.Field, level_set_out: fem.Field, velocity: fem.Field, dt: float):
        if self._discontinuous:
            # Discontinuous Galerkin version with (explicit) upwind transport:
            # Integrate in-cell transport + side flux
            transport_integrals = wp.empty(
                shape=self._ls_space.node_count(),
                dtype=float,
                requires_grad=True,
            )
            fem.integrate(
                level_set_transport_form,
                fields={"psi": self._ls_test, "level_set": level_set_in, "velocity": velocity},
                values={"dt": dt},
                output=transport_integrals,
            )

            fem.integrate(
                level_set_transport_form_upwind,
                fields={"level_set": level_set_in.trace(), "psi": self._ls_sides_test, "velocity": velocity.trace()},
                values={"dt": dt},
                output=transport_integrals,
                add=True,
            )

            # Divide by mass matrix and write back to advected field
            out_field = level_set_out if isinstance(level_set_out, fem.DiscreteField) else level_set_out.field
            fem.interpolate(
                advected_level_set_upwind,
                fields={"level_set": level_set_in, "velocity": velocity},
                values={"transport_integrals": transport_integrals},
                quadrature=fem.NodalQuadrature(self._p_test.domain, self._ls_space),
                dest=out_field.dof_values,
            )
        else:
            # Continuous Galerkin version with semi-Lagrangian transport:
            fem.interpolate(
                advected_level_set_semi_lagrangian,
                fields={"level_set": level_set_in, "velocity": velocity},
                values={"dt": dt},
                dest=level_set_out,
            )

    def render(self):
        # velocity field
        u_space = fem.make_polynomial_space(self._geo, degree=self._p_space.degree, dtype=wp.vec2)
        u_field = fem.make_discrete_field(space=u_space)
        fem.interpolate(
            velocity_field,
            fields={"level_set": self._level_set_field, "p": self._p_field},
            values={"smoothing": self._smoothing},
            dest=u_field,
        )

        # material fraction field
        mat_field = fem.make_discrete_field(space=self._ls_space)
        fem.interpolate(
            material_fraction,
            fields={"level_set": self._level_set_field},
            values={"smoothing": self._smoothing},
            dest=mat_field,
        )

        self.renderer.add_field("velocity", u_field)
        self.renderer.add_field("material", mat_field)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--resolution", type=int, default=100, help="Grid resolution.")
    parser.add_argument("--degree", type=int, default=1, help="Polynomial degree of shape functions.")
    parser.add_argument("--discontinuous", action="store_true", help="Use discontinuous level set advection.")
    parser.add_argument("--mesh", type=str, default="grid", help="Mesh type.")
    parser.add_argument("--num_iters", type=int, default=100, help="Number of iterations.")
    parser.add_argument("--dt", type=float, default=0.05, help="Level set update time step.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode, suppressing the opening of any graphical windows.",
    )

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(
            quiet=True,
            degree=args.degree,
            resolution=args.resolution,
            discontinuous=args.discontinuous,
            mesh=args.mesh,
            dt=args.dt,
        )

        for _iter in range(args.num_iters):
            example.step()
            example.render()

        if not args.headless:
            example.renderer.plot(
                options={
                    "velocity": {"arrows": {"glyph_scale": 0.1}},
                    "material": {"contours": {"levels": [0.0, 0.5, 1.0001]}},
                },
            )
