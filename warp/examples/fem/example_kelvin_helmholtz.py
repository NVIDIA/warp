# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# Example Compressible Euler
#
# Solves the 2D compressible Euler equations with Discontinuous Galerkin
# (DG) and explicit SSPRK3 time integration on a uniform Cartesian grid.
#
# Governing equations (conservative form):
#
#   dU/dt + div F(U) = 0
#
#   U = (rho, rhou, rhov, E),   F = | rhou         rhov        |
#                               | rhou^2+p      rhouv       |
#                               | rhouv        rhov^2+p     |
#                               | (E+p)u     (E+p)v    |
#
#   p = (gamma-1)(E - 1/2rho|u|^2),  gamma = 1.4
#
# DG weak form (same structure as example_shallow_water.py):
#
#   (dU/dt, V)_K = (F(U), grad V)_K  -  <F_num . n, V>_{dK}
#
# with the Rusanov (local Lax-Friedrichs) numerical flux:
#
#   F_num . n = 1/2(F(U_L)+F(U_R)).n - 1/2 lambda_max (U_R - U_L)
#
# where lambda_max = max(|u.n| + c) and c = sqrt(gamma p / rho) is the sound speed.
#
# Test case: Kelvin-Helmholtz instability -- a horizontal shear layer
# (rho=2, u=-0.5 in the middle strip, rho=1, u=0.5 outside) with a small
# sinusoidal perturbation in the v-velocity at the two interfaces.
# Reflective BCs on top/bottom, periodic left/right.
#
# A Zhang-Shu positivity limiter enforces rho > 0 and p > 0 at every DOF
# after each SSPRK3 stage.
#
# For a description of the warp.fem concepts used here (integrands, fields,
# spaces, sides, jump/average, .trace(), etc.), see the detailed comments
# in example_shallow_water.py.
###########################################################################

import math

import warp as wp
import warp.examples.fem.utils as fem_example_utils
import warp.fem as fem

GAMMA = wp.constant(1.4)
PI = wp.constant(3.141592653589793)


# ---------------------------------------------------------------------------
# Compressible Euler physics helpers
# ---------------------------------------------------------------------------


@wp.func
def euler_pressure(U: wp.vec4) -> float:
    """Pressure from equation of state: p = (gamma-1)(E - 1/2rho|u|^2)."""
    rho = wp.max(U[0], 1.0e-10)
    ke = 0.5 * (U[1] * U[1] + U[2] * U[2]) / rho
    return (GAMMA - 1.0) * (U[3] - ke)


@wp.func
def euler_velocity(U: wp.vec4) -> wp.vec2:
    """Primitive velocity u = (rhou, rhov) / rho."""
    rho = wp.max(U[0], 1.0e-10)
    return wp.vec2(U[1] / rho, U[2] / rho)


@wp.func
def euler_flux_dot_normal(U: wp.vec4, n: wp.vec2) -> wp.vec4:
    """Physical flux projected onto a face normal: F(U) . n."""
    rho = wp.max(U[0], 1.0e-10)
    vel = euler_velocity(U)
    p = euler_pressure(U)
    vn = wp.dot(vel, n)

    return wp.vec4(
        rho * vn,
        U[1] * vn + p * n[0],
        U[2] * vn + p * n[1],
        (U[3] + p) * vn,
    )


@wp.func
def euler_max_wavespeed(U: wp.vec4, n: wp.vec2) -> float:
    """Maximum wavespeed in direction n: |u.n| + c, c = sqrt(gamma p / rho)."""
    rho = wp.max(U[0], 1.0e-10)
    p = wp.max(euler_pressure(U), 1.0e-10)
    c = wp.sqrt(GAMMA * p / rho)
    return wp.abs(wp.dot(euler_velocity(U), n)) + c


@wp.func
def euler_reflect(U: wp.vec4, n: wp.vec2) -> wp.vec4:
    """Ghost state for a reflective wall: flip the normal momentum component."""
    mom = wp.vec2(U[1], U[2])
    mom_n = wp.dot(mom, n)
    mom_refl = mom - 2.0 * mom_n * n
    return wp.vec4(U[0], mom_refl[0], mom_refl[1], U[3])


# ---------------------------------------------------------------------------
# warp.fem integrands -- Euler equations
# ---------------------------------------------------------------------------


@fem.integrand
def initial_condition(
    s: fem.Sample, domain: fem.Domain, domain_size: float, domain_width: float, interface_width: float
):
    """Kelvin-Helmholtz instability initial condition.

    Horizontal shear layer at y = domain_size/4 and y = 3*domain_size/4
    with tanh transitions of configurable width. A sinusoidal
    perturbation in v triggers the physical KH instability.
    """
    x = domain(s)
    px = x[0]
    py = x[1]

    y_lo = 0.25 * domain_size
    y_hi = 0.75 * domain_size

    # Smooth profile: 0 outside the strip, 1 inside
    a = interface_width
    profile = 0.5 * (wp.tanh((py - y_lo) / a) - wp.tanh((py - y_hi) / a))

    # Shear layer: dense slow strip in the middle, light fast flow outside
    rho = 1.0 + profile  # 1 outside, 2 inside
    u = 0.5 - profile  # 0.5 outside, -0.5 inside

    # Small sinusoidal v-perturbation localized near the two interfaces
    sigma = wp.max(interface_width, 0.05 * domain_size)
    v = (
        0.01
        * wp.sin(4.0 * PI * px / domain_width)
        * (
            wp.exp(-0.5 * (py - y_lo) * (py - y_lo) / (sigma * sigma))
            + wp.exp(-0.5 * (py - y_hi) * (py - y_hi) / (sigma * sigma))
        )
    )

    p = 2.5  # uniform pressure
    E = p / (GAMMA - 1.0) + 0.5 * rho * (u * u + v * v)

    return wp.vec4(rho, rho * u, rho * v, E)


@fem.integrand
def cell_flux_form(s: fem.Sample, domain: fem.Domain, U: fem.Field, V: fem.Field):
    """Volume integral: (F(U), grad V)_K  (integration-by-parts form).

    Same structure as the shallow water volume term, but with 4 rows
    (mass, x-momentum, y-momentum, energy) instead of 3.
    ``fem.grad(V, s)`` returns a 4x2 matrix for the vec4-valued test space.
    """
    state = U(s)
    grad_v = fem.grad(V, s)
    rho = wp.max(state[0], 1.0e-10)
    vel = euler_velocity(state)
    p = euler_pressure(state)

    result = float(0.0)

    # Mass row: F_0 = rho vel^T
    result -= rho * wp.dot(vel, wp.vec2(grad_v[0, 0], grad_v[0, 1]))

    # x-momentum row: F_1 = (rhou^2 + p, rhou.v)
    fx1 = wp.vec2(state[1] * vel[0] + p, state[1] * vel[1])
    result -= wp.dot(fx1, wp.vec2(grad_v[1, 0], grad_v[1, 1]))

    # y-momentum row: F_2 = (rhov.u, rhov^2 + p)
    fx2 = wp.vec2(state[2] * vel[0], state[2] * vel[1] + p)
    result -= wp.dot(fx2, wp.vec2(grad_v[2, 0], grad_v[2, 1]))

    # Energy row: F_3 = (E+p) vel^T
    e_flux = (state[3] + p) * vel
    result -= wp.dot(e_flux, wp.vec2(grad_v[3, 0], grad_v[3, 1]))

    return result


@fem.integrand
def rusanov_flux_form(
    s: fem.Sample, domain: fem.Domain, U: fem.Field, V: fem.Field, bounds_lo: wp.vec2, bounds_hi: wp.vec2
):
    """Numerical flux integral over cell interfaces (interior + boundary).

    See example_shallow_water.py for detailed documentation of the
    warp.fem side conventions (normal, jump, average, .trace()).

    Boundary conditions:
      - Top/bottom (y = bounds_lo[1] or bounds_hi[1]): reflective wall
      - Left/right (x = bounds_lo[0] or bounds_hi[0]): periodic BC --
        ghost state looked up from opposite boundary via
        ``fem.cells(U)`` + ``fem.lookup``
    """
    nor = fem.normal(domain, s)
    pos = domain(s)

    # Boundary detection
    on_top_bottom = pos[1] <= bounds_lo[1] or pos[1] >= bounds_hi[1]
    on_left_right = pos[0] <= bounds_lo[0] or pos[0] >= bounds_hi[0]
    is_boundary = on_top_bottom or on_left_right

    if is_boundary:
        U_inner = U(s)

        if on_top_bottom:
            U_outer = euler_reflect(U_inner, nor)
        else:
            # Periodic BC: look up state at opposite boundary.
            # fem.cells(U) converts the traced field to a cell-space field
            # that correctly evaluates at cell samples from fem.lookup.
            domain_width = bounds_hi[0] - bounds_lo[0]
            on_left = pos[0] <= bounds_lo[0]
            eps = 1.0e-6 * domain_width
            wrapped_x = wp.where(on_left, bounds_hi[0] - eps, bounds_lo[0] + eps)
            wrapped_pos = wp.vec2(wrapped_x, pos[1])

            cell_domain = fem.cells(domain)
            wrapped_s = fem.lookup(cell_domain, wrapped_pos, s)
            U_cell = fem.cells(U)
            U_outer = U_cell(wrapped_s)

        fn_inner = euler_flux_dot_normal(U_inner, nor)
        fn_outer = euler_flux_dot_normal(U_outer, nor)
        lambda_max = wp.max(euler_max_wavespeed(U_inner, nor), euler_max_wavespeed(U_outer, nor))

        flux = 0.5 * (fn_inner + fn_outer) - 0.5 * lambda_max * (U_outer - U_inner)
        return wp.dot(flux, V(s))
    else:
        # Interior side: recover L/R states from jump/average
        U_avg = fem.average(U, s)
        U_jump = fem.jump(U, s)

        U_inner = U_avg + 0.5 * U_jump
        U_outer = U_avg - 0.5 * U_jump

        fn_inner = euler_flux_dot_normal(U_inner, nor)
        fn_outer = euler_flux_dot_normal(U_outer, nor)
        lambda_max = wp.max(euler_max_wavespeed(U_inner, nor), euler_max_wavespeed(U_outer, nor))

        flux = 0.5 * (fn_inner + fn_outer) - 0.5 * lambda_max * (U_outer - U_inner)
        return wp.dot(flux, fem.jump(V, s))


# ---------------------------------------------------------------------------
# warp.fem integrands -- scalar utilities
# ---------------------------------------------------------------------------


@fem.integrand
def density_field(s: fem.Sample, U: fem.Field):
    """Extract density for validation."""
    return U(s)[0]


@fem.integrand
def get_position(s: fem.Sample, domain: fem.Domain):
    """Return the physical position of the sample point."""
    return domain(s)


# ---------------------------------------------------------------------------
# warp.fem integrands -- slope limiter
# ---------------------------------------------------------------------------


@wp.func
def minmod(a: float, b: float):
    sa = wp.sign(a)
    sb = wp.sign(b)
    return wp.where(sa == sb, sa * wp.min(wp.abs(a), wp.abs(b)), 0.0)


@fem.integrand
def slope_limiter(domain: fem.Domain, s: fem.Sample, U: fem.Field, dx: wp.vec2, rho_jump_threshold: float):
    """Minmod slope limiter with density-based troubled cell indicator.

    Only activates in cells where the density jump to a neighbor exceeds
    ``rho_jump_threshold``, i.e. near contact discontinuities or shocks.
    Smooth regions (vortices, expansion fans) are left unlimited so that
    physical instabilities can develop at full DG accuracy.
    """
    center_coords = fem.Coords(0.5, 0.5, 0.0)
    cell_center = fem.make_free_sample(s.element_index, center_coords)
    center_pos = domain(cell_center)
    u_center = U(cell_center)
    delta_coords = s.element_coords - center_coords

    # Look up 4 neighbor cell centers
    neighbour_xp = fem.lookup(domain, center_pos + wp.vec2(dx[0], 0.0))
    neighbour_xm = fem.lookup(domain, center_pos - wp.vec2(dx[0], 0.0))
    neighbour_yp = fem.lookup(domain, center_pos + wp.vec2(0.0, dx[1]))
    neighbour_ym = fem.lookup(domain, center_pos - wp.vec2(0.0, dx[1]))

    u_nxp = U(neighbour_xp)
    u_nxm = U(neighbour_xm)
    u_nyp = U(neighbour_yp)
    u_nym = U(neighbour_ym)

    # Troubled cell indicator: max density jump to any neighbor
    max_rho_jump = wp.max(
        wp.max(wp.abs(u_nxp[0] - u_center[0]), wp.abs(u_nxm[0] - u_center[0])),
        wp.max(wp.abs(u_nyp[0] - u_center[0]), wp.abs(u_nym[0] - u_center[0])),
    )

    delta_u = U(s) - u_center

    # Only limit in troubled cells (near density discontinuities)
    if max_rho_jump < rho_jump_threshold:
        return U(s)

    # Component-wise minmod limiting
    result = u_center
    for i in range(4):
        gx = minmod(u_nxp[i] - u_center[i], u_center[i] - u_nxm[i]) * delta_coords[0]
        gy = minmod(u_nyp[i] - u_center[i], u_center[i] - u_nym[i]) * delta_coords[1]
        result[i] = result[i] + minmod(gx + gy, delta_u[i])

    return result


# ---------------------------------------------------------------------------
# warp.fem integrands -- positivity limiter
# ---------------------------------------------------------------------------


@fem.integrand
def positivity_limiter(s: fem.Sample, U: fem.Field, eps: float):
    """Zhang-Shu positivity-preserving limiter for compressible Euler.

    Ensures density rho > 0 and pressure p > 0 at every DOF by scaling
    deviations from the cell average. Applied to the full vec4 state.

    Uses ``fem.node_count(U, s)`` and ``fem.at_node(U, s, k)`` to read all
    DOFs in the current element. Must interpolate into a separate field
    to avoid read/write race conditions.
    """
    n = fem.node_count(U, s)

    # Compute cell average and find worst-case density and pressure
    avg = wp.vec4(0.0, 0.0, 0.0, 0.0)
    rho_min = float(1.0e10)
    p_min = float(1.0e10)
    for k in range(n):
        val = U(fem.at_node(U, s, k))
        avg = avg + val
        rho_min = wp.min(rho_min, val[0])
        p_min = wp.min(p_min, euler_pressure(val))
    avg = avg / float(n)

    u_here = U(s)

    if rho_min >= eps and p_min >= eps:
        return u_here

    # Limit density: scale so that min rho >= eps
    rho_avg = avg[0]
    theta = float(1.0)
    if rho_min < eps and rho_avg > eps:
        theta = (rho_avg - eps) / (rho_avg - rho_min)

    result = avg + theta * (u_here - avg)

    # Limit pressure: if pressure is still negative after density limiting,
    # further reduce theta until pressure is non-negative at all nodes.
    # For simplicity, use a conservative single-pass approach: if the
    # cell-average pressure is positive, squeeze towards the average.
    p_result = euler_pressure(result)
    if p_result < eps:
        p_avg = euler_pressure(avg)
        if p_avg > eps:
            # Binary-search-free conservative bound: just use the average
            result = avg
        else:
            # Both pointwise and average pressure are non-positive; reset
            result = wp.vec4(eps, 0.0, 0.0, eps / (GAMMA - 1.0))

    return result


# ---------------------------------------------------------------------------
# warp.fem integrands -- mass forms
# ---------------------------------------------------------------------------


@fem.integrand
def mass_form(s: fem.Sample, U: fem.Field, V: fem.Field):
    """Bilinear mass form M(U, V) = U . V for vec4 nodal diagonal mass matrix."""
    return wp.dot(U(s), V(s))


class Example:
    """Compressible Euler DG solver on a 2D Cartesian grid.

    Same explicit DG framework as example_shallow_water.py but with a
    4-component conserved state (rho, rhou, rhov, E) and a positivity limiter
    that enforces both rho > 0 and p > 0. Visualizes the density field.
    """

    def __init__(
        self,
        resolution=100,
        degree=1,
        num_frames=300,
        domain_size=1.0,
        sim_time=3.0,
        aspect=1.0,
        interface_width=None,
        slope_limiter_enabled=False,
    ):
        res = resolution
        dx = domain_size / res

        # CFL condition: dt < C * dx / (max_wavespeed * (2p+1))
        # Max wavespeed from initial conditions: max(|u| + c) over both fluids,
        # where c = sqrt(gamma p / rho) is the sound speed.
        gamma = 1.4
        p0 = 2.5  # initial uniform pressure
        max_wavespeed = max(
            0.5 + math.sqrt(gamma * p0 / 1.0),  # light fluid: rho=1, |u|=0.5
            0.5 + math.sqrt(gamma * p0 / 2.0),  # heavy fluid: rho=2, |u|=0.5
        )
        self.sim_dt = 0.2 * dx / (max_wavespeed * (2 * degree + 1))
        self.frame_dt = sim_time / num_frames
        self.current_frame = 0
        self.current_time = 0.0
        self._domain_size = domain_size
        self._slope_limiter_enabled = slope_limiter_enabled

        if interface_width is None:
            interface_width = 0.02 * domain_size
        self._interface_width = interface_width

        # Rectangular domain: width = aspect * domain_size, height = domain_size
        # Scale x-resolution to keep square cells
        res_x = int(aspect * resolution)
        domain_width = aspect * domain_size
        self._dx = wp.vec2(domain_width / res_x, domain_size / res)
        self._rho_jump_threshold = 0.1  # Only limit cells with density jump > this

        geo = fem.Grid2D(
            res=wp.vec2i(res_x, resolution),
            bounds_lo=wp.vec2(0.0),
            bounds_hi=wp.vec2(domain_width, domain_size),
        )

        domain = fem.Cells(geometry=geo)
        sides = fem.Sides(geo)

        basis_space = fem.make_polynomial_basis_space(geo, degree=degree, discontinuous=True)
        state_space = fem.make_collocated_function_space(basis_space, dtype=wp.vec4)

        # Euler test/trial (vec4)
        self._test = fem.make_test(space=state_space, domain=domain)
        self._side_test = fem.make_test(space=state_space, domain=sides)

        # Euler diagonal mass matrix (nodal assembly)
        trial = fem.make_trial(space=state_space, domain=domain)
        matrix_mass = fem.integrate(
            mass_form, fields={"U": trial, "V": self._test}, output_dtype=wp.float32, assembly="nodal"
        )
        self._inv_mass_matrix = wp.sparse.bsr_copy(matrix_mass)
        fem_example_utils.invert_diagonal_bsr_matrix(self._inv_mass_matrix)

        # Initial conditions
        self._bounds_lo = wp.vec2(0.0)
        self._bounds_hi = wp.vec2(domain_width, domain_size)
        self.state_field = state_space.make_field()
        fem.interpolate(
            initial_condition,
            dest=self.state_field,
            values={"domain_size": domain_size, "domain_width": domain_width, "interface_width": interface_width},
        )

        # Visualization: interpolate DG density onto a CG space for smoother rendering
        cg_scalar_space = fem.make_polynomial_space(geo, degree=degree, dtype=float, discontinuous=False)
        self.rho_field = cg_scalar_space.make_field()
        self._interpolate_density()

        # Temporary fields for SSPRK3 stages and limiters (pre-allocated)
        self._euler_tmp = state_space.make_field()
        self._limiter_tmp = state_space.make_field()

        # Store references for validation
        self._domain = domain
        self._basis_space = basis_space
        self._initial_mass = fem.integrate(density_field, domain=domain, fields={"U": self.state_field})

        self.renderer = fem_example_utils.Plot()
        self.renderer.add_field("rho", self.rho_field)

        # Capture CUDA graph for the simulation step
        self.use_cuda_graph = wp.get_device().is_cuda
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def _state_delta(self, trial_state):
        """Evaluate the DG spatial operator: M^{-1} [volume_rhs + side_rhs]."""
        rhs = fem.integrate(
            cell_flux_form,
            fields={"U": trial_state, "V": self._test},
            output_dtype=wp.vec4,
        )

        fem.linalg.array_axpy(
            x=fem.integrate(
                rusanov_flux_form,
                fields={"U": trial_state.trace(), "V": self._side_test},
                values={"bounds_lo": self._bounds_lo, "bounds_hi": self._bounds_hi},
                output_dtype=wp.vec4,
            ),
            y=rhs,
            alpha=1.0,
            beta=1.0,
        )

        return self._inv_mass_matrix @ rhs

    def _interpolate_density(self):
        """Interpolate DG density onto CG space for visualization."""
        fem.interpolate(density_field, dest=self.rho_field, fields={"U": self.state_field})

    def _limit_slope(self, field):
        """Apply minmod slope limiter (troubled cells only) to suppress DG oscillations."""
        if not self._slope_limiter_enabled:
            return
        fem.interpolate(
            slope_limiter,
            dest=self._limiter_tmp,
            fields={"U": field},
            values={"dx": self._dx, "rho_jump_threshold": self._rho_jump_threshold},
        )
        wp.copy(dest=field.dof_values, src=self._limiter_tmp.dof_values)

    def _limit_positivity(self, field):
        """Apply Zhang-Shu positivity limiter to ensure rho > 0 and p > 0."""
        fem.interpolate(positivity_limiter, dest=self._limiter_tmp, fields={"U": field}, values={"eps": 1.0e-6})
        wp.copy(dest=field.dof_values, src=self._limiter_tmp.dof_values)

    def _substep(self, dt):
        """One SSPRK3 substep advancing the Euler state by dt."""
        tmp = self._euler_tmp

        # ---- Stage 1 ----
        k1 = self._state_delta(self.state_field)

        fem.linalg.array_axpy(y=tmp.dof_values, x=self.state_field.dof_values, alpha=1.0, beta=0.0)
        fem.linalg.array_axpy(y=tmp.dof_values, x=k1, alpha=-dt, beta=1.0)
        self._limit_slope(tmp)
        self._limit_positivity(tmp)

        # ---- Stage 2 ----
        k2 = self._state_delta(tmp)

        fem.linalg.array_axpy(y=tmp.dof_values, x=k1, alpha=0.75 * dt, beta=1.0)
        fem.linalg.array_axpy(y=tmp.dof_values, x=k2, alpha=-0.25 * dt, beta=1.0)
        self._limit_slope(tmp)
        self._limit_positivity(tmp)

        # ---- Stage 3 ----
        k3 = self._state_delta(tmp)

        fem.linalg.array_axpy(y=self.state_field.dof_values, x=k1, alpha=-1.0 / 6.0 * dt, beta=1.0)
        fem.linalg.array_axpy(y=self.state_field.dof_values, x=k2, alpha=-1.0 / 6.0 * dt, beta=1.0)
        fem.linalg.array_axpy(y=self.state_field.dof_values, x=k3, alpha=-2.0 / 3.0 * dt, beta=1.0)
        self._limit_slope(self.state_field)
        self._limit_positivity(self.state_field)

    def simulate(self):
        """Run all substeps for one visualization frame (GPU-capturable).

        The substep count and dt values are deterministic (frame_dt and sim_dt
        are constant), so the entire frame can be captured as a CUDA graph.
        """
        t_remaining = self.frame_dt
        while t_remaining > 1.0e-12:
            dt = min(self.sim_dt, t_remaining)
            self._substep(dt)
            t_remaining -= dt

        # Interpolate density for visualization
        self._interpolate_density()

    def step(self):
        """Advance one visualization frame, using a captured CUDA graph when available."""
        self.current_frame += 1

        if self.use_cuda_graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.current_time += self.frame_dt

    def render(self):
        self.renderer.begin_frame(time=self.current_time)
        self.renderer.add_field("rho", self.rho_field)
        self.renderer.end_frame()

    def validate(self):
        """Validate the Euler solution against physical expectations.

        Checks finite values, density/pressure positivity, mass conservation,
        and that the Kelvin-Helmholtz shear layer retains its expected structure.
        """
        import numpy as np  # noqa: PLC0415

        state_np = self.state_field.dof_values.numpy()

        # All values must be finite
        assert np.all(np.isfinite(state_np)), "Non-finite values in state"

        rho = state_np[:, 0]
        rhou = state_np[:, 1]
        rhov = state_np[:, 2]
        E = state_np[:, 3]

        # Density must be positive
        assert np.all(rho > 0), f"Non-positive density: min rho = {rho.min()}"

        # Pressure must be positive: p = (gamma-1)(E - 1/2(rhou^2 + rhov^2)/rho)
        rho_safe = np.maximum(rho, 1e-10)
        ke = 0.5 * (rhou**2 + rhov**2) / rho_safe
        pressure = 0.4 * (E - ke)
        assert np.all(pressure > -1e-6), f"Negative pressure: min p = {pressure.min()}"

        # Mass conservation (periodic + reflective BCs form a nearly closed
        # system; small errors from positivity limiter and periodic lookup offset)
        current_mass = fem.integrate(density_field, domain=self._domain, fields={"U": self.state_field})
        rel_mass_err = abs(current_mass - self._initial_mass) / self._initial_mass
        assert rel_mass_err < 0.05, f"Mass not conserved: relative error = {rel_mass_err:.2e}"

        # Density should remain bounded (initial range is [1, 2])
        assert rho.max() < 5.0, f"Density too large: max rho = {rho.max()}"

        # Get DOF positions and check shear-layer structure
        pos_space = fem.make_collocated_function_space(self._basis_space, dtype=wp.vec2)
        pos_field = pos_space.make_field()
        fem.interpolate(get_position, dest=pos_field)
        pos_np = pos_field.dof_values.numpy()

        py = pos_np[:, 1]
        ds = self._domain_size

        # Middle strip (y in [0.3, 0.7] of domain) should be denser than edges
        mid_mask = (py > 0.3 * ds) & (py < 0.7 * ds)
        edge_mask = (py < 0.15 * ds) | (py > 0.85 * ds)

        if mid_mask.sum() > 10 and edge_mask.sum() > 10:
            assert rho[mid_mask].mean() > rho[edge_mask].mean(), (
                f"Middle strip should be denser: mid rho={rho[mid_mask].mean():.3f}, edge rho={rho[edge_mask].mean():.3f}"
            )


if __name__ == "__main__":
    import argparse

    wp.set_module_options({"enable_backward": False})

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--resolution", type=int, default=100, help="Grid resolution.")
    parser.add_argument("--degree", choices=(0, 1), type=int, default=1, help="Discretization order.")
    parser.add_argument("--num-frames", type=int, default=300, help="Total number of frames.")
    parser.add_argument("--sim-time", type=float, default=3.0, help="Total simulation time.")
    parser.add_argument("--domain-size", type=float, default=1.0, help="Domain height.")
    parser.add_argument("--aspect", type=float, default=1.0, help="Domain width/height aspect ratio.")
    parser.add_argument(
        "--interface-width", type=float, default=None, help="Interface width (default: 2%% of domain size)."
    )
    parser.add_argument("--slope-limiter", action="store_true", help="Enable minmod slope limiter.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode, suppressing the opening of any graphical windows.",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Export animation to file (e.g. animation.mp4) or directory (e.g. frames/).",
    )

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(
            resolution=args.resolution,
            degree=args.degree,
            num_frames=args.num_frames,
            domain_size=args.domain_size,
            sim_time=args.sim_time,
            aspect=args.aspect,
            interface_width=args.interface_width,
            slope_limiter_enabled=args.slope_limiter,
        )

        for _k, _ in fem_example_utils.progress_bar(args.num_frames, quiet=args.quiet):
            example.step()
            example.render()

        # example.validate()

        if not args.headless or args.export:
            example.renderer.plot(
                options={"rho": {"contours": {"levels": 5}, "clim": (1.0, 2.0), "cmap": "RdBu_r"}},
                save=args.export,
            )
