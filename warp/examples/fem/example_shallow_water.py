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
# Example Shallow Water
#
# Solves the 2D shallow water equations with Discontinuous Galerkin (DG)
# and explicit SSPRK3 time integration on a uniform Cartesian grid.
#
# Governing equations (conservative form):
#
#   dU/dt + div F(U) = 0
#
#   U = (h, hu, hv),   F = | h*u          h*v         |
#                           | hu*u+½gh²    hu*v        |
#                           | hv*u         hv*v+½gh²   |
#
# DG weak form (per cell K, for all test functions V in the DG space):
#
#   (dU/dt, V)_K = (F(U), grad V)_K  -  <F_num · n, V>_{dK}
#
# The first term is the volume integral (cell_flux_form); the second is the
# numerical flux integrated over cell boundaries (rusanov_flux_form). This
# example uses the Rusanov (local Lax-Friedrichs) numerical flux:
#
#   F_num · n = ½(F(U_L) + F(U_R))·n  -  ½ λ_max (U_R - U_L)
#
# where λ_max = max(|u|+c) over both sides and c = sqrt(g*h).
#
# Test case: circular dam break centered in the domain with h=2 inside
# a radius of 0.25 and h=1 outside, with reflective wall BCs.
#
# Key warp.fem concepts used:
#
#   @fem.integrand    Pointwise expression evaluated at quadrature points.
#                     Arguments typed fem.Sample, fem.Domain, fem.Field
#                     are automatically bound by warp.fem during integration.
#
#   fem.integrate()   Assembles a weak-form integral over a domain (cells
#                     or sides) by evaluating an integrand at quadrature
#                     points. Returns a BSR matrix (bilinear) or vector.
#
#   fem.interpolate() Evaluates an integrand at DOF locations and writes
#                     the result into a discrete field (no quadrature).
#
#   fem.Cells         Integration domain over cell interiors.
#   fem.Sides         Integration domain over all inter-cell faces and
#                     boundary faces—used for the numerical flux.
#
#   field.trace()     Restricts a cell-based (DG) field to the side
#                     domain so it can be evaluated in side integrands.
#                     On interior sides the field is double-valued.
#
#   fem.jump(f, s)    = f_inner(s) - f_outer(s)   (on a side)
#   fem.average(f, s) = ½(f_inner(s) + f_outer(s))
#   fem.normal()      Unit normal pointing from inner to outer.
#
#   assembly="nodal"  Uses DOF nodes as quadrature points, producing a
#                     diagonal mass matrix (exact for DG with collocated
#                     nodes). This avoids a global linear solve each step.
###########################################################################

import warp as wp
import warp.examples.fem.utils as fem_example_utils
import warp.fem as fem

GRAVITY = wp.constant(9.81)


# ---------------------------------------------------------------------------
# Shallow water physics helpers
#
# These are plain @wp.func (GPU-callable functions), not integrands.
# They encapsulate the flux algebra and can be called from any kernel.
# ---------------------------------------------------------------------------


@wp.func
def sw_velocity(U: wp.vec3) -> wp.vec2:
    """Primitive velocity u = (hu, hv) / h from conserved state U = (h, hu, hv)."""
    h = wp.max(U[0], 1.0e-6)
    return wp.vec2(U[1] / h, U[2] / h)


@wp.func
def sw_wavespeed(U: wp.vec3) -> float:
    """Maximum wavespeed |u| + sqrt(g*h) for CFL / Rusanov dissipation."""
    h = wp.max(U[0], 1.0e-6)
    vel = sw_velocity(U)
    return wp.length(vel) + wp.sqrt(GRAVITY * h)


@wp.func
def sw_flux_dot_normal(U: wp.vec3, n: wp.vec2) -> wp.vec3:
    """Physical flux projected onto a face normal: F(U) · n."""
    h = wp.max(U[0], 1.0e-6)
    vel = sw_velocity(U)
    vn = wp.dot(vel, n)

    f_h = h * vn
    f_hu = U[1] * vn + 0.5 * GRAVITY * h * h * n[0]
    f_hv = U[2] * vn + 0.5 * GRAVITY * h * h * n[1]

    return wp.vec3(f_h, f_hu, f_hv)


@wp.func
def sw_reflect(U: wp.vec3, n: wp.vec2) -> wp.vec3:
    """Ghost state for a reflective wall: flip the normal momentum component."""
    mom = wp.vec2(U[1], U[2])
    mom_n = wp.dot(mom, n)
    mom_refl = mom - 2.0 * mom_n * n
    return wp.vec3(U[0], mom_refl[0], mom_refl[1])


# ---------------------------------------------------------------------------
# warp.fem integrands
#
# An @fem.integrand is a pointwise expression that warp.fem evaluates at
# quadrature points during fem.integrate() or at DOF nodes during
# fem.interpolate(). Special argument types are automatically provided:
#
#   fem.Sample  s       The current quadrature/sample point.
#   fem.Domain  domain  Provides geometric queries: domain(s) returns the
#                        physical position; fem.normal(domain, s) returns
#                        the outward normal (on sides).
#   fem.Field   U, V    Discrete fields. U(s) evaluates the field value;
#                        fem.grad(V, s) evaluates the gradient.
#
# Additional scalar/vector arguments (center, dam_radius, bounds, …) are
# passed as ``values={...}`` when calling fem.integrate/interpolate.
# ---------------------------------------------------------------------------


@fem.integrand
def initial_condition(s: fem.Sample, domain: fem.Domain, center: wp.vec2, dam_radius: float):
    """Circular dam break: h=2 inside, h=1 outside, zero velocity.

    Used with fem.interpolate() to set DOF values of the state field directly
    (no quadrature—just evaluates the function at each DOF location).
    """
    x = domain(s)
    r = wp.length(x - center)
    h = wp.where(r < dam_radius, 2.0, 1.0)
    return wp.vec3(h, 0.0, 0.0)


@fem.integrand
def cell_flux_form(s: fem.Sample, domain: fem.Domain, U: fem.Field, V: fem.Field):
    """Volume integral: (F(U), grad V)_K  (integration-by-parts form).

    Integrated over fem.Cells (all cell interiors). The result is a vector
    of the same size as the DG DOF array—one entry per test function V.

    ``U(s)`` evaluates the discrete solution at the quadrature point s;
    ``fem.grad(V, s)`` returns the gradient of the test function, which for
    a vec3-valued function space is a 3x2 matrix (3 components x 2 spatial dims).
    The double contraction -F(U) : grad(V) is computed row by row.
    """
    state = U(s)
    grad_v = fem.grad(V, s)
    h = wp.max(state[0], 1.0e-6)
    vel = sw_velocity(state)

    result = float(0.0)

    # Mass equation row: F_0 = h * vel^T
    result -= h * wp.dot(vel, wp.vec2(grad_v[0, 0], grad_v[0, 1]))

    # x-momentum row: F_1 = (hu*u + ½gh², hu*v)
    p_term_x = wp.vec2(state[1] * vel[0] + 0.5 * GRAVITY * h * h, state[1] * vel[1])
    result -= wp.dot(p_term_x, wp.vec2(grad_v[1, 0], grad_v[1, 1]))

    # y-momentum row: F_2 = (hv*u, hv*v + ½gh²)
    p_term_y = wp.vec2(state[2] * vel[0], state[2] * vel[1] + 0.5 * GRAVITY * h * h)
    result -= wp.dot(p_term_y, wp.vec2(grad_v[2, 0], grad_v[2, 1]))

    return result


@fem.integrand
def rusanov_flux_form(
    s: fem.Sample, domain: fem.Domain, U: fem.Field, V: fem.Field, bounds_lo: wp.vec2, bounds_hi: wp.vec2
):
    """Numerical flux integral over cell interfaces (interior + boundary).

    Integrated over fem.Sides, which iterates over every inter-cell face and
    every boundary face of the mesh. The field U must be passed as
    ``trial_state.trace()`` so that it can be evaluated on the side domain;
    ``.trace()`` restricts a DG cell field to the side, making it double-valued
    on interior sides (accessible via fem.jump / fem.average).

    warp.fem side convention:
      - ``fem.normal(domain, s)`` points from the *inner* to the *outer* cell.
      - ``fem.jump(f, s) = f_inner - f_outer``
      - ``fem.average(f, s) = ½(f_inner + f_outer)``
      - On boundary sides, only the inner value exists (the cell touching
        the boundary); ``U(s)`` returns the inner value directly.

    The Rusanov (local Lax-Friedrichs) numerical flux is:
      F_num·n = ½(F(U_L)+F(U_R))·n - ½ λ_max (U_R - U_L)
    where n points from L (inner) to R (outer), and the dissipation term
    damps oscillations across the interface.
    """
    nor = fem.normal(domain, s)
    pos = domain(s)

    # Boundary detection: side center lies on a domain boundary face
    is_boundary = wp.min(pos - bounds_lo) <= 0.0 or wp.min(bounds_hi - pos) <= 0.0

    if is_boundary:
        # Reflective wall: fabricate a ghost state by mirroring the momentum
        U_inner = U(s)
        U_outer = sw_reflect(U_inner, nor)

        fn_inner = sw_flux_dot_normal(U_inner, nor)
        fn_outer = sw_flux_dot_normal(U_outer, nor)

        lambda_max = wp.max(sw_wavespeed(U_inner), sw_wavespeed(U_outer))

        flux = 0.5 * (fn_inner + fn_outer) - 0.5 * lambda_max * (U_outer - U_inner)

        # On boundary sides V is single-valued, so just V(s)
        return wp.dot(flux, V(s))
    else:
        # Interior side: recover L/R states from the jump/average operators.
        # Since jump = inner - outer:
        #   inner = average + ½ jump
        #   outer = average - ½ jump
        U_avg = fem.average(U, s)
        U_jump = fem.jump(U, s)

        U_inner = U_avg + 0.5 * U_jump
        U_outer = U_avg - 0.5 * U_jump

        fn_inner = sw_flux_dot_normal(U_inner, nor)
        fn_outer = sw_flux_dot_normal(U_outer, nor)

        lambda_max = wp.max(sw_wavespeed(U_inner), sw_wavespeed(U_outer))

        flux = 0.5 * (fn_inner + fn_outer) - 0.5 * lambda_max * (U_outer - U_inner)

        # On interior sides the test function is also double-valued;
        # fem.jump(V, s) ensures the flux contributes with opposite signs
        # to the two cells sharing this face, as required by the DG formulation.
        return wp.dot(flux, fem.jump(V, s))


@fem.integrand
def water_height(s: fem.Sample, U: fem.Field):
    """Extract scalar water height from the vec3 state.

    Used both for visualization (fem.interpolate into a scalar field)
    and for computing total mass (fem.integrate over cells).
    """
    return U(s)[0]


@fem.integrand
def get_position(s: fem.Sample, domain: fem.Domain):
    """Return the physical position of the sample point.

    Used with fem.interpolate() to build a position field that maps
    each DOF index to its spatial coordinates—needed for post-processing
    (e.g. computing radial profiles for validation).
    """
    return domain(s)


@fem.integrand
def positivity_limiter(s: fem.Sample, U: fem.Field, eps: float):
    """Zhang-Shu positivity-preserving limiter for DG water height.

    For each cell, linearly scales deviations from the cell-average state
    so that the minimum height across DOFs stays >= eps.  Because only the
    deviation is scaled (not the mean), the cell average is preserved and
    mass is exactly conserved.  Applied after every SSPRK3 stage to prevent
    negative water depths near steep gradients.

    Uses ``fem.node_count(U, s)`` to query the number of DOFs in the element
    containing sample ``s``, and ``U(fem.at_node(U, s, k))`` to read the
    field value at the k-th element node.  Must be interpolated into a
    *different* field than U to avoid read/write race conditions.
    """
    n = fem.node_count(U, s)

    # Compute cell average and minimum height over all element nodes
    avg = wp.vec3(0.0, 0.0, 0.0)
    h_min = float(1.0e10)
    for k in range(n):
        val = U(fem.at_node(U, s, k))
        avg = avg + val
        h_min = wp.min(h_min, val[0])
    avg = avg / float(n)

    # Current DOF value at this node
    u_here = U(s)

    if h_min >= eps:
        return u_here

    # Scale deviations from cell average to ensure h >= eps
    if avg[0] > eps:
        theta = (avg[0] - eps) / (avg[0] - h_min)
        return avg + theta * (u_here - avg)
    else:
        # Cell average itself is too small; clamp to minimum height with zero velocity
        return wp.vec3(eps, 0.0, 0.0)


@fem.integrand
def mass_form(s: fem.Sample, U: fem.Field, V: fem.Field):
    """Bilinear mass form M(U, V) = U · V.

    When integrated with ``assembly="nodal"`` (quadrature at DOF nodes), this
    produces a diagonal mass matrix—exact for DG with collocated nodes.  The
    inverse is computed once and used at every time step to convert the RHS
    vector to DOF increments: dU = M^{-1} RHS.
    """
    return wp.dot(U(s), V(s))


class Example:
    """Shallow water DG solver on a 2D Cartesian grid.

    The solver is fully explicit: the DG mass matrix is diagonal (nodal
    assembly) so no linear system needs to be solved.  Time integration
    uses the third-order strong-stability-preserving Runge-Kutta method
    (SSPRK3), which is a convex combination of forward-Euler steps and
    therefore preserves the TVD property of the spatial discretization.

    Each time step:
      1. Evaluate the spatial operator  dU/dt = M^{-1} [volume_rhs + side_rhs]
         three times (one per SSPRK3 stage).
      2. Apply the positivity limiter after each stage.
      3. Combine the stages with the SSPRK3 weights.
    """

    def __init__(self, quiet=False, resolution=50, degree=1, domain_size=1.0):
        self._quiet = quiet

        res = resolution
        dx = domain_size / res
        max_wavespeed = (9.81 * 2.0) ** 0.5
        # CFL condition for DG: dt ~ dx / (max_wavespeed * (2p+1))
        self.sim_dt = 0.2 * dx / (max_wavespeed * (2 * degree + 1))
        self.current_frame = 0
        self._domain_size = domain_size
        self._dam_radius = 0.25

        # --- Geometry and function spaces ---
        #
        # Grid2D creates a uniform quad mesh on [bounds_lo, bounds_hi]^2.
        geo = fem.Grid2D(
            res=wp.vec2i(resolution),
            bounds_lo=wp.vec2(0.0),
            bounds_hi=wp.vec2(domain_size),
        )

        # fem.Cells: integration domain over cell interiors (volume integrals).
        # fem.Sides: integration domain over all faces (numerical flux integrals),
        #            including both interior inter-cell faces and boundary faces.
        domain = fem.Cells(geometry=geo)
        sides = fem.Sides(geo)

        # DG function space: discontinuous polynomial basis of the given degree.
        # ``discontinuous=True`` means DOFs are not shared between cells, so each
        # cell has its own independent polynomial representation.
        # ``dtype=wp.vec3`` makes each DOF a 3-component vector (h, hu, hv).
        basis_space = fem.make_polynomial_basis_space(geo, degree=degree, discontinuous=True)
        state_space = fem.make_collocated_function_space(basis_space, dtype=wp.vec3)
        scalar_space = fem.make_collocated_function_space(basis_space, dtype=float)

        # Test and trial functions.
        # ``make_test`` / ``make_trial`` bind a function space to an integration
        # domain, producing objects that can be passed to fem.integrate().
        # The test on ``domain`` (cells) is used for the volume integral;
        # the test on ``sides`` is used for the numerical flux integral.
        self._test = fem.make_test(space=state_space, domain=domain)
        self._side_test = fem.make_test(space=state_space, domain=sides)

        # Diagonal mass matrix via nodal assembly.
        # With ``assembly="nodal"`` the quadrature points coincide with the DOF
        # nodes, so the mass matrix M_ij = integral(phi_i · phi_j) is diagonal.
        # We invert it once; each time step just does M^{-1} @ rhs.
        trial = fem.make_trial(space=state_space, domain=domain)
        matrix_mass = fem.integrate(
            mass_form, fields={"U": trial, "V": self._test}, output_dtype=wp.float32, assembly="nodal"
        )
        self._inv_mass_matrix = wp.sparse.bsr_copy(matrix_mass)
        fem_example_utils.invert_diagonal_bsr_matrix(self._inv_mass_matrix)

        # --- Initial condition ---
        #
        # fem.interpolate() evaluates the integrand at each DOF node and writes
        # the result directly into the field's DOF array (no quadrature needed).
        # Extra parameters (center, dam_radius) are passed via ``values={...}``.
        self._center = wp.vec2(domain_size / 2.0)
        self._bounds_lo = wp.vec2(0.0)
        self._bounds_hi = wp.vec2(domain_size)
        self.state_field = state_space.make_field()
        fem.interpolate(
            initial_condition, dest=self.state_field, values={"center": self._center, "dam_radius": self._dam_radius}
        )

        # Scalar field for visualization (just the h component)
        self.height_field = scalar_space.make_field()
        fem.interpolate(water_height, dest=self.height_field, fields={"U": self.state_field})

        # Temporary fields: one for SSPRK3 stages, one for positivity limiter
        # (double-buffering to avoid read/write conflicts: interpolate reads
        # from state, writes to tmp)
        self._stage_tmp = state_space.make_field()
        self._limiter_tmp = state_space.make_field()

        # Store references for validation
        self._domain = domain
        self._basis_space = basis_space
        self._initial_mass = fem.integrate(water_height, domain=domain, fields={"U": self.state_field})

        self.renderer = fem_example_utils.Plot()
        self.renderer.add_field("height", self.height_field)

        # Capture CUDA graph for the simulation step
        self.use_cuda_graph = wp.get_device().is_cuda
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def _state_delta(self, trial_state):
        """Evaluate the DG spatial operator: M^{-1} [volume_rhs + side_rhs].

        Returns the time derivative dU/dt as a DOF array (wp.array of vec3).

        1. ``fem.integrate(cell_flux_form, ...)`` assembles the volume integral
           (F(U), grad V)_K over all cells, returning a vector.

        2. ``fem.integrate(rusanov_flux_form, ...)`` assembles the numerical
           flux integral over all sides. The field ``trial_state.trace()``
           restricts the cell DG field to the side domain so that it becomes
           double-valued on interior faces (enabling jump/average queries).

        3. The two contributions are summed (array_axpy) and multiplied by
           M^{-1} (a simple diagonal scaling thanks to nodal assembly).
        """
        # Volume integral: (F(U), grad V)_K
        rhs = fem.integrate(
            cell_flux_form,
            fields={"U": trial_state, "V": self._test},
            output_dtype=wp.vec3,
        )

        # Side integral: -<F_num · n, [V]>  (flux across all faces)
        # trial_state.trace() restricts the DG field to the side domain.
        fem.integrate(
            rusanov_flux_form,
            fields={"U": trial_state.trace(), "V": self._side_test},
            values={"bounds_lo": self._bounds_lo, "bounds_hi": self._bounds_hi},
            output=rhs,
            add=True,
        )

        # Solve M dU = rhs  (diagonal M, so just element-wise multiply by M^{-1})
        return self._inv_mass_matrix @ rhs

    def _limit_positivity(self, field):
        """Apply Zhang-Shu positivity limiter to ensure h >= eps in all DOFs.

        Interpolates the limiter integrand from ``field`` into a temporary
        field, then copies the result back. The double-buffering avoids
        read/write race conditions (all nodes in a cell must read the
        *original* values to compute the cell average before any are modified).
        """
        fem.interpolate(positivity_limiter, dest=self._limiter_tmp, fields={"U": field}, values={"eps": 1.0e-6})
        wp.copy(dest=field.dof_values, src=self._limiter_tmp.dof_values)

    def simulate(self):
        """Run one SSPRK3 time step (GPU-capturable).

        The SSPRK3 method evaluates the spatial operator L(U) three times:

          U^(1) = U^n       - dt * L(U^n)
          U^(2) = ¾ U^n     + ¼ (U^(1) - dt * L(U^(1)))
          U^(n+1) = ⅓ U^n   + ⅔ (U^(2) - dt * L(U^(2)))

        ``fem.linalg.array_axpy(x, y, alpha, beta)`` computes y = alpha*x + beta*y
        in-place on the DOF arrays, avoiding temporary allocations.

        The positivity limiter is applied after each stage to prevent negative
        water heights from entering the next operator evaluation.
        """
        tmp = self._stage_tmp

        # Stage 1: U^(1) = U^n - dt * L(U^n)
        k1 = self._state_delta(self.state_field)

        fem.linalg.array_axpy(y=tmp.dof_values, x=self.state_field.dof_values, alpha=1.0, beta=0.0)
        fem.linalg.array_axpy(y=tmp.dof_values, x=k1, alpha=-self.sim_dt, beta=1.0)
        self._limit_positivity(tmp)

        # Stage 2: U^(2) = ¾ U^n + ¼ U^(1) - ¼ dt * L(U^(1))
        k2 = self._state_delta(tmp)

        fem.linalg.array_axpy(y=tmp.dof_values, x=k1, alpha=0.75 * self.sim_dt, beta=1.0)
        fem.linalg.array_axpy(y=tmp.dof_values, x=k2, alpha=-0.25 * self.sim_dt, beta=1.0)
        self._limit_positivity(tmp)

        # Stage 3: U^(n+1) = ⅓ U^n + ⅔ U^(2) - ⅔ dt * L(U^(2))
        k3 = self._state_delta(tmp)

        fem.linalg.array_axpy(y=self.state_field.dof_values, x=k1, alpha=-1.0 / 6.0 * self.sim_dt, beta=1.0)
        fem.linalg.array_axpy(y=self.state_field.dof_values, x=k2, alpha=-1.0 / 6.0 * self.sim_dt, beta=1.0)
        fem.linalg.array_axpy(y=self.state_field.dof_values, x=k3, alpha=-2.0 / 3.0 * self.sim_dt, beta=1.0)
        self._limit_positivity(self.state_field)

        # Project the vec3 state onto a scalar field for visualization
        fem.interpolate(water_height, dest=self.height_field, fields={"U": self.state_field})

    def step(self):
        """Advance one time step, using a captured CUDA graph when available."""
        self.current_frame += 1

        if self.use_cuda_graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

    def render(self):
        self.renderer.begin_frame(time=self.current_frame * self.sim_dt)
        self.renderer.add_field("height", self.height_field)
        self.renderer.end_frame()

    def validate(self):
        """Validate the shallow water solution against physical expectations.

        Checks finite values, height positivity, mass conservation,
        and radial structure of the circular dam break.
        """
        import numpy as np  # noqa: PLC0415

        state_np = self.state_field.dof_values.numpy()

        # All values must be finite
        assert np.all(np.isfinite(state_np)), "Non-finite values in state"

        h_all = state_np[:, 0]

        # Height must be positive and bounded
        assert np.all(h_all > 0), f"Non-positive height: min h = {h_all.min()}"
        assert h_all.max() < 3.0, f"Height too large: max h = {h_all.max()}"

        # Mass conservation (integral of h over domain)
        current_mass = fem.integrate(water_height, domain=self._domain, fields={"U": self.state_field})
        rel_mass_err = abs(current_mass - self._initial_mass) / self._initial_mass
        assert rel_mass_err < 1e-4, f"Mass not conserved: relative error = {rel_mass_err:.2e}"

        # Get DOF positions for radial analysis
        pos_space = fem.make_collocated_function_space(self._basis_space, dtype=wp.vec2)
        pos_field = pos_space.make_field()
        fem.interpolate(get_position, dest=pos_field)
        pos_np = pos_field.dof_values.numpy()

        cx, cy = self._domain_size / 2.0, self._domain_size / 2.0
        px, py = pos_np[:, 0], pos_np[:, 1]
        r_all = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)

        t = self.current_frame * self.sim_dt
        max_wavespeed = (9.81 * 2.0) ** 0.5

        # Undisturbed outer region (beyond the fastest wave) should have h ≈ 1
        shock_radius = self._dam_radius + max_wavespeed * t
        far_mask = r_all > shock_radius + 0.05
        if far_mask.sum() > 0:
            h_far = h_all[far_mask]
            assert abs(h_far.mean() - 1.0) < 0.05, f"Undisturbed outer region h = {h_far.mean():.4f}, expected ~1.0"

        # Radial velocity should be outward (positive) in the active region
        h_safe = np.maximum(h_all, 1e-6)
        r_safe = np.maximum(r_all, 1e-10)
        hu_all, hv_all = state_np[:, 1], state_np[:, 2]
        ur_all = ((px - cx) * hu_all + (py - cy) * hv_all) / (r_safe * h_safe)

        active_mask = (r_all > 0.20) & (r_all < shock_radius - 0.02)
        if active_mask.sum() > 10:
            assert ur_all[active_mask].mean() > 0, (
                f"Mean radial velocity in active region is negative: {ur_all[active_mask].mean():.4f}"
            )


if __name__ == "__main__":
    import argparse

    wp.set_module_options({"enable_backward": False})

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--resolution", type=int, default=50, help="Grid resolution.")
    parser.add_argument("--degree", choices=(0, 1), type=int, default=1, help="Discretization order.")
    parser.add_argument("--num-frames", type=int, default=200, help="Total number of frames.")
    parser.add_argument("--domain-size", type=float, default=1.0, help="Domain side length.")
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
            quiet=args.quiet,
            resolution=args.resolution,
            degree=args.degree,
            domain_size=args.domain_size,
        )

        for _k, _ in fem_example_utils.progress_bar(args.num_frames, quiet=args.quiet):
            example.step()
            example.render()

        # example.validate()

        if not args.headless or args.export:
            example.renderer.plot(save=args.export)
