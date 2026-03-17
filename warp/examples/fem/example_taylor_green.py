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
# Example Taylor-Green Vortex
#
# Solves the 2D incompressible Navier-Stokes equations on [0,1]^2 and
# validates the numerical solution against the exact Taylor-Green vortex:
#
#   u(x, y, t) = -cos(2*pi*x) * sin(2*pi*y) * exp(-8*pi^2*nu*t)
#   v(x, y, t) =  sin(2*pi*x) * cos(2*pi*y) * exp(-8*pi^2*nu*t)
#   p(x, y, t) = -1/4*(cos(4*pi*x) + cos(4*pi*y)) * exp(-16*pi^2*nu*t)
#
# This is a rare Navier-Stokes test case with a closed-form solution: the
# nonlinear advection term (u . grad)u happens to equal a pressure gradient,
# so the vortex decays purely by viscous diffusion.
#
# Discretization
# --------------
# - **Spatial**: Taylor-Hood Q_d / Q_{d-1} mixed finite elements.
#   The velocity is approximated in a vector-valued space of degree d
#   (e.g. Q2 — biquadratic), and the pressure in a scalar space of
#   degree d-1 (e.g. Q1 — bilinear). This pairing satisfies the
#   inf-sup (LBB) stability condition, avoiding pressure oscillations.
#
# - **Temporal**: BDF2 (second-order backward differentiation) for all
#   steps, with u^{n-1} initialized to the exact solution at t = -dt.
#   Advection is handled semi-Lagrangian: instead of discretizing the
#   material derivative Du/Dt, we trace characteristics backward in time
#   to find "departure points" and evaluate the old velocity there.
#   This yields an unconditionally stable treatment of the nonlinear
#   convection term.
#
# - **Linear system**: Each time step solves a saddle-point system
#
#       [ A   B^T ] [ u ]   [ f ]
#       [ B   0   ] [ p ] = [ g ]
#
#   where A combines the viscous stiffness and the implicit mass term,
#   B is the discrete divergence, f is the advection + BC right-hand side,
#   and g enforces incompressibility (div u = 0).
#
# - **Boundary conditions**: Time-dependent exact Dirichlet BCs are
#   imposed on all boundary walls via projection (see Section 10 of
#   fem.MD for details).
#
# warp.fem API overview (for readers new to the library)
# -------------------------------------------------------
# Warp.fem provides a high-level finite element API on top of Warp's
# GPU kernel infrastructure.  The key abstractions are:
#
# - **Geometry** (e.g. ``Grid2D``, ``Trimesh2D``): the mesh.
# - **Function space** (``make_polynomial_space``): defines the shape
#   functions.  Each space has a set of *nodes* (DOF locations) and a
#   *degree*.
# - **Field** (``space.make_field()``): a vector of DOF values living
#   in a function space — represents a discrete FE function.
# - **Test / Trial** (``make_test``, ``make_trial``): lightweight wrappers
#   around a space used to indicate which role a field plays during
#   assembly.
# - **Domain** (``fem.Cells``, ``fem.BoundarySides``): where integrals
#   are evaluated — either over all cells or over boundary facets.
# - **Integrand** (``@fem.integrand``): a function evaluated at each
#   quadrature point.  Inside an integrand:
#     - ``s`` (``fem.Sample``) is the current quadrature point
#     - ``u(s)`` evaluates a field at that point
#     - ``fem.D(u, s)`` is the gradient ∇u
#     - ``fem.div(u, s)`` is the divergence ∇·u
#     - ``domain(s)`` returns the physical coordinates x of the point
#     - ``fem.lookup(domain, x, s)`` finds the sample closest to x,
#       used for semi-Lagrangian departure-point queries
# - **``fem.integrate``**: assembles an integrand over a domain.
#   Depending on the fields passed (test, trial, or neither), it returns
#   a matrix, vector, or scalar.
# - **``fem.interpolate``**: evaluates an integrand at the *nodes* of a
#   destination field (not at quadrature points).  For Lagrange elements
#   this gives exact nodal values — no mass-matrix inversion needed.
###########################################################################

import warp as wp
import warp.examples.fem.utils as fem_example_utils
import warp.fem as fem
from warp.fem.linalg import array_axpy
from warp.optim.linear import cg as cg_solve

PI = wp.constant(3.141592653589793)
TWO_PI = wp.constant(2.0 * 3.141592653589793)
FOUR_PI = wp.constant(4.0 * 3.141592653589793)


@wp.func
def periodic_wrap(pos: wp.vec2) -> wp.vec2:
    """Wrap coordinates back into [0, 1]^2 for the periodic Taylor-Green solution."""
    return wp.vec2(pos[0] - wp.floor(pos[0]), pos[1] - wp.floor(pos[1]))


@wp.func
def exact_velocity(x: wp.vec2, t: float, nu: float) -> wp.vec2:
    decay = wp.exp(-8.0 * PI * PI * nu * t)
    u = -wp.cos(TWO_PI * x[0]) * wp.sin(TWO_PI * x[1]) * decay
    v = wp.sin(TWO_PI * x[0]) * wp.cos(TWO_PI * x[1]) * decay
    return wp.vec2(u, v)


@wp.func
def exact_pressure(x: wp.vec2, t: float, nu: float) -> float:
    decay = wp.exp(-16.0 * PI * PI * nu * t)
    return -0.25 * (wp.cos(FOUR_PI * x[0]) + wp.cos(FOUR_PI * x[1])) * decay


###########################################################################
# Weak-form integrands
#
# Each @fem.integrand function is evaluated at every quadrature point
# during assembly.  The first argument ``s`` (a ``fem.Sample``) identifies
# the current quadrature point.  Arguments annotated ``fem.Field`` are
# FE functions that can be evaluated at ``s`` via ``u(s)`` (value),
# ``fem.D(u, s)`` (gradient), or ``fem.div(u, s)`` (divergence).
# Scalar parameters (``dt``, ``nu``, …) are passed at call time through
# the ``values=`` dictionary of ``fem.integrate``.
#
# When both a test and trial field are present, ``fem.integrate`` returns
# a sparse matrix whose (i,j) entry is the integral of the integrand
# over the domain with basis function j as trial and basis function i as
# test.  With only a test field, it returns a vector (linear form).
# With no test/trial fields, it returns a scalar.
###########################################################################


@fem.integrand
def mass_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    """L2 inner product  ∫ u · v  dx.

    Used both for the velocity mass matrix and for building the boundary
    projector (see Dirichlet BC section below).
    """
    return wp.dot(u(s), v(s))


# ---- LHS bilinear forms (implicit part of the time discretization) ----
#
# The implicit Euler (BDF1) semi-discrete momentum equation is:
#
#   (u^{n+1} - u^n_dep) / dt  -  nu * Laplacian(u^{n+1})  +  grad(p) = 0
#
# Multiplying by a test function v, integrating by parts, and using the
# symmetric strain rate (factor 2*nu with ddot on gradients), the bilinear
# form for the velocity block of the saddle-point matrix is:
#
#   a(u, v) = (1/dt) ∫ u·v dx  +  2*nu ∫ D(u):D(v) dx
#
# For BDF2 the mass coefficient changes from 1/dt to 3/(2*dt).


@fem.integrand
def viscosity_and_inertia_form(s: fem.Sample, u: fem.Field, v: fem.Field, dt: float, nu: float):
    """BDF1 velocity bilinear form:  (1/dt) (u, v) + 2 nu (D(u), D(v)).

    ``fem.D(u, s)`` is the Jacobian matrix ∂u_i/∂x_j at quadrature point s.
    ``wp.ddot`` is the Frobenius (component-wise) double contraction A:B.
    """
    return 2.0 * nu * wp.ddot(fem.D(u, s), fem.D(v, s)) + wp.dot(u(s), v(s)) / dt


@fem.integrand
def viscosity_and_inertia_form_bdf2(s: fem.Sample, u: fem.Field, v: fem.Field, dt: float, nu: float):
    """BDF2 velocity bilinear form:  (3/(2 dt)) (u, v) + 2 nu (D(u), D(v))."""
    return 2.0 * nu * wp.ddot(fem.D(u, s), fem.D(v, s)) + 1.5 * wp.dot(u(s), v(s)) / dt


# ---- RHS linear forms (semi-Lagrangian advection) ----------------------
#
# Semi-Lagrangian advection replaces the nonlinear convection term
# (u · grad)u with a characteristic-tracing step.  For each quadrature
# point x, we find the "departure point" X by tracing backward along the
# velocity field, then evaluate the old velocity u^n(X).
#
# The departure point is found with a predictor-corrector (midpoint) method:
#   1. Predict:  x_mid = x - 0.5 * dt * u(x)
#   2. Correct:  X     = x_mid - 0.5 * dt * u(x_mid)
#
# ``fem.lookup(domain, pos, s)`` returns a new ``fem.Sample`` at position
# ``pos``, allowing evaluation of any field there.  The last argument ``s``
# is a hint for the spatial search (speeds up the BVH query).
#
# ``periodic_wrap`` maps the backtracked position back into [0,1]^2 to
# respect the periodicity of the Taylor-Green solution.
#
# BDF1 RHS:  f = (1/dt) ∫ u^n(X) · v dx
# BDF2 RHS:  f = (1/dt) ∫ (2 u^n(X_1) - 0.5 u^{n-1}(X_2)) · v dx
# where X_1 is the departure point at distance dt and X_2 at distance 2*dt.


@fem.integrand
def transported_inertia_form(s: fem.Sample, domain: fem.Domain, u: fem.Field, v: fem.Field, dt: float):
    """BDF1 semi-Lagrangian RHS:  (1/dt) ∫ u^n(X) · v dx."""
    pos = domain(s)
    vel = u(s)

    # Predictor-corrector backtracking (midpoint rule, one dt step)
    conv_pos = periodic_wrap(pos - 0.5 * vel * dt)
    conv_s = fem.lookup(domain, conv_pos, s)
    conv_vel = u(conv_s)

    conv_pos = periodic_wrap(conv_pos - 0.5 * conv_vel * dt)
    u_dep = u(fem.lookup(domain, conv_pos, conv_s))

    return wp.dot(u_dep, v(s)) / dt


@fem.integrand
def transported_inertia_form_bdf2(
    s: fem.Sample, domain: fem.Domain, u_cur: fem.Field, u_prev: fem.Field, v: fem.Field, dt: float
):
    """BDF2 semi-Lagrangian RHS:  (1/dt) ∫ (2 u^n(X_1) - 1/2 u^{n-1}(X_2)) · v dx.

    Two departure points are computed:
    - X_1: backtrack by dt   using u^n  →  evaluate u^n(X_1)
    - X_2: backtrack by 2*dt using u^n  →  evaluate u^{n-1}(X_2)
    """
    pos = domain(s)
    vel = u_cur(s)

    # Backtrack by dt to get u^n at departure point X_1
    mid1 = periodic_wrap(pos - 0.5 * vel * dt)
    mid1_s = fem.lookup(domain, mid1, s)
    mid1 = periodic_wrap(mid1 - 0.5 * u_cur(mid1_s) * dt)
    u_n_dep = u_cur(fem.lookup(domain, mid1, mid1_s))

    # Backtrack by 2*dt to get u^{n-1} at departure point X_2
    mid2 = periodic_wrap(pos - vel * dt)
    mid2_s = fem.lookup(domain, mid2, s)
    mid2 = periodic_wrap(mid2 - u_cur(mid2_s) * dt)
    u_nm1_dep = u_prev(fem.lookup(domain, mid2, mid2_s))

    return wp.dot(2.0 * u_n_dep - 0.5 * u_nm1_dep, v(s)) / dt


# ---- Divergence / pressure coupling ------------------------------------
#
# The incompressibility constraint  div(u) = 0  is enforced weakly:
#
#   b(u, q) = -∫ q div(u) dx
#
# When assembled with a velocity trial and a pressure test, this gives
# the B matrix of the saddle-point system.


@fem.integrand
def div_form(s: fem.Sample, u: fem.Field, q: fem.Field):
    """Discrete divergence operator:  -∫ q div(u) dx."""
    return -q(s) * fem.div(u, s)


# ---- Boundary and initial condition integrands --------------------------


@fem.integrand
def velocity_boundary_integrand(s: fem.Sample, domain: fem.Domain, v: fem.Field, t: float, nu: float):
    """Weighted boundary integral of exact velocity:  ∫_∂Ω u_exact · v ds.

    This is only used once at t=0 to provide an initial value for
    ``normalize_dirichlet_projector``.  For subsequent time steps we use
    ``fem.interpolate`` instead (see the ``step`` method).
    """
    x = domain(s)
    return wp.dot(exact_velocity(x, t, nu), v(s))


@fem.integrand
def initial_velocity(s: fem.Sample, domain: fem.Domain, nu: float):
    """Exact velocity at t=0, used with ``fem.interpolate`` to set ICs."""
    return exact_velocity(domain(s), 0.0, nu)


# ---- Error measurement integrands --------------------------------------


@fem.integrand
def velocity_error(s: fem.Sample, domain: fem.Domain, u: fem.Field, t: float, nu: float):
    """Pointwise squared velocity error |u_h - u_exact|^2 (integrated to get L2 norm)."""
    x = domain(s)
    diff = u(s) - exact_velocity(x, t, nu)
    return wp.dot(diff, diff)


@fem.integrand
def pressure_mean_integrand(s: fem.Sample, p: fem.Field):
    """∫ p dx — used to compute the mean pressure for centering."""
    return p(s)


@fem.integrand
def pressure_error(s: fem.Sample, domain: fem.Domain, p: fem.Field, t: float, nu: float, p_mean: float):
    """Pointwise squared pressure error (p_h - mean - p_exact)^2.

    Since the pressure in an incompressible flow is determined only up to
    a constant, we subtract its mean before comparing.
    """
    x = domain(s)
    diff = (p(s) - p_mean) - exact_pressure(x, t, nu)
    return diff * diff


@fem.integrand
def exact_velocity_at_time(s: fem.Sample, domain: fem.Domain, t: float, nu: float):
    """Exact velocity at arbitrary time t — used with ``fem.interpolate``."""
    return exact_velocity(domain(s), t, nu)


@fem.integrand
def exact_pressure_at_time(s: fem.Sample, domain: fem.Domain, t: float, nu: float):
    """Exact pressure at arbitrary time t — used with ``fem.interpolate``."""
    return exact_pressure(domain(s), t, nu)


# ---- Passive dye advection (visualization only) -------------------------


@fem.integrand
def initial_dye(s: fem.Sample, domain: fem.Domain):
    """Stripe pattern that will be deformed by the vortex flow."""
    x = domain(s)
    return wp.sin(FOUR_PI * x[0])


@fem.integrand
def advect_dye(s: fem.Sample, domain: fem.Domain, dye: fem.Field, u: fem.Field, dt: float):
    """Semi-Lagrangian advection of a passive scalar dye field.

    Same predictor-corrector backtracking as the velocity advection,
    but here we evaluate the dye field (not velocity) at the departure
    point.  Used with ``fem.interpolate`` (not ``fem.integrate``), so
    the result is written directly into the destination field's DOFs.
    """
    pos = domain(s)
    vel = u(s)

    conv_pos = periodic_wrap(pos - 0.5 * vel * dt)
    conv_s = fem.lookup(domain, conv_pos, s)
    conv_vel = u(conv_s)

    conv_pos = periodic_wrap(conv_pos - 0.5 * conv_vel * dt)
    return dye(fem.lookup(domain, conv_pos, conv_s))


class Example:
    """Taylor-Green vortex solver with analytical error tracking.

    The constructor assembles all constant matrices (viscosity, mass,
    divergence, boundary projector), pre-computes time-dependent boundary
    conditions for every frame, and captures the simulation step as a CUDA
    graph for reduced launch overhead.

    Args:
        quiet: Suppress solver residual output and skip error computation.
        degree: Polynomial degree for velocity (pressure is degree-1).
        resolution: Number of cells per spatial direction.
        Re: Reynolds number (viscosity = 1/Re).
        mesh: Mesh type — ``"grid"`` (structured quads), ``"tri"``, or ``"quad"``
            (unstructured).
    """

    def __init__(self, quiet=False, degree=2, resolution=25, Re=100.0, mesh: str = "grid"):
        self._quiet = quiet

        # Time step size: dt = h so that CFL ~ 1 for the decaying vortex.
        self.sim_dt = 1.0 / resolution
        self.current_frame = 0

        viscosity = 1.0 / Re

        # -----------------------------------------------------------------
        # 1. Mesh / geometry
        # -----------------------------------------------------------------
        # Grid2D creates a structured quad mesh on [0,1]^2 with the given
        # resolution.  Trimesh2D / Quadmesh2D create unstructured meshes.
        # ``build_bvh=True`` builds a bounding-volume hierarchy needed by
        # ``fem.lookup`` for the semi-Lagrangian departure-point queries.
        if mesh == "tri":
            positions, tri_vidx = fem_example_utils.gen_trimesh(res=wp.vec2i(resolution))
            geo = fem.Trimesh2D(tri_vertex_indices=tri_vidx, positions=positions, build_bvh=True)
        elif mesh == "quad":
            positions, quad_vidx = fem_example_utils.gen_quadmesh(res=wp.vec2i(resolution))
            geo = fem.Quadmesh2D(quad_vertex_indices=quad_vidx, positions=positions, build_bvh=True)
        else:
            geo = fem.Grid2D(res=wp.vec2i(resolution))

        # Integration domains: all cells (for volume integrals) and boundary
        # edges (for Dirichlet BC enforcement).
        domain = fem.Cells(geometry=geo)
        boundary = fem.BoundarySides(geo)

        # -----------------------------------------------------------------
        # 2. Function spaces (Taylor-Hood)
        # -----------------------------------------------------------------
        # Velocity: vector-valued (wp.vec2) polynomial space of the given
        # degree (default Q2 — biquadratic on quads).
        # Pressure: scalar polynomial space one degree lower (default Q1).
        u_space = fem.make_polynomial_space(geo, degree=degree, dtype=wp.vec2)
        p_space = fem.make_polynomial_space(geo, degree=degree - 1)

        # -----------------------------------------------------------------
        # 3. Assemble constant matrices
        # -----------------------------------------------------------------
        # Test and trial functions are lightweight wrappers that tell
        # ``fem.integrate`` which field plays which role.  Passing both a
        # trial and a test produces a sparse matrix (bilinear form).

        u_test = fem.make_test(space=u_space, domain=domain)
        u_trial = fem.make_trial(space=u_space, domain=domain)

        # Velocity block A = 3/(2 dt)*M + 2*nu*K  (BDF2)
        # M = mass matrix, K = stiffness matrix (viscous term).
        u_matrix = fem.integrate(
            viscosity_and_inertia_form_bdf2,
            fields={"u": u_trial, "v": u_test},
            values={"nu": viscosity, "dt": self.sim_dt},
        )

        # Divergence matrix B (pressure-velocity coupling).
        # Passing a velocity trial and a pressure test gives the rectangular
        # matrix B such that B @ u approximates -∫ q div(u) dx.
        p_test = fem.make_test(space=p_space, domain=domain)
        div_matrix = fem.integrate(div_form, fields={"u": u_trial, "q": p_test})

        # -----------------------------------------------------------------
        # 4. Dirichlet boundary conditions (projection method)
        # -----------------------------------------------------------------
        # We enforce u = u_exact on ∂Ω by modifying the linear system:
        #   A_proj = (I - P) A (I - P) + P
        #   rhs    = (I - P)(f - A * u_bc) + u_bc
        # where P is a projector that is identity on boundary DOFs and zero
        # on interior DOFs.
        #
        # Building the projector:
        #   1. Assemble the boundary mass matrix with ``assembly="nodal"``
        #      (row-sum lumping) — this gives a diagonal matrix whose
        #      nonzero entries correspond to boundary DOFs.
        #   2. ``normalize_dirichlet_projector`` scales it so the diagonal
        #      entries become exactly 1, making it an idempotent projector.
        #
        # IMPORTANT: ``normalize_dirichlet_projector`` also divides the
        # companion ``fixed_value`` vector by the raw diagonal, converting
        # mass-weighted integrals into nodal values.  It must be called
        # exactly once per projector — calling it again on an already-
        # normalized projector is a no-op on the values (see fem.MD §17.11).

        u_bd_test = fem.make_test(space=u_space, domain=boundary)
        u_bd_trial = fem.make_trial(space=u_space, domain=boundary)
        u_bd_projector = fem.integrate(mass_form, fields={"u": u_bd_trial, "v": u_bd_test}, assembly="nodal")

        u_bd_value_init = fem.integrate(
            velocity_boundary_integrand,
            fields={"v": u_bd_test},
            values={"t": 0.0, "nu": viscosity},
            assembly="nodal",
            output_dtype=wp.vec2d,
        )
        fem.normalize_dirichlet_projector(u_bd_projector, u_bd_value_init)

        # Save copies of the un-projected matrices — we need the originals
        # to pre-compute the BC contribution to the RHS for each time step.
        self._u_matrix_orig = wp.sparse.bsr_copy(u_matrix)
        self._div_matrix_orig = wp.sparse.bsr_copy(div_matrix)

        # Project the velocity matrix in-place.
        u_bd_rhs = wp.zeros_like(u_bd_value_init)
        fem.project_linear_system(u_matrix, u_bd_rhs, u_bd_projector, u_bd_value_init, normalize_projector=False)

        # Project the divergence block: B_proj = B (I - P), so that B
        # doesn't "see" the boundary DOFs (they're fixed, not unknowns).
        div_matrix -= div_matrix @ u_bd_projector

        # -----------------------------------------------------------------
        # 5. Build saddle-point system
        # -----------------------------------------------------------------
        # SaddleSystem wraps the projected [A, B^T; B, 0] block structure
        # and provides a CG-based Uzawa solver.
        self._saddle_system = fem_example_utils.SaddleSystem(u_matrix, div_matrix)

        # Save references needed during time stepping
        self._domain = domain
        self._u_bd_projector = u_bd_projector
        self._u_test = u_test
        self._viscosity = viscosity

        # -----------------------------------------------------------------
        # 6. Solution fields and initial conditions
        # -----------------------------------------------------------------
        # ``space.make_field()`` allocates a DOF vector and returns a Field
        # object.  The DOF values live in ``field.dof_values`` (a wp.array).
        self._u_field = u_space.make_field()
        self._u_field_prev = u_space.make_field()  # u^{n-1} for BDF2
        self._p_field = p_space.make_field()

        # Scratch field for computing time-dependent boundary values.
        self._u_bc_field = u_space.make_field()

        # Set u^0 = u_exact(t=0) and u^{-1} = u_exact(t=-dt) so that BDF2
        # can be used from the very first time step (it needs two levels).
        fem.interpolate(initial_velocity, dest=self._u_field, values={"nu": viscosity})
        fem.interpolate(exact_velocity_at_time, dest=self._u_field_prev, values={"t": -self.sim_dt, "nu": viscosity})

        # Passively advected dye field for visualization (same degree as velocity).
        dye_space = fem.make_polynomial_space(geo, degree=degree)
        self._dye_field = dye_space.make_field()
        self._dye_tmp = dye_space.make_field()
        fem.interpolate(initial_dye, dest=self._dye_field)

        # Error tracking
        self._velocity_errors = []
        self._pressure_errors = []
        self._times = []

        # Fields for plotting the exact solution at the final time
        self._exact_u_field = u_space.make_field()
        self._exact_p_field = p_space.make_field()

        # Pre-allocate scratch arrays used every time step
        n_u = self._u_field.space.node_count()
        n_p = self._p_field.space.node_count()
        self._u_bd_value = wp.zeros(n_u, dtype=wp.vec2d)
        self._x_u = wp.zeros(n_u, dtype=wp.vec2d)
        self._x_p = wp.zeros(n_p, dtype=wp.float64)
        self._u_rhs = wp.zeros(n_u, dtype=wp.vec2d)
        self._u_bd_rhs_buf = wp.zeros(n_u, dtype=wp.vec2d)
        self._p_rhs = wp.zeros(n_p, dtype=wp.float64)

        saddle = self._saddle_system
        self._saddle_x = wp.empty(dtype=saddle.scalar_type, shape=saddle.shape[0], device=saddle.device)
        self._saddle_b = wp.empty_like(self._saddle_x)

        self.renderer = fem_example_utils.Plot()
        self.renderer.add_field("dye", self._dye_field)

        # -----------------------------------------------------------------
        # 7. CUDA graph capture
        # -----------------------------------------------------------------
        # With BDF2 used from frame 1, simulate() is frame-independent.
        # We capture it as a CUDA graph and replay it each step for
        # reduced launch overhead.  Before each replay, step() computes
        # the time-dependent BC contributions into the working buffers
        # (_u_bd_rhs_buf, _p_rhs) that the graph references.
        self.use_cuda_graph = wp.get_device().is_cuda
        if self.use_cuda_graph:
            import gc  # noqa: PLC0415

            gc.disable()
            try:
                with wp.ScopedCapture() as capture:
                    self.simulate()
                self.graph = capture.graph
            finally:
                gc.enable()

    def _compute_bc(self, t):
        """Compute time-dependent Dirichlet BC contributions at time ``t``.

        Fills ``_u_bd_rhs_buf`` and ``_p_rhs`` with the velocity and
        pressure RHS contributions from the boundary conditions.  This is
        called by ``step()`` before each graph replay (or ``simulate()``
        call) so that the captured graph sees the correct BC data.
        """
        fem.interpolate(exact_velocity_at_time, dest=self._u_bc_field, values={"t": t, "nu": self._viscosity})
        self._u_bd_value.zero_()
        wp.utils.array_cast(out_array=self._u_bd_value, in_array=self._u_bc_field.dof_values)
        wp.sparse.bsr_mv(self._u_bd_projector, x=self._u_bd_value, y=self._u_bd_value, alpha=1.0, beta=0.0)

        wp.sparse.bsr_mv(self._u_matrix_orig, x=self._u_bd_value, y=self._u_bd_rhs_buf, alpha=1.0, beta=0.0)
        wp.sparse.bsr_mv(self._u_bd_projector, x=self._u_bd_rhs_buf, y=self._u_bd_rhs_buf, alpha=1.0, beta=-1.0)
        array_axpy(x=self._u_bd_value, y=self._u_bd_rhs_buf, alpha=1.0, beta=1.0)

        wp.sparse.bsr_mv(self._div_matrix_orig, x=self._u_bd_value, y=self._p_rhs, alpha=-1.0, beta=0.0)

    def simulate(self):
        """Run one time step of the Navier-Stokes solver.

        This method is CUDA-graph-capturable: it contains no time-dependent
        parameters and no host-side branches.  Time-dependent BC
        contributions must be loaded into ``_u_bd_rhs_buf`` and ``_p_rhs``
        by ``_compute_bc()`` before each call (or graph replay).

        The iterative solver uses ``check_every=0`` so that it runs a fixed
        number of iterations with device-side convergence checks, making it
        compatible with outer graph capture.
        """
        # --- Step 1: semi-Lagrangian advection RHS (BDF2) ----------------
        #
        # Assemble the linear form that represents the explicit part of the
        # time discretization — the old velocity evaluated at departure
        # points.  Passing only a test field (no trial) makes
        # ``fem.integrate`` return a vector.
        fem.integrate(
            transported_inertia_form_bdf2,
            fields={"u_cur": self._u_field, "u_prev": self._u_field_prev, "v": self._u_test},
            values={"dt": self.sim_dt},
            output_dtype=wp.vec2d,
            output=self._u_rhs,
        )

        # --- Step 2: combine advection RHS with pre-loaded BC contribution
        #
        # u_rhs = (I - P) * u_rhs + u_bd_rhs
        #       = (I - P) * f  -  (I - P) * A * u_bc  +  u_bc
        wp.sparse.bsr_mv(self._u_bd_projector, x=self._u_rhs, y=self._u_rhs, alpha=-1.0, beta=1.0)
        array_axpy(x=self._u_bd_rhs_buf, y=self._u_rhs, alpha=1.0, beta=1.0)

        # --- Step 3: solve the saddle-point system -----------------------
        #
        # Use the previous solution as initial guess (warm start) for the
        # iterative solver.  ``array_cast`` converts between float32 (field
        # storage) and float64 (solver precision).
        #
        # We call the CG solver directly (bypassing bsr_solve_saddle /
        # bsr_cg) to avoid host-side .numpy() calls that would break
        # outer graph capture.  With check_every=0, the solver detects
        # that it is inside a graph capture and uses wp.capture_while
        # for its iteration loop.
        x_u = self._x_u
        x_p = self._x_p
        wp.utils.array_cast(out_array=x_u, in_array=self._u_field.dof_values)
        wp.utils.array_cast(out_array=x_p, in_array=self._p_field.dof_values)

        saddle = self._saddle_system
        wp.copy(src=x_u, dest=saddle.u_slice(self._saddle_x))
        wp.copy(src=x_p, dest=saddle.p_slice(self._saddle_x))
        wp.copy(src=self._u_rhs, dest=saddle.u_slice(self._saddle_b))
        wp.copy(src=self._p_rhs, dest=saddle.p_slice(self._saddle_b))

        cg_solve(
            A=saddle,
            b=self._saddle_b,
            x=self._saddle_x,
            tol=1.0e-6,
            check_every=0,
            use_cuda_graph=True,
            M=saddle.preconditioner,
        )

        wp.copy(dest=x_u, src=saddle.u_slice(self._saddle_x))
        wp.copy(dest=x_p, src=saddle.p_slice(self._saddle_x))

        # Save u^n before overwriting (BDF2 needs both u^n and u^{n-1}).
        wp.copy(src=self._u_field.dof_values, dest=self._u_field_prev.dof_values)

        # Copy solver output back into the solution fields.
        wp.utils.array_cast(in_array=x_u, out_array=self._u_field.dof_values)
        wp.utils.array_cast(in_array=x_p, out_array=self._p_field.dof_values)

        # --- Step 4: advect dye ------------------------------------------
        #
        # The dye is a passive scalar with no feedback on the flow.
        # ``fem.interpolate`` with the ``advect_dye`` integrand evaluates
        # the dye at backtracked positions and writes the result into
        # ``_dye_tmp``.  We copy back instead of swapping so that the
        # graph always reads from / writes to the same memory addresses.
        fem.interpolate(
            advect_dye,
            dest=self._dye_tmp,
            fields={"dye": self._dye_field, "u": self._u_field},
            values={"dt": self.sim_dt},
        )
        wp.copy(src=self._dye_tmp.dof_values, dest=self._dye_field.dof_values)

    def step(self):
        """Advance the solution by one time step.

        Loads pre-computed BC data for the current frame, then either
        replays the captured CUDA graph or calls ``simulate()`` directly.
        Optionally computes L2 errors against the exact solution.
        """
        self.current_frame += 1
        t = self.current_frame * self.sim_dt

        # Compute time-dependent BC contributions for this frame.
        self._compute_bc(t)

        if self.use_cuda_graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        # --- Error measurement (host-side, outside graph) ----------------
        #
        # Compute L2 norms of velocity and pressure errors against the
        # exact Taylor-Green solution.  The pressure is only determined up
        # to a constant, so we subtract its mean before comparing.
        # ``fem.integrate`` with no test/trial fields returns a scalar.
        if not self._quiet:
            l2_vel_err_sq = fem.integrate(
                velocity_error,
                domain=self._domain,
                fields={"u": self._u_field},
                values={"t": t, "nu": self._viscosity},
            )

            p_mean = fem.integrate(
                pressure_mean_integrand,
                domain=self._domain,
                fields={"p": self._p_field},
            )
            l2_p_err_sq = fem.integrate(
                pressure_error,
                domain=self._domain,
                fields={"p": self._p_field},
                values={"t": t, "nu": self._viscosity, "p_mean": p_mean},
            )

            vel_err = l2_vel_err_sq**0.5
            p_err = l2_p_err_sq**0.5

            self._times.append(t)
            self._velocity_errors.append(vel_err)
            self._pressure_errors.append(p_err)

            return vel_err, p_err

        return None, None

    def render(self):
        self.renderer.begin_frame(time=self.current_frame * self.sim_dt)
        self.renderer.add_field("dye", self._dye_field)
        self.renderer.end_frame()

    def plot_comparison(self, save=None):
        """Plot L2 error history and side-by-side field comparison at final time.

        Produces a 2x3 figure:
          - (0,0) Error decay over time (log scale)
          - (0,1) Numerical velocity magnitude
          - (0,2) Exact velocity magnitude
          - (1,0) Pointwise velocity error
          - (1,1) Numerical pressure (mean-centered)
          - (1,2) Exact pressure

        Args:
            save: If not ``None``, save the figure to this path instead of
                displaying it interactively.
        """
        import matplotlib.pyplot as plt  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
        from matplotlib.tri import Triangulation  # noqa: PLC0415

        if not self._times:
            return

        t_final = self._times[-1]

        # Interpolate exact solutions at final time
        fem.interpolate(exact_velocity_at_time, dest=self._exact_u_field, values={"t": t_final, "nu": self._viscosity})
        fem.interpolate(exact_pressure_at_time, dest=self._exact_p_field, values={"t": t_final, "nu": self._viscosity})

        u_num = self._u_field.dof_values.numpy()
        u_exact = self._exact_u_field.dof_values.numpy()
        p_num = self._p_field.dof_values.numpy()
        p_exact = self._exact_p_field.dof_values.numpy()

        # Center numerical pressure (analytical pressure has zero mean)
        p_num_centered = p_num - np.mean(p_num)

        u_num_mag = np.linalg.norm(u_num, axis=1)
        u_exact_mag = np.linalg.norm(u_exact, axis=1)

        # Triangulations for contour plots
        u_pos = self._u_field.space.node_positions().numpy()
        u_tri = Triangulation(u_pos[:, 0], u_pos[:, 1], self._u_field.space.node_triangulation())
        p_pos = self._p_field.space.node_positions().numpy()
        p_tri = Triangulation(p_pos[:, 0], p_pos[:, 1], self._p_field.space.node_triangulation())

        times = np.array(self._times)

        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        fig.suptitle(f"Taylor-Green Vortex: Comparison with Analytical Solution (t = {t_final:.2f})")

        # --- Top row: error decay + velocity comparison ---

        ax = axes[0, 0]
        ax.semilogy(times, self._velocity_errors, "b-o", markersize=2, label="Velocity")
        ax.semilogy(times, self._pressure_errors, "r-s", markersize=2, label="Pressure")
        ax.set_xlabel("Time")
        ax.set_ylabel("L2 Error")
        ax.set_title("Error Decay")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Use independent color scales so each field shows its full dynamic range
        ax = axes[0, 1]
        cs = ax.tricontourf(u_tri, u_num_mag, levels=20)
        ax.set_title("Velocity |u| (Numerical)")
        ax.set_aspect("equal")
        fig.colorbar(cs, ax=ax)

        ax = axes[0, 2]
        cs = ax.tricontourf(u_tri, u_exact_mag, levels=20)
        ax.set_title("Velocity |u| (Exact)")
        ax.set_aspect("equal")
        fig.colorbar(cs, ax=ax)

        # --- Bottom row: velocity error + pressure comparison ---

        u_err_mag = np.linalg.norm(u_num - u_exact, axis=1)
        ax = axes[1, 0]
        cs = ax.tricontourf(u_tri, u_err_mag, levels=20)
        ax.set_title("Velocity Error |u - u_exact|")
        ax.set_aspect("equal")
        fig.colorbar(cs, ax=ax)

        ax = axes[1, 1]
        cs = ax.tricontourf(p_tri, p_num_centered, levels=20)
        ax.set_title("Pressure (Numerical, centered)")
        ax.set_aspect("equal")
        fig.colorbar(cs, ax=ax)

        ax = axes[1, 2]
        cs = ax.tricontourf(p_tri, p_exact, levels=20)
        ax.set_title("Pressure (Exact)")
        ax.set_aspect("equal")
        fig.colorbar(cs, ax=ax)

        plt.tight_layout()

        if save is not None:
            fig.savefig(save)
        else:
            plt.show()


if __name__ == "__main__":
    import argparse
    import os

    wp.set_module_options({"enable_backward": False})

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--resolution", type=int, default=64, help="Grid resolution.")
    parser.add_argument("--degree", type=int, default=2, help="Polynomial degree of shape functions.")
    parser.add_argument("--num-frames", type=int, default=200, help="Total number of frames.")
    parser.add_argument("--Re", type=float, default=250.0, help="Reynolds number.")
    parser.add_argument("--mesh", choices=("grid", "tri", "quad"), default="grid", help="Mesh type.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode, suppressing the opening of any graphical windows.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppresses the printing out of iteration residuals.")
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
            degree=args.degree,
            resolution=args.resolution,
            Re=args.Re,
            mesh=args.mesh,
        )

        for _k, set_info in fem_example_utils.progress_bar(args.num_frames, quiet=args.quiet):
            vel_err, p_err = example.step()
            if vel_err is not None:
                set_info("vel_err", f"{vel_err:.4e}")
                set_info("p_err", f"{p_err:.4e}")
            example.render()

        if not args.headless or args.export:
            # Derive a companion path for the static comparison figure
            comparison_save = None
            if args.export:
                base, ext = os.path.splitext(args.export)
                if ext:
                    comparison_save = base + "_comparison.png"
                else:
                    comparison_save = args.export.rstrip("/").rstrip(os.sep) + "_comparison.png"

            if not args.quiet:
                example.plot_comparison(save=comparison_save)
            example.renderer.plot(
                options={
                    "dye": {"contours": {"levels": 20}},
                },
                save=args.export,
            )
