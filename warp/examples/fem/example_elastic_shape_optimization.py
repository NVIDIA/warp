# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example: Shape Optimization of a 2D Elastic Bridge
#
# This example demonstrates shape optimization of a 2D elastic bridge using
# finite element analysis and gradient-based optimization.
#
# Both left and right edges are fixed supports. A downward load is applied
# on the bottom edge. The optimizer discovers an arch-like form that
# minimizes stress.
#
# Shape Optimization Strategy:
# - The goal is to optimize the shape of the structure to minimize the total squared norm of
#   the stress field over the domain.
# - The positions of the left and right boundary vertices are fixed throughout the optimization
#   to maintain the structure's support and loading conditions.
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
    left: wp.array[int],
    right: wp.array[int],
    bottom: wp.array[int],
):
    nor = fem.normal(domain, s)

    if nor[0] < -0.5:
        left[s.qp_index] = 1
    elif nor[0] > 0.5:
        right[s.qp_index] = 1
    elif nor[1] < -0.5:
        bottom[s.qp_index] = 1


@wp.func
def hooke_stress(strain: wp.mat22, lame: wp.vec2):
    """Hookean elasticity"""
    return 2.0 * lame[1] * strain + lame[0] * wp.trace(strain) * wp.identity(n=2, dtype=float)


@fem.integrand
def stress_field(s: fem.Sample, u: fem.Field, lame: wp.vec2):
    return hooke_stress(fem.D(u, s), lame)


@fem.integrand
def stress_norm_field(s: fem.Sample, u: fem.Field, lame: wp.vec2):
    stress = stress_field(s, u, lame)
    return wp.sqrt(wp.ddot(stress, stress))


@fem.integrand
def hooke_elasticity_form(s: fem.Sample, u: fem.Field, v: fem.Field, lame: wp.vec2):
    return wp.ddot(fem.D(v, s), stress_field(s, u, lame))


@fem.integrand
def normalized_load_form(s: fem.Sample, domain: fem.Domain, v: fem.Field, load: wp.vec2, inv_length: wp.array[float]):
    return wp.dot(v(s), load) * inv_length[0]


@fem.integrand
def loss_form(
    s: fem.Sample,
    domain: fem.Domain,
    u: fem.Field,
    lame: wp.vec2,
    quality_weight: float,
):
    stress = stress_field(s, u, lame)
    stress_norm_sq = wp.ddot(stress, stress)

    # As we're not remeshing, add a "quality" term
    # to avoid degenerate and inverted elements.
    # Uses smooth matrix invariants (det and Frobenius norm) instead of SVD,
    # as the SVD backward pass is numerically unstable for near-degenerate elements.
    F = fem.deformation_gradient(domain, s)
    det_F = wp.determinant(F)
    frob_sq = wp.ddot(F, F)  # ||F||^2_F = s0^2 + s1^2

    # quality = 2*det(F) / ||F||^2_F is the ratio of geometric to arithmetic mean
    # of squared singular values: equals 1 for isotropic elements, 0 for degenerate.
    quality_threshold = 0.5
    quality = 2.0 * det_F / (frob_sq + 1.0e-8) / quality_threshold
    quality_pen = -wp.log(wp.max(quality, 1.0e-4)) * wp.min(0.0, quality - 1.0) * wp.min(0.0, quality - 1.0)

    return stress_norm_sq + quality_pen * quality_weight


@fem.integrand
def volume_form():
    return 1.0


@wp.kernel
def invert_scalar(input: wp.array[float], output: wp.array[float]):
    output[0] = 1.0 / input[0]


@fem.integrand
def symmetrize_field(s: fem.Sample, domain: fem.Domain, field: fem.Field):
    """Average field value with its x-mirrored counterpart, flipping x."""
    pos = domain(s)

    mirror_pos = wp.vec2(1.0 - pos[0], pos[1])
    mirror_val = field(fem.lookup(domain, mirror_pos, s))
    val = field(s)
    return wp.vec2(
        (val[0] - mirror_val[0]) * 0.5,
        (val[1] + mirror_val[1]) * 0.5,
    )


@wp.kernel
def add_volume_loss(loss: wp.array[wp.float32], vol: wp.array[wp.float32], target_vol: wp.float32, weight: wp.float32):
    loss[0] += weight * (vol[0] - target_vol) * (vol[0] - target_vol)


def delaunay_edge_flip(positions_np, tri_indices_np, ref_positions_np=None):
    """Flip interior edges to restore the Delaunay property.

    Args:
        positions_np: (N, 2) array of vertex positions.
        tri_indices_np: (M, 3) array of triangle vertex indices (modified in-place).
        ref_positions_np: Optional (N, 2) array of reference vertex positions.
            When provided, flips that would create degenerate triangles on the
            reference configuration are rejected.

    Returns:
        Tuple of (tri_indices_np, total_num_flips).
    """

    def _in_circumcircle(ax, ay, bx, by, cx, cy, dx, dy):
        """Return True if point d lies strictly inside the circumcircle of CCW triangle (a, b, c)."""
        adx, ady = ax - dx, ay - dy
        bdx, bdy = bx - dx, by - dy
        cdx, cdy = cx - dx, cy - dy
        det = adx * (bdy * (cdx * cdx + cdy * cdy) - cdy * (bdx * bdx + bdy * bdy))
        det -= ady * (bdx * (cdx * cdx + cdy * cdy) - cdx * (bdx * bdx + bdy * bdy))
        det += (adx * adx + ady * ady) * (bdx * cdy - bdy * cdx)
        return det > 0.0

    def _signed_area(ax, ay, bx, by, cx, cy):
        return 0.5 * ((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))

    total_flips = 0

    while True:
        # Build edge adjacency: edge -> list of triangle indices
        edge_to_tris = {}
        for ti in range(len(tri_indices_np)):
            v = tri_indices_np[ti]
            for j in range(3):
                a, b = int(v[j]), int(v[(j + 1) % 3])
                edge = (min(a, b), max(a, b))
                if edge not in edge_to_tris:
                    edge_to_tris[edge] = []
                edge_to_tris[edge].append(ti)

        pass_flips = 0
        for (a, b), tris in edge_to_tris.items():
            if len(tris) != 2:
                continue
            t0, t1 = tris

            # Find opposite vertices
            v0 = tri_indices_np[t0]
            v1 = tri_indices_np[t1]
            c = int(next(x for x in v0 if x != a and x != b))
            d = int(next(x for x in v1 if x != a and x != b))

            ax, ay = positions_np[a]
            bx, by = positions_np[b]
            cx, cy = positions_np[c]
            dx, dy = positions_np[d]

            # Check convexity: both new triangles must have positive area
            if _signed_area(ax, ay, dx, dy, cx, cy) <= 0.0:
                continue
            if _signed_area(bx, by, cx, cy, dx, dy) <= 0.0:
                continue

            # Circumcircle test: flip if d is inside circumcircle of (a, b, c)
            # Swap coordinates (not vertex indices) to ensure CCW for the test
            test_ax, test_ay, test_bx, test_by = ax, ay, bx, by
            if _signed_area(ax, ay, bx, by, cx, cy) < 0.0:
                test_ax, test_ay, test_bx, test_by = bx, by, ax, ay

            if not _in_circumcircle(test_ax, test_ay, test_bx, test_by, cx, cy, dx, dy):
                continue

            # Reject flips that would create degenerate triangles on the reference mesh
            if ref_positions_np is not None:
                rax, ray = ref_positions_np[a]
                rbx, rby = ref_positions_np[b]
                rcx, rcy = ref_positions_np[c]
                rdx, rdy = ref_positions_np[d]
                if abs(_signed_area(rax, ray, rdx, rdy, rcx, rcy)) <= 1.0e-10:
                    continue
                if abs(_signed_area(rbx, rby, rcx, rcy, rdx, rdy)) <= 1.0e-10:
                    continue

            # Flip: replace edge (a,b) with edge (c,d)
            # Use original a,b — convexity check verified (a,d,c) and (b,c,d) are CCW
            tri_indices_np[t0] = [a, d, c]
            tri_indices_np[t1] = [b, c, d]
            pass_flips += 1

        total_flips += pass_flips
        if pass_flips == 0:
            break

    return tri_indices_np, total_flips


class Example:
    def __init__(
        self,
        quiet=False,
        degree=2,
        resolution=25,
        mesh="tri",
        poisson_ratio=0.45,
        load=(0.0, -0.2),
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

        positions, tri_vidx = fem_example_utils.gen_trimesh(
            res=wp.vec2i(resolution, resolution // 5), bounds_lo=bounds_lo, bounds_hi=bounds_hi
        )
        self._initial_positions = wp.clone(positions)
        self._tri_vertex_indices = tri_vidx
        self._start_geo = fem.Trimesh2D(tri_vertex_indices=tri_vidx, positions=self._initial_positions, build_bvh=True)
        self._use_deformed_geo = mesh == "deformed"

        if self._use_deformed_geo:
            # Use a DeformedGeometry: a degree-1 position field over the fixed
            # reference Trimesh2D defines the actual vertex positions.
            self._vertex_positions = wp.clone(positions)
            self._build_deformed_geo()
        else:
            # Direct approach: optimize the Trimesh2D vertex positions in-place.
            self._vertex_positions = positions

        # make sure positions are differentiable
        self._vertex_positions.requires_grad = True

        self._degree = degree

        # initialize renderer
        self.renderer = fem_example_utils.Plot()

        # Initialize Adam optimizer
        # Current implementation assumes scalar arrays, so cast our vec2 arrays to scalars
        self._vertex_positions_scalar = wp.array(self._vertex_positions, dtype=wp.float32).flatten()
        self._vertex_positions_scalar.grad = wp.array(self._vertex_positions.grad, dtype=wp.float32).flatten()
        self.optimizer = Adam([self._vertex_positions_scalar], lr=lr)

        self._rebuild_fem_structures(degree)

        # Store initial node positions (for rendering)
        start_space = fem.make_polynomial_space(self._start_geo, degree=degree, dtype=wp.vec2)
        self._start_node_positions = start_space.node_positions()

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

    def _build_deformed_geo(self):
        """(Re)build the deformation field and DeformedGeometry.

        Creates a degree-1 position field on ``_start_geo`` whose dof_values
        are ``_vertex_positions``, then wraps it in a DeformedGeometry.
        """
        deformation_space = fem.make_polynomial_space(self._start_geo, degree=1, dtype=wp.vec2)
        deformation_field = fem.make_discrete_field(space=deformation_space)
        deformation_field.dof_values = self._vertex_positions
        self._geo = deformation_field.make_deformed_geometry(relative=False)

    def _rebuild_fem_structures(self, degree):
        """Build all geometry-dependent FEM objects from scratch.

        Called once from ``__init__``.  Creates geometry objects, boundary
        subdomains, Dirichlet projectors, and function spaces.
        """

        if self._use_deformed_geo:
            self._build_deformed_geo()
        elif not hasattr(self, "_geo"):
            self._geo = fem.Trimesh2D(
                tri_vertex_indices=self._tri_vertex_indices, positions=self._vertex_positions, build_bvh=True
            )

        # Identify left, right, and bottom sides for boundary conditions
        boundary = fem.BoundarySides(self._geo)

        # Temporarily reset positions to initial configuration for boundary classification
        tmp = wp.clone(self._vertex_positions)
        self._vertex_positions.assign(self._initial_positions)

        left_mask = wp.zeros(shape=boundary.element_count(), dtype=int)
        right_mask = wp.zeros(shape=boundary.element_count(), dtype=int)
        bottom_mask = wp.zeros(shape=boundary.element_count(), dtype=int)

        fem.interpolate(
            classify_boundary_sides,
            at=boundary,
            values={"left": left_mask, "right": right_mask, "bottom": bottom_mask},
        )

        self._left = fem.Subdomain(boundary, element_mask=left_mask)
        self._right = fem.Subdomain(boundary, element_mask=right_mask)
        self._bottom = fem.Subdomain(boundary, element_mask=bottom_mask)

        # Build spaces, fields, test/trial functions, and projectors
        self._rebuild_spaces(degree)

        self._vertex_positions.assign(tmp)

    def _rebuild_spaces(self, degree):
        """Rebuild function spaces, fields, and test/trial functions.

        Called after edge flips update ``tri_vertex_indices`` in-place.
        Reuses the existing geometry objects (avoids non-deterministic topology
        rebuild) and boundary subdomains (unaffected by interior edge flips).
        """

        self._u_space = fem.make_polynomial_space(self._geo, degree=degree, dtype=wp.vec2)

        # displacement field, make sure gradient is stored
        self._u_field = fem.make_discrete_field(space=self._u_space)
        self._u_field.dof_values.requires_grad = True

        # Trial and test functions
        self._u_test = fem.make_test(space=self._u_space)
        self._u_trial = fem.make_trial(space=self._u_space)

        # Dirichlet projector for the displacement space
        u_left_bd_test = fem.make_test(space=self._u_space, domain=self._left)
        u_left_bd_trial = fem.make_trial(space=self._u_space, domain=self._left)

        u_bd_matrix = fem.integrate(
            boundary_projector_form,
            fields={"u": u_left_bd_trial, "v": u_left_bd_test},
            assembly="nodal",
            output_dtype=float,
        )

        u_right_bd_test = fem.make_test(space=self._u_space, domain=self._right)
        u_right_bd_trial = fem.make_trial(space=self._u_space, domain=self._right)
        u_bd_matrix += fem.integrate(
            boundary_projector_form,
            fields={"u": u_right_bd_trial, "v": u_right_bd_test},
            assembly="nodal",
            output_dtype=float,
        )

        fem.normalize_dirichlet_projector(u_bd_matrix)
        self._bd_projector = u_bd_matrix

        self._u_bottom_test = fem.make_test(space=self._u_space, domain=self._bottom)

    def remesh(self):
        """Perform Delaunay edge flips to improve triangle quality.

        Only active for triangle meshes; silently returns for other mesh types.
        """
        if self._tri_vertex_indices is None:
            return

        positions_np = self._vertex_positions.numpy()
        ref_positions_np = self._initial_positions.numpy()
        tri_np = self._tri_vertex_indices.numpy()
        tri_np, num_flips = delaunay_edge_flip(positions_np, tri_np, ref_positions_np=ref_positions_np)
        if num_flips == 0:
            return

        if not self._quiet:
            print(f"Remesh: {num_flips} edge flip(s)")

        # Write back updated connectivity in-place
        wp.copy(src=wp.array(tri_np, dtype=int), dest=self._tri_vertex_indices)

        # Rebuild spaces and fields on the existing geometry.
        # We intentionally avoid recreating the Trimesh2D objects: edge flips
        # only affect interior edges, so boundary structures and projectors
        # remain valid; and re-running _build_topology() would introduce
        # non-deterministic edge ordering that perturbs the boundary projector
        # enough to visibly affect the displacement on ill-conditioned systems.
        self._rebuild_fem_structures(self._degree)

    def step(self):
        # Forward step, record adjoint tape for forces
        u = self._u_field.dof_values
        u.zero_()

        u_rhs = wp.empty(self._u_space.node_count(), dtype=wp.vec2f, requires_grad=True)

        tape = wp.Tape()

        with tape:
            # Normalize the traction by the bottom boundary length
            # so that the total applied force is independent of boundary deformation
            bottom_length = wp.empty(shape=1, dtype=float, requires_grad=True)
            inv_bottom_length = wp.empty(shape=1, dtype=float, requires_grad=True)
            fem.integrate(
                volume_form,
                domain=self._u_bottom_test.domain,
                output=bottom_length,
            )
            wp.launch(invert_scalar, dim=1, inputs=[bottom_length, inv_bottom_length])
            fem.integrate(
                normalized_load_form,
                fields={"v": self._u_bottom_test},
                values={"load": self._load, "inv_length": inv_bottom_length},
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

        fem_example_utils.bsr_cg(u_matrix, b=u_rhs, x=u, quiet=self._quiet, tol=1e-8, max_iters=1000)

        # Record adjoint of linear solve
        # (For nonlinear elasticity, this should use the final hessian, as per implicit function theorem)
        def solve_linear_system():
            fem_example_utils.bsr_cg(u_matrix, b=u.grad, x=u_rhs.grad, quiet=self._quiet, tol=1e-8, max_iters=1000)
            u_rhs.grad -= self._bd_projector @ u_rhs.grad
            self._u_field.dof_values.grad.zero_()

        tape.record_func(solve_linear_system, arrays=(u_rhs, u))

        # Evaluate residual
        # Integral of squared difference between simulated position and target positions
        loss = wp.empty(shape=1, dtype=float, requires_grad=True)
        vol = wp.empty(shape=1, dtype=float, requires_grad=True)

        with tape:
            fem.integrate(
                loss_form,
                fields={"u": self._u_field},
                values={"lame": self._lame, "quality_weight": 50.0},
                domain=self._u_test.domain,
                output=loss,
            )

            # Add penalization term enforcing constant volume
            fem.integrate(
                volume_form,
                domain=self._u_test.domain,
                output=vol,
            )

            vol_loss_weight = 1000.0
            wp.launch(
                add_volume_loss,
                dim=1,
                inputs=(loss, vol, self._initial_volume, vol_loss_weight),
            )

        # perform backward step
        tape.backward(loss=loss)

        # enforce fixed vertices
        self._vertex_positions.grad -= self._fixed_vertex_projector @ self._vertex_positions.grad

        # enforce x-symmetry
        grad_space = fem.make_polynomial_space(self._start_geo, degree=1, dtype=wp.vec2)
        grad_field = fem.make_discrete_field(space=grad_space)
        grad_field.dof_values.assign(self._vertex_positions.grad)
        sym_field = fem.make_discrete_field(space=grad_space)
        fem.interpolate(
            symmetrize_field,
            dest=sym_field,
            fields={"field": grad_field},
        )
        self._vertex_positions.grad.assign(sym_field.dof_values)

        # update positions and reset tape
        self.optimizer.step([self._vertex_positions_scalar.grad])
        tape.zero()

        return float(loss.numpy()[0])

    def render(self):
        # Render using fields defined on start geometry
        # (renderer assumes geometry remains fixed for timesampled fields)
        u_space = fem.make_polynomial_space(self._start_geo, degree=self._u_space.degree, dtype=wp.vec2)
        u_field = fem.make_discrete_field(space=u_space)
        rest_field = fem.make_discrete_field(space=u_space)

        # Compute geo displacement at node level (works for any degree and geometry type)
        geo_displacement = self._u_space.node_positions() - self._start_node_positions
        u_field.dof_values = self._u_field.dof_values + geo_displacement
        rest_field.dof_values = geo_displacement

        self.renderer.add_field("displacement", u_field)
        self.renderer.add_field("rest", rest_field)

        # Interpolate stress norm on the current geometry, then display on the start geometry
        stress_field_cur = fem.make_discrete_field(
            space=fem.make_polynomial_space(self._geo, degree=self._u_space.degree, dtype=float)
        )
        fem.interpolate(
            stress_norm_field,
            dest=stress_field_cur,
            fields={"u": self._u_field},
            values={"lame": self._lame},
        )
        stress_field_vis = fem.make_discrete_field(
            space=fem.make_polynomial_space(self._start_geo, degree=self._u_space.degree, dtype=float)
        )
        stress_field_vis.dof_values = stress_field_cur.dof_values
        self.renderer.add_field("stress", stress_field_vis)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--resolution", type=int, default=25, help="Grid resolution.")
    parser.add_argument("--degree", type=int, default=1, help="Polynomial degree of shape functions.")
    parser.add_argument(
        "--mesh",
        choices=("tri", "deformed"),
        default="tri",
        help="Shape parametrization: 'tri' modifies vertex positions directly, "
        "'deformed' uses a DeformedGeometry over a fixed reference mesh.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode, suppressing the opening of any graphical windows.",
    )
    parser.add_argument("--lr", type=float, default=1.0e-3, help="Learning rate.")
    parser.add_argument("--num-iters", type=int, default=750, help="Number of iterations.")
    parser.add_argument(
        "--remesh-interval", type=int, default=10, help="Edge-flip remeshing every N iters (0 to disable)."
    )

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(
            quiet=True,
            degree=args.degree,
            resolution=args.resolution,
            mesh=args.mesh,
            lr=args.lr,
        )

        for _k, set_info in fem_example_utils.progress_bar(args.num_iters, quiet=args.headless):
            loss = example.step()
            set_info("loss", loss)
            if _k % 50 == 0:
                example.render()
            if args.remesh_interval > 0 and (_k + 1) % args.remesh_interval == 0:
                example.remesh()

        if not args.headless:
            example.renderer.plot(
                options={
                    "rest": {"displacement": {}, "color": "stress"},
                    "displacement": {"displacement": {}, "color": "stress"},
                },
            )
