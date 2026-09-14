# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FP64 density-based topology optimization of a cantilever with Warp FEM.

The rectangular beam is clamped on its left face and carries a downward surface
traction over a small patch centered on its right face. Cells next to the clamp
and load are kept solid. The optimization minimizes compliance (maximizes
stiffness) subject to a material-volume constraint.

Warp FEM assembles the Solid Isotropic Material with Penalization (SIMP)
linear-elasticity system, distributed traction, Dirichlet projection, and
element energies. A linear density filter regularizes the cell-wise design
variables. Warp automatic differentiation propagates implicit-adjoint element
sensitivities through SIMP and the filter to the raw decision variables, which
NLopt's Method of Moving Asymptotes (MMA) updates. The console reports
compliance relative to the initial design (``C/C0``) and mean physical density
(``V``). The live plot shows the unthresholded physical density used by the
finite-element analysis, viewed as a maximum-density projection through the
beam thickness.

`NLopt <https://nlopt.readthedocs.io/en/latest/>`_ is required to run the
optimization.
"""

import argparse
import math

import numpy as np

import warp as wp
import warp.fem as fem
from warp.examples.fem import utils as fem_example_utils

wp.set_module_options({"enable_backward": True})


@wp.func
def hooke_stress(strain: wp.mat33d, lame: wp.vec2d):
    """Evaluate isotropic Hooke stress."""
    return wp.float64(2.0) * lame[1] * strain + lame[0] * wp.trace(strain) * wp.identity(n=3, dtype=wp.float64)


@fem.integrand
def simp_elasticity_form(
    s: fem.Sample,
    u: fem.Field,
    v: fem.Field,
    density: fem.Field,
    lame: wp.vec2d,
    penal: wp.float64,
    emin_ratio: wp.float64,
):
    """Evaluate the SIMP-scaled elasticity bilinear form.

    Args:
        s: Integration sample.
        u: Trial displacement field.
        v: Test displacement field.
        density: Cell-wise physical-density field.
        lame: First and second Lame parameters.
        penal: SIMP penalization exponent.
        emin_ratio: Minimum-to-solid stiffness ratio.

    Returns:
        The local contribution to the stiffness bilinear form.
    """
    rho = density(s)
    scale = emin_ratio + (wp.float64(1.0) - emin_ratio) * wp.pow(rho, penal)
    return scale * wp.ddot(fem.D(v, s), hooke_stress(fem.D(u, s), lame))


@fem.integrand
def element_energy_form(s: fem.Sample, u: fem.Field, q: fem.Field, lame: wp.vec2d):
    """Evaluate elastic energy independently for every density cell."""
    return q(s) * wp.ddot(fem.D(u, s), hooke_stress(fem.D(u, s), lame))


@fem.integrand
def cantilever_clamp_form(s: fem.Sample, domain: fem.Domain, u: fem.Field, v: fem.Field, clamp_x: wp.float64):
    """Select displacement degrees of freedom on the clamped face."""
    x = fem.position(domain, s)
    if x[0] <= clamp_x:
        return wp.dot(u(s), v(s))
    return wp.float64(0.0)


@fem.integrand
def cantilever_load_form(
    s: fem.Sample,
    domain: fem.Domain,
    v: fem.Field,
    loaded_x: wp.float64,
    load_y_min: wp.float64,
    load_y_max: wp.float64,
    traction: wp.vec3d,
):
    """Evaluate the distributed traction on the free-end load patch.

    Args:
        s: Boundary integration sample.
        domain: Boundary-side domain used to evaluate positions.
        v: Boundary displacement test field.
        loaded_x: Minimum x-coordinate belonging to the free-end face.
        load_y_min: Lower y-coordinate of the load patch.
        load_y_max: Upper y-coordinate of the load patch.
        traction: Constant traction chosen to produce the requested total load.

    Returns:
        The local load-vector contribution, or zero outside the patch.
    """
    x = fem.position(domain, s)
    if x[0] >= loaded_x and x[1] >= load_y_min and x[1] <= load_y_max:
        return wp.dot(v(s), traction)
    return wp.float64(0.0)


@wp.kernel
def apply_density_filter(
    raw: wp.array[wp.float64],
    passive: wp.array[int],
    offsets: wp.array[int],
    neighbors: wp.array[int],
    weights: wp.array[wp.float64],
    physical: wp.array[wp.float64],
):
    """Apply the linear density filter and enforce passive solid cells.

    Each CSR row stores normalized distance weights for one physical-density
    cell. Passive clamp and load cells are assigned unit density.

    Args:
        raw: Raw optimizer decision variables.
        passive: Flags identifying cells that must remain solid.
        offsets: CSR row offsets for the filter neighborhoods.
        neighbors: CSR column indices of neighboring cells.
        weights: Normalized distance weights corresponding to ``neighbors``.
        physical: Output filtered density field.
    """
    e = wp.tid()
    if passive[e] != 0:
        physical[e] = wp.float64(1.0)
        return
    value = wp.float64(0.0)
    for p in range(offsets[e], offsets[e + 1]):
        n = neighbors[p]
        rho = raw[n]
        if passive[n] != 0:
            rho = wp.float64(1.0)
        value += weights[p] * rho
    physical[e] = value


@wp.kernel
def compliance_adjoint_loss(
    energy: wp.array[wp.float64],
    density: wp.array[wp.float64],
    penal: wp.float64,
    emin_ratio: wp.float64,
    normalization: wp.float64,
    loss: wp.array[wp.float64],
):
    """Accumulate the implicit-adjoint compliance loss.

    With displacement held at the converged equilibrium, differentiating the
    negative strain energy produces ``dC = -u^T (dK) u``. Warp Tape propagates
    this loss through SIMP and the density filter to the raw design variables.

    Args:
        energy: Unpenalized elastic energy integrated per cell.
        density: Filtered physical densities tracked by Warp Tape.
        penal: SIMP penalization exponent.
        emin_ratio: Minimum-to-solid stiffness ratio.
        normalization: Initial compliance used to scale the objective.
        loss: Scalar output accumulated atomically.
    """
    e = wp.tid()
    rho = wp.max(density[e], wp.float64(1.0e-12))
    stiffness_scale = emin_ratio + (wp.float64(1.0) - emin_ratio) * wp.pow(rho, penal)
    # With the converged displacement held fixed, negative strain energy is
    # the implicit compliance adjoint: dC = -u^T (dK) u.
    wp.atomic_add(loss, 0, -stiffness_scale * energy[e] / normalization)


@wp.kernel
def volume_loss(density: wp.array[wp.float64], count: wp.float64, loss: wp.array[wp.float64]):
    """Accumulate mean physical density for the volume constraint."""
    e = wp.tid()
    wp.atomic_add(loss, 0, density[e] / count)


def build_filter(resolution, radius):
    """Build normalized CSR rows for the linear density filter.

    Neighbor weights decrease linearly with cell-center distance and vanish at
    ``radius``. Because the grid and radius are fixed, these rows are built once
    and reused throughout optimization.

    Args:
        resolution: Number of density cells along each grid axis.
        radius: Filter support radius measured in cell units.

    Returns:
        A tuple containing CSR offsets, neighbor indices, and normalized weights.

    Raises:
        ValueError: If ``radius`` is not positive.
    """
    if radius <= 0.0:
        raise ValueError("Density filter radius must be positive")
    shape = np.asarray(resolution, dtype=np.int32)
    cells = np.indices(tuple(shape), dtype=np.int32).reshape(3, -1).T

    search = math.ceil(radius)
    axis = np.arange(-search, search + 1, dtype=np.int32)
    deltas = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)

    stencil_weights = radius - np.linalg.norm(deltas, axis=1)
    supported = stencil_weights > 0.0
    deltas = deltas[supported]
    stencil_weights = stencil_weights[supported]

    candidates = cells[:, None, :] + deltas[None, :, :]
    valid = np.all((candidates >= 0) & (candidates < shape), axis=-1)

    strides = np.asarray((shape[1] * shape[2], shape[2], 1), dtype=np.int64)
    linear_indices = candidates @ strides
    counts = valid.sum(axis=1)
    offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int32)
    neighbors = linear_indices[valid].astype(np.int32)

    row_totals = np.where(valid, stencil_weights[None, :], 0.0).sum(axis=1)
    weights = (stencil_weights[None, :] / row_totals[:, None])[valid]
    return offsets, neighbors, weights


class Example:
    """Optimize an FP64 cantilever using SIMP and Warp FEM.

    Construction allocates the FEM fields, boundary projector, distributed load,
    density-filter rows, and design arrays. Repeated evaluations update the
    stiffness values in place before solving with conjugate gradient.

    Args:
        args: Parsed command-line arguments defining the resolution, maximum
            evaluations, device, and visualization mode.
    """

    def __init__(self, args):
        self.args = args
        self.device = wp.get_device(args.device)
        self.length = 3.0
        self.height = 1.0
        self.thickness = 0.25
        self.load = 1.0
        self.volume_fraction = 0.30
        self.penal = 3.0
        self.filter_radius = 1.5
        self.emin = 1.0e-6
        self.young = 2.0e9
        self.poisson = 0.30
        self.resolution = np.asarray((args.nx, args.ny, args.nz), dtype=np.int32)
        self.active_count = int(np.prod(self.resolution))
        self.geometry = fem.Grid3D(
            res=wp.vec3i(*self.resolution.tolist()),
            scalar_type=wp.float64,
            bounds_lo=wp.vec3d(0.0, 0.0, 0.0),
            bounds_hi=wp.vec3d(self.length, self.height, self.thickness),
        )
        self.cells = fem.Cells(geometry=self.geometry)
        self.elasticity_quadrature = fem.RegularQuadrature(self.cells, order=2)

        self.u_space = fem.make_polynomial_space(self.geometry, degree=1, dtype=wp.vec3d)
        self.u_partition = fem.make_space_partition(self.u_space.topology, with_halo=False)
        self.u_test = fem.make_test(space=self.u_space, space_partition=self.u_partition, domain=self.cells)
        self.u_trial = fem.make_trial(space=self.u_space, space_partition=self.u_partition, domain=self.cells)
        self.displacement = fem.make_discrete_field(self.u_space, self.u_partition)

        self.rho_space = fem.make_polynomial_space(self.geometry, degree=0, dtype=wp.float64)
        self.rho_partition = fem.make_space_partition(self.rho_space.topology, with_halo=False)
        self.rho_field = fem.make_discrete_field(self.rho_space, self.rho_partition)
        self.rho_test = fem.make_test(space=self.rho_space, space_partition=self.rho_partition, domain=self.cells)
        if len(self.rho_field.dof_values) != self.active_count:
            raise RuntimeError("Unexpected Q0 density ordering/count for cantilever grid")

        boundary = fem.BoundarySides(self.geometry)
        boundary_test = fem.make_test(space=self.u_space, space_partition=self.u_partition, domain=boundary)
        boundary_trial = fem.make_trial(space=self.u_space, space_partition=self.u_partition, domain=boundary)
        spacing_x = self.length / args.nx
        self.projector = fem.integrate(
            cantilever_clamp_form,
            fields={"u": boundary_trial, "v": boundary_test},
            values={"clamp_x": 0.25 * spacing_x},
            assembly="nodal",
            output_dtype=wp.float64,
        )
        fem.normalize_dirichlet_projector(self.projector)

        patch_cells = max(1, args.ny // 8)
        patch_height = patch_cells * self.height / args.ny
        load_y_min = 0.5 * (self.height - patch_height)
        load_y_max = 0.5 * (self.height + patch_height)
        traction = wp.vec3d(0.0, -self.load / (patch_height * self.thickness), 0.0)
        self.load_vector = fem.integrate(
            cantilever_load_form,
            fields={"v": boundary_test},
            values={
                "loaded_x": self.length - 0.25 * spacing_x,
                "load_y_min": load_y_min,
                "load_y_max": load_y_max,
                "traction": traction,
            },
            output_dtype=wp.vec3d,
        )
        self.rhs = wp.empty_like(self.load_vector)

        full = np.arange(self.active_count, dtype=np.int32)
        yz = args.ny * args.nz
        ix = full // yz
        iy = (full - ix * yz) // args.nz
        center_y = 0.5 * (args.ny - 1)
        passive = ((ix == 0) | ((ix == args.nx - 1) & (np.abs(iy - center_y) <= 0.5 * patch_cells))).astype(np.int32)
        self.passive_np = passive
        offsets, neighbors, weights = build_filter(self.resolution, self.filter_radius)
        self.passive = wp.array(passive, dtype=int, device=self.device)
        self.filter_offsets = wp.array(offsets, dtype=int, device=self.device)
        self.filter_neighbors = wp.array(neighbors, dtype=int, device=self.device)
        self.filter_weights = wp.array(weights, dtype=wp.float64, device=self.device)

        design_count = int(np.count_nonzero(passive == 0))
        passive_fraction = float(passive.sum() / self.active_count)
        if self.volume_fraction <= passive_fraction:
            raise ValueError(f"volume fraction {self.volume_fraction} is below passive fraction {passive_fraction:.3f}")
        # Start from the uniform raw density that would satisfy the volume
        # constraint without filtering.
        initial = (self.volume_fraction * self.active_count - passive.sum()) / design_count
        raw_np = np.where(passive != 0, 1.0, initial).astype(np.float64)
        self.raw = wp.array(raw_np, dtype=wp.float64, device=self.device, requires_grad=True)
        self.physical = wp.zeros(
            self.active_count,
            dtype=wp.float64,
            device=self.device,
            requires_grad=True,
        )
        self.rho_field.dof_values = self.physical
        self._filter()

        volume_now = self._volume_numpy()
        if abs(volume_now - self.volume_fraction) > 1.0e-7:
            # Passive solid cells contribute to neighboring filter rows, so the
            # filtered volume may differ from the unfiltered estimate above.
            # The filter is linear, hence the volume for a uniform design value
            # ``a`` is affine: V(a) = V(0) + a * (V(1) - V(0)). Evaluate those
            # endpoints and interpolate the value that meets the target volume.
            raw_zero = np.where(passive != 0, 1.0, 0.0).astype(np.float64)
            raw_one = np.ones(self.active_count, dtype=np.float64)
            self.raw.assign(raw_zero)
            self._filter()
            v0 = self._volume_numpy()
            self.raw.assign(raw_one)
            self._filter()
            v1 = self._volume_numpy()
            initial = np.clip((self.volume_fraction - v0) / (v1 - v0), 0.001, 1.0)
            raw_np = np.where(passive != 0, 1.0, initial).astype(np.float64)
            self.raw.assign(raw_np)
            self._filter()
        self.initial_x = raw_np

        e = self.young
        nu = self.poisson
        self.lame = wp.vec2d(e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)), e / (2.0 * (1.0 + nu)))
        self.reference_compliance = None
        self.last_x = None
        self.cached = None
        self.analysis_count = 0
        self.matrix = None
        self.live_figure = None
        self.live_image = None
        self.live_title = None
        self.live_pyplot = None
        if not args.headless:
            self._initialize_live_plot()

    def evaluate(self, x):
        """Evaluate the objective, constraint, and design sensitivities.

        The method filters ``x``, assembles and projects the SIMP stiffness
        system, solves equilibrium, integrates cell energies, and differentiates
        the implicit-adjoint and volume losses through the filter.

        Args:
            x: Raw cell-density decision variables supplied by NLopt.

        Returns:
            A tuple containing normalized compliance, its raw-density gradient,
            mean physical density, its raw-density gradient, and the physical
            density field.
        """
        x64 = np.asarray(x, dtype=np.float64)
        if self.last_x is not None and np.array_equal(x64, self.last_x):
            return self.cached
        self.raw.assign(x64)
        tape = wp.Tape()
        with tape:
            self._filter()
        matrix = self._assemble_matrix()
        self.rhs.assign(self.load_vector)
        fem.project_linear_system(matrix, self.rhs, self.projector, normalize_projector=False)
        self.displacement.dof_values.zero_()
        _residual, iterations = fem_example_utils.bsr_cg(
            matrix,
            b=self.rhs,
            x=self.displacement.dof_values,
            max_iters=3000,
            tol=1.0e-6,
            quiet=True,
        )
        compliance = float(np.sum(self.rhs.numpy() * self.displacement.dof_values.numpy()))
        self.energy = fem.integrate(
            element_energy_form,
            quadrature=self.elasticity_quadrature,
            fields={"u": self.displacement, "q": self.rho_test},
            values={"lame": self.lame},
            output_dtype=wp.float64,
        )
        total_iterations = int(iterations)

        if self.reference_compliance is None:
            self.reference_compliance = compliance
        normalized = compliance / self.reference_compliance
        objective_gradient, volume_gradient = self._autodiff_gradients(tape)
        volume = self._volume_numpy()
        physical = self.physical.numpy()
        self.analysis_count += 1
        self._update_live_plot(physical, compliance, normalized, volume)
        print(
            f"analysis {self.analysis_count:3d}: C/C0={normalized:.6f}, volume={volume:.6f}, "
            f"total CG iterations={total_iterations}"
        )
        self.last_x = x64.copy()
        self.cached = normalized, objective_gradient, volume, volume_gradient, physical
        return self.cached

    def optimize(self):
        """Minimize compliance subject to the volume constraint using NLopt.

        Returns:
            The optimized raw density variables as a NumPy array.

        Raises:
            ImportError: If NLopt is not installed.
        """
        import nlopt  # noqa: PLC0415

        optimizer = nlopt.opt(nlopt.LD_MMA, self.active_count)
        lower = np.where(self.passive_np == 0, 0.001, 1.0).astype(np.float64)
        upper = np.ones(self.active_count, dtype=np.float64)
        optimizer.set_lower_bounds(lower)
        optimizer.set_upper_bounds(upper)

        def objective(x, grad):
            value, sensitivity, _, _, _ = self.evaluate(x)
            if grad.size:
                grad[:] = sensitivity
            return value

        def volume_constraint(x, grad):
            _, _, volume, sensitivity, _ = self.evaluate(x)
            if grad.size:
                grad[:] = sensitivity
            return volume - self.volume_fraction

        optimizer.set_min_objective(objective)
        optimizer.add_inequality_constraint(volume_constraint, 1.0e-5)
        optimizer.set_maxeval(self.args.max_evals)
        optimizer.set_xtol_rel(1.0e-6)
        return optimizer.optimize(self.initial_x.astype(np.float64))

    def show_live_plot(self):
        """Block on the final interactive physical-density view, when enabled."""
        if self.live_figure is not None:
            self.live_pyplot.show(block=True)

    def _filter(self):
        """Filter ``self.raw`` into the FEM physical-density field."""
        wp.launch(
            apply_density_filter,
            dim=self.active_count,
            inputs=[self.raw, self.passive, self.filter_offsets, self.filter_neighbors, self.filter_weights],
            outputs=[self.physical],
            device=self.device,
        )

    def _autodiff_gradients(self, tape):
        """Differentiate compliance and volume back to raw densities.

        Args:
            tape: Tape that recorded the forward density-filter launch.

        Returns:
            NumPy arrays containing objective and volume sensitivities.
        """
        objective_loss = wp.zeros(1, dtype=wp.float64, device=self.device, requires_grad=True)
        volume_loss_value = wp.zeros(1, dtype=wp.float64, device=self.device, requires_grad=True)
        with tape:
            wp.launch(
                compliance_adjoint_loss,
                dim=self.active_count,
                inputs=[self.energy, self.physical, self.penal, self.emin, self.reference_compliance],
                outputs=[objective_loss],
                device=self.device,
            )
            wp.launch(
                volume_loss,
                dim=self.active_count,
                inputs=[self.physical, float(self.active_count)],
                outputs=[volume_loss_value],
                device=self.device,
            )
        tape.backward(objective_loss)
        objective_gradient = self.raw.grad.numpy().astype(np.float64)
        tape.zero()
        tape.backward(volume_loss_value)
        volume_gradient = self.raw.grad.numpy().astype(np.float64)
        tape.zero()
        return objective_gradient, volume_gradient

    def _volume_numpy(self):
        """Return the mean filtered physical density."""
        return float(self.physical.numpy().sum() / self.active_count)

    def _assemble_matrix(self):
        """Assemble SIMP stiffness values while reusing sparse storage.

        Returns:
            The block-sparse stiffness matrix for the current physical density.
        """
        bsr_options = None if self.matrix is None else {"topology": "masked"}
        matrix = fem.integrate(
            simp_elasticity_form,
            quadrature=self.elasticity_quadrature,
            fields={"u": self.u_trial, "v": self.u_test, "density": self.rho_field},
            values={"lame": self.lame, "penal": self.penal, "emin_ratio": self.emin},
            output_dtype=wp.float64,
            output=self.matrix,
            bsr_options=bsr_options,
        )
        if self.matrix is None:
            self.matrix = matrix
        return self.matrix

    def _initialize_live_plot(self):
        """Create an interactive view of the physical-density projection."""
        import matplotlib.pyplot as plt  # noqa: PLC0415

        top = self.physical.numpy().reshape(tuple(self.resolution)).max(axis=2).T
        self.live_pyplot = plt
        plt.ion()
        self.live_figure, axis = plt.subplots(figsize=(10, 3.5))
        self.live_image = axis.imshow(
            top,
            origin="lower",
            cmap="gray_r",
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0,
            extent=(0.0, self.length, 0.0, self.height),
        )
        self.live_title = axis.set_title("Initializing topology optimization")
        axis.set_xlabel("length")
        axis.set_ylabel("height")
        axis.set_aspect("equal")
        self.live_figure.colorbar(self.live_image, ax=axis, label="physical density")
        self.live_figure.tight_layout()
        plt.show(block=False)
        self.live_figure.canvas.draw_idle()
        self.live_figure.canvas.flush_events()

    def _update_live_plot(self, physical, compliance, normalized, volume):
        """Refresh the interactive physical-density view after an analysis."""
        if self.live_figure is None:
            return
        top = physical.reshape(tuple(self.resolution)).max(axis=2).T
        self.live_image.set_data(top)
        self.live_title.set_text(
            f"Analysis {self.analysis_count}: C={compliance:.6e}, C/C0={normalized:.3f}, V={volume:.3f}"
        )
        self.live_figure.canvas.draw_idle()
        self.live_figure.canvas.flush_events()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", default=None)
    parser.add_argument("--nx", type=int, default=32, help="Cantilever cells along its length")
    parser.add_argument("--ny", type=int, default=12, help="Cantilever cells along its height")
    parser.add_argument("--nz", type=int, default=1, help="Cantilever cells through its thickness")
    parser.add_argument("--max-evals", type=int, default=100)
    parser.add_argument("--headless", action="store_true", help="Suppress the interactive density plot")
    args = parser.parse_known_args()[0]
    if min(args.nx, args.ny, args.nz) < 1:
        parser.error("--nx, --ny, and --nz must be positive")
    if args.max_evals < 1:
        parser.error("--max-evals must be positive")

    with wp.ScopedDevice(args.device):
        example = Example(args)
        print(
            f"Cantilever: LxHxT=({example.length}, {example.height}, {example.thickness}), "
            f"grid={tuple(example.resolution)}, load={example.load}, volume={example.volume_fraction}, "
            f"p={example.penal}, optimizer=NLopt MMA"
        )
        print("Sensitivities: implicit elasticity adjoint + Warp Tape through SIMP/filter")
        example.optimize()
        if not args.headless:
            example.show_live_plot()
