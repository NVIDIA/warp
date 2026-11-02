# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Periodic Homogenization
#
# This example computes the effective conductivity tensor of a periodic
# composite from one unit cell Y = [0, 1]^d. For each direction e_i, it
# solves for a periodic corrector w_i with
#
# integral_Y k (grad w_i + e_i) . grad v = 0    for all periodic v,
#
# then forms
#
# K_eff[i, j] = integral_Y k (e_i + grad w_i) . (e_j + grad w_j).
#
# Periodicity is imposed by eliminating the nodes on the high faces of the
# cell with fem.periodic_prolongation() and fem.make_constrained_basis_space().
# On an adaptive grid, it is combined with hanging-node elimination.
#
# The material is a matrix of unit conductivity with a circular (2D) or
# spherical (3D) inclusion, or a laminate of layers normal to x. For the
# laminate, the exact effective conductivity is the harmonic mean of the
# layer conductivities across the layers and their arithmetic mean along them.
#
# The inclusion is not aligned with grid cells, so on the grid, triangle and
# adaptive meshes the material jumps inside cells and the effective tensor
# converges at first order only. The "fitted" mesh deforms a triangulated grid
# so that cell edges follow the circular inclusion, which recovers the full
# convergence rate of the elements. For a circular inclusion, the result is
# compared with Rayleigh's multipole formula for a square array of cylinders.
###########################################################################

from typing import Any

import numpy as np

import warp as wp
import warp.examples.fem.utils as fem_example_utils
import warp.fem as fem

MATERIAL_INCLUSION = wp.constant(0)
MATERIAL_LAMINATE = wp.constant(1)


@wp.func
def conductivity(x: Any, material: int, radius: float, layer_fraction: float, contrast: float):
    """Conductivity at position ``x``: ``contrast`` in the inclusion or first layer, one elsewhere."""
    if material == MATERIAL_LAMINATE:
        return wp.where(x[0] < layer_fraction, contrast, 1.0)
    center = type(x)(0.5)
    return wp.where(wp.length(x - center) < radius, contrast, 1.0)


@fem.integrand
def conductivity_form(
    s: fem.Sample,
    domain: fem.Domain,
    u: fem.Field,
    v: fem.Field,
    material: int,
    radius: float,
    layer_fraction: float,
    contrast: float,
):
    k = conductivity(domain(s), material, radius, layer_fraction, contrast)
    return k * wp.dot(fem.grad(u, s), fem.grad(v, s))


@fem.integrand
def corrector_rhs_form(
    s: fem.Sample,
    domain: fem.Domain,
    v: fem.Field,
    axis: int,
    material: int,
    radius: float,
    layer_fraction: float,
    contrast: float,
):
    """Right-hand side of the corrector equation for the unit direction along ``axis``."""
    k = conductivity(domain(s), material, radius, layer_fraction, contrast)
    return -k * fem.grad(v, s)[axis]


@fem.integrand
def effective_conductivity_form(
    s: fem.Sample,
    domain: fem.Domain,
    w_i: fem.Field,
    w_j: fem.Field,
    i: int,
    j: int,
    material: int,
    radius: float,
    layer_fraction: float,
    contrast: float,
):
    k = conductivity(domain(s), material, radius, layer_fraction, contrast)
    flux_i = fem.grad(w_i, s)
    flux_i[i] += 1.0
    flux_j = fem.grad(w_j, s)
    flux_j[j] += 1.0
    return k * wp.dot(flux_i, flux_j)


@wp.func
def refinement_field(x: wp.vec3, material: int, radius: float, layer_fraction: float, margin: float):
    """Finest cells (zero) near the material interface, coarsest (one) far from it."""
    # Periodicity needs the same refinement on opposite faces. Cells are refined
    # from random samples, so keep the cells near the faces at the coarsest level.
    if wp.min(wp.min(x), 1.0 - wp.max(x)) < margin:
        return 1.0
    if material == MATERIAL_LAMINATE:
        distance = wp.abs(x[0] - layer_fraction)
    else:
        distance = wp.abs(wp.length(x - wp.vec3(0.5)) - radius)
    return wp.clamp(4.0 * distance, 0.0, 1.0)


def rayleigh_effective_conductivity(radius: float, contrast: float) -> float:
    """Effective conductivity of a square array of cylinders, from Rayleigh's multipole formula.

    Uses the coefficients of Perrins, McKenzie and McPhedran (1979), accurate to about 1e-6
    for area fractions up to 0.4.
    """
    fraction = np.pi * radius**2
    beta = (contrast - 1.0) / (contrast + 1.0)
    denominator = (
        1.0
        - fraction * beta
        - 0.305827 * beta**2 * fraction**4 / (1.0 - 1.402958 * beta**2 * fraction**8)
        - 0.013362 * beta**2 * fraction**8
    )
    return 1.0 + 2.0 * fraction * beta / denominator


def outward_diagonal_trimesh(resolution: int) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate a regular unit square grid with diagonals pointing away from its center.

    The mesh itself does not follow the inclusion; :func:`inclusion_fitting_map` deforms it.
    The diagonal of each cell passes through the cell corner farthest from the center,
    so when a centered block of cells is mapped onto a disk, the corner cells of the
    block are split into two triangles instead of being flattened.
    """
    n = resolution
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    ll = (i * (n + 1) + j).ravel()
    lr = ll + n + 1
    ul = ll + 1
    ur = lr + 1
    toward_lower_left = ((i + 0.5 < 0.5 * n) == (j + 0.5 < 0.5 * n)).ravel()
    triangles = np.where(
        toward_lower_left[:, None, None],
        np.stack([np.stack([ll, lr, ur], -1), np.stack([ll, ur, ul], -1)], 1),
        np.stack([np.stack([ll, lr, ul], -1), np.stack([lr, ur, ul], -1)], 1),
    ).reshape(-1, 3)
    vertices = np.stack(np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij"), -1).reshape(-1, 2) / n
    return vertices, triangles.astype(np.int32)


def inclusion_fitting_map(points: np.ndarray, resolution: int, radius: float) -> np.ndarray:
    """Map the unit cell onto itself so that a centered block of grid cells covers the inclusion.

    Points at grid distance ``half_width`` from the center (in the max norm) land on the
    inclusion boundary, the cell boundary stays fixed so periodic faces still match, and
    positions in between blend linearly along rays from the center.
    """
    half_width = max(1, round(radius * np.sqrt(np.pi) / 2.0 * resolution)) / resolution
    if half_width >= 0.5 or radius >= 0.5:
        raise ValueError("The fitted mesh requires the inclusion to lie strictly inside the unit cell")

    def square_to_disk(p):
        # Smooth map from [-1, 1]^2 onto the unit disk
        return np.stack(
            [p[:, 0] * np.sqrt(1.0 - 0.5 * p[:, 1] ** 2), p[:, 1] * np.sqrt(1.0 - 0.5 * p[:, 0] ** 2)], axis=1
        )

    offset = points - 0.5
    distance = np.max(np.abs(offset), axis=1)
    mapped = np.empty_like(points)

    inside = distance <= half_width
    mapped[inside] = 0.5 + radius * square_to_disk(offset[inside] / half_width)

    outside = ~inside
    direction = offset[outside] / distance[outside, None]
    on_circle = radius * square_to_disk(direction)
    on_boundary = 0.5 * direction
    blend = ((distance[outside] - half_width) / (0.5 - half_width))[:, None]
    mapped[outside] = 0.5 + (1.0 - blend) * on_circle + blend * on_boundary
    return mapped


def laminate_effective_conductivity(dimension: int, layer_fraction: float, contrast: float) -> np.ndarray:
    """Exact effective conductivity of a laminate of layers normal to x."""
    arithmetic = layer_fraction * contrast + (1.0 - layer_fraction)
    harmonic = 1.0 / (layer_fraction / contrast + (1.0 - layer_fraction))
    return np.diag([harmonic] + [arithmetic] * (dimension - 1))


class Example:
    def __init__(
        self,
        quiet=False,
        mesh="fitted",
        material="inclusion",
        degree=2,
        resolution=32,
        level_count=3,
        contrast=10.0,
        radius=0.25,
        layer_fraction=0.5,
    ):
        self._quiet = quiet
        self._values = {
            "material": MATERIAL_LAMINATE if material == "laminate" else MATERIAL_INCLUSION,
            "radius": radius,
            "layer_fraction": layer_fraction,
            "contrast": contrast,
        }

        # Unit cell geometry and raw (non-periodic) basis
        if mesh == "adaptive":
            # Adaptive 3D grid refined near the material interface, with ``resolution``
            # cells per side at the finest level. Hanging nodes are eliminated
            # first, then periodicity is imposed on the result.
            coarse_resolution = max(resolution >> (level_count - 1), 1)
            volume = fem_example_utils.gen_volume(res=wp.vec3i(coarse_resolution))
            refinement = fem.ImplicitField(
                domain=fem.Cells(fem.Nanogrid(volume)),
                func=refinement_field,
                values={
                    "material": self._values["material"],
                    "radius": radius,
                    "layer_fraction": layer_fraction,
                    "margin": 2.0 / coarse_resolution,
                },
            )
            self._geo = fem.adaptive_nanogrid_from_field(
                volume, level_count, refinement_field=refinement, grading="face"
            )
            raw_basis = fem.make_polynomial_basis_space(self._geo, degree=1)
            base_basis = fem.make_conforming_basis_space(raw_basis)
        else:
            if mesh == "fitted":
                # Start from a triangulated regular grid, then deform it with a quadratic
                # position field holding the fitting map at every node: edges on the
                # boundary of the centered block become curved arcs of the inclusion
                vertices, triangles = outward_diagonal_trimesh(resolution)
                base_geo = fem.Trimesh2D(
                    tri_vertex_indices=wp.array(triangles, dtype=int), positions=wp.array(vertices, dtype=wp.vec2)
                )
                position_space = fem.make_polynomial_space(base_geo, degree=2, dtype=wp.vec2)
                position_field = position_space.make_field()
                position_field.dof_values = wp.array(
                    inclusion_fitting_map(position_space.node_positions().numpy(), resolution, radius), dtype=wp.vec2
                )
                self._geo = position_field.make_deformed_geometry(relative=False)
            elif mesh == "tri":
                positions, tri_vidx = fem_example_utils.gen_trimesh(res=wp.vec2i(resolution))
                self._geo = fem.Trimesh2D(tri_vertex_indices=tri_vidx, positions=positions)
            else:
                self._geo = fem.Grid2D(res=wp.vec2i(resolution))
            raw_basis = fem.make_polynomial_basis_space(self._geo, degree=degree)
            base_basis = raw_basis

        # Curved cells make the integrands rational, so integrate them more accurately
        self._domain = fem.Cells(self._geo)
        self._quadrature = fem.RegularQuadrature(self._domain, order=2 * degree + 2) if mesh == "fitted" else None

        # Nodes on the high faces take the values of their copies on the low faces
        self._dimension = self._geo.dimension
        translations = [wp.types.vector(self._dimension, float)(*row) for row in np.eye(self._dimension)]
        prolongation = fem.periodic_prolongation(base_basis, translations=translations)
        basis = fem.make_constrained_basis_space(base_basis, prolongation)

        self._space = fem.make_collocated_function_space(basis)
        self._correctors = [self._space.make_field() for _ in range(self._dimension)]
        self.effective_conductivity = np.zeros((self._dimension, self._dimension))

        # Correctors expanded to all nodes of the raw basis, for rendering
        self._prolongation = basis.prolongation_matrix()
        raw_space = fem.make_collocated_function_space(raw_basis)
        self._raw_correctors = [raw_space.make_field() for _ in range(self._dimension)]

        self.renderer = fem_example_utils.Plot()

    def step(self):
        domain = self._domain
        test = fem.make_test(self._space, domain=domain)
        trial = fem.make_trial(self._space, domain=domain)
        matrix = fem.integrate(
            conductivity_form, quadrature=self._quadrature, fields={"u": trial, "v": test}, values=self._values
        )

        # With only periodic conditions, solutions are defined up to a constant;
        # pin the first independent node to zero.
        pinned = np.zeros(self._space.node_count(), dtype=np.float32)
        pinned[0] = 1.0
        projector = wp.sparse.bsr_diag(wp.array(pinned, dtype=float))
        fem.project_system_matrix(matrix, projector)

        for axis, corrector in enumerate(self._correctors):
            rhs = fem.integrate(
                corrector_rhs_form,
                quadrature=self._quadrature,
                fields={"v": test},
                values={"axis": axis, **self._values},
            )
            fem.project_system_rhs(matrix, rhs, projector)
            x = wp.zeros_like(rhs)
            fem_example_utils.bsr_cg(matrix, b=rhs, x=x, tol=1.0e-8, quiet=self._quiet)
            corrector.dof_values = x

        for i in range(self._dimension):
            for j in range(i, self._dimension):
                value = fem.integrate(
                    effective_conductivity_form,
                    domain=domain,
                    quadrature=self._quadrature,
                    fields={"w_i": self._correctors[i], "w_j": self._correctors[j]},
                    values={"i": i, "j": j, **self._values},
                )
                self.effective_conductivity[i, j] = self.effective_conductivity[j, i] = float(value)

        for corrector, raw_corrector in zip(self._correctors, self._raw_correctors, strict=True):
            raw_corrector.dof_values = self._prolongation @ corrector.dof_values

    def render(self):
        self.renderer.add_field("corrector_x", self._raw_correctors[0])


if __name__ == "__main__":
    import argparse

    wp.set_module_options({"enable_backward": False})

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--mesh",
        choices=("fitted", "grid", "tri", "adaptive"),
        default="fitted",
        help="Unit cell geometry: 2D triangle mesh fitted to the inclusion, 2D grid, 2D triangle mesh, "
        "or 3D adaptive grid refined near the material interface.",
    )
    parser.add_argument("--material", choices=("inclusion", "laminate"), default="inclusion", help="Material layout.")
    parser.add_argument(
        "--degree", type=int, default=2, help="Polynomial degree of shape functions (the adaptive grid uses 1)."
    )
    parser.add_argument(
        "--resolution", type=int, default=32, help="Grid resolution, or finest resolution of the adaptive grid."
    )
    parser.add_argument("--level-count", type=int, default=3, help="Number of refinement levels of the adaptive grid.")
    parser.add_argument(
        "--contrast", type=float, default=10.0, help="Conductivity of the inclusion or first layer; the matrix has 1."
    )
    parser.add_argument("--radius", type=float, default=0.25, help="Radius of the inclusion.")
    parser.add_argument("--layer-fraction", type=float, default=0.5, help="Volume fraction of the first layer.")
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
            mesh=args.mesh,
            material=args.material,
            degree=args.degree,
            resolution=args.resolution,
            level_count=args.level_count,
            contrast=args.contrast,
            radius=args.radius,
            layer_fraction=args.layer_fraction,
        )
        example.step()
        example.render()

        np.set_printoptions(precision=5, suppress=True)
        print("Effective conductivity tensor:")
        print(example.effective_conductivity)
        if args.material == "laminate":
            print("Exact laminate tensor:")
            print(
                laminate_effective_conductivity(
                    example.effective_conductivity.shape[0], args.layer_fraction, args.contrast
                )
            )
        elif example.effective_conductivity.shape[0] == 2:
            reference = rayleigh_effective_conductivity(args.radius, args.contrast)
            print(f"Rayleigh multipole reference: {reference:.6f}")
            print(f"Relative error: {example.effective_conductivity[0, 0] / reference - 1.0:+.2e}")

        if not args.headless:
            example.renderer.plot()
