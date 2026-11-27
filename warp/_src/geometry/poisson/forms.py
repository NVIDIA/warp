# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact volume and explicitly weighted point forms using Warp FEM."""

import warp as wp
import warp.fem as fem

from .point_location import locate_constraints


@fem.integrand
def stiffness(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.dot(fem.grad(u, s), fem.grad(v, s))


@fem.integrand
def mass(s: fem.Sample, u: fem.Field, v: fem.Field):
    return u(s) * v(s)


@fem.integrand
def value(s: fem.Sample, u: fem.Field):
    return u(s)


@fem.integrand
def gradient(s: fem.Sample, u: fem.Field):
    return fem.grad(u, s)


def uniform_space(depth, dtype=wp.float32):
    """Verification space only; never a maximum-depth reconstruction fallback."""
    if not isinstance(depth, int) or not 0 <= depth <= 5:
        raise ValueError("Uniform verification grid requires integer depth 0..5")
    geometry = fem.Grid3D(wp.vec3i(1 << depth))
    return fem.make_polynomial_space(geometry, degree=1, dtype=dtype, element_basis=fem.ElementBasis.BSPLINE)


def assemble_stiffness(space, device=None, trial_space=None, accumulation_dtype=None):
    domain = fem.Cells(space.geometry)
    u, v = (
        fem.make_trial(trial_space or space, domain=domain),
        fem.make_test(space, domain=domain),
    )
    quadrature = fem.RegularQuadrature(domain, order=2)
    return fem.integrate(
        stiffness,
        quadrature=quadrature,
        fields={"u": u, "v": v},
        output_dtype=accumulation_dtype or space.dtype,
        device=device,
    )


def point_quadrature(space, positions, effective_weights):
    domain = fem.Cells(space.geometry)
    coordinates = positions
    if space.geometry.scalar_type == wp.float64 and isinstance(space.geometry, (fem.AdaptiveNanogrid, fem.Nanogrid)):
        coordinates = locate_constraints(space.geometry, positions)
    quadrature = fem.PicQuadrature(domain, coordinates, measures=effective_weights)
    mask = wp.empty(space.geometry.cell_count(), dtype=int, device=positions.device)
    quadrature.fill_element_mask(mask)
    # Keep the original quadrature's domain stable when restricting matrix
    # triplets, so cached particle-binning closures cannot alias a changed domain.
    restricted = fem.Subdomain(domain, element_mask=mask)
    return fem.PicQuadrature(
        restricted,
        (quadrature.cell_indices, quadrature.particle_coords),
        measures=effective_weights,
    )


def assemble_screening(
    space,
    positions,
    effective_weights,
    trial_space=None,
    accumulation_dtype=None,
    quadrature=None,
):
    """Sum w_s phi_i(p_s) phi_j(p_s); weights are NOT the public pointWeight."""
    if positions.shape[0] != effective_weights.shape[0]:
        raise ValueError("One effective weight per point is required")
    if quadrature is None:
        quadrature = point_quadrature(space, positions, effective_weights)
    domain = quadrature.domain
    u, v = (
        fem.make_trial(trial_space or space, domain=domain),
        fem.make_test(space, domain=domain),
    )
    return fem.integrate(
        mass,
        quadrature=quadrature,
        fields={"u": u, "v": v},
        output_dtype=accumulation_dtype or space.dtype,
        device=positions.device,
    )


def assemble_point_load(space, positions, effective_weights, quadrature=None):
    if quadrature is None:
        quadrature = point_quadrature(space, positions, effective_weights)
    domain = quadrature.domain
    v = fem.make_test(space, domain=domain)
    return fem.integrate(
        value,
        quadrature=quadrature,
        fields={"u": v},
        output_dtype=wp.float64,
        device=positions.device,
    )
