# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for constrained basis spaces: conforming adaptive, periodic, and user-supplied prolongations."""

import unittest

import numpy as np

import warp as wp
import warp.fem as fem
from warp.fem.utils import grid_to_tris
from warp.tests.fem.utils import bilinear_form
from warp.tests.unittest_utils import add_function_test, get_selected_cuda_test_devices, get_test_devices


@fem.integrand
def _cross_trace_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return fem.inner(u, s) * fem.outer(v, s)


@fem.integrand
def _vector_mass_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.dot(u(s), v(s))


@fem.integrand
def _virtual_node_value(s: fem.Sample, v: fem.Field):
    return v(fem.at_node(v, s))


@fem.integrand
def _unit_virtual_value(s: fem.Sample, v: fem.Field):
    return 1.0


@fem.integrand
def _trial_at_test_node_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return u(fem.at_node(v, s)) * v(s)


@fem.integrand
def _test_at_trial_node_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return v(fem.at_node(u, s)) * u(s)


@fem.integrand
def _test_at_field_nodes(s: fem.Sample, w: fem.Field, v: fem.Field):
    total = float(0.0)
    for k in range(fem.node_count(w, s)):
        total += v(fem.at_node(w, s, k))
    return total


@fem.integrand
def _test_where_node_stays(s: fem.Sample, v: fem.Field):
    return wp.where(fem.at_node(v, s).element_index == s.element_index, v(s), 0.0)


@fem.integrand(kernel_options={"enable_backward": False})
def _count_moved_samples(s: fem.Sample, v: fem.Field, moved: wp.array[int]):
    if fem.at_node(v, s).element_index != s.element_index:
        wp.atomic_add(moved, 0, 1)
    return 0.0


@fem.integrand
def _linear_init(s: fem.Sample, domain: fem.Domain):
    pos = domain(s)
    return 1.0 + 0.5 * pos[0] - 0.25 * pos[1] + 0.125 * pos[2]


_LINEAR_GRADIENT = wp.constant(wp.vec3(0.5, -0.25, 0.125))
_VECTOR_DIRECTION = wp.constant(wp.vec3(1.0, 2.0, -1.0))


@fem.integrand
def _diffusion_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.dot(fem.grad(u, s), fem.grad(v, s))


@fem.integrand
def _side_gradient_form(s: fem.Sample, domain: fem.Domain, u: fem.Field, v: fem.Field):
    # Normal derivatives on sides depend on nodes away from the side
    n = fem.normal(domain, s)
    return wp.dot(fem.grad(u, s), n) * fem.inner(v, s) + fem.outer(u, s) * wp.dot(fem.grad_outer(v, s), n)


@fem.integrand
def _vector_diffusion_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.ddot(fem.grad(u, s), fem.grad(v, s))


@fem.integrand
def _linear_flux_form(s: fem.Sample, domain: fem.Domain, v: fem.Field):
    return wp.dot(_LINEAR_GRADIENT, fem.normal(domain, s)) * v(s)


@fem.integrand
def _vector_linear_flux_form(s: fem.Sample, domain: fem.Domain, v: fem.Field):
    return wp.dot(_LINEAR_GRADIENT, fem.normal(domain, s)) * wp.dot(_VECTOR_DIRECTION, v(s))


@fem.integrand
def _bottom_projector_form(s: fem.Sample, domain: fem.Domain, u: fem.Field, v: fem.Field):
    # Selects the z = 0 face
    return wp.where(fem.normal(domain, s)[2] < -0.5, u(s) * v(s), 0.0)


@fem.integrand
def _vector_bottom_projector_form(s: fem.Sample, domain: fem.Domain, u: fem.Field, v: fem.Field):
    return wp.where(fem.normal(domain, s)[2] < -0.5, wp.dot(u(s), v(s)), 0.0)


@fem.integrand
def _bottom_value_form(s: fem.Sample, domain: fem.Domain, v: fem.Field):
    value = 1.0 + wp.dot(_LINEAR_GRADIENT, domain(s))
    return wp.where(fem.normal(domain, s)[2] < -0.5, value * v(s), 0.0)


@fem.integrand
def _vector_bottom_value_form(s: fem.Sample, domain: fem.Domain, v: fem.Field):
    value = (1.0 + wp.dot(_LINEAR_GRADIENT, domain(s))) * _VECTOR_DIRECTION
    return wp.where(fem.normal(domain, s)[2] < -0.5, wp.dot(value, v(s)), 0.0)


@fem.integrand
def _jump_square(s: fem.Sample, u: fem.Field):
    value = fem.jump(u, s)
    return value * value


@wp.func
def _periodic_offset(offset: wp.vec3, period: wp.vec3):
    # Remove whole periods along the periodic axes, which have a nonzero period
    wrapped = offset
    for axis in range(3):
        if period[axis] > 0.0:
            wrapped[axis] = offset[axis] - period[axis] * wp.round(offset[axis] / period[axis])
    return wrapped


@fem.integrand(kernel_options={"enable_backward": False})
def _check_master_samples(
    s: fem.Sample,
    domain: fem.Domain,
    u: fem.Field,
    positions: wp.array[wp.vec3],
    expected: wp.array[float],
    period: wp.vec3,
    failures: wp.array[int],
):
    # at_node() samples each node at its anchor or at one of its periodic images
    for k in range(fem.node_count(u, s)):
        node = fem.node_index(u, s, k)
        sample = fem.at_node(u, s, k)
        if wp.length(_periodic_offset(domain(sample) - positions[node], period)) > 1.0e-4:
            wp.atomic_add(failures, 0, 1)
        if wp.abs(u(sample) - expected[node]) > 1.0e-4:
            wp.atomic_add(failures, 0, 1)
        # The sample still refers to the integrand's quadrature point
        if sample.qp_index != s.qp_index or sample.qp_weight != s.qp_weight:
            wp.atomic_add(failures, 0, 1)
    return 0.0


def _dense(matrix):
    dense = np.zeros(matrix.shape)
    offsets = matrix.offsets.numpy()
    columns = matrix.columns.numpy()
    values = matrix.values.numpy()
    row_counts = matrix.row_counts.numpy() if matrix.row_counts is not None else np.diff(offsets)
    block_rows, block_cols = matrix.block_shape
    for row in range(matrix.nrow):
        for index in range(offsets[row], offsets[row] + row_counts[row]):
            if matrix.block_size == 1:
                dense[row, columns[index]] = values[index]
            else:
                dense[
                    row * block_rows : (row + 1) * block_rows,
                    columns[index] * block_cols : (columns[index] + 1) * block_cols,
                ] = values[index]
    return dense


def _mass_matrix(space, domain, device, form=bilinear_form, assembly="generic", bsr_options=None):
    test = fem.make_test(space, domain=domain)
    trial = fem.make_trial(space, domain=domain)
    return fem.integrate(
        form,
        fields={"u": test, "v": trial},
        domain=domain,
        assembly=assembly,
        bsr_options=bsr_options,
        device=device,
    )


def test_q1_hanging_nodes(test, device):
    with wp.ScopedDevice(device):
        _test_q1_hanging_nodes(test, device)


def _two_cell_geometry(device):
    """One coarse cell next to one fine cell; the shared facet has three hanging nodes."""
    return fem.AdaptiveNanogrid.from_environment_voxels(
        points=wp.array([[0, 0, 0], [2, 0, 0]], dtype=wp.vec3i, device=device),
        cell_levels=wp.array([1, 0], dtype=wp.uint8, device=device),
        point_envs=wp.zeros(2, dtype=wp.int32, device=device),
        env_count=1,
        level_count=2,
        device=device,
    )


def _dense_prolongation(constrained):
    offsets = constrained.prolongation_offsets.numpy()
    indices = constrained.prolongation_indices.numpy()
    weights = constrained.prolongation_weights.numpy()
    prolongation = np.zeros((constrained.raw_basis.topology.node_count(), constrained.topology.node_count()))
    for raw_node in range(prolongation.shape[0]):
        prolongation[raw_node, indices[offsets[raw_node] : offsets[raw_node + 1]]] = weights[
            offsets[raw_node] : offsets[raw_node + 1]
        ]
    return prolongation


def _test_q1_hanging_nodes(test, device):
    geometry = _two_cell_geometry(device)
    raw = fem.make_polynomial_basis_space(geometry, degree=1)
    constrained = fem.make_conforming_basis_space(raw)
    test.assertIs(constrained.raw_basis, raw)
    test.assertIs(constrained.topology.trace(), constrained.topology.trace())
    test.assertIs(constrained.trace(), constrained.trace())
    test.assertEqual(raw.topology.node_count(), 15)
    test.assertEqual(constrained.topology.node_count(), 12)

    side_indices = constrained.topology.trace().element_node_indices().numpy()
    side_counts = constrained.topology._side_counts.numpy()
    for side, count in enumerate(side_counts):
        test.assertEqual(len(np.unique(side_indices[side, :count])), count)
        test.assertTrue(np.all(side_indices[side, count:] == fem.NULL_NODE_INDEX))
    interface_sides = np.nonzero(geometry.side_facet_incidence().outer_cell.numpy() >= 0)[0]
    test.assertEqual(len(interface_sides), 1)
    # Sides list every independent node of both cells, as gradients on the side depend on all of them
    test.assertEqual(side_counts[interface_sides[0]], 12)

    prolongation = _dense_prolongation(constrained)
    np.testing.assert_allclose(np.sum(prolongation, axis=1), 1.0)

    raw_space = fem.make_collocated_function_space(raw)
    space = fem.make_collocated_function_space(constrained)
    raw_nodal_boundary = _mass_matrix(raw_space, fem.BoundarySides(geometry), device, assembly="nodal")
    test.assertGreater(np.count_nonzero(_dense(raw_nodal_boundary)), 0)
    with test.assertRaisesRegex(ValueError, "Nodal assembly is not supported.*raw_basis.*reduce_dirichlet_projector"):
        _mass_matrix(space, fem.Cells(geometry), device, assembly="nodal")
    with test.assertRaisesRegex(ValueError, "Nodal quadrature is not supported"):
        fem.NodalQuadrature(domain=fem.Cells(geometry), basis_space=constrained)
    for domain, form in (
        (fem.Cells(geometry), bilinear_form),
        (fem.BoundarySides(geometry), bilinear_form),
        (fem.Sides(geometry), _side_gradient_form),
    ):
        raw_matrix = _mass_matrix(raw_space, domain, device, form)
        reduced_matrix = _mass_matrix(space, domain, device, form)
        np.testing.assert_allclose(
            _dense(reduced_matrix),
            prolongation.T @ _dense(raw_matrix) @ prolongation,
            rtol=1.0e-5,
            atol=1.0e-5,
        )

    sides = fem.Sides(geometry)
    for assembly, bsr_options in (
        ("generic", None),
        ("generic", {"construction": "row_compress"}),
        ("generic", {"topology": "padded"}),
        ("dispatch", None),
    ):
        raw_cross = _mass_matrix(
            raw_space, sides, device, _cross_trace_form, assembly=assembly, bsr_options=bsr_options
        )
        reduced_cross = _mass_matrix(
            space, sides, device, _cross_trace_form, assembly=assembly, bsr_options=bsr_options
        )
        np.testing.assert_allclose(
            _dense(reduced_cross),
            prolongation.T @ _dense(raw_cross) @ prolongation,
            rtol=1.0e-5,
            atol=1.0e-5,
        )

    vector_raw = fem.make_collocated_function_space(raw, dtype=wp.vec3)
    vector_space = fem.make_collocated_function_space(constrained, dtype=wp.vec3)
    vector_raw_mass = _mass_matrix(vector_raw, fem.Cells(geometry), device, _vector_mass_form)
    vector_mass = _mass_matrix(vector_space, fem.Cells(geometry), device, _vector_mass_form)
    vector_prolongation = np.kron(prolongation, np.eye(3))
    np.testing.assert_allclose(
        _dense(vector_mass),
        vector_prolongation.T @ _dense(vector_raw_mass) @ vector_prolongation,
        rtol=1.0e-5,
        atol=1.0e-5,
    )

    virtual = fem.make_test(space, domain=fem.Cells(geometry))
    virtual_at_node = fem.integrate(
        _virtual_node_value,
        domain=fem.Cells(geometry),
        fields={"v": virtual},
        assembly="generic",
        device=device,
    )
    virtual_unit = fem.integrate(
        _unit_virtual_value,
        domain=fem.Cells(geometry),
        fields={"v": virtual},
        assembly="generic",
        device=device,
    )
    np.testing.assert_allclose(virtual_at_node.numpy(), virtual_unit.numpy(), rtol=1.0e-5, atol=1.0e-5)

    field = space.make_field()
    fem.interpolate(_linear_init, dest=field, device=device)
    expected_positions = constrained.node_positions().numpy()
    np.testing.assert_allclose(
        field.dof_values.numpy(),
        1.0 + 0.5 * expected_positions[:, 0] - 0.25 * expected_positions[:, 1] + 0.125 * expected_positions[:, 2],
        rtol=1.0e-5,
        atol=1.0e-5,
    )
    field.dof_values = wp.array(np.linspace(1.0, 3.0, constrained.topology.node_count()), dtype=float, device=device)
    positions = constrained.node_positions()
    for domain, domain_field in ((fem.Cells(geometry), field), (fem.BoundarySides(geometry), field.trace())):
        failures = wp.zeros(1, dtype=int, device=device)
        fem.integrate(
            _check_master_samples,
            domain=domain,
            fields={"u": domain_field},
            values={"positions": positions, "expected": field.dof_values, "period": wp.vec3(0.0), "failures": failures},
            device=device,
        )
        test.assertEqual(int(failures.numpy()[0]), 0)


def _three_level_geometry(device, scalar_type=wp.float32):
    """Three levels with missing cells, so that levels differ by two across some facets."""
    base = [(4 * i, 4 * j, 4 * k) for i in range(2) for j in range(2) for k in range(2) if (i, j, k) != (1, 0, 0)]
    middle = [(4 + 2 * i, 2 * j, 2 * k) for i in range(2) for j in range(2) for k in range(2) if (i, j, k) != (0, 1, 0)]
    fine = [(4 + i, 2 + j, k) for i in range(2) for j in range(2) for k in range(2)]
    points = base + middle + fine
    return fem.AdaptiveNanogrid.from_environment_voxels(
        points=wp.array(points, dtype=wp.vec3i, device=device),
        cell_levels=wp.array([2] * len(base) + [1] * len(middle) + [0] * len(fine), dtype=wp.uint8, device=device),
        point_envs=wp.zeros(len(points), dtype=int, device=device),
        env_count=1,
        level_count=3,
        device=device,
        scalar_type=scalar_type,
    )


def test_q1_three_level(test, device):
    with wp.ScopedDevice(device):
        for scalar_type in (wp.float32, wp.float64):
            geometry = _three_level_geometry(device, scalar_type)
            raw = fem.make_polynomial_basis_space(geometry, degree=1)
            constrained = fem.make_conforming_basis_space(raw)
            test.assertEqual(raw.topology.node_count(), 65)
            test.assertEqual(constrained.topology.node_count(), 36)
            test.assertEqual(constrained.topology.MAX_NODES_PER_ELEMENT, 10)
            test.assertEqual(constrained.trace().topology.MAX_NODES_PER_ELEMENT, 13)

            offsets = constrained.prolongation_offsets.numpy()
            weights = constrained.prolongation_weights.numpy()
            for raw_node in range(raw.topology.node_count()):
                test.assertAlmostEqual(float(np.sum(weights[offsets[raw_node] : offsets[raw_node + 1]])), 1.0)

            if scalar_type == wp.float32:
                space = fem.make_collocated_function_space(constrained)
                field = space.make_field()
                field.dof_values = wp.array(
                    np.linspace(1.0, 3.0, constrained.topology.node_count()), dtype=float, device=device
                )
                failures = wp.zeros(1, dtype=int, device=device)
                fem.integrate(
                    _check_master_samples,
                    domain=fem.Sides(geometry),
                    fields={"u": field.trace()},
                    values={
                        "positions": constrained.node_positions(),
                        "expected": field.dof_values,
                        "period": wp.vec3(0.0),
                        "failures": failures,
                    },
                    device=device,
                )
                test.assertEqual(int(failures.numpy()[0]), 0)
                jump_norm = fem.integrate(
                    _jump_square, domain=fem.Sides(geometry), fields={"u": field.trace()}, device=device
                )
                test.assertLess(abs(float(jump_norm)), 1.0e-5)


def test_high_degree_hanging_nodes(test, device):
    with wp.ScopedDevice(device):
        for geometry in (_two_cell_geometry(device), _three_level_geometry(device)):
            for degree in (2, 3):
                raw = fem.make_polynomial_basis_space(geometry, degree=degree)
                constrained = fem.make_conforming_basis_space(raw)
                test.assertLess(constrained.topology.node_count(), raw.topology.node_count())
                prolongation = _dense_prolongation(constrained)
                np.testing.assert_allclose(np.sum(prolongation, axis=1), 1.0, atol=1.0e-5)

                # Polynomials of the basis degree are reproduced at every raw node
                positions = raw.node_positions().numpy() / 8.0
                polynomial = (1.0 + 0.3 * positions[:, 0] - 0.2 * positions[:, 1] + 0.1 * positions[:, 2]) ** degree
                independent = polynomial[constrained.independent_raw_nodes.numpy()]
                np.testing.assert_allclose(prolongation @ independent, polynomial, rtol=1.0e-4)

                # Fields do not jump across cells of different levels
                space = fem.make_collocated_function_space(constrained)
                field = space.make_field()
                field.dof_values = wp.array(
                    np.random.default_rng(degree).uniform(1.0, 2.0, constrained.topology.node_count()), dtype=float
                )
                jump_norm = fem.integrate(_jump_square, domain=fem.Sides(geometry), fields={"u": field.trace()})
                test.assertLess(abs(float(jump_norm)), 1.0e-6)

                # Assembly on the constrained space, including side gradients, is the reduction of the raw one
                raw_space = fem.make_collocated_function_space(raw)
                for domain, form in (
                    (fem.Cells(geometry), bilinear_form),
                    (fem.BoundarySides(geometry), bilinear_form),
                    (fem.Sides(geometry), _side_gradient_form),
                ):
                    np.testing.assert_allclose(
                        _dense(_mass_matrix(space, domain, device, form)),
                        prolongation.T @ _dense(_mass_matrix(raw_space, domain, device, form)) @ prolongation,
                        rtol=1.0e-4,
                        atol=1.0e-5,
                    )


def test_hanging_node_errors(test, device):
    with wp.ScopedDevice(device):
        geometry = _two_cell_geometry(device)
        with test.assertRaisesRegex(NotImplementedError, "continuous"):
            fem.make_conforming_basis_space(fem.make_polynomial_basis_space(geometry, degree=1, discontinuous=True))
        # Geometries that do not describe which sides cover each cell facet
        grid = fem.Grid3D(res=wp.vec3i(2))
        with test.assertRaises(NotImplementedError):
            fem.make_conforming_basis_space(fem.make_polynomial_basis_space(grid, degree=1))
        bspline = fem.make_polynomial_basis_space(grid, degree=2, element_basis=fem.ElementBasis.BSPLINE)
        with test.assertRaisesRegex(NotImplementedError, "interpolatory"):
            fem.make_conforming_basis_space(bspline)


@wp.func
def _shell_refinement(x: wp.vec3, center: wp.vec3, radius: float):
    distance = wp.length(x - center)
    # Carve a hole to expose facets, and refine near a spherical shell
    if distance < 0.4 * radius:
        return -1.0
    return wp.clamp(wp.abs(distance - radius) / radius, 0.0, 1.0)


def test_q1_graded_grid(test, device):
    with wp.ScopedDevice(device):
        res = 8
        voxels = np.stack(np.meshgrid(*[np.arange(res)] * 3, indexing="ij"), axis=-1).reshape(-1, 3)
        volume = wp.Volume.allocate_by_voxels(wp.array(voxels, dtype=wp.vec3i), voxel_size=1.0)
        refinement = fem.ImplicitField(
            domain=fem.Cells(fem.Nanogrid(volume)),
            func=_shell_refinement,
            values={"center": wp.vec3(0.5 * res), "radius": 0.3 * res},
        )
        geometry = fem.adaptive_nanogrid_from_field(volume, level_count=3, refinement_field=refinement, grading="face")
        raw = fem.make_polynomial_basis_space(geometry, degree=1)
        constrained = fem.make_conforming_basis_space(raw)

        raw_count = raw.topology.node_count()
        independent_count = constrained.topology.node_count()
        test.assertGreater(geometry.cell_count(), 1000)
        test.assertLess(independent_count, raw_count)

        # Independent nodes are increasing raw nodes and prolongate to themselves
        independent_raw_nodes = constrained.independent_raw_nodes.numpy()
        test.assertEqual(len(independent_raw_nodes), independent_count)
        test.assertTrue(np.all(np.diff(independent_raw_nodes) > 0))
        offsets = constrained.prolongation_offsets.numpy()
        indices = constrained.prolongation_indices.numpy()
        weights = constrained.prolongation_weights.numpy()
        test.assertEqual(len(offsets), raw_count + 1)
        np.testing.assert_array_equal(offsets[independent_raw_nodes + 1] - offsets[independent_raw_nodes], 1)
        np.testing.assert_array_equal(indices[offsets[independent_raw_nodes]], np.arange(independent_count))
        np.testing.assert_allclose(weights[offsets[independent_raw_nodes]], 1.0)

        # Rows are sorted by column and sum to one
        rows = np.repeat(np.arange(raw_count), np.diff(offsets))
        same_row = rows[1:] == rows[:-1]
        test.assertTrue(np.all(np.diff(indices)[same_row] > 0))
        np.testing.assert_allclose(np.bincount(rows, weights=weights, minlength=raw_count), 1.0, rtol=1.0e-6)

        space = fem.make_collocated_function_space(constrained)
        field = space.make_field()
        rng = np.random.default_rng(123)
        field.dof_values = wp.array(rng.uniform(1.0, 3.0, size=independent_count), dtype=float)
        jump_norm = fem.integrate(_jump_square, domain=fem.Sides(geometry), fields={"u": field.trace()})
        test.assertLess(abs(float(jump_norm)), 1.0e-6 * geometry.side_count())

        failures = wp.zeros(1, dtype=int)
        for domain, domain_field in ((fem.Cells(geometry), field), (fem.Sides(geometry), field.trace())):
            fem.integrate(
                _check_master_samples,
                domain=domain,
                fields={"u": domain_field},
                values={
                    "positions": constrained.node_positions(),
                    "expected": field.dof_values,
                    "period": wp.vec3(0.0),
                    "failures": failures,
                },
            )
        test.assertEqual(int(failures.numpy()[0]), 0)


def test_q1_prolongation_matrix(test, device):
    with wp.ScopedDevice(device):
        geometry = _two_cell_geometry(device)
        raw = fem.make_polynomial_basis_space(geometry, degree=1)
        constrained = fem.make_conforming_basis_space(raw)
        dense_prolongation = _dense_prolongation(constrained)
        raw_count = raw.topology.node_count()
        independent_count = constrained.topology.node_count()
        cells = fem.Cells(geometry)

        prolongation = constrained.prolongation_matrix()
        test.assertEqual((prolongation.nrow, prolongation.ncol), (raw_count, independent_count))
        test.assertEqual(prolongation.dtype, wp.float32)
        np.testing.assert_allclose(_dense(prolongation), dense_prolongation, rtol=1.0e-6)
        test.assertEqual(constrained.prolongation_matrix(wp.float64).dtype, wp.float64)

        raw_mass = _mass_matrix(fem.make_collocated_function_space(raw), cells, device)
        reduced_mass = _mass_matrix(fem.make_collocated_function_space(constrained), cells, device)
        restriction = wp.sparse.bsr_transposed(prolongation)
        np.testing.assert_allclose(
            _dense(wp.sparse.bsr_mm(restriction, wp.sparse.bsr_mm(raw_mass, prolongation))),
            _dense(reduced_mass),
            rtol=1.0e-5,
            atol=1.0e-6,
        )

        # A linear field on the independent nodes prolongates to the same linear field on the raw nodes.
        field = fem.make_collocated_function_space(constrained).make_field()
        fem.interpolate(_linear_init, dest=field)
        raw_values = prolongation @ field.dof_values
        raw_positions = raw.node_positions().numpy()
        np.testing.assert_allclose(
            raw_values.numpy(),
            1.0 + 0.5 * raw_positions[:, 0] - 0.25 * raw_positions[:, 1] + 0.125 * raw_positions[:, 2],
            rtol=1.0e-5,
        )

        vector_prolongation = constrained.prolongation_matrix(wp.vec3)
        test.assertEqual(vector_prolongation.block_shape, (3, 3))
        np.testing.assert_allclose(
            _dense(vector_prolongation), np.kron(dense_prolongation, np.eye(3)), rtol=1.0e-6, atol=0.0
        )
        vector_raw_mass = _mass_matrix(
            fem.make_collocated_function_space(raw, dtype=wp.vec3), cells, device, _vector_mass_form
        )
        vector_mass = _mass_matrix(
            fem.make_collocated_function_space(constrained, dtype=wp.vec3), cells, device, _vector_mass_form
        )
        np.testing.assert_allclose(
            _dense(
                wp.sparse.bsr_mm(
                    wp.sparse.bsr_transposed(vector_prolongation),
                    wp.sparse.bsr_mm(vector_raw_mass, vector_prolongation),
                )
            ),
            _dense(vector_mass),
            rtol=1.0e-5,
            atol=1.0e-6,
        )

        with test.assertRaisesRegex(TypeError, "floating-point"):
            constrained.prolongation_matrix(wp.int32)


def _raw_nodal_projector(raw, form, dtype, device, value_form=None):
    """Assemble a boundary projector, and optionally matching fixed values, with nodal assembly on the raw basis."""
    boundary = fem.BoundarySides(raw.geometry)
    raw_space = fem.make_collocated_function_space(raw, dtype=dtype)
    test = fem.make_test(raw_space, domain=boundary)
    projector = fem.integrate(
        form,
        fields={"u": fem.make_trial(raw_space, domain=boundary), "v": test},
        domain=boundary,
        assembly="nodal",
        output_dtype=float,
        device=device,
    )
    if value_form is None:
        return projector
    # Fixed values use the same nodal weights as the projector, as expected by ``normalize_dirichlet_projector``
    values = fem.integrate(
        value_form, fields={"v": test}, domain=boundary, assembly="nodal", output_dtype=dtype, device=device
    )
    return projector, values


def test_q1_dirichlet_across_interface(test, device):
    with wp.ScopedDevice(device):
        # Coarse cells fill [0, 8]^3, except [2, 6] x [0, 4] x [0, 4], which holds fine cells.
        # The z = 0 and y = 0 faces cross the coarse-to-fine interface.
        coarse = [
            (i, j, k)
            for i in range(0, 8, 2)
            for j in range(0, 8, 2)
            for k in range(0, 8, 2)
            if not (2 <= i < 6 and j < 4 and k < 4)
        ]
        fine = [(i, j, k) for i in range(2, 6) for j in range(4) for k in range(4)]
        points = coarse + fine
        geometry = fem.AdaptiveNanogrid.from_environment_voxels(
            points=wp.array(points, dtype=wp.vec3i, device=device),
            cell_levels=wp.array([1] * len(coarse) + [0] * len(fine), dtype=wp.uint8, device=device),
            point_envs=wp.zeros(len(points), dtype=int, device=device),
            env_count=1,
            level_count=2,
            device=device,
        )
        raw = fem.make_polynomial_basis_space(geometry, degree=1)
        constrained = fem.make_conforming_basis_space(raw)
        cells = fem.Cells(geometry)
        boundary = fem.BoundarySides(geometry)
        raw_positions = raw.node_positions().numpy()
        independent_positions = constrained.node_positions().numpy()
        exact_scalar = 1.0 + raw_positions @ np.array([0.5, -0.25, 0.125])

        independent_mask = np.zeros(raw.topology.node_count(), dtype=bool)
        independent_mask[constrained.independent_raw_nodes.numpy()] = True

        for dtype, projector_form, value_form, diffusion_form, flux_form, direction in (
            (float, _bottom_projector_form, _bottom_value_form, _diffusion_form, _linear_flux_form, np.ones(1)),
            (
                wp.vec3,
                _vector_bottom_projector_form,
                _vector_bottom_value_form,
                _vector_diffusion_form,
                _vector_linear_flux_form,
                np.array([1.0, 2.0, -1.0]),
            ),
        ):
            raw_projector, raw_fixed = _raw_nodal_projector(raw, projector_form, dtype, device, value_form)

            # The Dirichlet face contains hanging nodes, whose masters are on the face too
            raw_diagonal = wp.sparse.bsr_get_diag(raw_projector).numpy().reshape(raw.topology.node_count(), -1)
            constrained_raw = np.any(raw_diagonal != 0.0, axis=1)
            test.assertGreater(np.count_nonzero(constrained_raw & ~independent_mask), 0)

            projector = constrained.reduce_dirichlet_projector(raw_projector)
            fixed_value = constrained.gather_independent_values(raw_fixed)
            test.assertEqual(projector.nrow, constrained.topology.node_count())
            test.assertEqual(projector.dtype, raw_projector.dtype)
            test.assertEqual(fixed_value.dtype, raw_fixed.dtype)
            np.testing.assert_array_equal(
                wp.sparse.bsr_get_diag(projector).numpy().reshape(projector.nrow, -1),
                raw_diagonal[constrained.independent_raw_nodes.numpy()],
            )
            np.testing.assert_array_equal(
                fixed_value.numpy(), raw_fixed.numpy()[constrained.independent_raw_nodes.numpy()]
            )
            # Values given as a two-dimensional scalar array keep their layout
            flat_fixed = constrained.gather_independent_values(
                wp.array(raw_fixed.numpy().reshape(raw_fixed.shape[0], -1), dtype=float)
            )
            test.assertEqual(flat_fixed.ndim, 2)
            np.testing.assert_array_equal(flat_fixed.numpy(), fixed_value.numpy().reshape(flat_fixed.shape))

            space = fem.make_collocated_function_space(constrained, dtype=dtype)
            matrix = fem.integrate(
                diffusion_form,
                fields={"u": fem.make_trial(space, domain=cells), "v": fem.make_test(space, domain=cells)},
                domain=cells,
                device=device,
            )
            rhs = fem.integrate(
                flux_form, fields={"v": fem.make_test(space, domain=boundary)}, domain=boundary, output_dtype=dtype
            )
            fem.project_linear_system(matrix, rhs, projector, fixed_value)

            solution = np.linalg.solve(_dense(matrix), rhs.numpy().reshape(-1)).reshape(-1, len(direction))
            expected = (1.0 + independent_positions @ np.array([0.5, -0.25, 0.125]))[:, None] * direction
            np.testing.assert_allclose(solution, expected, rtol=1.0e-4, atol=1.0e-4)

            # Hanging nodes on the Dirichlet face take the exact value from their masters.
            raw_solution = (
                constrained.prolongation_matrix(dtype) @ wp.array(solution.squeeze(), dtype=dtype, device=device)
            ).numpy()
            np.testing.assert_allclose(
                raw_solution.reshape(-1, len(direction)),
                exact_scalar[:, None] * direction,
                rtol=1.0e-4,
                atol=1.0e-4,
            )


def test_q1_dirichlet_errors(test, device):
    with wp.ScopedDevice(device):
        geometry = _two_cell_geometry(device)
        raw = fem.make_polynomial_basis_space(geometry, degree=1)
        constrained = fem.make_conforming_basis_space(raw)

        with test.assertRaisesRegex(ValueError, "rows and columns"):
            constrained.reduce_dirichlet_projector(wp.sparse.bsr_identity(3, block_type=wp.float32, device=device))
        with test.assertRaisesRegex(ValueError, "entries"):
            constrained.gather_independent_values(wp.zeros(3, dtype=float))
        with test.assertRaisesRegex(ValueError, "dimensions"):
            constrained.gather_independent_values(wp.zeros((raw.topology.node_count(), 1, 1), dtype=float))


@fem.integrand
def _periodic_source(s: fem.Sample, domain: fem.Domain, v: fem.Field):
    x = domain(s)
    f = 8.0 * wp.pi * wp.pi * wp.sin(2.0 * wp.pi * x[0]) * wp.sin(2.0 * wp.pi * x[1])
    return f * v(s)


@fem.integrand
def _periodic_error(s: fem.Sample, domain: fem.Domain, u: fem.Field):
    x = domain(s)
    difference = u(s) - wp.sin(2.0 * wp.pi * x[0]) * wp.sin(2.0 * wp.pi * x[1])
    return difference * difference


def _unit_trimesh(res, device, perturb=False, shear=0.0):
    x = np.linspace(0.0, 1.0, res + 1)
    positions = np.transpose(np.meshgrid(x, x, indexing="ij"), axes=(1, 2, 0)).reshape(-1, 2)
    if perturb:
        # Move a vertex along the x = 1 face so that it has no partner on the x = 0 face
        positions[res * (res + 1) + 1, 1] += 0.25 / res
    # Map the unit square to the parallelogram spanned by (1, 0) and (shear, 1)
    positions[:, 0] += shear * positions[:, 1]
    return fem.Trimesh2D(
        tri_vertex_indices=wp.array(grid_to_tris(res, res), dtype=int, device=device),
        positions=wp.array(positions, dtype=wp.vec2, device=device),
    )


def _check_periodic_prolongation(test, basis, prolongation, axes):
    """Check a periodic prolongation of a basis on the unit square or cube, periodic along ``axes``."""
    positions = basis.node_positions().numpy()
    dense = _dense(prolongation)
    node_count = positions.shape[0]
    test.assertEqual(dense.shape[0], node_count)

    # Every node takes the value of exactly one independent node
    test.assertTrue(np.all(np.count_nonzero(dense, axis=1) == 1))
    np.testing.assert_array_equal(dense[dense != 0.0], 1.0)
    columns = np.argmax(dense != 0.0, axis=1)

    # High nodes are constrained; the anchor of each column is its lowest copy
    high = np.any(np.isclose(positions[:, axes], 1.0), axis=1)
    test.assertEqual(dense.shape[1], node_count - np.count_nonzero(high))
    anchors = np.full(dense.shape[1], -1)
    for node in range(node_count - 1, -1, -1):
        if not high[node]:
            test.assertEqual(anchors[columns[node]], -1)
            anchors[columns[node]] = node
    test.assertTrue(np.all(anchors >= 0))

    # Nodes and their anchors coincide modulo the periods
    offsets = positions - positions[anchors[columns]]
    np.testing.assert_allclose(offsets[:, axes], np.round(offsets[:, axes]), atol=1.0e-5)
    others = [axis for axis in range(positions.shape[1]) if axis not in axes]
    np.testing.assert_allclose(offsets[:, others], 0.0, atol=1.0e-5)
    return anchors


def test_periodic_prolongation(test, device):
    with wp.ScopedDevice(device):
        cases = [
            (fem.Grid2D(res=wp.vec2i(3)), 1, (0, 1)),
            (fem.Grid2D(res=wp.vec2i(3)), 2, (0, 1)),
            (fem.Grid2D(res=wp.vec2i(3), scalar_type=wp.float64), 1, (0,)),
            (fem.Grid3D(res=wp.vec3i(2)), 1, (0, 1, 2)),
            (fem.Grid3D(res=wp.vec3i(2)), 2, (0, 2)),
            (_unit_trimesh(3, device), 2, (0, 1)),
        ]
        for geometry, degree, axes in cases:
            raw = fem.make_polynomial_basis_space(geometry, degree=degree)
            dimension = geometry.dimension
            translations = [np.eye(dimension)[axis] for axis in axes]
            prolongation = fem.periodic_prolongation(raw, translations=translations)
            test.assertEqual(prolongation.dtype, geometry.scalar_type)
            anchors = _check_periodic_prolongation(test, raw, prolongation, list(axes))

            basis = fem.make_constrained_basis_space(raw, prolongation)
            test.assertEqual(basis.topology.node_count(), prolongation.ncol)
            np.testing.assert_array_equal(basis.independent_raw_nodes.numpy(), anchors)
            np.testing.assert_allclose(basis.node_positions().numpy(), raw.node_positions().numpy()[anchors])
            np.testing.assert_allclose(_dense(basis.prolongation_matrix()), _dense(prolongation))

        # Interpolation samples each independent node at its anchor, and at_node() at its anchor or a periodic image
        raw = fem.make_polynomial_basis_space(fem.Grid3D(res=wp.vec3i(2)), degree=1)
        prolongation = fem.periodic_prolongation(raw, translations=[wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 1.0)])
        basis = fem.make_constrained_basis_space(raw, prolongation)
        geometry = raw.geometry
        field = fem.make_collocated_function_space(basis).make_field()
        fem.interpolate(_linear_init, dest=field)
        positions = basis.node_positions()
        np.testing.assert_allclose(
            field.dof_values.numpy(), 1.0 + positions.numpy() @ np.array([0.5, -0.25, 0.125]), rtol=1.0e-5
        )
        for domain, domain_field in ((fem.Cells(geometry), field), (fem.BoundarySides(geometry), field.trace())):
            failures = wp.zeros(1, dtype=int)
            fem.integrate(
                _check_master_samples,
                domain=domain,
                fields={"u": domain_field},
                values={
                    "positions": positions,
                    "expected": field.dof_values,
                    "period": wp.vec3(1.0, 0.0, 1.0),
                    "failures": failures,
                },
            )
            test.assertEqual(int(failures.numpy()[0]), 0)


def test_periodic_geometry_device(test, device):
    """Periodic prolongations are built on the device holding the geometry, whatever the current device."""
    geometry = _unit_trimesh(3, device)
    raw = fem.make_polynomial_basis_space(geometry, degree=1)
    with wp.ScopedDevice("cpu"):
        prolongation = fem.periodic_prolongation(raw, translations=[wp.vec2(1.0, 0.0), wp.vec2(0.0, 1.0)])
    test.assertEqual(prolongation.device, wp.get_device(device))
    with wp.ScopedDevice(device):
        _check_periodic_prolongation(test, raw, prolongation, [0, 1])


def test_layered_prolongation_device(test, device):
    """Constraints layered on a basis from another device are built on the device of the new prolongation."""
    with wp.ScopedDevice("cpu"):
        raw = fem.make_polynomial_basis_space(fem.Grid2D(res=wp.vec2i(2)), degree=1)
        x_periodic = fem.periodic_prolongation(raw, translations=[wp.vec2(1.0, 0.0)])
        base = fem.make_constrained_basis_space(raw, x_periodic)
        y_periodic = _dense(fem.periodic_prolongation(base, translations=[wp.vec2(0.0, 1.0)]))

    rows, columns = np.nonzero(y_periodic)
    prolongation = _user_prolongation(rows, columns, y_periodic[rows, columns], *y_periodic.shape, device)
    basis = fem.make_constrained_basis_space(base, prolongation)
    test.assertIs(basis.raw_basis, raw)
    test.assertEqual(basis.prolongation_offsets.device, wp.get_device(device))
    np.testing.assert_allclose(_dense(basis.prolongation_matrix()), _dense(x_periodic) @ y_periodic, atol=1.0e-6)


def test_periodic_oblique(test, device):
    """Translations that are not perpendicular to the faces of a parallelogram cell."""
    with wp.ScopedDevice(device):
        res = 4
        edges = np.array([[1.0, 0.0], [0.5, 1.0]])
        for degree in (1, 2):
            raw = fem.make_polynomial_basis_space(_unit_trimesh(res, device, shear=0.5), degree=degree)
            positions = raw.node_positions().numpy()
            side_nodes = degree * res + 1
            for translations, independent_count in (
                (edges[:1], side_nodes * (side_nodes - 1)),
                (edges, (side_nodes - 1) ** 2),
            ):
                prolongation = fem.periodic_prolongation(raw, translations=[wp.vec2(*t) for t in translations])
                test.assertEqual(prolongation.ncol, independent_count)

                # Every node takes the value of a single copy shifted by a whole number of translations
                dense = _dense(prolongation)
                np.testing.assert_array_equal(np.count_nonzero(dense, axis=1), 1)
                np.testing.assert_allclose(dense.sum(axis=1), 1.0)
                rows, columns = np.nonzero(dense)
                # Each column is anchored at its first node
                anchors = np.array([np.flatnonzero(dense[:, column])[0] for column in range(prolongation.ncol)])
                shifts = np.linalg.solve(edges.T, (positions[rows] - positions[anchors[columns]]).T).T
                np.testing.assert_allclose(shifts, np.round(shifts), atol=1.0e-5)
                if len(translations) == 1:
                    np.testing.assert_allclose(shifts[:, 1], 0.0, atol=1.0e-5)


def test_periodic_errors(test, device):
    with wp.ScopedDevice(device):
        # The perturbed vertex has no partner and stays independent
        raw = fem.make_polynomial_basis_space(_unit_trimesh(3, device, perturb=True), degree=1)
        prolongation = fem.periodic_prolongation(raw, translations=[wp.vec2(1.0, 0.0)])
        test.assertEqual(prolongation.ncol, raw.topology.node_count() - 3)
        prolongation = fem.periodic_prolongation(raw, translations=[wp.vec2(0.0, 1.0)])
        test.assertEqual(prolongation.ncol, raw.topology.node_count() - 4)

        raw = fem.make_polynomial_basis_space(fem.Grid2D(res=wp.vec2i(2)), degree=1)
        for tolerance in (0.0, -1.0, 0.6):
            with test.assertRaisesRegex(ValueError, "tolerance"):
                fem.periodic_prolongation(raw, translations=[wp.vec2(1.0, 0.0)], tolerance=tolerance)
        with test.assertRaisesRegex(ValueError, "nonzero"):
            fem.periodic_prolongation(raw, translations=[wp.vec2(0.0, 0.0)])
        with test.assertRaisesRegex(ValueError, "components"):
            fem.periodic_prolongation(raw, translations=[wp.vec3(1.0, 0.0, 0.0)])
        with test.assertRaisesRegex(ValueError, "Cyclic"):
            fem.periodic_prolongation(raw, translations=[wp.vec2(1.0, 0.0), wp.vec2(-1.0, 0.0)])


def test_periodic_environments(test, device):
    with wp.ScopedDevice(device):
        # Environments share local coordinates: one voxel, and two voxels along x
        geo = fem.Nanogrid.from_environment_voxels(
            wp.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=wp.vec3i),
            wp.array([0, 1, 1], dtype=int),
            2,
            voxel_size=1.0,
        )
        raw = fem.make_polynomial_basis_space(geo, degree=1)
        prolongation = fem.periodic_prolongation(raw, translations=[wp.vec3(0.0, 1.0, 0.0), wp.vec3(0.0, 0.0, 1.0)])

        # Each environment is periodic on its own, leaving one node per x plane:
        # x = -0.5, 0.5 in the first environment and x = -0.5, 0.5, 1.5 in the second.
        # Each one takes the values of the 4 nodes of its plane in its environment only.
        dense = _dense(prolongation)
        test.assertTrue(np.all(np.count_nonzero(dense, axis=1) == 1))
        np.testing.assert_array_equal(np.count_nonzero(dense, axis=0), [4, 4, 4, 4, 4])
        positions = raw.node_positions().numpy()
        anchors = np.argmax(dense != 0.0, axis=0)
        np.testing.assert_array_equal(np.sort(positions[anchors, 0]), [-0.5, -0.5, 0.5, 0.5, 1.5])


def _periodic_poisson_error(geometry, degree):
    raw = fem.make_polynomial_basis_space(geometry, degree=degree)
    prolongation = fem.periodic_prolongation(raw, translations=[wp.vec2(1.0, 0.0), wp.vec2(0.0, 1.0)])
    basis = fem.make_constrained_basis_space(raw, prolongation)
    space = fem.make_collocated_function_space(basis)
    cells = fem.Cells(geometry)
    matrix = fem.integrate(
        _diffusion_form, fields={"u": fem.make_trial(space, domain=cells), "v": fem.make_test(space, domain=cells)}
    )
    rhs = fem.integrate(_periodic_source, fields={"v": fem.make_test(space, domain=cells)}).numpy()

    # Assembly on the periodic space is the reduction of the raw system
    raw_space = fem.make_collocated_function_space(raw)
    raw_matrix = fem.integrate(
        _diffusion_form,
        fields={"u": fem.make_trial(raw_space, domain=cells), "v": fem.make_test(raw_space, domain=cells)},
    )
    dense_prolongation = _dense(prolongation)
    dense_matrix = _dense(matrix)
    np.testing.assert_allclose(
        dense_matrix, dense_prolongation.T @ _dense(raw_matrix) @ dense_prolongation, rtol=1.0e-4, atol=1.0e-4
    )

    # Pin the node at the origin, where the solution vanishes, to remove the constant null space
    pinned = int(np.argmin(np.linalg.norm(basis.node_positions().numpy(), axis=1)))
    dense_matrix[pinned, :] = 0.0
    dense_matrix[:, pinned] = 0.0
    dense_matrix[pinned, pinned] = 1.0
    rhs[pinned] = 0.0
    field = space.make_field()
    field.dof_values = wp.array(np.linalg.solve(dense_matrix, rhs), dtype=float)
    quadrature = fem.RegularQuadrature(cells, order=2 * degree + 2)
    return np.sqrt(float(fem.integrate(_periodic_error, fields={"u": field}, quadrature=quadrature)))


def test_periodic_solve(test, device):
    with wp.ScopedDevice(device):
        for make_geometry, degree, min_ratio in (
            (lambda res: fem.Grid2D(res=wp.vec2i(res)), 1, 3.5),
            (lambda res: fem.Grid2D(res=wp.vec2i(res)), 2, 7.0),
            (lambda res: _unit_trimesh(res, device), 2, 7.0),
        ):
            coarse = _periodic_poisson_error(make_geometry(8), degree)
            fine = _periodic_poisson_error(make_geometry(16), degree)
            test.assertLess(fine, 0.02)
            test.assertGreater(coarse / fine, min_ratio)


def _user_prolongation(rows, columns, values, row_count, column_count, device, dtype=wp.float32):
    return wp.sparse.bsr_from_triplets(
        row_count,
        column_count,
        rows=wp.array(rows, dtype=int, device=device),
        columns=wp.array(columns, dtype=int, device=device),
        values=wp.array(values, dtype=dtype, device=device),
    )


def test_user_prolongation(test, device):
    with wp.ScopedDevice(device):
        geometry = fem.Grid2D(res=wp.vec2i(2))
        raw = fem.make_polynomial_basis_space(geometry, degree=1)
        raw_count = raw.topology.node_count()
        raw_space = fem.make_collocated_function_space(raw)
        cells = fem.Cells(geometry)
        raw_mass = _dense(_mass_matrix(raw_space, cells, device))
        raw_diffusion = _dense(_mass_matrix(raw_space, cells, device, _diffusion_form))
        raw_positions = raw.node_positions().numpy()

        # The identity reproduces the raw space
        identity = fem.make_constrained_basis_space(raw, wp.sparse.bsr_identity(raw_count, block_type=wp.float32))
        test.assertEqual(identity.topology.node_count(), raw_count)
        np.testing.assert_allclose(identity.node_positions().numpy(), raw_positions)
        identity_space = fem.make_collocated_function_space(identity)
        np.testing.assert_allclose(_dense(_mass_matrix(identity_space, cells, device)), raw_mass, rtol=1.0e-6)

        # Node 4 shares the value of node 0, and node 8 is a combination of nodes 3 and 5
        independent = [0, 1, 2, 3, 5, 6, 7]
        column_of = {node: column for column, node in enumerate(independent)}
        rows = [*independent, 4, 8, 8]
        columns = [*range(len(independent)), 0, column_of[3], column_of[5]]
        values = [1.0] * (len(independent) + 1) + [0.5, 0.25]
        expected = np.zeros((raw_count, len(independent)))
        expected[rows, columns] = values
        for dtype in (wp.float32, wp.float64):
            prolongation = _user_prolongation(rows, columns, values, raw_count, len(independent), device, dtype)
            for anchors, anchor_nodes in ((None, independent), ([4, 1, 2, 3, 5, 6, 7], [4, 1, 2, 3, 5, 6, 7])):
                basis = fem.make_constrained_basis_space(raw, prolongation, independent_raw_nodes=anchors)
                np.testing.assert_array_equal(basis.independent_raw_nodes.numpy(), anchor_nodes)
                np.testing.assert_allclose(basis.node_positions().numpy(), raw_positions[anchor_nodes])
                np.testing.assert_allclose(_dense(basis.prolongation_matrix()), expected)
                space = fem.make_collocated_function_space(basis)
                np.testing.assert_allclose(
                    _dense(_mass_matrix(space, cells, device)), expected.T @ raw_mass @ expected, atol=1.0e-6
                )
                np.testing.assert_allclose(
                    _dense(_mass_matrix(space, cells, device, _diffusion_form)),
                    expected.T @ raw_diffusion @ expected,
                    atol=1.0e-5,
                )

        with test.assertRaisesRegex(TypeError, "BSR matrix"):
            fem.make_constrained_basis_space(raw, expected)
        with test.assertRaisesRegex(ValueError, "scalar"):
            fem.make_constrained_basis_space(raw, wp.sparse.bsr_identity(raw_count, block_type=wp.mat22))
        with test.assertRaisesRegex(ValueError, "one row per base node"):
            fem.make_constrained_basis_space(raw, wp.sparse.bsr_identity(raw_count - 1, block_type=wp.float32))
        with test.assertRaisesRegex(ValueError, "no nonzero entries"):
            fem.make_constrained_basis_space(
                raw, _user_prolongation(range(raw_count), range(raw_count), [1.0] * raw_count, raw_count, 10, device)
            )
        with test.assertRaisesRegex(ValueError, "no anchor"):
            fem.make_constrained_basis_space(
                raw,
                _user_prolongation(range(raw_count), [*range(8), 8], [1.0] * 8 + [0.5], raw_count, raw_count, device),
            )
        with test.assertRaisesRegex(ValueError, "unit row"):
            fem.make_constrained_basis_space(
                raw,
                wp.sparse.bsr_identity(raw_count, block_type=wp.float32),
                independent_raw_nodes=[1, 0, *range(2, 9)],
            )
        with test.assertRaisesRegex(ValueError, "Raw node 8 .* unit row"):
            # The last row is empty, so its entries would start past the end of the matrix
            fem.make_constrained_basis_space(
                raw,
                _user_prolongation(range(8), range(8), [1.0] * 8, raw_count, 8, device),
                independent_raw_nodes=[*range(7), 8],
            )
        with test.assertRaisesRegex(ValueError, "max_nodes_per_element"):
            fem.make_constrained_basis_space(
                raw, wp.sparse.bsr_identity(raw_count, block_type=wp.float32), max_nodes_per_element=3
            )


def _slab_geometry(device, fine_y):
    """Coarse cells on [0, 4]^3, with fine cells on the slab z < 2 and y on the ``fine_y`` side.

    The slab crosses the domain along x, so refinement matches on the two x faces.
    """
    in_slab = lambda j, k: k < 2 and (j >= 2 if fine_y == "high" else j < 2)  # noqa: E731
    coarse = [(i, j, k) for i in range(0, 4, 2) for j in range(0, 4, 2) for k in range(0, 4, 2) if not in_slab(j, k)]
    fine = [(i, j, k) for i in range(4) for j in range(4) for k in range(4) if in_slab(j, k)]
    points = coarse + fine
    return fem.AdaptiveNanogrid.from_environment_voxels(
        points=wp.array(points, dtype=wp.vec3i, device=device),
        cell_levels=wp.array([1] * len(coarse) + [0] * len(fine), dtype=wp.uint8, device=device),
        point_envs=wp.zeros(len(points), dtype=int, device=device),
        env_count=1,
        level_count=2,
        device=device,
    )


def test_periodic_conforming(test, device):
    with wp.ScopedDevice(device):
        geometry = _slab_geometry(device, fine_y="high")
        raw = fem.make_polynomial_basis_space(geometry, degree=1)
        conforming = fem.make_conforming_basis_space(raw)
        hanging = fem.hanging_node_prolongation(raw)
        np.testing.assert_allclose(_dense(hanging), _dense(conforming.prolongation_matrix()))

        positions = conforming.node_positions().numpy()
        extent = positions.max(axis=0) - positions.min(axis=0)
        periodic = fem.periodic_prolongation(conforming, translations=[wp.vec3(extent[0], 0.0, 0.0)])
        basis = fem.make_constrained_basis_space(conforming, periodic)
        test.assertIs(basis.raw_basis, raw)
        test.assertEqual(basis.topology.node_count(), periodic.ncol)
        composed = _dense(hanging) @ _dense(periodic)
        np.testing.assert_allclose(_dense(basis.prolongation_matrix()), composed, atol=1.0e-6)

        # Random fields are continuous across refinement interfaces and equal across the seam
        space = fem.make_collocated_function_space(basis)
        field = space.make_field()
        rng = np.random.default_rng(42)
        field.dof_values = wp.array(rng.uniform(1.0, 3.0, size=basis.topology.node_count()), dtype=float)
        jump_norm = fem.integrate(_jump_square, domain=fem.Sides(geometry), fields={"u": field.trace()})
        test.assertLess(abs(float(jump_norm)), 1.0e-6 * geometry.side_count())

        # Side forms with gradients are the reduction of the raw ones
        sides = fem.Sides(geometry)
        raw_gradient = _dense(_mass_matrix(fem.make_collocated_function_space(raw), sides, device, _side_gradient_form))
        np.testing.assert_allclose(
            _dense(_mass_matrix(space, sides, device, _side_gradient_form)),
            composed.T @ raw_gradient @ composed,
            rtol=1.0e-4,
            atol=1.0e-5,
        )

        raw_values = (basis.prolongation_matrix() @ field.dof_values).numpy()
        raw_positions = raw.node_positions().numpy()
        low = np.nonzero(np.isclose(raw_positions[:, 0], raw_positions[:, 0].min()))[0]
        high = np.nonzero(np.isclose(raw_positions[:, 0], raw_positions[:, 0].max()))[0]
        test.assertGreater(len(high), 0)
        for node in high:
            partner = low[np.argmin(np.linalg.norm(raw_positions[low, 1:] - raw_positions[node, 1:], axis=1))]
            np.testing.assert_allclose(raw_positions[partner, 1:], raw_positions[node, 1:])
            test.assertAlmostEqual(float(raw_values[node]), float(raw_values[partner]), places=5)

        failures = wp.zeros(1, dtype=int)
        for domain, domain_field in ((fem.Cells(geometry), field), (fem.Sides(geometry), field.trace())):
            fem.integrate(
                _check_master_samples,
                domain=domain,
                fields={"u": domain_field},
                values={
                    "positions": basis.node_positions(),
                    "expected": field.dof_values,
                    "period": wp.vec3(extent[0], 0.0, 0.0),
                    "failures": failures,
                },
            )
        test.assertEqual(int(failures.numpy()[0]), 0)


def _check_at_node_forms(test, basis, domain):
    """Check forms evaluating test and trial functions at each other's nodes; return the number of moved samples."""
    space = fem.make_collocated_function_space(basis)
    test_field = fem.make_test(space, domain=domain)
    trial_field = fem.make_trial(space, domain=domain)

    # Each basis function is one at its own node and zero at the others, so
    # both forms are diagonal, holding the integral of each test function.
    # Samples moved to another element drop the other function, which then
    # evaluates to zero.
    expected = np.diag(fem.integrate(_test_where_node_stays, domain=domain, fields={"v": test_field}).numpy())
    for form in (_trial_at_test_node_form, _test_at_trial_node_form):
        matrix = fem.integrate(form, domain=domain, fields={"u": trial_field, "v": test_field}, assembly="generic")
        np.testing.assert_allclose(_dense(matrix), expected, atol=1.0e-6, err_msg=form.name)

    # Test functions are not defined at the nodes of a discrete field, even when they are sampled in another element
    field = space.make_field()
    if domain.element_kind == fem.ElementKind.SIDE:
        field = field.trace()
    got = fem.integrate(_test_at_field_nodes, domain=domain, fields={"w": field, "v": test_field}, assembly="generic")
    np.testing.assert_array_equal(got.numpy(), 0.0)

    moved = wp.zeros(1, dtype=int)
    fem.integrate(_count_moved_samples, domain=domain, fields={"v": test_field}, values={"moved": moved})
    return int(moved.numpy()[0])


def test_periodic_at_node(test, device):
    with wp.ScopedDevice(device):
        for degree in (1, 2):
            raw = fem.make_polynomial_basis_space(fem.Grid2D(res=wp.vec2i(3, 2)), degree=degree)
            prolongation = fem.periodic_prolongation(raw, translations=[wp.vec2(1.0, 0.0), wp.vec2(0.0, 1.0)])
            basis = fem.make_constrained_basis_space(raw, prolongation)
            # Every cell holds a periodic image of each of its nodes, so samples stay in the cell
            test.assertEqual(_check_at_node_forms(test, basis, fem.Cells(raw.geometry)), 0)
            if degree == 1:
                # Quadratic traces also list nodes inside the cells, which at_node() cannot place on a side
                _check_at_node_forms(test, basis, fem.Sides(raw.geometry))


def test_q1_at_node(test, device):
    with wp.ScopedDevice(device):
        geometry = _slab_geometry(device, fine_y="high")
        conforming = fem.make_conforming_basis_space(fem.make_polynomial_basis_space(geometry, degree=1))
        # Fine cells sample coarse masters in a coarse cell
        test.assertGreater(_check_at_node_forms(test, conforming, fem.Cells(geometry)), 0)
        _check_at_node_forms(test, conforming, fem.Sides(geometry))


class TestFemConstrainedSpace(unittest.TestCase):
    pass


add_function_test(
    TestFemConstrainedSpace, "test_q1_hanging_nodes", test_q1_hanging_nodes, devices=get_selected_cuda_test_devices()
)
add_function_test(
    TestFemConstrainedSpace, "test_q1_three_level", test_q1_three_level, devices=get_selected_cuda_test_devices()
)
add_function_test(
    TestFemConstrainedSpace,
    "test_high_degree_hanging_nodes",
    test_high_degree_hanging_nodes,
    devices=get_selected_cuda_test_devices(),
)
add_function_test(
    TestFemConstrainedSpace,
    "test_hanging_node_errors",
    test_hanging_node_errors,
    devices=get_selected_cuda_test_devices(),
)
add_function_test(
    TestFemConstrainedSpace, "test_q1_graded_grid", test_q1_graded_grid, devices=get_selected_cuda_test_devices()
)
add_function_test(
    TestFemConstrainedSpace,
    "test_q1_prolongation_matrix",
    test_q1_prolongation_matrix,
    devices=get_selected_cuda_test_devices(),
)
add_function_test(
    TestFemConstrainedSpace,
    "test_q1_dirichlet_across_interface",
    test_q1_dirichlet_across_interface,
    devices=get_selected_cuda_test_devices(),
)
add_function_test(
    TestFemConstrainedSpace,
    "test_q1_dirichlet_errors",
    test_q1_dirichlet_errors,
    devices=get_selected_cuda_test_devices(),
)
add_function_test(
    TestFemConstrainedSpace, "test_periodic_prolongation", test_periodic_prolongation, devices=get_test_devices()
)
add_function_test(
    TestFemConstrainedSpace,
    "test_periodic_geometry_device",
    test_periodic_geometry_device,
    devices=get_test_devices(),
)
add_function_test(
    TestFemConstrainedSpace,
    "test_layered_prolongation_device",
    test_layered_prolongation_device,
    devices=get_selected_cuda_test_devices(),
)
add_function_test(TestFemConstrainedSpace, "test_periodic_oblique", test_periodic_oblique, devices=get_test_devices())
add_function_test(TestFemConstrainedSpace, "test_periodic_errors", test_periodic_errors, devices=get_test_devices())
add_function_test(
    TestFemConstrainedSpace,
    "test_periodic_environments",
    test_periodic_environments,
    devices=get_selected_cuda_test_devices(),
)
add_function_test(TestFemConstrainedSpace, "test_periodic_solve", test_periodic_solve, devices=get_test_devices())
add_function_test(TestFemConstrainedSpace, "test_user_prolongation", test_user_prolongation, devices=get_test_devices())
add_function_test(
    TestFemConstrainedSpace,
    "test_periodic_conforming",
    test_periodic_conforming,
    devices=get_selected_cuda_test_devices(),
)
add_function_test(TestFemConstrainedSpace, "test_periodic_at_node", test_periodic_at_node, devices=get_test_devices())
add_function_test(TestFemConstrainedSpace, "test_q1_at_node", test_q1_at_node, devices=get_selected_cuda_test_devices())


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
