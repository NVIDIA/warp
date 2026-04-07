# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import warp as wp
import warp.fem as fem
from warp._src.fem.geometry.element import LinearEdge, Polynomial, Triangle
from warp._src.fem.operator import element_partition_index, node_partition_index
from warp.fem import Coords, Domain, Field, Sample, div, integrand
from warp.sparse import bsr_set_zero
from warp.tests.fem.utils import grad_field, linear_form, piecewise_constant
from warp.tests.unittest_utils import *


@integrand
def vector_divergence_form(s: Sample, u: Field, q: Field):
    return div(u, s) * q(s)


@wp.func
def _rbf_kernel_func(squared_dist: float, point_index: int, radius: float):
    return wp.exp(-squared_dist / (2.0 * radius * radius))


@wp.func
def _rbf_kernel_grad_func(squared_dist: float, point_index: int, radius: float):
    return -wp.exp(-squared_dist / (2.0 * radius * radius)) * (squared_dist / (2.0 * radius * radius))


@fem.integrand
def _bicubic(s: Sample, domain: Domain):
    x = domain(s)
    return wp.pow(x[0], 3.0) * wp.pow(x[1], 3.0)


@fem.integrand(kernel_options={"enable_backward": False})
def _value_at_node(domain: fem.Domain, s: fem.Sample, f: fem.Field, values: wp.array(dtype=float)):
    # lookup at node is ambiguous, check that partition_lookup retains sample on current partition
    s_partition = fem.partition_lookup(domain, domain(s))
    wp.expect_eq(s.element_index, s_partition.element_index)
    wp.expect_neq(element_partition_index(domain, s.element_index), fem.NULL_ELEMENT_INDEX)

    node_index = node_partition_index(f, s.qp_index)
    return values[node_index]


@fem.integrand(kernel_options={"enable_backward": False})
def _test_node_index(s: fem.Sample, u: fem.Field):
    wp.expect_eq(fem.node_index(u, s), s.qp_index)
    return 0.0


def test_regular_quadrature(test, device):
    for family in Polynomial:
        # test integrating monomials
        for degree in range(8):
            coords, weights = LinearEdge().instantiate_quadrature(degree, family=family)
            res = sum(w * pow(c[0], degree) for w, c in zip(weights, coords, strict=False))
            ref = 1.0 / (degree + 1)

            test.assertAlmostEqual(ref, res, places=4)

        # test integrating y^k1 (1 - x)^k2 on triangle using transformation to square
        for x_degree in range(4):
            for y_degree in range(4):
                coords, weights = Triangle().instantiate_quadrature(x_degree + y_degree, family=family)
                res = 0.5 * sum(
                    w * pow(1.0 - c[1], x_degree) * pow(c[2], y_degree) for w, c in zip(weights, coords, strict=False)
                )

                ref = 1.0 / ((x_degree + y_degree + 2) * (y_degree + 1))
                # print(x_degree, y_degree, family, len(coords), res, ref)
                test.assertAlmostEqual(ref, res, places=4)

    # test integrating y^k1 (1 - x)^k2 on triangle using direct formulas
    for x_degree in range(5):
        for y_degree in range(5):
            coords, weights = Triangle().instantiate_quadrature(x_degree + y_degree, family=None)
            res = 0.5 * sum(
                w * pow(1.0 - c[1], x_degree) * pow(c[2], y_degree) for w, c in zip(weights, coords, strict=False)
            )

            ref = 1.0 / ((x_degree + y_degree + 2) * (y_degree + 1))
            test.assertAlmostEqual(ref, res, places=4)


def test_point_basis(test, device):
    geo = fem.Grid2D(res=wp.vec2i(2))

    domain = fem.Cells(geo)

    quadrature = fem.RegularQuadrature(domain, order=2, family=fem.Polynomial.GAUSS_LEGENDRE)
    point_basis = fem.PointBasisSpace(quadrature)

    point_space = fem.make_collocated_function_space(point_basis)
    point_test = fem.make_test(point_space, domain=domain)

    # Sample at particle positions
    self_int = fem.integrate(linear_form, fields={"u": point_test}, assembly="nodal")
    test.assertAlmostEqual(np.sum(self_int.numpy()), 1.0, places=5)

    # Sampling outside of particle positions
    other_quadrature = fem.RegularQuadrature(domain, order=0, family=fem.Polynomial.GAUSS_LEGENDRE)
    other_int = fem.integrate(linear_form, quadrature=other_quadrature, fields={"u": point_test})

    test.assertAlmostEqual(np.sum(other_int.numpy()), 0.0, places=5)

    # test point basis with a finite-radius rbf kernel and varying points per element
    points = wp.array([[0.25, 0.33], [0.33, 0.25], [0.8, 0.8]], dtype=wp.vec2)
    neighbour_points_squared_dist = 2 * (0.25 - 0.33) ** 2  # dist between points in the same cell
    cell0_center_squared_dist = (0.25 - 0.33) ** 2  # dist with cell center
    cell3_center_squared_dist = 2.0 * (0.75 - 0.8) ** 2  # dist with cell center

    pic = fem.PicQuadrature(domain, positions=points)

    test.assertEqual(pic.active_cell_count(), 2)
    test.assertEqual(pic.total_point_count(), 3)
    test.assertEqual(pic.max_points_per_element(), 2)

    rbf_radius = 0.1
    point_basis = fem.PointBasisSpace(
        pic,
        kernel_func=_rbf_kernel_func,
        kernel_grad_func=_rbf_kernel_grad_func,
        kernel_values={"radius": rbf_radius},
        distance_space="world",
    )
    point_space = fem.make_collocated_function_space(point_basis)
    point_test = fem.make_test(point_space, domain=domain)
    test.assertEqual(point_test.space_restriction.node_count(), 3)

    self_int = fem.integrate(linear_form, fields={"u": point_test}, quadrature=pic)
    np.testing.assert_allclose(
        self_int.numpy(),
        [
            0.125 * (1.0 + _rbf_kernel_func(neighbour_points_squared_dist, 0, rbf_radius)),
            0.125 * (1.0 + _rbf_kernel_func(neighbour_points_squared_dist, 0, rbf_radius)),
            0.25,
        ],
        rtol=2.0e-7,
    )

    other_int = fem.integrate(linear_form, quadrature=other_quadrature, fields={"u": point_test})
    np.testing.assert_allclose(
        other_int.numpy(),
        [
            0.25 * _rbf_kernel_func(cell0_center_squared_dist, 0, rbf_radius),
            0.25 * _rbf_kernel_func(cell0_center_squared_dist, 0, rbf_radius),
            0.25 * _rbf_kernel_func(cell3_center_squared_dist, 0, rbf_radius),
        ],
    )

    # integration of bilinear form
    linear_vec_space = fem.make_polynomial_space(geo, dtype=wp.vec2)
    linear_test = fem.make_test(linear_vec_space)
    point_trial = fem.make_trial(point_space)

    mat = fem.integrate(
        vector_divergence_form, fields={"u": linear_test, "q": point_trial}, quadrature=pic, output_dtype=float
    )
    test.assertEqual(mat.nrow, 9)
    test.assertEqual(mat.ncol, 3)
    test.assertEqual(mat.nnz_sync(), 12)

    # test gradient
    bsr_set_zero(mat, rows_of_blocks=3, cols_of_blocks=3)
    fem.interpolate(
        grad_field,
        fields={"p": point_trial},
        dest=mat,
        dest_space=fem.make_collocated_function_space(point_basis, dtype=wp.vec2),
    )
    test.assertEqual(mat.nnz_sync(), 2)
    assert_np_equal(mat.columns[: mat.nnz].numpy(), [1, 0])

    delta_points = np.array([0.25 - 0.33, 0.33 - 0.25])
    ref_grad = 2.0 * _rbf_kernel_grad_func(neighbour_points_squared_dist, 0, rbf_radius) * delta_points
    np.testing.assert_allclose(
        mat.values[: mat.nnz].numpy().reshape(2, 2),
        np.array([ref_grad, -ref_grad], dtype=float),
        rtol=1.0e-6,
    )


def test_particle_quadratures(test, device):
    geo = fem.Grid2D(res=wp.vec2i(2))

    domain = fem.Cells(geo)

    # Explicit quadrature
    points, weights = domain.reference_element().prototype.instantiate_quadrature(
        order=4, family=fem.Polynomial.GAUSS_LEGENDRE
    )
    points_per_cell = len(points)

    points = points * domain.element_count()
    weights = weights * domain.element_count()

    points = wp.array(points, shape=(domain.element_count(), points_per_cell), dtype=Coords, device=device)
    weights = wp.array(weights, shape=(domain.element_count(), points_per_cell), dtype=float, device=device)

    explicit_quadrature = fem.ExplicitQuadrature(domain, points, weights)

    test.assertEqual(explicit_quadrature.max_points_per_element(), points_per_cell)
    test.assertEqual(explicit_quadrature.total_point_count(), points_per_cell * geo.cell_count())

    # test integration accuracy
    val = fem.integrate(_bicubic, quadrature=explicit_quadrature)
    test.assertAlmostEqual(val, 1.0 / 16, places=5)

    # test indexing validity
    arr = wp.empty(explicit_quadrature.total_point_count(), dtype=float)
    fem.interpolate(piecewise_constant, dest=arr, at=explicit_quadrature)
    assert_np_equal(arr.numpy(), np.arange(geo.cell_count()).repeat(points_per_cell))

    # PIC quadrature
    element_indices = wp.array([3, 3, 2], dtype=int, device=device)
    element_coords = wp.array(
        [
            [0.25, 0.5, 0.0],
            [0.5, 0.25, 0.0],
            [0.5, 0.5, 0.0],
        ],
        dtype=Coords,
        device=device,
    )

    pic_quadrature = fem.PicQuadrature(domain, positions=(element_indices, element_coords))

    test.assertEqual(pic_quadrature.max_points_per_element(), 2)
    test.assertEqual(pic_quadrature.total_point_count(), 3)
    test.assertEqual(pic_quadrature.active_cell_count(), 2)

    # Test integration accuracy
    val = fem.integrate(piecewise_constant, quadrature=pic_quadrature)
    test.assertAlmostEqual(val, 1.25, places=5)

    # Test differentiability of PicQuadrature w.r.t positions and measures
    points = wp.array([[0.25, 0.33], [0.33, 0.25], [0.8, 0.8]], dtype=wp.vec2, device=device, requires_grad=True)
    measures = wp.ones(3, dtype=float, device=device, requires_grad=True)

    tape = wp.Tape()
    with tape:
        pic = fem.PicQuadrature(domain, positions=points, measures=measures, requires_grad=True)

    pic.arg_value(device).cell_particle_coords.grad.fill_(1.0)
    pic.arg_value(device).cell_particle_fraction.grad.fill_(1.0)
    tape.backward()

    assert_np_equal(points.grad.numpy(), np.full((3, 2), 2.0))  # == 1.0 / cell_size
    assert_np_equal(measures.grad.numpy(), np.full(3, 4.0))  # == 1.0 / cell_area


def test_gimp_quadrature(test, device):
    # Test GIMP mode for PicQuadrature: particles spanning multiple cells

    geo = fem.Grid2D(res=wp.vec2i(2))
    domain = fem.Cells(geo)

    # Let's define 2 particles, each contributing to multiple cells (4 total evaluations)
    # Particle 0 overlaps cells 0, 1; Particle 1 overlaps cells 2, 3
    cell_indices = wp.array(
        [[0, 1], [2, 3]],  # shape=(2,2)
        dtype=int,
        device=device,
    )
    coords = wp.array(
        [
            [[0.25, 0.25, 0.0], [0.75, 0.25, 0.0]],  # Particle 0 in cell 0 and 1
            [[0.25, 0.75, 0.0], [0.75, 0.75, 0.0]],  # Particle 1 in cell 2 and 3
        ],
        dtype=Coords,
        device=device,
    )
    # Each "particle fraction" gives what fraction of the measure is in each cell overlap
    # For simplicity, split evenly: each overlapping cell gets 50% of the particle measure
    particle_fraction = wp.array(
        [[0.5, 0.5], [0.5, 0.5]],  # shape=(2,2)
        dtype=float,
        device=device,
    )

    # Shape: (num_particles, elements_per_particle)
    gimp_quadrature = fem.PicQuadrature(
        domain,
        (cell_indices, coords, particle_fraction),
    )

    # There are 2 unique particles, 4 total evaluation points (2*2 overlaps).
    test.assertEqual(gimp_quadrature.total_point_count(), 2)
    test.assertEqual(gimp_quadrature.evaluation_point_count(), 4)
    test.assertEqual(gimp_quadrature.max_points_per_element(), 1)  # each (2x2) cell gets at most 1

    # Now let's check the mapping: for each evaluation we should get the right cell, coords, and weight
    # The cells each have 1 GIMP "particle overlap"
    offsets = gimp_quadrature.cell_particle_offsets.numpy()
    indices = gimp_quadrature.cell_particle_indices.numpy()
    fractions = gimp_quadrature._cell_particle_fraction.numpy()
    coords_flat = gimp_quadrature._cell_particle_coords.numpy()
    # Build map: For cell in range(geo.cell_count()), collect indices into evaluation points
    found = {}
    for cell in range(geo.cell_count()):
        lo = offsets[cell]
        hi = offsets[cell + 1]
        found[cell] = []
        for eval_idx in range(lo, hi):
            pi = indices[eval_idx]  # particle index (row in our 2x2 arrays)
            # Find the per-particle entry this cell matched for
            for j in range(cell_indices.shape[1]):
                if cell_indices.numpy()[pi, j] == cell:
                    found[cell].append(
                        {"particle": pi, "coords": coords.numpy()[pi, j], "fraction": particle_fraction.numpy()[pi, j]}
                    )
    # Cells 0,1,2,3 each should get one particle overlap
    assert all(len(v) == 1 for v in found.values())
    test.assertTrue(np.allclose(found[0][0]["coords"], [0.25, 0.25, 0.0]))
    test.assertTrue(np.allclose(found[1][0]["coords"], [0.75, 0.25, 0.0]))
    test.assertTrue(np.allclose(found[2][0]["coords"], [0.25, 0.75, 0.0]))
    test.assertTrue(np.allclose(found[3][0]["coords"], [0.75, 0.75, 0.0]))
    test.assertAlmostEqual(found[0][0]["fraction"], 0.5)
    test.assertAlmostEqual(found[1][0]["fraction"], 0.5)
    test.assertAlmostEqual(found[2][0]["fraction"], 0.5)
    test.assertAlmostEqual(found[3][0]["fraction"], 0.5)

    # Check that integration with GIMP quadrature works and returns correct result with constant function
    total_volume = fem.integrate(piecewise_constant, quadrature=gimp_quadrature)
    test.assertAlmostEqual(total_volume, 0.5 * (0 + 1 + 2 + 3) / 4.0, places=5)


def test_nodal_quadrature(test, device):
    geo = fem.Grid2D(res=wp.vec2i(2))

    domain = fem.Cells(geo)

    space = fem.make_polynomial_space(geo, degree=2, discontinuous=True, family=fem.Polynomial.GAUSS_LEGENDRE)
    nodal_quadrature = fem.NodalQuadrature(domain, space)

    test.assertEqual(nodal_quadrature.max_points_per_element(), 9)
    test.assertEqual(nodal_quadrature.total_point_count(), 9 * geo.cell_count())

    val = fem.integrate(_bicubic, quadrature=nodal_quadrature)
    test.assertAlmostEqual(val, 1.0 / 16, places=5)

    # test accessing data associated to a given node

    piecewise_constant_space = fem.make_polynomial_space(geo, degree=1)
    geo_partition = fem.LinearGeometryPartition(geo, 2, 4)
    assert geo_partition.cell_count() == 1

    space_partition = fem.make_space_partition(
        space_topology=piecewise_constant_space.topology, geometry_partition=geo_partition, with_halo=False
    )

    field = fem.make_discrete_field(piecewise_constant_space, space_partition=space_partition)

    partition_domain = fem.Cells(geo_partition)
    partition_nodal_quadrature = fem.NodalQuadrature(partition_domain, piecewise_constant_space)

    partition_node_values = wp.full(value=5.0, shape=space_partition.node_count(), dtype=float)
    val = fem.integrate(
        _value_at_node,
        quadrature=partition_nodal_quadrature,
        fields={"f": field},
        values={"values": partition_node_values},
    )
    test.assertAlmostEqual(val, 5.0 / geo.cell_count(), places=5)

    u_test = fem.make_test(space)
    fem.integrate(_test_node_index, assembly="nodal", fields={"u": u_test})


# -- Device setup and test registration --


class TestFemQuadrature(unittest.TestCase):
    pass


add_function_test(TestFemQuadrature, "test_regular_quadrature", test_regular_quadrature)
add_function_test(TestFemQuadrature, "test_nodal_quadrature", test_nodal_quadrature)
add_function_test(TestFemQuadrature, "test_particle_quadratures", test_particle_quadratures)
add_function_test(TestFemQuadrature, "test_gimp_quadrature", test_gimp_quadrature)
add_function_test(TestFemQuadrature, "test_point_basis", test_point_basis)

if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
