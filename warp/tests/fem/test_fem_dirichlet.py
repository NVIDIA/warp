# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import warp as wp
import warp.fem as fem
from warp.sparse import (
    bsr_assign,
    bsr_copy,
    bsr_diag,
    bsr_from_triplets,
    bsr_get_diag,
    bsr_set_from_triplets,
    bsr_zeros,
)
from warp.tests.unittest_utils import *


@fem.integrand
def _mass_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.dot(u(s), v(s))


@fem.integrand
def _scalar_mass_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return u(s) * v(s)


@fem.integrand
def _stiffness_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.ddot(fem.grad(u, s), fem.grad(v, s)) + wp.dot(u(s), v(s))


@fem.integrand
def _scalar_stiffness_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.dot(fem.grad(u, s), fem.grad(v, s)) + u(s) * v(s)


@fem.integrand
def _normal_projector_form(s: fem.Sample, domain: fem.Domain, u: fem.Field, v: fem.Field):
    # Constrain only the normal direction, so projector blocks are not identities
    n = fem.normal(domain, s)
    return wp.dot(u(s), n) * wp.dot(v(s), n)


@fem.integrand
def _value_form(s: fem.Sample, domain: fem.Domain, v: fem.Field):
    x = fem.position(domain, s)
    return wp.dot(x, v(s))


def _bsr_to_dense(bsr):
    mat = np.zeros(bsr.shape)
    offsets = bsr.offsets.numpy()
    row_counts = None if bsr.row_counts is None else bsr.row_counts.numpy()
    columns = bsr.columns.numpy()
    values = bsr.values.numpy().reshape(-1, *bsr.block_shape)
    rows, cols = bsr.block_shape
    for row in range(bsr.nrow):
        beg = offsets[row]
        end = offsets[row + 1] if row_counts is None else beg + row_counts[row]
        for block in range(beg, end):
            col = columns[block]
            mat[row * rows : (row + 1) * rows, col * cols : (col + 1) * cols] += values[block]
    return mat


def _expected_projection(system, projector):
    complement = np.eye(system.shape[0]) - projector
    return complement @ system @ complement + projector


def _vector_system(device, scalar_type=wp.float32):
    geo = fem.Grid3D(res=wp.vec3i(3), scalar_type=scalar_type)
    vec_type = wp.vec3d if scalar_type == wp.float64 else wp.vec3
    space = fem.make_polynomial_space(geo, degree=1, dtype=vec_type)
    cells = fem.Cells(geo)
    boundary = fem.BoundarySides(geo)
    matrix = fem.integrate(
        _stiffness_form,
        fields={"u": fem.make_trial(space, domain=cells), "v": fem.make_test(space, domain=cells)},
        output_dtype=scalar_type,
        device=device,
    )
    bd_test = fem.make_test(space, domain=boundary)
    projector = fem.integrate(
        _normal_projector_form,
        fields={"u": fem.make_trial(space, domain=boundary), "v": bd_test},
        assembly="nodal",
        output_dtype=scalar_type,
        device=device,
    )
    values = fem.integrate(_value_form, fields={"v": bd_test}, assembly="nodal", output_dtype=vec_type, device=device)
    rhs = fem.integrate(
        _value_form, fields={"v": fem.make_test(space, domain=cells)}, output_dtype=vec_type, device=device
    )
    return matrix, projector, values, rhs


def test_project_block_diagonal(test, device):
    with wp.ScopedDevice(device):
        for scalar_type in (wp.float32, wp.float64):
            matrix, projector, values, rhs = _vector_system(device, scalar_type)
            fem.normalize_dirichlet_projector(projector, values)
            expected = _expected_projection(_bsr_to_dense(matrix), _bsr_to_dense(projector))
            tolerance = 1.0e-4 if scalar_type == wp.float32 else 1.0e-10

            for structure in ("block_diagonal", "general"):
                projected = bsr_copy(matrix)
                fem.project_system_matrix(projected, projector, projector_structure=structure)
                np.testing.assert_allclose(_bsr_to_dense(projected), expected, atol=tolerance * np.abs(expected).max())
                if structure != "general":
                    # The single-kernel projection keeps the pattern of the system matrix
                    test.assertEqual(projected.nnz_sync(), matrix.nnz_sync())

            # Right-hand side and matrix together match between methods
            results = {}
            for structure in ("block_diagonal", "general"):
                projected = bsr_copy(matrix)
                projected_rhs = wp.clone(rhs)
                fem.project_linear_system(
                    projected,
                    projected_rhs,
                    projector,
                    values,
                    normalize_projector=False,
                    projector_structure=structure,
                )
                results[structure] = (_bsr_to_dense(projected), projected_rhs.numpy())
            np.testing.assert_allclose(
                results["block_diagonal"][0],
                results["general"][0],
                atol=tolerance * np.abs(expected).max(),
            )
            np.testing.assert_array_equal(results["block_diagonal"][1], results["general"][1])


def test_project_block_diagonal_scalar_and_padded(test, device):
    with wp.ScopedDevice(device):
        geo = fem.Grid2D(res=wp.vec2i(4))
        space = fem.make_polynomial_space(geo, degree=2)
        cells = fem.Cells(geo)
        boundary = fem.BoundarySides(geo)
        matrix = fem.integrate(
            _scalar_stiffness_form,
            fields={"u": fem.make_trial(space, domain=cells), "v": fem.make_test(space, domain=cells)},
        )
        projector = fem.integrate(
            _scalar_mass_form,
            fields={"u": fem.make_trial(space, domain=boundary), "v": fem.make_test(space, domain=boundary)},
            assembly="nodal",
        )
        fem.normalize_dirichlet_projector(projector)
        expected = _expected_projection(_bsr_to_dense(matrix), _bsr_to_dense(projector))

        projected = bsr_copy(matrix)
        fem.project_system_matrix(projected, projector, projector_structure="block_diagonal")
        np.testing.assert_allclose(_bsr_to_dense(projected), expected, rtol=1.0e-5, atol=1.0e-5)

        # Padded storage: inactive slack blocks are skipped and left untouched
        row_capacity = int(np.diff(matrix.offsets.numpy()[: matrix.nrow + 1]).max()) + 4
        padded = bsr_zeros(
            matrix.nrow, matrix.ncol, block_type=matrix.dtype, topology="padded", row_capacity=row_capacity
        )
        bsr_assign(dest=padded, src=matrix, topology="padded")
        test.assertIsNotNone(padded.row_counts)
        test.assertEqual(padded.status_sync(), 0)
        fem.project_system_matrix(padded, projector, projector_structure="block_diagonal")
        np.testing.assert_allclose(_bsr_to_dense(padded), expected, rtol=1.0e-5, atol=1.0e-5)


def test_project_general(test, device):
    with wp.ScopedDevice(device):
        rng = np.random.default_rng(42)
        n = 6
        dense_system = rng.standard_normal((n, n))
        rows, cols = np.nonzero(np.abs(dense_system) > 0.3)
        system = bsr_from_triplets(
            n,
            n,
            wp.array(rows, dtype=int),
            wp.array(cols, dtype=int),
            wp.array(dense_system[rows, cols], dtype=wp.float64),
        )
        dense_system = _bsr_to_dense(system)

        # Off-diagonal projector block
        coupled_projector = bsr_from_triplets(
            n,
            n,
            wp.array([0, 1, 1, 2], dtype=int),
            wp.array([0, 1, 2, 2], dtype=int),
            wp.array([1.0, 0.5, 0.5, 0.5], dtype=wp.float64),
        )
        projected = bsr_copy(system)
        fem.project_system_matrix(projected, coupled_projector, projector_structure="general")
        np.testing.assert_allclose(
            _bsr_to_dense(projected), _expected_projection(dense_system, _bsr_to_dense(coupled_projector)), atol=1.0e-12
        )

        # Constrained row without a stored diagonal block
        keep = ~((rows == 0) & (cols == 0))
        system = bsr_from_triplets(
            n,
            n,
            wp.array(rows[keep], dtype=int),
            wp.array(cols[keep], dtype=int),
            wp.array(dense_system[rows[keep], cols[keep]], dtype=wp.float64),
        )
        dense_system = _bsr_to_dense(system)
        diagonal = np.zeros(n)
        diagonal[0] = 1.0
        projector = bsr_diag(wp.array(diagonal, dtype=wp.float64))
        projected = bsr_copy(system)
        fem.project_system_matrix(projected, projector, projector_structure="general")
        np.testing.assert_allclose(
            _bsr_to_dense(projected), _expected_projection(dense_system, np.diag(diagonal)), atol=1.0e-12
        )

        with test.assertRaisesRegex(ValueError, "Invalid projector structure"):
            fem.project_system_matrix(projected, projector, projector_structure="diagonal")
        with test.assertRaisesRegex(ValueError, "matching shapes"):
            fem.project_system_matrix(
                projected, bsr_diag(wp.ones(n + 1, dtype=wp.float64)), projector_structure="block_diagonal"
            )

        # Invalid arguments are rejected before the right-hand side and projector are modified
        rhs = wp.ones(n, dtype=wp.float64)
        scaled_projector = bsr_diag(wp.full(n, 2.0, dtype=wp.float64))
        larger_projector = bsr_diag(wp.full(n + 1, 2.0, dtype=wp.float64))
        with test.assertRaisesRegex(ValueError, "Invalid projector structure"):
            fem.project_linear_system(projected, rhs, scaled_projector, projector_structure="diagonal")
        with test.assertRaisesRegex(ValueError, "matching shapes"):
            fem.project_linear_system(projected, rhs, larger_projector, projector_structure="block_diagonal")
        np.testing.assert_array_equal(rhs.numpy(), np.ones(n))
        np.testing.assert_array_equal(bsr_get_diag(scaled_projector).numpy(), np.full(n, 2.0))
        np.testing.assert_array_equal(bsr_get_diag(larger_projector).numpy(), np.full(n + 1, 2.0))


def test_project_graph_capture(test, device):
    with wp.ScopedDevice(device):
        matrix, projector, values, rhs = _vector_system(device, wp.float64)
        fem.normalize_dirichlet_projector(projector, values)
        expected = _expected_projection(_bsr_to_dense(matrix), _bsr_to_dense(projector))

        def project(projected, projected_rhs):
            # The block-diagonal projection can be captured
            fem.project_linear_system(
                projected,
                projected_rhs,
                projector,
                values,
                normalize_projector=False,
                projector_structure="block_diagonal",
            )

        # Load the modules of the captured kernels beforehand, as older drivers cannot load them during capture
        project(bsr_copy(matrix), wp.clone(rhs))

        projected = bsr_copy(matrix)
        projected_rhs = wp.clone(rhs)
        with wp.ScopedCapture(force_module_load=False) as capture:
            project(projected, projected_rhs)
        wp.capture_launch(capture.graph)
        np.testing.assert_allclose(_bsr_to_dense(projected), expected, atol=1.0e-10 * np.abs(expected).max())

        reference_rhs = wp.clone(rhs)
        fem.project_system_rhs(matrix, reference_rhs, projector, values)
        np.testing.assert_allclose(projected_rhs.numpy(), reference_rhs.numpy(), rtol=1.0e-12, atol=1.0e-12)


def test_normalize_graph_capture(test, device):
    with wp.ScopedDevice(device):
        _matrix, projector, values, _rhs = _vector_system(device, wp.float64)
        expected_projector = bsr_copy(projector)
        expected_values = wp.clone(values)
        fem.normalize_dirichlet_projector(expected_projector, expected_values)

        # Normalization neither allocates nor reads back, so it can be captured; the
        # normalization above already loaded its module
        with wp.ScopedCapture(force_module_load=False) as capture:
            fem.normalize_dirichlet_projector(projector, values)
        wp.capture_launch(capture.graph)
        np.testing.assert_allclose(_bsr_to_dense(projector), _bsr_to_dense(expected_projector), atol=1.0e-12)
        np.testing.assert_allclose(values.numpy(), expected_values.numpy(), atol=1.0e-12)


def test_dirichlet_projector_off_diagonal(test, device):
    """Normalizing a projector only touches its diagonal blocks, whatever its structure."""
    rows = wp.array([0, 0, 1], dtype=int, device=device)
    columns = wp.array([0, 1, 1], dtype=int, device=device)
    blocks = np.stack([2.0 * np.eye(2), np.ones((2, 2)), 4.0 * np.eye(2)]).astype(np.float32)
    values = wp.array(blocks, dtype=wp.mat22, device=device)

    for topology in ("compact", "padded"):
        matrix = bsr_zeros(2, 2, wp.mat22, device=device, row_capacity=2 if topology == "padded" else None)
        bsr_set_from_triplets(matrix, rows, columns, values, topology=topology)
        fixed_values = wp.array([[1.0, 2.0], [3.0, 4.0]], dtype=wp.vec2, device=device)

        fem.normalize_dirichlet_projector(matrix, fixed_values)

        # Diagonal blocks become identities with their values scaled accordingly
        dense = matrix.values.numpy()
        active_columns = matrix.columns.numpy()
        offsets = matrix.offsets.numpy()
        for row in range(2):
            for block in range(offsets[row], offsets[row] + (1 if row == 1 else 2)):
                expected = np.eye(2) if active_columns[block] == row else np.ones((2, 2))
                np.testing.assert_allclose(dense[block], expected, rtol=1.0e-6)
        np.testing.assert_allclose(fixed_values.numpy(), [[0.5, 1.0], [0.75, 1.0]], rtol=1.0e-6)

    # An inactive padded slot may contain an arbitrary column index
    identity = np.eye(2, dtype=np.float32)
    diagonal = bsr_zeros(2, 2, wp.mat22, device=device, row_capacity=2)
    diagonal_rows = wp.array([0, 1], dtype=int, device=device)
    diagonal_values = wp.array(np.tile(identity, (2, 1, 1)), dtype=wp.mat22, device=device)
    bsr_set_from_triplets(diagonal, diagonal_rows, diagonal_rows, diagonal_values, topology="padded")
    offsets = diagonal.offsets.numpy()
    row_counts = diagonal.row_counts.numpy()
    padded_columns = diagonal.columns.numpy()
    for row in range(2):
        padded_columns[offsets[row] + row_counts[row]] = 1 - row
    diagonal.columns.assign(wp.array(padded_columns, dtype=int, device=device))
    fem.normalize_dirichlet_projector(diagonal)
    np.testing.assert_allclose(bsr_get_diag(diagonal).numpy(), np.tile(identity, (2, 1, 1)))


devices = get_test_devices()
cuda_devices = get_selected_cuda_test_devices()


class TestFemDirichlet(unittest.TestCase):
    pass


add_function_test(TestFemDirichlet, "test_project_block_diagonal", test_project_block_diagonal, devices=devices)
add_function_test(
    TestFemDirichlet,
    "test_project_block_diagonal_scalar_and_padded",
    test_project_block_diagonal_scalar_and_padded,
    devices=devices,
)
add_function_test(TestFemDirichlet, "test_project_general", test_project_general, devices=devices)
add_function_test(
    TestFemDirichlet, "test_dirichlet_projector_off_diagonal", test_dirichlet_projector_off_diagonal, devices=devices
)
add_function_test(TestFemDirichlet, "test_project_graph_capture", test_project_graph_capture, devices=cuda_devices)
add_function_test(TestFemDirichlet, "test_normalize_graph_capture", test_normalize_graph_capture, devices=cuda_devices)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
