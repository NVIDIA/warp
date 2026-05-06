# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np

import warp as wp
import warp.fem as fem
from warp.fem import (
    D,
    Domain,
    Field,
    Sample,
    curl,
    div,
    grad,
    integrand,
    normal,
)
from warp.sparse import bsr_set_zero, bsr_zeros
from warp.tests.fem.utils import (
    bilinear_field,
    bilinear_form,
    grad_field,
    linear_form,
    piecewise_constant,
    scaled_linear_form,
)
from warp.tests.unittest_utils import *

# -- Local integrands/kernels used only by tests in this file --


@wp.kernel
def atomic_sum(v: wp.array(dtype=float), sum: wp.array(dtype=float)):
    i = wp.tid()
    wp.atomic_add(sum, 0, v[i])


@integrand
def vector_divergence_form(s: Sample, u: Field, q: Field):
    return div(u, s) * q(s)


@integrand
def vector_grad_form(s: Sample, u: Field, q: Field):
    return wp.dot(u(s), grad(q, s))


@integrand
def vector_boundary_form(domain: Domain, s: Sample, u: Field, q: Field):
    return wp.dot(u(s) * q(s), normal(domain, s))


@integrand
def tensor_divergence_form(s: Sample, tau: Field, v: Field):
    return wp.dot(div(tau, s), v(s))


@integrand
def tensor_grad_form(s: Sample, tau: Field, v: Field):
    return wp.ddot(wp.transpose(tau(s)), grad(v, s))


@integrand
def tensor_boundary_form(domain: Domain, s: Sample, tau: Field, v: Field):
    return wp.dot(tau(s) * v(s), normal(domain, s))


@integrand
def grad_decomposition(s: Sample, u: Field, v: Field):
    return wp.length_sq(grad(u, s) * v(s) - D(u, s) * v(s) - wp.cross(curl(u, s), v(s)))


# -- Test functions --


def test_integrate_gradient(test, device):
    with wp.ScopedDevice(device):
        # Grid geometry
        geo = fem.Grid2D(res=wp.vec2i(5))

        # Domain and function spaces
        domain = fem.Cells(geometry=geo)
        quadrature = fem.RegularQuadrature(domain=domain, order=3)

        scalar_space = fem.make_polynomial_space(geo, degree=3)

        u = scalar_space.make_field()
        u.dof_values = wp.zeros_like(u.dof_values, requires_grad=True)

        result = wp.empty(dtype=wp.float32, shape=(1), requires_grad=True)
        tape = wp.Tape()

        # forward pass
        with tape:
            fem.integrate(linear_form, quadrature=quadrature, fields={"u": u}, output=result)
        tape.backward(result)

        test_field = fem.make_test(space=scalar_space, domain=domain)

        u_adj = wp.empty_like(u.dof_values, requires_grad=True)
        scale = wp.ones(1, requires_grad=True)
        loss = wp.zeros(1, requires_grad=True)

        tape2 = wp.Tape()
        with tape2:
            fem.integrate(
                scaled_linear_form,
                quadrature=quadrature,
                fields={"u": test_field},
                values={"scale": scale},
                assembly="generic",
                output=u_adj,
            )
            wp.launch(atomic_sum, dim=u_adj.shape, inputs=[u_adj, loss])

        # gradient of scalar integral w.r.t dofs should be equal to linear form vector
        assert_np_equal(u_adj.numpy(), u.dof_values.grad.numpy(), tol=1.0e-8)
        test.assertAlmostEqual(loss.numpy()[0], 1.0, places=4)

        # Check gradient of linear form vec w.r.t value params
        tape.zero()
        tape2.backward(loss=loss)

        test.assertAlmostEqual(loss.numpy()[0], scale.grad.numpy()[0], places=4)
        tape2.zero()
        test.assertEqual(scale.grad.numpy()[0], 0.0)

        # Same, with dispatched assembly
        tape2.reset()
        loss.zero_()
        with tape2:
            fem.integrate(
                scaled_linear_form,
                quadrature=quadrature,
                fields={"u": test_field},
                values={"scale": scale},
                assembly="dispatch",
                output=u_adj,
            )
            wp.launch(atomic_sum, dim=u_adj.shape, inputs=[u_adj, loss])
        tape2.backward(loss=loss)
        test.assertAlmostEqual(loss.numpy()[0], scale.grad.numpy()[0], places=4)


def test_interpolate_gradient(test, device):
    with wp.ScopedDevice(device):
        # Quad mesh with single element
        # so we can test gradient with respect to vertex positions
        positions = wp.array([[0.0, 0.0], [0.0, 2.0], [2.0, 0.0], [2.0, 2.0]], dtype=wp.vec2, requires_grad=True)
        quads = wp.array([[0, 2, 3, 1]], dtype=int)
        geo = fem.Quadmesh2D(quads, positions)

        # Quadratic scalar space
        scalar_space = fem.make_polynomial_space(geo, degree=2)

        # Point-based vector space
        # So we can test gradient with respect to interpolation point position
        point_coords = wp.array([[[0.5, 0.5, 0.0]]], dtype=fem.Coords, requires_grad=True)
        point_quadrature = fem.ExplicitQuadrature(
            domain=fem.Cells(geo), points=point_coords, weights=wp.array([[1.0]], dtype=float)
        )
        interpolation_nodes = fem.PointBasisSpace(point_quadrature)
        vector_space = fem.make_collocated_function_space(interpolation_nodes, dtype=wp.vec2)

        # Initialize scalar field with known function
        scalar_field = scalar_space.make_field()
        scalar_field.dof_values.requires_grad = True
        fem.interpolate(bilinear_field, dest=scalar_field)

        # Interpolate gradient at center point
        vector_field = vector_space.make_field()
        vector_field.dof_values.requires_grad = True
        vector_field_restriction = fem.make_restriction(vector_field)
        tape = wp.Tape()
        with tape:
            fem.interpolate(
                grad_field,
                dest=vector_field_restriction,
                fields={"p": scalar_field},
                kernel_options={"enable_backward": True},
            )

        assert_np_equal(vector_field.dof_values.numpy(), np.array([[1.0, 1.0]]))

        vector_field.dof_values.grad.assign([1.0, 0.0])
        tape.backward()

        assert_np_equal(scalar_field.dof_values.grad.numpy(), np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.5]))
        assert_np_equal(
            geo.positions.grad.numpy(),
            np.array(
                [
                    [0.25, 0.25],
                    [0.25, 0.25],
                    [-0.25, -0.25],
                    [-0.25, -0.25],
                ]
            ),
        )
        assert_np_equal(point_coords.grad.numpy(), np.array([[[0.0, 2.0, 0.0]]]))

        tape.zero()
        scalar_field.dof_values.grad.zero_()
        geo.positions.grad.zero_()
        point_coords.grad.zero_()

        vector_field.dof_values.grad.assign([0.0, 1.0])
        tape.backward()

        assert_np_equal(scalar_field.dof_values.grad.numpy(), np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.5, 0.0]))
        assert_np_equal(
            geo.positions.grad.numpy(),
            np.array(
                [
                    [0.25, 0.25],
                    [-0.25, -0.25],
                    [0.25, 0.25],
                    [-0.25, -0.25],
                ]
            ),
        )
        assert_np_equal(point_coords.grad.numpy(), np.array([[[2.0, 0.0, 0.0]]]))

        # Compare against jacobian at quadrature points
        scalar_trial = fem.make_trial(scalar_space)
        jacobian = bsr_zeros(
            rows_of_blocks=point_quadrature.total_point_count(),
            cols_of_blocks=scalar_space.node_count(),
            block_type=wp.types.matrix(shape=(2, 1), dtype=float),
        )
        fem.interpolate(
            grad_field,
            dest=jacobian,
            at=point_quadrature,
            fields={"p": scalar_trial},
            kernel_options={"enable_backward": False},
        )
        assert jacobian.nnz_sync() == 4  # one non-zero per cell center
        assert_np_equal((jacobian @ scalar_field.dof_values.grad).numpy(), [[0.0, 0.5]])

        # Compare against jacobian at nodes
        bsr_set_zero(jacobian)
        fem.interpolate(
            grad_field,
            dest=jacobian,
            dest_space=vector_space,
            at=vector_field_restriction.space_restriction,
            fields={"p": scalar_trial},
            kernel_options={"enable_backward": False},
        )
        assert jacobian.nnz_sync() == 4  # one non-zero per cell center
        assert_np_equal((jacobian @ scalar_field.dof_values.grad).numpy(), [[0.0, 0.5]])

        # Compare against jacobian at nodes (reduction="first")
        bsr_set_zero(jacobian)
        fem.interpolate(
            grad_field,
            dest=jacobian,
            dest_space=vector_space,
            at=vector_field_restriction.space_restriction,
            fields={"p": scalar_trial},
            kernel_options={"enable_backward": False},
            reduction="first",
        )
        assert jacobian.nnz_sync() == 4  # one non-zero per cell center
        assert_np_equal((jacobian @ scalar_field.dof_values.grad).numpy(), [[0.0, 0.5]])


def test_vector_divergence_theorem(test, device):
    rng = np.random.default_rng(123)

    with wp.ScopedDevice(device):
        # Grid geometry
        geo = fem.Grid2D(res=wp.vec2i(5))

        # Domain and function spaces
        interior = fem.Cells(geometry=geo)
        boundary = fem.BoundarySides(geometry=geo)

        vector_space = fem.make_polynomial_space(geo, degree=2, dtype=wp.vec2)
        scalar_space = fem.make_polynomial_space(geo, degree=1, dtype=float)

        u = vector_space.make_field()
        u.dof_values = rng.random(size=(u.dof_values.shape[0], 2))

        # Divergence theorem
        constant_one = scalar_space.make_field()
        constant_one.dof_values.fill_(1.0)

        interior_quadrature = fem.RegularQuadrature(domain=interior, order=vector_space.degree)
        boundary_quadrature = fem.RegularQuadrature(domain=boundary, order=vector_space.degree)
        div_int = fem.integrate(
            vector_divergence_form,
            quadrature=interior_quadrature,
            fields={"u": u, "q": constant_one},
            kernel_options={"enable_backward": False},
        )
        boundary_int = fem.integrate(
            vector_boundary_form,
            quadrature=boundary_quadrature,
            fields={"u": u.trace(), "q": constant_one.trace()},
            kernel_options={"enable_backward": False},
        )

        test.assertAlmostEqual(div_int, boundary_int, places=5)

        # Integration by parts
        q = scalar_space.make_field()
        q.dof_values = rng.random(size=q.dof_values.shape[0])

        interior_quadrature = fem.RegularQuadrature(domain=interior, order=vector_space.degree + scalar_space.degree)
        boundary_quadrature = fem.RegularQuadrature(domain=boundary, order=vector_space.degree + scalar_space.degree)
        div_int = fem.integrate(
            vector_divergence_form,
            quadrature=interior_quadrature,
            fields={"u": u, "q": q},
            kernel_options={"enable_backward": False},
        )
        grad_int = fem.integrate(
            vector_grad_form,
            quadrature=interior_quadrature,
            fields={"u": u, "q": q},
            kernel_options={"enable_backward": False},
        )
        boundary_int = fem.integrate(
            vector_boundary_form,
            quadrature=boundary_quadrature,
            fields={"u": u.trace(), "q": q.trace()},
            kernel_options={"enable_backward": False},
        )

        test.assertAlmostEqual(div_int + grad_int, boundary_int, places=5)


def test_tensor_divergence_theorem(test, device):
    rng = np.random.default_rng(123)

    with wp.ScopedDevice(device):
        # Grid geometry
        geo = fem.Grid2D(res=wp.vec2i(5))

        # Domain and function spaces
        interior = fem.Cells(geometry=geo)
        boundary = fem.BoundarySides(geometry=geo)

        tensor_space = fem.make_polynomial_space(geo, degree=2, dtype=wp.mat22)
        vector_space = fem.make_polynomial_space(geo, degree=1, dtype=wp.vec2)

        tau = tensor_space.make_field()
        tau.dof_values = rng.random(size=(tau.dof_values.shape[0], 2, 2))

        # Divergence theorem
        constant_vec = vector_space.make_field()
        constant_vec.dof_values.fill_(wp.vec2(0.5, 2.0))

        interior_quadrature = fem.RegularQuadrature(domain=interior, order=tensor_space.degree)
        boundary_quadrature = fem.RegularQuadrature(domain=boundary, order=tensor_space.degree)
        div_int = fem.integrate(
            tensor_divergence_form,
            quadrature=interior_quadrature,
            fields={"tau": tau, "v": constant_vec},
            kernel_options={"enable_backward": False},
        )
        boundary_int = fem.integrate(
            tensor_boundary_form,
            quadrature=boundary_quadrature,
            fields={"tau": tau.trace(), "v": constant_vec.trace()},
            kernel_options={"enable_backward": False},
        )

        test.assertAlmostEqual(div_int, boundary_int, places=5)

        # Integration by parts
        v = vector_space.make_field()
        v.dof_values = rng.random(size=(v.dof_values.shape[0], 2))

        interior_quadrature = fem.RegularQuadrature(domain=interior, order=tensor_space.degree + vector_space.degree)
        boundary_quadrature = fem.RegularQuadrature(domain=boundary, order=tensor_space.degree + vector_space.degree)
        div_int = fem.integrate(
            tensor_divergence_form,
            quadrature=interior_quadrature,
            fields={"tau": tau, "v": v},
            kernel_options={"enable_backward": False},
        )
        grad_int = fem.integrate(
            tensor_grad_form,
            quadrature=interior_quadrature,
            fields={"tau": tau, "v": v},
            kernel_options={"enable_backward": False},
        )
        boundary_int = fem.integrate(
            tensor_boundary_form,
            quadrature=boundary_quadrature,
            fields={"tau": tau.trace(), "v": v.trace()},
            kernel_options={"enable_backward": False},
        )

        test.assertAlmostEqual(div_int + grad_int, boundary_int, places=5)


def test_grad_decomposition(test, device):
    rng = np.random.default_rng(123)

    with wp.ScopedDevice(device):
        # Grid geometry
        geo = fem.Grid3D(res=wp.vec3i(5))

        # Domain and function spaces
        domain = fem.Cells(geometry=geo)
        quadrature = fem.RegularQuadrature(domain=domain, order=4)

        vector_space = fem.make_polynomial_space(geo, degree=2, dtype=wp.vec3)
        u = vector_space.make_field()

        u.dof_values = rng.random(size=(u.dof_values.shape[0], 3))

        err = fem.integrate(grad_decomposition, quadrature=quadrature, fields={"u": u, "v": u})
        test.assertLess(err, 1.0e-8)


def test_interpolate_reduction(test, device):
    # Test reduction modes of fem.interpolate() on a discontinuous field
    with wp.ScopedDevice(device):
        N = 2  # 2x2 cells → 3x3 vertices
        geo = fem.Grid2D(res=wp.vec2i(N))
        space = fem.make_polynomial_space(geo, degree=1)  # Q1 nodes at vertices
        field = space.make_field()

        # Helper: compute neighbor element indices for a vertex (ix,iy)
        def neighbor_elements(ix, iy):
            iset = [i for i in (ix - 1, ix) if 0 <= i < N]
            jset = [j for j in (iy - 1, iy) if 0 <= j < N]
            return [j * N + i for j in jset for i in iset]

        # Precompute expectations for each reduction mode
        node_count = (N + 1) * (N + 1)

        def expected_array(mode):
            out = np.zeros(node_count, dtype=float)
            for iy in range(N + 1):
                for ix in range(N + 1):
                    neigh = neighbor_elements(ix, iy)
                    idx = iy * (N + 1) + ix  # row-major vertex indexing
                    if len(neigh) == 0:
                        out[idx] = 0.0
                        continue
                    if mode == "sum":
                        out[idx] = float(sum(neigh))
                    elif mode == "mean" or mode == "weighted_average":
                        out[idx] = float(sum(neigh)) / float(len(neigh))
                    elif mode == "max":
                        out[idx] = float(max(neigh))
                    elif mode == "min":
                        out[idx] = float(min(neigh))
                    elif mode == "first":
                        # Deterministic order in restriction: expect smallest element index first
                        out[idx] = float(min(neigh))
                    else:
                        raise ValueError("Unknown mode")
            return out

        # Run interpolation for each reduction and compare
        for mode in ("weighted_average", "mean", "sum", "max", "min", "first"):
            field.dof_values.zero_()
            fem.interpolate(piecewise_constant, dest=field, reduction=mode)
            got = field.dof_values.numpy()
            exp = expected_array(mode)
            assert_np_equal(got, exp, tol=1.0e-6)


def test_integrate_high_order(test, device):
    with wp.ScopedDevice(device):
        geo = fem.Grid3D(res=(1, 1, 1))
        space = fem.make_polynomial_space(geo, degree=4)
        test_field = fem.make_test(space)
        trial_field = fem.make_trial(space)

        # compare consistency of tile-based "dispatch" assembly and generic
        v0 = fem.integrate(
            linear_form, fields={"u": test_field}, assembly="dispatch", kernel_options={"enable_backward": False}
        )
        v1 = fem.integrate(
            linear_form, fields={"u": test_field}, assembly="generic", kernel_options={"enable_backward": False}
        )

        assert_np_equal(v0.numpy(), v1.numpy(), tol=1.0e-6)

        h0 = fem.integrate(
            bilinear_form,
            fields={"v": test_field, "u": trial_field},
            assembly="dispatch",
            kernel_options={"enable_backward": False},
        )
        h1 = fem.integrate(
            bilinear_form,
            fields={"v": test_field, "u": trial_field},
            assembly="generic",
            kernel_options={"enable_backward": False},
        )

        h0_nnz = h0.nnz_sync()
        h1_nnz = h1.nnz_sync()
        assert h0.shape == h1.shape
        assert h0_nnz == h1_nnz
        assert_array_equal(h0.offsets[: h0.nrow + 1], h1.offsets[: h1.nrow + 1])
        assert_array_equal(h0.columns[:h0_nnz], h1.columns[:h1_nnz])
        assert_np_equal(h0.values[:h0_nnz].numpy(), h1.values[:h1_nnz].numpy(), tol=1.0e-6)


def test_capturability(test, device):
    A = bsr_zeros(0, 0, block_type=wp.float32, device=device)

    def test_body():
        geo = fem.Grid3D(res=(4, 4, 4))
        space = fem.make_polynomial_space(geo, degree=1)

        cell_mask = wp.zeros(geo.cell_count(), dtype=int, device=device)
        cell_mask[16:32].fill_(1)

        geo_partition = fem.ExplicitGeometryPartition(geo, cell_mask, max_cell_count=32, max_side_count=0)
        space_partition = fem.make_space_partition(
            space_topology=space.topology, geometry_partition=geo_partition, with_halo=False, max_node_count=64
        )

        test_field = fem.make_test(space, space_partition=space_partition)
        trial_field = fem.make_trial(space, space_partition=space_partition)
        bsr_set_zero(
            A,
            rows_of_blocks=test_field.space_partition.node_count(),
            cols_of_blocks=trial_field.space_partition.node_count(),
        )
        fem.integrate(
            bilinear_form,
            fields={"v": test_field, "u": trial_field},
            kernel_options={"enable_backward": False},
            output=A,
        )

    with wp.ScopedDevice(device):
        test_body()
        assert A.shape == (64, 64)
        nnz_ref = A.nnz_sync()
        values_ref = A.values.numpy()[:nnz_ref]
        columns_ref = A.columns.numpy()[:nnz_ref]
        bsr_set_zero(A)
        assert A.nnz_sync() == 0

        with wp.ScopedCapture() as capture:
            test_body()
        wp.capture_launch(capture.graph)
        assert A.nnz_sync() == nnz_ref
        assert_np_equal(A.values.numpy()[:nnz_ref], values_ref)
        assert_np_equal(A.columns.numpy()[:nnz_ref], columns_ref)


# -- Device setup and test registration --

devices = get_test_devices()
cuda_devices = get_selected_cuda_test_devices()
cuda_devices_with_mempool = get_selected_cuda_test_devices_with_mempool()


class TestFemIntegrate(unittest.TestCase):
    pass


add_function_test(TestFemIntegrate, "test_integrate_gradient", test_integrate_gradient, devices=devices)
add_function_test(TestFemIntegrate, "test_interpolate_gradient", test_interpolate_gradient, devices=devices)
add_function_test(TestFemIntegrate, "test_vector_divergence_theorem", test_vector_divergence_theorem, devices=devices)
add_function_test(TestFemIntegrate, "test_tensor_divergence_theorem", test_tensor_divergence_theorem, devices=devices)
add_function_test(TestFemIntegrate, "test_grad_decomposition", test_grad_decomposition, devices=devices)
add_function_test(TestFemIntegrate, "test_integrate_high_order", test_integrate_high_order, devices=cuda_devices)
add_function_test(TestFemIntegrate, "test_interpolate_reduction", test_interpolate_reduction, devices=devices)
add_function_test(TestFemIntegrate, "test_capturability", test_capturability, devices=cuda_devices_with_mempool)

if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
