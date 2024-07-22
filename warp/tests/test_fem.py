# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import math
import unittest

import numpy as np

import warp as wp
import warp.fem as fem
from warp.fem import Coords, D, Domain, Field, Sample, curl, div, grad, integrand, normal
from warp.fem.cache import dynamic_kernel
from warp.fem.geometry import DeformedGeometry
from warp.fem.geometry.closest_point import project_on_tet_at_origin, project_on_tri_at_origin
from warp.fem.space import shape
from warp.fem.types import make_free_sample
from warp.fem.utils import (
    grid_to_hexes,
    grid_to_quads,
    grid_to_tets,
    grid_to_tris,
    inverse_qr,
    symmetric_eigenvalues_qr,
)
from warp.tests.unittest_utils import *


@integrand
def linear_form(s: Sample, u: Field):
    return u(s)


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

        result = wp.empty(dtype=wp.float64, shape=(1), requires_grad=True)

        tape = wp.Tape()

        # forward pass
        with tape:
            fem.integrate(linear_form, quadrature=quadrature, fields={"u": u}, output=result)

        tape.backward(result)

        test_field = fem.make_test(space=scalar_space, domain=domain)
        rhs = fem.integrate(linear_form, quadrature=quadrature, fields={"u": test_field})

        err = np.linalg.norm(rhs.numpy() - u.dof_values.grad.numpy())
        test.assertLess(err, 1.0e-8)


@fem.integrand
def bilinear_field(s: fem.Sample, domain: fem.Domain):
    x = domain(s)
    return x[0] * x[1]


@fem.integrand
def grad_field(s: fem.Sample, p: fem.Field):
    return fem.grad(p, s)


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
        # So we can test gradient with respect to inteprolation point position
        point_coords = wp.array([[[0.5, 0.5, 0.0]]], dtype=fem.Coords, requires_grad=True)
        interpolation_nodes = fem.PointBasisSpace(
            fem.ExplicitQuadrature(domain=fem.Cells(geo), points=point_coords, weights=wp.array([[1.0]], dtype=float))
        )
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

        assert_np_equal(scalar_field.dof_values.grad.numpy(), np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.5, 0.0]))
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

        assert_np_equal(scalar_field.dof_values.grad.numpy(), np.array([0.0, 0.0, 0.0, 0.0, -0.5, 0.0, 0.5, 0.0, 0.0]))
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


@integrand
def vector_divergence_form(s: Sample, u: Field, q: Field):
    return div(u, s) * q(s)


@integrand
def vector_grad_form(s: Sample, u: Field, q: Field):
    return wp.dot(u(s), grad(q, s))


@integrand
def vector_boundary_form(domain: Domain, s: Sample, u: Field, q: Field):
    return wp.dot(u(s) * q(s), normal(domain, s))


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


@integrand
def tensor_divergence_form(s: Sample, tau: Field, v: Field):
    return wp.dot(div(tau, s), v(s))


@integrand
def tensor_grad_form(s: Sample, tau: Field, v: Field):
    return wp.ddot(wp.transpose(tau(s)), grad(v, s))


@integrand
def tensor_boundary_form(domain: Domain, s: Sample, tau: Field, v: Field):
    return wp.dot(tau(s) * v(s), normal(domain, s))


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


@integrand
def grad_decomposition(s: Sample, u: Field, v: Field):
    return wp.length_sq(grad(u, s) * v(s) - D(u, s) * v(s) - wp.cross(curl(u, s), v(s)))


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


def _gen_trimesh(N):
    x = np.linspace(0.0, 1.0, N + 1)
    y = np.linspace(0.0, 1.0, N + 1)

    positions = np.transpose(np.meshgrid(x, y, indexing="ij")).reshape(-1, 2)

    vidx = grid_to_tris(N, N)

    return wp.array(positions, dtype=wp.vec2), wp.array(vidx, dtype=int)


def _gen_quadmesh(N):
    x = np.linspace(0.0, 1.0, N + 1)
    y = np.linspace(0.0, 1.0, N + 1)

    positions = np.transpose(np.meshgrid(x, y, indexing="ij")).reshape(-1, 2)

    vidx = grid_to_quads(N, N)

    return wp.array(positions, dtype=wp.vec2), wp.array(vidx, dtype=int)


def _gen_tetmesh(N):
    x = np.linspace(0.0, 1.0, N + 1)
    y = np.linspace(0.0, 1.0, N + 1)
    z = np.linspace(0.0, 1.0, N + 1)

    positions = np.transpose(np.meshgrid(x, y, z, indexing="ij")).reshape(-1, 3)

    vidx = grid_to_tets(N, N, N)

    return wp.array(positions, dtype=wp.vec3), wp.array(vidx, dtype=int)


def _gen_hexmesh(N):
    x = np.linspace(0.0, 1.0, N + 1)
    y = np.linspace(0.0, 1.0, N + 1)
    z = np.linspace(0.0, 1.0, N + 1)

    positions = np.transpose(np.meshgrid(x, y, z, indexing="ij")).reshape(-1, 3)

    vidx = grid_to_hexes(N, N, N)

    return wp.array(positions, dtype=wp.vec3), wp.array(vidx, dtype=int)


def _launch_test_geometry_kernel(geo: fem.Geometry, device):
    @dynamic_kernel(suffix=geo.name, kernel_options={"enable_backward": False})
    def test_geo_cells_kernel(
        cell_arg: geo.CellArg,
        qps: wp.array(dtype=Coords),
        qp_weights: wp.array(dtype=float),
        cell_measures: wp.array(dtype=float),
    ):
        cell_index, q = wp.tid()

        coords = qps[q]
        s = make_free_sample(cell_index, coords)

        wp.atomic_add(cell_measures, cell_index, geo.cell_measure(cell_arg, s) * qp_weights[q])

    REF_MEASURE = geo.reference_side().measure()

    @dynamic_kernel(suffix=geo.name, kernel_options={"enable_backward": False, "max_unroll": 1})
    def test_geo_sides_kernel(
        side_arg: geo.SideArg,
        qps: wp.array(dtype=Coords),
        qp_weights: wp.array(dtype=float),
        side_measures: wp.array(dtype=float),
    ):
        side_index, q = wp.tid()

        coords = qps[q]
        s = make_free_sample(side_index, coords)

        cell_arg = geo.side_to_cell_arg(side_arg)
        inner_cell_index = geo.side_inner_cell_index(side_arg, side_index)
        outer_cell_index = geo.side_outer_cell_index(side_arg, side_index)
        inner_cell_coords = geo.side_inner_cell_coords(side_arg, side_index, coords)
        outer_cell_coords = geo.side_outer_cell_coords(side_arg, side_index, coords)

        inner_s = make_free_sample(inner_cell_index, inner_cell_coords)
        outer_s = make_free_sample(outer_cell_index, outer_cell_coords)

        pos_side = geo.side_position(side_arg, s)
        pos_inner = geo.cell_position(cell_arg, inner_s)
        pos_outer = geo.cell_position(cell_arg, outer_s)

        for k in range(type(pos_side).length):
            wp.expect_near(pos_side[k], pos_inner[k], 0.0001)
            wp.expect_near(pos_side[k], pos_outer[k], 0.0001)

        inner_side_coords = geo.side_from_cell_coords(side_arg, side_index, inner_cell_index, inner_cell_coords)
        outer_side_coords = geo.side_from_cell_coords(side_arg, side_index, outer_cell_index, outer_cell_coords)

        wp.expect_near(coords, inner_side_coords, 0.0001)
        wp.expect_near(coords, outer_side_coords, 0.0001)

        area = geo.side_measure(side_arg, s)
        wp.atomic_add(side_measures, side_index, area * qp_weights[q])

        # test consistency of side normal, measure, and deformation gradient
        F = geo.side_deformation_gradient(side_arg, s)
        F_det = DeformedGeometry._side_measure(F)
        wp.expect_near(F_det * REF_MEASURE, area)

        nor = geo.side_normal(side_arg, s)
        F_cross = DeformedGeometry._side_normal(F)

        for k in range(type(pos_side).length):
            wp.expect_near(F_cross[k], nor[k], 0.0001)

    cell_measures = wp.zeros(dtype=float, device=device, shape=geo.cell_count())

    cell_quadrature = fem.RegularQuadrature(fem.Cells(geo), order=2)
    cell_qps = wp.array(cell_quadrature.points, dtype=Coords, device=device)
    cell_qp_weights = wp.array(cell_quadrature.weights, dtype=float, device=device)

    wp.launch(
        kernel=test_geo_cells_kernel,
        dim=(geo.cell_count(), cell_qps.shape[0]),
        inputs=[geo.cell_arg_value(device), cell_qps, cell_qp_weights, cell_measures],
        device=device,
    )

    side_measures = wp.zeros(dtype=float, device=device, shape=geo.side_count())

    side_quadrature = fem.RegularQuadrature(fem.Sides(geo), order=2)
    side_qps = wp.array(side_quadrature.points, dtype=Coords, device=device)
    side_qp_weights = wp.array(side_quadrature.weights, dtype=float, device=device)

    wp.launch(
        kernel=test_geo_sides_kernel,
        dim=(geo.side_count(), side_qps.shape[0]),
        inputs=[geo.side_arg_value(device), side_qps, side_qp_weights, side_measures],
        device=device,
    )

    return side_measures, cell_measures


def test_grid_2d(test, device):
    N = 3

    geo = fem.Grid2D(res=wp.vec2i(N))

    test.assertEqual(geo.cell_count(), N**2)
    test.assertEqual(geo.vertex_count(), (N + 1) ** 2)
    test.assertEqual(geo.side_count(), 2 * (N + 1) * N)
    test.assertEqual(geo.boundary_side_count(), 4 * N)

    side_measures, cell_measures = _launch_test_geometry_kernel(geo, device)

    assert_np_equal(side_measures.numpy(), np.full(side_measures.shape, 1.0 / (N)), tol=1.0e-4)
    assert_np_equal(cell_measures.numpy(), np.full(cell_measures.shape, 1.0 / (N**2)), tol=1.0e-4)


def test_triangle_mesh(test, device):
    N = 3

    with wp.ScopedDevice(device):
        positions, tri_vidx = _gen_trimesh(N)

    geo = fem.Trimesh2D(tri_vertex_indices=tri_vidx, positions=positions)

    test.assertEqual(geo.cell_count(), 2 * (N) ** 2)
    test.assertEqual(geo.vertex_count(), (N + 1) ** 2)
    test.assertEqual(geo.side_count(), 2 * (N + 1) * N + (N**2))
    test.assertEqual(geo.boundary_side_count(), 4 * N)

    side_measures, cell_measures = _launch_test_geometry_kernel(geo, device)

    assert_np_equal(cell_measures.numpy(), np.full(cell_measures.shape, 0.5 / (N**2)), tol=1.0e-4)
    test.assertAlmostEqual(np.sum(side_measures.numpy()), 2 * (N + 1) + N * math.sqrt(2.0), places=4)


def test_quad_mesh(test, device):
    N = 3

    with wp.ScopedDevice(device):
        positions, quad_vidx = _gen_quadmesh(N)

    geo = fem.Quadmesh2D(quad_vertex_indices=quad_vidx, positions=positions)

    test.assertEqual(geo.cell_count(), N**2)
    test.assertEqual(geo.vertex_count(), (N + 1) ** 2)
    test.assertEqual(geo.side_count(), 2 * (N + 1) * N)
    test.assertEqual(geo.boundary_side_count(), 4 * N)

    side_measures, cell_measures = _launch_test_geometry_kernel(geo, device)

    assert_np_equal(side_measures.numpy(), np.full(side_measures.shape, 1.0 / (N)), tol=1.0e-4)
    assert_np_equal(cell_measures.numpy(), np.full(cell_measures.shape, 1.0 / (N**2)), tol=1.0e-4)


def test_grid_3d(test, device):
    N = 3

    geo = fem.Grid3D(res=wp.vec3i(N))

    test.assertEqual(geo.cell_count(), (N) ** 3)
    test.assertEqual(geo.vertex_count(), (N + 1) ** 3)
    test.assertEqual(geo.side_count(), 3 * (N + 1) * N**2)
    test.assertEqual(geo.boundary_side_count(), 6 * N * N)
    test.assertEqual(geo.edge_count(), 3 * N * (N + 1) ** 2)

    side_measures, cell_measures = _launch_test_geometry_kernel(geo, device)

    assert_np_equal(side_measures.numpy(), np.full(side_measures.shape, 1.0 / (N**2)), tol=1.0e-4)
    assert_np_equal(cell_measures.numpy(), np.full(cell_measures.shape, 1.0 / (N**3)), tol=1.0e-4)


def test_tet_mesh(test, device):
    N = 3

    with wp.ScopedDevice(device):
        positions, tet_vidx = _gen_tetmesh(N)

    geo = fem.Tetmesh(tet_vertex_indices=tet_vidx, positions=positions)

    test.assertEqual(geo.cell_count(), 5 * (N) ** 3)
    test.assertEqual(geo.vertex_count(), (N + 1) ** 3)
    test.assertEqual(geo.side_count(), 6 * (N + 1) * N**2 + (N**3) * 4)
    test.assertEqual(geo.boundary_side_count(), 12 * N * N)
    test.assertEqual(geo.edge_count(), 3 * N * (N + 1) * (2 * N + 1))

    side_measures, cell_measures = _launch_test_geometry_kernel(geo, device)

    test.assertAlmostEqual(np.sum(cell_measures.numpy()), 1.0, places=4)
    test.assertAlmostEqual(np.sum(side_measures.numpy()), 0.5 * 6 * (N + 1) + N * 2 * math.sqrt(3.0), places=4)


def test_hex_mesh(test, device):
    N = 3

    with wp.ScopedDevice(device):
        positions, tet_vidx = _gen_hexmesh(N)

    geo = fem.Hexmesh(hex_vertex_indices=tet_vidx, positions=positions)

    test.assertEqual(geo.cell_count(), (N) ** 3)
    test.assertEqual(geo.vertex_count(), (N + 1) ** 3)
    test.assertEqual(geo.side_count(), 3 * (N + 1) * N**2)
    test.assertEqual(geo.boundary_side_count(), 6 * N * N)
    test.assertEqual(geo.edge_count(), 3 * N * (N + 1) ** 2)

    side_measures, cell_measures = _launch_test_geometry_kernel(geo, device)

    assert_np_equal(side_measures.numpy(), np.full(side_measures.shape, 1.0 / (N**2)), tol=1.0e-4)
    assert_np_equal(cell_measures.numpy(), np.full(cell_measures.shape, 1.0 / (N**3)), tol=1.0e-4)


def test_nanogrid(test, device):
    N = 8

    points = wp.array([[0.5, 0.5, 0.5]], dtype=float, device=device)
    volume = wp.Volume.allocate_by_tiles(
        tile_points=points, voxel_size=1.0 / N, translation=(0.0, 0.0, 0.0), bg_value=None, device=device
    )

    geo = fem.Nanogrid(volume)

    test.assertEqual(geo.cell_count(), (N) ** 3)
    test.assertEqual(geo.vertex_count(), (N + 1) ** 3)
    test.assertEqual(geo.side_count(), 3 * (N + 1) * N**2)
    test.assertEqual(geo.boundary_side_count(), 6 * N * N)
    test.assertEqual(geo.edge_count(), 3 * N * (N + 1) ** 2)

    side_measures, cell_measures = _launch_test_geometry_kernel(geo, device)

    assert_np_equal(side_measures.numpy(), np.full(side_measures.shape, 1.0 / (N**2)), tol=1.0e-4)
    assert_np_equal(cell_measures.numpy(), np.full(cell_measures.shape, 1.0 / (N**3)), tol=1.0e-4)


@integrand
def _rigid_deformation_field(s: Sample, domain: Domain, translation: wp.vec3, rotation: wp.vec3, scale: float):
    q = wp.quat_from_axis_angle(wp.normalize(rotation), wp.length(rotation))
    return translation + scale * wp.quat_rotate(q, domain(s)) - domain(s)


def test_deformed_geometry(test, device):
    N = 3

    with wp.ScopedDevice(device):
        positions, tet_vidx = _gen_tetmesh(N)

        geo = fem.Tetmesh(tet_vertex_indices=tet_vidx, positions=positions)

        translation = [1.0, 2.0, 3.0]
        rotation = [0.0, math.pi / 4.0, 0.0]
        scale = 2.0

        vector_space = fem.make_polynomial_space(geo, dtype=wp.vec3, degree=2)
        pos_field = vector_space.make_field()
        fem.interpolate(
            _rigid_deformation_field,
            dest=pos_field,
            values={"translation": translation, "rotation": rotation, "scale": scale},
        )

        deformed_geo = pos_field.make_deformed_geometry()

        # rigidly-deformed geometry

        test.assertEqual(geo.cell_count(), 5 * (N) ** 3)
        test.assertEqual(geo.vertex_count(), (N + 1) ** 3)
        test.assertEqual(geo.side_count(), 6 * (N + 1) * N**2 + (N**3) * 4)
        test.assertEqual(geo.boundary_side_count(), 12 * N * N)

        side_measures, cell_measures = _launch_test_geometry_kernel(deformed_geo, wp.get_device())

        test.assertAlmostEqual(
            np.sum(cell_measures.numpy()), scale**3, places=4, msg=f"cell_measures = {cell_measures.numpy()}"
        )
        test.assertAlmostEqual(
            np.sum(side_measures.numpy()), scale**2 * (0.5 * 6 * (N + 1) + N * 2 * math.sqrt(3.0)), places=4
        )

        @wp.kernel
        def _test_deformed_geometry_normal(
            geo_index_arg: geo.SideIndexArg, geo_arg: geo.SideArg, def_arg: deformed_geo.SideArg, rotation: wp.vec3
        ):
            i = wp.tid()
            side_index = deformed_geo.boundary_side_index(geo_index_arg, i)

            s = make_free_sample(side_index, Coords(0.5, 0.5, 0.0))
            geo_n = geo.side_normal(geo_arg, s)
            def_n = deformed_geo.side_normal(def_arg, s)

            q = wp.quat_from_axis_angle(wp.normalize(rotation), wp.length(rotation))
            wp.expect_near(wp.quat_rotate(q, geo_n), def_n, 0.001)

        wp.launch(
            _test_deformed_geometry_normal,
            dim=geo.boundary_side_count(),
            inputs=[
                geo.side_index_arg_value(wp.get_device()),
                geo.side_arg_value(wp.get_device()),
                deformed_geo.side_arg_value(wp.get_device()),
                rotation,
            ],
        )

    wp.synchronize()


@wp.kernel
def _test_closest_point_on_tri_kernel(
    e0: wp.vec2,
    e1: wp.vec2,
    points: wp.array(dtype=wp.vec2),
    sq_dist: wp.array(dtype=float),
    coords: wp.array(dtype=Coords),
):
    i = wp.tid()
    d2, c = project_on_tri_at_origin(points[i], e0, e1)
    sq_dist[i] = d2
    coords[i] = c


@wp.kernel
def _test_closest_point_on_tet_kernel(
    e0: wp.vec3,
    e1: wp.vec3,
    e2: wp.vec3,
    points: wp.array(dtype=wp.vec3),
    sq_dist: wp.array(dtype=float),
    coords: wp.array(dtype=Coords),
):
    i = wp.tid()
    d2, c = project_on_tet_at_origin(points[i], e0, e1, e2)
    sq_dist[i] = d2
    coords[i] = c


def test_closest_point_queries(test, device):
    # Test some simple lookup queries
    e0 = wp.vec2(2.0, 0.0)
    e1 = wp.vec2(0.0, 2.0)

    points = wp.array(
        (
            [-1.0, -1.0],
            [0.5, 0.5],
            [1.0, 1.0],
            [2.0, 2.0],
        ),
        dtype=wp.vec2,
        device=device,
    )
    expected_sq_dist = np.array([2.0, 0.0, 0.0, 2.0])
    expected_coords = np.array([[1.0, 0.0, 0.0], [0.5, 0.25, 0.25], [0.0, 0.5, 0.5], [0.0, 0.5, 0.5]])

    sq_dist = wp.empty(shape=points.shape, dtype=float, device=device)
    coords = wp.empty(shape=points.shape, dtype=Coords, device=device)
    wp.launch(
        _test_closest_point_on_tri_kernel, dim=points.shape, device=device, inputs=[e0, e1, points, sq_dist, coords]
    )

    assert_np_equal(coords.numpy(), expected_coords)
    assert_np_equal(sq_dist.numpy(), expected_sq_dist)

    # Tet

    e0 = wp.vec3(3.0, 0.0, 0.0)
    e1 = wp.vec3(0.0, 3.0, 0.0)
    e2 = wp.vec3(0.0, 0.0, 3.0)

    points = wp.array(
        (
            [-1.0, -1.0, -1.0],
            [0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        ),
        dtype=wp.vec3,
        device=device,
    )
    expected_sq_dist = np.array([3.0, 0.0, 0.0, 3.0])
    expected_coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0],
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        ]
    )

    sq_dist = wp.empty(shape=points.shape, dtype=float, device=device)
    coords = wp.empty(shape=points.shape, dtype=Coords, device=device)
    wp.launch(
        _test_closest_point_on_tet_kernel, dim=points.shape, device=device, inputs=[e0, e1, e2, points, sq_dist, coords]
    )

    assert_np_equal(coords.numpy(), expected_coords, tol=1.0e-4)
    assert_np_equal(sq_dist.numpy(), expected_sq_dist, tol=1.0e-4)


def test_regular_quadrature(test, device):
    from warp.fem.geometry.element import LinearEdge, Polynomial, Triangle

    for family in Polynomial:
        # test integrating monomials
        for degree in range(8):
            coords, weights = LinearEdge().instantiate_quadrature(degree, family=family)
            res = sum(w * pow(c[0], degree) for w, c in zip(weights, coords))
            ref = 1.0 / (degree + 1)

            test.assertAlmostEqual(ref, res, places=4)

        # test integrating y^k1 (1 - x)^k2 on triangle using transformation to square
        for x_degree in range(4):
            for y_degree in range(4):
                coords, weights = Triangle().instantiate_quadrature(x_degree + y_degree, family=family)
                res = 0.5 * sum(w * pow(1.0 - c[1], x_degree) * pow(c[2], y_degree) for w, c in zip(weights, coords))

                ref = 1.0 / ((x_degree + y_degree + 2) * (y_degree + 1))
                # print(x_degree, y_degree, family, len(coords), res, ref)
                test.assertAlmostEqual(ref, res, places=4)

    # test integrating y^k1 (1 - x)^k2 on triangle using direct formulas
    for x_degree in range(5):
        for y_degree in range(5):
            coords, weights = Triangle().instantiate_quadrature(x_degree + y_degree, family=None)
            res = 0.5 * sum(w * pow(1.0 - c[1], x_degree) * pow(c[2], y_degree) for w, c in zip(weights, coords))

            ref = 1.0 / ((x_degree + y_degree + 2) * (y_degree + 1))
            test.assertAlmostEqual(ref, res, places=4)


def test_dof_mapper(test, device):
    matrix_types = [wp.mat22, wp.mat33]

    # Symmetric mapper
    for mapping in fem.SymmetricTensorMapper.Mapping:
        for dtype in matrix_types:
            mapper = fem.SymmetricTensorMapper(dtype, mapping=mapping)
            dof_dtype = mapper.dof_dtype

            for k in range(dof_dtype._length_):
                elem = np.array(dof_dtype(0.0))
                elem[k] = 1.0
                dof_vec = dof_dtype(elem)

                mat = mapper.dof_to_value(dof_vec)
                dof_round_trip = mapper.value_to_dof(mat)

                # Check that value_to_dof(dof_to_value) is idempotent
                assert_np_equal(np.array(dof_round_trip), np.array(dof_vec))

                # Check that value is unitary for Frobenius norm 0.5 * |tau:tau|
                frob_norm2 = 0.5 * wp.ddot(mat, mat)
                test.assertAlmostEqual(frob_norm2, 1.0, places=6)

    # Skew-symmetric mapper
    for dtype in matrix_types:
        mapper = fem.SkewSymmetricTensorMapper(dtype)
        dof_dtype = mapper.dof_dtype

        if hasattr(dof_dtype, "_length_"):
            for k in range(dof_dtype._length_):
                elem = np.array(dof_dtype(0.0))
                elem[k] = 1.0
                dof_vec = dof_dtype(elem)

                mat = mapper.dof_to_value(dof_vec)
                dof_round_trip = mapper.value_to_dof(mat)

                # Check that value_to_dof(dof_to_value) is idempotent
                assert_np_equal(np.array(dof_round_trip), np.array(dof_vec))

                # Check that value is unitary for Frobenius norm 0.5 * |tau:tau|
                frob_norm2 = 0.5 * wp.ddot(mat, mat)
                test.assertAlmostEqual(frob_norm2, 1.0, places=6)
        else:
            dof_val = 1.0

            mat = mapper.dof_to_value(dof_val)
            dof_round_trip = mapper.value_to_dof(mat)

            test.assertAlmostEqual(dof_round_trip, dof_val)

            # Check that value is unitary for Frobenius norm 0.5 * |tau:tau|
            frob_norm2 = 0.5 * wp.ddot(mat, mat)
            test.assertAlmostEqual(frob_norm2, 1.0, places=6)


def test_shape_function_weight(test, shape: shape.ShapeFunction, coord_sampler, CENTER_COORDS):
    NODE_COUNT = shape.NODES_PER_ELEMENT
    weight_fn = shape.make_element_inner_weight()
    node_coords_fn = shape.make_node_coords_in_element()

    # Weight at node should be 1
    @dynamic_kernel(suffix=shape.name, kernel_options={"enable_backward": False})
    def node_unity_test():
        n = wp.tid()
        node_w = weight_fn(node_coords_fn(n), n)
        wp.expect_near(node_w, 1.0, 1e-5)

    wp.launch(node_unity_test, dim=NODE_COUNT, inputs=[])

    # Sum of node quadrature weights should be one (order 0)
    # Sum of weighted quadrature coords should be element center (order 1)
    node_quadrature_weight_fn = shape.make_node_quadrature_weight()

    @dynamic_kernel(suffix=shape.name, kernel_options={"enable_backward": False})
    def node_quadrature_unity_test():
        sum_node_qp = float(0.0)
        sum_node_qp_coords = Coords(0.0)

        for n in range(NODE_COUNT):
            w = node_quadrature_weight_fn(n)
            sum_node_qp += w
            sum_node_qp_coords += w * node_coords_fn(n)

        wp.expect_near(sum_node_qp, 1.0, 0.0001)
        wp.expect_near(sum_node_qp_coords, CENTER_COORDS, 0.0001)

    wp.launch(node_quadrature_unity_test, dim=1, inputs=[])

    @dynamic_kernel(suffix=shape.name, kernel_options={"enable_backward": False})
    def partition_of_unity_test():
        rng_state = wp.rand_init(4321, wp.tid())
        coords = coord_sampler(rng_state)

        # sum of node weights anywhere should be 1.0
        w_sum = float(0.0)
        for n in range(NODE_COUNT):
            w_sum += weight_fn(coords, n)

        wp.expect_near(w_sum, 1.0, 0.0001)

    n_samples = 100
    wp.launch(partition_of_unity_test, dim=n_samples, inputs=[])


def test_shape_function_trace(test, shape: shape.ShapeFunction, CENTER_COORDS):
    NODE_COUNT = shape.NODES_PER_ELEMENT
    node_coords_fn = shape.make_node_coords_in_element()

    # Sum of node quadrature weights should be one (order 0)
    # Sum of weighted quadrature coords should be element center (order 1)
    trace_node_quadrature_weight_fn = shape.make_trace_node_quadrature_weight()

    @dynamic_kernel(suffix=shape.name, kernel_options={"enable_backward": False})
    def trace_node_quadrature_unity_test():
        sum_node_qp = float(0.0)
        sum_node_qp_coords = Coords(0.0)

        for n in range(NODE_COUNT):
            coords = node_coords_fn(n)

            if wp.abs(coords[0]) < 1.0e-6:
                w = trace_node_quadrature_weight_fn(n)
                sum_node_qp += w
                sum_node_qp_coords += w * node_coords_fn(n)

        wp.expect_near(sum_node_qp, 1.0, 0.0001)
        wp.expect_near(sum_node_qp_coords, CENTER_COORDS, 0.0001)

    wp.launch(trace_node_quadrature_unity_test, dim=1, inputs=[])


def test_shape_function_gradient(test, shape: shape.ShapeFunction, coord_sampler, coord_delta_sampler):
    weight_fn = shape.make_element_inner_weight()
    weight_gradient_fn = shape.make_element_inner_weight_gradient()

    @dynamic_kernel(suffix=shape.name, kernel_options={"enable_backward": False})
    def finite_difference_test():
        i, n = wp.tid()
        rng_state = wp.rand_init(1234, i)

        coords = coord_sampler(rng_state)

        epsilon = 0.003
        param_delta, coords_delta = coord_delta_sampler(epsilon, rng_state)

        w_p = weight_fn(coords + coords_delta, n)
        w_m = weight_fn(coords - coords_delta, n)

        gp = weight_gradient_fn(coords + coords_delta, n)
        gm = weight_gradient_fn(coords - coords_delta, n)

        # 2nd-order finite-difference test
        # See Schroeder 2019, Practical course on computing derivatives in code
        delta_ref = w_p - w_m
        delta_est = wp.dot(gp + gm, param_delta)

        # wp.printf("%d %f %f \n", n, delta_ref, delta_est)
        wp.expect_near(delta_ref, delta_est, 0.0001)

    n_samples = 100
    wp.launch(finite_difference_test, dim=(n_samples, shape.NODES_PER_ELEMENT), inputs=[])


def test_square_shape_functions(test, device):
    SQUARE_CENTER_COORDS = wp.constant(Coords(0.5, 0.5, 0.0))
    SQUARE_SIDE_CENTER_COORDS = wp.constant(Coords(0.0, 0.5, 0.0))

    @wp.func
    def square_coord_sampler(state: wp.uint32):
        return Coords(wp.randf(state), wp.randf(state), 0.0)

    @wp.func
    def square_coord_delta_sampler(epsilon: float, state: wp.uint32):
        param_delta = wp.normalize(wp.vec2(wp.randf(state), wp.randf(state))) * epsilon
        return param_delta, Coords(param_delta[0], param_delta[1], 0.0)

    Q_1 = shape.SquareBipolynomialShapeFunctions(degree=1, family=fem.Polynomial.LOBATTO_GAUSS_LEGENDRE)
    Q_2 = shape.SquareBipolynomialShapeFunctions(degree=2, family=fem.Polynomial.LOBATTO_GAUSS_LEGENDRE)
    Q_3 = shape.SquareBipolynomialShapeFunctions(degree=3, family=fem.Polynomial.LOBATTO_GAUSS_LEGENDRE)

    test_shape_function_weight(test, Q_1, square_coord_sampler, SQUARE_CENTER_COORDS)
    test_shape_function_weight(test, Q_2, square_coord_sampler, SQUARE_CENTER_COORDS)
    test_shape_function_weight(test, Q_3, square_coord_sampler, SQUARE_CENTER_COORDS)
    test_shape_function_trace(test, Q_1, SQUARE_SIDE_CENTER_COORDS)
    test_shape_function_trace(test, Q_2, SQUARE_SIDE_CENTER_COORDS)
    test_shape_function_trace(test, Q_3, SQUARE_SIDE_CENTER_COORDS)
    test_shape_function_gradient(test, Q_1, square_coord_sampler, square_coord_delta_sampler)
    test_shape_function_gradient(test, Q_2, square_coord_sampler, square_coord_delta_sampler)
    test_shape_function_gradient(test, Q_3, square_coord_sampler, square_coord_delta_sampler)

    Q_1 = shape.SquareBipolynomialShapeFunctions(degree=1, family=fem.Polynomial.GAUSS_LEGENDRE)
    Q_2 = shape.SquareBipolynomialShapeFunctions(degree=2, family=fem.Polynomial.GAUSS_LEGENDRE)
    Q_3 = shape.SquareBipolynomialShapeFunctions(degree=3, family=fem.Polynomial.GAUSS_LEGENDRE)

    test_shape_function_weight(test, Q_1, square_coord_sampler, SQUARE_CENTER_COORDS)
    test_shape_function_weight(test, Q_2, square_coord_sampler, SQUARE_CENTER_COORDS)
    test_shape_function_weight(test, Q_3, square_coord_sampler, SQUARE_CENTER_COORDS)
    test_shape_function_gradient(test, Q_1, square_coord_sampler, square_coord_delta_sampler)
    test_shape_function_gradient(test, Q_2, square_coord_sampler, square_coord_delta_sampler)
    test_shape_function_gradient(test, Q_3, square_coord_sampler, square_coord_delta_sampler)

    S_2 = shape.SquareSerendipityShapeFunctions(degree=2, family=fem.Polynomial.LOBATTO_GAUSS_LEGENDRE)
    S_3 = shape.SquareSerendipityShapeFunctions(degree=3, family=fem.Polynomial.LOBATTO_GAUSS_LEGENDRE)

    test_shape_function_weight(test, S_2, square_coord_sampler, SQUARE_CENTER_COORDS)
    test_shape_function_weight(test, S_3, square_coord_sampler, SQUARE_CENTER_COORDS)
    test_shape_function_trace(test, S_2, SQUARE_SIDE_CENTER_COORDS)
    test_shape_function_trace(test, S_3, SQUARE_SIDE_CENTER_COORDS)
    test_shape_function_gradient(test, S_2, square_coord_sampler, square_coord_delta_sampler)
    test_shape_function_gradient(test, S_3, square_coord_sampler, square_coord_delta_sampler)

    P_c1 = shape.SquareNonConformingPolynomialShapeFunctions(degree=1)
    P_c2 = shape.SquareNonConformingPolynomialShapeFunctions(degree=2)
    P_c3 = shape.SquareNonConformingPolynomialShapeFunctions(degree=3)

    test_shape_function_weight(test, P_c1, square_coord_sampler, SQUARE_CENTER_COORDS)
    test_shape_function_weight(test, P_c2, square_coord_sampler, SQUARE_CENTER_COORDS)
    test_shape_function_weight(test, P_c3, square_coord_sampler, SQUARE_CENTER_COORDS)
    test_shape_function_gradient(test, P_c1, square_coord_sampler, square_coord_delta_sampler)
    test_shape_function_gradient(test, P_c2, square_coord_sampler, square_coord_delta_sampler)
    test_shape_function_gradient(test, P_c3, square_coord_sampler, square_coord_delta_sampler)

    wp.synchronize()


def test_cube_shape_functions(test, device):
    CUBE_CENTER_COORDS = wp.constant(Coords(0.5, 0.5, 0.5))
    CUBE_SIDE_CENTER_COORDS = wp.constant(Coords(0.0, 0.5, 0.5))

    @wp.func
    def cube_coord_sampler(state: wp.uint32):
        return Coords(wp.randf(state), wp.randf(state), wp.randf(state))

    @wp.func
    def cube_coord_delta_sampler(epsilon: float, state: wp.uint32):
        param_delta = wp.normalize(wp.vec3(wp.randf(state), wp.randf(state), wp.randf(state))) * epsilon
        return param_delta, param_delta

    Q_1 = shape.CubeTripolynomialShapeFunctions(degree=1, family=fem.Polynomial.LOBATTO_GAUSS_LEGENDRE)
    Q_2 = shape.CubeTripolynomialShapeFunctions(degree=2, family=fem.Polynomial.LOBATTO_GAUSS_LEGENDRE)
    Q_3 = shape.CubeTripolynomialShapeFunctions(degree=3, family=fem.Polynomial.LOBATTO_GAUSS_LEGENDRE)

    test_shape_function_weight(test, Q_1, cube_coord_sampler, CUBE_CENTER_COORDS)
    test_shape_function_weight(test, Q_2, cube_coord_sampler, CUBE_CENTER_COORDS)
    test_shape_function_weight(test, Q_3, cube_coord_sampler, CUBE_CENTER_COORDS)
    test_shape_function_trace(test, Q_1, CUBE_SIDE_CENTER_COORDS)
    test_shape_function_trace(test, Q_2, CUBE_SIDE_CENTER_COORDS)
    test_shape_function_trace(test, Q_3, CUBE_SIDE_CENTER_COORDS)
    test_shape_function_gradient(test, Q_1, cube_coord_sampler, cube_coord_delta_sampler)
    test_shape_function_gradient(test, Q_2, cube_coord_sampler, cube_coord_delta_sampler)
    test_shape_function_gradient(test, Q_3, cube_coord_sampler, cube_coord_delta_sampler)

    Q_1 = shape.CubeTripolynomialShapeFunctions(degree=1, family=fem.Polynomial.GAUSS_LEGENDRE)
    Q_2 = shape.CubeTripolynomialShapeFunctions(degree=2, family=fem.Polynomial.GAUSS_LEGENDRE)
    Q_3 = shape.CubeTripolynomialShapeFunctions(degree=3, family=fem.Polynomial.GAUSS_LEGENDRE)

    test_shape_function_weight(test, Q_1, cube_coord_sampler, CUBE_CENTER_COORDS)
    test_shape_function_weight(test, Q_2, cube_coord_sampler, CUBE_CENTER_COORDS)
    test_shape_function_weight(test, Q_3, cube_coord_sampler, CUBE_CENTER_COORDS)
    test_shape_function_gradient(test, Q_1, cube_coord_sampler, cube_coord_delta_sampler)
    test_shape_function_gradient(test, Q_2, cube_coord_sampler, cube_coord_delta_sampler)
    test_shape_function_gradient(test, Q_3, cube_coord_sampler, cube_coord_delta_sampler)

    S_2 = shape.CubeSerendipityShapeFunctions(degree=2, family=fem.Polynomial.LOBATTO_GAUSS_LEGENDRE)
    S_3 = shape.CubeSerendipityShapeFunctions(degree=3, family=fem.Polynomial.LOBATTO_GAUSS_LEGENDRE)

    test_shape_function_weight(test, S_2, cube_coord_sampler, CUBE_CENTER_COORDS)
    test_shape_function_weight(test, S_3, cube_coord_sampler, CUBE_CENTER_COORDS)
    test_shape_function_trace(test, S_2, CUBE_SIDE_CENTER_COORDS)
    test_shape_function_trace(test, S_3, CUBE_SIDE_CENTER_COORDS)
    test_shape_function_gradient(test, S_2, cube_coord_sampler, cube_coord_delta_sampler)
    test_shape_function_gradient(test, S_3, cube_coord_sampler, cube_coord_delta_sampler)

    P_c1 = shape.CubeNonConformingPolynomialShapeFunctions(degree=1)
    P_c2 = shape.CubeNonConformingPolynomialShapeFunctions(degree=2)
    P_c3 = shape.CubeNonConformingPolynomialShapeFunctions(degree=3)

    test_shape_function_weight(test, P_c1, cube_coord_sampler, CUBE_CENTER_COORDS)
    test_shape_function_weight(test, P_c2, cube_coord_sampler, CUBE_CENTER_COORDS)
    test_shape_function_weight(test, P_c3, cube_coord_sampler, CUBE_CENTER_COORDS)
    test_shape_function_gradient(test, P_c1, cube_coord_sampler, cube_coord_delta_sampler)
    test_shape_function_gradient(test, P_c2, cube_coord_sampler, cube_coord_delta_sampler)
    test_shape_function_gradient(test, P_c3, cube_coord_sampler, cube_coord_delta_sampler)

    wp.synchronize()


def test_tri_shape_functions(test, device):
    TRI_CENTER_COORDS = wp.constant(Coords(1 / 3.0, 1 / 3.0, 1 / 3.0))
    TRI_SIDE_CENTER_COORDS = wp.constant(Coords(0.0, 0.5, 0.5))

    @wp.func
    def tri_coord_sampler(state: wp.uint32):
        a = wp.randf(state)
        b = wp.randf(state)
        return Coords(1.0 - a - b, a, b)

    @wp.func
    def tri_coord_delta_sampler(epsilon: float, state: wp.uint32):
        param_delta = wp.normalize(wp.vec2(wp.randf(state), wp.randf(state))) * epsilon
        a = param_delta[0]
        b = param_delta[1]
        return param_delta, Coords(-a - b, a, b)

    P_1 = shape.Triangle2DPolynomialShapeFunctions(degree=1)
    P_2 = shape.Triangle2DPolynomialShapeFunctions(degree=2)
    P_3 = shape.Triangle2DPolynomialShapeFunctions(degree=3)

    test_shape_function_weight(test, P_1, tri_coord_sampler, TRI_CENTER_COORDS)
    test_shape_function_weight(test, P_2, tri_coord_sampler, TRI_CENTER_COORDS)
    test_shape_function_weight(test, P_3, tri_coord_sampler, TRI_CENTER_COORDS)
    test_shape_function_trace(test, P_1, TRI_SIDE_CENTER_COORDS)
    test_shape_function_trace(test, P_2, TRI_SIDE_CENTER_COORDS)
    test_shape_function_trace(test, P_3, TRI_SIDE_CENTER_COORDS)
    test_shape_function_gradient(test, P_1, tri_coord_sampler, tri_coord_delta_sampler)
    test_shape_function_gradient(test, P_2, tri_coord_sampler, tri_coord_delta_sampler)
    test_shape_function_gradient(test, P_3, tri_coord_sampler, tri_coord_delta_sampler)

    P_1d = shape.Triangle2DNonConformingPolynomialShapeFunctions(degree=1)
    P_2d = shape.Triangle2DNonConformingPolynomialShapeFunctions(degree=2)
    P_3d = shape.Triangle2DNonConformingPolynomialShapeFunctions(degree=3)

    test_shape_function_weight(test, P_1d, tri_coord_sampler, TRI_CENTER_COORDS)
    test_shape_function_weight(test, P_2d, tri_coord_sampler, TRI_CENTER_COORDS)
    test_shape_function_weight(test, P_3d, tri_coord_sampler, TRI_CENTER_COORDS)
    test_shape_function_gradient(test, P_1d, tri_coord_sampler, tri_coord_delta_sampler)
    test_shape_function_gradient(test, P_2d, tri_coord_sampler, tri_coord_delta_sampler)
    test_shape_function_gradient(test, P_3d, tri_coord_sampler, tri_coord_delta_sampler)

    wp.synchronize()


def test_tet_shape_functions(test, device):
    TET_CENTER_COORDS = wp.constant(Coords(1 / 4.0, 1 / 4.0, 1 / 4.0))
    TET_SIDE_CENTER_COORDS = wp.constant(Coords(0.0, 1.0 / 3.0, 1.0 / 3.0))

    @wp.func
    def tet_coord_sampler(state: wp.uint32):
        return Coords(wp.randf(state), wp.randf(state), wp.randf(state))

    @wp.func
    def tet_coord_delta_sampler(epsilon: float, state: wp.uint32):
        param_delta = wp.normalize(wp.vec3(wp.randf(state), wp.randf(state), wp.randf(state))) * epsilon
        return param_delta, param_delta

    P_1 = shape.TetrahedronPolynomialShapeFunctions(degree=1)
    P_2 = shape.TetrahedronPolynomialShapeFunctions(degree=2)
    P_3 = shape.TetrahedronPolynomialShapeFunctions(degree=3)

    test_shape_function_weight(test, P_1, tet_coord_sampler, TET_CENTER_COORDS)
    test_shape_function_weight(test, P_2, tet_coord_sampler, TET_CENTER_COORDS)
    test_shape_function_weight(test, P_3, tet_coord_sampler, TET_CENTER_COORDS)
    test_shape_function_trace(test, P_1, TET_SIDE_CENTER_COORDS)
    test_shape_function_trace(test, P_2, TET_SIDE_CENTER_COORDS)
    test_shape_function_trace(test, P_3, TET_SIDE_CENTER_COORDS)
    test_shape_function_gradient(test, P_1, tet_coord_sampler, tet_coord_delta_sampler)
    test_shape_function_gradient(test, P_2, tet_coord_sampler, tet_coord_delta_sampler)
    test_shape_function_gradient(test, P_3, tet_coord_sampler, tet_coord_delta_sampler)

    P_1d = shape.TetrahedronNonConformingPolynomialShapeFunctions(degree=1)
    P_2d = shape.TetrahedronNonConformingPolynomialShapeFunctions(degree=2)
    P_3d = shape.TetrahedronNonConformingPolynomialShapeFunctions(degree=3)

    test_shape_function_weight(test, P_1d, tet_coord_sampler, TET_CENTER_COORDS)
    test_shape_function_weight(test, P_2d, tet_coord_sampler, TET_CENTER_COORDS)
    test_shape_function_weight(test, P_3d, tet_coord_sampler, TET_CENTER_COORDS)
    test_shape_function_gradient(test, P_1d, tet_coord_sampler, tet_coord_delta_sampler)
    test_shape_function_gradient(test, P_2d, tet_coord_sampler, tet_coord_delta_sampler)
    test_shape_function_gradient(test, P_3d, tet_coord_sampler, tet_coord_delta_sampler)

    wp.synchronize()


def test_point_basis(test, device):
    geo = fem.Grid2D(res=wp.vec2i(2))

    domain = fem.Cells(geo)

    quadrature = fem.RegularQuadrature(domain, order=2, family=fem.Polynomial.GAUSS_LEGENDRE)
    point_basis = fem.PointBasisSpace(quadrature)

    point_space = fem.make_collocated_function_space(point_basis)
    point_test = fem.make_test(point_space, domain=domain)

    # Sample at particle positions
    ones = fem.integrate(linear_form, fields={"u": point_test}, nodal=True)
    test.assertAlmostEqual(np.sum(ones.numpy()), 1.0, places=5)

    # Sampling outside of particle positions
    other_quadrature = fem.RegularQuadrature(domain, order=2, family=fem.Polynomial.LOBATTO_GAUSS_LEGENDRE)
    zeros = fem.integrate(linear_form, quadrature=other_quadrature, fields={"u": point_test})

    test.assertAlmostEqual(np.sum(zeros.numpy()), 0.0, places=5)

    # test point basis with variable points per cell
    points = wp.array([[0.25, 0.33], [0.33, 0.25], [0.8, 0.8]], dtype=wp.vec2)
    pic = fem.PicQuadrature(domain, positions=points)

    test.assertEqual(pic.active_cell_count(), 2)
    test.assertEqual(pic.total_point_count(), 3)
    test.assertEqual(pic.max_points_per_element(), 2)

    point_basis = fem.PointBasisSpace(pic)
    point_space = fem.make_collocated_function_space(point_basis)
    point_test = fem.make_test(point_space, domain=domain)
    test.assertEqual(point_test.space_restriction.node_count(), 3)

    ones = fem.integrate(linear_form, fields={"u": point_test}, quadrature=pic)
    test.assertAlmostEqual(np.sum(ones.numpy()), pic.active_cell_count() / geo.cell_count(), places=5)

    zeros = fem.integrate(linear_form, quadrature=other_quadrature, fields={"u": point_test})
    test.assertAlmostEqual(np.sum(zeros.numpy()), 0.0, places=5)

    linear_vec = fem.make_polynomial_space(geo, dtype=wp.vec2)
    linear_test = fem.make_test(linear_vec)
    point_trial = fem.make_trial(point_space)

    mat = fem.integrate(vector_divergence_form, fields={"u": linear_test, "q": point_trial}, quadrature=pic)
    test.assertEqual(mat.nrow, 9)
    test.assertEqual(mat.ncol, 3)
    test.assertEqual(mat.nnz_sync(), 12)


@fem.integrand
def _bicubic(s: Sample, domain: Domain):
    x = domain(s)
    return wp.pow(x[0], 3.0) * wp.pow(x[1], 3.0)


@fem.integrand
def _piecewise_constant(s: Sample):
    return float(s.element_index)


def test_particle_quadratures(test, device):
    geo = fem.Grid2D(res=wp.vec2i(2))

    domain = fem.Cells(geo)
    points, weights = domain.reference_element().instantiate_quadrature(order=4, family=fem.Polynomial.GAUSS_LEGENDRE)
    points_per_cell = len(points)

    points = points * domain.element_count()
    weights = weights * domain.element_count()

    points = wp.array(points, shape=(domain.element_count(), points_per_cell), dtype=Coords, device=device)
    weights = wp.array(weights, shape=(domain.element_count(), points_per_cell), dtype=float, device=device)

    explicit_quadrature = fem.ExplicitQuadrature(domain, points, weights)

    test.assertEqual(explicit_quadrature.max_points_per_element(), points_per_cell)
    test.assertEqual(explicit_quadrature.total_point_count(), points_per_cell * geo.cell_count())

    val = fem.integrate(_bicubic, quadrature=explicit_quadrature)
    test.assertAlmostEqual(val, 1.0 / 16, places=5)

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

    val = fem.integrate(_piecewise_constant, quadrature=pic_quadrature)
    test.assertAlmostEqual(val, 1.25, places=5)

    # Test differentiability of PicQuadrature w.r.t positions and measures
    points = wp.array([[0.25, 0.33], [0.33, 0.25], [0.8, 0.8]], dtype=wp.vec2, device=device, requires_grad=True)
    measures = wp.ones(3, dtype=float, device=device, requires_grad=True)

    tape = wp.Tape()
    with tape:
        pic = fem.PicQuadrature(domain, positions=points, measures=measures, requires_grad=True)

    pic.arg_value(device).particle_coords.grad.fill_(1.0)
    pic.arg_value(device).particle_fraction.grad.fill_(1.0)
    tape.backward()

    assert_np_equal(points.grad.numpy(), np.full((3, 2), 2.0))  # == 1.0 / cell_size
    assert_np_equal(measures.grad.numpy(), np.full(3, 4.0))  # == 1.0 / cell_area


@wp.func
def aniso_bicubic_fn(x: wp.vec2, scale: wp.vec2):
    return wp.pow(x[0] * scale[0], 3.0) * wp.pow(x[1] * scale[1], 3.0)


@wp.func
def aniso_bicubic_grad(x: wp.vec2, scale: wp.vec2):
    return wp.vec2(
        3.0 * scale[0] * wp.pow(x[0] * scale[0], 2.0) * wp.pow(x[1] * scale[1], 3.0),
        3.0 * scale[1] * wp.pow(x[0] * scale[0], 3.0) * wp.pow(x[1] * scale[1], 2.0),
    )


def test_implicit_fields(test, device):
    geo = fem.Grid2D(res=wp.vec2i(2))
    domain = fem.Cells(geo)
    boundary = fem.BoundarySides(geo)

    space = fem.make_polynomial_space(geo)
    vec_space = fem.make_polynomial_space(geo, dtype=wp.vec2)
    discrete_field = fem.make_discrete_field(space)
    discrete_vec_field = fem.make_discrete_field(vec_space)

    # Uniform

    uniform = fem.UniformField(domain, 5.0)
    fem.interpolate(uniform, dest=discrete_field)
    assert_np_equal(discrete_field.dof_values.numpy(), np.full(9, 5.0))

    fem.interpolate(grad_field, fields={"p": uniform}, dest=discrete_vec_field)
    assert_np_equal(discrete_vec_field.dof_values.numpy(), np.zeros((9, 2)))

    uniform.value = 2.0
    fem.interpolate(uniform.trace(), dest=fem.make_restriction(discrete_field, domain=boundary))
    assert_np_equal(discrete_field.dof_values.numpy(), np.array([2.0] * 4 + [5.0] + [2.0] * 4))

    # Implicit

    implicit = fem.ImplicitField(
        domain, func=aniso_bicubic_fn, values={"scale": wp.vec2(2.0, 4.0)}, grad_func=aniso_bicubic_grad
    )
    fem.interpolate(implicit, dest=discrete_field)
    assert_np_equal(
        discrete_field.dof_values.numpy(),
        np.array([0.0, 0.0, 0.0, 0.0, 2.0**3, 4.0**3, 0.0, 2.0**3 * 2.0**3, 4.0**3 * 2.0**3]),
    )

    fem.interpolate(grad_field, fields={"p": implicit}, dest=discrete_vec_field)
    assert_np_equal(discrete_vec_field.dof_values.numpy()[0], np.zeros(2))
    assert_np_equal(discrete_vec_field.dof_values.numpy()[-1], np.full(2, (2.0**9.0 * 3.0)))

    implicit.values.scale = wp.vec2(-2.0, -2.0)
    fem.interpolate(implicit.trace(), dest=fem.make_restriction(discrete_field, domain=boundary))
    assert_np_equal(
        discrete_field.dof_values.numpy(),
        np.array([0.0, 0.0, 0.0, 0.0, 2.0**3, 2.0**3, 0.0, 2.0**3, 4.0**3]),
    )

    # Nonconforming

    geo2 = fem.Grid2D(res=wp.vec2i(1), bounds_lo=wp.vec2(0.25, 0.25), bounds_hi=wp.vec2(2.0, 2.0))
    domain2 = fem.Cells(geo2)
    boundary2 = fem.BoundarySides(geo2)
    space2 = fem.make_polynomial_space(geo2)
    vec_space2 = fem.make_polynomial_space(geo2, dtype=wp.vec2)
    discrete_field2 = fem.make_discrete_field(space2)
    discrete_vec_field2 = fem.make_discrete_field(vec_space2)

    nonconforming = fem.NonconformingField(domain2, discrete_field, background=5.0)
    fem.interpolate(
        nonconforming,
        dest=discrete_field2,
    )
    assert_np_equal(discrete_field2.dof_values.numpy(), np.array([2.0] + [5.0] * 3))

    fem.interpolate(grad_field, fields={"p": nonconforming}, dest=discrete_vec_field2)
    assert_np_equal(discrete_vec_field2.dof_values.numpy()[0], np.full(2, 8.0))
    assert_np_equal(discrete_vec_field2.dof_values.numpy()[-1], np.zeros(2))

    discrete_field2.dof_values.zero_()
    fem.interpolate(
        nonconforming.trace(),
        dest=fem.make_restriction(discrete_field2, domain=boundary2),
    )
    assert_np_equal(discrete_field2.dof_values.numpy(), np.array([2.0] + [5.0] * 3))


@wp.kernel
def test_qr_eigenvalues():
    tol = 1.0e-6

    # zero
    Zero = wp.mat33(0.0)
    Id = wp.identity(n=3, dtype=float)
    D3, P3 = symmetric_eigenvalues_qr(Zero, tol * tol)
    wp.expect_eq(D3, wp.vec3(0.0))
    wp.expect_eq(P3, Id)

    # Identity
    D3, P3 = symmetric_eigenvalues_qr(Id, tol * tol)
    wp.expect_eq(D3, wp.vec3(1.0))
    wp.expect_eq(wp.transpose(P3) * P3, Id)

    # rank 1
    v = wp.vec4(0.0, 1.0, 1.0, 0.0)
    Rank1 = wp.outer(v, v)
    D4, P4 = symmetric_eigenvalues_qr(Rank1, tol * tol)
    wp.expect_near(wp.max(D4), wp.length_sq(v), tol)
    Err4 = wp.transpose(P4) * wp.diag(D4) * P4 - Rank1
    wp.expect_near(wp.ddot(Err4, Err4), 0.0, tol)

    # rank 2
    v2 = wp.vec4(0.0, 0.5, -0.5, 0.0)
    Rank2 = Rank1 + wp.outer(v2, v2)
    D4, P4 = symmetric_eigenvalues_qr(Rank2, tol * tol)
    wp.expect_near(wp.max(D4), wp.length_sq(v), tol)
    wp.expect_near(D4[0] + D4[1] + D4[2] + D4[3], wp.length_sq(v) + wp.length_sq(v2), tol)
    Err4 = wp.transpose(P4) * wp.diag(D4) * P4 - Rank2
    wp.expect_near(wp.ddot(Err4, Err4), 0.0, tol)

    # rank 4
    v3 = wp.vec4(1.0, 2.0, 3.0, 4.0)
    v4 = wp.vec4(2.0, 1.0, 0.0, -1.0)
    Rank4 = Rank2 + wp.outer(v3, v3) + wp.outer(v4, v4)
    D4, P4 = symmetric_eigenvalues_qr(Rank4, tol * tol)
    Err4 = wp.transpose(P4) * wp.diag(D4) * P4 - Rank4
    wp.expect_near(wp.ddot(Err4, Err4), 0.0, tol)


@wp.kernel
def test_qr_inverse():
    rng = wp.rand_init(4356, wp.tid())
    M = wp.mat33(
        wp.randf(rng, 0.0, 10.0),
        wp.randf(rng, 0.0, 10.0),
        wp.randf(rng, 0.0, 10.0),
        wp.randf(rng, 0.0, 10.0),
        wp.randf(rng, 0.0, 10.0),
        wp.randf(rng, 0.0, 10.0),
        wp.randf(rng, 0.0, 10.0),
        wp.randf(rng, 0.0, 10.0),
        wp.randf(rng, 0.0, 10.0),
    )

    if wp.determinant(M) != 0.0:
        tol = 1.0e-8
        Mi = inverse_qr(M)
        Id = wp.identity(n=3, dtype=float)
        Err = M * Mi - Id
        wp.expect_near(wp.ddot(Err, Err), 0.0, tol)
        Err = Mi * M - Id
        wp.expect_near(wp.ddot(Err, Err), 0.0, tol)


devices = get_test_devices()
cuda_devices = get_selected_cuda_test_devices()


class TestFem(unittest.TestCase):
    pass


add_function_test(TestFem, "test_regular_quadrature", test_regular_quadrature)
add_function_test(TestFem, "test_closest_point_queries", test_closest_point_queries)
add_function_test(TestFem, "test_grad_decomposition", test_grad_decomposition, devices=devices)
add_function_test(TestFem, "test_integrate_gradient", test_integrate_gradient, devices=devices)
add_function_test(TestFem, "test_interpolate_gradient", test_interpolate_gradient, devices=devices)
add_function_test(TestFem, "test_vector_divergence_theorem", test_vector_divergence_theorem, devices=devices)
add_function_test(TestFem, "test_tensor_divergence_theorem", test_tensor_divergence_theorem, devices=devices)
add_function_test(TestFem, "test_grid_2d", test_grid_2d, devices=devices)
add_function_test(TestFem, "test_triangle_mesh", test_triangle_mesh, devices=devices)
add_function_test(TestFem, "test_quad_mesh", test_quad_mesh, devices=devices)
add_function_test(TestFem, "test_grid_3d", test_grid_3d, devices=devices)
add_function_test(TestFem, "test_tet_mesh", test_tet_mesh, devices=devices)
add_function_test(TestFem, "test_hex_mesh", test_hex_mesh, devices=devices)
add_function_test(TestFem, "test_nanogrid", test_nanogrid, devices=cuda_devices)
add_function_test(TestFem, "test_deformed_geometry", test_deformed_geometry, devices=devices)
add_function_test(TestFem, "test_dof_mapper", test_dof_mapper)
add_function_test(TestFem, "test_point_basis", test_point_basis)
add_function_test(TestFem, "test_particle_quadratures", test_particle_quadratures)
add_function_test(TestFem, "test_implicit_fields", test_implicit_fields)
add_kernel_test(TestFem, test_qr_eigenvalues, dim=1, devices=devices)
add_kernel_test(TestFem, test_qr_inverse, dim=100, devices=devices)


class TestFemShapeFunctions(unittest.TestCase):
    pass


add_function_test(TestFemShapeFunctions, "test_square_shape_functions", test_square_shape_functions)
add_function_test(TestFemShapeFunctions, "test_cube_shape_functions", test_cube_shape_functions)
add_function_test(TestFemShapeFunctions, "test_tri_shape_functions", test_tri_shape_functions)
add_function_test(TestFemShapeFunctions, "test_tet_shape_functions", test_tet_shape_functions)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2, failfast=True)
