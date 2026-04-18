# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import platform
import unittest

import numpy as np

import warp as wp
import warp.fem as fem
from warp._src.fem.geometry.closest_point import project_on_tet_at_origin, project_on_tri_at_origin
from warp.fem import Coords, Domain, Sample, integrand, make_free_sample
from warp.fem.utils import grid_to_hexes, grid_to_quads, grid_to_tets, grid_to_tris
from warp.tests.unittest_utils import *


def _gen_trimesh(Nx, Ny):
    x = np.linspace(0.0, 1.0, Nx + 1)
    y = np.linspace(0.0, 1.0, Ny + 1)

    positions = np.transpose(np.meshgrid(x, y, indexing="ij"), axes=(1, 2, 0)).reshape(-1, 2)

    vidx = grid_to_tris(Nx, Ny)

    return wp.array(positions, dtype=wp.vec2), wp.array(vidx, dtype=int)


def _gen_quadmesh(N):
    x = np.linspace(0.0, 1.0, N + 1)
    y = np.linspace(0.0, 1.0, N + 1)

    positions = np.transpose(np.meshgrid(x, y, indexing="ij"), axes=(1, 2, 0)).reshape(-1, 2)

    vidx = grid_to_quads(N, N)

    return wp.array(positions, dtype=wp.vec2), wp.array(vidx, dtype=int)


def _gen_tetmesh(Nx, Ny, Nz):
    x = np.linspace(0.0, 1.0, Nx + 1)
    y = np.linspace(0.0, 1.0, Ny + 1)
    z = np.linspace(0.0, 1.0, Nz + 1)

    positions = np.transpose(np.meshgrid(x, y, z, indexing="ij"), axes=(1, 2, 3, 0)).reshape(-1, 3)

    vidx = grid_to_tets(Nx, Ny, Nz)

    return wp.array(positions, dtype=wp.vec3), wp.array(vidx, dtype=int)


def _gen_hexmesh(N):
    x = np.linspace(0.0, 1.0, N + 1)
    y = np.linspace(0.0, 1.0, N + 1)
    z = np.linspace(0.0, 1.0, N + 1)

    positions = np.transpose(np.meshgrid(x, y, z, indexing="ij"), axes=(1, 2, 3, 0)).reshape(-1, 3)

    vidx = grid_to_hexes(N, N, N)

    return wp.array(positions, dtype=wp.vec3), wp.array(vidx, dtype=int)


@fem.integrand(kernel_options={"enable_backward": False})
def _test_geo_cells(
    s: fem.Sample,
    domain: fem.Domain,
    cell_measures: wp.array(dtype=float),
):
    wp.atomic_add(cell_measures, s.element_index, fem.measure(domain, s) * s.qp_weight)


@fem.integrand(kernel_options={"enable_backward": False, "max_unroll": 2})
def _test_cell_lookup(s: fem.Sample, domain: fem.Domain, cell_filter: wp.array(dtype=int)):
    pos = domain(s)

    s_guess = fem.lookup(domain, pos, s)
    wp.expect_eq(s_guess.element_index, s.element_index)
    wp.expect_near(domain(s_guess), pos, 0.001)

    s_noguess = fem.lookup(domain, pos)
    wp.expect_eq(s_noguess.element_index, s.element_index)
    wp.expect_near(domain(s_noguess), pos, 0.001)

    # Filtered lookup
    max_dist = 10.0
    filter_target = 1
    s_filter = fem.lookup(domain, pos, max_dist, cell_filter, filter_target)
    wp.expect_eq(s_filter.element_index, 0)

    if s.element_index != 0:
        # test closest point optimality
        pos_f = domain(s_filter)
        pos_f += 0.1 * (pos - pos_f)
        coord_proj, _sq_dist = fem.element_closest_point(domain, s_filter.element_index, pos_f)
        wp.expect_near(coord_proj, s_filter.element_coords, 0.001)

    # test that extrapolated coordinates yield back correct position
    s_filter.element_coords = fem.element_coordinates(domain, s_filter.element_index, pos)
    wp.expect_near(domain(s_filter), pos, 0.001)


@fem.integrand(kernel_options={"enable_backward": False, "max_unroll": 1})
def _test_geo_sides(
    s: fem.Sample,
    domain: fem.Domain,
    ref_measure: float,
    side_measures: wp.array(dtype=float),
):
    side_index = s.element_index
    coords = s.element_coords

    cells = fem.cells(domain)

    inner_s = fem.to_inner_cell(domain, s)
    outer_s = fem.to_outer_cell(domain, s)

    pos_side = domain(s)
    pos_inner = cells(inner_s)
    pos_outer = cells(outer_s)

    for k in range(type(pos_side).length):
        wp.expect_near(pos_side[k], pos_inner[k], 0.0001)
        wp.expect_near(pos_side[k], pos_outer[k], 0.0001)

    inner_side_s = fem.to_cell_side(domain, inner_s, side_index)
    outer_side_s = fem.to_cell_side(domain, outer_s, side_index)

    wp.expect_near(coords, inner_side_s.element_coords, 0.0001)
    wp.expect_near(coords, outer_side_s.element_coords, 0.0001)
    wp.expect_near(coords, fem.element_coordinates(domain, side_index, pos_side), 0.001)

    area = fem.measure(domain, s)
    wp.atomic_add(side_measures, side_index, area * s.qp_weight)

    F = fem.deformation_gradient(domain, s)
    F_det = fem.Geometry._element_measure(F)
    wp.expect_near(F_det * ref_measure, area)


@fem.integrand(kernel_options={"enable_backward": False, "max_unroll": 1})
def _test_side_normals(
    s: fem.Sample,
    domain: fem.Domain,
):
    # test consistency of side normal, measure, and deformation gradient
    F = fem.deformation_gradient(domain, s)

    nor = fem.normal(domain, s)
    F_cross = fem.Geometry._element_normal(F)

    for k in range(type(nor).length):
        wp.expect_near(F_cross[k], nor[k], 0.0001)


def _launch_test_geometry_kernel(geo: fem.Geometry, device):
    cell_measures = wp.zeros(dtype=float, device=device, shape=geo.cell_count())
    cell_quadrature = fem.RegularQuadrature(fem.Cells(geo), order=2)

    side_measures = wp.zeros(dtype=float, device=device, shape=geo.side_count())
    side_quadrature = fem.RegularQuadrature(fem.Sides(geo), order=2)

    with wp.ScopedDevice(device):
        fem.interpolate(_test_geo_cells, at=cell_quadrature, values={"cell_measures": cell_measures})

        cell_filter = np.zeros(geo.cell_count(), dtype=int)
        cell_filter[0] = 1
        cell_filter = wp.array(cell_filter, dtype=int)
        fem.interpolate(_test_cell_lookup, at=cell_quadrature, values={"cell_filter": cell_filter})

        fem.interpolate(
            _test_geo_sides,
            at=side_quadrature,
            values={"side_measures": side_measures, "ref_measure": geo.reference_side().prototype.measure()},
        )

        if geo.side_normal is not None:
            fem.interpolate(_test_side_normals, at=side_quadrature)

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
        positions, tri_vidx = _gen_trimesh(N, N)

    geo = fem.Trimesh2D(tri_vertex_indices=tri_vidx, positions=positions, build_bvh=True)

    test.assertEqual(geo.cell_count(), 2 * (N) ** 2)
    test.assertEqual(geo.vertex_count(), (N + 1) ** 2)
    test.assertEqual(geo.side_count(), 2 * (N + 1) * N + (N**2))
    test.assertEqual(geo.boundary_side_count(), 4 * N)

    side_measures, cell_measures = _launch_test_geometry_kernel(geo, device)

    assert_np_equal(cell_measures.numpy(), np.full(cell_measures.shape, 0.5 / (N**2)), tol=1.0e-4)
    test.assertAlmostEqual(np.sum(side_measures.numpy()), 2 * (N + 1) + N * math.sqrt(2.0), places=4)

    # 3d

    positions = positions.numpy()
    positions = np.hstack((positions, np.ones((positions.shape[0], 1))))
    positions = wp.array(positions, device=device, dtype=wp.vec3)

    geo = fem.Trimesh3D(tri_vertex_indices=tri_vidx, positions=positions, build_bvh=True)

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

    geo = fem.Quadmesh2D(quad_vertex_indices=quad_vidx, positions=positions, build_bvh=True)

    test.assertEqual(geo.cell_count(), N**2)
    test.assertEqual(geo.vertex_count(), (N + 1) ** 2)
    test.assertEqual(geo.side_count(), 2 * (N + 1) * N)
    test.assertEqual(geo.boundary_side_count(), 4 * N)

    side_measures, cell_measures = _launch_test_geometry_kernel(geo, device)

    assert_np_equal(side_measures.numpy(), np.full(side_measures.shape, 1.0 / (N)), tol=1.0e-4)
    assert_np_equal(cell_measures.numpy(), np.full(cell_measures.shape, 1.0 / (N**2)), tol=1.0e-4)

    # 3d

    positions = positions.numpy()
    positions = np.hstack((positions, np.ones((positions.shape[0], 1))))
    positions = wp.array(positions, device=device, dtype=wp.vec3)

    geo = fem.Quadmesh3D(quad_vertex_indices=quad_vidx, positions=positions, build_bvh=True)

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
        positions, tet_vidx = _gen_tetmesh(N, N, N)

    geo = fem.Tetmesh(tet_vertex_indices=tet_vidx, positions=positions, build_bvh=True)

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

    geo = fem.Hexmesh(hex_vertex_indices=tet_vidx, positions=positions, build_bvh=True)

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


@wp.func
def _refinement_field(x: wp.vec3):
    return 4.0 * (wp.length(x) - 0.5)


def test_adaptive_nanogrid(test, device):
    # 3 res-1 voxels, 8 res-0 voxels

    if platform.system() == "Windows":
        test.skipTest("Skipping test due to NVRTC bug on Windows")

    res0 = wp.array(
        [
            [2, 2, 0],
            [2, 3, 0],
            [3, 2, 0],
            [3, 3, 0],
            [2, 2, 1],
            [2, 3, 1],
            [3, 2, 1],
            [3, 3, 1],
        ],
        dtype=int,
        device=device,
    )
    res1 = wp.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 1, 0],
        ],
        dtype=int,
        device=device,
    )
    grid0 = wp.Volume.allocate_by_voxels(res0, 0.5, device=device)
    grid1 = wp.Volume.allocate_by_voxels(res1, 1.0, device=device)
    geo = fem.adaptive_nanogrid_from_hierarchy([grid0, grid1])

    test.assertEqual(geo.cell_count(), 3 + 8)
    test.assertEqual(geo.vertex_count(), 2 * 9 + 27 - 8)
    test.assertEqual(geo.side_count(), 2 * 4 + 6 * 2 + (3 * (2 + 1) * 2**2 - 6))
    test.assertEqual(geo.boundary_side_count(), 2 * 4 + 4 * 2 + (4 * 4 - 4))
    # test.assertEqual(geo.edge_count(), 6 * 4 + 9 + (3 * 2 * (2 + 1) ** 2 - 12))
    test.assertEqual(geo.stacked_face_count(), geo.side_count() + 2)
    test.assertEqual(geo.stacked_edge_count(), 6 * 4 + 9 + (3 * 2 * (2 + 1) ** 2 - 12) + 7)

    side_measures, cell_measures = _launch_test_geometry_kernel(geo, device)

    test.assertAlmostEqual(np.sum(cell_measures.numpy()), 4.0, places=4)
    test.assertAlmostEqual(np.sum(side_measures.numpy()), 20 + 3.0, places=4)

    # Test with non-graded geometry
    ref_field = fem.ImplicitField(fem.Cells(geo), func=_refinement_field)
    non_graded_geo = fem.adaptive_nanogrid_from_field(grid1, level_count=3, refinement_field=ref_field)
    _launch_test_geometry_kernel(non_graded_geo, device)

    # Test automatic grading
    graded_geo = fem.adaptive_nanogrid_from_field(grid1, level_count=3, refinement_field=ref_field, grading="face")
    test.assertEqual(non_graded_geo.cell_count() + 7, graded_geo.cell_count())


@integrand
def _rigid_deformation_field(s: Sample, domain: Domain, translation: wp.vec3, rotation: wp.vec3, scale: float):
    q = wp.quat_from_axis_angle(wp.normalize(rotation), wp.length(rotation))
    return translation + scale * wp.quat_rotate(q, domain(s)) - domain(s)


def test_deformed_geometry(test, device):
    N = 3

    translation = [1.0, 2.0, 3.0]
    rotation = [0.0, math.pi / 4.0, 0.0]
    scale = 2.0

    with wp.ScopedDevice(device):
        positions, tet_vidx = _gen_tetmesh(N, N, N)

        geo = fem.Tetmesh(tet_vertex_indices=tet_vidx, positions=positions)

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

        deformed_geo.build_bvh()
        side_measures, cell_measures = _launch_test_geometry_kernel(deformed_geo, device)

        test.assertAlmostEqual(
            np.sum(cell_measures.numpy()), scale**3, places=4, msg=f"cell_measures = {cell_measures.numpy()}"
        )
        test.assertAlmostEqual(
            np.sum(side_measures.numpy()), scale**2 * (0.5 * 6 * (N + 1) + N * 2 * math.sqrt(3.0)), places=4
        )

        @fem.cache.dynamic_kernel(suffix=deformed_geo.name, kernel_options={"enable_backward": False})
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


def test_deformed_geometry_codimensional(test, device):
    N = 3

    translation = [1.0, 2.0, 3.0]
    rotation = [0.0, math.pi / 4.0, 0.0]
    scale = 2.0

    with wp.ScopedDevice(device):
        # Test with Trimesh3d (different space and cell dimensions)
        positions, tri_vidx = _gen_trimesh(N, N)
        positions = positions.numpy()
        positions = np.hstack((positions, np.ones((positions.shape[0], 1))))
        positions = wp.array(positions, device=device, dtype=wp.vec3)

        geo = fem.Trimesh3D(tri_vertex_indices=tri_vidx, positions=positions)

        vector_space = fem.make_polynomial_space(geo, dtype=wp.vec3, degree=1)
        pos_field = vector_space.make_field()
        fem.interpolate(
            _rigid_deformation_field,
            dest=pos_field,
            values={"translation": translation, "rotation": rotation, "scale": scale},
        )

        deformed_geo = pos_field.make_deformed_geometry()

        @fem.cache.dynamic_kernel(suffix=deformed_geo.name, kernel_options={"enable_backward": False})
        def _test_deformed_geometry_normal_codimensional(
            geo_arg: geo.CellArg, def_arg: deformed_geo.CellArg, rotation: wp.vec3
        ):
            i = wp.tid()

            s = make_free_sample(i, Coords(0.5, 0.5, 0.0))
            geo_n = geo.cell_normal(geo_arg, s)
            def_n = deformed_geo.cell_normal(def_arg, s)

            q = wp.quat_from_axis_angle(wp.normalize(rotation), wp.length(rotation))
            wp.expect_near(wp.quat_rotate(q, geo_n), def_n, 0.001)

        wp.launch(
            _test_deformed_geometry_normal_codimensional,
            dim=geo.cell_count(),
            inputs=[
                geo.cell_arg_value(wp.get_device()),
                deformed_geo.cell_arg_value(wp.get_device()),
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


# -- Device setup and test registration --

devices = get_test_devices()
cuda_devices = get_selected_cuda_test_devices()


class TestFemGeometry(unittest.TestCase):
    pass


add_function_test(TestFemGeometry, "test_grid_2d", test_grid_2d, devices=devices)
add_function_test(TestFemGeometry, "test_triangle_mesh", test_triangle_mesh, devices=devices)
add_function_test(TestFemGeometry, "test_quad_mesh", test_quad_mesh, devices=devices)
add_function_test(TestFemGeometry, "test_grid_3d", test_grid_3d, devices=devices)
add_function_test(TestFemGeometry, "test_tet_mesh", test_tet_mesh, devices=devices)
add_function_test(TestFemGeometry, "test_hex_mesh", test_hex_mesh, devices=devices)
add_function_test(TestFemGeometry, "test_nanogrid", test_nanogrid, devices=cuda_devices)
add_function_test(TestFemGeometry, "test_adaptive_nanogrid", test_adaptive_nanogrid, devices=cuda_devices)
add_function_test(TestFemGeometry, "test_deformed_geometry", test_deformed_geometry, devices=devices)
add_function_test(
    TestFemGeometry, "test_deformed_geometry_codimensional", test_deformed_geometry_codimensional, devices=devices
)
add_function_test(TestFemGeometry, "test_closest_point_queries", test_closest_point_queries)

if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
