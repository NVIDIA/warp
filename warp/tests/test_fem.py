# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import warp as wp
from warp.tests.test_base import *


from warp.fem.types import *
from warp.fem.geometry import Grid2D, Trimesh2D, Tetmesh
from warp.fem.geometry.closest_point import project_on_tri_at_origin, project_on_tet_at_origin
from warp.fem.space import make_polynomial_space, SymmetricTensorMapper
from warp.fem.field import make_test
from warp.fem.domain import Cells
from warp.fem.integrate import integrate
from warp.fem.operator import integrand
from warp.fem.quadrature import RegularQuadrature
from warp.fem.utils import unit_element

wp.init()

wp.config.mode = "debug"
wp.config.verify_cuda = True


@integrand
def linear_form(s: Sample, u: Field):
    return u(s)


def test_integrate_gradient(test_case, device):
    with wp.ScopedDevice(device):
        # Grid geometry
        geo = Grid2D(res=vec2i(5))

        # Domain and function spaces
        domain = Cells(geometry=geo)
        quadrature = RegularQuadrature(domain=domain, order=3)

        scalar_space = make_polynomial_space(geo, degree=3)

        u = scalar_space.make_field()
        u.dof_values = wp.zeros_like(u.dof_values, requires_grad=True)

        result = wp.empty(dtype=wp.float64, shape=(1), requires_grad=True)

        tape = wp.Tape()

        # forward pass
        with tape:
            integrate(linear_form, quadrature=quadrature, fields={"u": u}, output=result)

        tape.backward(result)

        test = make_test(space=scalar_space, domain=domain)
        rhs = integrate(linear_form, quadrature=quadrature, fields={"u": test})

        err = np.linalg.norm(rhs.numpy() - u.dof_values.grad.numpy())
        test_case.assertLess(err, 1.0e-8)


def _gen_trimesh(N):
    x = np.linspace(0.0, 1.0, N + 1)
    y = np.linspace(0.0, 1.0, N + 1)

    positions = np.transpose(np.meshgrid(x, y, indexing="ij")).reshape(-1, 2)

    cx, cy = np.meshgrid(np.arange(N, dtype=int), np.arange(N, dtype=int), indexing="ij")

    vidx = np.transpose(
        np.array(
            [
                (N + 1) * cx + cy,
                (N + 1) * (cx + 1) + cy,
                (N + 1) * (cx + 1) + (cy + 1),
                (N + 1) * cx + cy,
                (N + 1) * (cx + 1) + (cy + 1),
                (N + 1) * (cx) + (cy + 1),
            ]
        )
    ).reshape((-1, 3))

    return wp.array(positions, dtype=wp.vec2), wp.array(vidx, dtype=int)


def _gen_tetmesh(N):
    x = np.linspace(0.0, 1.0, N + 1)
    y = np.linspace(0.0, 1.0, N + 1)
    z = np.linspace(0.0, 1.0, N + 1)

    positions = np.transpose(np.meshgrid(x, y, z, indexing="ij")).reshape(-1, 3)

    # Global node indices for each cell
    cx, cy, cz = np.meshgrid(np.arange(N, dtype=int), np.arange(N, dtype=int), np.arange(N, dtype=int), indexing="ij")
    grid_vidx = np.array(
        [
            (N + 1) ** 2 * cx + (N + 1) * cy + cz,
            (N + 1) ** 2 * cx + (N + 1) * cy + cz + 1,
            (N + 1) ** 2 * cx + (N + 1) * (cy + 1) + cz,
            (N + 1) ** 2 * cx + (N + 1) * (cy + 1) + cz + 1,
            (N + 1) ** 2 * (cx + 1) + (N + 1) * cy + cz,
            (N + 1) ** 2 * (cx + 1) + (N + 1) * cy + cz + 1,
            (N + 1) ** 2 * (cx + 1) + (N + 1) * (cy + 1) + cz,
            (N + 1) ** 2 * (cx + 1) + (N + 1) * (cy + 1) + cz + 1,
        ]
    )

    # decompose grid cells into 5 tets
    tet_vidx = np.array(
        [
            [0, 1, 2, 4],
            [3, 2, 1, 7],
            [5, 1, 7, 4],
            [6, 7, 4, 2],
            [4, 1, 2, 7],
        ]
    )

    # Convert to 3d index coordinates
    vidx_coords = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ]
    )
    tet_coords = vidx_coords[tet_vidx]

    # Symmetry bits for each cell
    ox, oy, oz = np.meshgrid(
        np.arange(N, dtype=int) % 2, np.arange(N, dtype=int) % 2, np.arange(N, dtype=int) % 2, indexing="ij"
    )
    tet_coords = np.broadcast_to(tet_coords, shape=(*ox.shape, *tet_coords.shape))

    # Flip coordinates according to symmetry
    ox_bk = np.broadcast_to(ox.reshape(*ox.shape, 1, 1), tet_coords.shape[:-1])
    oy_bk = np.broadcast_to(oy.reshape(*ox.shape, 1, 1), tet_coords.shape[:-1])
    oz_bk = np.broadcast_to(oz.reshape(*ox.shape, 1, 1), tet_coords.shape[:-1])
    tet_coords_x = tet_coords[..., 0] ^ ox_bk
    tet_coords_y = tet_coords[..., 1] ^ oy_bk
    tet_coords_z = tet_coords[..., 2] ^ oz_bk

    # Back to local vertex indices
    corner_indices = 4 * tet_coords_x + 2 * tet_coords_y + tet_coords_z

    # Now go from cell-local to global node indices
    # There must be a nicer way than this, but for example purposes this works
    corner_indices = corner_indices.reshape(-1, 4)
    grid_vidx = grid_vidx.reshape((8, -1, 1))
    grid_vidx = np.broadcast_to(grid_vidx, shape=(8, grid_vidx.shape[1], 5))
    grid_vidx = grid_vidx.reshape((8, -1))

    node_indices = np.arange(corner_indices.shape[0])
    tet_grid_vidx = np.transpose(
        [
            grid_vidx[corner_indices[:, 0], node_indices],
            grid_vidx[corner_indices[:, 1], node_indices],
            grid_vidx[corner_indices[:, 2], node_indices],
            grid_vidx[corner_indices[:, 3], node_indices],
        ]
    )

    return wp.array(positions, dtype=wp.vec3), wp.array(tet_grid_vidx, dtype=int)


def test_triangle_mesh(test_case, device):
    N = 3

    with wp.ScopedDevice(device):
        positions, tri_vidx = _gen_trimesh(N)

    geo = Trimesh2D(tri_vertex_indices=tri_vidx, positions=positions)

    test_case.assertEqual(geo.cell_count(), 2 * (N) ** 2)
    test_case.assertEqual(geo.vertex_count(), (N + 1) ** 2)
    test_case.assertEqual(geo.side_count(), 2 * (N + 1) * N + (N**2))
    test_case.assertEqual(geo.boundary_side_count(), 4 * N)


def test_tet_mesh(test_case, device):
    N = 3

    with wp.ScopedDevice(device):
        positions, tet_vidx = _gen_tetmesh(N)

    geo = Tetmesh(tet_vertex_indices=tet_vidx, positions=positions)

    test_case.assertEqual(geo.cell_count(), 5 * (N) ** 3)
    test_case.assertEqual(geo.vertex_count(), (N + 1) ** 3)
    test_case.assertEqual(geo.side_count(), 6 * (N + 1) * N**2 + (N**3) * 4)
    test_case.assertEqual(geo.boundary_side_count(), 12 * N * N)


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


def test_closest_point_queries(test_case, device):
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
    expected_coords = np.array([[0.0, 0.0, 0.0], [1.0/6.0, 1.0/6.0, 1.0/6.0], [1.0/3.0, 1.0/3.0, 1.0/3.0], [1.0/3.0, 1.0/3.0, 1.0/3.0]])

    sq_dist = wp.empty(shape=points.shape, dtype=float, device=device)
    coords = wp.empty(shape=points.shape, dtype=Coords, device=device)
    wp.launch(
        _test_closest_point_on_tet_kernel, dim=points.shape, device=device, inputs=[e0, e1, e2, points, sq_dist, coords]
    )

    assert_np_equal(coords.numpy(), expected_coords, tol = 1.e-4)
    assert_np_equal(sq_dist.numpy(), expected_sq_dist, tol = 1.e-4)


def test_regular_quadrature(test_case, device):
    from warp.fem.geometry.element import LinearEdge, Triangle, Polynomial

    for family in Polynomial:
        # test integrating monomials
        for degree in range(8):
            coords, weights = LinearEdge().instantiate_quadrature(degree, family=family)
            res = sum(w * pow(c[0], degree) for w, c in zip(weights, coords))
            ref = 1.0 / (degree + 1)

            test_case.assertAlmostEqual(ref, res, places=4)

        # test integrating y^k1 (1 - x)^k2 on triangle using transformation to square
        for x_degree in range(4):
            for y_degree in range(4):
                coords, weights = Triangle().instantiate_quadrature(x_degree + y_degree, family=family)
                res = 0.5 * sum(w * pow(1.0 - c[1], x_degree) * pow(c[2], y_degree) for w, c in zip(weights, coords))

                ref = 1.0 / ((x_degree + y_degree + 2) * (y_degree + 1))
                # print(x_degree, y_degree, family, len(coords), res, ref)
                test_case.assertAlmostEqual(ref, res, places=4)

    # test integrating y^k1 (1 - x)^k2 on triangle using direct formulas
    for x_degree in range(5):
        for y_degree in range(5):
            coords, weights = Triangle().instantiate_quadrature(x_degree + y_degree, family=None)
            res = 0.5 * sum(w * pow(1.0 - c[1], x_degree) * pow(c[2], y_degree) for w, c in zip(weights, coords))

            ref = 1.0 / ((x_degree + y_degree + 2) * (y_degree + 1))
            test_case.assertAlmostEqual(ref, res, places=4)


def test_dof_mapper(test_case, device):
    matrix_types = [wp.mat22, wp.mat33]
    for mapping in SymmetricTensorMapper.Mapping:
        for dtype in matrix_types:
            mapper = SymmetricTensorMapper(dtype, mapping=mapping)
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
                test_case.assertAlmostEqual(frob_norm2, 1.0, places=6)


def register(parent):
    devices = get_test_devices()

    class TestFem(parent):
        pass

    add_function_test(TestFem, "test_regular_quadrature", test_regular_quadrature)
    add_function_test(TestFem, "test_closest_point_queries", test_closest_point_queries)
    add_function_test(TestFem, "test_integrate_gradient", test_integrate_gradient, devices=devices)
    add_function_test(TestFem, "test_triangle_mesh", test_triangle_mesh, devices=devices)
    add_function_test(TestFem, "test_tet_mesh", test_tet_mesh, devices=devices)
    add_function_test(TestFem, "test_dof_mapper", test_dof_mapper)

    return TestFem


if __name__ == "__main__":
    c = register(unittest.TestCase)
    unittest.main(verbosity=2)
