# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the fp64 FEM pipeline.

Verifies that the fp64 path preserves full double precision across all geometry
types, polynomial degrees, and operator combinations. Uses positions with an
offset of 1 + 2^{-30} that is exact in fp64 but rounds to 1.0 in fp32, so any
silent downcast is immediately detectable.
"""

import unittest

import numpy as np

import warp as wp
import warp.fem as fem
from warp.sparse import bsr_mv
from warp.tests.unittest_utils import add_function_test, get_test_devices

vec2d = wp.vec2d
vec3d = wp.vec3d

# Position offset that is exact in fp64 but truncates to 1.0 in fp32
OFFSET = 1.0 + 2.0**-30


# -- Integrands --


@fem.integrand
def volume_integrand(s: fem.Sample, domain: fem.Domain):
    return fem.scalar_type(domain)(1.0)


@fem.integrand
def position_x_integrand(s: fem.Sample, domain: fem.Domain):
    x = domain(s)
    return x[0]


@fem.integrand
def mass_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return u(s) * v(s)


@fem.integrand
def diffusion_form(s: fem.Sample, u: fem.Field, v: fem.Field):
    return wp.dot(fem.grad(u, s), fem.grad(v, s))


@fem.integrand
def boundary_integral(s: fem.Sample, domain: fem.Domain):
    return fem.scalar_type(domain)(1.0)


@fem.integrand
def lookup_integrand(s: fem.Sample, domain: fem.Domain):
    position = domain(s)
    closest = fem.lookup(domain, position, s)
    return domain(closest)[0]


# -- Geometry factories --


def _make_tetmesh():
    positions = wp.array(
        np.array([[OFFSET, 0, 0], [OFFSET + 1, 0, 0], [OFFSET, 1, 0], [OFFSET, 0, 1]], dtype=np.float64),
        dtype=vec3d,
    )
    indices = wp.array(np.array([[0, 1, 2, 3]], dtype=np.int32), ndim=2)
    return fem.Tetmesh(indices, positions)


def _make_trimesh2d():
    positions = wp.array(
        np.array([[OFFSET, 0], [OFFSET + 1, 0], [OFFSET, 1]], dtype=np.float64),
        dtype=vec2d,
    )
    indices = wp.array(np.array([[0, 1, 2]], dtype=np.int32), ndim=2)
    return fem.Trimesh2D(indices, positions)


def _make_trimesh3d():
    positions = wp.array(
        np.array([[OFFSET, 0, 0], [OFFSET + 1, 0, 0], [OFFSET, 1, 0]], dtype=np.float64),
        dtype=vec3d,
    )
    indices = wp.array(np.array([[0, 1, 2]], dtype=np.int32), ndim=2)
    return fem.Trimesh3D(indices, positions)


def _make_quadmesh2d():
    positions = wp.array(
        np.array([[OFFSET, 0], [OFFSET + 1, 0], [OFFSET + 1, 1], [OFFSET, 1]], dtype=np.float64),
        dtype=vec2d,
    )
    indices = wp.array(np.array([[0, 1, 2, 3]], dtype=np.int32), ndim=2)
    return fem.Quadmesh2D(indices, positions)


def _make_hexmesh():
    positions = wp.array(
        np.array(
            [
                [OFFSET, 0, 0],
                [OFFSET + 1, 0, 0],
                [OFFSET + 1, 1, 0],
                [OFFSET, 1, 0],
                [OFFSET, 0, 1],
                [OFFSET + 1, 0, 1],
                [OFFSET + 1, 1, 1],
                [OFFSET, 1, 1],
            ],
            dtype=np.float64,
        ),
        dtype=vec3d,
    )
    indices = wp.array(np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.int32), ndim=2)
    return fem.Hexmesh(indices, positions)


def _make_grid2d():
    return fem.Grid2D(
        res=wp.vec2i(1, 1),
        bounds_lo=vec2d(OFFSET, 0.0),
        bounds_hi=vec2d(OFFSET + 1.0, 1.0),
        scalar_type=wp.float64,
    )


def _make_grid3d():
    return fem.Grid3D(
        res=wp.vec3i(1, 1, 1),
        bounds_lo=vec3d(OFFSET, 0.0, 0.0),
        bounds_hi=vec3d(OFFSET + 1.0, 1.0, 1.0),
        scalar_type=wp.float64,
    )


def _make_nanogrid():
    voxel_points = wp.array([[0, 0, 0]], dtype=wp.vec3i)
    volume = wp.Volume.allocate_by_voxels(voxel_points=voxel_points, voxel_size=1.0, translation=(OFFSET, 0.0, 0.0))
    return fem.Nanogrid(volume, scalar_type=wp.float64)


GEOS = {
    "Tetmesh": (_make_tetmesh, 1.0 / 6.0, 3),
    "Trimesh2D": (_make_trimesh2d, 0.5, 2),
    "Trimesh3D": (_make_trimesh3d, 0.5, 2),
    "Quadmesh2D": (_make_quadmesh2d, 1.0, 2),
    "Hexmesh": (_make_hexmesh, 1.0, 3),
    "Grid2D": (_make_grid2d, 1.0, 2),
    "Grid3D": (_make_grid3d, 1.0, 3),
}
"""geo_name -> (factory, expected_volume, dimension)"""


# -- Test functions --


def test_fp64_volume_integral(test, device):
    """Test volume integration at fp64 for all geometries."""
    with wp.ScopedDevice(device):
        for geo_name, (factory, expected_vol, _dim) in GEOS.items():
            with test.subTest(geo=geo_name):
                geo = factory()
                test.assertEqual(geo.scalar_type, wp.float64)
                domain = fem.Cells(geo)
                vol = fem.integrate(volume_integrand, domain=domain)
                np.testing.assert_allclose(vol, expected_vol, rtol=1e-14, err_msg=f"{geo_name} volume")


def test_fp64_position_integral(test, device):
    """Verify that position integral detects OFFSET truncation to fp32."""
    with wp.ScopedDevice(device):
        for geo_name, (factory, expected_vol, _dim) in GEOS.items():
            with test.subTest(geo=geo_name):
                geo = factory()
                domain = fem.Cells(geo)
                result = fem.integrate(position_x_integrand, domain=domain)
                # Expected = integral of x over domain = volume * x_centroid
                if geo_name in ("Quadmesh2D", "Hexmesh", "Grid2D", "Grid3D"):
                    expected = expected_vol * (OFFSET + 0.5)
                elif geo_name in ("Tetmesh",):
                    expected = expected_vol * (OFFSET + 0.25)
                elif geo_name in ("Trimesh2D", "Trimesh3D"):
                    expected = expected_vol * (OFFSET + 1.0 / 3.0)
                else:
                    expected = result
                np.testing.assert_allclose(result, expected, rtol=1e-14, err_msg=f"{geo_name} position")


def test_fp64_mass_matrix(test, device):
    """Test mass matrix assembly at fp64 for all geometries, degrees 1-2."""
    with wp.ScopedDevice(device):
        for geo_name, (factory, expected_vol, _dim) in GEOS.items():
            for degree in (1, 2):
                with test.subTest(geo=geo_name, degree=degree):
                    geo = factory()
                    space = fem.make_polynomial_space(geo, degree=degree)
                    test.assertEqual(space.dtype, wp.float64)

                    domain = fem.Cells(geo)
                    trial = fem.make_trial(space=space, domain=domain)
                    test_f = fem.make_test(space=space, domain=domain)
                    matrix = fem.integrate(mass_form, fields={"u": trial, "v": test_f})

                    # Sum of mass matrix entries = integral of (sum phi_i)(sum phi_j) = volume
                    nnz = matrix.nnz_sync()
                    total = float(np.sum(matrix.values.numpy()[:nnz]))
                    np.testing.assert_allclose(
                        total, expected_vol, rtol=1e-12, err_msg=f"{geo_name} P{degree} mass matrix total"
                    )


def test_fp64_diffusion_matrix(test, device):
    """Test diffusion (stiffness) matrix assembly at fp64 for all geometries."""
    with wp.ScopedDevice(device):
        for geo_name, (factory, _expected_vol, _dim) in GEOS.items():
            for degree in (1, 2):
                with test.subTest(geo=geo_name, degree=degree):
                    geo = factory()
                    space = fem.make_polynomial_space(geo, degree=degree)
                    domain = fem.Cells(geo)
                    trial = fem.make_trial(space=space, domain=domain)
                    test_f = fem.make_test(space=space, domain=domain)
                    matrix = fem.integrate(diffusion_form, fields={"u": trial, "v": test_f})

                    test.assertEqual(matrix.scalar_type, wp.float64, f"{geo_name} P{degree} stiffness dtype")

                    # Stiffness matrix row sums should be zero (constant functions are in the kernel).
                    n = matrix.nrow
                    ones = wp.array(np.ones(n, dtype=np.float64), dtype=wp.float64)
                    row_sums = wp.zeros(n, dtype=wp.float64)

                    bsr_mv(matrix, ones, row_sums)
                    np.testing.assert_allclose(
                        row_sums.numpy(),
                        0.0,
                        atol=1e-12,
                        err_msg=f"{geo_name} P{degree} stiffness row sums",
                    )


def test_fp64_boundary_integral(test, device):
    """Test boundary side integration at fp64 for geometries with sides."""
    with wp.ScopedDevice(device):
        for geo_name, (factory, _expected_vol, _dim) in GEOS.items():
            if geo_name == "Trimesh3D":
                continue
            with test.subTest(geo=geo_name):
                geo = factory()
                boundary = fem.BoundarySides(geo)
                perimeter = fem.integrate(boundary_integral, domain=boundary)
                test.assertGreater(perimeter, 0.0, f"{geo_name} boundary integral should be positive")

                if geo_name in ("Grid2D", "Quadmesh2D"):
                    np.testing.assert_allclose(perimeter, 4.0, rtol=1e-14, err_msg=f"{geo_name} perimeter")
                elif geo_name in ("Grid3D", "Hexmesh"):
                    np.testing.assert_allclose(perimeter, 6.0, rtol=1e-14, err_msg=f"{geo_name} surface area")


def test_fp64_interpolation(test, device):
    """Verify that interpolation at fp64 preserves precision."""

    @fem.integrand
    def linear_field(s: fem.Sample, domain: fem.Domain):
        x = domain(s)
        return x[0]

    with wp.ScopedDevice(device):
        for geo_name in ("Grid2D", "Grid3D", "Tetmesh", "Quadmesh2D", "Hexmesh"):
            with test.subTest(geo=geo_name):
                factory, _vol, _dim = GEOS[geo_name]
                geo = factory()
                space = fem.make_polynomial_space(geo, degree=1)
                domain = fem.Cells(geo)
                field = space.make_field()
                fem.interpolate(linear_field, dest=field)

                values = field.dof_values.numpy()
                test.assertTrue(
                    np.all((np.abs(values - OFFSET) < 1e-10) | (np.abs(values - (OFFSET + 1.0)) < 1e-10)),
                    f"{geo_name}: interpolated values should be OFFSET or OFFSET+1, got {values}",
                )


def test_fp64_scalar_type_operator(test, device):
    """Verify that fem.scalar_type() operator resolves correctly in fp64 integrands."""

    @fem.integrand
    def scaled_volume(s: fem.Sample, domain: fem.Domain):
        half = fem.scalar_type(domain)(0.5)
        return half * fem.scalar_type(domain)(1.0)

    with wp.ScopedDevice(device):
        for geo_name in ("Grid2D", "Grid3D", "Tetmesh"):
            with test.subTest(geo=geo_name):
                factory, expected_vol, _dim = GEOS[geo_name]
                geo = factory()
                domain = fem.Cells(geo)
                result = fem.integrate(scaled_volume, domain=domain)
                np.testing.assert_allclose(result, 0.5 * expected_vol, rtol=1e-14, err_msg=f"{geo_name}")


def test_fp64_nanogrid_volume(test, device):
    """Verify that an fp64 Nanogrid integrates its unit cell to a volume of 1.0 at double precision."""
    with wp.ScopedDevice(device):
        geo = _make_nanogrid()
        test.assertEqual(geo.scalar_type, wp.float64)
        domain = fem.Cells(geo)
        vol = fem.integrate(volume_integrand, domain=domain)
        np.testing.assert_allclose(vol, 1.0, rtol=1e-14, err_msg="Nanogrid volume")


def test_fp64_lookup(test, device):
    with wp.ScopedDevice(device):
        for geo_name, (factory, _, _) in GEOS.items():
            geo = factory()
            # don't build BVH for structured geometries
            if geo_name not in {"Grid2D", "Grid3D"}:
                geo.build_bvh()
            domain = fem.Cells(geo)
            # for now just test that this doesn't crash
            _ = fem.integrate(lookup_integrand, domain=domain)


def test_fp64_grid_particle_lookup(test, device):
    """Preserve particle cell ownership and coordinates across FP64 boundaries."""
    with wp.ScopedDevice(device):
        for dim, grid_type, res_type, pos_type in (
            (2, fem.Grid2D, wp.vec2i, wp.vec2d),
            (3, fem.Grid3D, wp.vec3i, wp.vec3d),
        ):
            for env_count in (1, 2):
                for nearest in (False, True):
                    with test.subTest(dimension=dim, environments=env_count, nearest=nearest):
                        grid = grid_type(res_type(2), scalar_type=wp.float64, env_count=env_count)
                        cell_stride = 2 ** (dim - 1)
                        cells_per_env = 2**dim
                        data = np.full((3 * env_count, dim), 0.25, dtype=np.float64)
                        data[:, 0] = np.tile([0.5 - 2.0**-32, 0.5 + 2.0**-32, 0.4], env_count)
                        envs = np.repeat(np.arange(env_count, dtype=np.int32), 3)
                        local_cells = np.full(len(data), cell_stride) if nearest else (data[:, 0] >= 0.5) * cell_stride
                        expected_cells = local_cells + envs * cells_per_env
                        expected_coords = np.zeros((len(data), 3), dtype=np.float64)
                        expected_coords[:, :dim] = data * 2.0
                        expected_coords[:, 0] -= local_cells / cell_stride
                        if nearest:
                            mask = (np.arange(grid.cell_count()) % cells_per_env >= cell_stride).astype(np.int32)
                            domain = fem.Cells(fem.ExplicitGeometryPartition(grid, wp.array(mask, dtype=int)))
                        else:
                            domain = fem.Cells(grid)
                        quadrature = fem.PicQuadrature(
                            domain,
                            wp.array(data, dtype=pos_type),
                            measures=wp.ones(len(data), dtype=wp.float64),
                            env_indices=wp.array(envs, dtype=int) if env_count > 1 else None,
                            max_dist=0.2 if nearest else 0.0,
                        )
                        args = quadrature.arg_value(device)
                        order = args.cell_particle_indices.numpy()
                        cell_indices = np.repeat(
                            np.arange(grid.cell_count()), np.diff(args.cell_particle_offsets.numpy())
                        )
                        np.testing.assert_array_equal(cell_indices, expected_cells[order])
                        np.testing.assert_allclose(
                            args.cell_particle_coords.numpy(), expected_coords[order], rtol=0, atol=1e-14
                        )


# -- Test class --


class TestFemFp64(unittest.TestCase):
    pass


devices = get_test_devices(mode="basic")

add_function_test(TestFemFp64, "test_fp64_volume_integral", test_fp64_volume_integral, devices=devices)
add_function_test(TestFemFp64, "test_fp64_position_integral", test_fp64_position_integral, devices=devices)
add_function_test(TestFemFp64, "test_fp64_mass_matrix", test_fp64_mass_matrix, devices=devices)
add_function_test(TestFemFp64, "test_fp64_diffusion_matrix", test_fp64_diffusion_matrix, devices=devices)
add_function_test(TestFemFp64, "test_fp64_boundary_integral", test_fp64_boundary_integral, devices=devices)
add_function_test(TestFemFp64, "test_fp64_interpolation", test_fp64_interpolation, devices=devices)
add_function_test(TestFemFp64, "test_fp64_scalar_type_operator", test_fp64_scalar_type_operator, devices=devices)
add_function_test(TestFemFp64, "test_fp64_nanogrid_volume", test_fp64_nanogrid_volume, devices=devices)
add_function_test(TestFemFp64, "test_fp64_lookup", test_fp64_lookup, devices=devices)

add_function_test(TestFemFp64, "test_fp64_grid_particle_lookup", test_fp64_grid_particle_lookup, devices=devices)

if __name__ == "__main__":
    unittest.main()
