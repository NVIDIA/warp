# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from typing import Any

import numpy as np

import warp as wp
import warp.fem as fem
from warp.fem.linalg import spherical_part
from warp.fem.utils import (
    grid_to_hexes,
    grid_to_quads,
    grid_to_tets,
    grid_to_tris,
)
from warp.tests.fem.utils import grad_field
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


@wp.func
def aniso_bicubic_fn(x: wp.vec2, scale: wp.vec2):
    return wp.pow(x[0] * scale[0], 3.0) * wp.pow(x[1] * scale[1], 3.0)


@wp.func
def aniso_bicubic_grad(x: wp.vec2, scale: wp.vec2):
    return wp.vec2(
        3.0 * scale[0] * wp.pow(x[0] * scale[0], 2.0) * wp.pow(x[1] * scale[1], 3.0),
        3.0 * scale[1] * wp.pow(x[0] * scale[0], 3.0) * wp.pow(x[1] * scale[1], 2.0),
    )


@wp.func
def _expect_near(a: Any, b: Any, tol: float):
    wp.expect_near(a, b, tol)


@wp.func
def _expect_near(a: wp.vec2, b: wp.vec2, tol: float):
    for k in range(2):
        wp.expect_near(a[k], b[k], tol)


@fem.integrand
def _expect_pure_curl(s: fem.Sample, field: fem.Field):
    sym_grad = fem.D(field, s)
    wp.expect_near(wp.ddot(sym_grad, sym_grad), 0.0)
    return 0.0


@fem.integrand
def _expect_pure_spherical(s: fem.Sample, field: fem.Field):
    grad = fem.grad(field, s)
    deviatoric_part = grad - spherical_part(grad)
    wp.expect_near(wp.ddot(deviatoric_part, deviatoric_part), 0.0)
    return 0.0


@fem.integrand
def _expect_normal_continuity(s: fem.Sample, domain: fem.Domain, field: fem.Field):
    nor = fem.normal(domain, s)
    wp.expect_near(wp.dot(fem.inner(field, s), nor), wp.dot(fem.outer(field, s), nor), 0.0001)
    return 0.0


@fem.integrand
def _expect_tangential_continuity(s: fem.Sample, domain: fem.Domain, field: fem.Field):
    nor = fem.normal(domain, s)
    in_s = fem.inner(field, s)
    out_s = fem.outer(field, s)
    in_t = in_s - wp.dot(in_s, nor) * nor
    out_t = out_s - wp.dot(out_s, nor) * nor

    _expect_near(in_t, out_t, 0.0001)
    return 0.0


@fem.integrand
def _boundary_cells_field_lookup_integral(
    s: fem.Sample,
    domain: fem.Domain,
    U: fem.Field,
    bounds_lo: wp.vec2,
    bounds_hi: wp.vec2,
):
    """On left/right boundary sides, use fem.cells(U) for correct cell-space eval."""
    pos = domain(s)
    nor = fem.normal(domain, s)
    on_top_bottom = pos[1] <= bounds_lo[1] or pos[1] >= bounds_hi[1]
    on_left_right = pos[0] <= bounds_lo[0] or pos[0] >= bounds_hi[0]
    if on_left_right and not on_top_bottom:
        domain_width = bounds_hi[0] - bounds_lo[0]
        on_left = pos[0] <= bounds_lo[0]
        eps = 1.0e-6 * domain_width
        wrapped_x = wp.where(on_left, bounds_hi[0] - eps, bounds_lo[0] + eps)
        wrapped_pos = wp.vec2(wrapped_x, pos[1])
        cell_domain = fem.cells(domain)
        wrapped_s = fem.lookup(cell_domain, wrapped_pos, s)
        U_cell = fem.cells(U)
        U_cell_bis = fem.cells(U_cell)  # no op
        return U_cell_bis(wrapped_s)[0] * wp.length(nor)
    return 0.0


@fem.integrand
def _y_only_init(s: fem.Sample, domain: fem.Domain):
    """rho = 1 + y, all other components zero."""
    x = domain(s)
    return wp.vec4(1.0 + x[1], 0.0, 0.0, 0.0)


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
    fem.interpolate(uniform.trace(), dest=discrete_field, at=boundary)
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
    fem.interpolate(implicit.trace(), dest=discrete_field, at=boundary)
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
        dest=discrete_field2,
        at=boundary2,
    )
    assert_np_equal(discrete_field2.dof_values.numpy(), np.array([2.0] + [5.0] * 3))


def test_vector_spaces(test, device):
    # Test covariant / contravariant mappings

    with wp.ScopedDevice(device):
        positions, hex_vidx = _gen_quadmesh(3)

        geo = fem.Quadmesh2D(quad_vertex_indices=hex_vidx, positions=positions)

        curl_space = fem.make_polynomial_space(geo, element_basis=fem.ElementBasis.NEDELEC_FIRST_KIND)
        curl_test = fem.make_test(curl_space)

        curl_field = curl_space.make_field()
        curl_field.dof_values = wp.array(np.linspace(0.0, 1.0, curl_space.node_count()), dtype=float)

        fem.interpolate(
            _expect_tangential_continuity,
            at=fem.RegularQuadrature(fem.Sides(geo), order=2),
            fields={"field": curl_field.trace()},
        )

        div_space = fem.make_polynomial_space(geo, element_basis=fem.ElementBasis.RAVIART_THOMAS)
        div_test = fem.make_test(div_space)

        div_field = div_space.make_field()
        div_field.dof_values = wp.array(np.linspace(0.0, 1.0, div_space.node_count()), dtype=float)

        fem.interpolate(
            _expect_normal_continuity,
            at=fem.RegularQuadrature(fem.Sides(geo), order=2),
            fields={"field": div_field.trace()},
        )

    with wp.ScopedDevice(device):
        positions, hex_vidx = _gen_hexmesh(3)

        geo = fem.Hexmesh(hex_vertex_indices=hex_vidx, positions=positions)

        curl_space = fem.make_polynomial_space(geo, element_basis=fem.ElementBasis.NEDELEC_FIRST_KIND)
        curl_test = fem.make_test(curl_space)

        curl_field = curl_space.make_field()
        curl_field.dof_values = wp.array(np.linspace(0.0, 1.0, curl_space.node_count()), dtype=float)

        fem.interpolate(
            _expect_tangential_continuity,
            at=fem.RegularQuadrature(fem.Sides(geo), order=2),
            fields={"field": curl_field.trace()},
        )

        div_space = fem.make_polynomial_space(geo, element_basis=fem.ElementBasis.RAVIART_THOMAS)
        div_test = fem.make_test(div_space)

        div_field = div_space.make_field()
        div_field.dof_values = wp.array(np.linspace(0.0, 1.0, div_space.node_count()), dtype=float)

        fem.interpolate(
            _expect_normal_continuity,
            at=fem.RegularQuadrature(fem.Sides(geo), order=2),
            fields={"field": div_field.trace()},
        )

    with wp.ScopedDevice(device):
        positions, tri_vidx = _gen_trimesh(3, 5)

        geo = fem.Trimesh2D(tri_vertex_indices=tri_vidx, positions=positions)

        curl_space = fem.make_polynomial_space(geo, element_basis=fem.ElementBasis.NEDELEC_FIRST_KIND)
        curl_test = fem.make_test(curl_space)

        fem.integrate(_expect_pure_curl, fields={"field": curl_test}, assembly="generic")

        curl_field = curl_space.make_field()
        curl_field.dof_values.fill_(1.0)
        fem.interpolate(
            _expect_pure_curl, at=fem.RegularQuadrature(fem.Cells(geo), order=2), fields={"field": curl_field}
        )

        fem.interpolate(
            _expect_tangential_continuity,
            at=fem.RegularQuadrature(fem.Sides(geo), order=2),
            fields={"field": curl_field.trace()},
        )

        div_space = fem.make_polynomial_space(geo, element_basis=fem.ElementBasis.RAVIART_THOMAS)
        div_test = fem.make_test(div_space)

        fem.integrate(_expect_pure_spherical, fields={"field": div_test}, assembly="generic")

        div_field = div_space.make_field()
        div_field.dof_values.fill_(1.0)
        fem.interpolate(
            _expect_pure_spherical,
            at=fem.RegularQuadrature(fem.Cells(geo), order=2),
            fields={"field": div_field},
        )

        fem.interpolate(
            _expect_normal_continuity,
            at=fem.RegularQuadrature(fem.Sides(geo), order=2),
            fields={"field": div_field.trace()},
        )

    with wp.ScopedDevice(device):
        positions, tet_vidx = _gen_tetmesh(3, 5, 7)

        geo = fem.Tetmesh(tet_vertex_indices=tet_vidx, positions=positions)

        curl_space = fem.make_polynomial_space(geo, element_basis=fem.ElementBasis.NEDELEC_FIRST_KIND)
        curl_test = fem.make_test(curl_space)

        fem.integrate(_expect_pure_curl, fields={"field": curl_test}, assembly="generic")

        curl_field = curl_space.make_field()
        curl_field.dof_values.fill_(1.0)
        fem.interpolate(
            _expect_pure_curl, at=fem.RegularQuadrature(fem.Cells(geo), order=2), fields={"field": curl_field}
        )

        fem.interpolate(
            _expect_tangential_continuity,
            at=fem.RegularQuadrature(fem.Sides(geo), order=1),
            fields={"field": curl_field.trace()},
        )

        div_space = fem.make_polynomial_space(geo, element_basis=fem.ElementBasis.RAVIART_THOMAS)
        div_test = fem.make_test(div_space)

        fem.integrate(_expect_pure_spherical, fields={"field": div_test}, assembly="generic")

        div_field = div_space.make_field()
        div_field.dof_values.fill_(1.0)
        fem.interpolate(
            _expect_pure_spherical,
            at=fem.RegularQuadrature(fem.Cells(geo), order=2),
            fields={"field": div_field},
        )

        fem.interpolate(
            _expect_normal_continuity,
            at=fem.RegularQuadrature(fem.Sides(geo), order=0),
            fields={"field": div_field.trace()},
        )


def test_traced_cells_field_lookup_is_correct(test, device):
    """U(s) on boundary sides with traced field gives correct results."""

    def _setup_grid(device, res=10):
        aspect = 2.0
        domain_size = 1.0
        domain_width = aspect * domain_size
        res_x = int(aspect * res)

        geo = fem.Grid2D(
            res=wp.vec2i(res_x, res),
            bounds_lo=wp.vec2(0.0),
            bounds_hi=wp.vec2(domain_width, domain_size),
        )
        sides = fem.Sides(geo)
        basis = fem.make_polynomial_basis_space(geo, degree=1, discontinuous=True)
        space4 = fem.make_collocated_function_space(basis, dtype=wp.vec4)

        field = space4.make_field()
        fem.interpolate(_y_only_init, dest=field)

        bounds_lo = wp.vec2(0.0)
        bounds_hi = wp.vec2(domain_width, domain_size)

        return field, sides, bounds_lo, bounds_hi

    with wp.ScopedDevice(device):
        field, sides, bounds_lo, bounds_hi = _setup_grid(device)

        vals = {"bounds_lo": bounds_lo, "bounds_hi": bounds_hi}
        inner_integral = fem.integrate(
            _boundary_cells_field_lookup_integral,
            domain=sides,
            fields={"U": field.trace()},
            values=vals,
        )
        # 2 boundaries x int_0^1 (1+y) dy = 2 x 1.5 = 3.0
        test.assertAlmostEqual(
            inner_integral,
            3.0,
            places=4,
            msg=f"Traced U(s) on boundary sides wrong on {device}",
        )


devices = get_test_devices()


class TestFemField(unittest.TestCase):
    pass


add_function_test(TestFemField, "test_vector_spaces", test_vector_spaces, devices=devices)
add_function_test(TestFemField, "test_dof_mapper", test_dof_mapper)
add_function_test(TestFemField, "test_implicit_fields", test_implicit_fields)
add_function_test(
    TestFemField,
    "test_traced_cells_field_lookup_is_correct",
    test_traced_cells_field_lookup_is_correct,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
