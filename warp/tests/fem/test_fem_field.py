# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from typing import Any

import numpy as np

import warp as wp
import warp.fem as fem
import warp.tests.fem.aux_test_function_cache_1 as function_cache_module_1
import warp.tests.fem.aux_test_function_cache_2 as function_cache_module_2
from warp.fem.linalg import spherical_part
from warp.fem.utils import (
    grid_to_hexes,
    grid_to_quads,
    grid_to_tets,
    grid_to_tris,
)
from warp.sparse import bsr_zeros
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
def _implicit_constant_one(x: wp.vec2):
    return 1.0


@wp.func
def _implicit_constant_two(x: wp.vec2):
    return 2.0


def _make_deferred_static_implicit_func(values):
    @wp.func(name="deferred_static_implicit_func", module="unique")
    def implicit_func(x: wp.vec2):
        result = 0.0
        for i in range(wp.static(len(values))):
            result += wp.static(values[i])
        return result

    return implicit_func


@wp.func
def _implicit_constant_gradient(x: wp.vec2):
    return wp.vec2(3.0, 4.0)


@wp.func
def _implicit_vector_constant(x: wp.vec2):
    return wp.vec2(1.0, 2.0)


@wp.func
def _implicit_constant_divergence(x: wp.vec2):
    return 5.0


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


@wp.func
def _curl_conforming_exact_field(x: wp.vec2):
    # Constant plus rotation, spanned by lowest-order Nédélec elements
    return wp.vec2(1.0 - 0.5 * x[1], 2.0 + 0.5 * x[0])


@wp.func
def _curl_conforming_exact_field(x: wp.vec3):
    return wp.vec3(1.0, 2.0, 3.0) + wp.cross(wp.vec3(0.5, -0.25, 0.75), x)


@wp.func
def _div_conforming_exact_field(x: wp.vec2):
    # Constant plus dilation, spanned by lowest-order Raviart-Thomas elements
    return wp.vec2(1.0, 2.0) + 0.7 * x


@wp.func
def _div_conforming_exact_field(x: wp.vec3):
    return wp.vec3(1.0, 2.0, 3.0) + 0.7 * x


@wp.func
def _vector_space_exact_field(x: Any, curl_conforming: int):
    if curl_conforming:
        return _curl_conforming_exact_field(x)
    return _div_conforming_exact_field(x)


@fem.integrand
def _vector_space_exact_value(s: fem.Sample, domain: fem.Domain, curl_conforming: int):
    return _vector_space_exact_field(domain(s), curl_conforming)


@fem.integrand
def _vector_space_cell_error(s: fem.Sample, domain: fem.Domain, u: fem.Field, curl_conforming: int):
    return wp.length_sq(u(s) - _vector_space_exact_field(domain(s), curl_conforming))


@fem.integrand
def _vector_space_boundary_trace_error(s: fem.Sample, domain: fem.Domain, u: fem.Field, curl_conforming: int):
    # Tangential (curl-conforming) or normal (div-conforming) traces only depend on boundary nodes
    err = u(s) - _vector_space_exact_field(domain(s), curl_conforming)
    nor = fem.normal(domain, s)
    err_n = wp.dot(err, nor)
    if curl_conforming:
        return wp.length_sq(err - err_n * nor)
    return err_n * err_n


@fem.integrand
def _vector_space_trial_value(s: fem.Sample, u: fem.Field):
    return u(s)


@wp.func
def _bspline_exact_field(x: wp.vec2):
    # Linear field, reproduced exactly by B-spline interpolation
    return 2.0 + x[0] - 0.5 * x[1]


@wp.func
def _bspline_exact_field(x: wp.vec3):
    return 2.0 + x[0] - 0.5 * x[1] + 0.25 * x[2]


@fem.integrand
def _bspline_exact_value(s: fem.Sample, domain: fem.Domain):
    return _bspline_exact_field(domain(s))


@fem.integrand
def _bspline_cell_error(s: fem.Sample, domain: fem.Domain, u: fem.Field):
    err = u(s) - _bspline_exact_field(domain(s))
    return err * err


@fem.integrand
def _div_field(s: fem.Sample, p: fem.Field):
    return fem.div(p, s)


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
    """Initialize ``rho = 1 + y`` with all other components set to zero."""
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


def test_implicit_field_function_identity(test, device):
    """Verify that implicit fields with different functions do not share evaluators."""
    with wp.ScopedDevice(device):
        geo = fem.Grid2D(res=wp.vec2i(2))
        domain = fem.Cells(geo)
        space = fem.make_polynomial_space(geo)
        dest = fem.make_discrete_field(space)

        # Keep every structural cache input equal so only the value function distinguishes the fields.
        field_one = fem.ImplicitField(domain, func=_implicit_constant_one)
        field_two = fem.ImplicitField(domain, func=_implicit_constant_two)

        fem.interpolate(field_one, dest=dest)
        assert_np_equal(dest.dof_values.numpy(), np.ones(9))

        fem.interpolate(field_two, dest=dest)
        assert_np_equal(dest.dof_values.numpy(), np.full(9, 2.0))


def test_implicit_field_function_module_identity(test, device):
    """Verify that same-named functions from different modules do not share evaluators."""
    with wp.ScopedDevice(device):
        geo = fem.Grid2D(res=wp.vec2i(2))
        domain = fem.Cells(geo)
        dest = fem.make_polynomial_space(geo).make_field()

        # The paired functions share a Warp key and module name but have distinct native identities.
        field_one = fem.ImplicitField(domain, func=function_cache_module_1.implicit_func)
        field_two = fem.ImplicitField(domain, func=function_cache_module_2.implicit_func)

        fem.interpolate(field_one, dest=dest)
        assert_np_equal(dest.dof_values.numpy(), np.ones(9))

        fem.interpolate(field_two, dest=dest)
        assert_np_equal(dest.dof_values.numpy(), np.full(9, 2.0))


def test_implicit_field_deferred_static_identity(test, device):
    """Verify that deferred static values distinguish implicit field evaluators."""
    with wp.ScopedDevice(device):
        geo = fem.Grid2D(res=wp.vec2i(2))
        domain = fem.Cells(geo)
        dest = fem.make_polynomial_space(geo).make_field()

        field_one = fem.ImplicitField(domain, func=_make_deferred_static_implicit_func([1.0, 2.0]))
        field_two = fem.ImplicitField(domain, func=_make_deferred_static_implicit_func([10.0, 20.0]))

        fem.interpolate(field_one, dest=dest)
        assert_np_equal(dest.dof_values.numpy(), np.full(9, 3.0))

        fem.interpolate(field_two, dest=dest)
        assert_np_equal(dest.dof_values.numpy(), np.full(9, 30.0))


def test_implicit_field_gradient_module_identity(test, device):
    """Verify that same-named gradients from different modules do not share evaluators."""
    with wp.ScopedDevice(device):
        geo = fem.Grid2D(res=wp.vec2i(2))
        domain = fem.Cells(geo)
        dest = fem.make_polynomial_space(geo, dtype=wp.vec2).make_field()

        # Keep the value function fixed so only the gradient function changes the cache identity.
        field_one = fem.ImplicitField(
            domain,
            func=_implicit_constant_one,
            grad_func=function_cache_module_1.implicit_grad_func,
            degree=4,
        )
        field_two = fem.ImplicitField(
            domain,
            func=_implicit_constant_one,
            grad_func=function_cache_module_2.implicit_grad_func,
            degree=4,
        )

        fem.interpolate(grad_field, fields={"p": field_one}, dest=dest)
        assert_np_equal(dest.dof_values.numpy(), np.tile((3.0, 4.0), (9, 1)))

        fem.interpolate(grad_field, fields={"p": field_two}, dest=dest)
        assert_np_equal(dest.dof_values.numpy(), np.tile((7.0, 8.0), (9, 1)))


def test_implicit_field_divergence_module_identity(test, device):
    """Verify that same-named divergences from different modules do not share evaluators."""
    with wp.ScopedDevice(device):
        geo = fem.Grid2D(res=wp.vec2i(2))
        domain = fem.Cells(geo)
        dest = fem.make_polynomial_space(geo).make_field()

        # Keep the value function fixed so only the divergence function changes the cache identity.
        field_one = fem.ImplicitField(
            domain,
            func=_implicit_vector_constant,
            div_func=function_cache_module_1.implicit_div_func,
            degree=4,
        )
        field_two = fem.ImplicitField(
            domain,
            func=_implicit_vector_constant,
            div_func=function_cache_module_2.implicit_div_func,
            degree=4,
        )

        fem.interpolate(_div_field, fields={"p": field_one}, dest=dest)
        assert_np_equal(dest.dof_values.numpy(), np.full(9, 5.0))

        fem.interpolate(_div_field, fields={"p": field_two}, dest=dest)
        assert_np_equal(dest.dof_values.numpy(), np.full(9, 6.0))


def test_implicit_field_does_not_share_gradient(test, device):
    """Verify that a gradientless field still rejects gradient evaluation."""
    with wp.ScopedDevice(device):
        geo = fem.Grid2D(res=wp.vec2i(2))
        domain = fem.Cells(geo)
        vec_space = fem.make_polynomial_space(geo, dtype=wp.vec2)
        dest = fem.make_discrete_field(vec_space)

        # Build the gradient-capable field first to expose cache contamination in the later field.
        field_with_gradient = fem.ImplicitField(
            domain,
            func=_implicit_constant_one,
            grad_func=_implicit_constant_gradient,
            degree=1,
        )
        field_without_gradient = fem.ImplicitField(domain, func=_implicit_constant_one, degree=1)

        fem.interpolate(grad_field, fields={"p": field_with_gradient}, dest=dest)
        assert_np_equal(dest.dof_values.numpy(), np.tile((3.0, 4.0), (9, 1)))

        with test.assertRaisesRegex(TypeError, r"Operator `grad` is not defined"):
            fem.interpolate(grad_field, fields={"p": field_without_gradient}, dest=dest)


def test_implicit_field_does_not_drop_gradient(test, device):
    """Verify that a later gradient-capable field still evaluates its gradient."""
    with wp.ScopedDevice(device):
        geo = fem.Grid2D(res=wp.vec2i(2))
        domain = fem.Cells(geo)
        vec_space = fem.make_polynomial_space(geo, dtype=wp.vec2)
        dest = fem.make_discrete_field(vec_space)

        # Prime the cache without a gradient, then evaluate a later field through interpolation.
        fem.ImplicitField(domain, func=_implicit_constant_two, degree=2)
        field_with_gradient = fem.ImplicitField(
            domain,
            func=_implicit_constant_two,
            grad_func=_implicit_constant_gradient,
            degree=2,
        )

        fem.interpolate(grad_field, fields={"p": field_with_gradient}, dest=dest)
        assert_np_equal(dest.dof_values.numpy(), np.tile((3.0, 4.0), (9, 1)))


def test_implicit_field_does_not_share_divergence(test, device):
    """Verify that a divergence-free field still rejects divergence evaluation."""
    with wp.ScopedDevice(device):
        geo = fem.Grid2D(res=wp.vec2i(2))
        domain = fem.Cells(geo)
        space = fem.make_polynomial_space(geo)
        dest = fem.make_discrete_field(space)

        # Build the divergence-capable field first to expose cache contamination in the later field.
        field_with_divergence = fem.ImplicitField(
            domain,
            func=_implicit_vector_constant,
            div_func=_implicit_constant_divergence,
            degree=3,
        )
        field_without_divergence = fem.ImplicitField(domain, func=_implicit_vector_constant, degree=3)

        fem.interpolate(_div_field, fields={"p": field_with_divergence}, dest=dest)
        assert_np_equal(dest.dof_values.numpy(), np.full(9, 5.0))

        with test.assertRaisesRegex(TypeError, r"Operator `div` is not defined"):
            fem.interpolate(_div_field, fields={"p": field_without_divergence}, dest=dest)


def test_implicit_field_does_not_drop_divergence(test, device):
    """Verify that a later divergence-capable field still evaluates its divergence."""
    with wp.ScopedDevice(device):
        geo = fem.Grid2D(res=wp.vec2i(2))
        domain = fem.Cells(geo)
        space = fem.make_polynomial_space(geo)
        dest = fem.make_discrete_field(space)

        # Prime the cache without divergence, then evaluate a later field through interpolation.
        fem.ImplicitField(domain, func=_implicit_vector_constant, degree=5)
        field_with_divergence = fem.ImplicitField(
            domain,
            func=_implicit_vector_constant,
            div_func=_implicit_constant_divergence,
            degree=5,
        )

        fem.interpolate(_div_field, fields={"p": field_with_divergence}, dest=dest)
        assert_np_equal(dest.dof_values.numpy(), np.full(9, 5.0))


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
    """Test covariant and contravariant vector-space mappings."""

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


def test_vector_space_interpolation(test, device):
    """Test that interpolation reproduces fields spanned by H(curl) and H(div) spaces."""

    # Affine maps shear the reference grids, so that edges and sides are not axis-aligned
    shear_2d = np.array([[1.0, 0.6], [-0.3, 0.9]])
    shear_3d = np.array([[1.0, 0.6, 0.2], [-0.3, 0.9, 0.1], [0.1, -0.2, 1.1]])

    def _shear(positions, shear):
        return wp.array(positions.numpy() @ shear.T, dtype=positions.dtype)

    with wp.ScopedDevice(device):
        positions, vidx = _gen_trimesh(3, 4)
        tri_geo = fem.Trimesh2D(tri_vertex_indices=vidx, positions=_shear(positions, shear_2d))

        positions, vidx = _gen_quadmesh(3)
        quad_geo = fem.Quadmesh2D(quad_vertex_indices=vidx, positions=_shear(positions, shear_2d))

        positions, vidx = _gen_tetmesh(2, 3, 2)
        tet_geo = fem.Tetmesh(tet_vertex_indices=vidx, positions=_shear(positions, shear_3d))

        positions, vidx = _gen_hexmesh(2)
        hex_geo = fem.Hexmesh(hex_vertex_indices=vidx, positions=_shear(positions, shear_3d))

        for geo in (tri_geo, quad_geo, tet_geo, hex_geo):
            vec_dtype = wp.types.vector(length=geo.dimension, dtype=float)

            for element_basis in (fem.ElementBasis.NEDELEC_FIRST_KIND, fem.ElementBasis.RAVIART_THOMAS):
                curl_conforming = int(element_basis == fem.ElementBasis.NEDELEC_FIRST_KIND)
                values = {"curl_conforming": curl_conforming}
                msg = f"{geo.name}, {element_basis}"

                space = fem.make_polynomial_space(geo, degree=1, element_basis=element_basis, dtype=vec_dtype)

                # Interpolation at all nodes
                field = space.make_field()
                fem.interpolate(_vector_space_exact_value, dest=field, values=values)

                cell_error = fem.integrate(
                    _vector_space_cell_error,
                    quadrature=fem.RegularQuadrature(fem.Cells(geo), order=2),
                    fields={"u": field},
                    values=values,
                )
                test.assertLess(cell_error, 1.0e-8, msg=msg)

                # Interpolation at boundary nodes only, through the trace space
                boundary = fem.BoundarySides(geo)
                boundary_field = space.make_field()
                fem.interpolate(_vector_space_exact_value, dest=boundary_field, at=boundary, values=values)

                boundary_error = fem.integrate(
                    _vector_space_boundary_trace_error,
                    quadrature=fem.RegularQuadrature(boundary, order=2),
                    fields={"u": boundary_field.trace()},
                    values=values,
                )
                test.assertLess(boundary_error, 1.0e-8, msg=msg)

                # Jacobian of the interpolation from a Lagrange field, which also spans the exact field
                lagrange_space = fem.make_polynomial_space(geo, degree=1, dtype=vec_dtype)
                lagrange_field = lagrange_space.make_field()
                fem.interpolate(_vector_space_exact_value, dest=lagrange_field, values=values)

                jacobian = bsr_zeros(
                    rows_of_blocks=space.node_count(),
                    cols_of_blocks=lagrange_space.node_count(),
                    block_type=wp.types.matrix(shape=(1, geo.dimension), dtype=float),
                )
                fem.interpolate(
                    _vector_space_trial_value,
                    dest=jacobian,
                    dest_space=space,
                    at=fem.make_space_restriction(space_topology=space.topology, domain=fem.Cells(geo)),
                    fields={"u": fem.make_trial(lagrange_space)},
                )
                assert_np_equal(
                    (jacobian @ lagrange_field.dof_values).numpy().flatten(),
                    field.dof_values.numpy(),
                    tol=1.0e-5,
                )


def test_bspline_space_interpolation(test, device):
    """Test that interpolation into B-spline spaces reproduces linear fields."""

    with wp.ScopedDevice(device):
        for geo in (fem.Grid2D(res=wp.vec2i(4)), fem.Grid3D(res=wp.vec3i(4))):
            for degree in (1, 2, 3):
                space = fem.make_polynomial_space(geo, degree=degree, element_basis=fem.ElementBasis.BSPLINE)
                field = space.make_field()
                fem.interpolate(_bspline_exact_value, dest=field)

                cell_error = fem.integrate(
                    _bspline_cell_error,
                    quadrature=fem.RegularQuadrature(fem.Cells(geo), order=2 * degree),
                    fields={"u": field},
                )
                test.assertLess(cell_error, 1.0e-8, msg=f"{geo.name}, degree {degree}")


def test_traced_cells_field_lookup_is_correct(test, device):
    """Verify that U(s) on boundary sides with traced field gives correct results."""

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
    def test_uniform_field_rejects_value_type_change(self):
        domain = fem.Cells(fem.Grid2D(res=wp.vec2i(1, 1)))
        field = fem.UniformField(domain, 1.0)

        with self.assertRaisesRegex(TypeError, "UniformField value type must remain float32, got vec2f"):
            field.value = wp.vec2(1.0)


add_function_test(TestFemField, "test_vector_spaces", test_vector_spaces, devices=devices)
add_function_test(TestFemField, "test_vector_space_interpolation", test_vector_space_interpolation, devices=devices)
add_function_test(TestFemField, "test_bspline_space_interpolation", test_bspline_space_interpolation, devices=devices)
add_function_test(TestFemField, "test_dof_mapper", test_dof_mapper)
add_function_test(
    TestFemField, "test_implicit_field_function_identity", test_implicit_field_function_identity, devices=devices
)
add_function_test(
    TestFemField,
    "test_implicit_field_function_module_identity",
    test_implicit_field_function_module_identity,
    devices=devices,
)
add_function_test(
    TestFemField,
    "test_implicit_field_deferred_static_identity",
    test_implicit_field_deferred_static_identity,
    devices=devices,
)
add_function_test(
    TestFemField,
    "test_implicit_field_gradient_module_identity",
    test_implicit_field_gradient_module_identity,
    devices=devices,
)
add_function_test(
    TestFemField,
    "test_implicit_field_divergence_module_identity",
    test_implicit_field_divergence_module_identity,
    devices=devices,
)
add_function_test(
    TestFemField,
    "test_implicit_field_does_not_share_gradient",
    test_implicit_field_does_not_share_gradient,
    devices=devices,
)
add_function_test(
    TestFemField,
    "test_implicit_field_does_not_drop_gradient",
    test_implicit_field_does_not_drop_gradient,
    devices=devices,
)
add_function_test(
    TestFemField,
    "test_implicit_field_does_not_share_divergence",
    test_implicit_field_does_not_share_divergence,
    devices=devices,
)
add_function_test(
    TestFemField,
    "test_implicit_field_does_not_drop_divergence",
    test_implicit_field_does_not_drop_divergence,
    devices=devices,
)
add_function_test(TestFemField, "test_implicit_fields", test_implicit_fields)
add_function_test(
    TestFemField,
    "test_traced_cells_field_lookup_is_correct",
    test_traced_cells_field_lookup_is_correct,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
