# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import warp as wp
from warp.tests.test_base import *


from warp.fem import Field, Sample, Domain, Coords
from warp.fem import Grid2D, Grid3D, Trimesh2D, Tetmesh
from warp.fem import make_polynomial_space, SymmetricTensorMapper, SkewSymmetricTensorMapper
from warp.fem import make_test
from warp.fem import Cells, BoundarySides
from warp.fem import integrate
from warp.fem import integrand, div, grad, curl, D, normal
from warp.fem import RegularQuadrature, Polynomial
from warp.fem.geometry.closest_point import project_on_tri_at_origin, project_on_tet_at_origin
from warp.fem.space import shape
from warp.fem.cache import dynamic_kernel
from warp.fem.utils import grid_to_tets, grid_to_tris

wp.init()


@integrand
def linear_form(s: Sample, u: Field):
    return u(s)


def test_integrate_gradient(test_case, device):
    with wp.ScopedDevice(device):
        # Grid geometry
        geo = Grid2D(res=wp.vec2i(5))

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


@integrand
def vector_divergence_form(s: Sample, u: Field, q: Field):
    return div(u, s) * q(s)


@integrand
def vector_grad_form(s: Sample, u: Field, q: Field):
    return wp.dot(u(s), grad(q, s))


@integrand
def vector_boundary_form(domain: Domain, s: Sample, u: Field, q: Field):
    return wp.dot(u(s) * q(s), normal(domain, s))


def test_vector_divergence_theorem(test_case, device):
    with wp.ScopedDevice(device):
        # Grid geometry
        geo = Grid2D(res=wp.vec2i(5))

        # Domain and function spaces
        interior = Cells(geometry=geo)
        boundary = BoundarySides(geometry=geo)

        vector_space = make_polynomial_space(geo, degree=2, dtype=wp.vec2)
        scalar_space = make_polynomial_space(geo, degree=1, dtype=float)

        u = vector_space.make_field()
        u.dof_values = np.random.rand(u.dof_values.shape[0], 2)

        # Divergence theorem
        constant_one = scalar_space.make_field()
        constant_one.dof_values.fill_(1.0)

        interior_quadrature = RegularQuadrature(domain=interior, order=vector_space.degree)
        boundary_quadrature = RegularQuadrature(domain=boundary, order=vector_space.degree)
        div_int = integrate(vector_divergence_form, quadrature=interior_quadrature, fields={"u": u, "q": constant_one})
        boundary_int = integrate(
            vector_boundary_form, quadrature=boundary_quadrature, fields={"u": u.trace(), "q": constant_one.trace()}
        )

        test_case.assertAlmostEqual(div_int, boundary_int, places=5)

        # Integration by parts
        q = scalar_space.make_field()
        q.dof_values = np.random.rand(q.dof_values.shape[0])

        interior_quadrature = RegularQuadrature(domain=interior, order=vector_space.degree + scalar_space.degree)
        boundary_quadrature = RegularQuadrature(domain=boundary, order=vector_space.degree + scalar_space.degree)
        div_int = integrate(vector_divergence_form, quadrature=interior_quadrature, fields={"u": u, "q": q})
        grad_int = integrate(vector_grad_form, quadrature=interior_quadrature, fields={"u": u, "q": q})
        boundary_int = integrate(
            vector_boundary_form, quadrature=boundary_quadrature, fields={"u": u.trace(), "q": q.trace()}
        )

        test_case.assertAlmostEqual(div_int + grad_int, boundary_int, places=5)


@integrand
def tensor_divergence_form(s: Sample, tau: Field, v: Field):
    return wp.dot(div(tau, s), v(s))


@integrand
def tensor_grad_form(s: Sample, tau: Field, v: Field):
    return wp.ddot(wp.transpose(tau(s)), grad(v, s))


@integrand
def tensor_boundary_form(domain: Domain, s: Sample, tau: Field, v: Field):
    return wp.dot(tau(s) * v(s), normal(domain, s))


def test_tensor_divergence_theorem(test_case, device):
    with wp.ScopedDevice(device):
        # Grid geometry
        geo = Grid2D(res=wp.vec2i(5))

        # Domain and function spaces
        interior = Cells(geometry=geo)
        boundary = BoundarySides(geometry=geo)

        tensor_space = make_polynomial_space(geo, degree=2, dtype=wp.mat22)
        vector_space = make_polynomial_space(geo, degree=1, dtype=wp.vec2)

        tau = tensor_space.make_field()
        tau.dof_values = np.random.rand(tau.dof_values.shape[0], 2, 2)

        # Divergence theorem
        constant_vec = vector_space.make_field()
        constant_vec.dof_values.fill_(wp.vec2(0.5, 2.0))

        interior_quadrature = RegularQuadrature(domain=interior, order=tensor_space.degree)
        boundary_quadrature = RegularQuadrature(domain=boundary, order=tensor_space.degree)
        div_int = integrate(
            tensor_divergence_form, quadrature=interior_quadrature, fields={"tau": tau, "v": constant_vec}
        )
        boundary_int = integrate(
            tensor_boundary_form, quadrature=boundary_quadrature, fields={"tau": tau.trace(), "v": constant_vec.trace()}
        )

        test_case.assertAlmostEqual(div_int, boundary_int, places=5)

        # Integration by parts
        v = vector_space.make_field()
        v.dof_values = np.random.rand(v.dof_values.shape[0], 2)

        interior_quadrature = RegularQuadrature(domain=interior, order=tensor_space.degree + vector_space.degree)
        boundary_quadrature = RegularQuadrature(domain=boundary, order=tensor_space.degree + vector_space.degree)
        div_int = integrate(tensor_divergence_form, quadrature=interior_quadrature, fields={"tau": tau, "v": v})
        grad_int = integrate(tensor_grad_form, quadrature=interior_quadrature, fields={"tau": tau, "v": v})
        boundary_int = integrate(
            tensor_boundary_form, quadrature=boundary_quadrature, fields={"tau": tau.trace(), "v": v.trace()}
        )

        test_case.assertAlmostEqual(div_int + grad_int, boundary_int, places=5)


@integrand
def grad_decomposition(s: Sample, u: Field, v: Field):
    return wp.length_sq(grad(u, s) * v(s) - D(u, s) * v(s) - wp.cross(curl(u, s), v(s)))


def test_grad_decomposition(test_case, device):
    with wp.ScopedDevice(device):
        # Grid geometry
        geo = Grid3D(res=wp.vec3i(5))

        # Domain and function spaces
        domain = Cells(geometry=geo)
        quadrature = RegularQuadrature(domain=domain, order=4)

        vector_space = make_polynomial_space(geo, degree=2, dtype=wp.vec3)
        u = vector_space.make_field()

        u.dof_values = np.random.rand(u.dof_values.shape[0], 3)

        err = integrate(grad_decomposition, quadrature=quadrature, fields={"u": u, "v": u})
        test_case.assertLess(err, 1.0e-8)


def _gen_trimesh(N):
    x = np.linspace(0.0, 1.0, N + 1)
    y = np.linspace(0.0, 1.0, N + 1)

    positions = np.transpose(np.meshgrid(x, y, indexing="ij")).reshape(-1, 2)

    vidx = grid_to_tris(N, N)

    return wp.array(positions, dtype=wp.vec2), wp.array(vidx, dtype=int)


def _gen_tetmesh(N):
    x = np.linspace(0.0, 1.0, N + 1)
    y = np.linspace(0.0, 1.0, N + 1)
    z = np.linspace(0.0, 1.0, N + 1)

    positions = np.transpose(np.meshgrid(x, y, z, indexing="ij")).reshape(-1, 3)

    vidx = grid_to_tets(N, N, N)

    return wp.array(positions, dtype=wp.vec3), wp.array(vidx, dtype=int)


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

    # Symmetric mapper
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

    # Skew-symmetric mapper
    for dtype in matrix_types:
        mapper = SkewSymmetricTensorMapper(dtype)
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
                test_case.assertAlmostEqual(frob_norm2, 1.0, places=6)
        else:
            dof_val = 1.0

            mat = mapper.dof_to_value(dof_val)
            dof_round_trip = mapper.value_to_dof(mat)

            test_case.assertAlmostEqual(dof_round_trip, dof_val)

            # Check that value is unitary for Frobenius norm 0.5 * |tau:tau|
            frob_norm2 = 0.5 * wp.ddot(mat, mat)
            test_case.assertAlmostEqual(frob_norm2, 1.0, places=6)


def test_shape_function_weight(test_case, shape: shape.ShapeFunction, coord_sampler, CENTER_COORDS):
    NODE_COUNT = shape.NODES_PER_ELEMENT
    weight_fn = shape.make_element_inner_weight()
    node_coords_fn = shape.make_node_coords_in_element()

    # Weight at node should be 1
    @dynamic_kernel(suffix=shape.name)
    def node_unity_test():
        n = wp.tid()
        node_w = weight_fn(node_coords_fn(n), n)
        wp.expect_near(node_w, 1.0, places=5)

    wp.launch(node_unity_test, dim=NODE_COUNT, inputs=[])

    # Sum of node quadrature weights should be one (order 0)
    # Sum of weighted quadrature coords should be element center (order 1)
    node_quadrature_weight_fn = shape.make_node_quadrature_weight()

    @dynamic_kernel(suffix=shape.name)
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

    @dynamic_kernel(suffix=shape.name)
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


def test_shape_function_trace(test_case, shape: shape.ShapeFunction, CENTER_COORDS):
    NODE_COUNT = shape.NODES_PER_ELEMENT
    node_coords_fn = shape.make_node_coords_in_element()

    # Sum of node quadrature weights should be one (order 0)
    # Sum of weighted quadrature coords should be element center (order 1)
    trace_node_quadrature_weight_fn = shape.make_trace_node_quadrature_weight()

    @dynamic_kernel(suffix=shape.name)
    def trace_node_quadrature_unity_test():
        sum_node_qp = float(0.0)
        sum_node_qp_coords = Coords(0.0)

        for n in range(NODE_COUNT):
            coords = node_coords_fn(n)

            if wp.abs(coords[0]) < 1.e-6:
                w = trace_node_quadrature_weight_fn(n)
                sum_node_qp += w
                sum_node_qp_coords += w * node_coords_fn(n)

        wp.expect_near(sum_node_qp, 1.0, 0.0001)
        wp.expect_near(sum_node_qp_coords, CENTER_COORDS, 0.0001)

    wp.launch(trace_node_quadrature_unity_test, dim=1, inputs=[])


def test_shape_function_gradient(test_case, shape: shape.ShapeFunction, coord_sampler, coord_delta_sampler):
    weight_fn = shape.make_element_inner_weight()
    weight_gradient_fn = shape.make_element_inner_weight_gradient()

    @dynamic_kernel(suffix=shape.name)
    def finite_difference_test():
        i, n = wp.tid()
        rng_state = wp.rand_init(1234, i)

        coords = coord_sampler(rng_state)

        epsilon = 0.003
        param_delta, coords_delta = coord_delta_sampler(epsilon, rng_state)

        wp = weight_fn(coords + coords_delta, n)
        wm = weight_fn(coords - coords_delta, n)

        gp = weight_gradient_fn(coords + coords_delta, n)
        gm = weight_gradient_fn(coords - coords_delta, n)

        # 2nd-order finite-difference test
        # See Schroeder 2019, Practical course on computing derivatives in code
        delta_ref = wp - wm
        delta_est = wp.dot(gp + gm, param_delta)

        # wp.printf("%d %f %f \n", n, delta_ref, delta_est)
        wp.expect_near(delta_ref, delta_est, 0.0001)

    n_samples = 100
    wp.launch(finite_difference_test, dim=(n_samples, shape.NODES_PER_ELEMENT), inputs=[])


def test_square_shape_functions(test_case, device):
    SQUARE_CENTER_COORDS = wp.constant(Coords(0.5, 0.5, 0.0))
    SQUARE_SIDE_CENTER_COORDS = wp.constant(Coords(0.0, 0.5, 0.0))

    @wp.func
    def square_coord_sampler(state: wp.uint32):
        return Coords(wp.randf(state), wp.randf(state), 0.0)

    @wp.func
    def square_coord_delta_sampler(epsilon: float, state: wp.uint32):
        param_delta = wp.normalize(wp.vec2(wp.randf(state), wp.randf(state))) * epsilon
        return param_delta, Coords(param_delta[0], param_delta[1], 0.0)

    Q_1 = shape.SquareBipolynomialShapeFunctions(degree=1, family=Polynomial.LOBATTO_GAUSS_LEGENDRE)
    Q_2 = shape.SquareBipolynomialShapeFunctions(degree=2, family=Polynomial.LOBATTO_GAUSS_LEGENDRE)
    Q_3 = shape.SquareBipolynomialShapeFunctions(degree=3, family=Polynomial.LOBATTO_GAUSS_LEGENDRE)

    test_shape_function_weight(test_case, Q_1, square_coord_sampler, SQUARE_CENTER_COORDS)
    test_shape_function_weight(test_case, Q_2, square_coord_sampler, SQUARE_CENTER_COORDS)
    test_shape_function_weight(test_case, Q_3, square_coord_sampler, SQUARE_CENTER_COORDS)
    test_shape_function_trace(test_case, Q_1, SQUARE_SIDE_CENTER_COORDS)
    test_shape_function_trace(test_case, Q_2, SQUARE_SIDE_CENTER_COORDS)
    test_shape_function_trace(test_case, Q_3, SQUARE_SIDE_CENTER_COORDS)
    test_shape_function_gradient(test_case, Q_1, square_coord_sampler, square_coord_delta_sampler)
    test_shape_function_gradient(test_case, Q_2, square_coord_sampler, square_coord_delta_sampler)
    test_shape_function_gradient(test_case, Q_3, square_coord_sampler, square_coord_delta_sampler)

    S_2 = shape.SquareSerendipityShapeFunctions(degree=2, family=Polynomial.LOBATTO_GAUSS_LEGENDRE)
    S_3 = shape.SquareSerendipityShapeFunctions(degree=3, family=Polynomial.LOBATTO_GAUSS_LEGENDRE)

    test_shape_function_weight(test_case, S_2, square_coord_sampler, SQUARE_CENTER_COORDS)
    test_shape_function_weight(test_case, S_3, square_coord_sampler, SQUARE_CENTER_COORDS)
    test_shape_function_trace(test_case, S_2, SQUARE_SIDE_CENTER_COORDS)
    test_shape_function_trace(test_case, S_3, SQUARE_SIDE_CENTER_COORDS)
    test_shape_function_gradient(test_case, S_2, square_coord_sampler, square_coord_delta_sampler)
    test_shape_function_gradient(test_case, S_3, square_coord_sampler, square_coord_delta_sampler)

    P_c1 = shape.SquareNonConformingPolynomialShapeFunctions(degree=1)
    P_c2 = shape.SquareNonConformingPolynomialShapeFunctions(degree=2)
    P_c3 = shape.SquareNonConformingPolynomialShapeFunctions(degree=3)

    test_shape_function_weight(test_case, P_c1, square_coord_sampler, SQUARE_CENTER_COORDS)
    test_shape_function_weight(test_case, P_c2, square_coord_sampler, SQUARE_CENTER_COORDS)
    test_shape_function_weight(test_case, P_c3, square_coord_sampler, SQUARE_CENTER_COORDS)
    test_shape_function_gradient(test_case, P_c1, square_coord_sampler, square_coord_delta_sampler)
    test_shape_function_gradient(test_case, P_c2, square_coord_sampler, square_coord_delta_sampler)
    test_shape_function_gradient(test_case, P_c3, square_coord_sampler, square_coord_delta_sampler)

    wp.synchronize()


def test_cube_shape_functions(test_case, device):
    CUBE_CENTER_COORDS = wp.constant(Coords(0.5, 0.5, 0.5))
    CUBE_SIDE_CENTER_COORDS = wp.constant(Coords(0.0, 0.5, 0.5))

    @wp.func
    def cube_coord_sampler(state: wp.uint32):
        return Coords(wp.randf(state), wp.randf(state), wp.randf(state))

    @wp.func
    def cube_coord_delta_sampler(epsilon: float, state: wp.uint32):
        param_delta = wp.normalize(wp.vec3(wp.randf(state), wp.randf(state), wp.randf(state))) * epsilon
        return param_delta, param_delta

    Q_1 = shape.CubeTripolynomialShapeFunctions(degree=1, family=Polynomial.LOBATTO_GAUSS_LEGENDRE)
    Q_2 = shape.CubeTripolynomialShapeFunctions(degree=2, family=Polynomial.LOBATTO_GAUSS_LEGENDRE)
    Q_3 = shape.CubeTripolynomialShapeFunctions(degree=3, family=Polynomial.LOBATTO_GAUSS_LEGENDRE)

    test_shape_function_weight(test_case, Q_1, cube_coord_sampler, CUBE_CENTER_COORDS)
    test_shape_function_weight(test_case, Q_2, cube_coord_sampler, CUBE_CENTER_COORDS)
    test_shape_function_weight(test_case, Q_3, cube_coord_sampler, CUBE_CENTER_COORDS)
    test_shape_function_trace(test_case, Q_1, CUBE_SIDE_CENTER_COORDS)
    test_shape_function_trace(test_case, Q_2, CUBE_SIDE_CENTER_COORDS)
    test_shape_function_trace(test_case, Q_3, CUBE_SIDE_CENTER_COORDS)
    test_shape_function_gradient(test_case, Q_1, cube_coord_sampler, cube_coord_delta_sampler)
    test_shape_function_gradient(test_case, Q_2, cube_coord_sampler, cube_coord_delta_sampler)
    test_shape_function_gradient(test_case, Q_3, cube_coord_sampler, cube_coord_delta_sampler)

    S_2 = shape.CubeSerendipityShapeFunctions(degree=2, family=Polynomial.LOBATTO_GAUSS_LEGENDRE)
    S_3 = shape.CubeSerendipityShapeFunctions(degree=3, family=Polynomial.LOBATTO_GAUSS_LEGENDRE)

    test_shape_function_weight(test_case, S_2, cube_coord_sampler, CUBE_CENTER_COORDS)
    test_shape_function_weight(test_case, S_3, cube_coord_sampler, CUBE_CENTER_COORDS)
    test_shape_function_trace(test_case, S_2, CUBE_SIDE_CENTER_COORDS)
    test_shape_function_trace(test_case, S_3, CUBE_SIDE_CENTER_COORDS)
    test_shape_function_gradient(test_case, S_2, cube_coord_sampler, cube_coord_delta_sampler)
    test_shape_function_gradient(test_case, S_3, cube_coord_sampler, cube_coord_delta_sampler)

    P_c1 = shape.CubeNonConformingPolynomialShapeFunctions(degree=1)
    P_c2 = shape.CubeNonConformingPolynomialShapeFunctions(degree=2)
    P_c3 = shape.CubeNonConformingPolynomialShapeFunctions(degree=3)

    test_shape_function_weight(test_case, P_c1, cube_coord_sampler, CUBE_CENTER_COORDS)
    test_shape_function_weight(test_case, P_c2, cube_coord_sampler, CUBE_CENTER_COORDS)
    test_shape_function_weight(test_case, P_c3, cube_coord_sampler, CUBE_CENTER_COORDS)
    test_shape_function_gradient(test_case, P_c1, cube_coord_sampler, cube_coord_delta_sampler)
    test_shape_function_gradient(test_case, P_c2, cube_coord_sampler, cube_coord_delta_sampler)
    test_shape_function_gradient(test_case, P_c3, cube_coord_sampler, cube_coord_delta_sampler)

    wp.synchronize()


def test_tri_shape_functions(test_case, device):
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

    test_shape_function_weight(test_case, P_1, tri_coord_sampler, TRI_CENTER_COORDS)
    test_shape_function_weight(test_case, P_2, tri_coord_sampler, TRI_CENTER_COORDS)
    test_shape_function_weight(test_case, P_3, tri_coord_sampler, TRI_CENTER_COORDS)
    test_shape_function_trace(test_case, P_1, TRI_SIDE_CENTER_COORDS)
    test_shape_function_trace(test_case, P_2, TRI_SIDE_CENTER_COORDS)
    test_shape_function_trace(test_case, P_3, TRI_SIDE_CENTER_COORDS)
    test_shape_function_gradient(test_case, P_1, tri_coord_sampler, tri_coord_delta_sampler)
    test_shape_function_gradient(test_case, P_2, tri_coord_sampler, tri_coord_delta_sampler)
    test_shape_function_gradient(test_case, P_3, tri_coord_sampler, tri_coord_delta_sampler)

    P_1d = shape.Triangle2DNonConformingPolynomialShapeFunctions(degree=1)
    P_2d = shape.Triangle2DNonConformingPolynomialShapeFunctions(degree=2)
    P_3d = shape.Triangle2DNonConformingPolynomialShapeFunctions(degree=3)

    test_shape_function_weight(test_case, P_1d, tri_coord_sampler, TRI_CENTER_COORDS)
    test_shape_function_weight(test_case, P_2d, tri_coord_sampler, TRI_CENTER_COORDS)
    test_shape_function_weight(test_case, P_3d, tri_coord_sampler, TRI_CENTER_COORDS)
    test_shape_function_gradient(test_case, P_1d, tri_coord_sampler, tri_coord_delta_sampler)
    # test_shape_function_gradient(test_case, P_2d, tri_coord_sampler, tri_coord_delta_sampler)
    # test_shape_function_gradient(test_case, P_3d, tri_coord_sampler, tri_coord_delta_sampler)

    wp.synchronize()


def test_tet_shape_functions(test_case, device):
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

    test_shape_function_weight(test_case, P_1, tet_coord_sampler, TET_CENTER_COORDS)
    test_shape_function_weight(test_case, P_2, tet_coord_sampler, TET_CENTER_COORDS)
    test_shape_function_weight(test_case, P_3, tet_coord_sampler, TET_CENTER_COORDS)
    test_shape_function_trace(test_case, P_1, TET_SIDE_CENTER_COORDS)
    test_shape_function_trace(test_case, P_2, TET_SIDE_CENTER_COORDS)
    test_shape_function_trace(test_case, P_3, TET_SIDE_CENTER_COORDS)
    test_shape_function_gradient(test_case, P_1, tet_coord_sampler, tet_coord_delta_sampler)
    test_shape_function_gradient(test_case, P_2, tet_coord_sampler, tet_coord_delta_sampler)
    test_shape_function_gradient(test_case, P_3, tet_coord_sampler, tet_coord_delta_sampler)

    P_1d = shape.TetrahedronNonConformingPolynomialShapeFunctions(degree=1)
    P_2d = shape.TetrahedronNonConformingPolynomialShapeFunctions(degree=2)
    P_3d = shape.TetrahedronNonConformingPolynomialShapeFunctions(degree=3)

    test_shape_function_weight(test_case, P_1d, tet_coord_sampler, TET_CENTER_COORDS)
    test_shape_function_weight(test_case, P_2d, tet_coord_sampler, TET_CENTER_COORDS)
    test_shape_function_weight(test_case, P_3d, tet_coord_sampler, TET_CENTER_COORDS)
    test_shape_function_gradient(test_case, P_1d, tet_coord_sampler, tet_coord_delta_sampler)
    test_shape_function_gradient(test_case, P_2d, tet_coord_sampler, tet_coord_delta_sampler)
    test_shape_function_gradient(test_case, P_3d, tet_coord_sampler, tet_coord_delta_sampler)

    wp.synchronize()


def register(parent):
    devices = get_test_devices()

    class TestFem(parent):
        pass

    add_function_test(TestFem, "test_regular_quadrature", test_regular_quadrature)
    add_function_test(TestFem, "test_closest_point_queries", test_closest_point_queries)
    add_function_test(TestFem, "test_grad_decomposition", test_grad_decomposition, devices=devices)
    add_function_test(TestFem, "test_integrate_gradient", test_integrate_gradient, devices=devices)
    add_function_test(TestFem, "test_vector_divergence_theorem", test_vector_divergence_theorem, devices=devices)
    add_function_test(TestFem, "test_tensor_divergence_theorem", test_tensor_divergence_theorem, devices=devices)
    add_function_test(TestFem, "test_triangle_mesh", test_triangle_mesh, devices=devices)
    add_function_test(TestFem, "test_tet_mesh", test_tet_mesh, devices=devices)
    add_function_test(TestFem, "test_dof_mapper", test_dof_mapper)
    add_function_test(TestFem, "test_square_shape_functions", test_square_shape_functions)
    add_function_test(TestFem, "test_cube_shape_functions", test_cube_shape_functions)
    add_function_test(TestFem, "test_tri_shape_functions", test_tri_shape_functions)
    add_function_test(TestFem, "test_tet_shape_functions", test_tet_shape_functions)

    return TestFem


if __name__ == "__main__":
    c = register(unittest.TestCase)
    unittest.main(verbosity=2)
