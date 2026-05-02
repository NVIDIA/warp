# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from typing import Any

import warp as wp
import warp.fem as fem
from warp.fem import Coords, ShapeFunction
from warp.fem.cache import dynamic_kernel
from warp.fem.linalg import spherical_part, symmetric_part
from warp.fem.space.shape import (
    CubeBSplineShapeFunctions,
    CubeNedelecFirstKindShapeFunctions,
    CubeNonConformingPolynomialShapeFunctions,
    CubeRaviartThomasShapeFunctions,
    CubeSerendipityShapeFunctions,
    CubeTripolynomialShapeFunctions,
    SquareBipolynomialShapeFunctions,
    SquareBSplineShapeFunctions,
    SquareNedelecFirstKindShapeFunctions,
    SquareNonConformingPolynomialShapeFunctions,
    SquareRaviartThomasShapeFunctions,
    SquareSerendipityShapeFunctions,
    TetrahedronNedelecFirstKindShapeFunctions,
    TetrahedronNonConformingPolynomialShapeFunctions,
    TetrahedronPolynomialShapeFunctions,
    TetrahedronRaviartThomasShapeFunctions,
    TriangleNedelecFirstKindShapeFunctions,
    TriangleNonConformingPolynomialShapeFunctions,
    TrianglePolynomialShapeFunctions,
    TriangleRaviartThomasShapeFunctions,
)
from warp.tests.unittest_utils import *


@wp.func
def _expect_near(a: Any, b: Any, tol: float):
    wp.expect_near(a, b, tol)


@wp.func
def _expect_near(a: wp.vec2, b: wp.vec2, tol: float):
    for k in range(2):
        wp.expect_near(a[k], b[k], tol)


def test_shape_function_weight(
    test,
    shape: ShapeFunction,
    coord_sampler,
    center_coords,
    node_unity: bool = True,
    node_quadrature_unity: bool = True,
    partition_of_unity: bool = True,
):
    NODE_COUNT = shape.NODES_PER_ELEMENT
    weight_fn = shape.make_element_inner_weight()
    node_coords_fn = shape.make_node_coords_in_element()

    # Weight at node should be 1
    @dynamic_kernel(suffix=shape.name, kernel_options={"enable_backward": False})
    def node_unity_test():
        n = wp.tid()
        node_w = weight_fn(node_coords_fn(n), n)
        wp.expect_near(node_w, 1.0, 1e-5)

    if node_unity:
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
        wp.expect_near(sum_node_qp_coords, center_coords, 0.0001)

    if node_quadrature_unity:
        wp.launch(node_quadrature_unity_test, dim=1, inputs=[])

    @dynamic_kernel(suffix=shape.name, kernel_options={"enable_backward": False})
    def partition_of_unity_test():
        rng_state = wp.rand_init(4321, wp.tid())
        coords = coord_sampler(rng_state)

        # sum of node weights anywhere should be 1.0
        w_sum = type(weight_fn(coords, 0))(0.0)
        for n in range(NODE_COUNT):
            w_sum += weight_fn(coords, n)

        _expect_near(wp.abs(w_sum), type(w_sum)(1.0), 0.0001)

    n_samples = 100
    if partition_of_unity:
        wp.launch(partition_of_unity_test, dim=n_samples, inputs=[])


def test_shape_function_trace(test, shape: ShapeFunction, CENTER_COORDS):
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


def test_shape_function_gradient(
    test,
    shape: ShapeFunction,
    coord_sampler,
    coord_delta_sampler,
    pure_curl: bool = False,
    pure_spherical: bool = False,
):
    weight_fn = shape.make_element_inner_weight()
    weight_gradient_fn = shape.make_element_inner_weight_gradient()

    @wp.func
    def scalar_delta(avg_grad: Any, param_delta: Any):
        return wp.dot(avg_grad, param_delta)

    @wp.func
    def vector_delta(avg_grad: Any, param_delta: Any):
        return avg_grad * param_delta

    grad_delta_fn = scalar_delta if shape.value == shape.Value.Scalar else vector_delta

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
        delta_est = grad_delta_fn(gp + gm, param_delta)
        _expect_near(delta_ref, delta_est, 0.0001)

        if wp.static(pure_curl):
            wp.expect_near(wp.ddot(symmetric_part(gp), symmetric_part(gp)), gp.dtype(0.0))

        if wp.static(pure_spherical):
            deviatoric_part = gp - spherical_part(gp)
            wp.expect_near(wp.ddot(deviatoric_part, deviatoric_part), gp.dtype(0.0))

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

    Q_1 = SquareBipolynomialShapeFunctions(degree=1, family=fem.Polynomial.LOBATTO_GAUSS_LEGENDRE)
    Q_2 = SquareBipolynomialShapeFunctions(degree=2, family=fem.Polynomial.LOBATTO_GAUSS_LEGENDRE)
    Q_3 = SquareBipolynomialShapeFunctions(degree=3, family=fem.Polynomial.LOBATTO_GAUSS_LEGENDRE)

    test_shape_function_weight(test, Q_1, square_coord_sampler, SQUARE_CENTER_COORDS)
    test_shape_function_weight(test, Q_2, square_coord_sampler, SQUARE_CENTER_COORDS)
    test_shape_function_weight(test, Q_3, square_coord_sampler, SQUARE_CENTER_COORDS)
    test_shape_function_trace(test, Q_1, SQUARE_SIDE_CENTER_COORDS)
    test_shape_function_trace(test, Q_2, SQUARE_SIDE_CENTER_COORDS)
    test_shape_function_trace(test, Q_3, SQUARE_SIDE_CENTER_COORDS)
    test_shape_function_gradient(test, Q_1, square_coord_sampler, square_coord_delta_sampler)
    test_shape_function_gradient(test, Q_2, square_coord_sampler, square_coord_delta_sampler)
    test_shape_function_gradient(test, Q_3, square_coord_sampler, square_coord_delta_sampler)

    Q_1 = SquareBipolynomialShapeFunctions(degree=1, family=fem.Polynomial.GAUSS_LEGENDRE)
    Q_2 = SquareBipolynomialShapeFunctions(degree=2, family=fem.Polynomial.GAUSS_LEGENDRE)
    Q_3 = SquareBipolynomialShapeFunctions(degree=3, family=fem.Polynomial.GAUSS_LEGENDRE)

    test_shape_function_weight(test, Q_1, square_coord_sampler, SQUARE_CENTER_COORDS)
    test_shape_function_weight(test, Q_2, square_coord_sampler, SQUARE_CENTER_COORDS)
    test_shape_function_weight(test, Q_3, square_coord_sampler, SQUARE_CENTER_COORDS)
    test_shape_function_gradient(test, Q_1, square_coord_sampler, square_coord_delta_sampler)
    test_shape_function_gradient(test, Q_2, square_coord_sampler, square_coord_delta_sampler)
    test_shape_function_gradient(test, Q_3, square_coord_sampler, square_coord_delta_sampler)

    S_2 = SquareSerendipityShapeFunctions(degree=2, family=fem.Polynomial.LOBATTO_GAUSS_LEGENDRE)
    S_3 = SquareSerendipityShapeFunctions(degree=3, family=fem.Polynomial.LOBATTO_GAUSS_LEGENDRE)

    test_shape_function_weight(test, S_2, square_coord_sampler, SQUARE_CENTER_COORDS)
    test_shape_function_weight(test, S_3, square_coord_sampler, SQUARE_CENTER_COORDS)
    test_shape_function_trace(test, S_2, SQUARE_SIDE_CENTER_COORDS)
    test_shape_function_trace(test, S_3, SQUARE_SIDE_CENTER_COORDS)
    test_shape_function_gradient(test, S_2, square_coord_sampler, square_coord_delta_sampler)
    test_shape_function_gradient(test, S_3, square_coord_sampler, square_coord_delta_sampler)

    P_c1 = SquareNonConformingPolynomialShapeFunctions(degree=1)
    P_c2 = SquareNonConformingPolynomialShapeFunctions(degree=2)
    P_c3 = SquareNonConformingPolynomialShapeFunctions(degree=3)

    test_shape_function_weight(test, P_c1, square_coord_sampler, SQUARE_CENTER_COORDS)
    test_shape_function_weight(test, P_c2, square_coord_sampler, SQUARE_CENTER_COORDS)
    test_shape_function_weight(test, P_c3, square_coord_sampler, SQUARE_CENTER_COORDS)
    test_shape_function_gradient(test, P_c1, square_coord_sampler, square_coord_delta_sampler)
    test_shape_function_gradient(test, P_c2, square_coord_sampler, square_coord_delta_sampler)
    test_shape_function_gradient(test, P_c3, square_coord_sampler, square_coord_delta_sampler)

    N1_1 = SquareNedelecFirstKindShapeFunctions(degree=1)
    test_shape_function_gradient(test, N1_1, square_coord_sampler, square_coord_delta_sampler)
    RT_1 = SquareRaviartThomasShapeFunctions(degree=1)
    test_shape_function_gradient(test, RT_1, square_coord_sampler, square_coord_delta_sampler)

    B_1 = SquareBSplineShapeFunctions(degree=1)
    B_2 = SquareBSplineShapeFunctions(degree=2)
    B_3 = SquareBSplineShapeFunctions(degree=3)

    test_shape_function_weight(test, B_1, square_coord_sampler, SQUARE_CENTER_COORDS)
    test_shape_function_weight(test, B_2, square_coord_sampler, SQUARE_CENTER_COORDS, node_unity=False)
    test_shape_function_weight(test, B_3, square_coord_sampler, SQUARE_CENTER_COORDS, node_unity=False)
    test_shape_function_gradient(test, B_1, square_coord_sampler, square_coord_delta_sampler)
    test_shape_function_gradient(test, B_2, square_coord_sampler, square_coord_delta_sampler)
    test_shape_function_gradient(test, B_3, square_coord_sampler, square_coord_delta_sampler)

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

    Q_1 = CubeTripolynomialShapeFunctions(degree=1, family=fem.Polynomial.LOBATTO_GAUSS_LEGENDRE)
    Q_2 = CubeTripolynomialShapeFunctions(degree=2, family=fem.Polynomial.LOBATTO_GAUSS_LEGENDRE)
    Q_3 = CubeTripolynomialShapeFunctions(degree=3, family=fem.Polynomial.LOBATTO_GAUSS_LEGENDRE)

    test_shape_function_weight(test, Q_1, cube_coord_sampler, CUBE_CENTER_COORDS)
    test_shape_function_weight(test, Q_2, cube_coord_sampler, CUBE_CENTER_COORDS)
    test_shape_function_weight(test, Q_3, cube_coord_sampler, CUBE_CENTER_COORDS)
    test_shape_function_trace(test, Q_1, CUBE_SIDE_CENTER_COORDS)
    test_shape_function_trace(test, Q_2, CUBE_SIDE_CENTER_COORDS)
    test_shape_function_trace(test, Q_3, CUBE_SIDE_CENTER_COORDS)
    test_shape_function_gradient(test, Q_1, cube_coord_sampler, cube_coord_delta_sampler)
    test_shape_function_gradient(test, Q_2, cube_coord_sampler, cube_coord_delta_sampler)
    test_shape_function_gradient(test, Q_3, cube_coord_sampler, cube_coord_delta_sampler)

    Q_1 = CubeTripolynomialShapeFunctions(degree=1, family=fem.Polynomial.GAUSS_LEGENDRE)
    Q_2 = CubeTripolynomialShapeFunctions(degree=2, family=fem.Polynomial.GAUSS_LEGENDRE)
    Q_3 = CubeTripolynomialShapeFunctions(degree=3, family=fem.Polynomial.GAUSS_LEGENDRE)

    test_shape_function_weight(test, Q_1, cube_coord_sampler, CUBE_CENTER_COORDS)
    test_shape_function_weight(test, Q_2, cube_coord_sampler, CUBE_CENTER_COORDS)
    test_shape_function_weight(test, Q_3, cube_coord_sampler, CUBE_CENTER_COORDS)
    test_shape_function_gradient(test, Q_1, cube_coord_sampler, cube_coord_delta_sampler)
    test_shape_function_gradient(test, Q_2, cube_coord_sampler, cube_coord_delta_sampler)
    test_shape_function_gradient(test, Q_3, cube_coord_sampler, cube_coord_delta_sampler)

    S_2 = CubeSerendipityShapeFunctions(degree=2, family=fem.Polynomial.LOBATTO_GAUSS_LEGENDRE)
    S_3 = CubeSerendipityShapeFunctions(degree=3, family=fem.Polynomial.LOBATTO_GAUSS_LEGENDRE)

    test_shape_function_weight(test, S_2, cube_coord_sampler, CUBE_CENTER_COORDS)
    test_shape_function_weight(test, S_3, cube_coord_sampler, CUBE_CENTER_COORDS)
    test_shape_function_trace(test, S_2, CUBE_SIDE_CENTER_COORDS)
    test_shape_function_trace(test, S_3, CUBE_SIDE_CENTER_COORDS)
    test_shape_function_gradient(test, S_2, cube_coord_sampler, cube_coord_delta_sampler)
    test_shape_function_gradient(test, S_3, cube_coord_sampler, cube_coord_delta_sampler)

    P_c1 = CubeNonConformingPolynomialShapeFunctions(degree=1)
    P_c2 = CubeNonConformingPolynomialShapeFunctions(degree=2)
    P_c3 = CubeNonConformingPolynomialShapeFunctions(degree=3)

    test_shape_function_weight(test, P_c1, cube_coord_sampler, CUBE_CENTER_COORDS)
    test_shape_function_weight(test, P_c2, cube_coord_sampler, CUBE_CENTER_COORDS)
    test_shape_function_weight(test, P_c3, cube_coord_sampler, CUBE_CENTER_COORDS)
    test_shape_function_gradient(test, P_c1, cube_coord_sampler, cube_coord_delta_sampler)
    test_shape_function_gradient(test, P_c2, cube_coord_sampler, cube_coord_delta_sampler)
    test_shape_function_gradient(test, P_c3, cube_coord_sampler, cube_coord_delta_sampler)

    N1_1 = CubeNedelecFirstKindShapeFunctions(degree=1)
    test_shape_function_gradient(test, N1_1, cube_coord_sampler, cube_coord_delta_sampler)
    RT_1 = CubeRaviartThomasShapeFunctions(degree=1)
    test_shape_function_gradient(test, RT_1, cube_coord_sampler, cube_coord_delta_sampler)

    B_1 = CubeBSplineShapeFunctions(degree=1)
    B_2 = CubeBSplineShapeFunctions(degree=2)
    B_3 = CubeBSplineShapeFunctions(degree=3)

    test_shape_function_weight(test, B_1, cube_coord_sampler, CUBE_CENTER_COORDS)
    test_shape_function_weight(test, B_2, cube_coord_sampler, CUBE_CENTER_COORDS, node_unity=False)
    test_shape_function_weight(test, B_3, cube_coord_sampler, CUBE_CENTER_COORDS, node_unity=False)
    test_shape_function_gradient(test, B_1, cube_coord_sampler, cube_coord_delta_sampler)
    test_shape_function_gradient(test, B_2, cube_coord_sampler, cube_coord_delta_sampler)
    test_shape_function_gradient(test, B_3, cube_coord_sampler, cube_coord_delta_sampler)

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

    P_1 = TrianglePolynomialShapeFunctions(degree=1)
    P_2 = TrianglePolynomialShapeFunctions(degree=2)
    P_3 = TrianglePolynomialShapeFunctions(degree=3)

    test_shape_function_weight(test, P_1, tri_coord_sampler, TRI_CENTER_COORDS)
    test_shape_function_weight(test, P_2, tri_coord_sampler, TRI_CENTER_COORDS)
    test_shape_function_weight(test, P_3, tri_coord_sampler, TRI_CENTER_COORDS)
    test_shape_function_trace(test, P_1, TRI_SIDE_CENTER_COORDS)
    test_shape_function_trace(test, P_2, TRI_SIDE_CENTER_COORDS)
    test_shape_function_trace(test, P_3, TRI_SIDE_CENTER_COORDS)
    test_shape_function_gradient(test, P_1, tri_coord_sampler, tri_coord_delta_sampler)
    test_shape_function_gradient(test, P_2, tri_coord_sampler, tri_coord_delta_sampler)
    test_shape_function_gradient(test, P_3, tri_coord_sampler, tri_coord_delta_sampler)

    P_1d = TriangleNonConformingPolynomialShapeFunctions(degree=1)
    P_2d = TriangleNonConformingPolynomialShapeFunctions(degree=2)
    P_3d = TriangleNonConformingPolynomialShapeFunctions(degree=3)

    test_shape_function_weight(test, P_1d, tri_coord_sampler, TRI_CENTER_COORDS)
    test_shape_function_weight(test, P_2d, tri_coord_sampler, TRI_CENTER_COORDS)
    test_shape_function_weight(test, P_3d, tri_coord_sampler, TRI_CENTER_COORDS)
    test_shape_function_gradient(test, P_1d, tri_coord_sampler, tri_coord_delta_sampler)
    test_shape_function_gradient(test, P_2d, tri_coord_sampler, tri_coord_delta_sampler)
    test_shape_function_gradient(test, P_3d, tri_coord_sampler, tri_coord_delta_sampler)

    N1_1 = TriangleNedelecFirstKindShapeFunctions(degree=1)
    test_shape_function_gradient(test, N1_1, tri_coord_sampler, tri_coord_delta_sampler, pure_curl=True)

    RT_1 = TriangleRaviartThomasShapeFunctions(degree=1)
    test_shape_function_gradient(test, RT_1, tri_coord_sampler, tri_coord_delta_sampler, pure_spherical=True)

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

    P_1 = TetrahedronPolynomialShapeFunctions(degree=1)
    P_2 = TetrahedronPolynomialShapeFunctions(degree=2)
    P_3 = TetrahedronPolynomialShapeFunctions(degree=3)

    test_shape_function_weight(test, P_1, tet_coord_sampler, TET_CENTER_COORDS)
    test_shape_function_weight(test, P_2, tet_coord_sampler, TET_CENTER_COORDS)
    test_shape_function_weight(test, P_3, tet_coord_sampler, TET_CENTER_COORDS)
    test_shape_function_trace(test, P_1, TET_SIDE_CENTER_COORDS)
    test_shape_function_trace(test, P_2, TET_SIDE_CENTER_COORDS)
    test_shape_function_trace(test, P_3, TET_SIDE_CENTER_COORDS)
    test_shape_function_gradient(test, P_1, tet_coord_sampler, tet_coord_delta_sampler)
    test_shape_function_gradient(test, P_2, tet_coord_sampler, tet_coord_delta_sampler)
    test_shape_function_gradient(test, P_3, tet_coord_sampler, tet_coord_delta_sampler)

    P_1d = TetrahedronNonConformingPolynomialShapeFunctions(degree=1)
    P_2d = TetrahedronNonConformingPolynomialShapeFunctions(degree=2)
    P_3d = TetrahedronNonConformingPolynomialShapeFunctions(degree=3)

    test_shape_function_weight(test, P_1d, tet_coord_sampler, TET_CENTER_COORDS)
    test_shape_function_weight(test, P_2d, tet_coord_sampler, TET_CENTER_COORDS)
    test_shape_function_weight(test, P_3d, tet_coord_sampler, TET_CENTER_COORDS)
    test_shape_function_gradient(test, P_1d, tet_coord_sampler, tet_coord_delta_sampler)
    test_shape_function_gradient(test, P_2d, tet_coord_sampler, tet_coord_delta_sampler)
    test_shape_function_gradient(test, P_3d, tet_coord_sampler, tet_coord_delta_sampler)

    N1_1 = TetrahedronNedelecFirstKindShapeFunctions(degree=1)
    test_shape_function_gradient(test, N1_1, tet_coord_sampler, tet_coord_delta_sampler, pure_curl=True)

    RT_1 = TetrahedronRaviartThomasShapeFunctions(degree=1)
    test_shape_function_gradient(test, RT_1, tet_coord_sampler, tet_coord_delta_sampler, pure_spherical=True)

    wp.synchronize()


class TestFemShape(unittest.TestCase):
    pass


add_function_test(TestFemShape, "test_square_shape_functions", test_square_shape_functions)
add_function_test(TestFemShape, "test_cube_shape_functions", test_cube_shape_functions)
add_function_test(TestFemShape, "test_tri_shape_functions", test_tri_shape_functions)
add_function_test(TestFemShape, "test_tet_shape_functions", test_tet_shape_functions)

if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
