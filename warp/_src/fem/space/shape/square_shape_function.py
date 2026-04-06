# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Any

import numpy as np

import warp as wp
from warp._src.fem import cache
from warp._src.fem.polynomial import Polynomial, is_closed, lagrange_scales, quadrature_1d
from warp._src.fem.types import cached_coords_type

from .cube_shape_function import CubeBSplineShapeFunctions
from .shape_function import ShapeFunction
from .triangle_shape_function import TrianglePolynomialShapeFunctions

_wp_module_name_ = "warp.fem.space.shape.square_shape_function"


class SquareShapeFunction(ShapeFunction):
    """Base class for shape functions defined on quadrilateral (square) elements."""

    VERTEX = 0
    EDGE_X = 1
    EDGE_Y = 2
    INTERIOR = 3

    VERTEX_NODE_COUNT: int
    """Number of shape function nodes per vertex."""

    EDGE_NODE_COUNT: int
    """Number of shape function nodes per square edge (excluding vertex nodes)."""

    INTERIOR_NODE_COUNT: int
    """Number of shape function nodes per square (excluding edge and vertex nodes)."""

    @wp.func
    def _vertex_coords_f(vidx_in_cell: int):
        x = vidx_in_cell // 2
        y = vidx_in_cell - 2 * x
        return wp.vec2(float(x), float(y))


class SquareBipolynomialShapeFunctions(SquareShapeFunction):
    """Bipolynomial (tensor-product Lagrange) shape functions on quadrilateral elements."""

    def __init__(self, degree: int, family: Polynomial, scalar_type: type = wp.float32):
        self.family = family
        self.scalar_type = scalar_type
        self.CoordsType = cached_coords_type(scalar_type)

        self.ORDER = wp.constant(degree)
        self.NODES_PER_ELEMENT = wp.constant((degree + 1) * (degree + 1))
        self.NODES_PER_SIDE = wp.constant(degree + 1)

        if is_closed(self.family):
            self.VERTEX_NODE_COUNT = wp.constant(1)
            self.EDGE_NODE_COUNT = wp.constant(max(0, degree - 1))
            self.INTERIOR_NODE_COUNT = wp.constant(max(0, degree - 1) ** 2)
        else:
            self.VERTEX_NODE_COUNT = wp.constant(0)
            self.EDGE_NODE_COUNT = wp.constant(0)
            self.INTERIOR_NODE_COUNT = self.NODES_PER_ELEMENT

        lobatto_coords, lobatto_weight = quadrature_1d(point_count=degree + 1, family=family)
        lagrange_scale = lagrange_scales(lobatto_coords)

        NodeVec = cache.cached_vec_type(length=degree + 1, dtype=scalar_type)
        self.LOBATTO_COORDS = wp.constant(NodeVec(lobatto_coords))
        self.LOBATTO_WEIGHT = wp.constant(NodeVec(lobatto_weight))
        self.LAGRANGE_SCALE = wp.constant(NodeVec(lagrange_scale))
        self.ORDER_PLUS_ONE = wp.constant(self.ORDER + 1)

        self._node_ij = self._make_node_ij()
        self.node_type_and_type_index = self._make_node_type_and_type_index()

    @property
    def name(self) -> str:
        suffix = self._precision_suffix
        return f"Square_Q{self.ORDER}_{self.family}{suffix}"

    def _make_node_ij(self):
        ORDER_PLUS_ONE = self.ORDER_PLUS_ONE

        def node_ij(node_index_in_elt: int):
            node_i = node_index_in_elt // ORDER_PLUS_ONE
            node_j = node_index_in_elt - ORDER_PLUS_ONE * node_i
            return node_i, node_j

        return cache.get_func(node_ij, self.name)

    def _make_node_type_and_type_index(self):
        ORDER = self.ORDER

        @cache.dynamic_func(suffix=self.name)
        def node_type_and_type_index_open(
            node_index_in_elt: int,
        ):
            return SquareShapeFunction.INTERIOR, 0, node_index_in_elt

        @cache.dynamic_func(suffix=self.name)
        def node_type_and_type_index(
            node_index_in_elt: int,
        ):
            i, j = self._node_ij(node_index_in_elt)

            zi = int(i == 0)
            zj = int(j == 0)

            mi = int(i == ORDER)
            mj = int(j == ORDER)

            if zi + mi == 1:
                if zj + mj == 1:
                    # vertex
                    type_instance = mi * 2 + mj
                    return SquareShapeFunction.VERTEX, type_instance, 0
                # y edge
                type_index = j - 1
                type_instance = mi
                return SquareShapeFunction.EDGE_Y, type_instance, type_index
            elif zj + mj == 1:
                # x edge
                type_index = i - 1
                type_instance = mj
                return SquareShapeFunction.EDGE_X, type_instance, type_index

            type_index = (i - 1) * (ORDER - 1) + (j - 1)
            return SquareShapeFunction.INTERIOR, 0, type_index

        return node_type_and_type_index if is_closed(self.family) else node_type_and_type_index_open

    def make_node_coords_in_element(self):
        LOBATTO_COORDS = self.LOBATTO_COORDS
        CoordsType = self.CoordsType
        scalar = self.scalar_type

        @cache.dynamic_func(suffix=self.name)
        def node_coords_in_element(
            node_index_in_elt: int,
        ):
            node_i, node_j = self._node_ij(node_index_in_elt)
            return CoordsType(LOBATTO_COORDS[node_i], LOBATTO_COORDS[node_j], scalar(0.0))

        return node_coords_in_element

    def make_node_quadrature_weight(self):
        ORDER = self.ORDER
        LOBATTO_WEIGHT = self.LOBATTO_WEIGHT
        scalar = self.scalar_type

        def node_quadrature_weight(
            node_index_in_elt: int,
        ):
            node_i, node_j = self._node_ij(node_index_in_elt)
            return LOBATTO_WEIGHT[node_i] * LOBATTO_WEIGHT[node_j]

        LINEAR_WEIGHT = wp.constant(scalar(0.25))

        def node_quadrature_weight_linear(
            node_index_in_elt: int,
        ):
            return LINEAR_WEIGHT

        if ORDER == 1:
            return cache.get_func(node_quadrature_weight_linear, self.name)

        return cache.get_func(node_quadrature_weight, self.name)

    def make_trace_node_quadrature_weight(self):
        ORDER = self.ORDER
        LOBATTO_WEIGHT = self.LOBATTO_WEIGHT
        scalar = self.scalar_type

        def trace_node_quadrature_weight(
            node_index_in_elt: int,
        ):
            # We're either on a side interior or at a vertex
            # I.e., either both indices are at extrema, or only one is
            # Pick the interior one if possible, if both are at extrema pick any one
            node_i, node_j = self._node_ij(node_index_in_elt)
            if node_i > 0 and node_i < ORDER:
                return LOBATTO_WEIGHT[node_i]

            return LOBATTO_WEIGHT[node_j]

        LINEAR_WEIGHT = wp.constant(scalar(0.5))

        def trace_node_quadrature_weight_linear(
            node_index_in_elt: int,
        ):
            return LINEAR_WEIGHT

        ZERO_WEIGHT = wp.constant(scalar(0.0))

        def trace_node_quadrature_weight_open(
            node_index_in_elt: int,
        ):
            return ZERO_WEIGHT

        if not is_closed(self.family):
            return cache.get_func(trace_node_quadrature_weight_open, self.name)

        if ORDER == 1:
            return cache.get_func(trace_node_quadrature_weight_linear, self.name)

        return cache.get_func(trace_node_quadrature_weight, self.name)

    def make_element_inner_weight(self):
        ORDER_PLUS_ONE = self.ORDER_PLUS_ONE
        LOBATTO_COORDS = self.LOBATTO_COORDS
        LAGRANGE_SCALE = self.LAGRANGE_SCALE
        scalar = self.scalar_type

        def element_inner_weight(
            coords: Any,
            node_index_in_elt: int,
        ):
            node_i, node_j = self._node_ij(node_index_in_elt)

            w = scalar(1.0)
            for k in range(ORDER_PLUS_ONE):
                if k != node_i:
                    w *= coords[0] - LOBATTO_COORDS[k]
                if k != node_j:
                    w *= coords[1] - LOBATTO_COORDS[k]

            w *= LAGRANGE_SCALE[node_i] * LAGRANGE_SCALE[node_j]

            return w

        def element_inner_weight_linear(
            coords: Any,
            node_index_in_elt: int,
        ):
            v = SquareBipolynomialShapeFunctions._vertex_coords_f(node_index_in_elt)

            wx = (scalar(1.0) - coords[0]) * (scalar(1.0) - scalar(v[0])) + scalar(v[0]) * coords[0]
            wy = (scalar(1.0) - coords[1]) * (scalar(1.0) - scalar(v[1])) + scalar(v[1]) * coords[1]
            return wx * wy

        if self.ORDER == 1 and is_closed(self.family):
            return cache.get_func(element_inner_weight_linear, self.name)

        return cache.get_func(element_inner_weight, self.name)

    def make_element_inner_weight_gradient(self):
        ORDER_PLUS_ONE = self.ORDER_PLUS_ONE
        LOBATTO_COORDS = self.LOBATTO_COORDS
        LAGRANGE_SCALE = self.LAGRANGE_SCALE
        scalar = self.scalar_type
        vec2_type = cache.cached_vec_type(2, scalar)

        def element_inner_weight_gradient(
            coords: Any,
            node_index_in_elt: int,
        ):
            node_i, node_j = self._node_ij(node_index_in_elt)

            prefix_x = scalar(1.0)
            prefix_y = scalar(1.0)
            for k in range(ORDER_PLUS_ONE):
                if k != node_i:
                    prefix_y *= coords[0] - LOBATTO_COORDS[k]
                if k != node_j:
                    prefix_x *= coords[1] - LOBATTO_COORDS[k]

            grad_x = scalar(0.0)
            grad_y = scalar(0.0)

            for k in range(ORDER_PLUS_ONE):
                if k != node_i:
                    delta_x = coords[0] - LOBATTO_COORDS[k]
                    grad_x = grad_x * delta_x + prefix_x
                    prefix_x *= delta_x
                if k != node_j:
                    delta_y = coords[1] - LOBATTO_COORDS[k]
                    grad_y = grad_y * delta_y + prefix_y
                    prefix_y *= delta_y

            grad = LAGRANGE_SCALE[node_i] * LAGRANGE_SCALE[node_j] * vec2_type(grad_x, grad_y)

            return grad

        def element_inner_weight_gradient_linear(
            coords: Any,
            node_index_in_elt: int,
        ):
            v = SquareBipolynomialShapeFunctions._vertex_coords_f(node_index_in_elt)

            wx = (scalar(1.0) - coords[0]) * (scalar(1.0) - scalar(v[0])) + scalar(v[0]) * coords[0]
            wy = (scalar(1.0) - coords[1]) * (scalar(1.0) - scalar(v[1])) + scalar(v[1]) * coords[1]

            dx = scalar(2.0) * scalar(v[0]) - scalar(1.0)
            dy = scalar(2.0) * scalar(v[1]) - scalar(1.0)

            return vec2_type(dx * wy, dy * wx)

        if self.ORDER == 1 and is_closed(self.family):
            return cache.get_func(element_inner_weight_gradient_linear, self.name)

        return cache.get_func(element_inner_weight_gradient, self.name)

    def element_node_triangulation(self):
        from warp._src.fem.utils import grid_to_tris  # noqa: PLC0415

        return grid_to_tris(self.ORDER, self.ORDER)

    def element_vtk_cells(self):
        n = self.ORDER + 1

        # vertices
        cells = [[0, (n - 1) * n, n * n - 1, n - 1]]

        if self.ORDER == 1:
            cell_type = 9  # VTK_QUAD
        else:
            middle = np.arange(1, n - 1)

            # edges
            cells.append(middle * n)
            cells.append(middle + (n - 1) * n)
            cells.append(middle * n + n - 1)
            cells.append(middle)

            # faces
            interior = np.broadcast_to(middle, (n - 2, n - 2))
            cells.append((interior * n + interior.transpose()).flatten())

            cell_type = 70  # VTK_LAGRANGE_QUADRILATERAL

        return np.concatenate(cells)[np.newaxis, :], np.array([cell_type], dtype=np.int8)


class SquareSerendipityShapeFunctions(SquareShapeFunction):
    """Serendipity element: a tensor product space without interior nodes.

    Side shape functions are usual Lagrange shape functions times a linear function in the normal direction.
    Corner shape functions are bilinear shape functions times a function of (x^{d-1} + y^{d-1}).
    """

    def __init__(self, degree: int, family: Polynomial, scalar_type: type = wp.float32):
        if not is_closed(family):
            raise ValueError("A closed polynomial family is required to define serendipity elements")

        if degree not in [2, 3]:
            raise NotImplementedError("Serendipity element only implemented for order 2 or 3")

        self.family = family
        self.scalar_type = scalar_type
        self.CoordsType = cached_coords_type(scalar_type)

        self.ORDER = wp.constant(degree)
        self.NODES_PER_ELEMENT = wp.constant(4 * degree)
        self.NODES_PER_SIDE = wp.constant(degree + 1)

        self.VERTEX_NODE_COUNT = wp.constant(1)
        self.EDGE_NODE_COUNT = wp.constant(degree - 1)
        self.INTERIOR_NODE_COUNT = wp.constant(0)

        lobatto_coords, lobatto_weight = quadrature_1d(point_count=degree + 1, family=family)
        lagrange_scale = lagrange_scales(lobatto_coords)

        NodeVec = cache.cached_vec_type(length=degree + 1, dtype=scalar_type)
        self.LOBATTO_COORDS = wp.constant(NodeVec(lobatto_coords))
        self.LOBATTO_WEIGHT = wp.constant(NodeVec(lobatto_weight))
        self.LAGRANGE_SCALE = wp.constant(NodeVec(lagrange_scale))
        self.ORDER_PLUS_ONE = wp.constant(self.ORDER + 1)

        self.node_type_and_type_index = self._get_node_type_and_type_index()
        self._node_lobatto_indices = self._get_node_lobatto_indices()

    @property
    def name(self) -> str:
        suffix = self._precision_suffix
        return f"Square_S{self.ORDER}_{self.family}{suffix}"

    def _get_node_type_and_type_index(self):
        @cache.dynamic_func(suffix=self.name)
        def node_type_and_index(
            node_index_in_elt: int,
        ):
            if node_index_in_elt < 4:
                return SquareSerendipityShapeFunctions.VERTEX, node_index_in_elt, 0

            edge_index = (node_index_in_elt - 4) // 2
            edge_axis = node_index_in_elt - 4 - 2 * edge_index

            index_in_side = edge_index // 2
            side_offset = edge_index - 2 * index_in_side
            return SquareSerendipityShapeFunctions.EDGE_X + edge_axis, side_offset, index_in_side

        return node_type_and_index

    def _get_node_lobatto_indices(self):
        ORDER = self.ORDER

        @cache.dynamic_func(suffix=self.name)
        def node_lobatto_indices(node_type: int, type_instance: int, type_index: int):
            if node_type == SquareSerendipityShapeFunctions.VERTEX:
                node_i = type_instance // 2
                node_j = type_instance - 2 * node_i
                return node_i * ORDER, node_j * ORDER

            if node_type == SquareSerendipityShapeFunctions.EDGE_X:
                node_i = 1 + type_index
                node_j = type_instance * ORDER
            else:
                node_j = 1 + type_index
                node_i = type_instance * ORDER

            return node_i, node_j

        return node_lobatto_indices

    def make_node_coords_in_element(self):
        LOBATTO_COORDS = self.LOBATTO_COORDS
        CoordsType = self.CoordsType
        scalar = self.scalar_type

        @cache.dynamic_func(suffix=self.name)
        def node_coords_in_element(
            node_index_in_elt: int,
        ):
            node_type, type_instance, type_index = self.node_type_and_type_index(node_index_in_elt)
            node_i, node_j = self._node_lobatto_indices(node_type, type_instance, type_index)
            return CoordsType(LOBATTO_COORDS[node_i], LOBATTO_COORDS[node_j], scalar(0.0))

        return node_coords_in_element

    def make_node_quadrature_weight(self):
        scalar = self.scalar_type
        ORDER = self.ORDER
        VERTEX_WEIGHT = wp.constant(scalar(0.25 / (ORDER * ORDER)))
        EDGE_WEIGHT = wp.constant(scalar((0.25 - 0.25 / (ORDER * ORDER)) / (ORDER - 1)))

        @cache.dynamic_func(suffix=self.name)
        def node_quadrature_weight(
            node_index_in_elt: int,
        ):
            node_type, _type_instance, _type_index = self.node_type_and_type_index(node_index_in_elt)
            if node_type == SquareSerendipityShapeFunctions.VERTEX:
                return VERTEX_WEIGHT

            return EDGE_WEIGHT

        return node_quadrature_weight

    def make_trace_node_quadrature_weight(self):
        LOBATTO_WEIGHT = self.LOBATTO_WEIGHT

        @cache.dynamic_func(suffix=self.name)
        def trace_node_quadrature_weight(
            node_index_in_elt: int,
        ):
            node_type, _type_instance, type_index = self.node_type_and_type_index(node_index_in_elt)
            if node_type == SquareSerendipityShapeFunctions.VERTEX:
                return LOBATTO_WEIGHT[0]

            return LOBATTO_WEIGHT[1 + type_index]

        return trace_node_quadrature_weight

    def make_element_inner_weight(self):
        ORDER = self.ORDER
        ORDER_PLUS_ONE = self.ORDER_PLUS_ONE
        scalar = self.scalar_type

        LOBATTO_COORDS = self.LOBATTO_COORDS
        LAGRANGE_SCALE = self.LAGRANGE_SCALE

        DEGREE_3_CIRCLE_RAD = wp.constant(scalar(0.5**2 + (0.5 - float(LOBATTO_COORDS[1])) ** 2))
        DEGREE_3_CIRCLE_SCALE = scalar(1.0 / (0.5 - float(DEGREE_3_CIRCLE_RAD)))

        @cache.dynamic_func(suffix=self.name)
        def element_inner_weight(
            coords: Any,
            node_index_in_elt: int,
        ):
            node_type, type_instance, type_index = self.node_type_and_type_index(node_index_in_elt)
            node_i, node_j = self._node_lobatto_indices(node_type, type_instance, type_index)

            if node_type == SquareSerendipityShapeFunctions.VERTEX:
                cx = wp.where(node_i == 0, scalar(1.0) - coords[0], coords[0])
                cy = wp.where(node_j == 0, scalar(1.0) - coords[1], coords[1])

                w = cx * cy

                if ORDER == 2:
                    w *= cx + cy - scalar(2.0) + LOBATTO_COORDS[1]
                    return w * LAGRANGE_SCALE[0]
                if ORDER == 3:
                    w *= (
                        (cx - scalar(0.5)) * (cx - scalar(0.5))
                        + (cy - scalar(0.5)) * (cy - scalar(0.5))
                        - DEGREE_3_CIRCLE_RAD
                    )
                    return w * DEGREE_3_CIRCLE_SCALE

            w = scalar(1.0)
            if node_type == SquareSerendipityShapeFunctions.EDGE_Y:
                w *= wp.where(node_i == 0, scalar(1.0) - coords[0], coords[0])
            else:
                for k in range(ORDER_PLUS_ONE):
                    if k != node_i:
                        w *= coords[0] - LOBATTO_COORDS[k]

                w *= LAGRANGE_SCALE[node_i]

            if node_type == SquareSerendipityShapeFunctions.EDGE_X:
                w *= wp.where(node_j == 0, scalar(1.0) - coords[1], coords[1])
            else:
                for k in range(ORDER_PLUS_ONE):
                    if k != node_j:
                        w *= coords[1] - LOBATTO_COORDS[k]
                w *= LAGRANGE_SCALE[node_j]

            return w

        return element_inner_weight

    def make_element_inner_weight_gradient(self):
        ORDER = self.ORDER
        ORDER_PLUS_ONE = self.ORDER_PLUS_ONE
        LOBATTO_COORDS = self.LOBATTO_COORDS
        LAGRANGE_SCALE = self.LAGRANGE_SCALE
        scalar = self.scalar_type
        vec2_type = cache.cached_vec_type(2, scalar)

        DEGREE_3_CIRCLE_RAD = wp.constant(scalar(0.5**2 + (0.5 - float(LOBATTO_COORDS[1])) ** 2))
        DEGREE_3_CIRCLE_SCALE = scalar(1.0 / (0.5 - float(DEGREE_3_CIRCLE_RAD)))

        @cache.dynamic_func(suffix=self.name)
        def element_inner_weight_gradient(
            coords: Any,
            node_index_in_elt: int,
        ):
            node_type, type_instance, type_index = self.node_type_and_type_index(node_index_in_elt)
            node_i, node_j = self._node_lobatto_indices(node_type, type_instance, type_index)

            if node_type == SquareSerendipityShapeFunctions.VERTEX:
                cx = wp.where(node_i == 0, scalar(1.0) - coords[0], coords[0])
                cy = wp.where(node_j == 0, scalar(1.0) - coords[1], coords[1])

                gx = wp.where(node_i == 0, scalar(-1.0), scalar(1.0))
                gy = wp.where(node_j == 0, scalar(-1.0), scalar(1.0))

                if ORDER == 2:
                    w = cx + cy - scalar(2.0) + LOBATTO_COORDS[1]
                    grad_x = cy * gx * (w + cx)
                    grad_y = cx * gy * (w + cy)

                    return vec2_type(grad_x, grad_y) * LAGRANGE_SCALE[0]

                if ORDER == 3:
                    w = (
                        (cx - scalar(0.5)) * (cx - scalar(0.5))
                        + (cy - scalar(0.5)) * (cy - scalar(0.5))
                        - DEGREE_3_CIRCLE_RAD
                    )

                    dw_dcx = scalar(2.0) * cx - scalar(1.0)
                    dw_dcy = scalar(2.0) * cy - scalar(1.0)
                    grad_x = cy * gx * (w + cx * dw_dcx)
                    grad_y = cx * gy * (w + cy * dw_dcy)

                    return vec2_type(grad_x, grad_y) * DEGREE_3_CIRCLE_SCALE

            if node_type == SquareSerendipityShapeFunctions.EDGE_X:
                prefix_x = wp.where(node_j == 0, scalar(1.0) - coords[1], coords[1])
            else:
                prefix_x = LAGRANGE_SCALE[node_j]
                for k in range(ORDER_PLUS_ONE):
                    if k != node_j:
                        prefix_x *= coords[1] - LOBATTO_COORDS[k]

            if node_type == SquareSerendipityShapeFunctions.EDGE_Y:
                prefix_y = wp.where(node_i == 0, scalar(1.0) - coords[0], coords[0])
            else:
                prefix_y = LAGRANGE_SCALE[node_i]
                for k in range(ORDER_PLUS_ONE):
                    if k != node_i:
                        prefix_y *= coords[0] - LOBATTO_COORDS[k]

            if node_type == SquareSerendipityShapeFunctions.EDGE_X:
                grad_y = wp.where(node_j == 0, scalar(-1.0), scalar(1.0)) * prefix_y
            else:
                prefix_y *= LAGRANGE_SCALE[node_j]
                grad_y = scalar(0.0)
                for k in range(ORDER_PLUS_ONE):
                    if k != node_j:
                        delta_y = coords[1] - LOBATTO_COORDS[k]
                        grad_y = grad_y * delta_y + prefix_y
                        prefix_y *= delta_y

            if node_type == SquareSerendipityShapeFunctions.EDGE_Y:
                grad_x = wp.where(node_i == 0, scalar(-1.0), scalar(1.0)) * prefix_x
            else:
                prefix_x *= LAGRANGE_SCALE[node_i]
                grad_x = scalar(0.0)
                for k in range(ORDER_PLUS_ONE):
                    if k != node_i:
                        delta_x = coords[0] - LOBATTO_COORDS[k]
                        grad_x = grad_x * delta_x + prefix_x
                        prefix_x *= delta_x

            return vec2_type(grad_x, grad_y)

        return element_inner_weight_gradient

    def element_node_triangulation(self):
        if self.ORDER == 2:
            element_triangles = [
                [0, 4, 5],
                [5, 4, 6],
                [5, 6, 1],
                [4, 2, 7],
                [4, 7, 6],
                [6, 7, 3],
            ]
        else:
            element_triangles = [
                [0, 4, 5],
                [2, 7, 8],
                [3, 10, 11],
                [1, 9, 6],
                [5, 6, 9],
                [5, 4, 6],
                [8, 11, 10],
                [8, 7, 11],
                [4, 8, 10],
                [4, 10, 6],
            ]

        return element_triangles

    def element_vtk_cells(self):
        tris = np.array(self.element_node_triangulation())
        cell_type = 5  # VTK_TRIANGLE

        return tris, np.full(tris.shape[0], cell_type, dtype=np.int8)


class SquareNonConformingPolynomialShapeFunctions(ShapeFunction):
    """Non-conforming polynomial shape functions on quadrilateral elements using embedded triangles."""

    # embeds the largest equilateral triangle centered at (0.5, 0.5) into the reference square
    _tri_height = 0.75
    _tri_side = 2.0 / math.sqrt(3.0) * _tri_height
    _tri_to_square = np.array([[_tri_side, _tri_side / 2.0], [0.0, _tri_height]])

    _tri_offset_np = np.array([0.5 - 0.5 * _tri_side, 0.5 - _tri_height / 3.0])
    _TRI_OFFSET = wp.constant(wp.vec2(_tri_offset_np))

    def __init__(self, degree: int, scalar_type: type = wp.float32):
        self.scalar_type = scalar_type
        self.CoordsType = cached_coords_type(scalar_type)
        self._tri_shape = TrianglePolynomialShapeFunctions(degree=degree, scalar_type=scalar_type)
        self.ORDER = self._tri_shape.ORDER
        self.NODES_PER_ELEMENT = self._tri_shape.NODES_PER_ELEMENT

        self.element_node_triangulation = self._tri_shape.element_node_triangulation
        self.element_vtk_cells = self._tri_shape.element_vtk_cells

    @property
    def name(self) -> str:
        suffix = self._precision_suffix
        return f"Square_P{self.ORDER}d{suffix}"

    def make_node_coords_in_element(self):
        node_coords_in_tet = self._tri_shape.make_node_coords_in_element()
        CoordsType = self.CoordsType
        scalar = self.scalar_type
        vec2_type = cache.cached_vec_type(2, scalar)
        mat22_type = cache.cached_mat_type((2, 2), scalar)

        TRI_TO_SQUARE = wp.constant(mat22_type(self._tri_to_square))
        TRI_OFFSET = wp.constant(vec2_type(self._tri_offset_np))

        @cache.dynamic_func(suffix=self.name)
        def node_coords_in_element(
            node_index_in_elt: int,
        ):
            tri_coords = node_coords_in_tet(node_index_in_elt)
            coords = (TRI_TO_SQUARE * vec2_type(tri_coords[1], tri_coords[2])) + TRI_OFFSET
            return CoordsType(coords[0], coords[1], scalar(0.0))

        return node_coords_in_element

    def make_node_quadrature_weight(self):
        scalar = self.scalar_type

        if self.ORDER == 2:
            # Intrinsic quadrature (order 2)
            VERTEX_WEIGHT = wp.constant(scalar(0.18518521))
            EDGE_WEIGHT = wp.constant(scalar(0.14814811))

            @cache.dynamic_func(suffix=self.name)
            def node_quadrature_weight_quadratic(
                node_index_in_elt: int,
            ):
                node_type, _type_index = self._tri_shape.node_type_and_type_index(node_index_in_elt)
                if node_type == TrianglePolynomialShapeFunctions.VERTEX:
                    return VERTEX_WEIGHT
                return EDGE_WEIGHT

            return node_quadrature_weight_quadratic

        WEIGHT = wp.constant(scalar(1.0 / self.NODES_PER_ELEMENT))

        @cache.dynamic_func(suffix=self.name)
        def node_uniform_quadrature_weight(
            node_index_in_elt: int,
        ):
            return WEIGHT

        return node_uniform_quadrature_weight

    def make_trace_node_quadrature_weight(self):
        scalar = self.scalar_type

        # Non-conforming, zero measure on sides

        @cache.dynamic_func(suffix=self.name)
        def zero(node_index_in_elt: int):
            return scalar(0.0)

        return zero

    def make_element_inner_weight(self):
        tri_inner_weight = self._tri_shape.make_element_inner_weight()
        scalar = self.scalar_type
        vec2_type = cache.cached_vec_type(2, scalar)
        mat22_type = cache.cached_mat_type((2, 2), scalar)
        CoordsType = self.CoordsType

        SQUARE_TO_TRI = wp.constant(mat22_type(np.linalg.inv(self._tri_to_square)))
        TRI_OFFSET = wp.constant(vec2_type(self._tri_offset_np))

        @cache.dynamic_func(suffix=self.name)
        def element_inner_weight(
            coords: Any,
            node_index_in_elt: int,
        ):
            tri_param = SQUARE_TO_TRI * (vec2_type(coords[0], coords[1]) - TRI_OFFSET)
            tri_coords = CoordsType(scalar(1.0) - tri_param[0] - tri_param[1], tri_param[0], tri_param[1])

            return tri_inner_weight(tri_coords, node_index_in_elt)

        return element_inner_weight

    def make_element_inner_weight_gradient(self):
        tri_inner_weight_gradient = self._tri_shape.make_element_inner_weight_gradient()
        scalar = self.scalar_type
        vec2_type = cache.cached_vec_type(2, scalar)
        mat22_type = cache.cached_mat_type((2, 2), scalar)
        CoordsType = self.CoordsType

        SQUARE_TO_TRI = wp.constant(mat22_type(np.linalg.inv(self._tri_to_square)))
        TRI_OFFSET = wp.constant(vec2_type(self._tri_offset_np))

        @cache.dynamic_func(suffix=self.name)
        def element_inner_weight_gradient(
            coords: Any,
            node_index_in_elt: int,
        ):
            tri_param = SQUARE_TO_TRI * (vec2_type(coords[0], coords[1]) - TRI_OFFSET)
            tri_coords = CoordsType(scalar(1.0) - tri_param[0] - tri_param[1], tri_param[0], tri_param[1])

            grad = tri_inner_weight_gradient(tri_coords, node_index_in_elt)
            return wp.transpose(SQUARE_TO_TRI) * grad

        return element_inner_weight_gradient


class SquareNedelecFirstKindShapeFunctions(SquareShapeFunction):
    """Nédélec first-kind (edge) shape functions on quadrilateral elements for H(curl) spaces."""

    value = ShapeFunction.Value.CovariantVector

    def __init__(self, degree: int, scalar_type: type = wp.float32):
        self.scalar_type = scalar_type
        self.CoordsType = cached_coords_type(scalar_type)
        if degree != 1:
            raise NotImplementedError("Only linear Nédélec implemented right now")

        self.ORDER = wp.constant(degree)
        self.NODES_PER_ELEMENT = wp.constant(4)
        self.NODES_PER_SIDE = wp.constant(1)

        self.VERTEX_NODE_COUNT = wp.constant(0)
        self.EDGE_NODE_COUNT = wp.constant(1)
        self.INTERIOR_NODE_COUNT = wp.constant(0)

        self.node_type_and_type_index = self._get_node_type_and_type_index()

    @property
    def name(self) -> str:
        suffix = self._precision_suffix
        return f"SquareN1_{self.ORDER}{suffix}"

    def _get_node_type_and_type_index(self):
        @cache.dynamic_func(suffix=self.name)
        def node_type_and_index(
            node_index_in_elt: int,
        ):
            axis = node_index_in_elt // 2
            offset = node_index_in_elt - 2 * axis
            return SquareShapeFunction.EDGE_X + axis, offset, 0

        return node_type_and_index

    def make_node_coords_in_element(self):
        CoordsType = self.CoordsType
        scalar = self.scalar_type

        @cache.dynamic_func(suffix=self.name)
        def node_coords_in_element(
            node_index_in_elt: int,
        ):
            node_type, type_instance, _type_index = self.node_type_and_type_index(node_index_in_elt)
            axis = node_type - SquareShapeFunction.EDGE_X

            coords = CoordsType()
            coords[axis] = scalar(0.5)
            coords[1 - axis] = scalar(type_instance)

        return node_coords_in_element

    def make_node_quadrature_weight(self):
        scalar = self.scalar_type
        WEIGHT = wp.constant(scalar(1.0 / self.NODES_PER_ELEMENT))

        @cache.dynamic_func(suffix=self.name)
        def node_quadrature_weight(node_index_in_element: int):
            return WEIGHT

        return node_quadrature_weight

    def make_trace_node_quadrature_weight(self):
        scalar = self.scalar_type
        WEIGHT = wp.constant(scalar(1.0 / self.NODES_PER_SIDE))

        @cache.dynamic_func(suffix=self.name)
        def trace_node_quadrature_weight(node_index_in_element: int):
            return WEIGHT

        return trace_node_quadrature_weight

    def make_element_inner_weight(self):
        scalar = self.scalar_type
        vec2_type = cache.cached_vec_type(2, scalar)

        @cache.dynamic_func(suffix=self.name)
        def element_inner_weight(
            coords: Any,
            node_index_in_elt: int,
        ):
            node_type, type_instance, _type_index = self.node_type_and_type_index(node_index_in_elt)

            axis = node_type - SquareShapeFunction.EDGE_X
            a = scalar(2 * type_instance - 1)
            b = scalar(1 - type_instance)

            w = vec2_type(scalar(0.0))
            w[axis] = b + a * coords[1 - axis]

            return w

        return element_inner_weight

    def make_element_inner_weight_gradient(self):
        scalar = self.scalar_type
        mat22_type = cache.cached_mat_type((2, 2), scalar)

        @cache.dynamic_func(suffix=self.name)
        def element_inner_weight_gradient(
            coords: Any,
            node_index_in_elt: int,
        ):
            node_type, type_instance, _type_index = self.node_type_and_type_index(node_index_in_elt)

            axis = node_type - SquareShapeFunction.EDGE_X
            a = scalar(2 * type_instance - 1)

            grad = mat22_type(scalar(0.0))
            grad[axis, 1 - axis] = a

            return grad

        return element_inner_weight_gradient


class SquareRaviartThomasShapeFunctions(SquareShapeFunction):
    """Raviart-Thomas (face) shape functions on quadrilateral elements for H(div) spaces."""

    value = ShapeFunction.Value.ContravariantVector

    def __init__(self, degree: int, scalar_type: type = wp.float32):
        self.scalar_type = scalar_type
        self.CoordsType = cached_coords_type(scalar_type)
        if degree != 1:
            raise NotImplementedError("Only linear Nédélec implemented right now")

        self.ORDER = wp.constant(degree)
        self.NODES_PER_ELEMENT = wp.constant(4)
        self.NODES_PER_SIDE = wp.constant(1)

        self.VERTEX_NODE_COUNT = wp.constant(0)
        self.EDGE_NODE_COUNT = wp.constant(1)
        self.INTERIOR_NODE_COUNT = wp.constant(0)

        self.node_type_and_type_index = self._get_node_type_and_type_index()

    @property
    def name(self) -> str:
        suffix = self._precision_suffix
        return f"SquareRT_{self.ORDER}{suffix}"

    def _get_node_type_and_type_index(self):
        @cache.dynamic_func(suffix=self.name)
        def node_type_and_index(
            node_index_in_elt: int,
        ):
            axis = node_index_in_elt // 2
            offset = node_index_in_elt - 2 * axis
            return SquareShapeFunction.EDGE_X + axis, offset, 0

        return node_type_and_index

    def make_node_coords_in_element(self):
        CoordsType = self.CoordsType
        scalar = self.scalar_type

        @cache.dynamic_func(suffix=self.name)
        def node_coords_in_element(
            node_index_in_elt: int,
        ):
            node_type, type_instance, _type_index = self.node_type_and_type_index(node_index_in_elt)
            axis = node_type - SquareShapeFunction.EDGE_X

            coords = CoordsType()
            coords[axis] = scalar(0.5)
            coords[1 - axis] = scalar(type_instance)

        return node_coords_in_element

    def make_node_quadrature_weight(self):
        scalar = self.scalar_type
        WEIGHT = wp.constant(scalar(1.0 / self.NODES_PER_ELEMENT))

        @cache.dynamic_func(suffix=self.name)
        def node_quadrature_weight(node_index_in_element: int):
            return WEIGHT

        return node_quadrature_weight

    def make_trace_node_quadrature_weight(self):
        scalar = self.scalar_type
        WEIGHT = wp.constant(scalar(1.0 / self.NODES_PER_SIDE))

        @cache.dynamic_func(suffix=self.name)
        def trace_node_quadrature_weight(node_index_in_element: int):
            return WEIGHT

        return trace_node_quadrature_weight

    def make_element_inner_weight(self):
        scalar = self.scalar_type
        vec2_type = cache.cached_vec_type(2, scalar)

        @cache.dynamic_func(suffix=self.name)
        def element_inner_weight(
            coords: Any,
            node_index_in_elt: int,
        ):
            node_type, type_instance, _type_index = self.node_type_and_type_index(node_index_in_elt)

            axis = node_type - SquareShapeFunction.EDGE_X
            a = scalar(2 * type_instance - 1)
            b = scalar(1 - type_instance)

            w = vec2_type(scalar(0.0))
            w[1 - axis] = b + a * coords[1 - axis]

            return w

        return element_inner_weight

    def make_element_inner_weight_gradient(self):
        scalar = self.scalar_type
        mat22_type = cache.cached_mat_type((2, 2), scalar)

        @cache.dynamic_func(suffix=self.name)
        def element_inner_weight_gradient(
            coords: Any,
            node_index_in_elt: int,
        ):
            node_type, type_instance, _type_index = self.node_type_and_type_index(node_index_in_elt)

            axis = node_type - SquareShapeFunction.EDGE_X
            a = scalar(2 * type_instance - 1)

            grad = mat22_type(scalar(0.0))
            grad[1 - axis, 1 - axis] = a

            return grad

        return element_inner_weight_gradient


class SquareBSplineShapeFunctions(SquareShapeFunction):
    def __init__(self, degree: int, scalar_type: type = wp.float32):
        self.scalar_type = scalar_type
        self.CoordsType = cached_coords_type(scalar_type)
        if degree < 1 or degree > 3:
            raise ValueError("Only degrees 1, 2, and 3 are supported")

        self.ORDER = wp.constant(degree)

        self.PADDING = wp.constant(degree // 2)
        self.NODES_PER_DIM = wp.constant(2 * self.PADDING + 2)

        self.NODES_PER_ELEMENT = wp.constant(self.NODES_PER_DIM**2)
        self.NODES_PER_SIDE = wp.constant(2)

        self._node_ij = self._make_node_ij()

    @property
    def name(self) -> str:
        suffix = self._precision_suffix
        return f"SquareBSpline{self.ORDER}{suffix}"

    def _make_node_ij(self):
        NODES_PER_DIM = self.NODES_PER_DIM
        PADDING = self.PADDING

        def node_ij(
            node_index_in_elt: int,
        ):
            node_i = node_index_in_elt // NODES_PER_DIM
            node_j = node_index_in_elt - NODES_PER_DIM * node_i
            return node_i - PADDING, node_j - PADDING

        return cache.get_func(node_ij, self.name)

    def make_node_coords_in_element(self):
        CoordsType = self.CoordsType
        scalar = self.scalar_type

        @cache.dynamic_func(suffix=self.name)
        def node_coords_in_element(
            node_index_in_elt: int,
        ):
            node_i, node_j = self._node_ij(node_index_in_elt)
            return CoordsType(scalar(node_i), scalar(node_j), scalar(0.0))

        return node_coords_in_element

    def make_node_quadrature_weight(self):
        scalar = self.scalar_type
        WEIGHT = wp.constant(scalar(1.0 / self.NODES_PER_DIM**2))

        def node_quadrature_weight(
            node_index_in_elt: int,
        ):
            return WEIGHT

        return cache.get_func(node_quadrature_weight, self.name)

    def make_trace_node_quadrature_weight(self):
        scalar = self.scalar_type
        WEIGHT = wp.constant(scalar(1.0 / self.NODES_PER_DIM))

        def node_quadrature_weight(
            node_index_in_elt: int,
        ):
            return WEIGHT

        return cache.get_func(node_quadrature_weight, self.name)

    def make_element_inner_weight(self):
        if self.ORDER == 1:
            weight_fn = CubeBSplineShapeFunctions._linear_bspline_weight
        elif self.ORDER == 2:
            weight_fn = CubeBSplineShapeFunctions._quadratic_bspline_weight
        elif self.ORDER == 3:
            weight_fn = CubeBSplineShapeFunctions._cubic_bspline_weight

        node_coords_in_element = self.make_node_coords_in_element()

        def element_inner_weight(
            coords: Any,
            node_index_in_elt: int,
        ):
            node_coords = node_coords_in_element(node_index_in_elt)
            node_delta = coords - node_coords

            wx = weight_fn(node_delta[0])
            wy = weight_fn(node_delta[1])
            return wx * wy

        return cache.get_func(element_inner_weight, self.name)

    def make_element_inner_weight_gradient(self):
        if self.ORDER == 1:
            weight_fn = CubeBSplineShapeFunctions._linear_bspline_weight
            weight_gradient_fn = CubeBSplineShapeFunctions._linear_bspline_weight_gradient
        elif self.ORDER == 2:
            weight_fn = CubeBSplineShapeFunctions._quadratic_bspline_weight
            weight_gradient_fn = CubeBSplineShapeFunctions._quadratic_bspline_weight_gradient
        elif self.ORDER == 3:
            weight_fn = CubeBSplineShapeFunctions._cubic_bspline_weight
            weight_gradient_fn = CubeBSplineShapeFunctions._cubic_bspline_weight_gradient

        node_coords_in_element = self.make_node_coords_in_element()

        def element_inner_weight_gradient(
            coords: Any,
            node_index_in_elt: int,
        ):
            node_coords = node_coords_in_element(node_index_in_elt)
            node_delta = coords - node_coords

            wx = weight_fn(node_delta[0])
            wy = weight_fn(node_delta[1])

            dx = weight_gradient_fn(node_delta[0])
            dy = weight_gradient_fn(node_delta[1])

            return wp.vec2(dx * wy, dy * wx)

        return cache.get_func(element_inner_weight_gradient, self.name)

    def element_node_quads(self):
        from warp._src.fem.utils import grid_to_quads  # noqa: PLC0415

        whole_elt_quads = grid_to_quads(self.NODES_PER_DIM - 1, self.NODES_PER_DIM - 1)
        center_quad = whole_elt_quads[whole_elt_quads.shape[0] // 2]
        return center_quad[np.newaxis, :]

    def element_node_triangulation(self):
        quads = np.array(self.element_node_quads())

        return np.array(
            [
                [quads[0], quads[1], quads[2]],
                [quads[0], quads[2], quads[3]],
            ]
        )

    def element_vtk_cells(self):
        quads = np.array(self.element_node_quads())
        cell_type = 9  # VTK_QUADRILATERAL

        return quads, np.full(quads.shape[0], cell_type, dtype=np.int8)
