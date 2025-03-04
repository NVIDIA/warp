# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import numpy as np

import warp as wp
from warp.fem import cache
from warp.fem.polynomial import Polynomial, is_closed, lagrange_scales, quadrature_1d
from warp.fem.types import Coords

from .shape_function import ShapeFunction
from .triangle_shape_function import TrianglePolynomialShapeFunctions


class SquareShapeFunction(ShapeFunction):
    VERTEX = 0
    EDGE_X = 1
    EDGE_Y = 2
    INTERIOR = 3

    VERTEX_NODE_COUNT: int
    """Number of shape function nodes per vertex"""

    EDGE_NODE_COUNT: int
    """Number of shape function nodes per triangle edge (excluding vertex nodes)"""

    INTERIOR_NODE_COUNT: int
    """Number of shape function nodes per triangle (excluding edge and vertex nodes)"""

    @wp.func
    def _vertex_coords_f(vidx_in_cell: int):
        x = vidx_in_cell // 2
        y = vidx_in_cell - 2 * x
        return wp.vec2(float(x), float(y))


class SquareBipolynomialShapeFunctions(SquareShapeFunction):
    def __init__(self, degree: int, family: Polynomial):
        self.family = family

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

        NodeVec = wp.types.vector(length=degree + 1, dtype=wp.float32)
        self.LOBATTO_COORDS = wp.constant(NodeVec(lobatto_coords))
        self.LOBATTO_WEIGHT = wp.constant(NodeVec(lobatto_weight))
        self.LAGRANGE_SCALE = wp.constant(NodeVec(lagrange_scale))
        self.ORDER_PLUS_ONE = wp.constant(self.ORDER + 1)

        self._node_ij = self._make_node_ij()
        self.node_type_and_type_index = self._make_node_type_and_type_index()

    @property
    def name(self) -> str:
        return f"Square_Q{self.ORDER}_{self.family}"

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

        @cache.dynamic_func(suffix=self.name)
        def node_coords_in_element(
            node_index_in_elt: int,
        ):
            node_i, node_j = self._node_ij(node_index_in_elt)
            return Coords(LOBATTO_COORDS[node_i], LOBATTO_COORDS[node_j], 0.0)

        return node_coords_in_element

    def make_node_quadrature_weight(self):
        ORDER = self.ORDER
        LOBATTO_WEIGHT = self.LOBATTO_WEIGHT

        def node_quadrature_weight(
            node_index_in_elt: int,
        ):
            node_i, node_j = self._node_ij(node_index_in_elt)
            return LOBATTO_WEIGHT[node_i] * LOBATTO_WEIGHT[node_j]

        def node_quadrature_weight_linear(
            node_index_in_elt: int,
        ):
            return 0.25

        if ORDER == 1:
            return cache.get_func(node_quadrature_weight_linear, self.name)

        return cache.get_func(node_quadrature_weight, self.name)

    def make_trace_node_quadrature_weight(self):
        ORDER = self.ORDER
        LOBATTO_WEIGHT = self.LOBATTO_WEIGHT

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

        def trace_node_quadrature_weight_linear(
            node_index_in_elt: int,
        ):
            return 0.5

        def trace_node_quadrature_weight_open(
            node_index_in_elt: int,
        ):
            return 0.0

        if not is_closed(self.family):
            return cache.get_func(trace_node_quadrature_weight_open, self.name)

        if ORDER == 1:
            return cache.get_func(trace_node_quadrature_weight_linear, self.name)

        return cache.get_func(trace_node_quadrature_weight, self.name)

    def make_element_inner_weight(self):
        ORDER_PLUS_ONE = self.ORDER_PLUS_ONE
        LOBATTO_COORDS = self.LOBATTO_COORDS
        LAGRANGE_SCALE = self.LAGRANGE_SCALE

        def element_inner_weight(
            coords: Coords,
            node_index_in_elt: int,
        ):
            node_i, node_j = self._node_ij(node_index_in_elt)

            w = float(1.0)
            for k in range(ORDER_PLUS_ONE):
                if k != node_i:
                    w *= coords[0] - LOBATTO_COORDS[k]
                if k != node_j:
                    w *= coords[1] - LOBATTO_COORDS[k]

            w *= LAGRANGE_SCALE[node_i] * LAGRANGE_SCALE[node_j]

            return w

        def element_inner_weight_linear(
            coords: Coords,
            node_index_in_elt: int,
        ):
            v = SquareBipolynomialShapeFunctions._vertex_coords_f(node_index_in_elt)

            wx = (1.0 - coords[0]) * (1.0 - v[0]) + v[0] * coords[0]
            wy = (1.0 - coords[1]) * (1.0 - v[1]) + v[1] * coords[1]
            return wx * wy

        if self.ORDER == 1 and is_closed(self.family):
            return cache.get_func(element_inner_weight_linear, self.name)

        return cache.get_func(element_inner_weight, self.name)

    def make_element_inner_weight_gradient(self):
        ORDER_PLUS_ONE = self.ORDER_PLUS_ONE
        LOBATTO_COORDS = self.LOBATTO_COORDS
        LAGRANGE_SCALE = self.LAGRANGE_SCALE

        def element_inner_weight_gradient(
            coords: Coords,
            node_index_in_elt: int,
        ):
            node_i, node_j = self._node_ij(node_index_in_elt)

            prefix_x = float(1.0)
            prefix_y = float(1.0)
            for k in range(ORDER_PLUS_ONE):
                if k != node_i:
                    prefix_y *= coords[0] - LOBATTO_COORDS[k]
                if k != node_j:
                    prefix_x *= coords[1] - LOBATTO_COORDS[k]

            grad_x = float(0.0)
            grad_y = float(0.0)

            for k in range(ORDER_PLUS_ONE):
                if k != node_i:
                    delta_x = coords[0] - LOBATTO_COORDS[k]
                    grad_x = grad_x * delta_x + prefix_x
                    prefix_x *= delta_x
                if k != node_j:
                    delta_y = coords[1] - LOBATTO_COORDS[k]
                    grad_y = grad_y * delta_y + prefix_y
                    prefix_y *= delta_y

            grad = LAGRANGE_SCALE[node_i] * LAGRANGE_SCALE[node_j] * wp.vec2(grad_x, grad_y)

            return grad

        def element_inner_weight_gradient_linear(
            coords: Coords,
            node_index_in_elt: int,
        ):
            v = SquareBipolynomialShapeFunctions._vertex_coords_f(node_index_in_elt)

            wx = (1.0 - coords[0]) * (1.0 - v[0]) + v[0] * coords[0]
            wy = (1.0 - coords[1]) * (1.0 - v[1]) + v[1] * coords[1]

            dx = 2.0 * v[0] - 1.0
            dy = 2.0 * v[1] - 1.0

            return wp.vec2(dx * wy, dy * wx)

        if self.ORDER == 1 and is_closed(self.family):
            return cache.get_func(element_inner_weight_gradient_linear, self.name)

        return cache.get_func(element_inner_weight_gradient, self.name)

    def element_node_triangulation(self):
        from warp.fem.utils import grid_to_tris

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
    """
    Serendipity element ~ tensor product space without interior nodes
    Side shape functions are usual Lagrange shape functions times a linear function in the normal direction
    Corner shape functions are bilinear shape functions times a function of (x^{d-1} + y^{d-1})
    """

    def __init__(self, degree: int, family: Polynomial):
        if not is_closed(family):
            raise ValueError("A closed polynomial family is required to define serendipity elements")

        if degree not in [2, 3]:
            raise NotImplementedError("Serendipity element only implemented for order 2 or 3")

        self.family = family

        self.ORDER = wp.constant(degree)
        self.NODES_PER_ELEMENT = wp.constant(4 * degree)
        self.NODES_PER_SIDE = wp.constant(degree + 1)

        self.VERTEX_NODE_COUNT = wp.constant(1)
        self.EDGE_NODE_COUNT = wp.constant(degree - 1)
        self.INTERIOR_NODE_COUNT = wp.constant(0)

        lobatto_coords, lobatto_weight = quadrature_1d(point_count=degree + 1, family=family)
        lagrange_scale = lagrange_scales(lobatto_coords)

        NodeVec = wp.types.vector(length=degree + 1, dtype=wp.float32)
        self.LOBATTO_COORDS = wp.constant(NodeVec(lobatto_coords))
        self.LOBATTO_WEIGHT = wp.constant(NodeVec(lobatto_weight))
        self.LAGRANGE_SCALE = wp.constant(NodeVec(lagrange_scale))
        self.ORDER_PLUS_ONE = wp.constant(self.ORDER + 1)

        self.node_type_and_type_index = self._get_node_type_and_type_index()
        self._node_lobatto_indices = self._get_node_lobatto_indices()

    @property
    def name(self) -> str:
        return f"Square_S{self.ORDER}_{self.family}"

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

        @cache.dynamic_func(suffix=self.name)
        def node_coords_in_element(
            node_index_in_elt: int,
        ):
            node_type, type_instance, type_index = self.node_type_and_type_index(node_index_in_elt)
            node_i, node_j = self._node_lobatto_indices(node_type, type_instance, type_index)
            return Coords(LOBATTO_COORDS[node_i], LOBATTO_COORDS[node_j], 0.0)

        return node_coords_in_element

    def make_node_quadrature_weight(self):
        ORDER = self.ORDER

        @cache.dynamic_func(suffix=self.name)
        def node_quadrature_weight(
            node_index_in_elt: int,
        ):
            node_type, type_instance, type_index = self.node_type_and_type_index(node_index_in_elt)
            if node_type == SquareSerendipityShapeFunctions.VERTEX:
                return 0.25 / float(ORDER * ORDER)

            return (0.25 - 0.25 / float(ORDER * ORDER)) / float(ORDER - 1)

        return node_quadrature_weight

    def make_trace_node_quadrature_weight(self):
        LOBATTO_WEIGHT = self.LOBATTO_WEIGHT

        @cache.dynamic_func(suffix=self.name)
        def trace_node_quadrature_weight(
            node_index_in_elt: int,
        ):
            node_type, type_instance, type_index = self.node_type_and_type_index(node_index_in_elt)
            if node_type == SquareSerendipityShapeFunctions.VERTEX:
                return LOBATTO_WEIGHT[0]

            return LOBATTO_WEIGHT[1 + type_index]

        return trace_node_quadrature_weight

    def make_element_inner_weight(self):
        ORDER = self.ORDER
        ORDER_PLUS_ONE = self.ORDER_PLUS_ONE

        LOBATTO_COORDS = self.LOBATTO_COORDS
        LAGRANGE_SCALE = self.LAGRANGE_SCALE

        DEGREE_3_CIRCLE_RAD = wp.constant(0.5**2 + (0.5 - LOBATTO_COORDS[1]) ** 2)
        DEGREE_3_CIRCLE_SCALE = 1.0 / (0.5 - DEGREE_3_CIRCLE_RAD)

        @cache.dynamic_func(suffix=self.name)
        def element_inner_weight(
            coords: Coords,
            node_index_in_elt: int,
        ):
            node_type, type_instance, type_index = self.node_type_and_type_index(node_index_in_elt)
            node_i, node_j = self._node_lobatto_indices(node_type, type_instance, type_index)

            if node_type == SquareSerendipityShapeFunctions.VERTEX:
                cx = wp.where(node_i == 0, 1.0 - coords[0], coords[0])
                cy = wp.where(node_j == 0, 1.0 - coords[1], coords[1])

                w = cx * cy

                if ORDER == 2:
                    w *= cx + cy - 2.0 + LOBATTO_COORDS[1]
                    return w * LAGRANGE_SCALE[0]
                if ORDER == 3:
                    w *= (cx - 0.5) * (cx - 0.5) + (cy - 0.5) * (cy - 0.5) - DEGREE_3_CIRCLE_RAD
                    return w * DEGREE_3_CIRCLE_SCALE

            w = float(1.0)
            if node_type == SquareSerendipityShapeFunctions.EDGE_Y:
                w *= wp.where(node_i == 0, 1.0 - coords[0], coords[0])
            else:
                for k in range(ORDER_PLUS_ONE):
                    if k != node_i:
                        w *= coords[0] - LOBATTO_COORDS[k]

                w *= LAGRANGE_SCALE[node_i]

            if node_type == SquareSerendipityShapeFunctions.EDGE_X:
                w *= wp.where(node_j == 0, 1.0 - coords[1], coords[1])
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

        DEGREE_3_CIRCLE_RAD = wp.constant(0.5**2 + (0.5 - LOBATTO_COORDS[1]) ** 2)
        DEGREE_3_CIRCLE_SCALE = 1.0 / (0.5 - DEGREE_3_CIRCLE_RAD)

        @cache.dynamic_func(suffix=self.name)
        def element_inner_weight_gradient(
            coords: Coords,
            node_index_in_elt: int,
        ):
            node_type, type_instance, type_index = self.node_type_and_type_index(node_index_in_elt)
            node_i, node_j = self._node_lobatto_indices(node_type, type_instance, type_index)

            if node_type == SquareSerendipityShapeFunctions.VERTEX:
                cx = wp.where(node_i == 0, 1.0 - coords[0], coords[0])
                cy = wp.where(node_j == 0, 1.0 - coords[1], coords[1])

                gx = wp.where(node_i == 0, -1.0, 1.0)
                gy = wp.where(node_j == 0, -1.0, 1.0)

                if ORDER == 2:
                    w = cx + cy - 2.0 + LOBATTO_COORDS[1]
                    grad_x = cy * gx * (w + cx)
                    grad_y = cx * gy * (w + cy)

                    return wp.vec2(grad_x, grad_y) * LAGRANGE_SCALE[0]

                if ORDER == 3:
                    w = (cx - 0.5) * (cx - 0.5) + (cy - 0.5) * (cy - 0.5) - DEGREE_3_CIRCLE_RAD

                    dw_dcx = 2.0 * cx - 1.0
                    dw_dcy = 2.0 * cy - 1.0
                    grad_x = cy * gx * (w + cx * dw_dcx)
                    grad_y = cx * gy * (w + cy * dw_dcy)

                    return wp.vec2(grad_x, grad_y) * DEGREE_3_CIRCLE_SCALE

            if node_type == SquareSerendipityShapeFunctions.EDGE_X:
                prefix_x = wp.where(node_j == 0, 1.0 - coords[1], coords[1])
            else:
                prefix_x = LAGRANGE_SCALE[node_j]
                for k in range(ORDER_PLUS_ONE):
                    if k != node_j:
                        prefix_x *= coords[1] - LOBATTO_COORDS[k]

            if node_type == SquareSerendipityShapeFunctions.EDGE_Y:
                prefix_y = wp.where(node_i == 0, 1.0 - coords[0], coords[0])
            else:
                prefix_y = LAGRANGE_SCALE[node_i]
                for k in range(ORDER_PLUS_ONE):
                    if k != node_i:
                        prefix_y *= coords[0] - LOBATTO_COORDS[k]

            if node_type == SquareSerendipityShapeFunctions.EDGE_X:
                grad_y = wp.where(node_j == 0, -1.0, 1.0) * prefix_y
            else:
                prefix_y *= LAGRANGE_SCALE[node_j]
                grad_y = float(0.0)
                for k in range(ORDER_PLUS_ONE):
                    if k != node_j:
                        delta_y = coords[1] - LOBATTO_COORDS[k]
                        grad_y = grad_y * delta_y + prefix_y
                        prefix_y *= delta_y

            if node_type == SquareSerendipityShapeFunctions.EDGE_Y:
                grad_x = wp.where(node_i == 0, -1.0, 1.0) * prefix_x
            else:
                prefix_x *= LAGRANGE_SCALE[node_i]
                grad_x = float(0.0)
                for k in range(ORDER_PLUS_ONE):
                    if k != node_i:
                        delta_x = coords[0] - LOBATTO_COORDS[k]
                        grad_x = grad_x * delta_x + prefix_x
                        prefix_x *= delta_x

            grad = wp.vec2(grad_x, grad_y)
            return grad

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
    # embeds the largest equilateral triangle centered at (0.5, 0.5) into the reference square
    _tri_height = 0.75
    _tri_side = 2.0 / math.sqrt(3.0) * _tri_height
    _tri_to_square = np.array([[_tri_side, _tri_side / 2.0], [0.0, _tri_height]])

    _TRI_OFFSET = wp.constant(wp.vec2(0.5 - 0.5 * _tri_side, 0.5 - _tri_height / 3.0))

    def __init__(self, degree: int):
        self._tri_shape = TrianglePolynomialShapeFunctions(degree=degree)
        self.ORDER = self._tri_shape.ORDER
        self.NODES_PER_ELEMENT = self._tri_shape.NODES_PER_ELEMENT

        self.element_node_triangulation = self._tri_shape.element_node_triangulation
        self.element_vtk_cells = self._tri_shape.element_vtk_cells

    @property
    def name(self) -> str:
        return f"Square_P{self.ORDER}d"

    def make_node_coords_in_element(self):
        node_coords_in_tet = self._tri_shape.make_node_coords_in_element()

        TRI_TO_SQUARE = wp.constant(wp.mat22(self._tri_to_square))

        @cache.dynamic_func(suffix=self.name)
        def node_coords_in_element(
            node_index_in_elt: int,
        ):
            tri_coords = node_coords_in_tet(node_index_in_elt)
            coords = (
                TRI_TO_SQUARE * wp.vec2(tri_coords[1], tri_coords[2])
            ) + SquareNonConformingPolynomialShapeFunctions._TRI_OFFSET
            return Coords(coords[0], coords[1], 0.0)

        return node_coords_in_element

    def make_node_quadrature_weight(self):
        NODES_PER_ELEMENT = self.NODES_PER_ELEMENT

        if self.ORDER == 2:
            # Intrinsic quadrature (order 2)
            @cache.dynamic_func(suffix=self.name)
            def node_quadrature_weight_quadratic(
                node_index_in_elt: int,
            ):
                node_type, type_index = self._tri_shape.node_type_and_type_index(node_index_in_elt)
                if node_type == TrianglePolynomialShapeFunctions.VERTEX:
                    return 0.18518521
                return 0.14814811

            return node_quadrature_weight_quadratic

        @cache.dynamic_func(suffix=self.name)
        def node_uniform_quadrature_weight(
            node_index_in_elt: int,
        ):
            return 1.0 / float(NODES_PER_ELEMENT)

        return node_uniform_quadrature_weight

    def make_trace_node_quadrature_weight(self):
        # Non-conforming, zero measure on sides

        @wp.func
        def zero(node_index_in_elt: int):
            return 0.0

        return zero

    def make_element_inner_weight(self):
        tri_inner_weight = self._tri_shape.make_element_inner_weight()

        SQUARE_TO_TRI = wp.constant(wp.mat22(np.linalg.inv(self._tri_to_square)))

        @cache.dynamic_func(suffix=self.name)
        def element_inner_weight(
            coords: Coords,
            node_index_in_elt: int,
        ):
            tri_param = SQUARE_TO_TRI * (
                wp.vec2(coords[0], coords[1]) - SquareNonConformingPolynomialShapeFunctions._TRI_OFFSET
            )
            tri_coords = Coords(1.0 - tri_param[0] - tri_param[1], tri_param[0], tri_param[1])

            return tri_inner_weight(tri_coords, node_index_in_elt)

        return element_inner_weight

    def make_element_inner_weight_gradient(self):
        tri_inner_weight_gradient = self._tri_shape.make_element_inner_weight_gradient()

        SQUARE_TO_TRI = wp.constant(wp.mat22(np.linalg.inv(self._tri_to_square)))

        @cache.dynamic_func(suffix=self.name)
        def element_inner_weight_gradient(
            coords: Coords,
            node_index_in_elt: int,
        ):
            tri_param = SQUARE_TO_TRI * (
                wp.vec2(coords[0], coords[1]) - SquareNonConformingPolynomialShapeFunctions._TRI_OFFSET
            )
            tri_coords = Coords(1.0 - tri_param[0] - tri_param[1], tri_param[0], tri_param[1])

            grad = tri_inner_weight_gradient(tri_coords, node_index_in_elt)
            return wp.transpose(SQUARE_TO_TRI) * grad

        return element_inner_weight_gradient


class SquareNedelecFirstKindShapeFunctions(SquareShapeFunction):
    value = ShapeFunction.Value.CovariantVector

    def __init__(self, degree: int):
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
        return f"SquareN1_{self.ORDER}"

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
        @cache.dynamic_func(suffix=self.name)
        def node_coords_in_element(
            node_index_in_elt: int,
        ):
            node_type, type_instance, type_index = self.node_type_and_type_index(node_index_in_elt)
            axis = node_type - SquareShapeFunction.EDGE_X

            coords = Coords()
            coords[axis] = 0.5
            coords[1 - axis] = float(type_instance)

        return node_coords_in_element

    def make_node_quadrature_weight(self):
        NODES_PER_ELEMENT = self.NODES_PER_ELEMENT

        @cache.dynamic_func(suffix=self.name)
        def node_quadrature_weight(node_index_in_element: int):
            return 1.0 / float(NODES_PER_ELEMENT)

        return node_quadrature_weight

    def make_trace_node_quadrature_weight(self):
        NODES_PER_SIDE = self.NODES_PER_SIDE

        @cache.dynamic_func(suffix=self.name)
        def trace_node_quadrature_weight(node_index_in_element: int):
            return 1.0 / float(NODES_PER_SIDE)

        return trace_node_quadrature_weight

    def make_element_inner_weight(self):
        @cache.dynamic_func(suffix=self.name)
        def element_inner_weight(
            coords: Coords,
            node_index_in_elt: int,
        ):
            node_type, type_instance, type_index = self.node_type_and_type_index(node_index_in_elt)

            axis = node_type - SquareShapeFunction.EDGE_X
            a = float(2 * type_instance - 1)
            b = float(1 - type_instance)

            w = wp.vec2(0.0)
            w[axis] = b + a * coords[1 - axis]

            return w

        return element_inner_weight

    def make_element_inner_weight_gradient(self):
        @cache.dynamic_func(suffix=self.name)
        def element_inner_weight_gradient(
            coords: Coords,
            node_index_in_elt: int,
        ):
            node_type, type_instance, type_index = self.node_type_and_type_index(node_index_in_elt)

            axis = node_type - SquareShapeFunction.EDGE_X
            a = float(2 * type_instance - 1)

            grad = wp.mat22(0.0)
            grad[axis, 1 - axis] = a

            return grad

        return element_inner_weight_gradient


class SquareRaviartThomasShapeFunctions(SquareShapeFunction):
    value = ShapeFunction.Value.ContravariantVector

    def __init__(self, degree: int):
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
        return f"SquareRT_{self.ORDER}"

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
        @cache.dynamic_func(suffix=self.name)
        def node_coords_in_element(
            node_index_in_elt: int,
        ):
            node_type, type_instance, type_index = self.node_type_and_type_index(node_index_in_elt)
            axis = node_type - SquareShapeFunction.EDGE_X

            coords = Coords()
            coords[axis] = 0.5
            coords[1 - axis] = float(type_instance)

        return node_coords_in_element

    def make_node_quadrature_weight(self):
        NODES_PER_ELEMENT = self.NODES_PER_ELEMENT

        @cache.dynamic_func(suffix=self.name)
        def node_quadrature_weight(node_index_in_element: int):
            return 1.0 / float(NODES_PER_ELEMENT)

        return node_quadrature_weight

    def make_trace_node_quadrature_weight(self):
        NODES_PER_SIDE = self.NODES_PER_SIDE

        @cache.dynamic_func(suffix=self.name)
        def trace_node_quadrature_weight(node_index_in_element: int):
            return 1.0 / float(NODES_PER_SIDE)

        return trace_node_quadrature_weight

    def make_element_inner_weight(self):
        @cache.dynamic_func(suffix=self.name)
        def element_inner_weight(
            coords: Coords,
            node_index_in_elt: int,
        ):
            node_type, type_instance, type_index = self.node_type_and_type_index(node_index_in_elt)

            axis = node_type - SquareShapeFunction.EDGE_X
            a = float(2 * type_instance - 1)
            b = float(1 - type_instance)

            w = wp.vec2(0.0)
            w[1 - axis] = b + a * coords[1 - axis]

            return w

        return element_inner_weight

    def make_element_inner_weight_gradient(self):
        @cache.dynamic_func(suffix=self.name)
        def element_inner_weight_gradient(
            coords: Coords,
            node_index_in_elt: int,
        ):
            node_type, type_instance, type_index = self.node_type_and_type_index(node_index_in_elt)

            axis = node_type - SquareShapeFunction.EDGE_X
            a = float(2 * type_instance - 1)

            grad = wp.mat22(0.0)
            grad[1 - axis, 1 - axis] = a

            return grad

        return element_inner_weight_gradient
