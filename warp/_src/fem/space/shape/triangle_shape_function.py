# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import numpy as np

import warp as wp
from warp._src.fem import cache
from warp._src.fem.types import cached_coords_type

from .shape_function import ShapeFunction

_wp_module_name_ = "warp.fem.space.shape.triangle_shape_function"


def _triangle_node_index(tx: int, ty: int, degree: int):
    VERTEX_NODE_COUNT = 3
    SIDE_INTERIOR_NODE_COUNT = degree - 1

    # Index in similar order to e.g. VTK
    # First vertices, then edge (counterclockwise) then interior points (recursively)

    if tx == 0:
        if ty == 0:
            return 0
        elif ty == degree:
            return 2
        else:
            edge_index = 2
            return VERTEX_NODE_COUNT + SIDE_INTERIOR_NODE_COUNT * edge_index + (SIDE_INTERIOR_NODE_COUNT - ty)
    elif ty == 0:
        if tx == degree:
            return 1
        else:
            edge_index = 0
            return VERTEX_NODE_COUNT + SIDE_INTERIOR_NODE_COUNT * edge_index + tx - 1
    elif tx + ty == degree:
        edge_index = 1
        return VERTEX_NODE_COUNT + SIDE_INTERIOR_NODE_COUNT * edge_index + ty - 1

    vertex_edge_node_count = 3 * degree
    return vertex_edge_node_count + _triangle_node_index(tx - 1, ty - 1, degree - 3)


class TriangleShapeFunction(ShapeFunction):
    """Base class for shape functions defined on triangular elements."""

    VERTEX = wp.constant(0)
    EDGE = wp.constant(1)
    INTERIOR = wp.constant(2)

    VERTEX_NODE_COUNT: int
    """Number of shape function nodes per vertex."""

    EDGE_NODE_COUNT: int
    """Number of shape function nodes per triangle edge (excluding vertex nodes)."""

    INTERIOR_NODE_COUNT: int
    """Number of shape function nodes per triangle (excluding edge and vertex nodes)."""

    @staticmethod
    def node_type_and_index(node_index_in_elt: int):
        pass

    @wp.func
    def _vertex_coords(vidx: int):
        return wp.vec2(
            float(vidx == 1),
            float(vidx == 2),
        )


class TrianglePolynomialShapeFunctions(TriangleShapeFunction):
    """Polynomial Lagrange shape functions on triangular elements."""

    def __init__(self, degree: int, scalar_type: type = wp.float32):
        self.scalar_type = scalar_type
        self.CoordsType = cached_coords_type(scalar_type)
        self.ORDER = wp.constant(degree)

        self.NODES_PER_ELEMENT = wp.constant((degree + 1) * (degree + 2) // 2)
        self.NODES_PER_SIDE = wp.constant(degree + 1)

        self.VERTEX_NODE_COUNT = wp.constant(1)
        self.EDGE_NODE_COUNT = wp.constant(degree - 1)
        self.INTERIOR_NODE_COUNT = wp.constant(max(0, degree - 2) * max(0, degree - 1) // 2)

        triangle_coords = np.empty((self.NODES_PER_ELEMENT, 2), dtype=int)

        for tx in range(degree + 1):
            for ty in range(degree + 1 - tx):
                index = _triangle_node_index(tx, ty, degree)
                triangle_coords[index] = [tx, ty]

        CoordTypeVec = cache.cached_mat_type(dtype=int, shape=(self.NODES_PER_ELEMENT, 2))
        self.NODE_TRIANGLE_COORDS = wp.constant(CoordTypeVec(triangle_coords))

        self.node_type_and_type_index = self._get_node_type_and_type_index()
        self._node_triangle_coordinates = self._get_node_triangle_coordinates()

    @property
    def name(self) -> str:
        suffix = self._precision_suffix
        return f"Tri_P{self.ORDER}{suffix}"

    def _get_node_triangle_coordinates(self):
        NODE_TRIANGLE_COORDS = self.NODE_TRIANGLE_COORDS

        def node_triangle_coordinates(
            node_index_in_elt: int,
        ):
            return NODE_TRIANGLE_COORDS[node_index_in_elt]

        return cache.get_func(node_triangle_coordinates, self.name)

    def _get_node_type_and_type_index(self):
        ORDER = self.ORDER

        def node_type_and_index(
            node_index_in_elt: int,
        ):
            if node_index_in_elt < 3:
                return TrianglePolynomialShapeFunctions.VERTEX, node_index_in_elt

            if node_index_in_elt < 3 * ORDER:
                return TrianglePolynomialShapeFunctions.EDGE, (node_index_in_elt - 3)

            return TrianglePolynomialShapeFunctions.INTERIOR, (node_index_in_elt - 3 * ORDER)

        return cache.get_func(node_type_and_index, self.name)

    def make_node_coords_in_element(self):
        ORDER = self.ORDER
        scalar = self.scalar_type
        CoordsType = self.CoordsType

        def node_coords_in_element(
            node_index_in_elt: int,
        ):
            tri_coords = self._node_triangle_coordinates(node_index_in_elt)
            cx = scalar(tri_coords[0]) / scalar(ORDER)
            cy = scalar(tri_coords[1]) / scalar(ORDER)
            return CoordsType(scalar(1.0) - cx - cy, cx, cy)

        return cache.get_func(node_coords_in_element, self.name)

    def make_node_quadrature_weight(self):
        scalar = self.scalar_type

        if self.ORDER == 3:
            # P3 intrinsic quadrature
            vertex_weight = 1.0 / 30
            edge_weight = 0.075
            interior_weight = 0.45
        elif self.ORDER == 2:
            # Order 1, but optimized quadrature weights for monomials of order <= 4
            vertex_weight = 0.022335964126
            edge_weight = 0.310997369207
            interior_weight = 0.0
        else:
            vertex_weight = 1.0 / self.NODES_PER_ELEMENT
            edge_weight = 1.0 / self.NODES_PER_ELEMENT
            interior_weight = 1.0 / self.NODES_PER_ELEMENT

        VERTEX_WEIGHT = wp.constant(scalar(vertex_weight))
        EDGE_WEIGHT = wp.constant(scalar(edge_weight))
        INTERIOR_WEIGHT = wp.constant(scalar(interior_weight))

        @cache.dynamic_func(suffix=self.name)
        def node_quadrature_weight(node_index_in_element: int):
            node_type, _type_index = self.node_type_and_type_index(node_index_in_element)

            if node_type == TrianglePolynomialShapeFunctions.VERTEX:
                return VERTEX_WEIGHT
            elif node_type == TrianglePolynomialShapeFunctions.EDGE:
                return EDGE_WEIGHT

            return INTERIOR_WEIGHT

        return node_quadrature_weight

    def make_trace_node_quadrature_weight(self):
        scalar = self.scalar_type

        # Closed Newton-Cotes
        if self.ORDER == 3:
            vertex_weight = 1.0 / 8.0
            edge_weight = 3.0 / 8.0
        elif self.ORDER == 2:
            vertex_weight = 1.0 / 6.0
            edge_weight = 2.0 / 3.0
        else:
            vertex_weight = 1.0 / self.NODES_PER_SIDE
            edge_weight = 1.0 / self.NODES_PER_SIDE

        VERTEX_WEIGHT = wp.constant(scalar(vertex_weight))
        EDGE_WEIGHT = wp.constant(scalar(edge_weight))

        @cache.dynamic_func(suffix=self.name)
        def trace_node_quadrature_weight(node_index_in_element: int):
            node_type, _type_index = self.node_type_and_type_index(node_index_in_element)

            return wp.where(node_type == TrianglePolynomialShapeFunctions.VERTEX, VERTEX_WEIGHT, EDGE_WEIGHT)

        return trace_node_quadrature_weight

    def make_element_inner_weight(self):
        ORDER = self.ORDER

        def element_inner_weight_linear(
            coords: Any,
            node_index_in_elt: int,
        ):
            return coords[node_index_in_elt]

        def element_inner_weight_quadratic(
            coords: Any,
            node_index_in_elt: int,
        ):
            node_type, type_index = self.node_type_and_type_index(node_index_in_elt)

            if node_type == TrianglePolynomialShapeFunctions.VERTEX:
                # Vertex
                return coords[type_index] * (coords.dtype(2.0) * coords[type_index] - coords.dtype(1.0))

            # Edge
            c1 = type_index
            c2 = (type_index + 1) % 3
            return coords.dtype(4.0) * coords[c1] * coords[c2]

        def element_inner_weight_cubic(
            coords: Any,
            node_index_in_elt: int,
        ):
            node_type, type_index = self.node_type_and_type_index(node_index_in_elt)

            if node_type == TrianglePolynomialShapeFunctions.VERTEX:
                # Vertex
                return (
                    coords.dtype(0.5)
                    * coords[type_index]
                    * (coords.dtype(3.0) * coords[type_index] - coords.dtype(1.0))
                    * (coords.dtype(3.0) * coords[type_index] - coords.dtype(2.0))
                )

            elif node_type == TrianglePolynomialShapeFunctions.EDGE:
                # Edge
                edge = type_index // 2
                k = type_index - 2 * edge
                c1 = (edge + k) % 3
                c2 = (edge + 1 - k) % 3

                return (
                    coords.dtype(4.5) * coords[c1] * coords[c2] * (coords.dtype(3.0) * coords[c1] - coords.dtype(1.0))
                )

            # Interior
            return coords.dtype(27.0) * coords[0] * coords[1] * coords[2]

        if ORDER == 1:
            return cache.get_func(element_inner_weight_linear, self.name)
        elif ORDER == 2:
            return cache.get_func(element_inner_weight_quadratic, self.name)
        elif ORDER == 3:
            return cache.get_func(element_inner_weight_cubic, self.name)

        return None

    def make_element_inner_weight_gradient(self):
        ORDER = self.ORDER

        def element_inner_weight_gradient_linear(
            coords: Any,
            node_index_in_elt: int,
        ):
            dw_dc = wp.vector(coords.dtype(0.0), coords.dtype(0.0), coords.dtype(0.0), dtype=coords.dtype)
            dw_dc[node_index_in_elt] = coords.dtype(1.0)

            return wp.vector(dw_dc[1] - dw_dc[0], dw_dc[2] - dw_dc[0], dtype=coords.dtype)

        def element_inner_weight_gradient_quadratic(
            coords: Any,
            node_index_in_elt: int,
        ):
            node_type, type_index = self.node_type_and_type_index(node_index_in_elt)

            dw_dc = wp.vector(coords.dtype(0.0), coords.dtype(0.0), coords.dtype(0.0), dtype=coords.dtype)

            if node_type == TrianglePolynomialShapeFunctions.VERTEX:
                # Vertex
                dw_dc[type_index] = coords.dtype(4.0) * coords[type_index] - coords.dtype(1.0)

            else:
                # Edge
                c1 = type_index
                c2 = (type_index + 1) % 3
                dw_dc[c1] = coords.dtype(4.0) * coords[c2]
                dw_dc[c2] = coords.dtype(4.0) * coords[c1]

            return wp.vector(dw_dc[1] - dw_dc[0], dw_dc[2] - dw_dc[0], dtype=coords.dtype)

        def element_inner_weight_gradient_cubic(
            coords: Any,
            node_index_in_elt: int,
        ):
            node_type, type_index = self.node_type_and_type_index(node_index_in_elt)

            dw_dc = wp.vector(coords.dtype(0.0), coords.dtype(0.0), coords.dtype(0.0), dtype=coords.dtype)

            if node_type == TrianglePolynomialShapeFunctions.VERTEX:
                # Vertex
                dw_dc[type_index] = (
                    coords.dtype(0.5) * coords.dtype(27.0) * coords[type_index] * coords[type_index]
                    - coords.dtype(9.0) * coords[type_index]
                    + coords.dtype(1.0)
                )

            elif node_type == TrianglePolynomialShapeFunctions.EDGE:
                # Edge
                edge = type_index // 2
                k = type_index - 2 * edge
                c1 = (edge + k) % 3
                c2 = (edge + 1 - k) % 3

                dw_dc[c1] = coords.dtype(4.5) * coords[c2] * (coords.dtype(6.0) * coords[c1] - coords.dtype(1.0))
                dw_dc[c2] = coords.dtype(4.5) * coords[c1] * (coords.dtype(3.0) * coords[c1] - coords.dtype(1.0))

            else:
                # Interior
                dw_dc = wp.vector(
                    coords.dtype(27.0) * coords[1] * coords[2],
                    coords.dtype(27.0) * coords[2] * coords[0],
                    coords.dtype(27.0) * coords[0] * coords[1],
                    dtype=coords.dtype,
                )

            return wp.vector(dw_dc[1] - dw_dc[0], dw_dc[2] - dw_dc[0], dtype=coords.dtype)

        if ORDER == 1:
            return cache.get_func(element_inner_weight_gradient_linear, self.name)
        elif ORDER == 2:
            return cache.get_func(element_inner_weight_gradient_quadratic, self.name)
        elif ORDER == 3:
            return cache.get_func(element_inner_weight_gradient_cubic, self.name)

        return None

    def element_node_triangulation(self):
        if self.ORDER == 1:
            element_triangles = [[0, 1, 2]]
        if self.ORDER == 2:
            element_triangles = [[0, 3, 5], [3, 1, 4], [2, 5, 4], [3, 4, 5]]
        elif self.ORDER == 3:
            element_triangles = [
                [0, 3, 8],
                [3, 4, 9],
                [4, 1, 5],
                [8, 3, 9],
                [4, 5, 9],
                [8, 9, 7],
                [9, 5, 6],
                [6, 7, 9],
                [7, 6, 2],
            ]

        return np.array(element_triangles)

    def element_vtk_cells(self):
        cells = np.arange(self.NODES_PER_ELEMENT)
        if self.ORDER == 1:
            cell_type = 5  # VTK_TRIANGLE
        else:
            cell_type = 69  # VTK_LAGRANGE_TRIANGLE
        return cells[np.newaxis, :], np.array([cell_type], dtype=np.int8)


class TriangleNonConformingPolynomialShapeFunctions(ShapeFunction):
    """Non-conforming polynomial shape functions on triangular elements."""

    def __init__(self, degree: int, scalar_type: type = wp.float32):
        self.scalar_type = scalar_type
        self.CoordsType = cached_coords_type(scalar_type)
        self._tri_shape = TrianglePolynomialShapeFunctions(degree=degree, scalar_type=scalar_type)
        self.ORDER = self._tri_shape.ORDER
        self.NODES_PER_ELEMENT = self._tri_shape.NODES_PER_ELEMENT

        self.element_node_triangulation = self._tri_shape.element_node_triangulation
        self.element_vtk_cells = self._tri_shape.element_vtk_cells

        # Coordinates (a, b, b) of embedded triangle
        if self.ORDER == 1:
            # Order 2
            a = 2.0 / 3.0
        elif self.ORDER == 2:
            # Order 2, optimized for small intrinsic quadrature error up to degree 4
            a = 0.7790771484375001
        elif self.ORDER == 3:
            # Order 3, optimized for small intrinsic quadrature error up to degree 6
            a = 0.8429443359375002
        else:
            a = 1.0

        b = 0.5 * (1.0 - a)
        self._small_to_big = np.full((3, 3), b) + (a - b) * np.eye(3)
        self._tri_scale = a - b

    @property
    def name(self) -> str:
        suffix = self._precision_suffix
        return f"Tri_dP{self.ORDER}{suffix}"

    def make_node_quadrature_weight(self):
        scalar = self.scalar_type

        # Intrinsic quadrature -- precomputed integral of node shape functions
        # over element. Order equal to self.ORDER

        if self.ORDER == 2:
            vertex_weight = 0.13743348
            edge_weight = 0.19589985
            interior_weight = 0.0
        elif self.ORDER == 3:
            vertex_weight = 0.07462578
            edge_weight = 0.1019807
            interior_weight = 0.16423881
        else:
            vertex_weight = 1.0 / self.NODES_PER_ELEMENT
            edge_weight = 1.0 / self.NODES_PER_ELEMENT
            interior_weight = 1.0 / self.NODES_PER_ELEMENT

        VERTEX_WEIGHT = wp.constant(scalar(vertex_weight))
        EDGE_WEIGHT = wp.constant(scalar(edge_weight))
        INTERIOR_WEIGHT = wp.constant(scalar(interior_weight))

        @cache.dynamic_func(suffix=self.name)
        def node_quadrature_weight(node_index_in_element: int):
            node_type, _type_index = self._tri_shape.node_type_and_type_index(node_index_in_element)

            if node_type == TrianglePolynomialShapeFunctions.VERTEX:
                return VERTEX_WEIGHT
            elif node_type == TrianglePolynomialShapeFunctions.EDGE:
                return EDGE_WEIGHT

            return INTERIOR_WEIGHT

        return node_quadrature_weight

    def make_trace_node_quadrature_weight(self):
        scalar = self.scalar_type

        # Non-conforming, zero measure on sides

        @cache.dynamic_func(suffix=self.name)
        def zero(node_index_in_elt: int):
            return scalar(0.0)

        return zero

    def make_node_coords_in_element(self):
        node_coords_in_tet = self._tri_shape.make_node_coords_in_element()
        scalar = self.scalar_type
        mat33_type = cache.cached_mat_type((3, 3), scalar)

        SMALL_TO_BIG = wp.constant(mat33_type(self._small_to_big))

        @cache.dynamic_func(suffix=self.name)
        def node_coords_in_element(
            node_index_in_elt: int,
        ):
            tri_coords = node_coords_in_tet(node_index_in_elt)
            return SMALL_TO_BIG * tri_coords

        return node_coords_in_element

    def make_element_inner_weight(self):
        tri_inner_weight = self._tri_shape.make_element_inner_weight()
        scalar = self.scalar_type
        mat33_type = cache.cached_mat_type((3, 3), scalar)

        BIG_TO_SMALL = wp.constant(mat33_type(np.linalg.inv(self._small_to_big)))

        @cache.dynamic_func(suffix=self.name)
        def element_inner_weight(
            coords: Any,
            node_index_in_elt: int,
        ):
            tri_coords = BIG_TO_SMALL * coords
            return tri_inner_weight(tri_coords, node_index_in_elt)

        return element_inner_weight

    def make_element_inner_weight_gradient(self):
        tri_inner_weight_gradient = self._tri_shape.make_element_inner_weight_gradient()
        scalar = self.scalar_type
        mat33_type = cache.cached_mat_type((3, 3), scalar)

        BIG_TO_SMALL = wp.constant(mat33_type(np.linalg.inv(self._small_to_big)))
        INV_TRI_SCALE = wp.constant(scalar(1.0 / self._tri_scale))

        @cache.dynamic_func(suffix=self.name)
        def element_inner_weight_gradient(
            coords: Any,
            node_index_in_elt: int,
        ):
            tri_coords = BIG_TO_SMALL * coords
            grad = tri_inner_weight_gradient(tri_coords, node_index_in_elt)
            return INV_TRI_SCALE * grad

        return element_inner_weight_gradient


class TriangleNedelecFirstKindShapeFunctions(TriangleShapeFunction):
    """Nédélec first-kind (edge) shape functions on triangular elements for H(curl) spaces."""

    value = ShapeFunction.Value.CovariantVector

    def __init__(self, degree: int, scalar_type: type = wp.float32):
        self.scalar_type = scalar_type
        self.CoordsType = cached_coords_type(scalar_type)
        if degree != 1:
            raise NotImplementedError("Only linear Nédélec implemented right now")

        self.ORDER = wp.constant(degree)

        self.NODES_PER_ELEMENT = wp.constant(3)
        self.NODES_PER_SIDE = wp.constant(1)

        self.VERTEX_NODE_COUNT = wp.constant(0)
        self.EDGE_NODE_COUNT = wp.constant(1)
        self.INTERIOR_NODE_COUNT = wp.constant(0)

        self.node_type_and_type_index = self._get_node_type_and_type_index()

    @property
    def name(self) -> str:
        suffix = self._precision_suffix
        return f"TriN1_{self.ORDER}{suffix}"

    def _get_node_type_and_type_index(self):
        @cache.dynamic_func(suffix=self.name)
        def node_type_and_index(
            node_index_in_elt: int,
        ):
            return TriangleShapeFunction.EDGE, node_index_in_elt

        return node_type_and_index

    def make_node_coords_in_element(self):
        CoordsType = self.CoordsType
        scalar = self.scalar_type

        @cache.dynamic_func(suffix=self.name)
        def node_coords_in_element(
            node_index_in_elt: int,
        ):
            coords = CoordsType(scalar(0.5))
            coords[(node_index_in_elt + 2) % 3] = scalar(0.0)
            return coords

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

    @wp.func
    def _vertex_coords(vidx: int):
        return wp.vec2(
            float(vidx == 1),
            float(vidx == 2),
        )

    def make_element_inner_weight(self):
        ORDER = self.ORDER
        scalar = self.scalar_type
        vec2_type = wp.types.vector(length=2, dtype=scalar)

        def element_inner_weight_linear(
            coords: Any,
            node_index_in_elt: int,
        ):
            x = vec2_type(coords[1], coords[2])
            p = vec2_type(self._vertex_coords((node_index_in_elt + 2) % 3))

            d = x - p
            return vec2_type(-d[1], d[0])

        if ORDER == 1:
            return cache.get_func(element_inner_weight_linear, self.name)

        return None

    def make_element_inner_weight_gradient(self):
        scalar = self.scalar_type
        mat22_type = wp.types.matrix(shape=(2, 2), dtype=scalar)
        ROT = wp.constant(mat22_type(scalar(0.0), scalar(-1.0), scalar(1.0), scalar(0.0)))

        def element_inner_weight_gradient_linear(
            coords: Any,
            node_index_in_elt: int,
        ):
            return ROT

        if self.ORDER == 1:
            return cache.get_func(element_inner_weight_gradient_linear, self.name)

        return None


class TriangleRaviartThomasShapeFunctions(TriangleShapeFunction):
    """Raviart-Thomas (face) shape functions on triangular elements for H(div) spaces."""

    value = ShapeFunction.Value.ContravariantVector

    def __init__(self, degree: int, scalar_type: type = wp.float32):
        self.scalar_type = scalar_type
        self.CoordsType = cached_coords_type(scalar_type)
        if degree != 1:
            raise NotImplementedError("Only linear Raviart-Thomas implemented right now")

        self.ORDER = wp.constant(degree)

        self.NODES_PER_ELEMENT = wp.constant(3)
        self.NODES_PER_SIDE = wp.constant(1)

        self.VERTEX_NODE_COUNT = wp.constant(0)
        self.EDGE_NODE_COUNT = wp.constant(1)
        self.INTERIOR_NODE_COUNT = wp.constant(0)

        self.node_type_and_type_index = self._get_node_type_and_type_index()

    @property
    def name(self) -> str:
        suffix = self._precision_suffix
        return f"TriRT_{self.ORDER}{suffix}"

    def _get_node_type_and_type_index(self):
        @cache.dynamic_func(suffix=self.name)
        def node_type_and_index(
            node_index_in_elt: int,
        ):
            return TriangleShapeFunction.EDGE, node_index_in_elt

        return node_type_and_index

    def make_node_coords_in_element(self):
        CoordsType = self.CoordsType
        scalar = self.scalar_type

        @cache.dynamic_func(suffix=self.name)
        def node_coords_in_element(
            node_index_in_elt: int,
        ):
            coords = CoordsType(scalar(0.5))
            coords[(node_index_in_elt + 2) % 3] = scalar(0.0)
            return coords

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
        ORDER = self.ORDER
        scalar = self.scalar_type
        vec2_type = wp.types.vector(length=2, dtype=scalar)

        def element_inner_weight_linear(
            coords: Any,
            node_index_in_elt: int,
        ):
            x = vec2_type(coords[1], coords[2])
            p = vec2_type(self._vertex_coords((node_index_in_elt + 2) % 3))

            d = x - p
            return d

        if ORDER == 1:
            return cache.get_func(element_inner_weight_linear, self.name)

        return None

    def make_element_inner_weight_gradient(self):
        scalar = self.scalar_type

        def element_inner_weight_gradient_linear(
            coords: Any,
            node_index_in_elt: int,
        ):
            return wp.identity(n=2, dtype=scalar)

        if self.ORDER == 1:
            return cache.get_func(element_inner_weight_gradient_linear, self.name)

        return None
