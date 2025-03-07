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

import numpy as np

import warp as wp
from warp.fem import cache
from warp.fem.types import Coords

from .shape_function import ShapeFunction


def _tet_node_index(tx: int, ty: int, tz: int, degree: int):
    from .triangle_shape_function import _triangle_node_index

    VERTEX_NODE_COUNT = 4
    EDGE_INTERIOR_NODE_COUNT = degree - 1
    VERTEX_EDGE_NODE_COUNT = VERTEX_NODE_COUNT + 6 * EDGE_INTERIOR_NODE_COUNT
    FACE_INTERIOR_NODE_COUNT = (degree - 1) * (degree - 2) // 2
    VERTEX_EDGE_FACE_NODE_COUNT = VERTEX_EDGE_NODE_COUNT + 4 * FACE_INTERIOR_NODE_COUNT

    # Index in similar order to e.g. VTK
    # First vertices, then edges (counterclockwise), then faces, then interior points (recursively)

    if tx == 0:
        if ty == 0:
            if tz == 0:
                return 0
            elif tz == degree:
                return 3
            else:
                # 0-3 edge
                edge_index = 3
                return VERTEX_NODE_COUNT + EDGE_INTERIOR_NODE_COUNT * edge_index + (tz - 1)
        elif tz == 0:
            if ty == degree:
                return 2
            else:
                # 2-0 edge
                edge_index = 2
                return VERTEX_NODE_COUNT + EDGE_INTERIOR_NODE_COUNT * edge_index + (EDGE_INTERIOR_NODE_COUNT - ty)
        elif tz + ty == degree:
            # 2-3 edge
            edge_index = 5
            return VERTEX_NODE_COUNT + EDGE_INTERIOR_NODE_COUNT * edge_index + (tz - 1)
        else:
            # 2-3-0 face
            face_index = 2
            return (
                VERTEX_EDGE_NODE_COUNT
                + FACE_INTERIOR_NODE_COUNT * face_index
                + _triangle_node_index(degree - 1 - ty - tz, tz - 1, degree - 3)
            )
    elif ty == 0:
        if tz == 0:
            if tx == degree:
                return 1
            else:
                # 0-1 edge
                edge_index = 0
                return VERTEX_NODE_COUNT + EDGE_INTERIOR_NODE_COUNT * edge_index + (tx - 1)
        elif tz + tx == degree:
            # 1-3 edge
            edge_index = 4
            return VERTEX_NODE_COUNT + EDGE_INTERIOR_NODE_COUNT * edge_index + (tz - 1)
        else:
            # 3-0-1 face
            face_index = 3
            return (
                VERTEX_EDGE_NODE_COUNT
                + FACE_INTERIOR_NODE_COUNT * face_index
                + _triangle_node_index(tx - 1, tz - 1, degree - 3)
            )
    elif tz == 0:
        if tx + ty == degree:
            # 1-2 edge
            edge_index = 1
            return VERTEX_NODE_COUNT + EDGE_INTERIOR_NODE_COUNT * edge_index + (ty - 1)
        else:
            # 0-1-2 face
            face_index = 0
            return (
                VERTEX_EDGE_NODE_COUNT
                + FACE_INTERIOR_NODE_COUNT * face_index
                + _triangle_node_index(tx - 1, ty - 1, degree - 3)
            )
    elif tx + ty + tz == degree:
        # 1-2-3 face
        face_index = 1
        return (
            VERTEX_EDGE_NODE_COUNT
            + FACE_INTERIOR_NODE_COUNT * face_index
            + _triangle_node_index(tx - 1, tz - 1, degree - 3)
        )

    return VERTEX_EDGE_FACE_NODE_COUNT + _tet_node_index(tx - 1, ty - 1, tz - 1, degree - 4)


class TetrahedronShapeFunction(ShapeFunction):
    VERTEX = wp.constant(0)
    EDGE = wp.constant(1)
    FACE = wp.constant(2)
    INTERIOR = wp.constant(3)

    VERTEX_NODE_COUNT: int
    """Number of shape function nodes per vertex"""

    EDGE_NODE_COUNT: int
    """Number of shape function nodes per tet edge (excluding vertex nodes)"""

    FACE_NODE_COUNT: int
    """Number of shape function nodes per tet face (excluding edge and vertex nodes)"""

    INTERIOR_NODE_COUNT: int
    """Number of shape function nodes per tet (excluding face, edge and vertex nodes)"""

    @staticmethod
    def node_type_and_index(node_index_in_elt: int):
        pass

    @wp.func
    def edge_vidx(edge: int):
        if edge < 3:
            c1 = edge
            c2 = (edge + 1) % 3
        else:
            c1 = edge - 3
            c2 = 3
        return c1, c2

    @wp.func
    def opposite_edge_vidx(edge: int):
        if edge < 3:
            e1 = (edge + 2) % 3
            e2 = 3
        else:
            e1 = (edge - 2) % 3
            e2 = (edge - 1) % 3
        return e1, e2

    @wp.func
    def _vertex_coords(vidx: int):
        return wp.vec3(
            float(vidx == 1),
            float(vidx == 2),
            float(vidx == 3),
        )


class TetrahedronPolynomialShapeFunctions(TetrahedronShapeFunction):
    def __init__(self, degree: int):
        self.ORDER = wp.constant(degree)

        self.NODES_PER_ELEMENT = wp.constant((degree + 1) * (degree + 2) * (degree + 3) // 6)
        self.NODES_PER_SIDE = wp.constant((degree + 1) * (degree + 2) // 2)

        self.VERTEX_NODE_COUNT = wp.constant(1)
        self.EDGE_NODE_COUNT = wp.constant(degree - 1)
        self.NODES_PER_ELEMENT = wp.constant((degree + 1) * (degree + 2) * (degree + 3) // 6)
        self.NODES_PER_SIDE = wp.constant((degree + 1) * (degree + 2) // 2)

        self.SIDE_NODE_COUNT = wp.constant(self.NODES_PER_ELEMENT - 3 * (self.VERTEX_NODE_COUNT + self.EDGE_NODE_COUNT))
        self.INTERIOR_NODE_COUNT = wp.constant(
            self.NODES_PER_ELEMENT - 3 * (self.VERTEX_NODE_COUNT + self.EDGE_NODE_COUNT)
        )

        self.VERTEX_NODE_COUNT = wp.constant(1)
        self.EDGE_NODE_COUNT = wp.constant(degree - 1)
        self.FACE_NODE_COUNT = wp.constant(max(0, degree - 2) * max(0, degree - 1) // 2)
        self.INERIOR_NODE_COUNT = wp.constant(max(0, degree - 1) * max(0, degree - 2) * max(0, degree - 3) // 6)

        tet_coords = np.empty((self.NODES_PER_ELEMENT, 3), dtype=int)

        for tx in range(degree + 1):
            for ty in range(degree + 1 - tx):
                for tz in range(degree + 1 - tx - ty):
                    index = _tet_node_index(tx, ty, tz, degree)
                    tet_coords[index] = [tx, ty, tz]

        CoordTypeVec = wp.mat(dtype=int, shape=(self.NODES_PER_ELEMENT, 3))
        self.NODE_TET_COORDS = wp.constant(CoordTypeVec(tet_coords))

        self.node_type_and_type_index = self._get_node_type_and_type_index()
        self._node_tet_coordinates = self._get_node_tet_coordinates()

    @property
    def name(self) -> str:
        return f"Tet_P{self.ORDER}"

    def _get_node_tet_coordinates(self):
        NODE_TET_COORDS = self.NODE_TET_COORDS

        def node_tet_coordinates(
            node_index_in_elt: int,
        ):
            return NODE_TET_COORDS[node_index_in_elt]

        return cache.get_func(node_tet_coordinates, self.name)

    def _get_node_type_and_type_index(self):
        ORDER = self.ORDER

        def node_type_and_index(
            node_index_in_elt: int,
        ):
            if node_index_in_elt < 4:
                return TetrahedronPolynomialShapeFunctions.VERTEX, node_index_in_elt

            if node_index_in_elt < (6 * ORDER - 2):
                return TetrahedronPolynomialShapeFunctions.EDGE, (node_index_in_elt - 4)

            if node_index_in_elt < (2 * ORDER * ORDER + 2):
                return TetrahedronPolynomialShapeFunctions.FACE, (node_index_in_elt - (6 * ORDER - 2))

            return TetrahedronPolynomialShapeFunctions.INTERIOR, (node_index_in_elt - (2 * ORDER * ORDER + 2))

        return cache.get_func(node_type_and_index, self.name)

    def make_node_coords_in_element(self):
        ORDER = self.ORDER

        def node_coords_in_element(
            node_index_in_elt: int,
        ):
            tet_coords = self._node_tet_coordinates(node_index_in_elt)
            cx = float(tet_coords[0]) / float(ORDER)
            cy = float(tet_coords[1]) / float(ORDER)
            cz = float(tet_coords[2]) / float(ORDER)
            return Coords(cx, cy, cz)

        return cache.get_func(node_coords_in_element, self.name)

    def make_node_quadrature_weight(self):
        if self.ORDER == 3:
            # Order 1, but optimized quadrature weights for monomials of order <= 6
            vertex_weight = 0.007348845656
            edge_weight = 0.020688129855
            face_weight = 0.180586764778
            interior_weight = 0.0
        else:
            vertex_weight = 1.0 / self.NODES_PER_ELEMENT
            edge_weight = 1.0 / self.NODES_PER_ELEMENT
            face_weight = 1.0 / self.NODES_PER_ELEMENT
            interior_weight = 1.0 / self.NODES_PER_ELEMENT

        VERTEX_WEIGHT = wp.constant(vertex_weight)
        EDGE_WEIGHT = wp.constant(edge_weight)
        FACE_WEIGHT = wp.constant(face_weight)
        INTERIOR_WEIGHT = wp.constant(interior_weight)

        @cache.dynamic_func(suffix=self.name)
        def node_quadrature_weight(node_index_in_element: int):
            node_type, type_index = self.node_type_and_type_index(node_index_in_element)

            if node_type == TetrahedronPolynomialShapeFunctions.VERTEX:
                return VERTEX_WEIGHT
            elif node_type == TetrahedronPolynomialShapeFunctions.EDGE:
                return EDGE_WEIGHT
            elif node_type == TetrahedronPolynomialShapeFunctions.FACE:
                return FACE_WEIGHT

            return INTERIOR_WEIGHT

        return node_quadrature_weight

    def make_trace_node_quadrature_weight(self):
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
            vertex_weight = 1.0 / self.NODES_PER_SIDE
            edge_weight = 1.0 / self.NODES_PER_SIDE
            interior_weight = 1.0 / self.NODES_PER_SIDE

        VERTEX_WEIGHT = wp.constant(vertex_weight)
        EDGE_WEIGHT = wp.constant(edge_weight)
        FACE_INTERIOR_WEIGHT = wp.constant(interior_weight)

        @cache.dynamic_func(suffix=self.name)
        def trace_node_quadrature_weight(node_index_in_element: int):
            node_type, type_index = self.node_type_and_type_index(node_index_in_element)

            if node_type == TetrahedronPolynomialShapeFunctions.VERTEX:
                return VERTEX_WEIGHT
            elif node_type == TetrahedronPolynomialShapeFunctions.EDGE:
                return EDGE_WEIGHT

            return FACE_INTERIOR_WEIGHT

        return trace_node_quadrature_weight

    def make_element_inner_weight(self):
        ORDER = self.ORDER

        def element_inner_weight_linear(
            coords: Coords,
            node_index_in_elt: int,
        ):
            if node_index_in_elt < 0 or node_index_in_elt >= 4:
                return 0.0

            tet_coords = wp.vec4(1.0 - coords[0] - coords[1] - coords[2], coords[0], coords[1], coords[2])
            return tet_coords[node_index_in_elt]

        def element_inner_weight_quadratic(
            coords: Coords,
            node_index_in_elt: int,
        ):
            node_type, type_index = self.node_type_and_type_index(node_index_in_elt)

            tet_coords = wp.vec4(1.0 - coords[0] - coords[1] - coords[2], coords[0], coords[1], coords[2])

            if node_type == TetrahedronPolynomialShapeFunctions.VERTEX:
                # Vertex
                return tet_coords[type_index] * (2.0 * tet_coords[type_index] - 1.0)

            elif node_type == TetrahedronPolynomialShapeFunctions.EDGE:
                # Edge
                c1, c2 = TetrahedronShapeFunction.edge_vidx(type_index)
                return 4.0 * tet_coords[c1] * tet_coords[c2]

            return 0.0

        def element_inner_weight_cubic(
            coords: Coords,
            node_index_in_elt: int,
        ):
            node_type, type_index = self.node_type_and_type_index(node_index_in_elt)

            tet_coords = wp.vec4(1.0 - coords[0] - coords[1] - coords[2], coords[0], coords[1], coords[2])

            if node_type == TetrahedronPolynomialShapeFunctions.VERTEX:
                # Vertex
                return (
                    0.5
                    * tet_coords[type_index]
                    * (3.0 * tet_coords[type_index] - 1.0)
                    * (3.0 * tet_coords[type_index] - 2.0)
                )

            elif node_type == TetrahedronPolynomialShapeFunctions.EDGE:
                # Edge
                edge = type_index // 2
                edge_node = type_index - 2 * edge

                if edge < 3:
                    c1 = (edge + edge_node) % 3
                    c2 = (edge + 1 - edge_node) % 3
                elif edge_node == 0:
                    c1 = edge - 3
                    c2 = 3
                else:
                    c1 = 3
                    c2 = edge - 3

                return 4.5 * tet_coords[c1] * tet_coords[c2] * (3.0 * tet_coords[c1] - 1.0)

            elif node_type == TetrahedronPolynomialShapeFunctions.FACE:
                # Interior
                c1 = type_index
                c2 = (c1 + 1) % 4
                c3 = (c1 + 2) % 4
                return 27.0 * tet_coords[c1] * tet_coords[c2] * tet_coords[c3]

            return 0.0

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
            coords: Coords,
            node_index_in_elt: int,
        ):
            if node_index_in_elt < 0 or node_index_in_elt >= 4:
                return wp.vec3(0.0)

            dw_dc = wp.vec4(0.0)
            dw_dc[node_index_in_elt] = 1.0

            dw_du = wp.vec3(dw_dc[1] - dw_dc[0], dw_dc[2] - dw_dc[0], dw_dc[3] - dw_dc[0])

            return dw_du

        def element_inner_weight_gradient_quadratic(
            coords: Coords,
            node_index_in_elt: int,
        ):
            node_type, type_index = self.node_type_and_type_index(node_index_in_elt)

            tet_coords = wp.vec4(1.0 - coords[0] - coords[1] - coords[2], coords[0], coords[1], coords[2])
            dw_dc = wp.vec4(0.0)

            if node_type == TetrahedronPolynomialShapeFunctions.VERTEX:
                # Vertex
                dw_dc[type_index] = 4.0 * tet_coords[type_index] - 1.0

            elif node_type == TetrahedronPolynomialShapeFunctions.EDGE:
                # Edge
                c1, c2 = TetrahedronShapeFunction.edge_vidx(type_index)
                dw_dc[c1] = 4.0 * tet_coords[c2]
                dw_dc[c2] = 4.0 * tet_coords[c1]

            dw_du = wp.vec3(dw_dc[1] - dw_dc[0], dw_dc[2] - dw_dc[0], dw_dc[3] - dw_dc[0])
            return dw_du

        def element_inner_weight_gradient_cubic(
            coords: Coords,
            node_index_in_elt: int,
        ):
            node_type, type_index = self.node_type_and_type_index(node_index_in_elt)

            tet_coords = wp.vec4(1.0 - coords[0] - coords[1] - coords[2], coords[0], coords[1], coords[2])

            dw_dc = wp.vec4(0.0)

            if node_type == TetrahedronPolynomialShapeFunctions.VERTEX:
                # Vertex
                dw_dc[type_index] = (
                    0.5 * 27.0 * tet_coords[type_index] * tet_coords[type_index] - 9.0 * tet_coords[type_index] + 1.0
                )

            elif node_type == TetrahedronPolynomialShapeFunctions.EDGE:
                # Edge
                edge = type_index // 2
                edge_node = type_index - 2 * edge

                if edge < 3:
                    c1 = (edge + edge_node) % 3
                    c2 = (edge + 1 - edge_node) % 3
                elif edge_node == 0:
                    c1 = edge - 3
                    c2 = 3
                else:
                    c1 = 3
                    c2 = edge - 3

                dw_dc[c1] = 4.5 * tet_coords[c2] * (6.0 * tet_coords[c1] - 1.0)
                dw_dc[c2] = 4.5 * tet_coords[c1] * (3.0 * tet_coords[c1] - 1.0)

            elif node_type == TetrahedronPolynomialShapeFunctions.FACE:
                # Interior
                c1 = type_index
                c2 = (c1 + 1) % 4
                c3 = (c1 + 2) % 4

                dw_dc[c1] = 27.0 * tet_coords[c2] * tet_coords[c3]
                dw_dc[c2] = 27.0 * tet_coords[c3] * tet_coords[c1]
                dw_dc[c3] = 27.0 * tet_coords[c1] * tet_coords[c2]

            dw_du = wp.vec3(dw_dc[1] - dw_dc[0], dw_dc[2] - dw_dc[0], dw_dc[3] - dw_dc[0])
            return dw_du

        if ORDER == 1:
            return cache.get_func(element_inner_weight_gradient_linear, self.name)
        elif ORDER == 2:
            return cache.get_func(element_inner_weight_gradient_quadratic, self.name)
        elif ORDER == 3:
            return cache.get_func(element_inner_weight_gradient_cubic, self.name)

        return None

    def element_node_tets(self):
        if self.ORDER == 1:
            element_tets = [[0, 1, 2, 3]]
        if self.ORDER == 2:
            element_tets = [
                [0, 4, 6, 7],
                [1, 5, 4, 8],
                [2, 6, 5, 9],
                [3, 7, 8, 9],
                [4, 5, 6, 8],
                [8, 7, 9, 6],
                [6, 5, 9, 8],
                [6, 8, 7, 4],
            ]
        elif self.ORDER == 3:
            raise NotImplementedError()

        return np.array(element_tets)

    def element_vtk_cells(self):
        cells = np.arange(self.NODES_PER_ELEMENT)
        if self.ORDER == 1:
            cell_type = 10  # VTK_TETRA
        else:
            cell_type = 71  # VTK_LAGRANGE_TETRAHEDRON
        return cells[np.newaxis, :], np.array([cell_type], dtype=np.int8)


class TetrahedronNonConformingPolynomialShapeFunctions(ShapeFunction):
    def __init__(self, degree: int):
        self._tet_shape = TetrahedronPolynomialShapeFunctions(degree=degree)
        self.ORDER = self._tet_shape.ORDER
        self.NODES_PER_ELEMENT = self._tet_shape.NODES_PER_ELEMENT

        self.element_node_tets = self._tet_shape.element_node_tets
        self.element_vtk_cells = self._tet_shape.element_vtk_cells

        if self.ORDER == 1:
            self._TET_SCALE = 0.4472135955  # so v at 0.5854101966249680 (order 2)
        elif self.ORDER == 2:
            self._TET_SCALE = 0.6123779296874996  # optimized for low intrinsic quadrature error of deg 4
        elif self.ORDER == 3:
            self._TET_SCALE = 0.7153564453124999  # optimized for low intrinsic quadrature error of deg 6
        else:
            self._TET_SCALE = 1.0

        self._TET_SCALE = wp.constant(self._TET_SCALE)
        self._TET_OFFSET = wp.constant((1.0 - self._TET_SCALE) * wp.vec3(0.25, 0.25, 0.25))

    @property
    def name(self) -> str:
        return f"Tet_P{self.ORDER}d"

    def make_node_coords_in_element(self):
        node_coords_in_tet = self._tet_shape.make_node_coords_in_element()

        TET_SCALE = self._TET_SCALE
        TET_OFFSET = self._TET_OFFSET

        @cache.dynamic_func(suffix=self.name)
        def node_coords_in_element(
            node_index_in_elt: int,
        ):
            tet_coords = node_coords_in_tet(node_index_in_elt)
            return TET_SCALE * tet_coords + TET_OFFSET

        return node_coords_in_element

    def make_node_quadrature_weight(self):
        # Intrinsic quadrature -- precomputed integral of node shape functions
        # over element. Order equal to self.ORDER

        if self.ORDER == 2:
            vertex_weight = 0.07499641
            edge_weight = 0.11666908
            face_interior_weight = 0.0
        elif self.ORDER == 3:
            vertex_weight = 0.03345134
            edge_weight = 0.04521887
            face_interior_weight = 0.08089206
        else:
            vertex_weight = 1.0 / self.NODES_PER_ELEMENT
            edge_weight = 1.0 / self.NODES_PER_ELEMENT
            face_interior_weight = 1.0 / self.NODES_PER_ELEMENT

        VERTEX_WEIGHT = wp.constant(vertex_weight)
        EDGE_WEIGHT = wp.constant(edge_weight)
        FACE_INTERIOR_WEIGHT = wp.constant(face_interior_weight)

        @cache.dynamic_func(suffix=self.name)
        def node_quadrature_weight(node_index_in_element: int):
            node_type, type_index = self._tet_shape.node_type_and_type_index(node_index_in_element)

            if node_type == TetrahedronPolynomialShapeFunctions.VERTEX:
                return VERTEX_WEIGHT
            elif node_type == TetrahedronPolynomialShapeFunctions.EDGE:
                return EDGE_WEIGHT

            return FACE_INTERIOR_WEIGHT

        return node_quadrature_weight

    def make_trace_node_quadrature_weight(self):
        # Non-conforming, zero measure on sides

        @wp.func
        def zero(node_index_in_elt: int):
            return 0.0

        return zero

    def make_element_inner_weight(self):
        tet_inner_weight = self._tet_shape.make_element_inner_weight()

        TET_SCALE = self._TET_SCALE
        TET_OFFSET = self._TET_OFFSET

        @cache.dynamic_func(suffix=self.name)
        def element_inner_weight(
            coords: Coords,
            node_index_in_elt: int,
        ):
            tet_coords = (coords - TET_OFFSET) / TET_SCALE

            return tet_inner_weight(tet_coords, node_index_in_elt)

        return element_inner_weight

    def make_element_inner_weight_gradient(self):
        tet_inner_weight_gradient = self._tet_shape.make_element_inner_weight_gradient()

        TET_SCALE = self._TET_SCALE
        TET_OFFSET = self._TET_OFFSET

        @cache.dynamic_func(suffix=self.name)
        def element_inner_weight_gradient(
            coords: Coords,
            node_index_in_elt: int,
        ):
            tet_coords = (coords - TET_OFFSET) / TET_SCALE
            grad = tet_inner_weight_gradient(tet_coords, node_index_in_elt)
            return grad / TET_SCALE

        return element_inner_weight_gradient


class TetrahedronNedelecFirstKindShapeFunctions(TetrahedronShapeFunction):
    value = ShapeFunction.Value.CovariantVector

    def __init__(self, degree: int):
        if degree != 1:
            raise NotImplementedError("Only linear Nédélec implemented right now")

        self.ORDER = wp.constant(degree)

        self.NODES_PER_ELEMENT = wp.constant(6)
        self.NODES_PER_SIDE = wp.constant(3)

        self.VERTEX_NODE_COUNT = wp.constant(0)
        self.EDGE_NODE_COUNT = wp.constant(1)
        self.FACE_NODE_COUNT = wp.constant(0)
        self.INTERIOR_NODE_COUNT = wp.constant(0)

        self.node_type_and_type_index = self._get_node_type_and_type_index()

    @property
    def name(self) -> str:
        return f"TetN1_{self.ORDER}"

    def _get_node_type_and_type_index(self):
        @cache.dynamic_func(suffix=self.name)
        def node_type_and_index(
            node_index_in_elt: int,
        ):
            return TetrahedronShapeFunction.EDGE, node_index_in_elt

        return node_type_and_index

    def make_node_coords_in_element(self):
        @cache.dynamic_func(suffix=self.name)
        def node_coords_in_element(
            node_index_in_elt: int,
        ):
            c1, c2 = TetrahedronShapeFunction.edge_vidx(node_index_in_elt)

            coords = wp.vec4(0.0)
            coords[c1] = 0.5
            coords[c2] = 0.5

            return Coords(coords[1], coords[2], coords[3])

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
        ORDER = self.ORDER

        def element_inner_weight_linear(
            coords: Coords,
            node_index_in_elt: int,
        ):
            e1, e2 = TetrahedronShapeFunction.opposite_edge_vidx(node_index_in_elt)

            v1 = self._vertex_coords(e1)
            v2 = self._vertex_coords(e2)

            nor = v2 - v1
            return wp.cross(nor, coords - v1)

        if ORDER == 1:
            return cache.get_func(element_inner_weight_linear, self.name)

        return None

    def make_element_inner_weight_gradient(self):
        ORDER = self.ORDER

        def element_inner_weight_gradient_linear(
            coords: Coords,
            node_index_in_elt: int,
        ):
            e1, e2 = TetrahedronShapeFunction.opposite_edge_vidx(node_index_in_elt)

            v1 = self._vertex_coords(e1)
            v2 = self._vertex_coords(e2)

            nor = v2 - v1
            return wp.skew(nor)

        if ORDER == 1:
            return cache.get_func(element_inner_weight_gradient_linear, self.name)

        return None


class TetrahedronRaviartThomasShapeFunctions(TetrahedronShapeFunction):
    value = ShapeFunction.Value.ContravariantVector

    def __init__(self, degree: int):
        if degree != 1:
            raise NotImplementedError("Only linear Raviart-Thomas implemented right now")

        self.ORDER = wp.constant(degree)

        self.NODES_PER_ELEMENT = wp.constant(4)
        self.NODES_PER_SIDE = wp.constant(1)

        self.VERTEX_NODE_COUNT = wp.constant(0)
        self.EDGE_NODE_COUNT = wp.constant(0)
        self.FACE_NODE_COUNT = wp.constant(1)
        self.INTERIOR_NODE_COUNT = wp.constant(0)

        self.node_type_and_type_index = self._get_node_type_and_type_index()

    @property
    def name(self) -> str:
        return f"TetRT_{self.ORDER}"

    def _get_node_type_and_type_index(self):
        @cache.dynamic_func(suffix=self.name)
        def node_type_and_index(
            node_index_in_elt: int,
        ):
            return TetrahedronShapeFunction.FACE, node_index_in_elt

        return node_type_and_index

    def make_node_coords_in_element(self):
        @cache.dynamic_func(suffix=self.name)
        def node_coords_in_element(
            node_index_in_elt: int,
        ):
            v = (node_index_in_elt + 3) % 4

            coords = wp.vec4(1.0 / 3.0)
            coords[v] = 0.0

            return Coords(coords[1], coords[2], coords[3])

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
        ORDER = self.ORDER

        def element_inner_weight_linear(
            coords: Coords,
            node_index_in_elt: int,
        ):
            v = (node_index_in_elt + 3) % 4

            return 2.0 * (coords - self._vertex_coords(v))

        if ORDER == 1:
            return cache.get_func(element_inner_weight_linear, self.name)

        return None

    def make_element_inner_weight_gradient(self):
        ORDER = self.ORDER

        def element_inner_weight_gradient_linear(
            coords: Coords,
            node_index_in_elt: int,
        ):
            return 2.0 * wp.identity(n=3, dtype=float)

        if ORDER == 1:
            return cache.get_func(element_inner_weight_gradient_linear, self.name)

        return None
