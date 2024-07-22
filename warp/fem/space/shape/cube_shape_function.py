import math

import numpy as np

import warp as wp
from warp.fem import cache
from warp.fem.geometry import Grid3D
from warp.fem.polynomial import Polynomial, is_closed, lagrange_scales, quadrature_1d
from warp.fem.types import Coords

from .tet_shape_function import TetrahedronPolynomialShapeFunctions


class CubeTripolynomialShapeFunctions:
    VERTEX = 0
    EDGE = 1
    FACE = 2
    INTERIOR = 3

    def __init__(self, degree: int, family: Polynomial):
        self.family = family

        self.ORDER = wp.constant(degree)
        self.NODES_PER_ELEMENT = wp.constant((degree + 1) ** 3)
        self.NODES_PER_EDGE = wp.constant(degree + 1)

        lobatto_coords, lobatto_weight = quadrature_1d(point_count=degree + 1, family=family)
        lagrange_scale = lagrange_scales(lobatto_coords)

        NodeVec = wp.types.vector(length=degree + 1, dtype=wp.float32)
        self.LOBATTO_COORDS = wp.constant(NodeVec(lobatto_coords))
        self.LOBATTO_WEIGHT = wp.constant(NodeVec(lobatto_weight))
        self.LAGRANGE_SCALE = wp.constant(NodeVec(lagrange_scale))
        self.ORDER_PLUS_ONE = wp.constant(self.ORDER + 1)

        self._node_ijk = self._make_node_ijk()
        self.node_type_and_type_index = self._make_node_type_and_type_index()

    @property
    def name(self) -> str:
        return f"Cube_Q{self.ORDER}_{self.family}"

    @wp.func
    def _vertex_coords_f(vidx_in_cell: int):
        x = vidx_in_cell // 4
        y = (vidx_in_cell - 4 * x) // 2
        z = vidx_in_cell - 4 * x - 2 * y
        return wp.vec3(float(x), float(y), float(z))

    def _make_node_ijk(self):
        ORDER_PLUS_ONE = self.ORDER_PLUS_ONE

        def node_ijk(
            node_index_in_elt: int,
        ):
            node_i = node_index_in_elt // (ORDER_PLUS_ONE * ORDER_PLUS_ONE)
            node_jk = node_index_in_elt - ORDER_PLUS_ONE * ORDER_PLUS_ONE * node_i
            node_j = node_jk // ORDER_PLUS_ONE
            node_k = node_jk - ORDER_PLUS_ONE * node_j
            return node_i, node_j, node_k

        return cache.get_func(node_ijk, self.name)

    def _make_node_type_and_type_index(self):
        ORDER = self.ORDER

        @cache.dynamic_func(suffix=self.name)
        def node_type_and_type_index(
            node_index_in_elt: int,
        ):
            i, j, k = self._node_ijk(node_index_in_elt)

            zi = wp.select(i == 0, 0, 1)
            zj = wp.select(j == 0, 0, 1)
            zk = wp.select(k == 0, 0, 1)

            mi = wp.select(i == ORDER, 0, 1)
            mj = wp.select(j == ORDER, 0, 1)
            mk = wp.select(k == ORDER, 0, 1)

            if zi + mi == 1:
                if zj + mj == 1:
                    if zk + mk == 1:
                        # vertex
                        type_instance = mi * 4 + mj * 2 + mk
                        return CubeTripolynomialShapeFunctions.VERTEX, type_instance, 0

                    # z edge
                    type_instance = 8 + mi * 2 + mj
                    type_index = k - 1
                    return CubeTripolynomialShapeFunctions.EDGE, type_instance, type_index

                if zk + mk == 1:
                    # y edge
                    type_instance = 4 + mk * 2 + mi
                    type_index = j - 1
                    return CubeTripolynomialShapeFunctions.EDGE, type_instance, type_index

                # x face
                type_instance = mi
                type_index = (j - 1) * (ORDER - 1) + k - 1
                return CubeTripolynomialShapeFunctions.FACE, type_instance, type_index

            if zj + mj == 1:
                if zk + mk == 1:
                    # x edge
                    type_instance = mj * 2 + mk
                    type_index = i - 1
                    return CubeTripolynomialShapeFunctions.EDGE, type_instance, type_index

                # y face
                type_instance = 2 + mj
                type_index = (k - 1) * (ORDER - 1) + i - 1
                return CubeTripolynomialShapeFunctions.FACE, type_instance, type_index

            if zk + mk == 1:
                # z face
                type_instance = 4 + mk
                type_index = (i - 1) * (ORDER - 1) + j - 1
                return CubeTripolynomialShapeFunctions.FACE, type_instance, type_index

            type_index = ((i - 1) * (ORDER - 1) + (j - 1)) * (ORDER - 1) + k - 1
            return CubeTripolynomialShapeFunctions.INTERIOR, 0, type_index

        return node_type_and_type_index

    def make_node_coords_in_element(self):
        LOBATTO_COORDS = self.LOBATTO_COORDS

        @cache.dynamic_func(suffix=self.name)
        def node_coords_in_element(
            node_index_in_elt: int,
        ):
            node_i, node_j, node_k = self._node_ijk(node_index_in_elt)
            return Coords(LOBATTO_COORDS[node_i], LOBATTO_COORDS[node_j], LOBATTO_COORDS[node_k])

        return node_coords_in_element

    def make_node_quadrature_weight(self):
        ORDER = self.ORDER
        LOBATTO_WEIGHT = self.LOBATTO_WEIGHT

        def node_quadrature_weight(
            node_index_in_elt: int,
        ):
            node_i, node_j, node_k = self._node_ijk(node_index_in_elt)
            return LOBATTO_WEIGHT[node_i] * LOBATTO_WEIGHT[node_j] * LOBATTO_WEIGHT[node_k]

        def node_quadrature_weight_linear(
            node_index_in_elt: int,
        ):
            return 0.125

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
            # If we find one index at extremum, pick the two other

            node_i, node_j, node_k = self._node_ijk(node_index_in_elt)

            if node_i == 0 or node_i == ORDER:
                return LOBATTO_WEIGHT[node_j] * LOBATTO_WEIGHT[node_k]

            if node_j == 0 or node_j == ORDER:
                return LOBATTO_WEIGHT[node_i] * LOBATTO_WEIGHT[node_k]

            return LOBATTO_WEIGHT[node_i] * LOBATTO_WEIGHT[node_j]

        def trace_node_quadrature_weight_linear(
            node_index_in_elt: int,
        ):
            return 0.25

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
            node_i, node_j, node_k = self._node_ijk(node_index_in_elt)

            w = float(1.0)
            for k in range(ORDER_PLUS_ONE):
                if k != node_i:
                    w *= coords[0] - LOBATTO_COORDS[k]
                if k != node_j:
                    w *= coords[1] - LOBATTO_COORDS[k]
                if k != node_k:
                    w *= coords[2] - LOBATTO_COORDS[k]

            w *= LAGRANGE_SCALE[node_i] * LAGRANGE_SCALE[node_j] * LAGRANGE_SCALE[node_k]

            return w

        def element_inner_weight_linear(
            coords: Coords,
            node_index_in_elt: int,
        ):
            v = CubeTripolynomialShapeFunctions._vertex_coords_f(node_index_in_elt)

            wx = (1.0 - coords[0]) * (1.0 - v[0]) + v[0] * coords[0]
            wy = (1.0 - coords[1]) * (1.0 - v[1]) + v[1] * coords[1]
            wz = (1.0 - coords[2]) * (1.0 - v[2]) + v[2] * coords[2]
            return wx * wy * wz

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
            node_i, node_j, node_k = self._node_ijk(node_index_in_elt)

            prefix_xy = float(1.0)
            prefix_yz = float(1.0)
            prefix_zx = float(1.0)
            for k in range(ORDER_PLUS_ONE):
                if k != node_i:
                    prefix_yz *= coords[0] - LOBATTO_COORDS[k]
                if k != node_j:
                    prefix_zx *= coords[1] - LOBATTO_COORDS[k]
                if k != node_k:
                    prefix_xy *= coords[2] - LOBATTO_COORDS[k]

            prefix_x = prefix_zx * prefix_xy
            prefix_y = prefix_yz * prefix_xy
            prefix_z = prefix_zx * prefix_yz

            grad_x = float(0.0)
            grad_y = float(0.0)
            grad_z = float(0.0)

            for k in range(ORDER_PLUS_ONE):
                if k != node_i:
                    delta_x = coords[0] - LOBATTO_COORDS[k]
                    grad_x = grad_x * delta_x + prefix_x
                    prefix_x *= delta_x
                if k != node_j:
                    delta_y = coords[1] - LOBATTO_COORDS[k]
                    grad_y = grad_y * delta_y + prefix_y
                    prefix_y *= delta_y
                if k != node_k:
                    delta_z = coords[2] - LOBATTO_COORDS[k]
                    grad_z = grad_z * delta_z + prefix_z
                    prefix_z *= delta_z

            grad = (
                LAGRANGE_SCALE[node_i]
                * LAGRANGE_SCALE[node_j]
                * LAGRANGE_SCALE[node_k]
                * wp.vec3(
                    grad_x,
                    grad_y,
                    grad_z,
                )
            )

            return grad

        def element_inner_weight_gradient_linear(
            coords: Coords,
            node_index_in_elt: int,
        ):
            v = CubeTripolynomialShapeFunctions._vertex_coords_f(node_index_in_elt)

            wx = (1.0 - coords[0]) * (1.0 - v[0]) + v[0] * coords[0]
            wy = (1.0 - coords[1]) * (1.0 - v[1]) + v[1] * coords[1]
            wz = (1.0 - coords[2]) * (1.0 - v[2]) + v[2] * coords[2]

            dx = 2.0 * v[0] - 1.0
            dy = 2.0 * v[1] - 1.0
            dz = 2.0 * v[2] - 1.0

            return wp.vec3(dx * wy * wz, dy * wz * wx, dz * wx * wy)

        if self.ORDER == 1 and is_closed(self.family):
            return cache.get_func(element_inner_weight_gradient_linear, self.name)

        return cache.get_func(element_inner_weight_gradient, self.name)

    def element_node_hexes(self):
        from warp.fem.utils import grid_to_hexes

        return grid_to_hexes(self.ORDER, self.ORDER, self.ORDER)

    def element_node_tets(self):
        from warp.fem.utils import grid_to_tets

        return grid_to_tets(self.ORDER, self.ORDER, self.ORDER)

    def element_vtk_cells(self):
        n = self.ORDER + 1

        # vertices
        cells = [
            [
                [0, 0, 0],
                [n - 1, 0, 0],
                [n - 1, n - 1, 0],
                [0, n - 1, 0],
                [0, 0, n - 1],
                [n - 1, 0, n - 1],
                [n - 1, n - 1, n - 1],
                [0, n - 1, n - 1],
            ]
        ]

        if self.ORDER == 1:
            cell_type = 12  # vtk_hexahedron
        else:
            middle = np.arange(1, n - 1)
            front = np.zeros(n - 2, dtype=int)
            back = np.full(n - 2, n - 1)

            # edges
            cells.append(np.column_stack((middle, front, front)))
            cells.append(np.column_stack((back, middle, front)))
            cells.append(np.column_stack((middle, back, front)))
            cells.append(np.column_stack((front, middle, front)))

            cells.append(np.column_stack((middle, front, back)))
            cells.append(np.column_stack((back, middle, back)))
            cells.append(np.column_stack((middle, back, back)))
            cells.append(np.column_stack((front, middle, back)))

            cells.append(np.column_stack((front, front, middle)))
            cells.append(np.column_stack((back, front, middle)))
            cells.append(np.column_stack((back, back, middle)))
            cells.append(np.column_stack((front, back, middle)))

            # faces

            face = np.meshgrid(middle, middle)
            front = np.zeros((n - 2) ** 2, dtype=int)
            back = np.full((n - 2) ** 2, n - 1)

            # YZ
            cells.append(
                np.column_stack((front, face[0].flatten(), face[1].flatten())),
            )
            cells.append(
                np.column_stack((back, face[0].flatten(), face[1].flatten())),
            )
            # XZ
            cells.append(
                np.column_stack((face[0].flatten(), front, face[1].flatten())),
            )
            cells.append(
                np.column_stack((face[0].flatten(), back, face[1].flatten())),
            )
            # XY
            cells.append(
                np.column_stack((face[0].flatten(), face[1].flatten(), front)),
            )
            cells.append(
                np.column_stack((face[0].flatten(), face[1].flatten(), back)),
            )

            # interior
            interior = np.meshgrid(middle, middle, middle)
            cells.append(
                np.column_stack((interior[0].flatten(), interior[1].flatten(), interior[2].flatten())),
            )

            cell_type = 72  # vtk_lagrange_hexahedron

        cells = np.concatenate(cells)
        cell_indices = cells[:, 0] * n * n + cells[:, 1] * n + cells[:, 2]

        return cell_indices[np.newaxis, :], np.array([cell_type], dtype=np.int8)


class CubeSerendipityShapeFunctions:
    """
    Serendipity element ~ tensor product space without interior nodes
    Edge shape functions are usual Lagrange shape functions times a bilinear function in the normal directions
    Corner shape functions are trilinear shape functions times a function of (x^{d-1} + y^{d-1})
    """

    # Node categories
    VERTEX = wp.constant(0)
    EDGE_X = wp.constant(1)
    EDGE_Y = wp.constant(2)

    def __init__(self, degree: int, family: Polynomial):
        if not is_closed(family):
            raise ValueError("A closed polynomial family is required to define serendipity elements")

        if degree not in [2, 3]:
            raise NotImplementedError("Serendipity element only implemented for order 2 or 3")

        self.family = family

        self.ORDER = wp.constant(degree)
        self.NODES_PER_ELEMENT = wp.constant(8 + 12 * (degree - 1))
        self.NODES_PER_EDGE = wp.constant(degree + 1)

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
        return f"Cube_S{self.ORDER}_{self.family}"

    def _get_node_type_and_type_index(self):
        @cache.dynamic_func(suffix=self.name)
        def node_type_and_index(
            node_index_in_elt: int,
        ):
            if node_index_in_elt < 8:
                return CubeSerendipityShapeFunctions.VERTEX, node_index_in_elt

            type_index = (node_index_in_elt - 8) // 3
            side = node_index_in_elt - 8 - 3 * type_index
            return CubeSerendipityShapeFunctions.EDGE_X + side, type_index

        return node_type_and_index

    @wp.func
    def _vertex_coords(vidx_in_cell: int):
        x = vidx_in_cell // 4
        y = (vidx_in_cell - 4 * x) // 2
        z = vidx_in_cell - 4 * x - 2 * y
        return wp.vec3i(x, y, z)

    @wp.func
    def _edge_coords(type_index: int):
        index_in_side = type_index // 4
        side_offset = type_index - 4 * index_in_side
        return wp.vec3i(index_in_side + 1, side_offset // 2, side_offset & 1)

    @wp.func
    def _edge_axis(node_type: int):
        return node_type - CubeSerendipityShapeFunctions.EDGE_X

    @wp.func
    def _cube_edge_index(node_type: int, type_index: int):
        index_in_side = type_index // 4
        side_offset = type_index - 4 * index_in_side

        return 4 * (node_type - CubeSerendipityShapeFunctions.EDGE_X) + side_offset, index_in_side

    def _get_node_lobatto_indices(self):
        ORDER = self.ORDER

        @cache.dynamic_func(suffix=self.name)
        def node_lobatto_indices(node_type: int, type_index: int):
            if node_type == CubeSerendipityShapeFunctions.VERTEX:
                return CubeSerendipityShapeFunctions._vertex_coords(type_index) * ORDER

            axis = CubeSerendipityShapeFunctions._edge_axis(node_type)
            local_coords = CubeSerendipityShapeFunctions._edge_coords(type_index)

            local_indices = wp.vec3i(local_coords[0], local_coords[1] * ORDER, local_coords[2] * ORDER)

            return Grid3D._local_to_world(axis, local_indices)

        return node_lobatto_indices

    def make_node_coords_in_element(self):
        LOBATTO_COORDS = self.LOBATTO_COORDS

        @cache.dynamic_func(suffix=self.name)
        def node_coords_in_element(
            node_index_in_elt: int,
        ):
            node_type, type_index = self.node_type_and_type_index(node_index_in_elt)
            node_coords = self._node_lobatto_indices(node_type, type_index)
            return Coords(
                LOBATTO_COORDS[node_coords[0]], LOBATTO_COORDS[node_coords[1]], LOBATTO_COORDS[node_coords[2]]
            )

        return node_coords_in_element

    def make_node_quadrature_weight(self):
        ORDER = self.ORDER

        @cache.dynamic_func(suffix=self.name)
        def node_quadrature_weight(
            node_index_in_elt: int,
        ):
            node_type, type_index = self.node_type_and_type_index(node_index_in_elt)
            if node_type == CubeSerendipityShapeFunctions.VERTEX:
                return 1.0 / float(8 * ORDER * ORDER * ORDER)

            return (1.0 - 1.0 / float(ORDER * ORDER * ORDER)) / float(12 * (ORDER - 1))

        return node_quadrature_weight

    def make_trace_node_quadrature_weight(self):
        ORDER = self.ORDER

        @cache.dynamic_func(suffix=self.name)
        def trace_node_quadrature_weight(
            node_index_in_elt: int,
        ):
            node_type, type_index = self.node_type_and_type_index(node_index_in_elt)
            if node_type == CubeSerendipityShapeFunctions.VERTEX:
                return 0.25 / float(ORDER * ORDER)

            return (0.25 - 0.25 / float(ORDER * ORDER)) / float(ORDER - 1)

        return trace_node_quadrature_weight

    def make_element_inner_weight(self):
        ORDER = self.ORDER
        ORDER_PLUS_ONE = self.ORDER_PLUS_ONE

        LOBATTO_COORDS = self.LOBATTO_COORDS
        LAGRANGE_SCALE = self.LAGRANGE_SCALE

        DEGREE_3_SPHERE_RAD = wp.constant(2 * 0.5**2 + (0.5 - LOBATTO_COORDS[1]) ** 2)
        DEGREE_3_SPHERE_SCALE = 1.0 / (0.75 - DEGREE_3_SPHERE_RAD)

        @cache.dynamic_func(suffix=self.name)
        def element_inner_weight(
            coords: Coords,
            node_index_in_elt: int,
        ):
            node_type, type_index = self.node_type_and_type_index(node_index_in_elt)

            if node_type == CubeSerendipityShapeFunctions.VERTEX:
                node_ijk = CubeSerendipityShapeFunctions._vertex_coords(type_index)

                cx = wp.select(node_ijk[0] == 0, coords[0], 1.0 - coords[0])
                cy = wp.select(node_ijk[1] == 0, coords[1], 1.0 - coords[1])
                cz = wp.select(node_ijk[2] == 0, coords[2], 1.0 - coords[2])

                w = cx * cy * cz

                if ORDER == 2:
                    w *= cx + cy + cz - 3.0 + LOBATTO_COORDS[1]
                    return w * LAGRANGE_SCALE[0]
                if ORDER == 3:
                    w *= (
                        (cx - 0.5) * (cx - 0.5)
                        + (cy - 0.5) * (cy - 0.5)
                        + (cz - 0.5) * (cz - 0.5)
                        - DEGREE_3_SPHERE_RAD
                    )
                    return w * DEGREE_3_SPHERE_SCALE

            axis = CubeSerendipityShapeFunctions._edge_axis(node_type)

            node_all = CubeSerendipityShapeFunctions._edge_coords(type_index)

            local_coords = Grid3D._world_to_local(axis, coords)

            w = float(1.0)
            w *= wp.select(node_all[1] == 0, local_coords[1], 1.0 - local_coords[1])
            w *= wp.select(node_all[2] == 0, local_coords[2], 1.0 - local_coords[2])

            for k in range(ORDER_PLUS_ONE):
                if k != node_all[0]:
                    w *= local_coords[0] - LOBATTO_COORDS[k]
            w *= LAGRANGE_SCALE[node_all[0]]

            return w

        return element_inner_weight

    def make_element_inner_weight_gradient(self):
        ORDER = self.ORDER
        ORDER_PLUS_ONE = self.ORDER_PLUS_ONE
        LOBATTO_COORDS = self.LOBATTO_COORDS
        LAGRANGE_SCALE = self.LAGRANGE_SCALE

        DEGREE_3_SPHERE_RAD = wp.constant(2 * 0.5**2 + (0.5 - LOBATTO_COORDS[1]) ** 2)
        DEGREE_3_SPHERE_SCALE = 1.0 / (0.75 - DEGREE_3_SPHERE_RAD)

        @cache.dynamic_func(suffix=self.name)
        def element_inner_weight_gradient(
            coords: Coords,
            node_index_in_elt: int,
        ):
            node_type, type_index = self.node_type_and_type_index(node_index_in_elt)

            if node_type == CubeSerendipityShapeFunctions.VERTEX:
                node_ijk = CubeSerendipityShapeFunctions._vertex_coords(type_index)

                cx = wp.select(node_ijk[0] == 0, coords[0], 1.0 - coords[0])
                cy = wp.select(node_ijk[1] == 0, coords[1], 1.0 - coords[1])
                cz = wp.select(node_ijk[2] == 0, coords[2], 1.0 - coords[2])

                gx = wp.select(node_ijk[0] == 0, 1.0, -1.0)
                gy = wp.select(node_ijk[1] == 0, 1.0, -1.0)
                gz = wp.select(node_ijk[2] == 0, 1.0, -1.0)

                if ORDER == 2:
                    w = cx + cy + cz - 3.0 + LOBATTO_COORDS[1]
                    grad_x = cy * cz * gx * (w + cx)
                    grad_y = cz * cx * gy * (w + cy)
                    grad_z = cx * cy * gz * (w + cz)

                    return wp.vec3(grad_x, grad_y, grad_z) * LAGRANGE_SCALE[0]

                if ORDER == 3:
                    w = (
                        (cx - 0.5) * (cx - 0.5)
                        + (cy - 0.5) * (cy - 0.5)
                        + (cz - 0.5) * (cz - 0.5)
                        - DEGREE_3_SPHERE_RAD
                    )

                    dw_dcx = 2.0 * cx - 1.0
                    dw_dcy = 2.0 * cy - 1.0
                    dw_dcz = 2.0 * cz - 1.0
                    grad_x = cy * cz * gx * (w + dw_dcx * cx)
                    grad_y = cz * cx * gy * (w + dw_dcy * cy)
                    grad_z = cx * cy * gz * (w + dw_dcz * cz)

                    return wp.vec3(grad_x, grad_y, grad_z) * DEGREE_3_SPHERE_SCALE

            axis = CubeSerendipityShapeFunctions._edge_axis(node_type)
            node_all = CubeSerendipityShapeFunctions._edge_coords(type_index)

            local_coords = Grid3D._world_to_local(axis, coords)

            w_long = wp.select(node_all[1] == 0, local_coords[1], 1.0 - local_coords[1])
            w_lat = wp.select(node_all[2] == 0, local_coords[2], 1.0 - local_coords[2])

            g_long = wp.select(node_all[1] == 0, 1.0, -1.0)
            g_lat = wp.select(node_all[2] == 0, 1.0, -1.0)

            w_alt = LAGRANGE_SCALE[node_all[0]]
            g_alt = float(0.0)
            prefix_alt = LAGRANGE_SCALE[node_all[0]]
            for k in range(ORDER_PLUS_ONE):
                if k != node_all[0]:
                    delta_alt = local_coords[0] - LOBATTO_COORDS[k]
                    w_alt *= delta_alt
                    g_alt = g_alt * delta_alt + prefix_alt
                    prefix_alt *= delta_alt

            local_grad = wp.vec3(g_alt * w_long * w_lat, w_alt * g_long * w_lat, w_alt * w_long * g_lat)

            return Grid3D._local_to_world(axis, local_grad)

        return element_inner_weight_gradient

    def element_node_tets(self):
        from warp.fem.utils import grid_to_tets

        if self.ORDER == 2:
            element_tets = np.array(
                [
                    [0, 8, 9, 10],
                    [1, 11, 10, 15],
                    [2, 9, 14, 13],
                    [3, 15, 13, 17],
                    [4, 12, 8, 16],
                    [5, 18, 16, 11],
                    [6, 14, 12, 19],
                    [7, 19, 18, 17],
                    [16, 12, 18, 11],
                    [8, 16, 12, 11],
                    [12, 19, 18, 14],
                    [14, 19, 17, 18],
                    [10, 9, 15, 8],
                    [10, 8, 11, 15],
                    [9, 13, 15, 14],
                    [13, 14, 17, 15],
                ]
            )

            middle_hex = np.array([8, 11, 9, 15, 12, 18, 14, 17])
            middle_tets = middle_hex[grid_to_tets(1, 1, 1)]

            return np.concatenate((element_tets, middle_tets))

        raise NotImplementedError()

    def element_vtk_cells(self):
        tets = np.array(self.element_node_tets())
        cell_type = 10  # VTK_TETRA

        return tets, np.full(tets.shape[0], cell_type, dtype=np.int8)


class CubeNonConformingPolynomialShapeFunctions:
    # embeds the largest regular tet centered at (0.5, 0.5, 0.5) into the reference cube

    _tet_height = 2.0 / 3.0
    _tet_side = math.sqrt(3.0 / 2.0) * _tet_height
    _tet_face_height = math.sqrt(3.0) / 2.0 * _tet_side

    _tet_to_cube = np.array(
        [
            [_tet_side, _tet_side / 2.0, _tet_side / 2.0],
            [0.0, _tet_face_height, _tet_face_height / 3.0],
            [0.0, 0.0, _tet_height],
        ]
    )

    _TET_OFFSET = wp.constant(wp.vec3(0.5 - 0.5 * _tet_side, 0.5 - _tet_face_height / 3.0, 0.5 - 0.25 * _tet_height))

    def __init__(self, degree: int):
        self._tet_shape = TetrahedronPolynomialShapeFunctions(degree=degree)
        self.ORDER = self._tet_shape.ORDER
        self.NODES_PER_ELEMENT = self._tet_shape.NODES_PER_ELEMENT

        self.element_node_tets = self._tet_shape.element_node_tets
        self.element_vtk_cells = self._tet_shape.element_vtk_cells

    @property
    def name(self) -> str:
        return f"Cube_P{self.ORDER}d"

    def make_node_coords_in_element(self):
        node_coords_in_tet = self._tet_shape.make_node_coords_in_element()

        TET_TO_CUBE = wp.constant(wp.mat33(self._tet_to_cube))

        @cache.dynamic_func(suffix=self.name)
        def node_coords_in_element(
            node_index_in_elt: int,
        ):
            tet_coords = node_coords_in_tet(node_index_in_elt)
            return TET_TO_CUBE * tet_coords + CubeNonConformingPolynomialShapeFunctions._TET_OFFSET

        return node_coords_in_element

    def make_node_quadrature_weight(self):
        NODES_PER_ELEMENT = self.NODES_PER_ELEMENT

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
        tet_inner_weight = self._tet_shape.make_element_inner_weight()

        CUBE_TO_TET = wp.constant(wp.mat33(np.linalg.inv(self._tet_to_cube)))

        @cache.dynamic_func(suffix=self.name)
        def element_inner_weight(
            coords: Coords,
            node_index_in_elt: int,
        ):
            tet_coords = CUBE_TO_TET * (coords - CubeNonConformingPolynomialShapeFunctions._TET_OFFSET)

            return tet_inner_weight(tet_coords, node_index_in_elt)

        return element_inner_weight

    def make_element_inner_weight_gradient(self):
        tet_inner_weight_gradient = self._tet_shape.make_element_inner_weight_gradient()

        CUBE_TO_TET = wp.constant(wp.mat33(np.linalg.inv(self._tet_to_cube)))

        @cache.dynamic_func(suffix=self.name)
        def element_inner_weight_gradient(
            coords: Coords,
            node_index_in_elt: int,
        ):
            tet_coords = CUBE_TO_TET * (coords - CubeNonConformingPolynomialShapeFunctions._TET_OFFSET)
            grad = tet_inner_weight_gradient(tet_coords, node_index_in_elt)
            return wp.transpose(CUBE_TO_TET) * grad

        return element_inner_weight_gradient
