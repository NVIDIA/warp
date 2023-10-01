from typing import Optional
import warp as wp
import numpy as np


from warp.fem.types import ElementIndex, Coords, OUTSIDE
from warp.fem.geometry import Tetmesh
from warp.fem.cache import cached_arg_value

from .dof_mapper import DofMapper
from .nodal_function_space import NodalFunctionSpace, NodalFunctionSpaceTrace


class TetmeshFunctionSpace(NodalFunctionSpace):
    DIMENSION = wp.constant(3)

    @wp.struct
    class SpaceArg:
        geo_arg: Tetmesh.SideArg

        reference_transforms: wp.array(dtype=wp.mat33f)
        tet_edge_indices: wp.array2d(dtype=int)
        tet_face_indices: wp.array2d(dtype=int)

        vertex_count: int
        edge_count: int
        face_count: int

    def __init__(
        self,
        mesh: Tetmesh,
        dtype: type = float,
        dof_mapper: DofMapper = None,
    ):
        super().__init__(dtype, dof_mapper)
        self._mesh = mesh

        self._reference_transforms: wp.array = None
        self._tet_face_indices: wp.array = None

        self._compute_reference_transforms()
        self._compute_tet_face_indices()

    @property
    def geometry(self) -> Tetmesh:
        return self._mesh

    @cached_arg_value
    def space_arg_value(self, device):
        arg = self.SpaceArg()
        arg.geo_arg = self.geometry.side_arg_value(device)
        arg.reference_transforms = self._reference_transforms.to(device)
        arg.tet_face_indices = self._tet_face_indices.to(device)
        arg.tet_edge_indices = self._mesh._tet_edge_indices.to(device)

        arg.vertex_count = self._mesh.vertex_count()
        arg.face_count = self._mesh.side_count()
        arg.edge_count = self._mesh.edge_count()
        return arg

    class Trace(NodalFunctionSpaceTrace):
        def __init__(self, space: NodalFunctionSpace):
            super().__init__(space)
            self.ORDER = space.ORDER

    @wp.func
    def _inner_cell_index(args: SpaceArg, side_index: ElementIndex):
        return Tetmesh.side_inner_cell_index(args.geo_arg, side_index)

    @wp.func
    def _outer_cell_index(args: SpaceArg, side_index: ElementIndex):
        return Tetmesh.side_outer_cell_index(args.geo_arg, side_index)

    @wp.func
    def _inner_cell_coords(args: SpaceArg, side_index: ElementIndex, side_coords: Coords):
        tet_index = Tetmesh.side_inner_cell_index(args.geo_arg, side_index)
        return Tetmesh.face_to_tet_coords(args.geo_arg, side_index, tet_index, side_coords)

    @wp.func
    def _outer_cell_coords(args: SpaceArg, side_index: ElementIndex, side_coords: Coords):
        tet_index = Tetmesh.side_outer_cell_index(args.geo_arg, side_index)
        return Tetmesh.face_to_tet_coords(args.geo_arg, side_index, tet_index, side_coords)

    @wp.func
    def _cell_to_side_coords(
        args: SpaceArg,
        side_index: ElementIndex,
        element_index: ElementIndex,
        element_coords: Coords,
    ):
        return Tetmesh.tet_to_face_coords(args.geo_arg, side_index, element_index, element_coords)

    def _compute_reference_transforms(self):
        self._reference_transforms = wp.empty(
            dtype=wp.mat33f, device=self._mesh.positions.device, shape=(self._mesh.cell_count())
        )

        wp.launch(
            kernel=TetmeshFunctionSpace._compute_reference_transforms_kernel,
            dim=self._reference_transforms.shape,
            device=self._reference_transforms.device,
            inputs=[self._mesh.tet_vertex_indices, self._mesh.positions, self._reference_transforms],
        )

    def _compute_tet_face_indices(self):
        self._tet_face_indices = wp.empty(
            dtype=int, device=self._mesh.tet_vertex_indices.device, shape=(self._mesh.cell_count(), 4)
        )

        wp.launch(
            kernel=TetmeshFunctionSpace._compute_tet_face_indices_kernel,
            dim=self._mesh._face_tet_indices.shape,
            device=self._mesh.tet_vertex_indices.device,
            inputs=[
                self._mesh._face_tet_indices,
                self._mesh._face_vertex_indices,
                self._mesh.tet_vertex_indices,
                self._tet_face_indices,
            ],
        )

    @wp.kernel
    def _compute_reference_transforms_kernel(
        tet_vertex_indices: wp.array2d(dtype=int),
        positions: wp.array(dtype=wp.vec3f),
        transforms: wp.array(dtype=wp.mat33f),
    ):
        t = wp.tid()

        p0 = positions[tet_vertex_indices[t, 0]]
        p1 = positions[tet_vertex_indices[t, 1]]
        p2 = positions[tet_vertex_indices[t, 2]]
        p3 = positions[tet_vertex_indices[t, 3]]

        e1 = p1 - p0
        e2 = p2 - p0
        e3 = p3 - p0

        mat = wp.mat33(e1, e2, e3)
        transforms[t] = wp.transpose(wp.inverse(mat))

    @wp.func
    def _find_face_index_in_tet(
        face_vtx: wp.vec3i,
        tet_vtx: wp.vec4i,
    ):
        for k in range(3):
            tvk = wp.vec3i(tet_vtx[k], tet_vtx[(k + 1) % 4], tet_vtx[(k + 2) % 4])

            # Use fact that face always start with min vertex
            min_t = wp.min(tvk)
            max_t = wp.max(tvk)
            mid_t = tvk[0] + tvk[1] + tvk[2] - min_t - max_t

            if min_t == face_vtx[0] and (
                (face_vtx[2] == max_t and face_vtx[1] == mid_t) or (face_vtx[1] == max_t and face_vtx[2] == mid_t)
            ):
                return k

        return 3

    @wp.kernel
    def _compute_tet_face_indices_kernel(
        face_tet_indices: wp.array(dtype=wp.vec2i),
        face_vertex_indices: wp.array(dtype=wp.vec3i),
        tet_vertex_indices: wp.array2d(dtype=int),
        tet_face_indices: wp.array2d(dtype=int),
    ):
        e = wp.tid()

        face_vtx = face_vertex_indices[e]
        face_tets = face_tet_indices[e]

        t0 = face_tets[0]
        t0_vtx = wp.vec4i(
            tet_vertex_indices[t0, 0], tet_vertex_indices[t0, 1], tet_vertex_indices[t0, 2], tet_vertex_indices[t0, 3]
        )
        t0_face = TetmeshFunctionSpace._find_face_index_in_tet(face_vtx, t0_vtx)
        tet_face_indices[t0, t0_face] = e

        t1 = face_tets[1]
        if t1 != t0:
            t1_vtx = wp.vec4i(
                tet_vertex_indices[t1, 0],
                tet_vertex_indices[t1, 1],
                tet_vertex_indices[t1, 2],
                tet_vertex_indices[t1, 3],
            )
            t1_face = TetmeshFunctionSpace._find_face_index_in_tet(face_vtx, t1_vtx)
            tet_face_indices[t1, t1_face] = e


class TetmeshPiecewiseConstantSpace(TetmeshFunctionSpace):
    ORDER = wp.constant(0)
    NODES_PER_ELEMENT = wp.constant(1)

    def __init__(self, grid: Tetmesh, dtype: type = float, dof_mapper: DofMapper = None):
        super().__init__(grid, dtype, dof_mapper)

        self.element_outer_weight = self.element_inner_weight
        self.element_outer_weight_gradient = self.element_inner_weight_gradient

    def node_count(self) -> int:
        return self._mesh.cell_count()

    def node_positions(self):
        vtx_pos = self._mesh.positions.numpy()
        tet_vtx = self._mesh.tet_vertex_indices.numpy()

        tet_pos = vtx_pos[tet_vtx]
        centers = tet_pos.sum(axis=1) / 4.0

        return centers[:, 0], centers[:, 1], centers[:, 2]

    @wp.func
    def element_node_index(
        args: TetmeshFunctionSpace.SpaceArg,
        element_index: ElementIndex,
        node_index_in_elt: int,
    ):
        return element_index

    @wp.func
    def node_coords_in_element(
        args: TetmeshFunctionSpace.SpaceArg,
        element_index: ElementIndex,
        node_index_in_elt: int,
    ):
        if node_index_in_elt == 0:
            return Coords(1.0 / 4.0, 1.0 / 4.0, 1.0 / 4.0)

        return Coords(OUTSIDE)

    @wp.func
    def node_quadrature_weight(
        args: TetmeshFunctionSpace.SpaceArg,
        element_index: ElementIndex,
        node_index_in_elt: int,
    ):
        return 1.0

    @wp.func
    def element_inner_weight(
        args: TetmeshFunctionSpace.SpaceArg,
        element_index: ElementIndex,
        coords: Coords,
        node_index_in_elt: int,
    ):
        if node_index_in_elt == 0:
            return 1.0
        return 0.0

    @wp.func
    def element_inner_weight_gradient(
        args: TetmeshFunctionSpace.SpaceArg,
        element_index: ElementIndex,
        coords: Coords,
        node_index_in_elt: int,
    ):
        return wp.vec3(0.0)

    class Trace(TetmeshFunctionSpace.Trace):
        NODES_PER_ELEMENT = wp.constant(2)
        ORDER = wp.constant(0)

        def __init__(self, space: "TetmeshPiecewiseConstantSpace"):
            super().__init__(space)

            self.element_node_index = self._make_element_node_index(space)

            self.element_inner_weight = self._make_element_inner_weight(space)
            self.element_inner_weight_gradient = self._make_element_inner_weight_gradient(space)

            self.element_outer_weight = self._make_element_outer_weight(space)
            self.element_outer_weight_gradient = self._make_element_outer_weight_gradient(space)

        @wp.func
        def node_coords_in_element(
            args: TetmeshFunctionSpace.SpaceArg,
            element_index: ElementIndex,
            node_index_in_element: int,
        ):
            if node_index_in_element == 0:
                return Coords(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
            elif node_index_in_element == 1:
                return Coords(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

            return Coords(OUTSIDE)

        @wp.func
        def node_quadrature_weight(
            args: TetmeshFunctionSpace.SpaceArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            return 1.0

    def trace(self):
        return TetmeshPiecewiseConstantSpace.Trace(self)


def _tet_node_index(tx: int, ty: int, tz: int, degree: int):
    from .trimesh_2d_function_space import _triangle_node_index

    VERTEX_NODE_COUNT = 4
    EDGE_INTERIOR_NODE_COUNT = degree - 1
    VERTEX_EDGE_NODE_COUNT = VERTEX_NODE_COUNT + 6 * EDGE_INTERIOR_NODE_COUNT
    FACE_INTERIOR_NODE_COUNT = (degree - 1) * (degree - 2) // 2
    VERTEX_EDGE_FACE_NODE_COUNT = VERTEX_EDGE_NODE_COUNT + 4 * FACE_INTERIOR_NODE_COUNT

    # Index in similar order to e.g. VTK
    # First vertices, then edges (counterclokwise), then faces, then interior points (recursively)

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
                + _triangle_node_index(degree - 2 - ty, tz - 1, degree - 2)
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
                + _triangle_node_index(tx - 1, tz - 1, degree - 2)
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
                + _triangle_node_index(tx - 1, ty - 1, degree - 2)
            )
    elif tx + ty + tz == degree:
        # 1-2-3 face
        face_index = 1
        return (
            VERTEX_EDGE_NODE_COUNT
            + FACE_INTERIOR_NODE_COUNT * face_index
            + _triangle_node_index(tx - 1, tz - 1, degree - 2)
        )

    return VERTEX_EDGE_FACE_NODE_COUNT + _tet_node_index(tx - 1, ty - 1, tz - 1, degree - 3)


class TetmeshPolynomialShapeFunctions:
    INVALID = wp.constant(-1)
    VERTEX = wp.constant(0)
    EDGE = wp.constant(1)
    FACE = wp.constant(2)
    INTERIOR = wp.constant(3)

    def __init__(self, degree: int):
        self.ORDER = wp.constant(degree)

        self.NODES_PER_ELEMENT = wp.constant((degree + 1) * (degree + 2) * (degree + 3) // 6)
        self.NODES_PER_SIDE = wp.constant((degree + 1) * (degree + 2) // 2)

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
        return f"{self.ORDER}"

    def _get_node_tet_coordinates(self):
        NODE_TET_COORDS = self.NODE_TET_COORDS

        def node_tet_coordinates(
            node_index_in_elt: int,
        ):
            return wp.vec3i(
                NODE_TET_COORDS[node_index_in_elt, 0],
                NODE_TET_COORDS[node_index_in_elt, 1],
                NODE_TET_COORDS[node_index_in_elt, 2],
            )

        from warp.fem import cache

        return cache.get_func(node_tet_coordinates, self.name)

    def _get_node_type_and_type_index(self):
        ORDER = self.ORDER
        NODES_PER_ELEMENT = self.NODES_PER_ELEMENT

        def node_type_and_index(
            node_index_in_elt: int,
        ):
            if node_index_in_elt < 0 or node_index_in_elt >= NODES_PER_ELEMENT:
                return TetmeshPolynomialShapeFunctions.INVALID, TetmeshPolynomialShapeFunctions.INVALID

            if node_index_in_elt < 4:
                return TetmeshPolynomialShapeFunctions.VERTEX, node_index_in_elt

            if node_index_in_elt < (6 * ORDER - 2):
                return TetmeshPolynomialShapeFunctions.EDGE, (node_index_in_elt - 4)

            if node_index_in_elt < (2 * ORDER * ORDER + 2):
                return TetmeshPolynomialShapeFunctions.FACE, (node_index_in_elt - (6 * ORDER - 2))

            return TetmeshPolynomialShapeFunctions.INTERIOR, (node_index_in_elt - (2 * ORDER * ORDER + 2))

        from warp.fem import cache

        return cache.get_func(node_type_and_index, self.name)

    def make_node_coords_in_element(self):
        ORDER = self.ORDER

        def node_coords_in_element(
            args: TetmeshFunctionSpace.SpaceArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            tet_coords = self._node_tet_coordinates(node_index_in_elt)
            cx = float(tet_coords[0]) / float(ORDER)
            cy = float(tet_coords[1]) / float(ORDER)
            cz = float(tet_coords[2]) / float(ORDER)
            return Coords(cx, cy, cz)

        from warp.fem import cache

        return cache.get_func(node_coords_in_element, self.name)

    def make_node_quadrature_weight(self):
        ORDER = self.ORDER

        def node_uniform_quadrature_weight(
            args: TetmeshFunctionSpace.SpaceArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            node_type, type_index = self.node_type_and_type_index(node_index_in_elt)

            base_weight = 1.0 / float(4 * ORDER * ORDER * ORDER)
            if node_type == TetmeshPolynomialShapeFunctions.VERTEX:
                return base_weight
            if node_type == TetmeshPolynomialShapeFunctions.EDGE:
                return 2.0 * base_weight
            if node_type == TetmeshPolynomialShapeFunctions.FACE:
                return 4.0 * base_weight
            return 8.0 * base_weight

        def node_linear_quadrature_weight(
            args: TetmeshFunctionSpace.SpaceArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            return 1.0 / 4.0

        from warp.fem import cache

        if ORDER == 1:
            return cache.get_func(node_linear_quadrature_weight, self.name)
        return cache.get_func(node_uniform_quadrature_weight, self.name)

    def make_trace_node_quadrature_weight(self):
        ORDER = self.ORDER
        NODES_PER_ELEMENT = self.NODES_PER_ELEMENT

        def trace_uniform_node_quadrature_weight(
            args: TetmeshFunctionSpace.SpaceArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            if node_index_in_elt >= NODES_PER_ELEMENT:
                node_index_in_cell = node_index_in_elt - NODES_PER_ELEMENT
            else:
                node_index_in_cell = node_index_in_elt

            # We're either on a side interior or at a vertex
            node_type, type_index = self.node_type_and_type_index(node_index_in_cell)

            base_weight = 1.0 / float(3 * ORDER * ORDER)
            if node_type == TetmeshPolynomialShapeFunctions.VERTEX:
                return base_weight
            if node_type == TetmeshPolynomialShapeFunctions.EDGE:
                return 2.0 * base_weight

            return 4.0 * base_weight

        def trace_linear_node_quadrature_weight(
            args: TetmeshFunctionSpace.SpaceArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            return 1.0 / 3.0

        from warp.fem import cache

        if ORDER == 1:
            return cache.get_func(trace_linear_node_quadrature_weight, self.name)

        return cache.get_func(trace_uniform_node_quadrature_weight, self.name)

    def make_element_inner_weight(self):
        ORDER = self.ORDER

        def element_inner_weight_linear(
            args: TetmeshFunctionSpace.SpaceArg,
            element_index: ElementIndex,
            coords: Coords,
            node_index_in_elt: int,
        ):
            if node_index_in_elt < 0 or node_index_in_elt >= 4:
                return 0.0

            tet_coords = wp.vec4(1.0 - coords[0] - coords[1] - coords[2], coords[0], coords[1], coords[2])
            return tet_coords[node_index_in_elt]

        def element_inner_weight_quadratic(
            args: TetmeshFunctionSpace.SpaceArg,
            element_index: ElementIndex,
            coords: Coords,
            node_index_in_elt: int,
        ):
            node_type, type_index = self.node_type_and_type_index(node_index_in_elt)

            tet_coords = wp.vec4(1.0 - coords[0] - coords[1] - coords[2], coords[0], coords[1], coords[2])

            if node_type == TetmeshPolynomialShapeFunctions.VERTEX:
                # Vertex
                return tet_coords[type_index] * (2.0 * tet_coords[type_index] - 1.0)

            elif node_type == TetmeshPolynomialShapeFunctions.EDGE:
                # Edge
                if type_index < 3:
                    c1 = type_index
                    c2 = (type_index + 1) % 3
                else:
                    c1 = type_index - 3
                    c2 = 3
                return 4.0 * tet_coords[c1] * tet_coords[c2]

            return 0.0

        def element_inner_weight_cubic(
            args: TetmeshFunctionSpace.SpaceArg,
            element_index: ElementIndex,
            coords: Coords,
            node_index_in_elt: int,
        ):
            node_type, type_index = self.node_type_and_type_index(node_index_in_elt)

            tet_coords = wp.vec4(1.0 - coords[0] - coords[1] - coords[2], coords[0], coords[1], coords[2])

            if node_type == TetmeshPolynomialShapeFunctions.VERTEX:
                # Vertex
                return (
                    0.5
                    * tet_coords[type_index]
                    * (3.0 * tet_coords[type_index] - 1.0)
                    * (3.0 * tet_coords[type_index] - 2.0)
                )

            elif node_type == TetmeshPolynomialShapeFunctions.EDGE:
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

            elif node_type == TetmeshPolynomialShapeFunctions.FACE:
                # Interior
                c1 = type_index
                c2 = (c1 + 1) % 4
                c3 = (c1 + 2) % 4
                return 27.0 * tet_coords[c1] * tet_coords[c2] * tet_coords[c3]

            return 0.0

        from warp.fem import cache

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
            args: TetmeshFunctionSpace.SpaceArg,
            element_index: ElementIndex,
            coords: Coords,
            node_index_in_elt: int,
        ):
            if node_index_in_elt < 0 or node_index_in_elt >= 4:
                return wp.vec3(0.0)

            dw_dc = wp.vec4(0.0)
            dw_dc[node_index_in_elt] = 1.0

            dw_du = wp.vec3(dw_dc[1] - dw_dc[0], dw_dc[2] - dw_dc[0], dw_dc[3] - dw_dc[0])

            return args.reference_transforms[element_index] * dw_du

        def element_inner_weight_gradient_quadratic(
            args: TetmeshFunctionSpace.SpaceArg,
            element_index: ElementIndex,
            coords: Coords,
            node_index_in_elt: int,
        ):
            node_type, type_index = self.node_type_and_type_index(node_index_in_elt)

            tet_coords = wp.vec4(1.0 - coords[0] - coords[1] - coords[2], coords[0], coords[1], coords[2])
            dw_dc = wp.vec4(0.0)

            if node_type == TetmeshPolynomialShapeFunctions.VERTEX:
                # Vertex
                dw_dc[type_index] = 4.0 * tet_coords[type_index] - 1.0

            elif node_type == TetmeshPolynomialShapeFunctions.EDGE:
                # Edge
                if type_index < 3:
                    c1 = type_index
                    c2 = (type_index + 1) % 3
                else:
                    c1 = type_index - 3
                    c2 = 3
                dw_dc[c1] = 4.0 * tet_coords[c2]
                dw_dc[c2] = 4.0 * tet_coords[c1]

            dw_du = wp.vec3(dw_dc[1] - dw_dc[0], dw_dc[2] - dw_dc[0], dw_dc[3] - dw_dc[0])
            return args.reference_transforms[element_index] * dw_du

        def element_inner_weight_gradient_cubic(
            args: TetmeshFunctionSpace.SpaceArg,
            element_index: ElementIndex,
            coords: Coords,
            node_index_in_elt: int,
        ):
            node_type, type_index = self.node_type_and_type_index(node_index_in_elt)

            tet_coords = wp.vec4(1.0 - coords[0] - coords[1] - coords[2], coords[0], coords[1], coords[2])

            dw_dc = wp.vec4(0.0)

            if node_type == TetmeshPolynomialShapeFunctions.VERTEX:
                # Vertex
                dw_dc[type_index] = (
                    0.5 * 27.0 * tet_coords[type_index] * tet_coords[type_index] - 9.0 * tet_coords[type_index] + 1.0
                )

            elif node_type == TetmeshPolynomialShapeFunctions.EDGE:
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

            elif node_type == TetmeshPolynomialShapeFunctions.FACE:
                # Interior
                c1 = type_index
                c2 = (c1 + 1) % 4
                c3 = (c1 + 2) % 4

                dw_dc[c1] = 27.0 * tet_coords[c2] * tet_coords[c3]
                dw_dc[c2] = 27.0 * tet_coords[c3] * tet_coords[c1]
                dw_dc[c3] = 27.0 * tet_coords[c1] * tet_coords[c2]

            dw_du = wp.vec3(dw_dc[1] - dw_dc[0], dw_dc[2] - dw_dc[0], dw_dc[3] - dw_dc[0])
            return args.reference_transforms[element_index] * dw_du

        from warp.fem import cache

        if ORDER == 1:
            return cache.get_func(element_inner_weight_gradient_linear, self.name)
        elif ORDER == 2:
            return cache.get_func(element_inner_weight_gradient_quadratic, self.name)
        elif ORDER == 3:
            return cache.get_func(element_inner_weight_gradient_cubic, self.name)

        return None

    @staticmethod
    def node_positions(space):
        if space.ORDER == 1:
            node_positions = space._mesh.positions.numpy()
            return node_positions[:, 0], node_positions[:, 1], node_positions[:, 2]

        NODES_PER_ELEMENT = space.NODES_PER_ELEMENT

        def fill_node_positions_fn(
            space_arg: space.SpaceArg,
            node_positions: wp.array(dtype=wp.vec3),
        ):
            element_index = wp.tid()
            tet_idx = space_arg.geo_arg.tet_vertex_indices[element_index]
            p0 = space_arg.geo_arg.positions[tet_idx[0]]
            p1 = space_arg.geo_arg.positions[tet_idx[1]]
            p2 = space_arg.geo_arg.positions[tet_idx[2]]
            p3 = space_arg.geo_arg.positions[tet_idx[3]]

            for n in range(NODES_PER_ELEMENT):
                node_index = space.element_node_index(space_arg, element_index, n)
                coords = space.node_coords_in_element(space_arg, element_index, n)

                pos = p0 + coords[0] * (p1 - p0) + coords[1] * (p2 - p0) + coords[2] * (p3 - p0)

                node_positions[node_index] = pos

        from warp.fem import cache

        fill_node_positions = cache.get_kernel(
            fill_node_positions_fn,
            suffix=space.name,
        )

        device = space._mesh.tet_vertex_indices.device
        node_positions = wp.empty(
            shape=space.node_count(),
            dtype=wp.vec3,
            device=device,
        )
        wp.launch(
            dim=space._mesh.cell_count(),
            kernel=fill_node_positions,
            inputs=[
                space.space_arg_value(device),
                node_positions,
            ],
            device=device,
        )

        node_positions = node_positions.numpy()
        return node_positions[:, 0], node_positions[:, 1], node_positions[:, 2]


class TetmeshPolynomialSpace(TetmeshFunctionSpace):
    def __init__(self, grid: Tetmesh, degree: int, dtype: type = float, dof_mapper: DofMapper = None):
        super().__init__(grid, dtype, dof_mapper)

        self._shape = TetmeshPolynomialShapeFunctions(degree)

        self.ORDER = self._shape.ORDER
        self.NODES_PER_ELEMENT = self._shape.NODES_PER_ELEMENT

        self.element_node_index = self._make_element_node_index()
        self.node_coords_in_element = self._shape.make_node_coords_in_element()
        self.node_quadrature_weight = self._shape.make_node_quadrature_weight()
        self.element_inner_weight = self._shape.make_element_inner_weight()
        self.element_inner_weight_gradient = self._shape.make_element_inner_weight_gradient()

        self.element_outer_weight = self.element_inner_weight
        self.element_outer_weight_gradient = self.element_inner_weight_gradient

    def _make_element_node_index(self):
        INTERIOR_NODES_PER_EDGE = wp.constant(max(0, self.ORDER - 1))
        INTERIOR_NODES_PER_FACE = wp.constant(max(0, self.ORDER - 2) * max(0, self.ORDER - 1) // 2)
        INTERIOR_NODES_PER_CELL = wp.constant(
            max(0, self.ORDER - 3) * max(0, self.ORDER - 2) * max(0, self.ORDER - 1) // 6
        )

        def element_node_index(
            args: TetmeshFunctionSpace.SpaceArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            node_type, type_index = self._shape.node_type_and_type_index(node_index_in_elt)

            if node_type == TetmeshPolynomialShapeFunctions.VERTEX:
                return args.geo_arg.tet_vertex_indices[element_index][type_index]

            global_offset = args.vertex_count

            if node_type == TetmeshPolynomialShapeFunctions.EDGE:
                edge = type_index // INTERIOR_NODES_PER_EDGE
                edge_node = type_index - INTERIOR_NODES_PER_EDGE * edge

                global_edge_index = args.tet_edge_indices[element_index][edge]

                # Test if we need to swap edge direction
                if INTERIOR_NODES_PER_EDGE > 1:
                    if edge < 3:
                        c1 = edge
                        c2 = (edge + 1) % 3
                    else:
                        c1 = edge - 3
                        c2 = 3

                    if (
                        args.geo_arg.tet_vertex_indices[element_index][c1]
                        > args.geo_arg.tet_vertex_indices[element_index][c2]
                    ):
                        edge_node = INTERIOR_NODES_PER_EDGE - 1 - edge_node

                return global_offset + INTERIOR_NODES_PER_EDGE * global_edge_index + edge_node

            global_offset += INTERIOR_NODES_PER_EDGE * args.edge_count

            if node_type == TetmeshPolynomialShapeFunctions.FACE:
                face = type_index // INTERIOR_NODES_PER_FACE
                face_node = type_index - INTERIOR_NODES_PER_FACE * face

                global_face_index = args.tet_face_indices[element_index][face]

                if INTERIOR_NODES_PER_FACE == 3:
                    # Hard code for P4 case, 3 nodes per face
                    # Higher orders would require rotating triangle coordinates, this is not supported yet

                    vidx = args.geo_arg.tet_vertex_indices[element_index][(face + face_node) % 4]
                    fvi = args.geo_arg.face_vertex_indices[global_face_index]

                    if vidx == fvi[0]:
                        face_node = 0
                    elif vidx == fvi[1]:
                        face_node = 1
                    else:
                        face_node = 2

                return global_offset + INTERIOR_NODES_PER_FACE * global_face_index + face_node

            global_offset += INTERIOR_NODES_PER_FACE * args.face_count

            return global_offset + INTERIOR_NODES_PER_CELL * element_index + type_index

        from warp.fem import cache

        return cache.get_func(element_node_index, self.name)

    def node_count(self) -> int:
        INTERIOR_NODES_PER_EDGE = wp.constant(max(0, self.ORDER - 1))
        INTERIOR_NODES_PER_FACE = wp.constant(max(0, self.ORDER - 2) * max(0, self.ORDER - 1) // 2)
        INTERIOR_NODES_PER_CELL = wp.constant(
            max(0, self.ORDER - 3) * max(0, self.ORDER - 2) * max(0, self.ORDER - 1) // 6
        )

        return (
            self._mesh.vertex_count()
            + self._mesh.edge_count() * INTERIOR_NODES_PER_EDGE
            + self._mesh.side_count() * INTERIOR_NODES_PER_FACE
            + self._mesh.cell_count() * INTERIOR_NODES_PER_CELL
        )

    def node_positions(self):
        return TetmeshPolynomialShapeFunctions.node_positions(self)

    class Trace(TetmeshFunctionSpace.Trace):
        NODES_PER_ELEMENT = wp.constant(2)
        ORDER = wp.constant(0)

        def __init__(self, space: "TetmeshPolynomialSpace"):
            super().__init__(space)

            self.element_node_index = self._make_element_node_index(space)
            self.node_coords_in_element = self._make_node_coords_in_element(space)
            self.node_quadrature_weight = space._shape.make_trace_node_quadrature_weight()

            self.element_inner_weight = self._make_element_inner_weight(space)
            self.element_inner_weight_gradient = self._make_element_inner_weight_gradient(space)

            self.element_outer_weight = self._make_element_outer_weight(space)
            self.element_outer_weight_gradient = self._make_element_outer_weight_gradient(space)

    def trace(self):
        return TetmeshPolynomialSpace.Trace(self)


class TetmeshDGPolynomialSpace(TetmeshFunctionSpace):
    def __init__(
        self,
        mesh: Tetmesh,
        degree: int,
        dtype: type = float,
        dof_mapper: DofMapper = None,
    ):
        super().__init__(mesh, dtype, dof_mapper)

        self._shape = TetmeshPolynomialShapeFunctions(degree)

        self.ORDER = self._shape.ORDER
        self.NODES_PER_ELEMENT = self._shape.NODES_PER_ELEMENT

        self.element_node_index = self._make_element_node_index()
        self.node_coords_in_element = self._shape.make_node_coords_in_element()
        self.node_quadrature_weight = self._shape.make_node_quadrature_weight()
        self.element_inner_weight = self._shape.make_element_inner_weight()
        self.element_inner_weight_gradient = self._shape.make_element_inner_weight_gradient()

        self.element_outer_weight = self.element_inner_weight
        self.element_outer_weight_gradient = self.element_inner_weight_gradient

    def node_count(self) -> int:
        return self._mesh.cell_count() * self.NODES_PER_ELEMENT

    def node_positions(self):
        return TetmeshPolynomialShapeFunctions.node_positions(self)

    def node_triangulation(self):
        return TetmeshPolynomialShapeFunctions.node_triangulation(self)

    def _make_element_node_index(self):
        NODES_PER_ELEMENT = self.NODES_PER_ELEMENT

        def element_node_index(
            args: TetmeshFunctionSpace.SpaceArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            return element_index * NODES_PER_ELEMENT + node_index_in_elt

        from warp.fem import cache

        return cache.get_func(element_node_index, f"{self.name}_{self.ORDER}")

    class Trace(TetmeshPolynomialSpace.Trace):
        def __init__(self, space: "TetmeshDGPolynomialSpace"):
            super().__init__(space)

    def trace(self):
        return TetmeshDGPolynomialSpace.Trace(self)
