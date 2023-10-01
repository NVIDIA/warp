import warp as wp
import numpy as np


from warp.fem.types import ElementIndex, Coords, OUTSIDE
from warp.fem.geometry import Trimesh2D
from warp.fem.cache import cached_arg_value

from .dof_mapper import DofMapper
from .nodal_function_space import NodalFunctionSpace, NodalFunctionSpaceTrace


class Trimesh2DFunctionSpace(NodalFunctionSpace):
    DIMENSION = wp.constant(2)

    @wp.struct
    class SpaceArg:
        geo_arg: Trimesh2D.SideArg

        reference_transforms: wp.array(dtype=wp.mat22f)
        tri_edge_indices: wp.array2d(dtype=int)

        vertex_count: int
        edge_count: int

    def __init__(self, mesh: Trimesh2D, dtype: type = float, dof_mapper: DofMapper = None):
        super().__init__(dtype, dof_mapper)
        self._mesh = mesh

        self._reference_transforms: wp.array = None
        self._tri_edge_indices: wp.array = None

        self._compute_reference_transforms()
        self._compute_tri_edge_indices()

    @property
    def geometry(self) -> Trimesh2D:
        return self._mesh

    @cached_arg_value
    def space_arg_value(self, device):
        arg = self.SpaceArg()
        arg.geo_arg = self.geometry.side_arg_value(device)
        arg.reference_transforms = self._reference_transforms.to(device)
        arg.tri_edge_indices = self._tri_edge_indices.to(device)

        arg.vertex_count = self._mesh.vertex_count()
        arg.edge_count = self._mesh.side_count()
        return arg

    class Trace(NodalFunctionSpaceTrace):
        def __init__(self, space: NodalFunctionSpace):
            super().__init__(space)
            self.ORDER = space.ORDER

    @wp.func
    def _inner_cell_index(args: SpaceArg, side_index: ElementIndex):
        return Trimesh2D.side_inner_cell_index(args.geo_arg, side_index)

    @wp.func
    def _outer_cell_index(args: SpaceArg, side_index: ElementIndex):
        return Trimesh2D.side_outer_cell_index(args.geo_arg, side_index)

    @wp.func
    def _inner_cell_coords(args: SpaceArg, side_index: ElementIndex, side_coords: Coords):
        tri_index = Trimesh2D.side_inner_cell_index(args.geo_arg, side_index)
        return Trimesh2D.edge_to_tri_coords(args.geo_arg, side_index, tri_index, side_coords)

    @wp.func
    def _outer_cell_coords(args: SpaceArg, side_index: ElementIndex, side_coords: Coords):
        tri_index = Trimesh2D.side_outer_cell_index(args.geo_arg, side_index)
        return Trimesh2D.edge_to_tri_coords(args.geo_arg, side_index, tri_index, side_coords)

    @wp.func
    def _cell_to_side_coords(
        args: SpaceArg,
        side_index: ElementIndex,
        element_index: ElementIndex,
        element_coords: Coords,
    ):
        return Trimesh2D.tri_to_edge_coords(args.geo_arg, side_index, element_index, element_coords)

    def _compute_reference_transforms(self):
        self._reference_transforms = wp.empty(
            dtype=wp.mat22f, device=self._mesh.positions.device, shape=(self._mesh.cell_count())
        )

        wp.launch(
            kernel=Trimesh2DFunctionSpace._compute_reference_transforms_kernel,
            dim=self._reference_transforms.shape,
            device=self._reference_transforms.device,
            inputs=[self._mesh.tri_vertex_indices, self._mesh.positions, self._reference_transforms],
        )

    def _compute_tri_edge_indices(self):
        self._tri_edge_indices = wp.empty(
            dtype=int, device=self._mesh.tri_vertex_indices.device, shape=(self._mesh.cell_count(), 3)
        )

        wp.launch(
            kernel=Trimesh2DFunctionSpace._compute_tri_edge_indices_kernel,
            dim=self._mesh._edge_tri_indices.shape,
            device=self._mesh.tri_vertex_indices.device,
            inputs=[
                self._mesh._edge_tri_indices,
                self._mesh._edge_vertex_indices,
                self._mesh.tri_vertex_indices,
                self._tri_edge_indices,
            ],
        )

    @wp.kernel
    def _compute_reference_transforms_kernel(
        tri_vertex_indices: wp.array2d(dtype=int),
        positions: wp.array(dtype=wp.vec2f),
        transforms: wp.array(dtype=wp.mat22f),
    ):
        t = wp.tid()

        p0 = positions[tri_vertex_indices[t, 0]]
        p1 = positions[tri_vertex_indices[t, 1]]
        p2 = positions[tri_vertex_indices[t, 2]]

        e1 = p1 - p0
        e2 = p2 - p0

        mat = wp.mat22(e1, e2)
        transforms[t] = wp.transpose(wp.inverse(mat))

    @wp.func
    def _find_edge_index_in_tri(
        edge_vtx: wp.vec2i,
        tri_vtx: wp.vec3i,
    ):
        for k in range(2):
            if (edge_vtx[0] == tri_vtx[k] and edge_vtx[1] == tri_vtx[k + 1]) or (
                edge_vtx[1] == tri_vtx[k] and edge_vtx[0] == tri_vtx[k + 1]
            ):
                return k
        return 2

    @wp.kernel
    def _compute_tri_edge_indices_kernel(
        edge_tri_indices: wp.array(dtype=wp.vec2i),
        edge_vertex_indices: wp.array(dtype=wp.vec2i),
        tri_vertex_indices: wp.array2d(dtype=int),
        tri_edge_indices: wp.array2d(dtype=int),
    ):
        e = wp.tid()

        edge_vtx = edge_vertex_indices[e]
        edge_tris = edge_tri_indices[e]

        t0 = edge_tris[0]
        t0_vtx = wp.vec3i(tri_vertex_indices[t0, 0], tri_vertex_indices[t0, 1], tri_vertex_indices[t0, 2])
        t0_edge = Trimesh2DFunctionSpace._find_edge_index_in_tri(edge_vtx, t0_vtx)
        tri_edge_indices[t0, t0_edge] = e

        t1 = edge_tris[1]
        if t1 != t0:
            t1_vtx = wp.vec3i(tri_vertex_indices[t1, 0], tri_vertex_indices[t1, 1], tri_vertex_indices[t1, 2])
            t1_edge = Trimesh2DFunctionSpace._find_edge_index_in_tri(edge_vtx, t1_vtx)
            tri_edge_indices[t1, t1_edge] = e


class Trimesh2DPiecewiseConstantSpace(Trimesh2DFunctionSpace):
    ORDER = wp.constant(0)
    NODES_PER_ELEMENT = wp.constant(1)

    def __init__(self, grid: Trimesh2D, dtype: type = float, dof_mapper: DofMapper = None):
        super().__init__(grid, dtype, dof_mapper)

        self.element_outer_weight = self.element_inner_weight
        self.element_outer_weight_gradient = self.element_inner_weight_gradient

    def node_count(self) -> int:
        return self._mesh.cell_count()

    def node_positions(self):
        vtx_pos = self._mesh.positions.numpy()
        tri_vtx = self._mesh.tri_vertex_indices.numpy()

        tri_pos = vtx_pos[tri_vtx]
        centers = tri_pos.sum(axis=1) / 3.0

        return centers[:,0], centers[:,1]

    @wp.func
    def element_node_index(
        args: Trimesh2DFunctionSpace.SpaceArg,
        element_index: ElementIndex,
        node_index_in_elt: int,
    ):
        return element_index

    @wp.func
    def node_coords_in_element(
        args: Trimesh2DFunctionSpace.SpaceArg,
        element_index: ElementIndex,
        node_index_in_elt: int,
    ):
        if node_index_in_elt == 0:
            return Coords(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

        return Coords(OUTSIDE)

    @wp.func
    def node_quadrature_weight(
        args: Trimesh2DFunctionSpace.SpaceArg,
        element_index: ElementIndex,
        node_index_in_elt: int,
    ):
        return 1.0

    @wp.func
    def element_inner_weight(
        args: Trimesh2DFunctionSpace.SpaceArg,
        element_index: ElementIndex,
        coords: Coords,
        node_index_in_elt: int,
    ):
        if node_index_in_elt == 0:
            return 1.0
        return 0.0

    @wp.func
    def element_inner_weight_gradient(
        args: Trimesh2DFunctionSpace.SpaceArg,
        element_index: ElementIndex,
        coords: Coords,
        node_index_in_elt: int,
    ):
        return wp.vec2(0.0)

    class Trace(Trimesh2DFunctionSpace.Trace):
        NODES_PER_ELEMENT = wp.constant(2)
        ORDER = wp.constant(0)

        def __init__(self, space: "Trimesh2DPiecewiseConstantSpace"):
            super().__init__(space)

            self.element_node_index = self._make_element_node_index(space)

            self.element_inner_weight = self._make_element_inner_weight(space)
            self.element_inner_weight_gradient = self._make_element_inner_weight_gradient(space)

            self.element_outer_weight = self._make_element_outer_weight(space)
            self.element_outer_weight_gradient = self._make_element_outer_weight_gradient(space)

        @wp.func
        def node_coords_in_element(
            args: Trimesh2DFunctionSpace.SpaceArg,
            element_index: ElementIndex,
            node_index_in_element: int,
        ):
            if node_index_in_element == 0:
                return Coords(0.5, 0.0, 0.0)
            elif node_index_in_element == 1:
                return Coords(0.5, 0.0, 0.0)

            return Coords(OUTSIDE)

        @wp.func
        def node_quadrature_weight(
            args: Trimesh2DFunctionSpace.SpaceArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            return 1.0

    def trace(self):
        return Trimesh2DPiecewiseConstantSpace.Trace(self)


def _triangle_node_index(tx: int, ty: int, degree: int):
    VERTEX_NODE_COUNT = 3
    SIDE_INTERIOR_NODE_COUNT = degree - 1

    # Index in similar order to e.g. VTK
    # First vertices, then edge (counterclokwise) then interior points (recursively)

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
    return vertex_edge_node_count + _triangle_node_index(tx - 1, ty - 1, degree - 2)


class Trimesh2DPolynomialShapeFunctions:
    INVALID = wp.constant(-1)
    VERTEX = wp.constant(0)
    EDGE = wp.constant(1)
    INTERIOR = wp.constant(2)

    def __init__(self, degree: int):
        self.ORDER = wp.constant(degree)

        self.NODES_PER_ELEMENT = wp.constant((degree + 1) * (degree + 2) // 2)
        self.NODES_PER_SIDE = wp.constant(degree + 1)

        triangle_coords = np.empty((self.NODES_PER_ELEMENT, 2), dtype=int)

        for tx in range(degree + 1):
            for ty in range(degree + 1 - tx):
                index = _triangle_node_index(tx, ty, degree)
                triangle_coords[index] = [tx, ty]

        CoordTypeVec = wp.mat(dtype=int, shape=(self.NODES_PER_ELEMENT, 2))
        self.NODE_TRIANGLE_COORDS = wp.constant(CoordTypeVec(triangle_coords))

        self.node_type_and_type_index = self._get_node_type_and_type_index()
        self._node_triangle_coordinates = self._get_node_triangle_coordinates()

    @property
    def name(self) -> str:
        return f"{self.ORDER}"

    def _get_node_triangle_coordinates(self):
        NODE_TRIANGLE_COORDS = self.NODE_TRIANGLE_COORDS

        def node_triangle_coordinates(
            node_index_in_elt: int,
        ):
            return wp.vec2i(NODE_TRIANGLE_COORDS[node_index_in_elt, 0], NODE_TRIANGLE_COORDS[node_index_in_elt, 1])

        from warp.fem import cache

        return cache.get_func(node_triangle_coordinates, self.name)

    def _get_node_type_and_type_index(self):
        ORDER = self.ORDER
        NODES_PER_ELEMENT = self.NODES_PER_ELEMENT

        def node_type_and_index(
            node_index_in_elt: int,
        ):
            if node_index_in_elt < 0 or node_index_in_elt >= NODES_PER_ELEMENT:
                return Trimesh2DPolynomialShapeFunctions.INVALID, Trimesh2DPolynomialShapeFunctions.INVALID

            if node_index_in_elt < 3:
                return Trimesh2DPolynomialShapeFunctions.VERTEX, node_index_in_elt

            if node_index_in_elt < 3 * ORDER:
                return Trimesh2DPolynomialShapeFunctions.EDGE, (node_index_in_elt - 3)

            return Trimesh2DPolynomialShapeFunctions.INTERIOR, (node_index_in_elt - 3 * ORDER)

        from warp.fem import cache

        return cache.get_func(node_type_and_index, self.name)

    def make_node_coords_in_element(self):
        ORDER = self.ORDER

        def node_coords_in_element(
            args: Trimesh2DFunctionSpace.SpaceArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            tri_coords = self._node_triangle_coordinates(node_index_in_elt)
            cx = float(tri_coords[0]) / float(ORDER)
            cy = float(tri_coords[1]) / float(ORDER)
            return Coords(1.0 - cx - cy, cx, cy)

        from warp.fem import cache

        return cache.get_func(node_coords_in_element, self.name)

    def make_node_quadrature_weight(self):
        ORDER = self.ORDER

        def node_uniform_quadrature_weight(
            args: Trimesh2DFunctionSpace.SpaceArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            node_type, type_index = self.node_type_and_type_index(node_index_in_elt)

            base_weight = 1.0 / float(3 * ORDER * ORDER)
            if node_type == Trimesh2DPolynomialShapeFunctions.VERTEX:
                return base_weight
            if node_type == Trimesh2DPolynomialShapeFunctions.EDGE:
                return 2.0 * base_weight
            return 4.0 * base_weight

        def node_linear_quadrature_weight(
            args: Trimesh2DFunctionSpace.SpaceArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            return 1.0 / 3.0

        from warp.fem import cache

        if ORDER == 1:
            return cache.get_func(node_linear_quadrature_weight, self.name)
        return cache.get_func(node_uniform_quadrature_weight, self.name)

    def make_trace_node_quadrature_weight(self):
        ORDER = self.ORDER
        NODES_PER_ELEMENT = self.NODES_PER_ELEMENT

        def trace_uniform_node_quadrature_weight(
            args: Trimesh2DFunctionSpace.SpaceArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            if node_index_in_elt >= NODES_PER_ELEMENT:
                node_index_in_cell = node_index_in_elt - NODES_PER_ELEMENT
            else:
                node_index_in_cell = node_index_in_elt

            # We're either on a side interior or at a vertex
            node_type, type_index = self.node_type_and_type_index(node_index_in_cell)

            base_weight = 1.0 / float(ORDER)
            return wp.select(node_type == Trimesh2DPolynomialShapeFunctions.VERTEX, base_weight, 0.5 * base_weight)

        def trace_linear_node_quadrature_weight(
            args: Trimesh2DFunctionSpace.SpaceArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            return 0.5

        from warp.fem import cache

        if ORDER == 1:
            return cache.get_func(trace_linear_node_quadrature_weight, self.name)

        return cache.get_func(trace_uniform_node_quadrature_weight, self.name)

    def make_element_inner_weight(self):
        ORDER = self.ORDER

        def element_inner_weight_linear(
            args: Trimesh2DFunctionSpace.SpaceArg,
            element_index: ElementIndex,
            coords: Coords,
            node_index_in_elt: int,
        ):
            if node_index_in_elt < 0 or node_index_in_elt >= 3:
                return 0.0

            return coords[node_index_in_elt]

        def element_inner_weight_quadratic(
            args: Trimesh2DFunctionSpace.SpaceArg,
            element_index: ElementIndex,
            coords: Coords,
            node_index_in_elt: int,
        ):
            node_type, type_index = self.node_type_and_type_index(node_index_in_elt)

            if node_type == Trimesh2DPolynomialShapeFunctions.VERTEX:
                # Vertex
                return coords[type_index] * (2.0 * coords[type_index] - 1.0)

            elif node_type == Trimesh2DPolynomialShapeFunctions.EDGE:
                # Edge
                c1 = type_index
                c2 = (type_index + 1) % 3
                return 4.0 * coords[c1] * coords[c2]

            return 0.0

        def element_inner_weight_cubic(
            args: Trimesh2DFunctionSpace.SpaceArg,
            element_index: ElementIndex,
            coords: Coords,
            node_index_in_elt: int,
        ):
            node_type, type_index = self.node_type_and_type_index(node_index_in_elt)

            if node_type == Trimesh2DPolynomialShapeFunctions.VERTEX:
                # Vertex
                return 0.5 * coords[type_index] * (3.0 * coords[type_index] - 1.0) * (3.0 * coords[type_index] - 2.0)

            elif node_type == Trimesh2DPolynomialShapeFunctions.EDGE:
                # Edge
                edge = type_index // 2
                k = type_index - 2 * edge
                c1 = (edge + k) % 3
                c2 = (edge + 1 - k) % 3

                return 4.5 * coords[c1] * coords[c2] * (3.0 * coords[c1] - 1.0)

            elif node_type == Trimesh2DPolynomialShapeFunctions.INTERIOR:
                # Interior
                return 27.0 * coords[0] * coords[1] * coords[2]

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
            args: Trimesh2DFunctionSpace.SpaceArg,
            element_index: ElementIndex,
            coords: Coords,
            node_index_in_elt: int,
        ):
            if node_index_in_elt < 0 or node_index_in_elt >= 3:
                return wp.vec2(0.0)

            dw_dc = wp.vec3(0.0)
            dw_dc[node_index_in_elt] = 1.0

            dw_du = wp.vec2(dw_dc[1] - dw_dc[0], dw_dc[2] - dw_dc[0])
            return args.reference_transforms[element_index] * dw_du

        def element_inner_weight_gradient_quadratic(
            args: Trimesh2DFunctionSpace.SpaceArg,
            element_index: ElementIndex,
            coords: Coords,
            node_index_in_elt: int,
        ):
            node_type, type_index = self.node_type_and_type_index(node_index_in_elt)

            dw_dc = wp.vec3(0.0)

            if node_type == Trimesh2DPolynomialShapeFunctions.VERTEX:
                # Vertex
                dw_dc[type_index] = 4.0 * coords[type_index] - 1.0

            elif node_type == Trimesh2DPolynomialShapeFunctions.EDGE:
                # Edge
                c1 = type_index
                c2 = (type_index + 1) % 3
                dw_dc[c1] = 4.0 * coords[c2]
                dw_dc[c2] = 4.0 * coords[c1]

            dw_du = wp.vec2(dw_dc[1] - dw_dc[0], dw_dc[2] - dw_dc[0])
            return args.reference_transforms[element_index] * dw_du

        def element_inner_weight_gradient_cubic(
            args: Trimesh2DFunctionSpace.SpaceArg,
            element_index: ElementIndex,
            coords: Coords,
            node_index_in_elt: int,
        ):
            node_type, type_index = self.node_type_and_type_index(node_index_in_elt)

            dw_dc = wp.vec3(0.0)

            if node_type == Trimesh2DPolynomialShapeFunctions.VERTEX:
                # Vertex
                dw_dc[type_index] = (
                    0.5 * 27.0 * coords[type_index] * coords[type_index] - 9.0 * coords[type_index] + 1.0
                )

            elif node_type == Trimesh2DPolynomialShapeFunctions.EDGE:
                # Edge
                edge = type_index // 2
                k = type_index - 2 * edge
                c1 = (edge + k) % 3
                c2 = (edge + 1 - k) % 3

                dw_dc[c1] = 4.5 * coords[c2] * (6.0 * coords[c1] - 1.0)
                dw_dc[c2] = 4.5 * coords[c1] * (3.0 * coords[c1] - 1.0)

            elif node_type == Trimesh2DPolynomialShapeFunctions.INTERIOR:
                # Interior
                dw_dc = wp.vec3(
                    27.0 * coords[1] * coords[2], 27.0 * coords[2] * coords[0], 27.0 * coords[0] * coords[1]
                )

            dw_du = wp.vec2(dw_dc[1] - dw_dc[0], dw_dc[2] - dw_dc[0])
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
            return np.transpose(space._mesh.positions.numpy())

        NODES_PER_ELEMENT = space.NODES_PER_ELEMENT

        def fill_node_positions_fn(
            space_arg: space.SpaceArg,
            node_positions: wp.array(dtype=wp.vec2),
        ):
            element_index = wp.tid()
            tri_idx = space_arg.geo_arg.tri_vertex_indices[element_index]
            p0 = space_arg.geo_arg.positions[tri_idx[0]]
            p1 = space_arg.geo_arg.positions[tri_idx[1]]
            p2 = space_arg.geo_arg.positions[tri_idx[2]]

            for n in range(NODES_PER_ELEMENT):
                node_index = space.element_node_index(space_arg, element_index, n)
                coords = space.node_coords_in_element(space_arg, element_index, n)

                pos = coords[0] * p0 + coords[1] * p1 + coords[2] * p2

                node_positions[node_index] = pos

        from warp.fem import cache

        fill_node_positions = cache.get_kernel(
            fill_node_positions_fn,
            suffix=space.name,
        )

        device = space._mesh.tri_vertex_indices.device
        node_positions = wp.empty(
            shape=space.node_count(),
            dtype=wp.vec2,
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

        return np.transpose(node_positions.numpy())

    @staticmethod
    def node_triangulation(space):
        if space.ORDER == 1:
            return space._mesh.tri_vertex_indices.numpy()

        NODES_PER_ELEMENT = space.NODES_PER_ELEMENT

        def fill_element_node_indices_fn(
            space_arg: space.SpaceArg,
            element_node_indices: wp.array2d(dtype=int),
        ):
            element_index = wp.tid()
            for n in range(NODES_PER_ELEMENT):
                element_node_indices[element_index, n] = space.element_node_index(space_arg, element_index, n)

        from warp.fem import cache

        fill_element_node_indices = cache.get_kernel(
            fill_element_node_indices_fn,
            suffix=space.name,
        )

        device = space._mesh.tri_vertex_indices.device
        element_node_indices = wp.empty(
            shape=(space._mesh.cell_count(), NODES_PER_ELEMENT),
            dtype=int,
            device=device,
        )
        wp.launch(
            dim=element_node_indices.shape[0],
            kernel=fill_element_node_indices,
            inputs=[
                space.space_arg_value(device),
                element_node_indices,
            ],
            device=device,
        )

        element_node_indices = element_node_indices.numpy()
        if space.ORDER == 2:
            element_triangles = [[0, 3, 5], [3, 1, 4], [2, 5, 4], [3, 4, 5]]
        elif space.ORDER == 3:
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

        tri_indices = element_node_indices[:, element_triangles].reshape(-1, 3)
        return tri_indices


class Trimesh2DPolynomialSpace(Trimesh2DFunctionSpace):

    def __init__(self, grid: Trimesh2D, degree: int, dtype: type = float, dof_mapper: DofMapper = None):
        super().__init__(grid, dtype, dof_mapper)

        self._shape = Trimesh2DPolynomialShapeFunctions(degree)

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
        INTERIOR_NODES_PER_SIDE = wp.constant(max(0, self.ORDER - 1))
        INTERIOR_NODES_PER_CELL = wp.constant(max(0, self.ORDER - 2) * max(0, self.ORDER - 1) // 2)

        def element_node_index(
            args: Trimesh2DFunctionSpace.SpaceArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            node_type, type_index = self._shape.node_type_and_type_index(node_index_in_elt)

            if node_type == Trimesh2DPolynomialShapeFunctions.VERTEX:
                return args.geo_arg.tri_vertex_indices[element_index][type_index]

            global_offset = args.vertex_count

            if node_type == Trimesh2DPolynomialShapeFunctions.EDGE:
                edge = type_index // INTERIOR_NODES_PER_SIDE
                edge_node = type_index - INTERIOR_NODES_PER_SIDE * edge

                global_edge_index = args.tri_edge_indices[element_index][edge]

                if (
                    args.geo_arg.edge_vertex_indices[global_edge_index][0]
                    != args.geo_arg.tri_vertex_indices[element_index][edge]
                ):
                    edge_node = INTERIOR_NODES_PER_SIDE - 1 - edge_node

                return global_offset + INTERIOR_NODES_PER_SIDE * global_edge_index + edge_node

            global_offset += INTERIOR_NODES_PER_SIDE * args.edge_count
            return global_offset + INTERIOR_NODES_PER_CELL * element_index + type_index

        from warp.fem import cache

        return cache.get_func(element_node_index, self.name)

    def node_count(self) -> int:
        INTERIOR_NODES_PER_SIDE = wp.constant(max(0, self.ORDER - 1))
        INTERIOR_NODES_PER_CELL = wp.constant(max(0, self.ORDER - 2) * max(0, self.ORDER - 1) // 2)

        return (
            self._mesh.vertex_count()
            + self._mesh.side_count() * INTERIOR_NODES_PER_SIDE
            + self._mesh.cell_count() * INTERIOR_NODES_PER_CELL
        )

    def node_positions(self):
        return Trimesh2DPolynomialShapeFunctions.node_positions(self)

    def node_triangulation(self):
        return Trimesh2DPolynomialShapeFunctions.node_triangulation(self)

    class Trace(Trimesh2DFunctionSpace.Trace):
        NODES_PER_ELEMENT = wp.constant(2)
        ORDER = wp.constant(0)

        def __init__(self, space: "Trimesh2DPolynomialSpace"):
            super().__init__(space)

            self.element_node_index = self._make_element_node_index(space)
            self.node_coords_in_element = self._make_node_coords_in_element(space)
            self.node_quadrature_weight = space._shape.make_trace_node_quadrature_weight()

            self.element_inner_weight = self._make_element_inner_weight(space)
            self.element_inner_weight_gradient = self._make_element_inner_weight_gradient(space)

            self.element_outer_weight = self._make_element_outer_weight(space)
            self.element_outer_weight_gradient = self._make_element_outer_weight_gradient(space)

    def trace(self):
        return Trimesh2DPolynomialSpace.Trace(self)


class Trimesh2DDGPolynomialSpace(Trimesh2DFunctionSpace):
    def __init__(
        self,
        mesh: Trimesh2D,
        degree: int,
        dtype: type = float,
        dof_mapper: DofMapper = None,
    ):
        super().__init__(mesh, dtype, dof_mapper)

        self._shape = Trimesh2DPolynomialShapeFunctions(degree)

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
        return Trimesh2DPolynomialShapeFunctions.node_positions(self)

    def node_triangulation(self):
        return Trimesh2DPolynomialShapeFunctions.node_triangulation(self)

    def _make_element_node_index(self):
        NODES_PER_ELEMENT = self.NODES_PER_ELEMENT

        def element_node_index(
            args: Trimesh2DFunctionSpace.SpaceArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            return element_index * NODES_PER_ELEMENT + node_index_in_elt

        from warp.fem import cache

        return cache.get_func(element_node_index, f"{self.name}_{self.ORDER}")

    class Trace(Trimesh2DPolynomialSpace.Trace):
        def __init__(self, space: "Trimesh2DDGPolynomialSpace"):
            super().__init__(space)

    def trace(self):
        return Trimesh2DDGPolynomialSpace.Trace(self)
