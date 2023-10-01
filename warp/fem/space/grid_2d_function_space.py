import warp as wp
import numpy as np


from warp.fem.types import ElementIndex, Coords, OUTSIDE
from warp.fem.polynomial import Polynomial, lagrange_scales, quadrature_1d, is_closed
from warp.fem.geometry import Grid2D

from .dof_mapper import DofMapper
from .nodal_function_space import NodalFunctionSpace, NodalFunctionSpaceTrace


class Grid2DFunctionSpace(NodalFunctionSpace):
    DIMENSION = wp.constant(2)

    @wp.struct
    class SpaceArg:
        geo_arg: Grid2D.SideArg
        inv_cell_size: wp.vec2

    def __init__(self, grid: Grid2D, dtype: type = float, dof_mapper: DofMapper = None):
        super().__init__(dtype, dof_mapper)
        self._grid = grid

    @property
    def geometry(self) -> Grid2D:
        return self._grid

    def space_arg_value(self, device):
        arg = self.SpaceArg()
        arg.geo_arg = self.geometry.side_arg_value(device)
        arg.inv_cell_size = wp.vec2(
            1.0 / self.geometry.cell_size[0],
            1.0 / self.geometry.cell_size[1],
        )

        return arg

    class Trace(NodalFunctionSpaceTrace):
        def __init__(self, space: NodalFunctionSpace):
            super().__init__(space)
            self.ORDER = space.ORDER

    @wp.func
    def _inner_cell_index(args: SpaceArg, side_index: ElementIndex):
        return Grid2D.side_inner_cell_index(args.geo_arg, side_index)

    @wp.func
    def _outer_cell_index(args: SpaceArg, side_index: ElementIndex):
        return Grid2D.side_outer_cell_index(args.geo_arg, side_index)

    @wp.func
    def _inner_cell_coords(args: SpaceArg, side_index: ElementIndex, side_coords: Coords):
        side = Grid2D.get_side(args.geo_arg, side_index)

        if side.origin[0] == 0:
            inner_alt = 0.0
        else:
            inner_alt = 1.0

        coords = Grid2D._rotate(side.axis, wp.vec2(inner_alt, side_coords[0]))
        return Coords(coords[0], coords[1], 0.0)

    @wp.func
    def _outer_cell_coords(args: SpaceArg, side_index: ElementIndex, side_coords: Coords):
        side = Grid2D.get_side(args.geo_arg, side_index)

        alt_axis = Grid2D.ROTATION[side.axis, 0]
        if side.origin[0] == args.geo_arg.cell_arg.res[alt_axis]:
            outer_alt = 1.0
        else:
            outer_alt = 0.0

        coords = Grid2D._rotate(side.axis, wp.vec2(outer_alt, side_coords[0]))
        return Coords(coords[0], coords[1], 0.0)

    @wp.func
    def _cell_to_side_coords(
        args: SpaceArg,
        side_index: ElementIndex,
        element_index: ElementIndex,
        element_coords: Coords,
    ):
        side = Grid2D.get_side(args.geo_arg, side_index)
        cell = Grid2D.get_cell(args.geo_arg.cell_arg.res, element_index)

        if float(side.origin[0] - cell[side.axis]) == element_coords[side.axis]:
            long_axis = Grid2D.ROTATION[side.axis, 1]
            return Coords(element_coords[long_axis], 0.0, 0.0)

        return Coords(OUTSIDE)

    @wp.func
    def _vertex_coords(vidx_in_cell: int):
        x = vidx_in_cell // 2
        y = vidx_in_cell - 2 * x
        return wp.vec2i(x, y)

    @wp.func
    def _vertex_coords_f(vidx_in_cell: int):
        x = vidx_in_cell // 2
        y = vidx_in_cell - 2 * x
        return wp.vec2(float(x), float(y))

    @wp.func
    def _vertex_index(args: SpaceArg, cell_index: ElementIndex, vidx_in_cell: int):
        res = args.geo_arg.cell_arg.res
        x_stride = res[1] + 1

        corner = Grid2D.get_cell(res, cell_index) + Grid2DFunctionSpace._vertex_coords(vidx_in_cell)
        return Grid2D._from_2d_index(x_stride, corner)


class GridPiecewiseConstantSpace(Grid2DFunctionSpace):
    ORDER = wp.constant(0)
    NODES_PER_ELEMENT = wp.constant(1)

    def __init__(self, grid: Grid2D, dtype: type = float, dof_mapper: DofMapper = None):
        super().__init__(grid, dtype, dof_mapper)

        self.element_outer_weight = self.element_inner_weight
        self.element_outer_weight_gradient = self.element_inner_weight_gradient

    def node_count(self) -> int:
        return self._grid.cell_count()

    def node_positions(self):
        res = self._grid.res

        X = (np.arange(0, res[0], dtype=float) + 0.5) * self._grid.cell_size[0] + self._grid.origin[0]
        Y = (np.arange(0, res[1], dtype=float) + 0.5) * self._grid.cell_size[1] + self._grid.origin[1]
        return np.meshgrid(X, Y, indexing="ij")

    @wp.func
    def element_node_index(
        args: Grid2DFunctionSpace.SpaceArg,
        element_index: ElementIndex,
        node_index_in_elt: int,
    ):
        return element_index

    @wp.func
    def node_coords_in_element(
        args: Grid2DFunctionSpace.SpaceArg,
        element_index: ElementIndex,
        node_index_in_elt: int,
    ):
        if node_index_in_elt == 0:
            return Coords(0.5, 0.5, 0.0)

        return Coords(OUTSIDE)

    @wp.func
    def node_quadrature_weight(
        args: Grid2DFunctionSpace.SpaceArg,
        element_index: ElementIndex,
        node_index_in_elt: int,
    ):
        return 1.0

    @wp.func
    def element_inner_weight(
        args: Grid2DFunctionSpace.SpaceArg,
        element_index: ElementIndex,
        coords: Coords,
        node_index_in_elt: int,
    ):
        if node_index_in_elt == 0:
            return 1.0
        return 0.0

    @wp.func
    def element_inner_weight_gradient(
        args: Grid2DFunctionSpace.SpaceArg,
        element_index: ElementIndex,
        coords: Coords,
        node_index_in_elt: int,
    ):
        return wp.vec2(0.0)

    class Trace(Grid2DFunctionSpace.Trace):
        NODES_PER_ELEMENT = wp.constant(2)
        ORDER = wp.constant(0)

        def __init__(self, space: "GridPiecewiseConstantSpace"):
            super().__init__(space)

            self.element_node_index = self._make_element_node_index(space)

            self.element_inner_weight = self._make_element_inner_weight(space)
            self.element_inner_weight_gradient = self._make_element_inner_weight_gradient(space)

            self.element_outer_weight = self._make_element_outer_weight(space)
            self.element_outer_weight_gradient = self._make_element_outer_weight_gradient(space)

        @wp.func
        def node_coords_in_element(
            args: Grid2DFunctionSpace.SpaceArg,
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
            args: Grid2DFunctionSpace.SpaceArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            return 1.0

    def trace(self):
        return GridPiecewiseConstantSpace.Trace(self)


class GridBipolynomialShapeFunctions:
    def __init__(self, degree: int, family: Polynomial):
        self.family = family

        self.ORDER = wp.constant(degree)
        self.NODES_PER_ELEMENT = wp.constant((degree + 1) * (degree + 1))
        self.NODES_PER_SIDE = wp.constant(degree + 1)

        lobatto_coords, lobatto_weight = quadrature_1d(point_count=degree + 1, family=family)
        lagrange_scale = lagrange_scales(lobatto_coords)

        NodeVec = wp.types.vector(length=degree + 1, dtype=wp.float32)
        self.LOBATTO_COORDS = wp.constant(NodeVec(lobatto_coords))
        self.LOBATTO_WEIGHT = wp.constant(NodeVec(lobatto_weight))
        self.LAGRANGE_SCALE = wp.constant(NodeVec(lagrange_scale))

    @property
    def name(self) -> str:
        return f"{self.family}_{self.ORDER}"

    def make_node_coords_in_element(self):
        ORDER = self.ORDER
        LOBATTO_COORDS = self.LOBATTO_COORDS

        def node_coords_in_element(
            args: Grid2DFunctionSpace.SpaceArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            node_i = node_index_in_elt // (ORDER + 1)
            node_j = node_index_in_elt - (ORDER + 1) * node_i
            return Coords(LOBATTO_COORDS[node_i], LOBATTO_COORDS[node_j], 0.0)

        from warp.fem import cache

        return cache.get_func(node_coords_in_element, self.name)

    def make_node_quadrature_weight(self):
        ORDER = self.ORDER
        LOBATTO_WEIGHT = self.LOBATTO_WEIGHT

        def node_quadrature_weight(
            args: Grid2DFunctionSpace.SpaceArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            node_i = node_index_in_elt // (ORDER + 1)
            node_j = node_index_in_elt - (ORDER + 1) * node_i
            return LOBATTO_WEIGHT[node_i] * LOBATTO_WEIGHT[node_j]

        def node_quadrature_weight_linear(
            args: Grid2DFunctionSpace.SpaceArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            return 0.25

        from warp.fem import cache

        if ORDER == 1:
            return cache.get_func(node_quadrature_weight_linear, self.name)

        return cache.get_func(node_quadrature_weight, self.name)

    def make_trace_node_quadrature_weight(self):
        ORDER = self.ORDER
        NODES_PER_ELEMENT = self.NODES_PER_ELEMENT
        LOBATTO_WEIGHT = self.LOBATTO_WEIGHT

        def trace_node_quadrature_weight(
            args: Grid2DFunctionSpace.SpaceArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            if node_index_in_elt >= NODES_PER_ELEMENT:
                node_index_in_cell = node_index_in_elt - NODES_PER_ELEMENT
            else:
                node_index_in_cell = node_index_in_elt

            # We're either on a side interior or at a vertex
            # I.e., either both indices are at extrema, or only one is
            # Pick the interior one if possible, if both are at extrema pick any one
            node_i = node_index_in_cell // (ORDER + 1)
            if node_i > 0 and node_i < ORDER:
                return LOBATTO_WEIGHT[node_i]

            node_j = node_index_in_cell - (ORDER + 1) * node_i
            return LOBATTO_WEIGHT[node_j]

        def trace_node_quadrature_weight_linear(
            args: Grid2DFunctionSpace.SpaceArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            return 0.5

        from warp.fem import cache

        if ORDER == 1:
            return cache.get_func(trace_node_quadrature_weight_linear, self.name)

        return cache.get_func(trace_node_quadrature_weight, self.name)

    def make_element_inner_weight(self):
        ORDER = self.ORDER
        NODES_PER_ELEMENT = self.NODES_PER_ELEMENT
        LOBATTO_COORDS = self.LOBATTO_COORDS
        LAGRANGE_SCALE = self.LAGRANGE_SCALE

        def element_inner_weight(
            args: Grid2DFunctionSpace.SpaceArg,
            element_index: ElementIndex,
            coords: Coords,
            node_index_in_elt: int,
        ):
            if node_index_in_elt < 0 or node_index_in_elt >= NODES_PER_ELEMENT:
                return 0.0

            node_i = node_index_in_elt // (ORDER + 1)
            node_j = node_index_in_elt - (ORDER + 1) * node_i

            w = float(1.0)
            for k in range(ORDER + 1):
                if k != node_i:
                    w *= coords[0] - LOBATTO_COORDS[k]
                if k != node_j:
                    w *= coords[1] - LOBATTO_COORDS[k]

            w *= LAGRANGE_SCALE[node_i] * LAGRANGE_SCALE[node_j]

            return w

        def element_inner_weight_linear(
            args: Grid2DFunctionSpace.SpaceArg,
            element_index: ElementIndex,
            coords: Coords,
            node_index_in_elt: int,
        ):
            if node_index_in_elt < 0 or node_index_in_elt >= 4:
                return 0.0

            v = Grid2DFunctionSpace._vertex_coords_f(node_index_in_elt)

            wx = (1.0 - coords[0]) * (1.0 - v[0]) + v[0] * coords[0]
            wy = (1.0 - coords[1]) * (1.0 - v[1]) + v[1] * coords[1]
            return wx * wy

        from warp.fem import cache

        if ORDER == 1:
            return cache.get_func(element_inner_weight_linear, self.name)

        return cache.get_func(element_inner_weight, self.name)

    def make_element_inner_weight_gradient(self):
        ORDER = self.ORDER
        NODES_PER_ELEMENT = self.NODES_PER_ELEMENT
        LOBATTO_COORDS = self.LOBATTO_COORDS
        LAGRANGE_SCALE = self.LAGRANGE_SCALE

        def element_inner_weight_gradient(
            args: Grid2DFunctionSpace.SpaceArg,
            element_index: ElementIndex,
            coords: Coords,
            node_index_in_elt: int,
        ):
            if node_index_in_elt < 0 or node_index_in_elt >= NODES_PER_ELEMENT:
                return wp.vec2(0.0)

            node_i = node_index_in_elt // (ORDER + 1)
            node_j = node_index_in_elt - (ORDER + 1) * node_i

            prefix_x = float(1.0)
            prefix_y = float(1.0)
            for k in range(ORDER + 1):
                if k != node_i:
                    prefix_y *= coords[0] - LOBATTO_COORDS[k]
                if k != node_j:
                    prefix_x *= coords[1] - LOBATTO_COORDS[k]

            grad_x = float(0.0)
            grad_y = float(0.0)

            for k in range(ORDER + 1):
                if k != node_i:
                    delta_x = coords[0] - LOBATTO_COORDS[k]
                    grad_x = grad_x * delta_x + prefix_x
                    prefix_x *= delta_x
                if k != node_j:
                    delta_y = coords[1] - LOBATTO_COORDS[k]
                    grad_y = grad_y * delta_y + prefix_y
                    prefix_y *= delta_y

            grad = (
                LAGRANGE_SCALE[node_i]
                * LAGRANGE_SCALE[node_j]
                * wp.vec2(grad_x * args.inv_cell_size[0], grad_y * args.inv_cell_size[1])
            )

            return grad

        def element_inner_weight_gradient_linear(
            args: Grid2DFunctionSpace.SpaceArg,
            element_index: ElementIndex,
            coords: Coords,
            node_index_in_elt: int,
        ):
            if node_index_in_elt < 0 or node_index_in_elt >= 4:
                return wp.vec2(0.0)

            v = Grid2DFunctionSpace._vertex_coords_f(node_index_in_elt)

            wx = (1.0 - coords[0]) * (1.0 - v[0]) + v[0] * coords[0]
            wy = (1.0 - coords[1]) * (1.0 - v[1]) + v[1] * coords[1]

            dx = (2.0 * v[0] - 1.0) * args.inv_cell_size[0]
            dy = (2.0 * v[1] - 1.0) * args.inv_cell_size[1]

            return wp.vec2(dx * wy, dy * wx)

        from warp.fem import cache

        if ORDER == 1:
            return cache.get_func(element_inner_weight_gradient_linear, self.name)

        return cache.get_func(element_inner_weight_gradient, self.name)


class GridBipolynomialSpace(Grid2DFunctionSpace):
    def __init__(
        self,
        grid: Grid2D,
        degree: int,
        family: int,
        dtype: type = float,
        dof_mapper: DofMapper = None,
    ):
        super().__init__(grid, dtype, dof_mapper)

        if family is None:
            family = Polynomial.LOBATTO_GAUSS_LEGENDRE

        if not is_closed(family):
            raise ValueError("A closed polynomial family is required to defined a continuous funciton space")

        self._shape = GridBipolynomialShapeFunctions(degree, family=family)

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
        return (self._grid.res[0] * self.ORDER + 1) * (self._grid.res[1] * self.ORDER + 1)

    def node_positions(self):
        res = self._grid.res

        cell_coords = np.array(self._shape.LOBATTO_COORDS)[:-1]

        grid_coords_x = np.repeat(np.arange(0, res[0], dtype=float), len(cell_coords)) + np.tile(
            cell_coords, reps=res[0]
        )
        grid_coords_x = np.append(grid_coords_x, res[0])
        X = grid_coords_x * self._grid.cell_size[0] + self._grid.origin[0]

        grid_coords_y = np.repeat(np.arange(0, res[1], dtype=float), len(cell_coords)) + np.tile(
            cell_coords, reps=res[1]
        )
        grid_coords_y = np.append(grid_coords_y, res[1])
        Y = grid_coords_y * self._grid.cell_size[1] + self._grid.origin[1]

        return np.meshgrid(X, Y, indexing="ij")

    def _make_element_node_index(self):
        ORDER = self.ORDER

        def element_node_index(
            args: Grid2DFunctionSpace.SpaceArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            res = args.geo_arg.cell_arg.res
            cell = Grid2D.get_cell(res, element_index)

            node_i = node_index_in_elt // (ORDER + 1)
            node_j = node_index_in_elt - (ORDER + 1) * node_i

            node_x = ORDER * cell[0] + node_i
            node_y = ORDER * cell[1] + node_j

            node_pitch = (res[1] * ORDER) + 1
            node_index = node_pitch * node_x + node_y

            return node_index

        from warp.fem import cache

        return cache.get_func(element_node_index, f"{self.name}_{ORDER}")

    class Trace(Grid2DFunctionSpace.Trace):
        def __init__(self, space: "GridBipolynomialSpace"):
            super().__init__(space)

            self.element_node_index = self._make_element_node_index(space)
            self.node_coords_in_element = self._make_node_coords_in_element(space)
            self.node_quadrature_weight = space._shape.make_trace_node_quadrature_weight()

            self.element_inner_weight = self._make_element_inner_weight(space)
            self.element_inner_weight_gradient = self._make_element_inner_weight_gradient(space)

            self.element_outer_weight = self._make_element_outer_weight(space)
            self.element_outer_weight_gradient = self._make_element_outer_weight_gradient(space)

    def trace(self):
        return GridBipolynomialSpace.Trace(self)


class GridDGBipolynomialSpace(Grid2DFunctionSpace):
    def __init__(
        self,
        grid: Grid2D,
        degree: int,
        family: Polynomial,
        dtype: type = float,
        dof_mapper: DofMapper = None,
    ):
        super().__init__(grid, dtype, dof_mapper)

        if family is None:
            family = Polynomial.LOBATTO_GAUSS_LEGENDRE

        self._shape = GridBipolynomialShapeFunctions(degree, family=family)

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
        return self._grid.cell_count() * (self.ORDER + 1) ** 2

    def node_positions(self):
        res = self._grid.res

        cell_coords = np.array(self._shape.LOBATTO_COORDS)

        grid_coords_x = np.repeat(np.arange(0, res[0], dtype=float), len(cell_coords)) + np.tile(
            cell_coords, reps=res[0]
        )
        X = grid_coords_x * self._grid.cell_size[0] + self._grid.origin[0]

        grid_coords_y = np.repeat(np.arange(0, res[1], dtype=float), len(cell_coords)) + np.tile(
            cell_coords, reps=res[1]
        )
        Y = grid_coords_y * self._grid.cell_size[1] + self._grid.origin[1]

        return np.meshgrid(X, Y, indexing="ij")

    def _make_element_node_index(self):
        NODES_PER_ELEMENT = self.NODES_PER_ELEMENT

        def element_node_index(
            args: Grid2DFunctionSpace.SpaceArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            return element_index * NODES_PER_ELEMENT + node_index_in_elt

        from warp.fem import cache

        return cache.get_func(element_node_index, f"{self.name}_{self.ORDER}")

    class Trace(GridBipolynomialSpace.Trace):
        def __init__(self, space: "GridBipolynomialSpace"):
            super().__init__(space)

    def trace(self):
        return GridDGBipolynomialSpace.Trace(self)
