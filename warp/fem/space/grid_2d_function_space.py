import numpy as np

import warp as wp
from warp.fem import cache
from warp.fem.geometry import Grid2D
from warp.fem.polynomial import Polynomial, is_closed
from warp.fem.types import Coords, ElementIndex

from .basis_space import ShapeBasisSpace, TraceBasisSpace
from .shape import (
    ConstantShapeFunction,
    ShapeFunction,
    SquareBipolynomialShapeFunctions,
    SquareNonConformingPolynomialShapeFunctions,
    SquareSerendipityShapeFunctions,
)
from .topology import DiscontinuousSpaceTopologyMixin, SpaceTopology, forward_base_topology


class Grid2DSpaceTopology(SpaceTopology):
    def __init__(self, grid: Grid2D, shape: ShapeFunction):
        super().__init__(grid, shape.NODES_PER_ELEMENT)
        self._shape = shape

    @wp.func
    def _vertex_coords(vidx_in_cell: int):
        x = vidx_in_cell // 2
        y = vidx_in_cell - 2 * x
        return wp.vec2i(x, y)

    @wp.func
    def _vertex_index(cell_arg: Grid2D.CellArg, cell_index: ElementIndex, vidx_in_cell: int):
        res = cell_arg.res
        x_stride = res[1] + 1

        corner = Grid2D.get_cell(res, cell_index) + Grid2DSpaceTopology._vertex_coords(vidx_in_cell)
        return Grid2D._from_2d_index(x_stride, corner)


class Grid2DDiscontinuousSpaceTopology(
    DiscontinuousSpaceTopologyMixin,
    Grid2DSpaceTopology,
):
    pass


class Grid2DBasisSpace(ShapeBasisSpace):
    def __init__(self, topology: Grid2DSpaceTopology, shape: ShapeFunction):
        super().__init__(topology, shape)

        self._grid: Grid2D = topology.geometry


class GridPiecewiseConstantBasis(Grid2DBasisSpace):
    def __init__(self, grid: Grid2D):
        shape = ConstantShapeFunction(grid.reference_cell(), space_dimension=2)
        topology = Grid2DDiscontinuousSpaceTopology(grid, shape)
        super().__init__(shape=shape, topology=topology)

        if isinstance(grid, Grid2D):
            self.node_grid = self._node_grid

    def _node_grid(self):
        res = self._grid.res

        X = (np.arange(0, res[0], dtype=float) + 0.5) * self._grid.cell_size[0] + self._grid.origin[0]
        Y = (np.arange(0, res[1], dtype=float) + 0.5) * self._grid.cell_size[1] + self._grid.origin[1]
        return np.meshgrid(X, Y, indexing="ij")

    class Trace(TraceBasisSpace):
        @wp.func
        def _node_coords_in_element(
            side_arg: Grid2D.SideArg,
            basis_arg: Grid2DBasisSpace.BasisArg,
            element_index: ElementIndex,
            node_index_in_element: int,
        ):
            return Coords(0.5, 0.0, 0.0)

        def make_node_coords_in_element(self):
            return self._node_coords_in_element

    def trace(self):
        return GridPiecewiseConstantBasis.Trace(self)


class GridBipolynomialSpaceTopology(Grid2DSpaceTopology):
    def __init__(self, grid: Grid2D, shape: SquareBipolynomialShapeFunctions):
        super().__init__(grid, shape)

        self.element_node_index = self._make_element_node_index()

    def node_count(self) -> int:
        return (self.geometry.res[0] * self._shape.ORDER + 1) * (self.geometry.res[1] * self._shape.ORDER + 1)

    def _make_element_node_index(self):
        ORDER = self._shape.ORDER

        @cache.dynamic_func(suffix=self.name)
        def element_node_index(
            cell_arg: Grid2D.CellArg,
            topo_arg: Grid2DSpaceTopology.TopologyArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            res = cell_arg.res
            cell = Grid2D.get_cell(res, element_index)

            node_i = node_index_in_elt // (ORDER + 1)
            node_j = node_index_in_elt - (ORDER + 1) * node_i

            node_x = ORDER * cell[0] + node_i
            node_y = ORDER * cell[1] + node_j

            node_pitch = (res[1] * ORDER) + 1
            node_index = node_pitch * node_x + node_y

            return node_index

        return element_node_index


class GridBipolynomialBasisSpace(Grid2DBasisSpace):
    def __init__(
        self,
        grid: Grid2D,
        degree: int,
        family: Polynomial,
    ):
        if family is None:
            family = Polynomial.LOBATTO_GAUSS_LEGENDRE

        if not is_closed(family):
            raise ValueError("A closed polynomial family is required to define a continuous function space")

        shape = SquareBipolynomialShapeFunctions(degree, family=family)
        topology = forward_base_topology(GridBipolynomialSpaceTopology, grid, shape)

        super().__init__(topology, shape)

        if isinstance(grid, Grid2D):
            self.node_grid = self._node_grid

    def _node_grid(self):
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


class GridDGBipolynomialBasisSpace(Grid2DBasisSpace):
    def __init__(
        self,
        grid: Grid2D,
        degree: int,
        family: Polynomial,
    ):
        if family is None:
            family = Polynomial.LOBATTO_GAUSS_LEGENDRE

        shape = SquareBipolynomialShapeFunctions(degree, family=family)
        topology = Grid2DDiscontinuousSpaceTopology(grid, shape)

        super().__init__(shape=shape, topology=topology)


class GridSerendipitySpaceTopology(Grid2DSpaceTopology):
    def __init__(self, grid: Grid2D, shape: SquareSerendipityShapeFunctions):
        super().__init__(grid, shape)

        self.element_node_index = self._make_element_node_index()

    TopologyArg = Grid2D.SideArg

    def topo_arg_value(self, device):
        return self.geometry.side_arg_value(device)

    def node_count(self) -> int:
        return self.geometry.vertex_count() + (self._shape.ORDER - 1) * self.geometry.side_count()

    def _make_element_node_index(self):
        ORDER = self._shape.ORDER

        @cache.dynamic_func(suffix=self.name)
        def element_node_index(
            cell_arg: Grid2D.CellArg,
            topo_arg: Grid2D.SideArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            node_type, type_index = self._shape.node_type_and_type_index(node_index_in_elt)

            if node_type == SquareSerendipityShapeFunctions.VERTEX:
                return Grid2DSpaceTopology._vertex_index(cell_arg, element_index, type_index)

            side_offset, index_in_side = SquareSerendipityShapeFunctions.side_offset_and_index(type_index)
            axis = 1 - (node_type - SquareSerendipityShapeFunctions.EDGE_X)

            cell = Grid2D.get_cell(cell_arg.res, element_index)
            origin = wp.vec2i(cell[Grid2D.ROTATION[axis, 0]] + side_offset, cell[Grid2D.ROTATION[axis, 1]])

            side = Grid2D.Side(axis, origin)
            side_index = Grid2D.side_index(topo_arg, side)

            res = cell_arg.res
            vertex_count = (res[0] + 1) * (res[1] + 1)

            return vertex_count + (ORDER - 1) * side_index + index_in_side

        return element_node_index


class GridSerendipityBasisSpace(Grid2DBasisSpace):
    def __init__(
        self,
        grid: Grid2D,
        degree: int,
        family: Polynomial,
    ):
        if family is None:
            family = Polynomial.LOBATTO_GAUSS_LEGENDRE

        shape = SquareSerendipityShapeFunctions(degree, family=family)
        topology = forward_base_topology(GridSerendipitySpaceTopology, grid, shape=shape)

        super().__init__(topology=topology, shape=shape)


class GridDGSerendipityBasisSpace(Grid2DBasisSpace):
    def __init__(
        self,
        grid: Grid2D,
        degree: int,
        family: Polynomial,
    ):
        if family is None:
            family = Polynomial.LOBATTO_GAUSS_LEGENDRE

        shape = SquareSerendipityShapeFunctions(degree, family=family)
        topology = Grid2DDiscontinuousSpaceTopology(grid, shape=shape)

        super().__init__(topology=topology, shape=shape)


class GridDGPolynomialBasisSpace(Grid2DBasisSpace):
    def __init__(
        self,
        grid: Grid2D,
        degree: int,
    ):
        shape = SquareNonConformingPolynomialShapeFunctions(degree)
        topology = Grid2DDiscontinuousSpaceTopology(grid, shape=shape)

        super().__init__(topology=topology, shape=shape)
