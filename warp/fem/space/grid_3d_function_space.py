import warp as wp
import numpy as np

from warp.fem.types import ElementIndex, Coords
from warp.fem.polynomial import Polynomial, is_closed
from warp.fem.geometry import Grid3D
from warp.fem import cache

from .topology import SpaceTopology, DiscontinuousSpaceTopologyMixin, forward_base_topology
from .basis_space import ShapeBasisSpace, TraceBasisSpace

from .shape import ShapeFunction, ConstantShapeFunction
from .shape.cube_shape_function import (
    CubeTripolynomialShapeFunctions,
    CubeSerendipityShapeFunctions,
    CubeNonConformingPolynomialShapeFunctions,
)


class Grid3DSpaceTopology(SpaceTopology):
    def __init__(self, grid: Grid3D, shape: ShapeFunction):
        super().__init__(grid, shape.NODES_PER_ELEMENT)
        self._shape = shape

    @wp.func
    def _vertex_coords(vidx_in_cell: int):
        x = vidx_in_cell // 4
        y = (vidx_in_cell - 4 * x) // 2
        z = vidx_in_cell - 4 * x - 2 * y
        return wp.vec3i(x, y, z)

    @wp.func
    def _vertex_index(cell_arg: Grid3D.CellArg, cell_index: ElementIndex, vidx_in_cell: int):
        res = cell_arg.res
        strides = wp.vec2i((res[1] + 1) * (res[2] + 1), res[2] + 1)

        corner = Grid3D.get_cell(res, cell_index) + Grid3DSpaceTopology._vertex_coords(vidx_in_cell)
        return Grid3D._from_3d_index(strides, corner)


class Grid3DDiscontinuousSpaceTopology(
    DiscontinuousSpaceTopologyMixin,
    Grid3DSpaceTopology,
):
    pass


class Grid3DBasisSpace(ShapeBasisSpace):
    def __init__(self, topology: Grid3DSpaceTopology, shape: ShapeFunction):
        super().__init__(topology, shape)

        self._grid: Grid3D = topology.geometry


class Grid3DPiecewiseConstantBasis(Grid3DBasisSpace):
    def __init__(self, grid: Grid3D):
        shape = ConstantShapeFunction(grid.reference_cell(), space_dimension=3)
        topology = Grid3DDiscontinuousSpaceTopology(grid, shape)
        super().__init__(shape=shape, topology=topology)

        if isinstance(grid, Grid3D):
            self.node_grid = self._node_grid

    def _node_grid(self):
        X = (np.arange(0, self.geometry.res[0], dtype=float) + 0.5) * self._grid.cell_size[0] + self._grid.bounds_lo[0]
        Y = (np.arange(0, self.geometry.res[1], dtype=float) + 0.5) * self._grid.cell_size[1] + self._grid.bounds_lo[1]
        Z = (np.arange(0, self.geometry.res[2], dtype=float) + 0.5) * self._grid.cell_size[2] + self._grid.bounds_lo[2]
        return np.meshgrid(X, Y, Z, indexing="ij")

    class Trace(TraceBasisSpace):
        @wp.func
        def _node_coords_in_element(
            side_arg: Grid3D.SideArg,
            basis_arg: Grid3DBasisSpace.BasisArg,
            element_index: ElementIndex,
            node_index_in_element: int,
        ):
            return Coords(0.5, 0.5, 0.0)

        def make_node_coords_in_element(self):
            return self._node_coords_in_element

    def trace(self):
        return Grid3DPiecewiseConstantBasis.Trace(self)


class GridTripolynomialSpaceTopology(Grid3DSpaceTopology):
    def __init__(self, grid: Grid3D, shape: CubeTripolynomialShapeFunctions):
        super().__init__(grid, shape)

        self.element_node_index = self._make_element_node_index()

    def node_count(self) -> int:
        return (
            (self.geometry.res[0] * self._shape.ORDER + 1)
            * (self.geometry.res[1] * self._shape.ORDER + 1)
            * (self.geometry.res[2] * self._shape.ORDER + 1)
        )

    def _make_element_node_index(self):
        ORDER = self._shape.ORDER

        @cache.dynamic_func(suffix=self.name)
        def element_node_index(
            cell_arg: Grid3D.CellArg,
            topo_arg: Grid3DSpaceTopology.TopologyArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            res = cell_arg.res
            cell = Grid3D.get_cell(res, element_index)

            node_i, node_j, node_k = self._shape._node_ijk(node_index_in_elt)

            node_x = ORDER * cell[0] + node_i
            node_y = ORDER * cell[1] + node_j
            node_z = ORDER * cell[2] + node_k

            node_pitch_y = (res[2] * ORDER) + 1
            node_pitch_x = node_pitch_y * ((res[1] * ORDER) + 1)
            node_index = node_pitch_x * node_x + node_pitch_y * node_y + node_z

            return node_index

        return element_node_index


class GridTripolynomialBasisSpace(Grid3DBasisSpace):
    def __init__(
        self,
        grid: Grid3D,
        degree: int,
        family: Polynomial,
    ):
        if family is None:
            family = Polynomial.LOBATTO_GAUSS_LEGENDRE

        if not is_closed(family):
            raise ValueError("A closed polynomial family is required to define a continuous function space")

        shape = CubeTripolynomialShapeFunctions(degree, family=family)
        topology = forward_base_topology(GridTripolynomialSpaceTopology, grid, shape)

        super().__init__(topology, shape)

        if isinstance(grid, Grid3D):
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

        grid_coords_z = np.repeat(np.arange(0, res[2], dtype=float), len(cell_coords)) + np.tile(
            cell_coords, reps=res[2]
        )
        grid_coords_z = np.append(grid_coords_z, res[2])
        Z = grid_coords_z * self._grid.cell_size[2] + self._grid.origin[2]

        return np.meshgrid(X, Y, Z, indexing="ij")


class GridDGTripolynomialBasisSpace(Grid3DBasisSpace):
    def __init__(
        self,
        grid: Grid3D,
        degree: int,
        family: Polynomial,
    ):
        if family is None:
            family = Polynomial.LOBATTO_GAUSS_LEGENDRE

        shape = CubeTripolynomialShapeFunctions(degree, family=family)
        topology = Grid3DDiscontinuousSpaceTopology(grid, shape)

        super().__init__(shape=shape, topology=topology)

    def node_grid(self):
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

        grid_coords_z = np.repeat(np.arange(0, res[2], dtype=float), len(cell_coords)) + np.tile(
            cell_coords, reps=res[2]
        )
        Z = grid_coords_z * self._grid.cell_size[2] + self._grid.origin[2]

        return np.meshgrid(X, Y, Z, indexing="ij")


class Grid3DSerendipitySpaceTopology(Grid3DSpaceTopology):
    def __init__(self, grid: Grid3D, shape: CubeSerendipityShapeFunctions):
        super().__init__(grid, shape)

        self.element_node_index = self._make_element_node_index()

    def node_count(self) -> int:
        return self.geometry.vertex_count() + (self._shape.ORDER - 1) * self.geometry.edge_count()

    def _make_element_node_index(self):
        ORDER = self._shape.ORDER

        @cache.dynamic_func(suffix=self.name)
        def element_node_index(
            cell_arg: Grid3D.CellArg,
            topo_arg: Grid3DSpaceTopology.TopologyArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            res = cell_arg.res
            cell = Grid3D.get_cell(res, element_index)

            node_type, type_index = self._shape.node_type_and_type_index(node_index_in_elt)

            if node_type == CubeSerendipityShapeFunctions.VERTEX:
                return Grid3DSpaceTopology._vertex_index(cell_arg, element_index, type_index)

            axis = CubeSerendipityShapeFunctions._edge_axis(node_type)
            node_all = CubeSerendipityShapeFunctions._edge_coords(type_index)

            res = cell_arg.res

            edge_index = 0
            if axis > 0:
                edge_index += (res[1] + 1) * (res[2] + 1) * res[0]
            if axis > 1:
                edge_index += (res[0] + 1) * (res[2] + 1) * res[1]

            res_loc = Grid3D._world_to_local(axis, res)
            cell_loc = Grid3D._world_to_local(axis, cell)

            edge_index += (res_loc[1] + 1) * (res_loc[2] + 1) * cell_loc[0]
            edge_index += (res_loc[2] + 1) * (cell_loc[1] + node_all[1])
            edge_index += cell_loc[2] + node_all[2]

            vertex_count = (res[0] + 1) * (res[1] + 1) * (res[2] + 1)

            return vertex_count + (ORDER - 1) * edge_index + (node_all[0] - 1)

        return element_node_index


class Grid3DSerendipityBasisSpace(Grid3DBasisSpace):
    def __init__(
        self,
        grid: Grid3D,
        degree: int,
        family: Polynomial,
    ):
        if family is None:
            family = Polynomial.LOBATTO_GAUSS_LEGENDRE

        shape = CubeSerendipityShapeFunctions(degree, family=family)
        topology = forward_base_topology(Grid3DSerendipitySpaceTopology, grid, shape=shape)

        super().__init__(topology=topology, shape=shape)


class Grid3DDGSerendipityBasisSpace(Grid3DBasisSpace):
    def __init__(
        self,
        grid: Grid3D,
        degree: int,
        family: Polynomial,
    ):
        if family is None:
            family = Polynomial.LOBATTO_GAUSS_LEGENDRE

        shape = CubeSerendipityShapeFunctions(degree, family=family)
        topology = Grid3DDiscontinuousSpaceTopology(grid, shape=shape)

        super().__init__(topology=topology, shape=shape)


class Grid3DDGPolynomialBasisSpace(Grid3DBasisSpace):
    def __init__(
        self,
        grid: Grid3D,
        degree: int,
    ):
        shape = CubeNonConformingPolynomialShapeFunctions(degree)
        topology = Grid3DDiscontinuousSpaceTopology(grid, shape=shape)

        super().__init__(topology=topology, shape=shape)
