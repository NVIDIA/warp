import warp as wp

from warp.fem.types import ElementIndex, Coords
from warp.fem.polynomial import Polynomial, is_closed
from warp.fem.geometry import Quadmesh2D
from warp.fem import cache

from .topology import SpaceTopology, DiscontinuousSpaceTopologyMixin, forward_base_topology
from .basis_space import ShapeBasisSpace, TraceBasisSpace

from .shape import ShapeFunction, ConstantShapeFunction
from .shape import (
    SquareBipolynomialShapeFunctions,
    SquareSerendipityShapeFunctions,
    SquareNonConformingPolynomialShapeFunctions,
)


@wp.struct
class Quadmesh2DTopologyArg:
    edge_vertex_indices: wp.array(dtype=wp.vec2i)
    quad_edge_indices: wp.array2d(dtype=int)

    vertex_count: int
    edge_count: int


class Quadmesh2DSpaceTopology(SpaceTopology):
    TopologyArg = Quadmesh2DTopologyArg

    def __init__(self, mesh: Quadmesh2D, shape: ShapeFunction):
        super().__init__(mesh, shape.NODES_PER_ELEMENT)
        self._mesh = mesh
        self._shape = shape

        self._compute_quad_edge_indices()

    @cache.cached_arg_value
    def topo_arg_value(self, device):
        arg = Quadmesh2DTopologyArg()
        arg.quad_edge_indices = self._quad_edge_indices.to(device)
        arg.edge_vertex_indices = self._mesh.edge_vertex_indices.to(device)

        arg.vertex_count = self._mesh.vertex_count()
        arg.edge_count = self._mesh.side_count()
        return arg

    def _compute_quad_edge_indices(self):
        self._quad_edge_indices = wp.empty(
            dtype=int, device=self._mesh.quad_vertex_indices.device, shape=(self._mesh.cell_count(), 4)
        )

        wp.launch(
            kernel=Quadmesh2DSpaceTopology._compute_quad_edge_indices_kernel,
            dim=self._mesh.edge_quad_indices.shape,
            device=self._mesh.quad_vertex_indices.device,
            inputs=[
                self._mesh.edge_quad_indices,
                self._mesh.edge_vertex_indices,
                self._mesh.quad_vertex_indices,
                self._quad_edge_indices,
            ],
        )

    @wp.func
    def _find_edge_index_in_quad(
        edge_vtx: wp.vec2i,
        quad_vtx: wp.vec4i,
    ):
        for k in range(3):
            if (edge_vtx[0] == quad_vtx[k] and edge_vtx[1] == quad_vtx[k + 1]) or (
                edge_vtx[1] == quad_vtx[k] and edge_vtx[0] == quad_vtx[k + 1]
            ):
                return k
        return 3

    @wp.kernel
    def _compute_quad_edge_indices_kernel(
        edge_quad_indices: wp.array(dtype=wp.vec2i),
        edge_vertex_indices: wp.array(dtype=wp.vec2i),
        quad_vertex_indices: wp.array2d(dtype=int),
        quad_edge_indices: wp.array2d(dtype=int),
    ):
        e = wp.tid()

        edge_vtx = edge_vertex_indices[e]
        edge_quads = edge_quad_indices[e]

        q0 = edge_quads[0]
        q0_vtx = wp.vec4i(
            quad_vertex_indices[q0, 0],
            quad_vertex_indices[q0, 1],
            quad_vertex_indices[q0, 2],
            quad_vertex_indices[q0, 3],
        )
        q0_edge = Quadmesh2DSpaceTopology._find_edge_index_in_quad(edge_vtx, q0_vtx)
        quad_edge_indices[q0, q0_edge] = e

        q1 = edge_quads[1]
        if q1 != q0:
            t1_vtx = wp.vec4i(
                quad_vertex_indices[q1, 0],
                quad_vertex_indices[q1, 1],
                quad_vertex_indices[q1, 2],
                quad_vertex_indices[q1, 3],
            )
            t1_edge = Quadmesh2DSpaceTopology._find_edge_index_in_quad(edge_vtx, t1_vtx)
            quad_edge_indices[q1, t1_edge] = e


class Quadmesh2DDiscontinuousSpaceTopology(
    DiscontinuousSpaceTopologyMixin,
    SpaceTopology,
):
    def __init__(self, mesh: Quadmesh2D, shape: ShapeFunction):
        super().__init__(mesh, shape.NODES_PER_ELEMENT)


class Quadmesh2DBasisSpace(ShapeBasisSpace):
    def __init__(self, topology: Quadmesh2DSpaceTopology, shape: ShapeFunction):
        super().__init__(topology, shape)

        self._mesh: Quadmesh2D = topology.geometry


class Quadmesh2DPiecewiseConstantBasis(Quadmesh2DBasisSpace):
    def __init__(self, mesh: Quadmesh2D):
        shape = ConstantShapeFunction(mesh.reference_cell(), space_dimension=2)
        topology = Quadmesh2DDiscontinuousSpaceTopology(mesh, shape)
        super().__init__(shape=shape, topology=topology)

    class Trace(TraceBasisSpace):
        @wp.func
        def _node_coords_in_element(
            side_arg: Quadmesh2D.SideArg,
            basis_arg: Quadmesh2DBasisSpace.BasisArg,
            element_index: ElementIndex,
            node_index_in_element: int,
        ):
            return Coords(0.5, 0.0, 0.0)

        def make_node_coords_in_element(self):
            return self._node_coords_in_element

    def trace(self):
        return Quadmesh2DPiecewiseConstantBasis.Trace(self)


class Quadmesh2DBipolynomialSpaceTopology(Quadmesh2DSpaceTopology):
    def __init__(self, mesh: Quadmesh2D, shape: SquareBipolynomialShapeFunctions):
        super().__init__(mesh, shape)

        self.element_node_index = self._make_element_node_index()

    def node_count(self) -> int:
        ORDER = self._shape.ORDER
        INTERIOR_NODES_PER_SIDE = max(0, ORDER - 1)
        INTERIOR_NODES_PER_CELL = INTERIOR_NODES_PER_SIDE**2

        return (
            self._mesh.vertex_count()
            + self._mesh.side_count() * INTERIOR_NODES_PER_SIDE
            + self._mesh.cell_count() * INTERIOR_NODES_PER_CELL
        )

    def _make_element_node_index(self):
        ORDER = self._shape.ORDER
        INTERIOR_NODES_PER_SIDE = wp.constant(max(0, ORDER - 1))
        INTERIOR_NODES_PER_CELL = wp.constant(INTERIOR_NODES_PER_SIDE**2)

        @cache.dynamic_func(suffix=self.name)
        def element_node_index(
            geo_arg: Quadmesh2D.CellArg,
            topo_arg: Quadmesh2DTopologyArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            node_i = node_index_in_elt // (ORDER + 1)
            node_j = node_index_in_elt - (ORDER + 1) * node_i

            # Vertices
            if node_i == 0:
                if node_j == 0:
                    return geo_arg.quad_vertex_indices[element_index, 0]
                elif node_j == ORDER:
                    return geo_arg.quad_vertex_indices[element_index, 3]

                # 3-0 edge
                side_index = topo_arg.quad_edge_indices[element_index, 3]
                local_vs = geo_arg.quad_vertex_indices[element_index, 3]
                global_vs = topo_arg.edge_vertex_indices[side_index][0]
                index_in_side = wp.select(local_vs == global_vs, ORDER - node_j, node_j) - 1

                return topo_arg.vertex_count + (ORDER - 1) * side_index + index_in_side

            elif node_i == ORDER:
                if node_j == 0:
                    return geo_arg.quad_vertex_indices[element_index, 1]
                elif node_j == ORDER:
                    return geo_arg.quad_vertex_indices[element_index, 2]

                # 1-2 edge
                side_index = topo_arg.quad_edge_indices[element_index, 1]
                local_vs = geo_arg.quad_vertex_indices[element_index, 1]
                global_vs = topo_arg.edge_vertex_indices[side_index][0]
                index_in_side = wp.select(local_vs == global_vs, ORDER - node_j, node_j) - 1

                return topo_arg.vertex_count + (ORDER - 1) * side_index + index_in_side

            if node_j == 0:
                # 0-1 edge
                side_index = topo_arg.quad_edge_indices[element_index, 0]
                local_vs = geo_arg.quad_vertex_indices[element_index, 0]
                global_vs = topo_arg.edge_vertex_indices[side_index][0]
                index_in_side = wp.select(local_vs == global_vs, node_i, ORDER - node_i) - 1

                return topo_arg.vertex_count + (ORDER - 1) * side_index + index_in_side

            elif node_j == ORDER:
                # 2-3 edge
                side_index = topo_arg.quad_edge_indices[element_index, 2]
                local_vs = geo_arg.quad_vertex_indices[element_index, 2]
                global_vs = topo_arg.edge_vertex_indices[side_index][0]
                index_in_side = wp.select(local_vs == global_vs, node_i, ORDER - node_i) - 1

                return topo_arg.vertex_count + (ORDER - 1) * side_index + index_in_side

            return (
                topo_arg.vertex_count
                + topo_arg.edge_count * INTERIOR_NODES_PER_SIDE
                + element_index * INTERIOR_NODES_PER_CELL
                + (node_i - 1) * INTERIOR_NODES_PER_SIDE
                + node_j
                - 1
            )

        return element_node_index


class Quadmesh2DBipolynomialBasisSpace(Quadmesh2DBasisSpace):
    def __init__(
        self,
        mesh: Quadmesh2D,
        degree: int,
        family: Polynomial,
    ):
        if family is None:
            family = Polynomial.LOBATTO_GAUSS_LEGENDRE

        if not is_closed(family):
            raise ValueError("A closed polynomial family is required to define a continuous function space")

        shape = SquareBipolynomialShapeFunctions(degree, family=family)
        topology = forward_base_topology(Quadmesh2DBipolynomialSpaceTopology, mesh, shape)

        super().__init__(topology, shape)


class Quadmesh2DDGBipolynomialBasisSpace(Quadmesh2DBasisSpace):
    def __init__(
        self,
        mesh: Quadmesh2D,
        degree: int,
        family: Polynomial,
    ):
        if family is None:
            family = Polynomial.LOBATTO_GAUSS_LEGENDRE

        shape = SquareBipolynomialShapeFunctions(degree, family=family)
        topology = Quadmesh2DDiscontinuousSpaceTopology(mesh, shape)

        super().__init__(topology, shape)


class Quadmesh2DSerendipitySpaceTopology(Quadmesh2DSpaceTopology):
    def __init__(self, grid: Quadmesh2D, shape: SquareSerendipityShapeFunctions):
        super().__init__(grid, shape)

        self.element_node_index = self._make_element_node_index()

    def node_count(self) -> int:
        return self.geometry.vertex_count() + (self._shape.ORDER - 1) * self.geometry.side_count()

    def _make_element_node_index(self):
        ORDER = self._shape.ORDER

        SHAPE_TO_QUAD_IDX = wp.constant(wp.vec4i([0, 3, 1, 2]))

        @cache.dynamic_func(suffix=self.name)
        def element_node_index(
            cell_arg: Quadmesh2D.CellArg,
            topo_arg: Quadmesh2DSpaceTopology.TopologyArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            node_type, type_index = self._shape.node_type_and_type_index(node_index_in_elt)

            if node_type == SquareSerendipityShapeFunctions.VERTEX:
                return cell_arg.quad_vertex_indices[element_index, SHAPE_TO_QUAD_IDX[type_index]]

            side_offset, index_in_side = SquareSerendipityShapeFunctions.side_offset_and_index(type_index)

            if node_type == SquareSerendipityShapeFunctions.EDGE_X:
                if side_offset == 0:
                    side_start = 0
                else:
                    side_start = 2
                    index_in_side = ORDER - 2 - index_in_side
            else:
                if side_offset == 0:
                    side_start = 3
                    index_in_side = ORDER - 2 - index_in_side
                else:
                    side_start = 1

            side_index = topo_arg.quad_edge_indices[element_index, side_start]
            local_vs = cell_arg.quad_vertex_indices[element_index, side_start]
            global_vs = topo_arg.edge_vertex_indices[side_index][0]
            if local_vs != global_vs:
                # Flip indexing direction
                index_in_side = ORDER - 2 - index_in_side

            return topo_arg.vertex_count + (ORDER - 1) * side_index + index_in_side

        return element_node_index


class Quadmesh2DSerendipityBasisSpace(Quadmesh2DBasisSpace):
    def __init__(
        self,
        mesh: Quadmesh2D,
        degree: int,
        family: Polynomial,
    ):
        if family is None:
            family = Polynomial.LOBATTO_GAUSS_LEGENDRE

        shape = SquareSerendipityShapeFunctions(degree, family=family)
        topology = forward_base_topology(Quadmesh2DSerendipitySpaceTopology, mesh, shape=shape)

        super().__init__(topology=topology, shape=shape)


class Quadmesh2DDGSerendipityBasisSpace(Quadmesh2DBasisSpace):
    def __init__(
        self,
        mesh: Quadmesh2D,
        degree: int,
        family: Polynomial,
    ):
        if family is None:
            family = Polynomial.LOBATTO_GAUSS_LEGENDRE

        shape = SquareSerendipityShapeFunctions(degree, family=family)
        topology = Quadmesh2DDiscontinuousSpaceTopology(mesh, shape=shape)

        super().__init__(topology=topology, shape=shape)


class Quadmesh2DPolynomialBasisSpace(Quadmesh2DBasisSpace):
    def __init__(
        self,
        mesh: Quadmesh2D,
        degree: int,
    ):
        shape = SquareNonConformingPolynomialShapeFunctions(degree)
        topology = Quadmesh2DDiscontinuousSpaceTopology(mesh, shape)

        super().__init__(topology, shape)
