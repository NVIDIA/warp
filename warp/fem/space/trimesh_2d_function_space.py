import warp as wp
from warp.fem import cache
from warp.fem.geometry import Trimesh2D
from warp.fem.types import Coords, ElementIndex

from .basis_space import ShapeBasisSpace, TraceBasisSpace
from .shape import (
    ConstantShapeFunction,
    ShapeFunction,
    Triangle2DNonConformingPolynomialShapeFunctions,
    Triangle2DPolynomialShapeFunctions,
)
from .topology import DiscontinuousSpaceTopologyMixin, SpaceTopology, forward_base_topology


@wp.struct
class Trimesh2DTopologyArg:
    edge_vertex_indices: wp.array(dtype=wp.vec2i)
    tri_edge_indices: wp.array2d(dtype=int)

    vertex_count: int
    edge_count: int


class Trimesh2DSpaceTopology(SpaceTopology):
    TopologyArg = Trimesh2DTopologyArg

    def __init__(self, mesh: Trimesh2D, shape: ShapeFunction):
        super().__init__(mesh, shape.NODES_PER_ELEMENT)
        self._mesh = mesh
        self._shape = shape

        self._compute_tri_edge_indices()

    @cache.cached_arg_value
    def topo_arg_value(self, device):
        arg = Trimesh2DTopologyArg()
        arg.tri_edge_indices = self._tri_edge_indices.to(device)
        arg.edge_vertex_indices = self._mesh.edge_vertex_indices.to(device)

        arg.vertex_count = self._mesh.vertex_count()
        arg.edge_count = self._mesh.side_count()
        return arg

    def _compute_tri_edge_indices(self):
        self._tri_edge_indices = wp.empty(
            dtype=int, device=self._mesh.tri_vertex_indices.device, shape=(self._mesh.cell_count(), 3)
        )

        wp.launch(
            kernel=Trimesh2DSpaceTopology._compute_tri_edge_indices_kernel,
            dim=self._mesh.edge_tri_indices.shape,
            device=self._mesh.tri_vertex_indices.device,
            inputs=[
                self._mesh.edge_tri_indices,
                self._mesh.edge_vertex_indices,
                self._mesh.tri_vertex_indices,
                self._tri_edge_indices,
            ],
        )

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
        t0_edge = Trimesh2DSpaceTopology._find_edge_index_in_tri(edge_vtx, t0_vtx)
        tri_edge_indices[t0, t0_edge] = e

        t1 = edge_tris[1]
        if t1 != t0:
            t1_vtx = wp.vec3i(tri_vertex_indices[t1, 0], tri_vertex_indices[t1, 1], tri_vertex_indices[t1, 2])
            t1_edge = Trimesh2DSpaceTopology._find_edge_index_in_tri(edge_vtx, t1_vtx)
            tri_edge_indices[t1, t1_edge] = e


class Trimesh2DDiscontinuousSpaceTopology(
    DiscontinuousSpaceTopologyMixin,
    SpaceTopology,
):
    def __init__(self, mesh: Trimesh2D, shape: ShapeFunction):
        super().__init__(mesh, shape.NODES_PER_ELEMENT)


class Trimesh2DBasisSpace(ShapeBasisSpace):
    def __init__(self, topology: Trimesh2DSpaceTopology, shape: ShapeFunction):
        super().__init__(topology, shape)

        self._mesh: Trimesh2D = topology.geometry


class Trimesh2DPiecewiseConstantBasis(Trimesh2DBasisSpace):
    def __init__(self, mesh: Trimesh2D):
        shape = ConstantShapeFunction(mesh.reference_cell(), space_dimension=2)
        topology = Trimesh2DDiscontinuousSpaceTopology(mesh, shape)
        super().__init__(shape=shape, topology=topology)

    class Trace(TraceBasisSpace):
        @wp.func
        def _node_coords_in_element(
            side_arg: Trimesh2D.SideArg,
            basis_arg: Trimesh2DBasisSpace.BasisArg,
            element_index: ElementIndex,
            node_index_in_element: int,
        ):
            return Coords(0.5, 0.0, 0.0)

        def make_node_coords_in_element(self):
            return self._node_coords_in_element

    def trace(self):
        return Trimesh2DPiecewiseConstantBasis.Trace(self)


class Trimesh2DPolynomialSpaceTopology(Trimesh2DSpaceTopology):
    def __init__(self, mesh: Trimesh2D, shape: Triangle2DPolynomialShapeFunctions):
        super().__init__(mesh, shape)

        self.element_node_index = self._make_element_node_index()

    def node_count(self) -> int:
        INTERIOR_NODES_PER_SIDE = max(0, self._shape.ORDER - 1)
        INTERIOR_NODES_PER_CELL = max(0, self._shape.ORDER - 2) * max(0, self._shape.ORDER - 1) // 2

        return (
            self._mesh.vertex_count()
            + self._mesh.side_count() * INTERIOR_NODES_PER_SIDE
            + self._mesh.cell_count() * INTERIOR_NODES_PER_CELL
        )

    def _make_element_node_index(self):
        INTERIOR_NODES_PER_SIDE = wp.constant(max(0, self._shape.ORDER - 1))
        INTERIOR_NODES_PER_CELL = wp.constant(max(0, self._shape.ORDER - 2) * max(0, self._shape.ORDER - 1) // 2)

        @cache.dynamic_func(suffix=self.name)
        def element_node_index(
            geo_arg: Trimesh2D.CellArg,
            topo_arg: Trimesh2DTopologyArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            node_type, type_index = self._shape.node_type_and_type_index(node_index_in_elt)

            if node_type == Triangle2DPolynomialShapeFunctions.VERTEX:
                return geo_arg.tri_vertex_indices[element_index][type_index]

            global_offset = topo_arg.vertex_count

            if node_type == Triangle2DPolynomialShapeFunctions.EDGE:
                edge = type_index // INTERIOR_NODES_PER_SIDE
                edge_node = type_index - INTERIOR_NODES_PER_SIDE * edge

                global_edge_index = topo_arg.tri_edge_indices[element_index][edge]

                if (
                    topo_arg.edge_vertex_indices[global_edge_index][0]
                    != geo_arg.tri_vertex_indices[element_index][edge]
                ):
                    edge_node = INTERIOR_NODES_PER_SIDE - 1 - edge_node

                return global_offset + INTERIOR_NODES_PER_SIDE * global_edge_index + edge_node

            global_offset += INTERIOR_NODES_PER_SIDE * topo_arg.edge_count
            return global_offset + INTERIOR_NODES_PER_CELL * element_index + type_index

        return element_node_index


class Trimesh2DPolynomialBasisSpace(Trimesh2DBasisSpace):
    def __init__(
        self,
        mesh: Trimesh2D,
        degree: int,
    ):
        shape = Triangle2DPolynomialShapeFunctions(degree)
        topology = forward_base_topology(Trimesh2DPolynomialSpaceTopology, mesh, shape)

        super().__init__(topology, shape)


class Trimesh2DDGPolynomialBasisSpace(Trimesh2DBasisSpace):
    def __init__(
        self,
        mesh: Trimesh2D,
        degree: int,
    ):
        shape = Triangle2DPolynomialShapeFunctions(degree)
        topology = Trimesh2DDiscontinuousSpaceTopology(mesh, shape)

        super().__init__(topology, shape)


class Trimesh2DNonConformingPolynomialBasisSpace(Trimesh2DBasisSpace):
    def __init__(
        self,
        mesh: Trimesh2D,
        degree: int,
    ):
        shape = Triangle2DNonConformingPolynomialShapeFunctions(degree)
        topology = Trimesh2DDiscontinuousSpaceTopology(mesh, shape)

        super().__init__(topology, shape)
