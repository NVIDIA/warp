import warp as wp

from warp.fem.types import ElementIndex, Coords
from warp.fem.geometry import Tetmesh
from warp.fem import cache

from .topology import SpaceTopology, DiscontinuousSpaceTopologyMixin, forward_base_topology
from .basis_space import ShapeBasisSpace, TraceBasisSpace

from .shape import ShapeFunction, ConstantShapeFunction
from .shape import TetrahedronPolynomialShapeFunctions, TetrahedronNonConformingPolynomialShapeFunctions


@wp.struct
class TetmeshTopologyArg:
    tet_edge_indices: wp.array2d(dtype=int)
    tet_face_indices: wp.array2d(dtype=int)
    face_vertex_indices: wp.array(dtype=wp.vec3i)

    vertex_count: int
    edge_count: int
    face_count: int


class TetmeshSpaceTopology(SpaceTopology):
    TopologyArg = TetmeshTopologyArg

    def __init__(
        self,
        mesh: Tetmesh,
        shape: ShapeFunction,
        need_tet_edge_indices: bool = True,
        need_tet_face_indices: bool = True,
    ):
        super().__init__(mesh, shape.NODES_PER_ELEMENT)
        self._mesh = mesh
        self._shape = shape

        if need_tet_edge_indices:
            self._tet_edge_indices = self._mesh.tet_edge_indices
            self._edge_count = self._mesh.edge_count()
        else:
            self._tet_edge_indices = wp.empty(shape=(0, 0), dtype=int)
            self._edge_count = 0

        if need_tet_face_indices:
            self._compute_tet_face_indices()
        else:
            self._tet_face_indices = wp.empty(shape=(0, 0), dtype=int)

    @cache.cached_arg_value
    def topo_arg_value(self, device):
        arg = TetmeshTopologyArg()
        arg.tet_face_indices = self._tet_face_indices.to(device)
        arg.tet_edge_indices = self._tet_edge_indices.to(device)
        arg.face_vertex_indices = self._mesh.face_vertex_indices.to(device)

        arg.vertex_count = self._mesh.vertex_count()
        arg.face_count = self._mesh.side_count()
        arg.edge_count = self._edge_count
        return arg

    def _compute_tet_face_indices(self):
        self._tet_face_indices = wp.empty(
            dtype=int, device=self._mesh.tet_vertex_indices.device, shape=(self._mesh.cell_count(), 4)
        )

        wp.launch(
            kernel=TetmeshSpaceTopology._compute_tet_face_indices_kernel,
            dim=self._mesh._face_tet_indices.shape,
            device=self._mesh.tet_vertex_indices.device,
            inputs=[
                self._mesh.face_tet_indices,
                self._mesh.face_vertex_indices,
                self._mesh.tet_vertex_indices,
                self._tet_face_indices,
            ],
        )

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
        t0_face = TetmeshSpaceTopology._find_face_index_in_tet(face_vtx, t0_vtx)
        tet_face_indices[t0, t0_face] = e

        t1 = face_tets[1]
        if t1 != t0:
            t1_vtx = wp.vec4i(
                tet_vertex_indices[t1, 0],
                tet_vertex_indices[t1, 1],
                tet_vertex_indices[t1, 2],
                tet_vertex_indices[t1, 3],
            )
            t1_face = TetmeshSpaceTopology._find_face_index_in_tet(face_vtx, t1_vtx)
            tet_face_indices[t1, t1_face] = e


class TetmeshDiscontinuousSpaceTopology(
    DiscontinuousSpaceTopologyMixin,
    SpaceTopology,
):
    def __init__(self, mesh: Tetmesh, shape: ShapeFunction):
        super().__init__(mesh, shape.NODES_PER_ELEMENT)


class TetmeshBasisSpace(ShapeBasisSpace):
    def __init__(self, topology: TetmeshSpaceTopology, shape: ShapeFunction):
        super().__init__(topology, shape)

        self._mesh: Tetmesh = topology.geometry


class TetmeshPiecewiseConstantBasis(TetmeshBasisSpace):
    def __init__(self, mesh: Tetmesh):
        shape = ConstantShapeFunction(mesh.reference_cell(), space_dimension=3)
        topology = TetmeshDiscontinuousSpaceTopology(mesh, shape)
        super().__init__(shape=shape, topology=topology)

    class Trace(TraceBasisSpace):
        @wp.func
        def _node_coords_in_element(
            side_arg: Tetmesh.SideArg,
            basis_arg: TetmeshBasisSpace.BasisArg,
            element_index: ElementIndex,
            node_index_in_element: int,
        ):
            return Coords(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

        def make_node_coords_in_element(self):
            return self._node_coords_in_element

    def trace(self):
        return TetmeshPiecewiseConstantBasis.Trace(self)


class TetmeshPolynomialSpaceTopology(TetmeshSpaceTopology):
    def __init__(self, mesh: Tetmesh, shape: TetrahedronPolynomialShapeFunctions):
        super().__init__(mesh, shape, need_tet_edge_indices=shape.ORDER >= 2, need_tet_face_indices=shape.ORDER >= 3)

        self.element_node_index = self._make_element_node_index()

    def node_count(self) -> int:
        ORDER = self._shape.ORDER
        INTERIOR_NODES_PER_EDGE = max(0, ORDER - 1)
        INTERIOR_NODES_PER_FACE = max(0, ORDER - 2) * max(0, ORDER - 1) // 2
        INTERIOR_NODES_PER_CELL = max(0, ORDER - 3) * max(0, ORDER - 2) * max(0, ORDER - 1) // 6

        return (
            self._mesh.vertex_count()
            + self._mesh.edge_count() * INTERIOR_NODES_PER_EDGE
            + self._mesh.side_count() * INTERIOR_NODES_PER_FACE
            + self._mesh.cell_count() * INTERIOR_NODES_PER_CELL
        )

    def _make_element_node_index(self):
        ORDER = self._shape.ORDER
        INTERIOR_NODES_PER_EDGE = wp.constant(max(0, ORDER - 1))
        INTERIOR_NODES_PER_FACE = wp.constant(max(0, ORDER - 2) * max(0, ORDER - 1) // 2)
        INTERIOR_NODES_PER_CELL = wp.constant(max(0, ORDER - 3) * max(0, ORDER - 2) * max(0, ORDER - 1) // 6)

        @cache.dynamic_func(suffix=self.name)
        def element_node_index(
            geo_arg: Tetmesh.CellArg,
            topo_arg: TetmeshTopologyArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            node_type, type_index = self._shape.node_type_and_type_index(node_index_in_elt)

            if node_type == TetrahedronPolynomialShapeFunctions.VERTEX:
                return geo_arg.tet_vertex_indices[element_index][type_index]

            global_offset = topo_arg.vertex_count

            if node_type == TetrahedronPolynomialShapeFunctions.EDGE:
                edge = type_index // INTERIOR_NODES_PER_EDGE
                edge_node = type_index - INTERIOR_NODES_PER_EDGE * edge

                global_edge_index = topo_arg.tet_edge_indices[element_index][edge]

                # Test if we need to swap edge direction
                if INTERIOR_NODES_PER_EDGE > 1:
                    if edge < 3:
                        c1 = edge
                        c2 = (edge + 1) % 3
                    else:
                        c1 = edge - 3
                        c2 = 3

                    if geo_arg.tet_vertex_indices[element_index][c1] > geo_arg.tet_vertex_indices[element_index][c2]:
                        edge_node = INTERIOR_NODES_PER_EDGE - 1 - edge_node

                return global_offset + INTERIOR_NODES_PER_EDGE * global_edge_index + edge_node

            global_offset += INTERIOR_NODES_PER_EDGE * topo_arg.edge_count

            if node_type == TetrahedronPolynomialShapeFunctions.FACE:
                face = type_index // INTERIOR_NODES_PER_FACE
                face_node = type_index - INTERIOR_NODES_PER_FACE * face

                global_face_index = topo_arg.tet_face_indices[element_index][face]

                if INTERIOR_NODES_PER_FACE == 3:
                    # Hard code for P4 case, 3 nodes per face
                    # Higher orders would require rotating triangle coordinates, this is not supported yet

                    vidx = geo_arg.tet_vertex_indices[element_index][(face + face_node) % 4]
                    fvi = topo_arg.face_vertex_indices[global_face_index]

                    if vidx == fvi[0]:
                        face_node = 0
                    elif vidx == fvi[1]:
                        face_node = 1
                    else:
                        face_node = 2

                return global_offset + INTERIOR_NODES_PER_FACE * global_face_index + face_node

            global_offset += INTERIOR_NODES_PER_FACE * topo_arg.face_count

            return global_offset + INTERIOR_NODES_PER_CELL * element_index + type_index

        return element_node_index


class TetmeshPolynomialBasisSpace(TetmeshBasisSpace):
    def __init__(
        self,
        mesh: Tetmesh,
        degree: int,
    ):
        shape = TetrahedronPolynomialShapeFunctions(degree)
        topology = forward_base_topology(TetmeshPolynomialSpaceTopology, mesh, shape)

        super().__init__(topology, shape)


class TetmeshDGPolynomialBasisSpace(TetmeshBasisSpace):
    def __init__(
        self,
        mesh: Tetmesh,
        degree: int,
    ):
        shape = TetrahedronPolynomialShapeFunctions(degree)
        topology = TetmeshDiscontinuousSpaceTopology(mesh, shape)

        super().__init__(topology, shape)


class TetmeshNonConformingPolynomialBasisSpace(TetmeshBasisSpace):
    def __init__(
        self,
        mesh: Tetmesh,
        degree: int,
    ):
        shape = TetrahedronNonConformingPolynomialShapeFunctions(degree)
        topology = TetmeshDiscontinuousSpaceTopology(mesh, shape)

        super().__init__(topology, shape)
