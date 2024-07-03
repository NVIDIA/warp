import warp as wp
from warp.fem import cache
from warp.fem.geometry import Hexmesh
from warp.fem.geometry.hexmesh import (
    EDGE_VERTEX_INDICES,
    FACE_ORIENTATION,
    FACE_TRANSLATION,
)
from warp.fem.polynomial import is_closed
from warp.fem.types import ElementIndex

from .shape import (
    CubeSerendipityShapeFunctions,
    CubeTripolynomialShapeFunctions,
    ShapeFunction,
)
from .topology import SpaceTopology, forward_base_topology

_FACE_ORIENTATION_I = wp.constant(wp.mat(shape=(16, 2), dtype=int)(FACE_ORIENTATION))
_FACE_TRANSLATION_I = wp.constant(wp.mat(shape=(4, 2), dtype=int)(FACE_TRANSLATION))

# map from shape function vertex indexing to hexmesh vertex indexing
_CUBE_TO_HEX_VERTEX = wp.constant(wp.vec(length=8, dtype=int)([0, 4, 3, 7, 1, 5, 2, 6]))

# map from shape function edge indexing to hexmesh edge indexing
_CUBE_TO_HEX_EDGE = wp.constant(wp.vec(length=12, dtype=int)([0, 4, 2, 6, 3, 1, 7, 5, 8, 11, 9, 10]))


@wp.struct
class HexmeshTopologyArg:
    hex_edge_indices: wp.array2d(dtype=int)
    hex_face_indices: wp.array2d(dtype=wp.vec2i)

    vertex_count: int
    edge_count: int
    face_count: int


class HexmeshSpaceTopology(SpaceTopology):
    TopologyArg = HexmeshTopologyArg

    def __init__(
        self,
        mesh: Hexmesh,
        shape: ShapeFunction,
        need_hex_edge_indices: bool = True,
        need_hex_face_indices: bool = True,
    ):
        if not is_closed(shape.family):
            raise ValueError("A closed polynomial family is required to define a continuous function space")

        super().__init__(mesh, shape.NODES_PER_ELEMENT)
        self._mesh = mesh
        self.shape = shape

        if need_hex_edge_indices:
            self._hex_edge_indices = self._mesh.hex_edge_indices
            self._edge_count = self._mesh.edge_count()
        else:
            self._hex_edge_indices = wp.empty(shape=(0, 0), dtype=int)
            self._edge_count = 0

        if need_hex_face_indices:
            self._compute_hex_face_indices()
        else:
            self._hex_face_indices = wp.empty(shape=(0, 0), dtype=wp.vec2i)

        self._compute_hex_face_indices()

    @cache.cached_arg_value
    def topo_arg_value(self, device):
        arg = HexmeshTopologyArg()
        arg.hex_edge_indices = self._hex_edge_indices.to(device)
        arg.hex_face_indices = self._hex_face_indices.to(device)

        arg.vertex_count = self._mesh.vertex_count()
        arg.face_count = self._mesh.side_count()
        arg.edge_count = self._edge_count
        return arg

    def _compute_hex_face_indices(self):
        self._hex_face_indices = wp.empty(
            dtype=wp.vec2i, device=self._mesh.hex_vertex_indices.device, shape=(self._mesh.cell_count(), 6)
        )

        wp.launch(
            kernel=HexmeshSpaceTopology._compute_hex_face_indices_kernel,
            dim=self._mesh.side_count(),
            device=self._mesh.hex_vertex_indices.device,
            inputs=[
                self._mesh.face_hex_indices,
                self._mesh._face_hex_face_orientation,
                self._hex_face_indices,
            ],
        )

    @wp.kernel
    def _compute_hex_face_indices_kernel(
        face_hex_indices: wp.array(dtype=wp.vec2i),
        face_hex_face_ori: wp.array(dtype=wp.vec4i),
        hex_face_indices: wp.array2d(dtype=wp.vec2i),
    ):
        f = wp.tid()

        hx0 = face_hex_indices[f][0]
        local_face_0 = face_hex_face_ori[f][0]
        ori_0 = face_hex_face_ori[f][1]

        hex_face_indices[hx0, local_face_0] = wp.vec2i(f, ori_0)

        hx1 = face_hex_indices[f][1]
        local_face_1 = face_hex_face_ori[f][2]
        ori_1 = face_hex_face_ori[f][3]

        hex_face_indices[hx1, local_face_1] = wp.vec2i(f, ori_1)


class HexmeshTripolynomialSpaceTopology(HexmeshSpaceTopology):
    def __init__(self, mesh: Hexmesh, shape: CubeTripolynomialShapeFunctions):
        super().__init__(mesh, shape, need_hex_edge_indices=shape.ORDER >= 2, need_hex_face_indices=shape.ORDER >= 2)

        self.element_node_index = self._make_element_node_index()

    def node_count(self) -> int:
        ORDER = self.shape.ORDER
        INTERIOR_NODES_PER_EDGE = max(0, ORDER - 1)
        INTERIOR_NODES_PER_FACE = INTERIOR_NODES_PER_EDGE**2
        INTERIOR_NODES_PER_CELL = INTERIOR_NODES_PER_EDGE**3

        return (
            self._mesh.vertex_count()
            + self._mesh.edge_count() * INTERIOR_NODES_PER_EDGE
            + self._mesh.side_count() * INTERIOR_NODES_PER_FACE
            + self._mesh.cell_count() * INTERIOR_NODES_PER_CELL
        )

    @wp.func
    def _rotate_face_index(type_index: int, ori: int, size: int):
        i = type_index // size
        j = type_index - i * size
        coords = wp.vec2i(i, j)

        fv = ori // 2

        # face indices from shape function always have positive orientation, drop `ori % 2`
        rot_i = wp.dot(_FACE_ORIENTATION_I[4 * fv], coords) + _FACE_TRANSLATION_I[fv, 0]
        rot_j = wp.dot(_FACE_ORIENTATION_I[4 * fv + 1], coords) + _FACE_TRANSLATION_I[fv, 1]

        return rot_i * size + rot_j

    def _make_element_node_index(self):
        ORDER = self.shape.ORDER
        INTERIOR_NODES_PER_EDGE = wp.constant(max(0, ORDER - 1))
        INTERIOR_NODES_PER_FACE = wp.constant(INTERIOR_NODES_PER_EDGE**2)
        INTERIOR_NODES_PER_CELL = wp.constant(INTERIOR_NODES_PER_EDGE**3)

        @cache.dynamic_func(suffix=self.name)
        def element_node_index(
            geo_arg: Hexmesh.CellArg,
            topo_arg: HexmeshTopologyArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            node_type, type_instance, type_index = self.shape.node_type_and_type_index(node_index_in_elt)

            if node_type == CubeTripolynomialShapeFunctions.VERTEX:
                return geo_arg.hex_vertex_indices[element_index, _CUBE_TO_HEX_VERTEX[type_instance]]

            offset = topo_arg.vertex_count

            if node_type == CubeTripolynomialShapeFunctions.EDGE:
                hex_edge = _CUBE_TO_HEX_EDGE[type_instance]
                edge_index = topo_arg.hex_edge_indices[element_index, hex_edge]

                v0 = geo_arg.hex_vertex_indices[element_index, EDGE_VERTEX_INDICES[hex_edge, 0]]
                v1 = geo_arg.hex_vertex_indices[element_index, EDGE_VERTEX_INDICES[hex_edge, 1]]

                if v0 > v1:
                    type_index = ORDER - 1 - type_index

                return offset + INTERIOR_NODES_PER_EDGE * edge_index + type_index

            offset += INTERIOR_NODES_PER_EDGE * topo_arg.edge_count

            if node_type == CubeTripolynomialShapeFunctions.FACE:
                face_index_and_ori = topo_arg.hex_face_indices[element_index, type_instance]
                face_index = face_index_and_ori[0]
                face_orientation = face_index_and_ori[1]

                type_index = HexmeshTripolynomialSpaceTopology._rotate_face_index(
                    type_index, face_orientation, ORDER - 1
                )

                return offset + INTERIOR_NODES_PER_FACE * face_index + type_index

            offset += INTERIOR_NODES_PER_FACE * topo_arg.face_count

            return offset + INTERIOR_NODES_PER_CELL * element_index + type_index

        return element_node_index


class HexmeshSerendipitySpaceTopology(HexmeshSpaceTopology):
    def __init__(
        self,
        grid: Hexmesh,
        shape: CubeSerendipityShapeFunctions,
    ):
        super().__init__(grid, shape, need_hex_edge_indices=True, need_hex_face_indices=False)

        self.element_node_index = self._make_element_node_index()

    def node_count(self) -> int:
        return self.geometry.vertex_count() + (self.shape.ORDER - 1) * self.geometry.edge_count()

    def _make_element_node_index(self):
        ORDER = self.shape.ORDER

        @cache.dynamic_func(suffix=self.name)
        def element_node_index(
            cell_arg: Hexmesh.CellArg,
            topo_arg: HexmeshSpaceTopology.TopologyArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            node_type, type_index = self.shape.node_type_and_type_index(node_index_in_elt)

            if node_type == CubeSerendipityShapeFunctions.VERTEX:
                return cell_arg.hex_vertex_indices[element_index, _CUBE_TO_HEX_VERTEX[type_index]]

            type_instance, index_in_edge = CubeSerendipityShapeFunctions._cube_edge_index(node_type, type_index)
            hex_edge = _CUBE_TO_HEX_EDGE[type_instance]

            edge_index = topo_arg.hex_edge_indices[element_index, hex_edge]

            v0 = cell_arg.hex_vertex_indices[element_index, EDGE_VERTEX_INDICES[hex_edge, 0]]
            v1 = cell_arg.hex_vertex_indices[element_index, EDGE_VERTEX_INDICES[hex_edge, 1]]

            if v0 > v1:
                index_in_edge = ORDER - 1 - index_in_edge

            return topo_arg.vertex_count + (ORDER - 1) * edge_index + index_in_edge

        return element_node_index


def make_hexmesh_space_topology(mesh: Hexmesh, shape: ShapeFunction):
    if isinstance(shape, CubeSerendipityShapeFunctions):
        return forward_base_topology(HexmeshSerendipitySpaceTopology, mesh, shape)

    if isinstance(shape, CubeTripolynomialShapeFunctions):
        return forward_base_topology(HexmeshTripolynomialSpaceTopology, mesh, shape)

    raise ValueError(f"Unsupported shape function {shape.name}")
