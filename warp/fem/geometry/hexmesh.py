from typing import Optional

import warp as wp
from warp.fem.cache import (
    TemporaryStore,
    borrow_temporary,
    borrow_temporary_like,
    cached_arg_value,
)
from warp.fem.types import OUTSIDE, Coords, ElementIndex, Sample, make_free_sample

from .element import Cube, Square
from .geometry import Geometry


@wp.struct
class HexmeshCellArg:
    hex_vertex_indices: wp.array2d(dtype=int)
    positions: wp.array(dtype=wp.vec3)

    # for neighbor cell lookup
    vertex_hex_offsets: wp.array(dtype=int)
    vertex_hex_indices: wp.array(dtype=int)


@wp.struct
class HexmeshSideArg:
    cell_arg: HexmeshCellArg
    face_vertex_indices: wp.array(dtype=wp.vec4i)
    face_hex_indices: wp.array(dtype=wp.vec2i)
    face_hex_face_orientation: wp.array(dtype=wp.vec4i)


_mat32 = wp.mat(shape=(3, 2), dtype=float)

FACE_VERTEX_INDICES = wp.constant(
    wp.mat(shape=(6, 4), dtype=int)(
        [
            [0, 4, 7, 3],  # x = 0
            [1, 2, 6, 5],  # x = 1
            [0, 1, 5, 4],  # y = 0
            [3, 7, 6, 2],  # y = 1
            [0, 3, 2, 1],  # z = 0
            [4, 5, 6, 7],  # z = 1
        ]
    )
)

EDGE_VERTEX_INDICES = wp.constant(
    wp.mat(shape=(12, 2), dtype=int)(
        [
            [0, 1],
            [1, 2],
            [3, 2],
            [0, 3],
            [4, 5],
            [5, 6],
            [7, 6],
            [4, 7],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ]
    )
)

# orthogonal transform for face coordinates given first vertex + winding
# (two rows per entry)

FACE_ORIENTATION = [
    [1, 0],  # FV: 0, det: +
    [0, 1],
    [0, 1],  # FV: 0, det: -
    [1, 0],
    [0, -1],  # FV: 1, det: +
    [1, 0],
    [-1, 0],  # FV: 1, det: -
    [0, 1],
    [-1, 0],  # FV: 2, det: +
    [0, -1],
    [0, -1],  # FV: 2, det: -
    [-1, 0],
    [0, 1],  # FV: 3, det: +
    [-1, 0],
    [1, 0],  # FV: 3, det: -
    [0, -1],
]

FACE_TRANSLATION = [
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1],
]

# local face coordinate system
_FACE_COORD_INDICES = wp.constant(
    wp.mat(shape=(6, 4), dtype=int)(
        [
            [2, 1, 0, 0],  # 0: z y -x
            [1, 2, 0, 1],  # 1: y z  x-1
            [0, 2, 1, 0],  # 2: x z -y
            [2, 0, 1, 1],  # 3: z x  y-1
            [1, 0, 2, 0],  # 4: y x -z
            [0, 1, 2, 1],  # 5: x y  z-1
        ]
    )
)

_FACE_ORIENTATION_F = wp.constant(wp.mat(shape=(16, 2), dtype=float)(FACE_ORIENTATION))
_FACE_TRANSLATION_F = wp.constant(wp.mat(shape=(4, 2), dtype=float)(FACE_TRANSLATION))


class Hexmesh(Geometry):
    """Hexahedral mesh geometry"""

    dimension = 3

    def __init__(
        self, hex_vertex_indices: wp.array, positions: wp.array, temporary_store: Optional[TemporaryStore] = None
    ):
        """
        Constructs a tetrahedral mesh.

        Args:
            hex_vertex_indices: warp array of shape (num_hexes, 8) containing vertex indices for each hex
                following standard ordering (bottom face vertices in counter-clockwise order, then similarly for upper face)
            positions: warp array of shape (num_vertices, 3) containing 3d position for each vertex
            temporary_store: shared pool from which to allocate temporary arrays
        """

        self.hex_vertex_indices = hex_vertex_indices
        self.positions = positions

        self._face_vertex_indices: wp.array = None
        self._face_hex_indices: wp.array = None
        self._face_hex_face_orientation: wp.array = None
        self._vertex_hex_offsets: wp.array = None
        self._vertex_hex_indices: wp.array = None
        self._hex_edge_indices: wp.array = None
        self._edge_count = 0
        self._build_topology(temporary_store)

    def cell_count(self):
        return self.hex_vertex_indices.shape[0]

    def vertex_count(self):
        return self.positions.shape[0]

    def side_count(self):
        return self._face_vertex_indices.shape[0]

    def edge_count(self):
        if self._hex_edge_indices is None:
            self._compute_hex_edges()
        return self._edge_count

    def boundary_side_count(self):
        return self._boundary_face_indices.shape[0]

    def reference_cell(self) -> Cube:
        return Cube()

    def reference_side(self) -> Square:
        return Square()

    @property
    def hex_edge_indices(self) -> wp.array:
        if self._hex_edge_indices is None:
            self._compute_hex_edges()
        return self._hex_edge_indices

    @property
    def face_hex_indices(self) -> wp.array:
        return self._face_hex_indices

    @property
    def face_vertex_indices(self) -> wp.array:
        return self._face_vertex_indices

    CellArg = HexmeshCellArg
    SideArg = HexmeshSideArg

    @wp.struct
    class SideIndexArg:
        boundary_face_indices: wp.array(dtype=int)

    # Geometry device interface

    @cached_arg_value
    def cell_arg_value(self, device) -> CellArg:
        args = self.CellArg()

        args.hex_vertex_indices = self.hex_vertex_indices.to(device)
        args.positions = self.positions.to(device)
        args.vertex_hex_offsets = self._vertex_hex_offsets.to(device)
        args.vertex_hex_indices = self._vertex_hex_indices.to(device)

        return args

    @wp.func
    def cell_position(args: CellArg, s: Sample):
        hex_idx = args.hex_vertex_indices[s.element_index]

        w_p = s.element_coords
        w_m = Coords(1.0) - s.element_coords

        # 0 : m m m
        # 1 : p m m
        # 2 : p p m
        # 3 : m p m
        # 4 : m m p
        # 5 : p m p
        # 6 : p p p
        # 7 : m p p

        return (
            w_m[0] * w_m[1] * w_m[2] * args.positions[hex_idx[0]]
            + w_p[0] * w_m[1] * w_m[2] * args.positions[hex_idx[1]]
            + w_p[0] * w_p[1] * w_m[2] * args.positions[hex_idx[2]]
            + w_m[0] * w_p[1] * w_m[2] * args.positions[hex_idx[3]]
            + w_m[0] * w_m[1] * w_p[2] * args.positions[hex_idx[4]]
            + w_p[0] * w_m[1] * w_p[2] * args.positions[hex_idx[5]]
            + w_p[0] * w_p[1] * w_p[2] * args.positions[hex_idx[6]]
            + w_m[0] * w_p[1] * w_p[2] * args.positions[hex_idx[7]]
        )

    @wp.func
    def cell_deformation_gradient(cell_arg: CellArg, s: Sample):
        """Deformation gradient at `coords`"""
        """Transposed deformation gradient at `coords`"""
        hex_idx = cell_arg.hex_vertex_indices[s.element_index]

        w_p = s.element_coords
        w_m = Coords(1.0) - s.element_coords

        return (
            wp.outer(cell_arg.positions[hex_idx[0]], wp.vec3(-w_m[1] * w_m[2], -w_m[0] * w_m[2], -w_m[0] * w_m[1]))
            + wp.outer(cell_arg.positions[hex_idx[1]], wp.vec3(w_m[1] * w_m[2], -w_p[0] * w_m[2], -w_p[0] * w_m[1]))
            + wp.outer(cell_arg.positions[hex_idx[2]], wp.vec3(w_p[1] * w_m[2], w_p[0] * w_m[2], -w_p[0] * w_p[1]))
            + wp.outer(cell_arg.positions[hex_idx[3]], wp.vec3(-w_p[1] * w_m[2], w_m[0] * w_m[2], -w_m[0] * w_p[1]))
            + wp.outer(cell_arg.positions[hex_idx[4]], wp.vec3(-w_m[1] * w_p[2], -w_m[0] * w_p[2], w_m[0] * w_m[1]))
            + wp.outer(cell_arg.positions[hex_idx[5]], wp.vec3(w_m[1] * w_p[2], -w_p[0] * w_p[2], w_p[0] * w_m[1]))
            + wp.outer(cell_arg.positions[hex_idx[6]], wp.vec3(w_p[1] * w_p[2], w_p[0] * w_p[2], w_p[0] * w_p[1]))
            + wp.outer(cell_arg.positions[hex_idx[7]], wp.vec3(-w_p[1] * w_p[2], w_m[0] * w_p[2], w_m[0] * w_p[1]))
        )

    @wp.func
    def cell_inverse_deformation_gradient(cell_arg: CellArg, s: Sample):
        return wp.inverse(Hexmesh.cell_deformation_gradient(cell_arg, s))

    @wp.func
    def cell_measure(args: CellArg, s: Sample):
        return wp.abs(wp.determinant(Hexmesh.cell_deformation_gradient(args, s)))

    @wp.func
    def cell_normal(args: CellArg, s: Sample):
        return wp.vec3(0.0)

    @cached_arg_value
    def side_index_arg_value(self, device) -> SideIndexArg:
        args = self.SideIndexArg()

        args.boundary_face_indices = self._boundary_face_indices.to(device)

        return args

    @wp.func
    def boundary_side_index(args: SideIndexArg, boundary_side_index: int):
        """Boundary side to side index"""

        return args.boundary_face_indices[boundary_side_index]

    @cached_arg_value
    def side_arg_value(self, device) -> CellArg:
        args = self.SideArg()

        args.cell_arg = self.cell_arg_value(device)
        args.face_vertex_indices = self._face_vertex_indices.to(device)
        args.face_hex_indices = self._face_hex_indices.to(device)
        args.face_hex_face_orientation = self._face_hex_face_orientation.to(device)

        return args

    @wp.func
    def side_position(args: SideArg, s: Sample):
        face_idx = args.face_vertex_indices[s.element_index]

        w_p = s.element_coords
        w_m = Coords(1.0) - s.element_coords

        return (
            w_m[0] * w_m[1] * args.cell_arg.positions[face_idx[0]]
            + w_p[0] * w_m[1] * args.cell_arg.positions[face_idx[1]]
            + w_p[0] * w_p[1] * args.cell_arg.positions[face_idx[2]]
            + w_m[0] * w_p[1] * args.cell_arg.positions[face_idx[3]]
        )

    @wp.func
    def _side_deformation_vecs(args: SideArg, side_index: ElementIndex, coords: Coords):
        face_idx = args.face_vertex_indices[side_index]

        p0 = args.cell_arg.positions[face_idx[0]]
        p1 = args.cell_arg.positions[face_idx[1]]
        p2 = args.cell_arg.positions[face_idx[2]]
        p3 = args.cell_arg.positions[face_idx[3]]

        w_p = coords
        w_m = Coords(1.0) - coords

        v1 = w_m[1] * (p1 - p0) + w_p[1] * (p2 - p3)
        v2 = w_p[0] * (p2 - p1) + w_m[0] * (p3 - p0)
        return v1, v2

    @wp.func
    def side_deformation_gradient(args: SideArg, s: Sample):
        """Transposed side deformation gradient at `coords`"""
        v1, v2 = Hexmesh._side_deformation_vecs(args, s.element_index, s.element_coords)
        return _mat32(v1, v2)

    @wp.func
    def side_inner_inverse_deformation_gradient(args: SideArg, s: Sample):
        cell_index = Hexmesh.side_inner_cell_index(args, s.element_index)
        cell_coords = Hexmesh.side_inner_cell_coords(args, s.element_index, s.element_coords)
        return Hexmesh.cell_inverse_deformation_gradient(args.cell_arg, make_free_sample(cell_index, cell_coords))

    @wp.func
    def side_outer_inverse_deformation_gradient(args: SideArg, s: Sample):
        cell_index = Hexmesh.side_outer_cell_index(args, s.element_index)
        cell_coords = Hexmesh.side_outer_cell_coords(args, s.element_index, s.element_coords)
        return Hexmesh.cell_inverse_deformation_gradient(args.cell_arg, make_free_sample(cell_index, cell_coords))

    @wp.func
    def side_measure(args: SideArg, s: Sample):
        v1, v2 = Hexmesh._side_deformation_vecs(args, s.element_index, s.element_coords)
        return wp.length(wp.cross(v1, v2))

    @wp.func
    def side_measure_ratio(args: SideArg, s: Sample):
        inner = Hexmesh.side_inner_cell_index(args, s.element_index)
        outer = Hexmesh.side_outer_cell_index(args, s.element_index)
        inner_coords = Hexmesh.side_inner_cell_coords(args, s.element_index, s.element_coords)
        outer_coords = Hexmesh.side_outer_cell_coords(args, s.element_index, s.element_coords)
        return Hexmesh.side_measure(args, s) / wp.min(
            Hexmesh.cell_measure(args.cell_arg, make_free_sample(inner, inner_coords)),
            Hexmesh.cell_measure(args.cell_arg, make_free_sample(outer, outer_coords)),
        )

    @wp.func
    def side_normal(args: SideArg, s: Sample):
        v1, v2 = Hexmesh._side_deformation_vecs(args, s.element_index, s.element_coords)
        return wp.normalize(wp.cross(v1, v2))

    @wp.func
    def side_inner_cell_index(arg: SideArg, side_index: ElementIndex):
        return arg.face_hex_indices[side_index][0]

    @wp.func
    def side_outer_cell_index(arg: SideArg, side_index: ElementIndex):
        return arg.face_hex_indices[side_index][1]

    @wp.func
    def _hex_local_face_coords(hex_coords: Coords, face_index: int):
        # Coordinatex in local face coordinates system
        # Sign of last coordinate (out of face)

        face_coords = wp.vec2(
            hex_coords[_FACE_COORD_INDICES[face_index, 0]], hex_coords[_FACE_COORD_INDICES[face_index, 1]]
        )

        normal_coord = hex_coords[_FACE_COORD_INDICES[face_index, 2]]
        normal_coord = wp.select(_FACE_COORD_INDICES[face_index, 3] == 0, normal_coord - 1.0, -normal_coord)

        return face_coords, normal_coord

    @wp.func
    def _local_face_hex_coords(face_coords: wp.vec2, face_index: int):
        # Coordinates in hex from local face coordinates system

        hex_coords = Coords()
        hex_coords[_FACE_COORD_INDICES[face_index, 0]] = face_coords[0]
        hex_coords[_FACE_COORD_INDICES[face_index, 1]] = face_coords[1]
        hex_coords[_FACE_COORD_INDICES[face_index, 2]] = wp.select(_FACE_COORD_INDICES[face_index, 3] == 0, 1.0, 0.0)

        return hex_coords

    @wp.func
    def _local_from_oriented_face_coords(ori: int, oriented_coords: Coords):
        fv = ori // 2
        return (oriented_coords[0] - _FACE_TRANSLATION_F[fv, 0]) * _FACE_ORIENTATION_F[2 * ori] + (
            oriented_coords[1] - _FACE_TRANSLATION_F[fv, 1]
        ) * _FACE_ORIENTATION_F[2 * ori + 1]

    @wp.func
    def _local_to_oriented_face_coords(ori: int, coords: wp.vec2):
        fv = ori // 2
        return Coords(
            wp.dot(_FACE_ORIENTATION_F[2 * ori], coords) + _FACE_TRANSLATION_F[fv, 0],
            wp.dot(_FACE_ORIENTATION_F[2 * ori + 1], coords) + _FACE_TRANSLATION_F[fv, 1],
            0.0,
        )

    @wp.func
    def face_to_hex_coords(local_face_index: int, face_orientation: int, side_coords: Coords):
        local_coords = Hexmesh._local_from_oriented_face_coords(face_orientation, side_coords)
        return Hexmesh._local_face_hex_coords(local_coords, local_face_index)

    @wp.func
    def side_inner_cell_coords(args: SideArg, side_index: ElementIndex, side_coords: Coords):
        local_face_index = args.face_hex_face_orientation[side_index][0]
        face_orientation = args.face_hex_face_orientation[side_index][1]

        return Hexmesh.face_to_hex_coords(local_face_index, face_orientation, side_coords)

    @wp.func
    def side_outer_cell_coords(args: SideArg, side_index: ElementIndex, side_coords: Coords):
        local_face_index = args.face_hex_face_orientation[side_index][2]
        face_orientation = args.face_hex_face_orientation[side_index][3]

        return Hexmesh.face_to_hex_coords(local_face_index, face_orientation, side_coords)

    @wp.func
    def side_from_cell_coords(args: SideArg, side_index: ElementIndex, hex_index: ElementIndex, hex_coords: Coords):
        if Hexmesh.side_inner_cell_index(args, side_index) == hex_index:
            local_face_index = args.face_hex_face_orientation[side_index][0]
            face_orientation = args.face_hex_face_orientation[side_index][1]
        else:
            local_face_index = args.face_hex_face_orientation[side_index][2]
            face_orientation = args.face_hex_face_orientation[side_index][3]

        face_coords, normal_coord = Hexmesh._hex_local_face_coords(hex_coords, local_face_index)
        return wp.select(
            normal_coord == 0.0, Coords(OUTSIDE), Hexmesh._local_to_oriented_face_coords(face_orientation, face_coords)
        )

    @wp.func
    def side_to_cell_arg(side_arg: SideArg):
        return side_arg.cell_arg

    def _build_topology(self, temporary_store: TemporaryStore):
        from warp.fem.utils import compress_node_indices, host_read_at_index, masked_indices
        from warp.utils import array_scan

        device = self.hex_vertex_indices.device

        vertex_hex_offsets, vertex_hex_indices = compress_node_indices(
            self.vertex_count(), self.hex_vertex_indices, temporary_store=temporary_store
        )
        self._vertex_hex_offsets = vertex_hex_offsets.detach()
        self._vertex_hex_indices = vertex_hex_indices.detach()

        vertex_start_face_count = borrow_temporary(temporary_store, dtype=int, device=device, shape=self.vertex_count())
        vertex_start_face_count.array.zero_()
        vertex_start_face_offsets = borrow_temporary_like(vertex_start_face_count, temporary_store=temporary_store)

        vertex_face_other_vs = borrow_temporary(
            temporary_store, dtype=wp.vec3i, device=device, shape=(8 * self.cell_count())
        )
        vertex_face_hexes = borrow_temporary(
            temporary_store, dtype=int, device=device, shape=(8 * self.cell_count(), 2)
        )

        # Count face edges starting at each vertex
        wp.launch(
            kernel=Hexmesh._count_starting_faces_kernel,
            device=device,
            dim=self.cell_count(),
            inputs=[self.hex_vertex_indices, vertex_start_face_count.array],
        )

        array_scan(in_array=vertex_start_face_count.array, out_array=vertex_start_face_offsets.array, inclusive=False)

        # Count number of unique edges (deduplicate across faces)
        vertex_unique_face_count = vertex_start_face_count
        wp.launch(
            kernel=Hexmesh._count_unique_starting_faces_kernel,
            device=device,
            dim=self.vertex_count(),
            inputs=[
                self._vertex_hex_offsets,
                self._vertex_hex_indices,
                self.hex_vertex_indices,
                vertex_start_face_offsets.array,
                vertex_unique_face_count.array,
                vertex_face_other_vs.array,
                vertex_face_hexes.array,
            ],
        )

        vertex_unique_face_offsets = borrow_temporary_like(vertex_start_face_offsets, temporary_store=temporary_store)
        array_scan(in_array=vertex_start_face_count.array, out_array=vertex_unique_face_offsets.array, inclusive=False)

        # Get back edge count to host
        face_count = int(
            host_read_at_index(
                vertex_unique_face_offsets.array, self.vertex_count() - 1, temporary_store=temporary_store
            )
        )

        self._face_vertex_indices = wp.empty(shape=(face_count,), dtype=wp.vec4i, device=device)
        self._face_hex_indices = wp.empty(shape=(face_count,), dtype=wp.vec2i, device=device)
        self._face_hex_face_orientation = wp.empty(shape=(face_count,), dtype=wp.vec4i, device=device)

        boundary_mask = borrow_temporary(temporary_store, shape=(face_count,), dtype=int, device=device)

        # Compress edge data
        wp.launch(
            kernel=Hexmesh._compress_faces_kernel,
            device=device,
            dim=self.vertex_count(),
            inputs=[
                vertex_start_face_offsets.array,
                vertex_unique_face_offsets.array,
                vertex_unique_face_count.array,
                vertex_face_other_vs.array,
                vertex_face_hexes.array,
                self._face_vertex_indices,
                self._face_hex_indices,
                boundary_mask.array,
            ],
        )

        vertex_start_face_offsets.release()
        vertex_unique_face_offsets.release()
        vertex_unique_face_count.release()
        vertex_face_other_vs.release()
        vertex_face_hexes.release()

        # Flip normals if necessary
        wp.launch(
            kernel=Hexmesh._flip_face_normals,
            device=device,
            dim=self.side_count(),
            inputs=[self._face_vertex_indices, self._face_hex_indices, self.hex_vertex_indices, self.positions],
        )

        # Compute and store face orientation
        wp.launch(
            kernel=Hexmesh._compute_face_orientation,
            device=device,
            dim=self.side_count(),
            inputs=[
                self._face_vertex_indices,
                self._face_hex_indices,
                self.hex_vertex_indices,
                self._face_hex_face_orientation,
            ],
        )

        boundary_face_indices, _ = masked_indices(boundary_mask.array)
        self._boundary_face_indices = boundary_face_indices.detach()

    def _compute_hex_edges(self, temporary_store: Optional[TemporaryStore] = None):
        from warp.fem.utils import host_read_at_index
        from warp.utils import array_scan

        device = self.hex_vertex_indices.device

        vertex_start_edge_count = borrow_temporary(temporary_store, dtype=int, device=device, shape=self.vertex_count())
        vertex_start_edge_count.array.zero_()
        vertex_start_edge_offsets = borrow_temporary_like(vertex_start_edge_count, temporary_store=temporary_store)

        vertex_edge_ends = borrow_temporary(temporary_store, dtype=int, device=device, shape=(12 * self.cell_count()))

        # Count face edges starting at each vertex
        wp.launch(
            kernel=Hexmesh._count_starting_edges_kernel,
            device=device,
            dim=self.cell_count(),
            inputs=[self.hex_vertex_indices, vertex_start_edge_count.array],
        )

        array_scan(in_array=vertex_start_edge_count.array, out_array=vertex_start_edge_offsets.array, inclusive=False)

        # Count number of unique edges (deduplicate across faces)
        vertex_unique_edge_count = vertex_start_edge_count
        wp.launch(
            kernel=Hexmesh._count_unique_starting_edges_kernel,
            device=device,
            dim=self.vertex_count(),
            inputs=[
                self._vertex_hex_offsets,
                self._vertex_hex_indices,
                self.hex_vertex_indices,
                vertex_start_edge_offsets.array,
                vertex_unique_edge_count.array,
                vertex_edge_ends.array,
            ],
        )

        vertex_unique_edge_offsets = borrow_temporary_like(
            vertex_start_edge_offsets.array, temporary_store=temporary_store
        )
        array_scan(in_array=vertex_start_edge_count.array, out_array=vertex_unique_edge_offsets.array, inclusive=False)

        # Get back edge count to host
        self._edge_count = int(
            host_read_at_index(
                vertex_unique_edge_offsets.array, self.vertex_count() - 1, temporary_store=temporary_store
            )
        )

        self._hex_edge_indices = wp.empty(
            dtype=int, device=self.hex_vertex_indices.device, shape=(self.cell_count(), 12)
        )

        # Compress edge data
        wp.launch(
            kernel=Hexmesh._compress_edges_kernel,
            device=device,
            dim=self.vertex_count(),
            inputs=[
                self._vertex_hex_offsets,
                self._vertex_hex_indices,
                self.hex_vertex_indices,
                vertex_start_edge_offsets.array,
                vertex_unique_edge_offsets.array,
                vertex_unique_edge_count.array,
                vertex_edge_ends.array,
                self._hex_edge_indices,
            ],
        )

        vertex_start_edge_offsets.release()
        vertex_unique_edge_offsets.release()
        vertex_unique_edge_count.release()
        vertex_edge_ends.release()

    @wp.kernel
    def _count_starting_faces_kernel(
        hex_vertex_indices: wp.array2d(dtype=int), vertex_start_face_count: wp.array(dtype=int)
    ):
        t = wp.tid()
        for k in range(6):
            vi = wp.vec4i(
                hex_vertex_indices[t, FACE_VERTEX_INDICES[k, 0]],
                hex_vertex_indices[t, FACE_VERTEX_INDICES[k, 1]],
                hex_vertex_indices[t, FACE_VERTEX_INDICES[k, 2]],
                hex_vertex_indices[t, FACE_VERTEX_INDICES[k, 3]],
            )
            vm = wp.min(vi)

            for i in range(4):
                if vm == vi[i]:
                    wp.atomic_add(vertex_start_face_count, vm, 1)

    @wp.func
    def _face_sort(vidx: wp.vec4i, min_k: int):
        v1 = vidx[(min_k + 1) % 4]
        v2 = vidx[(min_k + 2) % 4]
        v3 = vidx[(min_k + 3) % 4]

        if v1 < v3:
            return wp.vec3i(v1, v2, v3)
        return wp.vec3i(v3, v2, v1)

    @wp.func
    def _find_face(
        needle: wp.vec3i,
        values: wp.array(dtype=wp.vec3i),
        beg: int,
        end: int,
    ):
        for i in range(beg, end):
            if values[i] == needle:
                return i

        return -1

    @wp.kernel
    def _count_unique_starting_faces_kernel(
        vertex_hex_offsets: wp.array(dtype=int),
        vertex_hex_indices: wp.array(dtype=int),
        hex_vertex_indices: wp.array2d(dtype=int),
        vertex_start_face_offsets: wp.array(dtype=int),
        vertex_start_face_count: wp.array(dtype=int),
        face_other_vs: wp.array(dtype=wp.vec3i),
        face_hexes: wp.array2d(dtype=int),
    ):
        v = wp.tid()

        face_beg = vertex_start_face_offsets[v]

        hex_beg = vertex_hex_offsets[v]
        hex_end = vertex_hex_offsets[v + 1]

        face_cur = face_beg

        for hexa in range(hex_beg, hex_end):
            hx = vertex_hex_indices[hexa]

            for k in range(6):
                vi = wp.vec4i(
                    hex_vertex_indices[hx, FACE_VERTEX_INDICES[k, 0]],
                    hex_vertex_indices[hx, FACE_VERTEX_INDICES[k, 1]],
                    hex_vertex_indices[hx, FACE_VERTEX_INDICES[k, 2]],
                    hex_vertex_indices[hx, FACE_VERTEX_INDICES[k, 3]],
                )
                min_i = int(wp.argmin(vi))

                if v == vi[min_i]:
                    other_v = Hexmesh._face_sort(vi, min_i)

                    # Check if other_v has been seen
                    seen_idx = Hexmesh._find_face(other_v, face_other_vs, face_beg, face_cur)

                    if seen_idx == -1:
                        face_other_vs[face_cur] = other_v
                        face_hexes[face_cur, 0] = hx
                        face_hexes[face_cur, 1] = hx
                        face_cur += 1
                    else:
                        face_hexes[seen_idx, 1] = hx

        vertex_start_face_count[v] = face_cur - face_beg

    @wp.kernel
    def _compress_faces_kernel(
        vertex_start_face_offsets: wp.array(dtype=int),
        vertex_unique_face_offsets: wp.array(dtype=int),
        vertex_unique_face_count: wp.array(dtype=int),
        uncompressed_face_other_vs: wp.array(dtype=wp.vec3i),
        uncompressed_face_hexes: wp.array2d(dtype=int),
        face_vertex_indices: wp.array(dtype=wp.vec4i),
        face_hex_indices: wp.array(dtype=wp.vec2i),
        boundary_mask: wp.array(dtype=int),
    ):
        v = wp.tid()

        start_beg = vertex_start_face_offsets[v]
        unique_beg = vertex_unique_face_offsets[v]
        unique_count = vertex_unique_face_count[v]

        for f in range(unique_count):
            src_index = start_beg + f
            face_index = unique_beg + f

            face_vertex_indices[face_index] = wp.vec4i(
                v,
                uncompressed_face_other_vs[src_index][0],
                uncompressed_face_other_vs[src_index][1],
                uncompressed_face_other_vs[src_index][2],
            )

            hx0 = uncompressed_face_hexes[src_index, 0]
            hx1 = uncompressed_face_hexes[src_index, 1]
            face_hex_indices[face_index] = wp.vec2i(hx0, hx1)
            if hx0 == hx1:
                boundary_mask[face_index] = 1
            else:
                boundary_mask[face_index] = 0

    @wp.kernel
    def _flip_face_normals(
        face_vertex_indices: wp.array(dtype=wp.vec4i),
        face_hex_indices: wp.array(dtype=wp.vec2i),
        hex_vertex_indices: wp.array2d(dtype=int),
        positions: wp.array(dtype=wp.vec3),
    ):
        f = wp.tid()

        hexa = face_hex_indices[f][0]

        hex_vidx = hex_vertex_indices[hexa]
        face_vidx = face_vertex_indices[f]

        hex_centroid = (
            positions[hex_vidx[0]]
            + positions[hex_vidx[1]]
            + positions[hex_vidx[2]]
            + positions[hex_vidx[3]]
            + positions[hex_vidx[4]]
            + positions[hex_vidx[5]]
            + positions[hex_vidx[6]]
            + positions[hex_vidx[7]]
        ) / 8.0

        v0 = positions[face_vidx[0]]
        v1 = positions[face_vidx[1]]
        v2 = positions[face_vidx[2]]
        v3 = positions[face_vidx[3]]

        face_center = (v1 + v0 + v2 + v3) / 4.0
        face_normal = wp.cross(v2 - v0, v3 - v1)

        # if face normal points toward first tet centroid, flip indices
        if wp.dot(hex_centroid - face_center, face_normal) > 0.0:
            face_vertex_indices[f] = wp.vec4i(face_vidx[0], face_vidx[3], face_vidx[2], face_vidx[1])

    @wp.func
    def _find_face_orientation(face_vidx: wp.vec4i, hex_index: int, hex_vertex_indices: wp.array2d(dtype=int)):
        hex_vidx = hex_vertex_indices[hex_index]

        # Find local index in hex corresponding to face

        face_min_i = int(wp.argmin(face_vidx))
        face_other_v = Hexmesh._face_sort(face_vidx, face_min_i)

        for k in range(6):
            hex_face_vi = wp.vec4i(
                hex_vidx[FACE_VERTEX_INDICES[k, 0]],
                hex_vidx[FACE_VERTEX_INDICES[k, 1]],
                hex_vidx[FACE_VERTEX_INDICES[k, 2]],
                hex_vidx[FACE_VERTEX_INDICES[k, 3]],
            )
            hex_min_i = int(wp.argmin(hex_face_vi))
            hex_other_v = Hexmesh._face_sort(hex_face_vi, hex_min_i)

            if hex_other_v == face_other_v:
                local_face_index = k
                break

        # Find starting vertex index
        for k in range(4):
            if face_vidx[k] == hex_face_vi[0]:
                face_orientation = 2 * k
                if face_vidx[(k + 1) % 4] != hex_face_vi[1]:
                    face_orientation += 1

        return local_face_index, face_orientation

    @wp.kernel
    def _compute_face_orientation(
        face_vertex_indices: wp.array(dtype=wp.vec4i),
        face_hex_indices: wp.array(dtype=wp.vec2i),
        hex_vertex_indices: wp.array2d(dtype=int),
        face_hex_face_ori: wp.array(dtype=wp.vec4i),
    ):
        f = wp.tid()

        face_vidx = face_vertex_indices[f]

        hx0 = face_hex_indices[f][0]
        local_face_0, ori_0 = Hexmesh._find_face_orientation(face_vidx, hx0, hex_vertex_indices)

        hx1 = face_hex_indices[f][1]
        if hx0 == hx1:
            face_hex_face_ori[f] = wp.vec4i(local_face_0, ori_0, local_face_0, ori_0)
        else:
            local_face_1, ori_1 = Hexmesh._find_face_orientation(face_vidx, hx1, hex_vertex_indices)
            face_hex_face_ori[f] = wp.vec4i(local_face_0, ori_0, local_face_1, ori_1)

    @wp.kernel
    def _count_starting_edges_kernel(
        hex_vertex_indices: wp.array2d(dtype=int), vertex_start_edge_count: wp.array(dtype=int)
    ):
        t = wp.tid()
        for k in range(12):
            v0 = hex_vertex_indices[t, EDGE_VERTEX_INDICES[k, 0]]
            v1 = hex_vertex_indices[t, EDGE_VERTEX_INDICES[k, 1]]

            if v0 < v1:
                wp.atomic_add(vertex_start_edge_count, v0, 1)
            else:
                wp.atomic_add(vertex_start_edge_count, v1, 1)

    @wp.func
    def _find_edge(
        needle: int,
        values: wp.array(dtype=int),
        beg: int,
        end: int,
    ):
        for i in range(beg, end):
            if values[i] == needle:
                return i

        return -1

    @wp.kernel
    def _count_unique_starting_edges_kernel(
        vertex_hex_offsets: wp.array(dtype=int),
        vertex_hex_indices: wp.array(dtype=int),
        hex_vertex_indices: wp.array2d(dtype=int),
        vertex_start_edge_offsets: wp.array(dtype=int),
        vertex_start_edge_count: wp.array(dtype=int),
        edge_ends: wp.array(dtype=int),
    ):
        v = wp.tid()

        edge_beg = vertex_start_edge_offsets[v]

        hex_beg = vertex_hex_offsets[v]
        hex_end = vertex_hex_offsets[v + 1]

        edge_cur = edge_beg

        for tet in range(hex_beg, hex_end):
            t = vertex_hex_indices[tet]

            for k in range(12):
                v0 = hex_vertex_indices[t, EDGE_VERTEX_INDICES[k, 0]]
                v1 = hex_vertex_indices[t, EDGE_VERTEX_INDICES[k, 1]]

                if v == wp.min(v0, v1):
                    other_v = wp.max(v0, v1)
                    if Hexmesh._find_edge(other_v, edge_ends, edge_beg, edge_cur) == -1:
                        edge_ends[edge_cur] = other_v
                        edge_cur += 1

        vertex_start_edge_count[v] = edge_cur - edge_beg

    @wp.kernel
    def _compress_edges_kernel(
        vertex_hex_offsets: wp.array(dtype=int),
        vertex_hex_indices: wp.array(dtype=int),
        hex_vertex_indices: wp.array2d(dtype=int),
        vertex_start_edge_offsets: wp.array(dtype=int),
        vertex_unique_edge_offsets: wp.array(dtype=int),
        vertex_unique_edge_count: wp.array(dtype=int),
        uncompressed_edge_ends: wp.array(dtype=int),
        hex_edge_indices: wp.array2d(dtype=int),
    ):
        v = wp.tid()

        uncompressed_beg = vertex_start_edge_offsets[v]

        unique_beg = vertex_unique_edge_offsets[v]
        unique_count = vertex_unique_edge_count[v]

        hex_beg = vertex_hex_offsets[v]
        hex_end = vertex_hex_offsets[v + 1]

        for tet in range(hex_beg, hex_end):
            t = vertex_hex_indices[tet]

            for k in range(12):
                v0 = hex_vertex_indices[t, EDGE_VERTEX_INDICES[k, 0]]
                v1 = hex_vertex_indices[t, EDGE_VERTEX_INDICES[k, 1]]

                if v == wp.min(v0, v1):
                    other_v = wp.max(v0, v1)
                    edge_id = (
                        Hexmesh._find_edge(
                            other_v, uncompressed_edge_ends, uncompressed_beg, uncompressed_beg + unique_count
                        )
                        - uncompressed_beg
                        + unique_beg
                    )
                    hex_edge_indices[t][k] = edge_id
