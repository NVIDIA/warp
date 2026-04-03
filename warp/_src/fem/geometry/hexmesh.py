# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, ClassVar

import warp as wp
from warp._src.fem import cache
from warp._src.fem.cache import (
    TemporaryStore,
    borrow_temporary,
    borrow_temporary_like,
    cached_vec_type,
)
from warp._src.fem.types import (
    OUTSIDE,
    Coords,
    ElementIndex,
)
from warp._src.fem.utils import compress_node_indices, host_read_at_index, masked_indices
from warp._src.types import type_scalar_type
from warp._src.utils import array_scan

from .element import Element
from .geometry import Geometry

_wp_module_name_ = "warp.fem.geometry.hexmesh"


# Topology-only Arg structs (no position data, shared across precisions)
@wp.struct
class HexmeshCellArg:
    """Arguments for cell topology device functions."""

    hex_vertex_indices: wp.array2d(dtype=int)

    # for global cell lookup
    hex_bvh: wp.uint64


@wp.struct
class HexmeshSideArg:
    """Arguments for side topology device functions."""

    cell_arg: HexmeshCellArg
    face_vertex_indices: wp.array(dtype=wp.vec4i)
    face_hex_indices: wp.array(dtype=wp.vec2i)
    face_hex_face_orientation: wp.array(dtype=wp.vec4i)


def _make_hexmesh_cell_arg(topo_type, pos_vec_type):
    """Generate a CellArg struct for the given position vector type."""

    @cache.dynamic_struct(suffix=type_scalar_type(pos_vec_type))
    class _CellArg:
        topology: topo_type
        positions: wp.array(dtype=pos_vec_type)

    return _CellArg


def _make_hexmesh_side_arg(topo_type, pos_vec_type):
    """Generate a SideArg struct for the given position vector type."""

    @cache.dynamic_struct(suffix=type_scalar_type(pos_vec_type))
    class _SideArg:
        topology: topo_type
        positions: wp.array(dtype=pos_vec_type)

    return _SideArg


FACE_VERTEX_INDICES = wp.constant(
    wp.types.matrix(shape=(6, 4), dtype=int)(
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
    wp.types.matrix(shape=(12, 2), dtype=int)(
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
    wp.types.matrix(shape=(6, 4), dtype=int)(
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

_FACE_ORIENTATION_F = wp.constant(wp.types.matrix(shape=(16, 2), dtype=float)(FACE_ORIENTATION))
_FACE_TRANSLATION_F = wp.constant(wp.types.matrix(shape=(4, 2), dtype=float)(FACE_TRANSLATION))


class Hexmesh(Geometry):
    """Hexahedral mesh geometry."""

    dimension = 3

    # Class-level defaults (overridden per-instance in __init__)
    CellArg = HexmeshCellArg
    SideArg = HexmeshSideArg

    _dynamic_attribute_constructors: ClassVar = {
        # Hexmesh-specific functions that require closures (instance branching or struct construction)
        "cell_position": lambda obj: obj._make_hex_cell_position(),
        "cell_deformation_gradient": lambda obj: obj._make_hex_cell_deformation_gradient(),
        "side_to_cell_arg": lambda obj: obj._make_hex_side_to_cell_arg(),
        **Geometry._dynamic_attribute_constructors,
    }

    def __init__(
        self,
        hex_vertex_indices: wp.array,
        positions: wp.array,
        assume_parallelepiped_cells=False,
        build_bvh: bool = False,
        temporary_store: TemporaryStore | None = None,
    ):
        """Construct a hexahedral mesh.

        Args:
            hex_vertex_indices: warp array of shape (num_hexes, 8) containing vertex indices for each hex
                following standard ordering (bottom face vertices in counter-clockwise order, then similarly for upper face)
            positions: warp array of shape (num_vertices, 3) containing 3d position for each vertex
            assume_parallelepiped: If true, assume that all cells are parallelepipeds (cheaper position/gradient evaluations)
            build_bvh: Whether to also build the hex BVH, which is necessary for the global ``fem.lookup`` operator
            temporary_store: shared pool from which to allocate temporary arrays
        """

        self.hex_vertex_indices = hex_vertex_indices
        self.positions = positions
        self.parallelepiped_cells = assume_parallelepiped_cells

        # Infer scalar type from position array dtype
        self._scalar_type = type_scalar_type(positions.dtype)

        # Generate precision-appropriate Arg structs
        pos_vec = cached_vec_type(3, self._scalar_type)
        self.CellArg = _make_hexmesh_cell_arg(HexmeshCellArg, pos_vec)
        self.SideArg = _make_hexmesh_side_arg(HexmeshSideArg, pos_vec)

        self._face_vertex_indices: wp.array = None
        self._face_hex_indices: wp.array = None
        self._face_hex_face_orientation: wp.array = None
        self._vertex_hex_offsets: wp.array = None
        self._vertex_hex_indices: wp.array = None
        self._hex_edge_indices: wp.array = None
        self._edge_count = 0
        self._build_topology(temporary_store)

        # Process all dynamic attributes (Hexmesh primitives + Geometry dependents)
        cache.setup_dynamic_attributes(self)
        self.cell_coordinates = self._make_cell_coordinates(assume_linear=assume_parallelepiped_cells)
        self.side_coordinates = self._make_side_coordinates(assume_linear=assume_parallelepiped_cells)
        self.cell_closest_point = self._make_cell_closest_point(assume_linear=assume_parallelepiped_cells)
        self.side_closest_point = self._make_side_closest_point(assume_linear=assume_parallelepiped_cells)

        if build_bvh:
            self.build_bvh(self.positions.device)

    @property
    def scalar_type(self):
        return self._scalar_type

    def cell_count(self):
        """Number of cells in the mesh."""
        return self.hex_vertex_indices.shape[0]

    def vertex_count(self):
        """Number of vertices in the mesh."""
        return self.positions.shape[0]

    def side_count(self):
        """Number of sides in the mesh."""
        return self._face_vertex_indices.shape[0]

    def edge_count(self):
        """Number of edges in the mesh."""
        if self._hex_edge_indices is None:
            self._compute_hex_edges()
        return self._edge_count

    def boundary_side_count(self):
        """Number of boundary sides in the mesh."""
        return self._boundary_face_indices.shape[0]

    def reference_cell(self) -> Element:
        """Reference element for mesh cells."""
        return Element.CUBE

    def reference_side(self) -> Element:
        """Reference element for mesh sides."""
        return Element.SQUARE

    @property
    def hex_edge_indices(self) -> wp.array:
        """Edge indices for each hex element."""
        if self._hex_edge_indices is None:
            self._compute_hex_edges()
        return self._hex_edge_indices

    @property
    def face_hex_indices(self) -> wp.array:
        """Hex indices for each face."""
        return self._face_hex_indices

    @property
    def face_vertex_indices(self) -> wp.array:
        """Vertex indices for each face."""
        return self._face_vertex_indices

    @wp.struct
    class SideIndexArg:
        """Arguments for side-index device functions."""

        boundary_face_indices: wp.array(dtype=int)

    # Geometry device interface — topology fill helpers

    def _fill_cell_topo_arg(self, args: HexmeshCellArg, device):
        args.hex_vertex_indices = self.hex_vertex_indices.to(device)
        args.hex_bvh = self.bvh_id(device)

    def _fill_side_topo_arg(self, args: HexmeshSideArg, device):
        self._fill_cell_topo_arg(args.cell_arg, device)
        args.face_vertex_indices = self._face_vertex_indices.to(device)
        args.face_hex_indices = self._face_hex_indices.to(device)
        args.face_hex_face_orientation = self._face_hex_face_orientation.to(device)

    def fill_cell_arg(self, args, device):
        """Fill the arguments to be passed to cell-related device functions."""
        self._fill_cell_topo_arg(args.topology, device)
        args.positions = self.positions.to(device)

    def fill_side_arg(self, args, device):
        """Fill the arguments to be passed to side-related device functions."""
        self._fill_side_topo_arg(args.topology, device)
        args.positions = self.positions.to(device)

    def fill_side_index_arg(self, args: SideIndexArg, device):
        """Fill the arguments to be passed to side-index device functions."""
        args.boundary_face_indices = self._boundary_face_indices.to(device)

    @wp.func
    def boundary_side_index(args: SideIndexArg, boundary_side_index: int):
        """Boundary side to side index"""

        return args.boundary_face_indices[boundary_side_index]

    # -- Dynamic device function constructors --

    def _make_hex_cell_position(self):
        SampleType = self.sample_type
        CoordsType = self.coords_type

        if self.parallelepiped_cells:

            @cache.dynamic_func(suffix=self.name)
            def cell_position(args: self.CellArg, s: SampleType):
                hex_idx = args.topology.hex_vertex_indices[s.element_index]
                w = s.element_coords
                p0 = args.positions[hex_idx[0]]
                p1 = args.positions[hex_idx[1]]
                p2 = args.positions[hex_idx[3]]
                p3 = args.positions[hex_idx[4]]
                return w[0] * p1 + w[1] * p2 + w[2] * p3 + (s.element_coords.dtype(1.0) - w[0] - w[1] - w[2]) * p0

        else:

            @cache.dynamic_func(suffix=self.name)
            def cell_position(args: self.CellArg, s: SampleType):
                hex_idx = args.topology.hex_vertex_indices[s.element_index]

                w_p = s.element_coords
                w_m = CoordsType(s.element_coords.dtype(1.0)) - s.element_coords

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

        return cell_position

    def _make_hex_cell_deformation_gradient(self):
        SampleType = self.sample_type
        CoordsType = self.coords_type
        vec3_type = cached_vec_type(3, self._scalar_type)

        if self.parallelepiped_cells:

            @cache.dynamic_func(suffix=self.name)
            def cell_deformation_gradient(cell_arg: self.CellArg, s: SampleType):
                """Deformation gradient at ``coords``"""
                hex_idx = cell_arg.topology.hex_vertex_indices[s.element_index]

                p0 = cell_arg.positions[hex_idx[0]]
                p1 = cell_arg.positions[hex_idx[1]]
                p2 = cell_arg.positions[hex_idx[3]]
                p3 = cell_arg.positions[hex_idx[4]]
                return wp.matrix_from_cols(p1 - p0, p2 - p0, p3 - p0)

        else:

            @cache.dynamic_func(suffix=self.name)
            def cell_deformation_gradient(cell_arg: self.CellArg, s: SampleType):
                """Deformation gradient at ``coords``"""
                hex_idx = cell_arg.topology.hex_vertex_indices[s.element_index]

                w_p = s.element_coords
                w_m = CoordsType(s.element_coords.dtype(1.0)) - s.element_coords

                return (
                    wp.outer(
                        cell_arg.positions[hex_idx[0]],
                        vec3_type(-w_m[1] * w_m[2], -w_m[0] * w_m[2], -w_m[0] * w_m[1]),
                    )
                    + wp.outer(
                        cell_arg.positions[hex_idx[1]],
                        vec3_type(w_m[1] * w_m[2], -w_p[0] * w_m[2], -w_p[0] * w_m[1]),
                    )
                    + wp.outer(
                        cell_arg.positions[hex_idx[2]],
                        vec3_type(w_p[1] * w_m[2], w_p[0] * w_m[2], -w_p[0] * w_p[1]),
                    )
                    + wp.outer(
                        cell_arg.positions[hex_idx[3]],
                        vec3_type(-w_p[1] * w_m[2], w_m[0] * w_m[2], -w_m[0] * w_p[1]),
                    )
                    + wp.outer(
                        cell_arg.positions[hex_idx[4]],
                        vec3_type(-w_m[1] * w_p[2], -w_m[0] * w_p[2], w_m[0] * w_m[1]),
                    )
                    + wp.outer(
                        cell_arg.positions[hex_idx[5]],
                        vec3_type(w_m[1] * w_p[2], -w_p[0] * w_p[2], w_p[0] * w_m[1]),
                    )
                    + wp.outer(
                        cell_arg.positions[hex_idx[6]],
                        vec3_type(w_p[1] * w_p[2], w_p[0] * w_p[2], w_p[0] * w_p[1]),
                    )
                    + wp.outer(
                        cell_arg.positions[hex_idx[7]],
                        vec3_type(-w_p[1] * w_p[2], w_m[0] * w_p[2], w_m[0] * w_p[1]),
                    )
                )

        return cell_deformation_gradient

    @wp.func
    def side_position(args: Any, s: Any):
        face_idx = args.topology.face_vertex_indices[s.element_index]

        w_p = s.element_coords
        w_m = type(s.element_coords)(s.element_coords.dtype(1.0)) - s.element_coords

        return (
            w_m[0] * w_m[1] * args.positions[face_idx[0]]
            + w_p[0] * w_m[1] * args.positions[face_idx[1]]
            + w_p[0] * w_p[1] * args.positions[face_idx[2]]
            + w_m[0] * w_p[1] * args.positions[face_idx[3]]
        )

    @wp.func
    def _side_deformation_vecs(args: Any, side_index: ElementIndex, coords: Any):
        face_idx = args.topology.face_vertex_indices[side_index]

        p0 = args.positions[face_idx[0]]
        p1 = args.positions[face_idx[1]]
        p2 = args.positions[face_idx[2]]
        p3 = args.positions[face_idx[3]]

        w_p = coords
        w_m = type(coords)(coords.dtype(1.0)) - coords

        v1 = w_m[1] * (p1 - p0) + w_p[1] * (p2 - p3)
        v2 = w_p[0] * (p2 - p1) + w_m[0] * (p3 - p0)
        return v1, v2

    @wp.func
    def side_deformation_gradient(args: Any, s: Any):
        v1, v2 = Hexmesh._side_deformation_vecs(args, s.element_index, s.element_coords)
        return wp.matrix_from_cols(v1, v2)

    @wp.func
    def side_inner_cell_index(arg: Any, side_index: ElementIndex):
        return arg.topology.face_hex_indices[side_index][0]

    @wp.func
    def side_outer_cell_index(arg: Any, side_index: ElementIndex):
        return arg.topology.face_hex_indices[side_index][1]

    @wp.func
    def _hex_local_face_coords(hex_coords: Coords, face_index: int):
        face_coords = wp.vec2(
            hex_coords[_FACE_COORD_INDICES[face_index, 0]], hex_coords[_FACE_COORD_INDICES[face_index, 1]]
        )
        normal_coord = hex_coords[_FACE_COORD_INDICES[face_index, 2]]
        normal_coord = wp.where(_FACE_COORD_INDICES[face_index, 3] == 0, -normal_coord, normal_coord - 1.0)
        return face_coords, normal_coord

    @wp.func
    def _hex_local_face_coords(hex_coords: wp.vec3d, face_index: int):
        face_coords = wp.vec2d(
            hex_coords[_FACE_COORD_INDICES[face_index, 0]], hex_coords[_FACE_COORD_INDICES[face_index, 1]]
        )
        normal_coord = hex_coords[_FACE_COORD_INDICES[face_index, 2]]
        normal_coord = wp.where(_FACE_COORD_INDICES[face_index, 3] == 0, -normal_coord, normal_coord - wp.float64(1.0))
        return face_coords, normal_coord

    @wp.func
    def _local_face_hex_coords(face_coords: Any, face_index: int):
        zero = face_coords.dtype(0.0)
        hex_coords = wp.vector(zero, zero, zero, dtype=face_coords.dtype)
        hex_coords[_FACE_COORD_INDICES[face_index, 0]] = face_coords[0]
        hex_coords[_FACE_COORD_INDICES[face_index, 1]] = face_coords[1]
        hex_coords[_FACE_COORD_INDICES[face_index, 2]] = wp.where(
            _FACE_COORD_INDICES[face_index, 3] == 0,
            zero,
            face_coords.dtype(1.0),
        )
        return hex_coords

    @wp.func
    def _local_from_oriented_face_coords(ori: int, oriented_coords: Coords):
        fv = ori // 2
        return (oriented_coords[0] - _FACE_TRANSLATION_F[fv, 0]) * _FACE_ORIENTATION_F[2 * ori] + (
            oriented_coords[1] - _FACE_TRANSLATION_F[fv, 1]
        ) * _FACE_ORIENTATION_F[2 * ori + 1]

    @wp.func
    def _local_from_oriented_face_coords(ori: int, oriented_coords: wp.vec3d):
        fv = ori // 2
        c0 = wp.float64(oriented_coords[0]) - wp.float64(_FACE_TRANSLATION_F[fv, 0])
        c1 = wp.float64(oriented_coords[1]) - wp.float64(_FACE_TRANSLATION_F[fv, 1])
        o0 = _FACE_ORIENTATION_F[2 * ori]
        o1 = _FACE_ORIENTATION_F[2 * ori + 1]
        return wp.vec2d(
            c0 * wp.float64(o0[0]) + c1 * wp.float64(o1[0]), c0 * wp.float64(o0[1]) + c1 * wp.float64(o1[1])
        )

    @wp.func
    def _local_to_oriented_face_coords(ori: int, coords: Any):
        fv = ori // 2
        return wp.vector(
            wp.dot(type(coords)(_FACE_ORIENTATION_F[2 * ori]), coords) + coords.dtype(_FACE_TRANSLATION_F[fv, 0]),
            wp.dot(type(coords)(_FACE_ORIENTATION_F[2 * ori + 1]), coords) + coords.dtype(_FACE_TRANSLATION_F[fv, 1]),
            coords.dtype(0.0),
            dtype=coords.dtype,
        )

    @wp.func
    def face_to_hex_coords(local_face_index: int, face_orientation: int, side_coords: Any):
        local_coords = Hexmesh._local_from_oriented_face_coords(face_orientation, side_coords)
        return Hexmesh._local_face_hex_coords(local_coords, local_face_index)

    @wp.func
    def side_inner_cell_coords(args: Any, side_index: ElementIndex, side_coords: Any):
        local_face_index = args.topology.face_hex_face_orientation[side_index][0]
        face_orientation = args.topology.face_hex_face_orientation[side_index][1]
        return Hexmesh.face_to_hex_coords(local_face_index, face_orientation, side_coords)

    @wp.func
    def side_outer_cell_coords(args: Any, side_index: ElementIndex, side_coords: Any):
        local_face_index = args.topology.face_hex_face_orientation[side_index][2]
        face_orientation = args.topology.face_hex_face_orientation[side_index][3]
        return Hexmesh.face_to_hex_coords(local_face_index, face_orientation, side_coords)

    @wp.func
    def side_from_cell_coords(args: Any, side_index: ElementIndex, hex_index: ElementIndex, hex_coords: Any):
        if Hexmesh.side_inner_cell_index(args, side_index) == hex_index:
            local_face_index = args.topology.face_hex_face_orientation[side_index][0]
            face_orientation = args.topology.face_hex_face_orientation[side_index][1]
        else:
            local_face_index = args.topology.face_hex_face_orientation[side_index][2]
            face_orientation = args.topology.face_hex_face_orientation[side_index][3]

        face_coords, normal_coord = Hexmesh._hex_local_face_coords(hex_coords, local_face_index)
        return wp.where(
            normal_coord == hex_coords.dtype(0.0),
            Hexmesh._local_to_oriented_face_coords(face_orientation, face_coords),
            type(hex_coords)(hex_coords.dtype(OUTSIDE)),
        )

    def _make_hex_side_to_cell_arg(self):
        CellArgType = self.CellArg

        @cache.dynamic_func(suffix=self.name)
        def side_to_cell_arg(side_arg: self.SideArg):
            """Return the cell argument associated with a side argument."""
            return CellArgType(side_arg.topology.cell_arg, side_arg.positions)

        return side_to_cell_arg

    @wp.func
    def cell_bvh_id(cell_arg: Any):
        return cell_arg.topology.hex_bvh

    @wp.func
    def cell_bounds(cell_arg: Any, cell_index: ElementIndex):
        vidx = cell_arg.topology.hex_vertex_indices[cell_index]
        p0 = cell_arg.positions[vidx[0]]
        p1 = cell_arg.positions[vidx[1]]
        p2 = cell_arg.positions[vidx[2]]
        p3 = cell_arg.positions[vidx[3]]
        lo0, up0 = wp.min(wp.min(p0, p1), wp.min(p2, p3)), wp.max(wp.max(p0, p1), wp.max(p2, p3))

        p4 = cell_arg.positions[vidx[4]]
        p5 = cell_arg.positions[vidx[5]]
        p6 = cell_arg.positions[vidx[6]]
        p7 = cell_arg.positions[vidx[7]]
        lo1, up1 = wp.min(wp.min(p4, p5), wp.min(p6, p7)), wp.max(wp.max(p4, p5), wp.max(p6, p7))

        return wp.min(lo0, lo1), wp.max(up0, up1)

    # -- Topology building (precision-independent) --

    def _build_topology(self, temporary_store: TemporaryStore):
        device = self.hex_vertex_indices.device

        vertex_hex_offsets, vertex_hex_indices = compress_node_indices(
            self.vertex_count(), self.hex_vertex_indices, temporary_store=temporary_store
        )
        self._vertex_hex_offsets = vertex_hex_offsets.detach()
        self._vertex_hex_indices = vertex_hex_indices.detach()

        vertex_start_face_count = borrow_temporary(temporary_store, dtype=int, device=device, shape=self.vertex_count())
        vertex_start_face_count.zero_()
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
            inputs=[self.hex_vertex_indices, vertex_start_face_count],
        )

        array_scan(in_array=vertex_start_face_count, out_array=vertex_start_face_offsets, inclusive=False)

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
                vertex_start_face_offsets,
                vertex_unique_face_count,
                vertex_face_other_vs,
                vertex_face_hexes,
            ],
        )

        vertex_unique_face_offsets = borrow_temporary_like(vertex_start_face_offsets, temporary_store=temporary_store)
        array_scan(in_array=vertex_start_face_count, out_array=vertex_unique_face_offsets, inclusive=False)

        # Get back edge count to host
        face_count = int(
            host_read_at_index(vertex_unique_face_offsets, self.vertex_count() - 1, temporary_store=temporary_store)
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
                vertex_start_face_offsets,
                vertex_unique_face_offsets,
                vertex_unique_face_count,
                vertex_face_other_vs,
                vertex_face_hexes,
                self._face_vertex_indices,
                self._face_hex_indices,
                boundary_mask,
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

        boundary_face_indices, _ = masked_indices(boundary_mask)
        self._boundary_face_indices = boundary_face_indices.detach()

    def _compute_hex_edges(self, temporary_store: TemporaryStore | None = None):
        device = self.hex_vertex_indices.device

        vertex_start_edge_count = borrow_temporary(temporary_store, dtype=int, device=device, shape=self.vertex_count())
        vertex_start_edge_count.zero_()
        vertex_start_edge_offsets = borrow_temporary_like(vertex_start_edge_count, temporary_store=temporary_store)

        vertex_edge_ends = borrow_temporary(temporary_store, dtype=int, device=device, shape=(12 * self.cell_count()))

        # Count face edges starting at each vertex
        wp.launch(
            kernel=Hexmesh._count_starting_edges_kernel,
            device=device,
            dim=self.cell_count(),
            inputs=[self.hex_vertex_indices, vertex_start_edge_count],
        )

        array_scan(in_array=vertex_start_edge_count, out_array=vertex_start_edge_offsets, inclusive=False)

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
                vertex_start_edge_offsets,
                vertex_unique_edge_count,
                vertex_edge_ends,
            ],
        )

        vertex_unique_edge_offsets = borrow_temporary_like(vertex_start_edge_offsets, temporary_store=temporary_store)
        array_scan(in_array=vertex_start_edge_count, out_array=vertex_unique_edge_offsets, inclusive=False)

        # Get back edge count to host
        self._edge_count = int(
            host_read_at_index(vertex_unique_edge_offsets, self.vertex_count() - 1, temporary_store=temporary_store)
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
                vertex_start_edge_offsets,
                vertex_unique_edge_offsets,
                vertex_unique_edge_count,
                vertex_edge_ends,
                self._hex_edge_indices,
            ],
        )

        vertex_start_edge_offsets.release()
        vertex_unique_edge_offsets.release()
        vertex_unique_edge_count.release()
        vertex_edge_ends.release()

    # -- Topology kernels (precision-independent) --

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
        positions: wp.array(dtype=Any),
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
        ) / positions[hex_vidx[0]].dtype(8.0)

        v0 = positions[face_vidx[0]]
        v1 = positions[face_vidx[1]]
        v2 = positions[face_vidx[2]]
        v3 = positions[face_vidx[3]]

        face_center = (v1 + v0 + v2 + v3) / v0.dtype(4.0)
        face_normal = wp.cross(v2 - v0, v3 - v1)

        # if face normal points toward first hex centroid, flip indices
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
