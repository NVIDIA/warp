from typing import Optional

import numpy as np

import warp as wp
from warp.fem import cache, utils
from warp.fem.types import NULL_ELEMENT_INDEX, OUTSIDE, Coords, ElementIndex, Sample, make_free_sample

from .element import Cube, Square
from .geometry import Geometry

# Flag used for building edge/face grids to disambiguiate axis within the grid
GRID_AXIS_FLAG = wp.constant(wp.int32(1 << 20))
FACE_AXIS_MASK = wp.constant(wp.uint8((1 << 3) - 1))
FACE_INNER_OFFSET_BIT = wp.constant(wp.uint8(3))
FACE_OUTER_OFFSET_BIT = wp.constant(wp.uint8(4))

_mat32 = wp.mat(shape=(3, 2), dtype=float)


@wp.func
def _add_axis_flag(ijk: wp.vec3i, axis: int):
    coord = ijk[axis]
    ijk[axis] = wp.select(coord < 0, coord | GRID_AXIS_FLAG, coord & (~GRID_AXIS_FLAG))
    return ijk


@wp.func
def _extract_axis_flag(ijk: wp.vec3i):
    for ax in range(3):
        coord = ijk[ax]
        if coord < 0:
            if (ijk[ax] & GRID_AXIS_FLAG) == 0:
                ijk[ax] = ijk[ax] | GRID_AXIS_FLAG
                return ax, ijk
        else:
            if (ijk[ax] & GRID_AXIS_FLAG) != 0:
                ijk[ax] = ijk[ax] & (~GRID_AXIS_FLAG)
                return ax, ijk

    return -1, ijk


@wp.struct
class NanogridCellArg:
    # Utility device functions
    cell_grid: wp.uint64
    cell_ijk: wp.array(dtype=wp.vec3i)
    inverse_transform: wp.mat33
    cell_volume: float


@wp.struct
class NanogridSideArg:
    # Utility device functions
    cell_arg: NanogridCellArg
    face_ijk: wp.array(dtype=wp.vec3i)
    face_flags: wp.array(dtype=wp.uint8)
    face_areas: wp.vec3


class Nanogrid(Geometry):
    dimension = 3

    def __init__(self, grid: wp.Volume, temporary_store: Optional[cache.TemporaryStore] = None):
        self._cell_grid = grid
        self._cell_grid_info = grid.get_grid_info()

        device = grid.device

        cell_count = grid.get_voxel_count()
        self._cell_ijk = wp.array(shape=(cell_count,), dtype=wp.vec3i, device=device)
        grid.get_voxels(out=self._cell_ijk)

        self._node_grid = _build_node_grid(self._cell_ijk, grid, temporary_store)
        node_count = self._node_grid.get_voxel_count()
        self._node_ijk = wp.array(shape=(node_count,), dtype=wp.vec3i, device=device)
        self._node_grid.get_voxels(out=self._node_ijk)

        self._face_grid = _build_face_grid(self._cell_ijk, grid, temporary_store)
        face_count = self._face_grid.get_voxel_count()
        self._face_ijk = wp.array(shape=(face_count,), dtype=wp.vec3i, device=device)
        self._face_grid.get_voxels(out=self._face_ijk)

        self._face_flags = wp.array(shape=(face_count,), dtype=wp.uint8, device=device)
        boundary_face_mask = cache.borrow_temporary(temporary_store, shape=(face_count,), dtype=wp.int32, device=device)

        wp.launch(
            _build_face_flags,
            dim=face_count,
            device=device,
            inputs=[grid.id, self._face_ijk, self._face_flags, boundary_face_mask.array],
        )
        boundary_face_indices, _ = utils.masked_indices(boundary_face_mask.array)
        self._boundary_face_indices = boundary_face_indices.detach()

        self._edge_grid = None
        self._edge_ijk = None

    def _build_edge_grid(self, temporary_store: Optional[cache.TemporaryStore] = None):
        self._edge_grid = _build_edge_grid(self._cell_ijk, self._cell_grid, temporary_store)
        edge_count = self._edge_grid.get_voxel_count()
        self._edge_ijk = wp.array(shape=(edge_count,), dtype=wp.vec3i, device=self._edge_grid.device)
        self._edge_grid.get_voxels(out=self._edge_ijk)

    def cell_count(self):
        return self._cell_ijk.shape[0]

    def vertex_count(self):
        return self._node_ijk.shape[0]

    def side_count(self):
        return self._face_ijk.shape[0]

    def edge_count(self):
        if self._edge_ijk is None:
            self._build_edge_grid()

        return self._edge_ijk.shape[0]

    def boundary_side_count(self):
        return self._boundary_face_indices.shape[0]

    def reference_cell(self) -> Cube:
        return Cube()

    def reference_side(self) -> Square:
        return Square()

    CellArg = NanogridCellArg

    @cache.cached_arg_value
    def cell_arg_value(self, device) -> CellArg:
        args = self.CellArg()
        args.cell_grid = self._cell_grid.id
        args.cell_ijk = self._cell_ijk

        transform = np.array(self._cell_grid_info.transform_matrix).reshape(3, 3)
        args.inverse_transform = wp.mat33f(np.linalg.inv(transform))
        args.cell_volume = abs(np.linalg.det(transform))

        return args

    @wp.func
    def cell_position(args: CellArg, s: Sample):
        uvw = wp.vec3(args.cell_ijk[s.element_index]) + s.element_coords
        return wp.volume_index_to_world(args.cell_grid, uvw)

    @wp.func
    def cell_deformation_gradient(args: CellArg, s: Sample):
        return wp.inverse(args.inverse_transform)

    @wp.func
    def cell_inverse_deformation_gradient(args: CellArg, s: Sample):
        return args.inverse_transform

    @wp.func
    def cell_lookup(args: CellArg, pos: wp.vec3):
        uvw = wp.volume_world_to_index(args.cell_grid, pos)
        ijk = wp.vec3i(int(wp.floor(uvw[0])), int(wp.floor(uvw[1])), int(wp.floor(uvw[2])))
        element_index = wp.volume_lookup_index(args.cell_grid, ijk[0], ijk[1], ijk[2])

        return wp.select(
            element_index == -1,
            make_free_sample(element_index, uvw - wp.vec3(ijk)),
            make_free_sample(NULL_ELEMENT_INDEX, Coords(OUTSIDE)),
        )

    @wp.func
    def cell_lookup(args: CellArg, pos: wp.vec3, guess: Sample):
        return Nanogrid.cell_lookup(args, pos)

    @wp.func
    def cell_measure(args: CellArg, s: Sample):
        return args.cell_volume

    @wp.func
    def cell_normal(args: CellArg, s: Sample):
        return wp.vec3(0.0)

    SideArg = NanogridSideArg

    @cache.cached_arg_value
    def side_arg_value(self, device) -> SideArg:
        args = self.SideArg()
        args.cell_arg = self.cell_arg_value(device)
        args.face_ijk = self._face_ijk.to(device)
        args.face_flags = self._face_flags.to(device)
        transform = np.array(self._cell_grid_info.transform_matrix).reshape(3, 3)
        args.face_areas = wp.vec3(
            tuple(np.linalg.norm(np.cross(transform[:, k - 2], transform[:, k - 1])) for k in range(3))
        )

        return args

    @wp.struct
    class SideIndexArg:
        boundary_face_indices: wp.array(dtype=int)

    @cache.cached_arg_value
    def side_index_arg_value(self, device) -> SideIndexArg:
        args = self.SideIndexArg()
        args.boundary_face_indices = self._boundary_face_indices.to(device)
        return args

    @wp.func
    def boundary_side_index(args: SideIndexArg, boundary_side_index: int):
        return args.boundary_face_indices[boundary_side_index]

    @wp.func
    def _side_to_cell_coords(axis: int, inner: float, side_coords: Coords):
        uvw = wp.vec3()
        uvw[axis] = inner
        uvw[(axis + 1) % 3] = side_coords[0]
        uvw[(axis + 2) % 3] = side_coords[1]
        return uvw

    @wp.func
    def _get_face_axis(flags: wp.uint8):
        return wp.int32(flags & FACE_AXIS_MASK)

    @wp.func
    def _get_face_inner_offset(flags: wp.uint8):
        return wp.int32(flags >> FACE_INNER_OFFSET_BIT) & 1

    @wp.func
    def _get_face_outer_offset(flags: wp.uint8):
        return wp.int32(flags >> FACE_OUTER_OFFSET_BIT) & 1

    @wp.func
    def side_position(args: SideArg, s: Sample):
        ijk = args.face_ijk[s.element_index]
        axis = Nanogrid._get_face_axis(args.face_flags[s.element_index])

        uvw = wp.vec3(ijk) + Nanogrid._side_to_cell_coords(axis, 0.0, s.element_coords)

        cell_grid = args.cell_arg.cell_grid
        return wp.volume_index_to_world(cell_grid, uvw)

    @wp.func
    def _face_tangent_vecs(args: SideArg, axis: int, flip: int):
        u_axis = utils.unit_element(wp.vec3(), (axis + 1 + flip) % 3)
        v_axis = utils.unit_element(wp.vec3(), (axis + 2 - flip) % 3)

        cell_grid = args.cell_arg.cell_grid

        return wp.volume_index_to_world_dir(cell_grid, u_axis), wp.volume_index_to_world_dir(cell_grid, v_axis)

    @wp.func
    def side_deformation_gradient(args: SideArg, s: Sample):
        flags = args.face_flags[s.element_index]
        axis = Nanogrid._get_face_axis(flags)
        flip = Nanogrid._get_face_inner_offset(flags)
        v1, v2 = Nanogrid._face_tangent_vecs(args, axis, flip)
        return _mat32(v1, v2)

    @wp.func
    def side_inner_inverse_deformation_gradient(args: SideArg, s: Sample):
        return Nanogrid.cell_inverse_deformation_gradient(args.cell_arg, s)

    @wp.func
    def side_outer_inverse_deformation_gradient(args: SideArg, s: Sample):
        return Nanogrid.cell_inverse_deformation_gradient(args.cell_arg, s)

    @wp.func
    def side_measure(args: SideArg, s: Sample):
        axis = Nanogrid._get_face_axis(args.face_flags[s.element_index])
        return args.face_areas[axis]

    @wp.func
    def side_measure_ratio(args: SideArg, s: Sample):
        axis = Nanogrid._get_face_axis(args.face_flags[s.element_index])
        return args.face_areas[axis] / args.cell_arg.cell_volume

    @wp.func
    def side_normal(args: SideArg, s: Sample):
        flags = args.face_flags[s.element_index]
        axis = Nanogrid._get_face_axis(flags)
        flip = Nanogrid._get_face_inner_offset(flags)

        v1, v2 = Nanogrid._face_tangent_vecs(args, axis, flip)
        return wp.cross(v1, v2) / args.face_areas[axis]

    @wp.func
    def side_inner_cell_index(args: SideArg, side_index: ElementIndex):
        ijk = args.face_ijk[side_index]
        flags = args.face_flags[side_index]
        axis = Nanogrid._get_face_axis(flags)
        offset = Nanogrid._get_face_inner_offset(flags)

        ijk[axis] += offset - 1
        cell_grid = args.cell_arg.cell_grid

        return wp.volume_lookup_index(cell_grid, ijk[0], ijk[1], ijk[2])

    @wp.func
    def side_outer_cell_index(args: SideArg, side_index: ElementIndex):
        ijk = args.face_ijk[side_index]
        flags = args.face_flags[side_index]
        axis = Nanogrid._get_face_axis(flags)
        offset = Nanogrid._get_face_outer_offset(flags)

        ijk[axis] -= offset
        cell_grid = args.cell_arg.cell_grid

        return wp.volume_lookup_index(cell_grid, ijk[0], ijk[1], ijk[2])

    @wp.func
    def side_inner_cell_coords(args: SideArg, side_index: ElementIndex, side_coords: Coords):
        flags = args.face_flags[side_index]
        axis = Nanogrid._get_face_axis(flags)
        offset = float(Nanogrid._get_face_inner_offset(flags))
        return Nanogrid._side_to_cell_coords(axis, 1.0 - offset, side_coords)

    @wp.func
    def side_outer_cell_coords(args: SideArg, side_index: ElementIndex, side_coords: Coords):
        flags = args.face_flags[side_index]
        axis = Nanogrid._get_face_axis(flags)
        offset = float(Nanogrid._get_face_outer_offset(flags))
        return Nanogrid._side_to_cell_coords(axis, offset, side_coords)

    @wp.func
    def side_from_cell_coords(
        args: SideArg,
        side_index: ElementIndex,
        element_index: ElementIndex,
        element_coords: Coords,
    ):
        flags = args.face_flags[side_index]
        axis = Nanogrid._get_face_axis(flags)

        cell_ijk = args.cell_arg.cell_ijk[element_index]
        side_ijk = args.face_ijk[side_index]

        on_side = float(side_ijk[axis] - cell_ijk[axis]) == element_coords[axis]

        return wp.select(
            on_side, Coords(OUTSIDE), Coords(element_coords[(axis + 1) % 3], element_coords[(axis + 2) % 3], 0.0)
        )

    @wp.func
    def side_to_cell_arg(side_arg: SideArg):
        return side_arg.cell_arg


@wp.kernel
def _cell_node_indices(
    cell_ijk: wp.array(dtype=wp.vec3i),
    node_ijk: wp.array2d(dtype=wp.vec3i),
):
    cell, n = wp.tid()
    node_ijk[cell, n] = cell_ijk[cell] + wp.vec3i((n & 4) >> 2, (n & 2) >> 1, n & 1)


@wp.kernel
def _cell_face_indices(
    cell_ijk: wp.array(dtype=wp.vec3i),
    node_ijk: wp.array2d(dtype=wp.vec3i),
):
    cell = wp.tid()
    ijk = cell_ijk[cell]
    node_ijk[cell, 0] = _add_axis_flag(ijk, 0)
    node_ijk[cell, 1] = _add_axis_flag(ijk, 1)
    node_ijk[cell, 2] = _add_axis_flag(ijk, 2)

    node_ijk[cell, 3] = _add_axis_flag(ijk + wp.vec3i(1, 0, 0), 0)
    node_ijk[cell, 4] = _add_axis_flag(ijk + wp.vec3i(0, 1, 0), 1)
    node_ijk[cell, 5] = _add_axis_flag(ijk + wp.vec3i(0, 0, 1), 2)


@wp.kernel
def _cell_edge_indices(
    cell_ijk: wp.array(dtype=wp.vec3i),
    edge_ijk: wp.array2d(dtype=wp.vec3i),
):
    cell = wp.tid()
    ijk = cell_ijk[cell]
    edge_ijk[cell, 0] = _add_axis_flag(ijk, 0)
    edge_ijk[cell, 1] = _add_axis_flag(ijk, 1)
    edge_ijk[cell, 2] = _add_axis_flag(ijk, 2)

    edge_ijk[cell, 3] = _add_axis_flag(ijk + wp.vec3i(0, 1, 0), 0)
    edge_ijk[cell, 4] = _add_axis_flag(ijk + wp.vec3i(0, 0, 1), 1)
    edge_ijk[cell, 5] = _add_axis_flag(ijk + wp.vec3i(1, 0, 0), 2)

    edge_ijk[cell, 6] = _add_axis_flag(ijk + wp.vec3i(0, 1, 1), 0)
    edge_ijk[cell, 7] = _add_axis_flag(ijk + wp.vec3i(1, 0, 1), 1)
    edge_ijk[cell, 8] = _add_axis_flag(ijk + wp.vec3i(1, 1, 0), 2)

    edge_ijk[cell, 9] = _add_axis_flag(ijk + wp.vec3i(0, 0, 1), 0)
    edge_ijk[cell, 10] = _add_axis_flag(ijk + wp.vec3i(1, 0, 0), 1)
    edge_ijk[cell, 11] = _add_axis_flag(ijk + wp.vec3i(0, 1, 0), 2)


def _build_node_grid(cell_ijk, grid: wp.Volume, temporary_store: cache.TemporaryStore):
    cell_count = cell_ijk.shape[0]

    cell_nodes = cache.borrow_temporary(temporary_store, shape=(cell_count, 8), dtype=wp.vec3i, device=cell_ijk.device)
    wp.launch(
        _cell_node_indices, dim=cell_nodes.array.shape, inputs=[cell_ijk, cell_nodes.array], device=cell_ijk.device
    )
    node_grid = wp.Volume.allocate_by_voxels(
        cell_nodes.array.flatten(), voxel_size=grid.get_voxel_size()[0], device=cell_ijk.device
    )

    return node_grid


def _build_face_grid(cell_ijk, grid: wp.Volume, temporary_store: cache.TemporaryStore):
    cell_count = cell_ijk.shape[0]

    cell_faces = cache.borrow_temporary(temporary_store, shape=(cell_count, 6), dtype=wp.vec3i, device=cell_ijk.device)
    wp.launch(_cell_face_indices, dim=cell_count, inputs=[cell_ijk, cell_faces.array], device=cell_ijk.device)
    face_grid = wp.Volume.allocate_by_voxels(
        cell_faces.array.flatten(), voxel_size=grid.get_voxel_size()[0], device=cell_ijk.device
    )

    return face_grid


def _build_edge_grid(cell_ijk, grid: wp.Volume, temporary_store: cache.TemporaryStore):
    cell_count = cell_ijk.shape[0]

    cell_edges = cache.borrow_temporary(temporary_store, shape=(cell_count, 12), dtype=wp.vec3i, device=cell_ijk.device)
    wp.launch(_cell_edge_indices, dim=cell_count, inputs=[cell_ijk, cell_edges.array], device=cell_ijk.device)
    edge_grid = wp.Volume.allocate_by_voxels(
        cell_edges.array.flatten(), voxel_size=grid.get_voxel_size()[0], device=cell_ijk.device
    )

    return edge_grid


@wp.kernel
def _build_face_flags(
    cell_grid: wp.uint64,
    face_ijk: wp.array(dtype=wp.vec3i),
    face_flags: wp.array(dtype=wp.uint8),
    boundary_face_mask: wp.array(dtype=int),
):
    face = wp.tid()

    axis, ijk = _extract_axis_flag(face_ijk[face])

    ijk_minus = ijk
    ijk_minus[axis] -= 1

    plus_cell_index = wp.volume_lookup_index(cell_grid, ijk[0], ijk[1], ijk[2])
    minus_cell_index = wp.volume_lookup_index(cell_grid, ijk_minus[0], ijk_minus[1], ijk_minus[2])

    plus_boundary = wp.uint8(wp.select(plus_cell_index == -1, 0, 1)) << FACE_OUTER_OFFSET_BIT
    minus_boundary = wp.uint8(wp.select(minus_cell_index == -1, 0, 1)) << FACE_INNER_OFFSET_BIT

    face_ijk[face] = ijk
    face_flags[face] = wp.uint8(axis) | plus_boundary | minus_boundary
    boundary_face_mask[face] = wp.select((plus_boundary | minus_boundary) == 0, 1, 0)
