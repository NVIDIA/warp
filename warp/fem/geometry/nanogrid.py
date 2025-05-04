# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Optional

import numpy as np

import warp as wp
from warp.fem import cache, utils
from warp.fem.types import NULL_ELEMENT_INDEX, OUTSIDE, Coords, ElementIndex, Sample, make_free_sample

from .element import Cube, Square
from .geometry import Geometry

# Flag used for building edge/face grids to disambiguiate axis within the grid
# Morton indexing allows for
GRID_AXIS_FLAG = wp.constant(wp.int32(1 << 20))

FACE_AXIS_MASK = wp.constant(wp.uint8((1 << 2) - 1))
FACE_INNER_OFFSET_BIT = wp.constant(wp.uint8(2))
FACE_OUTER_OFFSET_BIT = wp.constant(wp.uint8(3))


@wp.func
def _add_axis_flag(ijk: wp.vec3i, axis: int):
    coord = ijk[axis]
    ijk[axis] = wp.where(coord < 0, coord & (~GRID_AXIS_FLAG), coord | GRID_AXIS_FLAG)
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
    """Sparse grid geometry"""

    dimension = 3

    def __init__(self, grid: wp.Volume, temporary_store: Optional[cache.TemporaryStore] = None):
        """
        Constructs a sparse grid geometry from an in-memory NanoVDB volume.

        Args:
            grid: The NanoVDB volume. Any type is accepted, but for indexing efficiency an index grid is recommended.
                If `grid` is an 'on' index grid, cells will be created for active voxels only, otherwise cells will
                be created for all leaf voxels.
            temporary_store: shared pool from which to allocate temporary arrays
        """

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

        self._face_grid = None
        self._face_ijk = None

        self._edge_grid = None
        self._edge_count = 0

    @property
    def cell_grid(self) -> wp.Volume:
        return self._cell_grid

    @property
    def vertex_grid(self) -> wp.Volume:
        return self._node_grid

    @property
    def face_grid(self) -> wp.Volume:
        self._ensure_face_grid()
        return self._face_grid

    @property
    def edge_grid(self) -> wp.Volume:
        self._ensure_edge_grid()
        return self._edge_grid

    def cell_count(self):
        return self._cell_ijk.shape[0]

    def vertex_count(self):
        return self._node_ijk.shape[0]

    def side_count(self):
        self._ensure_face_grid()
        return self._face_ijk.shape[0]

    def boundary_side_count(self):
        self._ensure_face_grid()
        return self._boundary_face_indices.shape[0]

    def edge_count(self):
        self._ensure_edge_grid()
        return self._edge_count

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
        return wp.volume_index_to_world(args.cell_grid, uvw - wp.vec3(0.5))

    @wp.func
    def cell_deformation_gradient(args: CellArg, s: Sample):
        return wp.inverse(args.inverse_transform)

    @wp.func
    def cell_inverse_deformation_gradient(args: CellArg, s: Sample):
        return args.inverse_transform

    def supports_cell_lookup(self, device):
        return True

    @wp.func
    def _lookup_cell_index(args: NanogridCellArg, i: int, j: int, k: int):
        return wp.volume_lookup_index(args.cell_grid, i, j, k)

    @wp.func
    def _cell_coordinates_local(args: NanogridCellArg, cell_index: int, uvw: wp.vec3):
        ijk = wp.vec3(args.cell_ijk[cell_index])
        rel_pos = uvw - ijk
        return rel_pos

    @wp.func
    def _cell_closest_point_local(args: NanogridCellArg, cell_index: int, uvw: wp.vec3):
        ijk = wp.vec3(args.cell_ijk[cell_index])
        rel_pos = uvw - ijk
        coords = wp.min(wp.max(rel_pos, wp.vec3(0.0)), wp.vec3(1.0))
        return wp.length_sq(wp.volume_index_to_world_dir(args.cell_grid, coords - rel_pos)), coords

    @wp.func
    def cell_coordinates(args: NanogridCellArg, cell_index: int, pos: wp.vec3):
        uvw = wp.volume_world_to_index(args.cell_grid, pos) + wp.vec3(0.5)
        return Nanogrid._cell_coordinates_local(args, cell_index, uvw)

    @wp.func
    def cell_closest_point(args: NanogridCellArg, cell_index: int, pos: wp.vec3):
        uvw = wp.volume_world_to_index(args.cell_grid, pos) + wp.vec3(0.5)
        dist, coords = Nanogrid._cell_closest_point_local(args, cell_index, uvw)
        return coords, dist

    @staticmethod
    def _make_filtered_cell_lookup(grid_geo, filter_func: wp.Function = None):
        suffix = f"{grid_geo.name}{filter_func.func.__qualname__ if filter_func is not None else ''}"

        @cache.dynamic_func(suffix=suffix)
        def cell_lookup(args: grid_geo.CellArg, pos: wp.vec3, max_dist: float, filter_data: Any, filter_target: Any):
            grid = args.cell_grid

            # Start at corresponding voxel
            uvw = wp.volume_world_to_index(grid, pos) + wp.vec3(0.5)
            i, j, k = int(wp.floor(uvw[0])), int(wp.floor(uvw[1])), int(wp.floor(uvw[2]))
            cell_index = grid_geo._lookup_cell_index(args, i, j, k)

            if cell_index != -1:
                coords = grid_geo._cell_coordinates_local(args, cell_index, uvw)
                if wp.static(filter_func is None):
                    return make_free_sample(cell_index, coords)
                else:
                    if filter_func(filter_data, cell_index) == filter_target:
                        return make_free_sample(cell_index, coords)

            # Iterate over increasingly larger neighborhoods
            cell_size = wp.vec3(
                wp.length(wp.volume_index_to_world_dir(grid, wp.vec3(1.0, 0.0, 0.0))),
                wp.length(wp.volume_index_to_world_dir(grid, wp.vec3(0.0, 1.0, 0.0))),
                wp.length(wp.volume_index_to_world_dir(grid, wp.vec3(0.0, 0.0, 1.0))),
            )

            offset = float(0.5)
            min_cell_size = wp.min(cell_size)
            max_offset = wp.ceil(max_dist / min_cell_size)
            scales = wp.cw_div(wp.vec3(min_cell_size), wp.vec3(cell_size))

            closest_cell = NULL_ELEMENT_INDEX
            closest_coords = Coords()

            while closest_cell == NULL_ELEMENT_INDEX:
                uvw_min = wp.vec3i(uvw - offset * scales)
                uvw_max = wp.vec3i(uvw + offset * scales) + wp.vec3i(1)

                closest_dist = min_cell_size * min_cell_size * float(offset * offset)

                for i in range(uvw_min[0], uvw_max[0]):
                    for j in range(uvw_min[1], uvw_max[1]):
                        for k in range(uvw_min[2], uvw_max[2]):
                            cell_index = grid_geo._lookup_cell_index(args, i, j, k)
                            if cell_index == -1:
                                continue

                            if wp.static(filter_func is not None):
                                if filter_func(filter_data, cell_index) != filter_target:
                                    continue
                            dist, coords = grid_geo._cell_closest_point_local(args, cell_index, uvw)

                            if dist <= closest_dist:
                                closest_dist = dist
                                closest_coords = coords
                                closest_cell = cell_index

                if offset >= max_offset:
                    break
                offset = wp.min(3.0 * offset, max_offset)

            return make_free_sample(closest_cell, closest_coords)

        return cell_lookup

    def make_filtered_cell_lookup(self, filter_func):
        return Nanogrid._make_filtered_cell_lookup(self, filter_func)

    @wp.func
    def cell_measure(args: CellArg, s: Sample):
        return args.cell_volume

    @wp.func
    def cell_normal(args: CellArg, s: Sample):
        return wp.vec3(0.0)

    SideArg = NanogridSideArg

    @cache.cached_arg_value
    def side_arg_value(self, device) -> SideArg:
        self._ensure_face_grid()

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
        self._ensure_face_grid()

        args = self.SideIndexArg()
        args.boundary_face_indices = self._boundary_face_indices.to(device)
        return args

    @wp.func
    def boundary_side_index(args: SideIndexArg, boundary_side_index: int):
        return args.boundary_face_indices[boundary_side_index]

    @wp.func
    def _side_to_cell_coords(axis: int, flip: int, inner: float, side_coords: Coords):
        uvw = wp.vec3()
        uvw[axis] = inner
        uvw[(axis + 1 + flip) % 3] = side_coords[0]
        uvw[(axis + 2 - flip) % 3] = side_coords[1]
        return uvw

    @wp.func
    def _cell_to_side_coords(axis: int, flip: int, cell_coords: Coords):
        return Coords(cell_coords[(axis + 1 + flip) % 3], cell_coords[(axis + 2 - flip) % 3], 0.0)

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
        flags = args.face_flags[s.element_index]
        axis = Nanogrid._get_face_axis(flags)
        flip = Nanogrid._get_face_inner_offset(flags)

        uvw = wp.vec3(ijk) + Nanogrid._side_to_cell_coords(axis, flip, 0.0, s.element_coords)

        cell_grid = args.cell_arg.cell_grid
        return wp.volume_index_to_world(cell_grid, uvw - wp.vec3(0.5))

    @wp.func
    def _face_tangent_vecs(cell_grid: wp.uint64, axis: int, flip: int):
        u_axis = wp.vec3()
        v_axis = wp.vec3()
        u_axis[(axis + 1 + flip) % 3] = 1.0
        v_axis[(axis + 2 - flip) % 3] = 1.0
        return wp.volume_index_to_world_dir(cell_grid, u_axis), wp.volume_index_to_world_dir(cell_grid, v_axis)

    @wp.func
    def side_deformation_gradient(args: SideArg, s: Sample):
        flags = args.face_flags[s.element_index]
        axis = Nanogrid._get_face_axis(flags)
        flip = Nanogrid._get_face_inner_offset(flags)
        v1, v2 = Nanogrid._face_tangent_vecs(args.cell_arg.cell_grid, axis, flip)
        return wp.matrix_from_cols(v1, v2)

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

        v1, v2 = Nanogrid._face_tangent_vecs(args.cell_arg.cell_grid, axis, flip)
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
        flip = Nanogrid._get_face_inner_offset(flags)
        offset = float(Nanogrid._get_face_inner_offset(flags))
        return Nanogrid._side_to_cell_coords(axis, flip, 1.0 - offset, side_coords)

    @wp.func
    def side_outer_cell_coords(args: SideArg, side_index: ElementIndex, side_coords: Coords):
        flags = args.face_flags[side_index]
        axis = Nanogrid._get_face_axis(flags)
        flip = Nanogrid._get_face_inner_offset(flags)
        offset = float(Nanogrid._get_face_outer_offset(flags))
        return Nanogrid._side_to_cell_coords(axis, flip, offset, side_coords)

    @wp.func
    def side_from_cell_coords(
        args: SideArg,
        side_index: ElementIndex,
        element_index: ElementIndex,
        element_coords: Coords,
    ):
        flags = args.face_flags[side_index]
        axis = Nanogrid._get_face_axis(flags)
        flip = Nanogrid._get_face_inner_offset(flags)

        cell_ijk = args.cell_arg.cell_ijk[element_index]
        side_ijk = args.face_ijk[side_index]

        on_side = float(side_ijk[axis] - cell_ijk[axis]) == element_coords[axis]

        return wp.where(on_side, Nanogrid._cell_to_side_coords(axis, flip, element_coords), Coords(OUTSIDE))

    @wp.func
    def side_to_cell_arg(side_arg: SideArg):
        return side_arg.cell_arg

    @wp.func
    def side_coordinates(args: SideArg, side_index: int, pos: wp.vec3):
        cell_arg = args.cell_arg

        ijk = args.face_ijk[side_index]
        cell_coords = wp.volume_world_to_index(cell_arg.cell_grid, pos) + wp.vec3(0.5) - wp.vec3(ijk)

        flags = args.face_flags[side_index]
        axis = Nanogrid._get_face_axis(flags)
        flip = Nanogrid._get_face_inner_offset(flags)
        return Nanogrid._cell_to_side_coords(axis, flip, cell_coords)

    @wp.func
    def side_closest_point(args: SideArg, side_index: int, pos: wp.vec3):
        coords = Nanogrid.side_coordinates(args, side_index, pos)

        proj_coords = Coords(wp.clamp(coords[0], 0.0, 1.0), wp.clamp(coords[1], 0.0, 1.0), 0.0)

        flags = args.face_flags[side_index]
        axis = Nanogrid._get_face_axis(flags)
        flip = Nanogrid._get_face_inner_offset(flags)
        cell_coord_offset = Nanogrid._side_to_cell_coords(axis, flip, 0, coords - proj_coords)

        return proj_coords, wp.length_sq(wp.volume_index_to_world_dir(args.cell_grid, cell_coord_offset))

    def _build_face_grid(self, temporary_store: Optional[cache.TemporaryStore] = None):
        device = self._cell_grid.device
        self._face_grid = _build_face_grid(self._cell_ijk, self._cell_grid, temporary_store)
        face_count = self._face_grid.get_voxel_count()
        self._face_ijk = wp.array(shape=(face_count,), dtype=wp.vec3i, device=device)
        self._face_grid.get_voxels(out=self._face_ijk)

        self._face_flags = wp.array(shape=(face_count,), dtype=wp.uint8, device=device)
        boundary_face_mask = cache.borrow_temporary(temporary_store, shape=(face_count,), dtype=wp.int32, device=device)

        wp.launch(
            _build_face_flags,
            dim=face_count,
            device=device,
            inputs=[self._cell_grid.id, self._face_ijk, self._face_flags, boundary_face_mask.array],
        )
        boundary_face_indices, _ = utils.masked_indices(boundary_face_mask.array)
        self._boundary_face_indices = boundary_face_indices.detach()

    def _build_edge_grid(self, temporary_store: Optional[cache.TemporaryStore] = None):
        self._edge_grid = _build_edge_grid(self._cell_ijk, self._cell_grid, temporary_store)
        self._edge_count = self._edge_grid.get_voxel_count()

    def _ensure_face_grid(self):
        if self._face_ijk is None:
            self._build_face_grid()

    def _ensure_edge_grid(self):
        if self._edge_grid is None:
            self._build_edge_grid()


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
        cell_nodes.array.flatten(), voxel_size=grid.get_voxel_size(), device=cell_ijk.device
    )

    return node_grid


def _build_face_grid(cell_ijk, grid: wp.Volume, temporary_store: cache.TemporaryStore):
    cell_count = cell_ijk.shape[0]

    cell_faces = cache.borrow_temporary(temporary_store, shape=(cell_count, 6), dtype=wp.vec3i, device=cell_ijk.device)
    wp.launch(_cell_face_indices, dim=cell_count, inputs=[cell_ijk, cell_faces.array], device=cell_ijk.device)
    face_grid = wp.Volume.allocate_by_voxels(
        cell_faces.array.flatten(), voxel_size=grid.get_voxel_size(), device=cell_ijk.device
    )

    return face_grid


def _build_edge_grid(cell_ijk, grid: wp.Volume, temporary_store: cache.TemporaryStore):
    cell_count = cell_ijk.shape[0]

    cell_edges = cache.borrow_temporary(temporary_store, shape=(cell_count, 12), dtype=wp.vec3i, device=cell_ijk.device)
    wp.launch(_cell_edge_indices, dim=cell_count, inputs=[cell_ijk, cell_edges.array], device=cell_ijk.device)
    edge_grid = wp.Volume.allocate_by_voxels(
        cell_edges.array.flatten(), voxel_size=grid.get_voxel_size(), device=cell_ijk.device
    )

    return edge_grid


@wp.func
def _make_face_flags(axis: int, plus_cell_index: int, minus_cell_index: int):
    plus_boundary = wp.uint8(wp.where(plus_cell_index == -1, 1, 0)) << FACE_OUTER_OFFSET_BIT
    minus_boundary = wp.uint8(wp.where(minus_cell_index == -1, 1, 0)) << FACE_INNER_OFFSET_BIT

    return wp.uint8(axis) | plus_boundary | minus_boundary


@wp.func
def _get_boundary_mask(flags: wp.uint8):
    return int((flags >> FACE_OUTER_OFFSET_BIT) | (flags >> FACE_INNER_OFFSET_BIT)) & 1


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

    face_ijk[face] = ijk

    flags = _make_face_flags(axis, plus_cell_index, minus_cell_index)
    face_flags[face] = flags
    boundary_face_mask[face] = _get_boundary_mask(flags)
